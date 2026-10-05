"""Expanded, deterministic development validation; never claims a blind test.

Uses ALL previously reserved internal IDs, not any final-test composition.
Weak-vocal remixes reuse source windows and are not independent recordings.
Selection cannot be won by instrumental-only improvements or a muted output.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load("selection_training_core", "110_train_residual_ablation.py")
audit = load("selection_waveform_audit", "109_audit_model_limits.py")
VOCAL_DOMAINS = ("musdb/native", "musdb/weak_minus12", "mir1k/native", "mir1k/weak_minus12")
POLICY = {
    "domain_snr_max_regression_db": 0.25,
    "instrumental_snr_max_regression_db": 0.25,
    "accompaniment_gain_max_drop": 0.03,
    "residual_vocal_max_increase_db": 0.25,
    "mean_residual_vocal_max_increase_db": 0.0,
    "minimum_mean_vocal_snr_gain_db": 0.15,
    "rank": "equal-weight mean vocal error-SNR gain over four normal/weak vocal domains",
    "scope": "Development checkpoint selection only; not a production or blind-test gate",
}


def tensor_digest(*tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(str((array.shape, str(array.dtype))).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def scoring_slice(length, warmup):
    start, end = (warmup + 2) * core.HOP, length - 2 * core.HOP
    if end <= start:
        raise ValueError("No context/edge-safe scoring interval")
    return slice(start, end)


def guard_split(corpus):
    train = {core.composition_key(r["track_id"]) for pool in corpus.train.values() for r in pool}
    val = {core.composition_key(r["track_id"]) for pool in corpus.val.values() for r in pool}
    final = {core.composition_key(s) for s in corpus.final_ids}
    teachers = {core.composition_key(s) for s in corpus.teacher_meta["source_ids"]}
    if train & val or (train | val | teachers) & final or val & teachers:
        raise ValueError("Composition overlap in train/selection/final/teacher inventories")


def build_suite(corpus, length, warmup):
    guard_split(corpus)
    interval = scoring_slice(length, warmup)
    rows = []
    for domain in ("musdb", "mir1k", "instrumental"):
        for record in sorted(corpus.val[domain], key=lambda r: r["track_id"]):
            mix, vocal = corpus.audio(record)
            if mix.shape != vocal.shape or mix.shape[0] != 2 or mix.shape[-1] < length:
                raise ValueError("Selection source is too short or not aligned stereo")
            starts = [round((mix.shape[-1] - length) * fraction) for fraction in (.1, .5, .9)]
            if len(set(starts)) != 3:
                raise ValueError("Three distinct source windows are required")
            for start in starts:
                x, v = mix[:, start:start+length].clone(), vocal[:, start:start+length].clone()
                for variant, db in (("native", 0), ("weak_minus12", -12)):
                    if domain == "instrumental" and db:
                        continue
                    ref = v * 10 ** (db / 20)
                    remixed = x - v + ref
                    gain = min(1., .95 / max(float(remixed.abs().max()), 1e-12))
                    remixed, ref = remixed * gain, ref * gain
                    vocal_present = float(ref[..., interval].square().sum()) > 1e-10
                    rows.append({"domain": f"{domain}/{variant}", "track": record["track_id"],
                                 "start_sample": start, "vocal_gain_db": db,
                                 "common_peak_gain": gain, "vocal_present": vocal_present,
                                 "samples_sha256": tensor_digest(remixed, ref),
                                 "x": remixed, "v": ref})
    return rows


def suite_manifest(rows, length, warmup):
    descriptors = [{k: v for k, v in row.items() if k not in ("x", "v")} for row in rows]
    interval = scoring_slice(length, warmup)
    original_windows = {(r["domain"].split("/")[0], r["track"], r["start_sample"]) for r in rows}
    return {
        "schema": 1, "clips": len(rows), "tracks": len({r["track"] for r in rows}),
        "source_windows": len(original_windows),
        "scored_seconds_including_remixes": len(rows) * (interval.stop-interval.start) / core.SR,
        "source_window_scored_seconds_before_overlap": len(original_windows) * (interval.stop-interval.start) / core.SR,
        "crop_samples": length, "score_start_sample": interval.start, "score_end_sample": interval.stop,
        "sha256": hashlib.sha256(json.dumps(descriptors, sort_keys=True).encode()).hexdigest(),
        "rows": descriptors,
        "caveats": ["Internal development IDs may have been seen by frozen warm-start weights",
                    "Three windows can overlap; remixes are NOT independent songs",
                    "No OnAir selection IDs: only two train songs; its holdout is regression-only",
                    "Weak remixes are not a substitute for real electronic-vocal validation"],
    }


def separation_metrics(pred_vocal, mix, vocal):
    metrics = audit.waveform_metrics(pred_vocal, mix, vocal)
    ca, cv = metrics["joint_fit_accompaniment_gain"], metrics["joint_fit_residual_vocal_gain"]
    metrics["residual_vocal_abs_gain"] = abs(cv) if cv is not None else None
    metrics["accompaniment_gain_error_abs"] = abs(1-ca) if ca is not None else None
    return metrics


def aggregate(rows):
    summary = {}
    for domain in sorted({row["domain"] for row in rows}):
        group = [row for row in rows if row["domain"] == domain]
        summary[domain] = {"tracks": len({r["track"] for r in group}), "clips": len(group), "metrics": {}}
        for metric in group[0]["metrics"]:
            by_track = {}
            for row in group:
                value = row["metrics"][metric]
                if value is not None:
                    if not math.isfinite(value):
                        raise ValueError("Non-finite selection metric")
                    by_track.setdefault(row["track"], []).append(value)
            means = [statistics.fmean(values) for values in by_track.values()]
            summary[domain]["metrics"][metric] = {
                "n_tracks": len(means), "mean": statistics.fmean(means) if means else None,
                "by_track": {k: statistics.fmean(v) for k, v in sorted(by_track.items())},
            }
    return summary


@torch.no_grad()
def evaluate(net, data, wa, gs, device, warmup, kill):
    was_training = net.training
    net.eval()
    rows = []
    interval = scoring_slice(data[0]["x"].shape[-1], warmup)
    for row in data:
        x, v = row["x"][None].to(device), row["v"][None].to(device)
        spectrum, bands, _ = core.prepare_truth(x, v, wa)
        output = net(bands)
        pv = core.product_vocal(spectrum, (output[:, :2]+1)/2, gs, x.shape[-1], kill)
        if not torch.isfinite(pv).all():
            raise ValueError("Non-finite candidate waveform")
        # CPU double metrics avoid float cancellation and device-specific dot sums.
        metrics = separation_metrics(pv[0, ..., interval].cpu(), x[0, ..., interval].cpu(), v[0, ..., interval].cpu())
        rows.append({k: value for k, value in row.items() if k not in ("x", "v")} | {"metrics": metrics})
    net.train(was_training)
    return {"rows": rows, "summary": aggregate(rows)}


def assess(candidate, baseline):
    """No silent/missing-domain fallback; pure instrumental gains never rank."""
    reasons, deltas, gains, residue_changes = [], {}, [], []
    for domain in (*VOCAL_DOMAINS, "instrumental/native"):
        if domain not in candidate or domain not in baseline:
            return {"eligible": False, "rank_gain_db": None, "reasons": [f"Missing domain: {domain}"]}
        c, b = candidate[domain]["metrics"], baseline[domain]["metrics"]
        deltas[domain] = {}
        # Identical track coverage is essential for a paired selection score.
        checked = ("accompaniment_error_snr_db",) if domain == "instrumental/native" else (
            "vocal_error_snr_db", "accompaniment_error_snr_db",
            "joint_fit_accompaniment_gain", "residual_vocal_abs_gain")
        for metric in checked:
            if set(c[metric]["by_track"]) != set(b[metric]["by_track"]) or not c[metric]["by_track"]:
                reasons.append(f"Unpaired/unavailable scores: {domain}/{metric}")
        if any(c[m]["mean"] is None or b[m]["mean"] is None for m in checked):
            reasons.append(f"Missing usable reference metrics: {domain}")
            continue
        if domain == "instrumental/native":
            delta = c["accompaniment_error_snr_db"]["mean"]-b["accompaniment_error_snr_db"]["mean"]
            deltas[domain]["accompaniment_error_snr_db"] = delta
            if delta < -POLICY["instrumental_snr_max_regression_db"]:
                reasons.append("Pure instrumental false removal regressed")
            continue
        gain = c["vocal_error_snr_db"]["mean"]-b["vocal_error_snr_db"]["mean"]
        gains.append(gain)
        deltas[domain]["vocal_error_snr_db"] = gain
        if gain < -POLICY["domain_snr_max_regression_db"]:
            reasons.append(f"Vocal reconstruction regressed: {domain}")
        preservation = c["joint_fit_accompaniment_gain"]["mean"]-b["joint_fit_accompaniment_gain"]["mean"]
        deltas[domain]["joint_fit_accompaniment_gain"] = preservation
        if preservation < -POLICY["accompaniment_gain_max_drop"]:
            reasons.append(f"Projected accompaniment preservation regressed: {domain}")
        residue = 20 * math.log10(max(c["residual_vocal_abs_gain"]["mean"], 1e-12) /
                                 max(b["residual_vocal_abs_gain"]["mean"], 1e-12))
        residue_changes.append(residue)
        deltas[domain]["residual_vocal_abs_gain_db_change"] = residue
        if residue > POLICY["residual_vocal_max_increase_db"]:
            reasons.append(f"Projected remaining vocal increased: {domain}")
    rank = statistics.fmean(gains) if len(gains) == len(VOCAL_DOMAINS) else None
    residue_mean = statistics.fmean(residue_changes) if len(residue_changes) == len(VOCAL_DOMAINS) else None
    if rank is None or rank < POLICY["minimum_mean_vocal_snr_gain_db"]:
        reasons.append("Insufficient mean vocal reconstruction gain")
    if residue_mean is None or residue_mean > POLICY["mean_residual_vocal_max_increase_db"]:
        reasons.append("Mean projected remaining vocal did not decrease")
    return {"eligible": not reasons, "rank_gain_db": rank, "mean_residual_vocal_change_db": residue_mean,
            "reasons": reasons, "deltas": deltas}
