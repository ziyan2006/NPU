"""Regression-only evaluation of matched FINAL-step layout-control models.

Reuses the known 143-clip/78-song suite, plus weak-vocal Cambridge remixes.
Neither regression scores nor bootstrap intervals select a training checkpoint.
No candidate is deployed and no SD media is touched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("layout_regression_suite", ROOT / "scripts/119_model_selection_suite.py")
suite = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = suite
spec.loader.exec_module(suite)
audit, core = suite.audit, suite.core
helpers = suite.load("layout_regression_helpers", "111_evaluate_residual_ablation.py")


@torch.no_grad()
def bounded_masks(net, bands):
    """Same convolution-only bott2 inference, preserving stride phase/history.

    128 history frames cover this pinned graph's receptive field. Keep all
    starts divisible by eight; pad once at the global end. This is not the
    old 16-frame-reset board inference and introduces no blockwise smoothing.
    """
    count = bands.shape[-1]
    padded = F.pad(bands, (0, (-count) % 8))
    outputs = []
    for start in range(0, padded.shape[-1], 256):
        left, right = max(0, start-128), min(start+256, padded.shape[-1])
        predicted = audit.masks(net, padded[..., left:right])
        outputs.append(predicted[..., start-left:])
    return torch.cat(outputs, dim=-1)[..., :count]


def paired_intervals(rows, labels):
    result = {}
    rng = np.random.default_rng(20261001)
    for dataset in sorted({row["dataset"] for row in rows}):
        result[dataset] = {}
        group = [row for row in rows if row["dataset"] == dataset]
        for label in labels:
            result[dataset][label] = {}
            for metric in ("vocal_error_snr_db", "vocal_si_sdr_db", "accompaniment_si_sdr_db",
                           "accompaniment_error_snr_db", "residual_vocal_abs_gain", "joint_fit_accompaniment_gain"):
                by_track = {}
                for row in group:
                    if (metric.startswith("vocal_") or metric == "residual_vocal_abs_gain") and not row["vocals_present"]:
                        continue
                    a, b = row["models"][label][metric], row["models"]["baseline"][metric]
                    if a is not None and b is not None:
                        if not math.isfinite(a-b):
                            raise ValueError("Invalid paired regression metric")
                        by_track.setdefault(row["track"], []).append(a-b)
                means = np.asarray([statistics.fmean(v) for _, v in sorted(by_track.items())])
                # 1-4 songs cannot support useful song-population interval claims.
                ci = (np.percentile(rng.choice(means, (10000, len(means)), replace=True).mean(1), [2.5, 97.5]).tolist()
                      if len(means) >= 5 else None)
                result[dataset][label][metric] = {
                    "n_tracks": len(means), "mean": float(means.mean()) if len(means) else None,
                    "bootstrap_95pct": ci,
                    "by_track": {k: statistics.fmean(v) for k, v in sorted(by_track.items())},
                }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=ROOT / "results/layout_control_microbatch_20261001")
    ap.add_argument("--previous", type=Path, default=ROOT / "results/mel_unique_finetune_cont_20261001/holdout_evaluation.json")
    ap.add_argument("--musdb-root", type=Path, default=ROOT / "data/datasets/MUSDB18-7-STEMS")
    args = ap.parse_args()
    path = args.experiment / "experiment.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    meta = report["metadata"]
    if report.get("status") != "complete" or not report.get("paired_batches_identical") or not report.get("deployed_checkpoint_unchanged"):
        raise ValueError("Training has not completed its paired/frozen integrity checks")
    source_checks = {"120_train_layout_control.py": meta["script_sha256"],
                     "119_model_selection_suite.py": meta["suite_script_sha256"],
                     "110_train_residual_ablation.py": meta["core_script_sha256"], **meta["dependency_sha256"]}
    for name, expected in source_checks.items():
        if core.digest(ROOT / "scripts" / name) != expected:
            raise ValueError(f"Training source changed: {name}")
    for name, expected in (("selection_suite.json", meta["selection_suite_sha256"]),
                           ("frozen_selection_scores.json", meta["frozen_selection_scores_sha256"])):
        if core.digest(args.experiment / name) != expected:
            raise ValueError(f"Recorded selection evidence changed: {name}")
    output = args.experiment / "regression_evaluation.json"
    if output.exists():
        raise ValueError("Preserve previous evaluation; output already exists")
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    if previous["models"]["baseline"]["sha256"] != meta["sha256"]:
        raise ValueError("Known regression suite used a different frozen baseline")
    old_sources = {row["dataset"]: row for row in previous["sources"]}
    previous_keys = {(r["dataset"], r["track"], round(r["start_s"], 6)) for r in previous["rows"]}
    previous_baseline = {(r["dataset"], r["track"], round(r["start_s"], 6)): r["models"]["baseline"]
                         for r in previous["rows"]}
    if len(previous_keys) != 143:
        raise ValueError("Known regression inventory is not the expected 143 clips")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = {"baseline": Path(meta["checkpoint"]),
             **{layout: args.experiment / f"{layout}_final.pt" for layout in report["arms"]}}
    nets, banks, model_meta = {}, {}, {}
    for label, checkpoint in paths.items():
        expected = meta["sha256"] if label == "baseline" else report["arms"][label]["final_sha256"]
        if core.digest(checkpoint) != expected:
            raise ValueError(f"Checkpoint hash changed: {label}")
        net, blob = core.t13.load_student(checkpoint, device)
        layout = "legacy_log" if label == "baseline" else label
        if net.band_layout != layout or net.frontend != ("linear", 1.) or net.n_bands != 128:
            raise ValueError("Mismatched candidate representation")
        if label != "baseline" and blob["step"] != meta["steps"]:
            raise ValueError("Final comparison must use the identical step budget")
        nets[label] = net
        wa = core.t09.make_analysis_matrix(128, layout=layout)
        gs = core.t09.make_synthesis_matrix(128, layout=layout)
        if (__import__("hashlib").sha256(wa.tobytes()).hexdigest() != meta["filterbanks"][layout]["analysis_sha256"] or
                __import__("hashlib").sha256(gs.tobytes()).hexdigest() != meta["filterbanks"][layout]["synthesis_sha256"]):
            raise ValueError("Inference filterbank differs from training")
        banks[label] = (torch.from_numpy(wa).to(device), torch.from_numpy(gs).to(device),
                        core.t11.lf_kill_band_for(250., layout, 128))
        model_meta[label] = {"path": str(checkpoint), "sha256": expected, "step": blob["step"],
                             "band_layout": layout, "internally_selected": label != "baseline" and
                             report["arms"][label]["best_step"] == meta["steps"],
                             "purpose": "Equal-budget FINAL-step diagnostic; NOT automatic deployment"}
    documents = []
    for folder in ("mir1k_gainmix_cache", "onair_true_cache", "mshoxx_stereo_instrumental_cache"):
        manifest_path = ROOT / "results" / folder / "manifest.json"
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        domain = "instrumental" if document["dataset"] == "mshoxx" else document["dataset"]
        if (core.digest(manifest_path) != meta["manifest_provenance"][domain]["sha256"] or
                core.digest(manifest_path) != old_sources[document["dataset"]]["sha256"]):
            raise ValueError("Training/regression manifest changed")
        documents.append(document)
    cambridge = audit.cambridge_manifest(ROOT / "data/datasets/CambridgeMTK-electronic-excerpts")
    if cambridge["source_sha256"] != old_sources["cambridge_multitrack"]["source_sha256"]:
        raise ValueError("Cambridge recordings changed since previous regression")
    documents.append(cambridge)
    proxy_mix = ROOT / "results/demons_mir1k_candidate_lf/01_mixture.wav"
    proxy_vocal = ROOT / "results/demons_mir1k_candidate_lf/04_teacher_vocals.wav"
    if (core.digest(proxy_mix) != previous["teacher_proxy"]["mixture_sha256"] or
            core.digest(proxy_vocal) != previous["teacher_proxy"]["teacher_sha256"]):
        raise ValueError("Teacher proxy recording changed")
    documents.append({"dataset": "dj_teacher_proxy", "tracks": [{
        "track_id": "Jerro, Sophia Bel - Demons.mp3", "dataset": "dj_teacher_proxy", "split": "holdout",
        "mix_files": [str(proxy_mix)], "vocal_files": [str(proxy_vocal)]}]})
    musdb = sorted((args.musdb_root / "test").glob("*.stem.mp4"))
    if len(musdb) != 50 or {str(p): core.digest(p) for p in musdb} != old_sources["musdb"]["source_sha256"]:
        raise ValueError("MUSDB regression inventory/audio changed")
    documents.append({"dataset": "musdb", "tracks": [{"track_id": p.stem, "dataset": "musdb", "split": "holdout",
                      "mix_files": [str(p)], "vocal_files": []} for p in musdb]})
    rows, seen, baseline_max_delta = [], set(), 0.
    with torch.no_grad():
        for document in documents:
            for record in document["tracks"]:
                if record["split"] != "holdout":
                    continue
                helpers.reject_overlap(meta, document["dataset"], record["track_id"])
                if record["dataset"] == "musdb":
                    mix, vocal = core.decode_musdb(Path(record["mix_files"][0]))
                elif record["dataset"] == "dj_teacher_proxy":
                    mix, vocal = [torch.from_numpy(core.t23._stereo(core.t23._read(Path(p))[0]))
                                  for p in (record["mix_files"][0], record["vocal_files"][0])]
                elif record["dataset"] == "cambridge_multitrack":
                    read = lambda paths: [torch.from_numpy(core.t23._stereo(core.t23._read(Path(p))[0])) for p in paths]
                    mix, vocal = audit.mix_multitrack(read(record["vocal_files"]), read(record["stem_files"]))
                else:
                    spec = core.t23.TrackSpec(record["track_id"], record["dataset"], record["mix_files"],
                                             record["vocal_files"], record.get("stem_files", []))
                    mix, vocal = core.t23.load_track(spec)
                if mix.shape != vocal.shape:
                    raise ValueError("Reference/mix sample alignment differs")
                for start in range(0, mix.shape[-1], 30*core.SR):
                    end = min(start+30*core.SR, mix.shape[-1])
                    if end-start < 5*core.SR:
                        continue
                    key = (document["dataset"], record["track_id"], round(start/core.SR, 6))
                    if key not in previous_keys or key in seen:
                        raise ValueError("Regression clip segmentation differs")
                    seen.add(key)
                    for db in ((0, -12) if record["dataset"] == "cambridge_multitrack" else (0,)):
                        v = vocal[:, start:end]*10**(db/20)
                        x = mix[:, start:end] if not db else mix[:, start:end]-vocal[:, start:end]+v
                        x, v = x.to(device), v.to(device)
                        spectrum = core.t09._stft(x)
                        vrms, xrms = v.square().mean().sqrt(), x.square().mean().sqrt()
                        present = bool(vrms > 1e-5 and 20*torch.log10((vrms+1e-12)/(xrms+1e-12)) >= (-45 if db else -30))
                        metrics = {}
                        for label, net in nets.items():
                            wa, gs, kill = banks[label]
                            masks = bounded_masks(net, torch.einsum("fb,cft->cbt", wa, spectrum.abs()))
                            prediction = audit.render(spectrum, masks[:2], gs, x.shape[-1], kill)
                            metrics[label] = suite.separation_metrics(prediction.cpu(), x.cpu(), v.cpu())
                        if not db:
                            for metric in ("vocal_si_sdr_db", "accompaniment_si_sdr_db",
                                           "vocal_error_snr_db", "accompaniment_error_snr_db"):
                                old, new = previous_baseline[key][metric], metrics["baseline"][metric]
                                if (old is None) != (new is None):
                                    raise ValueError("Frozen regression score availability changed")
                                if old is not None:
                                    baseline_max_delta = max(baseline_max_delta, abs(old-new))
                            if baseline_max_delta > .005:
                                raise ValueError(f"Frozen regression metrics differ from known continuous inference: {baseline_max_delta} dB")
                        rows.append({"dataset": document["dataset"]+ ("_weak_minus12" if db else ""),
                                     "track": record["track_id"], "start_s": start/core.SR, "seconds": (end-start)/core.SR,
                                     "vocal_gain_db": db, "vocals_present": present, "models": metrics})
                print(f"REGRESSION dataset={document['dataset']} track={record['track_id']} clips={len(seen)}/143", flush=True)
    if seen != previous_keys:
        raise ValueError("Incomplete known regression coverage")
    summary = helpers.grouped_summary(rows, list(nets))
    payload = {"schema": 1, "script_sha256": core.digest(__file__), "experiment_sha256": core.digest(path),
               "previous_suite_sha256": core.digest(args.previous), "models": model_meta, "rows": rows,
               "baseline_vs_known_continuous_max_metric_delta_db": baseline_max_delta,
               "summary": summary, "paired_intervals": paired_intervals(rows, list(report["arms"])),
               "provisional_float_gate": {label: helpers.provisional_gate(rows, summary, label) for label in report["arms"]},
               "sources": previous["sources"], "teacher_proxy": previous["teacher_proxy"],
               "mask_execution": "Convolution-only bott2, 256-frame blocks with 128-frame overlapping causal input history, stride phase preserved; STFT/iSTFT remain continuous",
               "scope": "143 known regression clips plus Cambridge weak-vocal remixes; never used for checkpoint selection",
               "caveats": ["Repeatedly inspected development/regression suite, NOT new blind evidence",
                           "Bootstrap samples whole song-level paired differences; intervals omitted for fewer than five songs",
                           "Descriptive intervals do not correct adaptive reuse, source correlations or warm-start exposure",
                           "Two Cambridge songs and remixes cannot establish general EDM performance",
                           "Final-step diagnostics may be internally INELIGIBLE; no weight/SD deployment"]}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"LAYOUT_REGRESSION PASS rows={len(rows)} known_clips={len(seen)} output={output}", flush=True)


if __name__ == "__main__":
    main()
