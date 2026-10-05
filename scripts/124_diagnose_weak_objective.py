"""Training-only error attribution and scalar mask-gain loss sensitivity.

Joint source projection is an algebraic diagnostic, not an isolated source or
perceptual metric. A scalar gain derivative is one local direction, NOT the
gradient of all model parameters or a proposed production gain setting.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weak_attribution_training", ROOT / "scripts/122_train_weak_vocal_objective.py")
training = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = training
spec.loader.exec_module(training)
suite, core = training.suite, training.core


def decompose(predicted, mix, vocal):
    """Exact energy identity including correlated-source cross term."""
    p, x, v = [value.detach().cpu().double().flatten() for value in (predicted, mix, vocal)]
    a, y = x-v, x-p
    aa, vv, av = a@a, v@v, a@v
    error_energy = (p-v).square().sum()
    denominator = aa*vv-av*av
    if vv <= 1e-10 or denominator <= 1e-8*aa*vv:
        return None
    ca = ((y@a)*vv-(y@v)*av)/denominator
    cv = ((y@v)*aa-(y@a)*av)/denominator
    fit_residual = y-ca*a-cv*v
    terms = {"projected_accompaniment_damage": (ca-1).square()*aa,
             "projected_remaining_vocal": cv.square()*vv,
             "correlated_source_cross_term": 2*(ca-1)*cv*av,
             "unmodeled_fit_residual": fit_residual.square().sum()}
    closure = abs(float(sum(terms.values())-error_energy))/max(float(error_energy), 1e-12)
    if closure > 1e-8:
        raise ValueError("Joint projection energy decomposition does not close")
    return {"joint_accompaniment_gain": float(ca), "joint_remaining_vocal_gain": float(cv),
            "error_energy": float(error_energy), "relative_closure_error": closure,
            "error_energy_fractions": {name: float(value/error_energy.clamp_min(1e-12)) for name, value in terms.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=ROOT / "results/weak_vocal_objective_20261001")
    ap.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()
    if not 1 <= args.batches <= 200:
        ap.error("batches must be 1..200")
    report_path = args.experiment / "experiment.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or not report.get("paired_batches_identical"):
        raise ValueError("Requires completed weak-objective training")
    meta = report["metadata"]
    if core.digest(training.__file__) != meta["script_sha256"]:
        raise ValueError("Training implementation changed")
    parent = training.load_control(Path(meta["control_directory"]))
    output = args.experiment / "training_loss_attribution.json"
    if output.exists():
        raise ValueError("Preserve previous diagnostic")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, blob = core.t13.load_student(Path(meta["checkpoint"]), device)
    if net.band_layout != "legacy_log" or net.frontend != ("linear", 1.) or blob.get("temporal_dilations"):
        raise ValueError("Pinned linear convolution graph required")
    corpus = core.Corpus(Path(meta["manifest_provenance"]["musdb"]["root"]), Path(meta["teacher"]["path"]), meta["seed"])
    suite.guard_split(corpus)
    length = (meta["crop"]+meta["warmup"]-1)*core.HOP
    interval = suite.scoring_slice(length, meta["warmup"])
    wa = torch.from_numpy(core.t09.make_analysis_matrix(128, layout="legacy_log")).to(device)
    gs = torch.from_numpy(core.t09.make_synthesis_matrix(128, layout="legacy_log")).to(device)
    kill = core.t11.lf_kill_band_for(250., "legacy_log", 128)
    generator = torch.Generator().manual_seed(meta["seed"])
    rows = []
    for batch in range(args.batches):
        x, v = corpus.batch(generator, length)
        fingerprint = suite.tensor_digest(x, v)
        if fingerprint != parent["arms"]["legacy_log"]["batch_sha256"][batch]:
            raise ValueError("Training attribution did not replay the same source batch")
        weights, stats = training.vocal_weights(x, v, meta["warmup"])
        for item, domain in enumerate(meta["batch_domains"]):
            xb, vb = x[item:item+1].to(device), v[item:item+1].to(device)
            with torch.no_grad():
                spectrum, bands, target = core.prepare_truth(xb, vb, wa)
                masks = (net(bands)[:, :2]+1)/2
            g = torch.ones((), device=device, requires_grad=True)
            scaled = (masks*g).clamp(0, 1)
            band = core.objective("complement", scaled[..., meta["warmup"]:], 1-scaled[..., meta["warmup"]:],
                                  target[..., meta["warmup"]:], bands[..., meta["warmup"]:])
            prediction = core.product_vocal(spectrum, scaled, gs, length, kill)
            wave = core.waveform_loss(prediction, xb, vb, meta["warmup"])
            band_grad = float(torch.autograd.grad(band, g, retain_graph=True)[0])
            wave_grad = float(torch.autograd.grad(wave, g)[0])
            ratio = stats["vocal_to_mix_power_ratio"][item]
            bucket = "instrumental_or_inactive" if not stats["active"][item] else ("weak_ratio_le_0.1" if ratio <= .1 else "normal_ratio_gt_0.1")
            rows.append({"batch": batch+1, "item": item, "domain": domain, "bucket": bucket,
                         "waveform_sha256": suite.tensor_digest(x[item], v[item]),
                         "vocal_to_mix_power_ratio": ratio, "sample_weight": float(weights[item]),
                         "band_loss": float(band.detach()), "wave_loss": float(wave.detach()),
                         "d_band_d_mask_gain": band_grad, "d_wave_d_mask_gain": wave_grad,
                         "d_control_total_d_mask_gain": band_grad+wave_grad,
                         "d_weighted_total_d_mask_gain": band_grad+float(weights[item])*wave_grad,
                         "decomposition": decompose(prediction[0, ..., interval], xb[0, ..., interval], vb[0, ..., interval])})
        print(f"WEAK_ATTRIBUTION batches={batch+1}/{args.batches}", flush=True)
    summary = {}
    for bucket in sorted({row["bucket"] for row in rows}):
        group = [row for row in rows if row["bucket"] == bucket]
        fits = [row["decomposition"] for row in group if row["decomposition"] is not None]
        summary[bucket] = {
            "crops": len(group), "identifiable_fits": len(fits),
            "mean_sample_weight": statistics.fmean(row["sample_weight"] for row in group),
            "control_crops_loss_locally_prefers_less_mask": sum(row["d_control_total_d_mask_gain"] > 0 for row in group),
            "weighted_crops_loss_locally_prefers_less_mask": sum(row["d_weighted_total_d_mask_gain"] > 0 for row in group),
            "wave_crops_locally_prefers_less_mask": sum(row["d_wave_d_mask_gain"] > 0 for row in group),
            "mean_projected_error_fractions": {name: statistics.fmean(fit["error_energy_fractions"][name] for fit in fits)
                                                for name in fits[0]["error_energy_fractions"]} if fits else None,
        }
    payload = {"schema": 1, "script_sha256": core.digest(__file__), "training_script_sha256": meta["script_sha256"],
               "experiment_sha256": core.digest(report_path), "checkpoint_sha256": meta["sha256"],
               "batches": args.batches, "rows": rows, "summary": summary,
               "scope": "Frozen weights; first matched TRAINING batches only, no selection or regression audio",
               "derivative_interpretation": "Positive dL/dg locally favors lowering ALL vocal masks; negative favors increasing. One scalar direction, not the full parameter gradient.",
               "caveats": ["Means are over sampled training crops, not song-level benchmark scores",
                           "Correlated-source cross term is signed; other fractions need not independently sum to 100 percent",
                           "Projected residual is not an independently isolated perceptual vocal signal",
                           "No loss/target modification, parameter update, SD or board write"]}
    training.control.write_json(output, payload)
    if core.digest(meta["checkpoint"]) != meta["sha256"]:
        raise ValueError("Frozen weights changed")
    print(f"WEAK_ATTRIBUTION PASS crops={len(rows)} output={output}", flush=True)


if __name__ == "__main__":
    main()
