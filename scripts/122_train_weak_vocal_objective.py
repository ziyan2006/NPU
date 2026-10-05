"""Single-variable, bounded weak-vocal waveform weighting against script 120.

Keeps frozen legacy graph, LF250, magnitude band target, truth-only batches,
optimizer and budget. Reuses the audited 200-step legacy control, NEVER a
continued checkpoint. No final-test audio is used to select a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weak_objective_control", ROOT / "scripts/120_train_layout_control.py")
control = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = control
spec.loader.exec_module(control)
suite, core = control.suite, control.core
LOSS_POLICY = {
    "mixture_floor": .02, "raw_weight_min": .25, "raw_weight_max": 10.,
    "active_vocal_min_mean_power": 1e-10, "active_vocal_min_ratio": 1e-6,
    "rule": "r=clip((M+1e-6)/(V+0.02*M+1e-6),0.25,10); active-vocal weights=r/mean_active(r); inactive weights=1",
    "budget": "mean full effective batch weight equals 1; pure instrumental weight stays 1",
}


def load_control(folder):
    path = folder / "experiment.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    meta = report["metadata"]
    if report.get("status") != "complete" or not report.get("paired_batches_identical") or not report.get("deployed_checkpoint_unchanged"):
        raise ValueError("Matched control is not a completed, audited experiment")
    for name, value in {"120_train_layout_control.py": meta["script_sha256"],
                        "119_model_selection_suite.py": meta["suite_script_sha256"],
                        "110_train_residual_ablation.py": meta["core_script_sha256"], **meta["dependency_sha256"]}.items():
        if core.digest(ROOT / "scripts" / name) != value:
            raise ValueError(f"Control training source changed: {name}")
    if (meta["steps"], meta["seed"], meta["crop"], meta["warmup"], meta["lr"], meta["wave_weight"],
            meta["microbatch"], meta["effective_batch"], meta["validate_every"]) != (200, 20261001, 256, 96, 5e-6, 1., 1, 6, 50):
        raise ValueError("This probe is pinned to the exact 200-step control budget")
    if meta["training_target"] != "1.0 true band complement objective + wave_weight * mixture-normalized product-wave MSE; no teacher loss in EITHER arm":
        raise ValueError("Control objective is not the expected mixture-normalized objective")
    if meta["selection_policy"] != suite.POLICY:
        raise ValueError("Selection policy changed; no retroactive gate tuning")
    if core.digest(meta["checkpoint"]) != meta["sha256"]:
        raise ValueError("Frozen starting model changed")
    for filename, expected in (("selection_suite.json", meta["selection_suite_sha256"]),
                               ("frozen_selection_scores.json", meta["frozen_selection_scores_sha256"]),
                               ("legacy_log_final.pt", report["arms"]["legacy_log"]["final_sha256"])):
        if core.digest(folder / filename) != expected:
            raise ValueError(f"Control evidence changed: {filename}")
    if len(report["arms"]["legacy_log"]["batch_sha256"]) != 200:
        raise ValueError("Control lacks all 200 waveform batch fingerprints")
    return report


def per_example_wave_error(predicted, mix, vocal, warmup):
    interval = suite.scoring_slice(mix.shape[-1], warmup)
    error = (predicted[..., interval]-vocal[..., interval]).square().mean(dim=(-1, -2))
    mix_power = mix[..., interval].square().mean(dim=(-1, -2))
    return error/(mix_power+1e-6)


@torch.no_grad()
def vocal_weights(mix, vocal, warmup):
    interval = suite.scoring_slice(mix.shape[-1], warmup)
    m = mix[..., interval].square().mean(dim=(-1, -2))
    v = vocal[..., interval].square().mean(dim=(-1, -2))
    active = (v > LOSS_POLICY["active_vocal_min_mean_power"]) & (
        v/(m+1e-12) >= LOSS_POLICY["active_vocal_min_ratio"])
    raw = ((m+1e-6)/(v+LOSS_POLICY["mixture_floor"]*m+1e-6)).clamp(
        LOSS_POLICY["raw_weight_min"], LOSS_POLICY["raw_weight_max"])
    weights = torch.ones_like(raw)
    if active.any():
        weights[active] = raw[active]/raw[active].mean()
    if not torch.isfinite(weights).all() or not torch.allclose(weights.mean(), weights.new_tensor(1.), atol=1e-6):
        raise ValueError("Invalid sample weighting or changed effective loss budget")
    return weights, {"weights": weights.cpu().tolist(), "active": active.cpu().tolist(),
                     "vocal_to_mix_power_ratio": (v/(m+1e-12)).cpu().tolist()}


def backward_weighted_batch(model, mix, vocal, wa, gs, device, warmup, kill, weights, microbatch=1):
    if (mix.shape != vocal.shape or weights.shape != (mix.shape[0],) or microbatch < 1 or
            not torch.isfinite(weights).all() or (weights < 0).any() or
            any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout,
                               torch.nn.Dropout2d, torch.nn.Dropout3d)) for m in model.modules())):
        raise ValueError("Invalid batch-independent accumulation contract")
    count, band_value, wave_value = mix.shape[0], 0., 0.
    for start in range(0, count, microbatch):
        x, v = mix[start:start+microbatch].to(device), vocal[start:start+microbatch].to(device)
        spectrum, bands, target = core.prepare_truth(x, v, wa)
        output = model(bands)
        mv, ma = (output[:, :2]+1)/2, (output[:, 2:]+1)/2
        region = slice(warmup, None)
        band = core.objective("complement", mv[..., region], ma[..., region], target[..., region], bands[..., region])
        pv = core.product_vocal(spectrum, mv, gs, x.shape[-1], kill)
        wave = (per_example_wave_error(pv, x, v, warmup)*weights[start:start+microbatch].to(device)).mean()
        share = x.shape[0]/count
        loss = (band+wave)*share
        if not torch.isfinite(loss):
            raise ValueError("Non-finite weighted objective")
        loss.backward()
        band_value += float(band.detach())*share
        wave_value += float(wave.detach())*share
    return band_value+wave_value, band_value, wave_value


def save(path, net, initial, metadata, step, assessment):
    payload = {k: v for k, v in initial.items() if k != "model"}
    payload.update(model={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                   step=step, steps=200, lr=metadata["lr"], best_val=None,
                   mask_mode="complement", band_layout="legacy_log", n_bands=128,
                   finetune={**metadata, "arm": "weak_vocal", "step": step, "selection": assessment,
                             "initial_checkpoint_step": initial.get("step")})
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, default=ROOT / "results/layout_control_microbatch_20261001")
    ap.add_argument("--out", type=Path, default=ROOT / "results/weak_vocal_objective_20261001")
    args = ap.parse_args()
    parent = load_control(args.control)
    old = parent["metadata"]
    old_regression = args.control / "regression_evaluation.json"
    if not old_regression.is_file():
        raise ValueError("Audited control regression is required for matched evaluation")
    if args.out.exists():
        raise ValueError("Fresh output directory required; no old experiment is overwritten")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.__version__ != old["torch_version"] or device != old["device"]:
        raise ValueError("Control replay requires the same PyTorch version/device class")
    model, initial = core.t13.load_student(Path(old["checkpoint"]), device)
    if (model.band_layout != "legacy_log" or model.frontend != ("linear", 1.) or model.n_bands != 128 or
            initial.get("temporal_dilations") or initial.get("bottleneck_blocks") != 2):
        raise ValueError("Requires the unchanged frozen bott2 graph")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    corpus = core.Corpus(Path(old["manifest_provenance"]["musdb"]["root"]), Path(old["teacher"]["path"]), old["seed"])
    if corpus.manifests != old["manifest_provenance"] or corpus.teacher_meta != {k: v for k, v in old["teacher"].items() if k != "used_for_loss"}:
        raise ValueError("Control corpus provenance changed")
    length = (old["crop"]+old["warmup"]-1)*core.HOP
    data = suite.build_suite(corpus, length, old["warmup"])
    control.write_json(args.out / "selection_suite.json", suite.suite_manifest(data, length, old["warmup"]))
    if core.digest(args.out / "selection_suite.json") != old["selection_suite_sha256"]:
        raise ValueError("Expanded selection waveforms/positions changed")
    wa = torch.from_numpy(core.t09.make_analysis_matrix(128, layout="legacy_log")).to(device)
    gs = torch.from_numpy(core.t09.make_synthesis_matrix(128, layout="legacy_log")).to(device)
    kill = core.t11.lf_kill_band_for(250., "legacy_log", 128)
    baseline = suite.evaluate(model, data, wa, gs, device, old["warmup"], kill)
    control.write_json(args.out / "frozen_selection_scores.json", baseline)
    # Same frozen graph + same input + deterministic execution must reproduce.
    if core.digest(args.out / "frozen_selection_scores.json") != old["frozen_selection_scores_sha256"]:
        raise ValueError("Frozen development scores differ from audited control")
    metadata = copy.deepcopy(old)
    metadata.update(script_sha256=core.digest(__file__), training_script="122_train_weak_vocal_objective.py",
                    control_directory=str(args.control.resolve()), control_experiment_sha256=core.digest(args.control / "experiment.json"),
                    control_regression_sha256=core.digest(old_regression), control_training_script_sha256=core.digest(control.__file__),
                    control_final_sha256=parent["arms"]["legacy_log"]["final_sha256"],
                    objective_policy=LOSS_POLICY, objective_arm="weak_vocal", band_layout="legacy_log",
                    training_target="Unchanged complement magnitude-band loss + bounded active-vocal-energy weighting of per-example mixture-normalized product-wave MSE",
                    comparison="Replayed exact 200-step waveform batches from original frozen weights; prior legacy_log arm is the immutable matched control",
                    caveat="Same previously inspected development/regression data; NOT fresh blind evidence. No layout, LF, target, data or graph change.")
    report = {"status": "running", "metadata": metadata, "arms": {}, "curve": [], "batch_weights": []}
    control.write_json(args.out / "experiment.json", report)
    generator = torch.Generator().manual_seed(old["seed"])
    torch.manual_seed(old["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=old["lr"])
    best_rank, best_step, batch_hashes = -math.inf, None, []
    frozen_assessment = suite.assess(baseline["summary"], baseline["summary"])
    save(args.out / "weak_vocal_initial.pt", model, initial, metadata, 0, frozen_assessment)
    report["curve"].append({"step": 0, "assessment": frozen_assessment, "summary": baseline["summary"]})
    for step in range(1, 201):
        model.train()
        lr = old["lr"]*min(step/20, 1.)*(.2+.8*(1+math.cos(math.pi*step/200))/2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, v = corpus.batch(generator, length)
        batch_hashes.append(suite.tensor_digest(x, v))
        if batch_hashes[-1] != parent["arms"]["legacy_log"]["batch_sha256"][step-1]:
            raise ValueError(f"Control waveform batch differs at step {step}")
        weights, stats = vocal_weights(x, v, old["warmup"])
        report["batch_weights"].append({"step": step, **stats})
        optimizer.zero_grad(set_to_none=True)
        loss, band, wave = backward_weighted_batch(model, x, v, wa, gs, device, old["warmup"], kill, weights, old["microbatch"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 10 == 0:
            print(f"WEAK_VOCAL step={step}/200 loss={loss:.5f} band={band:.5f} wave={wave:.5f}", flush=True)
        if step % old["validate_every"] == 0:
            scores = suite.evaluate(model, data, wa, gs, device, old["warmup"], kill)
            assessment = suite.assess(scores["summary"], baseline["summary"])
            previous_point = next(row for row in parent["arms"]["legacy_log"]["curve"] if row["step"] == step)
            vs_control = suite.assess(scores["summary"], previous_point["summary"])
            control.write_json(args.out / f"weak_vocal_step{step:04d}_selection.json", scores)
            report["curve"].append({"step": step, "assessment": assessment, "vs_same_step_control": vs_control,
                                    "summary": scores["summary"], "loss": loss, "band": band, "wave": wave,
                                    "wall_s": time.perf_counter()-started})
            if assessment["eligible"] and assessment["rank_gain_db"] > best_rank:
                best_rank, best_step = assessment["rank_gain_db"], step
                save(args.out / "weak_vocal_selected.pt", model, initial, metadata, step, assessment)
            print(f"SELECT_WEAK step={step} eligible={assessment['eligible']} rank={assessment['rank_gain_db']} "
                  f"residue_db={assessment['mean_residual_vocal_change_db']} best_step={best_step}", flush=True)
            control.write_json(args.out / "experiment.json", report)
    save(args.out / "weak_vocal_final.pt", model, initial, metadata, 200, assessment)
    report["arms"]["weak_vocal"] = {
        "best_step": best_step, "best_rank_gain_db": best_rank if best_step is not None else None,
        "selected_sha256": core.digest(args.out / "weak_vocal_selected.pt") if best_step is not None else None,
        "final_sha256": core.digest(args.out / "weak_vocal_final.pt"), "batch_sha256": batch_hashes,
        "parameters_changed": any(not torch.equal(v.cpu(), initial["model"][k].cpu()) for k, v in model.state_dict().items()),
    }
    report["paired_batches_identical"] = batch_hashes == parent["arms"]["legacy_log"]["batch_sha256"]
    report["deployed_checkpoint_unchanged"] = core.digest(old["checkpoint"]) == old["sha256"]
    report["control_unchanged"] = core.digest(args.control / "experiment.json") == metadata["control_experiment_sha256"] and (
        core.digest(args.control / "legacy_log_final.pt") == metadata["control_final_sha256"])
    if not (report["paired_batches_identical"] and report["deployed_checkpoint_unchanged"] and
            report["control_unchanged"] and report["arms"]["weak_vocal"]["parameters_changed"]):
        raise ValueError("Training/frozen/control integrity check failed")
    report.update(status="complete", total_wall_s=time.perf_counter()-started)
    control.write_json(args.out / "experiment.json", report)
    print(f"WEAK_VOCAL_PROBE PASS out={args.out}; deployment remains frozen", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
