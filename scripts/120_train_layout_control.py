"""Matched legacy_log/mel_unique warm-start experiment with expanded selection.

Both arms use true stems ONLY, identical random audio batches, optimizer, loss,
LR schedule, initial weights and nominal LF250 policy. No hardware/SD writes.
This isolates the layout + its matched reconstruction/protection mapping, not
teacher removal, convergence from scratch, or a deployable quality upgrade.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("layout_control_selection", ROOT / "scripts/119_model_selection_suite.py")
suite = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = suite
spec.loader.exec_module(suite)
core = suite.core


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save(path, model, initial, metadata, layout, step, assessment):
    payload = {k: value for k, value in initial.items() if k != "model"}
    payload.update(model={k: value.detach().cpu().clone() for k, value in model.state_dict().items()},
                   step=step, steps=metadata["steps"], lr=metadata["lr"], best_val=None,
                   mask_mode="complement", band_layout=layout, n_bands=128,
                   finetune={**metadata, "arm": layout, "step": step, "selection": assessment,
                             "initial_checkpoint_step": initial.get("step")})
    torch.save(payload, path)


def backward_batch(model, x, v, wa, gs, device, warmup, kill, wave_weight, microbatch):
    """Mean per-example losses: exact effective batch, one caller Adam update.

    This graph has neither batch statistics nor dropout; a microbatch is ONLY
    a memory/execution choice. Numerical accumulation ordering can differ.
    """
    if microbatch < 1 or any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,
                                         torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d))
                             for m in model.modules()):
        raise ValueError("Microbatch contract requires a deterministic, batch-independent graph")
    band_value, wave_value = 0., 0.
    count = x.shape[0]
    for start in range(0, count, microbatch):
        xb, vb = x[start:start+microbatch].to(device), v[start:start+microbatch].to(device)
        spectrum, bands, target = core.prepare_truth(xb, vb, wa)
        output = model(bands)
        mv, ma = (output[:, :2]+1)/2, (output[:, 2:]+1)/2
        score_region = slice(warmup, None)
        band = core.objective("complement", mv[..., score_region], ma[..., score_region],
                              target[..., score_region], bands[..., score_region])
        pv = core.product_vocal(spectrum, mv, gs, xb.shape[-1], kill)
        wave = core.waveform_loss(pv, xb, vb, warmup)
        weight = xb.shape[0]/count
        loss = (band+wave_weight*wave)*weight
        if not torch.isfinite(loss):
            raise ValueError("Non-finite matched training loss")
        loss.backward()
        band_value += float(band.detach())*weight
        wave_value += float(wave.detach())*weight
    return band_value+wave_weight*wave_value, band_value, wave_value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/student_bott2_mir1k_candidate.pt")
    ap.add_argument("--musdb-root", type=Path, default=ROOT / "data/datasets/MUSDB18-7-STEMS")
    ap.add_argument("--teacher-cache", type=Path, default=ROOT / "results/opt_bott2_5m_cache")
    ap.add_argument("--out", type=Path, default=ROOT / "results/layout_control_microbatch_20261001")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--wave-weight", type=float, default=1.)
    ap.add_argument("--validate-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--microbatch", type=int, default=1)
    args = ap.parse_args()
    if (args.steps < 1 or args.crop < 32 or args.crop % 8 or args.warmup < 96 or args.warmup % 8 or
            args.validate_every < 1 or args.log_every < 1 or args.microbatch < 1 or not math.isfinite(args.lr) or args.lr <= 0 or
            not math.isfinite(args.wave_weight) or args.wave_weight < 0):
        ap.error("Invalid training budget/context/rate/loss settings")
    if args.out.exists():
        raise ValueError("Use a fresh output directory; no historical candidates are overwritten")
    original_hash = core.digest(args.checkpoint)
    if original_hash != "cb16d333c372d9558ef563b9b4baa9b9f7f46f0ec72d5b71f9e26557a1abfb55":
        raise ValueError("This comparison is pinned to the frozen deployed starting checkpoint")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, initial = core.t13.load_student(args.checkpoint, device)
    if (initial.get("bottleneck_blocks") != 2 or initial.get("temporal_dilations") or
            model.frontend != ("linear", 1.) or model.n_bands != 128 or model.band_layout != "legacy_log"):
        raise ValueError("Requires the frozen convolution-only, linear 128-band graph")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    corpus = core.Corpus(args.musdb_root, args.teacher_cache, args.seed)
    length = (args.crop + args.warmup - 1) * core.HOP
    validation = suite.build_suite(corpus, length, args.warmup)
    manifest = suite.suite_manifest(validation, length, args.warmup)
    write_json(args.out / "selection_suite.json", manifest)
    banks, bank_metadata = {}, {}
    for layout in ("legacy_log", "mel_unique"):
        arrays = [fn(128, layout=layout) for fn in (core.t09.make_analysis_matrix, core.t09.make_synthesis_matrix)]
        kill = core.t11.lf_kill_band_for(250., layout, 128)
        banks[layout] = (*[torch.from_numpy(array).to(device) for array in arrays], kill)
        bank_metadata[layout] = {"analysis_sha256": hashlib.sha256(arrays[0].tobytes()).hexdigest(),
                                 "synthesis_sha256": hashlib.sha256(arrays[1].tobytes()).hexdigest(),
                                 "lf_hz": 250., "kill_bands": kill}
    metadata = {
        "checkpoint": str(args.checkpoint), "sha256": original_hash,
        "script_sha256": core.digest(__file__), "suite_script_sha256": core.digest(suite.__file__),
        "core_script_sha256": core.digest(core.__file__),
        "dependency_sha256": {name: core.digest(ROOT / "scripts" / name) for name in (
            "09_target_model.py", "11_smoke_train.py", "13_ab_compare.py",
            "23_build_true_stem_cache.py", "109_audit_model_limits.py")},
        "steps": args.steps, "seed": args.seed, "lr": args.lr, "wave_weight": args.wave_weight,
        "crop": args.crop, "warmup": args.warmup, "validate_every": args.validate_every,
        "microbatch": args.microbatch, "effective_batch": 6,
        "accumulation": "Sample-weighted loss/gradient sum; one clipping and Adam update per six-example batch; no batch normalization/dropout",
        "selection_suite_sha256": core.digest(args.out / "selection_suite.json"),
        "selection_policy": suite.POLICY, "filterbanks": bank_metadata,
        "batch_domains": ["musdb", "musdb", "mir1k", "mir1k", "onair", "instrumental"],
        "training_target": "1.0 true band complement objective + wave_weight * mixture-normalized product-wave MSE; no teacher loss in EITHER arm",
        "teacher": {**corpus.teacher_meta, "used_for_loss": False},
        "manifest_provenance": corpus.manifests, "excluded_before_split": corpus.excluded,
        "final_holdout_ids": corpus.final_ids,
        "splits": {d: {"train": [r["track_id"] for r in corpus.train[d]],
                       "validation": [r["track_id"] for r in corpus.val.get(d, [])]} for d in corpus.train},
        "initialization": "Identical frozen legacy weights, no remapping, fresh Adam in each arm",
        "caveat": "Warm-start representation-adaptation probe, NOT a from-scratch architectural comparison. Nominal LF250 same; its quantized band mapping differs. Selection pool is development only.",
        "torch_version": torch.__version__, "device": device,
        "determinism": "deterministic algorithms; CUBLAS_WORKSPACE_CONFIG=:4096:8; TF32 disabled",
    }
    report = {"metadata": metadata, "status": "running", "arms": {}}
    write_json(args.out / "experiment.json", report)
    baseline = suite.evaluate(model, validation, *banks["legacy_log"][:2], device, args.warmup, banks["legacy_log"][2])
    write_json(args.out / "frozen_selection_scores.json", baseline)
    metadata["frozen_selection_scores_sha256"] = core.digest(args.out / "frozen_selection_scores.json")
    print(f"CONTROL suite_tracks={manifest['tracks']} clips={manifest['clips']} "
          f"scored_s={manifest['scored_seconds_including_remixes']:.2f}", flush=True)
    reference_batches = None
    for layout in ("legacy_log", "mel_unique"):
        wa, gs, kill = banks[layout]
        model.load_state_dict(initial["model"])
        model.band_layout = layout
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        arm_started = time.perf_counter()
        best_rank, best_step = -math.inf, None
        curves, batch_hashes = [], []
        for step in range(args.steps+1):
            if step:
                model.train()
                lr = args.lr * min(step/20, 1.) * (.2+.8*(1+math.cos(math.pi*step/args.steps))/2)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                x, v = corpus.batch(generator, length)
                batch_hashes.append(suite.tensor_digest(x, v))
                if reference_batches is not None and batch_hashes[-1] != reference_batches[step-1]:
                    raise ValueError(f"Paired arms saw different waveform batches at step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss, band, wave = backward_batch(model, x, v, wa, gs, device, args.warmup, kill,
                                                  args.wave_weight, args.microbatch)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                if step % args.log_every == 0:
                    print(f"CONTROL arm={layout} step={step}/{args.steps} loss={float(loss):.5f}", flush=True)
            if not step or step % args.validate_every == 0 or step == args.steps:
                scored = (baseline if not step and layout == "legacy_log" else
                          suite.evaluate(model, validation, wa, gs, device, args.warmup, kill))
                assessment = suite.assess(scored["summary"], baseline["summary"])
                write_json(args.out / f"{layout}_step{step:04d}_selection.json", scored)
                curves.append({"step": step, "assessment": assessment, "summary": scored["summary"],
                               "loss": float(loss) if step else None, "wall_s": time.perf_counter()-arm_started})
                if not step:
                    save(args.out / f"{layout}_initial.pt", model, initial, metadata, layout, step, assessment)
                if step and assessment["eligible"] and assessment["rank_gain_db"] > best_rank:
                    best_rank, best_step = assessment["rank_gain_db"], step
                    save(args.out / f"{layout}_selected.pt", model, initial, metadata, layout, step, assessment)
                print(f"SELECT arm={layout} step={step} eligible={assessment['eligible']} "
                      f"rank={assessment['rank_gain_db']} best_step={best_step}", flush=True)
        save(args.out / f"{layout}_final.pt", model, initial, metadata, layout, args.steps, assessment)
        if reference_batches is None:
            reference_batches = batch_hashes
        report["arms"][layout] = {
            "best_step": best_step, "best_rank_gain_db": best_rank if best_step is not None else None,
            "selected_sha256": core.digest(args.out / f"{layout}_selected.pt") if best_step is not None else None,
            "final_sha256": core.digest(args.out / f"{layout}_final.pt"),
            "batch_sha256": batch_hashes, "curve": curves, "wall_s": time.perf_counter()-arm_started,
            "parameters_changed": any(not torch.equal(value.cpu(), initial["model"][name].cpu())
                                      for name, value in model.state_dict().items()),
        }
        write_json(args.out / "experiment.json", report)
    report["paired_batches_identical"] = report["arms"]["legacy_log"]["batch_sha256"] == report["arms"]["mel_unique"]["batch_sha256"]
    report["deployed_checkpoint_unchanged"] = core.digest(args.checkpoint) == original_hash
    if not report["paired_batches_identical"] or not report["deployed_checkpoint_unchanged"]:
        raise ValueError("Comparison or frozen checkpoint integrity failed")
    if not all(arm["parameters_changed"] for arm in report["arms"].values()):
        raise ValueError("No learned weight change in an arm")
    report.update(status="complete", total_wall_s=time.perf_counter()-started)
    write_json(args.out / "experiment.json", report)
    print(f"LAYOUT_CONTROL PASS out={args.out}; selected checkpoint may be NONE", flush=True)


if __name__ == "__main__":
    # Required before the first CUDA GEMM when deterministic algorithms are on.
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
