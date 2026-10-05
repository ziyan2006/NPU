"""Short supervised probe for the 128-band mel_unique front end.

Keeps the frozen convolution graph and training budget, but recomputes true
stem inputs/targets using mel_unique bands. It never edits the deployed model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
CORE = importlib.util.spec_from_file_location("train_mel_core", ROOT / "scripts/110_train_residual_ablation.py")
core = importlib.util.module_from_spec(CORE)
CORE.loader.exec_module(core)


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(path, model, initial, step, metadata, validation):
    payload = {k: v for k, v in initial.items() if k != "model"}
    payload.update(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   step=step, steps=metadata["steps"], lr=metadata["lr"], best_val=None,
                   mask_mode="complement", band_layout="mel_unique", n_bands=128,
                   finetune={**metadata, "step": step, "internal_validation": validation,
                             "initial_checkpoint_step": initial.get("step")})
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/student_bott2_mir1k_candidate.pt")
    ap.add_argument("--musdb-root", type=Path, default=ROOT / "data/datasets/MUSDB18-7-STEMS")
    ap.add_argument("--teacher-cache", type=Path, default=ROOT / "results/opt_bott2_5m_cache")
    ap.add_argument("--out", type=Path, default=ROOT / "results/mel_unique_finetune_20261001")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--wave-weight", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("Choose a fresh output directory; candidates are never overwritten")
    if args.steps < 1 or args.crop < 32 or args.crop % 8 or args.warmup < 96 or args.warmup % 8:
        ap.error("Invalid step, crop or warm-up size")
    if args.log_every < 1:
        ap.error("log-every must be positive")

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, initial = core.t13.load_student(args.checkpoint, device)
    if (initial.get("bottleneck_blocks") != 2 or initial.get("temporal_dilations") or
            model.frontend != ("linear", 1.) or model.n_bands != 128):
        raise ValueError("Requires the frozen 824,900-parameter convolution graph")
    start_hash = sha(args.checkpoint)
    args.out.mkdir(parents=True)

    # Corpus applies the cross-dataset title guard, with Cambridge/DJ/MIR and
    # official MUSDB test compositions reserved before train/validation split.
    corpus = core.Corpus(args.musdb_root, args.teacher_cache, args.seed)
    length = (args.crop + args.warmup - 1) * core.HOP
    validation = corpus.validation(length)
    wa_np = core.t09.make_analysis_matrix(128, layout="mel_unique")
    gs_np = core.t09.make_synthesis_matrix(128, layout="mel_unique")
    wa, gs = torch.from_numpy(wa_np).to(device), torch.from_numpy(gs_np).to(device)
    kill = core.t11.lf_kill_band_for(250., "mel_unique", 128)

    metadata = {"checkpoint": str(args.checkpoint), "sha256": start_hash,
                "script_sha256": sha(__file__), "core_script_sha256": sha(core.__file__),
                "filterbank_source_sha256": {str(ROOT / "scripts/09_target_model.py"): sha(ROOT / "scripts/09_target_model.py"),
                                              str(ROOT / "scripts/11_smoke_train.py"): sha(ROOT / "scripts/11_smoke_train.py")},
                "steps": args.steps, "seed": args.seed, "lr": args.lr,
                "wave_weight": args.wave_weight, "crop": args.crop, "warmup": args.warmup,
                "band_layout": "mel_unique", "n_bands": 128, "kill_bands": kill,
                "band_matrix_sha256": {"analysis": __import__("hashlib").sha256(wa_np.tobytes()).hexdigest(),
                                        "synthesis": __import__("hashlib").sha256(gs_np.tobytes()).hexdigest()},
                "batch_domains": ["musdb", "musdb", "mir1k", "mir1k", "onair", "instrumental"],
                "manifest_provenance": corpus.manifests, "excluded_before_split": corpus.excluded,
                "final_holdout_ids": corpus.final_ids,
                "splits": {d: {"train": [r["track_id"] for r in corpus.train[d]],
                                "validation": [r["track_id"] for r in corpus.val.get(d, [])]}
                           for d in corpus.train},
                "training_target": "true stems only; teacher cache is loaded for corpus guard but contributes no loss",
                "musdb_mix": "Decoded D+B+O+V synthetic sum; no test audio decoded during training",
                "validation_scope": "Reserved from former train pools; may have been seen by warm-start weights",
                "torch_version": torch.__version__, "device": device}
    report = {"metadata": metadata, "arms": {}}
    (args.out / "experiment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    model.band_layout = "mel_unique"
    model.n_bands = 128
    net_initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base_score, domains = core.validate(model, validation, wa, gs, device, args.warmup, kill)
    save_checkpoint(args.out / "mel_unique_best.pt", model, initial, 0, metadata, domains)
    best, best_step = base_score, 0
    curve = [{"step": 0, "validation": base_score, "domains": domains}]
    generator = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(step / 20, 1.) * (0.2 + 0.8 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, v = corpus.batch(generator, length)
        x, v = x.to(device), v.to(device)
        spectrum, bands, target = core.prepare_truth(x, v, wa)
        optimizer.zero_grad(set_to_none=True)
        output = model(bands)
        mv, ma = (output[:, :2] + 1) / 2, (output[:, 2:] + 1) / 2
        selection = slice(args.warmup, None)
        band_loss = core.objective("complement", mv[..., selection], ma[..., selection],
                                   target[..., selection], bands[..., selection])
        predicted = core.product_vocal(spectrum, mv, gs, length, kill)
        wave = core.waveform_loss(predicted, x, v, args.warmup)
        loss = band_loss + args.wave_weight * wave
        if not torch.isfinite(loss):
            raise ValueError("Non-finite mel_unique training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            score, domains = core.validate(model, validation, wa, gs, device, args.warmup, kill)
            curve.append({"step": step, "loss": float(loss), "band": float(band_loss),
                          "wave": float(wave), "validation": score, "domains": domains,
                          "wall_s": time.perf_counter() - started})
            if score < best:
                best, best_step = score, step
                save_checkpoint(args.out / "mel_unique_best.pt", model, initial, step, metadata, domains)
            print(f"MEL_UNIQUE step={step}/{args.steps} loss={float(loss):.5f} "
                  f"val={score:.6f} best_step={best_step}", flush=True)

    save_checkpoint(args.out / "mel_unique_final.pt", model, initial, args.steps, metadata, domains)
    report["arms"]["mel_unique"] = {"best_step": best_step, "best_validation": best,
        "baseline_validation": base_score, "curve": curve, "wall_s": time.perf_counter() - started,
        "best_sha256": sha(args.out / "mel_unique_best.pt"),
        "final_sha256": sha(args.out / "mel_unique_final.pt")}
    report["total_wall_s"] = time.perf_counter() - started
    report["deployed_checkpoint_unchanged"] = sha(args.checkpoint) == start_hash
    if not report["deployed_checkpoint_unchanged"]:
        raise ValueError("Frozen checkpoint changed")
    if not any(not torch.equal(t, net_initial[k]) for k, t in model.state_dict().items()):
        raise ValueError("No trainable parameters changed")
    (args.out / "experiment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MEL_UNIQUE_PROBE PASS out={args.out} params={sum(p.numel() for p in model.parameters())}", flush=True)


if __name__ == "__main__":
    main()
