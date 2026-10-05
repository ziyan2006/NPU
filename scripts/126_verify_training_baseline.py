"""Paired TRAIN-only small-fit test, pinned graph and real reconstruction losses.

This is NOT a quality-upgrade or deployment test. It saves complete Adam/RNG
resume state and refuses changed inputs/config/code when resuming. No SD writes.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("small_fit_data_lock", ROOT / "scripts/125_lock_training_data.py")
lock = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lock
spec.loader.exec_module(lock)
suite, core = lock.suite, lock.core
control = suite.load("small_fit_control", "120_train_layout_control.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def protected_full_mask(mask, gs, kill):
    masked = mask.clone()
    masked[:, :, :kill] = 0
    return torch.einsum("fk,bckt->bcft", gs, masked).clamp(0, 1)


def reconstruction_loss(mask, spectrum, x, v, gs, warmup, kill):
    full = protected_full_mask(mask, gs, kill)
    estimated = spectrum*full
    pv = core.t09._istft(estimated.flatten(0, 1), x.shape[-1]).reshape_as(x)
    region = suite.scoring_slice(x.shape[-1], warmup)
    wave_error = (pv[..., region]-v[..., region]).abs().mean(dim=(-1, -2))
    rms = x[..., region].square().mean(dim=(-1, -2)).sqrt()
    wave = (wave_error/(rms+1e-3)).mean()
    truth = core.stft_batch(v)
    score_frames = slice(warmup+2, spectrum.shape[-1]-2)
    error = (estimated-truth)[..., score_frames]
    reference = spectrum[..., score_frames]
    error_l1 = (error.real.abs()+error.imag.abs()).mean(dim=(1, 2, 3))
    norm_l1 = (reference.real.abs()+reference.imag.abs()).mean(dim=(1, 2, 3))
    spectral = (error_l1/(norm_l1+1e-3)).mean()
    return wave+spectral, {"wave_l1": wave, "complex_l1": spectral}, pv


def backward_batch(net, x, v, wa, gs, device, warmup, kill, arm, microbatch):
    if arm == "band_wave_control":
        total, band, wave = control.backward_batch(net, x, v, wa, gs, device, warmup, kill, 1., microbatch)
        return {"loss": total, "band": band, "wave_mse": wave}
    if arm != "reconstruction_l1" or microbatch < 1 or any(
        isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout,
                       torch.nn.Dropout2d, torch.nn.Dropout3d)) for m in net.modules()):
        raise ValueError("Microbatch requires a deterministic batch-independent graph and known loss")
    if x.shape != v.shape or x.ndim != 3 or x.shape[0] < 1:
        raise ValueError("Expected aligned nonempty (batch, stereo, samples)")
    values = {"loss": 0., "wave_l1": 0., "complex_l1": 0.}
    for start in range(0, x.shape[0], microbatch):
        xb, vb = x[start:start+microbatch].to(device), v[start:start+microbatch].to(device)
        spectrum = core.stft_batch(xb)
        bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
        output = net(bands)
        loss, parts, _ = reconstruction_loss((output[:, :2]+1)/2, spectrum, xb, vb, gs, warmup, kill)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite reconstruction loss")
        weight = xb.shape[0]/x.shape[0]
        (loss*weight).backward()
        for name, value in {"loss": loss, **parts}.items():
            values[name] += float(value.detach())*weight
    return values


def build_cohort(corpus, config):
    length = (config["crop_frames"]+config["warmup_frames"]-1)*core.HOP
    region = suite.scoring_slice(length, config["warmup_frames"])
    rows, count = [], 0
    # Alphabetical order, central crop and energy criteria are fixed BEFORE
    # fitting; neither development nor old test audio influences the choice.
    for record in sorted(corpus.train["musdb"], key=lambda r: r["track_id"]):
        mix, vocal = corpus.audio(record)
        if mix.shape != vocal.shape or mix.shape[0] != 2 or mix.shape[-1] < length:
            continue
        start = (mix.shape[-1]-length)//2
        x, v = mix[:, start:start+length], vocal[:, start:start+length]
        m, vv = float(x[..., region].square().mean()), float(v[..., region].square().mean())
        if not (m > 1e-6 and vv > 1e-6 and vv/m >= .01):
            continue
        for db in config["variants_db"]:
            ref = v*10**(db/20)
            remixed = x-v+ref
            gain = min(1., .95/max(float(remixed.abs().max()), 1e-12))
            remixed, ref = (remixed*gain).clone(), (ref*gain).clone()
            rows.append({"track": record["track_id"], "domain": "musdb/train_fit",
                         "start_sample": start, "vocal_gain_db": db, "common_peak_gain": gain,
                         "samples_sha256": suite.tensor_digest(remixed, ref), "x": remixed, "v": ref})
        count += 1
        if count == config["source_songs"]:
            break
    if count != config["source_songs"]:
        raise ValueError("Insufficient active vocal training sources for the locked small-fit cohort")
    return rows


@torch.no_grad()
def evaluate(net, rows, wa, gs, device, config, arm):
    net.eval()
    clips, losses = [], []
    for row in rows:
        x, v = row["x"][None].to(device), row["v"][None].to(device)
        spectrum, bands, target = core.prepare_truth(x, v, wa)
        out = net(bands)
        mv, ma = (out[:, :2]+1)/2, (out[:, 2:]+1)/2
        if arm == "reconstruction_l1":
            loss, parts, pv = reconstruction_loss(mv, spectrum, x, v, gs, config["warmup_frames"], 44)
        else:
            pv = core.product_vocal(spectrum, mv, gs, x.shape[-1], 44)
            wave = core.waveform_loss(pv, x, v, config["warmup_frames"])
            s = slice(config["warmup_frames"], None)
            band = core.objective("complement", mv[..., s], ma[..., s], target[..., s], bands[..., s])
            loss, parts = band+wave, {"band": band, "wave_mse": wave}
        interval = suite.scoring_slice(x.shape[-1], config["warmup_frames"])
        if not torch.isfinite(pv).all() or not torch.isfinite(loss):
            raise ValueError("Non-finite small-fit evaluation")
        metrics = suite.separation_metrics(pv[0, ..., interval].cpu(), x[0, ..., interval].cpu(), v[0, ..., interval].cpu())
        clips.append({k: value for k, value in row.items() if k not in ("x", "v")} |
                     {"loss": float(loss), "parts": {k: float(v) for k, v in parts.items()}, **metrics})
        losses.append(float(loss))
    return {"loss": statistics.fmean(losses), "clips": clips}


def assess_fit(candidate, baseline, policy):
    if len(candidate["clips"]) != len(baseline["clips"]) or not candidate["clips"]:
        raise ValueError("Unpaired/empty small-fit scores")
    reasons, domains, gains = [], {}, []
    for c, b in zip(candidate["clips"], baseline["clips"]):
        for key in ("track", "samples_sha256", "vocal_gain_db"):
            if c.get(key) != b.get(key):
                raise ValueError("Different scored small-fit waveforms")
        gain = c["vocal_error_snr_db"]-b["vocal_error_snr_db"]
        if not math.isfinite(gain):
            raise ValueError("Non-finite learning gain")
        domains.setdefault(c["vocal_gain_db"], []).append(gain)
        gains.append(gain)
    loss_reduction = 1-candidate["loss"]/max(baseline["loss"], 1e-12)
    if not math.isfinite(loss_reduction) or loss_reduction < policy["minimum_loss_reduction_fraction"]:
        reasons.append("Insufficient fitted training-loss reduction")
    for db, key in ((0, "minimum_native_error_snr_gain_db"), (-12, "minimum_weak_error_snr_gain_db")):
        if db not in domains or statistics.fmean(domains[db]) < policy[key]:
            reasons.append(f"Insufficient fixed-cohort waveform gain at vocal {db} dB")
    if min(gains) < -policy["maximum_any_clip_regression_db"]:
        reasons.append("A fixed training clip regressed")
    return {"passed": not reasons, "reasons": reasons, "loss_reduction_fraction": loss_reduction,
            "mean_error_snr_gain_db_by_vocal_db": {str(db): statistics.fmean(v) for db, v in domains.items()},
            "minimum_clip_gain_db": min(gains), "scope": "TRAIN-only mechanics, not generalization or production acceptance"}


def capture_state(net, optimizer, step, binding, arm):
    return {"model": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()), "step": step, "binding": binding, "arm": arm,
            "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "purpose": "Resumable small-fit research state; never a release checkpoint"}


def restore_state(state, net, optimizer, binding, arm):
    if state["binding"] != binding or state["arm"] != arm:
        raise ValueError("Resume protocol/data/code/arm binding mismatch")
    net.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"].cpu())
    if state["cuda_rng"]:
        if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA runtime mismatch")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    return state["step"]


def validate_protocol(protocol):
    c = protocol["small_fit"]
    expected_graph = {"bands": 128, "layout": "legacy_log", "bottleneck_blocks": 2,
                      "frontend": ["linear", 1.0], "lf_kill_bands": 44}
    if protocol["schema"] != 1 or protocol["frozen_sha256"] != lock.FROZEN_SHA or protocol["graph"] != expected_graph:
        raise ValueError("Unsupported graph or frozen model")
    if (c["crop_frames"] < 32 or c["crop_frames"] % 8 or c["warmup_frames"] < 96 or c["warmup_frames"] % 8 or
        c["steps"] < 1 or c["warmup_steps"] < 1 or c["check_every"] < 1 or c["microbatch"] < 1 or
        not math.isfinite(c["learning_rate"]) or c["learning_rate"] <= 0 or c["source_songs"] < 1 or
        c["variants_db"] != [0, -12] or c["arms"] != ["band_wave_control", "reconstruction_l1"]):
        raise ValueError("Invalid locked small-fit configuration")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-lock", type=Path, default=ROOT / "results/training_protocol_20261001/dataset_lock.json")
    ap.add_argument("--protocol", type=Path, default=ROOT / "docs/model_training_protocol_v1.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/training_small_fit_20261001")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pause-after-step", type=int, help="Resume smoke test: return after this saved check in the first incomplete arm")
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    config = protocol["small_fit"]
    if args.pause_after_step is not None and (args.pause_after_step < 1 or
        args.pause_after_step >= config["steps"] or args.pause_after_step % config["check_every"]):
        ap.error("Pause only at an intermediate saved check; it never changes the locked training budget")
    if args.out.exists() != args.resume:
        raise ValueError("Use a fresh output directory, or explicitly --resume an existing run")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    locked = lock.verify_lock(args.data_lock)
    if locked["seed"] != protocol["seed"]:
        raise ValueError("Protocol/data split seeds differ")
    corpus = core.Corpus(ROOT / "data/datasets/MUSDB18-7-STEMS", ROOT / "results/opt_bott2_5m_cache", protocol["seed"])
    suite.guard_split(corpus)
    current_roles = {(d, r["track_id"], role) for role, groups in (("train", corpus.train), ("development", corpus.val))
                     for d, records in groups.items() for r in records}
    if current_roles != {(r["domain"], r["track_id"], r["role"]) for r in locked["records"]}:
        raise ValueError("Current corpus differs from the locked split")
    rows = build_cohort(corpus, config)
    cohort = [{k: v for k, v in row.items() if k not in ("x", "v")} for row in rows]
    x, v = torch.stack([r["x"] for r in rows]), torch.stack([r["v"] for r in rows])
    binding_data = {"data_lock_sha256": lock.sha256(args.data_lock), "protocol_sha256": lock.sha256(args.protocol),
                    "frozen_sha256": lock.FROZEN_SHA, "cohort": cohort,
                    "code_sha256": {name: lock.sha256(ROOT / "scripts" / name) for name in (
                        "09_target_model.py", "11_smoke_train.py", "13_ab_compare.py", "23_build_true_stem_cache.py",
                        "109_audit_model_limits.py", "110_train_residual_ablation.py", "119_model_selection_suite.py",
                        "120_train_layout_control.py", "125_lock_training_data.py", "126_verify_training_baseline.py")},
                    "torch_version": str(torch.__version__), "device": device,
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    binding = lock.document_digest(binding_data)
    if args.resume:
        report = json.loads((args.out / "experiment.json").read_text(encoding="utf-8"))
        if report["binding"] != binding:
            raise ValueError("Resume data/protocol/code/runtime changed")
    else:
        args.out.mkdir(parents=True)
        report = {"schema": 1, "binding": binding, "inputs": binding_data, "protocol": protocol,
                  "status": "running", "arms": {}, "blind_acceptance": "missing", "deployed_model_promoted": False}
        write_json(args.out / "experiment.json", report)
        write_json(args.out / "cohort.json", {"rows": cohort, "scope": "Fixed TRAIN-only; 2 songs, 4 remixed views"})
    net, initial = core.t13.load_student(Path(locked["checkpoint"]), device)
    if (initial.get("bottleneck_blocks") != 2 or initial.get("temporal_dilations") or
        net.frontend != ("linear", 1.) or net.band_layout != "legacy_log" or net.n_bands != 128):
        raise ValueError("Starting checkpoint graph does not match locked protocol")
    wa = torch.from_numpy(core.t09.make_analysis_matrix()).to(device)
    gs = torch.from_numpy(core.t09.make_synthesis_matrix()).to(device)
    run_started = time.perf_counter()
    try:
        for arm in config["arms"]:
            existing = report["arms"].get(arm)
            if existing and existing["status"] == "complete":
                if lock.sha256(args.out / f"{arm}_final.pt") != existing["final_sha256"]:
                    raise ValueError("A completed arm checkpoint was changed")
                continue
            net.load_state_dict(initial["model"])
            torch.manual_seed(protocol["seed"])
            optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
            resume_path = args.out / f"{arm}_resume.pt"
            start_step = 0
            if existing:
                start_step = restore_state(torch.load(resume_path, map_location="cpu", weights_only=False),
                                           net, optimizer, binding, arm)
                if existing["curve"][-1]["step"] != start_step:
                    raise ValueError("Resume state/report step mismatch")
            else:
                baseline = evaluate(net, rows, wa, gs, device, config, arm)
                existing = {"status": "running", "baseline": baseline, "curve": [{"step": 0, "scores": baseline}],
                            "wall_s": 0.}
                report["arms"][arm] = existing
                torch.save(capture_state(net, optimizer, 0, binding, arm), resume_path)
                write_json(args.out / "experiment.json", report)
            print(f"SMALL_FIT arm={arm} start_step={start_step} cohort={len(rows)} TRAIN clips; blind=missing", flush=True)
            arm_started = time.perf_counter()
            for step in range(start_step+1, config["steps"]+1):
                net.train()
                lr = config["learning_rate"]*min(step/config["warmup_steps"], 1.)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                losses = backward_batch(net, x, v, wa, gs, device, config["warmup_frames"], 44, arm, config["microbatch"])
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                if step % config["check_every"] == 0 or step == config["steps"]:
                    scores = evaluate(net, rows, wa, gs, device, config, arm)
                    assessment = assess_fit(scores, existing["baseline"], config["gate"])
                    existing["curve"].append({"step": step, "scores": scores, "assessment": assessment,
                                              "training_loss": losses, "gradient_norm": float(grad_norm), "lr": lr})
                    existing["wall_s"] += time.perf_counter()-arm_started
                    arm_started = time.perf_counter()
                    torch.save(capture_state(net, optimizer, step, binding, arm), resume_path)
                    write_json(args.out / "experiment.json", report)
                    print(f"SMALL_FIT arm={arm} step={step}/{config['steps']} loss={scores['loss']:.6f} "
                          f"fit_pass={assessment['passed']} gains={assessment['mean_error_snr_gain_db_by_vocal_db']}", flush=True)
                    if args.pause_after_step == step:
                        report.update(status="yielded_for_resume_test", deployed_checkpoint_unchanged=
                                      lock.sha256(locked["checkpoint"]) == lock.FROZEN_SHA)
                        write_json(args.out / "experiment.json", report)
                        print(f"SMALL_FIT SAVED arm={arm} step={step}; rerun with --resume to finish the SAME budget", flush=True)
                        return
            final = existing["curve"][-1]
            payload = {k: value for k, value in initial.items() if k != "model"}
            payload.update(model={k: value.detach().cpu().clone() for k, value in net.state_dict().items()},
                           step=config["steps"], steps=config["steps"], lr=config["learning_rate"],
                           crop=config["crop_frames"], mask_mode="complement", best_val=None,
                           small_fit={"binding": binding, "arm": arm, "assessment": final["assessment"],
                                      "config": config, "initial_checkpoint_step": initial.get("step"),
                                      "purpose": "TRAIN-only fit, never deploy or treat as generalization"})
            final_path = args.out / f"{arm}_final.pt"
            torch.save(payload, final_path)
            existing.update(status="complete", final_sha256=lock.sha256(final_path),
                            parameters_changed=any(not torch.equal(value.cpu(), initial["model"][key].cpu())
                                                   for key, value in net.state_dict().items()),
                            final_assessment=final["assessment"])
            write_json(args.out / "experiment.json", report)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="interrupted_or_failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.out / "experiment.json", report)
        raise
    if lock.sha256(locked["checkpoint"]) != lock.FROZEN_SHA:
        raise ValueError("Deployed checkpoint changed")
    report.update(status="complete", invocation_wall_s=time.perf_counter()-run_started,
                  deployed_checkpoint_unchanged=True,
                  paired_fixed_waveforms=True,
                  learning_mechanics_passed=report["arms"]["reconstruction_l1"]["final_assessment"]["passed"],
                  formal_training_started=False)
    report.pop("error", None)
    write_json(args.out / "experiment.json", report)
    print(f"SMALL_FIT COMPLETE learning_gate={report['learning_mechanics_passed']}; no quality/deployment acceptance", flush=True)


if __name__ == "__main__":
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
