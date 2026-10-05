"""Sealed TRAIN-only CPU gradient-strength diagnosis. No optimizer, resume or CUDA."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import torch

spec = importlib.util.spec_from_file_location("strength_source", Path(__file__).with_name("161_train_mel_source_aux.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
p, m, d, acq, ROOT = t.p, t.m, t.d, t.acq, t.ROOT
PURPOSE = "NONRELEASE_SOURCE_AUX_STRENGTH_TRAIN_DIAGNOSTIC"
GRID = (0., .02, .1, .2)
CURSORS = tuple(range(2000, 2012))
SOURCE = ROOT / "results/mel_source_aux_20261003"
DEFAULT_OUT = ROOT / "results/mel_source_aux_strength_20261003"
REPORT = ROOT / "reports/45_source_aux_strength_diagnostic_plan.md"
TEST = Path(__file__).with_name("_test_source_aux_strength.py")


def strength_rows(base, auxiliary, d_base, d_aux):
    if any(type(v) is not float or not math.isfinite(v) for v in (base, auxiliary, d_base, d_aux)) or base < 0 or auxiliary < 0:
        raise ValueError("Finite nonnegative FP32-derived losses required")
    return [{"lambda": w, "loss": base+w*auxiliary, "auxiliary_contribution": w*auxiliary,
        "contribution_over_base": w*auxiliary/base if base > 0 else None,
        "d_combined_d_mask_gain": d_base+w*d_aux,
        "local_gain_descent_direction": "less_vocal" if d_base+w*d_aux > 1e-8 else
                                       "more_vocal" if d_base+w*d_aux < -1e-8 else "flat"} for w in GRID]


def gradient_summary(base, auxiliary):
    if (base.ndim != 1 or base.shape != auxiliary.shape or base.numel() == 0 or
        base.dtype != torch.float32 or auxiliary.dtype != torch.float32 or
        base.device.type != "cpu" or auxiliary.device.type != "cpu" or
        not torch.isfinite(base).all() or not torch.isfinite(auxiliary).all() or float(base.norm()) == 0):
        raise ValueError("Aligned finite nonzero CPU FP32 base gradient required")
    nb, na = float(base.norm()), float(auxiliary.norm())
    return {"base_l2": nb, "auxiliary_l2": na,
        "base_auxiliary_cosine": float((base@auxiliary)/(base.norm()*auxiliary.norm())) if na else None,
        "strengths": [{"lambda": w, "combined_l2": float((base+w*auxiliary).norm()),
                       "auxiliary_change_l2": w*na, "change_over_base_l2": w*na/nb,
                       "simulated_clip_scale_at_3": min(1., 3./(float((base+w*auxiliary).norm())+1e-6)),
                       "simulated_postclip_l2_at_3": float((base+w*auxiliary).norm())*min(1., 3./(float((base+w*auxiliary).norm())+1e-6)),
                       "cosine_with_base": float((base@(base+w*auxiliary))/(base.norm()*(base+w*auxiliary).norm()))
                                          if float((base+w*auxiliary).norm()) else None} for w in GRID]}


def check_plan(doc):
    exact = {"purpose": PURPOSE, "diagnostic_only": True, "training_authorized": False,
        "cuda_used": False, "model_updates": 0, "cursors": list(CURSORS), "lambda_grid": list(GRID),
        "source_arm": t.ARMS[1], "slot_weights": [1]*6, "teacher": "kim_melband",
        "selection": "Fixed TRAIN counters2000..2011; first eligible batch for full gradients",
        "release_selection": "NONE", "deployment": False}
    if any(doc.get(k) != v or type(doc.get(k)) is not type(v) for k, v in exact.items()):
        raise ValueError("Changed diagnostic scope/roles/grid/budget")
    if any(type(x) is not int for x in doc["cursors"]) or any(type(x) is not float for x in doc["lambda_grid"]):
        raise ValueError("Changed diagnostic counter/grid types")


def no_active_trainer():
    # Read-only actual processes, not a stale status.running flag. CPU diagnosis
    # also avoids competing with an active worker's development scoring.
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '(150_train_paired_exploration|154_train_mel_weak_weight|161_train_mel_source_aux)[.]py' } | Select-Object -ExpandProperty ProcessId"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise ValueError("Active historical training process: preserve it and postpone diagnosis")


def checked_source():
    doc = p.verified_approval(p.DEFAULT_APPROVAL)
    completion = acq.read_sealed(SOURCE / "completion.json")
    receipt = acq.read_sealed(SOURCE / "checkpoint_2000.json")
    exit_path = ROOT / "results/mel_source_aux_launch_20261003/detached_exit_20261003_012859_7811676.json"
    exit_doc = json.loads(exit_path.read_text(encoding="utf-8-sig"))
    launch_path = Path(exit_doc["launch_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    if (exit_doc["exit_code"] != 0 or launch["probe_only"] or Path(launch["out"]) != SOURCE or
        exit_doc["launcher_process_id"] != launch["worker"]["LauncherProcessId"] or
        completion["binding"] != t.binding(p.DEFAULT_APPROVAL) or completion["step"] != 2000 or
        completion["additional_steps"] != 500 or completion["final_checkpoint"] != receipt):
        raise ValueError("Missing exact normal completed source")
    proof_path = ROOT / "results/mel_source_aux_monitor_20261003/completion_review.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if (proof["trainer_verify"]["exit_code"] != 0 or proof["score_review"]["exit_code"] != 0 or
        proof["processes"] or proof["normal_exit_verified"] is not True or proof["matched_exit"] != exit_doc or
        proof["score_review"]["tracks"] != 31 or proof["score_review"]["views"] != 177 or proof["score_review"]["pending"]):
        raise ValueError("Missing independent completion evidence")
    state = m.load_checked_checkpoint(SOURCE / receipt["checkpoint"], receipt["sha256"])
    if (state["binding"] != completion["binding"] or state["purpose"] != p.PURPOSE or
        state["step"] != 2000 or state["limit"] != 2000 or state["sampler"]["cursor"] != 2000 or
        state["schedule"]["step"] != 2000 or state["schedule"]["last_validation"] != 2000 or
        state["smoke"] is not False or state["deployment_authorized"] is not False or
        any(a["updates"] != 2000 for a in state["arms"].values()) or not d.finite_state(state)):
        raise ValueError("Exact complete source state required")
    files = [p.DEFAULT_APPROVAL, SOURCE / "completion.json", SOURCE / "checkpoint_2000.json",
        SOURCE / receipt["checkpoint"], exit_path, launch_path, proof_path,
        ROOT / "results/mel_source_aux_review_20261003/summary_step_2000/review.json",
        Path(__file__), TEST, REPORT, Path(t.__file__), Path(p.__file__), Path(t.k.__file__)]
    return doc, state, {str(path): acq.sha256(path) for path in files}


def prepare(out):
    t.old.require_fresh(out)
    no_active_trainer()
    _, _, bindings = checked_source()
    plan = {"schema": 1, "purpose": PURPOSE, "diagnostic_only": True, "training_authorized": False,
        "cuda_used": False, "model_updates": 0, "cursors": list(CURSORS), "lambda_grid": list(GRID),
        "source_arm": t.ARMS[1], "slot_weights": [1]*6, "teacher": "kim_melband",
        "selection": "Fixed TRAIN counters2000..2011; first eligible batch for full gradients",
        "bindings_sha256": bindings, "release_selection": "NONE", "deployment": False}
    check_plan(plan)
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    print("SOURCE_AUX_STRENGTH PLAN SEALED; no training", flush=True)


def verified_plan(out):
    plan = acq.read_sealed(out / "plan.json")
    check_plan(plan)
    _, _, bindings = checked_source()
    if bindings != plan["bindings_sha256"]:
        raise ValueError("Changed completed source or diagnostic binding")
    return plan


def collect(dataset, true, cursor):
    if type(cursor) is not int or cursor not in CURSORS:
        raise ValueError("Fixed diagnostic TRAIN counters only, not a resume stream")
    xs, vs, metadata = [], [], []
    for domain in m.DOMAINS[:3]:
        item = true.crop(domain, dataset.seed, cursor)
        xs.append(item["x"]); vs.append(item["v"]); metadata.append(item["meta"])
    for index in range(3):
        recipe = m.data.crop_recipe(dataset.rows, dataset.config, dataset.seed, cursor*3+index)
        item = dataset.crop(recipe)
        xs.append(item["x"]); vs.append(item["v"])
        metadata.append(item["meta"] | {"domain": "pseudo", "role": "pseudo_label_train_candidate"})
    t.validate_metadata(metadata)
    return {"x": torch.stack(xs), "v": torch.stack(vs), "metadata": metadata, "diagnostic_counter": cursor}


def slot_losses(net, xb, vb, wa, gs, meta, scalar_only=False):
    if scalar_only:
        with torch.no_grad():
            spectrum = m.core.stft_batch(xb)
            mask = (net(torch.einsum("fk,bcft->bckt", wa, spectrum.abs()))[:, :2]+1)/2
        gain = torch.tensor(1., requires_grad=True)
        mask = mask*gain
    else:
        spectrum = m.core.stft_batch(xb)
        mask = (net(torch.einsum("fk,bcft->bckt", wa, spectrum.abs()))[:, :2]+1)/2
        gain = None
    base, _, pv = m.fit.reconstruction_loss(mask, spectrum, xb, vb, gs, 96, 44)
    region = m.fit.suite.scoring_slice(xb.shape[-1], 96)
    auxiliary, info = t.k.source_projection_auxiliary(pv[0, :, region], xb[0, :, region], vb[0, :, region], meta)
    return base, auxiliary, info, gain


def flatten_gradients(grads, parameters):
    result = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(grads, parameters)])
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite full parameter gradient")
    return result.clone()


def parameter_probe(net, batch, wa, gs):
    params = tuple(net.parameters())
    gb = torch.zeros(sum(x.numel() for x in params))
    ga = torch.zeros_like(gb)
    for i, meta in enumerate(batch["metadata"]):
        base, aux, _, _ = slot_losses(net, batch["x"][i:i+1], batch["v"][i:i+1], wa, gs, meta)
        gb += flatten_gradients(torch.autograd.grad(base/6, params, retain_graph=True, allow_unused=True), params)
        ga += flatten_gradients(torch.autograd.grad(aux/6, params, allow_unused=True), params)
    actual = torch.zeros_like(gb)
    for i, meta in enumerate(batch["metadata"]):
        base, aux, _, _ = slot_losses(net, batch["x"][i:i+1], batch["v"][i:i+1], wa, gs, meta)
        actual += flatten_gradients(torch.autograd.grad((base+.2*aux)/6, params, allow_unused=True), params)
    expected = gb+.2*ga
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-7)
    if float(ga.norm()) <= 0:
        raise ValueError("No nonzero actual auxiliary parameter gradient")
    return gradient_summary(gb, ga) | {"diagnostic_counter": batch["diagnostic_counter"],
        "combined_gradient_independently_checked_lambda": .2,
        "max_absolute_combination_error": float((actual-expected).abs().max()),
        "parameter_count": gb.numel(), "gradients_populated_on_model": any(p.grad is not None for p in params)}


def run(out):
    if (out / "diagnostic.json").exists():
        raise ValueError("Already generated; use verify and preserve evidence")
    plan = verified_plan(out)
    no_active_trainer()
    doc, state, _ = checked_source()
    rows, selected = [], None
    torch.set_num_threads(4)
    with d.deterministic_runtime("cpu"):
        dataset = p.inp.ApprovedTeacherDataset(p.inp.verified_approval(Path(doc["origin_approval"])), "kim_melband")
        true = m.LockedTruePool(m.bulk.OLD_LOCK, dataset.config)
        if state["sampler"]["seed"] != dataset.seed or state["sampler"]["true_lock_sha256"] != true.bound:
            raise ValueError("Source input seed/lock changed")
        net = m.frozen_factory(doc["source_protocol"])()
        net.load_state_dict(state["arms"][t.ARMS[1]]["model"], strict=True)
        net.train()
        if any(isinstance(x, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)) for x in net.modules()):
            raise ValueError("Read-only deterministic batch-independent graph required")
        before, mode = t.old.dev.state_digest(net.state_dict()), [(n, x.training) for n, x in net.named_modules()]
        wa, gs = torch.from_numpy(m.core.t09.make_analysis_matrix()), torch.from_numpy(m.core.t09.make_synthesis_matrix())
        for cursor in CURSORS:
            batch = collect(dataset, true, cursor)
            for i, meta in enumerate(batch["metadata"]):
                xb, vb = batch["x"][i:i+1], batch["v"][i:i+1]
                if m.pilot.wave_digest(xb[0]) != meta["input_pcm_sha256"]:
                    raise ValueError("TRAIN input PCM identity mismatch")
                base, aux, info, gain = slot_losses(net, xb, vb, wa, gs, meta, scalar_only=True)
                gb = torch.autograd.grad(base, gain, retain_graph=True)[0]
                ga = torch.autograd.grad(aux, gain)[0]
                values = [float(x.detach()) for x in (base, aux, gb, ga)]
                rows.append({"diagnostic_counter": cursor, "slot": i, "metadata": meta, **info,
                    "base_loss": values[0], "auxiliary_loss": values[1], "d_base_d_mask_gain": values[2],
                    "d_auxiliary_d_mask_gain": values[3], "strengths": strength_rows(*values)})
                if selected is None and info["active"] and values[1] > 0:
                    selected = d.portable(batch)
            print(f"SOURCE_AUX_STRENGTH counter={cursor} slots={len(rows)}/72", flush=True)
        if selected is None:
            raise ValueError("Fixed TRAIN scope has no active nonzero auxiliary")
        gradients = parameter_probe(net, selected, wa, gs)
        if before != t.old.dev.state_digest(net.state_dict()) or mode != [(n, x.training) for n, x in net.named_modules()]:
            raise ValueError("Diagnosis mutated model or mode")
        result = {"schema": 1, "purpose": PURPOSE, "plan_sha256": acq.sha256(out / "plan.json"),
            "bindings_sha256": plan["bindings_sha256"], "rows": rows, "gradients": gradients,
            "active_count": sum(r["active"] for r in rows), "skipped_count": sum(not r["active"] for r in rows),
            "model_digest": before, "model_and_modes_unchanged": True, "model_updates": 0,
            "optimizer_constructed": False, "cuda_used": False, "runtime": d.runtime_identity("cpu"),
            "release_selection": "NONE", "deployment": False,
            "scope": "TRAIN small-sample local/full-gradient mechanism; not student quality, generalization or causal proof"}
    if not d.finite_state(result):
        raise ValueError("Nonfinite diagnosis")
    acq.write_new_json(out / "diagnostic.json", acq.seal(result))
    verify(out)


def verify(out):
    plan = verified_plan(out)
    result = acq.read_sealed(out / "diagnostic.json")
    if (result["purpose"] != PURPOSE or result["plan_sha256"] != acq.sha256(out / "plan.json") or
        result["bindings_sha256"] != plan["bindings_sha256"] or result["model_updates"] != 0 or
        result["model_and_modes_unchanged"] is not True or result["optimizer_constructed"] is not False or
        result["cuda_used"] is not False or result["deployment"] is not False or
        result["release_selection"] != "NONE" or len(result["rows"]) != 72 or not d.finite_state(result) or
        result["runtime"]["device"] != "cpu" or result["runtime"]["threads"] != 4 or
        result["runtime"]["deterministic"] is not True):
        raise ValueError("Changed diagnostic result/scope")
    active = 0
    for j, row in enumerate(result["rows"]):
        if row["diagnostic_counter"] != CURSORS[j//6] or row["slot"] != j%6:
            raise ValueError("Changed fixed diagnostic ordering")
        if row["strengths"] != strength_rows(row["base_loss"], row["auxiliary_loss"], row["d_base_d_mask_gain"], row["d_auxiliary_d_mask_gain"]):
            raise ValueError("Strength arithmetic differs")
        if (type(row["active"]) is not bool or
            (row["active"] and j%6 not in (0, 1)) or
            (not row["active"] and (row["auxiliary_loss"] != 0. or row["d_auxiliary_d_mask_gain"] != 0.))):
            raise ValueError("Inactive/nontruth slots received auxiliary")
        active += row["active"]
    for j in range(12):
        t.validate_metadata([r["metadata"] for r in result["rows"][j*6:j*6+6]])
    if active != result["active_count"] or 72-active != result["skipped_count"] or not active:
        raise ValueError("Invalid activity accounting")
    if result["gradients"]["gradients_populated_on_model"] or result["gradients"]["auxiliary_l2"] <= 0:
        raise ValueError("Invalid read-only parameter gradient proof")
    print("SOURCE_AUX_STRENGTH VERIFIED; CPU TRAIN-only; updates0; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    globals()[args.command](args.out)
