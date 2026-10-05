"""Sealed TRAIN-only CPU gradient contributions. No optimizer/model update/CUDA.

Forward/backward remain FP32. Only detached geometry reductions use FP64 to
avoid large-vector FP32 cosine artifacts; this does not change inference.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import torch
import importlib.util

spec = importlib.util.spec_from_file_location("contribution_review_source", Path(__file__).with_name("168_review_mel_source_strength.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
t, p, m, d, acq, ROOT = r.t, r.p, r.m, r.t.d, r.acq, r.ROOT
s = p.s
PURPOSE = "NONRELEASE_TRAIN_GRADIENT_CONTRIBUTION_DIAGNOSTIC"
CURSORS = tuple(range(2000, 2012))
MODELS = ("origin_aux002_2000", "control_aux002_2500", "candidate_aux020_2500")
GROUPS = ("true_vocal_base", "instrumental_base", "pseudo_base", "true_auxiliary")
STRENGTHS = (.02, .2)
THREADS = 2
SOURCE = t.DEFAULT_OUT
DEFAULT_OUT = ROOT / "results/train_gradient_contributions_20261003"
PROOF = ROOT / "results/mel_source_strength_monitor_20261003/completion_review.json"
REPORT = ROOT / "reports/51_train_gradient_contribution_plan.md"
TEST = Path(__file__).with_name("_test_gradient_contributions.py")


def fixed_scope():
    return {"schema": 1, "purpose": PURPOSE, "diagnostic_only": True,
            "training_authorized": False, "cuda_used": False, "model_updates": 0,
            "cursors": list(CURSORS), "models": list(MODELS), "groups": list(GROUPS),
            "strengths": list(STRENGTHS), "slot_weights": [1]*6,
            "forward_backward_dtype": "float32", "metric_reduction_dtype": "float64",
            "threads": THREADS, "teacher": "kim_melband",
            "selection": "Same fixed12 TRAIN counters as164; all3 models; no score selection",
            "release_selection": "NONE", "deployment": False}


def check_plan(doc):
    if any(doc.get(k) != value or type(doc.get(k)) is not type(value) for k, value in fixed_scope().items()):
        raise ValueError("Changed fixed TRAIN-only diagnostic scope")
    if (any(type(c) is not int for c in doc["cursors"]) or
        any(type(w) is not int for w in doc["slot_weights"]) or
        any(type(v) is not float for v in doc["strengths"])):
        raise ValueError("Changed counter/weight/strength types")


def slot_group(index, metadata):
    if type(index) is not int or not 0 <= index < 6:
        raise ValueError("Exact original six-slot index required")
    if metadata.get("domain") != m.DOMAINS[index] or metadata.get("role") != (
            "train" if index < 3 else "pseudo_label_train_candidate"):
        raise ValueError("TRAIN/pseudo roles changed; no development allowed")
    return GROUPS[0] if index < 2 else GROUPS[1] if index == 2 else GROUPS[2]


def geometry(vectors):
    if set(vectors) != set(GROUPS):
        raise ValueError("All four aligned contribution vectors required")
    shape = vectors[GROUPS[0]].shape
    if len(shape) != 1 or shape[0] == 0 or any(v.shape != shape or v.dtype != torch.float32 or
            v.device.type != "cpu" or not torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError("Finite aligned CPU FP32 gradients required")
    values = {key: vector.detach().double() for key, vector in vectors.items()}
    norms = {key: float(value.norm()) for key, value in values.items()}
    pairs = {}
    for left, right in itertools.combinations(GROUPS, 2):
        dot = float(values[left]@values[right])
        cosine = dot/(norms[left]*norms[right]) if norms[left] and norms[right] else None
        if cosine is not None and abs(cosine) > 1+1e-12:
            raise ValueError("Invalid gradient geometry")
        pairs[f"{left}|{right}"] = {"dot": dot, "cosine": None if cosine is None else max(-1., min(1., cosine)),
                                   "roundoff_clipped": cosine is not None and abs(cosine) > 1}
    base = sum(values[key] for key in GROUPS[:3])
    auxiliary = values[GROUPS[3]]
    nb, na = float(base.norm()), norms[GROUPS[3]]
    return {"group_l2": norms, "pairs": pairs, "base_l2": nb,
            "base_auxiliary_dot": float(base@auxiliary), "metric_reduction_dtype": "float64",
            "gradient_sha256": {key: hashlib.sha256(vector.detach().contiguous().numpy().tobytes()).hexdigest()
                                 for key, vector in vectors.items()},
            "strengths": [{"lambda": weight, "combined_l2": float((base+weight*auxiliary).norm()),
                           "auxiliary_over_base_l2": weight*na/nb if nb else None}
                          for weight in STRENGTHS]}


def verify_geometry(doc):
    if set(doc["group_l2"]) != set(GROUPS) or set(doc["gradient_sha256"]) != set(GROUPS):
        raise ValueError("Changed gradient groups")
    expected_pairs = {f"{a}|{b}" for a, b in itertools.combinations(GROUPS, 2)}
    if set(doc["pairs"]) != expected_pairs or doc["metric_reduction_dtype"] != "float64":
        raise ValueError("Changed pair coverage/statistic precision")
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in doc["gradient_sha256"].values()):
        raise ValueError("Invalid gradient digests")
    norms = doc["group_l2"]
    if any(type(v) is not float or not math.isfinite(v) or v < 0 for v in norms.values()):
        raise ValueError("Invalid gradient norms")
    for key, pair in doc["pairs"].items():
        a, b = key.split("|")
        expected = pair["dot"]/(norms[a]*norms[b]) if norms[a] and norms[b] else None
        if expected is None:
            if pair["cosine"] is not None:
                raise ValueError("Zero norm cannot have a cosine")
        elif abs(expected) > 1+1e-12 or not math.isclose(pair["cosine"], max(-1., min(1., expected)), abs_tol=1e-12):
            raise ValueError("Cosine differs from retained dot products")
    squared = sum(norms[key]**2 for key in GROUPS[:3])+2*sum(
        doc["pairs"][f"{a}|{b}"]["dot"] for a, b in itertools.combinations(GROUPS[:3], 2))
    if squared < -1e-10:
        raise ValueError("Invalid base gradient squared norm")
    if not math.isclose(doc["base_l2"], math.sqrt(max(0., squared)), rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("Base norm differs from component geometry")
    dot = sum(doc["pairs"][f"{key}|true_auxiliary"]["dot"] for key in GROUPS[:3])
    if not math.isclose(dot, doc["base_auxiliary_dot"], rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("Base/auxiliary dot differs")
    if [row["lambda"] for row in doc["strengths"]] != list(STRENGTHS):
        raise ValueError("Changed diagnostic strength coverage")
    for row in doc["strengths"]:
        expected = math.sqrt(max(0., squared+2*row["lambda"]*dot+(row["lambda"]*norms[GROUPS[3]])**2))
        if not math.isclose(row["combined_l2"], expected, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("Combined norm arithmetic differs")
        ratio = row["lambda"]*norms[GROUPS[3]]/doc["base_l2"] if doc["base_l2"] else None
        if (ratio is None and row["auxiliary_over_base_l2"] is not None) or (ratio is not None and
                not math.isclose(row["auxiliary_over_base_l2"], ratio, rel_tol=1e-7, abs_tol=1e-9)):
            raise ValueError("Contribution ratio differs")


def parameter_probe(net, batch, wa, gs, coefficient, independent=False):
    params = tuple(net.parameters())
    if not params or any(parameter.grad is not None or parameter.dtype != torch.float32 or parameter.device.type != "cpu"
                         for parameter in params):
        raise ValueError("Unpopulated FP32 CPU parameter gradients required")
    t.validate_metadata(batch["metadata"])
    if type(coefficient) is not float or coefficient not in STRENGTHS:
        raise ValueError("Fixed diagnostic strengths only")
    before = r.dev.state_digest(net.state_dict())
    modes = [module.training for module in net.modules()]
    vectors = {key: torch.zeros(sum(parameter.numel() for parameter in params)) for key in GROUPS}
    slots = []
    for index, meta in enumerate(batch["metadata"]):
        group = slot_group(index, meta)
        base, auxiliary, info, _ = s.slot_losses(net, batch["x"][index:index+1], batch["v"][index:index+1], wa, gs, meta)
        if type(info["active"]) is not bool or (info["active"] and index >= 2) or (
                not info["active"] and float(auxiliary.detach()) != 0.):
            raise ValueError("Auxiliary leaked outside eligible true vocal slots")
        gb = s.flatten_gradients(torch.autograd.grad(base/6, params, retain_graph=info["active"], allow_unused=True), params)
        ga = s.flatten_gradients(torch.autograd.grad(auxiliary/6, params, allow_unused=True), params) if info["active"] else torch.zeros_like(gb)
        vectors[group] += gb
        vectors[GROUPS[3]] += ga
        slots.append({"slot": index, "group": group, "base_loss_divided_by6": float(base.detach())/6,
                      "auxiliary_loss_divided_by6": float(auxiliary.detach())/6,
                      "base_gradient_l2": float(gb.double().norm()), "auxiliary_gradient_l2": float(ga.double().norm()),
                      "auxiliary": info, "metadata": meta})
    result = {"geometry": geometry(vectors), "slots": slots, "parameter_count": vectors[GROUPS[0]].numel(),
              "active_count": sum(row["auxiliary"]["active"] for row in slots),
              "skipped_count": sum(not row["auxiliary"]["active"] for row in slots),
              "independent_combination_lambda": coefficient if independent else None,
              "independent_combination_max_error": None, "independent_combination_error_bound": None}
    if independent:
        actual = torch.zeros_like(vectors[GROUPS[0]])
        for index, meta in enumerate(batch["metadata"]):
            base, auxiliary, _, _ = s.slot_losses(net, batch["x"][index:index+1], batch["v"][index:index+1], wa, gs, meta)
            actual += s.flatten_gradients(torch.autograd.grad((base+coefficient*auxiliary)/6, params, allow_unused=True), params)
        expected = sum(vectors[key] for key in GROUPS[:3])+coefficient*vectors[GROUPS[3]]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-7)
        result["independent_combination_max_error"] = float((actual-expected).abs().max())
        result["independent_combination_error_bound"] = 2e-7+2e-4*float(expected.abs().max())
    if before != r.dev.state_digest(net.state_dict()) or modes != [module.training for module in net.modules()] or any(
            parameter.grad is not None for parameter in params):
        raise ValueError("Diagnostic mutated weights/modes/gradient storage")
    verify_geometry(result["geometry"])
    return result


def no_active_trainer():
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '(134_generate_teacher_library|139_generate_paired_htdemucs|150_train_paired_exploration|154_train_mel_weak_weight|161_train_mel_source_aux|166_train_mel_source_strength)[.]py[\" ]+(run|train|smoke)' } | Select-Object -ExpandProperty ProcessId"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise ValueError("Active teacher/student worker; postpone CPU diagnosis")


def checked_sources():
    approval = p.verified_approval(p.DEFAULT_APPROVAL)
    completion = acq.read_sealed(SOURCE / "completion.json")
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    exit_path = ROOT / "results/mel_source_strength_launch_20261003/detached_exit_20261003_042125_5173887.json"
    exit_doc = json.loads(exit_path.read_text(encoding="utf-8-sig"))
    launch_path = Path(exit_doc["launch_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    if (exit_doc["exit_code"] != 0 or launch["probe_only"] or Path(launch["out"]) != SOURCE or
        exit_doc["launcher_process_id"] != launch["worker"]["LauncherProcessId"] or
        completion["step"] != 2500 or completion["additional_steps"] != 500 or
        completion["binding"] != t.binding(p.DEFAULT_APPROVAL) or
        proof["normal_exit_verified"] is not True or proof["matched_exit"] != exit_doc or proof["processes"] or
        proof["trainer_verify"]["exit_code"] != 0 or proof["score_review"]["exit_code"] != 0 or
        proof["score_review"]["tracks"] != 31 or proof["score_review"]["views"] != 177 or proof["score_review"]["pending"] or
        proof["completion_sha256"] != acq.sha256(SOURCE / "completion.json") or
        proof["final_checkpoint_sha256"] != completion["final_checkpoint"]["sha256"]):
        raise ValueError("Exact independently verified normal500 completion required")
    review_path = ROOT / proof["score_review"]["path"]
    if acq.sha256(review_path) != proof["score_review"]["sha256"]:
        raise ValueError("Independent review changed")
    r.verify_output(review_path.parent)
    sources, paths = {}, []
    for step in (2000, 2500):
        receipt_path = SOURCE / f"checkpoint_{step:04d}.json"
        receipt = acq.read_sealed(receipt_path)
        state_path = SOURCE / receipt["checkpoint"]
        state = m.load_checked_checkpoint(state_path, receipt["sha256"])
        if (state["step"] != step or state["binding"] != completion["binding"] or state["smoke"] or
            state["sampler"]["cursor"] != step or state["deployment_authorized"] is not False or
            any(saved["updates"] != step for saved in state["arms"].values())):
            raise ValueError("Changed exact model/checkpoint source")
        if step == 2000:
            sources[MODELS[0]] = state["arms"][t.ARMS[1]]
            sampler = state["sampler"]
        else:
            sources[MODELS[1]], sources[MODELS[2]] = (state["arms"][arm] for arm in t.ARMS)
        paths += [receipt_path, state_path]
    paths += [p.DEFAULT_APPROVAL, PROOF, SOURCE / "completion.json", exit_path, launch_path, review_path,
              ROOT / proof["trainer_verify"]["path"], ROOT / "reports/50_mel_source_strength_completion.md",
              Path(__file__), TEST, REPORT, Path(s.__file__), Path(t.__file__), Path(r.__file__), Path(t.k.__file__)]
    return approval, sources, sampler, {str(path.resolve()): acq.sha256(path) for path in paths}


def prepare(out):
    t.old.require_fresh(out)
    no_active_trainer()
    _, _, _, files = checked_sources()
    plan = fixed_scope() | {"bindings_sha256": files}
    check_plan(plan)
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    print("GRADIENT_CONTRIBUTION PLAN SEALED; CPU-only, updates0", flush=True)


def verified_plan(out):
    plan = acq.read_sealed(out / "plan.json")
    check_plan(plan)
    _, _, _, files = checked_sources()
    if files != plan["bindings_sha256"]:
        raise ValueError("Source/code/test/plan binding changed")
    return plan


def status(out, values):
    temporary = out / f"status_{os.getpid()}.tmp"
    acq.write_new_json(temporary, values | {"pid": os.getpid(), "updated_utc": m.bulk.now(),
        "purpose": PURPOSE, "model_updates": 0, "cuda_used": False, "release_selection": "NONE"})
    os.replace(temporary, out / "run_status.json")


def run(out):
    if (out / "diagnostic.json").exists() or (out / "run_status.json").exists():
        raise ValueError("Existing run/evidence: verify or diagnose, do not repeat automatically")
    plan = verified_plan(out)
    no_active_trainer()
    approval, sources, sampler, _ = checked_sources()
    torch.set_num_threads(THREADS)
    rows, inputs, model_digests = [], [], {}
    with m.bulk.worker_lock(out), d.deterministic_runtime("cpu"):
        status(out, {"status": "running", "phase": "collect_fixed_train", "completed_model_batches": 0, "limit": 36, "error": None})
        try:
            dataset = p.inp.ApprovedTeacherDataset(p.inp.verified_approval(Path(approval["origin_approval"])), "kim_melband")
            true = m.LockedTruePool(m.bulk.OLD_LOCK, dataset.config)
            if sampler["seed"] != dataset.seed or sampler["true_lock_sha256"] != true.bound:
                raise ValueError("Input seed/true lock changed")
            batches = []
            for cursor in CURSORS:
                batch = s.collect(dataset, true, cursor)
                if batch["x"].shape != (6, 2, 89856) or batch["v"].shape != batch["x"].shape:
                    raise ValueError("Original crop layout changed")
                hashes = [m.pilot.wave_digest(wave) for wave in batch["x"]]
                if hashes != [meta["input_pcm_sha256"] for meta in batch["metadata"]]:
                    raise ValueError("TRAIN input PCM differs")
                inputs.append({"cursor": cursor, "input_sha256": hashes,
                               "target_sha256": [m.pilot.wave_digest(wave) for wave in batch["v"]], "metadata": batch["metadata"]})
                batches.append(batch)
            wa = torch.from_numpy(m.core.t09.make_analysis_matrix())
            gs = torch.from_numpy(m.core.t09.make_synthesis_matrix())
            for name in MODELS:
                net = m.frozen_factory(approval["source_protocol"])()
                saved = sources[name]
                net.load_state_dict(saved["model"], strict=True)
                if list(saved["parameter_names"]) != [key for key, _ in net.named_parameters()] or len(saved["modes"]) != len(list(net.modules())):
                    raise ValueError("Parameter order/module mode layout differs")
                for module, mode in zip(net.modules(), saved["modes"]):
                    module.training = mode
                if any(isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout,
                                          torch.nn.Dropout2d, torch.nn.Dropout3d)) for module in net.modules()):
                    raise ValueError("Batch-independent deterministic graph required")
                model_digests[name] = r.dev.state_digest(net.state_dict())
                if model_digests[name] != r.dev.state_digest(saved["model"]):
                    raise ValueError("Model identity differs")
                for index, batch in enumerate(batches):
                    probe = parameter_probe(net, batch, wa, gs, .2 if name == MODELS[2] else .02, independent=index == 0)
                    rows.append({"model": name, "diagnostic_counter": batch["diagnostic_counter"], **probe})
                    status(out, {"status": "running", "phase": "cpu_gradient_diagnostic", "model": name,
                        "counter": batch["diagnostic_counter"], "completed_model_batches": len(rows), "limit": 36, "error": None})
                    print(f"GRADIENT_CONTRIBUTION model={name} counter={batch['diagnostic_counter']} completed={len(rows)}/36", flush=True)
            result = {"schema": 1, "purpose": PURPOSE, "plan_sha256": acq.sha256(out / "plan.json"),
                "bindings_sha256": plan["bindings_sha256"], "rows": rows, "inputs": inputs, "model_digests": model_digests,
                "model_and_modes_unchanged": True, "gradients_populated_on_model": False, "model_updates": 0,
                "optimizer_constructed": False, "cuda_used": False, "runtime": d.runtime_identity("cpu"),
                "metric_reduction_dtype": "float64", "release_selection": "NONE", "deployment": False,
                "scope": "Fixed TRAIN parameter-gradient geometry; not Adam update direction, quality, capacity upper bound or full-library causality"}
            if not d.finite_state(result) or torch.cuda.is_initialized():
                raise ValueError("Nonfinite diagnostic or unexpected CUDA initialization")
            acq.write_new_json(out / "diagnostic.json", acq.seal(result))
            verify(out)
            status(out, {"status": "complete", "phase": "cpu_gradient_diagnostic_complete", "completed_model_batches": 36, "limit": 36, "error": None})
        except BaseException as error:
            status(out, {"status": "failed", "phase": "cpu_gradient_diagnostic", "completed_model_batches": len(rows), "limit": 36, "error": repr(error)})
            raise


def verify(out):
    plan = verified_plan(out)
    result = acq.read_sealed(out / "diagnostic.json")
    if (result["purpose"] != PURPOSE or result["plan_sha256"] != acq.sha256(out / "plan.json") or
        result["bindings_sha256"] != plan["bindings_sha256"] or len(result["rows"]) != 36 or len(result["inputs"]) != 12 or
        result["model_updates"] != 0 or result["model_and_modes_unchanged"] is not True or result["cuda_used"] is not False or
        result["optimizer_constructed"] is not False or result["gradients_populated_on_model"] is not False or
        result["deployment"] is not False or result["release_selection"] != "NONE" or not d.finite_state(result) or
        result["runtime"]["device"] != "cpu" or result["runtime"]["threads"] != THREADS or
        result["runtime"]["deterministic"] is not True or result["metric_reduction_dtype"] != "float64"):
        raise ValueError("Changed result scope/accounting/runtime")
    _, models, _, _ = checked_sources()
    if result["model_digests"] != {name: r.dev.state_digest(models[name]["model"]) for name in MODELS}:
        raise ValueError("Diagnostic model identity differs")
    for index, entry in enumerate(result["inputs"]):
        if entry["cursor"] != CURSORS[index] or len(entry["target_sha256"]) != 6 or entry["input_sha256"] != [meta["input_pcm_sha256"] for meta in entry["metadata"]]:
            raise ValueError("Changed common input recipe")
        t.validate_metadata(entry["metadata"])
    for index, row in enumerate(result["rows"]):
        counter = index % 12
        if row["model"] != MODELS[index//12] or row["diagnostic_counter"] != CURSORS[counter] or len(row["slots"]) != 6:
            raise ValueError("Changed model/counter/slot ordering")
        if [slot["metadata"] for slot in row["slots"]] != result["inputs"][counter]["metadata"]:
            raise ValueError("Models did not use the same inputs")
        active = 0
        for slot_index, slot in enumerate(row["slots"]):
            if slot["slot"] != slot_index or slot["group"] != slot_group(slot_index, slot["metadata"]):
                raise ValueError("Wrong contribution role")
            if type(slot["auxiliary"]["active"]) is not bool or (slot["auxiliary"]["active"] and slot_index >= 2):
                raise ValueError("Auxiliary role leakage")
            if not slot["auxiliary"]["active"] and (slot["auxiliary_loss_divided_by6"] != 0. or slot["auxiliary_gradient_l2"] != 0.):
                raise ValueError("Skipped slot has nonzero auxiliary")
            active += slot["auxiliary"]["active"]
        if row["active_count"] != active or row["skipped_count"] != 6-active or row["parameter_count"] <= 0:
            raise ValueError("Invalid activity/parameter accounting")
        if counter == 0:
            expected = .2 if row["model"] == MODELS[2] else .02
            if (row["independent_combination_lambda"] != expected or row["independent_combination_max_error"] is None or
                row["independent_combination_error_bound"] is None or row["independent_combination_max_error"] < 0 or
                row["independent_combination_max_error"] > row["independent_combination_error_bound"]):
                raise ValueError("Missing actual independent combination check")
        elif (row["independent_combination_lambda"] is not None or row["independent_combination_max_error"] is not None or
              row["independent_combination_error_bound"] is not None):
            raise ValueError("Changed combination probe scope")
        verify_geometry(row["geometry"])
    print("GRADIENT_CONTRIBUTION VERIFIED; CPU-only TRAIN; updates0; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    globals()[args.operation](args.out)
