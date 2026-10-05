"""Sealed TRAIN-only CPU auxiliary-component diagnosis; no optimizer or updates.

FP32 forward/backward exactly splits159's cv^2+(ca-1)^2. Only detached
gradient geometry uses FP64. Neither this diagnostic nor its plan authorizes
training, changes a loss, selects a release, or alters historical evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import torch

spec = importlib.util.spec_from_file_location("aux_component_review", Path(__file__).with_name("174_review_mel_instrumental_protection.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
t, p, m, d, acq, ROOT = r.t, r.p, r.m, r.t.d, r.acq, r.ROOT
s, k = p.q.s, t.k
PURPOSE = "NONRELEASE_TRAIN_AUXILIARY_COMPONENT_DIAGNOSTIC"
CURSORS = tuple(range(2000, 2012))
MODELS = ("origin_aux020_2500", "instrumental_weight1_3000", "instrumental_weight4_3000")
WEIGHTS = dict(zip(MODELS, (1, 1, 4)))
GROUPS = ("true_vocal_base", "instrumental_base", "pseudo_base", "residual_cv2", "accompaniment_ca1_squared")
THREADS = 2
SOURCE = t.DEFAULT_OUT
DEFAULT_OUT = ROOT / "results/auxiliary_components_20261003"
PROOF = ROOT / "results/mel_instrumental_protection_monitor_20261003/completion_review.json"
REPORT = ROOT / "reports/57_auxiliary_component_diagnostic_plan.md"
TEST = Path(__file__).with_name("_test_auxiliary_components.py")


def fixed_scope():
    return {"schema": 1, "purpose": PURPOSE, "diagnostic_only": True,
            "training_authorized": False, "cuda_used": False, "model_updates": 0,
            "cursors": list(CURSORS), "models": list(MODELS), "groups": list(GROUPS),
            "instrumental_weights": WEIGHTS, "auxiliary_lambda": .2, "normalizer": 6,
            "forward_backward_dtype": "float32", "metric_reduction_dtype": "float64",
            "threads": THREADS, "teacher": "kim_melband",
            "selection": "Fixed12 TRAIN counters2000..2011; predefined3 completed models; no score selection",
            "unique_input_batches": 12, "model_batches": 36, "slot_checks": 216,
            "release_selection": "NONE", "deployment": False}


def check_plan(doc):
    if any(doc.get(key) != value or type(doc.get(key)) is not type(value)
           for key, value in fixed_scope().items()):
        raise ValueError("Changed zero-update diagnostic scope")
    if any(type(c) is not int for c in doc["cursors"]) or any(type(w) is not int for w in doc["instrumental_weights"].values()):
        raise ValueError("Exact integer counters/weights required")


def split_auxiliary(predicted, mix, vocal, meta):
    # Validate roles/layout/finite values and obtain the immutable original
    # scalar from159 before constructing the diagnostic split. No new kernel
    # is substituted into training. The active calculation order is identical.
    original, info = k.source_projection_auxiliary(predicted, mix, vocal, meta)
    if not info["active"]:
        zero = predicted.sum()*0
        return zero, zero, original, info
    a, v = (mix-vocal).detach().flatten(), vocal.detach().flatten()
    na, nv = a.norm(), v.norm()
    u, w = a/na, v/nv
    rho = u@w
    determinant = 1-rho.square()
    y = (mix-predicted).flatten()
    ya, yv = y@u, y@w
    ca = (ya-rho*yv)/(na*determinant)
    cv = (yv-rho*ya)/(nv*determinant)
    residual, accompaniment = cv.square(), (ca-1).square()
    if not torch.equal(residual+accompaniment, original):
        raise ValueError("Split scalar is not bit-identical to159")
    return residual, accompaniment, original, info


def slot_losses(net, xb, vb, wa, gs, meta):
    spectrum = m.core.stft_batch(xb)
    bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
    mask = (net(bands)[:, :2]+1)/2
    base, _, pv = m.fit.reconstruction_loss(mask, spectrum, xb, vb, gs, 96, 44)
    region = m.fit.suite.scoring_slice(xb.shape[-1], 96)
    residual, accompaniment, original, info = split_auxiliary(pv[0, :, region], xb[0, :, region], vb[0, :, region], meta)
    return base, residual, accompaniment, original, info


def group_for_slot(index, meta):
    if (type(index) is not int or not 0 <= index < 6 or meta.get("domain") != m.DOMAINS[index] or
        meta.get("role") != ("train" if index < 3 else "pseudo_label_train_candidate")):
        raise ValueError("Original TRAIN six-slot roles required")
    return GROUPS[0] if index < 2 else GROUPS[1] if index == 2 else GROUPS[2]


def geometry(vectors, weight):
    if type(weight) is not int or weight not in (1, 4) or set(vectors) != set(GROUPS):
        raise ValueError("All5 original component gradients and weight1/4 required")
    shape = vectors[GROUPS[0]].shape
    if len(shape) != 1 or not shape[0] or any(v.shape != shape or v.dtype != torch.float32 or v.device.type != "cpu" or
                                            not torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError("Aligned finite CPU FP32 vectors required")
    values = {key: value.detach().double() for key, value in vectors.items()}
    norms = {key: float(value.norm()) for key, value in values.items()}
    pairs = {}
    for left, right in itertools.combinations(GROUPS, 2):
        dot = float(values[left]@values[right])
        cosine = dot/(norms[left]*norms[right]) if norms[left] and norms[right] else None
        if cosine is not None and abs(cosine) > 1+1e-12:
            raise ValueError("Invalid detached gradient geometry")
        pairs[f"{left}|{right}"] = {"dot": dot, "cosine": None if cosine is None else max(-1., min(1., cosine))}
    coefficients = (1., float(weight), 1., .2, .2)
    combined = sum(values[key]*coefficient for key, coefficient in zip(GROUPS, coefficients))
    return {"group_l2": norms, "pairs": pairs, "instrumental_weight": weight,
            "combined_coefficients": list(coefficients), "combined_l2": float(combined.norm()),
            "gradient_sha256": {key: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest() for key, value in vectors.items()},
            "metric_reduction_dtype": "float64"}


def verify_geometry(doc):
    if set(doc["group_l2"]) != set(GROUPS) or set(doc["gradient_sha256"]) != set(GROUPS):
        raise ValueError("Changed geometry groups")
    norms = doc["group_l2"]
    weight = doc["instrumental_weight"]
    coefficients = [1., float(weight), 1., .2, .2]
    if (type(weight) is not int or weight not in (1, 4) or doc["combined_coefficients"] != coefficients or
        doc["metric_reduction_dtype"] != "float64" or any(type(n) is not float or not math.isfinite(n) or n < 0 for n in norms.values()) or
        any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in doc["gradient_sha256"].values())):
        raise ValueError("Invalid gradient accounting")
    expected_pairs = {f"{a}|{b}" for a, b in itertools.combinations(GROUPS, 2)}
    if set(doc["pairs"]) != expected_pairs:
        raise ValueError("Missing component pair")
    for key, pair in doc["pairs"].items():
        a, b = key.split("|")
        if type(pair["dot"]) is not float or not math.isfinite(pair["dot"]):
            raise ValueError("Nonfinite geometry")
        expected = pair["dot"]/(norms[a]*norms[b]) if norms[a] and norms[b] else None
        if expected is None:
            if pair["cosine"] is not None or pair["dot"] != 0.:
                raise ValueError("Zero norm is unavailable, not a conflict")
        elif abs(expected) > 1+1e-12 or not math.isclose(pair["cosine"], max(-1., min(1., expected)), abs_tol=1e-12):
            raise ValueError("Cosine differs from retained dots")
    squared = sum((coefficients[i]*norms[key])**2 for i, key in enumerate(GROUPS))
    squared += 2*sum(coefficients[i]*coefficients[j]*doc["pairs"][f"{a}|{b}"]["dot"]
                     for (i, a), (j, b) in itertools.combinations(enumerate(GROUPS), 2))
    if squared < -1e-10 or not math.isclose(doc["combined_l2"], math.sqrt(max(0., squared)), rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("Combined norm differs from component geometry")


def compare_gradients(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-7)
    return {"max_error": float((actual-expected).abs().max()), "error_bound": 2e-7+2e-4*float(expected.abs().max())}


def parameter_probe(net, batch, wa, gs, weight, independent=False):
    params = tuple(net.parameters())
    if (type(weight) is not int or weight not in (1, 4) or not params or
        any(param.grad is not None or param.dtype != torch.float32 or param.device.type != "cpu" for param in params)):
        raise ValueError("Unpopulated FP32 CPU gradient storage and weight1/4 required")
    t.validate_metadata(batch["metadata"])
    before = r.dev.state_digest(net.state_dict())
    modes = [module.training for module in net.modules()]
    vectors = {key: torch.zeros(sum(param.numel() for param in params)) for key in GROUPS}
    slots = []
    for index, meta in enumerate(batch["metadata"]):
        group = group_for_slot(index, meta)
        base, residual, accompaniment, original, info = slot_losses(net, batch["x"][index:index+1], batch["v"][index:index+1], wa, gs, meta)
        active = info["active"]
        if type(active) is not bool or (active and index >= 2) or not torch.equal(residual+accompaniment, original):
            raise ValueError("Changed auxiliary split/role")
        gb = s.flatten_gradients(torch.autograd.grad(base/6, params, retain_graph=active, allow_unused=True), params)
        zero = torch.zeros_like(gb)
        gr = s.flatten_gradients(torch.autograd.grad(residual/6, params, retain_graph=True, allow_unused=True), params) if active else zero
        ga = s.flatten_gradients(torch.autograd.grad(accompaniment/6, params, retain_graph=True, allow_unused=True), params) if active else zero
        original_grad = s.flatten_gradients(torch.autograd.grad(original/6, params, allow_unused=True), params) if active else zero
        equivalence = compare_gradients(original_grad, gr+ga)
        if not active and any(float(value.detach()) != 0. for value in (residual, accompaniment, original)):
            raise ValueError("Skipped slot has nonzero auxiliary")
        vectors[group] += gb
        vectors[GROUPS[3]] += gr
        vectors[GROUPS[4]] += ga
        slots.append({"slot": index, "group": group, "metadata": meta, "auxiliary": info,
                      "base_loss_divided_by6": float(base.detach())/6,
                      "residual_loss_divided_by6": float(residual.detach())/6,
                      "accompaniment_loss_divided_by6": float(accompaniment.detach())/6,
                      "original_auxiliary_loss_divided_by6": float(original.detach())/6,
                      "base_gradient_l2": float(gb.double().norm()), "residual_gradient_l2": float(gr.double().norm()),
                      "accompaniment_gradient_l2": float(ga.double().norm()), "split_gradient_equivalence": equivalence})
    result = {"geometry": geometry(vectors, weight), "slots": slots,
              "active_count": sum(slot["auxiliary"]["active"] for slot in slots),
              "skipped_count": sum(not slot["auxiliary"]["active"] for slot in slots),
              "parameter_count": vectors[GROUPS[0]].numel(), "independent_combination": None}
    if independent:
        actual = torch.zeros_like(vectors[GROUPS[0]])
        for index, meta in enumerate(batch["metadata"]):
            base, _, _, original, info = slot_losses(net, batch["x"][index:index+1], batch["v"][index:index+1], wa, gs, meta)
            # Actual original172 operation order, including fixed denominator6.
            loss, _ = t.w.combine_slot_loss(base, original, meta, index, weight, info)
            actual += s.flatten_gradients(torch.autograd.grad(loss, params, allow_unused=True), params)
        expected = vectors[GROUPS[0]]+weight*vectors[GROUPS[1]]+vectors[GROUPS[2]]+.2*(vectors[GROUPS[3]]+vectors[GROUPS[4]])
        result["independent_combination"] = compare_gradients(actual, expected)
    if before != r.dev.state_digest(net.state_dict()) or modes != [module.training for module in net.modules()] or any(param.grad is not None for param in params):
        raise ValueError("Diagnostic mutated weights/modes/parameter.grad")
    verify_geometry(result["geometry"])
    return result


def no_active_worker():
    command = """Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '(134_generate_teacher_library|139_generate_paired_htdemucs|150_train_paired_exploration|154_train_mel_weak_weight|161_train_mel_source_aux|166_train_mel_source_strength|172_train_mel_instrumental_protection|169_diagnose_gradient_contributions|175_diagnose_auxiliary_components)[.]py[\" ]+(run|train|smoke)' } | Select-Object -ExpandProperty ProcessId"""
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    # This check is made before run; for run ignore only this process/shim chain.
    ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    own = {os.getpid(), os.getppid()}
    if any(pid not in own for pid in ids):
        raise ValueError("Active teacher/student/diagnostic worker; preserve it and postpone")


def checked_sources():
    approval = p.verified_approval(p.DEFAULT_APPROVAL)
    completion_path = SOURCE / "completion.json"
    completion = acq.read_sealed(completion_path)
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    exit_path, launch_path = ROOT / proof["detached_exit_receipt"], ROOT / proof["launch_receipt"]
    exit_doc = json.loads(exit_path.read_text(encoding="utf-8-sig"))
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    if (proof["purpose"] != "NONRELEASE_MEL_INSTRUMENTAL_PROTECTION_COMPLETION_REVIEW" or
        proof["terminal_status"] != "complete" or proof["final_step"] != 3000 or proof["additional_common_updates"] != 500 or
        proof["active_matching_training_processes"] or proof["detached_exit_code"] != 0 or
        proof["launch_and_shim_receipt_match"] is not True or exit_doc["exit_code"] != 0 or
        Path(exit_doc["launch_receipt"]) != launch_path or launch["probe_only"] or Path(launch["out"]) != SOURCE or
        exit_doc["launcher_process_id"] != launch["worker"]["LauncherProcessId"] or
        completion["step"] != 3000 or completion["additional_steps"] != 500 or completion["binding"] != t.binding(p.DEFAULT_APPROVAL) or
        proof["verification"]["trainer_verify_exit_code"] != 0 or proof["verification"]["summary_exit_code"] != 0 or
        proof["verification"]["pending_partial_steps"] or proof["verification"]["unchanged_manifest_tracks"] != 31 or
        proof["verification"]["unchanged_manifest_views"] != 177):
        raise ValueError("Exact independent normal500 completion required")
    retained = ["results/mel_instrumental_protection_monitor_20261003/completion_verify.log",
                "results/mel_instrumental_protection_monitor_20261003/summary3000.log",
                "results/mel_instrumental_protection_review_20261003/summary_step_3000/review.json",
                str(completion_path.relative_to(ROOT)).replace("\\", "/"), str(exit_path.relative_to(ROOT)).replace("\\", "/")]
    if acq.sha256(launch_path) != proof["launch_receipt_sha256"] or any(acq.sha256(ROOT / name) != proof["evidence_sha256"][name] for name in retained):
        raise ValueError("Changed retained independent evidence")
    files = [ROOT / name for name in retained]+[PROOF, launch_path, p.DEFAULT_APPROVAL,
        ROOT / "reports/56_mel_instrumental_protection_completion.md", REPORT, TEST, Path(__file__),
        Path(r.__file__), Path(t.__file__), Path(p.__file__), Path(s.__file__), Path(k.__file__), Path(t.w.__file__)]
    models = {}
    for step in (2500, 3000):
        receipt_path = SOURCE / f"checkpoint_{step:04d}.json"
        receipt = acq.read_sealed(receipt_path)
        state_path = SOURCE / receipt["checkpoint"]
        state = m.load_checked_checkpoint(state_path, receipt["sha256"])
        if (state["purpose"] != t.PURPOSE or state["binding"] != completion["binding"] or state["step"] != step or
            state["limit"] != 3000 or state["smoke"] is not False or state["deployment_authorized"] is not False or
            state["sampler"]["cursor"] != step or state["schedule"]["step"] != step or state["schedule"]["last_validation"] != step or
            state["schedule"]["stopped_at"] != (3000 if step == 3000 else None) or
            state["legacy_stop_events"] != ([] if step == 2500 else [2750, 3000]) or
            state["arm_roles"] != p.ROLES or state["arm_lambdas"] != p.LAMBDAS or state["arm_instrumental_weights"] != p.WEIGHTS or
            set(state["arms"]) != set(t.ARMS) or any(saved["updates"] != step for saved in state["arms"].values()) or not d.finite_state(state)):
            raise ValueError("Exact full model source/roles/stops required")
        if step == 2500:
            if not m.equal_state(state["arms"][t.ARMS[0]], state["arms"][t.ARMS[1]]):
                raise ValueError("Fork origins differ")
            models[MODELS[0]] = state["arms"][t.ARMS[1]]
            sampler = state["sampler"]
        else:
            if completion["final_checkpoint"] != receipt or acq.sha256(state_path) != proof["evidence_sha256"][str(state_path.relative_to(ROOT)).replace("\\", "/")]:
                raise ValueError("Completion/checkpoint evidence differs")
            models[MODELS[1]], models[MODELS[2]] = (state["arms"][arm] for arm in t.ARMS)
        files += [receipt_path, state_path]
    return approval, models, sampler, {str(path.resolve()): acq.sha256(path) for path in files}


def prepare(out):
    t.old.require_fresh(out)
    no_active_worker()
    _, _, _, files = checked_sources()
    plan = fixed_scope() | {"bindings_sha256": files}
    check_plan(plan)
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    print("AUX_COMPONENT PLAN SEALED; fixed36 CPU model-batches; updates0", flush=True)


def verified_plan(out):
    plan = acq.read_sealed(out / "plan.json")
    check_plan(plan)
    _, _, _, files = checked_sources()
    if plan["bindings_sha256"] != files:
        raise ValueError("Changed source/code/test/plan binding")
    return plan


def status(out, values):
    temporary = out / f"status_{os.getpid()}.tmp"
    acq.write_new_json(temporary, values | {"pid": os.getpid(), "updated_utc": m.bulk.now(), "purpose": PURPOSE,
        "model_updates": 0, "cuda_used": False, "release_selection": "NONE"})
    os.replace(temporary, out / "run_status.json")


def run(out):
    if (out / "diagnostic.json").exists() or (out / "run_status.json").exists():
        raise ValueError("Existing run/evidence: verify or diagnose, never repeat automatically")
    no_active_worker()
    plan = verified_plan(out)
    approval, models, sampler, _ = checked_sources()
    torch.set_num_threads(THREADS)
    rows, inputs, model_digests = [], [], {}
    with m.bulk.worker_lock(out), d.deterministic_runtime("cpu"):
        status(out, {"status": "running", "phase": "collect_fixed_train", "completed_model_batches": 0, "limit": 36, "error": None})
        try:
            dataset = p.inp.ApprovedTeacherDataset(p.inp.verified_approval(Path(approval["origin_approval"])), "kim_melband")
            true = m.LockedTruePool(m.bulk.OLD_LOCK, dataset.config)
            if sampler["seed"] != dataset.seed or sampler["true_lock_sha256"] != true.bound:
                raise ValueError("Input seed/true lock differs")
            batches = []
            for cursor in CURSORS:
                batch = s.collect(dataset, true, cursor)
                if batch["x"].shape != (6, 2, 89856) or batch["v"].shape != batch["x"].shape:
                    raise ValueError("Original crop layout changed")
                hashes = [m.pilot.wave_digest(wave) for wave in batch["x"]]
                if hashes != [meta["input_pcm_sha256"] for meta in batch["metadata"]]:
                    raise ValueError("TRAIN PCM differs")
                inputs.append({"cursor": cursor, "input_sha256": hashes, "target_sha256": [m.pilot.wave_digest(wave) for wave in batch["v"]], "metadata": batch["metadata"]})
                batches.append(batch)
            wa = torch.from_numpy(m.core.t09.make_analysis_matrix())
            gs = torch.from_numpy(m.core.t09.make_synthesis_matrix())
            for name in MODELS:
                net = m.frozen_factory(approval["source_protocol"])()
                saved = models[name]
                net.load_state_dict(saved["model"], strict=True)
                if saved["parameter_names"] != [key for key, _ in net.named_parameters()] or len(saved["modes"]) != len(list(net.modules())):
                    raise ValueError("Parameter order/mode layout differs")
                for module, mode in zip(net.modules(), saved["modes"]):
                    module.training = mode
                if any(isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)) for module in net.modules()):
                    raise ValueError("Deterministic batch-independent graph required")
                model_digests[name] = r.dev.state_digest(net.state_dict())
                if model_digests[name] != r.dev.state_digest(saved["model"]):
                    raise ValueError("Model identity differs")
                for index, batch in enumerate(batches):
                    probe = parameter_probe(net, batch, wa, gs, WEIGHTS[name], independent=index == 0)
                    rows.append({"model": name, "diagnostic_counter": batch["diagnostic_counter"], **probe})
                    status(out, {"status": "running", "phase": "cpu_gradient_diagnostic", "model": name,
                        "counter": batch["diagnostic_counter"], "completed_model_batches": len(rows), "limit": 36, "error": None})
                    print(f"AUX_COMPONENT model={name} counter={batch['diagnostic_counter']} completed={len(rows)}/36", flush=True)
            result = {"schema": 1, "purpose": PURPOSE, "plan_sha256": acq.sha256(out / "plan.json"),
                "bindings_sha256": plan["bindings_sha256"], "rows": rows, "inputs": inputs, "model_digests": model_digests,
                "model_and_modes_unchanged": True, "gradients_populated_on_model": False, "model_updates": 0,
                "optimizer_constructed": False, "cuda_used": False, "runtime": d.runtime_identity("cpu"),
                "metric_reduction_dtype": "float64", "release_selection": "NONE", "deployment": False,
                "scope": "Fixed TRAIN local parameter-gradient components; not Adam update direction, quality, full-library causality or LF44 capacity proof"}
            if not d.finite_state(result) or torch.cuda.is_initialized():
                raise ValueError("Nonfinite result or unexpected CUDA initialization")
            acq.write_new_json(out / "diagnostic.json", acq.seal(result))
            verify(out)
            status(out, {"status": "complete", "phase": "cpu_gradient_diagnostic_complete", "completed_model_batches": 36, "limit": 36, "error": None})
        except BaseException as error:
            status(out, {"status": "failed", "phase": "cpu_gradient_diagnostic", "completed_model_batches": len(rows), "limit": 36, "error": repr(error)})
            raise


def verify(out):
    plan = verified_plan(out)
    result = acq.read_sealed(out / "diagnostic.json")
    if (result["purpose"] != PURPOSE or result["plan_sha256"] != acq.sha256(out / "plan.json") or result["bindings_sha256"] != plan["bindings_sha256"] or
        len(result["rows"]) != 36 or len(result["inputs"]) != 12 or result["model_updates"] != 0 or result["model_and_modes_unchanged"] is not True or
        result["cuda_used"] is not False or result["optimizer_constructed"] is not False or result["gradients_populated_on_model"] is not False or
        result["deployment"] is not False or result["release_selection"] != "NONE" or not d.finite_state(result) or result["runtime"]["device"] != "cpu" or
        result["runtime"]["threads"] != THREADS or result["runtime"]["deterministic"] is not True or result["metric_reduction_dtype"] != "float64"):
        raise ValueError("Changed diagnostic scope/accounting/runtime")
    _, models, _, _ = checked_sources()
    if result["model_digests"] != {name: r.dev.state_digest(models[name]["model"]) for name in MODELS}:
        raise ValueError("Model identity differs")
    for i, entry in enumerate(result["inputs"]):
        if entry["cursor"] != CURSORS[i] or len(entry["target_sha256"]) != 6 or entry["input_sha256"] != [meta["input_pcm_sha256"] for meta in entry["metadata"]]:
            raise ValueError("Changed fixed shared input")
        t.validate_metadata(entry["metadata"])
    for i, row in enumerate(result["rows"]):
        model, index = MODELS[i//12], i % 12
        if row["model"] != model or row["diagnostic_counter"] != CURSORS[index] or len(row["slots"]) != 6 or row["geometry"]["instrumental_weight"] != WEIGHTS[model] or row["parameter_count"] <= 0:
            raise ValueError("Changed model/counter/slot/weight ordering")
        active = 0
        for j, slot in enumerate(row["slots"]):
            if slot["slot"] != j or slot["metadata"] != result["inputs"][index]["metadata"][j] or slot["group"] != group_for_slot(j, slot["metadata"]):
                raise ValueError("Changed input or contribution role")
            flag = slot["auxiliary"]["active"]
            if type(flag) is not bool or (flag and j >= 2):
                raise ValueError("Auxiliary role leakage")
            losses = [slot[key] for key in ("residual_loss_divided_by6", "accompaniment_loss_divided_by6", "original_auxiliary_loss_divided_by6")]
            if any(value < 0 for value in losses) or not math.isclose(sum(losses[:2]), losses[2], rel_tol=2e-7, abs_tol=1e-9):
                raise ValueError("Component scalar arithmetic differs")
            if not flag and (any(value != 0. for value in losses) or slot["residual_gradient_l2"] != 0. or slot["accompaniment_gradient_l2"] != 0.):
                raise ValueError("Skipped slot has nonzero auxiliary")
            equivalent = slot["split_gradient_equivalence"]
            if not 0 <= equivalent["max_error"] <= equivalent["error_bound"]:
                raise ValueError("Missing original159 gradient equivalence")
            active += flag
        if row["active_count"] != active or row["skipped_count"] != 6-active:
            raise ValueError("Activity accounting differs")
        combined = row["independent_combination"]
        if (index == 0 and (combined is None or not 0 <= combined["max_error"] <= combined["error_bound"])) or (index != 0 and combined is not None):
            raise ValueError("Missing fixed first-batch original-loss gradient check")
        verify_geometry(row["geometry"])
    print("AUX_COMPONENT VERIFIED;36 model-batches216 TRAIN slots; CPU-only updates0; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    globals()[args.operation](args.out)
