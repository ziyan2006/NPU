"""Fixed TRAIN cross-boundary probe: one forward, two isolated reconstructions, zero updates."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import types
import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n = load("sealed_reference184_for_cross", "184_diagnose_train_low_frequency_reference.py")
h = load("sealed_boundary188_for_cross", "188_review_mel_lf_boundary.py")
k, q, m, acq, ROOT = n.k, n.q, n.m, n.acq, n.ROOT
PURPOSE = "NONRELEASE_TRAIN_LF_CROSS_BOUNDARY_DIAGNOSTIC"
DEFAULT_OUT = ROOT / "results/train_lf_cross_boundary_20261004"
MONITOR = ROOT / "results/train_lf_cross_boundary_monitor_20261004"
TEST = Path(__file__).with_name("_test_train_lf_cross_boundary.py")
PROTOCOL = ROOT / "docs/train_lf_cross_boundary_protocol_20261004.json"
CURSORS = tuple(range(3000, 3012))
MODELS = ("source3500_full_control", "endpoint4000_lf44_control", "endpoint4000_lf32_candidate")
KILLS = (44, 32)
SAMPLES, FRAMES, THREADS = 89856, 352, 2
INTERVAL = (98 * 256, SAMPLES - 2 * 256)
SOURCE = ROOT / "results/mel_component_ablation_20261003"
ENDPOINT = ROOT / "results/mel_lf_boundary_20261003"
APPROVAL = ROOT / "results/mel_lf_boundary_gate_recovery_import_20261003/approval.json"
PROOF = ROOT / "results/mel_lf_boundary_monitor_20261003/completion_review.json"
PINNED = {
    "reports/79_mel_lf_boundary_completion.md": "dbf3d3afbe2a8c15c09eca9e936c1c94f2de4f1563f4a6d77e649777cfdf933e",
    "reports/80_train_lf_cross_boundary_plan.md": "df5b56dd95e669f72ace2ba6945a7866f8e3bce4e03ea60c364d79b6b0e8a089",
    "results/mel_lf_boundary_monitor_20261003/completion_review.json": "b863b1c4793c92d36b3d5aebfbd824f33f56289fc3624e83054791e1318c5d63",
    "results/mel_component_ablation_20261003/NONRELEASE_component_step_3500.pt": "62d3e91beeb75ecd06ff5ff204b3c1704819772a7b82458ae2c3b424277bdcfc",
    "results/mel_lf_boundary_20261003/NONRELEASE_lf_boundary_step_4000.pt": "94323f8a30fb5e5decb65d4c110fd5252d0916de68507f42af555b3f539a8f6d",
    "results/mel_lf_boundary_gate_recovery_import_20261003/approval.json": "97e6dd1d174a9dcde1dcada3b3833ffa1f2fa509c32ae249a549b9bec082e82a",
}
METRICS = ("residual_vocal_abs_gain", "joint_fit_residual_vocal_gain_db", "vocal_error_snr_db",
           "accompaniment_error_snr_db", "accompaniment_gain_error_abs", "removed_energy_relative_mix_db",
           "lf_backing_error_snr_db", "lf_backing_error_power", "lf_backing_reference_power",
           "pseudo_target_mae", "pseudo_target_mse")
canonical = n.canonical


def scope():
    return {"schema": 1, "purpose": PURPOSE, "cursors": list(CURSORS), "models": list(MODELS),
        "kill_bands": list(KILLS), "unique_input_batches": 12, "unique_input_slots": 72,
        "model_batches": 36, "model_input_slots": 216, "slot_boundary_reconstructions": 432,
        "samples": SAMPLES, "frames": FRAMES, "warmup": 96, "sample_interval": list(INTERVAL),
        "threads": THREADS, "device": "cpu", "forward_reconstruction_dtype": "float32",
        "spectrum_dtype": "complex64", "detached_statistics_dtype": "float64",
        "forward_calls_per_model_batch": 1, "model_updates": 0, "backward_used": False,
        "optimizer_constructed": False, "cuda_used": False, "audio_exported": False,
        "development_used": False, "training_authorized": False, "deployment": False,
        "release_selection": "NONE", "eligible_assessment": False,
        "interpretation": "fixed_TRAIN_local_cross_boundary_not_blind_quality_or_capacity_proof"}


def check_scope(doc):
    if any(canonical(doc.get(key)) != canonical(value) for key, value in scope().items()):
        raise ValueError("Changed fixed cross-boundary diagnostic scope")


def no_active_worker():
    n.no_active_worker()
    command = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' "
        "-and $_.CommandLine -match '(186_train_mel_lf_boundary|188_review_mel_lf_boundary|"
        "191_diagnose_train_lf_cross_boundary)[.]py[\" ]+(prepare|train|run|smoke|verify|summary|listen)' } "
        "| Select-Object -ExpandProperty ProcessId")
    reply = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                           check=True, capture_output=True, text=True)
    if any(int(line.strip()) not in {os.getpid(), os.getppid()} for line in reply.stdout.splitlines() if line.strip()):
        raise ValueError("Active training/diagnostic/review; preserve and postpone")


def module_files():
    seen, files, queue = set(), set(), [n, h]
    while queue:
        module = queue.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        filename = getattr(module, "__file__", None)
        if not filename or not Path(filename).resolve().is_relative_to(ROOT / "scripts"):
            continue
        files.add(Path(filename).resolve())
        queue.extend(value for value in vars(module).values() if isinstance(value, types.ModuleType))
    return files


def tree_digest(value):
    """Typed full-state digest, including Adam/RNG/cursor; never construct an optimizer."""
    digest = hashlib.sha256()
    def visit(item):
        digest.update((type(item).__name__ + ":").encode())
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or type(item) in (str, int, float, bool):
            digest.update(canonical(item).encode())
        else:
            raise ValueError(f"Unsupported saved state type: {type(item).__name__}")
        digest.update(b";")
    visit(value)
    return digest.hexdigest()


def geometry():
    wa = torch.from_numpy(q.core.t09.make_analysis_matrix())
    gs = torch.from_numpy(q.core.t09.make_synthesis_matrix())
    q.cpu_float(wa, (513, 128)); q.cpu_float(gs, (513, 128))
    if (wa < 0).any() or (gs < 0).any():
        raise ValueError("Original nonnegative matrices required")
    zero_bins = {str(kill): torch.where(gs[:, kill:].sum(1) == 0)[0].tolist() for kill in KILLS}
    if zero_bins != {"44": list(range(6)), "32": list(range(4))}:
        raise ValueError("Changed literal44/32 synthesis support")
    identity = {"analysis_sha256": q.suite.tensor_digest(wa), "synthesis_sha256": q.suite.tensor_digest(gs),
        "analysis_shape": [513, 128], "synthesis_shape": [513, 128], "forced_zero_fft_bins": zero_bins,
        "bin_hz": 44100 / 1024, "other_transition_bins_may_change": True,
        "native_sample_interval": list(INTERVAL), "lf_sidecar_fft_bins": list(range(6)),
        "lf_sidecar_frame_interval": [98, 350]}
    return wa, gs, identity


def describe(saved, state, arm, checkpoint):
    if saved["updates"] != state["step"] or not q.d.finite_state(state):
        raise ValueError("Complete finite model/Adam exposure required")
    group = saved["optimizer"]["param_groups"]
    if len(group) != 1 or set(saved["optimizer"]["state"]) != set(group[0]["params"]):
        raise ValueError("Incomplete saved Adam exposure")
    if any(float(value["step"]) != state["step"] for value in saved["optimizer"]["state"].values()):
        raise ValueError("Wrong Adam update exposure")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": acq.sha256(checkpoint), "internal_arm": arm,
        "source_step": state["step"], "updates": saved["updates"], "parameter_names": saved["parameter_names"],
        "source_modes": saved["modes"], "model_sha256": h.dev.state_digest(saved["model"]),
        "saved_arm_sha256": tree_digest(saved), "optimizer_sha256": tree_digest(saved["optimizer"]),
        "source_schedule_sha256": tree_digest(state["schedule"]), "source_sampler_sha256": tree_digest(state["sampler"]),
        "source_rng_sha256": tree_digest(state["rng"]), "source_legacy_stop_events": state["legacy_stop_events"],
        "source_stopped_at": state["schedule"]["stopped_at"], "diagnostic_running_modes": saved["modes"]}


def source_evidence():
    """Reuse immutable completion/verification. No old forward, scoring, or verification command."""
    if torch.cuda.is_initialized():
        raise ValueError("Fresh CPU-only process required")
    approval, expected, _, bindings = n.source_evidence()
    bindings |= {str(ROOT / name): digest for name, digest in PINNED.items()}
    q.check_bindings(bindings)
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    term, ending = proof["terminal"], proof["actual_launch_exit"]
    launch_path, exit_path = ROOT / ending["launch_path"], ROOT / ending["exit_path"]
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    exit_doc = json.loads(exit_path.read_text(encoding="utf-8-sig"))
    if (term["status"] != "complete" or term["step"] != 4000 or term["additional_steps"] != 500 or
        term["limit"] != 4000 or term["error"] is not None or term["formal_round_completed"] is not True or
        ending["match_confirmed"] is not True or not q.same_exit_record(exit_doc, ending["receipt"]) or
        canonical(exit_doc["exit_code"]) != "0" or Path(exit_doc["launch_receipt"]) != launch_path or
        launch["probe_only"] or Path(launch["out"]) != ENDPOINT or
        exit_doc["launcher_process_id"] != launch["worker"]["LauncherProcessId"] or
        proof["worker_evidence"]["alive"] is not False or proof["worker_evidence"]["active_repo_python"] or
        proof["final_verify"]["actual_native_exit_code"] != 0 or proof["summary"]["actual_native_exit_code"] != 0 or
        proof["summary"]["built_in_verify_passed"] is not True or
        (proof["summary"]["tracks"], proof["summary"]["views"]) != (31, 177) or
        proof["release_selection"] != "NONE" or proof["deployment"] is not False):
        raise ValueError("Matching actual complete500 exit0 and independent review required")
    bindings |= {str(ROOT / name): digest for name, digest in proof["bindings"].items()}
    recovered = h.p.verified_approval(APPROVAL)
    bindings |= recovered["bindings_sha256"]
    q.check_bindings(bindings)
    completion = acq.read_sealed(ENDPOINT / "completion.json")
    receipt = acq.read_sealed(ENDPOINT / "checkpoint_4000.json")
    origin_receipt = acq.read_sealed(SOURCE / "checkpoint_3500.json")
    origin_path, final_path = SOURCE / origin_receipt["checkpoint"], ENDPOINT / receipt["checkpoint"]
    origin = m.load_checked_checkpoint(origin_path, origin_receipt["sha256"])
    final = m.load_checked_checkpoint(final_path, receipt["sha256"])
    bound = h.t.binding(APPROVAL)
    if (origin["step"] != 3500 or origin["limit"] != 3500 or origin["sampler"]["cursor"] != 3500 or
        origin["schedule"]["step"] != 3500 or origin["schedule"]["stopped_at"] != 3500 or
        origin["legacy_stop_events"] != [3250, 3500] or origin["arm_roles"][h.ARMS[0]] != "aux_full_control" or
        final["purpose"] != h.t.PURPOSE or final["binding"] != bound or receipt["binding"] != bound or
        completion["binding"] != bound or final["step"] != 4000 or final["limit"] != 4000 or
        final["sampler"]["cursor"] != 4000 or final["schedule"]["step"] != 4000 or
        final["schedule"]["last_validation"] != 4000 or final["schedule"]["stopped_at"] != 4000 or
        final["origin_sha256"] != origin_receipt["sha256"] or final["source_stopped_at"] != 3500 or
        final["source_legacy_stop_events"] != [3250, 3500] or final["legacy_stop_events"] != [3750, 4000] or
        final["arm_roles"] != h.p.ROLES or final["arm_kill_bands"] != h.p.KILLS or
        final["arm_lambdas"] != h.p.LAMBDAS or final["arm_accompaniment_weights"] != h.p.CA_WEIGHTS or
        final["arm_instrumental_weights"] != h.p.WEIGHTS or final["teacher"] != "kim_melband" or
        final["smoke"] is not False or final["deployment_authorized"] is not False or
        any(completion["outputs_sha256"].get(path.name) != acq.sha256(path)
            for path in (ENDPOINT / "checkpoint_4000.json", final_path))):
        raise ValueError("Changed completed source/full-state stopping/boundary identity")
    h.check_layout_and_adam(final, origin)
    packet = acq.read_sealed(ENDPOINT / "development_step_4000.json")
    selected = {MODELS[0]: (origin, h.ARMS[0], origin_path),
                MODELS[1]: (final, h.ARMS[0], final_path), MODELS[2]: (final, h.ARMS[1], final_path)}
    models, descriptors = {}, {}
    for name, (state, arm, checkpoint) in selected.items():
        saved = state["arms"][arm]
        if state is final and h.dev.state_digest(saved["model"]) != packet["model_state_sha256"][arm]:
            raise ValueError("Completed score/model identity mismatch")
        models[name] = saved
        descriptors[name] = describe(saved, state, arm, checkpoint)
    files = module_files() | {Path(__file__).resolve(), TEST, PROTOCOL, ROOT / "reports/36_weak_weight_authorization.md",
        ROOT / "reports/72_train_low_frequency_reference_result.md", SOURCE / "checkpoint_3500.json", APPROVAL}
    bindings |= {str(path.resolve()): acq.sha256(path) for path in files}
    q.check_bindings(bindings)
    return approval, expected, models, descriptors, origin["sampler"], bindings


def check_unit_gate(path):
    gate = acq.read_sealed(path)
    if (gate["actual_exit_code"] != 0 or type(gate["actual_exit_code"]) is not int or
        gate["test_sha256"] != acq.sha256(TEST) or gate["tool_sha256"] != acq.sha256(__file__) or
        gate["draft_reviewed"] is not True or gate["tests_passed"] < 20 or
        acq.sha256(gate["log"]) != gate["log_sha256"]):
        raise ValueError("Actual new unit exit0 and draft review required before sealing")
    log = Path(gate["log"]).read_text(encoding="utf-8-sig")
    if not re.search(r"Ran " + str(gate["tests_passed"]) + r" tests? in", log) or not re.search(r"(?m)^OK\s*$", log):
        raise ValueError("Complete passing native unittest output required")
    return gate


def prepare(out, unit_evidence):
    out = q.fresh_output(out)
    no_active_worker(); q.check_disk()
    gate = check_unit_gate(unit_evidence)
    approval, expected, _, descriptors, sampler, bindings = source_evidence()
    wa, gs, geom = geometry()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if canonical(protocol) != canonical(scope()):
        raise ValueError("Protocol must exactly match the predeclared fixed scope")
    torch.set_num_threads(THREADS)
    with q.d.deterministic_runtime("cpu"):
        runtime = q.d.runtime_identity("cpu")
    prior_runtime = acq.read_sealed(k.DEFAULT_OUT / "diagnostic.json")["runtime"]
    if canonical(runtime) != canonical(prior_runtime) or torch.cuda.is_initialized():
        raise ValueError("Strict original CPU FP32 runtime required")
    bindings |= {str(Path(unit_evidence).resolve()): acq.sha256(unit_evidence), gate["log"]: gate["log_sha256"]}
    plan = scope() | {"bindings_sha256": bindings, "expected_inputs": expected, "models_identity": descriptors,
        "source_sampler_identity": tree_digest(sampler), "geometry": geom, "expected_runtime": runtime,
        "unit_gate_sha256": acq.sha256(unit_evidence), "input_reader": "sealed182.collect_fixed_train_same_177_inputaudit",
        "source_mode_policy": "preserve_saved_modes_in_isolated_copies_no_BN_or_dropout"}
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    print("CROSS_BOUNDARY_PREPARED; CPU-only zero updates; NONRELEASE", flush=True)


def raw_mask(net, bands):
    q.cpu_float(bands, (6, 2, 128, FRAMES))
    output = net(bands)  # Exactly one batched forward, shared by both reconstructions.
    q.cpu_float(output)
    if output.ndim != 4 or output.shape[:2] != (6, 4) or output.shape[2:] != (128, FRAMES):
        raise ValueError("Changed original batched output layout")
    mask = (output[:, :2] + 1) / 2
    q.cpu_float(mask, (6, 2, 128, FRAMES))
    if (mask < 0).any() or (mask > 1).any():
        raise ValueError("Original bounded mask required")
    return mask


def paired_reconstruction(net, bands, spectrum, gs):
    q.check_net(net); q.cpu_float(gs, (513, 128))
    if spectrum.device.type != "cpu" or spectrum.dtype != torch.complex64 or spectrum.shape != (6, 2, 513, FRAMES) or not torch.isfinite(spectrum).all():
        raise ValueError("Original CPU complex64 batched spectrum required")
    original_bands, original_spectrum = bands.clone(), spectrum.clone()
    with q.readonly_model(net), torch.no_grad():
        mask = raw_mask(net, bands)
        original_mask = mask.clone()
        outputs = {str(kill): q.core.product_vocal(spectrum, mask, gs, SAMPLES, kill) for kill in KILLS}
        for wave in outputs.values():
            q.cpu_float(wave, (6, 2, SAMPLES))
        if (not torch.equal(mask, original_mask) or not torch.equal(bands, original_bands) or
            not torch.equal(spectrum, original_spectrum) or outputs["44"].data_ptr() == outputs["32"].data_ptr()):
            raise ValueError("Input/mask changed or cross-boundary wave alias")
    return outputs, {"unprotected_mask_sha256": q.suite.tensor_digest(mask), "forward_calls": 1,
        "same_raw_mask": True, "inputs_mask_unchanged": True, "weights_modes_rng_grad_unchanged": True,
        "boundary_waves_no_alias": True}


def bucket(meta):
    if meta["domain"] == "pseudo":
        if meta["role"] != "pseudo_label_train_candidate":
            raise ValueError("Pseudo role changed")
        return "pseudo_not_final_truth"
    if meta["role"] != "train" or meta["domain"] not in ("musdb", "mir1k", "instrumental"):
        raise ValueError("Fixed true TRAIN roles required")
    if meta["domain"] == "instrumental":
        return "instrumental_zero_reference"
    db = meta["vocal_db"]
    if type(db) is not int or db > 0:
        raise ValueError("Original gainmix activity identity required")
    return meta["domain"] + ("_native" if db == 0 else "_weak")


def activity(mix, vocal):
    """Original FP32 source geometry, descriptive only, no new gate or sampling."""
    q.cpu_float(mix, (1, 2, SAMPLES)); q.cpu_float(vocal, (1, 2, SAMPLES))
    start, end = INTERVAL
    v = vocal[..., start:end].flatten()
    a = (mix[..., start:end] - vocal[..., start:end]).flatten()
    vr, ar = v.square().mean().sqrt(), a.square().mean().sqrt()
    gram = None
    if float(vr) > 1e-4 and float(ar) > 1e-4:
        gram = float(1 - ((a / a.norm()) @ (v / v.norm())).square())
    return {"vocal_rms_fp32": float(vr), "accompaniment_rms_fp32": float(ar), "gram_determinant_fp32": gram,
        "original_source_geometry_active": gram is not None and gram >= 1e-3,
        "exact_zero_vocal_reference": bool((v == 0).all()), "descriptive_only": True}


def metrics(predicted, mix, vocal, meta):
    for value in (predicted, mix, vocal):
        q.cpu_float(value, (1, 2, SAMPLES))
    start, end = INTERVAL
    result = dict.fromkeys(METRICS)
    kind = bucket(meta)
    if kind == "pseudo_not_final_truth":
        delta = (predicted[..., start:end].detach().double() - vocal[..., start:end].detach().double())
        result["pseudo_target_mae"], result["pseudo_target_mse"] = float(delta.abs().mean()), float(delta.square().mean())
        skip = "pseudo_target_not_final_truth_no_true_source_projection"
    else:
        original = q.suite.separation_metrics(predicted[..., start:end], mix[..., start:end], vocal[..., start:end])
        result.update({key: original[key] for key in METRICS if key in original})
        result.update(h.t.low_frequency_backing_metrics(predicted, mix, vocal))
        skip = ("zero_vocal_reference" if not bool(vocal[..., start:end].any()) else
                "legacy_joint_projection_ill_conditioned_or_inactive" if original["residual_vocal_abs_gain"] is None else None)
    if not q.d.finite_state(result):
        raise ValueError("Nonfinite detached metrics")
    return {"metrics": result, "projection_skip_reason": skip,
        "reference_kind": "pseudo_label_not_final_truth" if kind == "pseudo_not_final_truth" else "true_TRAIN",
        "sample_interval": list(INTERVAL), "lf_sidecar_is_descriptive": True}


def stats(values):
    kept = [value for value in values if value is not None]
    return {"n": len(kept), "mean": statistics.fmean(kept) if kept else None,
        "median": statistics.median(kept) if kept else None,
        "minimum": min(kept) if kept else None, "maximum": max(kept) if kept else None}


def summary(rows):
    result = {}
    for name in MODELS:
        result[name] = {}
        for kill in KILLS:
            slots = [slot for row in rows if row["model"] == name for slot in row["slots"]]
            grouped = {}
            for kind in ("musdb_native", "musdb_weak", "mir1k_native", "mir1k_weak", "instrumental_zero_reference", "pseudo_not_final_truth"):
                selected = [slot for slot in slots if slot["bucket"] == kind]
                grouped[kind] = {"unique_fixed_input_slots": len(selected), "coverage_missing": not selected,
                    "metrics": {key: stats([slot["boundaries"][str(kill)]["metrics"][key] for slot in selected]) for key in METRICS}}
            result[name][str(kill)] = grouped
    return result


def contrasts(rows):
    indexed = {(row["model"], row["counter"], slot["slot"]): slot for row in rows for slot in row["slots"]}
    # Each contrast is a signed sum over the SAME unique input; no partial-row ranking.
    terms = {"direct_boundary_" + name: [(1, name, "32"), (-1, name, "44")] for name in MODELS}
    for kill in ("44", "32"):
        for candidate, baseline in ((MODELS[1], MODELS[0]), (MODELS[2], MODELS[0]), (MODELS[2], MODELS[1])):
            terms[f"weights_at_{kill}_{candidate}_minus_{baseline}"] = [(1, candidate, kill), (-1, baseline, kill)]
    terms["endpoint_boundary_interaction_candidate_minus_control"] = [(1, MODELS[2], "32"), (-1, MODELS[2], "44"), (-1, MODELS[1], "32"), (1, MODELS[1], "44")]
    output = {}
    for label, signed in terms.items():
        paired = []
        for cursor in CURSORS:
            for slot_id in range(6):
                reference = indexed[MODELS[0], cursor, slot_id]
                values = {}
                for metric in METRICS:
                    operands = [indexed[name, cursor, slot_id]["boundaries"][kill]["metrics"][metric] for _, name, kill in signed]
                    values[metric] = None if any(value is None for value in operands) else sum(sign * value for (sign, _, _), value in zip(signed, operands))
                paired.append({"counter": cursor, "slot": slot_id, "bucket": reference["bucket"], "delta": values})
        output[label] = {"terms": [list(item) for item in signed], "paired_slots": paired,
            "groups": {kind: {key: stats([row["delta"][key] for row in paired if row["bucket"] == kind]) for key in METRICS}
                       for kind in sorted({row["bucket"] for row in paired})},
            "interpretation": "same_fixed_TRAIN_support_local_descriptive_not_whole_library_causal_proof"}
    return output


def check_input(batch, cursor, expected):
    if type(cursor) is not int or cursor not in CURSORS or batch["diagnostic_counter"] != cursor:
        raise ValueError("Fixed TRAIN3000..3011 only")
    q.cpu_float(batch["x"], (6, 2, SAMPLES)); q.cpu_float(batch["v"], (6, 2, SAMPLES))
    q.t.validate_metadata(batch["metadata"])
    actual = k.input_entry(batch, cursor)
    if canonical(actual) != canonical(expected):
        raise ValueError("Changed real PCM/target/metadata/role versus sealed182 inputs")
    if bool(batch["v"][2].any()):
        raise ValueError("Original instrumental reference must be exactly zero")
    return actual


def validate_result(plan, result):
    check_scope(plan); check_scope(result)
    for key in ("geometry", "expected_runtime", "expected_inputs", "models_identity", "bindings_sha256", "input_reader", "source_sampler_identity"):
        if canonical(result.get(key)) != canonical(plan[key]):
            raise ValueError(f"Changed fixed result identity: {key}")
    if (len(result["rows"]) != 36 or set(result["row_sha256"]) != {f"row_{index:02d}.json" for index in range(36)} or
        result["completed_model_batches"] != 36 or result["completed_model_input_slots"] != 216 or
        result["completed_slot_boundary_reconstructions"] != 432 or result["cpu_rng_unchanged"] is not True or
        result["source_full_states_unchanged"] is not True or result["input_target_metadata_unchanged"] is not True or
        canonical(result["runtime"]) != canonical(plan["expected_runtime"]) or canonical(result["inputs"]) != canonical(plan["expected_inputs"])):
        raise ValueError("Complete36/216/432 with unchanged states/runtime/inputs required")
    for index, row in enumerate(result["rows"]):
        name, cursor = MODELS[index // 12], CURSORS[index % 12]
        entry = plan["expected_inputs"][index % 12]
        identity = plan["models_identity"][name]
        if (row["model"] != name or type(row["counter"]) is not int or row["counter"] != cursor or len(row["slots"]) != 6 or
            row["model_sha256"] != identity["model_sha256"] or row["source_modes"] != identity["source_modes"] or
            row["diagnostic_running_modes"] != identity["source_modes"] or row["forward_calls"] != 1 or
            any(row[key] is not True for key in ("same_raw_mask", "inputs_mask_unchanged", "weights_modes_rng_grad_unchanged", "boundary_waves_no_alias")) or
            type(row["seconds"]) is not float or not math.isfinite(row["seconds"]) or row["seconds"] < 0):
            raise ValueError("Changed batched forward accounting/identity/state")
        for j, slot in enumerate(row["slots"]):
            if (type(slot["slot"]) is not int or slot["slot"] != j or canonical(slot["metadata"]) != canonical(entry["metadata"][j]) or
                slot["input_sha256"] != entry["input_sha256"][j] or slot["target_sha256"] != entry["target_sha256"][j] or
                slot["metadata_sha256"] != hashlib.sha256(canonical(entry["metadata"][j]).encode()).hexdigest() or
                slot["bucket"] != bucket(entry["metadata"][j]) or set(slot["boundaries"]) != {"44", "32"} or
                slot["activity"]["descriptive_only"] is not True):
                raise ValueError("Changed slot hash/role/support")
            for kill in ("44", "32"):
                boundary = slot["boundaries"][kill]
                if (boundary["kill_bands"] != int(kill) or canonical(boundary["sample_interval"]) != canonical(list(INTERVAL)) or
                    set(boundary["metrics"]) != set(METRICS) or boundary["lf_sidecar_is_descriptive"] is not True or
                    not re.fullmatch("[0-9a-f]{64}", boundary["predicted_pcm_sha256"])):
                    raise ValueError("Changed isolated reconstruction/support/metric schema")
                if j == 2 and (boundary["projection_skip_reason"] != "zero_vocal_reference" or
                    boundary["metrics"]["residual_vocal_abs_gain"] is not None or boundary["metrics"]["vocal_error_snr_db"] is not None):
                    raise ValueError("Zero instrumental reference cannot make valid vocal ratio")
                if j >= 3 and (boundary["reference_kind"] != "pseudo_label_not_final_truth" or
                    boundary["metrics"]["residual_vocal_abs_gain"] is not None):
                    raise ValueError("Pseudo target must not become final true-source projection")
    if (canonical(result["summary"]) != canonical(summary(result["rows"])) or
        canonical(result["contrasts"]) != canonical(contrasts(result["rows"])) or not q.d.finite_state(result)):
        raise ValueError("Changed/nonfinite full paired summaries")


def status(out, phase, completed, error=None):
    doc = {"purpose": PURPOSE, "status": "failed" if error else "complete" if phase == "complete" else "running",
        "phase": phase, "pid": os.getpid(), "utc": m.bulk.now(), "completed_model_batches": completed,
        "completed_model_input_slots": completed * 6, "completed_slot_boundary_reconstructions": completed * 12,
        "total_model_batches": 36, "model_updates": 0, "cuda_used": torch.cuda.is_initialized(), "error": error}
    temp = out / "run_status.tmp.json"
    temp.write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(out / "run_status.json")


def run(out):
    plan = acq.read_sealed(out / "plan.json")
    check_scope(plan); q.check_bindings(plan["bindings_sha256"])
    q.reject_existing_run(out); no_active_worker(); q.check_disk()
    completed = 0
    torch.set_num_threads(THREADS)
    initial_rng = m.capture_rng("cpu")
    try:
        status(out, "collect_fixed_train", completed)
        approval, expected, saved, descriptors, sampler, _ = source_evidence()
        if canonical(descriptors) != canonical(plan["models_identity"]) or tree_digest(sampler) != plan["source_sampler_identity"]:
            raise ValueError("Source full-state identity changed")
        full_before = {name: tree_digest(value) for name, value in saved.items()}
        dataset, true = k.dataset_pair(approval, sampler)
        batches, entries = [], []
        for cursor, target in zip(CURSORS, expected):
            batch = k.collect_fixed_train(dataset, true, cursor)
            entries.append(check_input(batch, cursor, target)); batches.append(batch)
            if not m.equal_state(initial_rng, m.capture_rng("cpu")) or torch.cuda.is_initialized():
                raise ValueError("Input collection changed RNG/CUDA")
        acq.write_new_json(out / "inputs.json", acq.seal({"purpose": PURPOSE, "inputs": entries, "unique_slots": 72}))
        construction_rng = m.capture_rng("cpu")
        try:
            factory = m.frozen_factory(approval["source_protocol"])
            nets = {name: factory() for name in MODELS}
        finally:
            m.restore_rng(construction_rng, "cpu")
        for name, net in nets.items():
            net.load_state_dict(saved[name]["model"], strict=True)
            if [key for key, _ in net.named_parameters()] != saved[name]["parameter_names"] or len(list(net.modules())) != len(saved[name]["modes"]):
                raise ValueError("Graph/parameter/mode layout changed")
            for module, mode in zip(net.modules(), saved[name]["modes"]):
                module.training = mode
            q.check_net(net)
        wa, gs, geom = geometry()
        if canonical(geom) != canonical(plan["geometry"]):
            raise ValueError("Original matrix geometry changed")
        rows, hashes = [], {}
        with q.d.deterministic_runtime("cpu"), torch.no_grad():
            runtime = q.d.runtime_identity("cpu")
            if canonical(runtime) != canonical(plan["expected_runtime"]):
                raise ValueError("Strict original CPU runtime changed")
            for name in MODELS:
                for batch_index, (cursor, batch) in enumerate(zip(CURSORS, batches)):
                    q.check_disk()
                    status(out, name + "_counter_" + str(cursor), completed)
                    started = time.perf_counter()
                    before = check_input(batch, cursor, expected[batch_index])
                    spectrum = q.core.stft_batch(batch["x"])
                    bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
                    waves, invariant = paired_reconstruction(nets[name], bands, spectrum, gs)
                    slots = []
                    for j, meta in enumerate(batch["metadata"]):
                        x, v = batch["x"][j:j+1], batch["v"][j:j+1]
                        slot = {"slot": j, "metadata": meta, "bucket": bucket(meta),
                            "metadata_sha256": hashlib.sha256(canonical(meta).encode()).hexdigest(),
                            "input_sha256": before["input_sha256"][j], "target_sha256": before["target_sha256"][j],
                            "activity": activity(x, v), "boundaries": {}}
                        for kill in KILLS:
                            predicted = waves[str(kill)][j:j+1]
                            slot["boundaries"][str(kill)] = metrics(predicted, x, v, meta) | {
                                "kill_bands": kill, "predicted_pcm_sha256": m.pilot.wave_digest(predicted[0])}
                        slots.append(slot)
                    if canonical(check_input(batch, cursor, expected[batch_index])) != canonical(before):
                        raise ValueError("Real input/target/metadata changed during probe")
                    if not m.equal_state(initial_rng, m.capture_rng("cpu")) or torch.cuda.is_initialized():
                        raise ValueError("Diagnostic changed RNG or initialized CUDA")
                    row = {"purpose": PURPOSE, "model": name, "counter": cursor, "slots": slots,
                        "model_sha256": descriptors[name]["model_sha256"], "source_modes": descriptors[name]["source_modes"],
                        "diagnostic_running_modes": [module.training for module in nets[name].modules()],
                        "seconds": float(time.perf_counter() - started)} | invariant
                    path = out / f"row_{completed:02d}.json"
                    acq.write_new_json(path, acq.seal(row))
                    hashes[path.name] = acq.sha256(path); rows.append(row); completed += 1
                    print(f"CROSS_BOUNDARY {completed}/36 {name} counter={cursor} seconds={row['seconds']:.3f}; updates=0", flush=True)
                    status(out, "row_committed", completed)
            unchanged = full_before == {name: tree_digest(value) for name, value in saved.items()}
            result = {key: copy.deepcopy(value) for key, value in plan.items() if key != "content_sha256"} | {
                "plan_sha256": acq.sha256(out / "plan.json"), "inputs_sha256": acq.sha256(out / "inputs.json"),
                "inputs": entries, "rows": rows, "row_sha256": hashes, "runtime": runtime,
                "completed_model_batches": completed, "completed_model_input_slots": completed * 6,
                "completed_slot_boundary_reconstructions": completed * 12,
                "cpu_rng_unchanged": m.equal_state(initial_rng, m.capture_rng("cpu")),
                "source_full_states_unchanged": unchanged, "input_target_metadata_unchanged": True,
                "summary": summary(rows), "contrasts": contrasts(rows)}
            validate_result(plan, result)
            q.check_bindings(plan["bindings_sha256"])
            acq.write_new_json(out / "diagnostic.json", acq.seal(result))
        status(out, "complete", completed)
        print("CROSS_BOUNDARY_COMPLETE36/216/432; updates=0; NONRELEASE", flush=True)
    except Exception as exc:
        status(out, "failed", completed, str(exc))
        raise


def verify(out):
    if (out / "verification.json").exists():
        raise ValueError("Existing verification must be preserved; no duplicate verify")
    no_active_worker(); q.check_disk()
    plan = acq.read_sealed(out / "plan.json")
    check_scope(plan); q.check_bindings(plan["bindings_sha256"])
    terminal = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    if (terminal["status"] != "complete" or terminal["phase"] != "complete" or terminal["error"] is not None or
        terminal["completed_model_batches"] != 36 or terminal["completed_model_input_slots"] != 216 or
        terminal["completed_slot_boundary_reconstructions"] != 432 or terminal["model_updates"] != 0 or terminal["cuda_used"] is not False):
        raise ValueError("Complete unchanged CPU36/216/432 required")
    result = acq.read_sealed(out / "diagnostic.json")
    inputs = acq.read_sealed(out / "inputs.json")
    if (result["plan_sha256"] != acq.sha256(out / "plan.json") or result["inputs_sha256"] != acq.sha256(out / "inputs.json") or
        canonical(inputs["inputs"]) != canonical(plan["expected_inputs"])):
        raise ValueError("Changed plan/canonical input provenance")
    if set(path.name for path in out.glob("row_*.json")) != set(result["row_sha256"]):
        raise ValueError("Missing/extra committed row files")
    for index, embedded in enumerate(result["rows"]):
        path = out / f"row_{index:02d}.json"
        file_doc = json.loads(path.read_text(encoding="utf-8"))
        n.v.checked_row(file_doc, embedded, acq.sha256(path), result["row_sha256"][path.name])
    validate_result(plan, result)
    torch.set_num_threads(THREADS)
    with q.d.deterministic_runtime("cpu"):
        if canonical(q.d.runtime_identity("cpu")) != canonical(plan["expected_runtime"]) or torch.cuda.is_initialized():
            raise ValueError("Changed verification CPU runtime")
    # Only read full saved states/metadata to verify identity; no model instance/forward.
    _, expected, _, descriptors, sampler, _ = source_evidence()
    if (canonical(expected) != canonical(plan["expected_inputs"]) or canonical(descriptors) != canonical(plan["models_identity"]) or
        tree_digest(sampler) != plan["source_sampler_identity"] or canonical(geometry()[2]) != canonical(plan["geometry"])):
        raise ValueError("Changed independent source/input/matrix identity")
    verification = scope() | {"plan_sha256": acq.sha256(out / "plan.json"), "diagnostic_sha256": acq.sha256(out / "diagnostic.json"),
        "inputs_sha256": acq.sha256(out / "inputs.json"), "symmetric_rows_verified": 36,
        "independently_verified": True, "verification_model_forward_calls": 0,
        "complete_identity_scope_role_accounting_summary_gates_passed": True}
    acq.write_new_json(out / "verification.json", acq.seal(verification))
    print("CROSS_BOUNDARY_VERIFY PASS36/216/432; zero forward in verify; NONRELEASE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--unit-evidence", type=Path, default=MONITOR / "unit_gate.json")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out, args.unit_evidence)
    elif args.mode == "run":
        run(args.out)
    else:
        verify(args.out)


if __name__ == "__main__":
    main()
