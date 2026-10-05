"""Bounded canonical-gradient bridge; old split failures remain measurements.

No optimizer, parameter.grad write, CUDA, update, DEV, audio or new checkpoint.
Only the direct original194 scalar gradient is authoritative, not a parts sum.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import re
import torch

import importlib.util
spec = importlib.util.spec_from_file_location("bridge_retained198", Path(__file__).with_name("198_diagnose_train_adam_split_numerics.py"))
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)
k, m, acq, ROOT = n.k, n.m, n.acq, n.ROOT
PURPOSE = "NONRELEASE_TRAIN_ADAM_CANONICAL_BRIDGE"
OUT = ROOT / "results/train_adam_canonical_bridge_20261004"
MONITOR = ROOT / "results/train_adam_canonical_bridge_monitor_20261004"
TEST = Path(__file__).with_name("_test_train_adam_canonical_bridge.py")
PROTOCOL = ROOT / "docs/train_adam_canonical_bridge_protocol_20261004.json"
REVIEW198 = n.MONITOR / "completion_review.json"
PATH_KEYS = ("parameter_full_vs_parts", "wave_full_vs_parts", "parameter_full_vs_vjp_full", "parameter_full_vs_vjp_parts")


def scope():
    return {"schema": 1, "purpose": PURPOSE, "model": "lr1_4500", "counter": 4500,
        "unique_input_batches": 1, "unique_input_slots": 6, "primary_slot_forwards": 6,
        "reference_slot_forwards": 6, "total_slot_forwards": 12, "samples": 89856,
        "frames": 352, "warmup": 96, "native_support": [25088, 89344], "kill_bands": 32,
        "normalizer": 6, "instrumental_weight": 4, "auxiliary_lambda": .2,
        "rtol": .0002, "atol": .0000002, "threads": 2, "forward_backward_dtype": "float32",
        "spectrum_dtype": "complex64", "statistics_dtype": "float64",
        "authority": "direct_original194_combine_slot_loss_autograd_grad_six_slot_FP32_order",
        "reference_requirement": "bit_exact_each_slot_and_total", "old_split_failures_retained": True,
        "original197_recovered": False, "training_authorized": False, "optimizer_constructed": False,
        "model_updates": 0, "cuda_used": False, "deployment": False, "release_selection": "NONE"}


def check_scope(doc):
    if any(not k.exact(doc.get(key), value) for key, value in scope().items()):
        raise ValueError("Changed bounded bridge scope/types")


def grad(params, scalar, retain=True):
    k.cpu_float(scalar, ())
    value = k.q.s.flatten_gradients(torch.autograd.grad(scalar, params, retain_graph=retain, allow_unused=True), params)
    k.cpu_float(value)
    return value


def require_comparison(doc, exact=False):
    n.validate_comparison(doc)
    if not doc["old_strict_pass"] or (exact and (doc["actual_sha256"] != doc["expected_sha256"] or doc["max_abs_error"] != 0.)):
        raise ValueError("Canonical bridge structural gradient gate failed; retain evidence")


def primary_probe(params, base, wave, losses, info, meta, index, direction):
    """Same graph: direct scalar authority, full-source canonical, measured parts.

    Never alter a group to force additivity. Every vector below comes from its
    named scalar or VJP. Scaling and sequential FP32 accumulation are explicit.
    """
    u, descent = direction
    residual, accompaniment, full = losses
    if not torch.equal(residual+accompaniment, full):
        raise ValueError("Original full auxiliary scalar identity changed")
    active = info["active"]
    if type(active) is not bool or (active and index >= 2):
        raise ValueError("Original activity/role required")
    gb = grad(params, base*(1/6))
    zero = torch.zeros_like(gb)
    gr = grad(params, residual*(1/6)) if active else zero.clone()
    ga = grad(params, accompaniment*(1/6)) if active else zero.clone()
    go = grad(params, full*(1/6)) if active else zero.clone()
    paths = None
    if active:
        wr = torch.autograd.grad(residual*(1/6), wave, retain_graph=True)[0]
        wa = torch.autograd.grad(accompaniment*(1/6), wave, retain_graph=True)[0]
        wf = torch.autograd.grad(full*(1/6), wave, retain_graph=True)[0]
        vf = k.q.s.flatten_gradients(torch.autograd.grad(wave, params, grad_outputs=wf, retain_graph=True, allow_unused=True), params)
        vp = k.q.s.flatten_gradients(torch.autograd.grad(wave, params, grad_outputs=wr+wa, retain_graph=True, allow_unused=True), params)
        paths = {"parameter_full_vs_parts": n.vector_comparison(go, gr+ga, (gr, ga), direction),
            "wave_full_vs_parts": n.vector_comparison(wf.flatten(), (wr+wa).flatten(), (wr.flatten(), wa.flatten())),
            "parameter_full_vs_vjp_full": n.vector_comparison(go, vf, direction=direction),
            "parameter_full_vs_vjp_parts": n.vector_comparison(go, vp, direction=direction)}
    elif any(float(value.detach()) != 0. for value in losses):
        raise ValueError("Skipped auxiliary must stay zero")
    loss, composition = k.t.w.combine_slot_loss(base, full, meta, index, 4, info)
    authority = grad(params, loss, False)
    weight = 4 if index == 2 else 1
    canonical = weight*gb+.2*go
    parts_sum = weight*gb+.2*(gr+ga)
    doc = {"loss": float(loss.detach()), "composition": composition,
        "direct_gradient_sha256": k.tree_digest(authority), "paths": paths,
        "canonical_full_vs_direct": n.vector_comparison(canonical, authority, direction=direction),
        "independent_parts_vs_direct_measurement": n.vector_comparison(parts_sum, authority, direction=direction),
        "alignment": {key: k.alignment(value, u, descent) for key, value in {
            "base_div6": gb, "cv2_div6": gr, "ca1_squared_div6": ga, "full_aux_div6": go,
            "canonical_full": canonical, "direct_original194": authority}.items()},
        "parameter_grad_storage_written": False, "additional_forwards_in_probe": 0}
    return doc, {"base": gb, "cv2": gr, "ca1_squared": ga, "full": go,
                 "canonical": canonical, "parts": parts_sum, "authority": authority}


def check_primary(doc, active):
    if doc["parameter_grad_storage_written"] is not False or not k.exact(doc["additional_forwards_in_probe"], 0):
        raise ValueError("No gradient storage writes or extra forwards")
    if active:
        if set(doc["paths"]) != set(PATH_KEYS):
            raise ValueError("All fixed structural numerical paths required")
        # The old failing independent parts comparison is retained, NOT a gate.
        n.validate_comparison(doc["paths"]["parameter_full_vs_parts"])
        require_comparison(doc["paths"]["parameter_full_vs_vjp_full"], exact=True)
        require_comparison(doc["paths"]["wave_full_vs_parts"])
        require_comparison(doc["paths"]["parameter_full_vs_vjp_parts"])
    elif doc["paths"] is not None:
        raise ValueError("Inactive original auxiliary paths must be skipped")
    require_comparison(doc["canonical_full_vs_direct"])
    n.validate_comparison(doc["independent_parts_vs_direct_measurement"])
    if set(doc["alignment"]) != {"base_div6", "cv2_div6", "ca1_squared_div6", "full_aux_div6", "canonical_full", "direct_original194"}:
        raise ValueError("All named scalar gradients required")
    for value in doc["alignment"].values():
        k.validate_alignment(value)
    if doc["alignment"]["direct_original194"]["gradient_sha256"] != doc["direct_gradient_sha256"]:
        raise ValueError("Direct authority gradient hash differs")
    if (doc["canonical_full_vs_direct"]["actual_sha256"] != doc["alignment"]["canonical_full"]["gradient_sha256"] or
        doc["canonical_full_vs_direct"]["expected_sha256"] != doc["direct_gradient_sha256"]):
        raise ValueError("Canonical/direct comparison identity differs")
    if not active and any(doc["alignment"][key]["g_l2"] != 0. for key in ("cv2_div6", "ca1_squared_div6", "full_aux_div6")):
        raise ValueError("Skipped original auxiliary must have zero gradient")


def reference_slot(net, xb, vb, wa, gs, meta, index):
    """Independent original194 operations, without constructing split scalars."""
    spectrum = m.core.stft_batch(xb)
    if spectrum.dtype != torch.complex64 or spectrum.shape != (1, 2, 513, 352):
        raise ValueError("Original frontend required")
    bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
    output = net(bands); k.cpu_float(output, (1, 4, 128, 352))
    mask = (output[:, :2]+1)/2
    base, _, wave = m.fit.reconstruction_loss(mask, spectrum, xb, vb, gs, 96, 32)
    k.cpu_float(base, ()); k.cpu_float(wave, xb.shape)
    region = m.fit.suite.scoring_slice(xb.shape[-1], 96)
    if (region.start, region.stop) != (25088, 89344):
        raise ValueError("Changed support")
    full, info = k.t.k.source_projection_component_loss(wave[0, :, region], xb[0, :, region], vb[0, :, region], meta, 1)
    loss, composition = k.t.w.combine_slot_loss(base, full, meta, index, 4, info)
    identity = {"wave_sha256": k.tree_digest(wave), "mask_sha256": k.tree_digest(mask),
                "base": float(base.detach()), "full": float(full.detach())}
    return grad(tuple(net.parameters()), loss, False), identity, info, float(loss.detach()), composition


def no_active_task():
    # Match relative new filenames too, not just an absolute workspace command.
    pattern = re.escape(str(ROOT))+r"|199_verify_train_adam_canonical_bridge|_test_train_adam_canonical_bridge"
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '"+pattern+"' } | Select-Object -ExpandProperty ProcessId"
    proc = k.subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    own = {os.getpid(), os.getppid()}
    if any(int(line.strip()) not in own for line in proc.stdout.splitlines() if line.strip()):
        raise ValueError("Active repository task; preserve it and postpone")
    k.no_active_worker()


def source():
    authority, saved, descriptor, expected, bindings = n.source()
    if acq.sha256(REVIEW198) != "18d3ef0111c0be1c3e74182c35837dae03d7451769128bafa1d6caf086070baa":
        raise ValueError("Changed completed198 independent proof")
    review = acq.read_sealed(REVIEW198)
    if (not k.exact(review["real_run"]["actual_native_exit_code"], 0) or
        not k.exact(review["real_verify"]["actual_native_exit_code"], 0) or
        review["real_verify"]["independently_verified"] is not True or review["recovery_authorized"] is not False):
        raise ValueError("Actual completed measurement, not recovered197 proof required")
    for item in review["immutable_bindings"]+review["row_bindings"]:
        path = ROOT / item["path"]
        if acq.sha256(path) != item["sha256"]:
            raise ValueError("Changed completed198 evidence")
        bindings[str(path.resolve())] = item["sha256"]
    prior = acq.read_sealed(n.OUT/"inputs.json")
    if not k.exact(prior["input"], expected):
        raise ValueError("Same197/198 fixed input required")
    for path in (Path(__file__), TEST, PROTOCOL, REVIEW198, n.MONITOR/"aggregation.json",
                 ROOT/"reports/93_train_adam_split_numerics_result.md", ROOT/"reports/94_train_adam_canonical_bridge_plan.md"):
        bindings[str(path.resolve())] = acq.sha256(path)
    return authority, saved, descriptor, expected, bindings


def unit_gate(path):
    doc = acq.read_sealed(path)
    if (not k.exact(doc["actual_exit_code"], 0) or doc["draft_reviewed"] is not True or
        type(doc["tests_passed"]) is not int or doc["tests_passed"] < 40 or
        any(doc[key] != acq.sha256(file) for key,file in (("tool_sha256", Path(__file__)), ("test_sha256", TEST), ("protocol_sha256", PROTOCOL), ("log_sha256", Path(doc["log"])) ))):
        raise ValueError("Actual new units and reviewed draft required before prepare")
    log = Path(doc["log"]).read_text(encoding="utf-8-sig")
    if not re.search(r"Ran "+str(doc["tests_passed"])+r" tests? in", log) or not re.search(r"(?m)^OK\s*$", log):
        raise ValueError("Full native unit output required")
    return doc


def prepare(out, unit):
    k.fresh(out); no_active_task(); k.resources(); gate = unit_gate(unit)
    if not k.exact(json.loads(PROTOCOL.read_text(encoding="utf-8")), scope()):
        raise ValueError("Changed fixed new protocol")
    _, _, descriptor, expected, bindings = source()
    torch.set_num_threads(2)
    with k.d.deterministic_runtime("cpu"):
        runtime = k.d.runtime_identity("cpu")
    bindings |= {str(unit.resolve()): acq.sha256(unit), gate["log"]: gate["log_sha256"]}
    plan = scope() | {"bindings_sha256": bindings, "model_descriptor": descriptor,
        "expected_input": expected, "matrix": k.matrix_identity()[2], "expected_runtime": runtime,
        "unit_gate": str(unit.resolve())}
    out.mkdir(parents=True); acq.write_new_json(out/"plan.json", acq.seal(plan))
    print("CANONICAL_BRIDGE PLAN SEALED; six primary+six reference; no recovery/training", flush=True)


def read_plan(out):
    plan = acq.read_sealed(out/"plan.json"); check_scope(plan)
    unit_gate(Path(plan["unit_gate"])); k.p.q.check_bindings(plan["bindings_sha256"])
    if not k.exact(k.matrix_identity()[2], plan["matrix"]):
        raise ValueError("Changed LF32 matrices")
    return plan


def new_counts():
    return {"primary_started": 0, "primary_completed": 0, "reference_started": 0, "reference_completed": 0}


@contextlib.contextmanager
def counted_forward(net, counts, kind):
    if kind not in ("primary", "reference") or counts[kind+"_started"] >= 6:
        raise ValueError("Six actual forwards per path only")
    before = counts[kind+"_started"]
    def start(_net, _args):
        counts[kind+"_started"] += 1
        if counts[kind+"_started"] > 6 or counts[kind+"_started"] != before+1:
            raise ValueError("Unexpected extra model forward")
    def end(_net, _args, _result):
        counts[kind+"_completed"] += 1
    left = net.register_forward_pre_hook(start); right = net.register_forward_hook(end)
    try:
        yield
        if counts[kind+"_completed"] != counts[kind+"_started"] or counts[kind+"_started"] != before+1:
            raise ValueError("Exactly one completed forward required per slot")
    finally:
        left.remove(); right.remove()


def status(out, state, phase, counts, primary_rows, reference_rows, error=None):
    temp = out/f"status199_{os.getpid()}.tmp"
    acq.write_new_json(temp, {"purpose": PURPOSE, "status": state, "phase": phase, "counts": counts,
        "committed_primary_rows": primary_rows, "committed_reference_rows": reference_rows,
        "pid": os.getpid(), "updated_utc": m.bulk.now(), "error": error,
        "model_updates": 0, "cuda_used": False, "optimizer_constructed": False, "release_selection": "NONE"})
    os.replace(temp, out/"run_status.json")


def run(out):
    if any((out/name).exists() for name in ("run_status.json", "inputs.json", "rows", "reference_rows", "diagnostic.json")):
        raise ValueError("Existing run/evidence; do not overwrite or restart")
    no_active_task(); k.resources(); plan = read_plan(out)
    authority, saved, descriptor, expected, _ = source()
    if not k.exact(descriptor, plan["model_descriptor"]) or not k.exact(expected, plan["expected_input"]):
        raise ValueError("Changed full source/input identity")
    stamps = k.binding_stamps(list(plan["bindings_sha256"])+[out/"plan.json"])
    source_digest = k.tree_digest((authority, saved, descriptor))
    rows, refs, counts = [], [], new_counts(); phase = "initialization"
    torch.set_num_threads(2)
    with m.bulk.worker_lock(out), k.d.deterministic_runtime("cpu"):
        if not k.exact(k.d.runtime_identity("cpu"), plan["expected_runtime"]):
            raise ValueError("Changed CPU FP32 runtime")
        status(out, "running", phase, counts, 0, 0)
        try:
            dataset = k.p.inp.ApprovedTeacherDataset(k.p.inp.verified_approval(Path(authority["origin_approval"])), "kim_melband")
            true = m.LockedTruePool(m.bulk.OLD_LOCK, dataset.config)
            before_rng = k.tree_digest(m.capture_rng("cpu"))
            batch = k.collect_next(dataset, true, 4500)
            if before_rng != k.tree_digest(m.capture_rng("cpu")) or not k.exact(k.input_entry(batch), expected):
                raise ValueError("Same retained197/198 input; no resampling")
            acq.write_new_json(out/"inputs.json", acq.seal({"purpose": PURPOSE, "input": expected, "rng_unchanged": True}))
            rng = k.d.portable(m.capture_rng("cpu"))
            try:
                net = m.frozen_factory(authority["source_protocol"])()
            finally:
                m.restore_rng(rng, "cpu")
            net.load_state_dict(saved["model"], strict=True)
            if len(saved["modes"]) != len(list(net.modules())) or any(type(flag) is not bool for flag in saved["modes"]):
                raise ValueError("All source modes required")
            for mod, flag in zip(net.modules(), saved["modes"]): mod.training = flag
            if (k.tree_digest(dict(net.state_dict())) != k.tree_digest(dict(saved["model"])) or
                saved["parameter_names"] != [name for name,_ in net.named_parameters()] or
                any(not param.requires_grad for param in net.parameters()) or
                any(isinstance(mod, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)) for mod in net.modules())):
                raise ValueError("Original full deterministic model/order required")
            params = tuple(net.parameters()); u, descent, direction = k.stored_direction(saved)
            wa, gs, _ = k.matrix_identity(); total = {key: torch.zeros_like(u) for key in ("authority", "canonical", "parts", "reference")}
            groups = {name: torch.zeros_like(u) for name in k.GROUPS}
            full_group = torch.zeros_like(u); primary_vectors = []
            (out/"rows").mkdir(); (out/"reference_rows").mkdir()
            # Original six-slot FP32 order; no flatten-all-batch/backward substitution.
            for index, meta in enumerate(batch["metadata"]):
                phase = "primary"; k.resources(); k.unchanged_bindings(stamps)
                with k.readonly(net, saved, batch), counted_forward(net, counts, phase):
                    base, wave, losses, info, identity = n.slot_wave(net, batch["x"][index:index+1], batch["v"][index:index+1], wa, gs, meta)
                    doc, vectors = primary_probe(params, base, wave, losses, info, meta, index, (u, descent))
                row = acq.seal({"slot": index, "metadata": copy.deepcopy(meta), "bucket": k.bucket(meta), "auxiliary": info,
                    "identity": identity, "probe": doc, "primary_slot_forwards": 1,
                    "scalar_split_bit_exact": True, "unchanged_weights_modes_grad_rng_moments_inputs": True})
                # Commit measurements before enforcing gates, preserving a real failure.
                acq.write_new_json(out/"rows"/f"row_{index:02d}.json", row); rows.append(row)
                check_primary(doc, info["active"])
                for key in ("authority", "canonical", "parts"): total[key] += vectors[key]
                groups[k.group_for_slot(index, meta)] += vectors["base"]
                groups[k.GROUPS[3]] += vectors["cv2"]; groups[k.GROUPS[4]] += vectors["ca1_squared"]
                full_group += vectors["full"]; primary_vectors.append(vectors["authority"])
                k.unchanged_bindings(stamps); status(out, "running", phase, counts, len(rows), len(refs))
                print(f"CANONICAL_BRIDGE primary={len(rows)}/6", flush=True)
            for index, meta in enumerate(batch["metadata"]):
                phase = "reference"; k.resources(); k.unchanged_bindings(stamps)
                with k.readonly(net, saved, batch), counted_forward(net, counts, phase):
                    vector, identity, info, loss, composition = reference_slot(net, batch["x"][index:index+1], batch["v"][index:index+1], wa, gs, meta, index)
                expected_identity = {key: rows[index]["identity"][key] for key in identity}
                comparison = n.vector_comparison(primary_vectors[index], vector, direction=(u, descent))
                ref = acq.seal({"slot": index, "metadata": copy.deepcopy(meta), "input_sha256": expected["input_sha256"][index],
                    "target_sha256": expected["target_sha256"][index], "identity": identity, "auxiliary": info,
                    "loss": loss, "composition": composition, "authority_vs_reference": comparison,
                    "scalar_wave_mask_identity_bit_exact": k.exact(identity, expected_identity) and k.exact(info, rows[index]["auxiliary"]) and
                        k.exact(loss, rows[index]["probe"]["loss"]) and k.exact(composition, rows[index]["probe"]["composition"]),
                    "reference_slot_forwards": 1, "unchanged_weights_modes_grad_rng_moments_inputs": True})
                acq.write_new_json(out/"reference_rows"/f"reference_{index:02d}.json", ref); refs.append(ref)
                if ref["scalar_wave_mask_identity_bit_exact"] is not True: raise ValueError("Independent original194 scalar/wave/mask identity differs")
                require_comparison(comparison, exact=True); total["reference"] += vector
                k.unchanged_bindings(stamps); status(out, "running", phase, counts, len(rows), len(refs))
                print(f"CANONICAL_BRIDGE reference={len(refs)}/6", flush=True)
            phase = "totals"
            group_sum = groups[k.GROUPS[0]]+4*groups[k.GROUPS[1]]+groups[k.GROUPS[2]]+.2*(groups[k.GROUPS[3]]+groups[k.GROUPS[4]])
            complete_group_sum = groups[k.GROUPS[0]]+4*groups[k.GROUPS[1]]+groups[k.GROUPS[2]]+.2*full_group
            totals = {"canonical_full_vs_authority": n.vector_comparison(total["canonical"], total["authority"], direction=(u,descent)),
                "authority_vs_reference": n.vector_comparison(total["authority"], total["reference"], direction=(u,descent)),
                "independent_slot_parts_vs_authority_measurement": n.vector_comparison(total["parts"], total["authority"], direction=(u,descent)),
                "independent_group_parts_vs_authority_measurement": n.vector_comparison(group_sum, total["authority"], direction=(u,descent)),
                "full_group_sum_vs_authority_measurement": n.vector_comparison(complete_group_sum, total["authority"], direction=(u,descent)),
                "alignment": {key: k.alignment(value,u,descent) for key,value in total.items()},
                "group_alignment": {key: k.alignment(value,u,descent) for key,value in groups.items()},
                "weighted_group_alignment": {key: k.alignment(value*c,u,descent) for (key,value),c in zip(groups.items(), k.COEFFICIENTS)},
                "full_auxiliary_group_alignment": k.alignment(full_group,u,descent),
                "direction": direction, "group_additivity_not_assumed": True}
            acq.write_new_json(out/"totals.json", acq.seal(totals))
            require_comparison(totals["canonical_full_vs_authority"]); require_comparison(totals["authority_vs_reference"], exact=True)
            if source_digest != k.tree_digest((authority, saved, descriptor)): raise ValueError("Mutated source authority/schedule/descriptor")
            k.unchanged_bindings(stamps); k.p.q.check_bindings(plan["bindings_sha256"])
            result = scope() | {"plan_sha256": acq.sha256(out/"plan.json"), "inputs_sha256": acq.sha256(out/"inputs.json"),
                "rows": rows, "reference_rows": refs, "totals": totals, "totals_sha256": acq.sha256(out/"totals.json"),
                "counts": counts, "runtime": k.d.runtime_identity("cpu"), "error": None,
                "source_state_unchanged": True, "bridge_gates_passed": True, "recovery_authorized": False}
            validate_result(result, plan); acq.write_new_json(out/"diagnostic.json", acq.seal(result))
            status(out, "complete", "bridge_requires_external_verify", counts, 6, 6)
            print("CANONICAL_BRIDGE COMPLETE6+6; structural gates passed;197 NOT RECOVERED; NONRELEASE", flush=True)
        except BaseException as error:
            status(out, "failed", phase, counts, len(rows), len(refs), repr(error)); raise


def validate_result(result, plan):
    check_scope(result)
    expected_counts = {key: 6 for key in new_counts()}
    if (not k.exact(result["counts"], expected_counts) or len(result["rows"]) != 6 or len(result["reference_rows"]) != 6 or
        result["error"] is not None or result["source_state_unchanged"] is not True or result["bridge_gates_passed"] is not True or
        result["recovery_authorized"] is not False or not k.d.finite_state(result) or not k.exact(result["runtime"], plan["expected_runtime"])):
        raise ValueError("Incomplete bridge/count/state/runtime identity")
    elements = sum(math.prod(item["shape"]) for item in plan["model_descriptor"]["direction"]["parameter_mapping"])
    for index, (row, ref) in enumerate(zip(result["rows"], result["reference_rows"])):
        meta = plan["expected_input"]["metadata"][index]
        if set(ref["identity"]) != {"wave_sha256", "mask_sha256", "base", "full"} or set(row["identity"]) != {"wave_sha256", "mask_sha256", "base", "residual", "accompaniment", "full"}:
            raise ValueError("Complete independent mask/wave/base/full identity required")
        if (not k.exact(row["slot"], index) or not k.exact(ref["slot"], index) or not k.exact(row["metadata"], meta) or
            not k.exact(ref["metadata"], meta) or row["bucket"] != k.bucket(meta) or not k.exact(row["primary_slot_forwards"], 1) or
            not k.exact(ref["reference_slot_forwards"], 1) or row["scalar_split_bit_exact"] is not True or
            any(doc["unchanged_weights_modes_grad_rng_moments_inputs"] is not True for doc in (row, ref)) or
            ref["scalar_wave_mask_identity_bit_exact"] is not True or not k.exact(row["auxiliary"], ref["auxiliary"]) or
            not k.exact(row["probe"]["loss"], ref["loss"]) or not k.exact(row["probe"]["composition"], ref["composition"]) or
            not k.exact({key: row["identity"][key] for key in ref["identity"]}, ref["identity"]) or
            ref["input_sha256"] != plan["expected_input"]["input_sha256"][index] or
            ref["target_sha256"] != plan["expected_input"]["target_sha256"][index]):
            raise ValueError("Changed paired original input/role/forward/scalar/wave identity")
        active = row["auxiliary"]["active"]
        if type(active) is not bool or (active and index >= 2) or (not active and any(row["identity"][key] != 0. for key in ("residual", "accompaniment", "full"))):
            raise ValueError("Original activity/skip identity required")
        check_primary(row["probe"], active); require_comparison(ref["authority_vs_reference"], exact=True)
        if (ref["authority_vs_reference"]["elements"] != elements or
            ref["authority_vs_reference"]["actual_sha256"] != row["probe"]["direct_gradient_sha256"]):
            raise ValueError("Reference authority hash/flattened parameter order differs")
        for key in ("canonical_full_vs_direct", "independent_parts_vs_direct_measurement"):
            if row["probe"][key]["elements"] != elements or row["probe"][key]["expected_sha256"] != row["probe"]["direct_gradient_sha256"]:
                raise ValueError("Missing original direct authority")
        for key, value in (row["probe"]["paths"] or {}).items():
            if value["elements"] != (179712 if key == "wave_full_vs_parts" else elements):
                raise ValueError("Changed wave/parameter support")
    totals = result["totals"]
    if totals["group_additivity_not_assumed"] is not True or not k.exact(totals["direction"], plan["model_descriptor"]["direction"]):
        raise ValueError("Changed original moments/order or numerical interpretation")
    for key in ("canonical_full_vs_authority", "authority_vs_reference", "independent_slot_parts_vs_authority_measurement",
                "independent_group_parts_vs_authority_measurement", "full_group_sum_vs_authority_measurement"):
        n.validate_comparison(totals[key])
        if totals[key]["elements"] != elements: raise ValueError("Changed total shape")
    require_comparison(totals["canonical_full_vs_authority"]); require_comparison(totals["authority_vs_reference"], exact=True)
    for collection in (totals["alignment"], totals["group_alignment"], totals["weighted_group_alignment"]):
        for doc in collection.values(): k.validate_alignment(doc)
    if set(totals["alignment"]) != {"authority", "canonical", "parts", "reference"}:
        raise ValueError("Separately named total vectors required")
    if set(totals["group_alignment"]) != set(k.GROUPS) or set(totals["weighted_group_alignment"]) != set(k.GROUPS):
        raise ValueError("All independently measured original groups required")
    k.validate_alignment(totals["full_auxiliary_group_alignment"])
    if totals["alignment"]["authority"]["gradient_sha256"] != totals["authority_vs_reference"]["actual_sha256"]:
        raise ValueError("Total authority identity differs")
    if (totals["alignment"]["reference"]["gradient_sha256"] != totals["authority_vs_reference"]["expected_sha256"] or
        totals["alignment"]["canonical"]["gradient_sha256"] != totals["canonical_full_vs_authority"]["actual_sha256"] or
        totals["alignment"]["authority"]["gradient_sha256"] != totals["canonical_full_vs_authority"]["expected_sha256"]):
        raise ValueError("Total canonical/reference identities differ")


def symmetric_rows(out, embedded, directory="rows", prefix="row"):
    files = sorted((out/directory).glob(prefix+"_*.json"))
    if len(files) != 6 or len(embedded) != 6: raise ValueError("Exactly six separately sealed rows required")
    for index, (path, inside) in enumerate(zip(files, embedded)):
        outside = acq.read_sealed(path)
        if acq.content_digest(inside) != inside.get("content_sha256") or path.name != f"{prefix}_{index:02d}.json" or not k.exact(outside, inside):
            raise ValueError("Both seals and full symmetric type-sensitive row identity required")


def verify(out):
    if (out/"verification.json").exists(): raise ValueError("Existing independent verification; do not repeat")
    no_active_task(); k.resources(); plan = read_plan(out)
    ending = json.loads((out/"run_status.json").read_text(encoding="utf-8"))
    expected = {"status": "complete", "counts": {key:6 for key in new_counts()}, "committed_primary_rows": 6,
                "committed_reference_rows": 6, "model_updates": 0, "cuda_used": False, "optimizer_constructed": False, "error": None}
    if any(not k.exact(ending.get(key), value) for key,value in expected.items()): raise ValueError("Complete actual6+6 status required")
    result = acq.read_sealed(out/"diagnostic.json")
    if result["plan_sha256"] != acq.sha256(out/"plan.json") or result["inputs_sha256"] != acq.sha256(out/"inputs.json") or result["totals_sha256"] != acq.sha256(out/"totals.json"):
        raise ValueError("Changed plan/inputs/totals")
    inp = acq.read_sealed(out/"inputs.json"); totals = acq.read_sealed(out/"totals.json")
    if inp["rng_unchanged"] is not True or not k.exact(inp["input"], plan["expected_input"]) or not k.exact({key:value for key,value in totals.items() if key != "content_sha256"}, result["totals"]):
        raise ValueError("Changed input/totals or missing source guard")
    symmetric_rows(out, result["rows"]); symmetric_rows(out, result["reference_rows"], "reference_rows", "reference")
    validate_result(result, plan)
    acq.write_new_json(out/"verification.json", acq.seal({"purpose": PURPOSE, "diagnostic_sha256": acq.sha256(out/"diagnostic.json"),
        "plan_sha256": acq.sha256(out/"plan.json"), "primary_slots": 6, "reference_slots": 6,
        "model_forward_count": 0, "independently_verified": True, "bridge_gates_passed": True,
        "both_row_sets_sealed_symmetric_type_sensitive": True, "original197_recovered": False,
        "recovery_authorized": False, "model_updates": 0, "optimizer_constructed": False, "cuda_used": False}))
    print("CANONICAL_BRIDGE VERIFIED6+6;0 forward;197 NOT RECOVERED; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--unit-evidence", type=Path, default=MONITOR/"unit_gate.json")
    args = parser.parse_args()
    prepare(args.out, args.unit_evidence) if args.operation == "prepare" else globals()[args.operation](args.out)
