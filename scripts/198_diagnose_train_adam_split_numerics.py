"""Bounded readonly numerical-path measurement for the retained197 failure."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import torch

spec = importlib.util.spec_from_file_location("retained197", Path(__file__).with_name("197_diagnose_train_adam_memory.py"))
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
m, acq, ROOT = k.m, k.acq, k.ROOT
PURPOSE = "NONRELEASE_TRAIN_ADAM_SPLIT_NUMERICS"
OUT = ROOT / "results/train_adam_split_numerics_20261004"
MONITOR = ROOT / "results/train_adam_split_numerics_monitor_20261004"
TEST = Path(__file__).with_name("_test_train_adam_split_numerics.py")
PROTOCOL = ROOT / "docs/train_adam_split_numerics_protocol_20261004.json"
FAILURE = k.MONITOR / "failure_review.json"
COMPARISONS = ("parameter_full_vs_parts", "wave_full_vs_parts", "parameter_full_vs_vjp_full",
               "parameter_full_vs_vjp_parts", "parameter_vjp_parts_vs_parameter_parts")


def scope():
    return {"schema": 1, "purpose": PURPOSE, "counter": 4500, "model": "lr1_4500",
        "unique_input_batches": 1, "unique_input_slots": 6, "slot_forwards": 6,
        "frames": 352, "warmup": 96, "native_support": [25088, 89344], "kill_bands": 32,
        "normalizer": 6, "auxiliary_lambda_unchanged": .2, "instrumental_weight_unchanged": 4,
        "rtol": .0002, "atol": .0000002, "threads": 2, "forward_backward_dtype": "float32",
        "spectrum_dtype": "complex64", "statistics_dtype": "float64", "comparisons": list(COMPARISONS),
        "measurement_complete_not_equivalence_pass": True, "training_authorized": False,
        "optimizer_constructed": False, "model_updates": 0, "cuda_used": False, "deployment": False,
        "release_selection": "NONE"}


def check_scope(doc):
    if any(not k.exact(doc.get(key), value) for key, value in scope().items()):
        raise ValueError("Changed fixed readonly scope/types")


def vector_comparison(actual, expected, terms=None, direction=None):
    """Retain failures under unchanged torch.assert_close tolerances, no gate relaxation."""
    k.cpu_float(actual); k.cpu_float(expected, actual.shape)
    if actual.ndim != 1 or not actual.numel():
        raise ValueError("Aligned nonempty flattened CPU FP32 vectors")
    if terms is not None:
        for value in terms:
            k.cpu_float(value, actual.shape)
    av, ev = actual.detach().double(), expected.detach().double()
    diff = av-ev
    bad = torch.nonzero(~torch.isclose(actual, expected, rtol=.0002, atol=.0000002)).flatten().tolist()
    failures = []
    for index in bad:
        item = {"index": index, "actual": float(av[index]), "expected": float(ev[index]),
                "absolute_error": float(diff[index].abs()), "old_bound": .0000002+.0002*float(ev[index].abs())}
        if terms is not None:
            values = [float(value[index]) for value in terms]
            denom = sum(abs(value) for value in values)
            item |= {"terms": values, "sum_abs_terms": denom,
                     "cancellation_ratio": abs(sum(values))/denom if denom else None}
        failures.append(item)
    maximum = int(diff.abs().argmax())
    cosine_denominator = float(av.norm())*float(ev.norm())
    doc = {"elements": actual.numel(), "mismatched_elements": len(bad), "old_strict_pass": not bad,
        "rtol": .0002, "atol": .0000002, "max_abs_error": float(diff.abs().max()), "max_error_index": maximum,
        "error_l2": float(diff.norm()), "actual_l2": float(av.norm()), "expected_l2": float(ev.norm()),
        "cosine": float(av@ev)/cosine_denominator if cosine_denominator else None,
        "actual_sha256": k.tree_digest(actual), "expected_sha256": k.tree_digest(expected),
        "all_mismatched_elements": failures, "statistics_dtype": "float64"}
    if direction is not None:
        u, descent = direction
        k.cpu_float(u, actual.shape); k.cpu_float(descent, actual.shape)
        uv, dv = u.detach().double(), descent.detach().double()
        doc["stored_direction_dot"] = {"actual_dot_u": float(av@uv), "expected_dot_u": float(ev@uv),
            "error_dot_u": float(diff@uv), "actual_dot_d": float(av@dv), "expected_dot_d": float(ev@dv),
            "error_dot_d": float(diff@dv), "not_quality_or_descent_acceptance": True}
    validate_comparison(doc)
    return doc


def validate_comparison(doc):
    if (type(doc["elements"]) is not int or doc["elements"] <= 0 or type(doc["mismatched_elements"]) is not int or
        doc["mismatched_elements"] != len(doc["all_mismatched_elements"]) or not k.exact(doc["old_strict_pass"], doc["mismatched_elements"] == 0) or
        not k.exact(doc["rtol"], .0002) or not k.exact(doc["atol"], .0000002) or doc["statistics_dtype"] != "float64" or
        not k.d.finite_state(doc)):
        raise ValueError("Invalid strict measurement/count/type")
    for key in ("max_abs_error", "error_l2", "actual_l2", "expected_l2"):
        if type(doc[key]) is not float or doc[key] < 0:
            raise ValueError("Literal finite FP64 norm/error required")
    if type(doc["max_error_index"]) is not int or not 0 <= doc["max_error_index"] < doc["elements"]:
        raise ValueError("Invalid max error index")
    if doc["error_l2"]+1e-15 < doc["max_abs_error"]:
        raise ValueError("Inconsistent error norms")
    for key in ("actual_sha256", "expected_sha256"):
        if type(doc[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", doc[key]):
            raise ValueError("Exact vector hashes required")
    if doc["actual_l2"] == 0. or doc["expected_l2"] == 0.:
        if doc["cosine"] is not None:
            raise ValueError("Zero norm cosine must be unavailable")
    elif type(doc["cosine"]) is not float or abs(doc["cosine"]) > 1+1e-12:
        raise ValueError("Invalid detached cosine")
    indices = []
    for item in doc["all_mismatched_elements"]:
        index = item["index"]; indices.append(index)
        if type(index) is not int or not 0 <= index < doc["elements"]:
            raise ValueError("Invalid mismatch index")
        a, e = torch.tensor(item["actual"], dtype=torch.float32), torch.tensor(item["expected"], dtype=torch.float32)
        if torch.isclose(a, e, rtol=.0002, atol=.0000002):
            raise ValueError("Stored failed element satisfies old strict threshold")
        if (item["absolute_error"] != abs(item["actual"]-item["expected"]) or
            item["old_bound"] != .0000002+.0002*abs(item["expected"])):
            raise ValueError("Mismatch arithmetic changed")
        if "terms" in item:
            denom = sum(abs(value) for value in item["terms"])
            ratio = abs(sum(item["terms"]))/denom if denom else None
            if not k.exact(item["sum_abs_terms"], denom) or not k.exact(item["cancellation_ratio"], ratio):
                raise ValueError("Cancellation arithmetic changed")
    if indices != sorted(set(indices)):
        raise ValueError("All failed elements must be unique and ordered")
    if "stored_direction_dot" in doc:
        dot = doc["stored_direction_dot"]
        if dot["not_quality_or_descent_acceptance"] is not True:
            raise ValueError("No quality inference")
        for suffix in ("u", "d"):
            scale = max(abs(dot["actual_dot_"+suffix]), abs(dot["expected_dot_"+suffix]), 1.)
            if not math.isclose(dot["actual_dot_"+suffix]-dot["expected_dot_"+suffix], dot["error_dot_"+suffix], abs_tol=1e-12*scale):
                raise ValueError("Detached dot error arithmetic changed")


def trace_paths(params, wave, losses, base, direction):
    """Same197 base/gr/ga/go order, then extra VJPs on the same single forward."""
    def grad(loss):
        return k.q.s.flatten_gradients(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True), params)
    gb = grad(base*(1/6))
    residual, accompaniment, original = losses
    gr, ga, go = (grad(value*(1/6)) for value in losses)
    wr, wa, wf = (torch.autograd.grad(value*(1/6), wave, retain_graph=True)[0] for value in losses)
    for value in (wr, wa, wf):
        k.cpu_float(value, wave.shape)
    def vjp(cotangent, retain):
        return k.q.s.flatten_gradients(torch.autograd.grad(wave, params, grad_outputs=cotangent, retain_graph=retain, allow_unused=True), params)
    vp, vf = vjp(wr+wa, True), vjp(wf, False)
    return {"base_gradient_sha256": k.tree_digest(gb), "comparisons": {
        COMPARISONS[0]: vector_comparison(go, gr+ga, (gr, ga), direction),
        COMPARISONS[1]: vector_comparison(wf.flatten(), (wr+wa).flatten(), (wr.flatten(), wa.flatten())),
        COMPARISONS[2]: vector_comparison(go, vf, direction=direction),
        COMPARISONS[3]: vector_comparison(go, vp, direction=direction),
        COMPARISONS[4]: vector_comparison(vp, gr+ga, (gr, ga), direction)},
        "parameter_gradient_sha256": {"full": k.tree_digest(go), "residual": k.tree_digest(gr), "accompaniment": k.tree_digest(ga),
            "vjp_full": k.tree_digest(vf), "vjp_parts": k.tree_digest(vp)},
        "wave_cotangent_sha256": {"full": k.tree_digest(wf), "residual": k.tree_digest(wr), "accompaniment": k.tree_digest(wa)},
        "parameter_grad_storage_written": False, "forward_count_in_trace": 0}


def slot_wave(net, xb, vb, wa, gs, meta):
    # Exact197 forward and immutable original194 full auxiliary; expose only wave.
    spectrum = m.core.stft_batch(xb)
    if spectrum.dtype != torch.complex64 or spectrum.shape != (1, 2, 513, 352):
        raise ValueError("Original FP32/complex64 352-frame frontend required")
    bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
    output = net(bands); k.cpu_float(output, (1, 4, 128, 352))
    mask = (output[:, :2]+1)/2
    base, _, wave = m.fit.reconstruction_loss(mask, spectrum, xb, vb, gs, 96, 32)
    k.cpu_float(base, ()); k.cpu_float(wave, xb.shape)
    region = m.fit.suite.scoring_slice(xb.shape[-1], 96)
    if (region.start, region.stop) != (25088, 89344):
        raise ValueError("Changed original support")
    residual, accompaniment, original, info = k.q.split_auxiliary(wave[0, :, region], xb[0, :, region], vb[0, :, region], meta)
    full, full_info = k.t.k.source_projection_component_loss(wave[0, :, region], xb[0, :, region], vb[0, :, region], meta, 1)
    if not torch.equal(original, full) or not k.exact(info, full_info):
        raise ValueError("Original full scalar/activity must remain bit identical")
    return base, wave, (residual, accompaniment, original), info, {
        "wave_sha256": k.tree_digest(wave), "mask_sha256": k.tree_digest(mask), "base": float(base.detach()),
        "residual": float(residual.detach()), "accompaniment": float(accompaniment.detach()), "full": float(original.detach())}


def source():
    if acq.sha256(FAILURE) != "9cd55bd2c4c1d06eca73a4691746068450b1f4e1e48b1a24e854b9cb391d85d7":
        raise ValueError("Changed retained197 failure review")
    failure = acq.read_sealed(FAILURE)
    bindings = dict(failure["evidence_sha256"])
    k.p.q.check_bindings(bindings)
    if (failure["actual_run_exit_code"] != 1 or failure["committed_model_batches"] != 3 or
        failure["diagnostic_exists"] is not False or failure["verification_exists"] is not False or
        any((k.DEFAULT_OUT/name).exists() for name in ("diagnostic.json", "verification.json"))):
        raise ValueError("Actual failed, not completed197 evidence required")
    authority, models, descriptors, _, inherited = k.source_evidence()
    bindings |= inherited
    expected = acq.read_sealed(k.DEFAULT_OUT/"inputs.json")["inputs"][0]
    if expected["counter"] != 4500:
        raise ValueError("Same failure input counter required")
    for path in (Path(__file__), TEST, PROTOCOL, FAILURE, ROOT/"reports/91_train_adam_memory_failure.md", ROOT/"reports/92_train_adam_split_numerics_plan.md"):
        bindings[str(path.resolve())] = acq.sha256(path)
    return authority, models["lr1_4500"], descriptors["lr1_4500"], expected, bindings


def gate(path):
    doc = acq.read_sealed(path)
    if (type(doc["actual_exit_code"]) is not int or doc["actual_exit_code"] != 0 or doc["draft_reviewed"] is not True or
        type(doc["tests_passed"]) is not int or doc["tests_passed"] < 20 or doc["tool_sha256"] != acq.sha256(__file__) or
        doc["test_sha256"] != acq.sha256(TEST) or doc["protocol_sha256"] != acq.sha256(PROTOCOL) or
        doc["log_sha256"] != acq.sha256(doc["log"])):
        raise ValueError("Actual new units/draft review before prepare required")
    log = Path(doc["log"]).read_text(encoding="utf-8-sig")
    if not re.search(r"Ran "+str(doc["tests_passed"])+r" tests? in", log) or not re.search(r"(?m)^OK\s*$", log):
        raise ValueError("Full native unit output required")
    return doc


def prepare(out, unit):
    k.fresh(out); k.no_active_worker(); k.resources(); unit_doc = gate(unit)
    if not k.exact(json.loads(PROTOCOL.read_text(encoding="utf-8")), scope()):
        raise ValueError("Changed new protocol")
    _, _, descriptor, expected, bindings = source()
    _, _, matrix = k.matrix_identity()
    torch.set_num_threads(2)
    with k.d.deterministic_runtime("cpu"):
        runtime = k.d.runtime_identity("cpu")
    bindings |= {str(unit.resolve()): acq.sha256(unit), unit_doc["log"]: unit_doc["log_sha256"]}
    doc = scope() | {"bindings_sha256": bindings, "model_descriptor": descriptor, "expected_input": expected,
                     "matrix": matrix, "expected_runtime": runtime, "unit_gate": str(unit.resolve())}
    out.mkdir(parents=True); acq.write_new_json(out/"plan.json", acq.seal(doc))
    print("SPLIT_NUMERICS PLAN SEALED; six forwards, no recovery/training authorization", flush=True)


def read_plan(out):
    plan = acq.read_sealed(out/"plan.json"); check_scope(plan); gate(Path(plan["unit_gate"]))
    k.p.q.check_bindings(plan["bindings_sha256"])
    if not k.exact(k.matrix_identity()[2], plan["matrix"]):
        raise ValueError("Changed LF32 matrix")
    return plan


def status(out, state, completed, error=None):
    temp = out/f"status198_{os.getpid()}.tmp"
    acq.write_new_json(temp, {"purpose": PURPOSE, "status": state, "completed_slots": completed,
        "slot_forwards_committed": completed, "limit": 6, "error": error, "pid": os.getpid(), "updated_utc": m.bulk.now(),
        "model_updates": 0, "cuda_used": False, "optimizer_constructed": False})
    os.replace(temp, out/"run_status.json")


def validate_rows(rows, plan):
    if len(rows) != 6:
        raise ValueError("Exactly six original slots required")
    for index, row in enumerate(rows):
        meta = plan["expected_input"]["metadata"][index]
        if (not k.exact(row["slot"], index) or not k.exact(row["metadata"], meta) or row["bucket"] != k.bucket(meta) or
            row["unchanged_weights_modes_grad_rng_moments_inputs"] is not True or not k.exact(row["slot_forwards"], 1) or
            row["scalar_split_bit_exact"] is not True or not k.d.finite_state(row) or type(row["auxiliary"]["active"]) is not bool):
            raise ValueError("Changed fixed slot/role/state/finite identity")
        if row["auxiliary"]["active"]:
            if index >= 2 or row["numerics"] is None or set(row["numerics"]["comparisons"]) != set(COMPARISONS):
                raise ValueError("All numerical paths required for true active slot")
            if row["numerics"]["parameter_grad_storage_written"] is not False or not k.exact(row["numerics"]["forward_count_in_trace"], 0):
                raise ValueError("No gradient storage writes or extra forwards")
            for key, doc in row["numerics"]["comparisons"].items():
                validate_comparison(doc)
                expected_elements = 179712 if key == "wave_full_vs_parts" else plan["model_descriptor"]["direction"]["parameter_mapping"]
                if key != "wave_full_vs_parts":
                    expected_elements = sum(math.prod(item["shape"]) for item in expected_elements)
                if doc["elements"] != expected_elements:
                    raise ValueError("Changed flattened wave/parameter support")
        elif row["numerics"] is not None or any(row["identity"][key] != 0. for key in ("residual", "accompaniment", "full")):
            raise ValueError("Zero-reference/pseudo/activity skip must remain unavailable")


def run(out):
    if any((out/name).exists() for name in ("run_status.json", "inputs.json", "rows", "diagnostic.json")):
        raise ValueError("Existing run/evidence; never overwrite or restart")
    k.no_active_worker(); k.resources(); plan = read_plan(out)
    authority, saved, descriptor, expected, _ = source()
    if not k.exact(descriptor, plan["model_descriptor"]) or not k.exact(expected, plan["expected_input"]):
        raise ValueError("Changed fixed full source/input identity")
    stamps = k.binding_stamps(list(plan["bindings_sha256"])+[out/"plan.json"])
    torch.set_num_threads(2); rows = []
    with m.bulk.worker_lock(out), k.d.deterministic_runtime("cpu"):
        if not k.exact(k.d.runtime_identity("cpu"), plan["expected_runtime"]):
            raise ValueError("Changed strict CPU FP32 runtime")
        status(out, "running_initialization", 0)
        try:
            dataset = k.p.inp.ApprovedTeacherDataset(k.p.inp.verified_approval(Path(authority["origin_approval"])), "kim_melband")
            true = m.LockedTruePool(m.bulk.OLD_LOCK, dataset.config)
            before = k.tree_digest(m.capture_rng("cpu"))
            batch = k.collect_next(dataset, true, 4500)
            if before != k.tree_digest(m.capture_rng("cpu")) or not k.exact(k.input_entry(batch), expected):
                raise ValueError("Same retained197 x/v/meta/PCM hash required; no resampling")
            acq.write_new_json(out/"inputs.json", acq.seal({"purpose": PURPOSE, "input": expected, "rng_unchanged": True}))
            rng = k.d.portable(m.capture_rng("cpu"))
            try:
                net = m.frozen_factory(authority["source_protocol"])()
            finally:
                m.restore_rng(rng, "cpu")
            net.load_state_dict(saved["model"], strict=True)
            if len(saved["modes"]) != len(list(net.modules())) or any(type(flag) is not bool for flag in saved["modes"]):
                raise ValueError("All saved modes required")
            for module, flag in zip(net.modules(), saved["modes"]):
                module.training = flag
            if (k.tree_digest(dict(net.state_dict())) != k.tree_digest(dict(saved["model"])) or
                saved["parameter_names"] != [name for name,_ in net.named_parameters()]):
                raise ValueError("Complete copied model/order identity required")
            params = tuple(net.parameters()); u, descent, _ = k.stored_direction(saved)
            if any(not param.requires_grad for param in params) or any(isinstance(module, (
                torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)) for module in net.modules()):
                raise ValueError("Original deterministic graph/no mode-mutating BN/dropout required")
            wa, gs, _ = k.matrix_identity(); (out/"rows").mkdir()
            for index, meta in enumerate(batch["metadata"]):
                k.resources(); k.unchanged_bindings(stamps)
                with k.readonly(net, saved, batch):
                    base, wave, losses, info, identity = slot_wave(net, batch["x"][index:index+1], batch["v"][index:index+1], wa, gs, meta)
                    numeric = trace_paths(params, wave, losses, base, (u, descent)) if info["active"] else None
                k.unchanged_bindings(stamps)
                row = acq.seal({"slot": index, "metadata": copy.deepcopy(meta), "bucket": k.bucket(meta), "auxiliary": info,
                    "identity": identity, "numerics": numeric, "slot_forwards": 1, "scalar_split_bit_exact": True,
                    "unchanged_weights_modes_grad_rng_moments_inputs": True})
                acq.write_new_json(out/"rows"/f"row_{index:02d}.json", row); rows.append(row)
                status(out, "running_numeric_paths", len(rows))
                print(f"SPLIT_NUMERICS slot={index} active={info['active']} complete={len(rows)}/6", flush=True)
            validate_rows(rows, plan); k.unchanged_bindings(stamps); k.p.q.check_bindings(plan["bindings_sha256"])
            result = scope() | {"plan_sha256": acq.sha256(out/"plan.json"), "inputs_sha256": acq.sha256(out/"inputs.json"),
                "rows": rows, "error": None, "runtime": k.d.runtime_identity("cpu"), "recovery_authorized": False}
            acq.write_new_json(out/"diagnostic.json", acq.seal(result)); status(out, "complete", 6)
            print("SPLIT_NUMERICS MEASUREMENT COMPLETE6; strict failures retained;197 NOT RECOVERED", flush=True)
        except BaseException as error:
            status(out, "failed", len(rows), repr(error)); raise


def symmetric_rows(out, embedded):
    files = sorted((out/"rows").glob("row_*.json"))
    if len(files) != 6 or len(embedded) != 6:
        raise ValueError("Six committed and embedded rows required")
    for index, (path, inside) in enumerate(zip(files, embedded)):
        outside = acq.read_sealed(path)
        if (acq.content_digest(inside) != inside.get("content_sha256") or path.name != f"row_{index:02d}.json" or
            not k.exact(outside, inside)):
            raise ValueError("Each row seal and full symmetric typed equality required")


def verify(out):
    if (out/"verification.json").exists():
        raise ValueError("Existing verification: do not repeat")
    k.no_active_worker(); k.resources(); plan = read_plan(out)
    ending = json.loads((out/"run_status.json").read_text(encoding="utf-8"))
    if any(not k.exact(ending.get(key), value) for key,value in {"status":"complete","completed_slots":6,"error":None,"model_updates":0,"cuda_used":False}.items()):
        raise ValueError("Complete6 readonly status required")
    result = acq.read_sealed(out/"diagnostic.json"); check_scope(result)
    if (result["error"] is not None or result["recovery_authorized"] is not False or
        result["plan_sha256"] != acq.sha256(out/"plan.json") or result["inputs_sha256"] != acq.sha256(out/"inputs.json") or
        not k.exact(result["runtime"], plan["expected_runtime"])):
        raise ValueError("Incomplete/changed measurement; not recovery authority")
    inp = acq.read_sealed(out/"inputs.json")
    if not k.exact(inp["input"], plan["expected_input"]) or inp["rng_unchanged"] is not True:
        raise ValueError("Changed same retained input")
    symmetric_rows(out, result["rows"]); validate_rows(result["rows"], plan)
    acq.write_new_json(out/"verification.json", acq.seal({"purpose":PURPOSE,"diagnostic_sha256":acq.sha256(out/"diagnostic.json"),
        "plan_sha256":acq.sha256(out/"plan.json"),"verified_slots":6,"model_forward_count":0,"cuda_used":False,
        "model_updates":0,"independently_verified":True,"measurement_not_equivalence_acceptance":True,"recovery_authorized":False}))
    print("SPLIT_NUMERICS VERIFIED6;0 forward;197 recovery remains NOT AUTHORIZED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare","run","verify"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--unit-evidence",type=Path,default=MONITOR/"unit_gate.json")
    args=parser.parse_args()
    prepare(args.out,args.unit_evidence) if args.operation=="prepare" else globals()[args.operation](args.out)
