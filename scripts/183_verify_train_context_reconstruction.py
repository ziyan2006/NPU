"""Read-only independent verification; symmetric full sealed rows, no forward."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("immutable_recovery182", Path(__file__).with_name("182_diagnose_train_context_reconstruction_recovery.py"))
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
q, acq, ROOT = k.q, k.acq, k.ROOT
PURPOSE = "NONRELEASE_TRAIN_CONTEXT_RECONSTRUCTION_INDEPENDENT_RECOVERED_VERIFY"
SOURCE = k.DEFAULT_OUT
DEFAULT_OUT = ROOT / "results/train_context_reconstruction_review_20261003"
REPORT = ROOT / "reports/68_train_context_independent_verifier_recovery.md"
TEST = Path(__file__).with_name("_test_train_context_independent_verifier.py")
PROOF = k.MONITOR / "run_completion_unverified.json"


def canonical(doc):
    return json.dumps(doc, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def checked_row(file_doc, embedded, actual_sha, expected_sha):
    if actual_sha != expected_sha:
        raise ValueError("Committed file SHA differs from original manifest")
    for doc in (file_doc, embedded):
        if not isinstance(doc, dict) or not isinstance(doc.get("content_sha256"), str) or acq.content_digest(doc) != doc["content_sha256"]:
            raise ValueError("Invalid file or embedded row content seal")
    if canonical(file_doc) != canonical(embedded):
        raise ValueError("Sealed file/embedded row differs, including exact types")


def expected_scope():
    return {"schema": 1, "purpose": PURPOSE, "source": str(SOURCE),
        "method": "symmetric_full_sealed_row_equality", "model_batches": 36,
        "slot_checks": 216, "unique_input_batches": 12, "unique_input_slots": 72,
        "run_exit_code": 0, "initial_verify_exit_code": 1, "model_updates": 0,
        "model_forward_used": False, "backward_used": False, "cuda_used": False,
        "training_authorized": False, "release_selection": "NONE", "deployment": False}


def check_plan(doc):
    if any(canonical(doc.get(name)) != canonical(value) for name, value in expected_scope().items()):
        raise ValueError("Changed independent read-only verification scope")


def check_row_accounting(result):
    names = [f"row_{index:02d}.json" for index in range(36)]
    if set(result["row_sha256"]) != set(names) or len(result["rows"]) != 36:
        raise ValueError("All36 original committed rows required")
    return names


def source_evidence():
    k.no_active_worker(); q.check_disk()
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only read-only verifier required")
    plan = k.verified_plan(SOURCE)
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    if (proof["purpose"] != "NONRELEASE_TRAIN_CONTEXT_RECONSTRUCTION_COMPLETE_RUN_WITH_FAILED_INITIAL_VERIFY" or
        type(proof["actual_run_exit_code"]) is not int or proof["actual_run_exit_code"] != 0 or
        proof["foreground_run_session"] != 14585 or proof["not_wmi"] is not True or
        proof["no_active_worker_or_shim"] is not True or proof["processes"] or
        proof["initial_verify"]["actual_exit_code"] != 1 or proof["initial_verify"]["error"] != "ValueError('Changed committed row')" or
        proof["model_updates"] != 0 or proof["cuda_used"] is not False or
        proof["independently_verified"] is not False or proof["analysis_pending"] is not True):
        raise ValueError("Actual complete run and preserved initial verifier failure required")
    status = json.loads((SOURCE / "run_status.json").read_text(encoding="utf-8"))
    if (canonical(status) != canonical(proof["terminal_status"]) or status["status"] != "complete" or
        status["error"] is not None or status["completed_model_batches"] != 36 or status["limit"] != 36):
        raise ValueError("Exact complete36 terminal state required")
    evidence = {str((ROOT / name).resolve()): digest for name, digest in proof["raw_evidence_sha256"].items()}
    q.check_bindings(evidence)
    result = acq.read_sealed(SOURCE / "diagnostic.json")
    inputs = acq.read_sealed(SOURCE / "inputs.json")
    if (result["plan_sha256"] != acq.sha256(SOURCE / "plan.json") or result["inputs_sha256"] != acq.sha256(SOURCE / "inputs.json") or
        canonical(inputs["inputs"]) != canonical(result["inputs"]) or inputs["purpose"] != q.PURPOSE or
        inputs["tool"] != k.TOOL or inputs["plan_sha256"] != result["plan_sha256"]):
        raise ValueError("Changed original plan/input/artifact identity")
    names = check_row_accounting(result)
    files = {PROOF, REPORT, TEST, Path(__file__), SOURCE / "plan.json"}
    bindings = plan["bindings_sha256"] | evidence
    for index, name in enumerate(names):
        path = SOURCE / name
        digest = acq.sha256(path)
        checked_row(acq.read_sealed(path), result["rows"][index], digest, result["row_sha256"][name])
        bindings[str(path.resolve())] = digest
    # All unchanged original scope/roles/inputs/model digest/runtime/geometry,
    # model mode/RNG/grad immutability and recomputed summary gates remain.
    k.validate_result(plan, result)
    bindings |= {str(path.resolve()): acq.sha256(path) for path in files}
    q.check_bindings(bindings)
    if torch.cuda.is_initialized():
        raise ValueError("Unexpected CUDA")
    return result, bindings


def prepare(out):
    out = q.fresh_output(out)
    result, bindings = source_evidence()
    plan = expected_scope() | {"bindings_sha256": bindings,
        "model_digests": result["model_digests"], "source_plan_sha256": result["plan_sha256"],
        "source_diagnostic_sha256": acq.sha256(SOURCE / "diagnostic.json")}
    check_plan(plan)
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    print("INDEPENDENT FRONTEND VERIFY PLAN SEALED; existing36 rows only; no model forward", flush=True)


def verify(out):
    if (out / "verification.json").exists():
        raise ValueError("Existing independent verification: inspect, never overwrite/repeat")
    plan = acq.read_sealed(out / "plan.json")
    check_plan(plan); q.check_bindings(plan["bindings_sha256"])
    result, bindings = source_evidence()
    if (bindings != plan["bindings_sha256"] or result["model_digests"] != plan["model_digests"] or
        result["plan_sha256"] != plan["source_plan_sha256"] or acq.sha256(SOURCE / "diagnostic.json") != plan["source_diagnostic_sha256"]):
        raise ValueError("Changed immutable verification dependency/source")
    review = expected_scope() | {"plan_sha256": acq.sha256(out / "plan.json"),
        "source_plan_sha256": result["plan_sha256"], "source_diagnostic_sha256": plan["source_diagnostic_sha256"],
        "source_inputs_sha256": result["inputs_sha256"], "bindings_sha256": bindings,
        "model_digests": result["model_digests"], "symmetric_rows_verified": 36,
        "original_all_scope_identity_accounting_and_summary_gates_passed": True,
        "independently_verified": True, "initial182_verify_not_claimed_passed": True,
        "model_state_modes_rng_grad_unchanged": result["model_state_modes_rng_grad_unchanged"],
        "summary": result["summary"]}
    acq.write_new_json(out / "verification.json", acq.seal(review))
    print("INDEPENDENT FRONTEND VERIFIED36 model-batches216 slots; symmetric sealed rows; no forward; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    globals()[args.operation](args.out)
