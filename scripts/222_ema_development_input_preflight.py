"""New guarded31/177 DEV input preparation and actual Windows source leases.

No model, PT read, optimizer, forward/backward, CUDA initialization or training.
Original source selection/PCM/windows are preserved. Closed mechanisms are
not run. Prepared descriptors are compared with the existing full own-sealed
manifest; no historical DEV prediction or score is recomputed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA222_REAL_DEVELOPMENT_INPUT_AND_SOURCE_LEASE_PREFLIGHT"
SOURCE = "scripts/217_ema_actual_update_mechanism.py"
PINS = {
    SOURCE: "468734d23c353c805a87fa10f3c6d134a14bf789adabb2659fb63237e4bc71fd",
    "scripts/219_ema_continuous_training_core.py": "cfd19e0b4059164621c190eda4f74819e0c8b73b358396b94e8f1bbb7f3619ca",
    "scripts/220_ema_original_development_adapter.py": "1eb6be95980268c2d1ef239712073cbafbd8d20b11af14f82b9b1da10369c857",
    "scripts/221_ema_supervised_training.py": "f1eb83d4abe087a182c761aa6e2e736a07bac0084d71822b2a70816ad6f3cc65",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_files():
    for name, expected in PINS.items():
        require(sha(ROOT / name) == expected, "222 preflight dependency changed: " + name)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_report(report):
    require(report["purpose"] == PURPOSE and report["device"] == "cpu"
            and report["formal_training_updates"] == 0 and report["training_authorized"] is False
            and report["model_PT_Adam_forward_backward_calls"] == 0 and report["cuda_initialized"] is False
            and report["CPU_Python_NumPy_RNG_unchanged"] is True, "CPU preparation is not training/evaluation")
    require(report["development_manifest"]["tracks"] == 31
            and report["development_manifest"]["clips"] == 177
            and report["development_manifest"]["source_windows"] == 93
            and report["development_manifest"]["sha256"] == "e5990cdd2faf71e83469cbef7f94dea6b80d0b376ba49d5b81bca69fe442fb07"
            and report["actual_windows_lease"]["write_and_delete_denied_while_held"] is True
            and report["actual_windows_lease"]["wrong_sha_refused_and_handles_closed"] is True,
            "Actual complete original DEV preparation/Windows lease required")
    return True


def collector_ast():
    check_files()
    tree = ast.parse((ROOT / SOURCE).read_text(encoding="utf-8"))
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "collect"]
    require(len(nodes) == 1, "One SHA-bound native supervisor")
    class Change(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 900:
                return ast.copy_location(ast.Constant(1800), node)
            mapping = {"mechanism_result.json": "supervised_preflight_result.json",
                       "ACTUAL_MECHANISM_VERIFIED ": "DEV_INPUT_PREFLIGHT_VERIFIED ",
                       "ACTUAL_MECHANISM_ARCHIVE ": "DEV_INPUT_PREFLIGHT_ARCHIVE ",
                       "Bounded900s mechanism": "Bounded1800s new DEV input preparation",
                       "Fresh217 output, refuse overwrite/repeat": "Fresh222 preparation output, refuse overwrite/repeat"}
            if type(node.value) is str and node.value in mapping:
                return ast.copy_location(ast.Constant(mapping[node.value]), node)
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "require" and len(node.args) == 2:
                if isinstance(node.args[1], ast.Constant) and node.args[1].value == "Actual mechanism scope":
                    node.args[0] = ast.parse("validate_report(report)", mode="eval").body
                    node.args[1] = ast.Constant("Actual new DEV input preparation scope")
            return node
    return ast.fix_missing_locations(ast.Module(body=[Change().visit(nodes[0])], type_ignores=[]))


def collect(out, device):
    q = load("ema222_native_gate", SOURCE)
    launch = load("ema222_duplicate_inventory", "scripts/221_ema_supervised_training.py")
    require(device == "cpu", "New input-only preflight is CPU/no CUDA")
    def resource():
        record = q.r.resource()
        record["no_duplicate_task_preflight"] = launch.duplicate_preflight()
        return record
    namespace = dict(q.__dict__)
    namespace.update(__file__=__file__, PURPOSE=PURPOSE,
                     validate_exposure=lambda v: require(type(v) is dict and v == {}, "No model/update exposure"),
                     validate_report=validate_report,
                     r=type("ResourceBoundary", (), {"resource": staticmethod(resource)})(),
                     PINS=q.PINS | PINS)
    exec(compile(collector_ast(), "217native[222CPU-input-only]", "exec"), namespace)
    namespace["collect"](out, device)


def actual_lease_checks(t, native, out):
    """Actual Win32 sharing/identity checks on one NEW temporary fixture.

    CreateFile requests WRITE/DELETE access but never writes/deletes while
    held. Fixture is retained as evidence; no existing source is modified.
    """
    import ctypes
    path = out / "new_lease_fixture_Unicode_音频_&%!.bin"
    with path.open("xb") as handle:
        handle.write(b"new222-read-only-lease-fixture\x00\xff")
    expected = sha(path)
    with t.SourceReadLease({str(path): expected}, native) as lease:
        require(len(lease.handles) == 1, "Actual new fixture held")
        refusals = []
        for access in (0x40000000, 0x10000):
            attempted = native.dll.CreateFileW(str(path), access, 3, None, 3, 0x80, None)
            code = ctypes.get_last_error()
            if attempted not in (None, 0, ctypes.c_void_p(-1).value):
                native.checked("CloseHandle", attempted)
                raise ValueError("WRITE/DELETE access unexpectedly allowed during READ-only lease")
            require(code == 32, "Actual ERROR_SHARING_VIOLATION required")
            refusals.append({"desired_access": access, "error": code})
        lease.check(); before = copy.deepcopy(lease.before[0])
    require(not lease.handles, "All successful lease handles closed")
    after = native.dll.CreateFileW(str(path), 0x40000000, 3, None, 3, 0x80, None)
    require(after not in (None, 0, ctypes.c_void_p(-1).value), "Lease leaked/restricted after close")
    native.checked("CloseHandle", after)
    wrong = t.SourceReadLease({str(path): "0"*64}, native)
    try:
        wrong.__enter__()
    except ValueError as error:
        require("BEFORE draw" in str(error) and not wrong.handles, "Wrong-SHA failure closes all handles")
    else:
        wrong.close()
        raise ValueError("Wrong source SHA was accepted")
    require(sha(path) == expected, "Fixture bytes changed")
    return {"fixture": path.name, "sha256": expected, "actual_held_identity": before,
            "refused_access_requests": refusals, "write_and_delete_denied_while_held": True,
            "wrong_sha_refused_and_handles_closed": True, "fixture_retained_not_deleted": True}


def worker(out, device):
    check_files(); require(device == "cpu", "CPU input preparation only")
    out = Path(out).resolve()
    require(out.parent == ROOT / "results" and out.is_dir(), "Fresh supervisor-created output required")
    q = load("ema222_worker_native_gate", SOURCE)
    t = load("ema222_source_lease", "scripts/219_ema_continuous_training_core.py")
    d = load("ema222_old_DEV_adapter", "scripts/220_ema_original_development_adapter.py")
    a = t.load("ema222_original_input_components", "scripts/208_ema_authenticated_audio_input.py")
    import torch
    import torch._dynamo
    from contextlib import ExitStack
    from types import MethodType
    from unittest.mock import patch
    e, c = t.e, t.c
    require(not torch.cuda.is_initialized(), "Fresh no-CUDA preparation")
    def allowed():
        require(not (out / "supervisor_refused.json").exists(), "Native supervisor refused actual load")
    def forbidden(*args, **kwargs):
        raise RuntimeError("New input preparation forbids PT/model/forward/autograd/Adam/CUDA")
    decoder, native = q.g.DirectDecoder(q.g.read_manifest()), q.p.LargeNative()
    with ExitStack() as stack:
        stack.enter_context(c.source.zero_execution_guard())
        stack.enter_context(patch.object(torch, "load", forbidden))
        journal = stack.enter_context((out / "decoder_native_journal.jsonl").open("x", encoding="utf-8", buffering=1))
        original_decode = decoder.check_output
        def journaled(instance, argv, **kwargs):
            previous = len(instance.rows)
            try:
                return original_decode(argv, **kwargs)
            finally:
                for record in instance.rows[previous:]:
                    journal.write(json.dumps(record, ensure_ascii=True, allow_nan=False)+"\n")
                journal.flush()
        decoder.check_output = MethodType(journaled, decoder)
        approval = a.inp.verified_approval(a.inp.DEFAULT_APPROVAL)
        dataset = a.inp.ApprovedTeacherDataset(approval, "kim_melband")
        loss = e.OriginalBoundaryLoss("cpu")
        original_m = loss.function.__globals__["m"]
        stack.enter_context(q.g.original_decoder_functions(original_m.core, original_m.bulk, decoder))
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        rng = e.capture_rng("cpu")
        lease_result = actual_lease_checks(t, native, out)
        teacher_ids = [Path(row["source"]["path"]).name for row in dataset.rows]
        print("DEV_INPUT_PREFLIGHT preparing original31 songs /177views; model/Adam/CUDA=0", flush=True)
        evaluator = d.OriginalDevelopmentEvaluator.prepare(original_m, teacher_ids,
            lambda bindings: t.SourceReadLease(bindings, native), allowed)
        evaluator._check(); require(evaluator.exposure == {"forward_started": 0, "forward_completed": 0,
            "pair_started": 0, "pair_completed": 0}, "No DEV predictions or historical scoring")
        Corpus, _ = d.corpus_types(original_m)
        metadata_corpus = Corpus(original_m.bulk.OLD_LOCK, teacher_ids)
        bindings = d.development_source_bindings(metadata_corpus.lock)
        require(c.equal(e.capture_rng("cpu"), rng) and not torch.cuda.is_initialized(), "Preparation changed RNG/CUDA")
        journal.flush()
        report = q.p.seal_metadata({"purpose": PURPOSE, "device": "cpu", "exposure": {},
            "formal_training_updates": 0, "training_authorized": False,
            "model_PT_Adam_forward_backward_calls": 0, "cuda_initialized": False,
            "CPU_Python_NumPy_RNG_unchanged": True, "CPU_threads": torch.get_num_threads(),
            "actual_windows_lease": lease_result, "development_manifest": evaluator.manifest,
            "original_evaluator_binding": evaluator.binding, "development_source_bindings_sha256": bindings,
            "complete_input_target_descriptor_PCM_guard_passed": True,
            "decoder_request_count": len(decoder.rows), "decoder_native_journal_sha256": sha(out / "decoder_native_journal.jsonl"),
            "actual_DEV_predictions": 0, "actual_Adam_steps": 0, "release_selection": "NONE",
            "prepared_inputs_not_stored_or_used_as_new_independent_samples": True,
            "scope_note": "New entry-point input/native/Win32 integration only. Original old DEV score/model forward not repeated."})
        validate_report(report); check_files()
        q.obs.write_new(out / "input_preflight.json", report)
        print("DEV_INPUT_PREFLIGHT_VERIFIED "+json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("collect", "worker"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), required=True)
    args = parser.parse_args()
    (collect if args.operation == "collect" else worker)(args.out, args.device)
