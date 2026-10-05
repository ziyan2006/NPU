"""Fresh CUDA runtime/RNG restoration observation, zero model or updates.

The215 zero-update model-audio gate must pass first. Known229 actual native
files are preheld; newly observed CUDA-native files are measured/held only AFTER
mapping and are inventory for the NEXT preheld mechanism, not update authority.
No model/Adam/forward/backward/PCM/DEV/output audio, no original training PT.
No existing process is attached or terminated. Shared GPU utilization is allowed.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA216_CUDA_RUNTIME_RNG_OBSERVATION_ZERO_MODEL_UPDATES"
AUDIT = "results/ema215_real_model_audio_audit_attempt01/supervised_audit_result.json"
REVIEW = "results/ema215_real_model_audio_audit_attempt01/independent_readonly_review.json"
PINS = {
    "scripts/215_ema_model_audio_audit.py": "8e9cce90b934b465e8946d805a53a7d34a881de1d752f3634ccc8f0ea72a5b3c",
    AUDIT: "82b84cd280d2e91061e2f94db2aadb329c7fa9e301b7a0865b6846e5798b0014",
    REVIEW: "57d98b07712eeaec630b24397205b142b5cea42d2f93f74e7726b6013ea81dd4",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_files():
    for relative, expected in PINS.items():
        require(sha(ROOT / relative) == expected, "Changed216 prior gate: " + relative)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_files()
a = load("ema216_known_cpu_native_gate", "scripts/215_ema_model_audio_audit.py")
p, g, obs = a.p, a.g, a.obs


def gate():
    check_files()
    audit = json.loads((ROOT / AUDIT).read_text(encoding="utf-8"))
    review = json.loads((ROOT / REVIEW).read_text(encoding="utf-8"))
    p.check_metadata(audit); p.check_metadata(audit["audit_report"]); p.check_metadata(review)
    require(audit["error"] is None and audit["launcher_exit_code"] == 0
            and audit["actual_debug_supervised_native_audio_scope_passed"] is True
            and review["verified_actual_audio_draws"] == 1
            and review["verified_actual_microbatch_forwards"] == 6
            and review["CUDA_initialized"] is False, "Completed independent real CPU zero-update audio gate")
    require(review["bound_artifacts_sha256"][str(ROOT / AUDIT)] == PINS[AUDIT], "Independent review binds exact audit")
    return audit


def resource():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15, check=True)
    rows = [row.split(",") for row in result.stdout.strip().splitlines()]
    require(len(rows) == 1 and len(rows[0]) == 3, "One actual readable GPU")
    index, free, utilization = [int(field.strip()) for field in rows[0]]
    require(index == 0 and free >= 2300 and shutil.disk_usage(ROOT).free >= 12 * 1024**3,
            "Fresh resource gate: freeVRAM>=2300MiB/disk>=12GiB")
    return {"checked_utc": obs.now(), "index": index, "free_mib": free,
            "utilization_percent": utilization, "shared_GPU_utilization_authorized": True}


def worker(out):
    gate()
    e = load("ema216_rng_helpers_only", "scripts/212_ema_single_trajectory_engine.py")
    import torch
    import torch._dynamo
    from unittest.mock import patch
    from contextlib import ExitStack
    require(not torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "Fresh explicit source CUDA workspace before initialization")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(out).resolve()
    require(out.parent == (ROOT / "results").resolve() and out.is_dir(), "Owned diagnostic directory")
    def forbidden(*args, **kwargs):
        raise RuntimeError("Runtime observation forbids model/Adam/forward/autograd/audio")
    def allowed():
        require(not (out / "supervisor_refused.json").exists(), "Supervisor runtime refusal")
    with ExitStack() as stack:
        for obj, name in ((torch.nn.Module, "__init__"), (torch.nn.Module, "_call_impl"),
                          (torch.optim.Adam, "__init__"), (torch.optim.Adam, "step"),
                          (torch.autograd, "backward"), (torch.autograd, "grad")):
            stack.enter_context(patch.object(obj, name, forbidden))
        source_path = ROOT / e.j.STORAGE_REL
        require(sha(source_path) == e.j.STORAGE_SHA, "Pinned source CPU storage before read")
        saved = torch.load(source_path, map_location="cpu", weights_only=True)
        require(sha(source_path) == e.j.STORAGE_SHA, "Pinned source storage changed")
        e.c.check_seal(saved)
        source = saved["parent"]["parent_metadata"]
        expected = source["runtime"]
        require(expected["device"] == "cuda" and expected["workspace"] == ":4096:8"
                and expected["threads"] == 4 and expected["deterministic"] is True,
                "Exact original CUDA source runtime, no CPU substitute")
        before_cpu = e.capture_rng("cpu")
        allowed()
        print("CUDA_RUNTIME_PHASE explicit_initialization_no_model", flush=True)
        torch.cuda.init()
        actual = e.runtime_identity("cuda")
        require(e.equal(actual, expected), "Actual CUDA runtime differs from immutable4500 source")
        active = e.capture_rng("cuda")
        require(e.equal(active | {"torch_cuda": []}, before_cpu), "CUDA init changed CPU/Python/NumPy RNG")
        original_rng = e.portable(source["rng"])
        e.validate_rng(original_rng, "cuda")
        allowed()
        e.restore_rng(original_rng, "cuda")
        restored = e.capture_rng("cuda")
        require(e.equal(original_rng, restored), "Original CPU/Python/NumPy/CUDA RNG restoration differs by bits/type")
        # No random tensor draw; this gate measures runtime + state restoration,
        # not actual update, nonlinear frontend or CUDA transaction equivalence.
        torch.cuda.synchronize()
        allowed()
        report = p.seal_metadata({"purpose": PURPOSE, "actual_source_runtime": actual,
            "runtime_equal_immutable_source": True, "source_cpu_storage_reads": 1,
            "source_training_PT_reads": 0, "actual_model_constructions": 0,
            "actual_Adam_constructions_or_steps": 0, "actual_forward_backward_audio_draws": 0,
            "CUDA_explicitly_initialized": True, "original_complete_RNG_restore_bit_type_equal": True,
            "original_RNG_typed205_digest": e.c.digest(original_rng),
            "restored_RNG_typed205_digest": e.c.digest(restored),
            "CUDA_RNG_byte_lengths": [state.numel() for state in restored["torch_cuda"]],
            "cuda_allocated_bytes": torch.cuda.memory_allocated(), "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "CUDA_update_mechanism_verified": False, "training_authorized": False, "release_selection": "NONE"})
        print("CUDA_RUNTIME_OBSERVED " + json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)


def collect(out):
    prior = gate()
    out = Path(out).resolve()
    require(out.parent == (ROOT / "results").resolve() and not out.exists(), "Fresh216 runtime observation, no repeat/overwrite")
    require(not any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile")), "Light observer only")
    preflight = resource()
    native = p.LargeNative()
    known = prior["preheld_native_physical_files"]
    native.dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    native.dll.CreateFileW.restype = wintypes.HANDLE
    out.mkdir()
    source_handles, event_handles, observed, active, breaks = [], [], {}, set(), set()
    process, pending = None, None
    buffers, readers, pipe_errors = {"stdout": bytearray(), "stderr": bytearray()}, [], []
    row = {"purpose": PURPOSE, "started_utc": obs.now(), "resource_preflight": preflight,
           "worker_source_sha256": sha(__file__), "bindings_sha256": dict(PINS), "processes": {},
           "newly_observed_native_files": {}, "error": None, "training_authorized": False}
    started = time.monotonic()
    journal_path = out / "native_event_journal.jsonl"
    with journal_path.open("x", encoding="utf-8") as journal:
        def record(item):
            journal.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + "\n"); journal.flush()
        try:
            bindings = prior["preheld_source_bindings_sha256"] | {str(ROOT / relative): expected for relative, expected in PINS.items()}
            bindings[str(Path(__file__).resolve())] = sha(__file__)
            for name, expected in bindings.items():
                handle = native.dll.CreateFileW(name, 0x80000000, 1, None, 3, 0x80, None)
                require(handle not in (None, 0, ctypes.c_void_p(-1).value), "Cannot prehold runtime source bytes: " + name)
                source_handles.append(handle)
                require(sha(name) == expected, "Changed bound runtime source")
            row["preheld_source_bindings_sha256"] = bindings
            with g.HeldFiles(known, native) as held:
                row["preheld_known_native_files"] = known
                startup = subprocess.STARTUPINFO()
                startup.dwFlags, startup.wShowWindow = subprocess.STARTF_USESHOWWINDOW, 0
                argv = [sys.executable, "-B", "-X", "pycache_prefix=" + str(out / "unused_pycache"),
                        str(Path(__file__).resolve()), "worker", "--out", str(out)]
                environment = dict(os.environ) | {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
                row["request"] = {"argv": argv, "executable": sys.executable, "shell": False,
                                  "DEBUG_PROCESS": True, "child_workspace_override": ":4096:8",
                                  "parent_environment_modified": False, "cwd": None}
                process = subprocess.Popen(argv, executable=sys.executable, shell=False, close_fds=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startup,
                    creationflags=subprocess.CREATE_NO_WINDOW | 1, env=environment, cwd=None)
                row["launcher_pid"] = process.pid
                native.checked("DebugSetProcessKillOnExit", False)
                def consume(name):
                    try:
                        while True:
                            block = getattr(process, name).read(1024 * 1024)
                            if not block:
                                break
                            require(len(buffers[name]) + len(block) <= 16 * 1024**2, "Bounded runtime metadata pipe")
                            buffers[name].extend(block)
                    except BaseException as error:
                        pipe_errors.append(error)
                for name in buffers:
                    thread = threading.Thread(target=consume, args=(name,), daemon=True)
                    thread.start(); readers.append(thread)
                created, events = 0, 0
                while not created or active:
                    require(time.monotonic() - started < 300 and not pipe_errors, "Bounded300s CUDA runtime observation")
                    event = obs.DebugEvent()
                    if not native.dll.WaitForDebugEventEx(ctypes.byref(event), 100):
                        require(ctypes.get_last_error() == 121, "Native runtime wait failed")
                        continue
                    pending, events = event, events + 1
                    require(events <= 20000 and 1 <= event.code <= 8, "Known bounded native events")
                    item = {"sequence": events, "pid": event.pid, "tid": event.tid, "code": event.code}
                    if event.code == 3:
                        require(event.pid not in active and created < 4, "Owned redirector/base/console chain")
                        active.add(event.pid); created += 1
                        row["processes"][str(event.pid)] = {"events": 0, "image_file_events": 0, "exit_code": None}
                    require(event.pid in active, "Owned runtime observation PID only")
                    current = row["processes"][str(event.pid)]; current["events"] += 1
                    if event.code in (3, 6):
                        info = event.data.create if event.code == 3 else event.data.load
                        require(info.file and info.file not in event_handles, "Actual unique load-event file handle")
                        event_handles.append(info.file)
                        actual = native.measure(info.file); item["file"] = actual
                        current["image_file_events"] += 1
                        record({"kind": "prevalidation_image", "event": item})
                        if event.code == 3:
                            p.check_executable_identity(actual, a.preparation()["pre_bound_python_executables"])
                        name = g.canonical_path(actual["final_path"])
                        require(name not in observed or obs.meta.typed(observed[name]) == obs.meta.typed(actual), "Conflicting observed native identity")
                        observed[name] = actual
                        try:
                            a.validate_native(actual, known)
                            item["matched_preheld_known_identity"] = True
                        except ValueError:
                            # This worker cannot execute models/updates/audio.
                            # An unknown loaded DLL is inventory, NOT approval.
                            item["matched_preheld_known_identity"] = False
                            item["new_cuda_native_observation_only_not_preimport_certified"] = True
                            row["newly_observed_native_files"][name] = actual
                    elif event.code == 1:
                        item.update(exception_code=int(event.data.exception.record.code), first_chance=int(event.data.exception.first_chance))
                    elif event.code == 5:
                        current["exit_code"] = int(event.data.exit_code); active.remove(event.pid)
                    continuation, seen = obs.continuation(event, event.pid in breaks)
                    if seen:
                        breaks.add(event.pid)
                    item["continuation"] = continuation
                    record(item)
                    native.checked("ContinueDebugEvent", event.pid, event.tid, continuation); pending = None
                row["launcher_exit_code"] = process.wait(timeout=2)
                for thread in readers:
                    thread.join(timeout=2)
                require(all(not thread.is_alive() for thread in readers) and not pipe_errors, "Complete runtime output pipes")
                require(row["launcher_exit_code"] == 0 and all(r["exit_code"] == 0 for r in row["processes"].values()), "Runtime observation actual EXIT not0")
                reports = [json.loads(line.removeprefix("CUDA_RUNTIME_OBSERVED ")) for line in buffers["stdout"].decode("utf-8").splitlines()
                           if line.startswith("CUDA_RUNTIME_OBSERVED ")]
                require(len(reports) == 1, "One complete runtime result")
                report = reports[0]; p.check_metadata(report)
                require(report["runtime_equal_immutable_source"] is True
                        and report["original_complete_RNG_restore_bit_type_equal"] is True
                        and report["actual_model_constructions"] == report["actual_Adam_constructions_or_steps"] == 0
                        and report["actual_forward_backward_audio_draws"] == 0
                        and report["training_authorized"] is False, "Runtime observation scope")
                entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
                images = [item for item in entries if "sequence" in item and "file" in item]
                require(len(images) == len(event_handles), "Full native runtime handle coverage")
                for handle, item in zip(event_handles, images):
                    require(obs.meta.typed(native.measure(handle)) == obs.meta.typed(item["file"]), "Native runtime bytes changed through EXIT")
                held.check()
                for name, expected in bindings.items():
                    require(sha(name) == expected, "Source bytes changed through runtime EXIT")
                check_files()
                row.update(runtime_report=report, native_file_events=len(images), all_event_handles_post_exit_equal=True,
                    observed_native_physical_files=observed, new_native_preimport_authenticated=False,
                    CUDA_update_mechanism_verified=False, full_training_ready=False, completed_utc=obs.now())
        except BaseException as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
            record({"kind": "failure", "row": row})
            obs.write_new(out / "supervisor_refused.json", p.seal_metadata(row["error"]))
            if pending is not None:
                continuation, _ = obs.continuation(pending, pending.pid in breaks)
                native.checked("ContinueDebugEvent", pending.pid, pending.tid, continuation)
            for pid in list(active):
                native.checked("DebugActiveProcessStop", pid)
            row["detached_new_owned_pids_on_failure"] = list(active)
            raise
        finally:
            for handle in reversed(event_handles):
                native.checked("CloseHandle", handle)
            for handle in reversed(source_handles):
                native.checked("CloseHandle", handle)
            for name in buffers:
                (out / (name + ".bin")).write_bytes(buffers[name])
            obs.write_new(out / "runtime_observation.json", p.seal_metadata(row))
    print("CUDA_RUNTIME_RNG_OBSERVATION " + str(out / "runtime_observation.json"), flush=True)
    print(json.dumps({"runtime_matches_source": True, "complete_source_RNG_restored": True,
        "actual_model_updates": 0, "new_native_files_observed": len(row["newly_observed_native_files"]),
        "CUDA_update_mechanism_verified": False, "full_training_ready": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("collect", "worker"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "collect":
        collect(args.out)
    else:
        worker(args.out)
