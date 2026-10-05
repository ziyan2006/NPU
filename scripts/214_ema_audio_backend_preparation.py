"""Prepare the new real audio backend under an isolated Python loader trace.

Preparation only: no PCM draw, model/Adam construction, PT deserialize, forward,
CUDA initialization or training. Captures actual Python native image handles,
including transient loads/unloads, to construct a FUTURE pre-draw allow-list.
Import-time files are observed AFTER mapping; they are NOT pre-import certified.
The closed208 stream and211 version run are not constructed or repeated.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE = 2 * 1024**3
PINS = {
    "scripts/211_ema_decoder_loaded_image_observation.py": "4d700a762fa998450571eb993e0702c430dfed4a15d2cd5e22df200597513eeb",
    "scripts/208_ema_authenticated_audio_input.py": "8d208e34ed1ce6b18997c64d75ad1aa03d67401b7752fa8d995da23d64982825",
    "scripts/213_ema_direct_decoder_runtime.py": "c699c0f54b5817da6fc187ff065df757823857c852e8c9b3bdb0144a5d81af84",
    "scripts/212_ema_single_trajectory_engine.py": "8d10c304f3c3c2b4d2d6509dcec719e454b6afdabe651a329d1bc99b4a85d666",
}
PURPOSE = "NONRELEASE_EMA214_REAL_BACKEND_PREPARATION_NO_PCM_DRAW"
CONSOLE_OBSERVATION = "results/ema214_audio_backend_preparation_attempt02/preparation_result.json"
CONSOLE_OBSERVATION_SHA = "bf0a213244072a76622bbf6046f24068bc31c73e65391d1e667e6f664d86e0e2"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_files():
    for relative, expected in PINS.items():
        require(sha(ROOT / relative) == expected, "Changed integration dependency: " + relative)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_files()
obs = load("ema214_loader_abi_only", "scripts/211_ema_decoder_loaded_image_observation.py")


def typed_metadata(value):
    """214 schema: exact primitive types plus finite float HEX; no tensors.

    Do not apply209 integer-only seals to actual input config float fields.
    Closed209/205 algorithms and original seals remain untouched.
    """
    if type(value) is float:
        require(math.isfinite(value), "Finite input metadata float only")
        return ["float", value.hex()]
    if type(value) is dict:
        return ["dict", [[typed_metadata(k), typed_metadata(v)] for k, v in value.items()]]
    if type(value) in (list, tuple):
        return [type(value).__name__, [typed_metadata(v) for v in value]]
    return obs.meta.typed(value)


def seal_metadata(doc):
    require(type(doc) is dict and "content_sha256" not in doc, "New214 unsealed metadata dict")
    digest = hashlib.sha256(json.dumps(typed_metadata(doc), ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return doc | {"content_sha256": digest}


def check_metadata(doc):
    require(type(doc) is dict and "content_sha256" in doc, "Own214 seal required")
    payload = {k: v for k, v in doc.items() if k != "content_sha256"}
    require(seal_metadata(payload)["content_sha256"] == doc["content_sha256"], "Changed214 exact primitive/float metadata")


class LargeNative(obs.Native):
    """Own streaming byte reader for actual larger installed Torch DLLs."""
    def measure(self, handle):
        require(handle not in (None, 0, -1, ctypes.c_void_p(-1).value), "Missing actual native event file handle")
        buffer = ctypes.create_unicode_buffer(32768)
        count = self.checked("GetFinalPathNameByHandleW", handle, buffer, len(buffer), 0)
        require(count < len(buffer), "Actual image path too long")
        info = obs.FileInfo()
        self.checked("GetFileInformationByHandle", handle, ctypes.byref(info))
        size = (info.size_high << 32) | info.size_low
        require(0 < size <= MAX_IMAGE, "Bounded installed native image file")
        self.checked("SetFilePointerEx", handle, 0, None, 0)
        digest, total = hashlib.sha256(), 0
        block, used = ctypes.create_string_buffer(1024 * 1024), wintypes.DWORD()
        while total < size:
            self.checked("ReadFile", handle, block, min(len(block), size - total), ctypes.byref(used), None)
            require(0 < used.value <= size - total, "Short actual native image read")
            digest.update(block.raw[:used.value]); total += used.value
        return {"final_path": buffer.value, "bytes": size, "sha256": digest.hexdigest(),
                "volume_serial": info.volume, "file_index": (info.index_high << 32) | info.index_low,
                "last_write_ticks": (info.written.dwHighDateTime << 32) | info.written.dwLowDateTime}


def check_executable_identity(actual, expected):
    """Compare exact native file identity, not redirected/hard-link spelling."""
    names = ("bytes", "sha256", "volume_serial", "file_index", "last_write_ticks")
    require(any(all(obs.meta.typed(actual[name]) == obs.meta.typed(item[name]) for name in names)
                for item in expected), "Preparation executable is not a pre-bound preparation process image: " + actual["final_path"])


def observed_console_host():
    """Bind the exact auxiliary image actually refused in draft02, not a guess."""
    path = ROOT / CONSOLE_OBSERVATION
    require(sha(path) == CONSOLE_OBSERVATION_SHA, "Changed retained console-host discovery evidence")
    doc = json.loads(path.read_text(encoding="utf-8"))
    obs.meta.check_seal(doc)
    rows = [value["actual_executable_identity"] for value in doc["processes"].values()
            if value.get("actual_executable", "").lower().endswith("\\conhost.exe")]
    require(len(rows) == 1 and rows[0]["sha256"] == "e449bce01f275cd08f3d4e64bb73b3b43ae845a0dbdb3e6131426e66537705e5",
            "Exact actually observed console-host byte identity")
    return rows[0]


def worker_prepare():
    """New assembly using actual original pool/dataset, but NO next_batch."""
    a = load("ema214_original_input_components", "scripts/208_ema_authenticated_audio_input.py")
    e = load("ema214_new_single_trajectory", "scripts/212_ema_single_trajectory_engine.py")
    g = load("ema214_new_direct_decoder", "scripts/213_ema_direct_decoder_runtime.py")
    import torch
    require(not torch.cuda.is_initialized(), "Preparation is CPU/no CUDA")
    # Register installed optimizer rules before RNG snapshot; no Module, Adam
    # construction/step, compilation or accelerator probing is performed.
    import torch._dynamo
    before = e.capture_rng("cpu")
    header_calls, sf_module = [], a.inp.sf
    from types import SimpleNamespace
    def header_info(path):
        header_calls.append(str(path))
        return sf_module.info(path)
    from unittest.mock import patch
    from contextlib import ExitStack
    def forbidden(*args, **kwargs):
        raise RuntimeError("Preparation forbids PCM source decode/Module/autograd/Adam/PT/CUDA/children")
    with ExitStack() as stack:
        stack.enter_context(a.storage.source.zero_execution_guard())
        stack.enter_context(patch.object(torch, "load", forbidden))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        stack.enter_context(patch.object(a.inp.m.core, "decode_musdb", forbidden))
        stack.enter_context(patch.object(a.inp.m.bulk, "decode", forbidden))
        stack.enter_context(patch.object(a.inp.m.core.t23, "load_track", forbidden))
        approval = a.inp.verified_approval(a.inp.DEFAULT_APPROVAL)
        # Only THIS old module's sf reference is scoped; no soundfile global or
        # original function/file mutation. Count actual metadata header calls.
        a.inp.sf = SimpleNamespace(info=header_info)
        try:
            dataset = a.inp.ApprovedTeacherDataset(approval, "kim_melband")
        finally:
            a.inp.sf = sf_module
        true = a.inp.m.LockedTruePool(a.inp.m.bulk.OLD_LOCK, dataset.config)
        stream = e.DeviceInputStream(a.ORIGINAL_SAMPLER, true, dataset, a.data.crop_recipe, device="cpu")
        loss = e.OriginalBoundaryLoss("cpu")  # Full original matrices/functions; no model/forward.
    manifest = g.read_manifest()  # Read-only previous physical identities; no version child.
    require(e.equal(before, e.capture_rng("cpu")) and not torch.cuda.is_initialized(), "Preparation changed RNG/CUDA")
    require(not true.cache and not dataset.cache and stream.cursor == 4500 and stream.last_metadata is None,
            "No hidden PCM draw/cursor exposure")
    # Full structured traversal, not a truncated file listing. Actual routine
    # validators have already checked the whole lock/snapshot own schemas.
    records, files = true.lock["records"], true.lock["files"]
    require(type(records) is list and type(files) is dict and all(type(row) is dict for row in records), "Whole input-lock schema")
    referenced = set()
    for row in records:
        require(type(row["track_id"]) is str and type(row["domain"]) is str and type(row["role"]) is str,
                "Every input-lock record identity/role")
        for name in ("mix_files", "vocal_files", "stem_files"):
            paths = row.get(name, [])
            require(type(paths) is list and all(type(path) is str and path in files for path in paths), "Every input-lock file reference")
            referenced.update(paths)
    bindings = {str(ROOT / p): h for p, h in a.PINS.items()}
    bindings.update(approval["bindings_sha256"])
    for snapshot in approval["snapshots"].values():
        bindings[snapshot["path"]] = snapshot["sha256"]
    for row in dataset.rows:
        for item in (row["source"], *row["label_files"].values()):
            bindings[item["path"]] = item["sha256"]
    a.verify_files(bindings)
    # Bind all actually loaded Python-source modules under repository/runtime
    # roots. Native modules are bound by the host's real debug file events.
    imported = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).is_file() and Path(filename).suffix in (".py", ".pyd", ".dll"):
            imported[str(Path(filename).resolve())] = sha(filename)
    require(e.equal(before, e.capture_rng("cpu")), "Hash/inventory review changed training RNG")
    prepared = seal_metadata({"purpose": PURPOSE, "seal_schema": "214_exact_primitive_metadata_finite_float_hex_v1",
        "source_sampler": e.portable(a.ORIGINAL_SAMPLER),
        "bindings_sha256": bindings, "imported_python_files_sha256": imported,
        "whole_lock": {"records": len(records), "file_entries": len(files), "unique_referenced_files": len(referenced),
                       "entire_structured_tree_sha256": e.c.digest(true.lock), "manual_dependency_review_complete": False},
        "true_train_counts": {domain: len(rows) for domain, rows in true.pools.items()},
        "pseudo_ids": [row["song_id"] for row in dataset.rows], "input_config": e.portable(dataset.config),
        "original_loss_identity_sha256": e.c.digest(loss.identity), "decoder_allowlist_sha256": obs.meta.digest(manifest),
        "actual_pcm_draws": 0, "audio_header_info_calls": len(header_calls), "student_pt_deserializations": 0,
        "model_forward_adam_updates": 0, "cuda_initialized": False, "CPU_Python_NumPy_RNG_unchanged": True,
        "Python_native_pre_import_authenticated": False, "Python_native_pre_draw_authenticated": False,
        "training_authorized": False, "release_selection": "NONE"})
    print("AUDIO_BACKEND_PREPARED " + json.dumps(prepared, ensure_ascii=True, allow_nan=False), flush=True)


def collect(out):
    """New preparation process tree only, never attach/kill existing processes."""
    check_files()
    out = Path(out).resolve()
    require(out.parent == (ROOT / "results").resolve() and not out.exists(), "New direct child result directory; no repeat/overwrite")
    require(not any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile")), "Light stdlib-only observer host")
    native = LargeNative()
    # Resolve identity through actual read handles BEFORE launch. Normal venv
    # redirector/base chains and hard-link spelling are not guessed by basename.
    native.dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    native.dll.CreateFileW.restype = wintypes.HANDLE
    exe_handles, allowed = [], []
    console = observed_console_host()
    environment, started = dict(os.environ), time.monotonic()
    out.mkdir()
    row = {"purpose": PURPOSE, "started_utc": obs.now(), "source_hashes": dict(PINS),
           "worker_source_sha256": sha(__file__), "requests": [], "processes": {}, "error": None,
           "actual_pcm_draws": 0, "training_authorized": False}
    process, pending, active, handles, breaks = None, None, set(), [], set()
    pipe_errors, readers, buffers = [], [], {"stdout": bytearray(), "stderr": bytearray()}
    journal_path = out / "native_import_event_journal.jsonl"
    with journal_path.open("x", encoding="utf-8") as journal:
        def record(item):
            journal.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + "\n"); journal.flush()
        try:
            for filename in (sys.executable, sys._base_executable, console["final_path"]):
                handle = native.dll.CreateFileW(filename, 0x80000000, 1, None, 3, 0x80, None)
                require(handle not in (None, 0, ctypes.c_void_p(-1).value), "Unable to hold Python executable before launch")
                exe_handles.append(handle)
                info = native.measure(handle)
                require(info["sha256"] == sha(filename), "Python requested file/native identity differs before spawn")
                allowed.append(info)
            require(obs.meta.typed(allowed[-1]) == obs.meta.typed(console), "Observed console host changed before new launch")
            row["pre_bound_python_executables"] = allowed
            row["auxiliary_console_discovery_binding"] = {"path": CONSOLE_OBSERVATION, "sha256": CONSOLE_OBSERVATION_SHA}
            startup = subprocess.STARTUPINFO()
            startup.dwFlags, startup.wShowWindow = subprocess.STARTF_USESHOWWINDOW, 0
            argv = [sys.executable, str(Path(__file__).resolve()), "worker-prepare"]
            row["requests"].append({"argv": argv, "explicit_executable": sys.executable,
                                    "DEBUG_PROCESS": True, "shell": False, "env": None, "cwd": None})
            process = subprocess.Popen(argv, executable=sys.executable, shell=False, close_fds=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startup,
                creationflags=subprocess.CREATE_NO_WINDOW | 1, env=None, cwd=None)
            row["launcher_pid"] = process.pid
            native.checked("DebugSetProcessKillOnExit", False)
            def consume(name):
                try:
                    while True:
                        block = getattr(process, name).read(1024 * 1024)
                        if not block:
                            break
                        require(len(buffers[name]) + len(block) <= 16 * 1024**2, "Preparation metadata output too large")
                        buffers[name].extend(block)
                except BaseException as error:
                    pipe_errors.append(error)
            for name in buffers:
                reader = threading.Thread(target=consume, args=(name,), daemon=True)
                reader.start(); readers.append(reader)
            created, events = 0, 0
            while not created or active:
                require(time.monotonic() - started < 240 and not pipe_errors, "Preparation bounded240s/no pipe errors; no retry")
                event = obs.DebugEvent()
                if not native.dll.WaitForDebugEventEx(ctypes.byref(event), 100):
                    require(ctypes.get_last_error() == 121, "Native wait failed")
                    continue
                pending, events = event, events + 1
                require(events <= 10000 and 1 <= event.code <= 8, "Native event budget/known types")
                item = {"sequence": events, "pid": event.pid, "tid": event.tid, "code": event.code, "observed_utc": obs.now()}
                if event.code == 3:
                    require(event.pid not in active and created < 4, "Only bounded redirector/base Python/observed auxiliary console process tree")
                    created += 1; active.add(event.pid)
                    row["processes"][str(event.pid)] = {"events": 0, "image_file_events": 0, "exit_code": None}
                require(event.pid in active, "Only this new owned process tree")
                current = row["processes"][str(event.pid)]
                current["events"] += 1
                if event.code in (3, 6):
                    info = event.data.create if event.code == 3 else event.data.load
                    require(info.file and info.file not in handles, "Missing/duplicate actual image handle")
                    handles.append(info.file)
                    actual = native.measure(info.file)
                    item["file"] = actual
                    current["image_file_events"] += 1
                    if event.code == 3:
                        # Preserve the full observed image BEFORE validation so
                        # a refusal cannot discard the very identity at issue.
                        current["actual_executable"] = actual["final_path"]
                        current["actual_executable_identity"] = actual
                        record({"kind": "prevalidation_process_image", "event": item})
                        check_executable_identity(actual, allowed)
                        current["actual_executable"] = actual["final_path"]
                elif event.code == 1:
                    item.update(exception_code=int(event.data.exception.record.code), first_chance=int(event.data.exception.first_chance))
                elif event.code == 5:
                    current["exit_code"] = int(event.data.exit_code)
                    active.remove(event.pid)
                continuation, break_seen = obs.continuation(event, event.pid in breaks)
                if break_seen:
                    breaks.add(event.pid)
                item["continuation"] = continuation
                record(item)
                native.checked("ContinueDebugEvent", event.pid, event.tid, continuation)
                pending = None
            row["launcher_exit_code"] = process.wait(timeout=2)
            for reader in readers:
                reader.join(timeout=2)
            require(all(not t.is_alive() for t in readers) and not pipe_errors, "Preparation pipes not fully consumed")
            require(row["launcher_exit_code"] == 0 and all(p["exit_code"] == 0 for p in row["processes"].values()), "Preparation actual exit not0")
            lines = buffers["stdout"].decode("utf-8").splitlines()
            prepared_lines = [line for line in lines if line.startswith("AUDIO_BACKEND_PREPARED ")]
            require(len(prepared_lines) == 1, "Missing/duplicate prepared backend result")
            prepared = json.loads(prepared_lines[0].removeprefix("AUDIO_BACKEND_PREPARED "))
            check_metadata(prepared)
            require(prepared["purpose"] == PURPOSE and prepared["actual_pcm_draws"] == 0
                    and prepared["cuda_initialized"] is False and prepared["training_authorized"] is False,
                    "Preparation scope changed")
            row["prepared_backend"] = prepared
            check_files()
            require(environment == dict(os.environ), "Observer changed environment")
            # All OS load-event handles remain held through full process-tree EXIT.
            # Re-read each same handle; import observation is not a pre-import lock.
            unique = {}
            journal.flush()
            entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
            file_events = [item for item in entries if "file" in item and "sequence" in item]
            require(len(file_events) == len(handles), "Full native event handle accounting")
            for handle, item in zip(handles, file_events):
                require(obs.meta.typed(native.measure(handle)) == obs.meta.typed(item["file"]), "Actual Python-native file changed through EXIT")
                name = os.path.normcase(item["file"]["final_path"].removeprefix("\\\\?\\"))
                require(name not in unique or obs.meta.typed(unique[name]) == obs.meta.typed(item["file"]), "Native file identity conflict")
                unique[name] = item["file"]
            for handle, info in zip(exe_handles, allowed):
                require(obs.meta.typed(native.measure(handle)) == obs.meta.typed(info), "Preheld Python/console image changed through EXIT")
            row.update(native_physical_files=unique, native_file_event_count=len(file_events),
                       all_event_handles_post_exit_equal=True, Python_native_pre_import_authenticated=False,
                       Python_native_pre_draw_authenticated=False, completed_utc=obs.now())
        except BaseException as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
            record({"kind": "failure", "row": row})
            if pending is not None:
                continuation, _ = obs.continuation(pending, pending.pid in breaks)
                native.checked("ContinueDebugEvent", pending.pid, pending.tid, continuation)
            for pid in list(active):
                native.checked("DebugActiveProcessStop", pid)
            row["detached_new_owned_pids_on_failure"] = list(active)
            raise
        finally:
            for handle in reversed(handles):
                native.checked("CloseHandle", handle)
            for handle in reversed(exe_handles):
                native.checked("CloseHandle", handle)
            for name in buffers:
                (out / (name + ".bin")).write_bytes(buffers[name])
            obs.write_new(out / "preparation_result.json", seal_metadata(row))
    print("BACKEND_PREPARATION " + str(out / "preparation_result.json"))
    print(json.dumps({"processes": row["processes"], "native_file_events": row["native_file_event_count"],
                      "unique_native_files": len(row["native_physical_files"]), "audio_draws": 0,
                      "student_updates": 0, "CUDA_initialized": False, "training_ready": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("worker-prepare", "collect"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.operation == "worker-prepare":
        require(args.out is None, "Preparation worker stdout only")
        worker_prepare()
    else:
        require(args.out is not None, "Explicit fresh preparation result directory required")
        collect(args.out)
