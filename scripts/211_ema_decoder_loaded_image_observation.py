"""Two new version-only debug observations, NOT audio/runtime certification.

No existing process is attached. No media path is accepted. The only launches
are the two pinned EXEs with exactly -version. Actual load-event file handles
identify observed files; they do not resolve API-set/forwarder causality or prove
the conditional decode path. All audio, model, CUDA and training gates stay shut.
"""
from __future__ import annotations

from contextlib import ExitStack
import ctypes as c
from ctypes import wintypes as w
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/mel_ema_loaded_image_observation_20261004"
PURPOSE = "NONRELEASE_EMA211_VERSION_ONLY_ACTUAL_LOADED_FILE_OBSERVATION"
GUARD_SHA = "6b904f9f1b8497d16f3ecca85d9122b000604f4e357456eef0a0e9ddf28fe7d2"
PROOF_REL = "results/mel_ema_decoder_launch_request_monitor_20261004/request_interception_evidence.json"
PROOF_SHA = "fd1937ada6f393c824133be7668dc72a0502701cfcc46fe6b8b0b1aab52dc89a"
MAX_FILE_BYTES, MAX_EVENTS = 256 * 1024 * 1024, 1000
DBG_CONTINUE, DBG_NOT_HANDLED = 0x10002, 0x80010001
HEAVY = ("torch", "numpy", "scipy", "soundfile")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


require(sha(ROOT / "scripts/209_ema_decoder_target_guard.py") == GUARD_SHA, "Changed sealed209")
_spec = importlib.util.spec_from_file_location("ema211_metadata209", ROOT / "scripts/209_ema_decoder_target_guard.py")
meta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(meta)  # Metadata helpers only; NEVER construct old guard.


def now():
    return datetime.now(timezone.utc).isoformat()


class ExceptionRecord(c.Structure):
    _fields_ = [("code", w.DWORD), ("flags", w.DWORD), ("record", w.LPVOID),
                ("address", w.LPVOID), ("count", w.DWORD), ("information", c.c_size_t * 15)]


class ExceptionInfo(c.Structure):
    _fields_ = [("record", ExceptionRecord), ("first_chance", w.DWORD)]


class CreateInfo(c.Structure):
    _fields_ = [("file", w.HANDLE), ("process", w.HANDLE), ("thread", w.HANDLE),
                ("base", w.LPVOID), ("debug_offset", w.DWORD), ("debug_size", w.DWORD),
                ("tls", w.LPVOID), ("start", w.LPVOID), ("image_name", w.LPVOID), ("unicode", w.WORD)]


class LoadInfo(c.Structure):
    _fields_ = [("file", w.HANDLE), ("base", w.LPVOID), ("debug_offset", w.DWORD),
                ("debug_size", w.DWORD), ("image_name", w.LPVOID), ("unicode", w.WORD)]


class EventData(c.Union):
    _fields_ = [("exception", ExceptionInfo), ("create", CreateInfo), ("load", LoadInfo),
                ("exit_code", w.DWORD), ("unload_base", w.LPVOID)]


class DebugEvent(c.Structure):
    _fields_ = [("code", w.DWORD), ("pid", w.DWORD), ("tid", w.DWORD), ("data", EventData)]


class FileInfo(c.Structure):
    _fields_ = [("attributes", w.DWORD), ("created", w.FILETIME), ("accessed", w.FILETIME),
                ("written", w.FILETIME), ("volume", w.DWORD), ("size_high", w.DWORD),
                ("size_low", w.DWORD), ("links", w.DWORD), ("index_high", w.DWORD), ("index_low", w.DWORD)]


def assert_abi():
    require(os.name == "nt" and c.sizeof(w.LPVOID) == 8, "This isolated observer requires Windows x64")
    require((c.sizeof(ExceptionRecord), c.sizeof(ExceptionInfo), c.sizeof(CreateInfo),
             c.sizeof(LoadInfo), DebugEvent.data.offset, c.sizeof(DebugEvent), c.sizeof(FileInfo))
            == (152, 160, 72, 40, 16, 176, 52), "Unexpected Win32 ABI")
    require([getattr(CreateInfo, name).offset for name, _ in CreateInfo._fields_]
            == [0, 8, 16, 24, 32, 36, 40, 48, 56, 64], "Unexpected create-event member offsets")


class Native:
    def __init__(self):
        assert_abi()
        self.dll = c.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "WaitForDebugEventEx": ([c.POINTER(DebugEvent), w.DWORD], w.BOOL),
            "ContinueDebugEvent": ([w.DWORD, w.DWORD, w.DWORD], w.BOOL),
            "DebugSetProcessKillOnExit": ([w.BOOL], w.BOOL),
            "DebugActiveProcessStop": ([w.DWORD], w.BOOL),
            "GetFinalPathNameByHandleW": ([w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD], w.DWORD),
            "GetFileInformationByHandle": ([w.HANDLE, c.POINTER(FileInfo)], w.BOOL),
            "SetFilePointerEx": ([w.HANDLE, c.c_longlong, c.POINTER(c.c_longlong), w.DWORD], w.BOOL),
            "ReadFile": ([w.HANDLE, w.LPVOID, w.DWORD, c.POINTER(w.DWORD), w.LPVOID], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            func = getattr(self.dll, name)
            func.argtypes, func.restype = args, result

    def checked(self, name, *args):
        result = getattr(self.dll, name)(*args)
        if not result:
            raise c.WinError(c.get_last_error())
        return result

    def measure(self, handle):
        """Read ONLY the OS event's file handle, not a guessed DLL basename."""
        require(handle not in (None, 0, -1, c.c_void_p(-1).value), "Missing actual load-event file handle")
        buffer = c.create_unicode_buffer(32768)
        count = self.checked("GetFinalPathNameByHandleW", handle, buffer, len(buffer), 0)
        require(count < len(buffer), "Unbounded loaded image path")
        info = FileInfo()
        self.checked("GetFileInformationByHandle", handle, c.byref(info))
        size = (info.size_high << 32) | info.size_low
        require(0 < size <= MAX_FILE_BYTES, "Unbounded/empty loaded image")
        self.checked("SetFilePointerEx", handle, 0, None, 0)
        digest, total = hashlib.sha256(), 0
        block, used = c.create_string_buffer(1024 * 1024), w.DWORD()
        while total < size:
            self.checked("ReadFile", handle, block, min(len(block), size - total), c.byref(used), None)
            require(0 < used.value <= size - total, "Short/invalid actual image read")
            digest.update(block.raw[:used.value])
            total += used.value
        return {"final_path": buffer.value, "bytes": size, "sha256": digest.hexdigest(),
                "volume_serial": info.volume, "file_index": (info.index_high << 32) | info.index_low,
                "last_write_ticks": (info.written.dwHighDateTime << 32) | info.written.dwLowDateTime}


def command(label):
    require(type(label) is str and label in meta.EXPECTED, "Only two fixed version diagnostics")
    return [meta.EXPECTED[label]["target"], "-version"]


def continuation(event, initial_break_seen):
    if event.code != 1:
        return DBG_CONTINUE, initial_break_seen
    record = event.data.exception
    if not initial_break_seen and record.record.code == 0x80000003 and record.first_chance == 1:
        return DBG_CONTINUE, True  # Only the initial loader breakpoint is swallowed.
    return DBG_NOT_HANDLED, initial_break_seen


def check_bindings(bindings):
    for path, digest in bindings.items():
        require(sha(path) == digest, "Changed bound input: " + path)


def write_new(path, doc):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(doc, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def read_unit_log(path):
    data = Path(path).read_bytes()
    # PS5.1 Tee defaults to BOM UTF16; this host's newer shell may emit UTF8.
    # Decode exactly by BOM, not by a lossy fallback or changing the old log.
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return data.decode(encoding)


def observe(label, directory, native, journal):
    argv = command(label)
    started, tick = now(), time.monotonic()
    row = {"label": label, "argv": argv, "started_utc": started, "events": [],
           "actual_child_created": False, "debug_exit_code": None, "popen_exit_code": None,
           "detached_on_failure": False, "error": None}
    process, pending, handles, exited, break_seen = None, None, [], False, False
    journal({"kind": "version_request", "label": label, "argv": argv, "utc": started})
    try:
        with ExitStack() as stack:
            stdout = stack.enter_context((directory / (label + "_version_stdout.bin")).open("xb"))
            stderr = stack.enter_context((directory / (label + "_version_stderr.bin")).open("xb"))
            startup = subprocess.STARTUPINFO()
            startup.dwFlags, startup.wShowWindow = subprocess.STARTF_USESHOWWINDOW, 0
            process = subprocess.Popen(argv, executable=argv[0], shell=False, close_fds=True,
                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW | 2,
                                       env=None, cwd=None)
            row.update(actual_child_created=True, pid=process.pid)
            journal({"kind": "child_created", "label": label, "pid": process.pid, "utc": now()})
            native.checked("DebugSetProcessKillOnExit", False)
            while not exited:
                require(time.monotonic() - tick < 30, "Version debug observation reached30s; no retry/kill")
                event = DebugEvent()
                if not native.dll.WaitForDebugEventEx(c.byref(event), 100):
                    require(c.get_last_error() == 121, "Debug wait failed")
                    continue
                pending = event
                require(event.pid == process.pid, "No other process may be observed")
                require(len(row["events"]) < MAX_EVENTS, "Version event budget exceeded")
                item = {"sequence": len(row["events"]), "code": event.code, "pid": event.pid,
                        "tid": event.tid, "observed_utc": now()}
                # Add before measurement: retain failed/unresolved exposure in journal.
                row["events"].append(item)
                if event.code in (3, 6):
                    info = event.data.create if event.code == 3 else event.data.load
                    require(info.file not in handles, "Duplicate live debug file handle")
                    if info.file:
                        handles.append(info.file)
                    item.update(base_address=info.base, file=native.measure(info.file),
                                measured_from_actual_event_handle=True)
                    if event.code == 3:
                        require(sum(v["code"] == 3 for v in row["events"]) == 1, "Multiple create events")
                        require(os.path.normcase(item["file"]["final_path"].removeprefix("\\\\?\\")) == os.path.normcase(argv[0]),
                                "Actual process image is not the requested pinned target")
                        require(item["file"]["sha256"] == meta.EXPECTED[label]["target_sha256"], "Loaded target bytes differ")
                elif event.code == 1:
                    item.update(exception_code=event.data.exception.record.code,
                                first_chance=event.data.exception.first_chance)
                elif event.code == 5:
                    row["debug_exit_code"] = event.data.exit_code
                    exited = True
                elif event.code == 7:
                    item["unload_base"] = event.data.unload_base
                require(1 <= event.code <= 9 and event.code != 9, "Unknown/RIP debug event")
                status, break_seen = continuation(event, break_seen)
                item["continuation"] = status
                journal({"kind": "debug_event", "label": label, "event": item})
                native.checked("ContinueDebugEvent", event.pid, event.tid, status)
                pending = None
            row["popen_exit_code"] = process.wait(timeout=2)
        require(row["debug_exit_code"] == row["popen_exit_code"] == 0, "Actual version exit is not0")
        require(break_seen and sum(e["code"] == 3 for e in row["events"]) == 1,
                "Incomplete loader event sequence")
        row["observed_file_event_count"] = sum(e["code"] in (3, 6) for e in row["events"])
        require(row["observed_file_event_count"] > 1, "No actual DLL events")
        # Files remain held through EXIT; re-read the SAME event handles afterward.
        file_events = [e for e in row["events"] if e["code"] in (3, 6)]
        for handle, item in zip(handles, file_events):
            require(meta.typed(native.measure(handle)) == meta.typed(item["file"]), "Loaded file changed while held")
        row["held_file_handles_post_exit_equal"] = True
        for stream in ("stdout", "stderr"):
            path = directory / (label + "_version_" + stream + ".bin")
            require(path.stat().st_size <= 1024 * 1024, "Version metadata output exceeded1MiB")
            row[stream] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        row["completed_utc"] = now()
    except BaseException as error:
        row["error"] = {"type": type(error).__name__, "message": str(error)}
        journal({"kind": "failure", "label": label, "row": row})
        if process is not None and not exited:
            # Release ONLY this newly-created diagnostic. Never kill any process.
            if pending is not None:
                status, _ = continuation(pending, break_seen)
                native.checked("ContinueDebugEvent", pending.pid, pending.tid, status)
                pending = None
            native.checked("DebugActiveProcessStop", process.pid)
            row["detached_on_failure"] = True
        raise
    finally:
        for handle in reversed(handles):
            native.checked("CloseHandle", handle)
        journal({"kind": "row_final", "label": label, "row": row})
    return meta.seal(row)


def run():
    require(not OUT.exists(), "Existing diagnostic output retained; refuse repeat/overwrite")
    require(not any(v in sys.modules for v in HEAVY), "Heavy training/audio libraries forbidden")
    require(sha(ROOT / PROOF_REL) == PROOF_SHA, "Changed sealed210 evidence")
    proof = json.loads((ROOT / PROOF_REL).read_text(encoding="utf-8"))
    meta.check_seal(proof)
    bindings = dict(proof["bindings_sha256"])
    for rel in ("scripts/211_ema_decoder_loaded_image_observation.py", "scripts/_test_ema_decoder_loaded_image_observation.py",
                "docs/ema_loaded_image_observation_scope_20261004.json", "results/ema211_observer_unit_tests_attempt01.log",
                "results/ema211_observer_unit_tests_attempt02.log", "results/ema211_observer_unit_tests_attempt03.log"):
        bindings[str(ROOT / rel)] = sha(ROOT / rel)
    unit_log = read_unit_log(ROOT / "results/ema211_observer_unit_tests_attempt03.log")
    require("Ran 24 tests" in unit_log and "\nOK\n" in unit_log.replace("\r\n", "\n")
            and "FAILED (" not in unit_log, "New final24 observer units must actually pass first")
    bindings[str(ROOT / PROOF_REL)] = PROOF_SHA
    check_bindings(bindings)
    native, rng, environment = Native(), random.getstate(), dict(os.environ)
    OUT.mkdir()  # New diagnostic only, NOT a training plan or approval.
    journal_path = OUT / "actual_event_journal.jsonl"
    with journal_path.open("x", encoding="utf-8", newline="\n") as handle:
        def journal(value):
            handle.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
            handle.flush()
        write_new(OUT / "diagnostic_scope.json", {"purpose": PURPOSE, "budget": "exactly2 version-only children,30s each,1000events each",
                  "bindings_sha256": bindings, "music_inputs": 0, "training_authorized": False})
        rows = []
        try:
            for label in ("ffmpeg", "ffprobe"):
                rows.append(observe(label, OUT, native, journal))
            check_bindings(bindings)
            require(random.getstate() == rng and dict(os.environ) == environment, "Parent RNG/environment changed")
            require(not any(v in sys.modules for v in HEAVY), "Unexpected heavy library import")
        except BaseException as error:
            write_new(OUT / "failure.json", meta.seal({"purpose": PURPOSE, "completed_rows": rows,
                      "error": {"type": type(error).__name__, "message": str(error)},
                      "new_run_retry_authorized": False, "training_authorized": False, "utc": now()}))
            raise
    evidence = meta.seal({"purpose": PURPOSE, "checked_utc": now(), "bindings_sha256": bindings,
               "rows": rows, "actual_version_children": 2, "actual_audio_draws": 0,
               "student_pt_model_adam_cuda": 0, "closed_component_reruns": 0,
               "parent_python_rng_environment_unchanged": True,
               "observed_version_load_event_file_paths_and_bytes": True,
               "observed_files_authenticated_before_original_loading": False,
               "API_set_forwarder_SxS_causal_mapping_verified": False,
               "nondebug_launch_equivalence_verified": False,
               "conditional_audio_decode_runtime_verified": False,
               "Python_native_audio_runtime_verified": False,
               "historical208_actual_childpath_verified": False,
               "full_audio_backend_gate": "PENDING", "training_authorized": False, "release_selection": "NONE"})
    write_new(OUT / "loaded_image_evidence.json", evidence)
    print("VERSION_LOADED_IMAGE_EVIDENCE " + str(OUT / "loaded_image_evidence.json") + " SHA256=" + sha(OUT / "loaded_image_evidence.json"))
    print("ACTUAL_VERSION_CHILDREN=2 ACTUAL_AUDIO_DRAWS=0 TRAINING_UPDATES=0 FULL_RUNTIME=PENDING")


if __name__ == "__main__":
    require(sys.argv[1:] == ["observe-two-versions"], "Only explicit new version-only diagnostic entry")
    run()
