"""Unsealed direct decoder integration; not formal training authorization.

Reuse211 ABI/event measurement primitives, NEVER its closed version observer.
The prior observed physical file inventory is an allow-list, not a prediction
of API-set hosts by basename. Lock every allow-listed file BEFORE spawning and
hold through EXIT. Reject any unlisted actual loaded file before returning PCM.
This certifies only accepted debug-supervised calls, not historical208 calls,
nondebug equivalence, parent Python native imports or future decode availability.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
PINS = {
    "scripts/211_ema_decoder_loaded_image_observation.py": "4d700a762fa998450571eb993e0702c430dfed4a15d2cd5e22df200597513eeb",
    "scripts/210_ema_decoder_launch_request.py": "47bc072538ca057e73c80c0b83d86cbebcabd42227eda4b4095e47de158acd65",
    "results/mel_ema_loaded_image_observation_20261004/loaded_image_evidence.json":
        "5dd237dfeea7861bdddc519a6979997ebe5b460a8528d2ab1b605d9c818dbb75",
}
MAX_OUTPUT = 2 * 1024**3


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def load(name, relative):
    require(sha(ROOT / relative) == PINS[relative], "Changed sealed helper: " + relative)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


obs = load("ema213_debug_abi_only", "scripts/211_ema_decoder_loaded_image_observation.py")
request = load("ema213_original_argv_only", "scripts/210_ema_decoder_launch_request.py")
meta = obs.meta


def canonical_path(path):
    require(type(path) is str and path and "\0" not in path, "Exact physical Windows path")
    return os.path.normcase(os.path.normpath(path.removeprefix("\\\\?\\")))


def manifest_from_evidence(evidence):
    meta.check_seal(evidence)
    require(evidence["purpose"] == obs.PURPOSE and evidence["actual_version_children"] == 2,
            "Original version-file inventory only")
    files = {}
    require([row["label"] for row in evidence["rows"]] == ["ffmpeg", "ffprobe"], "Two original rows/order")
    for row in evidence["rows"]:
        meta.check_seal(row)
        require(row["debug_exit_code"] == row["popen_exit_code"] == 0
                and row["held_file_handles_post_exit_equal"] is True, "Original complete EXIT/held files")
        for event in row["events"]:
            if event["code"] not in (3, 6):
                continue
            require(event["measured_from_actual_event_handle"] is True, "No guessed physical dependency path")
            info = event["file"]
            require(list(info) == ["final_path", "bytes", "sha256", "volume_serial", "file_index", "last_write_ticks"]
                    and type(info["bytes"]) is int and 0 < info["bytes"] <= obs.MAX_FILE_BYTES,
                    "Actual finite event-file identity")
            name = canonical_path(info["final_path"])
            require(name not in files or meta.typed(files[name]) == meta.typed(info), "Conflicting observed physical file")
            files[name] = copy.deepcopy(info)
    require(len(files) == 37, "Original37 distinct physically observed files")
    for label, desc in meta.EXPECTED.items():
        require(files[canonical_path(desc["target"])]["sha256"] == desc["target_sha256"], "Direct target inventory mismatch")
    return meta.seal({"purpose": "EMA213_PRIOR_PHYSICAL_FILE_ALLOWLIST_NOT_ALL_DECODE_AVAILABILITY",
                      "files": files, "historical_audio_runtime_verified": False,
                      "Python_native_runtime_verified": False, "training_authorized": False})


def read_manifest():
    path = ROOT / "results/mel_ema_loaded_image_observation_20261004/loaded_image_evidence.json"
    require(sha(path) == PINS[str(path.relative_to(ROOT)).replace("\\", "/")], "Changed existing211 evidence")
    return manifest_from_evidence(json.loads(path.read_text(encoding="utf-8")))


def validate_loaded(actual, manifest, *, initial_target=None):
    name = canonical_path(actual["final_path"])
    require(name in manifest["files"] and meta.typed(actual) == meta.typed(manifest["files"][name]),
            "Actual loaded file not pre-bound/held: " + name)
    if initial_target is not None:
        require(name == canonical_path(initial_target), "Actual initial image is not explicit EXE")
    return name


class HeldFiles:
    """Read-only deny-WRITE/DELETE physical handles, measured before spawn."""
    def __init__(self, files, native):
        self.files, self.native, self.handles = copy.deepcopy(files), native, []
        dll = native.dll
        dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        dll.CreateFileW.restype = wintypes.HANDLE

    def __enter__(self):
        require(not self.handles, "No nested physical-file hold")
        try:
            for name, expected in self.files.items():
                handle = self.native.dll.CreateFileW(name, 0x80000000, 1, None, 3, 0x80, None)
                require(handle not in (None, 0, ctypes.c_void_p(-1).value), "Could not deny write/delete: " + name)
                self.handles.append(handle)
                require(meta.typed(self.native.measure(handle)) == meta.typed(expected), "Physical runtime changed BEFORE spawn")
            return self
        except BaseException:
            self.close()
            raise

    def check(self):
        require(len(self.handles) == len(self.files), "Full runtime hold missing")
        for handle, info in zip(self.handles, self.files.values()):
            require(meta.typed(self.native.measure(handle)) == meta.typed(info), "Held physical file changed")

    def close(self):
        for handle in reversed(self.handles):
            self.native.checked("CloseHandle", handle)
        self.handles.clear()

    def __exit__(self, typ, value, trace):
        try:
            if typ is None:
                self.check()
        finally:
            self.close()


def checked_request(argv, kwargs):
    """Exactly six original command/typed timeout shapes; change only argv0."""
    require(type(argv) is list and argv and type(kwargs) is dict, "Exact original check_output call")
    require(all(type(v) is str and "\0" not in v for v in argv), "Original string tokens only")
    require(argv[0] in meta.EXPECTED, "Bare original command label; caller cannot choose EXE")
    path = argv[argv.index("-i") + 1] if "-i" in argv else argv[-1]
    if argv[-1] == "-version":
        path = "NONEXISTENT_METADATA_ONLY"
    candidates = request.original_requests(path)  # Pure AST; no210 captures/children.
    matches = [r for r in candidates if meta.typed(r["argv"]) == meta.typed(argv)
               and meta.typed(r["kwargs"]) == meta.typed(kwargs)]
    require(len(matches) == 1, "Changed token, graph, sample rate, pipe or typed timeout")
    policy = request.explicit_request(matches[0])
    return policy


class DirectDecoder:
    """Accepted calls have complete per-child physical image event coverage.

    Unknown/missing images or exceptions fail closed and poison this adapter.
    No output reaches the audio loader until EXIT, pre/post held-file equality
    and complete pipe reads. Debug mode is intentional, NOT nondebug equality.
    """
    def __init__(self, manifest):
        meta.check_seal(manifest)
        require(manifest.get("training_authorized") is False and len(manifest["files"]) == 37
                and meta.typed(manifest) == meta.typed(read_manifest()),
                "Observed physical allow-list; not training approval")
        self.manifest = copy.deepcopy(manifest)
        self._digest = meta.digest(self.manifest)
        self.native = obs.Native()
        self.environment = dict(os.environ)
        self.cwd = str(Path.cwd().resolve())
        self.poisoned, self.active = False, False
        self.rows = []

    def guard(self):
        require(not self.poisoned and not self.active, "Poisoned/nested decoder adapter")
        require(meta.digest(self.manifest) == self._digest and dict(os.environ) == self.environment
                and str(Path.cwd().resolve()) == self.cwd, "Decoder manifest/environment/cwd changed")
        for path, expected in PINS.items():
            require(sha(ROOT / path) == expected, "Decoder integration dependency changed")

    def check_output(self, argv, **kwargs):
        self.guard()
        policy = checked_request(argv, kwargs)  # Reject BEFORE handles/process.
        self.active = True
        started = time.monotonic()
        row = {"request": copy.deepcopy(policy), "request_authority": "new213_debug_supervised_call_not210_authority",
               "started_utc": obs.now(), "events": [],
               "child_created": False, "exit_code": None, "error": None, "detached_on_failure": False}
        process, pending, event_handles, break_seen, exited = None, None, [], False, False
        buffers, reader_errors, readers = {"stdout": bytearray(), "stderr": bytearray()}, [], []
        try:
            with HeldFiles(self.manifest["files"], self.native) as held:
                startup = subprocess.STARTUPINFO()
                startup.dwFlags, startup.wShowWindow = subprocess.STARTF_USESHOWWINDOW, 0
                process = subprocess.Popen(policy["argv"], executable=policy["executable"],
                    shell=False, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW | 2, env=None, cwd=None)
                row.update(child_created=True, pid=process.pid, actual_application_name=policy["executable"],
                           pre_spawn_runtime_files_held=len(held.handles), debug_only_this_process=True)
                self.native.checked("DebugSetProcessKillOnExit", False)
                def consume(name):
                    try:
                        stream = getattr(process, name)
                        while True:
                            block = stream.read(1024 * 1024)
                            if not block:
                                break
                            require(len(buffers[name]) + len(block) <= MAX_OUTPUT, "Bounded PCM/stderr output exceeded")
                            buffers[name].extend(block)
                    except BaseException as error:
                        reader_errors.append(error)
                for name in buffers:
                    reader = threading.Thread(target=consume, args=(name,), daemon=True)
                    reader.start(); readers.append(reader)
                while not exited:
                    if "timeout" in kwargs:
                        require(time.monotonic() - started < kwargs["timeout"], "Original typed decoder timeout")
                    require(not reader_errors, "Pipe reader failed")
                    event = obs.DebugEvent()
                    if not self.native.dll.WaitForDebugEventEx(ctypes.byref(event), 100):
                        require(ctypes.get_last_error() == 121, "Native debug wait failed")
                        continue
                    pending = event
                    require(event.pid == process.pid and len(row["events"]) < 10000, "Unexpected child/event budget")
                    item = {"code": event.code, "pid": event.pid, "tid": event.tid}
                    row["events"].append(item)  # Record failed/unlisted exposure too.
                    if event.code in (3, 6):
                        info = event.data.create if event.code == 3 else event.data.load
                        require(info.file and info.file not in event_handles, "Missing/duplicate actual image handle")
                        event_handles.append(info.file)
                        actual = self.native.measure(info.file)
                        item["file"] = actual
                        name = validate_loaded(actual, self.manifest,
                                initial_target=policy["executable"] if event.code == 3 else None)
                        item["matched_preheld_physical_identity"] = True
                    elif event.code == 5:
                        row["exit_code"], exited = int(event.data.exit_code), True
                    elif event.code == 1:
                        item.update(exception_code=int(event.data.exception.record.code),
                                    first_chance=int(event.data.exception.first_chance))
                    require(1 <= event.code <= 8, "Unknown/RIP debug event")
                    continuation, break_seen = obs.continuation(event, break_seen)
                    item["continuation"] = continuation
                    self.native.checked("ContinueDebugEvent", event.pid, event.tid, continuation)
                    pending = None
                row["popen_exit_code"] = process.wait(timeout=2)
                for reader in readers:
                    reader.join(timeout=2)
                require(all(not t.is_alive() for t in readers) and not reader_errors, "Incomplete stdout/stderr pipe consumption")
                require(row["exit_code"] == row["popen_exit_code"] == 0 and break_seen
                        and sum(e["code"] == 3 for e in row["events"]) == 1, "Incomplete/nonzero actual decoder exit")
                for handle, event in zip(event_handles, (e for e in row["events"] if e["code"] in (3, 6))):
                    require(meta.typed(self.native.measure(handle)) == meta.typed(event["file"]), "Actual event file changed through EXIT")
                held.check()
                row.update(accepted_debug_call=True, held_files_post_exit_equal=True,
                           stdout_bytes=len(buffers["stdout"]), stdout_sha256=hashlib.sha256(buffers["stdout"]).hexdigest(),
                           stderr_bytes=len(buffers["stderr"]), stderr_sha256=hashlib.sha256(buffers["stderr"]).hexdigest())
            return bytes(buffers["stdout"])
        except BaseException as error:
            self.poisoned = True
            row["error"] = {"type": type(error).__name__, "message": str(error)}
            if process is not None and not exited:
                # Release ONLY this new read-only diagnostic, never attach or
                # terminate a user process. Failed/unaccepted PCM stays private.
                if pending is not None:
                    continuation, _ = obs.continuation(pending, break_seen)
                    self.native.checked("ContinueDebugEvent", pending.pid, pending.tid, continuation)
                self.native.checked("DebugActiveProcessStop", process.pid)
                row["detached_on_failure"] = True
            raise
        finally:
            for handle in reversed(event_handles):
                self.native.checked("CloseHandle", handle)
            if process is not None and exited:
                for name in buffers:
                    getattr(process, name).close()
            row["completed_utc"] = obs.now()
            self.rows.append(meta.seal(row))
            self.active = False


@contextmanager
def original_decoder_functions(core, bulk, adapter):
    """Scoped module-only facade; no file/global subprocess/PATH changes."""
    require(type(adapter) is DirectDecoder, "Exact isolated direct decoder adapter")
    old_core, old_bulk = core.subprocess, bulk.subprocess
    require(old_core is subprocess and old_bulk is subprocess, "Original unchanged subprocess identities")
    class Facade:
        check_output = staticmethod(adapter.check_output)
    facade = Facade()
    core.subprocess = bulk.subprocess = facade
    try:
        yield
        require(core.subprocess is facade and bulk.subprocess is facade, "Scoped facade changed")
    finally:
        core.subprocess, bulk.subprocess = old_core, old_bulk
