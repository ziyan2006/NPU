"""Call-scoped decoder hashing optimization; no training/launch authority.

The complete original213 process/event/pipe/timeout loop is compiled unchanged
from its pinned AST. Only its HeldFiles implementation is replaced locally.
All37 files still receive full SHA reads BEFORE spawn and AFTER EXIT. During
that SAME deny-WRITE/DELETE hold, actual load-event handles are matched by path,
volume, FILE_INDEX, size and write time to those byte-bound physical files.
No cache survives a call, handle close, failure or lock release. This is not an
adversarial-kernel, writable-mapping or filesystem integrity guarantee.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "scripts/213_ema_direct_decoder_runtime.py"
BASE_SHA = "c699c0f54b5817da6fc187ff065df757823857c852e8c9b3bdb0144a5d81af84"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_base():
    require(sha(BASE) == BASE_SHA, "Pinned decoder source changed")
    spec = importlib.util.spec_from_file_location("ema226_decoder_primitives", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Actual source SHA is checked before any import, native call or child process.
g = load_base()
FIELDS = ("final_path", "bytes", "volume_serial", "file_index", "last_write_ticks")


def checked_identity(actual, expected):
    require(type(actual) is dict and list(actual) == list(FIELDS), "Exact physical metadata fields")
    require(type(expected) is dict and set(expected) == set(FIELDS) | {"sha256"}, "Full byte-bound identity")
    require(type(actual["final_path"]) is str and actual["final_path"]
            and all(type(actual[k]) is int for k in FIELDS[1:]), "Exact physical metadata types")
    require(actual["bytes"] > 0 and actual["file_index"] > 0, "Nonempty identified physical file")
    require(type(expected["sha256"]) is str and len(expected["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in expected["sha256"]), "SHA-bound bytes required")
    require(all(g.meta.typed(actual[k]) == g.meta.typed(expected[k]) for k in FIELDS),
            "Actual event handle is not the continuously held physical file")
    # Schema/order stays compatible, but the SHA's provenance is explicitly
    # the independently read held handle, NOT another event-handle byte read.
    return {"final_path": actual["final_path"], "bytes": actual["bytes"], "sha256": expected["sha256"],
            "volume_serial": actual["volume_serial"], "file_index": actual["file_index"],
            "last_write_ticks": actual["last_write_ticks"]}


class ScopedNative(g.obs.Native):
    def __init__(self):
        super().__init__()
        self.custody = None
        self.full_reads = self.full_bytes = self.identity_reads = 0

    def full_measure(self, handle):
        result = g.obs.Native.measure(self, handle)
        self.full_reads += 1
        self.full_bytes += result["bytes"]
        return result

    def identity(self, handle):
        require(handle not in (None, 0, -1, ctypes.c_void_p(-1).value), "Missing actual file handle")
        path = ctypes.create_unicode_buffer(32768)
        count = self.checked("GetFinalPathNameByHandleW", handle, path, len(path), 0)
        require(count < len(path), "Unbounded event path")
        info = g.obs.FileInfo()
        self.checked("GetFileInformationByHandle", handle, ctypes.byref(info))
        self.identity_reads += 1
        return {"final_path": path.value, "bytes": (info.size_high << 32) | info.size_low,
                "volume_serial": info.volume, "file_index": (info.index_high << 32) | info.index_low,
                "last_write_ticks": (info.written.dwHighDateTime << 32) | info.written.dwLowDateTime}

    def measure(self, handle):
        require(self.custody is not None, "No cached identity outside an active per-call hold")
        return self.custody.event_identity(handle)


class CallProof:
    def __init__(self, files, native):
        require(type(native) is ScopedNative, "Exact call-scoped native adapter")
        self.files, self.native, self.handles = copy.deepcopy(files), native, []
        self.opened, self.post_checked, self.failed, self.used = False, False, False, False
        self.commitment = g.meta.digest(self.files)
        dll = native.dll
        dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        dll.CreateFileW.restype = wintypes.HANDLE

    def valid(self):
        require(self.opened and not self.failed and self.native.custody is self
                and len(self.handles) == len(self.files) and self.handles
                and g.meta.digest(self.files) == self.commitment, "Closed/partial/changed call custody")

    def __enter__(self):
        require(not self.used and not self.opened and not self.handles and self.native.custody is None,
                "No nested or reused custody")
        self.used = True
        try:
            for path, expected in self.files.items():
                handle = self.native.dll.CreateFileW(path, 0x80000000, 1, None, 3, 0x80, None)
                require(handle not in (None, 0, ctypes.c_void_p(-1).value), "Cannot deny WRITE/DELETE")
                self.handles.append(handle)
                require(g.meta.typed(self.native.full_measure(handle)) == g.meta.typed(expected),
                        "Full runtime SHA/physical identity changed BEFORE spawn")
            self.opened = True
            self.native.custody = self
            self.valid()
            return self
        except BaseException:
            self.failed = True
            self.close()
            raise

    def event_identity(self, handle):
        try:
            self.valid()
            actual = self.native.identity(handle)
            path = g.canonical_path(actual["final_path"])
            require(path in self.files, "Unlisted actual loaded file; no basename inference")
            return checked_identity(actual, self.files[path])
        except BaseException:
            self.failed = True
            raise

    def check(self):
        try:
            self.valid()
            for handle, expected in zip(self.handles, self.files.values()):
                if self.post_checked:
                    checked_identity(self.native.identity(handle), expected)
                else:
                    require(g.meta.typed(self.native.full_measure(handle)) == g.meta.typed(expected),
                            "Full held runtime SHA/identity changed AFTER EXIT")
            self.post_checked = True
        except BaseException:
            self.failed = True
            raise

    def close(self):
        # Invalidate BEFORE closing, including partial cleanup failures.
        self.native.custody = None
        self.opened = False
        handles, self.handles = self.handles, []
        first = None
        for handle in reversed(handles):
            try:
                self.native.checked("CloseHandle", handle)
            except BaseException as error:
                first = first or error
        if first is not None:
            self.failed = True
            raise first

    def __exit__(self, typ, value, trace):
        try:
            if typ is None:
                self.check()
            else:
                self.failed = True
        finally:
            self.close()


def unchanged_call(module):
    require(Path(module.__file__).resolve() == BASE and sha(BASE) == BASE_SHA, "Exact baseline decoder module")
    tree = ast.parse(BASE.read_text(encoding="utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DirectDecoder")
    original = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "check_output")
    namespace = dict(vars(module))
    namespace["HeldFiles"] = CallProof
    # No AST modification: request policy, debugger flags, events, timeouts,
    # output buffers, reader errors, EXIT0 and failure handling stay identical.
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(BASE), "exec"), namespace)
    return namespace["check_output"], hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest()


def make_decoder(module=None):
    module = g if module is None else module
    call, ast_digest = unchanged_call(module)
    adapter = module.DirectDecoder(module.read_manifest())
    adapter.native = ScopedNative()
    def optimized(instance, argv, **kwargs):
        before = len(instance.rows)
        counts = (instance.native.full_reads, instance.native.full_bytes, instance.native.identity_reads)
        try:
            return call(instance, argv, **kwargs)
        finally:
            for index in range(before, len(instance.rows)):
                old = instance.rows[index]
                g.meta.check_seal(old)
                body = {k: copy.deepcopy(v) for k, v in old.items() if k != "content_sha256"}
                body["request_authority"] = "new226_call_scoped_physical_custody_not_training_authority"
                body["hash_optimization"] = {
                    "unchanged_original213_call_AST_sha256": ast_digest,
                    "full_sha_before_spawn_and_after_EXIT": body.get("accepted_debug_call", False),
                    "event_sha_source": "same physical file under continuously held deny-WRITE/DELETE handles; metadata checked at every event",
                    "full_reads": instance.native.full_reads-counts[0],
                    "full_bytes": instance.native.full_bytes-counts[1],
                    "identity_reads": instance.native.identity_reads-counts[2],
                    "cross_call_cache": False, "training_authorized": False}
                instance.rows[index] = g.meta.seal(body)
    adapter.check_output = MethodType(optimized, adapter)
    return adapter


@contextmanager
def optimized_decoder_functions(core, bulk):
    """Drop-in input facade, not a trainer, activation or CLI launch."""
    adapter = make_decoder()
    with g.original_decoder_functions(core, bulk, adapter):
        yield adapter


if __name__ == "__main__":
    raise SystemExit("Component only: use new unit/synthetic benchmark; no training authority")
