"""Prospective decoder TARGET byte guard; NOT complete runtime certification.

No decoder launch, audio draw, model, optimizer, CUDA or training entry point.
Strictly recognizes the two existing wrappers, pins their real EXE targets,
and inventories normal/delay PE imports without pretending to resolve the
Windows loader. API sets, forwarders, SxS and dynamic loads remain PENDING.
This module never upgrades208's historical, wrapper-only evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA_DECODER_TARGET_GUARD_NOT_FULL_RUNTIME"
WRAPPER_ROOT = Path(r"C:\Users\30519\AppData\Local\Microsoft\WindowsApps")
EXPECTED = {
    "ffmpeg": {"wrapper": str(WRAPPER_ROOT / "ffmpeg.cmd"),
               "wrapper_sha256": "632b62597b7f0a982601dcb0bafe59029ba9dbc21837abe0f4235aaef01db982",
               "target": r"C:\ffmpeg\bin\ffmpeg.exe",
               "target_sha256": "72a489eccd008c2ec2c0a5856c5c75bc3d8bbfa90166c4566865c246445e6aa3"},
    "ffprobe": {"wrapper": str(WRAPPER_ROOT / "ffprobe.cmd"),
                "wrapper_sha256": "2b2905bfb7bdfcb7e437e04341871f8e40a7d9851d4f1c358dc8434509df1b5b",
                "target": r"C:\ffmpeg\bin\ffprobe.exe",
                "target_sha256": "19202b23c0043f15ad1b7bce2344f406fd52bd6efd8f995ce02e7392a1cec52f"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def typed(value):
    if type(value) is dict:
        return ["dict", [[typed(k), typed(v)] for k, v in value.items()]]
    if type(value) in (list, tuple):
        return [type(value).__name__, [typed(v) for v in value]]
    if type(value) in (str, int, bool, type(None)):
        return [type(value).__name__, value]
    raise ValueError("Only exact finite integer/string metadata in209")


def digest(value):
    return hashlib.sha256(json.dumps(typed(value), ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(doc):
    require(type(doc) is dict and "content_sha256" not in doc, "No resealing")
    return {**doc, "content_sha256": digest(doc)}


def check_seal(doc):
    require(type(doc) is dict and type(doc.get("content_sha256")) is str, "Own seal required")
    require(digest({k: v for k, v in doc.items() if k != "content_sha256"}) == doc["content_sha256"],
            "Changed209 own typed seal")


def compare_sealed(left, right):
    check_seal(left)
    check_seal(right)
    require(typed(left) == typed(right), "Full symmetric typed documents differ")


def wrapper_target(data):
    require(type(data) is bytes and len(data) < 1024, "Bounded wrapper bytes")
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeError as error:
        raise ValueError("ASCII fixed wrapper required") from error
    require(len(lines) == 2 and lines[0] == "@echo off", "Only reviewed two-line wrapper")
    match = re.fullmatch(r'"([A-Za-z]:\\[^"%&|<>^!()\r\n]+\.exe)" %\*', lines[1])
    require(match is not None, "No shell chaining, variable expansion or unquoted target")
    return match.group(1)


def pe_imports(data):
    """Bounded on-disk PE32/PE32+ import inventory, never loads the image.

    RVA mapped through actual raw sections (not VA=file offset). Delay VA
    format is explicitly rejected rather than silently treated as an RVA.
    This is an inventory, not a loader/ABI trust or complete dependency proof.
    """
    require(type(data) is bytes and len(data) >= 64 and data[:2] == b"MZ", "DOS image header")

    def unpack(fmt, offset):
        size = struct.calcsize(fmt)
        require(type(offset) is int and 0 <= offset <= len(data)-size, "PE file bounds")
        return struct.unpack_from(fmt, data, offset)

    pe = unpack("<I", 0x3c)[0]
    require(data[pe:pe+4] == b"PE\0\0", "PE signature")
    machine, sections = unpack("<HH", pe+4)
    optional_size = unpack("<H", pe+20)[0]
    optional = pe+24
    require(1 <= sections <= 96 and optional+optional_size <= len(data), "PE section/optional bounds")
    magic = unpack("<H", optional)[0]
    require(magic in (0x10b, 0x20b), "PE32 or PE32+ only")
    directory = optional+(96 if magic == 0x10b else 112)
    require(directory <= optional+optional_size, "Complete fixed optional header required")
    count = unpack("<I", directory-4)[0]
    require(count <= 16 and directory+count*8 <= optional+optional_size, "PE directory bounds")
    header_size = unpack("<I", optional+60)[0]
    require(header_size <= len(data), "PE raw headers bound")
    table = optional+optional_size
    require(table+sections*40 <= len(data), "Complete section table required")
    raw_sections = []
    for index in range(sections):
        virtual_size, address, raw_size, pointer = unpack("<IIII", table+index*40+8)
        require(pointer+raw_size <= len(data), "PE raw section extends beyond file")
        raw_sections.append((address, raw_size, pointer))

    def raw(rva, size):
        require(type(rva) is int and rva > 0 and size >= 0, "Nonzero PE RVA")
        if rva+size <= header_size:
            return rva
        matches = [pointer+rva-address for address, length, pointer in raw_sections
                   if address <= rva and rva+size <= address+length]
        require(len(matches) == 1, "RVA must map to exactly one raw section")
        return matches[0]

    def string(rva):
        offset = raw(rva, 1)
        end = data.find(b"\0", offset, min(len(data), offset+513))
        require(end != -1, "Bounded terminated PE name")
        raw(rva, end-offset+1)
        try:
            return data[offset:end].decode("ascii")
        except UnicodeError as error:
            raise ValueError("ASCII PE import name") from error

    word = 4 if magic == 0x10b else 8
    word_format = "<I" if word == 4 else "<Q"
    ordinal_mask = 1 << (word*8-1)

    def symbols(rva):
        require(rva != 0, "Unbound import lookup table required")
        result = []
        for index in range(65536):
            value = unpack(word_format, raw(rva+index*word, word))[0]
            if value == 0:
                return result
            if value & ordinal_mask:
                require(value & ~(ordinal_mask | 65535) == 0, "Ordinal import reserved bits")
                result.append({"ordinal": value & 65535})
            else:
                raw(value, 2)
                result.append({"name": string(value+2)})
        raise ValueError("Unterminated or excessive PE import symbols")

    def imports(directory_index, delay):
        if directory_index >= count:
            return []
        rva, size = unpack("<II", directory+directory_index*8)
        require((rva == 0) == (size == 0), "PE import directory address/size mismatch")
        if rva == 0:
            return []
        stride = 32 if delay else 20
        require(stride <= size <= 16*1024*1024, "Bounded PE import directory")
        result = []
        for index in range(min(size//stride, 4096)):
            fields = unpack("<8I" if delay else "<5I", raw(rva+index*stride, stride))
            if not any(fields):
                return result
            if delay:
                require(fields[0] == 1, "Delay VA/unknown attributes require separate adapter")
                name_rva, lookup = fields[1], fields[4]
            else:
                name_rva, lookup = fields[3], fields[0]
                require(lookup != 0, "Bound IAT-only image requires separate adapter")
            name = string(name_rva)
            require(re.fullmatch(r"[A-Za-z0-9_.-]+\.dll", name, re.IGNORECASE) is not None,
                    "Simple import DLL basename required")
            result.append({"dll": name, "symbols": symbols(lookup)})
        raise ValueError("No null import directory terminator within declared bounds")

    normal, delay = imports(1, False), imports(13, True)
    dynamic = sorted({s["name"] for row in normal+delay for s in row["symbols"]
                      if "name" in s and s["name"] in {
                          "LoadLibraryA", "LoadLibraryW", "LoadLibraryExA", "LoadLibraryExW",
                          "GetProcAddress", "LdrLoadDll", "LdrGetProcedureAddress"}})
    return {"machine": machine, "optional_magic": magic, "normal": normal,
            "delay": delay, "dynamic_loader_symbols": dynamic,
            "loader_resolution_authenticated": False}


@contextmanager
def locked_bytes(paths):
    """Windows read handles deny WRITE/DELETE sharing while held. No execution.

    Not a guarantee against an adversarial loader or arbitrary directory/OS
    mutation, and not a lock on unenumerated transitive runtime dependencies.
    """
    require(os.name == "nt", "Reviewed Windows target scope only")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handles, payload = [], {}
    try:
        for path in paths:
            handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
            chunks, total = [], 0
            buffer, received = ctypes.create_string_buffer(1024*1024), wintypes.DWORD()
            while True:
                if not kernel.ReadFile(handle, buffer, len(buffer), ctypes.byref(received), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if received.value == 0:
                    break
                total += received.value
                require(total <= 256*1024*1024, "Bounded decoder file read")
                chunks.append(buffer.raw[:received.value])
            payload[str(path)] = b"".join(chunks)
        yield payload
    finally:
        for handle in reversed(handles):
            kernel.CloseHandle(handle)


def environment():
    return {"path": os.environ.get("PATH"), "pathext": os.environ.get("PATHEXT"),
            "comspec": os.environ.get("COMSPEC"), "systemroot": os.environ.get("SYSTEMROOT"),
            "cwd": str(Path.cwd().resolve()), "python": str(Path(sys.executable).resolve()),
            "windows_version": platform.version(), "pointer_bits": struct.calcsize("P")*8}


class DecoderTargetGuard:
    """Direct wrapper/EXE/cmd guard and fail-closed unresolved-runtime boundary."""
    def __init__(self):
        require(os.name == "nt" and struct.calcsize("P") == 8, "Actual Windows x64 scope")
        self._expected = copy.deepcopy(EXPECTED)
        self._environment = environment()
        command = Path(os.environ.get("COMSPEC", ""))
        system = Path(os.environ.get("SYSTEMROOT", "")) / "System32" / "cmd.exe"
        require(command.is_absolute() and command.resolve() == system.resolve() and system.is_file(),
                "Reviewed absolute System32 command processor")
        self._command = str(system.resolve())
        self._paths = [p for row in self._expected.values() for p in (row["wrapper"], row["target"])] + [self._command]
        self._in_hold, self.poisoned = False, False
        self._identity = digest([self._expected, self._environment, self._paths])
        with locked_bytes(self._paths) as payload:
            self._validate_payload(payload)
            images = {name: pe_imports(payload[row["target"]]) for name, row in self._expected.items()}
            images["command_processor"] = pe_imports(payload[self._command])
            require(all(image["machine"] == 0x8664 for image in images.values()), "Actual x64 images")
            self._files = {p: hashlib.sha256(payload[p]).hexdigest() for p in self._paths}
        self._contract = seal({"schema": 1, "purpose": PURPOSE,
            "bindings_sha256": copy.deepcopy(self._files), "environment": copy.deepcopy(self._environment),
            "decoder_targets": copy.deepcopy(self._expected), "pe_import_inventory": images,
            "direct_targets_prospectively_bound": True, "retroactive208_target_authentication": False,
            "unresolved_runtime": ["Windows API-set host mapping", "import/forwarder/SxS loader resolution",
                                   "dynamic LoadLibrary/GetProcAddress paths", "Python soundfile/scipy/numpy native runtime"],
            "complete_audio_runtime_authenticated": False, "actual_audio_draws": 0,
            "training_authorized": False, "release_selection": "NONE"})
        self._contract_digest = digest(self._contract)

    def _validate_payload(self, payload):
        require(environment() == self._environment, "Decoder lookup environment changed")
        for name, row in self._expected.items():
            selected = shutil.which(name)
            require(selected is not None and str(Path(selected).resolve()) == str(Path(row["wrapper"]).resolve()),
                    "PATH-selected wrapper changed")
            require(wrapper_target(payload[row["wrapper"]]) == row["target"], "Wrapper target changed")
            for kind in ("wrapper", "target"):
                require(hashlib.sha256(payload[row[kind]]).hexdigest() == row[kind+"_sha256"],
                        "Decoder " + kind + " bytes changed")

    def _guard_identity(self):
        try:
            require(not self.poisoned, "Poisoned target guard cannot continue")
            require(digest([self._expected, self._environment, self._paths]) == self._identity and
                    digest(self._contract) == self._contract_digest, "Target guard identity changed")
            check_seal(self._contract)
            require(typed(self._files) == typed(self._contract["bindings_sha256"]), "Target binding map changed")
        except BaseException:
            self.poisoned = True
            raise

    @contextmanager
    def hold_targets(self):
        """Pre/post byte checks with held direct-file handles; NOT audio authority.

        New tests place metadata-only synthetic operations inside this boundary.
        Future audio use still needs ALL unresolved runtime and input gates in
        a separately implemented integration. This is not a training rollback.
        """
        if self._in_hold:
            self.poisoned = True
            raise ValueError("Nested target hold forbidden")
        self._guard_identity()
        self._in_hold = True
        try:
            with locked_bytes(self._paths) as payload:
                self._validate_payload(payload)
                require(all(hashlib.sha256(payload[p]).hexdigest() == h for p, h in self._files.items()),
                        "Direct-file bytes changed before guarded scope")
                yield
                self._guard_identity()
                self._validate_payload(payload)
                # Reopen/read while original deny-WRITE/DELETE handles still held.
                with locked_bytes(self._paths) as current:
                    require(all(hashlib.sha256(current[p]).hexdigest() == h for p, h in self._files.items()),
                            "Direct-file bytes changed across guarded scope")
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self._in_hold = False

    def contract(self):
        self._guard_identity()
        return copy.deepcopy(self._contract)

    def require_complete_audio_runtime(self):
        self._guard_identity()
        raise ValueError("Complete runtime unresolved;209 never authorizes actual audio/training")
