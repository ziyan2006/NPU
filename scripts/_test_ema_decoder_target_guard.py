"""NEW209 metadata tests. No decoder, old input replay, model or training.

One actual read-only constructor and one empty held-target scope are recorded.
All mutations and sharing-denial probes use tiny synthetic metadata/files.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema209", ROOT / "scripts/209_ema_decoder_target_guard.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
ACTUAL_EVIDENCE = None


def image(bits=64, delay=False):
    data = bytearray(4096)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3c, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HH", data, 0x84, 0x8664 if bits == 64 else 0x14c, 1)
    optional_size = 240 if bits == 64 else 224
    struct.pack_into("<H", data, 0x94, optional_size)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x20b if bits == 64 else 0x10b)
    struct.pack_into("<I", data, optional+60, 0x400)
    directory = optional+(112 if bits == 64 else 96)
    struct.pack_into("<I", data, directory-4, 16)
    struct.pack_into("<II", data, directory+8, 0x1000, 40)
    section = optional+optional_size
    data[section:section+8] = b".idata\0\0"
    struct.pack_into("<IIII", data, section+8, 0xc00, 0x1000, 0xc00, 0x400)
    struct.pack_into("<5I", data, 0x400, 0x1050, 0, 0, 0x1080, 0x1050)
    fmt, width = ("<Q", 8) if bits == 64 else ("<I", 4)
    struct.pack_into(fmt, data, 0x450, 0x10a0)
    struct.pack_into(fmt, data, 0x450+width, (1 << (bits-1)) | 7)
    data[0x480:0x480+13] = b"KERNEL32.dll\0"
    data[0x4a0:0x4a0+15] = b"\0\0LoadLibraryW\0"
    if delay:
        struct.pack_into("<II", data, directory+13*8, 0x1200, 64)
        struct.pack_into("<8I", data, 0x600, 1, 0x1260, 0, 0x1280, 0x1280, 0, 0, 0)
        data[0x660:0x660+11] = b"USER32.dll\0"
        struct.pack_into(fmt, data, 0x680, 0x12a0)
        data[0x6a0:0x6a0+17] = b"\0\0GetProcAddress\0"
    return bytes(data)


def edited(data, fmt, offset, *values):
    result = bytearray(data)
    struct.pack_into(fmt, result, offset, *values)
    return bytes(result)


def fixture_guard():
    guard = g.DecoderTargetGuard.__new__(g.DecoderTargetGuard)
    payload = {}
    expected = {}
    for name in ("ffmpeg", "ffprobe"):
        wrapper, target = "C:\\fixture\\"+name+".cmd", "C:\\fixture\\"+name+".exe"
        payload[wrapper] = ('@echo off\r\n"'+target+'" %*\r\n').encode("ascii")
        payload[target] = image()
        expected[name] = {"wrapper": wrapper, "wrapper_sha256": hashlib.sha256(payload[wrapper]).hexdigest(),
                          "target": target, "target_sha256": hashlib.sha256(payload[target]).hexdigest()}
    command = "C:\\fixture\\cmd.exe"
    payload[command] = image()
    guard._expected, guard._environment = expected, g.environment()
    guard._command = command
    guard._paths = [p for row in expected.values() for p in (row["wrapper"], row["target"])] + [command]
    guard._identity = g.digest([expected, guard._environment, guard._paths])
    guard._files = {p: hashlib.sha256(payload[p]).hexdigest() for p in guard._paths}
    guard._contract = g.seal({"bindings_sha256": copy.deepcopy(guard._files),
                             "complete_audio_runtime_authenticated": False, "training_authorized": False})
    guard._contract_digest = g.digest(guard._contract)
    guard._in_hold, guard.poisoned = False, False
    return guard, payload


@contextmanager
def fixture_scope(guard, payload, change_reopen=None):
    calls = []

    @contextmanager
    def reader(paths):
        calls.append(list(paths))
        current = copy.deepcopy(payload)
        if len(calls) == 2 and change_reopen:
            change_reopen(current)
        yield current

    with mock.patch.object(g, "locked_bytes", reader), mock.patch.object(
            g.shutil, "which", lambda name: guard._expected[name]["wrapper"]):
        yield calls


class Metadata(unittest.TestCase):
    def test_wrapper_actual_form(self):
        self.assertEqual(g.wrapper_target(b'@echo off\r\n"C:\\ffmpeg\\bin\\ffmpeg.exe" %*\r\n'),
                         r"C:\ffmpeg\bin\ffmpeg.exe")

    def test_wrapper_rejects_shell_variants(self):
        for line in ('C:\\f.exe %*', '"%ROOT%\\f.exe" %*', '"C:\\f.exe" %* & echo x',
                     '"C:\\f!.exe" %*', '"C:\\f^.exe" %*', '"C:\\f(x).exe" %*',
                     '"\\\\host\\f.exe" %*', '"C:\\f.exe" %1', '"C:\\f.exe" %*\necho x'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                g.wrapper_target(('@echo off\n'+line).encode())

    def test_wrapper_rejects_size_encoding_header(self):
        for data in (b'x'*1024, b'\xff', b'@ECHO OFF\n"C:\\f.exe" %*', bytearray(b"x")):
            with self.assertRaises(ValueError):
                g.wrapper_target(data)

    def test_symmetric_own_seals(self):
        a = g.seal({"a": 1, "b": False})
        g.compare_sealed(a, copy.deepcopy(a))
        self.assertNotIn("content_sha256", {"a": 1})
        with self.assertRaises(ValueError):
            g.seal(a)

    def test_typed_bool_int_and_order(self):
        for a, b in (({"x": 1}, {"x": True}), ({"x": [1]}, {"x": (1,)}),
                     ({"x": 1, "y": 2}, {"y": 2, "x": 1})):
            with self.assertRaises(ValueError):
                g.compare_sealed(g.seal(a), g.seal(b))

    def test_typed_rejects_float_custom(self):
        for value in (1.0, float("nan"), {"x": 1.0}, set(), bytearray()):
            with self.assertRaises(ValueError):
                g.digest(value)

    def test_own_seal_mutation_missing(self):
        doc = g.seal({"x": 1})
        doc["x"] = 2
        with self.assertRaises(ValueError):
            g.compare_sealed(doc, g.seal({"x": 2}))
        with self.assertRaises(ValueError):
            g.compare_sealed({"x": 2}, g.seal({"x": 2}))

    def test_json_disk_roundtrip_exclusive(self):
        doc = g.seal({"i": 1, "flag": False, "n": None, "rows": [{"s": "ascii"}]})
        with tempfile.TemporaryDirectory(prefix="ema209_metadata_") as folder:
            path = Path(folder) / "metadata.json"
            with path.open("x", encoding="utf8") as stream:
                json.dump(doc, stream, ensure_ascii=True)
            with self.assertRaises(FileExistsError), path.open("x"):
                pass
            g.compare_sealed(doc, json.loads(path.read_text(encoding="utf8")))


class PE(unittest.TestCase):
    def test_pe32_plus_named_ordinal_rva(self):
        result = g.pe_imports(image())
        self.assertEqual(result["normal"], [{"dll": "KERNEL32.dll", "symbols": [{"name": "LoadLibraryW"}, {"ordinal": 7}]}])
        self.assertEqual(result["machine"], 0x8664)
        self.assertFalse(result["loader_resolution_authenticated"])

    def test_pe32_word_width(self):
        result = g.pe_imports(image(32))
        self.assertEqual(result["machine"], 0x14c)
        self.assertEqual(result["normal"][0]["symbols"][1], {"ordinal": 7})

    def test_delay_import_rva_and_dynamic_union(self):
        result = g.pe_imports(image(delay=True))
        self.assertEqual(result["delay"], [{"dll": "USER32.dll", "symbols": [{"name": "GetProcAddress"}]}])
        self.assertEqual(result["dynamic_loader_symbols"], ["GetProcAddress", "LoadLibraryW"])

    def test_empty_directories(self):
        result = g.pe_imports(edited(image(), "<II", 0x98+112+8, 0, 0))
        self.assertEqual(result["normal"], [])
        self.assertEqual(result["delay"], [])

    def test_no_dos_or_pe_header(self):
        for data in (b"", b"x"*4096, edited(image(), "<I", 0x3c, 4096), image()[:120]):
            with self.assertRaises(ValueError):
                g.pe_imports(data)

    def test_optional_header_complete(self):
        for data in (edited(image(), "<H", 0x94, 8), edited(image(), "<H", 0x98, 0x999),
                     edited(image(), "<I", 0x98+108, 17)):
            with self.assertRaises(ValueError):
                g.pe_imports(data)

    def test_full_section_table_bound(self):
        with self.assertRaises(ValueError):
            g.pe_imports(edited(image(), "<H", 0x86, 96))

    def test_raw_file_bound_and_missing_rva(self):
        section = 0x98+240
        for data in (edited(image(), "<I", section+16, 0x10000),
                     edited(image(), "<I", 0x400, 0x3000)):
            with self.assertRaises(ValueError):
                g.pe_imports(data)

    def test_overlapping_sections_rejected(self):
        data = bytearray(edited(image(), "<H", 0x86, 2))
        section = 0x98+240
        data[section+40:section+80] = data[section:section+40]
        with self.assertRaises(ValueError):
            g.pe_imports(bytes(data))

    def test_directory_size_pair_and_terminator(self):
        for data in (edited(image(), "<II", 0x98+112+8, 0x1000, 0),
                     edited(image(), "<II", 0x98+112+8, 0x1000, 20)):
            with self.assertRaises(ValueError):
                g.pe_imports(data)

    def test_delay_va_unknown_attrs_rejected(self):
        for attr in (0, 2, 3):
            with self.assertRaises(ValueError):
                g.pe_imports(edited(image(delay=True), "<I", 0x600, attr))

    def test_iat_only_rejected(self):
        with self.assertRaises(ValueError):
            g.pe_imports(edited(image(), "<I", 0x400, 0))

    def test_bad_dll_name(self):
        data = bytearray(image())
        data[0x480:0x480+13] = b"C:\\bad.dll\0\0\0"
        with self.assertRaises(ValueError):
            g.pe_imports(bytes(data))

    def test_ordinal_reserved_bits(self):
        with self.assertRaises(ValueError):
            g.pe_imports(edited(image(), "<Q", 0x458, (1 << 63) | (1 << 32) | 7))

    def test_unterminated_name(self):
        data = bytearray(image())
        data[0x480:0x480+513] = b"a"*513
        with self.assertRaises(ValueError):
            g.pe_imports(bytes(data))


class Guard(unittest.TestCase):
    def test_synthetic_pre_post_locked_reopen(self):
        guard, payload = fixture_guard()
        with fixture_scope(guard, payload) as calls, guard.hold_targets():
            self.assertTrue(guard._in_hold)
        self.assertEqual(calls, [guard._paths, guard._paths])
        self.assertFalse(guard.poisoned)

    def test_pre_bytes_change_poison(self):
        guard, payload = fixture_guard()
        payload[guard._expected["ffmpeg"]["target"]] += b"changed"
        with fixture_scope(guard, payload), self.assertRaises(ValueError), guard.hold_targets():
            self.fail("must reject before entry")
        self.assertTrue(guard.poisoned)
        with self.assertRaises(ValueError):
            guard.contract()

    def test_post_bytes_change_poison(self):
        guard, payload = fixture_guard()
        def change(current):
            current[guard._command] += b"changed"
        with fixture_scope(guard, payload, change), self.assertRaises(ValueError), guard.hold_targets():
            pass
        self.assertTrue(guard.poisoned)

    def test_baseexception_poison(self):
        guard, payload = fixture_guard()
        with fixture_scope(guard, payload), self.assertRaises(KeyboardInterrupt), guard.hold_targets():
            raise KeyboardInterrupt("synthetic metadata scope")
        self.assertTrue(guard.poisoned)
        self.assertFalse(guard._in_hold)

    def test_path_environment_change_before_and_after(self):
        for when in ("before", "after"):
            guard, payload = fixture_guard()
            with fixture_scope(guard, payload), mock.patch.dict(os.environ, {}, clear=False):
                with self.assertRaises(ValueError):
                    if when == "before":
                        os.environ["PATH"] += ";synthetic_changed"
                    with guard.hold_targets():
                        if when == "after":
                            os.environ["PATH"] += ";synthetic_changed"
            self.assertTrue(guard.poisoned)

    def test_selected_wrapper_change(self):
        guard, payload = fixture_guard()
        with fixture_scope(guard, payload), mock.patch.object(g.shutil, "which", return_value=None), \
                self.assertRaises(ValueError), guard.hold_targets():
            self.fail("unresolved path must not enter")
        self.assertTrue(guard.poisoned)

    def test_internal_identity_mutation_poison(self):
        for key in ("_paths", "_expected", "_files", "_contract"):
            guard, _ = fixture_guard()
            value = getattr(guard, key)
            if type(value) is list:
                value.append("changed")
            else:
                value["changed"] = True
            with self.assertRaises(ValueError):
                guard.contract()
            self.assertTrue(guard.poisoned)

    def test_nested_hold_poison_even_when_caught(self):
        guard, payload = fixture_guard()
        with fixture_scope(guard, payload), self.assertRaises(ValueError), guard.hold_targets():
            with self.assertRaises(ValueError), guard.hold_targets():
                self.fail("nested must reject")
        self.assertTrue(guard.poisoned)

    def test_contract_noalias_and_unresolved_authority(self):
        guard, _ = fixture_guard()
        doc = guard.contract()
        doc["bindings_sha256"]["fake"] = "mutated"
        self.assertNotIn("fake", guard.contract()["bindings_sha256"])
        with self.assertRaisesRegex(ValueError, "never authorizes"):
            guard.require_complete_audio_runtime()
        self.assertFalse(guard.poisoned)

    def test_windows_temp_read_handle_denies_write_delete(self):
        with tempfile.TemporaryDirectory(prefix="ema209_handle_") as folder:
            path = Path(folder) / "tiny.bin"
            with path.open("xb") as stream:
                stream.write(b"synthetic metadata")
            with g.locked_bytes([str(path)]) as payload:
                self.assertEqual(payload[str(path)], b"synthetic metadata")
                for action in (lambda: path.open("r+b"), lambda: path.unlink()):
                    with self.assertRaises(OSError):
                        action()
                self.assertEqual(path.read_bytes(), b"synthetic metadata")
            self.assertEqual(path.read_bytes(), b"synthetic metadata")

    def test_windows_partial_open_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="ema209_cleanup_") as folder:
            path = Path(folder) / "tiny.bin"
            with path.open("xb") as stream:
                stream.write(b"synthetic")
            with self.assertRaises(OSError), g.locked_bytes([str(path), str(Path(folder) / "absent.bin")]):
                self.fail("second open must fail")
            with path.open("r+b") as stream:
                self.assertEqual(stream.read(), b"synthetic")


class ZActual(unittest.TestCase):
    def test_one_actual_read_only_guard_not_audio(self):
        global ACTUAL_EVIDENCE
        self.assertIsNone(ACTUAL_EVIDENCE)
        rng = random.getstate()
        contract_started = datetime.now(timezone.utc).isoformat()
        guard = g.DecoderTargetGuard()
        original = guard.contract()
        held_started = datetime.now(timezone.utc).isoformat()
        with guard.hold_targets():
            # Intentionally EMPTY: no decoder, real input or student operation.
            pass
        held_completed = datetime.now(timezone.utc).isoformat()
        g.compare_sealed(original, guard.contract())
        self.assertEqual(random.getstate(), rng)
        self.assertEqual(len(original["bindings_sha256"]), 5)
        self.assertTrue(original["direct_targets_prospectively_bound"])
        self.assertFalse(original["complete_audio_runtime_authenticated"])
        self.assertFalse(original["retroactive208_target_authentication"])
        self.assertFalse(original["training_authorized"])
        self.assertEqual(original["actual_audio_draws"], 0)
        for row in original["pe_import_inventory"].values():
            self.assertEqual(row["machine"], 0x8664)
            self.assertFalse(row["loader_resolution_authenticated"])
        with self.assertRaises(ValueError):
            guard.require_complete_audio_runtime()
        ACTUAL_EVIDENCE = g.seal({"schema": 1, "purpose": g.PURPOSE,
            "contract_started_utc": contract_started, "held_started_utc": held_started,
            "held_completed_utc": held_completed, "contract": original,
            "constructor_actual_read_only": 1, "actual_empty_held_scopes": 1,
            "actual_audio_draws": 0, "decoder_launches": 0, "student_pt_loads": 0,
            "model_forwards": 0, "adam_construction_or_steps": 0, "student_updates": 0,
            "cuda_initialization": False, "full_audio_runtime_gate": "PENDING",
            "complete_audio_runtime_authority_rejected": True,
            "python_rng_unchanged": True, "release_selection": "NONE"})


def main():
    if len(sys.argv) != 2:
        raise ValueError("One fresh evidence path required")
    output = Path(sys.argv.pop())
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Fresh evidence file in existing results only")
    rng = random.getstate()
    forbidden = []
    def deny(*args, **kwargs):
        forbidden.append("subprocess blocked before execution")
        raise AssertionError("209 cannot execute a decoder or any child process")
    with ExitStack() as stack:
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            stack.enter_context(mock.patch.object(subprocess, name, deny))
        stack.enter_context(mock.patch.object(os, "system", deny))
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if random.getstate() != rng:
        random.setstate(rng)
        raise AssertionError("Outer Python RNG changed")
    if not result.wasSuccessful():
        sys.exit(1)
    if forbidden or ACTUAL_EVIDENCE is None:
        raise AssertionError("No attempted external execution allowed in209")
    if any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile")):
        raise AssertionError("209 target-only suite must not load numerical/audio libraries")
    g.check_seal(ACTUAL_EVIDENCE)
    g.check_seal(ACTUAL_EVIDENCE["contract"])
    with output.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(ACTUAL_EVIDENCE, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"tests_run": result.testsRun, "evidence": str(output.resolve()),
                      "audio_draws": 0, "decoder_launches": 0,
                      "numerical_audio_libraries_loaded": False,
                      "full_audio_runtime_gate": "PENDING"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
