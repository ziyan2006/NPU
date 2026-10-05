"""New214 metadata/ABI preparation tests; no actual backend worker or audio."""
import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema214_preparation_test", ROOT / "scripts/214_ema_audio_backend_preparation.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PreparationScopeTests(unittest.TestCase):
    def test_light_import_no_heavy_library(self):
        self.assertFalse(any(v in sys.modules for v in ("torch", "numpy", "scipy", "soundfile")))

    def test_original_helpers_bytes_bound(self):
        p.check_files()
        self.assertEqual(len(p.PINS), 4)

    def test_windows_x64_debug_abi(self):
        p.obs.assert_abi()
        self.assertEqual(ctypes.sizeof(p.obs.DebugEvent), 176)

    def test_native_streaming_reader_own_file(self):
        with tempfile.TemporaryDirectory(prefix="ema214_native_") as directory:
            path = Path(directory) / "metadata.tmp"
            path.write_bytes(b"new214 isolated metadata")
            native = p.LargeNative()
            native.dll.CreateFileW.argtypes = [p.wintypes.LPCWSTR, p.wintypes.DWORD, p.wintypes.DWORD,
                p.wintypes.LPVOID, p.wintypes.DWORD, p.wintypes.DWORD, p.wintypes.HANDLE]
            native.dll.CreateFileW.restype = p.wintypes.HANDLE
            handle = native.dll.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
            try:
                measured = native.measure(handle)
                self.assertEqual(measured["sha256"], p.sha(path))
                self.assertEqual(measured["bytes"], path.stat().st_size)
            finally:
                native.checked("CloseHandle", handle)

    def test_missing_native_handle_rejected(self):
        native = p.LargeNative()
        for handle in (None, 0, -1, ctypes.c_void_p(-1).value):
            self.assertRaises(ValueError, native.measure, handle)

    def test_new_scope_never_old_closed_directory(self):
        with patch.object(subprocess, "Popen", side_effect=AssertionError("No child")):
            for path in (ROOT / "results", ROOT / "results/mel_ema_loaded_image_observation_20261004", ROOT / "scripts/unrelated"):
                self.assertRaises(ValueError, p.collect, path)

    def test_change_source_detected_before_child(self):
        original = p.PINS.copy()
        with patch.dict(p.PINS, {"scripts/208_ema_authenticated_audio_input.py": "0" * 64}):
            with patch.object(subprocess, "Popen", side_effect=AssertionError("No child")):
                self.assertRaises(ValueError, p.check_files)
        self.assertEqual(p.PINS, original)

    def test_initial_loader_break_only_swallowed_once(self):
        event = p.obs.DebugEvent()
        event.code = 1; event.data.exception.record.code = 0x80000003
        event.data.exception.first_chance = 1
        self.assertEqual(p.obs.continuation(event, False), (p.obs.DBG_CONTINUE, True))
        self.assertEqual(p.obs.continuation(event, True), (p.obs.DBG_NOT_HANDLED, True))

    def test_other_first_chance_exception_not_swallowed(self):
        event = p.obs.DebugEvent()
        event.code = 1; event.data.exception.record.code = 0xC0000005
        event.data.exception.first_chance = 1
        self.assertEqual(p.obs.continuation(event, False), (p.obs.DBG_NOT_HANDLED, False))

    def test_same_physical_python_different_hardlink_spelling(self):
        original = {"final_path": r"\\?\C:\pinned\python.exe", "bytes": 100, "sha256": "a" * 64,
                    "volume_serial": 3, "file_index": 5, "last_write_ticks": 8}
        actual = original | {"final_path": r"\\?\C:\another_hardlink\python.exe"}
        p.check_executable_identity(actual, [original])

    def test_any_executable_physical_identity_difference_rejected(self):
        original = {"final_path": r"\\?\C:\pinned\python.exe", "bytes": 100, "sha256": "a" * 64,
                    "volume_serial": 3, "file_index": 5, "last_write_ticks": 8}
        for name in ("bytes", "sha256", "volume_serial", "file_index", "last_write_ticks"):
            actual = original | {name: "b" * 64 if name == "sha256" else original[name] + 1}
            self.assertRaises(ValueError, p.check_executable_identity, actual, [original])

    def test_executable_bool_int_type_difference_rejected(self):
        original = {"final_path": r"\\?\C:\pinned\python.exe", "bytes": 100, "sha256": "a" * 64,
                    "volume_serial": 3, "file_index": 1, "last_write_ticks": 8}
        self.assertRaises(ValueError, p.check_executable_identity, original | {"file_index": True}, [original])

    def test_exact_retained_console_discovery_not_arbitrary_system_basename(self):
        observed = p.observed_console_host()
        self.assertEqual(observed["bytes"], 1003520)
        self.assertEqual(observed["file_index"], 3377699722507974)
        self.assertEqual(observed["sha256"], "e449bce01f275cd08f3d4e64bb73b3b43ae845a0dbdb3e6131426e66537705e5")

    def test_console_discovery_parent_sha_tampering_rejected(self):
        with patch.object(p, "CONSOLE_OBSERVATION_SHA", "0" * 64):
            self.assertRaises(ValueError, p.observed_console_host)

    def test_real_float_config_json_roundtrip_own214_not209(self):
        fields = {"learning_rate": .0001, "minimum_rank_gain": .02, "gain_range": [-3.0, 3.0], "instrumental": 4}
        doc = p.seal_metadata(fields)
        p.check_metadata(json.loads(json.dumps(doc)))
        self.assertRaises(ValueError, p.obs.meta.check_seal, doc)

    def test_float_signed_zero_and_int_bool_not_conflated(self):
        self.assertNotEqual(p.seal_metadata({"value": -0.0}), p.seal_metadata({"value": 0.0}))
        self.assertNotEqual(p.seal_metadata({"value": 1.0}), p.seal_metadata({"value": 1}))
        self.assertNotEqual(p.seal_metadata({"value": True}), p.seal_metadata({"value": 1}))

    def test_new214_seal_failure_retained_no_side_seal_stripping(self):
        doc = p.seal_metadata({"config": {"rate": .0001}})
        doc["config"]["rate"] = .0002
        self.assertRaises(ValueError, p.check_metadata, doc)
        for value in (float("inf"), float("nan")):
            self.assertRaises(ValueError, p.seal_metadata, {"value": value})


if __name__ == "__main__":
    unittest.main(verbosity=2)
