"""New211 pure/mock tests: never invoke a native decoder/debug child."""
import ctypes as c
import importlib.util
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema211_unit", ROOT / "scripts/211_ema_decoder_loaded_image_observation.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ObserverTests(unittest.TestCase):
    def test_01_abi(self):
        m.assert_abi()

    def test_02_only_two_fixed_commands(self):
        for label in ("ffmpeg", "ffprobe"):
            self.assertEqual(m.command(label), [m.meta.EXPECTED[label]["target"], "-version"])

    def test_03_no_arbitrary_executable(self):
        for label in ("cmd", "ffmpeg -i song", None, True, 1, ["ffmpeg"], "FFMPEG"):
            with self.assertRaises(ValueError): m.command(label)

    def test_04_no_command_alias(self):
        value = m.command("ffmpeg")
        value[0] = "wrong"
        self.assertNotEqual(value, m.command("ffmpeg"))

    def test_05_nonexception_continue(self):
        event = m.DebugEvent(code=6)
        self.assertEqual(m.continuation(event, False), (m.DBG_CONTINUE, False))

    def test_06_only_first_loader_breakpoint_handled(self):
        event = m.DebugEvent(code=1)
        event.data.exception.record.code = 0x80000003
        event.data.exception.first_chance = 1
        self.assertEqual(m.continuation(event, False), (m.DBG_CONTINUE, True))
        self.assertEqual(m.continuation(event, True), (m.DBG_NOT_HANDLED, True))

    def test_07_other_exceptions_not_swallowed(self):
        event = m.DebugEvent(code=1)
        event.data.exception.record.code = 0xC0000005
        event.data.exception.first_chance = 1
        self.assertEqual(m.continuation(event, False), (m.DBG_NOT_HANDLED, False))

    def test_08_second_chance_break_not_swallowed(self):
        event = m.DebugEvent(code=1)
        event.data.exception.record.code = 0x80000003
        event.data.exception.first_chance = 0
        self.assertEqual(m.continuation(event, False), (m.DBG_NOT_HANDLED, False))

    def test_09_missing_image_handle_rejected_before_native(self):
        native = object.__new__(m.Native)
        for handle in (None, 0, -1, c.c_void_p(-1).value):
            with self.assertRaises(ValueError): native.measure(handle)

    def test_10_bound_file_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata"
            path.write_bytes(b"fixture")
            bindings = {str(path): m.sha(path)}
            m.check_bindings(bindings)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError): m.check_bindings(bindings)

    def test_11_exclusive_evidence_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            m.write_new(path, {"value": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): m.write_new(path, {"value": 2})
            self.assertEqual(path.read_bytes(), before)

    def test_12_own_seal_rejects_edit(self):
        value = m.meta.seal({"version_only": True})
        m.meta.check_seal(value)
        value["version_only"] = 1
        with self.assertRaises(ValueError): m.meta.check_seal(value)

    def test_13_nested_symmetric_typed_comparison(self):
        row = m.meta.seal({"exit": 0})
        proof = m.meta.seal({"rows": [row]})
        m.meta.compare_sealed(proof, m.meta.seal({"rows": [dict(row)]}))
        changed = m.meta.seal({"rows": [{"exit": 0}]})
        with self.assertRaises(ValueError): m.meta.compare_sealed(proof, changed)

    def test_14_existing_run_refused_before_native(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, "OUT", Path(directory)), patch.object(m, "Native") as native:
            with self.assertRaises(ValueError): m.run()
            native.assert_not_called()

    def test_15_heavy_import_refused_before_native(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, "OUT", Path(directory) / "absent"), patch.dict(sys.modules, {"torch": object()}), patch.object(m, "Native") as native:
            with self.assertRaises(ValueError): m.run()
            native.assert_not_called()

    def test_16_failed_popen_retained_no_attach(self):
        class Abort(BaseException): pass
        with tempfile.TemporaryDirectory() as directory, patch.object(m.subprocess, "Popen", side_effect=Abort("injected-before-child")):
            events = []
            with self.assertRaises(Abort): m.observe("ffmpeg", Path(directory), object(), events.append)
            self.assertEqual(events[-1]["row"]["error"]["type"], "Abort")
            self.assertFalse(events[-1]["row"]["actual_child_created"])
            self.assertEqual(events[-1]["kind"], "row_final")

    def test_17_version_policy_hidden_no_shell_fixed_args(self):
        class Abort(BaseException): pass
        with tempfile.TemporaryDirectory() as directory, patch.object(m.subprocess, "Popen", side_effect=Abort) as popen:
            with self.assertRaises(Abort): m.observe("ffprobe", Path(directory), object(), lambda value: None)
            args, kwargs = popen.call_args
            self.assertEqual(args, ([m.meta.EXPECTED["ffprobe"]["target"], "-version"],))
            self.assertFalse(kwargs["shell"])
            self.assertTrue(kwargs["close_fds"])
            self.assertEqual(kwargs["creationflags"], m.subprocess.CREATE_NO_WINDOW | 2)
            self.assertEqual(kwargs["startupinfo"].wShowWindow, 0)
            self.assertIsNone(kwargs["env"])
            self.assertIsNone(kwargs["cwd"])

    def test_18_no_heavy_libraries(self):
        self.assertFalse(any(value in sys.modules for value in m.HEAVY))

    def fake_measure(self, size=3, short=False):
        native = object.__new__(m.Native)
        calls = []
        def checked(name, *args):
            calls.append(name)
            if name == "GetFinalPathNameByHandleW":
                args[1].value = r"\\?\C:\fixture\loaded.dll"
                return len(args[1].value)
            if name == "GetFileInformationByHandle":
                info = args[1]._obj
                info.size_low, info.volume, info.index_low = size, 123, 456
            if name == "ReadFile":
                args[1].raw = b"abc" + b"\0" * (len(args[1]) - 3)
                args[3]._obj.value = 0 if short else 3
            return 1
        native.checked = checked
        return native, calls

    def test_19_actual_handle_measure_metadata_and_bytes_mock(self):
        native, calls = self.fake_measure()
        value = native.measure(555)
        self.assertEqual(value["sha256"], m.hashlib.sha256(b"abc").hexdigest())
        self.assertEqual((value["volume_serial"], value["file_index"], value["bytes"]), (123, 456, 3))
        self.assertEqual(calls, ["GetFinalPathNameByHandleW", "GetFileInformationByHandle", "SetFilePointerEx", "ReadFile"])

    def test_20_empty_file_rejected_before_read(self):
        native, calls = self.fake_measure(size=0)
        with self.assertRaises(ValueError): native.measure(555)
        self.assertNotIn("ReadFile", calls)

    def test_21_overbudget_file_rejected_before_read(self):
        native, calls = self.fake_measure(size=m.MAX_FILE_BYTES + 1)
        with self.assertRaises(ValueError): native.measure(555)
        self.assertNotIn("ReadFile", calls)

    def test_22_short_file_read_rejected(self):
        native, calls = self.fake_measure(short=True)
        with self.assertRaises(ValueError): native.measure(555)
        self.assertIn("ReadFile", calls)

    def test_23_baseexception_detaches_only_new_own_pid_closes_event_handle(self):
        class Abort(BaseException): pass
        class Process:
            pid = 4242
        class NativeFake:
            def __init__(self): self.dll, self.calls = self, []
            def checked(self, name, *args):
                self.calls.append((name, args))
                return 1
            def WaitForDebugEventEx(self, pointer, timeout):
                event = pointer._obj
                event.code, event.pid, event.tid = 3, 4242, 43
                event.data.create.file, event.data.create.base = 555, 1234
                return 1
            def measure(self, handle): raise Abort("actual-handle-read-injected")
        native, events = NativeFake(), []
        with tempfile.TemporaryDirectory() as directory, patch.object(m.subprocess, "Popen", return_value=Process()):
            with self.assertRaises(Abort): m.observe("ffmpeg", Path(directory), native, events.append)
        self.assertEqual(native.calls, [("DebugSetProcessKillOnExit", (False,)),
                                       ("ContinueDebugEvent", (4242, 43, m.DBG_CONTINUE)),
                                       ("DebugActiveProcessStop", (4242,)), ("CloseHandle", (555,))])
        self.assertTrue(events[-1]["row"]["detached_on_failure"])
        self.assertEqual(events[-1]["row"]["events"][0]["code"], 3)

    def test_24_actual_log_byte_encoding_without_lossy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            text = "Ran 24 tests\r\n\r\nOK\r\n"
            for encoding in ("utf-8", "utf-8-sig", "utf-16"):
                path.write_bytes(text.encode(encoding))
                self.assertEqual(m.read_unit_log(path), text)
            path.write_bytes(b"\xffinvalid")
            with self.assertRaises(UnicodeError): m.read_unit_log(path)


if __name__ == "__main__":
    rng, environment = random.getstate(), dict(os.environ)
    with patch.object(m.subprocess, "Popen", side_effect=AssertionError("Native child forbidden in unit suite")):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObserverTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if random.getstate() != rng or dict(os.environ) != environment:
        raise RuntimeError("Unit suite changed RNG/environment")
    print("NEW211_UNIT_ONLY CHILDREN=0 AUDIO=0 MODEL_ADAM_CUDA=0 OLD_UNIT_RERUNS=0")
    sys.exit(0 if result.wasSuccessful() else 1)
