"""New226 units; temporary metadata files only, no decoder/music/ML execution."""
import ast
import copy
import ctypes
import importlib.util
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("ema226_unit", Path(__file__).with_name("226_ema_decoder_hotpath.py"))
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class HotpathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = random.getstate()
        cls.ban = patch.object(subprocess, "Popen", side_effect=AssertionError("No decoder in226 units"))
        cls.ban.start()

    @classmethod
    def tearDownClass(cls):
        cls.ban.stop()
        assert cls.rng == random.getstate()
        assert not any(x in sys.modules for x in ("torch", "numpy", "scipy", "soundfile"))
        print("NEW226_UNITS decoder/audio/PT/model/Adam/CUDA=0; PythonRNG unchanged")

    def sample(self):
        actual = dict(zip(f.FIELDS, [r"\\?\D:\metadata.bin", 8, 7, 11, 13]))
        expected = actual | {"sha256": "a"*64}
        return actual, expected

    def test_exact_identity_and_noalias(self):
        a, e = self.sample()
        result = f.checked_identity(a, e)
        self.assertEqual(result["sha256"], e["sha256"])
        result["file_index"] = 9
        self.assertEqual(e["file_index"], 11)

    def test_every_physical_field_changed_rejected(self):
        a, e = self.sample()
        for key in f.FIELDS:
            with self.subTest(key=key):
                bad = dict(a); bad[key] = (bad[key]+"other") if isinstance(bad[key], str) else bad[key]+1
                self.assertRaises(ValueError, f.checked_identity, bad, e)

    def test_bool_integer_rejected(self):
        a, e = self.sample()
        for key in f.FIELDS[1:]:
            with self.subTest(key=key):
                bad = dict(a); bad[key] = True
                self.assertRaises(ValueError, f.checked_identity, bad, e)

    def test_missing_extra_and_reordered_metadata_rejected(self):
        a, e = self.sample()
        self.assertRaises(ValueError, f.checked_identity, {k:v for k,v in a.items() if k!='bytes'}, e)
        self.assertRaises(ValueError, f.checked_identity, a | {"unknown": 0}, e)
        self.assertRaises(ValueError, f.checked_identity, dict(reversed(list(a.items()))), e)

    def test_unbound_sha_rejected(self):
        a, e = self.sample()
        for bad in (None, 5, "f"*63, "Z"*64):
            self.assertRaises(ValueError, f.checked_identity, a, e | {"sha256": bad})

    def test_zero_size_or_file_index_rejected(self):
        a, e = self.sample()
        for key in ("bytes", "file_index"):
            self.assertRaises(ValueError, f.checked_identity, a | {key: 0}, e | {key: 0})

    def test_exact_original_call_AST_unchanged(self):
        call, digest = f.unchanged_call(f.g)
        tree = ast.parse(f.BASE.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DirectDecoder")
        original = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "check_output")
        self.assertEqual(digest, f.hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest())
        self.assertIs(call.__globals__["HeldFiles"], f.CallProof)
        self.assertIs(f.g.DirectDecoder.check_output.__globals__["HeldFiles"], f.g.HeldFiles)

    def test_all_six_command_shapes_and_graph_unchanged(self):
        for row in f.g.request.original_requests(r"D:\nonexistent\new & % !.wav"):
            original = f.g.checked_request(row["argv"], row["kwargs"])
            self.assertEqual(original["argv"][1:], row["argv"][1:])
            self.assertEqual(f.g.meta.typed(original["kwargs"]), f.g.meta.typed(row["kwargs"]))

    def test_adapter_is_scoped_original_type_not_global_patch(self):
        old = f.g.DirectDecoder.check_output
        adapter = f.make_decoder()
        self.assertIs(type(adapter), f.g.DirectDecoder)
        self.assertIs(f.g.DirectDecoder.check_output, old)
        self.assertIsInstance(adapter.native, f.ScopedNative)

    def test_measure_without_custody_rejected(self):
        native = f.ScopedNative()
        self.assertRaises(ValueError, native.measure, 999)

    def temporary(self, directory):
        path = Path(directory) / "temporary_metadata.bin"
        path.write_bytes(b"new226 synthetic metadata only")
        native = f.ScopedNative()
        h = native.dll.CreateFileW  # argtypes installed by CallProof below
        proof = f.CallProof({}, native)
        handle = h(str(path), 0x80000000, 1, None, 3, 0x80, None)
        try:
            info = native.full_measure(handle)
        finally:
            native.checked("CloseHandle", handle)
        native.full_reads = native.full_bytes = native.identity_reads = 0
        return path, native, f.CallProof({f.g.canonical_path(info["final_path"]): info}, native)

    def test_native_two_full_reads_and_close_invalidates(self):
        with tempfile.TemporaryDirectory(prefix="ema226_unit_") as d:
            path, n, p = self.temporary(d)
            with p:
                p.event_identity(p.handles[0]); p.check(); p.check()
                self.assertEqual(n.full_reads, 2)
                self.assertRaises(PermissionError, path.write_bytes, b"must not write")
                self.assertRaises(PermissionError, path.unlink)
            self.assertEqual(n.full_reads, 2)
            self.assertIsNone(n.custody)
            self.assertEqual(p.handles, [])
            self.assertRaises(ValueError, p.__enter__)
            self.assertRaises(ValueError, n.measure, 999)
            self.assertEqual(path.read_bytes(), b"new226 synthetic metadata only")

    def test_native_nested_custody_rejected(self):
        with tempfile.TemporaryDirectory(prefix="ema226_unit_") as d:
            _, n, p = self.temporary(d)
            with p:
                self.assertRaises(ValueError, f.CallProof(p.files, n).__enter__)

    def test_native_mutated_registry_poisoned_and_closed(self):
        with tempfile.TemporaryDirectory(prefix="ema226_unit_") as d:
            _, n, p = self.temporary(d)
            with self.assertRaises(ValueError), p:
                next(iter(p.files.values()))["sha256"] = "0"*64
                n.measure(p.handles[0])
            self.assertTrue(p.failed)
            self.assertIsNone(n.custody)
            self.assertEqual(p.handles, [])

    def test_native_post_hash_failure_not_accepted(self):
        with tempfile.TemporaryDirectory(prefix="ema226_unit_") as d:
            _, n, p = self.temporary(d)
            with self.assertRaises(ValueError), p:
                old = n.full_measure
                with patch.object(n, "full_measure", side_effect=lambda h: old(h) | {"sha256": "0"*64}):
                    p.check()
            self.assertTrue(p.failed)
            self.assertIsNone(n.custody)

    def test_native_base_exception_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="ema226_unit_") as d:
            _, n, p = self.temporary(d)
            with self.assertRaises(KeyboardInterrupt), p:
                raise KeyboardInterrupt("new fixture only")
            self.assertTrue(p.failed)
            self.assertIsNone(n.custody)
            self.assertEqual(p.handles, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
