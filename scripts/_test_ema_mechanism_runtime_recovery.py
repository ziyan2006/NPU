"""New218 exact recovery AST and post-import runtime-order tests only."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("unit218", Path(__file__).with_name("218_ema_mechanism_runtime_recovery.py"))
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse((k.ROOT / k.SOURCE).read_text(encoding="utf-8"))

    def test_pinned_executed_source(self):
        self.assertEqual(k.sha(k.ROOT / k.SOURCE), k.SOURCE_SHA)

    def test_exact_inserted_count(self):
        _, count = k.recovery_tree(self.tree)
        self.assertEqual(count, 8)

    def test_original_worker_preserved_after_removing_insert(self):
        recovered, _ = k.recovery_tree(self.tree)
        original = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
        body = next(n for n in recovered.body[0].body if isinstance(n, ast.Try)).body
        at = next(i for i, n in enumerate(body) if ast.unparse(n) == "loss = e.OriginalBoundaryLoss(device)") + 1
        del body[at:at+8]
        self.assertEqual(ast.dump(recovered.body[0], include_attributes=False), ast.dump(original, include_attributes=False))

    def test_config_after_loss_before_target_model(self):
        recovered, _ = k.recovery_tree(self.tree)
        body = next(n for n in recovered.body[0].body if isinstance(n, ast.Try)).body
        text = [ast.unparse(n) for n in body]
        at = text.index("loss = e.OriginalBoundaryLoss(device)")
        self.assertTrue(text[at+1].startswith("torch.set_num_threads"))
        self.assertTrue(text[at+3].startswith("if device == 'cuda':"))
        self.assertEqual(text[at+9], "target = loss.function.__globals__['m'].core.t09")

    def test_no_loss_construct_refused(self):
        for n in ast.walk(self.tree):
            if isinstance(n, ast.FunctionDef) and n.name == "worker":
                body = next(n for n in n.body if isinstance(n, ast.Try)).body
                body[:] = [v for v in body if ast.unparse(v) != "loss = e.OriginalBoundaryLoss(device)"]
        with self.assertRaisesRegex(ValueError, "one original loss"):
            k.recovery_tree(self.tree)

    def test_duplicate_loss_construct_refused(self):
        worker = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
        next(n for n in worker.body if isinstance(n, ast.Try)).body.append(ast.parse("loss = e.OriginalBoundaryLoss(device)").body[0])
        with self.assertRaises(ValueError):
            k.recovery_tree(self.tree)

    def test_no_worker_refused(self):
        with self.assertRaises(ValueError):
            k.recovery_tree(ast.parse("x=1"))

    def test_built_worker_entry_only_new(self):
        worker, collect, count = k.build()
        self.assertEqual(Path(collect.__globals__["__file__"]).name, "218_ema_mechanism_runtime_recovery.py")
        self.assertEqual(worker.__globals__["PURPOSE"], k.PURPOSE)
        self.assertEqual(collect.__globals__["PINS"][k.SOURCE], k.SOURCE_SHA)

    def test_existing_marker_preserved(self):
        _, collect, _ = k.build()
        self.assertIn("exists", collect.__code__.co_names)

    def test_no_heavy_imports(self):
        self.assertFalse(any(v in sys.modules for v in ("torch", "numpy", "scipy", "soundfile")))

    def runtime_insert(self, device, expected_threads, mismatch=False):
        tree, _ = k.recovery_tree(self.tree)
        body = next(n for n in tree.body[0].body if isinstance(n, ast.Try)).body
        index = next(i for i, n in enumerate(body) if ast.unparse(n) == "loss = e.OriginalBoundaryLoss(device)") + 1
        calls = []
        current = {"threads": 20, "device": device}
        def setter(threads):
            calls.append(threads); current["threads"] = threads
        state = {"torch": SimpleNamespace(set_num_threads=setter), "e": SimpleNamespace(runtime_identity=lambda d: dict(current)),
                 "c": SimpleNamespace(equal=lambda a, b: a == b), "device": device, "require": k.require,
                 "parent": {"parent_metadata": {"runtime": {"threads": expected_threads, "device": device, **({"bad": True} if mismatch else {})}}}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(body[index:index+3]), type_ignores=[])), "unit218-runtime", "exec"), state)
        return calls, current

    def test_cpu20_import_recovered_to2(self):
        calls, current = self.runtime_insert("cpu", 2)
        self.assertEqual(calls, [2]); self.assertEqual(current["threads"], 2)

    def test_cuda20_import_recovered_to4(self):
        calls, current = self.runtime_insert("cuda", 4)
        self.assertEqual(calls, [4]); self.assertEqual(current["threads"], 4)

    def test_cuda_full_runtime_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "POST-IMPORT"):
            self.runtime_insert("cuda", 4, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
