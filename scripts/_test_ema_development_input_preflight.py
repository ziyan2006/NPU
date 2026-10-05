"""222 source/scope checks only. No input decoding/native handles/Model/CUDA."""
import ast
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("preflight222", Path(__file__).with_name("222_ema_development_input_preflight.py"))
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


class PreflightTests(unittest.TestCase):
    def test_01_all_native_identity_EXIT_pipe_checks_remain(self):
        source = ast.unparse(w.collector_ast())
        for text in ("a.validate_native(actual, known)", "held.check()", "native.measure(handle)",
                     "row['launcher_exit_code'] == 0", "process.wait(timeout=2)", "thread.join(timeout=2)",
                     "p.check_executable_identity(actual, a.preparation()['pre_bound_python_executables'])"):
            self.assertIn(text, source)

    def test_02_bounded_new_CPU_preparation_not_formal_training(self):
        source = ast.unparse(w.collector_ast())
        self.assertIn("time.monotonic() - start < 1800", source)
        self.assertIn("'formal_training_updates': 0", source)
        self.assertIn("'training_authorized': False", source)
        self.assertIn("validate_report(report)", source)

    def test_03_no_closed_CLI_or_prediction_is_called(self):
        tree = ast.parse(Path(w.__file__).read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
        calls = [ast.unparse(n.func) for n in ast.walk(function) if isinstance(n, ast.Call)]
        self.assertNotIn("torch.load", calls)
        self.assertFalse(any(n in calls for n in ("torch.optim.Adam", "e.SingleTrajectoryEngine", "evaluator.evaluate", "evaluator._score")))
        self.assertIn("d.OriginalDevelopmentEvaluator.prepare", calls)
        self.assertIn("c.source.zero_execution_guard", calls)

    def test_04_actual_WRITE_DELETE_share_failure_and_retained_fixture(self):
        tree = ast.parse(Path(w.__file__).read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "actual_lease_checks")
        source = ast.unparse(function)
        self.assertIn("(1073741824, 65536)", source)
        self.assertIn("code == 32", source)
        self.assertIn("not wrong.handles", source)
        self.assertNotIn("unlink", source)
        self.assertNotIn("WriteFile", source)

    def test_05_stdlib_supervisor_never_imports_Torch_at_top(self):
        tree = ast.parse(Path(w.__file__).read_text(encoding="utf-8"))
        imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("torch" in text or "numpy" in text for text in imports))

    def test_06_own_metadata_scope_rejects_any_formal_update(self):
        for count in (1, 500):
            with self.assertRaises(ValueError):
                w.validate_report({"purpose": w.PURPOSE, "device": "cpu", "formal_training_updates": count})


if __name__ == "__main__":
    unittest.main(verbosity=2)
