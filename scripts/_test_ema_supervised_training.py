"""221 launcher checks: stdlib only; NO worker, training, audio or CUDA."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("launcher221", Path(__file__).with_name("221_ema_supervised_training.py"))
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


def report():
    return {"purpose": w.PURPOSE, "training_authorized": True, "formal_training_updates": 500, "step": 5000,
            "development_steps": [4750, 5000], "single_training_Adam": True, "release_selection": "NONE",
            "exposure": {"forward_started": 3000, "forward_completed": 3000, "backward_completed": 3000,
                         "adam_started": 500, "adam_completed": 500}}


class LauncherTests(unittest.TestCase):
    def test_01_original_supervisor_file_native_checks_remain(self):
        tree = w.collector_ast(); source = ast.unparse(tree)
        for text in ("a.validate_native(actual, known)", "held.check()", "p.check_executable_identity(actual, a.preparation()['pre_bound_python_executables'])",
                     "native.checked('ContinueDebugEvent'", "native.measure(handle)", "process.wait(timeout=2)", "close_fds=True", "shell=False"):
            self.assertIn(text, source)

    def test_02_fixed500_scope_900s_to36000s_no_unbounded_loop(self):
        source = ast.unparse(w.collector_ast())
        self.assertIn("time.monotonic() - start < 36000", source)
        self.assertIn("events <= 250000", source)
        self.assertIn("created < 4", source)
        self.assertNotIn("< 900", source)

    def test_03_activation_and_native_guard_are_before_Popen(self):
        source = ast.unparse(w.collector_ast())
        self.assertIn("'--activation', str(ACTIVATION_PATH)", source)
        self.assertLess(source.index("supervisor_guard.json"), source.index("process = subprocess.Popen"))
        self.assertIn("'preheld_native_files': len(known)", source)

    def test_04_unknown_images_refuse_and_existing_failure_marker_preserved(self):
        source = ast.unparse(w.collector_ast())
        self.assertIn("if not (out / 'supervisor_refused.json').exists()", source)
        self.assertIn("native.checked('DebugActiveProcessStop', pid)", source)
        self.assertNotIn("TerminateProcess", source)
        self.assertNotIn("kill(", source)

    def test_05_actual_native_EXIT_and_full_pipe_consumption_not_replaced(self):
        tree = w.collector_ast()
        source = ast.unparse(tree)
        self.assertIn("row['launcher_exit_code'] == 0", source)
        expected = ast.parse("all(v['exit_code'] == 0 for v in row['processes'].values())", mode="eval").body
        self.assertTrue(any(ast.dump(node) == ast.dump(expected) for node in ast.walk(tree)))
        self.assertIn("all((not thread.is_alive() for thread in readers))", source)
        self.assertIn("len(images) == len(handles)", source)

    def test_06_complete500_report_is_typed_and_both_stages_required(self):
        self.assertTrue(w.valid_training_report(report()))
        for field, value in (("formal_training_updates", 499), ("step", 5001), ("training_authorized", False),
                             ("development_steps", [5000]), ("single_training_Adam", False), ("release_selection", "BEST")):
            changed = report(); changed[field] = value
            with self.assertRaises(ValueError):
                w.valid_training_report(changed)

    def test_07_partial_forward_Adam_and_boolean_counts_refused(self):
        for field, value in (("forward_started", 3001), ("forward_completed", 2999), ("backward_completed", 2999),
                             ("adam_started", 501), ("adam_completed", 499), ("adam_completed", True)):
            changed = report(); changed["exposure"][field] = value
            with self.assertRaises(ValueError):
                w.valid_training_report(changed)

    def test_08_missing_activation_refuses_without_opening_or_spawning_worker(self):
        with self.assertRaises(ValueError), patch.object(w.subprocess, "Popen", side_effect=AssertionError("worker forbidden")):
            w.authority(w.ROOT / "results/ema221_NONEXISTENT_activation.json", SimpleNamespace())

    def test_09_light_supervisor_does_not_import_torch_or_owner(self):
        tree = ast.parse(Path(w.__file__).read_text(encoding="utf-8"))
        top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("torch" in ast.unparse(n) or "numpy" in ast.unparse(n) for n in top_imports))
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        self.assertNotIn("ContinuousTrajectory", ast.unparse(train))

    def test_10_worker_has_one_training_Adam_and_no_old_CLI_or_resume(self):
        tree = ast.parse(Path(w.__file__).read_text(encoding="utf-8"))
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
        calls = [ast.unparse(n.func) for n in ast.walk(worker) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count("torch.optim.Adam"), 1)
        self.assertFalse(any(s.endswith(".train") or s.endswith(".evaluate_pair") for s in calls))
        self.assertIn("owner.observe_stage(saved)", ast.unparse(worker))
        self.assertNotIn("resume", ast.unparse(worker))

    def test_11_long_decoder_records_go_to_separate_journal_not_report_pipe(self):
        source = Path(w.__file__).read_text(encoding="utf-8")
        self.assertIn('"decoder_request_count": len(decoder.rows)', source)
        self.assertIn('"decoder_native_journal_sha256"', source)
        self.assertNotIn("decoder.receipts", source)
        self.assertIn('"decoder_native_journal.jsonl"', source)

    def test_12_supervisor_lifetime_is_OS_checked_and_disk_reserve_preserved(self):
        source = Path(w.__file__).read_text(encoding="utf-8")
        self.assertIn("kernel.GetExitCodeProcess(supervisor", source)
        self.assertIn("code.value == 259", source)
        self.assertIn("shutil.disk_usage(out).free >= 12*1024**3", source)
        self.assertIn("duplicate_preflight()", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
