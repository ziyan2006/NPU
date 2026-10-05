"""New static integration tests: no original worker execution/heavy imports."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema230_new_units", ROOT / "scripts/230_ema_fast_worker_integration.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = {path: f.sha(path) for path in f.PINS}
        cls.rng = random.getstate()
        cls.child = patch.object(subprocess, "Popen", side_effect=AssertionError("No process in static integration"))
        cls.child.start()

    @classmethod
    def tearDownClass(cls):
        cls.child.stop()
        assert cls.before == {path: f.sha(path) for path in f.PINS}
        assert cls.rng == random.getstate()
        assert not any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile"))
        print("NEW230_STATIC_UNITS child/audio/PT/model/Adam/CUDA=0; original bytes/Python RNG unchanged")

    def test_exact_two_hotpath_changes_and_full_reverse(self):
        row = f.audit()
        self.assertTrue(row["exactly_two_changes"])
        self.assertTrue(row["original_worker_reconstructed_exactly"])
        self.assertEqual(len(row["replacement_expressions"]), 2)

    def test_candidate_compiles_without_execution(self):
        code = compile(f.worker_ast(), "new230_test_only", "exec")
        self.assertIsNotNone(code)

    def test_correct_private_hot_bindings(self):
        tree = f.worker_ast()
        values = {node.targets[0].id: ast.unparse(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in ("decoder", "_cache_identity")}
        self.assertEqual(values, {"decoder": "hot.make_decoder(q.g)", "_cache_identity": "hot.fast_cache_identity(a)"})

    def test_changed_source_refused_before_compilation(self):
        with patch.object(f, "sha", return_value="0" * 64):
            self.assertRaises(ValueError, f.worker_ast)

    def test_one_missing_hotpath_assignment_refused(self):
        text = f.WORKER.read_text(encoding="utf-8").replace(
            "_cache_identity = a.AuthenticatedAudioStream._cache_identity", "_cache_identity = None")
        with patch.object(Path, "read_text", return_value=text):
            self.assertRaises(ValueError, f.worker_ast)

    def test_deterministic_ast_digest(self):
        first, second = f.audit(), f.audit()
        self.assertEqual(first, second)
        self.assertEqual(first["candidate_worker_AST_sha256"],
                         hashlib.sha256(f.dump(f.worker_ast()).encode()).hexdigest())

    def test_no_training_or_replay_authority_claim(self):
        row = f.audit()
        self.assertFalse(row["training_authorized"])
        self.assertFalse(row["completed_fixed500_replay_authorized"])
        self.assertEqual(row["actual_CPU_CUDA_mechanism"], "PENDING")
        self.assertEqual(row["end_to_end_training_speedup"], "UNMEASURED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
