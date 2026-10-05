"""New215 policy/AST tests. No real model/audio worker or decoder launch."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema215_audit_tests", ROOT / "scripts/215_ema_model_audio_audit.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def tiny_tree(expression="scaled.backward()"):
    return ast.parse("def validate_metadata(metadata):\n    return metadata\n"
                     "def boundary_backward(scaled):\n    value = 3\n    " + expression + "\n    return value\n")


class AuditScopeTests(unittest.TestCase):
    def test_light_import(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile")))

    def test_fixed_dependency_bytes(self):
        a.check_files()
        self.assertEqual(len(a.PINS), 4)

    def test_preparation_complete_nested_own_seals(self):
        doc = a.preparation()
        self.assertEqual(len(doc["native_physical_files"]), 218)
        self.assertFalse(doc["prepared_backend"]["training_authorized"])

    def test_preparation_tampered_outer_seal(self):
        doc = a.preparation(); doc["launcher_exit_code"] = 1
        self.assertRaises(ValueError, a.p.check_metadata, doc)

    def test_exact_float_seal_not_integer209(self):
        doc = a.p.seal_metadata({"gain": 1.0})
        a.p.check_metadata(json.loads(json.dumps(doc)))
        self.assertRaises(ValueError, a.obs.meta.check_seal, doc)

    def test_native_manifest_union_not_basename_guesses(self):
        files = a.native_manifest(a.preparation())
        self.assertGreaterEqual(len(files), 218)
        self.assertTrue(any("winsxs" in name.lower() for name in files))

    def test_exact_native_physical_identity(self):
        actual = {"final_path": "actual", "bytes": 2, "sha256": "a" * 64,
                  "volume_serial": 3, "file_index": 9007199254740993, "last_write_ticks": 7}
        a.validate_native(actual, {"observed": actual})
        a.validate_native(actual | {"final_path": "hardlink"}, {"observed": actual})

    def test_large_file_identity_int_not_rounded(self):
        actual = {"final_path": "actual", "bytes": 2, "sha256": "a" * 64,
                  "volume_serial": 3, "file_index": 9007199254740993, "last_write_ticks": 7}
        self.assertRaises(ValueError, a.validate_native,
                          actual | {"file_index": 9007199254740992}, {"observed": actual})

    def test_unknown_native_image_rejected(self):
        actual = {"final_path": "unlisted", "bytes": 2, "sha256": "a" * 64,
                  "volume_serial": 3, "file_index": 5, "last_write_ticks": 7}
        self.assertRaises(ValueError, a.validate_native, actual, {})

    def test_all_native_identity_fields_exact(self):
        actual = {"final_path": "actual", "bytes": 2, "sha256": "a" * 64,
                  "volume_serial": 3, "file_index": 5, "last_write_ticks": 7}
        for field in ("bytes", "sha256", "volume_serial", "file_index", "last_write_ticks"):
            changed = actual | {field: "b" * 64 if field == "sha256" else actual[field] + 1}
            self.assertRaises(ValueError, a.validate_native, changed, {"observed": actual})

    def test_native_bool_int_not_equal(self):
        actual = {"final_path": "actual", "bytes": 2, "sha256": "a" * 64,
                  "volume_serial": 3, "file_index": 1, "last_write_ticks": 7}
        self.assertRaises(ValueError, a.validate_native, actual | {"file_index": True}, {"observed": actual})

    def test_only_original_backward_removed(self):
        source = tiny_tree(); original = ast.dump(source)
        transformed, removed = a.forward_only_ast(source)
        self.assertEqual(len(removed), 1)
        namespace = {}; exec(compile(transformed, "fixture", "exec"), namespace)
        self.assertEqual(namespace["boundary_backward"](object()), 3)
        self.assertEqual(ast.dump(source), original)

    def test_full_actual194_transform_preserves_other_statements(self):
        path = ROOT / "scripts/194_train_mel_lr_scale.py"
        source = ast.parse(path.read_text(encoding="utf-8-sig"))
        transformed, removed = a.forward_only_ast(source)
        names = [n.name for n in transformed.body]
        self.assertEqual(names, ["validate_metadata", "boundary_backward"])
        original = copy.deepcopy(next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "boundary_backward"))
        for node in ast.walk(original):
            if hasattr(node, "body") and isinstance(node.body, list):
                node.body[:] = [n for n in node.body if ast.dump(n, include_attributes=False) not in removed]
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(transformed.body[1], include_attributes=False))

    def test_no_backward_refused(self):
        self.assertRaises(ValueError, a.forward_only_ast, tiny_tree("pass"))

    def test_backward_args_refused(self):
        self.assertRaises(ValueError, a.forward_only_ast, tiny_tree("scaled.backward(retain_graph=True)"))

    def test_different_backward_target_refused(self):
        self.assertRaises(ValueError, a.forward_only_ast, tiny_tree("another.backward()"))

    def test_hidden_optimizer_step_refused(self):
        tree = tiny_tree(); tree.body[1].body.insert(1, ast.parse("optimizer.step()").body[0])
        self.assertRaises(ValueError, a.forward_only_ast, tree)

    def test_complete_function_order_required(self):
        tree = tiny_tree(); tree.body.reverse()
        self.assertRaises(ValueError, a.forward_only_ast, tree)

    def test_predetermined_true_selection_local_rng(self):
        doc = a.preparation()["prepared_backend"]
        lock = json.loads((ROOT / "results/training_protocol_20261001/dataset_lock.json").read_text(encoding="utf-8"))
        before = random.getstate()
        selected, files = a.selected_true_files(lock, doc["source_sampler"])
        self.assertEqual(random.getstate(), before)
        self.assertEqual([row["domain"] for row in selected], ["musdb", "mir1k", "instrumental"])
        self.assertEqual([row["track_id"] for row in selected],
                         ["Patrick Talbot - A Reason To Leave.stem", "bug_5", "Guete_des_Geschicks_v1"])
        self.assertEqual(len(files), 19)

    def test_other_counter_refused_before_draw(self):
        self.assertRaises(ValueError, a.selected_true_files, {}, {"seed": 20261002, "cursor": 4501})

    def test_existing_and_outside_results_refused_before_child(self):
        with patch.object(subprocess, "Popen", side_effect=AssertionError("No child")):
            for path in (ROOT / "results", ROOT / "scripts/ema215", ROOT / "results/ema214_audio_backend_preparation_attempt04"):
                self.assertRaises(ValueError, a.collect, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
