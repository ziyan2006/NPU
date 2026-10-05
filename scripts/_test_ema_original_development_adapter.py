"""New220 source/metadata checks; no audio/model/optimizer/PT/CUDA run."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("original220", Path(__file__).with_name("220_ema_original_development_adapter.py"))
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def aggregate_function():
    namespace = {"statistics": statistics, "math": d.math}
    tree = d.original_ast("scripts/119_model_selection_suite.py", ("aggregate",))
    exec(compile(tree, "original119[220metadata-only]", "exec"), namespace)
    return namespace["aggregate"]


class AdapterMetadataTests(unittest.TestCase):
    def test_01_complete_original_LF32_functions_and_decorators_unchanged(self):
        selected = d.original_ast("scripts/194_train_mel_lr_scale.py", ("low_frequency_backing_metrics", "boundary_evaluation"))
        original = ast.parse((d.ROOT / "scripts/194_train_mel_lr_scale.py").read_text(encoding="utf-8"))
        nodes = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in {"low_frequency_backing_metrics", "boundary_evaluation"}]
        self.assertEqual(ast.dump(selected, include_attributes=False), ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False))
        self.assertEqual(ast.unparse(selected.body[1].decorator_list[0]), "torch.no_grad()")

    def test_02_original_corpus_checked_suite_definitions_unchanged(self):
        tree = d.original_ast("scripts/146_evaluate_paired_development.py", ("LockedDevelopmentCorpus", "checked_suite"))
        self.assertEqual([n.name for n in tree.body], ["LockedDevelopmentCorpus", "checked_suite"])
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in ast.walk(tree)))

    def test_03_definition_order_missing_or_unbound_source_refuses(self):
        for relative, names in (("scripts/194_train_mel_lr_scale.py", ("boundary_evaluation", "low_frequency_backing_metrics")),
                                ("scripts/194_train_mel_lr_scale.py", ("missing",)),
                                ("scripts/143_paired_distillation_mechanics.py", ("LockedTruePool",))):
            with self.assertRaises(ValueError):
                d.original_ast(relative, names)

    def test_04_changed_source_SHA_refuses_before_compile(self):
        with patch.object(d, "sha", return_value="0"*64):
            with self.assertRaises(ValueError):
                d.original_ast("scripts/194_train_mel_lr_scale.py", ("boundary_evaluation",))

    def test_05_original_saved_frozen44_full177_aggregate_metadata_only(self):
        doc = json.loads((d.ROOT / "results/mel_lr_scale_20261004/frozen_scores.json").read_text(encoding="utf-8"))
        body = {k: v for k, v in doc.items() if k != "content_sha256"}
        own = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(doc["content_sha256"], own)
        self.assertEqual(len(doc["rows"]), 177)
        self.assertEqual(doc["summary"], aggregate_function()(doc["rows"]))

    def test_06_original_manifest_31tracks_177views_93windows(self):
        doc = json.loads((d.ROOT / "results/mel_lr_scale_20261004/selection_suite.json").read_text(encoding="utf-8"))
        self.assertEqual((doc["tracks"], doc["clips"], doc["source_windows"]), (31, 177, 93))
        self.assertEqual(doc["sha256"], d.EXPECTED_SUITE)
        self.assertEqual((doc["crop_samples"], doc["score_start_sample"], doc["score_end_sample"]), (89856, 25088, 89344))

    def lock(self, count=31):
        records = [{"domain": ("musdb", "mir1k", "instrumental")[i % 3], "role": "development", "track_id": "dev"+str(i),
                    "mix_files": ["file"+str(i)]} for i in range(count)]
        records += [{"domain": "musdb", "role": role, "track_id": role, "mix_files": [role+"_NEVER_READ"]}
                    for role in ("train", "blind", "regression")]
        return {"records": records, "files": {"file"+str(i): {"sha256": "a"*64} for i in range(count)}}

    def test_07_development_only_all31_no_train_or_final_source(self):
        bindings = d.development_source_bindings(self.lock())
        self.assertEqual(len(bindings), 31)
        self.assertFalse(any("NEVER_READ" in p for p in bindings))

    def test_08_missing_track_duplicate_or_foreign_domain_refused(self):
        locks = [self.lock(30)]
        duplicate = self.lock(); duplicate["records"][1] = copy.deepcopy(duplicate["records"][0]); locks.append(duplicate)
        foreign = self.lock(); foreign["records"][0]["domain"] = "acceptance"; locks.append(foreign)
        for lock in locks:
            with self.assertRaises(ValueError):
                d.development_source_bindings(lock)

    def test_09_no_optimizer_backward_teacher_or_audio_output_entry(self):
        tree = ast.parse(Path(d.__file__).read_text(encoding="utf-8"))
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(s.endswith((".Adam", ".backward", ".step", ".write", ".save")) for s in calls))
        self.assertFalse(any(s.endswith(".evaluate_pair") for s in calls))

    def test_10_real_stage_gate_rejects_fixture_before_Model_and_RNG(self):
        validator = object.__new__(d.OriginalDevelopmentEvaluator)
        validator._check = Mock()
        factory = Mock(side_effect=AssertionError("No model construction"))
        owner = SimpleNamespace(_actual=False, live=SimpleNamespace(validation_due=True, step=4750))
        with self.assertRaises(ValueError):
            validator.evaluate(owner, factory)
        factory.assert_not_called()

    def test_11_no_early_or_duplicate_stage_even_with_actual_marker(self):
        for step, due in ((4501, True), (4750, False), (5001, True)):
            validator = object.__new__(d.OriginalDevelopmentEvaluator); validator._check = Mock()
            owner = SimpleNamespace(_actual=True, live=SimpleNamespace(validation_due=due, step=step))
            with self.assertRaises(ValueError):
                validator.evaluate(owner, Mock())

    def test_12_original_policy_assess_AST_no_threshold_or_rank_rewrite(self):
        tree = d.original_ast("scripts/119_model_selection_suite.py", ("assess",))
        source = ast.parse((d.ROOT / "scripts/119_model_selection_suite.py").read_text(encoding="utf-8"))
        original = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "assess")
        self.assertEqual(ast.dump(tree.body[0], include_attributes=False), ast.dump(original, include_attributes=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
