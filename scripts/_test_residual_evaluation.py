"""Final-holdout guards and equal-song aggregation regression tests."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("residual_eval", Path(__file__).with_name("111_evaluate_residual_ablation.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResidualEvaluationTests(unittest.TestCase):
    def test_cross_dataset_composition_alias_rejected(self):
        for split in ("train", "validation"):
            meta = {"splits": {"musdb": {split: ["Skelpolu - Human Mistakes.stem"]}}}
            with self.assertRaises(ValueError):
                module.reject_overlap(meta, "cambridge_multitrack", "Skelpolu_HumanMistakes")
        with self.assertRaises(ValueError):
            module.reject_overlap({"teacher": {"source_ids": ["Triviul - Angelsaint.mp3"]}},
                                  "cambridge_multitrack", "Triviul_Angelsaint")

    def test_training_and_selection_overlap_rejected(self):
        meta = {"splits": {"instrumental": {"train": ["a"], "validation": ["b"]}}}
        for name in ("a", "b"):
            with self.assertRaises(ValueError):
                module.reject_overlap(meta, "mshoxx", name)
        module.reject_overlap(meta, "mshoxx", "c")

    def test_teacher_overlap_rejected(self):
        with self.assertRaises(ValueError):
            module.reject_overlap({"teacher": {"source_ids": ["song"]}}, "dj_teacher_proxy", "song")

    def test_equal_song_not_equal_segment_weight(self):
        rows = []
        for name, count, value in (("a", 3, 2.), ("b", 1, 10.)):
            for _ in range(count):
                rows.append({"dataset": "test", "track": name, "vocals_present": True,
                    "models": {"baseline": {"vocal_si_sdr_db": 0., "accompaniment_si_sdr_db": 0., "accompaniment_error_snr_db": 0.},
                               "candidate": {"vocal_si_sdr_db": value, "accompaniment_si_sdr_db": value, "accompaniment_error_snr_db": value}}})
        result = module.grouped_summary(rows, ["baseline", "candidate"])["test"]
        self.assertEqual(result["models"]["candidate"]["accompaniment_si_sdr_db"]["mean"], 6.)
        self.assertEqual(result["paired_vs_baseline"]["candidate"]["vocal_si_sdr_db"]["n_tracks"], 2)

    def test_counterexample_regression_blocks_upgrade(self):
        delta = {m: {"mean": .2} for m in ("vocal_si_sdr_db", "accompaniment_si_sdr_db", "accompaniment_error_snr_db")}
        summary = {d: {"paired_vs_baseline": {"candidate": delta}} for d in ("mir1k", "musdb", "onair", "mshoxx")}
        self.assertTrue(module.provisional_gate([], summary, "candidate")["passed"])
        row = {"dataset": "cambridge_multitrack", "track": "test_counterexample", "models": {
            "baseline": {"vocal_si_sdr_db": 2., "accompaniment_si_sdr_db": 3.},
            "candidate": {"vocal_si_sdr_db": 1.7, "accompaniment_si_sdr_db": 3.2}}}
        self.assertFalse(module.provisional_gate([row], summary, "candidate")["passed"])

    def test_missing_prediction_score_cannot_silently_pass(self):
        delta = {m: {"mean": .2} for m in ("vocal_si_sdr_db", "accompaniment_si_sdr_db", "accompaniment_error_snr_db")}
        summary = {d: {"paired_vs_baseline": {"candidate": delta}} for d in ("mir1k", "musdb", "onair", "mshoxx")}
        row = {"dataset": "cambridge_multitrack", "track": "missing_mask", "vocals_present": True,
               "models": {"baseline": {"vocal_si_sdr_db": 2., "accompaniment_si_sdr_db": 3.},
                          "candidate": {"vocal_si_sdr_db": None, "accompaniment_si_sdr_db": 3.2}}}
        self.assertFalse(module.provisional_gate([row], summary, "candidate")["passed"])


if __name__ == "__main__":
    unittest.main()
