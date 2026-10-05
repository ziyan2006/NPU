"""Real script119 assessment and read-only/transactional paired adapter contracts."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("paired_development_test", Path(__file__).with_name("146_evaluate_paired_development.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
torch.set_num_threads(2)


def net(weight=0.):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight)
    return model


def fixture_rows():
    t = torch.arange(99856, dtype=torch.float32)
    a = torch.sin(t * .079).repeat(2, 1) * .08
    v = torch.sin(t * .137).repeat(2, 1) * .04
    corpus = SimpleNamespace(train={"x": [{"track_id": "train"}]}, final_ids=["never-test"],
        teacher_meta={"source_ids": ["teacher"]},
        val={domain: [{"track_id": domain}] for domain in ("musdb", "mir1k", "instrumental")},
        audio=lambda row: (a, torch.zeros_like(v)) if row["track_id"] == "instrumental" else (a + v, v))
    return d.suite.build_suite(corpus, 89856, 96)


def metric_evaluator(model, rows, *_):
    scored = []
    region = d.suite.scoring_slice(89856, 96)
    # Synthetic predictions, but REAL waveform metrics, aggregation and POLICY.
    scale = .1 + float(model.weight[0, 0])
    for row in rows:
        x, v = row["x"][..., region], row["v"][..., region]
        scores = d.suite.separation_metrics(v * scale, x, v)
        scored.append({k: value for k, value in row.items() if k not in ("x", "v")} | {"metrics": scores})
    return {"rows": scored, "summary": d.suite.aggregate(scored)}


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture_rows()
        self.validator = d.PairedDevelopmentValidator(self.rows, net(), metric_evaluator)
        self.models = dict(zip(d.m.ARMS, (net(.2), net(.3))))

    def schedule(self):
        protocol = d.m.json.loads(d.m.data.PROTOCOL.read_text(encoding="utf-8"))
        schedule = d.m.SharedSchedule(d.m.validate_protocol(protocol))
        for _ in range(250):
            schedule.complete_step()
        return schedule

    def test_both_real_assessments_and_model_digests_share_inputs_and_policy(self):
        result = self.validator.evaluate_pair(self.models, 250)
        self.assertEqual(set(result["scores"]), set(d.m.ARMS))
        self.assertEqual(result["suite_sha256"], self.validator.manifest["sha256"])
        for arm in d.m.ARMS:
            self.assertEqual(result["scores"][arm], d.suite.assess(result["evaluations"][arm]["summary"], self.validator.baseline["summary"]))
            self.assertTrue(result["scores"][arm]["eligible"])
            self.assertEqual(result["model_state_sha256"][arm], d.state_digest(self.models[arm].state_dict()))

    def test_identical_frozen_models_do_not_fake_a_gain(self):
        scores = self.validator.evaluate_pair({arm: net() for arm in d.m.ARMS}, 0)["scores"]
        for value in scores.values():
            self.assertFalse(value["eligible"])
            self.assertEqual(value["rank_gain_db"], 0.)

    def test_real_scores_commit_to_shared_schedule_once(self):
        schedule = self.schedule()
        result = self.validator.observe_pair(schedule, self.models, 250)
        self.assertEqual(schedule.last_validation, 250)
        for arm in d.m.ARMS:
            self.assertEqual(schedule.best[arm]["rank_gain_db"], result["scores"][arm]["rank_gain_db"])
        with self.assertRaises(ValueError):
            self.validator.observe_pair(schedule, self.models, 250)

    def test_failed_second_evaluation_cannot_commit_first_arm(self):
        schedule = self.schedule()
        before = schedule.state_dict()
        def fail_second(model, *args):
            if model is self.models[d.m.ARMS[1]]:
                raise RuntimeError("second evaluation failed")
            return metric_evaluator(model, *args)
        self.validator.evaluator = fail_second
        with self.assertRaises(RuntimeError):
            self.validator.observe_pair(schedule, self.models, 250)
        self.assertEqual(schedule.state_dict(), before)

    def test_schedule_partial_mutation_is_rolled_back(self):
        schedule = self.schedule()
        before = schedule.state_dict()
        def fail_observe(_):
            schedule.stale[d.m.ARMS[0]] = 99
            raise RuntimeError("synthetic failed commit")
        with patch.object(schedule, "observe", side_effect=fail_observe):
            with self.assertRaises(RuntimeError):
                self.validator.observe_pair(schedule, self.models, 250)
        self.assertEqual(schedule.state_dict(), before)

    def test_wrong_boundary_missing_arm_or_bad_step_never_scores(self):
        schedule = self.schedule()
        with patch.object(self.validator, "_score", side_effect=AssertionError("scoring started")):
            for models, step in (({d.m.ARMS[0]: net()}, 250), (self.models, -1), (self.models, True)):
                with self.assertRaises(ValueError):
                    self.validator.evaluate_pair(models, step)
            for step in (249, 500):
                with self.assertRaises(ValueError):
                    self.validator.observe_pair(schedule, self.models, step)

    def test_changed_input_hash_coverage_baseline_or_policy_rejected(self):
        self.rows[0]["x"][0, 0] += .01
        with self.assertRaises(ValueError):
            self.validator.evaluate_pair(self.models, 250)
        self.setUp()
        self.rows[0]["start_sample"] += 1
        with self.assertRaises(ValueError):
            self.validator.evaluate_pair(self.models, 250)
        self.setUp()
        self.validator.baseline["summary"]["musdb/native"]["clips"] += 1
        with self.assertRaises(ValueError):
            self.validator.evaluate_pair(self.models, 250)
        self.setUp()
        with patch.dict(d.suite.POLICY, {"minimum_mean_vocal_snr_gain_db": 0.}):
            with self.assertRaises(ValueError):
                self.validator.evaluate_pair(self.models, 250)

    def test_missing_row_or_forged_summary_rejected(self):
        for kind in ("row", "summary"):
            def changed(model, *args):
                result = metric_evaluator(model, *args)
                if kind == "row":
                    result["rows"].pop()
                else:
                    result["summary"]["musdb/native"]["metrics"]["vocal_error_snr_db"]["mean"] += 10
                return result
            self.validator.evaluator = changed
            with self.assertRaises(ValueError):
                self.validator.evaluate_pair(self.models, 250)

    def test_model_mode_state_and_rng_preserved_even_on_failure(self):
        model = self.models[d.m.ARMS[0]]
        model.train()
        state, rng = copy.deepcopy(model.state_dict()), d.m.capture_rng("cpu")
        def mutating_eval(model, *args):
            model.eval()
            torch.rand(13)
            with torch.no_grad():
                model.weight.add_(.1)
            return metric_evaluator(model, *args)
        self.validator.evaluator = mutating_eval
        with self.assertRaises(ValueError):
            self.validator.evaluate_pair(self.models, 250)
        self.assertTrue(model.training)
        self.assertTrue(d.m.equal_state(model.state_dict(), state))
        self.assertTrue(d.m.equal_state(d.m.capture_rng("cpu"), rng))

    def test_real_script119_forward_no_optimizer_or_cuda_queries(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
        original = copy.deepcopy(model.state_dict())
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer started")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
             patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")):
            validator = d.PairedDevelopmentValidator(self.rows, model)
            packet = validator.evaluate_pair({arm: model for arm in d.m.ARMS}, 0)
        self.assertTrue(d.m.equal_state(model.state_dict(), original))
        self.assertEqual(set(packet["evaluations"][d.m.ARMS[0]]["summary"]), set(d.suite.VOCAL_DOMAINS) | {"instrumental/native"})
        self.assertFalse(any(score["eligible"] for score in packet["scores"].values()))

    def test_nonfinite_model_state_and_empty_or_changed_historical_suite_rejected(self):
        with self.assertRaises(ValueError):
            d.state_digest({"parameter": torch.tensor([float("nan")])})
        with self.assertRaises(ValueError):
            d.PairedDevelopmentValidator([], net())
        with patch.object(d.suite, "build_suite", return_value=self.rows):
            with self.assertRaises(ValueError):
                d.checked_suite(object(), self.validator.manifest | {"sha256": "changed"})

    def test_locked_reader_rejects_train_or_replaced_paths_before_decoding(self):
        reader = object.__new__(d.LockedDevelopmentCorpus)
        row = {"domain": "musdb", "track_id": "dev", "role": "development", "mix_files": ["exact"]}
        reader.allowed = {("musdb", "dev"): row}
        with patch.object(d.core, "decode_musdb", side_effect=AssertionError("decoded forbidden audio")):
            for changed in (row | {"role": "train"}, row | {"mix_files": ["other"]}, row | {"track_id": "acceptance"}):
                with self.assertRaises(ValueError):
                    reader.audio(changed)

    def test_complete_cpu_checkpoint_only_hash_and_exposure_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "NONRELEASE.pt"
            state = {"purpose": "NONRELEASE_CPU_MECHANISM", "device": "cpu", "binding": "fixture", "cpu_update_limit": 3,
                "step": 3, "arms": {arm: {"updates": 3} for arm in d.m.ARMS}, "sampler": {"cursor": 3}, "schedule": {"step": 3}}
            d.m.save_new(path, state)
            sha = d.acq.sha256(path)
            self.assertEqual(d.checked_cpu_state(path, sha, "fixture", 3), state)
            for binding, step in (("other", 3), ("fixture", 2), ("fixture", True)):
                with self.assertRaises(ValueError):
                    d.checked_cpu_state(path, sha, binding, step)
            with self.assertRaises(ValueError):
                d.checked_cpu_state(path, "0" * 64, "fixture", 3)

    def test_no_overwrite_or_broad_output_target(self):
        for target in (d.ROOT / "results", d.ROOT / "models", d.ROOT / "results/paired_distillation_cpu_20261002"):
            with self.assertRaises(ValueError):
                d.run(target)


if __name__ == "__main__":
    unittest.main()
