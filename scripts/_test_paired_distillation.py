"""CPU mechanics contracts; no real teacher generation or formal student run."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("paired_mechanics_test", Path(__file__).with_name("143_paired_distillation_mechanics.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.set_num_threads(2)
PROTOCOL = json.loads(m.data.PROTOCOL.read_text(encoding="utf-8"))


class FakeStream:
    def __init__(self):
        self.cursor = 0

    def state_dict(self):
        return {"cursor": self.cursor, "binding": "fixture"}

    def load_state_dict(self, state):
        if state["binding"] != "fixture":
            raise ValueError("Changed fixture source binding")
        self.cursor = state["cursor"]

    def next_batch(self):
        gen = torch.Generator(device="cpu").manual_seed(90+self.cursor)
        x = torch.randn(6, 2, 89856, generator=gen)*.02
        a, b = x*.2, x*.2
        # Different pseudo gradients are required; with three shared true slots,
        # the mean signs and Adam's first float32 weight update may still match.
        b[3:] = x[3:]*.8
        result = {"x": x, "targets": dict(zip(m.ARMS, (a, b))), "domains": m.DOMAINS, "cursor": self.cursor}
        self.cursor += 1
        return result


def factory():
    gen = torch.Generator(device="cpu").manual_seed(80)
    net = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=gen)*.05)
    return net


def engine():
    return m.PairedEngine(factory, PROTOCOL, "fixture-bound", purpose="cpu_mechanism")


class PairedTests(unittest.TestCase):
    def test_protocol_is_not_an_ablation_of_graph_loss_optimizer_or_budget(self):
        m.validate_protocol(PROTOCOL)
        for field, value in (("optimizer", "SGD"), ("maximum_steps", 300), ("microbatch", 6),
                             ("learning_rate", .001), ("patience_validation_checks", 2)):
            p = copy.deepcopy(PROTOCOL)
            p["paired_comparison_planned"][field] = value
            with self.assertRaises(ValueError):
                m.validate_protocol(p)

    def test_learning_rate_has_fixed_warmup_cosine_and_step_limits(self):
        c = m.validate_protocol(PROTOCOL)
        self.assertAlmostEqual(m.learning_rate(1, c), 1e-6)
        self.assertAlmostEqual(m.learning_rate(100, c), 1e-4)
        self.assertAlmostEqual(m.learning_rate(10000, c), 1e-5)
        for step in (0, 10001, 1.):
            with self.assertRaises(ValueError):
                m.learning_rate(step, c)

    def test_default_or_cuda_entry_never_constructs_optimizer(self):
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer started")):
            for device, purpose in (("cpu", "training"), ("cuda", "cpu_mechanism")):
                with self.assertRaises(ValueError):
                    m.PairedEngine(factory, PROTOCOL, "bound", device=device, purpose=purpose)
            with self.assertRaises(ValueError):
                m.SharedBatchStream(Path("missing"))

    def test_initial_weights_must_match(self):
        count = 0
        def changed_factory():
            nonlocal count
            net = factory()
            count += 1
            with torch.no_grad():
                next(net.parameters()).add_(count)
            return net
        with self.assertRaises(ValueError):
            m.PairedEngine(changed_factory, PROTOCOL, "bound", purpose="cpu_mechanism")

    def test_pair_update_uses_same_inputs_same_true_targets_and_two_teacher_targets(self):
        a, stream = engine(), FakeStream()
        result = a.update_next(stream)
        self.assertEqual(result["step"], 1)
        self.assertEqual(tuple(result["domains"]), m.DOMAINS)
        self.assertEqual(len(result["input_sha256"]), 6)
        self.assertNotEqual(result["losses"][m.ARMS[0]]["loss"], result["losses"][m.ARMS[1]]["loss"])
        self.assertFalse(m.equal_state(a.optimizers[m.ARMS[0]].state_dict(), a.optimizers[m.ARMS[1]].state_dict()))

    def test_failed_second_arm_rolls_back_models_adam_rng_budget_and_sampler(self):
        a, stream = engine(), FakeStream()
        before = a.state_dict(stream)
        original = m.fit.backward_batch
        def fail_second(net, *args, **kwargs):
            if net is a.models[m.ARMS[1]]:
                raise RuntimeError("synthetic second-arm failure")
            return original(net, *args, **kwargs)
        with patch.object(m.fit, "backward_batch", side_effect=fail_second):
            with self.assertRaises(RuntimeError):
                a.update_next(stream)
        self.assertTrue(m.equal_state(before, a.state_dict(stream)))

    def test_wrong_domain_or_different_true_target_rejected_without_exposure(self):
        for change in ("domain", "truth"):
            a, stream = engine(), FakeStream()
            original = stream.next_batch
            def changed():
                batch = original()
                if change == "domain":
                    batch["domains"] = ("onair",)+m.DOMAINS[1:]
                else:
                    batch["targets"][m.ARMS[1]][0] *= .9
                return batch
            stream.next_batch = changed
            with self.assertRaises(ValueError):
                a.update_next(stream)
            self.assertEqual(a.step, 0)
            self.assertEqual(stream.cursor, 0)

    def test_complete_disk_resume_is_identical_and_checkpoint_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            a, stream = engine(), FakeStream()
            a.update_next(stream)
            path = Path(folder)/"NONRELEASE.pt"
            m.save_new(path, a.state_dict(stream))
            digest = m.pilot.acq.sha256(path)
            expected = a.update_next(stream)
            state = a.state_dict(stream)
            b, other = engine(), FakeStream()
            b.load_state_dict(m.load_checked_checkpoint(path, digest), other)
            self.assertEqual(b.update_next(other), expected)
            self.assertTrue(m.equal_state(state, b.state_dict(other)))
            with path.open("ab") as f:
                f.write(b"changed")
            with self.assertRaises(ValueError):
                m.load_checked_checkpoint(path, digest)

    def test_partial_or_changed_backend_binding_budget_resume_rejected(self):
        a, stream = engine(), FakeStream()
        a.update_next(stream)
        state = a.state_dict(stream)
        for key, value in (("binding", "changed"), ("device", "cuda"), ("cpu_update_limit", 10000), ("step", 0)):
            with self.assertRaises(ValueError):
                a.load_state_dict(state | {key: value}, stream)
        bad = copy.deepcopy(state)
        bad["arms"][m.ARMS[1]]["updates"] = 0
        with self.assertRaises(ValueError):
            a.load_state_dict(bad, stream)

    def test_python_numpy_and_cpu_rng_restore_without_cuda_queries(self):
        with patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CUDA queried")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
             patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")):
            state = m.capture_rng("cpu")
            expected = (random.random(), np.random.random(), torch.rand(5))
            m.restore_rng(state, "cpu")
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.random(), expected[1])
            self.assertTrue(torch.equal(torch.rand(5), expected[2]))

    def test_cuda_rng_helper_requires_matching_device_count_no_cpu_fallback(self):
        with patch.object(torch.cuda, "get_rng_state_all", return_value=[torch.tensor([1], dtype=torch.uint8)]):
            state = m.capture_rng("cuda")
        with self.assertRaises(ValueError):
            m.restore_rng(state, "cpu")
        with patch.object(torch.cuda, "device_count", return_value=0):
            with self.assertRaises(ValueError):
                m.restore_rng(state, "cuda")

    def test_no_eligible_checkpoint_is_none_and_shared_stop_not_before_2000(self):
        schedule = m.SharedSchedule(m.validate_protocol(PROTOCOL))
        scores = {arm: {"eligible": False, "rank_gain_db": None} for arm in m.ARMS}
        for _ in range(2000):
            schedule.complete_step()
            if schedule.step % 250 == 0:
                schedule.observe(scores)
                if schedule.step < 2000:
                    self.assertIsNone(schedule.stopped_at)
        self.assertEqual(schedule.stopped_at, 2000)
        self.assertEqual(schedule.selection(), {arm: "NONE" for arm in m.ARMS})

    def test_one_arm_improvement_keeps_both_running_and_best_not_patience_thresholded(self):
        schedule = m.SharedSchedule(m.validate_protocol(PROTOCOL))
        for _ in range(2500):
            schedule.complete_step()
            if schedule.step % 250 == 0:
                i = schedule.step//250
                scores = {m.ARMS[0]: {"eligible": True, "rank_gain_db": .1},
                          m.ARMS[1]: {"eligible": True, "rank_gain_db": .1+i*.005}}
                schedule.observe(scores)
        self.assertIsNone(schedule.stopped_at)
        self.assertAlmostEqual(schedule.best[m.ARMS[1]]["rank_gain_db"], .15)
        restored = m.SharedSchedule(m.validate_protocol(PROTOCOL))
        restored.load_state_dict(schedule.state_dict())
        self.assertEqual(restored.state_dict(), schedule.state_dict())

    def test_paired_validation_cannot_omit_an_arm_or_duplicate_step(self):
        schedule = m.SharedSchedule(m.validate_protocol(PROTOCOL))
        for _ in range(250):
            schedule.complete_step()
        scores = {arm: {"eligible": True, "rank_gain_db": .1} for arm in m.ARMS}
        with self.assertRaises(ValueError):
            schedule.observe({m.ARMS[0]: scores[m.ARMS[0]]})
        schedule.observe(scores)
        with self.assertRaises(ValueError):
            schedule.observe(scores)

    def test_cpu_budget_cannot_be_extended_by_running_more_steps(self):
        a, stream = engine(), FakeStream()
        for _ in range(3):
            a.update_next(stream)
        with self.assertRaises(ValueError):
            a.update_next(stream)
        self.assertEqual(stream.cursor, 3)


if __name__ == "__main__":
    unittest.main()
