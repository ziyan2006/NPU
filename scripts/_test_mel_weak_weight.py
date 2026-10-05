"""Loss-only fork authority, identical target/state, rollback and gradient tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("weak_tests", Path(__file__).with_name("154_train_mel_weak_weight.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
p, m, d, acq, ARMS = t.p, t.m, t.d, t.acq, t.ARMS
POLICY = json.loads(p.PROTOCOL.read_text(encoding="utf-8"))
SOURCE = json.loads(m.data.PROTOCOL.read_text(encoding="utf-8"))
torch.set_num_threads(2)


def factory():
    net = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.fill_(.03)
    return net


def metadata():
    return [{"domain": name, "role": "train" if i < 3 else "pseudo_label_train_candidate",
        "vocal_db": -12 if i == 0 else 0, "purpose": t.old.PURPOSE,
        "exploratory_eligible": True, "deployment_eligible": False} for i, name in enumerate(m.DOMAINS)]


def cheap_backward(net, x, v, *args):
    loss = sum(parameter.square().sum() for parameter in net.parameters()) + v.mean()
    loss.backward()
    return {"loss": float(loss.detach())}


def fake_stream(path):
    # Tiny model state machinery test; does not stand in for real-input smoke.
    inp = t.old.inp
    origin = inp.ApprovedPairStream.__new__(inp.ApprovedPairStream)
    origin.path, origin.bound, origin.cursor, origin.seed = path, acq.sha256(path), 0, 20261002
    origin.true = type("TruePool", (), {"bound": "locked"})()
    ids = [f"song_{i}" for i in range(24)]
    origin.doc = {"purpose": t.old.PURPOSE, "schema": 1, "protocol": json.loads(inp.PROTOCOL.read_text(encoding="utf-8")),
        "pair_ids": ids, "snapshots": {"htdemucs": {}, "kim_melband": {}}, "exploratory_training_authorized": True,
        "deployment_authorized": False, "original_htdemucs_exit_code": None,
        "accepted_completion_basis": "independent_full_waveform_verify_for_exploration_only",
        "approved_records": [{"song_id": song, "exploratory_eligible": True, "deployment_eligible": False} for song in ids]}
    def origin_batch():
        x = torch.zeros(6, 2, 89856)
        targets = {arm: x.clone() for arm in ARMS}
        origin.cursor += 1
        return {"x": x, "targets": targets, "metadata": metadata(), "domains": m.DOMAINS, "cursor": origin.cursor-1}
    origin.next_batch = origin_batch
    engine = t.old.ExplorationEngine(factory, SOURCE, {"fixture": "origin"}, "cpu", origin, cheap_backward)
    engine.update_next(origin)
    state = engine.state_dict(origin)
    state["step"] = state["sampler"]["cursor"] = state["schedule"]["step"] = state["schedule"]["last_validation"] = 1000
    for saved in state["arms"].values():
        saved["updates"] = 1000
        saved["optimizer"]["param_groups"][0]["lr"] = m.learning_rate(1000, m.validate_protocol(SOURCE))
        for value in saved["optimizer"]["state"].values():
            value["step"].fill_(1000)
    value = p.MelForkStream.__new__(p.MelForkStream)
    value.path, value.bound, value.cursor, value.seed = path, acq.sha256(path), 1000, origin.seed
    value.true, value.source_state = origin.true, state
    value.doc = {"source_protocol": SOURCE, "protocol": POLICY, "purpose": p.PURPOSE}
    value.last_metadata = None
    def batch():
        cursor = value.cursor
        gen = torch.Generator().manual_seed(cursor)
        x = torch.randn(6, 2, 89856, generator=gen)*.025
        v = x*.2
        v[2].zero_()
        value.cursor += 1
        value.last_metadata = metadata()
        return {"x": x, "targets": {arm: v.clone() for arm in ARMS}, "metadata": copy.deepcopy(value.last_metadata),
            "domains": m.DOMAINS, "cursor": cursor}
    value.next_batch = batch
    return value


class WeakTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "approved.json"
        acq.write_new_json(self.path, {"fixture": "unit only"})
        runtime = d.deterministic_runtime("cpu")
        runtime.__enter__()
        self.addCleanup(runtime.__exit__, None, None, None)
        self.stream = fake_stream(self.path)

    def engine(self):
        with patch.object(p, "check_approval"):
            return t.ForkEngine(factory, self.stream, "cpu", smoke=True)

    def test_protocol_fixed_teacher_budget_and_no_release(self):
        p.check_protocol(POLICY)
        for key, value in (("teacher", "htdemucs"), ("absolute_limit", 2000), ("additional_common_steps", 1000),
                           ("deployment_authorized", True), ("origin_step", 0), ("approved_pseudo_songs", 275)):
            with self.assertRaises(ValueError):
                p.check_protocol(POLICY | {key: value})

    def test_boolean_weight_or_changed_roles_rejected(self):
        for replacement in ({"weights": {"uniform_control": True, "weak12_weight2": 2}},
                            {"weights": {"uniform_control": 1, "weak12_weight2": 3}}, {"internal_arm_keys": {}}):
            with self.assertRaises(ValueError):
                p.check_protocol(POLICY | replacement)

    def test_only_explicit_weak_true_slots_weighted(self):
        meta = metadata()
        self.assertEqual(t.slot_weights(meta, False), [1.]*6)
        self.assertEqual(t.slot_weights(meta, True), [2., 1., 1., 1., 1., 1.])
        meta[1]["vocal_db"] = -12
        meta[3]["vocal_db"] = -12
        self.assertEqual(t.slot_weights(meta, True), [2., 2., 1., 1., 1., 1.])

    def test_weight_recipe_role_geometry_rejected(self):
        for slot, key, value in ((0, "vocal_db", True), (0, "vocal_db", -18), (2, "vocal_db", -12),
                                  (1, "role", "development"), (4, "domain", "musdb")):
            meta = metadata(); meta[slot][key] = value
            with self.assertRaises(ValueError):
                t.slot_weights(meta, True)

    def test_initial_model_adam_modes_and_cpu_rng_equal_source(self):
        engine = self.engine()
        state = engine.state_dict(self.stream)
        for arm in ARMS:
            self.assertTrue(m.equal_state(state["arms"][arm], self.stream.source_state["arms"][ARMS[1]]))
        self.assertTrue(m.equal_state(state["rng"], self.stream.source_state["rng"]))
        self.assertEqual(state["sampler"]["cursor"], 1000)
        self.assertEqual(state["teacher"], "kim_melband")
        self.assertEqual(state["arm_roles"], p.ROLES)

    def test_cpu_long_training_or_unapproved_stream_before_optimizer(self):
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("Adam")):
            with self.assertRaises(ValueError):
                t.ForkEngine(factory, self.stream, "cpu", smoke=False)
            with self.assertRaises(ValueError):
                t.ForkEngine(factory, object(), "cuda")

    def test_equal_weights_identical_original_gradient(self):
        batch = self.stream.next_batch(); x, v = batch["x"], batch["targets"][ARMS[0]]
        engine = self.engine_after_reset()
        a, b = factory(), factory()
        left = t.weighted_backward(a, x, v, engine.wa, engine.gs, "cpu", [1.]*6)
        right = m.fit.backward_batch(b, x, v, engine.wa, engine.gs, "cpu", 96, 44, "reconstruction_l1", 1)
        self.assertEqual(left, right)
        self.assertTrue(all(torch.equal(pa.grad, pb.grad) for pa, pb in zip(a.parameters(), b.parameters())))

    def engine_after_reset(self):
        self.stream.cursor = 1000
        return self.engine()

    def test_weighted_gradient_matches_explicit_sum(self):
        engine = self.engine()
        batch = self.stream.next_batch(); x, v = batch["x"], batch["targets"][ARMS[0]]
        weights = [2., 1., 1., 1., 1., 1.]
        a, b = factory(), factory()
        result = t.weighted_backward(a, x, v, engine.wa, engine.gs, "cpu", weights)
        manual_loss = 0.
        for i, w in enumerate(weights):
            xb, vb = x[i:i+1], v[i:i+1]
            spec = m.core.stft_batch(xb)
            output = b(torch.einsum("fk,bcft->bckt", engine.wa, spec.abs()))
            loss, _, _ = m.fit.reconstruction_loss((output[:, :2]+1)/2, spec, xb, vb, engine.gs, 96, 44)
            (loss*w/sum(weights)).backward()
            manual_loss += float(loss.detach())*w/sum(weights)
        self.assertEqual(result["loss"], manual_loss)
        self.assertTrue(all(torch.equal(pa.grad, pb.grad) for pa, pb in zip(a.parameters(), b.parameters())))

    def test_disk_resume_saved_state_no_alias_and_budget(self):
        engine = self.engine(); engine.backward = cheap_backward
        engine.update_next(self.stream)
        receipt = t.save_checkpoint(Path(self.folder.name), engine, self.stream)
        saved = m.load_checked_checkpoint(Path(self.folder.name) / receipt["checkpoint"], receipt["sha256"])
        untouched = d.portable(saved)
        expected_rows = [engine.update_next(self.stream) for _ in range(2)]
        expected = engine.state_dict(self.stream)
        self.stream.cursor = 1000
        other = self.engine(); other.backward = cheap_backward
        other.load_state_dict(saved, self.stream)
        actual_rows = [other.update_next(self.stream) for _ in range(2)]
        self.assertEqual([{k:v for k,v in row.items() if k != "seconds"} for row in expected_rows],
                         [{k:v for k,v in row.items() if k != "seconds"} for row in actual_rows])
        self.assertTrue(m.equal_state(expected, other.state_dict(self.stream)))
        self.assertTrue(m.equal_state(saved, untouched))
        with self.assertRaises(ValueError):
            other.update_next(self.stream)

    def test_second_arm_failure_rolls_back_entire_pair(self):
        engine = self.engine(); before = engine.state_dict(self.stream)
        def fail(net, *args):
            if net is engine.models[ARMS[1]]:
                torch.rand(2); m.random.random(); m.np.random.random()
                raise RuntimeError("second")
            return cheap_backward(net, *args)
        engine.backward = fail
        with self.assertRaises(RuntimeError):
            engine.update_next(self.stream)
        self.assertTrue(m.equal_state(before, engine.state_dict(self.stream)))

    def test_changed_pseudo_target_rejected_and_cursor_rolled_back(self):
        engine = self.engine(); before = engine.state_dict(self.stream)
        original = self.stream.next_batch
        def bad():
            batch = original(); batch["targets"][ARMS[1]][3] += .01
            return batch
        self.stream.next_batch = bad
        with self.assertRaises(ValueError):
            engine.update_next(self.stream)
        self.assertTrue(m.equal_state(before, engine.state_dict(self.stream)))

    def test_changed_teacher_roles_phase_origin_or_cursor_refused(self):
        engine = self.engine(); state = engine.state_dict(self.stream)
        for key, value in (("teacher", "htdemucs"), ("arm_roles", {}), ("smoke", False), ("origin_sha256", "bad"), ("limit", 1500)):
            with self.assertRaises(ValueError):
                engine.load_state_dict(state | {key: value}, self.stream)
        for key, value in (("teacher", "htdemucs"), ("cursor", 999), ("cursor", 1501), ("cursor", True), ("approval_sha256", "bad")):
            with self.assertRaises(ValueError):
                self.stream.load_state_dict(state["sampler"] | {key: value})

    def test_old_origin_hash_checked_and_changed_stream_refuses(self):
        with self.assertRaises(ValueError):
            m.load_checked_checkpoint(self.path, "0"*64)
        engine = self.engine()
        other = Path(self.folder.name) / "changed.json"; acq.write_new_json(other, {"changed": True})
        self.stream.path = other
        with self.assertRaises(ValueError):
            engine.update_next(self.stream)

    def test_independent_listening_roles_and_common_gain(self):
        spec = importlib.util.spec_from_file_location("weak_review_unit", Path(__file__).with_name("156_review_mel_weak_weight.py"))
        review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
        models = {role: object() for role in ("frozen", "origin_mel1000", *p.ROLES.values())}
        x, v = torch.ones(2, 100), torch.ones(2, 100)*.2
        with patch.object(review.r, "vocal_wave", return_value=x*2):
            waves, gain = review.listening_waves(models, x, v)
        self.assertEqual(len(waves), 11)
        self.assertEqual(gain, .475)
        self.assertTrue(torch.equal(waves["reference_vocal"], v*gain))
        self.assertEqual(max(float(w.abs().max()) for w in waves.values()), float(torch.tensor(.95)))


if __name__ == "__main__":
    unittest.main()
