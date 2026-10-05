"""Exploratory authority, real-input wrapper and transactional engine contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import importlib.util
spec = importlib.util.spec_from_file_location("exploration_test", Path(__file__).with_name("150_train_paired_exploration.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
torch.set_num_threads(2)
SOURCE = json.loads(t.m.data.PROTOCOL.read_text(encoding="utf-8"))
POLICY = json.loads(t.inp.PROTOCOL.read_text(encoding="utf-8"))


def approval():
    ids = [f"song_{i:04d}" for i in range(24)]
    return {"schema": 1, "purpose": t.PURPOSE, "protocol": POLICY, "pair_ids": ids,
        "snapshots": {"htdemucs": {}, "kim_melband": {}}, "exploratory_training_authorized": True,
        "deployment_authorized": False, "original_htdemucs_exit_code": None,
        "accepted_completion_basis": "independent_full_waveform_verify_for_exploration_only",
        "approved_records": [{"song_id": i, "exploratory_eligible": True, "deployment_eligible": False} for i in ids]}


def stream(path):
    value = t.inp.ApprovedPairStream.__new__(t.inp.ApprovedPairStream)
    value.path, value.bound, value.doc, value.cursor = path, t.acq.sha256(path), approval(), 0
    value.seed = 20261002
    value.true = type("TruePool", (), {"bound": "locked"})()
    def batch():
        cursor = value.cursor
        generator = torch.Generator().manual_seed(cursor)
        x = torch.randn(6, 2, 89856, generator=generator) * .025
        a, b = x * .2, x * .2
        a[2].zero_(); b[2].zero_(); b[3:] = x[3:] * .7
        metadata = [{"role": "train"} for _ in range(3)] + [
            {"purpose": t.PURPOSE, "exploratory_eligible": True, "deployment_eligible": False} for _ in range(3)]
        value.cursor += 1
        return {"x": x, "targets": dict(zip(t.ARMS, (a, b))), "domains": t.m.DOMAINS,
                "cursor": cursor, "metadata": metadata}
    value.next_batch = batch
    return value


def factory():
    net = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(.03)
    return net


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "approved.json"
        t.acq.write_new_json(self.path, t.acq.seal(approval()))
        self.runtime = t.d.deterministic_runtime("cpu")
        self.runtime.__enter__()
        self.addCleanup(self.runtime.__exit__, None, None, None)
        self.stream = stream(self.path)

    def engine(self):
        return t.ExplorationEngine(factory, SOURCE, {"fixture": "same_input"}, "cpu", self.stream)

    def test_authority_not_implicit_or_release_or_all275(self):
        t.inp.check_approval(approval())
        for key, value in (("exploratory_training_authorized", False), ("deployment_authorized", True),
                           ("original_htdemucs_exit_code", 0), ("accepted_completion_basis", "old_exit_zero")):
            with self.assertRaises(ValueError):
                t.inp.check_approval(approval() | {key: value})
        for key, value in (("promotion_authorized", True), ("formal_training_authorized", True),
                           ("maximum_exploratory_steps", 10000), ("approved_pseudo_songs", 275)):
            with self.assertRaises(ValueError):
                t.inp.check_protocol(POLICY | {key: value})

    def test_each_id_approved_and_order_locked(self):
        for variant in range(3):
            doc = approval()
            if variant == 0:
                doc["approved_records"][0]["exploratory_eligible"] = False
            elif variant == 1:
                doc["approved_records"].reverse()
            else:
                doc["pair_ids"].append("extra")
            with self.assertRaises(ValueError):
                t.inp.check_approval(doc)

    def test_unapproved_stream_rejected_before_factory_adam_or_cuda(self):
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA")):
            with self.assertRaises(ValueError):
                t.ExplorationEngine(lambda: self.fail("factory"), SOURCE, {}, "cuda", object())

    def test_real_geometry_cpu_no_cuda_work(self):
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA init")), \
             patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CUDA RNG")):
            engine = self.engine()
            result = engine.update_next(self.stream)
        self.assertEqual(result["step"], 1)
        self.assertEqual(len(result["input_sha256"]), 6)
        self.assertNotEqual(result["losses"][t.ARMS[0]], result["losses"][t.ARMS[1]])

    def test_disk_resume_inputs_weights_adam_modes_rng_budget_identical(self):
        engine = self.engine()
        engine.update_next(self.stream)
        path = Path(self.folder.name) / "checkpoint.pt"
        t.m.save_new(path, engine.state_dict(self.stream))
        digest = t.acq.sha256(path)
        expected_row = engine.update_next(self.stream)
        expected = engine.state_dict(self.stream)
        other = stream(self.path)
        restored = t.ExplorationEngine(factory, SOURCE, {"fixture": "same_input"}, "cpu", other)
        restored.load_state_dict(t.m.load_checked_checkpoint(path, digest), other)
        row = restored.update_next(other)
        self.assertEqual({k:v for k,v in expected_row.items() if k != "seconds"}, {k:v for k,v in row.items() if k != "seconds"})
        self.assertTrue(t.m.equal_state(expected, restored.state_dict(other)))

    def test_second_arm_failure_and_missing_approval_in_batch_rollback(self):
        engine = self.engine()
        engine.update_next(self.stream)
        before, original = engine.state_dict(self.stream), engine.backward
        def failed(net, *args):
            if net is engine.models[t.ARMS[1]]:
                t.m.random.random(); t.m.np.random.random(); torch.rand(2)
                raise RuntimeError("second")
            return original(net, *args)
        engine.backward = failed
        with self.assertRaises(RuntimeError):
            engine.update_next(self.stream)
        self.assertTrue(t.m.equal_state(before, engine.state_dict(self.stream)))
        original_batch = self.stream.next_batch
        def wrong():
            row = original_batch()
            row["metadata"][3]["exploratory_eligible"] = False
            return row
        self.stream.next_batch = wrong
        with self.assertRaises(ValueError):
            engine.update_next(self.stream)
        self.assertTrue(t.m.equal_state(before, engine.state_dict(self.stream)))

    def test_bad_state_and_mid_load_failure_preserve_all_state(self):
        engine = self.engine()
        engine.update_next(self.stream)
        saved = engine.state_dict(self.stream)
        for key, value in (("limit", 10000), ("purpose", "RELEASE"), ("runtime", {}), ("deployment_authorized", True)):
            with self.assertRaises(ValueError):
                engine.load_state_dict(saved | {key: value}, self.stream)
        bad = copy.deepcopy(saved)
        bad["arms"][t.ARMS[0]]["optimizer"]["state"].clear()
        with self.assertRaises(ValueError):
            engine.load_state_dict(bad, self.stream)
        engine.update_next(self.stream)
        before = engine.state_dict(self.stream)
        optimizer = engine.optimizers[t.ARMS[1]]
        original, calls = optimizer.load_state_dict, 0
        def fail_once(state):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("partial load")
            return original(state)
        with patch.object(optimizer, "load_state_dict", side_effect=fail_once):
            with self.assertRaises(RuntimeError):
                engine.load_state_dict(saved, self.stream)
        self.assertTrue(t.m.equal_state(before, engine.state_dict(self.stream)))

    def test_terminal_limit_and_changed_file_guard(self):
        engine = self.engine()
        engine.step = engine.limit
        with self.assertRaises(ValueError):
            engine.update_next(self.stream)
        engine.step = 0
        replacement = Path(self.folder.name) / "changed.json"
        t.acq.write_new_json(replacement, {"different": True})
        self.stream.path = replacement
        with self.assertRaises(ValueError):
            engine.update_next(self.stream)

    def test_cursor_refuses_changed_approval_true_role_seed_or_budget(self):
        state = self.stream.state_dict()
        for key, value in (("approval_sha256", "bad"), ("true_lock_sha256", "bad"), ("seed", 1), ("cursor", 1001), ("cursor", True)):
            with self.assertRaises(ValueError):
                self.stream.load_state_dict(state | {key: value})

    def test_observation_preserves_live_state_rng_and_rolls_back_schedule(self):
        engine = self.engine()
        engine.step = engine.schedule.step = self.stream.cursor = 250
        before = engine.state_dict(self.stream)
        class FailedValidator:
            def observe_pair(self, schedule, models, step):
                torch.rand(3); t.m.random.random(); t.m.np.random.random()
                raise RuntimeError("second development score")
        with self.assertRaises(RuntimeError):
            t.observe(engine, FailedValidator(), factory)
        self.assertTrue(t.m.equal_state(before, engine.state_dict(self.stream)))

    def test_fresh_and_cpu_only_cli_never_launches_long_training(self):
        with tempfile.TemporaryDirectory(dir=t.ROOT / "results") as folder:
            with self.assertRaises(ValueError):
                t.require_fresh(Path(folder))
        with patch.object(t.inp, "verified_approval", return_value=approval()), \
             patch.object(t, "verify_mechanism", return_value={"runtime": {"device": "cuda"}}), patch.object(t.d, "verify"), \
             patch.object(t.acq, "read_sealed", return_value={"runtime": {"device": "cuda"}}), \
             patch.object(t, "exploration_gpu_preflight", return_value={"allowed": False}), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("Adam")):
            self.assertEqual(t.train(self.path, Path(self.folder.name), self.path, self.path, self.path), 2)

    def test_checkpoint_replayed_identical_reuses_no_overwrite_and_changed_state_refuses(self):
        engine = self.engine()
        engine.update_next(self.stream)
        out = Path(self.folder.name)
        first = t.save_checkpoint(out, engine, self.stream)
        self.assertEqual(t.save_checkpoint(out, engine, self.stream), first)
        with torch.no_grad():
            next(engine.models[t.ARMS[0]].parameters()).add_(.01)
        with self.assertRaises(ValueError):
            t.save_checkpoint(out, engine, self.stream)
        self.assertEqual(t.acq.sha256(out / first["checkpoint"]), first["sha256"])

    def test_approved_constructor_does_not_call_old_audit_constructor(self):
        import soundfile as sf
        import numpy as np
        out = Path(self.folder.name)
        label = out / "synthetic_float.wav"
        sf.write(label, np.zeros((100, 2), dtype=np.float32), 44100, subtype="FLOAT")
        doc = approval()
        source = {"path": str(self.path), "sha256": t.acq.sha256(self.path), "samples": 100, "pcm_sha256": "fixture"}
        info = {"path": str(label), "sha256": t.acq.sha256(label)}
        historical = {"input": SOURCE["input"], "sampler_seed": 20261002, "records": [
            {"song_id": i, "role": "pseudo_label_train_candidate", "training_eligible": False,
             "source": source, "label_files": {"vocals.wav": info, "accompaniment.wav": info}} for i in doc["pair_ids"]]}
        snapshot = out / "historical.json"
        t.acq.write_new_json(snapshot, t.acq.seal(historical))
        doc["snapshots"]["kim_melband"] = {"path": str(snapshot), "sha256": t.acq.sha256(snapshot)}
        with patch.object(t.inp.data.TeacherWaveformDataset, "__init__", side_effect=AssertionError("audit constructor used for training")):
            dataset = t.inp.ApprovedTeacherDataset(doc, "kim_melband")
        self.assertEqual(len(dataset.rows), 24)
        self.assertTrue(all(r["training_eligible"] is False for r in dataset.rows))
        self.assertEqual(t.acq.sha256(snapshot), doc["snapshots"]["kim_melband"]["sha256"])

    def test_unrecoverable_load_poison_stops_future_updates(self):
        engine = self.engine()
        before = engine.state_dict(self.stream)
        with patch.object(engine.optimizers[t.ARMS[1]], "load_state_dict", side_effect=RuntimeError("permanent")):
            with self.assertRaises(RuntimeError):
                engine.load_state_dict(before, self.stream)
        self.assertTrue(engine.poisoned)
        with self.assertRaises(ValueError):
            engine.update_next(self.stream)

    def test_adam_loading_never_aliases_reusable_saved_state(self):
        engine = self.engine()
        engine.update_next(self.stream)
        saved = engine.state_dict(self.stream)
        untouched = copy.deepcopy(saved)
        engine.load_state_dict(saved, self.stream)
        engine.update_next(self.stream)
        self.assertTrue(t.m.equal_state(saved, untouched))
        engine.load_state_dict(saved, self.stream)
        self.assertTrue(t.m.equal_state(engine.state_dict(self.stream), untouched))

    def test_concurrency_is_explicit_keeps_memory_guard_old_cli_unchanged(self):
        for memory, allowed in ((3000, True), (1200, False)):
            guard = {"gpu_index": 0, "free_mib": memory, "minimum_free_mib": 2300,
                     "utilization_percent": 94, "allowed": False}
            with patch.object(t.d, "gpu_preflight", return_value=guard):
                result = t.exploration_gpu_preflight(approval())
            self.assertEqual(result["allowed"], allowed)
            self.assertTrue(result["concurrency_authorized"])
            self.assertFalse(result["utilization_ceiling_applied"])
        doc = approval()
        doc["protocol"] = POLICY | {"gpu_concurrency_authorized": False}
        with self.assertRaises(ValueError):
            t.exploration_gpu_preflight(doc)


if __name__ == "__main__":
    unittest.main()
