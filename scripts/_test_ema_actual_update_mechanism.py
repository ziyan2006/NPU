"""New217 scope/native/PCM/state adapter tests; no source model or GPU."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("unit217", Path(__file__).with_name("217_ema_actual_update_mechanism.py"))
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)


class ScopeTests(unittest.TestCase):
    def test_actual_prior_gate(self):
        audit, runtime = q.gate()
        self.assertEqual(audit["audit_report"]["exposure"]["forward_completed"], 6)
        self.assertTrue(runtime["runtime_report"]["original_complete_RNG_restore_bit_type_equal"])

    def test_manifest_actual_union(self):
        audit, runtime = q.gate()
        self.assertEqual(len(q.native_manifest(audit, runtime)), 246)

    def test_manifest_conflict(self):
        audit, runtime = q.gate()
        name = next(iter(audit["preheld_native_physical_files"]))
        runtime = copy.deepcopy(runtime)
        runtime["newly_observed_native_files"][name] = {"bad": True}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            q.native_manifest(audit, runtime)

    def test_manifest_missing(self):
        audit, runtime = q.gate()
        runtime = copy.deepcopy(runtime); runtime["newly_observed_native_files"].pop(next(iter(runtime["newly_observed_native_files"])))
        with self.assertRaisesRegex(ValueError, "246"):
            q.native_manifest(audit, runtime)

    def test_scope_cpu(self):
        self.assertIn("not_formal", q.device_scope("cpu"))

    def test_scope_cuda(self):
        self.assertIn("not_formal", q.device_scope("cuda"))

    def test_no_device_fallback(self):
        for device in (None, True, "auto", "cuda:0", "training"):
            with self.assertRaises(ValueError):
                q.device_scope(device)

    def test_full_exposure(self):
        q.validate_exposure({"forward_started": 24, "forward_completed": 24, "backward_completed": 24,
                             "adam_started": 4, "adam_completed": 4})

    def test_exposure_false_not_int(self):
        with self.assertRaises(ValueError):
            q.validate_exposure({"forward_started": True})

    def test_incomplete_backward(self):
        with self.assertRaises(ValueError):
            q.validate_exposure({"forward_started": 24, "forward_completed": 24, "backward_completed": 23,
                                 "adam_started": 4, "adam_completed": 4})

    def test_baseexception_fault(self):
        self.assertTrue(issubclass(q.InjectedAfterCommit, BaseException))
        self.assertFalse(issubclass(q.InjectedAfterCommit, Exception))


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = q.load("unit217_new_storage_adapters", "scripts/212_ema_single_trajectory_engine.py")
        cls.before = cls.e.capture_rng("cpu")
        cls.guard = patch.object(cls.e.torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden in217 units"))
        cls.guard.start()
        cls.Owner, cls.Stream = q.mechanism_types(cls.e)

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop(); cls.e.restore_rng(cls.before, "cpu")
        print("EVIDENCE:217 synthetic PCM/CPU gradient adapters only; sourcePT/model/Adam-step/audio/CUDA=0", flush=True)

    def setUp(self):
        t = self.e.torch
        x = t.full((6, 2, 89856), .125)
        rows = []
        for i, domain in enumerate(self.e.live.DOMAINS):
            row = {"domain": domain, "role": "train" if i < 3 else "pseudo_label_train_candidate",
                   "score_start": 25088, "score_end": 89344, "vocal_db": 0,
                   "input_pcm_sha256": q.hashlib.sha256(x[i].numpy().tobytes()).hexdigest()}
            if i >= 3:
                row.update(purpose="NONRELEASE_PAIRED_EXPLORATION", exploratory_eligible=True,
                           deployment_eligible=False, training_eligible=False)
            rows.append(row)
        self.sampler = {"cursor": 4500, "seed": 20261002, "teacher": "kim_melband", "approval_sha256": "a"*64, "true_lock_sha256": "b"*64}
        self.batch = {"x": x, "v": x.clone(), "domains": self.e.live.DOMAINS, "cursor": 4500, "metadata": rows}
        self.stream = self.Stream(self.sampler, self.batch, "cpu")

    def test_full_clone_noalias(self):
        batch = self.stream.next_batch()
        self.e.c.noalias(batch, list(self.e.c.source.tensor_leaves(self.batch)))
        self.assertTrue(self.e.equal(batch, self.batch))

    def test_cursor_metadata(self):
        self.stream.next_batch()
        self.assertEqual(self.stream.cursor, 4501)
        self.assertTrue(self.e.equal(self.stream.last_metadata, self.batch["metadata"]))

    def test_second_unique_input_refused(self):
        self.stream.next_batch()
        with self.assertRaisesRegex(ValueError, "ONE unique"):
            self.stream.next_batch()

    def test_pcm_mutation_refused(self):
        self.stream.batch["x"][0, 0, 0] = -0.
        with self.assertRaisesRegex(ValueError, "mutated"):
            self.stream.next_batch()

    def test_poison_refused(self):
        self.stream.poisoned = True
        with self.assertRaises(ValueError):
            self.stream.state_dict()

    def test_changed_pcm_hash(self):
        self.batch["metadata"][0]["input_pcm_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "PCM identity"):
            self.Stream(self.sampler, self.batch, "cpu")

    def test_cpu_rng_unchanged(self):
        rng = self.e.capture_rng("cpu"); self.stream.next_batch()
        self.assertTrue(self.e.equal(rng, self.e.capture_rng("cpu")))

    def test_source_metadata_not_alias(self):
        self.stream.next_batch(); self.batch["metadata"][0]["vocal_db"] = 6
        self.assertEqual(self.stream.last_metadata[0]["vocal_db"], 0)

    def test_wrong_unique_counter(self):
        self.batch["cursor"] = 4501
        with self.assertRaises(ValueError):
            self.Stream(self.sampler, self.batch, "cpu")

    def test_target_nan_rejected(self):
        self.batch["v"][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.Stream(self.sampler, self.batch, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
