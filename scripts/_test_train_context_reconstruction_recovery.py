"""Frozen181 safety regressions plus scoped reader/provenance recovery tests."""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import torch

spec = importlib.util.spec_from_file_location("recovery182", Path(__file__).with_name("182_diagnose_train_context_reconstruction_recovery.py"))
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
spec = importlib.util.spec_from_file_location("frozen181_tests", Path(__file__).with_name("_test_train_context_reconstruction.py"))
old_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_tests)
KernelTests, ResultTests = old_tests.KernelTests, old_tests.ResultTests


class RecoveryTests(unittest.TestCase):
    def readers(self):
        dataset = SimpleNamespace(seed=123, rows=["original"], config={"original": True})
        true = SimpleNamespace()
        calls = []
        def true_crop(domain, seed, cursor):
            calls.append(("true", domain, seed, cursor))
            j = k.m.DOMAINS.index(domain)
            x = torch.full((2, k.q.SAMPLES), (cursor+j)*1e-5)
            return {"x": x, "v": x*.2, "meta": {"domain": domain, "role": "train", "vocal_db": 0, "input_pcm_sha256": k.m.pilot.wave_digest(x)}}
        def recipe(rows, config, seed, cursor):
            self.assertEqual(rows, dataset.rows); self.assertEqual(config, dataset.config)
            calls.append(("recipe", seed, cursor)); return cursor
        def pseudo_crop(cursor):
            x = torch.full((2, k.q.SAMPLES), cursor*1e-6)
            return {"x": x, "v": x*.3, "meta": {"input_pcm_sha256": k.m.pilot.wave_digest(x)}}
        true.crop, dataset.crop = true_crop, pseudo_crop
        return dataset, true, recipe, calls

    def test_39_new_range_and_canonical_reader_equal(self):
        torch.set_num_threads(2)
        dataset, true, recipe, calls = self.readers()
        with mock.patch.object(k.m.data, "crop_recipe", side_effect=recipe):
            for cursor in k.q.CURSORS:
                batch = k.collect_fixed_train(dataset, true, cursor)
                original = object.__new__(k.p.ComponentStream)
                original.dataset, original.true, original.seed = dataset, true, dataset.seed
                original.config, original.cursor = dataset.config, cursor
                expected = original.next_batch()
                self.assertTrue(torch.equal(expected["x"], batch["x"]))
                for arm in k.t.ARMS: self.assertTrue(torch.equal(expected["targets"][arm], batch["v"]))
                self.assertEqual(expected["metadata"], batch["metadata"])
                self.assertEqual(batch["diagnostic_counter"], cursor)
        self.assertEqual(calls[:6], [("true", name, 123, 3000) for name in k.m.DOMAINS[:3]]+[("recipe",123,j) for j in (9000,9001,9002)])

    def test_40_scope_rejects_invalid_counter_before_io(self):
        for cursor in (True, False, 3000., "3000", 2999, 3012, 2000, 3500):
            with self.assertRaises(ValueError): k.collect_fixed_train(None, None, cursor)

    def test_41_historical164_guard_unchanged(self):
        self.assertEqual(k.q.s.CURSORS, tuple(range(2000,2012)))
        with self.assertRaises(ValueError): k.q.s.collect(None, None, 3000)
        self.assertEqual(k.q.CURSORS, tuple(range(3000,3012)))

    def test_42_input_hash_mismatch_fails(self):
        dataset, true, recipe, _ = self.readers()
        with mock.patch.object(k.m.data,"crop_recipe",side_effect=recipe): batch=k.collect_fixed_train(dataset,true,3000)
        entry=k.input_entry(batch,3000)
        self.assertEqual(len(entry["target_sha256"]),6)
        batch["metadata"][0]["input_pcm_sha256"]="not_actual"
        with self.assertRaises(ValueError): k.input_entry(batch,3000)

    def test_43_new_identity_and_no_global_override(self):
        self.assertNotEqual(k.DEFAULT_OUT,k.q.DEFAULT_OUT)
        self.assertEqual(k.q.TEST.name,"_test_train_context_reconstruction.py")
        plan=k.q.fixed_scope()|{"tool":k.TOOL,"recovery_of":str(k.q.DEFAULT_OUT),"input_reader":"explicit_fixed3000..3011_canonical177"}
        k.check_plan(plan)
        for field in ("tool","recovery_of","input_reader"):
            with self.assertRaises(ValueError): k.check_plan(plan|{field:"different"})

    def test_44_duplicate182_worker_rejected(self):
        reply=SimpleNamespace(stdout="999999\n")
        with mock.patch.object(k.q,"no_active_worker"),mock.patch.object(k.subprocess,"run",return_value=reply):
            with self.assertRaises(ValueError): k.no_active_worker()
            reply.stdout=f"{os.getpid()}\n{os.getppid()}\n"; k.no_active_worker()

    def test_45_failure_manifest_wrong_exit_rejected(self):
        doc={"purpose":"NONRELEASE_TRAIN_CONTEXT_RECONSTRUCTION_FAILED_ATTEMPT_REVIEW","actual_exit_code":0}
        with mock.patch.object(Path,"read_text",return_value=k.json.dumps(doc)):
            with self.assertRaises(ValueError): k.check_failure()

    def test_46_real_audit_role_or_counter_mismatch_rejected(self):
        core=(None,{},None,{"frozen":"sha"})
        with mock.patch.object(k.q,"checked_sources",return_value=core), mock.patch.object(k,"check_failure",return_value={}),mock.patch.object(k.acq,"sha256",return_value="unit"), mock.patch.object(k.acq,"read_sealed",return_value={"tool":"wrong"}):
            with self.assertRaises(ValueError): k.sources()

    def test_47_complete_result_must_match_real_input_audit(self):
        plan=k.q.fixed_scope()|{"tool":k.TOOL,"recovery_of":str(k.q.DEFAULT_OUT),"input_reader":"explicit_fixed3000..3011_canonical177"}
        result=plan|{"inputs":["not_audited"]}
        with mock.patch.object(k.q,"validate_result"),mock.patch.object(k.acq,"read_sealed",return_value={"inputs":[]}):
            with self.assertRaises(ValueError): k.validate_result(plan,result)


if __name__=="__main__":
    unittest.main(verbosity=2)
