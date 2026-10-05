"""Seal mutation and strict symmetric verification tests; no model forward."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location("independent183",Path(__file__).with_name("183_verify_train_context_reconstruction.py"))
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class IndependentVerifierTests(unittest.TestCase):
    def row(self):
        return v.acq.seal({"model":"unit_model","counter":3000,"slots":[{"slot":0,"value":.125}],"seconds":1.})

    def test_01_real_seal_mutation_accepts_full_rows(self):
        row={"counter":3000,"slots":[],"seconds":1.}
        returned=v.acq.seal(row)
        self.assertIs(returned,row)
        self.assertIn("content_sha256",row)
        v.checked_row(copy.deepcopy(row),row,"original_file_sha","original_file_sha")

    def test_02_original_asymmetry_regression(self):
        row=self.row()
        self.assertNotEqual(v.k.q.r.r.plain(copy.deepcopy(row)),row)
        v.checked_row(copy.deepcopy(row),row,"sha","sha")

    def test_03_missing_embedded_seal_rejected(self):
        row=self.row(); embedded=v.k.q.r.r.plain(row)
        with self.assertRaises(ValueError): v.checked_row(row,embedded,"sha","sha")

    def test_04_missing_file_seal_rejected(self):
        row=self.row()
        with self.assertRaises(ValueError): v.checked_row(v.k.q.r.r.plain(row),row,"sha","sha")

    def test_05_bad_embedded_seal_rejected(self):
        row=self.row(); changed=copy.deepcopy(row); changed["content_sha256"]="0"*64
        with self.assertRaises(ValueError): v.checked_row(row,changed,"sha","sha")

    def test_06_body_tamper_not_ignored(self):
        row=self.row(); changed=copy.deepcopy(row); changed["slots"][0]["value"]+=.1
        with self.assertRaises(ValueError): v.checked_row(row,changed,"sha","sha")

    def test_07_resealed_different_body_rejected(self):
        row=self.row(); changed=v.acq.seal(row|{"counter":3001})
        with self.assertRaises(ValueError): v.checked_row(row,changed,"sha","sha")

    def test_08_manifest_file_sha_mismatch_rejected(self):
        row=self.row()
        with self.assertRaises(ValueError): v.checked_row(row,row,"new_sha","original_sha")

    def test_09_bool_and_integer_not_equal(self):
        left=v.acq.seal({"value":0}); right=v.acq.seal({"value":False})
        with self.assertRaises(ValueError): v.checked_row(left,right,"sha","sha")
        self.assertNotEqual(v.canonical({"value":0}),v.canonical({"value":False}))

    def test_10_nonfinite_rejected(self):
        row=self.row(); changed=copy.deepcopy(row); changed["slots"][0]["value"]=float("nan")
        with self.assertRaises(ValueError): v.checked_row(row,changed,"sha","sha")

    def test_11_nonobject_row_rejected(self):
        with self.assertRaises(ValueError): v.checked_row([],self.row(),"sha","sha")

    def test_12_complete36_row_accounting(self):
        result={"row_sha256":{f"row_{j:02d}.json":"sha" for j in range(36)},"rows":[{}]*36}
        self.assertEqual(len(v.check_row_accounting(result)),36)
        for altered in ({"rows":[{}]*35},{"row_sha256":{"row_00.json":"sha"}}):
            with self.assertRaises(ValueError): v.check_row_accounting(result|altered)

    def test_13_extra_committed_row_rejected(self):
        result={"row_sha256":{f"row_{j:02d}.json":"sha" for j in range(37)},"rows":[{}]*36}
        with self.assertRaises(ValueError): v.check_row_accounting(result)

    def test_14_verification_scope_types_and_budget(self):
        v.check_plan(v.expected_scope())
        for field,bad in (("slot_checks",215),("model_updates",False),("cuda_used",0),("model_forward_used",True),("run_exit_code",1),("initial_verify_exit_code",0)):
            with self.assertRaises(ValueError): v.check_plan(v.expected_scope()|{field:bad})

    def test_15_existing_verification_never_overwritten(self):
        with mock.patch.object(Path,"exists",return_value=True):
            with self.assertRaises(ValueError): v.verify(v.DEFAULT_OUT)

    def test_16_canonical_key_order_is_irrelevant(self):
        left=v.acq.seal({"a":1,"b":.5});right=v.acq.seal({"b":.5,"a":1})
        v.checked_row(left,right,"sha","sha")

    def test_17_extra_field_resealed_rejected(self):
        row=self.row(); changed=v.acq.seal(row|{"extra":"not_recorded"})
        with self.assertRaises(ValueError): v.checked_row(row,changed,"sha","sha")

    def test_18_schema_has_no_training_or_inference_authority(self):
        scope=v.expected_scope()
        for field in ("cuda_used","deployment","training_authorized","model_forward_used","backward_used"):
            self.assertIs(scope[field],False)
        self.assertEqual(scope["release_selection"],"NONE")


if __name__=="__main__":
    unittest.main(verbosity=2)
