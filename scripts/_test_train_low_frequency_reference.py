"""Synthetic reference-only mechanisms; no real TRAIN run or model inference."""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch

spec = importlib.util.spec_from_file_location("lowfreq184", Path(__file__).with_name("184_diagnose_train_low_frequency_reference.py"))
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def zeros(self):
        return torch.zeros(2, u.SAMPLES)

    def spectrum(self):
        spectrum = torch.zeros(2, 513, u.FRAMES, dtype=torch.complex64)
        spectrum[:, 0, :] = 1
        spectrum[:, 1, :] = 2
        spectrum[:, 5, :] = 3
        spectrum[:, 6, :] = 4
        spectrum[:, 512, :] = 5
        return spectrum

    def test_01_exact_scope(self):
        u.check_scope(u.scope())
        for field, bad in (("cursors", list(range(2000,2012))),("slot_checks",71),("model_updates",False),("cuda_used",0),
            ("model_forward_used",True),("lf_kill_bands",40),("training_authorized",True),("threads",4),("demeaning","new_target")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                u.check_scope(u.scope() | {field:bad})

    def test_02_geometry(self):
        geometry = u.original_geometry()
        self.assertEqual(geometry["forced_zero_fft_bins"],list(range(6)))
        self.assertEqual(geometry["forced_zero_bin_hz"][-1],215.33203125)

    def test_03_geometry_mutation_rejected(self):
        with mock.patch.object(u.q.core.t09,"make_synthesis_matrix",return_value=torch.ones(513,128).numpy()/128):
            with self.assertRaises(ValueError): u.original_geometry()

    def test_04_single_sided_weights(self):
        doc=u.energy(self.spectrum(),u.REGIONS["native"])
        frames=252
        self.assertEqual(doc["per_bin_energy"][0],2*frames)
        self.assertEqual(doc["per_bin_energy"][1],2*2*4*frames)
        self.assertEqual(doc["per_bin_energy"][512],2*25*frames)
        self.assertEqual(doc["partition_energy"]["other_forced_zero_bins1_5"],2*2*(4+9)*frames)

    def test_05_partition_and_channels_conserve(self):
        doc=u.energy(self.spectrum(),u.REGIONS["native"])
        u.validate_energy(doc)
        self.assertAlmostEqual(sum(doc["partition_fraction"].values()),1)
        self.assertAlmostEqual(sum(doc["per_bin_fraction"]),1)
        self.assertAlmostEqual(doc["total_energy"],sum(doc["per_bin_energy"]))

    def test_06_zero_has_no_epsilon(self):
        doc=u.energy(torch.zeros_like(self.spectrum()),u.REGIONS["native"])
        self.assertEqual(doc["total_energy"],0)
        self.assertIsNone(doc["forced_zero_fraction"])
        self.assertEqual(doc["fraction_skip"],"exact_zero_spectral_reference")
        self.assertTrue(all(x is None for x in doc["per_bin_fraction"]))

    def test_07_positive_tiny_not_skipped(self):
        doc=u.energy(self.spectrum()*1e-15,u.REGIONS["native"])
        self.assertGreater(doc["total_energy"],0)
        self.assertIsNotNone(doc["forced_zero_fraction"])

    def test_08_partition_tamper_rejected(self):
        doc=u.energy(self.spectrum(),u.REGIONS["native"])
        doc["partition_energy"]["windowed_dc_bin0"]+=1
        with self.assertRaises(ValueError):u.validate_energy(doc)

    def test_09_fraction_tamper_rejected(self):
        doc=u.energy(self.spectrum(),u.REGIONS["native"])
        doc["partition_fraction"]["windowed_dc_bin0"]+=.001
        with self.assertRaises(ValueError):u.validate_energy(doc)

    def test_10_bin_energy_tamper_rejected(self):
        doc=u.energy(self.spectrum(),u.REGIONS["native"])
        doc["per_channel_bin_energy"][0][0]+=1
        with self.assertRaises(ValueError):u.validate_energy(doc)

    def test_11_nonfinite_or_bool_energy_rejected(self):
        for bad in (float("nan"),float("inf"),True,-1):
            doc=u.energy(self.spectrum(),u.REGIONS["native"]);doc["per_bin_energy"][0]=bad
            with self.assertRaises(ValueError):u.validate_energy(doc)

    def test_12_shapes_dtypes(self):
        for value in (torch.zeros(1,u.SAMPLES),self.zeros().double(),torch.zeros(2,u.SAMPLES-1),self.zeros().to(torch.complex64)):
            with self.assertRaises(ValueError):u.checked_reference(value)
        for value in (self.spectrum().to(torch.complex128),self.spectrum()[None],self.spectrum()[...,:351]):
            with self.assertRaises(ValueError):u.energy(value,u.REGIONS["native"])

    def test_13_nonfinite_reference_or_spectrum(self):
        value=self.zeros();value[0,0]=float("nan")
        with self.assertRaises(ValueError):u.checked_reference(value)
        value=self.spectrum();value[0,0,0]=float("inf")
        with self.assertRaises(ValueError):u.energy(value,u.REGIONS["native"])

    def test_14_grad_reference_rejected(self):
        with self.assertRaises(ValueError):u.demean_copy(self.zeros().requires_grad_())

    def test_15_exact_interval_and_no_edges(self):
        spectrum=self.spectrum();spectrum[...,0:98]=10000;spectrum[...,350:]=10000
        clean=self.spectrum()
        self.assertEqual(u.energy(spectrum,u.REGIONS["native"]),u.energy(clean,u.REGIONS["native"]))
        for region in ((97,350),(98,351),[98,350],(True,350)):
            with self.assertRaises(ValueError):u.energy(spectrum,region)

    def test_16_supports_separate(self):
        native=u.energy(self.spectrum(),u.REGIONS["native"])
        common=u.energy(self.spectrum(),u.REGIONS["common_descriptive_only"])
        self.assertAlmostEqual(native["total_energy"]/common["total_energy"],252/220)
        self.assertEqual(common["frame_interval"],[130,350])

    def test_17_mean_subtraction_copy_only(self):
        value=torch.arange(u.SAMPLES,dtype=torch.float32)[None].repeat(2,1)*1e-7
        before=value.clone()
        changed,means=u.demean_copy(value)
        self.assertTrue(torch.equal(value,before))
        self.assertNotEqual(changed.data_ptr(),value.data_ptr())
        self.assertTrue(torch.equal(changed,value-means))
        self.assertTrue(torch.equal(means,value.mean(-1,keepdim=True)))

    def test_18_constant_hann_spreads_not_just_dc(self):
        value=torch.full((2,u.SAMPLES),.125)
        doc=u.energy(u.q.core.stft_batch(value),u.REGIONS["native"])
        self.assertGreater(doc["per_bin_energy"][1],0)
        self.assertAlmostEqual(doc["per_bin_energy"][1]/doc["per_bin_energy"][0],.5,places=5)
        changed,means=u.demean_copy(value)
        self.assertEqual(float(changed.abs().max()),0)
        self.assertEqual(float(means[0]),.125)

    def test_19_zero_mean_sine_has_windowed_dc(self):
        phase=torch.arange(u.SAMPLES,dtype=torch.float32)*2*torch.pi*1.25/1024
        value=torch.sin(phase)[None].repeat(2,1)*.01
        value,_=u.demean_copy(value)
        doc=u.energy(u.q.core.stft_batch(value),u.REGIONS["native"])
        self.assertLess(abs(float(value.mean())),1e-8)
        self.assertGreater(doc["per_bin_energy"][0],0)

    def test_20_moments_activity_and_zero(self):
        zero=u.moments(self.zeros())
        self.assertTrue(zero["whole"]["exact_zero"])
        self.assertFalse(zero["native"]["active_rms_gt_1e_4"])
        active=u.moments(torch.full((2,u.SAMPLES),.001))
        self.assertTrue(active["native"]["active_rms_gt_1e_4"])
        self.assertEqual(active["native"]["sample_interval"],[25088,89344])

    def test_21_old_reference_match_and_tamper(self):
        doc=u.energy(self.spectrum(),u.REGIONS["native"])
        old={"reference_spectral_energy":doc["total_energy"],"reference_forced_zero_energy_fraction":doc["forced_zero_fraction"],"reference_fraction_skip":None}
        u.check_old_native(doc,old)
        for change in ({"reference_spectral_energy":doc["total_energy"]+1},{"reference_forced_zero_energy_fraction":.5}):
            with self.assertRaises(ValueError):u.check_old_native(doc,old|change)

    def test_22_old_zero_compatible(self):
        doc=u.energy(torch.zeros_like(self.spectrum()),u.REGIONS["native"])
        u.check_old_native(doc,{"reference_spectral_energy":0,"reference_forced_zero_energy_fraction":None,"reference_fraction_skip":"zero_or_near_zero_reference"})

    def test_23_symmetric_seals_and_types(self):
        doc=u.acq.seal({"counter":3000})
        u.v.checked_row(copy.deepcopy(doc),doc,"sha","sha")
        for different in ({"counter":True},{"counter":3001}):
            with self.assertRaises(ValueError):u.v.checked_row(doc,u.acq.seal(different),"sha","sha")
        with self.assertRaises(ValueError):u.v.checked_row(doc,doc,"bad","sha")

    def test_24_bound_hash_changed_rejected(self):
        with mock.patch.object(u.acq,"sha256",return_value="changed"):
            with self.assertRaises(ValueError):u.q.check_bindings({"declared":"original"})

    def test_25_no_output_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);(path/"run_status.json").touch()
            with self.assertRaises(ValueError):u.q.reject_existing_run(path)
        with mock.patch.object(Path,"exists",return_value=True):
            with self.assertRaises(ValueError):u.verify(u.DEFAULT_OUT)
            with self.assertRaises(ValueError):u.prepare(u.DEFAULT_OUT)

    def test_26_range_or_bool_rejected(self):
        for cursor in (2000,2999,3012,True,3000.0):
            with self.assertRaises(ValueError):u.k.collect_fixed_train(None,None,cursor)

    def test_27_no_model_or_backward_load_calls(self):
        tree=ast.parse(Path(u.__file__).read_text(encoding="utf-8"))
        forbidden={"load_checked_checkpoint","frozen_factory","load_state_dict","backward","step","Adam","slot_probe","checked_sources","source_evidence_from183"}
        names={node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else "" for node in ast.walk(tree) if isinstance(node,ast.Call)}
        self.assertFalse(names & forbidden)

    def test_28_rng_and_cuda_not_initialized(self):
        before=u.m.capture_rng("cpu")
        value=self.zeros()
        with u.q.d.deterministic_runtime("cpu"):
            u.energy(u.q.core.stft_batch(value),u.REGIONS["native"]);u.demean_copy(value)
        self.assertTrue(u.m.equal_state(before,u.m.capture_rng("cpu")))
        self.assertFalse(torch.cuda.is_initialized())

    def test_29_probe_input_immutability(self):
        value=self.zeros()
        before=value.clone()
        probe=u.reference_probe(value,{"reference_spectral_energy":0,"reference_forced_zero_energy_fraction":None,"reference_fraction_skip":"zero_or_near_zero_reference"})
        self.assertTrue(probe["original_reference_unchanged"])
        self.assertTrue(torch.equal(value,before))
        self.assertFalse(probe["demeaned_copy_is_new_truth"])
        self.assertFalse(probe["waveform_quality_floor_claimed"])

    def test_30_zero_fraction_cannot_be_forged(self):
        doc=u.energy(torch.zeros_like(self.spectrum()),u.REGIONS["native"])
        doc["forced_zero_fraction"]=0.0
        with self.assertRaises(ValueError):u.validate_energy(doc)

    def result_fixture(self):
        previous={"reference_spectral_energy":0,"reference_forced_zero_energy_fraction":None,"reference_fraction_skip":"zero_or_near_zero_reference"}
        probe=u.reference_probe(self.zeros(),previous)
        inputs, rows, prior=[],[],[]
        for counter in u.CURSORS:
            metadata=[{"domain":name,"role":"pseudo_label_train_candidate" if name=="pseudo" else "train","vocal_db":-12 if j<2 else 0}
                for j,name in enumerate(("musdb","mir1k","instrumental","pseudo","pseudo","pseudo"))]
            inputs.append({"counter":counter,"input_sha256":["unit_input"]*6,"target_sha256":["unit_target"]*6,"metadata":metadata})
            rows.append({"counter":counter,"slots":[{"slot":j,"metadata":meta,"probe":copy.deepcopy(probe)} for j,meta in enumerate(metadata)],"seconds":1.})
            prior.append({"slots":[{"probe":{"lf44":previous}}]*6})
        plan=u.scope()|{"bindings_sha256":{"unit":"hash"},"geometry":u.original_geometry(),"expected_inputs":inputs,
            "expected_runtime":{"device":"cpu","threads":2},"input_reader":"immutable182_explicit_fixed3000..3011"}
        result=copy.deepcopy(plan)|{"inputs":inputs,"rows":rows,"row_sha256":{f"row_{j:02d}.json":"hash" for j in range(12)},
            "cpu_rng_unchanged":True,"input_reference_unchanged":True,"runtime":plan["expected_runtime"],"summary":u.group_summary(rows)}
        return plan,result,inputs,prior

    def test_31_full72_accounting(self):
        u.validate_result(*self.result_fixture())

    def test_32_short_or_extra_rows_rejected(self):
        for count in (11,13):
            plan,result,inputs,prior=self.result_fixture()
            result["rows"]=(result["rows"]+[copy.deepcopy(result["rows"][0])])[:count]
            with self.assertRaises(ValueError):u.validate_result(plan,result,inputs,prior)

    def test_33_role_or_metadata_tamper_rejected(self):
        plan,result,inputs,prior=self.result_fixture()
        result["rows"][0]["slots"][0]["metadata"]={"domain":"mir1k","role":"development","vocal_db":-12}
        with self.assertRaises(ValueError):u.validate_result(plan,result,inputs,prior)

    def test_34_runtime_or_rng_change_rejected(self):
        for field,value in (("runtime",{"device":"cuda","threads":2}),("cpu_rng_unchanged",False),("input_reference_unchanged",False)):
            plan,result,inputs,prior=self.result_fixture()
            result[field]=value
            with self.assertRaises(ValueError):u.validate_result(plan,result,inputs,prior)

    def test_35_summary_change_rejected(self):
        plan,result,inputs,prior=self.result_fixture()
        result["summary"]={}
        with self.assertRaises(ValueError):u.validate_result(plan,result,inputs,prior)

    def test_36_moments_activity_tamper_rejected(self):
        doc=u.moments(self.zeros());u.validate_moments(doc)
        doc["native"]["active_rms_gt_1e_4"]=True
        with self.assertRaises(ValueError):u.validate_moments(doc)

    def test_37_slot_order_or_target_hash_rejected(self):
        plan,result,inputs,prior=self.result_fixture()
        result["rows"][0]["slots"][0]["slot"]=True
        with self.assertRaises(ValueError):u.validate_result(plan,result,inputs,prior)
        plan,result,inputs,prior=self.result_fixture()
        result["inputs"]=copy.deepcopy(inputs)
        result["inputs"][0]["target_sha256"][0]="changed"
        with self.assertRaises(ValueError):u.validate_result(plan,result,inputs,prior)


if __name__=="__main__":
    unittest.main(verbosity=2)

