"""Synthetic and provenance units only; no Adam, CUDA, real-input model forward, or training."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("cross191_under_test", Path(__file__).with_name("191_diagnose_train_lf_cross_boundary.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.level = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return torch.zeros((6, 4, 128, c.FRAMES), dtype=torch.float32) + self.level


class CrossBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.wa, cls.gs, cls.geom = c.geometry()
        cls.meta = c.acq.read_sealed(c.k.DEFAULT_OUT / "inputs.json")["inputs"][0]["metadata"]
        axis = torch.arange(c.SAMPLES, dtype=torch.float32)
        cls.x = (0.1 * torch.sin(axis * .023))[None, None].repeat(6, 2, 1)
        cls.v = .3 * cls.x
        cls.v[2].zero_()
        cls.spectrum = c.q.core.stft_batch(cls.x)
        cls.bands = torch.einsum("fk,bcft->bckt", cls.wa, cls.spectrum.abs())

    def synthetic_batch(self):
        meta = copy.deepcopy(self.meta)
        for index, entry in enumerate(meta):
            entry["input_pcm_sha256"] = c.m.pilot.wave_digest(self.x[index])
        batch = {"x": self.x.clone(), "v": self.v.clone(), "metadata": meta, "diagnostic_counter": 3000}
        return batch, c.k.input_entry(batch, 3000)

    def dummy_rows(self):
        rows = []
        for model_id, model in enumerate(c.MODELS):
            for cursor in c.CURSORS:
                slots = []
                for j, meta in enumerate(self.meta):
                    boundaries = {str(kill): {"metrics": {key: float(model_id * 10 + kill + j) for key in c.METRICS}} for kill in c.KILLS}
                    slots.append({"slot": j, "bucket": c.bucket(meta), "boundaries": boundaries})
                rows.append({"model": model, "counter": cursor, "slots": slots})
        return rows

    def test_protocol_exact_scope(self):
        self.assertEqual(c.canonical(json.loads(c.PROTOCOL.read_text())), c.canonical(c.scope()))
    def test_scope_accounting(self):
        scope = c.scope()
        self.assertEqual((scope["unique_input_slots"], scope["model_batches"], scope["model_input_slots"], scope["slot_boundary_reconstructions"]), (72, 36, 216, 432))
    def test_scope_typed_boolean_rejected(self):
        doc = c.scope() | {"model_updates": False}
        with self.assertRaises(ValueError): c.check_scope(doc)
    def test_scope_range_rejected(self):
        with self.assertRaises(ValueError): c.check_scope(c.scope() | {"cursors": list(range(2000, 2012))})
    def test_literal_matrix_support(self):
        self.assertEqual(self.geom["forced_zero_fft_bins"], {"44": [0, 1, 2, 3, 4, 5], "32": [0, 1, 2, 3]})
    def test_matrix_identity_shape_dtype(self):
        self.assertEqual(tuple(self.gs.shape), (513, 128)); self.assertEqual(self.gs.dtype, torch.float32)
        self.assertEqual(c.q.suite.tensor_digest(self.gs), self.geom["synthesis_sha256"])
    def test_matrix_frequency(self):
        self.assertAlmostEqual(self.geom["bin_hz"] * 5, 215.33203125)
    def test_one_forward_two_waves(self):
        net = Tiny()
        waves, proof = c.paired_reconstruction(net, self.bands, self.spectrum, self.gs)
        self.assertEqual(net.calls, 1); self.assertEqual(set(waves), {"44", "32"}); self.assertEqual(proof["forward_calls"], 1)
        self.assertNotEqual(waves["44"].data_ptr(), waves["32"].data_ptr())
    def test_input_mask_spectrum_unchanged(self):
        b, s = self.bands.clone(), self.spectrum.clone()
        waves, proof = c.paired_reconstruction(Tiny(), self.bands, self.spectrum, self.gs)
        self.assertTrue(torch.equal(b, self.bands)); self.assertTrue(torch.equal(s, self.spectrum))
        self.assertTrue(proof["same_raw_mask"] and proof["inputs_mask_unchanged"])
    def test_batched_mask_layout(self):
        mask = c.raw_mask(Tiny(), self.bands)
        self.assertEqual(tuple(mask.shape), (6, 2, 128, 352)); self.assertEqual(mask.dtype, torch.float32)
    def test_double_bands_rejected(self):
        with self.assertRaises(ValueError): c.raw_mask(Tiny(), self.bands.double())
    def test_partial_batch_rejected(self):
        with self.assertRaises(ValueError): c.raw_mask(Tiny(), self.bands[:1])
    def test_nonfinite_bands_rejected(self):
        bad = self.bands.clone(); bad[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError): c.raw_mask(Tiny(), bad)
    def test_unbounded_mask_rejected(self):
        net = Tiny()
        with torch.no_grad(): net.level.fill_(3)
        with self.assertRaises(ValueError): c.raw_mask(net, self.bands)
    def test_complex128_rejected(self):
        with self.assertRaises(ValueError): c.paired_reconstruction(Tiny(), self.bands, self.spectrum.to(torch.complex128), self.gs)
    def test_model_modes_grad_rng_unchanged(self):
        net = Tiny(); net.eval(); net.level.grad = torch.tensor(.2)
        before, rng, grad = net.level.clone(), c.m.capture_rng("cpu"), net.level.grad
        c.paired_reconstruction(net, self.bands, self.spectrum, self.gs)
        self.assertFalse(net.training); self.assertTrue(torch.equal(net.level, before)); self.assertIs(net.level.grad, grad)
        self.assertEqual(float(grad), float(torch.tensor(.2))); self.assertTrue(c.m.equal_state(rng, c.m.capture_rng("cpu")))
    def test_mutating_model_rejected(self):
        class Bad(Tiny):
            def forward(self, x):
                self.level.add_(.1)
                return super().forward(x)
        with self.assertRaises(ValueError): c.paired_reconstruction(Bad(), self.bands, self.spectrum, self.gs)
    def test_cuda_not_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())
        c.paired_reconstruction(Tiny(), self.bands, self.spectrum, self.gs)
        self.assertFalse(torch.cuda.is_initialized())
    def test_native_same_support(self):
        self.assertEqual(c.INTERVAL, (25088, 89344)); self.assertEqual(c.q.suite.scoring_slice(c.SAMPLES, 96), slice(*c.INTERVAL))
    def test_outer_edge_not_wave_metric_benefit(self):
        x, v, pv = self.x[:1], self.v[:1], self.v[:1].clone()
        first = c.metrics(pv, x, v, self.meta[0])
        pv[..., :1000] += .1
        self.assertEqual(c.canonical(first), c.canonical(c.metrics(pv, x, v, self.meta[0])))
    def test_zero_vocal_skips_ratios(self):
        doc = c.metrics(self.v[2:3], self.x[2:3], self.v[2:3], self.meta[2])
        self.assertEqual(doc["projection_skip_reason"], "zero_vocal_reference")
        self.assertIsNone(doc["metrics"]["residual_vocal_abs_gain"]); self.assertIsNone(doc["metrics"]["vocal_error_snr_db"])
    def test_pseudo_not_true_projection(self):
        doc = c.metrics(self.v[3:4], self.x[3:4], self.v[3:4], self.meta[3])
        self.assertIsNone(doc["metrics"]["residual_vocal_abs_gain"]); self.assertEqual(doc["metrics"]["pseudo_target_mae"], 0)
        self.assertEqual(doc["reference_kind"], "pseudo_label_not_final_truth")
    def test_low_activity_not_filtered(self):
        doc = c.activity(self.x[:1], self.v[:1] * 1e-6)
        self.assertFalse(doc["original_source_geometry_active"]); self.assertFalse(doc["exact_zero_vocal_reference"])
    def test_true_roles_rejected(self):
        with self.assertRaises(ValueError): c.bucket(self.meta[0] | {"role": "development"})
    def test_pseudo_role_mismatch_rejected(self):
        with self.assertRaises(ValueError): c.bucket(self.meta[3] | {"role": "train"})
    def test_exact_fixed_input_and_rng(self):
        batch, expected = self.synthetic_batch(); rng = c.m.capture_rng("cpu")
        self.assertEqual(c.check_input(batch, 3000, expected), expected)
        self.assertTrue(c.m.equal_state(rng, c.m.capture_rng("cpu")))
    def test_changed_target_hash_rejected(self):
        batch, expected = self.synthetic_batch(); batch["v"][0, 0, 0] += .001
        with self.assertRaises(ValueError): c.check_input(batch, 3000, expected)
    def test_changed_pcm_hash_rejected(self):
        batch, expected = self.synthetic_batch(); batch["x"][0, 0, 0] += .001
        with self.assertRaises(ValueError): c.check_input(batch, 3000, expected)
    def test_fixed_range_rejected(self):
        batch, expected = self.synthetic_batch()
        with self.assertRaises(ValueError): c.check_input(batch, 2000, expected)
    def test_summary_unique_counts(self):
        result = c.summary(self.dummy_rows())
        for model in c.MODELS:
            for kill in ("44", "32"):
                self.assertEqual(sum(row["unique_fixed_input_slots"] for row in result[model][kill].values()), 72)
    def test_paired_direct_effect(self):
        result = c.contrasts(self.dummy_rows())
        values = result["direct_boundary_" + c.MODELS[0]]["paired_slots"]
        self.assertEqual(len(values), 72); self.assertEqual(values[0]["delta"][c.METRICS[0]], -12)
    def test_paired_interaction_not_double_counted(self):
        result = c.contrasts(self.dummy_rows())["endpoint_boundary_interaction_candidate_minus_control"]
        self.assertEqual(len(result["paired_slots"]), 72)
        self.assertTrue(all(row["delta"][c.METRICS[0]] == 0 for row in result["paired_slots"]))
    def test_partial_rows_cannot_contrast(self):
        with self.assertRaises(KeyError): c.contrasts(self.dummy_rows()[:-1])
    def test_symmetric_seal_full_types(self):
        doc = c.acq.seal({"scope": 0, "array": [True, 1]})
        c.n.v.checked_row(doc, copy.deepcopy(doc), "a", "a")
        wrong = c.acq.seal({"scope": False, "array": [True, 1]})
        with self.assertRaises(ValueError): c.n.v.checked_row(doc, wrong, "a", "a")
    def test_row_seal_and_file_hash_rejected(self):
        doc = c.acq.seal({"a": 1}); changed = doc | {"a": 2}
        with self.assertRaises(ValueError): c.n.v.checked_row(changed, doc, "a", "a")
        with self.assertRaises(ValueError): c.n.v.checked_row(doc, doc, "b", "a")
    def test_tree_digest_typed_and_tensor_identity(self):
        self.assertNotEqual(c.tree_digest({"a": False}), c.tree_digest({"a": 0}))
        self.assertNotEqual(c.tree_digest(torch.ones(2)), c.tree_digest(torch.ones(2).double()))
    def test_existing_run_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); (path / "run_status.json").touch()
            with self.assertRaises(ValueError): c.q.reject_existing_run(path)
    def test_existing_prepare_rejected(self):
        with self.assertRaises(ValueError): c.q.fresh_output(c.k.DEFAULT_OUT)
    def test_bound_dependency_changed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "immutable.txt"; path.write_text("fixture")
            with self.assertRaises(ValueError): c.q.check_bindings({str(path): "0" * 64})
    def test_status_correct_counts_and_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); c.status(path, "row_committed", 5)
            doc = json.loads((path / "run_status.json").read_text())
            self.assertEqual((doc["completed_model_input_slots"], doc["completed_slot_boundary_reconstructions"]), (30, 60))
            self.assertFalse(doc["cuda_used"]); self.assertEqual(doc["model_updates"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
