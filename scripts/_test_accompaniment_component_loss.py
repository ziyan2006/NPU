"""Synthetic prototype tests, not real TRAIN audit/resume/quality evidence."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location("component_loss_tests", Path(__file__).with_name("176_accompaniment_component_loss.py"))
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


def sources():
    a = torch.tensor([[1., -1., 0., 0.], [1., -1., 0., 0.]])
    v = torch.tensor([[0., 0., 1., -1.], [0., 0., 1., -1.]])
    return a+v, v


class ComponentLossTests(unittest.TestCase):
    def setUp(self):
        self.mix, self.vocal = sources()
        self.predicted = (.8*self.vocal+.1*(self.mix-self.vocal)).requires_grad_()
        self.meta = {"domain": "musdb", "role": "train"}

    def test_control_scalar_and_gradient_bit_exact159(self):
        full, info = k.original.source_projection_auxiliary(self.predicted, self.mix, self.vocal, self.meta)
        control, control_info = k.source_projection_component_loss(self.predicted, self.mix, self.vocal, self.meta, 1)
        self.assertTrue(torch.equal(full, control))
        self.assertEqual(info, control_info)
        gf = torch.autograd.grad(full, self.predicted, retain_graph=True)[0]
        gc = torch.autograd.grad(control, self.predicted)[0]
        self.assertTrue(torch.equal(gf, gc))

    def test_candidate_is_residual_only(self):
        loss, info = k.source_projection_component_loss(self.predicted, self.mix, self.vocal, self.meta, 0)
        self.assertAlmostEqual(float(loss.detach()), info["remaining_vocal_gain"]**2, places=7)
        self.assertAlmostEqual(float(loss.detach()), .04, places=6)
        self.assertTrue(info["active"])

    def test_candidate_gradient_difference_matches_removed_accompaniment(self):
        full, _ = k.source_projection_component_loss(self.predicted, self.mix, self.vocal, self.meta, 1)
        residual, _ = k.source_projection_component_loss(self.predicted, self.mix, self.vocal, self.meta, 0)
        gf = torch.autograd.grad(full, self.predicted, retain_graph=True)[0]
        gr = torch.autograd.grad(residual, self.predicted, retain_graph=True)[0]
        removed = torch.autograd.grad((full-residual), self.predicted)[0]
        torch.testing.assert_close(gf-gr, removed, rtol=1e-6, atol=1e-7)
        self.assertGreater(float(gr.norm()), 0.)
        self.assertGreater(float(removed.norm()), 0.)

    def test_exact_vocal_prediction_has_zero_residual(self):
        predicted = self.vocal.clone().requires_grad_()
        loss, _ = k.source_projection_component_loss(predicted, self.mix, self.vocal, self.meta, 0)
        self.assertLess(float(loss.detach()), 1e-12)

    def test_accompaniment_error_not_penalized_in_candidate(self):
        predicted = (self.vocal+.3*(self.mix-self.vocal)).requires_grad_()
        full, _ = k.source_projection_component_loss(predicted, self.mix, self.vocal, self.meta, 1)
        residual, _ = k.source_projection_component_loss(predicted, self.mix, self.vocal, self.meta, 0)
        self.assertGreater(float(full.detach()), .08)
        self.assertLess(float(residual.detach()), 1e-12)

    def test_pseudo_and_instrumental_original_skip(self):
        for meta in ({"domain": "pseudo", "role": "pseudo_label_train_candidate"}, {"domain": "instrumental", "role": "train"}):
            for weight in (1, 0):
                loss, info = k.source_projection_component_loss(self.predicted, self.mix, self.vocal, meta, weight)
                self.assertFalse(info["active"])
                self.assertEqual(float(loss.detach()), 0.)

    def test_inactive_and_correlated_sources_original_skip(self):
        for vocal in (torch.zeros_like(self.vocal), self.mix/2):
            for weight in (1, 0):
                loss, info = k.source_projection_component_loss(self.predicted, self.mix, vocal, self.meta, weight)
                self.assertFalse(info["active"])
                self.assertEqual(float(loss.detach()), 0.)

    def test_other_weight_types_and_values_refused(self):
        for value in (True, False, 0., 1., 2, -1, None):
            with self.assertRaises(ValueError):
                k.source_projection_component_loss(self.predicted, self.mix, self.vocal, self.meta, value)

    def test_no_development_acceptance_or_role_mismatch(self):
        for meta in ({"domain": "musdb", "role": "development"}, {"domain": "mir1k", "role": "acceptance"}, {"domain": "pseudo", "role": "train"}):
            with self.assertRaises(ValueError):
                k.source_projection_component_loss(self.predicted, self.mix, self.vocal, meta, 0)

    def test_nonfinite_or_wrong_precision_refused(self):
        for value in (self.predicted.double(), self.predicted*float("nan")):
            with self.assertRaises(ValueError):
                k.source_projection_component_loss(value, self.mix, self.vocal, self.meta, 0)

    def test_layout_mismatch_refused(self):
        with self.assertRaises(ValueError):
            k.source_projection_component_loss(self.predicted[:, :2], self.mix, self.vocal, self.meta, 1)

    def test_inputs_and_grad_storage_unchanged(self):
        before = self.predicted.detach().clone(), self.mix.clone(), self.vocal.clone()
        for weight in (1, 0):
            loss, _ = k.source_projection_component_loss(self.predicted, self.mix, self.vocal, self.meta, weight)
            torch.autograd.grad(loss, self.predicted)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, (self.predicted.detach(), self.mix, self.vocal))))
        self.assertIsNone(self.predicted.grad)


if __name__ == "__main__":
    unittest.main()
