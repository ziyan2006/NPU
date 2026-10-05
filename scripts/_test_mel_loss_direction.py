"""Synthetic CPU tests only; no real-data or training authority."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location("direction_tests", Path(__file__).with_name("158_diagnose_mel_loss_direction.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
torch.set_num_threads(2)


class DirectionTests(unittest.TestCase):
    def test_joint_projection_correlated_sources_closes(self):
        a = torch.tensor([1., 2., -1., .5], dtype=torch.float64)
        v = torch.tensor([.2, -1., .3, 2.], dtype=torch.float64)
        x, predicted = a+v, .9*v+.1*a
        fit = q.decompose(predicted, x, v)
        self.assertAlmostEqual(fit["accompaniment_gain"], .9)
        self.assertAlmostEqual(fit["remaining_vocal_gain"], .1)
        self.assertLess(fit["relative_closure_error"], 1e-8)

    def test_instrumental_or_collinear_not_identifiable(self):
        a = torch.ones(10)
        self.assertIsNone(q.decompose(a*.2, a, torch.zeros_like(a)))
        self.assertIsNone(q.decompose(a*.2, a, a*.3))

    def test_projection_rejects_misalignment_nonfinite(self):
        with self.assertRaises(ValueError):
            q.decompose(torch.ones(3), torch.ones(4), torch.ones(4))
        with self.assertRaises(ValueError):
            q.decompose(torch.tensor([float('nan')]), torch.ones(1), torch.ones(1))

    def test_single_band_analytic_spectral_solution(self):
        x = torch.ones(1, 2, 1, 5, dtype=torch.complex64)
        mask, info = q.bounded_spectral_mask(x, x*.3, torch.ones(1, 1), 0, 8)
        torch.testing.assert_close(mask, torch.full_like(mask, .3))
        self.assertLess(info["spectral_mse_trace"][-1], 1e-12)
        self.assertFalse(info["optimum_or_theoretical_ceiling_claimed"])

    def test_bounded_solution_clips_negative_and_over_one_targets(self):
        x = torch.ones(1, 2, 1, 3, dtype=torch.complex64)
        for value, expected in ((-1., 0.), (2., 1.)):
            mask, _ = q.bounded_spectral_mask(x, x*value, torch.ones(1, 1), 0, 4)
            torch.testing.assert_close(mask, torch.full_like(mask, expected))

    def test_protection_and_monotone_objective(self):
        x = torch.ones(1, 2, 2, 6, dtype=torch.complex64)
        gs = torch.eye(2)
        mask, info = q.bounded_spectral_mask(x, x*.6, gs, 1, 8)
        self.assertEqual(float(mask[:, :, 0].abs().max()), 0)
        torch.testing.assert_close(mask[:, :, 1], torch.full_like(mask[:, :, 1], .6))
        errors = info["spectral_mse_trace"]
        self.assertTrue(all(b <= a+1e-12 for a, b in zip(errors, errors[1:])))
        self.assertGreater(errors[-1], 0)

    def test_bad_geometry_or_budget_refused(self):
        x, gs = torch.ones(1, 2, 2, 3, dtype=torch.complex64), torch.eye(2)
        for kill, iterations in ((True, 8), (-1, 8), (2, 8), (0, 97), (0, True)):
            with self.assertRaises(ValueError):
                q.bounded_spectral_mask(x, x, gs, kill, iterations)
        with self.assertRaises(ValueError):
            q.bounded_spectral_mask(x, x, -gs, 0, 8)
        with self.assertRaises(ValueError):
            q.bounded_spectral_mask(x, x[..., :1], gs, 0, 8)

    def test_phase_mismatch_remains_even_for_answer_mask(self):
        x = torch.ones(1, 2, 1, 3, dtype=torch.complex64)
        mask, info = q.bounded_spectral_mask(x, x*1j, torch.ones(1, 1), 0, 8)
        self.assertEqual(float(mask.abs().max()), 0)
        self.assertAlmostEqual(info["spectral_mse_trace"][-1], 1.)

    def test_positive_weight_cannot_reverse_single_loss_direction(self):
        gain = torch.tensor(1., requires_grad=True)
        loss = (gain*.8-.2).abs()
        original = torch.autograd.grad(loss, gain, retain_graph=True)[0]
        weighted = torch.autograd.grad(loss*2, gain)[0]
        self.assertEqual(float(weighted), float(original*2))
        self.assertGreater(float(original), 0)


if __name__ == "__main__":
    unittest.main()
