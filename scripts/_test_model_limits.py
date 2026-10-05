"""Synthetic regression tests for the offline attribution metrics."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("model_limits", Path(__file__).with_name("109_audit_model_limits.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ModelLimitsTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(12)
        self.v = torch.randn(2, 4096, generator=generator)
        self.a = torch.randn(2, 4096, generator=generator)
        self.mix = self.a + self.v

    def test_exact_separation(self):
        row = audit.waveform_metrics(self.v, self.mix, self.v)
        self.assertGreater(row["accompaniment_error_snr_db"], 110)
        self.assertAlmostEqual(row["joint_fit_accompaniment_gain"], 1, places=6)
        self.assertAlmostEqual(row["joint_fit_residual_vocal_gain"], 0, places=6)

    def test_bypass_keeps_all_vocals(self):
        row = audit.waveform_metrics(torch.zeros_like(self.v), self.mix, self.v)
        self.assertIsNone(row["vocal_si_sdr_db"])
        self.assertAlmostEqual(row["joint_fit_residual_vocal_gain"], 1, places=6)
        self.assertAlmostEqual(row["joint_fit_accompaniment_gain"], 1, places=6)

    def test_partial_removal(self):
        row = audit.waveform_metrics(0.75 * self.v + 0.1 * self.a, self.mix, self.v)
        self.assertAlmostEqual(row["joint_fit_residual_vocal_gain"], 0.25, places=6)
        self.assertAlmostEqual(row["joint_fit_accompaniment_gain"], 0.9, places=6)

    def test_instrumental_not_scored_as_vocals(self):
        row = audit.waveform_metrics(0.1 * self.a, self.a, torch.zeros_like(self.a))
        self.assertIsNone(row["vocal_si_sdr_db"])
        self.assertIsNone(row["joint_fit_residual_vocal_gain"])
        self.assertAlmostEqual(row["accompaniment_error_snr_db"], 20, places=4)

    def test_dead_bands_do_not_inflate_zero_fraction(self):
        wa = torch.tensor([[0., 1., 1.]])
        bands = torch.tensor([[[0., 0.], [0.01, 0.01], [1., 1.]]])
        row = audit.mask_input_stats(bands, wa, 0.1)
        self.assertEqual(row["zero_rounding_pct_active_bands"], 50)
        self.assertEqual(row["saturation_pct_active_bands"], 0)

    def test_frequency_energy_preserves_imaginary_part(self):
        spectrum = torch.tensor([complex(0, 2), complex(3, 4)])
        self.assertEqual(float(spectrum.abs().double().square().sum()), 29)

    def test_multitrack_common_gain_preserves_true_sum_and_tail(self):
        v = torch.tensor([[0.2, 0.2]])
        a = torch.tensor([[0.3, 0.3, 0.4]])
        mix, vocal = audit.mix_multitrack([v], [a])
        self.assertEqual(mix.shape, (2, 3))
        self.assertAlmostEqual(float(mix.abs().max()), 0.95, places=6)
        self.assertEqual(float(vocal[:, -1].abs().max()), 0)
        self.assertTrue(torch.allclose(mix - vocal, a.expand(2, -1) * 1.9))


if __name__ == "__main__":
    unittest.main()
