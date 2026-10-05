"""CPU synthetic prototype tests; not formal training or GPU proof."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location("auxiliary_tests", Path(__file__).with_name("159_source_projection_auxiliary.py"))
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(77)
        self.a = torch.randn(2, 2048, generator=generator)*.03
        self.v = torch.randn(2, 2048, generator=generator)*.01+self.a*.1
        self.x = self.a+self.v
        self.meta = {"domain": "musdb", "role": "train"}

    def test_perfect_reference_zero(self):
        value, info = k.source_projection_auxiliary(self.v, self.x, self.v, self.meta)
        self.assertLess(float(value), 1e-10)
        self.assertAlmostEqual(info["accompaniment_gain"], 1., places=5)
        self.assertAlmostEqual(info["remaining_vocal_gain"], 0., places=5)

    def test_bypass_and_removing_entire_mix_both_penalized(self):
        for predicted in (torch.zeros_like(self.x), self.x):
            value, _ = k.source_projection_auxiliary(predicted, self.x, self.v, self.meta)
            self.assertAlmostEqual(float(value), 1., places=5)

    def test_common_gain_invariance(self):
        prediction = self.v*.3+self.a*.05
        expected, _ = k.source_projection_auxiliary(prediction, self.x, self.v, self.meta)
        for gain in (.1, 2.):
            actual, _ = k.source_projection_auxiliary(prediction*gain, self.x*gain, self.v*gain, self.meta)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_local_auxiliary_direction_increases_underestimated_vocal(self):
        gain = torch.tensor(1., requires_grad=True)
        predicted = gain*(self.v*.2+self.a*.05)
        value, info = k.source_projection_auxiliary(predicted, self.x, self.v, self.meta)
        direction = torch.autograd.grad(value, gain)[0]
        self.assertTrue(info["active"])
        self.assertLess(float(direction), 0)

    def test_overestimated_vocal_and_accompaniment_damage_penalized(self):
        value, info = k.source_projection_auxiliary(self.v*1.3+self.a*.2, self.x, self.v, self.meta)
        self.assertAlmostEqual(float(value), .3**2+.2**2, places=5)
        self.assertAlmostEqual(info["remaining_vocal_gain"], -.3, places=5)
        self.assertAlmostEqual(info["accompaniment_gain"], .8, places=5)

    def test_silent_and_collinear_sources_skip_with_zero_gradient(self):
        for vocal in (torch.zeros_like(self.x), self.x*.3):
            prediction = self.v.clone().requires_grad_()
            value, info = k.source_projection_auxiliary(prediction, self.x, vocal, self.meta)
            self.assertFalse(info["active"])
            self.assertEqual(float(value.detach()), 0)
            self.assertEqual(float(torch.autograd.grad(value, prediction)[0].abs().max()), 0)

    def test_pseudo_and_instrumental_never_get_auxiliary(self):
        for meta in ({"domain": "pseudo", "role": "pseudo_label_train_candidate"},
                     {"domain": "instrumental", "role": "train"}):
            value, info = k.source_projection_auxiliary(self.v, self.x, self.v, meta)
            self.assertFalse(info["active"])
            self.assertEqual(float(value), 0)

    def test_development_mismatch_and_nonfp32_refused(self):
        for meta in ({"domain": "musdb", "role": "development"}, {"domain": "pseudo", "role": "train"}):
            with self.assertRaises(ValueError):
                k.source_projection_auxiliary(self.v, self.x, self.v, meta)
        with self.assertRaises(ValueError):
            k.source_projection_auxiliary(self.v.double(), self.x.double(), self.v.double(), self.meta)


if __name__ == "__main__":
    unittest.main()
