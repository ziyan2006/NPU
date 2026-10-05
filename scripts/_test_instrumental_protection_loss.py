"""Synthetic FP32 loss-composition tests, not approved training or GPU proof."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location("protection_kernel", Path(__file__).with_name("170_instrumental_protection_loss.py"))
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


class ProtectionLossTests(unittest.TestCase):
    def setUp(self):
        self.domains = ("musdb", "mir1k", "instrumental", "pseudo", "pseudo", "pseudo")

    def meta(self, index):
        return {"domain": self.domains[index], "role": "train" if index < 3 else "pseudo_label_train_candidate"}

    def compose(self, index, weight, base=None, auxiliary=None, active=None):
        base = torch.tensor(2., requires_grad=True) if base is None else base
        auxiliary = torch.tensor(.5 if index < 2 else 0., requires_grad=True) if auxiliary is None else auxiliary
        active = index < 2 if active is None else active
        return k.combine_slot_loss(base, auxiliary, self.meta(index), index, weight, {"active": active})

    def test_uniform_exactly_matches_existing_formula(self):
        for index in range(6):
            base, auxiliary = torch.tensor(2.), torch.tensor(.5 if index < 2 else 0.)
            loss, parts = self.compose(index, 1, base, auxiliary)
            torch.testing.assert_close(loss, (base+.2*auxiliary)*(1/6), rtol=0, atol=0)
            self.assertEqual(parts["normalizer"], 6)

    def test_only_instrumental_base_changes(self):
        for index in range(6):
            control, _ = self.compose(index, 1)
            candidate, parts = self.compose(index, 4)
            if index == 2:
                self.assertAlmostEqual(float(candidate.detach()), 4*float(control.detach()), places=6)
                self.assertEqual(parts["base_weight"], 4)
            else:
                torch.testing.assert_close(control, candidate, rtol=0, atol=0)
                self.assertEqual(parts["base_weight"], 1)

    def test_uniform_fp32_value_and_gradient_operation_order(self):
        generator = torch.Generator().manual_seed(404)
        for _ in range(30):
            left = torch.rand((), generator=generator).requires_grad_()
            right = torch.rand((), generator=generator).requires_grad_()
            actual, _ = self.compose(0, 1, left, right)
            expected = (left+.2*right)*(1/6)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(torch.autograd.grad(actual, (left, right)), torch.autograd.grad(expected, (left, right))):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_instrumental_gradient_fourfold(self):
        gradients = []
        for weight in (1, 4):
            base = torch.tensor(2., requires_grad=True)
            loss, _ = self.compose(2, weight, base)
            gradients.append(torch.autograd.grad(loss, base)[0])
        torch.testing.assert_close(gradients[1], 4*gradients[0], rtol=0, atol=0)

    def test_true_and_pseudo_gradients_identical_across_arms(self):
        for index in (0, 1, 3, 4, 5):
            values = []
            for weight in (1, 4):
                base = torch.tensor(2., requires_grad=True)
                auxiliary = torch.tensor(.5 if index < 2 else 0., requires_grad=True)
                loss, _ = self.compose(index, weight, base, auxiliary)
                values.append(torch.autograd.grad(loss, (base, auxiliary)))
            for left, right in zip(values[0], values[1]):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_denominator_stays_six_not_sum_of_weights(self):
        candidate = sum(self.compose(index, 4)[0] for index in range(6))
        expected = (2*9+.2*.5*2)/6
        self.assertAlmostEqual(float(candidate.detach()), expected, places=6)
        self.assertNotAlmostEqual(float(candidate.detach()), (2*9+.2*.5*2)/9, places=3)

    def test_skipped_true_slot_keeps_original_base(self):
        loss, parts = self.compose(0, 4, auxiliary=torch.tensor(0.), active=False)
        self.assertAlmostEqual(float(loss.detach()), 2/6, places=6)
        self.assertEqual(parts["auxiliary_contribution"], 0.)

    def test_auxiliary_leak_or_nonzero_skipped_refused(self):
        for index, active, value in ((2, True, .1), (3, True, 0.), (0, False, .1), (2, False, .1)):
            with self.assertRaises(ValueError):
                self.compose(index, 4, auxiliary=torch.tensor(value), active=active)

    def test_nonboolean_activity_refused(self):
        for active in (1, 0, "true", None):
            with self.assertRaises(ValueError):
                k.combine_slot_loss(torch.tensor(2.), torch.tensor(0.), self.meta(2), 2, 4, {"active": active})

    def test_unplanned_weight_or_bool_refused(self):
        for weight in (True, False, 1., 4., 0, 2, 10, float("nan")):
            with self.assertRaises(ValueError):
                self.compose(2, weight)

    def test_index_roles_and_development_refused(self):
        for index, meta in ((True, self.meta(1)), (-1, self.meta(0)), (6, self.meta(0)),
                            (0, self.meta(2)), (3, {"domain": "pseudo", "role": "train"}),
                            (1, {"domain": "mir1k", "role": "development"})):
            with self.assertRaises(ValueError):
                k.combine_slot_loss(torch.tensor(2.), torch.tensor(0.), meta, index, 4, {"active": False})

    def test_invalid_dtype_shape_and_nonfinite_refused(self):
        for base, auxiliary in ((torch.tensor(2., dtype=torch.float64), torch.tensor(0.)),
                                (torch.tensor([2.]), torch.tensor(0.)),
                                (torch.tensor(2.), torch.tensor([0.])),
                                (torch.tensor(float("nan")), torch.tensor(0.)),
                                (torch.tensor(2.), torch.tensor(float("inf")))):
            with self.assertRaises(ValueError):
                self.compose(2, 4, base, auxiliary)

    def test_parts_account_for_same_value(self):
        for index in range(6):
            loss, parts = self.compose(index, 4)
            self.assertAlmostEqual(float(loss.detach()), parts["weighted_base_loss"]+parts["auxiliary_contribution"], places=6)
            self.assertEqual(parts["coefficient"], .2)


if __name__ == "__main__":
    unittest.main()
