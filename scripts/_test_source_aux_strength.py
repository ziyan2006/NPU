"""Synthetic CPU diagnostics tests. No teacher data, model updates or CUDA."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("strength_tests", Path(__file__).with_name("164_diagnose_source_aux_strength.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class StrengthTests(unittest.TestCase):
    def test_fixed_grid_and_local_directions(self):
        rows = s.strength_rows(.2, .5, .05, -.4)
        self.assertEqual([r["lambda"] for r in rows], [0., .02, .1, .2])
        self.assertEqual(rows[0]["local_gain_descent_direction"], "less_vocal")
        self.assertEqual(rows[3]["local_gain_descent_direction"], "more_vocal")
        self.assertAlmostEqual(rows[3]["loss"], .3)

    def test_inactive_aux_keeps_base(self):
        for r in s.strength_rows(.3, 0., -.01, 0.):
            self.assertEqual(r["loss"], .3)
            self.assertEqual(r["d_combined_d_mask_gain"], -.01)

    def test_zero_base_fraction_unavailable(self):
        self.assertIsNone(s.strength_rows(0., .4, 0., -.1)[2]["contribution_over_base"])

    def test_nonfinite_negative_and_wrong_type_refused(self):
        for args in ((float("nan"), .5, 0., 0.), (.1, -.5, 0., 0.), (1, .5, 0., 0.), (.1, .5, float("inf"), 0.)):
            with self.assertRaises(ValueError):
                s.strength_rows(*args)

    def test_gradient_metrics_match_linear_combination(self):
        b, a = torch.tensor([1., 2.]), torch.tensor([-2., 1.])
        before = b.clone(), a.clone()
        metrics = s.gradient_summary(b, a)
        self.assertAlmostEqual(metrics["base_auxiliary_cosine"], 0.)
        self.assertAlmostEqual(metrics["strengths"][3]["change_over_base_l2"], .2)
        self.assertAlmostEqual(metrics["strengths"][3]["combined_l2"], float((b+.2*a).norm()))
        self.assertTrue(torch.equal(before[0], b) and torch.equal(before[1], a))

    def test_zero_aux_gradient_is_reported_not_quality(self):
        metrics = s.gradient_summary(torch.ones(3), torch.zeros(3))
        self.assertIsNone(metrics["base_auxiliary_cosine"])
        self.assertEqual(metrics["strengths"][2]["change_over_base_l2"], 0.)

    def test_bad_gradients_refused(self):
        for b, a in ((torch.zeros(3), torch.ones(3)), (torch.ones(3), torch.ones(2)),
                     (torch.ones(3).double(), torch.ones(3)), (torch.ones(3), torch.tensor([1., float("inf"), 2.]))):
            with self.assertRaises(ValueError):
                s.gradient_summary(b, a)

    def test_flatten_unused_is_finite_and_unaliased(self):
        p, q = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))
        g = torch.tensor([1., 2.])
        vector = s.flatten_gradients([g, None], [p, q])
        self.assertTrue(torch.equal(vector, torch.tensor([1., 2., 0., 0., 0.])))
        vector[0] = 9.
        self.assertEqual(float(g[0]), 1.)
        self.assertIsNone(p.grad)

    def test_diagnostic_counter_no_resume_or_limit_bypass(self):
        for counter in (1999, 2012, True):
            with self.assertRaises(ValueError):
                s.collect(None, None, counter)

    def test_development_auxiliary_refused(self):
        v = torch.ones(2, 16)
        with self.assertRaises(ValueError):
            s.t.k.source_projection_auxiliary(v, v, v, {"domain": "musdb", "role": "development"})

    def test_cpu_process_check_refuses_active_worker(self):
        with patch.object(s.subprocess, "run") as mocked:
            mocked.return_value.stdout = "54752\n"
            with self.assertRaises(ValueError):
                s.no_active_trainer()

    def test_scope_tampering_refused(self):
        doc = {"purpose": s.PURPOSE, "diagnostic_only": True, "training_authorized": False,
            "cuda_used": False, "model_updates": 0, "cursors": list(s.CURSORS), "lambda_grid": list(s.GRID),
            "source_arm": s.t.ARMS[1], "slot_weights": [1]*6, "teacher": "kim_melband",
            "selection": "Fixed TRAIN counters2000..2011; first eligible batch for full gradients",
            "release_selection": "NONE", "deployment": False}
        s.check_plan(doc)
        for field, value in (("training_authorized", True), ("cuda_used", True), ("model_updates", 1),
                             ("cursors", list(range(2001, 2013))), ("lambda_grid", [0., .02, .1, .4]),
                             ("source_arm", s.t.ARMS[0])):
            changed = copy.deepcopy(doc)
            changed[field] = value
            with self.assertRaises(ValueError):
                s.check_plan(changed)

    def test_parameter_probe_independent_combination_without_mutation(self):
        net = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            net.weight.copy_(torch.tensor([[.3, .2], [.1, .4]]))
        before = copy.deepcopy(net.state_dict())
        batch = {"x": torch.ones(6, 2), "v": torch.zeros(6, 2), "metadata": [{}]*6, "diagnostic_counter": 2000}
        def losses(net, x, v, *args):
            y = net(x)
            return (y-v).square().mean(), (y-.2).square().mean(), None, None
        with patch.object(s, "slot_losses", losses):
            metrics = s.parameter_probe(net, batch, None, None)
        self.assertGreater(metrics["auxiliary_l2"], 0.)
        self.assertFalse(metrics["gradients_populated_on_model"])
        self.assertTrue(all(torch.equal(before[n], p) for n, p in net.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
