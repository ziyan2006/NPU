"""Weak-weight budget, clipping, silence, context and gradient contracts."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("weak_loss_under_test", Path(__file__).with_name("122_train_weak_vocal_objective.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
attribution_spec = importlib.util.spec_from_file_location("weak_attribution_under_test", Path(__file__).with_name("124_diagnose_weak_objective.py"))
attribution = importlib.util.module_from_spec(attribution_spec)
attribution_spec.loader.exec_module(attribution)
torch.set_num_threads(4)


class WeakObjectiveTests(unittest.TestCase):
    def test_weak_vocals_get_more_weight_without_global_loss_increase(self):
        x = torch.ones(6, 2, 89856)
        v = x*torch.tensor([.5, .05, .3, .03, .7, 0.])[:, None, None]
        weights, stats = module.vocal_weights(x, v, 96)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertGreater(float(weights[3]), float(weights[2]))
        self.assertAlmostEqual(float(weights.mean()), 1., places=6)
        self.assertEqual(float(weights[-1]), 1.)
        self.assertFalse(stats["active"][-1])
        self.assertLessEqual(float(weights.max()), 6.)

    def test_pure_instrumental_and_near_silence_fall_back_to_unit_weights(self):
        x = torch.ones(6, 2, 89856)*.1
        for v in (torch.zeros_like(x), torch.ones_like(x)*1e-8):
            w, _ = module.vocal_weights(x, v, 96)
            self.assertTrue(torch.equal(w, torch.ones(6)))
        w, _ = module.vocal_weights(torch.zeros_like(x), torch.zeros_like(x), 96)
        self.assertTrue(torch.isfinite(w).all())
        self.assertTrue(torch.equal(w, torch.ones(6)))

    def test_low_energy_floor_caps_relative_pre_normalization_weight(self):
        x = torch.ones(3, 2, 89856)
        v = x*torch.tensor([.01, .0011, 0.])[:, None, None]
        w, _ = module.vocal_weights(x, v, 96)
        self.assertTrue(torch.equal(w, torch.ones(3)))

    def test_context_and_stft_edges_do_not_change_weights_or_loss(self):
        x = torch.ones(2, 2, 89856)
        v = x*.2
        before, _ = module.vocal_weights(x, v, 96)
        v[..., :25088] = 100.
        v[..., 89344:] = 100.
        after, _ = module.vocal_weights(x, v, 96)
        self.assertTrue(torch.equal(before, after))
        pred = torch.ones_like(x)*.2
        self.assertEqual(float(module.per_example_wave_error(pred, x, v, 96).sum()), 0.)

    def test_uniform_weight_is_the_old_objective_with_identical_gradients(self):
        torch.manual_seed(8)
        x = torch.randn(6, 2, 89856)*.01
        v = torch.randn_like(x)*.002
        wa = torch.from_numpy(module.core.t09.make_analysis_matrix())
        gs = torch.from_numpy(module.core.t09.make_synthesis_matrix())
        old = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
        new = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
        new.load_state_dict(old.state_dict())
        a = module.control.backward_batch(old, x, v, wa, gs, "cpu", 96, 44, 1., 1)
        b = module.backward_weighted_batch(new, x, v, wa, gs, "cpu", 96, 44, torch.ones(6), 1)
        self.assertEqual(a, b)
        for p, q in zip(old.parameters(), new.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))

    def test_full_batch_weights_preserve_microbatch_gradient_and_parameter_update(self):
        torch.manual_seed(19)
        x = torch.randn(6, 2, 89856)*.1
        v = x*torch.tensor([.5, .05, .3, .03, .7, 0.])[:, None, None]
        weights, _ = module.vocal_weights(x, v, 96)
        wa = torch.from_numpy(module.core.t09.make_analysis_matrix())
        gs = torch.from_numpy(module.core.t09.make_synthesis_matrix())
        full = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
        accum = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
        accum.load_state_dict(full.state_dict())
        opt_full = torch.optim.Adam(full.parameters(), lr=5e-6)
        opt_accum = torch.optim.Adam(accum.parameters(), lr=5e-6)
        a = module.backward_weighted_batch(full, x, v, wa, gs, "cpu", 96, 44, weights, 6)
        b = module.backward_weighted_batch(accum, x, v, wa, gs, "cpu", 96, 44, weights, 1)
        for l, r in zip(a, b):
            self.assertAlmostEqual(l, r, places=6)
        for p, q in zip(full.parameters(), accum.parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, atol=2e-6, rtol=2e-5))
        for net in (full, accum):
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt_full.step(); opt_accum.step()
        for p, q in zip(full.parameters(), accum.parameters()):
            self.assertTrue(torch.allclose(p, q, atol=1e-7, rtol=1e-6))

    def test_invalid_weights_and_batch_dependent_graph_are_rejected(self):
        x = torch.ones(2, 2, 89856)
        net = torch.nn.Conv2d(2, 4, 1)
        for weights in (torch.ones(3), torch.tensor([1., -1.]), torch.tensor([1., float("nan")])):
            with self.assertRaises(ValueError):
                module.backward_weighted_batch(net, x, x, None, None, "cpu", 96, 44, weights)
        net = torch.nn.Sequential(net, torch.nn.BatchNorm2d(4))
        with self.assertRaises(ValueError):
            module.backward_weighted_batch(net, x, x, None, None, "cpu", 96, 44, torch.ones(2))

    def test_no_scoring_interval_fails(self):
        with self.assertRaises(ValueError):
            module.vocal_weights(torch.ones(1, 2, 25000), torch.zeros(1, 2, 25000), 96)

    def test_joint_error_energy_closes_including_correlated_source_cross_term(self):
        t = torch.arange(4096, dtype=torch.float64)
        a = torch.sin(2*torch.pi*13*t/4096).repeat(2, 1)
        v = (torch.sin(2*torch.pi*37*t/4096)*.1+a*.02)
        other = torch.sin(2*torch.pi*89*t/4096).repeat(2, 1)*.01
        mix = a+v
        output_accompaniment = .9*a+.6*v+other
        result = attribution.decompose(mix-output_accompaniment, mix, v)
        self.assertAlmostEqual(result["joint_accompaniment_gain"], .9, places=9)
        self.assertAlmostEqual(result["joint_remaining_vocal_gain"], .6, places=9)
        self.assertAlmostEqual(sum(result["error_energy_fractions"].values()), 1., places=9)
        self.assertLess(result["relative_closure_error"], 1e-9)
        self.assertLess(result["error_energy_fractions"]["correlated_source_cross_term"], 0.)
        self.assertIsNone(attribution.decompose(a*.05, a*1.1, a*.1))

    def test_uniform_vocal_energy_normalization_cannot_change_single_crop_gradient_direction(self):
        x = torch.ones(1, 2, 89856)
        v = x*.1
        scale = torch.ones((), requires_grad=True)
        p = x*.2*scale
        old_loss = module.per_example_wave_error(p, x, v, 96).mean()
        old_grad = torch.autograd.grad(old_loss, scale, retain_graph=True)[0]
        new_grad = torch.autograd.grad(old_loss*10., scale)[0]
        self.assertGreater(float(old_grad), 0.)
        self.assertAlmostEqual(float(new_grad), float(old_grad)*10., places=6)


if __name__ == "__main__":
    unittest.main()
