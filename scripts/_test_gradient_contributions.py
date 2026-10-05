"""Synthetic CPU-only contribution diagnostics; no data, training or CUDA."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("gradient_contribution_tests", Path(__file__).with_name("169_diagnose_gradient_contributions.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def metadata():
    return [{"domain": d, "role": "train" if i < 3 else "pseudo_label_train_candidate", "vocal_db": 0}
            for i, d in enumerate(s.m.DOMAINS)]


class ContributionTests(unittest.TestCase):
    def vectors(self):
        return {"true_vocal_base": torch.tensor([1., 0.]), "instrumental_base": torch.tensor([0., 1.]),
                "pseudo_base": torch.tensor([-1., 0.]), "true_auxiliary": torch.tensor([2., -1.])}

    def test_exact_slot_grouping(self):
        self.assertEqual([s.slot_group(i, m) for i, m in enumerate(metadata())],
                         ["true_vocal_base"]*2+["instrumental_base"]+["pseudo_base"]*3)

    def test_wrong_role_or_index_refused(self):
        for i, meta in ((0, {"domain": "musdb", "role": "development"}),
                        (2, {"domain": "pseudo", "role": "pseudo_label_train_candidate"}),
                        (6, metadata()[0]), (True, metadata()[0])):
            with self.assertRaises(ValueError):
                s.slot_group(i, meta)

    def test_negative_and_orthogonal_cosines(self):
        before = copy.deepcopy(self.vectors())
        result = s.geometry(before)
        self.assertEqual(result["pairs"]["true_vocal_base|pseudo_base"]["cosine"], -1.)
        self.assertEqual(result["pairs"]["true_vocal_base|instrumental_base"]["cosine"], 0.)
        self.assertTrue(all(torch.equal(before[n], x) for n, x in self.vectors().items()))

    def test_group_total_uses_original_six_slot_denominator(self):
        vectors = self.vectors()
        result = s.geometry(vectors)
        self.assertAlmostEqual(result["base_l2"], 1.)
        self.assertAlmostEqual(result["strengths"][1]["combined_l2"], float((torch.tensor([0., 1.])+0.2*vectors["true_auxiliary"]).double().norm()), places=7)

    def test_zero_norm_is_unavailable_not_conflict(self):
        vectors = {name: torch.zeros(2) for name in s.GROUPS}
        result = s.geometry(vectors)
        self.assertIsNone(result["pairs"]["true_vocal_base|pseudo_base"]["cosine"])
        self.assertIsNone(result["strengths"][0]["auxiliary_over_base_l2"])

    def test_float64_statistics_do_not_change_fp32_vectors(self):
        vector = torch.arange(10000, dtype=torch.float32)/10000
        result = s.geometry({name: vector.clone() for name in s.GROUPS})
        self.assertAlmostEqual(result["pairs"]["true_vocal_base|pseudo_base"]["cosine"], 1., places=12)
        self.assertEqual(result["metric_reduction_dtype"], "float64")
        self.assertEqual(vector.dtype, torch.float32)

    def test_nonfinite_wrong_dtype_shape_or_group_refused(self):
        for changed in (torch.ones(3), torch.ones(2).double(), torch.tensor([float("nan"), 0.])):
            vectors = self.vectors(); vectors["pseudo_base"] = changed
            with self.assertRaises(ValueError):
                s.geometry(vectors)
        with self.assertRaises(ValueError):
            s.geometry({"true_vocal_base": torch.ones(2)})

    def test_changed_fixed_scope_refused(self):
        plan = s.fixed_scope()
        s.check_plan(plan)
        for name, value in (("cuda_used", True), ("model_updates", 1), ("training_authorized", True),
                            ("cursors", list(range(2001, 2013))), ("models", list(reversed(s.MODELS))),
                            ("metric_reduction_dtype", "float32"), ("slot_weights", [1, 1, 1, 2, 2, 2])):
            changed = copy.deepcopy(plan); changed[name] = value
            with self.assertRaises(ValueError):
                s.check_plan(changed)

    def test_bool_budget_and_float_slot_weights_refused(self):
        for name, value in (("model_updates", False), ("cursors", [True]+list(s.CURSORS[1:])),
                            ("slot_weights", [1.]*6)):
            changed = s.fixed_scope(); changed[name] = value
            with self.assertRaises(ValueError):
                s.check_plan(changed)

    def test_process_check_refuses_active_trainer(self):
        with patch.object(s.subprocess, "run") as process:
            process.return_value.stdout = "57996\n"
            with self.assertRaises(ValueError):
                s.no_active_trainer()

    def test_parameter_probe_independent_combination_no_update_or_grad_storage(self):
        net = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            net.weight.copy_(torch.tensor([[.3, .2], [.1, .4]]))
        before = copy.deepcopy(net.state_dict())
        batch = {"x": torch.ones(6, 2), "v": torch.zeros(6, 2), "metadata": metadata(), "diagnostic_counter": 2000}
        def losses(net, x, v, wa, gs, meta):
            y = net(x)
            active = meta["domain"] in ("musdb", "mir1k")
            return y.square().mean(), (y-.2).square().mean() if active else y.sum()*0, {"active": active}, None
        with patch.object(s.s, "slot_losses", losses), patch.object(torch.optim, "Adam", side_effect=AssertionError("No optimizer")):
            result = s.parameter_probe(net, batch, None, None, .2, independent=True)
        self.assertEqual(result["active_count"], 2)
        self.assertEqual(result["skipped_count"], 4)
        self.assertLess(result["independent_combination_max_error"], 1e-6)
        self.assertTrue(all(torch.equal(before[n], p) for n, p in net.state_dict().items()))
        self.assertFalse(any(p.grad is not None for p in net.parameters()))

    def test_auxiliary_cannot_leak_into_instrumental_or_pseudo(self):
        net = torch.nn.Linear(2, 2, bias=False)
        batch = {"x": torch.ones(6, 2), "v": torch.zeros(6, 2), "metadata": metadata(), "diagnostic_counter": 2000}
        def bad(net, x, v, *args):
            y = net(x)
            return y.square().mean(), y.square().mean(), {"active": True}, None
        with patch.object(s.s, "slot_losses", bad):
            with self.assertRaises(ValueError):
                s.parameter_probe(net, batch, None, None, .2)

    def test_existing_grad_storage_refused(self):
        net = torch.nn.Linear(2, 2)
        net.weight.grad = torch.ones_like(net.weight)
        with self.assertRaises(ValueError):
            s.parameter_probe(net, {"metadata": metadata()}, None, None, .02)

    def test_sealed_geometry_arithmetic_refuses_tampering(self):
        result = s.geometry(self.vectors())
        s.verify_geometry(result)
        for field in ("base_l2", "base_auxiliary_dot"):
            changed = copy.deepcopy(result); changed[field] += 1.
            with self.assertRaises(ValueError):
                s.verify_geometry(changed)
        changed = copy.deepcopy(result); changed["strengths"][1]["auxiliary_over_base_l2"] += 1.
        with self.assertRaises(ValueError):
            s.verify_geometry(changed)

    def test_zero_geometry_independently_verifies(self):
        s.verify_geometry(s.geometry({name: torch.zeros(2) for name in s.GROUPS}))

    def test_unapproved_strength_refused(self):
        net = torch.nn.Linear(2, 2)
        for value in (.1, 1, True):
            with self.assertRaises(ValueError):
                s.parameter_probe(net, {"metadata": metadata()}, None, None, value)

    def test_graph_mutation_is_refused(self):
        net = torch.nn.Linear(2, 2, bias=False)
        batch = {"x": torch.ones(6, 2), "v": torch.zeros(6, 2), "metadata": metadata(), "diagnostic_counter": 2000}
        def mutable(net, x, v, wa, gs, meta):
            net.training = not net.training if meta["domain"] == "instrumental" else net.training
            y = net(x)
            return y.square().mean(), y.sum()*0, {"active": False}, None
        with patch.object(s.s, "slot_losses", mutable):
            with self.assertRaises(ValueError):
                s.parameter_probe(net, batch, None, None, .02)


if __name__ == "__main__":
    unittest.main()
