"""Synthetic CPU-only tests; no data/model update/CUDA/optimizer."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("auxiliary_component_tests", Path(__file__).with_name("175_diagnose_auxiliary_components.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def metadata():
    return [{"domain": domain, "role": "train" if i < 3 else "pseudo_label_train_candidate", "vocal_db": 0}
            for i, domain in enumerate(s.m.DOMAINS)]


def sources():
    a = torch.tensor([[1., -1., 0., 0.], [1., -1., 0., 0.]])
    v = torch.tensor([[0., 0., 1., -1.], [0., 0., 1., -1.]])
    return a+v, v


class ComponentTests(unittest.TestCase):
    def test_active_split_bit_exact_original_scalar_and_gradients(self):
        mix, vocal = sources()
        predicted = (.8*vocal+.1*(mix-vocal)).requires_grad_()
        residual, accompaniment, original, info = s.split_auxiliary(predicted, mix, vocal, metadata()[0])
        self.assertTrue(info["active"])
        self.assertTrue(torch.equal(residual+accompaniment, original))
        combined_grad = torch.autograd.grad(residual+accompaniment, predicted, retain_graph=True)[0]
        original_grad = torch.autograd.grad(original, predicted)[0]
        self.assertTrue(torch.equal(combined_grad, original_grad))
        self.assertGreater(float(residual.detach()), 0.)
        self.assertGreater(float(accompaniment.detach()), 0.)

    def test_perfect_prediction_zero_components(self):
        mix, vocal = sources()
        residual, accompaniment, _, _ = s.split_auxiliary(vocal.clone().requires_grad_(), mix, vocal, metadata()[0])
        self.assertLess(float(residual.detach()), 1e-12)
        self.assertLess(float(accompaniment.detach()), 1e-12)

    def test_instrumental_and_pseudo_skip(self):
        mix, vocal = sources()
        for meta in metadata()[2:]:
            pieces = s.split_auxiliary(vocal.clone().requires_grad_(), mix, vocal, meta)
            self.assertFalse(pieces[3]["active"])
            self.assertTrue(all(float(value.detach()) == 0. for value in pieces[:3]))

    def test_inactive_or_correlated_sources_skip(self):
        mix, vocal = sources()
        for x, v in ((mix, torch.zeros_like(vocal)), (mix, mix/2)):
            pieces = s.split_auxiliary(v.clone().requires_grad_(), x, v, metadata()[0])
            self.assertFalse(pieces[3]["active"])

    def test_development_acceptance_or_role_mismatch_refused(self):
        mix, vocal = sources()
        for meta in ({"domain": "musdb", "role": "development"}, {"domain": "musdb", "role": "acceptance"},
                     {"domain": "pseudo", "role": "train"}):
            with self.assertRaises(ValueError):
                s.split_auxiliary(vocal, mix, vocal, meta)

    def test_wrong_shape_precision_or_nonfinite_refused(self):
        mix, vocal = sources()
        for value in (vocal.double(), vocal[:, :2], vocal*float("nan")):
            with self.assertRaises(ValueError):
                s.split_auxiliary(value, mix, vocal, metadata()[0])

    def vectors(self):
        return {key: torch.tensor([float(i+1), float(-i)]) for i, key in enumerate(s.GROUPS)}

    def test_geometry_weights_and_coefficients(self):
        values = self.vectors()
        for weight in (1, 4):
            result = s.geometry(values, weight)
            s.verify_geometry(result)
            expected = values[s.GROUPS[0]]+weight*values[s.GROUPS[1]]+values[s.GROUPS[2]]+.2*(values[s.GROUPS[3]]+values[s.GROUPS[4]])
            self.assertAlmostEqual(result["combined_l2"], float(expected.double().norm()), places=6)
            self.assertEqual(result["combined_coefficients"], [1., float(weight), 1., .2, .2])

    def test_zero_gradient_not_conflict(self):
        doc = s.geometry({key: torch.zeros(2) for key in s.GROUPS}, 4)
        s.verify_geometry(doc)
        self.assertTrue(all(pair["cosine"] is None for pair in doc["pairs"].values()))

    def test_float64_geometry_does_not_change_fp32_gradient(self):
        vector = torch.arange(10000, dtype=torch.float32)/10000
        before = vector.clone()
        doc = s.geometry({key: vector for key in s.GROUPS}, 1)
        self.assertAlmostEqual(doc["pairs"][f"{s.GROUPS[3]}|{s.GROUPS[4]}"]["cosine"], 1., places=12)
        self.assertTrue(torch.equal(before, vector))
        self.assertEqual(vector.dtype, torch.float32)

    def test_wrong_geometry_inputs_refused(self):
        for value in (torch.ones(3), torch.ones(2).double(), torch.tensor([float("nan"), 1.])):
            changed = self.vectors(); changed[s.GROUPS[0]] = value
            with self.assertRaises(ValueError):
                s.geometry(changed, 1)
        for weight in (True, 1., 2):
            with self.assertRaises(ValueError):
                s.geometry(self.vectors(), weight)

    def test_geometry_tampering_refused(self):
        doc = s.geometry(self.vectors(), 1)
        for key, value in (("combined_l2", doc["combined_l2"]+1), ("combined_coefficients", [1.]*5)):
            changed = copy.deepcopy(doc); changed[key] = value
            with self.assertRaises(ValueError):
                s.verify_geometry(changed)
        changed = copy.deepcopy(doc); changed["pairs"].pop(next(iter(changed["pairs"])))
        with self.assertRaises(ValueError):
            s.verify_geometry(changed)

    def test_fixed_scope_cannot_authorize_training_or_change_input(self):
        s.check_plan(s.fixed_scope())
        for key, value in (("cuda_used", True), ("model_updates", 1), ("training_authorized", True),
                           ("cursors", list(range(3000, 3012))), ("normalizer", 9), ("models", list(reversed(s.MODELS))),
                           ("instrumental_weights", dict(zip(s.MODELS, (1, 1, 2)))), ("auxiliary_lambda", .02)):
            changed = s.fixed_scope(); changed[key] = value
            with self.assertRaises(ValueError):
                s.check_plan(changed)

    def test_bool_integer_fields_refused(self):
        for key, value in (("model_updates", False), ("cursors", [True]+list(s.CURSORS[1:])),
                           ("instrumental_weights", dict(zip(s.MODELS, (True, 1, 4))))):
            changed = s.fixed_scope(); changed[key] = value
            with self.assertRaises(ValueError):
                s.check_plan(changed)

    def test_slot_roles_and_order(self):
        self.assertEqual([s.group_for_slot(i, meta) for i, meta in enumerate(metadata())], [s.GROUPS[0]]*2+[s.GROUPS[1]]+[s.GROUPS[2]]*3)
        for index, meta in ((True, metadata()[0]), (6, metadata()[0]), (0, {"domain": "musdb", "role": "development"})):
            with self.assertRaises(ValueError):
                s.group_for_slot(index, meta)

    def test_active_process_preserved_and_refused(self):
        with patch.object(s.subprocess, "run") as process:
            process.return_value.stdout = "123456789\n"
            with self.assertRaises(ValueError):
                s.no_active_worker()

    def probe(self, weight):
        net = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            net.weight.copy_(torch.tensor([[.3, .2], [.1, .4]]))
        batch = {"x": torch.ones(6, 2), "v": torch.zeros(6, 2), "metadata": metadata()}
        def losses(net, x, v, wa, gs, meta):
            y = net(x)
            active = meta["domain"] in ("musdb", "mir1k")
            residual = (y-.2).square().mean() if active else y.sum()*0
            accompaniment = (y-.4).square().mean() if active else y.sum()*0
            return y.square().mean(), residual, accompaniment, residual+accompaniment, {"active": active}
        return net, batch, losses

    def test_parameter_probe_real_composition_no_optimizer_or_model_updates(self):
        for weight in (1, 4):
            net, batch, losses = self.probe(weight)
            before = copy.deepcopy(net.state_dict())
            with patch.object(s, "slot_losses", losses), patch.object(torch.optim, "Adam", side_effect=AssertionError("No optimizer")):
                result = s.parameter_probe(net, batch, None, None, weight, independent=True)
            self.assertEqual(result["active_count"], 2)
            self.assertEqual(result["skipped_count"], 4)
            self.assertLess(result["independent_combination"]["max_error"], 1e-6)
            self.assertTrue(all(torch.equal(value, before[key]) for key, value in net.state_dict().items()))
            self.assertFalse(any(param.grad is not None for param in net.parameters()))

    def test_existing_gradient_storage_refused(self):
        net, batch, _ = self.probe(1)
        net.weight.grad = torch.ones_like(net.weight)
        with self.assertRaises(ValueError):
            s.parameter_probe(net, batch, None, None, 1)

    def test_split_gradient_mismatch_refused(self):
        with self.assertRaises(AssertionError):
            s.compare_gradients(torch.tensor([1.]), torch.tensor([2.]))

    def test_model_mode_mutation_refused(self):
        net, batch, losses = self.probe(1)
        def mutable(*args):
            if args[-1]["domain"] == "instrumental":
                net.training = not net.training
            return losses(*args)
        with patch.object(s, "slot_losses", mutable), self.assertRaises(ValueError):
            s.parameter_probe(net, batch, None, None, 1)

    def test_auxiliary_leakage_into_pseudo_refused(self):
        net, batch, losses = self.probe(1)
        def leaked(*args):
            base, residual, accompaniment, original, info = losses(*args)
            if args[-1]["domain"] == "pseudo":
                info = {"active": True}
            return base, residual, accompaniment, original, info
        with patch.object(s, "slot_losses", leaked), self.assertRaises(ValueError):
            s.parameter_probe(net, batch, None, None, 1)

    def test_skipped_nonzero_auxiliary_refused(self):
        net, batch, losses = self.probe(1)
        def bad(*args):
            base, residual, accompaniment, original, info = losses(*args)
            if not info["active"]:
                residual = residual+1
                original = residual+accompaniment
            return base, residual, accompaniment, original, info
        with patch.object(s, "slot_losses", bad), self.assertRaises(ValueError):
            s.parameter_probe(net, batch, None, None, 1)

    def test_original_auxiliary_scalar_mismatch_refused(self):
        net, batch, losses = self.probe(1)
        def bad(*args):
            base, residual, accompaniment, original, info = losses(*args)
            return base, residual, accompaniment, original+1, info
        with patch.object(s, "slot_losses", bad), self.assertRaises(ValueError):
            s.parameter_probe(net, batch, None, None, 1)


if __name__ == "__main__":
    unittest.main()
