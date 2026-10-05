"""New synthetic CPU-only EMA component gates; no real student or audio IO.

Not a full resume/transaction/Adam/device/quality proof. Real audit and CPU/CUDA
next-input mechanisms are separate PENDING gates. Units do not authorize GPU.
"""
from collections import OrderedDict
import copy
import importlib.util
import io
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("ema_core", Path(__file__).with_name("203_ema_shadow_state.py"))
ema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ema)


def fixture():
    net = torch.nn.Module()
    net.register_parameter("weight", torch.nn.Parameter(torch.tensor([1., -2., 0., 1000.], dtype=torch.float32)))
    net.register_parameter("fixed", torch.nn.Parameter(torch.tensor([3., -0.], dtype=torch.float32), requires_grad=False))
    net.branch = torch.nn.Module()
    net.branch.register_parameter("bias", torch.nn.Parameter(torch.tensor([.125, -4.], dtype=torch.float32)))
    net.branch.register_buffer("floating", torch.tensor([1., -0.], dtype=torch.float64))
    net.branch.register_buffer("counter", torch.tensor(4500, dtype=torch.int64))
    net.branch.register_buffer("flags", torch.tensor([True, False]))
    net.branch.register_buffer("complex", torch.tensor([1+2j], dtype=torch.complex64))
    net.branch.register_buffer("scratch", torch.tensor([7.], dtype=torch.float32), persistent=False)
    net.branch.training = False
    net.weight.grad = torch.tensor([5., 6., 7., 8.], dtype=torch.float32)
    return net


def shadow(net):
    return ema.EmaShadow(net, source_sha256=ema.SOURCE_SHA256)


def bits(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape and left.device == right.device and
            torch.equal(left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)))


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and bits(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def forbidden(*args, **kwargs):
    raise AssertionError("Forbidden forward/autograd/Adam/CUDA initialization in CPU units")


class EmaStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patches = [patch.object(torch.nn.Module, "_call_impl", forbidden),
                       patch.object(torch.autograd, "grad", forbidden),
                       patch.object(torch.autograd, "backward", forbidden),
                       patch.object(torch.Tensor, "backward", forbidden),
                       patch.object(torch.optim.Adam, "__init__", forbidden),
                       patch.object(torch.cuda, "_lazy_init", forbidden)]
        for context in cls.patches:
            context.start()

    @classmethod
    def tearDownClass(cls):
        for context in reversed(cls.patches):
            context.stop()

    def setUp(self):
        self.net = fixture()
        self.core = shadow(self.net)

    def packet(self):
        return self.core.state_dict(self.net, raw_step=self.core.step)

    def bump(self, step=4501):
        with torch.no_grad():
            self.net.weight.add_(.25)
            self.net.branch.bias.sub_(.0625)
        self.core.update_after_raw_step(self.net, raw_step=step)

    def rejected_packet(self, change):
        before = self.packet()
        altered = copy.deepcopy(before)
        change(altered)
        with self.assertRaises(ValueError):
            self.core.load_state_dict(self.net, altered, raw_step=self.core.step)
        self.assertTrue(equal(before, self.packet()))

    def test_e0_all_tensors_bit_exact(self):
        packet = self.packet()
        for name, value in [*self.net.named_parameters(), *self.net.named_buffers()]:
            self.assertTrue(bits(packet["tensors"][name], value.detach()))
        self.assertEqual((packet["step"], packet["updates"]), (4500, 0))

    def test_fixed_source_required(self):
        for digest in ("0"*64, ema.SOURCE_SHA256.upper(), None, 1):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                ema.EmaShadow(self.net, source_sha256=digest)

    def test_fp32_parameters_only(self):
        self.net.weight.data = self.net.weight.data.double()
        with self.assertRaises(ValueError):
            shadow(self.net)

    def test_nonfinite_parameter_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            self.net.weight.data[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                shadow(self.net)

    def test_nonfinite_buffer_rejected(self):
        self.net.branch.floating[0] = float("nan")
        with self.assertRaises(ValueError):
            shadow(self.net)

    def test_shared_parameter_object_rejected(self):
        self.net.branch.register_parameter("shared", self.net.weight)
        with self.assertRaises(ValueError):
            shadow(self.net)

    def test_shared_distinct_parameter_storage_rejected(self):
        self.net.branch.register_parameter("shared", torch.nn.Parameter(self.net.weight.detach()))
        with self.assertRaises(ValueError):
            shadow(self.net)

    def test_shared_parameter_buffer_storage_rejected(self):
        self.net.register_buffer("alias", self.net.weight.detach())
        with self.assertRaises(ValueError):
            shadow(self.net)

    def test_shared_module_rejected(self):
        self.net.alias = self.net.branch
        with self.assertRaises(ValueError):
            shadow(self.net)

    def test_custom_extra_state_rejected(self):
        class Extra(torch.nn.Module):
            def get_extra_state(self):
                return {"x": 1}
        self.net.extra = Extra()
        with self.assertRaises(ValueError):
            shadow(self.net)

    def test_state_hook_rejected(self):
        handle = self.net.register_state_dict_pre_hook(lambda *args: None)
        try:
            with self.assertRaises(ValueError):
                shadow(self.net)
        finally:
            handle.remove()

    def test_exact_initial_modes(self):
        self.assertEqual(self.packet()["modes_at_copy"], [True, False])

    def test_initial_nonboolean_modes_rejected(self):
        self.net.branch.training = 0
        with self.assertRaises(ValueError):
            shadow(self.net)

    def test_one_update_independent_fp32_scalar_reference(self):
        previous = self.packet()["tensors"]
        self.bump()
        packet = self.packet()
        for name, value in self.net.named_parameters():
            if value.requires_grad:
                # NumPy independent scalar sequence, each multiplication and sum
                # rounded to FP32. Not torch add(alpha) or a FP64 rearrangement.
                expected = np.array([np.float32(np.float32(a*np.float32(.99)) + np.float32(b*np.float32(.01)))
                                     for a, b in zip(previous[name].numpy().flat, value.detach().numpy().flat)], dtype=np.float32)
                self.assertTrue(bits(packet["tensors"][name], torch.from_numpy(expected).reshape(value.shape)))

    def test_full_fixed500_shadow_recurrence_and_hard_limit(self):
        expected = np.array([1., -2., 0., 1000.], dtype=np.float32)
        for step in range(4501, 5001):
            self.bump(step)
            raw = self.net.weight.detach().numpy()
            expected = np.float32(np.float32(expected*np.float32(.99)) + np.float32(raw*np.float32(.01)))
        self.assertTrue(bits(self.packet()["tensors"]["weight"], torch.from_numpy(expected)))
        self.assertEqual(self.core.updates, 500)
        with self.assertRaises(ValueError):
            self.core.update_after_raw_step(self.net, raw_step=5001)

    def test_typed_buffers_and_frozen_parameter_copy_not_average(self):
        with torch.no_grad():
            self.net.fixed.add_(100)
            self.net.branch.floating.add_(123.456)
            self.net.branch.counter.add_(1)
            self.net.branch.flags.logical_not_()
            self.net.branch.complex.add_(2-1j)
            self.net.branch.scratch.mul_(2)
        self.bump()
        packet = self.packet()
        for name, value in [*(item for item in self.net.named_parameters() if not item[1].requires_grad), *self.net.named_buffers()]:
            self.assertTrue(bits(packet["tensors"][name], value.detach()))

    def test_nonpersistent_buffer_saved_but_not_evaluation_key(self):
        self.assertIn("branch.scratch", self.packet()["tensors"])
        evaluated = self.core.model_state_dict(self.net, raw_step=4500)
        self.assertNotIn("branch.scratch", evaluated)
        self.assertEqual(list(evaluated), list(self.net.state_dict()))

    def test_detached_shadow_and_nonzero_synthetic_difference(self):
        self.bump()
        for name, value in self.packet()["tensors"].items():
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        self.assertFalse(bits(self.packet()["tensors"]["weight"], self.net.weight.detach()))

    def test_raw_parameters_modes_and_existing_grads_not_mutated(self):
        before = {"model": copy.deepcopy(self.net.state_dict()), "modes": [m.training for m in self.net.modules()],
                  "grads": [None if p.grad is None else p.grad.clone() for p in self.net.parameters()]}
        self.core.update_after_raw_step(self.net, raw_step=4501)
        after = {"model": self.net.state_dict(), "modes": [m.training for m in self.net.modules()],
                 "grads": [p.grad for p in self.net.parameters()]}
        self.assertTrue(equal(before, after))

    def test_rng_unchanged_and_cpu_no_cuda_initialization(self):
        before_torch = torch.get_rng_state().clone()
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        self.bump()
        state = self.packet()
        self.core.load_state_dict(self.net, state, raw_step=4501)
        self.core.model_state_dict(self.net, raw_step=4501)
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertTrue(np.array_equal(before_numpy[1], after_numpy[1]))
        self.assertEqual(before_numpy[0], after_numpy[0])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        self.assertFalse(torch.cuda.is_initialized())

    def test_export_no_alias_raw_or_internal(self):
        packet = self.packet()
        raw = [*self.net.parameters(), *self.net.buffers()]
        ema._unaliased(packet["tensors"].values(), [*raw, *self.core._values.values()])
        packet["tensors"]["weight"].add_(10)
        self.assertTrue(bits(self.packet()["tensors"]["weight"], self.net.weight.detach()))

    def test_mutating_raw_does_not_alias_shadow(self):
        original = self.core._values["weight"].clone()
        self.net.weight.data.add_(10)
        self.assertTrue(bits(original, self.core._values["weight"]))
        with self.assertRaises(ValueError):
            self.packet()  # E0 identity must not be falsely asserted anymore.

    def test_changed_raw_names_rejected(self):
        self.net.register_parameter("new", torch.nn.Parameter(torch.ones(1)))
        with self.assertRaises(ValueError):
            self.core.update_after_raw_step(self.net, raw_step=4501)

    def test_changed_raw_parameter_order_rejected(self):
        self.net._parameters = dict(reversed(list(self.net._parameters.items())))
        with self.assertRaises(ValueError):
            self.core.update_after_raw_step(self.net, raw_step=4501)

    def test_changed_raw_shape_rejected(self):
        self.net.weight.data = self.net.weight.data.reshape(2, 2)
        with self.assertRaises(ValueError):
            self.core.update_after_raw_step(self.net, raw_step=4501)

    def test_changed_raw_trainability_rejected(self):
        self.net.fixed.requires_grad_(True)
        with self.assertRaises(ValueError):
            self.core.update_after_raw_step(self.net, raw_step=4501)

    def test_duplicate_skipped_old_boolean_float_steps_rejected(self):
        before = self.packet()
        for step in (True, 4501., 4500, 4502, 4499, 5001):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.core.update_after_raw_step(self.net, raw_step=step)
            self.assertTrue(equal(before, self.packet()))
        self.bump()
        with self.assertRaises(ValueError):
            self.core.update_after_raw_step(self.net, raw_step=4501)

    def test_partial_save_rejected(self):
        for step in (4501, True, 4500.):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.core.state_dict(self.net, raw_step=step)

    def test_failed_late_buffer_validation_does_not_commit_shadow(self):
        before = {name: value.clone() for name, value in self.core._values.items()}
        self.net.branch.scratch[0] = float("inf")
        with self.assertRaises(ValueError):
            self.core.update_after_raw_step(self.net, raw_step=4501)
        self.assertEqual(self.core.step, 4500)
        self.assertTrue(equal(before, dict(self.core._values)))

    def test_state_component_byte_serialization_roundtrip(self):
        self.bump()
        saved = self.packet()
        memory = io.BytesIO()
        torch.save(saved, memory)
        memory.seek(0)
        loaded = torch.load(memory, weights_only=True, map_location="cpu")
        self.assertTrue(equal(saved, loaded))
        other = shadow(fixture())
        other.load_state_dict(self.net, loaded, raw_step=4501)
        self.assertTrue(equal(saved, other.state_dict(self.net, raw_step=4501)))

    def test_component_load_no_alias_or_input_mutation(self):
        self.bump()
        saved = self.packet()
        untouched = copy.deepcopy(saved)
        self.core.load_state_dict(self.net, saved, raw_step=4501)
        self.bump(4502)
        self.assertTrue(equal(saved, untouched))
        ema._unaliased(self.core._values.values(), saved["tensors"].values())

    def test_raw_and_shadow_component_rollback_synthetic_only(self):
        # Explicitly NOT a full Adam/sampler/schedule/CUDA RNG transaction.
        saved = self.packet()
        raw_model = copy.deepcopy(self.net.state_dict())
        scratch = self.net.branch.scratch.clone()
        self.bump()
        self.net.load_state_dict(raw_model)
        self.net.branch.scratch.copy_(scratch)
        self.core.load_state_dict(self.net, saved, raw_step=4500)
        self.assertTrue(equal(saved, self.packet()))

    def test_changed_source_schema_decay_limit_rejected(self):
        changes = {"source_sha256": "0"*64, "schema": True, "source_step": 4500.,
                   "limit": 5001, "decay": .9, "raw_weight": .1, "purpose": "training",
                   "arithmetic": "lerp", "standalone_resume_authorized": True,
                   "contains_matching_adam": True, "release_selection": "EMA"}
        for key, value in changes.items():
            with self.subTest(key=key):
                self.rejected_packet(lambda packet, key=key, value=value: packet.__setitem__(key, value))

    def test_changed_counter_type_or_count_rejected(self):
        for key, value in (("step", 4500.), ("updates", False), ("updates", 1)):
            with self.subTest(key=key, value=value):
                self.rejected_packet(lambda packet, key=key, value=value: packet.__setitem__(key, value))

    def test_changed_layout_order_and_typed_flags_rejected(self):
        self.rejected_packet(lambda packet: packet["layout"]["tensors"].reverse())
        self.rejected_packet(lambda packet: packet["layout"]["tensors"][0].__setitem__("trainable", 1))

    def test_changed_or_nonboolean_saved_modes_rejected(self):
        self.rejected_packet(lambda packet: packet["modes_at_copy"].__setitem__(1, True))
        self.rejected_packet(lambda packet: packet["modes_at_copy"].__setitem__(1, 0))

    def test_changed_tensor_order_or_missing_tensor_rejected(self):
        self.rejected_packet(lambda packet: packet["tensors"].move_to_end("weight"))
        self.rejected_packet(lambda packet: packet["tensors"].pop("branch.scratch"))

    def test_malformed_saved_tensor_dtype_shape_nonfinite_rejected(self):
        for transform in (lambda tensor: tensor.double(), lambda tensor: tensor.reshape(2, 2),
                          lambda tensor: tensor*float("nan"), lambda tensor: "not a tensor"):
            with self.subTest(transform=transform):
                self.rejected_packet(lambda packet: packet["tensors"].__setitem__("weight", transform(packet["tensors"]["weight"])))

    def test_saved_grad_ownership_rejected(self):
        self.rejected_packet(lambda packet: packet["tensors"]["weight"].requires_grad_(True))

    def test_saved_alias_to_raw_rejected(self):
        self.rejected_packet(lambda packet: packet["tensors"].__setitem__("weight", self.net.weight.detach()))

    def test_saved_internal_alias_rejected(self):
        self.rejected_packet(lambda packet: packet["tensors"].__setitem__("weight", self.core._values["weight"]))

    def test_signed_zero_e0_bit_change_rejected(self):
        self.rejected_packet(lambda packet: packet["tensors"]["fixed"].__setitem__(1, 0.))

    def test_copied_buffer_or_frozen_parameter_tampering_rejected(self):
        self.bump()
        for name in ("fixed", "branch.counter", "branch.scratch"):
            with self.subTest(name=name):
                self.rejected_packet(lambda packet, name=name: packet["tensors"][name].add_(1))

    def test_full_ordered_schema_missing_extra_keys_rejected(self):
        self.rejected_packet(lambda packet: packet.pop("limit"))
        self.rejected_packet(lambda packet: packet.__setitem__("extra", 1))

    def test_not_model_empty_or_no_trainable_parameters_rejected(self):
        for net in (None, torch.nn.Module()):
            with self.subTest(net=net), self.assertRaises(ValueError):
                shadow(net)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        with self.assertRaises(ValueError):
            shadow(self.net)

    def test_guards_forbidden_operations_are_live(self):
        with self.assertRaises(AssertionError):
            self.net(torch.ones(1))
        with self.assertRaises(AssertionError):
            torch.optim.Adam(self.net.parameters())
        with self.assertRaises(AssertionError):
            torch.autograd.grad(None, None)
        with self.assertRaises(AssertionError):
            torch.cuda._lazy_init()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
