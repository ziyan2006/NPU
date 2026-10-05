"""NONRELEASE EMA tensor-state core for report102; NOT a trainer or resume container.

No model call, autograd, optimizer, sampling, RNG access, file IO or device setup.
The future transaction owner must prove an actual raw Adam update, keep full raw
model/Adam/modes/grad/RNG/schedule/stop/input state, and roll back BOTH states.
This module checks shadow exposure only; passing raw_step is not training proof.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import re

import torch

SOURCE_SHA256 = "b3a0450ecd9bba259981bd47f8a49c0fee1a997d7a087a72570f589cc14ed5b3"
START, LIMIT = 4500, 5000
DECAY, RAW_WEIGHT = 0.99, 0.01
PURPOSE = "NONRELEASE_EMA_SHADOW_STATE_ONLY"


def exact_metadata(left, right):
    """Type- and order-sensitive, never bool==int or int==float coercion."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return list(left) == list(right) and all(exact_metadata(left[k], right[k]) for k in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(exact_metadata(a, b) for a, b in zip(left, right))
    if type(left) in (str, bool, int, float, type(None)):
        return left == right
    raise ValueError("Unsupported metadata type")


def _step(value):
    if type(value) is not int or not START <= value <= LIMIT:
        raise ValueError("Exact integer exposure4500..5000 required")


def _tensor(value, device=None):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or
            value.device.type not in ("cpu", "cuda") or value.is_quantized or
            (device is not None and value.device != device) or
            not bool(torch.isfinite(value).all())):
        raise ValueError("Finite dense tensor on the unchanged CPU/CUDA device required")


def _storage(value):
    # Empty storage has no writable elements and may legitimately have ptr0.
    return None if value.untyped_storage().nbytes() == 0 else (str(value.device), value.untyped_storage().data_ptr())


def _bit_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype and left.device == right.device and
            torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                        right.detach().contiguous().reshape(-1).view(torch.uint8)))


def _unaliased(values, external=()):
    occupied = {_storage(v) for v in external if _storage(v) is not None}
    for value in values:
        token = _storage(value)
        if token is not None:
            if token in occupied:
                raise ValueError("Shared tensor storage is not permitted")
            occupied.add(token)


def _model_layout(model):
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Explicit raw model required")
    modules = list(model.named_modules(remove_duplicate=False))
    if len({id(mod) for _, mod in modules}) != len(modules):
        raise ValueError("Shared modules are not supported by this fixed-order core")
    for _, mod in modules:
        if (type(mod).get_extra_state is not torch.nn.Module.get_extra_state or
                mod._state_dict_hooks or mod._state_dict_pre_hooks or
                mod._load_state_dict_pre_hooks or mod._load_state_dict_post_hooks):
            raise ValueError("Custom state serialization requires a separate reviewed adapter")
        if type(mod.training) is not bool:
            raise ValueError("Exact boolean module modes required")
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    if not parameters or not any(p.requires_grad for _, p in parameters):
        raise ValueError("At least one trainable parameter required")
    values = OrderedDict(parameters + buffers)
    if len(values) != len(parameters) + len(buffers):
        raise ValueError("Duplicate state names")
    device = parameters[0][1].device
    descriptors = []
    for kind, items in (("parameter", parameters), ("buffer", buffers)):
        for name, value in items:
            _tensor(value, device)
            if kind == "parameter" and value.dtype != torch.float32:
                raise ValueError("All raw parameters must remain FP32")
            if kind == "parameter" and type(value.requires_grad) is not bool:
                raise ValueError("Exact parameter trainability required")
            owner_name, _, local_name = name.rpartition(".")
            owner = model.get_submodule(owner_name) if owner_name else model
            persistent = kind == "parameter" or local_name not in owner._non_persistent_buffers_set
            descriptors.append({"name": name, "kind": kind, "shape": list(value.shape),
                                "dtype": str(value.dtype), "trainable": kind == "parameter" and value.requires_grad,
                                "persistent": persistent})
    _unaliased(values.values())
    # Respect state_dict interleaving; copied nonpersistent buffers stay in the
    # shadow resume packet but do not get invented as model.state_dict keys.
    model_keys = list(model.state_dict())
    expected = {r["name"] for r in descriptors if r["persistent"]}
    if set(model_keys) != expected:
        raise ValueError("Unexpected raw model state entries")
    return {"tensors": descriptors, "model_keys": model_keys,
            "modules": [name for name, _ in modules]}, values, device, [mod.training for _, mod in modules]


class EmaShadow:
    """Independent detached tensors, not nn.Module and never optimizer-owned."""

    def __init__(self, raw_model, *, source_sha256):
        if type(source_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", source_sha256) or source_sha256 != SOURCE_SHA256:
            raise ValueError("Report102 fixed control4500 source hash required")
        layout, raw, device, modes = _model_layout(raw_model)
        self._layout, self._device = layout, device
        self._values = OrderedDict((name, value.detach().clone()) for name, value in raw.items())
        self._modes, self._step = list(modes), START
        _unaliased(self._values.values(), raw.values())

    @property
    def step(self):
        return self._step

    @property
    def updates(self):
        return self._step - START

    def _guard(self, raw_model):
        _step(self._step)
        layout, raw, device, modes = _model_layout(raw_model)
        if device != self._device or not exact_metadata(layout, self._layout):
            raise ValueError("Raw parameter/buffer/module identity or device changed")
        if list(self._values) != list(raw):
            raise ValueError("Shadow names/order changed")
        for name, value in self._values.items():
            _tensor(value, device)
            if (value.shape != raw[name].shape or value.dtype != raw[name].dtype or
                    value.requires_grad or value.grad_fn is not None):
                raise ValueError("Shadow shape/dtype/gradient ownership changed")
        _unaliased(self._values.values(), raw.values())
        return raw, modes

    @torch.no_grad()
    def update_after_raw_step(self, raw_model, *, raw_step):
        """Caller-owned transaction: once AFTER real Adam, never apply to raw.

        Separate FP32 mul, mul, add, in original named_parameters order. No
        lerp, add(alpha=...), reassociation, fused operation or FP64 averaging.
        Build/validate all replacements before committing; a bad late buffer
        cannot leave a partial shadow update. Outer raw rollback remains pending.
        """
        _step(raw_step)
        if raw_step != self._step + 1:
            raise ValueError("Exactly one contiguous shadow update per raw step required")
        raw, modes = self._guard(raw_model)
        candidate = OrderedDict()
        for descriptor in self._layout["tensors"]:
            name = descriptor["name"]
            if descriptor["trainable"]:
                old_scaled = self._values[name] * DECAY
                raw_scaled = raw[name].detach() * RAW_WEIGHT
                candidate[name] = old_scaled + raw_scaled
            else:
                # Frozen parameters and EVERY buffer are typed raw copies.
                candidate[name] = raw[name].detach().clone()
            _tensor(candidate[name], self._device)
        _unaliased(candidate.values(), [*raw.values(), *self._values.values()])
        self._values, self._modes, self._step = candidate, list(modes), raw_step

    def state_dict(self, raw_model, *, raw_step):
        """CPU-portable SHADOW component only; cannot replace full raw container."""
        _step(raw_step)
        if raw_step != self._step:
            raise ValueError("Cannot save half-updated raw/shadow exposure")
        raw, modes = self._guard(raw_model)
        if not exact_metadata(modes, self._modes):
            raise ValueError("Raw modes changed after the last shadow copy")
        for descriptor in self._layout["tensors"]:
            name = descriptor["name"]
            if (not descriptor["trainable"] or self._step == START) and not _bit_equal(self._values[name], raw[name]):
                raise ValueError("E0 or copied nontrainable state differs from corresponding raw state")
        return self._snapshot()

    def _snapshot(self):
        return {"schema": 1, "purpose": PURPOSE, "source_sha256": SOURCE_SHA256,
                "source_step": START, "limit": LIMIT, "decay": DECAY, "raw_weight": RAW_WEIGHT,
                "arithmetic": "FP32_separate_mul_old_mul_raw_add_in_named_parameter_order",
                "step": self._step, "updates": self.updates, "layout": copy.deepcopy(self._layout),
                "modes_at_copy": list(self._modes), "tensors": OrderedDict(
                    (name, value.detach().cpu().clone()) for name, value in self._values.items()),
                "contains_matching_adam": False, "standalone_resume_authorized": False,
                "release_selection": "NONE"}

    @torch.no_grad()
    def load_state_dict(self, raw_model, state, *, raw_step):
        """Validated independent component load, no raw/Adam/RNG mutation."""
        _step(raw_step)
        raw, _ = self._guard(raw_model)
        # Raw was already restored by the future outer transaction owner; it
        # may be at a different exposure from the current shadow. Do not assert
        # old E0 equality against the newly restored raw before validating load.
        template = self._snapshot()
        if type(state) is not dict or list(state) != list(template):
            raise ValueError("Full ordered shadow schema required")
        varying = {"step", "updates", "modes_at_copy", "tensors"}
        if any(not exact_metadata(state[key], template[key]) for key in template if key not in varying):
            raise ValueError("Changed shadow source/schema/arithmetic/layout/budget")
        if (type(state["step"]) is not int or state["step"] != raw_step or
                type(state["updates"]) is not int or state["updates"] != raw_step - START or
                type(state["modes_at_copy"]) is not list or len(state["modes_at_copy"]) != len(self._layout["modules"]) or
                any(type(mode) is not bool for mode in state["modes_at_copy"]) or
                type(state["tensors"]) is not OrderedDict or list(state["tensors"]) != list(raw)):
            raise ValueError("Partial/changed shadow counters, modes or tensor order")
        candidate = OrderedDict()
        for name, value in state["tensors"].items():
            _tensor(value, torch.device("cpu"))
            if (value.shape != raw[name].shape or value.dtype != raw[name].dtype or
                    value.requires_grad or value.grad_fn is not None):
                raise ValueError("Malformed portable shadow tensor")
            candidate[name] = value.detach().to(self._device).clone()
        _unaliased(state["tensors"].values(), [*raw.values(), *self._values.values()])
        _unaliased(candidate.values(), [*raw.values(), *state["tensors"].values(), *self._values.values()])
        current_modes = [module.training for module in raw_model.modules()]
        if not exact_metadata(state["modes_at_copy"], current_modes):
            raise ValueError("Raw modes were not restored with the shadow packet")
        for descriptor in self._layout["tensors"]:
            name = descriptor["name"]
            if (not descriptor["trainable"] or raw_step == START) and not _bit_equal(candidate[name], raw[name]):
                raise ValueError("Restored E0/copied state must match its restored raw counterpart")
        self._values, self._modes, self._step = candidate, list(state["modes_at_copy"]), raw_step

    def model_state_dict(self, raw_model, *, raw_step):
        """Detached CPU evaluation weights. No optimizer or resume implication."""
        packet = self.state_dict(raw_model, raw_step=raw_step)
        return OrderedDict((name, packet["tensors"][name]) for name in self._layout["model_keys"])
