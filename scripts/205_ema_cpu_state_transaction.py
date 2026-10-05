"""NONRELEASE raw/Adam/shadow CPU state owner, NOT a trainer or CUDA gate.

No forward, backward, Adam step, input sampler, scoring, or training entry point.
This stage tests complete CPU storage/rollback, including already-existing grads.
Source CUDA RNG remains in the immutable parent; CPU migration is explicit and
cannot be called a same-device numerical resume or CUDA transaction proof.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA_CPU_STATE_OWNER_NOT_TRAINER"
PINS = {
    "scripts/203_ema_shadow_state.py": "7fb5fd1b3c5f66f3e9a2ebf0b020997c3454c8cf334e90de03282ebe78b939ae",
    "scripts/204_mel_ema_source_contract.py": "e36f6045ea3eec18bc28dd89c5b521bbc887a55a900425b7b55d3dec4ebf2ea9",
    "scripts/09_target_model.py": "e7fd2833a89b97b2355260beb125fd3ab72b06e040c82aa4c962945b7bb87b81",
    "scripts/common.py": "2bb3cff47058eb09a2ee63bcdf77fe19fa7e954873846ee8e622e2bec7fc15a8",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dependencies():
    for relative, digest in PINS.items():
        require(sha256(ROOT / relative) == digest, "Changed frozen dependency: " + relative)


def _module(name, relative):
    check_dependencies()
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ema = _module("ema205_shadow203", "scripts/203_ema_shadow_state.py")
source = _module("ema205_source204", "scripts/204_mel_ema_source_contract.py")


def portable(value):
    if isinstance(value, torch.Tensor):
        require(value.layout == torch.strided and not value.is_quantized, "Dense state required")
        return value.detach().cpu().clone()
    if type(value) in (dict, OrderedDict):
        return type(value)((k, portable(v)) for k, v in value.items())
    if type(value) in (list, tuple):
        return type(value)(portable(v) for v in value)
    if type(value) in (str, bool, int, float, type(None)):
        require(type(value) is not float or math.isfinite(value), "Finite metadata required")
        return value
    raise ValueError("Unsupported portable state type")


def typed_tree(value):
    """Canonical typed commitment; signed zeros and list/tuple/int/float differ."""
    if type(value) is torch.Tensor:
        require(value.device.type == "cpu" and not value.requires_grad and value.grad_fn is None
                and value.layout == torch.strided and bool(torch.isfinite(value).all()), "Finite detached CPU packet")
        data = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return ["tensor", str(value.dtype), list(value.shape), hashlib.sha256(data).hexdigest()]
    if type(value) in (dict, OrderedDict):
        return [type(value).__name__, [[typed_tree(k), typed_tree(v)] for k, v in value.items()]]
    if type(value) in (tuple, list):
        return [type(value).__name__, [typed_tree(v) for v in value]]
    if type(value) is float:
        require(math.isfinite(value), "Nonfinite metadata")
        return ["float", value.hex()]
    if type(value) in (str, int, bool, type(None)):
        return [type(value).__name__, value]
    raise ValueError("Unknown commitment type")


def digest(value):
    return hashlib.sha256(json.dumps(typed_tree(value), ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def equal(left, right):
    return typed_tree(left) == typed_tree(right)


def seal(packet):
    require("content_sha256" not in packet, "Refuse resealing existing packet")
    return {**packet, "content_sha256": digest(packet)}


def check_seal(packet):
    require(type(packet) is dict and type(packet.get("content_sha256")) is str, "Missing packet seal")
    require(digest({k: v for k, v in packet.items() if k != "content_sha256"}) == packet["content_sha256"],
            "Changed typed packet seal")


def noalias(value, external=()):
    values = list(source.tensor_leaves(value))
    ema._unaliased(values, external)


def capture_cpu_rng():
    require(not torch.cuda.is_initialized(), "Fresh CPU state stage cannot access live CUDA")
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], torch.from_numpy(state[1].astype(np.int64)).clone(),
                      int(state[2]), int(state[3]), float(state[4])],
            "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": []}


def validate_cpu_rng(state):
    require(type(state) is dict and list(state) == ["python", "numpy", "torch_cpu", "torch_cuda"], "Full CPU RNG fields")
    require(type(state["torch_cuda"]) is list and not state["torch_cuda"], "CPU migration active RNG must not claim CUDA")
    py = state["python"]
    require(type(py) is tuple and len(py) == 3 and type(py[0]) is int and py[0] == 3
            and type(py[1]) is tuple and len(py[1]) == 625 and all(type(v) is int for v in py[1])
            and all(0 <= v < 2**32 for v in py[1][:-1]) and 0 <= py[1][-1] <= 624
            and (py[2] is None or type(py[2]) is float and math.isfinite(py[2])), "Python RNG types/ranges")
    nr = state["numpy"]
    require(type(nr) is list and len(nr) == 5 and nr[0] == "MT19937"
            and type(nr[2]) is int and 0 <= nr[2] <= 624 and type(nr[3]) is int and nr[3] in (0, 1)
            and type(nr[4]) is float and math.isfinite(nr[4]), "NumPy RNG types")
    source.check_tensor(nr[1], torch.int64, (624,), "Active NumPy RNG")
    require(bool(((nr[1] >= 0) & (nr[1] < 2**32)).all()), "NumPy RNG words")
    source.check_tensor(state["torch_cpu"], torch.uint8, (5056,), "Active CPU RNG")


def restore_cpu_rng(state):
    validate_cpu_rng(state)
    random.setstate(state["python"])
    nr = state["numpy"]
    np.random.set_state((nr[0], nr[1].numpy().astype(np.uint32), nr[2], nr[3], nr[4]))
    torch.set_rng_state(state["torch_cpu"])


@contextmanager
def preserve_cpu_rng():
    previous = capture_cpu_rng()
    try:
        yield
    finally:
        restore_cpu_rng(previous)


def runtime_identity():
    require(not torch.cuda.is_initialized(), "CPU state stage forbids initialized CUDA")
    return {"device": "cpu", "torch": str(torch.__version__), "numpy": str(np.__version__),
            "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


class CpuStateOwner:
    """Owns one CPU raw model/Adam and independent203 shadow; never trains.

    Mutable context is storage for sampler/schedule state, NOT a real stream or
    scheduler implementation. Transaction closure checks counters/moments, not
    evidence that a caller performed an Adam update. Formal entry stays CLOSED.
    """

    def __init__(self, model, optimizer, shadow, parent_packet, context, *, evidence_scope):
        require(evidence_scope in ("synthetic_cpu_state_fixture", "fixed_source_cpu_explicit_migration"), "State scope only")
        require(type(optimizer) is torch.optim.Adam, "One original Adam type required")
        self.model, self.optimizer, self.shadow, self.context = model, optimizer, shadow, context
        self._parent = portable(parent_packet)
        self._parent_digest = digest(self._parent)
        self._scope, self._runtime = evidence_scope, runtime_identity()
        self._layout, values, device, _ = ema._model_layout(model)
        require(device.type == "cpu", "This state owner is CPU-only, not a CUDA implementation")
        self._names = [name for name, _ in model.named_parameters()]
        self._parameters = list(model.parameters())
        require(len(optimizer.param_groups) == 1
                and len(optimizer.param_groups[0]["params"]) == len(self._parameters)
                and all(a is b for a, b in zip(optimizer.param_groups[0]["params"], self._parameters)), "Adam exact live parameter order")
        self._group = portable(optimizer.state_dict()["param_groups"][0])
        self._defaults = portable(optimizer.defaults)
        self._context_template = portable(context)
        self._poisoned, self._in_transaction = False, False
        self.state_dict()

    @classmethod
    def from_fixed_source(cls, factory):
        """Load204 once, restore a fresh CPU raw/Adam/E0, retain full parent.

        Original saved CPU/Python/NumPy RNG is restored explicitly; source CUDA
        RNG is retained byte-for-byte in parent, not cleared or initialized.
        Caller may use preserve_cpu_rng when this is an audit rather than resume.
        """
        packet, evidence = source.load_fixed_source()
        with preserve_cpu_rng():
            model = factory()
            require([n for n, _ in model.named_parameters()] == packet["raw_arm"]["parameter_names"]
                    and len(list(model.modules())) == 27, "Original architecture/name/mode layout")
            require([n for n, _ in model.named_buffers()] == [], "Original source has no buffers")
            model.load_state_dict(packet["raw_arm"]["model"], strict=True)
            for mod, mode in zip(model.modules(), packet["raw_arm"]["modes"]):
                mod.training = mode
            group = copy.deepcopy(packet["raw_arm"]["optimizer"]["param_groups"][0])
            group.pop("params")
            optimizer = torch.optim.Adam(model.parameters(), **group)
            optimizer.load_state_dict(portable(packet["raw_arm"]["optimizer"]))
            require(equal(portable(optimizer.state_dict()), packet["raw_arm"]["optimizer"]), "Adam loading changed original types/bits")
            require(source.state_digest(model.state_dict()) == evidence["model_sha256"], "Raw source loading changed bits")
            shadow = ema.EmaShadow(model, source_sha256=source.SOURCE_SHA)
        active_rng = portable(packet["parent_metadata"]["rng"])
        active_rng["torch_cuda"] = []  # Explicit active CPU migration; parent is NOT changed.
        restore_cpu_rng(active_rng)
        context = {"step": 4500, "sampler": portable(packet["parent_metadata"]["sampler"]),
                   "schedule": portable(packet["parent_metadata"]["schedule"])}
        owner = cls(model, optimizer, shadow, packet, context,
                    evidence_scope="fixed_source_cpu_explicit_migration")
        noalias(owner.state_dict(), list(source.tensor_leaves(packet)))
        return owner, evidence

    def _guard(self):
        require(not self._poisoned, "Poisoned owner cannot continue")
        require(equal(runtime_identity(), self._runtime), "CPU numerical runtime changed")
        require(digest(self._parent) == self._parent_digest, "Immutable parent/source/RNG/old stops changed")
        require(all(a is b for a, b in zip(self.model.parameters(), self._parameters))
                and len(list(self.model.parameters())) == len(self._parameters), "Live parameter ownership changed")
        require(len(self.optimizer.param_groups) == 1
                and len(self.optimizer.param_groups[0]["params"]) == len(self._parameters)
                and all(a is b for a, b in zip(self.optimizer.param_groups[0]["params"], self._parameters)), "Adam must own raw parameters only")
        for name in ("_optimizer_step_pre_hooks", "_optimizer_step_post_hooks",
                     "_optimizer_state_dict_pre_hooks", "_optimizer_state_dict_post_hooks",
                     "_optimizer_load_state_dict_pre_hooks", "_optimizer_load_state_dict_post_hooks"):
            require(not getattr(self.optimizer, name, {}), "Custom Adam hooks need a separately reviewed adapter")

    def _validate_context(self, context):
        require(type(context) is dict and list(context) == ["step", "sampler", "schedule"], "Complete context fields")
        step = context["step"]
        require(type(step) is int and 4500 <= step <= 5000, "Exact state exposure4500..5000")
        for key, cursor_key in (("sampler", "cursor"), ("schedule", "step")):
            current, original = context[key], self._context_template[key]
            require(type(current) is dict and list(current) == list(original), "Full ordered context metadata")
            require(type(current[cursor_key]) is int and current[cursor_key] == step, "Partial sampler/schedule exposure")
            variable = {cursor_key} if key == "sampler" else {cursor_key, "last_validation", "best", "stale", "patience_anchor"}
            for name in original:
                if name not in variable:
                    require(equal(current[name], original[name]), "Changed source sampler/config/old stop: " + name)
        sch = context["schedule"]
        require(type(sch["last_validation"]) is int and sch["last_validation"] in (4500, 4750, 5000)
                and sch["last_validation"] <= step, "Validation exposure")
        for field in ("best", "stale", "patience_anchor"):
            require(type(sch[field]) is dict and list(sch[field]) == list(self._context_template["schedule"][field]), "Full cumulative map")
            for arm, value in sch[field].items():
                original = self._context_template["schedule"][field][arm]
                if arm != source.RAW_ARM or step == 4500:
                    require(equal(value, original), "Historical arm/source cumulative value changed")
                elif field == "stale":
                    require(type(value) is int and value >= original, "Do not reset cumulative stale")
                else:
                    require(value is None or type(value) is float and math.isfinite(value), "Typed cumulative metric")
        return step

    def _lr(self, step):
        config = self._context_template["schedule"]["config"]
        if step <= config["warmup_steps"]:
            return config["learning_rate"] * step / config["warmup_steps"]
        progress = (step - config["warmup_steps"]) / (config["maximum_steps"] - config["warmup_steps"])
        return config["cosine_min_learning_rate"] + (config["learning_rate"] - config["cosine_min_learning_rate"]) * (1 + math.cos(math.pi * progress)) / 2

    def _validate_raw(self, raw, step):
        require(type(raw) is dict and list(raw) == ["tensors", "gradients", "modes", "parameter_names", "optimizer", "optimizer_defaults", "updates"], "Complete raw fields")
        require(equal(raw["optimizer_defaults"], self._defaults), "All original typed Adam defaults")
        require(equal(raw["parameter_names"], self._names) and type(raw["updates"]) is int and raw["updates"] == step, "Raw order/exposure")
        descriptors = self._layout["tensors"]
        require(type(raw["tensors"]) is OrderedDict and list(raw["tensors"]) == [d["name"] for d in descriptors], "All parameters and nonpersistent buffers")
        for d in descriptors:
            tensor = raw["tensors"][d["name"]]
            ema._tensor(tensor, torch.device("cpu"))
            require(type(tensor) is torch.Tensor and not tensor.requires_grad and tensor.grad_fn is None
                    and str(tensor.dtype) == d["dtype"] and list(tensor.shape) == d["shape"], "Saved raw tensor descriptor")
        require(type(raw["modes"]) is list and len(raw["modes"]) == len(self._layout["modules"])
                and all(type(v) is bool for v in raw["modes"]), "Full typed module modes")
        require(type(raw["gradients"]) is OrderedDict and list(raw["gradients"]) == self._names, "Every existing grad/None saved")
        for name, grad in raw["gradients"].items():
            if grad is not None:
                source.check_tensor(grad, torch.float32, raw["tensors"][name].shape, name + "/grad")
        opt = raw["optimizer"]
        require(type(opt) is dict and list(opt) == ["state", "param_groups"]
                and type(opt["param_groups"]) is list and len(opt["param_groups"]) == 1, "Complete Adam")
        group = opt["param_groups"][0]
        require(type(group) is dict and set(group) == set(self._group), "All original Adam options")
        for key in self._group:
            require(equal(group[key], self._lr(step) if key == "lr" else self._group[key]), "Changed typed Adam option/order: " + key)
        require(type(opt["state"]) is dict and list(opt["state"]) == list(range(len(self._names)))
                and all(type(k) is int for k in opt["state"]), "All Adam parameter IDs/states")
        for index, name in enumerate(self._names):
            values = opt["state"][index]
            require(type(values) is dict and set(values) == {"step", "exp_avg", "exp_avg_sq"}, "Full moments")
            source.check_tensor(values["step"], torch.float32, (), name + "/step")
            require(float(values["step"]) == step, "Every Adam step matches exposure")
            for key in ("exp_avg", "exp_avg_sq"):
                source.check_tensor(values[key], torch.float32, raw["tensors"][name].shape, name + "/" + key)
            require(bool((values["exp_avg_sq"] >= 0).all()), "Negative Adam variance")

    def _validate_shadow(self, shadow, raw, step):
        template = self.shadow._snapshot()
        require(type(shadow) is dict and list(shadow) == list(template), "Full ordered203 shadow")
        for key in template:
            if key not in {"step", "updates", "modes_at_copy", "tensors"}:
                require(equal(shadow[key], template[key]), "Changed shadow source/layout/arithmetic/limit")
        require(type(shadow["step"]) is int and shadow["step"] == step
                and type(shadow["updates"]) is int and shadow["updates"] == step - 4500
                and equal(shadow["modes_at_copy"], raw["modes"]), "Partial shadow exposure/modes")
        require(type(shadow["tensors"]) is OrderedDict and list(shadow["tensors"]) == list(raw["tensors"]), "Shadow complete tensor order")
        for d in self._layout["tensors"]:
            name, value = d["name"], shadow["tensors"][d["name"]]
            ema._tensor(value, torch.device("cpu"))
            require(type(value) is torch.Tensor and not value.requires_grad and value.grad_fn is None
                    and value.shape == raw["tensors"][name].shape and value.dtype == raw["tensors"][name].dtype, "Shadow tensor descriptor")
            if step == 4500 or not d["trainable"]:
                require(equal(value, raw["tensors"][name]), "E0/copied buffer/frozen parameter differs")

    def validate(self, packet):
        check_seal(packet)
        keys = ["schema", "purpose", "scope", "source_sha256", "parent", "parent_digest", "runtime", "layout",
                "context", "raw", "shadow", "rng", "training_authorized", "cuda_transaction_verified", "release_selection", "content_sha256"]
        require(list(packet) == keys, "Complete ordered CPU container schema")
        for key, value in {"schema": 1, "purpose": PURPOSE, "scope": self._scope, "source_sha256": source.SOURCE_SHA,
                           "parent_digest": self._parent_digest, "runtime": self._runtime, "layout": self._layout,
                           "training_authorized": False, "cuda_transaction_verified": False, "release_selection": "NONE"}.items():
            require(equal(packet[key], value), "Container identity/type changed: " + key)
        require(equal(packet["parent"], self._parent), "Full immutable parent differs")
        step = self._validate_context(packet["context"])
        self._validate_raw(packet["raw"], step)
        self._validate_shadow(packet["shadow"], packet["raw"], step)
        validate_cpu_rng(packet["rng"])
        external = [*self.model.parameters(), *self.model.buffers(),
                    *(p.grad for p in self._parameters if p.grad is not None),
                    *source.tensor_leaves(self.optimizer.state_dict()), *self.shadow._values.values(),
                    *source.tensor_leaves(self._parent)]
        noalias(packet, external)

    def state_dict(self):
        self._guard()
        layout, values, _, modes = ema._model_layout(self.model)
        require(equal(layout, self._layout), "Raw architecture/trainability changed")
        step = self.context["step"]
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                source.check_tensor(parameter.grad, torch.float32, parameter.shape, name + "/existing live grad")
        raw = {"tensors": portable(values), "gradients": OrderedDict((n, None if p.grad is None else portable(p.grad))
                                                                           for n, p in self.model.named_parameters()),
               "modes": modes, "parameter_names": list(self._names), "optimizer": portable(self.optimizer.state_dict()),
               "optimizer_defaults": portable(self.optimizer.defaults), "updates": step}
        live = [*values.values(), *(p.grad for p in self._parameters if p.grad is not None),
                *source.tensor_leaves(self.optimizer.state_dict()), *self.shadow._values.values()]
        ema._unaliased(live)
        packet = seal({"schema": 1, "purpose": PURPOSE, "scope": self._scope, "source_sha256": source.SOURCE_SHA,
                       "parent": portable(self._parent), "parent_digest": self._parent_digest, "runtime": portable(self._runtime),
                       "layout": portable(self._layout), "context": portable(self.context), "raw": raw,
                       "shadow": self.shadow.state_dict(self.model, raw_step=step), "rng": capture_cpu_rng(),
                       "training_authorized": False, "cuda_transaction_verified": False, "release_selection": "NONE"})
        self.validate(packet)
        noalias(packet, live + list(source.tensor_leaves(self._parent)))
        return packet

    @torch.no_grad()
    def _apply(self, packet):
        values = OrderedDict([*self.model.named_parameters(), *self.model.named_buffers()])
        for d in self._layout["tensors"]:
            value = values[d["name"]]
            if d["kind"] == "parameter":
                value.requires_grad_(d["trainable"])
            value.copy_(packet["raw"]["tensors"][d["name"]])
        self.optimizer.load_state_dict(portable(packet["raw"]["optimizer"]))
        self.optimizer.defaults.clear()
        self.optimizer.defaults.update(portable(packet["raw"]["optimizer_defaults"]))
        for name, parameter in self.model.named_parameters():
            grad = packet["raw"]["gradients"][name]
            parameter.grad = None if grad is None else grad.clone()
        for mod, mode in zip(self.model.modules(), packet["raw"]["modes"]):
            mod.training = mode
        self.context.clear()
        self.context.update(portable(packet["context"]))
        self.shadow.load_state_dict(self.model, packet["shadow"], raw_step=packet["context"]["step"])
        restore_cpu_rng(packet["rng"])

    def load_state_dict(self, packet):
        self._guard()
        self.validate(packet)  # All validation before any live mutation.
        previous, untouched = self.state_dict(), digest(packet)
        try:
            self._apply(packet)
            require(equal(self.state_dict(), packet) and digest(packet) == untouched, "Full CPU restore changed state/input")
        except BaseException:
            self._rollback(previous)
            raise

    def _rollback(self, previous):
        try:
            self._apply(previous)
            require(equal(self.state_dict(), previous), "Whole CPU rollback differs")
        except BaseException as error:
            self._poisoned = True
            raise RuntimeError("CPU rollback failed; poisoned, no continuation") from error

    @contextmanager
    def transaction(self):
        require(not self._in_transaction, "No nested state transaction")
        previous = self.state_dict()
        self._in_transaction = True
        try:
            yield self
            self.state_dict()  # Reject half-updated shadow/Adam/context.
        except BaseException:
            self._rollback(previous)
            raise
        finally:
            self._in_transaction = False

    def save_new(self, path):
        """Exclusive new state artifact; incomplete IO is retained, never deleted."""
        path = Path(path)
        require(not path.is_symlink(), "No symlink output")
        packet = self.state_dict()
        with path.open("xb") as handle:
            torch.save(packet, handle)
            handle.flush()
            os.fsync(handle.fileno())
        loaded = self.read_checked(path, sha256(path))
        require(equal(loaded, packet), "Disk state differs")
        return sha256(path)

    def read_checked(self, path, expected_sha256):
        require(type(expected_sha256) is str and len(expected_sha256) == 64
                and sha256(path) == expected_sha256, "Changed disk SHA before deserialize")
        packet = torch.load(path, map_location="cpu", weights_only=True)
        require(sha256(path) == expected_sha256, "Disk changed during deserialize")
        self.validate(packet)
        return packet
