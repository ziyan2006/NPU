"""Single raw/Adam/EMA update and full CPU/CUDA state integration.

Unsealed integration work. Synthetic CPU tests do real tiny-model Adam steps,
not student training. Actual-source mechanisms and training require a separate
fully populated activation plan. No plan is generated here. Closed historical
engines, limits, evidence and suites are not run or edited.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
from types import MethodType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA_SINGLE_RAW_ADAM_TRAJECTORY"
PINS = {
    "scripts/207_ema_live_cpu_state.py": "7a1ab6a2dbcf9b2af99030784b34bcf6c2a73732227e0abadbb0cf2f88f18d90",
    "scripts/194_train_mel_lr_scale.py": "56455bab403f5ffbab941980dde4cfc2d9fe88934d35f7dd5e841f34b5279f48",
    "scripts/143_paired_distillation_mechanics.py": "bb3ad26dcf5c94e3909f8c4f05c9fb4f249e1a64b03f8b3d3155a26619ebddab",
    "scripts/170_instrumental_protection_loss.py": "bd2fdeeae3cd511056d2aa179feb3ba67e1b91619412926bf52bda8203137e82",
    "scripts/176_accompaniment_component_loss.py": "5036d4d23fe1c96d45e330b6d3e2bcf84220955d1f1ffda651c1c084c047db1b",
}
TORCH_OPTIM_PINS = {
    "adam.py": "e333d196bd24f6a44146839139ad0222bfaab0f71dcdb341585a4e3f9f4c2d87",
    "optimizer.py": "39e279bd90c1a0e016e1635c26591c666513c2e7648a72a0e5470295ed58ce56",
}


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


j = load_module("ema212_joint_primitives", "scripts/207_ema_live_cpu_state.py")
c, live = j.c, j.live
require, portable, equal = c.require, c.portable, c.equal


def check_dependencies():
    for relative, expected in PINS.items():
        require(c.sha256(ROOT / relative) == expected, "Changed212 dependency: " + relative)
    j.check_dependencies()
    for filename, expected in TORCH_OPTIM_PINS.items():
        require(c.sha256(Path(torch.__file__).parent / "optim" / filename) == expected,
                "Installed original Adam implementation changed: " + filename)


def cpu_adam_step(optimizer):
    """Original Adam.step/kernel, per-instance CPU graph-health adapter only.

    Installed PyTorch probes the default accelerator even with ALL state on
    CPU. Replace only that instance's graph-health check for the duration of
    one CPU step. No global patches, optimizer arithmetic/options/state edits,
    accelerator initialization or CUDA-path bypass. Missing/foreign state,
    capture flags, compilation, hooks and prior per-instance overrides reject.
    """
    check_dependencies()
    require(type(optimizer) is torch.optim.Adam and not torch.cuda.is_initialized()
            and not torch.compiler.is_compiling(), "Fresh eager CPU Adam only")
    name = "_accelerator_graph_capture_health_check"
    require(name not in optimizer.__dict__, "Existing instance health override requires separate review")
    def cpu_health(instance):
        require(instance is optimizer and not torch.cuda.is_initialized()
                and not torch.compiler.is_compiling(), "CPU health boundary changed")
        require(all(g["capturable"] is False and g["fused"] is False
                    and g["foreach"] is False and g["differentiable"] is False
                    for g in instance.param_groups), "CPU non-capturing original flags required")
        tensors = [*(p for g in instance.param_groups for p in g["params"]),
                   *(p.grad for g in instance.param_groups for p in g["params"] if p.grad is not None),
                   *c.source.tensor_leaves(instance.state_dict())]
        require(tensors and all(t.device.type == "cpu" for t in tensors), "All raw/grad/Adam state must be CPU")
    cpu_health(optimizer)
    setattr(optimizer, name, MethodType(cpu_health, optimizer))
    try:
        optimizer.step()  # Original installed step and original single-tensor kernel.
    finally:
        delattr(optimizer, name)


def validate_rng(state, device):
    require(device in ("cpu", "cuda"), "Explicit CPU/CUDA required")
    require(type(state) is dict and type(state.get("torch_cuda")) is list, "Complete RNG schema")
    # Reuse exact CPU/Python/NumPy validator without clearing any source bytes.
    c.validate_cpu_rng(state | {"torch_cuda": []})
    if device == "cpu":
        require(not state["torch_cuda"], "CPU migration must be explicit; not CUDA resume")
    else:
        require(len(state["torch_cuda"]) == 1, "Original single CUDA RNG required")
        c.source.check_tensor(state["torch_cuda"][0], torch.uint8, (16,), "Active CUDA RNG")


def capture_rng(device):
    require(device in ("cpu", "cuda"), "Explicit CPU/CUDA RNG capture")
    if device == "cpu":
        require(not torch.cuda.is_initialized(), "Fresh CPU process; no hidden CUDA initialization")
    else:
        require(torch.cuda.is_initialized(), "CUDA must be explicitly initialized before capture")
    n = np.random.get_state()
    state = {"python": random.getstate(),
             "numpy": [n[0], torch.from_numpy(n[1].astype(np.int64)).clone(), int(n[2]), int(n[3]), float(n[4])],
             "torch_cpu": torch.get_rng_state().clone(),
             "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if device == "cuda" else []}
    validate_rng(state, device)
    return state


def restore_rng(state, device):
    validate_rng(state, device)  # Full validation before first mutation.
    if device == "cpu":
        require(not torch.cuda.is_initialized(), "Cannot restore CPU migration into initialized CUDA process")
    else:
        require(torch.cuda.is_initialized() and torch.cuda.device_count() == 1, "Exact initialized CUDA device")
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if device == "cuda":
        torch.cuda.set_rng_state_all([s.clone() for s in state["torch_cuda"]])


def runtime_identity(device):
    require(device in ("cpu", "cuda"), "Explicit device; no fallback")
    if device == "cpu":
        require(not torch.cuda.is_initialized(), "CPU integration forbids CUDA initialization")
    result = {"device": device, "torch": str(torch.__version__), "numpy": str(np.__version__),
              "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
              "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32}
    if device == "cuda":
        require(torch.cuda.is_initialized() and torch.cuda.device_count() == 1, "Exact explicitly initialized CUDA runtime")
        result.update(cuda=str(torch.version.cuda), cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    return result


class DeviceInputStream(live.BoundedSingleTargetStream):
    """Original six-crop routing with full RNG capture, including live CUDA.

    Replaces only206's CPU-only RNG boundary. Supplied backends are NOT made
    authentic by this adapter. Foreign cache failures poison the backend and
    are never silently retried; semantic cursor restoration is separate.
    """
    def __init__(self, sampler, true_pool, teacher_dataset, crop_recipe, *, device):
        require(device in ("cpu", "cuda"), "Explicit input RNG device")
        super().__init__(sampler, true_pool, teacher_dataset, crop_recipe)
        self.device = device

    def next_batch(self):
        before, metadata_before, rng = self.state_dict(), portable(self.last_metadata), capture_rng(self.device)
        require(self.cursor < live.LIMIT, "Hard5000 input limit")
        try:
            xs, vs, rows = [], [], []
            for domain in live.DOMAINS[:3]:
                row = self.true.crop(domain, self._source["seed"], self.cursor)
                xs.append(row["x"]); vs.append(row["v"]); rows.append(portable(row["meta"]))
            for index in range(3):
                recipe = self.recipe(self.dataset.rows, self.dataset.config, self._source["seed"], self.cursor*3+index)
                row = self.dataset.crop(recipe)
                xs.append(row["x"]); vs.append(row["v"])
                rows.append(portable(row["meta"]) | {"domain": "pseudo", "role": "pseudo_label_train_candidate"})
            live.validate_metadata(rows)
            require(all(type(v) is torch.Tensor and v.device.type == "cpu" and v.dtype == torch.float32
                        and v.shape == (2, 89856) and not v.requires_grad and v.grad_fn is None
                        and bool(torch.isfinite(v).all()) for v in xs+vs), "Original finite detached FP32 crops")
            x, v = torch.stack(xs), torch.stack(vs)
            hashes = [hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest() for w in x]
            require(hashes == [r["input_pcm_sha256"] for r in rows], "Input PCM identity mismatch")
            require(equal(rng, capture_rng(self.device)), "Local sampler changed global CPU/CUDA RNG")
            result = {"x": x, "v": v, "domains": live.DOMAINS, "cursor": self.cursor, "metadata": rows}
            self.cursor += 1
            self.last_metadata = portable(rows)
            return result
        except BaseException:
            # Preserve input failure as fatal to THIS stream even when raw state
            # rollback succeeds. Do not claim foreign cache rollback.
            self.cursor, self.last_metadata = before["cursor"], metadata_before
            try:
                restore_rng(rng, self.device)
            finally:
                self.poisoned = True
            raise


class OriginalBoundaryLoss:
    """Execute exactly194's two pure functions, not its closed two-arm CLI.

    AST selection is source-SHA bound. No loss rewrite, different reduction,
    extra forward, new graph or old guard removal. Imported original143
    frontend/loss modules and matrices remain an actual-runtime gate to verify.
    """
    def __init__(self, device):
        check_dependencies()
        m = load_module("ema212_original143", "scripts/143_paired_distillation_mechanics.py")
        k = load_module("ema212_original176", "scripts/176_accompaniment_component_loss.py")
        w = load_module("ema212_original170", "scripts/170_instrumental_protection_loss.py")
        source = ROOT / "scripts/194_train_mel_lr_scale.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("validate_metadata", "boundary_backward")]
        require([n.name for n in nodes] == ["validate_metadata", "boundary_backward"], "Exact original functions")
        def finite(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, (tuple, list)):
                return all(finite(v) for v in value)
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            return True
        from types import SimpleNamespace
        ns = {"torch": torch, "m": m, "k": k, "w": w, "d": SimpleNamespace(finite_state=finite)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
        self.function = ns["boundary_backward"]
        self.wa = torch.from_numpy(m.core.t09.make_analysis_matrix()).to(device)
        self.gs = torch.from_numpy(m.core.t09.make_synthesis_matrix()).to(device)
        self.identity = {"kind": "original194_boundary_backward", "source_sha256": PINS["scripts/194_train_mel_lr_scale.py"],
                         "kill": 32, "warmup": 96, "auxiliary": .2, "instrumental_weight": 4, "denominator": 6,
                         "analysis_matrix": portable(self.wa), "synthesis_matrix": portable(self.gs)}
        self._binding = c.digest(self.identity)
        self._function = self.function

    def __call__(self, model, batch, device):
        check_dependencies()
        require(self.function is self._function and c.digest(self.identity) == self._binding
                and equal(portable(self.wa), self.identity["analysis_matrix"])
                and equal(portable(self.gs), self.identity["synthesis_matrix"]), "Original loss/matrix changed")
        return self.function(model, batch["x"], batch["v"], self.wa, self.gs, device, batch["metadata"], 32)


class SingleTrajectoryEngine(j.LiveCpuStateOwner):
    """One actual raw Adam update, then EMA, in ONE complete transaction.

    Inherits only portable validators/storage/load/transaction algorithms;
    overrides207's CPU-only constructor, schema, runtime, RNG and _apply.
    CUDA implementation is not CUDA evidence. Current public construction is
    CPU fixture integration only; actual importer/activation remains closed.
    """
    def __init__(self, model, optimizer, parent_packet, context, stream, backward, *,
                 device="cpu", scope="synthetic_cpu_adam_integration", stop_provenance=None):
        check_dependencies()
        # No user-controlled flag can turn a fixture into formal authorization.
        require(scope in ("synthetic_cpu_adam_integration", "fixed_source_cpu_restoration") and device == "cpu",
                "Actual audio importer/zero audit/CUDA mechanism and formal activation still pending")
        require(type(stream) is DeviceInputStream and stream.device == device,
                "New device-aware single-target route required")
        require(type(optimizer) is torch.optim.Adam and callable(backward), "One original raw Adam/backward required")
        self.model, self.optimizer, self.stream, self.backward = model, optimizer, stream, backward
        self.device, self._scope = device, scope
        self.shadow = c.ema.EmaShadow(model, source_sha256=c.source.SOURCE_SHA)
        self._parent = portable(parent_packet)
        self._parent_digest = c.digest(self._parent)
        if scope == "synthetic_cpu_adam_integration":
            require(parent_packet.get("synthetic_fixture") is True, "Fixture is not authenticated source PT")
        else:
            require(parent_packet.get("source_sha256") == c.source.SOURCE_SHA
                    and parent_packet.get("selected_arm") == c.source.RAW_ARM,
                    "Exact control4500 source packet required")
            c.source.check_arm(parent_packet["raw_arm"], parent_packet["raw_arm"]["optimizer"]["param_groups"][0]["lr"])
        self._layout, _, actual_device, _ = c.ema._model_layout(model)
        require(actual_device == torch.device(device) and all(p.requires_grad for p in model.parameters()),
                "Original all-trainable raw FP32 parameter layout required")
        self._parameters = list(model.parameters())
        self._names = [n for n, _ in model.named_parameters()]
        self._group = portable(optimizer.state_dict()["param_groups"][0])
        self._defaults = portable(optimizer.defaults)
        expected = {"betas": (.9, .999), "eps": 1e-8, "weight_decay": 0, "amsgrad": False,
                    "maximize": False, "foreach": False, "capturable": False, "differentiable": False,
                    "fused": False, "decoupled_weight_decay": False, "params": list(range(len(self._names)))}
        require(all(equal(self._group.get(k), v) for k, v in expected.items()), "Original typed Adam options/order required")
        self._context_template = portable(context)
        require(equal(stream._source, context["sampler"]), "Input/source context mismatch")
        self._provenance = portable(stop_provenance)
        self.live = live.LiveContext(context, self._provenance)
        require(context["schedule"]["config"]["gradient_clip"] == 3, "Original gradient clip3")
        self._backend_identity = portable(stream._backend_identity)
        self._runtime = runtime_identity(device)
        require(self._runtime["deterministic"] and not self._runtime["warn_only"]
                and not self._runtime["cudnn_benchmark"] and self._runtime["cudnn_deterministic"]
                and not self._runtime["matmul_tf32"] and not self._runtime["cudnn_tf32"], "Strict original FP32 runtime")
        self._loss_identity = portable(backward.identity)
        self._objects = (model, optimizer, self.shadow, stream, self.live, backward)
        self._immutable = c.digest([self._scope, self.device, self._layout, self._names, self._group,
                                   self._defaults, self._context_template, self._provenance,
                                   self._backend_identity, self._loss_identity])
        self._poisoned, self._in_transaction = False, False
        # Exposure counters are diagnostic, NOT rolled back as training state.
        self.exposure = {"updates_started": 0, "updates_committed": 0, "updates_failed": 0,
                         "adam_started": 0, "adam_completed": 0}
        self.state_dict()

    @classmethod
    def from_fixed_source(cls, *args, **kwargs):
        raise ValueError("Use pinned CPU-storage restoration; actual CUDA importer not yet activated")

    @classmethod
    def from_fixed_cpu_storage(cls, factory, stream_factory, backward):
        """One authenticated zero-update CPU migration, never real updates.

        Reads the pinned205 storage once, not the original training PT. Retains
        full original parent CUDA RNG/old stops and exact CPU Adam defaults.
        CPU active RNG has an explicitly empty CUDA list. No audio constructor.
        """
        check_dependencies()
        for path, expected in c.source.PINS.items():
            require(c.sha256(ROOT / path) == expected, "Changed source-interface binding")
        path = ROOT / j.STORAGE_REL
        require(c.sha256(path) == j.STORAGE_SHA, "Changed CPU storage before deserialize")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        require(c.sha256(path) == j.STORAGE_SHA, "CPU storage changed during read")
        c.check_seal(saved)
        require(saved["purpose"] == c.PURPOSE and saved["scope"] == "fixed_source_cpu_explicit_migration"
                and saved["source_sha256"] == c.source.SOURCE_SHA, "Fixed zero-update CPU storage only")
        parent, raw = saved["parent"], saved["raw"]
        require(type(parent["raw_arm"]["model"]) is dict and type(raw["tensors"]) is OrderedDict
                and list(parent["raw_arm"]["model"]) == list(raw["tensors"])
                and all(equal(parent["raw_arm"]["model"][n], raw["tensors"][n]) for n in raw["tensors"]),
                "Actual dict/OrderedDict source interface and per-value bits")
        require(equal(parent["raw_arm"]["optimizer"], raw["optimizer"])
                and equal(parent["raw_arm"]["modes"], raw["modes"]), "Complete original raw Adam/modes")
        with c.preserve_cpu_rng():
            model = factory()
            require([n for n, _ in model.named_parameters()] == parent["raw_arm"]["parameter_names"]
                    and len(list(model.modules())) == 27 and not list(model.buffers()), "Exact original architecture")
            model.load_state_dict(raw["tensors"], strict=True)
            for module, mode in zip(model.modules(), raw["modes"]):
                module.training = mode
            for name, p in model.named_parameters():
                p.grad = None if raw["gradients"][name] is None else raw["gradients"][name].clone()
            group = portable(raw["optimizer"]["param_groups"][0]); group.pop("params")
            optimizer = torch.optim.Adam(model.parameters(), **group)
            optimizer.load_state_dict(portable(raw["optimizer"]))
            optimizer.defaults.clear(); optimizer.defaults.update(portable(raw["optimizer_defaults"]))
            stream = stream_factory(portable(saved["context"]["sampler"]))
        restore_rng(saved["rng"], "cpu")
        owner = cls(model, optimizer, parent, saved["context"], stream, backward,
                    scope="fixed_source_cpu_restoration", stop_provenance=j.provenance(parent))
        restored = owner.state_dict()
        require(equal(restored["raw"], raw) and equal(restored["shadow"], saved["shadow"])
                and equal(restored["rng"], saved["rng"]), "Whole zero-update source migration differs")
        c.noalias(restored, c.source.tensor_leaves(saved))
        for path, expected in c.source.PINS.items():
            require(c.sha256(ROOT / path) == expected, "Source binding changed during migration")
        return owner

    def _guard(self):
        check_dependencies()
        require(not self._poisoned, "Poisoned engine cannot continue")
        require(all(a is b for a, b in zip((self.model, self.optimizer, self.shadow, self.stream, self.live, self.backward),
                                         self._objects)), "Engine ownership changed")
        require(equal(runtime_identity(self.device), self._runtime), "Numerical runtime changed")
        require(c.digest(self._parent) == self._parent_digest, "Source/old stop/parent CUDA RNG changed")
        identity = [self._scope, self.device, self._layout, self._names, self._group, self._defaults,
                    self._context_template, self._provenance, self._backend_identity, self._loss_identity]
        require(c.digest(identity) == self._immutable and equal(self.backward.identity, self._loss_identity), "Immutable engine identity changed")
        require(len(self.optimizer.param_groups) == 1
                and len(self.optimizer.param_groups[0]["params"]) == len(self._parameters)
                and all(a is b for a, b in zip(self.optimizer.param_groups[0]["params"], self._parameters))
                and all(a is b for a, b in zip(self.model.parameters(), self._parameters)), "Adam raw ownership/order changed")
        for name in ("_optimizer_step_pre_hooks", "_optimizer_step_post_hooks", "_optimizer_state_dict_pre_hooks",
                     "_optimizer_state_dict_post_hooks", "_optimizer_load_state_dict_pre_hooks", "_optimizer_load_state_dict_post_hooks"):
            require(not getattr(self.optimizer, name, {}), "Custom optimizer hooks forbidden")
        self.stream._guard()
        require(equal(self.stream._backend_identity, self._backend_identity)
                and equal(self.stream._source, self._context_template["sampler"]), "Changed input/source identity")

    def validate(self, packet):
        c.check_seal(packet)
        require(list(packet) == ["schema", "purpose", "scope", "device", "source_sha256", "parent", "parent_digest",
                                "runtime", "layout", "loss_identity", "live_context", "input_state", "raw", "shadow", "rng",
                                "training_authorized", "actual_audio_backend_verified", "cuda_transaction_verified", "release_selection",
                                "content_sha256"], "Full single trajectory schema")
        fixed = {"schema": 1, "purpose": PURPOSE, "scope": self._scope, "device": self.device,
                 "source_sha256": c.source.SOURCE_SHA, "parent": self._parent, "parent_digest": self._parent_digest,
                 "runtime": self._runtime, "layout": self._layout, "loss_identity": self._loss_identity,
                 "training_authorized": False, "actual_audio_backend_verified": False,
                 "cuda_transaction_verified": False, "release_selection": "NONE"}
        require(all(equal(packet[k], v) for k, v in fixed.items()), "Changed identity/scope/runtime")
        context = self._checked_live(packet["live_context"]).context
        self._validate_input(packet["input_state"], context)
        self._validate_raw(packet["raw"], context["step"])
        self._validate_shadow(packet["shadow"], packet["raw"], context["step"])
        validate_rng(packet["rng"], self.device)
        c.noalias(packet, self._external())

    def state_dict(self):
        self._guard()
        context = self.live.state_dict()
        step = context["context"]["step"]
        c.ema._unaliased(self._external())
        packet = c.seal({"schema": 1, "purpose": PURPOSE, "scope": self._scope, "device": self.device,
                         "source_sha256": c.source.SOURCE_SHA, "parent": portable(self._parent), "parent_digest": self._parent_digest,
                         "runtime": portable(self._runtime), "layout": portable(self._layout), "loss_identity": portable(self._loss_identity),
                         "live_context": context,
                         "input_state": c.seal({"purpose": j.INPUT_PURPOSE, "backend_identity": portable(self._backend_identity),
                                                "sampler": self.stream.state_dict(), "last_metadata": portable(self.stream.last_metadata)}),
                         "raw": self._raw_packet(step), "shadow": self.shadow.state_dict(self.model, raw_step=step),
                         "rng": capture_rng(self.device), "training_authorized": False,
                         "actual_audio_backend_verified": False, "cuda_transaction_verified": False, "release_selection": "NONE"})
        self.validate(packet)
        return packet

    @torch.no_grad()
    def _apply(self, packet):
        verified = self._checked_live(packet["live_context"])
        values = OrderedDict([*self.model.named_parameters(), *self.model.named_buffers()])
        for desc in self._layout["tensors"]:
            value = values[desc["name"]]
            if desc["kind"] == "parameter":
                value.requires_grad_(desc["trainable"])
            value.copy_(packet["raw"]["tensors"][desc["name"]])
        self.optimizer.load_state_dict(portable(packet["raw"]["optimizer"]))
        self.optimizer.defaults.clear(); self.optimizer.defaults.update(portable(packet["raw"]["optimizer_defaults"]))
        for name, parameter in self.model.named_parameters():
            grad = packet["raw"]["gradients"][name]
            parameter.grad = None if grad is None else grad.to(self.device).clone()
        for module, mode in zip(self.model.modules(), packet["raw"]["modes"]):
            module.training = mode
        self.shadow.load_state_dict(self.model, packet["shadow"], raw_step=verified.step)
        self.live.context, self.live.journal = portable(verified.context), portable(verified.journal)
        self.live.patience_events = list(verified.patience_events)
        # A failed foreign input is fatal. Its semantic state can still be
        # restored for evidence without invoking an unsafe retry/guard.
        live.validate_sampler(packet["input_state"]["sampler"], self.stream._source)
        self.stream.cursor = packet["input_state"]["sampler"]["cursor"]
        self.stream.last_metadata = portable(packet["input_state"]["last_metadata"])
        restore_rng(packet["rng"], self.device)

    def update_next(self):
        self._guard()
        require(self._scope == "synthetic_cpu_adam_integration", "Source restoration does not authorize real audio/update")
        lr = self.live.learning_rate_next()  # Gate4750/5000 BEFORE input or Adam.
        require(self.live.step < 4503, "Fixture integration is at most3 steps; not formal500-step authority")
        self.exposure["updates_started"] += 1
        try:
            with self.transaction():
                batch = self.stream.next_batch()
                require(batch["cursor"] == self.live.step and equal(batch["domains"], live.DOMAINS), "Input exposure/domain mismatch")
                before_input = c.digest(batch)
                self.model.train()
                self.optimizer.param_groups[0]["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                losses = self.backward(self.model, batch, self.device)
                c.typed_tree(losses)  # Exact finite result before any Adam mutation.
                require(all(p.grad is not None and p.grad.dtype == torch.float32 and p.grad.device == p.device
                            and bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()), "Complete finite raw gradients")
                require(c.digest(batch) == before_input, "Backward changed source PCM/metadata")
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3, error_if_nonfinite=True)
                self.exposure["adam_started"] += 1
                if self.device == "cpu":
                    cpu_adam_step(self.optimizer)
                else:
                    self.optimizer.step()  # Never adapt the CUDA graph-health path.
                self.exposure["adam_completed"] += 1
                self.optimizer.zero_grad(set_to_none=True)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                self.complete_raw_storage_step()
                result = {"step": self.live.step, "lr": lr, "losses": portable(losses),
                          "gradient_norm_before_clip": float(norm), "metadata": portable(batch["metadata"]),
                          "input_sha256": [r["input_pcm_sha256"] for r in batch["metadata"]],
                          "synthetic_fixture": True, "student_training_updates": 0}
            self.exposure["updates_committed"] += 1
            return result
        except BaseException:
            self.exposure["updates_failed"] += 1
            raise

    def _rollback(self, previous):
        failed_input = self.stream.poisoned
        try:
            self._apply(previous)
            # Permit a single READ-ONLY whole-state equality check of restored
            # semantic state; do not unpoison or expose a retry afterwards.
            if failed_input:
                self.stream.poisoned = False
            require(equal(self.state_dict(), previous), "Full raw/Adam/EMA/input/schedule/RNG rollback differs")
        except BaseException as error:
            self._poisoned = True
            raise RuntimeError("Whole-state rollback failed; engine poisoned") from error
        finally:
            if failed_input:
                self.stream.poisoned = True
                self._poisoned = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("status", "train"))
    args = parser.parse_args()
    if args.operation == "train":
        raise SystemExit("Training CLOSED: actual source/audio importer, real zero-update audit, <=3 CPU-CUDA mechanisms and activation plan pending")
    print(json.dumps({"purpose": PURPOSE, "single_raw_adam_update_implementation": True,
                      "actual_source_student_updates": 0, "formal_training_authorized": False,
                      "cuda_implementation_exercised": False, "release_selection": "NONE"}))


if __name__ == "__main__":
    main()
