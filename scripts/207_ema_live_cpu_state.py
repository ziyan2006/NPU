"""Joint raw/Adam/EMA/206 input-and-schedule CPU storage, NOT a trainer.

New schema replaces205's incompatible frozen-context validator. Reuses its
typed storage/raw/Adam/shadow checks, not its live-context check. Supplied206
backends are not authenticated audio inputs. Manual state transitions used by
units are NOT optimizer updates. No CUDA resume, scoring or launch authority.
"""
from collections import OrderedDict
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA_JOINT_LIVE_CPU_STORAGE_NOT_TRAINER"
INPUT_PURPOSE = "NONRELEASE_EMA_JOINT_INPUT_STORAGE"
STORAGE_REL = "results/mel_ema_cpu_state_monitor_20261004/source_cpu_container_4500_complete_defaults.pt"
STORAGE_SHA = "b003b48e46e1e573491c7359251899a600e3daa542462204d38140aa7a11254a"
PINS = {
    "scripts/206_ema_live_input_schedule.py": "eaf1112263df6f74f97ea3e9f9a262db401dd24a819197ce0dbabdc6333d61ba",
    "scripts/205_ema_cpu_state_transaction.py": "8a8f2da566b664abfb19e6c24fe5fb4bf78bd61c13408ebf385b5cf425cd90f5",
    "scripts/203_ema_shadow_state.py": "7fb5fd1b3c5f66f3e9a2ebf0b020997c3454c8cf334e90de03282ebe78b939ae",
    "scripts/204_mel_ema_source_contract.py": "e36f6045ea3eec18bc28dd89c5b521bbc887a55a900425b7b55d3dec4ebf2ea9",
    "scripts/09_target_model.py": "e7fd2833a89b97b2355260beb125fd3ab72b06e040c82aa4c962945b7bb87b81",
    "scripts/common.py": "2bb3cff47058eb09a2ee63bcdf77fe19fa7e954873846ee8e622e2bec7fc15a8",
}
spec = importlib.util.spec_from_file_location("ema207_live206", ROOT / "scripts/206_ema_live_input_schedule.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)
c = live.storage
require, portable, equal = c.require, c.portable, c.equal


def check_dependencies():
    for path, expected in PINS.items():
        require(c.sha256(ROOT / path) == expected, "Changed joint-state dependency: " + path)
    c.check_dependencies()
    live.check_dependencies()


def provenance(parent):
    meta = parent["parent_metadata"]
    return {"source_sha256": c.source.SOURCE_SHA, "source_limit": meta["limit"],
            "source_legacy_stop_events": portable(meta["source_legacy_stop_events"]),
            "legacy_stop_events": portable(meta["legacy_stop_events"])}


class LiveCpuStateOwner(c.CpuStateOwner):
    """One full CPU ownership/rollback boundary including206's replay journal.

    Inherited load/save/transaction use this new full schema and new _apply.
    The205 _validate_context, validate, state_dict and constructor are NEVER
    used. CPU source migration retains original CUDA RNG in immutable parent.
    Raw counters/moments alone do not attest that an Adam update happened.
    """
    def __init__(self, model, optimizer, shadow, parent_packet, context, stream, *, evidence_scope,
                 source_storage=None, stop_provenance=None):
        check_dependencies()
        require(evidence_scope in ("synthetic_cpu_live_state_fixture", "fixed_cpu_source_synthetic_routing"),
                "Explicit non-training CPU scope")
        require(type(optimizer) is torch.optim.Adam and type(stream) is live.BoundedSingleTargetStream
                and type(shadow) is c.ema.EmaShadow,
                "One raw Adam and sealed206 stream required")
        self.model, self.optimizer, self.shadow, self.stream = model, optimizer, shadow, stream
        self._parent = portable(parent_packet)
        self._parent_digest = c.digest(self._parent)
        self._scope, self._runtime = evidence_scope, c.runtime_identity()
        self._layout, _, device, _ = c.ema._model_layout(model)
        require(device.type == "cpu", "CPU storage only")
        self._names = [name for name, _ in model.named_parameters()]
        self._parameters = list(model.parameters())
        self._group = portable(optimizer.state_dict()["param_groups"][0])
        self._defaults = portable(optimizer.defaults)
        self._context_template = portable(context)
        self._provenance = portable(stop_provenance if stop_provenance is not None else provenance(parent_packet))
        self.live = live.LiveContext(context, self._provenance)
        self._source_storage = portable(source_storage)
        self._objects = (model, optimizer, shadow, stream, self.live)
        self._poisoned, self._in_transaction = False, False
        require(equal(stream._source, context["sampler"]), "Stream original source differs from live context")
        self._backend_identity = portable(stream._backend_identity)
        self._fixed_digest = c.digest(self._fixed_identity())
        self.state_dict()

    def _fixed_identity(self):
        return [self._scope, self._runtime, self._layout, self._names, self._group, self._defaults,
                self._context_template, self._provenance, self._source_storage, self._backend_identity]

    @classmethod
    def from_fixed_cpu_storage(cls, factory, stream_factory):
        """One read of pinned205 zero-update CPU artifact, not old training PT.

        Caller supplies separately scoped synthetic routing in this stage.
        Original source files are hash checked, but not deserialized again.
        This is not the CUDA importer required for formal training.
        """
        check_dependencies()
        for path, expected in c.source.PINS.items():
            require(c.sha256(ROOT / path) == expected, "Changed original source interface")
        path = ROOT / STORAGE_REL
        require(c.sha256(path) == STORAGE_SHA, "Changed storage source before deserialize")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        require(c.sha256(path) == STORAGE_SHA, "Storage changed during deserialize")
        c.check_seal(saved)
        require(saved["purpose"] == c.PURPOSE and saved["scope"] == "fixed_source_cpu_explicit_migration"
                and saved["source_sha256"] == c.source.SOURCE_SHA, "Pinned zero-update CPU source scope")
        parent, context = saved["parent"], saved["context"]
        require(equal(context, {"step": 4500, "sampler": parent["parent_metadata"]["sampler"],
                                "schedule": parent["parent_metadata"]["schedule"]}), "Actual full initial context")
        # Original204 deliberately stores its model as dict;205 complete tensor
        # storage deliberately uses OrderedDict. Require both actual interface
        # types and identical name order, then compare every value by bits/type.
        # The full immutable parent and full stored trees keep their own types.
        source_model, stored_model = parent["raw_arm"]["model"], saved["raw"]["tensors"]
        require(type(source_model) is dict and type(stored_model) is OrderedDict
                and list(source_model) == list(stored_model)
                and all(equal(source_model[n], stored_model[n]) for n in source_model), "Original dict to ordered storage interface")
        require(equal(parent["raw_arm"]["optimizer"], saved["raw"]["optimizer"])
                and equal(parent["raw_arm"]["modes"], saved["raw"]["modes"]), "Stored actual raw source identity")
        with c.preserve_cpu_rng():
            model = factory()
            require([n for n, _ in model.named_parameters()] == parent["raw_arm"]["parameter_names"]
                    and len(list(model.modules())) == 27 and not list(model.buffers()), "Actual architecture/layout")
            model.load_state_dict(saved["raw"]["tensors"], strict=True)
            for module, mode in zip(model.modules(), saved["raw"]["modes"]):
                module.training = mode
            for name, parameter in model.named_parameters():
                grad = saved["raw"]["gradients"][name]
                parameter.grad = None if grad is None else grad.clone()
            group = portable(saved["raw"]["optimizer"]["param_groups"][0]); group.pop("params")
            optimizer = torch.optim.Adam(model.parameters(), **group)
            optimizer.load_state_dict(portable(saved["raw"]["optimizer"]))
            optimizer.defaults.clear(); optimizer.defaults.update(portable(saved["raw"]["optimizer_defaults"]))
            shadow = c.ema.EmaShadow(model, source_sha256=c.source.SOURCE_SHA)
            stream = stream_factory(portable(context["sampler"]))
        c.restore_cpu_rng(saved["rng"])
        owner = cls(model, optimizer, shadow, parent, context, stream,
                    evidence_scope="fixed_cpu_source_synthetic_routing",
                    source_storage={"path": STORAGE_REL, "sha256": STORAGE_SHA, "zero_update_cpu_storage": True})
        packet = owner.state_dict()
        require(equal(packet["raw"], saved["raw"]) and equal(packet["shadow"], saved["shadow"])
                and equal(packet["rng"], saved["rng"]), "Whole source CPU migration changed bits/types")
        c.noalias(packet, list(c.source.tensor_leaves(saved)))
        for path, expected in c.source.PINS.items():
            require(c.sha256(ROOT / path) == expected, "Original source interface changed during load")
        return owner

    def _guard(self):
        check_dependencies()
        require(all(a is b for a, b in zip((self.model, self.optimizer, self.shadow, self.stream, self.live), self._objects)),
                "Joint state ownership changed")
        require(c.digest(self._fixed_identity()) == self._fixed_digest, "Joint immutable identity changed")
        super()._guard()
        self.stream._guard()
        require(not self.live._poisoned and equal(self.live._source, self._context_template)
                and equal(self.live._provenance, self._provenance)
                and equal(self.stream._source, self._context_template["sampler"]), "Live context ownership/source changed")

    def _raw_packet(self, step):
        layout, values, _, modes = c.ema._model_layout(self.model)
        require(equal(layout, self._layout), "Raw architecture/trainability changed")
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                c.source.check_tensor(parameter.grad, torch.float32, parameter.shape, name + "/existing grad")
        return {"tensors": portable(values),
                "gradients": OrderedDict((n, None if p.grad is None else portable(p.grad)) for n, p in self.model.named_parameters()),
                "modes": modes, "parameter_names": list(self._names), "optimizer": portable(self.optimizer.state_dict()),
                "optimizer_defaults": portable(self.optimizer.defaults), "updates": step}

    def _external(self):
        return [*self.model.parameters(), *self.model.buffers(),
                *(p.grad for p in self._parameters if p.grad is not None),
                *c.source.tensor_leaves(self.optimizer.state_dict()), *self.shadow._values.values(),
                *c.source.tensor_leaves(self._parent), *c.source.tensor_leaves(self.live.context),
                *c.source.tensor_leaves(self.live.journal), *c.source.tensor_leaves(self.stream.last_metadata)]

    def _checked_live(self, packet):
        # A fresh verifier can check an incoming packet even if current state is
        # partial during rollback. Both nested own seal and full typed tree stay.
        verifier = live.LiveContext(self._context_template, self._provenance)
        verifier.load_state_dict(packet)
        require(equal(verifier.state_dict(), packet), "Full nested live context differs")
        return verifier

    def _validate_input(self, packet, context):
        c.check_seal(packet)
        require(list(packet) == ["purpose", "backend_identity", "sampler", "last_metadata", "content_sha256"]
                and equal(packet["purpose"], INPUT_PURPOSE)
                and equal(packet["backend_identity"], self._backend_identity), "Full fixed input storage schema")
        require(equal(packet["sampler"], context["sampler"]), "Input/live/raw exposure mismatch")
        metadata, step = packet["last_metadata"], context["step"]
        require((metadata is None) == (step == 4500), "Source versus consumed-input metadata exposure")
        if metadata is not None:
            live.validate_metadata(metadata)
            for row in metadata:
                require(type(row["score_start"]) is int and type(row["score_end"]) is int
                        and all(v in "0123456789abcdef" for v in row["input_pcm_sha256"]), "Exact support/PCM metadata types")

    def validate(self, packet):
        c.check_seal(packet)
        require(list(packet) == ["schema", "purpose", "scope", "source_sha256", "source_storage", "parent", "parent_digest",
                                "runtime", "layout", "live_context", "input_state", "raw", "shadow", "rng",
                                "training_authorized", "actual_audio_backend_verified", "cuda_transaction_verified",
                                "release_selection", "content_sha256"], "Complete joint CPU schema")
        fixed = {"schema": 1, "purpose": PURPOSE, "scope": self._scope, "source_sha256": c.source.SOURCE_SHA,
                 "source_storage": self._source_storage, "parent": self._parent, "parent_digest": self._parent_digest,
                 "runtime": self._runtime, "layout": self._layout, "training_authorized": False,
                 "actual_audio_backend_verified": False, "cuda_transaction_verified": False, "release_selection": "NONE"}
        require(all(equal(packet[k], v) for k, v in fixed.items()), "Joint container identity/types changed")
        context = self._checked_live(packet["live_context"]).context
        self._validate_input(packet["input_state"], context)
        self._validate_raw(packet["raw"], context["step"])
        self._validate_shadow(packet["shadow"], packet["raw"], context["step"])
        c.validate_cpu_rng(packet["rng"])
        c.noalias(packet, self._external())

    def state_dict(self):
        self._guard()
        context = self.live.state_dict()
        step = context["context"]["step"]
        c.ema._unaliased(self._external())
        packet = c.seal({"schema": 1, "purpose": PURPOSE, "scope": self._scope, "source_sha256": c.source.SOURCE_SHA,
                         "source_storage": portable(self._source_storage), "parent": portable(self._parent),
                         "parent_digest": self._parent_digest, "runtime": portable(self._runtime), "layout": portable(self._layout),
                         "live_context": context,
                         "input_state": c.seal({"purpose": INPUT_PURPOSE, "backend_identity": portable(self._backend_identity),
                                                "sampler": self.stream.state_dict(), "last_metadata": portable(self.stream.last_metadata)}),
                         "raw": self._raw_packet(step), "shadow": self.shadow.state_dict(self.model, raw_step=step),
                         "rng": c.capture_cpu_rng(), "training_authorized": False, "actual_audio_backend_verified": False,
                         "cuda_transaction_verified": False, "release_selection": "NONE"})
        self.validate(packet)
        return packet

    def complete_raw_storage_step(self):
        """Close caller-supplied raw state exposure; NOT evidence of Adam.step."""
        require(self._in_transaction, "Joint transaction required for storage step")
        self._guard()
        self.live.learning_rate_next()  # Enforce due observation / hard5000.
        step = self.live.step + 1
        require(self.stream.state_dict()["cursor"] == step, "One input exposure per storage step")
        self._validate_raw(self._raw_packet(step), step)
        self.shadow.update_after_raw_step(self.model, raw_step=step)
        self.live.complete_step(self.stream.state_dict())

    def observe_raw_policy_metadata(self, score):
        """Synthetic/unverified score storage; future trainer binds real files."""
        require(self._in_transaction, "Joint transaction required for observation")
        self.state_dict()
        self.live.observe_raw(score)

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
            parameter.grad = None if grad is None else grad.clone()
        for module, mode in zip(self.model.modules(), packet["raw"]["modes"]):
            module.training = mode
        self.shadow.load_state_dict(self.model, packet["shadow"], raw_step=verified.step)
        self.live.context, self.live.journal = portable(verified.context), portable(verified.journal)
        self.live.patience_events = list(verified.patience_events)
        self.stream.load_state_dict(packet["input_state"]["sampler"], last_metadata=packet["input_state"]["last_metadata"])
        c.restore_cpu_rng(packet["rng"])
