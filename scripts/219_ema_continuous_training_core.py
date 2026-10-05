"""Continuous single raw/Adam trajectory and atomic raw/EMA DEV checkpoints.

New integration, not a rerun or modification of closed mechanisms. Public
CPU fixtures never authorize student training. Actual construction requires a
separate complete activation record; this file does not manufacture one.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA219_CONTINUOUS_SINGLE_TRAJECTORY"
FIXTURE_SCOPE = "synthetic_continuous_CPU_not_student_training"
ACTUAL_SCOPE = "activated_single_raw_original_Adam_CUDA_fixed500"
PINS = {
    "scripts/212_ema_single_trajectory_engine.py": "8d10c304f3c3c2b4d2d6509dcec719e454b6afdabe651a329d1bc99b4a85d666",
    "scripts/215_ema_model_audio_audit.py": "8e9cce90b934b465e8946d805a53a7d34a881de1d752f3634ccc8f0ea72a5b3c",
    "scripts/217_ema_actual_update_mechanism.py": "468734d23c353c805a87fa10f3c6d134a14bf789adabb2659fb63237e4bc71fd",
    "scripts/218_ema_mechanism_runtime_recovery.py": "2652de693f6300e274feb102379e095f4d0a747755a38c2a2c4908d3a76eba4a",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_dependencies():
    for relative, expected in PINS.items():
        require(sha(ROOT / relative) == expected, "Changed219 component: " + relative)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_dependencies()
e = load("ema219_single_trajectory_primitives", "scripts/212_ema_single_trajectory_engine.py")
c, live, torch = e.c, e.live, e.torch


def checked_activation(path):
    """An implementation cannot activate itself with a boolean switch."""
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / "results") and path.is_file(), "Existing complete activation record required")
    doc = json.loads(path.read_text(encoding="utf-8"))
    # Tensor-free authority uses its own typed214 seal, not a plain JSON seal.
    a = load("ema219_activation_metadata", "scripts/215_ema_model_audio_audit.py")
    a.p.check_metadata(doc)
    fixed = {"purpose": PURPOSE, "source_step": 4500, "limit": 5000,
             "additional_updates": 500, "stages": [4750, 5000], "source_sha256": c.source.SOURCE_SHA,
             "single_raw_original_Adam": True, "EMA_feedback": False,
             "training_authorized": True, "release_selection": "NONE", "deployment": False}
    require(all(c.equal(doc.get(k), v) for k, v in fixed.items()), "Complete fixed500 activation identity")
    require(type(doc.get("bindings_sha256")) is dict and doc["bindings_sha256"]
            and doc["bindings_sha256"].get(str(Path(__file__).resolve())) == sha(__file__), "Exact executed trainer binding")
    for name, expected in doc["bindings_sha256"].items():
        require(sha(name) == expected, "Activation binding changed: " + name)
    required = ("actual_zero_update_audio_audit", "strict_source_CUDA_runtime_RNG",
                "actual_CPU_CUDA_first_raw_original_Adam_bit_identity", "actual_EMA_noalias",
                "complete_disk_replay_and_joint_CPU_CUDA_RNG_rollback", "cross_device_input_target_PCM_metadata",
                "continuous_engine_units", "original_DEVELOPMENT_evaluator_verified",
                "per_draw_source_and_native_runtime_guard", "complete_dependency_review",
                "human_training_authorization_record")
    require(type(doc.get("gates")) is dict and set(doc["gates"]) == set(required), "No missing real activation gates")
    for key in required:
        row = doc["gates"][key]
        require(type(row) is dict and row.get("passed") is True and type(row.get("path")) is str
                and doc["bindings_sha256"].get(row["path"]) == row.get("sha256"), "Unbound actual gate: " + key)
    return c.portable(doc | {"activation_path": str(path), "activation_file_sha256": sha(path)})


def selected_draw_files(true, dataset, recipe, sampler, cursor):
    """Only reconstruct original local selection; no audio/crop or RNG draw."""
    require(type(cursor) is int and 4500 <= cursor < 5000, "Contiguous bounded TRAIN counter")
    selected, bindings = [], {}
    for domain in live.DOMAINS[:3]:
        rng = random.Random(int(hashlib.sha256(f'{sampler["seed"]}:true:{domain}:{cursor}'.encode()).hexdigest(), 16))
        pool = true.pools[domain]
        require(pool and all(r["role"] == "train" and r["domain"] == domain for r in pool), "Original locked TRAIN pool")
        row = pool[rng.randrange(len(pool))]
        selected.append(row["track_id"])
        for name in sorted({p for field in ("mix_files", "vocal_files", "stem_files") for p in row.get(field, [])}):
            bindings[name] = true.lock["files"][name]["sha256"]
    for slot in range(3):
        crop = recipe(dataset.rows, dataset.config, sampler["seed"], cursor*3+slot)
        row = dataset.by_id[crop["song_id"]]
        require(row["role"] == "pseudo_label_train_candidate" and row["training_eligible"] is False,
                "Mel pseudo target is not final truth")
        selected.append(row["song_id"])
        for info in (row["source"], *row["label_files"].values()):
            name, digest = info["path"], info["sha256"]
            require(name not in bindings or bindings[name] == digest, "Conflicting source/target bytes")
            bindings[name] = digest
    return selected, bindings


class SourceReadLease:
    """Deny WRITE/DELETE of selected actual source bytes during ONE draw.

    Full SHA is checked from held handles, before and after. No directory or
    adversarial-OS guarantee, no native-dependency discovery by basename.
    Native-image supervision remains a separate whole-worker requirement.
    """
    def __init__(self, bindings, native):
        self.bindings, self.native, self.handles = dict(bindings), native, []
        self.before = []
        native.dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                          wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        native.dll.CreateFileW.restype = wintypes.HANDLE

    def __enter__(self):
        try:
            for name, expected in self.bindings.items():
                handle = self.native.dll.CreateFileW(name, 0x80000000, 1, None, 3, 0x80, None)
                require(handle not in (None, 0, ctypes.c_void_p(-1).value), "Cannot hold selected source: " + name)
                self.handles.append(handle)
                info = self.native.measure(handle)
                require(info["sha256"] == expected, "Selected source changed BEFORE draw: " + name)
                self.before.append(info)
            return self
        except BaseException:
            self.close()
            raise

    def check(self):
        require(len(self.handles) == len(self.bindings) == len(self.before), "Full selected-source lease required")
        for handle, expected in zip(self.handles, self.before):
            require(c.equal(self.native.measure(handle), expected), "Held source physical identity/bytes changed")

    def close(self):
        for handle in reversed(self.handles):
            self.native.checked("CloseHandle", handle)
        self.handles.clear()

    def __exit__(self, typ, value, trace):
        try:
            if typ is None:
                self.check()
        finally:
            self.close()


class GuardedAudioStream:
    """New continuous actual route; no closed208 constructor or draw replay."""
    def __init__(self, route, backend, decoder, native, allowed):
        require(type(route) is e.DeviceInputStream and route.device == "cuda", "Actual CUDA-aware single Mel route")
        self.route, self.backend, self.decoder, self.native, self.allowed = route, backend, decoder, native, allowed
        self._source = c.portable(route._source)
        self._static = backend._backend_identity()
        self._functions = backend._function_identity()
        self._cache = backend._cache_identity()
        self._backend_identity = {"route": c.portable(route._backend_identity), "complete_original_backend": c.portable(self._static),
                                  "decoder_manifest": c.portable(decoder.manifest), "per_draw_source_lease": True}
        self.draw_receipts = []  # Exposure evidence, not rollback-able cache/training state.

    @property
    def cursor(self):
        return self.route.cursor

    @cursor.setter
    def cursor(self, value):
        self.route.cursor = value

    @property
    def last_metadata(self):
        return self.route.last_metadata

    @last_metadata.setter
    def last_metadata(self, value):
        self.route.last_metadata = value

    @property
    def poisoned(self):
        return self.route.poisoned

    @poisoned.setter
    def poisoned(self, value):
        self.route.poisoned = value

    def _guard(self):
        self.allowed(); self.route._guard(); self.decoder.guard()
        require(c.equal(self.backend._backend_identity(), self._static)
                and self.backend._function_identity() == self._functions, "Actual backend functions/rows/config changed")
        require(c.equal(self.backend._cache_identity(), self._cache), "Foreign LRU PCM/signature bytes changed")

    def state_dict(self):
        self._guard()
        return self.route.state_dict()

    def next_batch(self):
        self._guard()
        cursor = self.cursor
        selected, bindings = selected_draw_files(self.route.true, self.route.dataset, self.route.recipe, self._source, cursor)
        receipt = {"cursor": cursor, "selected_ids": selected, "source_bindings_sha256": bindings,
                   "started": True, "completed": False}
        self.draw_receipts.append(receipt)
        try:
            with SourceReadLease(bindings, self.native) as held:
                result = self.route.next_batch()
                actual = [r.get("track_id") if i < 3 else r.get("song_id") for i, r in enumerate(result["metadata"])]
                require(actual == selected, "Pure original selection differs from actual draw")
                self._cache = self.backend._cache_identity()
                require(len(self.route.true.cache) <= 3 and len(self.route.dataset.cache) <= 1, "Original bounded LRUs")
                held.check(); self._guard()
                receipt["completed"] = True
                receipt["input_target_packet_digest"] = c.digest(result)
                return result
        except BaseException:
            self.poisoned = True
            raise


class ContinuousTrajectory(e.SingleTrajectoryEngine):
    """Same single original Adam update, now with all500/stage boundaries.

    Nested212 packet is explicitly a state-storage primitive, not an approval.
    New219 envelope binds activation and whole raw/EMA development receipts to
    the same atomic raw/Adam/EMA/input/context/all-RNG disk transaction.
    """
    def __init__(self, model, optimizer, parent, context, stream, backward, *, activation=None, provenance=None):
        self._activation = c.portable(activation)
        self.development_receipts = []
        self._actual = activation is not None
        if not self._actual:
            e.SingleTrajectoryEngine.__init__(self, model, optimizer, parent, context, stream, backward,
                                             stop_provenance=provenance)
        else:
            require(type(stream) is GuardedAudioStream and stream.route.device == "cuda"
                    and type(optimizer) is torch.optim.Adam and type(backward) is e.OriginalBoundaryLoss,
                    "One actual CUDA raw/Adam/original loss with guarded real input")
            verified = checked_activation(activation["activation_path"])
            require(c.equal(verified, activation), "Full activation bytes/types differ")
            c.source.check_arm(parent["raw_arm"], parent["raw_arm"]["optimizer"]["param_groups"][0]["lr"])
            require(parent["source_sha256"] == c.source.SOURCE_SHA and parent["selected_arm"] == c.source.RAW_ARM, "Exact original control4500")
            self.model, self.optimizer, self.stream, self.backward = model, optimizer, stream, backward
            self.device, self._scope = "cuda", ACTUAL_SCOPE
            self.shadow = c.ema.EmaShadow(model, source_sha256=c.source.SOURCE_SHA)
            self._parent, self._parent_digest = c.portable(parent), c.digest(parent)
            self._layout, _, actual_device, _ = c.ema._model_layout(model)
            require(actual_device == torch.device("cuda:0") and len(list(model.parameters())) == 22
                    and len(list(model.modules())) == 27 and not list(model.buffers()), "Original actual architecture/device")
            self._parameters, self._names = list(model.parameters()), [n for n, _ in model.named_parameters()]
            self._group = c.portable(optimizer.state_dict()["param_groups"][0]); self._defaults = c.portable(optimizer.defaults)
            self._context_template, self._provenance = c.portable(context), e.j.provenance(parent)
            self.live = live.LiveContext(context, self._provenance)
            self._backend_identity, self._runtime = c.portable(stream._backend_identity), e.runtime_identity("cuda")
            require(c.equal(self._runtime, parent["parent_metadata"]["runtime"]), "Post-import exact source CUDA4 FP32 runtime")
            self._loss_identity = c.portable(backward.identity)
            self._objects = (model, optimizer, self.shadow, stream, self.live, backward)
            self._immutable = c.digest([self._scope, self.device, self._layout, self._names, self._group, self._defaults,
                                       self._context_template, self._provenance, self._backend_identity, self._loss_identity])
            self._poisoned, self._in_transaction = False, False
            self.exposure = {"updates_started": 0, "updates_committed": 0, "updates_failed": 0, "adam_started": 0, "adam_completed": 0}
        self._activation_digest = c.digest(self._activation)
        self.state_dict()

    def _guard(self):
        super()._guard()
        # Base constructor calls state_dict before final envelope initialization.
        if hasattr(self, "_activation_digest"):
            require(c.digest(self._activation) == self._activation_digest, "Immutable activation changed")
        if self._actual:
            require(self._activation["training_authorized"] is True
                    and sha(self._activation["activation_path"]) == self._activation["activation_file_sha256"], "Formal authority changed")

    def _raw_packet(self, step):
        layout, values, _, modes = c.ema._model_layout(self.model)
        require(c.equal(layout, self._layout), "Raw architecture changed")
        for name, parameter in self.model.named_parameters():
            grad = parameter.grad
            require(grad is None or type(grad) is torch.Tensor and grad.device == parameter.device
                    and grad.dtype == torch.float32 and grad.shape == parameter.shape and not grad.requires_grad
                    and grad.grad_fn is None and bool(torch.isfinite(grad).all()), "Complete detached gradient " + name)
        return {"tensors": c.portable(values), "gradients": OrderedDict((n, None if v.grad is None else c.portable(v.grad))
                for n, v in self.model.named_parameters()), "modes": modes, "parameter_names": list(self._names),
                "optimizer": c.portable(self.optimizer.state_dict()), "optimizer_defaults": c.portable(self.optimizer.defaults), "updates": step}

    def validate(self, packet):
        if packet.get("purpose") == e.PURPOSE:
            # Only the base state_dict's internal primitive validation uses this.
            return e.SingleTrajectoryEngine.validate(self, packet)
        c.check_seal(packet)
        require(list(packet) == ["schema", "purpose", "scope", "activation", "trajectory", "development_receipts",
                                "training_authorized", "release_selection", "content_sha256"], "Full219 envelope schema")
        require(packet["schema"] == 1 and type(packet["schema"]) is int and packet["purpose"] == PURPOSE
                and packet["scope"] == (ACTUAL_SCOPE if self._actual else FIXTURE_SCOPE)
                and c.equal(packet["activation"], self._activation) and packet["training_authorized"] is self._actual
                and packet["release_selection"] == "NONE", "Full219 identity/types")
        trajectory = packet["trajectory"]
        e.SingleTrajectoryEngine.validate(self, trajectory)
        receipts = packet["development_receipts"]
        journal = trajectory["live_context"]["journal"]
        require(type(receipts) is list and len(receipts) == len(journal), "Coupled full DEV receipts/journal")
        for receipt, row in zip(receipts, journal):
            self._validate_score(receipt)
            require(receipt["step"] == row["step"] and c.equal(receipt["scores"]["raw"], row["raw_policy_score"]),
                    "Only actual raw policy score feeds original patience")
        c.noalias(packet, self._external())

    def state_dict(self):
        primitive = e.SingleTrajectoryEngine.state_dict(self)
        packet = c.seal({"schema": 1, "purpose": PURPOSE, "scope": ACTUAL_SCOPE if self._actual else FIXTURE_SCOPE,
                         "activation": c.portable(self._activation), "trajectory": primitive,
                         "development_receipts": c.portable(self.development_receipts),
                         "training_authorized": self._actual, "release_selection": "NONE"})
        self.validate(packet)
        return packet

    def load_state_dict(self, packet):
        require(packet.get("purpose") == PURPOSE, "A bare component or diagnostic packet is not a219 checkpoint")
        return super().load_state_dict(packet)

    @torch.no_grad()
    def _apply(self, packet):
        e.SingleTrajectoryEngine._apply(self, packet["trajectory"])
        self.development_receipts = c.portable(packet["development_receipts"])

    def _validate_score(self, packet):
        c.check_seal(packet)
        require(type(packet.get("step")) is int and packet["step"] in live.STAGES
                and packet.get("purpose") == "NONRELEASE_EMA_RAW_SHADOW_OLD_DEVELOPMENT"
                and packet.get("scope") == ("actual_original31track177view_DEV" if self._actual else "synthetic_only_NOT_real_DEV")
                and packet.get("release_selection") == "NONE" and set(packet.get("scores", {})) == {"raw", "ema"}
                and set(packet.get("model_digest", {})) == {"raw", "ema"}, "Complete stage pair; not release or loss selection")
        live.checked_score(packet["scores"]["raw"]); live.checked_score(packet["scores"]["ema"])
        if self._actual:
            require(packet.get("evaluator_binding") == self._activation["evaluator_binding"], "Original DEV policy/source binding")
            require(packet.get("suite_sha256") == "e5990cdd2faf71e83469cbef7f94dea6b80d0b376ba49d5b81bca69fe442fb07"
                    and set(packet.get("evaluations", {})) == {"raw", "ema"}
                    and all(len(v["rows"]) == 177 for v in packet["evaluations"].values()), "Whole unchanged old DEVELOPMENT suite")

    def observe_stage(self, packet):
        self._guard(); self._validate_score(packet)
        require(self.live.validation_due and packet["step"] == self.live.step, "One complete due4750/5000 observation")
        before = self.state_dict()["trajectory"]
        require(packet["model_digest"] == {"raw": c.digest(before["raw"]["tensors"]),
                                           "ema": c.digest(before["shadow"]["tensors"])}, "DEV evaluated these exact committed snapshots")
        with self.transaction():
            self.live.observe_raw(packet["scores"]["raw"])
            self.development_receipts.append(c.portable(packet))
        return self.state_dict()

    def update_next(self, allowed=lambda: None, *, after_joint_mutation=None):
        self._guard(); allowed()
        lr = self.live.learning_rate_next()  # Reject due DEV/hard5000 BEFORE a draw.
        self.exposure["updates_started"] += 1
        started = time.monotonic()
        try:
            with self.transaction():
                batch = self.stream.next_batch()
                require(batch["cursor"] == self.live.step and c.equal(batch["domains"], live.DOMAINS), "One original six-slot input")
                original = c.digest(batch)
                self.model.train(); self.optimizer.param_groups[0]["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                losses = self.backward(self.model, batch, self.device)
                c.typed_tree(losses); allowed()
                require(c.digest(batch) == original and all(v.grad is not None and v.grad.device == v.device
                        and v.grad.dtype == torch.float32 and bool(torch.isfinite(v.grad).all()) for v in self.model.parameters()),
                        "Complete original raw gradients and unchanged input")
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3, error_if_nonfinite=True)
                allowed(); self.exposure["adam_started"] += 1
                if self.device == "cuda":
                    self.optimizer.step()  # Original CUDA Adam path; never adapted.
                else:
                    e.cpu_adam_step(self.optimizer)
                self.exposure["adam_completed"] += 1
                self.optimizer.zero_grad(set_to_none=True)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                allowed(); self.complete_raw_storage_step()
                if after_joint_mutation is not None:
                    require(not self._actual, "Fault callback is a fixture interface, not arbitrary formal training code")
                    after_joint_mutation(self)
                result = {"step": self.live.step, "lr": lr, "losses": c.portable(losses), "gradient_norm_before_clip": float(norm),
                          "metadata": c.portable(batch["metadata"]), "input_target_packet_digest": original,
                          "seconds": time.monotonic()-started, "formal_student_update": self._actual}
            self.exposure["updates_committed"] += 1
            return result
        except BaseException:
            self.exposure["updates_failed"] += 1
            raise


if __name__ == "__main__":
    print(json.dumps({"purpose": PURPOSE, "continuous_update_implementation": True,
                      "actual_training_worker_started": False, "activation_record_generated": False,
                      "formal_updates": 0, "release_selection": "NONE"}))
