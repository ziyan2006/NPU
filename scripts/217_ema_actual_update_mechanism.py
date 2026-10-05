"""Bounded actual source4501 CPU/CUDA mechanism, NOT formal training.

Uses215's authenticated real PCM packet, no new audio draw. One unique input;
main/reference/disk replay/fault repetitions are not independent examples.
All246 previously observed native physical files are held before each fresh
worker; unknown conditional loads fail closed, with actual exposure retained.
No closed historical CLI, training PT deserialize, DEV or teacher inference.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA217_ACTUAL_SOURCE_ONE_UNIQUE_INPUT_UPDATE_MECHANISM"
RUNTIME = "results/ema216_cuda_runtime_observation_attempt01/runtime_observation.json"
REVIEW = "results/ema216_cuda_runtime_observation_attempt01/independent_readonly_review.json"
ARTIFACT = "results/ema215_real_model_audio_audit_attempt01/audit_result.pt"
PINS = {
    "scripts/216_ema_cuda_runtime_observation.py": "a933754eb194b1e2661ebc72c4a7667d441a431a86dff01a435d4f7a77264003",
    RUNTIME: "9ddd58826ce617ee345e22856a4e524fdf9e055d9ae940a9a75c0c3e4a80015f",
    REVIEW: "bc3954c9f0ce9f6bbbd8d3d68c436d6dddf1874b30dc794589bb9c7cc1cff89f",
    ARTIFACT: "95df81b34df7ef454d935d1293d452e0cc8f22f25213b9f3c07d150939f6aef5",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_files():
    for relative, expected in PINS.items():
        require(sha(ROOT / relative) == expected, "Changed217 gate: " + relative)


check_files()
r = load("ema217_actual_runtime_gate", "scripts/216_ema_cuda_runtime_observation.py")
a, p, g, obs = r.a, r.p, r.g, r.obs


def gate():
    check_files()
    audit = r.gate()
    doc = json.loads((ROOT / RUNTIME).read_text(encoding="utf-8"))
    review = json.loads((ROOT / REVIEW).read_text(encoding="utf-8"))
    p.check_metadata(doc); p.check_metadata(doc["runtime_report"]); p.check_metadata(review)
    require(doc["error"] is None and doc["launcher_exit_code"] == 0
            and all(row["exit_code"] == 0 for row in doc["processes"].values())
            and doc["all_event_handles_post_exit_equal"] is True
            and doc["runtime_report"]["runtime_equal_immutable_source"] is True
            and doc["runtime_report"]["original_complete_RNG_restore_bit_type_equal"] is True,
            "Complete actual source runtime/RNG gate")
    # Own independent review is SHA bound; no flattening/stripping nested seals.
    require(review["purpose"] and doc["training_authorized"] is False, "Runtime gate never formal authority")
    return audit, doc


def native_manifest(audit, runtime):
    import copy
    known = copy.deepcopy(audit["preheld_native_physical_files"])
    for name, record in runtime["newly_observed_native_files"].items():
        require(name not in known or obs.meta.typed(known[name]) == obs.meta.typed(record), "Conflicting CUDA physical file")
        known[name] = copy.deepcopy(record)
    require(len(known) == 246, "Exact actual observed246 files, not basename resolution")
    return known


def device_scope(device):
    require(type(device) is str and device in ("cpu", "cuda"), "Explicit bounded CPU or CUDA; no fallback")
    return "actual_source_217_" + device + "_one_input_not_formal"


def validate_exposure(exposure):
    require(type(exposure) is dict and all(type(v) is int and v >= 0 for v in exposure.values()), "Exact nonnegative exposure counters")
    require(exposure["forward_started"] == exposure["forward_completed"] == 24
            and exposure["backward_completed"] == 24 and exposure["adam_completed"] == 4
            and exposure["adam_started"] == 4, "Main, replay, fault, independent raw reference exposure")


class InjectedAfterCommit(BaseException):
    """Deliberate bounded fault AFTER raw/Adam/shadow/input/context mutation."""


def mechanism_types(e):
    """Heavy helpers only in explicit worker, never lightweight supervisor."""
    from collections import OrderedDict
    torch, c, live = e.torch, e.c, e.live

    class RecordedStream:
        def __init__(self, sampler, batch, device):
            self._source = c.portable(sampler)
            self.batch = c.portable(batch)
            self.device = device
            require(batch["cursor"] == sampler["cursor"] == 4500 and c.equal(batch["domains"], live.DOMAINS), "Only audited counter4500")
            live.validate_metadata(batch["metadata"])
            for field in ("x", "v"):
                c.source.check_tensor(batch[field], torch.float32, (6, 2, 89856), "Recorded actual PCM " + field)
            hashes = [hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for v in batch["x"]]
            require(hashes == [row["input_pcm_sha256"] for row in batch["metadata"]], "Audited PCM identity")
            self._digest = c.digest(self.batch)
            self._backend_identity = {"kind": "215_actual_PCM_disk_packet_NOT_new_audio_draw", "digest": self._digest,
                                      "artifact_sha256": PINS[ARTIFACT], "unique_input_counters": [4500]}
            self.cursor, self.last_metadata, self.poisoned = 4500, None, False

        def _guard(self):
            require(not self.poisoned and c.digest(self.batch) == self._digest
                    and self._backend_identity["digest"] == self._digest, "Recorded input poisoned or mutated")

        def state_dict(self):
            self._guard()
            state = self._source | {"cursor": self.cursor}
            live.validate_sampler(state, self._source)
            return c.portable(state)

        def next_batch(self):
            self._guard()
            require(self.cursor == 4500, "ONE unique fixed input only; not a formal500-step stream")
            before = e.capture_rng(self.device)
            result = c.portable(self.batch)
            require(c.equal(before, e.capture_rng(self.device)), "Disk PCM copy changed complete RNG")
            self.cursor = 4501
            self.last_metadata = c.portable(result["metadata"])
            return result

    class Owner(e.SingleTrajectoryEngine):
        def __init__(self, model, optimizer, parent, context, stream, backward, device):
            e.check_dependencies()
            require(type(stream) is RecordedStream and type(optimizer) is torch.optim.Adam, "One actual raw/Adam and authenticated recorded PCM")
            require(parent["source_sha256"] == c.source.SOURCE_SHA and parent["selected_arm"] == c.source.RAW_ARM, "Fixed control4500 source")
            c.source.check_arm(parent["raw_arm"], parent["raw_arm"]["optimizer"]["param_groups"][0]["lr"])
            self.model, self.optimizer, self.stream, self.backward = model, optimizer, stream, backward
            self.device, self._scope = device, device_scope(device)
            self.shadow = c.ema.EmaShadow(model, source_sha256=c.source.SOURCE_SHA)
            self._parent, self._parent_digest = c.portable(parent), c.digest(parent)
            self._layout, _, actual_device, _ = c.ema._model_layout(model)
            require(actual_device == torch.device("cuda:0" if device == "cuda" else "cpu")
                    and len(list(model.parameters())) == 22 and len(list(model.modules())) == 27
                    and not list(model.buffers()) and all(v.requires_grad for v in model.parameters()), "Original actual raw architecture/device")
            self._parameters, self._names = list(model.parameters()), [n for n, _ in model.named_parameters()]
            self._group = c.portable(optimizer.state_dict()["param_groups"][0])
            self._defaults = c.portable(optimizer.defaults)
            self._context_template, self._provenance = c.portable(context), e.j.provenance(parent)
            self.live = live.LiveContext(context, self._provenance)
            self._backend_identity, self._runtime = c.portable(stream._backend_identity), e.runtime_identity(device)
            self._loss_identity = c.portable(backward.identity)
            self._objects = (model, optimizer, self.shadow, stream, self.live, backward)
            self._immutable = c.digest([self._scope, self.device, self._layout, self._names, self._group, self._defaults,
                                       self._context_template, self._provenance, self._backend_identity, self._loss_identity])
            self._poisoned, self._in_transaction = False, False
            self.exposure = {"updates_started": 0, "updates_committed": 0, "updates_failed": 0, "adam_started": 0, "adam_completed": 0}
            self.state_dict()

        def _raw_packet(self, step):
            layout, values, _, modes = c.ema._model_layout(self.model)
            require(c.equal(layout, self._layout), "Actual raw layout changed")
            for name, parameter in self.model.named_parameters():
                grad = parameter.grad
                require(grad is None or type(grad) is torch.Tensor and grad.device == parameter.device
                        and grad.dtype == torch.float32 and grad.shape == parameter.shape
                        and not grad.requires_grad and grad.grad_fn is None and bool(torch.isfinite(grad).all()), "Complete detached live gradient device/type " + name)
            return {"tensors": c.portable(values),
                    "gradients": OrderedDict((n, None if v.grad is None else c.portable(v.grad)) for n, v in self.model.named_parameters()),
                    "modes": modes, "parameter_names": list(self._names), "optimizer": c.portable(self.optimizer.state_dict()),
                    "optimizer_defaults": c.portable(self.optimizer.defaults), "updates": step}

        def update_once(self, allowed, exposure, *, inject=False):
            self._guard(); allowed()
            require(self.live.step == 4500, "Mechanism hard4501, not formal training activation")
            self.exposure["updates_started"] += 1
            try:
                with self.transaction():
                    batch = self.stream.next_batch()
                    require(batch["cursor"] == self.live.step, "Exactly one matching input")
                    original_input = c.digest(batch)
                    self.model.train(); self.optimizer.param_groups[0]["lr"] = self.live.learning_rate_next()
                    self.optimizer.zero_grad(set_to_none=True)
                    losses = self.backward(self.model, batch, self.device)
                    c.typed_tree(losses); allowed()
                    require(c.digest(batch) == original_input and all(v.grad is not None and v.grad.device == v.device
                            and v.grad.dtype == torch.float32 and bool(torch.isfinite(v.grad).all()) for v in self.model.parameters()), "Complete original gradients/input")
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3, error_if_nonfinite=True)
                    allowed(); exposure["adam_started"] += 1; self.exposure["adam_started"] += 1
                    if self.device == "cuda":
                        self.optimizer.step()
                    else:
                        e.cpu_adam_step(self.optimizer)
                    exposure["adam_completed"] += 1; self.exposure["adam_completed"] += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    allowed(); self.complete_raw_storage_step()
                    if inject:
                        import random
                        random.random(); e.np.random.random(); torch.rand(1)
                        if self.device == "cuda":
                            torch.rand(1, device="cuda"); torch.cuda.synchronize()
                        require(self.shadow.updates == 1 and self.live.step == 4501, "Fault after REAL joint mutation")
                        raise InjectedAfterCommit("actual4501 raw Adam EMA counts and CPU-CUDA RNG mutated")
                    result = {"losses": c.portable(losses), "gradient_norm_before_clip": float(norm)}
                self.exposure["updates_committed"] += 1
                return result
            except BaseException:
                self.exposure["updates_failed"] += 1
                raise

    return Owner, RecordedStream


def worker(out, device):
    gate(); device_scope(device)
    out = Path(out).resolve()
    require(out.parent == (ROOT / "results").resolve() and out.is_dir(), "Owned new mechanism output")
    exposure = {"forward_started": 0, "forward_completed": 0, "backward_completed": 0, "adam_started": 0, "adam_completed": 0}
    def allowed():
        require(not (out / "supervisor_refused.json").exists(), "Supervisor refused conditional native image")
    try:
        e = load("ema217_complete_trajectory_primitives", "scripts/212_ema_single_trajectory_engine.py")
        import torch
        import torch._dynamo
        from types import MethodType
        c = e.c
        torch.set_num_threads(4 if device == "cuda" else 2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        require(not torch.cuda.is_initialized(), "Fresh mechanism worker")
        require(sha(ROOT / ARTIFACT) == PINS[ARTIFACT], "SHA before one new audit proof read")
        proof = torch.load(ROOT / ARTIFACT, map_location="cpu", weights_only=True)
        require(sha(ROOT / ARTIFACT) == PINS[ARTIFACT], "Audit proof changed")
        c.check_seal(proof); c.check_seal(proof["full_before"]); c.check_seal(proof["full_after"])
        require(c.equal(proof["full_before"], proof["full_after"]) and proof["updates"] == 0
                and c.digest(proof["input_packet"]) == proof["input_packet_digest"], "Full accepted215 proof")
        source = proof["full_before"]
        parent, raw = source["parent"], source["raw"]
        context = source["live_context"]["source_context"]
        rng = c.portable(parent["parent_metadata"]["rng"])
        allowed()
        if device == "cuda":
            require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Exact source CUDA workspace")
            torch.cuda.init()
            require(c.equal(e.runtime_identity("cuda"), parent["parent_metadata"]["runtime"]), "Exact immutable source CUDA FP32 runtime")
        else:
            rng["torch_cuda"] = []  # Explicit CPU mechanism migration, NOT CUDA numerical resume.
        allowed()
        loss = e.OriginalBoundaryLoss(device)
        target = loss.function.__globals__["m"].core.t09
        def restore_model_optimizer():
            allowed()
            net = target.CausalSpectralUNet(bottleneck_blocks=2).to(device)
            net.load_state_dict(raw["tensors"], strict=True)
            for mod, mode in zip(net.modules(), raw["modes"]):
                mod.training = mode
            for name, param in net.named_parameters():
                param.grad = None if raw["gradients"][name] is None else raw["gradients"][name].to(device).clone()
            group = c.portable(raw["optimizer"]["param_groups"][0]); group.pop("params")
            optimizer = torch.optim.Adam(net.parameters(), **group)
            optimizer.load_state_dict(c.portable(raw["optimizer"]))
            optimizer.defaults.clear(); optimizer.defaults.update(c.portable(raw["optimizer_defaults"]))
            for param in net.parameters():
                values = optimizer.state[param]
                require(values["exp_avg"].device == param.device and values["exp_avg_sq"].device == param.device
                        and values["step"].device.type == "cpu", "Original noncapturable Adam state device placement")
            return net, optimizer
        Owner, Stream = mechanism_types(e)
        net, optimizer = restore_model_optimizer()
        stream = Stream(context["sampler"], proof["input_packet"], device)
        e.restore_rng(rng, device)
        owner = Owner(net, optimizer, parent, context, stream, loss, device)
        initial = owner.state_dict()
        require(c.equal(initial["raw"], raw) and c.equal(initial["shadow"], source["shadow"])
                and c.equal(initial["rng"], rng), "Whole actual source restore bits/types")
        def instrument(model):
            original = model.forward
            require("forward" not in model.__dict__, "No original forward override")
            def counted(instance, *args, **kwargs):
                allowed(); exposure["forward_started"] += 1
                result = original(*args, **kwargs)
                exposure["forward_completed"] += 1; allowed()
                def backward_done(gradient):
                    exposure["backward_completed"] += 1
                    return gradient
                result.register_hook(backward_done)
                return result
            model.forward = MethodType(counted, model)
        instrument(net)
        print("MECHANISM_PHASE source_restored_" + device, flush=True)
        before_sha = owner.save_new(out / "source4500_complete.pt")
        first = owner.update_once(allowed, exposure)
        after = owner.state_dict()
        require(after["shadow"]["updates"] == 1 and any(not c.equal(after["shadow"]["tensors"][n], after["raw"]["tensors"][n])
                for n in owner._names) and any(not c.equal(after["raw"]["tensors"][n], initial["raw"]["tensors"][n])
                for n in owner._names), "Actual nonzero raw update and distinct independent EMA")
        c.noalias(after, owner._external())
        after_sha = owner.save_new(out / "diagnostic4501_complete.pt")
        print("MECHANISM_PHASE first_update_saved_" + device, flush=True)
        # Independent raw original194 reference sequence, not a second training arm.
        reference, reference_optimizer = restore_model_optimizer()
        instrument(reference)
        e.restore_rng(rng, device)
        reference.train(); reference_optimizer.param_groups[0]["lr"] = owner._lr(4501)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_losses = loss.function(reference, proof["input_packet"]["x"], proof["input_packet"]["v"],
                                         loss.wa, loss.gs, device, proof["input_packet"]["metadata"], 32)
        allowed(); reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 3, error_if_nonfinite=True)
        exposure["adam_started"] += 1
        if device == "cuda":
            reference_optimizer.step()
        else:
            e.cpu_adam_step(reference_optimizer)
        exposure["adam_completed"] += 1
        reference_optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        require(c.equal(c.portable(reference.state_dict()), after["raw"]["tensors"])
                and c.equal(c.portable(reference_optimizer.state_dict()), after["raw"]["optimizer"])
                and c.equal(reference_optimizer.defaults, after["raw"]["optimizer_defaults"])
                and c.equal([m.training for m in reference.modules()], after["raw"]["modes"])
                and all(v.grad is None for v in reference.parameters())
                and c.equal(reference_losses, first["losses"]) and float(reference_norm) == first["gradient_norm_before_clip"]
                and c.equal(e.capture_rng(device), after["rng"]), "Original194 independent raw/Adam first update not bit identical")
        reference_digest = c.digest({"raw": c.portable(reference.state_dict()), "optimizer": c.portable(reference_optimizer.state_dict()),
                                     "rng": e.capture_rng(device)})
        del reference.forward
        del reference, reference_optimizer
        # Whole disk state restores using fresh tensors with SHA before deserialize.
        owner.load_state_dict(owner.read_checked(out / "source4500_complete.pt", before_sha))
        require(c.equal(owner.state_dict(), initial), "Full initial disk state restore")
        replay = owner.update_once(allowed, exposure)
        require(c.equal(owner.state_dict(), after) and c.equal(replay, first), "Full raw/Adam/EMA/input/context/RNG disk replay bits/types")
        owner.load_state_dict(owner.read_checked(out / "source4500_complete.pt", before_sha))
        try:
            owner.update_once(allowed, exposure, inject=True)
        except InjectedAfterCommit:
            pass
        else:
            raise AssertionError("Actual joint fault was not raised")
        require(c.equal(owner.state_dict(), initial) and not owner._poisoned,
                "Actual raw/Adam/EMA count/input/context/fullCPU-CUDA RNG rollback differs")
        rollback_sha = owner.save_new(out / "rolled_back4500_complete.pt")
        restored_after = owner.read_checked(out / "diagnostic4501_complete.pt", after_sha)
        owner.load_state_dict(restored_after)
        require(c.equal(owner.state_dict(), after), "Full post-update disk restore bits/types")
        validate_exposure(exposure); allowed()
        del net.forward
        report = p.seal_metadata({"purpose": PURPOSE, "device": device, "source_step": 4500,
            "logical_diagnostic_step": 4501, "unique_input_counters": [4500], "unique_slots": 6,
            "new_audio_draws": 0, "original_training_PT_reads": 0, "audit_proof_reads": 1,
            "real_model_Adam_constructions": 2, "exposure": exposure,
            "actual_source_runtime": e.runtime_identity(device), "input_packet_digest": proof["input_packet_digest"],
            "input_metadata_digest": c.digest(proof["input_packet"]["metadata"]),
            "initial_state_digest": c.digest(initial), "after_state_digest": c.digest(after),
            "source4500_disk_sha256": before_sha, "diagnostic4501_disk_sha256": after_sha, "rolled_back4500_disk_sha256": rollback_sha,
            "original194_first_raw_Adam_bit_equal": True, "reference_digest": reference_digest,
            "real_raw_and_EMA_nonzero_independent": True, "whole_disk_replay_bit_type_equal": True,
            "fault_after_real_Adam_and_EMA_count_and_CPU_CUDA_RNG_mutation": True,
            "whole_joint_rollback_bit_type_equal": True, "source_parent_CUDA_rng_digest": c.digest(parent["parent_metadata"]["rng"]),
            "CPU_explicit_migration_not_CUDA_resume": device == "cpu", "CUDA_initialized": torch.cuda.is_initialized(),
            "formal_training_updates": 0, "training_authorized": False, "release_selection": "NONE"})
        print("ACTUAL_MECHANISM_VERIFIED " + json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)
    except BaseException as error:
        obs.write_new(out / "worker_failure.json", p.seal_metadata({"purpose": PURPOSE, "device": device,
            "error": {"type": type(error).__name__, "message": str(error)}, "actual_exposure": exposure,
            "formal_training_updates": 0, "training_authorized": False}))
        raise


def collect(out, device):
    prior, runtime = gate(); device_scope(device)
    out = Path(out).resolve()
    require(out.parent == (ROOT / "results").resolve() and not out.exists(), "Fresh217 output, refuse overwrite/repeat")
    require(not any(n in sys.modules for n in ("torch", "numpy", "scipy", "soundfile")), "Light native supervisor only")
    preflight = r.resource()
    native, known = p.LargeNative(), native_manifest(prior, runtime)
    native.dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    native.dll.CreateFileW.restype = wintypes.HANDLE
    out.mkdir()
    sources, handles, active, breaks = [], [], set(), set()
    process, pending = None, None
    buffers, readers, errors = {"stdout": bytearray(), "stderr": bytearray()}, [], []
    row = {"purpose": PURPOSE, "device": device, "started_utc": obs.now(), "resource_preflight": preflight,
           "processes": {}, "error": None, "formal_training_updates": 0, "training_authorized": False}
    start = time.monotonic(); journal_path = out / "native_event_journal.jsonl"
    with journal_path.open("x", encoding="utf-8") as journal:
        def record(item):
            journal.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + "\n"); journal.flush()
        try:
            bindings = runtime["preheld_source_bindings_sha256"] | {str(ROOT / f): h for f, h in PINS.items()}
            bindings[str(Path(__file__).resolve())] = sha(__file__)
            for name, expected in bindings.items():
                handle = native.dll.CreateFileW(name, 0x80000000, 1, None, 3, 0x80, None)
                require(handle not in (None, 0, ctypes.c_void_p(-1).value), "Cannot prehold mechanism bytes: " + name)
                sources.append(handle); require(sha(name) == expected, "Changed bound mechanism bytes")
            row["preheld_source_bindings_sha256"], row["preheld_native_physical_files"] = bindings, known
            with g.HeldFiles(known, native) as held:
                startup = subprocess.STARTUPINFO(); startup.dwFlags, startup.wShowWindow = subprocess.STARTF_USESHOWWINDOW, 0
                argv = [sys.executable, "-B", "-X", "pycache_prefix=" + str(out / "unused_pycache"), str(Path(__file__).resolve()),
                        "worker", "--out", str(out), "--device", device]
                row["request"] = {"argv": argv, "executable": sys.executable, "shell": False, "DEBUG_PROCESS": True,
                                  "child_workspace_override": ":4096:8", "parent_environment_modified": False, "cwd": None}
                process = subprocess.Popen(argv, executable=sys.executable, shell=False, close_fds=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW | 1,
                    env=dict(os.environ) | {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, cwd=None)
                row["launcher_pid"] = process.pid; native.checked("DebugSetProcessKillOnExit", False)
                def consume(name):
                    try:
                        while True:
                            block = getattr(process, name).read(1024 * 1024)
                            if not block:
                                break
                            require(len(buffers[name]) + len(block) <= 16 * 1024**2, "Bounded mechanism metadata pipe")
                            buffers[name].extend(block)
                    except BaseException as error:
                        errors.append(error)
                for name in buffers:
                    thread = threading.Thread(target=consume, args=(name,), daemon=True); thread.start(); readers.append(thread)
                created, events = 0, 0
                while not created or active:
                    require(time.monotonic() - start < 900 and not errors, "Bounded900s mechanism")
                    event = obs.DebugEvent()
                    if not native.dll.WaitForDebugEventEx(ctypes.byref(event), 100):
                        require(ctypes.get_last_error() == 121, "Native mechanism wait failed"); continue
                    pending, events = event, events + 1
                    require(events <= 40000 and 1 <= event.code <= 8, "Known bounded native mechanism event")
                    item = {"sequence": events, "pid": event.pid, "tid": event.tid, "code": event.code}
                    if event.code == 3:
                        require(event.pid not in active and created < 4, "Owned new worker chain only")
                        active.add(event.pid); created += 1
                        row["processes"][str(event.pid)] = {"events": 0, "image_file_events": 0, "exit_code": None}
                    require(event.pid in active, "Only owned mechanism process")
                    current = row["processes"][str(event.pid)]; current["events"] += 1
                    if event.code in (3, 6):
                        info = event.data.create if event.code == 3 else event.data.load
                        require(info.file and info.file not in handles, "Unique actual loaded file handle")
                        handles.append(info.file); actual = native.measure(info.file); item["file"] = actual
                        current["image_file_events"] += 1
                        record({"kind": "prevalidation_image", "event": item})
                        a.validate_native(actual, known)
                        if event.code == 3:
                            p.check_executable_identity(actual, a.preparation()["pre_bound_python_executables"])
                        item["matched_preheld_physical_identity"] = True
                    elif event.code == 1:
                        item.update(exception_code=int(event.data.exception.record.code), first_chance=int(event.data.exception.first_chance))
                    elif event.code == 5:
                        current["exit_code"] = int(event.data.exit_code); active.remove(event.pid)
                    continuation, seen = obs.continuation(event, event.pid in breaks)
                    if seen:
                        breaks.add(event.pid)
                    item["continuation"] = continuation; record(item)
                    native.checked("ContinueDebugEvent", event.pid, event.tid, continuation); pending = None
                row["launcher_exit_code"] = process.wait(timeout=2)
                for thread in readers:
                    thread.join(timeout=2)
                require(all(not thread.is_alive() for thread in readers) and not errors, "Complete mechanism output consumed")
                require(row["launcher_exit_code"] == 0 and all(v["exit_code"] == 0 for v in row["processes"].values()), "Actual mechanism EXIT not0")
                reports = [json.loads(line.removeprefix("ACTUAL_MECHANISM_VERIFIED ")) for line in buffers["stdout"].decode("utf-8").splitlines()
                           if line.startswith("ACTUAL_MECHANISM_VERIFIED ")]
                require(len(reports) == 1, "One complete actual mechanism report")
                report = reports[0]; p.check_metadata(report); validate_exposure(report["exposure"])
                require(report["device"] == device and report["whole_joint_rollback_bit_type_equal"] is True
                        and report["formal_training_updates"] == 0 and report["training_authorized"] is False, "Actual mechanism scope")
                entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
                images = [v for v in entries if "sequence" in v and "file" in v]
                require(len(images) == len(handles), "Every actual file event retained")
                for handle, entry in zip(handles, images):
                    require(obs.meta.typed(native.measure(handle)) == obs.meta.typed(entry["file"]), "Native mechanism bytes changed through EXIT")
                held.check()
                for name, expected in bindings.items():
                    require(sha(name) == expected, "Source changed through mechanism EXIT")
                row.update(mechanism_report=report, native_file_events=len(images), all_event_handles_post_exit_equal=True,
                           actual_preheld_native_mechanism_verified=True, full_training_ready=False, completed_utc=obs.now())
        except BaseException as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
            record({"kind": "failure", "row": row})
            obs.write_new(out / "supervisor_refused.json", p.seal_metadata(row["error"]))
            if pending is not None:
                continuation, _ = obs.continuation(pending, pending.pid in breaks)
                native.checked("ContinueDebugEvent", pending.pid, pending.tid, continuation)
            for pid in list(active):
                native.checked("DebugActiveProcessStop", pid)
            row["detached_new_owned_pids_on_failure"] = list(active)
            raise
        finally:
            for handle in reversed(handles):
                native.checked("CloseHandle", handle)
            for handle in reversed(sources):
                native.checked("CloseHandle", handle)
            for name in buffers:
                (out / (name + ".bin")).write_bytes(buffers[name])
            obs.write_new(out / "mechanism_result.json", p.seal_metadata(row))
    print("ACTUAL_MECHANISM_ARCHIVE " + str(out / "mechanism_result.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("collect", "worker"))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.operation == "collect" else worker)(args.out, args.device)
