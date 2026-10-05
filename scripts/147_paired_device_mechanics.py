"""FP32 paired CPU/CUDA update core; CLI permits synthetic mechanism tests ONLY.

No real audio/teacher loader, approved importer or formal-training switch exists.
Three logical updates, disk resume and injected second-arm failure are checked.
Original script143/checkpoints/protocols remain untouched. Outputs are NONRELEASE.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("device_mechanics", Path(__file__).with_name("143_paired_distillation_mechanics.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ARMS, ROOT, acq = m.ARMS, m.bulk.ROOT, m.pilot.acq
LIMIT = 3
PURPOSE = "NONRELEASE_SYNTHETIC_DEVICE_MECHANISM"
DEFAULT_OUT = ROOT / "results/paired_device_cpu_20261002_r2"


def formal_entry():
    raise ValueError("Formal training CLOSED: separately approved importer, truth roles and user authorization required")


def portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(portable(v) for v in value)
    return copy.deepcopy(value)


def finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def gpu_preflight():
    """Read-only, conservative guard BEFORE any CUDA initialization; no fallback."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15, check=True)
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("Need one known GPU with readable availability; no automatic device choice")
    index, free, utilization = (int(v.strip()) for v in rows[0])
    allowed = index == 0 and free >= 2300 and utilization <= 15
    return {"gpu_index": index, "free_mib": free, "utilization_percent": utilization,
            "minimum_free_mib": 2300, "maximum_utilization_percent": 15, "allowed": allowed,
            "note": "Point-in-time guard, not a GPU reservation or proof of no foreign process"}


@contextmanager
def deterministic_runtime(device):
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit cpu/cuda only; never silently fall back")
    old = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    prior_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device == "cuda":
        if torch.cuda.is_initialized():
            raise ValueError("Configure CUDA determinism before initialization in a fresh process")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[2:4]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[4:6]
        if device == "cuda":
            if prior_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = prior_workspace


def runtime_identity(device):
    result = {"device": device, "torch": str(torch.__version__), "numpy": str(np.__version__),
              "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
              "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32}
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("One available CUDA GPU required; no CPU substitution")
        result.update(cuda=str(torch.version.cuda), cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    return result


class SyntheticBatchStream:
    """Six generated waveforms; slot names mimic geometry, NOT dataset truth."""
    def __init__(self, seed=20261002):
        if type(seed) is not int or seed < 0:
            raise ValueError("Integer synthetic seed required")
        self.seed, self.cursor = seed, 0

    def state_dict(self):
        return {"source": "synthetic_only_v1", "seed": self.seed, "cursor": self.cursor}

    def load_state_dict(self, state):
        if (state.get("source") != "synthetic_only_v1" or state.get("seed") != self.seed or
            type(state.get("cursor")) is not int or not 0 <= state["cursor"] <= LIMIT):
            raise ValueError("Changed synthetic source/cursor")
        self.cursor = state["cursor"]

    def next_batch(self):
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.cursor)
        x = torch.randn(6, 2, 89856, generator=generator) * .025
        a, b = x * .2, x * .2
        a[2].zero_()
        b[2].zero_()
        b[3:] = x[3:] * .8
        result = {"x": x, "targets": dict(zip(ARMS, (a, b))), "domains": m.DOMAINS,
                  "cursor": self.cursor, "synthetic_only": True}
        self.cursor += 1
        return result


class DevicePairedEngine:
    def __init__(self, factory, protocol, binding, device="cpu", purpose="training", backward=None):
        if purpose != PURPOSE or device not in ("cpu", "cuda"):
            formal_entry()  # Reject before optimizer, model creation or CUDA query.
        self.config = m.validate_protocol(protocol)
        self.binding, self.device, self.step = copy.deepcopy(binding), device, 0
        self.runtime = runtime_identity(device)
        if (not self.runtime["deterministic"] or self.runtime["warn_only"] or self.runtime["cudnn_benchmark"] or
            not self.runtime["cudnn_deterministic"] or self.runtime["matmul_tf32"] or self.runtime["cudnn_tf32"]):
            raise ValueError("Strict deterministic runtime required")
        self.models = {arm: factory().to(device) for arm in ARMS}
        if not m.equal_state(portable(self.models[ARMS[0]].state_dict()), portable(self.models[ARMS[1]].state_dict())):
            raise ValueError("Different initializations")
        self.names = {arm: [name for name, _ in net.named_parameters()] for arm, net in self.models.items()}
        self.optimizers = {arm: torch.optim.Adam(net.parameters(), lr=self.config["learning_rate"], foreach=False, fused=False)
                           for arm, net in self.models.items()}
        self.schedule = m.SharedSchedule(self.config)
        self.wa = torch.from_numpy(m.core.t09.make_analysis_matrix()).to(device)
        self.gs = torch.from_numpy(m.core.t09.make_synthesis_matrix()).to(device)
        self.backward = backward or m.fit.backward_batch
        self.poisoned = False

    def _stream_guard(self, stream):
        if type(stream) is not SyntheticBatchStream or self.poisoned:
            raise ValueError("Synthetic-only stream required; poisoned engine must not continue")
        if runtime_identity(self.device) != self.runtime:
            raise ValueError("Numerical runtime changed")

    def state_dict(self, stream):
        self._stream_guard(stream)
        if stream.cursor != self.step or self.schedule.step != self.step:
            raise ValueError("Cannot checkpoint a partial pair")
        return {"schema": 1, "purpose": PURPOSE, "limit": LIMIT, "binding": copy.deepcopy(self.binding),
                "runtime": copy.deepcopy(self.runtime), "step": self.step, "sampler": stream.state_dict(),
                "schedule": self.schedule.state_dict(), "rng": portable(m.capture_rng(self.device)),
                "arms": {arm: {"model": portable(net.state_dict()), "optimizer": portable(self.optimizers[arm].state_dict()),
                               "parameter_names": list(self.names[arm]), "modes": [mod.training for mod in net.modules()],
                               "updates": self.step} for arm, net in self.models.items()}}

    def _validate(self, state):
        if (state.get("schema") != 1 or state.get("purpose") != PURPOSE or state.get("limit") != LIMIT or
            state.get("binding") != self.binding or state.get("runtime") != self.runtime or
            type(state.get("step")) is not int or not 0 <= state["step"] <= LIMIT or set(state.get("arms", {})) != set(ARMS) or
            state["sampler"]["cursor"] != state["step"] or state["schedule"]["step"] != state["step"] or not finite_state(state)):
            raise ValueError("Changed/partial/nonfinite paired state")
        candidate_schedule = m.SharedSchedule(self.config)
        candidate_schedule.load_state_dict(state["schedule"])
        for arm in ARMS:
            saved, current = state["arms"][arm], self.models[arm].state_dict()
            if (saved["updates"] != state["step"] or saved["parameter_names"] != self.names[arm] or
                set(saved["model"]) != set(current) or any(saved["model"][k].shape != v.shape or saved["model"][k].dtype != v.dtype for k, v in current.items()) or
                len(saved["modes"]) != len(list(self.models[arm].modules())) or any(type(v) is not bool for v in saved["modes"])):
                raise ValueError("Model/parameter order/exposure changed")
            optimizer = saved["optimizer"]
            groups = optimizer["param_groups"]
            expected_group = self.optimizers[arm].state_dict()["param_groups"][0]
            if len(groups) != 1 or groups[0]["params"] != expected_group["params"]:
                raise ValueError("Adam parameter layout changed")
            for name in expected_group:
                if name not in ("lr", "params") and groups[0].get(name) != expected_group[name]:
                    raise ValueError("Adam configuration changed")
            expected_lr = m.learning_rate(state["step"], self.config) if state["step"] else self.config["learning_rate"]
            if groups[0]["lr"] != expected_lr:
                raise ValueError("Adam LR differs from shared step")
            expected_ids = set(groups[0]["params"]) if state["step"] else set()
            if set(optimizer["state"]) != expected_ids:
                raise ValueError("Missing Adam moments; weights-only is not resume")
            for index, values in optimizer["state"].items():
                if index not in groups[0]["params"] or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("Malformed Adam moments")
                parameter = dict(self.models[arm].named_parameters())[self.names[arm][index]]
                if (float(values["step"]) != state["step"] or values["exp_avg"].shape != parameter.shape or
                    values["exp_avg_sq"].shape != parameter.shape or values["exp_avg"].dtype != parameter.dtype or
                    values["exp_avg_sq"].dtype != parameter.dtype or bool((values["exp_avg_sq"] < 0).any())):
                    raise ValueError("Adam moments/step mismatch")

    def _apply(self, state, stream):
        stream.load_state_dict(state["sampler"])
        self.schedule.load_state_dict(state["schedule"])
        for arm in ARMS:
            self.models[arm].load_state_dict(state["arms"][arm]["model"], strict=True)
            self.optimizers[arm].load_state_dict(state["arms"][arm]["optimizer"])
            self.optimizers[arm].zero_grad(set_to_none=True)
            for mod, mode in zip(self.models[arm].modules(), state["arms"][arm]["modes"]):
                mod.training = mode
        self.step = state["step"]
        m.restore_rng(state["rng"], self.device)

    def load_state_dict(self, state, stream):
        self._stream_guard(stream)
        self._validate(state)
        previous = self.state_dict(stream)
        try:
            self._apply(state, stream)
        except BaseException:
            try:
                self._apply(previous, stream)
            except BaseException:
                self.poisoned = True
                raise
            raise

    def update_next(self, stream):
        if self.step >= LIMIT:
            raise ValueError("Synthetic update cap reached; no formal training escalation")
        previous = self.state_dict(stream)
        started = time.perf_counter()
        try:
            batch = stream.next_batch()
            x = batch["x"]
            if (batch.get("synthetic_only") is not True or batch["cursor"] != self.step or tuple(batch["domains"]) != m.DOMAINS or
                x.shape != (6, 2, 89856) or x.dtype != torch.float32 or x.device.type != "cpu" or
                set(batch["targets"]) != set(ARMS) or any(v.shape != x.shape or v.dtype != x.dtype or v.device.type != "cpu" for v in batch["targets"].values()) or
                not finite_state(batch) or not torch.equal(batch["targets"][ARMS[0]][:3], batch["targets"][ARMS[1]][:3])):
                raise ValueError("Invalid same-input synthetic batch")
            lr, losses = m.learning_rate(self.step + 1, self.config), {}
            for arm in ARMS:
                net, optimizer = self.models[arm], self.optimizers[arm]
                net.train()
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                losses[arm] = self.backward(net, x, batch["targets"][arm], self.wa, self.gs, self.device, 96, 44, "reconstruction_l1", 1)
                if not finite_state(losses[arm]):
                    raise ValueError("Nonfinite loss")
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                if not finite_state(net.state_dict()) or not finite_state(optimizer.state_dict()):
                    raise ValueError("Nonfinite model/Adam after step")
            self.step += 1
            self.schedule.complete_step()
            return {"step": self.step, "lr": lr, "losses": losses,
                    "input_sha256": [m.pilot.wave_digest(v) for v in x], "seconds": time.perf_counter() - started}
        except BaseException:
            try:
                self._apply(previous, stream)
                if not m.equal_state(previous, self.state_dict(stream)):
                    raise RuntimeError("Rollback state differs")
            except BaseException:
                self.poisoned = True
                raise
            raise


def binding(protocol):
    recipe = ("143_paired_distillation_mechanics.py", "136_prepare_distillation_data.py", "134_generate_teacher_library.py",
              "131_run_teacher_pilot.py", "130_acquire_teacher_pilot.py", "128_acquire_cambridge_candidates.py",
              "126_verify_training_baseline.py", "125_lock_training_data.py", "119_model_selection_suite.py",
              "110_train_residual_ablation.py", "109_audit_model_limits.py", "13_ab_compare.py", "11_smoke_train.py",
              "09_target_model.py", "23_build_true_stem_cache.py", "common.py")
    paths = [Path(__file__), m.data.PROTOCOL, ROOT / "docs/model_training_protocol_v1.json",
             ROOT / "models/student_bott2_mir1k_candidate.pt", *(ROOT / "scripts" / name for name in recipe)]
    return {"files": {str(p): acq.sha256(p) for p in paths}, "protocol_sha256": acq.content_digest(protocol), "data": "generated_synthetic_only"}


def verify(out):
    report = acq.read_sealed(out / "device_mechanism.json")
    if report.get("formal_student_training") is not False or report.get("checkpoint_selected") != "NONE":
        raise ValueError("Not a NONRELEASE mechanism report")
    for path, expected in report["binding"]["files"].items():
        if acq.sha256(path) != expected:
            raise ValueError("Mechanism source binding changed")
    for name, digest in report["checkpoints"].items():
        state = m.load_checked_checkpoint(out / name, digest)
        if state.get("purpose") != PURPOSE or state.get("binding") != report["binding"] or state.get("limit") != LIMIT:
            raise ValueError("Changed mechanism checkpoint")
    if report["updates_per_arm"] != LIMIT or not report["disk_resume_identical"] or not report["second_arm_failure_rollback_identical"]:
        raise ValueError("Incomplete mechanism evidence")
    print(f"PAIRED_DEVICE_VERIFY PASS device={report['runtime']['device']} synthetic_only", flush=True)


def smoke(out, device):
    m.bulk.guard_output(out)
    if out.exists() or shutil.disk_usage(out.parent).free < 12 * 1024**3 + 100 * 1024**2:
        raise ValueError("Fresh output and >=12 GiB reserve required")
    preflight = gpu_preflight() if device == "cuda" else None
    out.mkdir(parents=True)
    if preflight is not None:
        acq.write_new_json(out / "gpu_preflight.json", acq.seal(preflight | {"cuda_initialized": False, "trainer_sha256": acq.sha256(__file__)}))
        if not preflight["allowed"]:
            print("PAIRED_DEVICE DEFERRED GPU busy/low free memory; no CUDA init or optimizer", flush=True)
            return 2
    protocol = json.loads(m.data.PROTOCOL.read_text(encoding="utf-8"))
    bound = binding(protocol)
    with deterministic_runtime(device):
        factory = m.frozen_factory(protocol)
        engine = DevicePairedEngine(factory, protocol, bound, device, PURPOSE)
        stream, hashes, draws = SyntheticBatchStream(), {}, []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(LIMIT):
            draws.append(engine.update_next(stream))
            path = out / f"NONRELEASE_synthetic_pair_step_{engine.step:04d}.pt"
            m.save_new(path, engine.state_dict(stream))
            hashes[path.name] = acq.sha256(path)
            print(f"PAIRED_DEVICE committed_step={engine.step}/{LIMIT} device={device} seconds={draws[-1]['seconds']:.3f}", flush=True)
        expected = engine.state_dict(stream)
        restored, other = DevicePairedEngine(factory, protocol, bound, device, PURPOSE), SyntheticBatchStream()
        name = "NONRELEASE_synthetic_pair_step_0001.pt"
        restored.load_state_dict(m.load_checked_checkpoint(out / name, hashes[name]), other)
        resumed = [restored.update_next(other) for _ in range(2)]
        comparable = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]
        if comparable(resumed) != comparable(draws[1:]) or not m.equal_state(expected, restored.state_dict(other)):
            raise ValueError("Device disk resume changed full state")
        restored.load_state_dict(m.load_checked_checkpoint(out / name, hashes[name]), other)
        before = restored.state_dict(other)
        original = restored.backward
        def fail_second(net, *args):
            if net is restored.models[ARMS[1]]:
                random.random(); np.random.random(); torch.rand(3, device=device)
                raise RuntimeError("injected second-arm failure")
            return original(net, *args)
        restored.backward = fail_second
        try:
            restored.update_next(other)
        except RuntimeError as error:
            if str(error) != "injected second-arm failure":
                raise
        else:
            raise ValueError("Failure injection not executed")
        if not m.equal_state(before, restored.state_dict(other)):
            raise ValueError("Second-arm failure changed paired state")
        report = {"schema": 1, "binding": bound, "runtime": engine.runtime, "updates_per_arm": LIMIT,
                  "synthetic_only": True, "formal_student_training": False, "deployment": False,
                  "checkpoint_selected": "NONE", "disk_resume_identical": True,
                  "second_arm_failure_rollback_identical": True, "draws": draws, "checkpoints": hashes,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                  "scope": "Same-backend mechanism only; no quality, cross-device bit-exactness or long-run throughput claim"}
        acq.write_new_json(out / "device_mechanism.json", acq.seal(report))
    verify(out)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--formal-train", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.formal_train or not (args.synthetic_smoke or args.verify):
        formal_entry()
    torch.set_num_threads(4)
    if args.verify:
        verify(args.out)
        return 0
    return smoke(args.out, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
