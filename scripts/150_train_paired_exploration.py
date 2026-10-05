"""Authorized fixed-24 exploratory paired training, never formal release/export.

Independent engine: does not remove old mechanism caps or modify old protocols.
Initial tranche hard-stops at 1000 shared steps; full-state commits every 250.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import time

import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inp = load("exploratory_import", "149_prepare_exploratory_import.py")
dev = load("exploratory_development", "146_evaluate_paired_development.py")
d = load("exploratory_device_helpers", "147_paired_device_mechanics.py")
m, acq, ROOT, ARMS, PURPOSE = inp.m, inp.acq, inp.ROOT, inp.m.ARMS, inp.PURPOSE
DEFAULT_OUT = ROOT / "results/paired_exploration_20261003"


class ExplorationEngine:
    def __init__(self, factory, source_protocol, binding, device, stream, backward=None):
        if type(stream) is not inp.ApprovedPairStream:
            raise ValueError("Approved exploration stream required before model/optimizer creation")
        inp.check_approval(stream.doc)
        if stream.cursor != 0 or device not in ("cpu", "cuda"):
            raise ValueError("Fresh approved stream and explicit backend required")
        self.limit = stream.doc["protocol"]["maximum_exploratory_steps"]
        self.config = m.validate_protocol(source_protocol)
        self.binding, self.device, self.step = copy.deepcopy(binding), device, 0
        self.runtime = d.runtime_identity(device)
        if (not self.runtime["deterministic"] or self.runtime["warn_only"] or self.runtime["cudnn_benchmark"] or
            not self.runtime["cudnn_deterministic"] or self.runtime["matmul_tf32"] or self.runtime["cudnn_tf32"]):
            raise ValueError("Strict deterministic FP32 runtime required")
        self.models = {arm: factory().to(device) for arm in ARMS}
        if not m.equal_state(d.portable(self.models[ARMS[0]].state_dict()), d.portable(self.models[ARMS[1]].state_dict())):
            raise ValueError("Different frozen initialization")
        self.names = {arm: [name for name, _ in net.named_parameters()] for arm, net in self.models.items()}
        self.optimizers = {arm: torch.optim.Adam(net.parameters(), lr=self.config["learning_rate"], foreach=False, fused=False)
                           for arm, net in self.models.items()}
        self.schedule = m.SharedSchedule(self.config)
        self.wa = torch.from_numpy(m.core.t09.make_analysis_matrix()).to(device)
        self.gs = torch.from_numpy(m.core.t09.make_synthesis_matrix()).to(device)
        self.backward = backward or m.fit.backward_batch
        self.poisoned = False

    def guard(self, stream):
        if (type(stream) is not inp.ApprovedPairStream or self.poisoned or
            acq.sha256(stream.path) != stream.bound or d.runtime_identity(self.device) != self.runtime):
            raise ValueError("Changed approval/runtime, wrong stream or poisoned instance")

    def state_dict(self, stream):
        self.guard(stream)
        if stream.cursor != self.step or self.schedule.step != self.step:
            raise ValueError("Cannot checkpoint partial paired update")
        return {"schema": 1, "purpose": PURPOSE, "limit": self.limit, "binding": copy.deepcopy(self.binding),
            "runtime": copy.deepcopy(self.runtime), "step": self.step, "sampler": stream.state_dict(),
            "schedule": self.schedule.state_dict(), "rng": d.portable(m.capture_rng(self.device)),
            "arms": {arm: {"model": d.portable(net.state_dict()), "optimizer": d.portable(self.optimizers[arm].state_dict()),
                "parameter_names": list(self.names[arm]), "modes": [mod.training for mod in net.modules()], "updates": self.step}
                for arm, net in self.models.items()}, "deployment_authorized": False}

    def validate_state(self, state):
        if (state.get("schema") != 1 or state.get("purpose") != PURPOSE or state.get("limit") != self.limit or
            state.get("binding") != self.binding or state.get("runtime") != self.runtime or
            state.get("deployment_authorized") is not False or type(state.get("step")) is not int or
            not 0 <= state["step"] <= self.limit or set(state.get("arms", {})) != set(ARMS) or
            state["sampler"]["cursor"] != state["step"] or state["schedule"]["step"] != state["step"] or not d.finite_state(state)):
            raise ValueError("Changed binding/budget/runtime or partial/nonfinite state")
        candidate = m.SharedSchedule(self.config)
        candidate.load_state_dict(state["schedule"])
        schedule = state["schedule"]
        if (type(schedule["last_validation"]) is not int or schedule["last_validation"] < 0 or
            schedule["last_validation"] % self.config["validation_every"] or
            (schedule["stopped_at"] is not None and type(schedule["stopped_at"]) is not int) or
            schedule["stopped_at"] not in (None, state["step"]) or set(schedule["patience_anchor"]) != set(ARMS) or
            any(type(v) is not int or v < 0 for v in schedule["stale"].values())):
            raise ValueError("Malformed shared validation/stop state")
        for arm in ARMS:
            saved, net = state["arms"][arm], self.models[arm]
            current = net.state_dict()
            if (saved["updates"] != state["step"] or saved["parameter_names"] != self.names[arm] or
                set(saved["model"]) != set(current) or any(saved["model"][k].shape != v.shape or saved["model"][k].dtype != v.dtype for k, v in current.items()) or
                len(saved["modes"]) != len(list(net.modules())) or any(type(v) is not bool for v in saved["modes"])):
                raise ValueError("Model exposure/names/shape/modes mismatch")
            optimizer = saved["optimizer"]
            groups, expected = optimizer["param_groups"], self.optimizers[arm].state_dict()["param_groups"][0]
            if len(groups) != 1 or groups[0]["params"] != expected["params"]:
                raise ValueError("Adam parameter order changed")
            if any(groups[0].get(k) != v for k, v in expected.items() if k not in ("lr", "params")):
                raise ValueError("Adam settings changed")
            lr = m.learning_rate(state["step"], self.config) if state["step"] else self.config["learning_rate"]
            if groups[0]["lr"] != lr or set(optimizer["state"]) != (set(groups[0]["params"]) if state["step"] else set()):
                raise ValueError("LR/missing Adam moments differ from committed step")
            parameters = dict(net.named_parameters())
            for index, values in optimizer["state"].items():
                p = parameters[self.names[arm][index]]
                if (set(values) != {"step", "exp_avg", "exp_avg_sq"} or float(values["step"]) != state["step"] or
                    any(values[k].shape != p.shape or values[k].dtype != p.dtype for k in ("exp_avg", "exp_avg_sq")) or
                    bool((values["exp_avg_sq"] < 0).any())):
                    raise ValueError("Malformed Adam state")

    def apply_state(self, state, stream):
        stream.load_state_dict(state["sampler"])
        self.schedule.load_state_dict(state["schedule"])
        for arm in ARMS:
            self.models[arm].load_state_dict(state["arms"][arm]["model"], strict=True)
            # Adam's CPU load may alias input tensors. Clone before load so a
            # later optimizer step cannot mutate a reusable saved checkpoint.
            self.optimizers[arm].load_state_dict(d.portable(state["arms"][arm]["optimizer"]))
            self.optimizers[arm].zero_grad(set_to_none=True)
            for mod, mode in zip(self.models[arm].modules(), state["arms"][arm]["modes"]):
                mod.training = mode
        self.step = state["step"]
        m.restore_rng(state["rng"], self.device)

    def load_state_dict(self, state, stream):
        self.guard(stream)
        self.validate_state(state)
        previous = self.state_dict(stream)
        try:
            self.apply_state(state, stream)
        except BaseException:
            try:
                self.apply_state(previous, stream)
            except BaseException:
                self.poisoned = True
                raise
            raise

    def update_next(self, stream):
        if self.step >= self.limit or self.schedule.stopped_at is not None:
            raise ValueError("Authorized tranche/shared stopping limit reached")
        previous = self.state_dict(stream)
        started = time.perf_counter()
        try:
            batch = stream.next_batch()
            x = batch["x"]
            if (batch["cursor"] != self.step or tuple(batch["domains"]) != m.DOMAINS or x.shape != (6, 2, 89856) or
                x.dtype != torch.float32 or x.device.type != "cpu" or set(batch["targets"]) != set(ARMS) or
                any(v.shape != x.shape or v.dtype != x.dtype or v.device.type != "cpu" for v in batch["targets"].values()) or
                not d.finite_state(batch) or not torch.equal(batch["targets"][ARMS[0]][:3], batch["targets"][ARMS[1]][:3]) or
                len(batch["metadata"]) != 6 or any(r.get("role") != "train" for r in batch["metadata"][:3]) or
                any(r.get("purpose") != PURPOSE or r.get("exploratory_eligible") is not True or r.get("deployment_eligible") is not False
                    for r in batch["metadata"][3:])):
                raise ValueError("Unapproved, changed, unpaired or nonfinite training batch")
            lr, losses = m.learning_rate(self.step + 1, self.config), {}
            for arm in ARMS:
                net, optimizer = self.models[arm], self.optimizers[arm]
                net.train()
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                losses[arm] = self.backward(net, x, batch["targets"][arm], self.wa, self.gs, self.device, 96, 44, "reconstruction_l1", 1)
                if not d.finite_state(losses[arm]):
                    raise ValueError("Nonfinite loss")
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                if not d.finite_state(net.state_dict()) or not d.finite_state(optimizer.state_dict()):
                    raise ValueError("Nonfinite model/Adam")
            self.step += 1
            self.schedule.complete_step()
            return {"step": self.step, "lr": lr, "losses": losses, "input_sha256": [m.pilot.wave_digest(v) for v in x],
                    "metadata": batch["metadata"], "seconds": time.perf_counter() - started}
        except BaseException:
            try:
                self.apply_state(previous, stream)
                if not m.equal_state(previous, self.state_dict(stream)):
                    raise RuntimeError("Rollback differs")
            except BaseException:
                self.poisoned = True
                raise
            raise


def binding(approval_path):
    return {"approval_sha256": acq.sha256(approval_path), "trainer_sha256": acq.sha256(__file__),
            "importer_sha256": acq.sha256(inp.__file__), "device_helpers_sha256": acq.sha256(d.__file__),
            "development_adapter_sha256": acq.sha256(dev.__file__)}


def exploration_gpu_preflight(doc):
    inp.check_approval(doc)
    guard = d.gpu_preflight()
    # Independent, explicitly authorized sharing policy, NOT a silent change to
    # the old synthetic CLI guard. Memory/index checks remain mandatory.
    guard["allowed"] = guard["gpu_index"] == 0 and guard["free_mib"] >= guard["minimum_free_mib"]
    guard["concurrency_authorized"] = True
    guard["utilization_ceiling_applied"] = False
    return guard


def atomic_status(out, doc):
    path = out / "run_status.json"
    temporary = out / f"status_{os.getpid()}.tmp"
    acq.write_new_json(temporary, doc | {"pid": os.getpid(), "updated_utc": m.bulk.now(), "purpose": PURPOSE,
        "formal_student_training": False, "deployment": False, "release_selection": "NONE"})
    os.replace(temporary, path)


def save_checkpoint(out, engine, stream):
    name = f"NONRELEASE_pair_step_{engine.step:04d}.pt"
    path = out / name
    state = engine.state_dict(stream)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not m.equal_state(saved, state):
            raise ValueError("Existing checkpoint differs; preserve failed/replayed evidence")
    else:
        m.save_new(path, state)
    digest = acq.sha256(path)
    receipt = {"step": engine.step, "checkpoint": name, "sha256": digest, "binding": engine.binding,
               "purpose": PURPOSE, "deployment_authorized": False}
    acq.seal(receipt)
    receipt_path = out / f"checkpoint_{engine.step:04d}.json"
    if receipt_path.exists():
        if acq.content_digest(acq.read_sealed(receipt_path)) != acq.content_digest(receipt):
            raise ValueError("Existing checkpoint receipt differs")
    else:
        acq.write_new_json(receipt_path, receipt)
    return receipt


def require_fresh(out):
    m.bulk.guard_output(out)
    if out.exists() or shutil.disk_usage(out.parent).free < 12 * 1024**3 + 512 * 1024**2:
        raise ValueError("Fresh ignored result directory and 12 GiB reserve required")


def smoke(approval_path, out, device):
    require_fresh(out)
    doc = inp.verified_approval(approval_path)
    if device == "cuda":
        guard = exploration_gpu_preflight(doc)
        if not guard["allowed"]:
            print("EXPLORATION_REAL DEFERRED GPU busy; no CUDA initialization", flush=True)
            return 2
    with d.deterministic_runtime(device):
        stream = inp.ApprovedPairStream(approval_path)
        factory = m.frozen_factory(doc["source_protocol"])
        engine = ExplorationEngine(factory, doc["source_protocol"], binding(approval_path), device, stream)
        out.mkdir(parents=True)
        receipts, draws = {}, []
        for _ in range(3):
            draws.append(engine.update_next(stream))
            receipt = save_checkpoint(out, engine, stream)
            receipts[receipt["checkpoint"]] = receipt["sha256"]
            print(f"EXPLORATION_REAL step={engine.step}/3 device={device}", flush=True)
        expected = engine.state_dict(stream)
        other = inp.ApprovedPairStream(approval_path)
        restored = ExplorationEngine(factory, doc["source_protocol"], binding(approval_path), device, other)
        name = "NONRELEASE_pair_step_0001.pt"
        saved = m.load_checked_checkpoint(out / name, receipts[name])
        restored.load_state_dict(saved, other)
        replay = [restored.update_next(other) for _ in range(2)]
        compare = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]
        if compare(draws[1:]) != compare(replay) or not m.equal_state(expected, restored.state_dict(other)):
            raise ValueError("Real-input disk resume not bit-identical")
        restored.load_state_dict(saved, other)
        before, original = restored.state_dict(other), restored.backward
        def fail_second(net, *args):
            if net is restored.models[ARMS[1]]:
                random.random(); m.np.random.random(); torch.rand(3, device=device)
                raise RuntimeError("injected second arm")
            return original(net, *args)
        restored.backward = fail_second
        try:
            restored.update_next(other)
        except RuntimeError as error:
            if str(error) != "injected second arm":
                raise
        else:
            raise ValueError("Failure injection absent")
        if not m.equal_state(before, restored.state_dict(other)):
            raise ValueError("Real-input rollback differs")
        report = {"schema": 1, "purpose": PURPOSE, "binding": engine.binding, "runtime": engine.runtime,
            "approval": str(approval_path.resolve()), "updates_per_arm": 3, "draws": draws, "checkpoints": receipts,
            "disk_resume_identical": True, "second_arm_rollback_identical": True,
            "formal_student_training": False, "deployment": False, "release_selection": "NONE"}
        acq.write_new_json(out / "real_mechanism.json", acq.seal(report))
    print(f"EXPLORATION_REAL PASS device={device}; resume and rollback identical", flush=True)
    return 0


def verify_mechanism(out, approval_path, device):
    report = acq.read_sealed(out / "real_mechanism.json")
    if (report["purpose"] != PURPOSE or report["binding"] != binding(approval_path) or report["runtime"]["device"] != device or
        report["updates_per_arm"] != 3 or not report["disk_resume_identical"] or not report["second_arm_rollback_identical"] or
        report["formal_student_training"] is not False or report["deployment"] is not False):
        raise ValueError("Missing/wrong current real-input mechanism evidence")
    for name, expected in report["checkpoints"].items():
        state = m.load_checked_checkpoint(out / name, expected)
        if state.get("purpose") != PURPOSE or state.get("binding") != binding(approval_path):
            raise ValueError("Mechanism checkpoint mismatch")
    return report


def synthetic_smoke(approval_path, out):
    """Same old three-step core, new explicitly authorized sharing entry only."""
    require_fresh(out)
    doc = inp.verified_approval(approval_path)
    guard = exploration_gpu_preflight(doc)
    if not guard["allowed"]:
        print("EXPLORATORY_SYNTHETIC DEFERRED insufficient free VRAM", flush=True)
        return 2
    out.mkdir(parents=True)
    acq.write_new_json(out / "gpu_preflight.json", acq.seal(guard | {"cuda_initialized": False, "trainer_sha256": acq.sha256(__file__)}))
    protocol = doc["source_protocol"]
    bound = d.binding(protocol)
    bound["files"].update({str(p.resolve()): acq.sha256(p) for p in (Path(__file__), Path(inp.__file__), inp.PROTOCOL, approval_path)})
    with d.deterministic_runtime("cuda"):
        factory = m.frozen_factory(protocol)
        engine = d.DevicePairedEngine(factory, protocol, bound, "cuda", d.PURPOSE)
        stream, receipts, draws = d.SyntheticBatchStream(), {}, []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            draws.append(engine.update_next(stream))
            path = out / f"NONRELEASE_synthetic_pair_step_{engine.step:04d}.pt"
            m.save_new(path, engine.state_dict(stream))
            receipts[path.name] = acq.sha256(path)
            print(f"EXPLORATORY_SYNTHETIC step={engine.step}/3 device=cuda", flush=True)
        expected = engine.state_dict(stream)
        restored = d.DevicePairedEngine(factory, protocol, bound, "cuda", d.PURPOSE)
        other = d.SyntheticBatchStream()
        name = "NONRELEASE_synthetic_pair_step_0001.pt"
        restored.load_state_dict(m.load_checked_checkpoint(out / name, receipts[name]), other)
        replay = [restored.update_next(other) for _ in range(2)]
        comparable = lambda rows: [{k:v for k,v in r.items() if k != "seconds"} for r in rows]
        if comparable(replay) != comparable(draws[1:]) or not m.equal_state(expected, restored.state_dict(other)):
            raise ValueError("CUDA synthetic disk replay differs")
        restored.load_state_dict(m.load_checked_checkpoint(out / name, receipts[name]), other)
        before, original = restored.state_dict(other), restored.backward
        def failed(net, *args):
            if net is restored.models[ARMS[1]]:
                random.random(); m.np.random.random(); torch.rand(3, device="cuda")
                raise RuntimeError("injected second arm")
            return original(net, *args)
        restored.backward = failed
        try:
            restored.update_next(other)
        except RuntimeError as error:
            if str(error) != "injected second arm":
                raise
        else:
            raise ValueError("Missing failure injection")
        if not m.equal_state(before, restored.state_dict(other)):
            raise ValueError("CUDA synthetic rollback differs")
        report = {"schema": 1, "binding": bound, "runtime": engine.runtime, "updates_per_arm": 3,
            "synthetic_only": True, "formal_student_training": False, "deployment": False, "checkpoint_selected": "NONE",
            "disk_resume_identical": True, "second_arm_failure_rollback_identical": True, "draws": draws, "checkpoints": receipts,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "resource_policy": guard,
            "scope": "Synthetic CUDA mechanism, not training or quality; explicitly permitted GPU concurrency"}
        acq.write_new_json(out / "device_mechanism.json", acq.seal(report))
    d.verify(out)
    return 0


def make_validator(approval):
    snapshot = acq.read_sealed(approval["snapshots"]["kim_melband"]["path"])
    teacher_ids = [Path(r["source"]["path"]).name for r in snapshot["records"]]
    corpus = dev.LockedDevelopmentCorpus(m.bulk.OLD_LOCK, teacher_ids)
    historical = json.loads(dev.OLD_MANIFEST.read_text(encoding="utf-8"))
    rows, manifest = dev.checked_suite(corpus, historical)
    frozen = m.frozen_factory(approval["source_protocol"])()
    validator = dev.PairedDevelopmentValidator(rows, frozen)
    return validator, manifest


def observe(engine, validator, factory):
    # CPU copies avoid changing the live CUDA net, its modes, buffers or RNG.
    rng = d.portable(m.capture_rng(engine.device))
    try:
        models = {arm: factory() for arm in ARMS}
        for arm in ARMS:
            models[arm].load_state_dict(d.portable(engine.models[arm].state_dict()), strict=True)
        return validator.observe_pair(engine.schedule, models, engine.step)
    finally:
        m.restore_rng(rng, engine.device)


def verify_run(out, approval_path):
    inp.verified_approval(approval_path)
    doc = acq.read_sealed(out / "completion.json")
    if (doc["purpose"] != PURPOSE or doc["binding"] != binding(approval_path) or doc["deployment"] is not False or
        doc["release_selection"] != "NONE" or not 0 < doc["step"] <= 1000):
        raise ValueError("Wrong exploration completion")
    for name, digest in doc["outputs_sha256"].items():
        if acq.sha256(out / name) != digest:
            raise ValueError("Exploration output changed")
    if doc["final_checkpoint"]["binding"] != binding(approval_path):
        raise ValueError("Final checkpoint recipe mismatch")
    state = m.load_checked_checkpoint(out / doc["final_checkpoint"]["checkpoint"], doc["final_checkpoint"]["sha256"])
    if (state["purpose"] != PURPOSE or state["binding"] != binding(approval_path) or state["deployment_authorized"] is not False or
        state["step"] != doc["step"] or state["sampler"]["cursor"] != doc["step"] or state["schedule"]["step"] != doc["step"] or
        any(v["updates"] != doc["step"] for v in state["arms"].values())):
        raise ValueError("Incomplete final pair")
    for step in range(250, doc["step"] + 1, 250):
        packet = acq.read_sealed(out / f"development_step_{step:04d}.json")
        if packet["step"] != step or set(packet["scores"]) != set(ARMS) or packet["suite_sha256"] != dev.EXPECTED_SUITE:
            raise ValueError("Missing/incorrect paired development coverage")
    print(f"EXPLORATION_VERIFY PASS step={doc['step']}; NONRELEASE", flush=True)


def train(approval_path, out, cpu_proof, cuda_proof, synthetic_proof, resume=False):
    doc = inp.verified_approval(approval_path)
    verify_mechanism(cpu_proof, approval_path, "cpu")
    cuda_runtime = verify_mechanism(cuda_proof, approval_path, "cuda")["runtime"]
    d.verify(synthetic_proof)
    synthetic = acq.read_sealed(synthetic_proof / "device_mechanism.json")
    if synthetic["runtime"]["device"] != "cuda" or synthetic["runtime"] != cuda_runtime:
        raise ValueError("CUDA synthetic mechanism proof required, not CPU substitute")
    guard = exploration_gpu_preflight(doc)
    if not guard["allowed"]:
        print("EXPLORATION_TRAIN DEFERRED GPU busy; no optimizer", flush=True)
        return 2
    m.bulk.guard_output(out)
    if resume:
        if not out.exists() or (out / "completion.json").exists():
            raise ValueError("Resume only unfinished existing run")
    else:
        require_fresh(out)
        out.mkdir(parents=True)
    with m.bulk.worker_lock(out), d.deterministic_runtime("cuda"):
        stream = inp.ApprovedPairStream(approval_path)
        factory = m.frozen_factory(doc["source_protocol"])
        engine = ExplorationEngine(factory, doc["source_protocol"], binding(approval_path), "cuda", stream)
        if engine.runtime != cuda_runtime:
            raise ValueError("CUDA runtime differs from tested mechanism backend")
        phase, latest = "initializing", None
        try:
            if resume:
                receipts = sorted(out.glob("checkpoint_*.json"))
                if not receipts:
                    raise ValueError("No full-state commit to resume")
                latest = acq.read_sealed(receipts[-1])
                if latest["binding"] != engine.binding:
                    raise ValueError("Changed resume recipe")
                engine.load_state_dict(m.load_checked_checkpoint(out / latest["checkpoint"], latest["sha256"]), stream)
            else:
                latest = save_checkpoint(out, engine, stream)
            atomic_status(out, {"status": "running", "phase": "development_baseline", "step": engine.step, "limit": engine.limit, "error": None})
            rng = d.portable(m.capture_rng("cuda"))
            try:
                validator, manifest = make_validator(doc)
            finally:
                m.restore_rng(rng, "cuda")
            if not resume:
                acq.write_new_json(out / "selection_suite.json", acq.seal(copy.deepcopy(manifest)))
                acq.write_new_json(out / "frozen_scores.json", acq.seal(copy.deepcopy(validator.baseline)))
            else:
                if acq.read_sealed(out / "selection_suite.json")["sha256"] != manifest["sha256"] or acq.read_sealed(out / "frozen_scores.json")["summary"] != validator.baseline["summary"]:
                    raise ValueError("Development baseline changed across resume")
            phase = "training"
            log_name = f"updates_from_{engine.step:04d}_{time.time_ns()}.jsonl"
            with (out / log_name).open("x", encoding="utf-8", buffering=1) as log:
                while engine.step < engine.limit and engine.schedule.stopped_at is None:
                    if shutil.disk_usage(out).free < 12 * 1024**3 + 256 * 1024**2:
                        raise ValueError("Disk reserve reached; no cleanup or continued writes")
                    row = engine.update_next(stream)
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    if engine.step % 10 == 0 or engine.step == 1:
                        atomic_status(out, {"status": "running", "phase": "training", "step": engine.step, "limit": engine.limit,
                            "latest_checkpoint": latest, "last_losses": row["losses"], "last_step_seconds": row["seconds"], "error": None})
                        print(f"EXPLORATION step={engine.step}/{engine.limit} seconds={row['seconds']:.3f} losses={row['losses']}", flush=True)
                    if engine.step % engine.config["validation_every"] == 0:
                        phase = "development_validation"
                        atomic_status(out, {"status": "running", "phase": phase, "step": engine.step, "limit": engine.limit, "error": None})
                        packet = observe(engine, validator, factory)
                        filename = f"development_step_{engine.step:04d}.json"
                        if (out / filename).exists():
                            old = acq.read_sealed(out / filename)
                            if acq.content_digest(old) != acq.content_digest(packet):
                                raise ValueError("Replayed validation differs; preserve old scores")
                        else:
                            acq.write_new_json(out / filename, acq.seal(packet))
                        latest = save_checkpoint(out, engine, stream)
                        print(f"EXPLORATION_DEV step={engine.step} scores={packet['scores']}", flush=True)
                        phase = "training"
            outputs = {p.name: acq.sha256(p) for p in out.iterdir() if p.suffix in ("json", "jsonl", "pt") and p.name != "run_status.json" and not p.name.startswith("detached_")}
            completion = {"schema": 1, "purpose": PURPOSE, "binding": engine.binding, "step": engine.step,
                "limit": engine.limit, "runtime": engine.runtime, "final_checkpoint": latest,
                "development_selection": engine.schedule.selection(), "release_selection": "NONE", "deployment": False,
                "formal_student_training": False, "exploratory_training": True, "independent_acceptance_ready": False,
                "outputs_sha256": outputs, "scope": "Old-development trend only, no new blind acceptance or model replacement"}
            acq.write_new_json(out / "completion.json", acq.seal(completion))
            atomic_status(out, {"status": "complete", "phase": "exploration_tranche_complete", "step": engine.step, "limit": engine.limit,
                "latest_checkpoint": latest, "development_selection": engine.schedule.selection(), "error": None})
            print(f"EXPLORATION COMPLETE step={engine.step}; release_selection=NONE", flush=True)
        except BaseException as error:
            atomic_status(out, {"status": "failed", "phase": phase, "step": engine.step, "limit": engine.limit,
                "latest_checkpoint": latest, "error": repr(error), "recovery": "Explicit resume from last full-state checkpoint; no automatic restart"})
            raise
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("synthetic-smoke", "smoke", "train", "verify"))
    parser.add_argument("--approval", type=Path, default=inp.DEFAULT_APPROVAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-proof", type=Path, default=ROOT / "results/paired_exploratory_cpu_20261003_r2")
    parser.add_argument("--cuda-proof", type=Path, default=ROOT / "results/paired_exploratory_cuda_20261003_r2")
    parser.add_argument("--synthetic-proof", type=Path, default=ROOT / "results/paired_device_cuda_20261003_r2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.operation == "synthetic-smoke":
        return synthetic_smoke(args.approval, args.out)
    if args.operation == "smoke":
        return smoke(args.approval, args.out, args.device)
    if args.operation == "verify":
        verify_run(args.out, args.approval)
        return 0
    if args.device != "cuda":
        raise ValueError("Exploration training requires explicit CUDA; no CPU long-run fallback")
    return train(args.approval, args.out, args.cpu_proof, args.cuda_proof, args.synthetic_proof, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
