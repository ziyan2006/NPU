"""Bounded same-Mel loss-weight fork. Explicit state migration, never deployment."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time
import torch
import importlib.util

spec = importlib.util.spec_from_file_location("weak_import", Path(__file__).with_name("153_prepare_mel_weak_weight.py"))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
old, m, d, acq, ROOT, ARMS, PURPOSE = p.old, p.m, p.d, p.acq, p.ROOT, p.ARMS, p.PURPOSE
DEFAULT_OUT = ROOT / "results/mel_weak_weight_20261003"
CPU_PROOF = ROOT / "results/mel_weak_weight_cpu_20261003"
CUDA_PROOF = ROOT / "results/mel_weak_weight_cuda_20261003"
AUDIT_OUT = ROOT / "results/mel_weak_weight_audit_20261003"


def binding(path):
    return {"approval_sha256": acq.sha256(path), "trainer_sha256": acq.sha256(__file__),
            "importer_sha256": acq.sha256(p.__file__), "arm_roles": p.ROLES, "teacher": "kim_melband"}


def slot_weights(metadata, weighted):
    if len(metadata) != 6 or [r.get("domain") for r in metadata] != list(m.DOMAINS):
        raise ValueError("Exact six domain slots required")
    if any(r.get("role") != "train" or type(r.get("vocal_db")) is not int for r in metadata[:3]):
        raise ValueError("Exact true TRAIN remix metadata required")
    if any(r.get("vocal_db") not in (-12, -6, 0, 6) for r in metadata[:2]) or metadata[2]["vocal_db"] != 0:
        raise ValueError("Changed true remix recipe")
    return [2. if weighted and i < 2 and r["vocal_db"] == -12 else 1. for i, r in enumerate(metadata)]


def weighted_backward(net, x, v, wa, gs, device, weights):
    if len(weights) != x.shape[0] or any(w not in (1., 2.) for w in weights):
        raise ValueError("Only the approved discrete slot weights allowed")
    if all(w == 1 for w in weights):
        return m.fit.backward_batch(net, x, v, wa, gs, device, 96, 44, "reconstruction_l1", 1)
    if x.shape != v.shape or x.ndim != 3 or not d.finite_state((x, v)):
        raise ValueError("Aligned finite waveform targets required")
    if any(isinstance(mod, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout,
                           torch.nn.Dropout2d, torch.nn.Dropout3d)) for mod in net.modules()):
        raise ValueError("Batch-independent deterministic graph required")
    total = {"loss": 0., "wave_l1": 0., "complex_l1": 0.}
    for i, w in enumerate(weights):
        xb, vb = x[i:i+1].to(device), v[i:i+1].to(device)
        spectrum = m.core.stft_batch(xb)
        bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
        output = net(bands)
        loss, parts, _ = m.fit.reconstruction_loss((output[:, :2]+1)/2, spectrum, xb, vb, gs, 96, 44)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite weighted loss")
        scale = w/sum(weights)
        (loss*scale).backward()
        for key, value in {"loss": loss, **parts}.items():
            total[key] += float(value.detach())*scale
    return total


class ForkEngine(old.ExplorationEngine):
    """Reuse tested transaction/Adam validation with explicit new authority, not old resume."""
    def __init__(self, factory, stream, device, smoke=False):
        if type(stream) is not p.MelForkStream or device not in ("cpu", "cuda") or (device == "cpu" and not smoke):
            raise ValueError("Approved fork stream; CPU permitted only for bounded mechanism")
        p.check_approval(stream.doc)
        if stream.cursor != 1000:
            raise ValueError("Explicit fork begins at cursor1000")
        self.config = m.validate_protocol(stream.doc["source_protocol"])
        self.binding = binding(stream.path)
        self.device, self.step = device, 1000
        self.limit = 1003 if smoke else 1500
        self.runtime = d.runtime_identity(device)
        if (not self.runtime["deterministic"] or self.runtime["warn_only"] or self.runtime["cudnn_benchmark"] or
            not self.runtime["cudnn_deterministic"] or self.runtime["matmul_tf32"] or self.runtime["cudnn_tf32"]):
            raise ValueError("Strict FP32 runtime required")
        source = stream.source_state
        if device == "cuda" and self.runtime != source["runtime"]:
            raise ValueError("CUDA fork runtime must equal original tested runtime")
        self.models = {arm: factory().to(device) for arm in ARMS}
        self.names = {arm: [name for name, _ in net.named_parameters()] for arm, net in self.models.items()}
        self.optimizers = {arm: torch.optim.Adam(net.parameters(), lr=self.config["learning_rate"], foreach=False, fused=False) for arm, net in self.models.items()}
        self.schedule = m.SharedSchedule(self.config)
        self.wa = torch.from_numpy(m.core.t09.make_analysis_matrix()).to(device)
        self.gs = torch.from_numpy(m.core.t09.make_synthesis_matrix()).to(device)
        self.poisoned, self.smoke = False, smoke
        self.backward = self._backward
        self.origin_sha256 = stream.doc["protocol"]["origin_checkpoint_sha256"]
        transferred = d.portable(source)
        chosen = d.portable(source["arms"][ARMS[1]])
        transferred.update(purpose=PURPOSE, binding=self.binding, limit=self.limit, runtime=self.runtime,
            sampler=stream.state_dict(), arm_roles=p.ROLES, teacher="kim_melband", origin_sha256=self.origin_sha256,
            smoke=smoke, arms={arm: d.portable(chosen) for arm in ARMS})
        for field in ("best", "stale", "patience_anchor"):
            transferred["schedule"][field] = {arm: copy.deepcopy(source["schedule"][field][ARMS[1]]) for arm in ARMS}
        if device == "cpu":
            # Explicit CPU mechanism fork ONLY. Never claim cross-device numerical resume.
            transferred["rng"]["torch_cuda"] = []
        self.validate_state(transferred)
        self.apply_state(transferred, stream)
        for arm in ARMS:
            if not m.equal_state(self.state_dict(stream)["arms"][arm], chosen):
                raise ValueError("Model/Adam/names/modes/exposure migration differs")
        self.initial_rng = d.portable(transferred["rng"])

    def guard(self, stream):
        if (type(stream) is not p.MelForkStream or self.poisoned or acq.sha256(stream.path) != stream.bound or
            d.runtime_identity(self.device) != self.runtime):
            raise ValueError("Changed fork authority/runtime or poisoned instance")

    def state_dict(self, stream):
        state = super().state_dict(stream)
        state.update(purpose=PURPOSE, arm_roles=p.ROLES, teacher="kim_melband", smoke=self.smoke,
                     origin_sha256=self.origin_sha256)
        return state

    def validate_state(self, state):
        if (state.get("purpose") != PURPOSE or state.get("arm_roles") != p.ROLES or
            state.get("teacher") != "kim_melband" or state.get("smoke") is not self.smoke or
            state.get("origin_sha256") != self.origin_sha256 or not 1000 <= state.get("step", -1) <= self.limit or
            state["sampler"].get("teacher") != "kim_melband" or state["sampler"].get("approval_sha256") != self.binding["approval_sha256"]):
            raise ValueError("Changed fork origin, loss roles, teacher, phase or cursor")
        super().validate_state(state | {"purpose": old.PURPOSE})

    def _backward(self, net, x, v, wa, gs, device, warmup, kill, objective, microbatch):
        if (warmup, kill, objective, microbatch) != (96, 44, "reconstruction_l1", 1):
            raise ValueError("Original reconstruction geometry required")
        arm = next((arm for arm in ARMS if net is self.models[arm]), None)
        if arm is None:
            raise ValueError("Unknown loss arm")
        weights = slot_weights(self._stream.last_metadata, arm == ARMS[1])
        return weighted_backward(net, x, v, wa, gs, device, weights)

    def update_next(self, stream):
        self._stream = stream
        original = stream.next_batch
        def checked_batch():
            batch = original()
            slot_weights(batch["metadata"], True)
            if not torch.equal(batch["targets"][ARMS[0]], batch["targets"][ARMS[1]]):
                raise ValueError("Loss-only experiment must share ALL Mel targets")
            return batch
        stream.next_batch = checked_batch
        try:
            row = super().update_next(stream)
            row.update(additional_step=self.step-1000, arm_roles=p.ROLES, teacher="kim_melband",
                slot_weights={arm: slot_weights(row["metadata"], arm == ARMS[1]) for arm in ARMS})
            return row
        finally:
            stream.next_batch = original


def save_checkpoint(out, engine, stream):
    state = engine.state_dict(stream)
    name = f"NONRELEASE_weak_step_{engine.step:04d}.pt"
    path = out / name
    if path.exists():
        if not m.equal_state(m.load_checked_checkpoint(path, acq.sha256(path)), state):
            raise ValueError("Existing checkpoint conflicts; preserve it")
    else:
        m.save_new(path, state)
    receipt = acq.seal({"schema": 1, "purpose": PURPOSE, "step": engine.step, "additional_step": engine.step-1000,
        "checkpoint": name, "sha256": acq.sha256(path), "binding": engine.binding, "arm_roles": p.ROLES,
        "teacher": "kim_melband", "deployment_authorized": False, "smoke": engine.smoke})
    target = out / f"checkpoint_{engine.step:04d}.json"
    if target.exists():
        if acq.read_sealed(target) != receipt:
            raise ValueError("Existing receipt conflicts")
    else:
        acq.write_new_json(target, receipt)
    return receipt


def status(out, doc):
    temporary = out / f"status_{os.getpid()}.tmp"
    acq.write_new_json(temporary, doc | {"pid": os.getpid(), "updated_utc": m.bulk.now(), "purpose": PURPOSE,
        "teacher": "kim_melband", "arm_roles": p.ROLES, "deployment": False, "release_selection": "NONE"})
    os.replace(temporary, out / "run_status.json")


def smoke(approval, out, device):
    old.require_fresh(out)
    doc = p.verified_approval(approval)
    if device == "cuda" and not old.exploration_gpu_preflight(p.inp.verified_approval(Path(doc["origin_approval"])))['allowed']:
        return 2
    out.mkdir(parents=True)
    with d.deterministic_runtime(device):
        stream = p.MelForkStream(approval)
        factory = m.frozen_factory(doc["source_protocol"])
        engine = ForkEngine(factory, stream, device, smoke=True)
        initial = engine.state_dict(stream)
        draws, states = [], []
        for _ in range(3):
            draws.append(engine.update_next(stream))
            states.append(engine.state_dict(stream))
        m.save_new(out / "resume1001.pt", states[0])
        other = p.MelForkStream(approval)
        replay = ForkEngine(factory, other, device, smoke=True)
        saved = m.load_checked_checkpoint(out / "resume1001.pt", acq.sha256(out / "resume1001.pt"))
        untouched = d.portable(saved)
        replay.load_state_dict(saved, other)
        for i in (1, 2):
            row = replay.update_next(other)
            if any(row[k] != draws[i][k] for k in row if k != "seconds") or not m.equal_state(replay.state_dict(other), states[i]):
                raise ValueError("Disk resume differs")
        if not m.equal_state(saved, untouched):
            raise ValueError("Adam load aliased saved state")
        replay.load_state_dict(saved, other)
        before, original = replay.state_dict(other), replay.backward
        def fail_second(net, *args):
            if net is replay.models[ARMS[1]]:
                torch.rand(2)
                raise RuntimeError("injected weak arm")
            return original(net, *args)
        replay.backward = fail_second
        try:
            replay.update_next(other)
        except RuntimeError as error:
            if str(error) != "injected weak arm":
                raise
        else:
            raise ValueError("No failure injection")
        if not m.equal_state(before, replay.state_dict(other)):
            raise ValueError("Rollback differs")
        proof = {"schema": 1, "purpose": PURPOSE, "binding": binding(approval), "runtime": engine.runtime,
            "device": device, "mechanism_only": True, "updates_per_arm": 3, "initial_equal": m.equal_state(initial["arms"][ARMS[0]], initial["arms"][ARMS[1]]),
            "initial_source_model_digest": old.dev.state_digest(initial["arms"][ARMS[0]]["model"]),
            "source_device": stream.source_state["runtime"]["device"], "cpu_rng_device_migration": device == "cpu",
            "source_rng_restored": m.equal_state(initial["rng"], stream.source_state["rng"] | ({"torch_cuda": []} if device == "cpu" else {})),
            "disk_resume_identical": True, "rollback_identical": True, "saved_state_unaliased": True,
            "resume_sha256": acq.sha256(out / "resume1001.pt"), "draws": draws, "release_selection": "NONE", "deployment": False}
        acq.write_new_json(out / "mechanism.json", acq.seal(proof))
    print(f"MEL_WEAK_MECHANISM PASS device={device}; real input3; disk resume/rollback; no quality claim", flush=True)
    return 0


def audit(approval, out):
    old.require_fresh(out)
    doc = p.verified_approval(approval)
    stream = p.MelForkStream(approval)
    rows, selected = [], None
    for _ in range(12):
        batch = stream.next_batch()
        weights = slot_weights(batch["metadata"], True)
        for i, meta in enumerate(batch["metadata"]):
            region = slice(meta["score_start"], meta["score_end"])
            x, v = batch["x"][i, :, region], batch["targets"][ARMS[0]][i, :, region]
            backing = x-v
            vv, aa = float(v.square().mean()), float(backing.square().mean())
            rows.append(meta | {"step": batch["cursor"]+1, "slot": i, "input_rms": float(x.square().mean().sqrt()),
                "vocal_rms": math.sqrt(vv), "vocal_to_backing_db": 10*math.log10(max(vv, 1e-12)/max(aa, 1e-12)),
                "vocal_active_above_rms_1e-4": vv > 1e-8, "nominal_weight": weights[i]})
        if selected is None and 2. in weights:
            selected = d.portable(batch)
    if selected is None:
        raise ValueError("No explicit weak TRAIN slot in fixed12 draws; preserve selection rule")
    net = m.frozen_factory(doc["source_protocol"])()
    net.load_state_dict(stream.source_state["arms"][ARMS[1]]["model"])
    net.train()
    digest = old.dev.state_digest(net.state_dict())
    wa, gs = torch.from_numpy(m.core.t09.make_analysis_matrix()), torch.from_numpy(m.core.t09.make_synthesis_matrix())
    gradients = []
    for i, meta in enumerate(selected["metadata"]):
        net.zero_grad(set_to_none=True)
        values = m.fit.backward_batch(net, selected["x"][i:i+1], selected["targets"][ARMS[0]][i:i+1], wa, gs, "cpu", 96, 44, "reconstruction_l1", 1)
        norm = math.sqrt(sum(float(param.grad.square().sum()) for param in net.parameters() if param.grad is not None))
        gradients.append({"slot": i, "metadata": meta, "unweighted_loss": values, "preclip_parameter_gradient_l2": norm})
    if old.dev.state_digest(net.state_dict()) != digest:
        raise ValueError("Read-only audit changed weights")
    out.mkdir(parents=True)
    acq.write_new_json(out / "audit.json", acq.seal({"schema": 1, "purpose": PURPOSE, "binding": binding(approval),
        "selection": "fixed TRAIN cursors1000..1011; gradient batch first explicit-12 among12, not development-selected",
        "rows": rows, "gradient_absolute_step": selected["cursor"]+1, "gradients": gradients,
        "source_model_digest": digest, "model_unchanged": True, "cuda_used": False, "optimizer_constructed": False,
        "release_selection": "NONE", "deployment": False, "scope": "Small activity/gradient diagnostic, not dataset-wide contribution or causal proof"}))
    print("MEL_WEAK_AUDIT PASS inputs12x6; one predeclared TRAIN gradient batch; CPU-only, no updates", flush=True)


def verify_proof(path, approval, device):
    doc = acq.read_sealed(path / "mechanism.json")
    if (doc["binding"] != binding(approval) or doc["device"] != device or not doc["mechanism_only"] or
        doc["updates_per_arm"] != 3 or not all(doc[k] is True for k in ("initial_equal", "source_rng_restored", "disk_resume_identical", "rollback_identical", "saved_state_unaliased")) or
        acq.sha256(path / "resume1001.pt") != doc["resume_sha256"]):
        raise ValueError("Missing current migrated-state mechanism proof")
    return doc


def train(approval, out, cpu_proof, cuda_proof, audit_out, resume=False):
    doc = p.verified_approval(approval)
    verify_proof(cpu_proof, approval, "cpu")
    cuda = verify_proof(cuda_proof, approval, "cuda")
    audit_doc = acq.read_sealed(audit_out / "audit.json")
    if audit_doc["binding"] != binding(approval) or not audit_doc["model_unchanged"]:
        raise ValueError("Read-only activity/gradient audit missing")
    guard = old.exploration_gpu_preflight(inp_doc := p.inp.verified_approval(Path(doc["origin_approval"])))
    if not guard["allowed"]:
        print("MEL_WEAK_TRAIN DEFERRED insufficient free GPU memory", flush=True)
        return 2
    if resume:
        if not out.exists() or (out / "completion.json").exists():
            raise ValueError("Explicit resume only unfinished existing run")
    else:
        old.require_fresh(out)
        out.mkdir(parents=True)
    with m.bulk.worker_lock(out), d.deterministic_runtime("cuda"):
        stream = p.MelForkStream(approval)
        factory = m.frozen_factory(doc["source_protocol"])
        engine = ForkEngine(factory, stream, "cuda")
        if engine.runtime != cuda["runtime"]:
            raise ValueError("Actual CUDA runtime differs from fork mechanism proof")
        phase, latest = "initializing", None
        try:
            if resume:
                receipts = sorted(out.glob("checkpoint_*.json"))
                if not receipts:
                    raise ValueError("No complete checkpoint receipt to resume")
                latest = acq.read_sealed(receipts[-1])
                if latest["binding"] != engine.binding or Path(latest["checkpoint"]).name != latest["checkpoint"]:
                    raise ValueError("Changed resume binding/path")
                engine.load_state_dict(m.load_checked_checkpoint(out / latest["checkpoint"], latest["sha256"]), stream)
            else:
                latest = save_checkpoint(out, engine, stream)
            status(out, {"status": "running", "phase": "development_baseline", "step": engine.step, "additional_step": engine.step-1000, "limit": 1500, "latest_checkpoint": latest, "error": None})
            rng = d.portable(m.capture_rng("cuda"))
            try:
                validator, manifest = old.make_validator(inp_doc)
            finally:
                m.restore_rng(rng, "cuda")
            if not resume:
                acq.write_new_json(out / "selection_suite.json", acq.seal(copy.deepcopy(manifest)))
                acq.write_new_json(out / "frozen_scores.json", acq.seal(copy.deepcopy(validator.baseline)))
                rng = d.portable(m.capture_rng("cuda"))
                try:
                    models = {arm: factory() for arm in ARMS}
                    for arm in ARMS:
                        models[arm].load_state_dict(d.portable(engine.models[arm].state_dict()))
                    start = validator.evaluate_pair(models, 1000)
                finally:
                    m.restore_rng(rng, "cuda")
                # Both copied models must reproduce the original Mel score on the SAME truth suite.
                original = acq.read_sealed(ROOT / doc["protocol"]["origin_run"] / "development_step_1000.json")
                if any(start["evaluations"][arm] != original["evaluations"][ARMS[1]] for arm in ARMS):
                    raise ValueError("Fork baseline differs from original Mel1000 score")
                start.update(purpose=PURPOSE, arm_roles=p.ROLES, teacher="kim_melband", additional_step=0)
                acq.write_new_json(out / "development_step_1000.json", acq.seal(start))
            else:
                if acq.read_sealed(out / "selection_suite.json")["sha256"] != manifest["sha256"] or acq.read_sealed(out / "frozen_scores.json") != acq.seal(copy.deepcopy(validator.baseline)):
                    raise ValueError("Development suite/baseline changed")
            phase = "training"
            with (out / f"updates_from_{engine.step:04d}_{time.time_ns()}.jsonl").open("x", encoding="utf-8", buffering=1) as log:
                while engine.step < 1500 and engine.schedule.stopped_at is None:
                    if shutil.disk_usage(out).free < 12*1024**3 + 256*1024**2:
                        raise ValueError("Disk reserve reached; no cleanup")
                    row = engine.update_next(stream)
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                    if engine.step % 10 == 0 or engine.step == 1001:
                        status(out, {"status": "running", "phase": "training", "step": engine.step, "additional_step": engine.step-1000,
                            "limit": 1500, "latest_checkpoint": latest, "last_losses": row["losses"], "last_step_seconds": row["seconds"], "error": None})
                        print(f"MEL_WEAK step={engine.step}/1500 additional={engine.step-1000}/500 seconds={row['seconds']:.3f}", flush=True)
                    if engine.step % 250 == 0:
                        phase = "development_validation"
                        status(out, {"status": "running", "phase": phase, "step": engine.step, "additional_step": engine.step-1000, "limit": 1500, "latest_checkpoint": latest, "error": None})
                        packet = old.observe(engine, validator, factory)
                        packet.update(purpose=PURPOSE, arm_roles=p.ROLES, teacher="kim_melband", additional_step=engine.step-1000)
                        path = out / f"development_step_{engine.step:04d}.json"
                        if path.exists():
                            if acq.read_sealed(path) != acq.seal(packet):
                                raise ValueError("Replayed development differs")
                        else:
                            acq.write_new_json(path, acq.seal(packet))
                        latest = save_checkpoint(out, engine, stream)
                        print(f"MEL_WEAK_DEV step={engine.step} scores={packet['scores']}", flush=True)
                        phase = "training"
            files = {path.name: acq.sha256(path) for path in out.iterdir() if path.suffix in (".json", ".jsonl", ".pt") and path.name != "run_status.json"}
            acq.write_new_json(out / "completion.json", acq.seal({"schema": 1, "purpose": PURPOSE, "binding": engine.binding,
                "step": engine.step, "additional_steps": engine.step-1000, "limit": 1500, "final_checkpoint": latest,
                "runtime": engine.runtime, "outputs_sha256": files, "teacher": "kim_melband", "arm_roles": p.ROLES,
                "release_selection": "NONE", "deployment": False, "independent_acceptance_ready": False}))
            status(out, {"status": "complete", "phase": "weak_weight_tranche_complete", "step": engine.step,
                "additional_step": engine.step-1000, "limit": 1500, "latest_checkpoint": latest, "error": None})
            print("MEL_WEAK COMPLETE500; release_selection=NONE", flush=True)
        except BaseException as error:
            status(out, {"status": "failed", "phase": phase, "step": engine.step, "additional_step": engine.step-1000,
                "limit": 1500, "latest_checkpoint": latest, "error": repr(error), "recovery": "Diagnose; explicit resume only from last full commit"})
            raise
    return 0


def verify(out, approval):
    p.verified_approval(approval)
    doc = acq.read_sealed(out / "completion.json")
    if (doc["purpose"] != PURPOSE or doc["binding"] != binding(approval) or doc["step"] != 1500 or
        doc["additional_steps"] != 500 or doc["teacher"] != "kim_melband" or doc["arm_roles"] != p.ROLES or
        doc["release_selection"] != "NONE" or doc["deployment"] is not False or not doc["outputs_sha256"]):
        raise ValueError("Invalid completion or missing output hashes")
    for name, digest in doc["outputs_sha256"].items():
        if Path(name).name != name or acq.sha256(out / name) != digest:
            raise ValueError("Completion output differs")
    state = m.load_checked_checkpoint(out / doc["final_checkpoint"]["checkpoint"], doc["final_checkpoint"]["sha256"])
    if state["step"] != 1500 or state["smoke"] or state["sampler"]["cursor"] != 1500 or any(a["updates"] != 1500 for a in state["arms"].values()):
        raise ValueError("Incomplete final state")
    for step in (1000, 1250, 1500):
        packet = acq.read_sealed(out / f"development_step_{step:04d}.json")
        if packet["step"] != step or packet["suite_sha256"] != old.dev.EXPECTED_SUITE:
            raise ValueError("Development coverage changed")
    print("MEL_WEAK_VERIFY PASS500; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("audit", "smoke", "train", "verify"))
    parser.add_argument("--approval", type=Path, default=p.DEFAULT_APPROVAL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-proof", type=Path, default=CPU_PROOF)
    parser.add_argument("--cuda-proof", type=Path, default=CUDA_PROOF)
    parser.add_argument("--audit-out", type=Path, default=AUDIT_OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = args.out or (AUDIT_OUT if args.operation == "audit" else (CPU_PROOF if args.device == "cpu" else CUDA_PROOF) if args.operation == "smoke" else DEFAULT_OUT)
    if args.operation == "audit":
        audit(args.approval, out)
    elif args.operation == "smoke":
        raise SystemExit(smoke(args.approval, out, args.device))
    elif args.operation == "verify":
        verify(out, args.approval)
    else:
        if args.device != "cuda":
            raise ValueError("No CPU long-training fallback")
        raise SystemExit(train(args.approval, out, args.cpu_proof, args.cuda_proof, args.audit_out, args.resume))
