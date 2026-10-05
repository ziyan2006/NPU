"""Independent LF44/LF32 reconstruction and evaluation, fixed5003501..4000."""
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

spec = importlib.util.spec_from_file_location("aux_import", Path(__file__).with_name("185_prepare_mel_lf_boundary.py"))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
old, m, d, acq, ROOT, ARMS, PURPOSE = p.old, p.m, p.d, p.acq, p.ROOT, p.ARMS, p.PURPOSE
DEFAULT_OUT = ROOT / "results/mel_lf_boundary_20261003"
CPU_PROOF = ROOT / "results/mel_lf_boundary_cpu_20261003"
CUDA_PROOF = ROOT / "results/mel_lf_boundary_cuda_20261003"

spec = importlib.util.spec_from_file_location("aux_kernel", Path(__file__).with_name("176_accompaniment_component_loss.py"))
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


spec = importlib.util.spec_from_file_location("protection_loss", Path(__file__).with_name("170_instrumental_protection_loss.py"))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

def binding(path):
    return {"approval_sha256": acq.sha256(path), "trainer_sha256": acq.sha256(__file__),
            "importer_sha256": acq.sha256(p.__file__), "protection_kernel_sha256": acq.sha256(w.__file__), "auxiliary_kernel_sha256": acq.sha256(k.__file__), "arm_roles": p.ROLES, "arm_lambdas": p.LAMBDAS, "arm_instrumental_weights": p.WEIGHTS, "arm_accompaniment_weights": p.CA_WEIGHTS, "arm_kill_bands": p.KILLS, "teacher": "kim_melband"}


def validate_metadata(metadata):
    if len(metadata) != 6 or [r.get("domain") for r in metadata] != list(m.DOMAINS):
        raise ValueError("Exact six domain slots required")
    if any(r.get("role") != "train" or type(r.get("vocal_db")) is not int for r in metadata[:3]):
        raise ValueError("Exact true TRAIN remix metadata required")
    if any(r.get("vocal_db") not in (-12, -6, 0, 6) for r in metadata[:2]) or metadata[2]["vocal_db"] != 0:
        raise ValueError("Changed true remix recipe")
    if any(r.get("role") != "pseudo_label_train_candidate" for r in metadata[3:]):
        raise ValueError("Pseudo role changed")


def boundary_backward(net, x, v, wa, gs, device, metadata, kill):
    accompaniment_weight = 1
    instrumental_weight = 4
    validate_metadata(metadata)
    if (x.shape != v.shape or x.shape != (6, 2, 89856) or
        x.dtype != torch.float32 or v.dtype != torch.float32 or x.device != v.device or
        type(kill) is not int or kill not in (44, 32) or not d.finite_state((x, v))):
        raise ValueError("Aligned original FP32 crop and explicit LF44/32 required")
    if any(isinstance(mod, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout,
                           torch.nn.Dropout2d, torch.nn.Dropout3d)) for mod in net.modules()):
        raise ValueError("Batch-independent deterministic graph required")
    total = {"loss": 0., "base_loss": 0., "wave_l1": 0., "complex_l1": 0.,
             "auxiliary_loss": 0., "auxiliary_contribution": 0.,
             "auxiliary_active_count": 0, "auxiliary_skip_count": 0, "auxiliary_slots": [],
             "weighted_base_loss": 0., "instrumental_base_loss": 0.,
             "instrumental_weighted_contribution": 0., "instrumental_extra_contribution": 0.,
             "instrumental_weight": instrumental_weight, "base_normalizer": 6,
             "accompaniment_weight": accompaniment_weight, "accompaniment_component_contribution": 0., "remaining_vocal_component_contribution": 0.}
    for i, meta in enumerate(metadata):
        xb, vb = x[i:i+1].to(device), v[i:i+1].to(device)
        spectrum = m.core.stft_batch(xb)
        bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
        output = net(bands)
        base, parts, pv = m.fit.reconstruction_loss((output[:, :2]+1)/2, spectrum, xb, vb, gs, 96, kill)
        region = m.fit.suite.scoring_slice(xb.shape[-1], 96)
        auxiliary, info = k.source_projection_component_loss(pv[0, :, region], xb[0, :, region], vb[0, :, region], meta, accompaniment_weight)
        scaled, composition = w.combine_slot_loss(base, auxiliary, meta, i, instrumental_weight, info)
        scaled.backward()
        weighted = (instrumental_weight if i == 2 else 1)*base
        loss = weighted+.2*auxiliary
        for key, value in {"loss": loss, "base_loss": base, **parts, "auxiliary_loss": auxiliary,
                           "auxiliary_contribution": .2*auxiliary, "weighted_base_loss": weighted}.items():
            total[key] += float(value.detach())*(1/6)
        if i == 2:
            total["instrumental_base_loss"] = float(base.detach())*(1/6)
            total["instrumental_weighted_contribution"] = float(weighted.detach())*(1/6)
            total["instrumental_extra_contribution"] = (instrumental_weight-1)*float(base.detach())*(1/6)
        if info["active"]:
            total["accompaniment_component_contribution"] += accompaniment_weight*.2*(info["accompaniment_gain"]-1)**2*(1/6)
            total["remaining_vocal_component_contribution"] += .2*info["remaining_vocal_gain"]**2*(1/6)
        total["auxiliary_active_count"] += int(info["active"])
        total["auxiliary_skip_count"] += int(not info["active"])
        total["auxiliary_slots"].append({"slot": i, **info})
    return total


class ForkEngine(old.ExplorationEngine):
    """Reuse tested transaction/Adam validation with explicit new authority, not old resume."""
    def __init__(self, factory, stream, device, smoke=False):
        if type(stream) is not p.BoundaryStream or stream.path is None or device not in ("cpu", "cuda") or (device == "cpu" and not smoke):
            raise ValueError("Approved fork stream; CPU permitted only for bounded mechanism")
        p.check_approval(stream.doc)
        if stream.cursor != 3500:
            raise ValueError("Explicit fork begins at cursor3500")
        self.config = m.validate_protocol(stream.doc["source_protocol"])
        self.binding = binding(stream.path)
        self.device, self.step = device, 3500
        self.limit = 3503 if smoke else 4000
        self.runtime = d.runtime_identity(device)
        if (not self.runtime["deterministic"] or self.runtime["warn_only"] or self.runtime["cudnn_benchmark"] or
            not self.runtime["cudnn_deterministic"] or self.runtime["matmul_tf32"] or self.runtime["cudnn_tf32"]):
            raise ValueError("Strict FP32 runtime required")
        source = stream.source_state
        self.source_stopped_at = source["schedule"]["stopped_at"]
        self.legacy_stop_events = []
        self.source_legacy_stop_events = list(source.get("legacy_stop_events", []))
        if (self.source_stopped_at != stream.doc["protocol"]["source_stopped_at"] or
            self.source_legacy_stop_events != stream.doc["protocol"]["source_legacy_stop_events"]):
            raise ValueError("Source closed-tranche stop evidence changed")
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
        chosen = d.portable(source["arms"][p.SOURCE_ARM])
        transferred.update(purpose=PURPOSE, binding=self.binding, limit=self.limit, runtime=self.runtime,
            sampler=stream.state_dict(), arm_roles=p.ROLES, arm_lambdas=p.LAMBDAS, arm_instrumental_weights=p.WEIGHTS, arm_accompaniment_weights=p.CA_WEIGHTS, arm_kill_bands=p.KILLS, teacher="kim_melband", origin_sha256=self.origin_sha256,
            smoke=smoke, arms={arm: d.portable(chosen) for arm in ARMS}, source_stopped_at=self.source_stopped_at, source_legacy_stop_events=self.source_legacy_stop_events,
            tranche_stopping=stream.doc["protocol"]["tranche_stopping"], legacy_stop_events=[])
        # A new, explicitly authorized fixed budget, NOT resuming a stopped old run.
        # Keep source stop as provenance; retain LR config, best/stale/anchors below.
        transferred["schedule"]["stopped_at"] = None
        for field in ("best", "stale", "patience_anchor"):
            transferred["schedule"][field] = {arm: copy.deepcopy(source["schedule"][field][p.SOURCE_ARM]) for arm in ARMS}
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
        if (type(stream) is not p.BoundaryStream or self.poisoned or acq.sha256(stream.path) != stream.bound or
            d.runtime_identity(self.device) != self.runtime):
            raise ValueError("Changed fork authority/runtime or poisoned instance")

    def state_dict(self, stream):
        state = super().state_dict(stream)
        state.update(purpose=PURPOSE, arm_roles=p.ROLES, arm_lambdas=p.LAMBDAS, arm_instrumental_weights=p.WEIGHTS, arm_accompaniment_weights=p.CA_WEIGHTS, arm_kill_bands=p.KILLS, teacher="kim_melband", smoke=self.smoke,
                     origin_sha256=self.origin_sha256, source_stopped_at=self.source_stopped_at, source_legacy_stop_events=self.source_legacy_stop_events,
                     tranche_stopping="new_common_fixed500_budget_preserve_legacy_stop_evidence",
                     legacy_stop_events=list(self.legacy_stop_events))
        return d.portable(state)

    def validate_state(self, state):
        p.require_weights(state.get("arm_instrumental_weights"))
        p.require_ca_weights(state.get("arm_accompaniment_weights"))
        p.require_kills(state.get("arm_kill_bands"))
        if (state.get("purpose") != PURPOSE or state.get("arm_roles") != p.ROLES or state.get("arm_lambdas") != p.LAMBDAS or state.get("arm_instrumental_weights") != p.WEIGHTS or state.get("arm_accompaniment_weights") != p.CA_WEIGHTS or
            state.get("teacher") != "kim_melband" or state.get("smoke") is not self.smoke or
            state.get("source_stopped_at") != self.source_stopped_at or
            state.get("source_legacy_stop_events") != self.source_legacy_stop_events or
            state.get("tranche_stopping") != "new_common_fixed500_budget_preserve_legacy_stop_evidence" or
            type(state.get("legacy_stop_events")) is not list or
            any(type(step) is not int or step not in (3750, 4000) or step > state.get("step", -1) for step in state["legacy_stop_events"]) or
            state["legacy_stop_events"] != sorted(set(state["legacy_stop_events"])) or
            (state.get("step", -1) < self.limit and state["schedule"]["stopped_at"] is not None) or
            state.get("origin_sha256") != self.origin_sha256 or not 3500 <= state.get("step", -1) <= self.limit or
            state["sampler"].get("teacher") != "kim_melband" or state["sampler"].get("approval_sha256") != self.binding["approval_sha256"]):
            raise ValueError("Changed fork origin, loss roles, teacher, phase or cursor")
        super().validate_state(state | {"purpose": old.PURPOSE})

    def apply_state(self, state, stream):
        super().apply_state(state, stream)
        self.legacy_stop_events = list(state["legacy_stop_events"])

    def observe(self, validator, factory):
        packet = old.observe(self, validator, factory)
        if self.schedule.stopped_at is not None:
            self.legacy_stop_events.append(self.schedule.stopped_at)
        if self.step < self.limit:
            self.schedule.stopped_at = None
        else:
            self.schedule.stopped_at = self.step
        packet.update(source_stopped_at=self.source_stopped_at, source_legacy_stop_events=self.source_legacy_stop_events,
            tranche_stopping="new_common_fixed500_budget_preserve_legacy_stop_evidence",
            legacy_stop_events=list(self.legacy_stop_events))
        return packet

    def _backward(self, net, x, v, wa, gs, device, warmup, kill, objective, microbatch):
        if (warmup, kill, objective, microbatch) != (96, 44, "reconstruction_l1", 1):
            raise ValueError("Original reconstruction geometry required")
        arm = next((arm for arm in ARMS if net is self.models[arm]), None)
        if arm is None:
            raise ValueError("Unknown loss arm")
        result = boundary_backward(net, x, v, wa, gs, device, self._stream.last_metadata, p.KILLS[arm])
        result["kill_bands"] = p.KILLS[arm]
        return result

    def update_next(self, stream):
        self._stream = stream
        original = stream.next_batch
        def checked_batch():
            batch = original()
            validate_metadata(batch["metadata"])
            if not torch.equal(batch["targets"][ARMS[0]], batch["targets"][ARMS[1]]):
                raise ValueError("Loss-only experiment must share ALL Mel targets")
            return batch
        stream.next_batch = checked_batch
        try:
            row = super().update_next(stream)
            row.update(additional_step=self.step-3500, arm_roles=p.ROLES, teacher="kim_melband",
                slot_weights=p.SLOT_WEIGHTS, arm_lambdas=p.LAMBDAS, arm_instrumental_weights=p.WEIGHTS, arm_accompaniment_weights=p.CA_WEIGHTS, arm_kill_bands=p.KILLS)
            return row
        finally:
            stream.next_batch = original


def save_checkpoint(out, engine, stream):
    state = engine.state_dict(stream)
    name = f"NONRELEASE_lf_boundary_step_{engine.step:04d}.pt"
    path = out / name
    if path.exists():
        if not m.equal_state(m.load_checked_checkpoint(path, acq.sha256(path)), state):
            raise ValueError("Existing checkpoint conflicts; preserve it")
    else:
        m.save_new(path, state)
    receipt = acq.seal({"schema": 1, "purpose": PURPOSE, "step": engine.step, "additional_step": engine.step-3500,
        "checkpoint": name, "sha256": acq.sha256(path), "binding": engine.binding, "arm_roles": p.ROLES, "arm_lambdas": p.LAMBDAS, "arm_instrumental_weights": p.WEIGHTS, "arm_accompaniment_weights": p.CA_WEIGHTS, "arm_kill_bands": p.KILLS,
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


def auxiliary_evidence(draws):
    result = {}
    for arm in ARMS:
        if any(row.get("arm_kill_bands") != p.KILLS or row["losses"][arm].get("kill_bands") != p.KILLS[arm] for row in draws):
            raise ValueError("Actual per-arm LF reconstruction identity required")
        active = sum(row["losses"][arm]["auxiliary_active_count"] for row in draws)
        contribution = sum(row["losses"][arm]["auxiliary_contribution"] for row in draws)
        if active < 1 or not math.isfinite(contribution) or contribution <= 0:
            raise ValueError("Both real strength branches must exercise nonzero auxiliary")
        base = sum(row["losses"][arm]["instrumental_base_loss"] for row in draws)
        weighted = sum(row["losses"][arm]["instrumental_weighted_contribution"] for row in draws)
        extra = sum(row["losses"][arm]["instrumental_extra_contribution"] for row in draws)
        if (not math.isfinite(base) or base <= 0 or not math.isfinite(weighted) or weighted <= 0 or
            any(type(row["losses"][arm]["instrumental_weight"]) is not int or
                row["losses"][arm]["instrumental_weight"] != p.WEIGHTS[arm] for row in draws) or
            not math.isclose(weighted, p.WEIGHTS[arm]*base, rel_tol=1e-6) or
            not math.isclose(extra, (p.WEIGHTS[arm]-1)*base, rel_tol=1e-6, abs_tol=1e-12)):
            raise ValueError("Both real branches must exercise declared nonzero instrumental protection")
        ca_contribution = sum(row["losses"][arm]["accompaniment_component_contribution"] for row in draws)
        cv_contribution = sum(row["losses"][arm]["remaining_vocal_component_contribution"] for row in draws)
        if (not math.isfinite(ca_contribution) or not math.isfinite(cv_contribution) or cv_contribution <= 0 or
            any(type(row["losses"][arm]["accompaniment_weight"]) is not int or row["losses"][arm]["accompaniment_weight"] != p.CA_WEIGHTS[arm] for row in draws) or
            (p.CA_WEIGHTS[arm] == 0 and ca_contribution != 0.) or (p.CA_WEIGHTS[arm] == 1 and ca_contribution <= 0)):
            raise ValueError("Actual nonzero residual and declared full/residual-only branches required")
        result[arm] = {"accompaniment_weight": p.CA_WEIGHTS[arm], "accompaniment_component_contribution": ca_contribution,
            "remaining_vocal_component_contribution": cv_contribution, "coefficient": p.LAMBDAS[arm], "active_count": active, "contribution": contribution,
            "instrumental_weight": p.WEIGHTS[arm], "instrumental_base_loss": base,
            "instrumental_weighted_contribution": weighted, "instrumental_extra_contribution": extra}
    return result


def check_cross_device_inputs(cpu, cuda):
    def identity(proof):
        for row in proof["draws"]:
            p.require_weights(row["arm_instrumental_weights"])
            p.require_ca_weights(row["arm_accompaniment_weights"])
        return [{key: row[key] for key in ("step", "input_sha256", "metadata", "arm_roles", "arm_lambdas", "arm_instrumental_weights", "arm_accompaniment_weights", "arm_kill_bands")}
                for row in proof["draws"]]
    if identity(cpu) != identity(cuda):
        raise ValueError("CPU/CUDA mechanism used different PCM or input metadata")


def smoke(approval, out, device):
    old.require_fresh(out)
    doc = p.verified_approval(approval)
    if device == "cuda" and not old.exploration_gpu_preflight(p.inp.verified_approval(Path(doc["origin_approval"])))['allowed']:
        return 2
    out.mkdir(parents=True)
    with d.deterministic_runtime(device):
        stream = p.BoundaryStream(approval)
        factory = m.frozen_factory(doc["source_protocol"])
        engine = ForkEngine(factory, stream, device, smoke=True)
        initial = engine.state_dict(stream)
        source = stream.source_state
        source_equal = all(m.equal_state(initial["arms"][arm], source["arms"][p.SOURCE_ARM]) for arm in ARMS)
        schedule_equal = all(initial["schedule"][field] == {arm: source["schedule"][field][p.SOURCE_ARM] for arm in ARMS}
                             for field in ("best", "stale", "patience_anchor"))
        if not source_equal or not schedule_equal:
            raise ValueError("Mechanism did not migrate full-control3500 ARMS[0] state and schedule")
        draws, states = [], []
        for _ in range(3):
            draws.append(engine.update_next(stream))
            states.append(engine.state_dict(stream))
        m.save_new(out / "resume3501.pt", states[0])
        other = p.BoundaryStream(approval)
        replay = ForkEngine(factory, other, device, smoke=True)
        saved = m.load_checked_checkpoint(out / "resume3501.pt", acq.sha256(out / "resume3501.pt"))
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
                m.random.random()
                m.np.random.random()
                if device == "cuda":
                    torch.rand(2, device="cuda")
                raise RuntimeError("injected auxiliary arm")
            return original(net, *args)
        replay.backward = fail_second
        try:
            replay.update_next(other)
        except RuntimeError as error:
            if str(error) != "injected auxiliary arm":
                raise
        else:
            raise ValueError("No failure injection")
        if not m.equal_state(before, replay.state_dict(other)):
            raise ValueError("Rollback differs")
        exercised = auxiliary_evidence(draws)
        proof = {"schema": 1, "purpose": PURPOSE, "binding": binding(approval), "runtime": engine.runtime,
            "device": device, "mechanism_only": True, "updates_per_arm": 3,
            "distinct_input_steps": [3501, 3502, 3503], "successful_update_executions_including_replay": 5,
            "arm_lambdas": p.LAMBDAS, "arm_instrumental_weights": p.WEIGHTS, "arm_accompaniment_weights": p.CA_WEIGHTS, "arm_kill_bands": p.KILLS, "auxiliary_by_arm": exercised,
            "source_arm": p.SOURCE_ARM, "source_full_arms_equal": source_equal, "source_schedule_equal": schedule_equal,
            "source_stopped_at": engine.source_stopped_at, "source_legacy_stop_events": engine.source_legacy_stop_events, "new_tranche_stop_reset": initial["schedule"]["stopped_at"] is None,
            "tranche_stopping": initial["tranche_stopping"],
            "initial_equal": m.equal_state(initial["arms"][ARMS[0]], initial["arms"][ARMS[1]]),
            "initial_source_model_digest": old.dev.state_digest(initial["arms"][ARMS[0]]["model"]),
            "source_device": stream.source_state["runtime"]["device"], "cpu_rng_device_migration": device == "cpu",
            "source_rng_restored": m.equal_state(initial["rng"], stream.source_state["rng"] | ({"torch_cuda": []} if device == "cpu" else {})),
            "disk_resume_identical": True, "rollback_identical": True, "saved_state_unaliased": True,
            "resume_sha256": acq.sha256(out / "resume3501.pt"), "draws": draws, "release_selection": "NONE", "deployment": False}
        acq.write_new_json(out / "mechanism.json", acq.seal(proof))
    print(f"MEL_LF_BOUNDARY_MECHANISM PASS device={device}; real input3; disk resume/rollback; no quality claim", flush=True)
    return 0


def verify_proof(path, approval, device):
    doc = acq.read_sealed(path / "mechanism.json")
    p.require_weights(doc.get("arm_instrumental_weights"))
    p.require_ca_weights(doc.get("arm_accompaniment_weights"))
    p.require_kills(doc.get("arm_kill_bands"))
    authority = p.verified_approval(approval)
    source = p.origin_state(authority)
    if (doc["purpose"] != PURPOSE or doc["binding"] != binding(approval) or doc["device"] != device or doc["mechanism_only"] is not True or
        doc["updates_per_arm"] != 3 or doc["source_arm"] != p.SOURCE_ARM or doc["arm_lambdas"] != p.LAMBDAS or doc["arm_instrumental_weights"] != p.WEIGHTS or doc.get("arm_accompaniment_weights") != p.CA_WEIGHTS or
        doc["source_stopped_at"] != source["schedule"]["stopped_at"] or doc["source_legacy_stop_events"] != source["legacy_stop_events"] or doc["new_tranche_stop_reset"] is not True or
        doc["tranche_stopping"] != authority["protocol"]["tranche_stopping"] or
        doc["initial_source_model_digest"] != old.dev.state_digest(source["arms"][p.SOURCE_ARM]["model"]) or
        doc["distinct_input_steps"] != [3501, 3502, 3503] or [row["step"] for row in doc["draws"]] != [3501, 3502, 3503] or
        doc["auxiliary_by_arm"] != auxiliary_evidence(doc["draws"]) or
        not all(doc[k] is True for k in ("initial_equal", "source_full_arms_equal", "source_schedule_equal", "source_rng_restored", "disk_resume_identical", "rollback_identical", "saved_state_unaliased")) or
        doc["cpu_rng_device_migration"] is not (device == "cpu") or
        doc["source_device"] != source["runtime"]["device"] or (device == "cuda" and doc["runtime"] != source["runtime"]) or
        doc["release_selection"] != "NONE" or doc["deployment"] is not False or
        acq.sha256(path / "resume3501.pt") != doc["resume_sha256"]):
        raise ValueError("Missing current migrated-state mechanism proof")
    saved = m.load_checked_checkpoint(path / "resume3501.pt", doc["resume_sha256"])
    p.require_weights(saved.get("arm_instrumental_weights"))
    p.require_ca_weights(saved.get("arm_accompaniment_weights"))
    p.require_kills(saved.get("arm_kill_bands"))
    if (saved["purpose"] != PURPOSE or saved["binding"] != doc["binding"] or saved["step"] != 3501 or saved["limit"] != 3503 or
        saved["runtime"] != doc["runtime"] or saved["arm_lambdas"] != p.LAMBDAS or saved["arm_instrumental_weights"] != p.WEIGHTS or saved.get("arm_accompaniment_weights") != p.CA_WEIGHTS or not saved["smoke"] or
        saved["source_stopped_at"] != doc["source_stopped_at"] or saved["source_legacy_stop_events"] != doc["source_legacy_stop_events"] or saved["tranche_stopping"] != doc["tranche_stopping"] or
        saved["legacy_stop_events"] != [] or saved["schedule"]["stopped_at"] is not None or
        saved["sampler"]["cursor"] != 3501 or saved["origin_sha256"] != authority["protocol"]["origin_checkpoint_sha256"] or
        any(saved["arms"][arm]["updates"] != 3501 for arm in ARMS) or not d.finite_state(saved)):
        raise ValueError("Mechanism resume state is incomplete")
    for row in doc["draws"]:
        p.require_kills(row.get("arm_kill_bands"))
        validate_metadata(row["metadata"])
        if len(row["input_sha256"]) != 6 or row["arm_lambdas"] != p.LAMBDAS or row["arm_instrumental_weights"] != p.WEIGHTS or row.get("arm_accompaniment_weights") != p.CA_WEIGHTS or row["arm_roles"] != p.ROLES:
            raise ValueError("Mechanism input/strength identity changed")
    return doc


def low_frequency_backing_metrics(predicted_vocal, mix, true_vocal):
    """Fixed bins0..5/native frames98..349, descriptive ONLY, no policy change."""
    if (predicted_vocal.shape != mix.shape or true_vocal.shape != mix.shape or
        mix.shape != (1, 2, 89856) or any(v.device.type != "cpu" or v.dtype != torch.float32 or
        not torch.isfinite(v).all() for v in (predicted_vocal, mix, true_vocal))):
        raise ValueError("Original finite CPU FP32 development geometry")
    error = m.core.stft_batch(true_vocal-predicted_vocal)[:, :, :6, 98:350]
    reference = m.core.stft_batch(mix-true_vocal)[:, :, :6, 98:350]
    weights = torch.tensor([1., 2., 2., 2., 2., 2.], dtype=torch.float64)[None, None, :, None]
    error_power = float((error.detach().abs().double().square()*weights).mean())
    reference_power = float((reference.detach().abs().double().square()*weights).mean())
    return {"lf_backing_error_power": error_power, "lf_backing_reference_power": reference_power,
            "lf_backing_error_snr_db": None if reference_power == 0 else
                10*math.log10(reference_power/max(error_power, 1e-12))}


@torch.no_grad()
def boundary_evaluation(net, rows, wa, gs, kill):
    """Same119 FP32 prediction and original metrics, plus a separate LF sidecar."""
    net.eval()
    measured, low = [], []
    interval = old.dev.suite.scoring_slice(rows[0]["x"].shape[-1], 96)
    for row in rows:
        x, v = row["x"][None], row["v"][None]
        spectrum, bands, _ = m.core.prepare_truth(x, v, wa)
        output = net(bands)
        pv = m.core.product_vocal(spectrum, (output[:, :2]+1)/2, gs, x.shape[-1], kill)
        if not torch.isfinite(pv).all():
            raise ValueError("Nonfinite development prediction")
        descriptor = {key:value for key,value in row.items() if key not in ("x", "v")}
        metrics = old.dev.suite.separation_metrics(pv[0, ..., interval], x[0, ..., interval], v[0, ..., interval])
        measured.append(descriptor | {"metrics": metrics})
        low.append(descriptor | {"metrics": low_frequency_backing_metrics(pv, x, v)})
    return ({"rows": measured, "summary": old.dev.suite.aggregate(measured)},
            {"rows": low, "summary": old.dev.suite.aggregate(low)})


class BoundaryValidator(old.dev.PairedDevelopmentValidator):
    """Unchanged suite/policy/frozen44, explicit per-arm LF; no global overrides."""
    def __init__(self, legacy):
        self.rows, self.manifest = legacy.rows, copy.deepcopy(legacy.manifest)
        self.policy, self.policy_digest = copy.deepcopy(legacy.policy), legacy.policy_digest
        self.evaluator, self.wa, self.gs = legacy.evaluator, legacy.wa, legacy.gs
        self.baseline, self.baseline_digest = copy.deepcopy(legacy.baseline), legacy.baseline_digest
        self.low_frequency = {}

    def _score_at(self, net, kill):
        if type(kill) is not int or kill not in (44, 32):
            raise ValueError("Explicit per-arm LF44/32 required")
        self._check_inputs()
        if any(value.device.type != "cpu" for value in (*net.parameters(), *net.buffers())):
            raise ValueError("Development is CPU-only")
        original = d.portable(net.state_dict())
        modes, rng = [module.training for module in net.modules()], m.capture_rng("cpu")
        try:
            with torch.no_grad():
                result, low = boundary_evaluation(net, self.rows, self.wa, self.gs, kill)
                self.low_frequency[kill] = low
            if not m.equal_state(original, net.state_dict()):
                raise ValueError("Read-only evaluation changed weights")
            self._check_inputs()
            if ([{k:v for k,v in row.items() if k != "metrics"} for row in result["rows"]] != self.manifest["rows"] or
                result["summary"] != old.dev.suite.aggregate(result["rows"])):
                raise ValueError("Original coverage/aggregation differs")
            return result
        finally:
            net.load_state_dict(original, strict=True)
            for module, mode in zip(net.modules(), modes):
                module.training = mode
            m.restore_rng(rng, "cpu")

    def evaluate_pair(self, models, step):
        if set(models) != set(ARMS) or type(step) is not int or step not in (3500, 3750, 4000):
            raise ValueError("Both LF arms at an authorized complete stage")
        if acq.content_digest(self.baseline) != self.baseline_digest:
            raise ValueError("Frozen44 baseline changed")
        self.low_frequency = {}
        evaluations = {arm: self._score_at(models[arm], p.KILLS[arm]) for arm in ARMS}
        scores = {arm: old.dev.suite.assess(evaluations[arm]["summary"], self.baseline["summary"]) for arm in ARMS}
        return {"step": step, "suite_sha256": self.manifest["sha256"], "policy_sha256": self.policy_digest,
            "baseline_scores_sha256": self.baseline_digest, "scores": scores, "evaluations": evaluations,
            "model_state_sha256": {arm: old.dev.state_digest(models[arm].state_dict()) for arm in ARMS},
            "arm_kill_bands": p.KILLS, "frozen_kill_bands": 44,
            "low_frequency_backing": {arm: self.low_frequency.get(p.KILLS[arm]) for arm in ARMS},
            "low_frequency_scope": "Separate descriptive bins0..5/native98..349 spectrum; unchanged119 eligibility",
            "baseline_interpretation": "LF32 baseline is zero-update direct effect, not training gain",
            "scope": "Old true-reference development only; not independent acceptance or release approval"}


def train(approval, out, cpu_proof, cuda_proof, resume=False):
    doc = p.verified_approval(approval)
    cpu = verify_proof(cpu_proof, approval, "cpu")
    cuda = verify_proof(cuda_proof, approval, "cuda")
    check_cross_device_inputs(cpu, cuda)
    p.verify_audit()
    guard = old.exploration_gpu_preflight(inp_doc := p.inp.verified_approval(Path(doc["origin_approval"])))
    if not guard["allowed"]:
        print("MEL_LF_BOUNDARY_TRAIN DEFERRED insufficient free GPU memory", flush=True)
        return 2
    if resume:
        if not out.exists() or (out / "completion.json").exists():
            raise ValueError("Explicit resume only unfinished existing run")
    else:
        old.require_fresh(out)
        out.mkdir(parents=True)
    with m.bulk.worker_lock(out), d.deterministic_runtime("cuda"):
        stream = p.BoundaryStream(approval)
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
            status(out, {"status": "running", "phase": "development_baseline", "step": engine.step, "additional_step": engine.step-3500, "limit": 4000, "latest_checkpoint": latest, "error": None})
            rng = d.portable(m.capture_rng("cuda"))
            try:
                legacy_validator, manifest = old.make_validator(inp_doc)
                validator = BoundaryValidator(legacy_validator)
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
                    start = validator.evaluate_pair(models, 3500)
                finally:
                    m.restore_rng(rng, "cuda")
                # Same model/Adam, but different LF outputs. Only LF44 must reproduce the source full-control score.
                original = acq.read_sealed(p.SOURCE / "development_step_3500.json")
                if start["evaluations"][ARMS[0]] != original["evaluations"][p.SOURCE_ARM]:
                    raise ValueError("LF44 baseline differs from source full-control3500 score")
                start.update(purpose=PURPOSE, arm_roles=p.ROLES, arm_lambdas=p.LAMBDAS, arm_instrumental_weights=p.WEIGHTS, arm_accompaniment_weights=p.CA_WEIGHTS, arm_kill_bands=p.KILLS, teacher="kim_melband", additional_step=0,
                    source_stopped_at=engine.source_stopped_at, source_legacy_stop_events=engine.source_legacy_stop_events, tranche_stopping=doc["protocol"]["tranche_stopping"], legacy_stop_events=[])
                acq.write_new_json(out / "development_step_3500.json", acq.seal(start))
            else:
                if acq.read_sealed(out / "selection_suite.json")["sha256"] != manifest["sha256"] or acq.read_sealed(out / "frozen_scores.json") != acq.seal(copy.deepcopy(validator.baseline)):
                    raise ValueError("Development suite/baseline changed")
            phase = "training"
            with (out / f"updates_from_{engine.step:04d}_{time.time_ns()}.jsonl").open("x", encoding="utf-8", buffering=1) as log:
                while engine.step < 4000 and engine.schedule.stopped_at is None:
                    if shutil.disk_usage(out).free < 12*1024**3 + 256*1024**2:
                        raise ValueError("Disk reserve reached; no cleanup")
                    row = engine.update_next(stream)
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                    if engine.step % 10 == 0 or engine.step == 3501:
                        status(out, {"status": "running", "phase": "training", "step": engine.step, "additional_step": engine.step-3500,
                            "limit": 4000, "latest_checkpoint": latest, "last_losses": row["losses"], "last_step_seconds": row["seconds"], "error": None})
                        print(f"MEL_LF_BOUNDARY step={engine.step}/4000 additional={engine.step-3500}/500 seconds={row['seconds']:.3f}", flush=True)
                    if engine.step % 250 == 0:
                        phase = "development_validation"
                        status(out, {"status": "running", "phase": phase, "step": engine.step, "additional_step": engine.step-3500, "limit": 4000, "latest_checkpoint": latest, "error": None})
                        packet = engine.observe(validator, factory)
                        packet.update(purpose=PURPOSE, arm_roles=p.ROLES, arm_lambdas=p.LAMBDAS, arm_instrumental_weights=p.WEIGHTS, arm_accompaniment_weights=p.CA_WEIGHTS, arm_kill_bands=p.KILLS, teacher="kim_melband", additional_step=engine.step-3500)
                        path = out / f"development_step_{engine.step:04d}.json"
                        if path.exists():
                            if acq.read_sealed(path) != acq.seal(packet):
                                raise ValueError("Replayed development differs")
                        else:
                            acq.write_new_json(path, acq.seal(packet))
                        latest = save_checkpoint(out, engine, stream)
                        print(f"MEL_LF_BOUNDARY_DEV step={engine.step} scores={packet['scores']}", flush=True)
                        phase = "training"
            if engine.step != 4000 or stream.cursor != 4000:
                raise ValueError("Not500 updates: refuse zero/partial-step completion")
            files = {path.name: acq.sha256(path) for path in out.iterdir() if path.suffix in (".json", ".jsonl", ".pt") and path.name != "run_status.json"}
            acq.write_new_json(out / "completion.json", acq.seal({"schema": 1, "purpose": PURPOSE, "binding": engine.binding,
                "step": engine.step, "additional_steps": engine.step-3500, "limit": 4000, "final_checkpoint": latest,
                "runtime": engine.runtime, "outputs_sha256": files, "teacher": "kim_melband", "arm_roles": p.ROLES, "arm_lambdas": p.LAMBDAS, "arm_instrumental_weights": p.WEIGHTS, "arm_accompaniment_weights": p.CA_WEIGHTS, "arm_kill_bands": p.KILLS,
                "source_stopped_at": engine.source_stopped_at, "source_legacy_stop_events": engine.source_legacy_stop_events, "tranche_stopping": doc["protocol"]["tranche_stopping"],
                "legacy_stop_events": list(engine.legacy_stop_events),
                "release_selection": "NONE", "deployment": False, "independent_acceptance_ready": False}))
            status(out, {"status": "complete", "phase": "lf_boundary_tranche_complete", "step": engine.step,
                "additional_step": engine.step-3500, "limit": 4000, "latest_checkpoint": latest, "error": None})
            print("MEL_LF_BOUNDARY COMPLETE500; release_selection=NONE", flush=True)
        except BaseException as error:
            status(out, {"status": "failed", "phase": phase, "step": engine.step, "additional_step": engine.step-3500,
                "limit": 4000, "latest_checkpoint": latest, "error": repr(error), "recovery": "Diagnose; explicit resume only from last full commit"})
            raise
    return 0


def verify(out, approval):
    p.verified_approval(approval)
    doc = acq.read_sealed(out / "completion.json")
    p.require_kills(doc.get("arm_kill_bands"))
    if (doc["purpose"] != PURPOSE or doc["binding"] != binding(approval) or doc["step"] != 4000 or
        doc["additional_steps"] != 500 or doc["teacher"] != "kim_melband" or doc["arm_roles"] != p.ROLES or
        doc["arm_lambdas"] != p.LAMBDAS or doc["arm_instrumental_weights"] != p.WEIGHTS or doc.get("arm_accompaniment_weights") != p.CA_WEIGHTS or doc["source_stopped_at"] != 3500 or
        doc["tranche_stopping"] != "new_common_fixed500_budget_preserve_legacy_stop_evidence" or
        doc["release_selection"] != "NONE" or doc["deployment"] is not False or not doc["outputs_sha256"]):
        raise ValueError("Invalid completion or missing output hashes")
    for name, digest in doc["outputs_sha256"].items():
        if Path(name).name != name or acq.sha256(out / name) != digest:
            raise ValueError("Completion output differs")
    state = m.load_checked_checkpoint(out / doc["final_checkpoint"]["checkpoint"], doc["final_checkpoint"]["sha256"])
    if state["step"] != 4000 or state["smoke"] or state["sampler"]["cursor"] != 4000 or any(a["updates"] != 4000 for a in state["arms"].values()):
        raise ValueError("Incomplete final state")
    for step in (3500, 3750, 4000):
        packet = acq.read_sealed(out / f"development_step_{step:04d}.json")
        if packet["step"] != step or packet["suite_sha256"] != old.dev.EXPECTED_SUITE:
            raise ValueError("Development coverage changed")
    print("MEL_LF_BOUNDARY_VERIFY PASS500; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("smoke", "train", "verify"))
    parser.add_argument("--approval", type=Path, default=p.DEFAULT_APPROVAL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-proof", type=Path, default=CPU_PROOF)
    parser.add_argument("--cuda-proof", type=Path, default=CUDA_PROOF)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = args.out or ((CPU_PROOF if args.device == "cpu" else CUDA_PROOF) if args.operation == "smoke" else DEFAULT_OUT)
    if args.operation == "smoke":
        raise SystemExit(smoke(args.approval, out, args.device))
    elif args.operation == "verify":
        verify(out, args.approval)
    else:
        if args.device != "cuda":
            raise ValueError("No CPU long-training fallback")
        raise SystemExit(train(args.approval, out, args.cpu_proof, args.cuda_proof, args.resume))


