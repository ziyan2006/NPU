"""Fresh instrumental base-weight1/4 fork with fixed auxiliary.2/denominator6."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import torch

spec = importlib.util.spec_from_file_location("protection_sources", Path(__file__).with_name("169_diagnose_gradient_contributions.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
t, p, m, d, acq, ROOT = q.r.t, q.r.p, q.m, q.d, q.acq, q.ROOT
old, inp, ARMS = t.old, p.inp, t.ARMS
PURPOSE = "NONRELEASE_MEL_INSTRUMENTAL_PROTECTION_FORK"
ROLES = dict(zip(ARMS, ("instrumental_weight1_control", "instrumental_weight4")))
LAMBDAS = dict(zip(ARMS, (.2, .2)))
WEIGHTS = dict(zip(ARMS, (1, 4)))
SLOT_WEIGHTS = {arm: [1, 1, WEIGHTS[arm], 1, 1, 1] for arm in ARMS}
SOURCE = t.DEFAULT_OUT
PROTOCOL = ROOT / "docs/mel_instrumental_protection_protocol_20261003.json"
DEFAULT_OUT = ROOT / "results/mel_instrumental_protection_import_20261003"
DEFAULT_APPROVAL = DEFAULT_OUT / "approval.json"
AUDIT_OUT = ROOT / "results/mel_instrumental_protection_audit_20261003"
REQUIRED = ("172_train_mel_instrumental_protection.py", "173_start_mel_instrumental_protection.ps1",
            "174_review_mel_instrumental_protection.py", "_test_mel_instrumental_protection.py",
            "170_instrumental_protection_loss.py", "_test_instrumental_protection_loss.py")


def require_weights(value):
    if (type(value) is not dict or value != WEIGHTS or set(value) != set(ARMS) or
        any(type(value[arm]) is not int for arm in ARMS)):
        raise ValueError("Exact instrumental arm weights required; bool is not weight1")


def check_protocol(doc):
    exact = {"schema": 1, "purpose": PURPOSE, "exploratory_training_authorized": True,
        "formal_training_authorized": False, "deployment_authorized": False, "teacher": "kim_melband",
        "approved_pseudo_songs": 24, "origin_step": 2500, "additional_common_steps": 500,
        "absolute_limit": 3000, "checkpoint_every": 250, "source_arm": ARMS[1], "source_stopped_at": 2500,
        "source_legacy_stop_events": [2250, 2500],
        "tranche_stopping": "new_common_fixed500_budget_preserve_legacy_stop_evidence",
        "internal_arm_keys": ROLES, "arm_lambdas": LAMBDAS, "arm_instrumental_weights": WEIGHTS,
        "arm_slot_weights": SLOT_WEIGHTS, "base_normalizer": 6,
        "gpu_concurrency_authorized": True, "minimum_free_mib": 2300, "minimum_disk_gib": 12,
        "mechanism_step_limit": 3, "release_selection": "NONE", "independent_acceptance_ready": False,
        "auxiliary_formula": "cv^2+(ca-1)^2", "rms_floor": .0001, "minimum_gram_determinant": .001,
        "auxiliary_domains": ["musdb", "mir1k"], "auxiliary_role": "train"}
    if any(doc.get(key) != value or type(doc.get(key)) is not type(value) for key, value in exact.items()):
        raise ValueError("Fresh fixed500 instrumental-only authority required")
    if (any(type(v) is not int for v in doc["arm_instrumental_weights"].values()) or
        any(type(v) is not int for row in doc["arm_slot_weights"].values() for v in row) or
        any(type(v) is not float for v in doc["arm_lambdas"].values())):
        raise ValueError("Exact weight/lambda types required")


def no_active_trainer():
    q.no_active_trainer()
    command = """Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '(172_train_mel_instrumental_protection)[.]py[" ]+(train|smoke)' } | Select-Object -ExpandProperty ProcessId"""
    active = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    if active.stdout.strip():
        raise ValueError("Active protection worker; preserve and postpone")


def checked_source():
    authority = p.verified_approval(p.DEFAULT_APPROVAL)
    completion_path = SOURCE / "completion.json"
    completion = acq.read_sealed(completion_path)
    proof_path = q.PROOF
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    checkpoint = SOURCE / "NONRELEASE_strength_step_2500.pt"
    receipt_path = SOURCE / "checkpoint_2500.json"
    receipt = acq.read_sealed(receipt_path)
    if (completion["binding"] != t.binding(p.DEFAULT_APPROVAL) or completion["step"] != 2500 or
        completion["additional_steps"] != 500 or completion["final_checkpoint"] != receipt or
        proof["normal_exit_verified"] is not True or proof["processes"] or
        proof["matched_exit"]["exit_code"] != 0 or proof["trainer_verify"]["exit_code"] != 0 or
        proof["score_review"]["exit_code"] != 0 or proof["score_review"]["pending"] or
        (proof["score_review"]["tracks"], proof["score_review"]["views"]) != (31, 177) or
        acq.sha256(completion_path) != proof["completion_sha256"] or
        acq.sha256(checkpoint) != proof["final_checkpoint_sha256"] or receipt["sha256"] != proof["final_checkpoint_sha256"] or
        receipt["checkpoint"] != checkpoint.name):
        raise ValueError("Independent normal500 complete candidate2500 evidence required")
    state = m.load_checked_checkpoint(checkpoint, receipt["sha256"])
    if (state["purpose"] != t.PURPOSE or state["binding"] != completion["binding"] or state["step"] != 2500 or
        state["limit"] != 2500 or state["smoke"] or state["sampler"]["cursor"] != 2500 or
        state["schedule"]["stopped_at"] != 2500 or state["legacy_stop_events"] != [2250, 2500] or
        state["deployment_authorized"] is not False or any(a["updates"] != 2500 for a in state["arms"].values())):
        raise ValueError("Source candidate/full-state/stop evidence changed")
    paths = [p.DEFAULT_APPROVAL, completion_path, proof_path, receipt_path, checkpoint, q.TEST,
             SOURCE / "development_step_2500.json", Path(q.__file__), q.REPORT,
             q.DEFAULT_OUT / "plan.json", q.DEFAULT_OUT / "diagnostic.json",
             ROOT / "results/train_gradient_contributions_monitor_20261003/completion_review.json"]
    paths += [ROOT / proof[key]["path"] for key in ("trainer_verify", "score_review")]
    for key in ("trainer_verify", "score_review"):
        if acq.sha256(ROOT / proof[key]["path"]) != proof[key]["sha256"]:
            raise ValueError("Recorded source verification changed")
    diagnostic = acq.read_sealed(q.DEFAULT_OUT / "diagnostic.json")
    plan = acq.read_sealed(q.DEFAULT_OUT / "plan.json")
    q.check_plan(plan)
    if (diagnostic["plan_sha256"] != acq.sha256(q.DEFAULT_OUT / "plan.json") or
        diagnostic["bindings_sha256"] != plan["bindings_sha256"] or diagnostic["model_updates"] != 0 or
        diagnostic["model_and_modes_unchanged"] is not True or len(diagnostic["rows"]) != 36):
        raise ValueError("Sealed complete TRAIN diagnosis required")
    bindings = authority["bindings_sha256"] | plan["bindings_sha256"]
    for path, digest in bindings.items():
        if acq.sha256(path) != digest:
            raise ValueError("Source historical/diagnostic binding changed")
    bindings |= {str(path.resolve()): acq.sha256(path) for path in paths}
    return authority, state, bindings


def check_approval(doc):
    check_protocol(doc["protocol"])
    require_weights(doc.get("arm_instrumental_weights"))
    if (doc.get("purpose") != PURPOSE or doc.get("deployment_authorized") is not False or
        doc.get("release_selection") != "NONE" or doc.get("arm_roles") != ROLES or
        doc.get("arm_instrumental_weights") != WEIGHTS or doc.get("teacher") != "kim_melband" or
        doc.get("source_binding") != t.binding(Path(doc["source_approval"]))):
        raise ValueError("Changed source/loss/release authority")


def verified_approval(path):
    doc = acq.read_sealed(path)
    check_approval(doc)
    for filename, digest in doc["bindings_sha256"].items():
        if acq.sha256(filename) != digest:
            raise ValueError(f"Protection dependency changed: {filename}")
    p.verified_approval(Path(doc["source_approval"]))
    inp.verified_approval(Path(doc["origin_approval"]))
    return doc


def origin_state(doc):
    source_doc, state, _ = checked_source()
    if (Path(doc["source_approval"]) != p.DEFAULT_APPROVAL or
        Path(doc["origin_approval"]) != Path(source_doc["origin_approval"]) or
        acq.sha256(doc["origin_checkpoint"]) != doc["protocol"]["origin_checkpoint_sha256"] or
        Path(doc["origin_checkpoint"]) != SOURCE / "NONRELEASE_strength_step_2500.pt"):
        raise ValueError("Exact complete candidate2500 source required")
    return state


class ProtectionStream:
    def __init__(self, path=None, preparation_doc=None):
        if (path is None) == (preparation_doc is None):
            raise ValueError("Explicit sealed OR preparation-only stream required")
        self.path = Path(path) if path is not None else None
        self.doc = verified_approval(self.path) if self.path else copy.deepcopy(preparation_doc)
        check_approval(self.doc)
        self.bound = acq.sha256(self.path) if self.path else "PREPARATION_ONLY_NO_TRAINING"
        self.dataset = inp.ApprovedTeacherDataset(inp.verified_approval(Path(self.doc["origin_approval"])), "kim_melband")
        self.seed, self.config = self.dataset.seed, self.dataset.config
        self.true = m.LockedTruePool(m.bulk.OLD_LOCK, self.config)
        self.source_state = origin_state(self.doc)
        sampler = self.source_state["sampler"]
        if (sampler["seed"] != self.seed or sampler["true_lock_sha256"] != self.true.bound or
            sampler["approval_sha256"] != acq.sha256(self.doc["source_approval"]) or sampler["teacher"] != "kim_melband"):
            raise ValueError("Changed common input/seed/teacher")
        self.cursor, self.last_metadata = 2500, None

    def state_dict(self):
        return {"approval_sha256": self.bound, "true_lock_sha256": self.true.bound,
                "seed": self.seed, "cursor": self.cursor, "teacher": "kim_melband"}

    def load_state_dict(self, state):
        if (self.path is None or state != self.state_dict() | {"cursor": state.get("cursor")} or
            type(state.get("cursor")) is not int or not 2500 <= state["cursor"] <= 3000 or
            acq.sha256(self.path) != self.bound):
            raise ValueError("Changed protection stream/cursor")
        self.cursor, self.last_metadata = state["cursor"], None

    def next_batch(self):
        if self.cursor >= 3000:
            raise ValueError("New protection sampler500 budget exhausted")
        xs, vs, metadata = [], [], []
        for domain in m.DOMAINS[:3]:
            item = self.true.crop(domain, self.seed, self.cursor)
            xs.append(item["x"]); vs.append(item["v"]); metadata.append(item["meta"])
        for index in range(3):
            recipe = m.data.crop_recipe(self.dataset.rows, self.config, self.seed, self.cursor*3+index)
            item = self.dataset.crop(recipe)
            xs.append(item["x"]); vs.append(item["v"])
            metadata.append(item["meta"] | {"domain": "pseudo", "role": "pseudo_label_train_candidate"})
        t.validate_metadata(metadata)
        x, v = torch.stack(xs), torch.stack(vs)
        row = {"x": x, "targets": {arm: v.clone() for arm in ARMS}, "domains": m.DOMAINS,
               "cursor": self.cursor, "metadata": metadata}
        self.cursor += 1
        self.last_metadata = copy.deepcopy(metadata)
        return row


def draft_document():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    check_protocol(protocol)
    source_doc, _, source_bindings = checked_source()
    if (acq.sha256(q.DEFAULT_OUT / "diagnostic.json") != protocol["diagnostic_sha256"]):
        raise ValueError("Wrong sealed TRAIN diagnosis")
    files = [Path(__file__), PROTOCOL, ROOT / "reports/52_train_gradient_contribution_result.md",
             ROOT / "reports/53_instrumental_protection_trial_plan.md"]
    files += [Path(__file__).with_name(name) for name in REQUIRED]
    bindings = source_bindings | {str(path.resolve()): acq.sha256(path) for path in files}
    doc = {"schema": 1, "purpose": PURPOSE, "protocol": protocol, "source_protocol": source_doc["source_protocol"],
        "source_approval": str(p.DEFAULT_APPROVAL), "source_binding": t.binding(p.DEFAULT_APPROVAL),
        "origin_approval": source_doc["origin_approval"], "origin_checkpoint": str(SOURCE / "NONRELEASE_strength_step_2500.pt"),
        "arm_roles": ROLES, "arm_instrumental_weights": WEIGHTS, "teacher": "kim_melband", "bindings_sha256": bindings,
        "deployment_authorized": False, "release_selection": "NONE"}
    check_approval(doc)
    origin_state(doc)
    return doc


def verify_audit():
    doc = draft_document()
    plan = acq.read_sealed(AUDIT_OUT / "plan.json")
    result = acq.read_sealed(AUDIT_OUT / "audit.json")
    if (plan["bindings_sha256"] != doc["bindings_sha256"] or
        result["plan_sha256"] != acq.sha256(AUDIT_OUT / "plan.json") or result["purpose"] != PURPOSE or
        result["counter"] != 2500 or result["model_updates"] != 0 or result["optimizer_constructed"] is not False or
        result["cuda_used"] is not False or result["model_and_modes_unchanged"] is not True or
        result["control_bit_exact"] is not True or result["gradient_delta_checked"] is not True or
        result["instrumental_extra_contribution"] <= 0 or result["auxiliary_active_count"] < 1 or
        result["source_model_digest"] != old.dev.state_digest(origin_state(doc)["arms"][ARMS[1]]["model"])):
        raise ValueError("Incomplete bound real TRAIN weight audit")
    t.validate_metadata(result["metadata"])
    if len(result["input_sha256"]) != 6 or result["input_sha256"] != [x["input_pcm_sha256"] for x in result["metadata"]]:
        raise ValueError("Audit common input changed")
    print("PROTECTION TRAIN_AUDIT VERIFIED; CPU, model updates0", flush=True)
    return result


def audit(out):
    if out != AUDIT_OUT:
        raise ValueError("Fixed predeclared audit directory")
    old.require_fresh(out)
    no_active_trainer()
    doc = draft_document()
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal({"schema": 1, "purpose": PURPOSE,
        "counter": 2500, "model_updates": 0, "cuda_used": False, "bindings_sha256": doc["bindings_sha256"]}))
    spec = importlib.util.spec_from_file_location("audit_fresh_trainer", Path(__file__).with_name(REQUIRED[0]))
    fresh = importlib.util.module_from_spec(spec); spec.loader.exec_module(fresh)
    torch.set_num_threads(2)
    with d.deterministic_runtime("cpu"):
        stream = ProtectionStream(preparation_doc=doc)
        batch = stream.next_batch()
        x, v = batch["x"], batch["targets"][ARMS[0]]
        wa = torch.from_numpy(m.core.t09.make_analysis_matrix())
        gs = torch.from_numpy(m.core.t09.make_synthesis_matrix())
        saved = stream.source_state["arms"][ARMS[1]]
        nets = []
        for _ in range(4):
            net = m.frozen_factory(doc["source_protocol"])()
            net.load_state_dict(saved["model"], strict=True)
            for module, mode in zip(net.modules(), saved["modes"]):
                module.training = mode
            nets.append(net)
        def gradients(net):
            return torch.cat([parameter.grad.detach().flatten() for parameter in net.parameters()])
        source = t.aux_backward(nets[0], x, v, wa, gs, "cpu", batch["metadata"], .2)
        control = fresh.protection_backward(nets[1], x, v, wa, gs, "cpu", batch["metadata"], 1)
        candidate = fresh.protection_backward(nets[2], x, v, wa, gs, "cpu", batch["metadata"], 4)
        if any(control[key] != value for key, value in source.items()) or not torch.equal(gradients(nets[0]), gradients(nets[1])):
            raise ValueError("Weight1 does not bit-reproduce source.2 loss/gradient")
        spectrum = m.core.stft_batch(x[2:3])
        bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
        output = nets[3](bands)
        base, _, _ = m.fit.reconstruction_loss((output[:, :2]+1)/2, spectrum, x[2:3], v[2:3], gs, 96, 44)
        (base*(1/6)).backward()
        expected = gradients(nets[1])+3*gradients(nets[3])
        actual = gradients(nets[2])
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-7)
        unchanged = all(old.dev.state_digest(net.state_dict()) == old.dev.state_digest(saved["model"]) and
            [module.training for module in net.modules()] == saved["modes"] for net in nets)
        result = {"schema": 1, "purpose": PURPOSE, "counter": 2500, "plan_sha256": acq.sha256(out / "plan.json"),
            "input_sha256": [m.pilot.wave_digest(wave) for wave in x], "metadata": batch["metadata"],
            "target_sha256": [m.pilot.wave_digest(wave) for wave in v],
            "source_model_digest": old.dev.state_digest(saved["model"]), "model_and_modes_unchanged": unchanged,
            "control_bit_exact": True, "gradient_delta_checked": True,
            "gradient_max_error": float((actual-expected).abs().max()), "gradient_delta_l2": float((actual-gradients(nets[1])).norm()),
            "instrumental_extra_contribution": candidate["instrumental_extra_contribution"],
            "auxiliary_active_count": candidate["auxiliary_active_count"], "candidate_losses": candidate,
            "model_updates": 0, "optimizer_constructed": False, "cuda_used": False,
            "scope": "One predeclared real TRAIN batch; contribution mechanism, not quality or Adam causality",
            "release_selection": "NONE", "deployment": False}
        if not unchanged or not d.finite_state(result) or torch.cuda.is_initialized():
            raise ValueError("Audit nonfinite/mutating/CUDA")
        acq.write_new_json(out / "audit.json", acq.seal(result))
    verify_audit()


def prepare(out):
    old.require_fresh(out)
    no_active_trainer()
    doc = draft_document()
    verify_audit()
    for path in (AUDIT_OUT / "plan.json", AUDIT_OUT / "audit.json"):
        doc["bindings_sha256"][str(path.resolve())] = acq.sha256(path)
    out.mkdir(parents=True)
    acq.write_new_json(out / "approval.json", acq.seal(doc))
    verified_approval(out / "approval.json")
    print("PROTECTION IMPORT SEALED; requires fresh real CPU/CUDA proofs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "verify_audit", "prepare", "verify"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.out or AUDIT_OUT)
    elif args.command == "verify_audit":
        verify_audit()
    elif args.command == "prepare":
        prepare(args.out or DEFAULT_OUT)
    else:
        verified_approval((args.out or DEFAULT_OUT) / "approval.json")
        print("PROTECTION IMPORT VERIFIED; NONRELEASE", flush=True)
