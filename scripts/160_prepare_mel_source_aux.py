"""Fresh equal-slot source-projection auxiliary authority; historical artifacts stay immutable."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("aux_source", Path(__file__).with_name("154_train_mel_weak_weight.py"))
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
old, inp, m, d, acq, ROOT = source.old, source.p.inp, source.m, source.d, source.acq, source.ROOT
ARMS = old.ARMS
PURPOSE = "NONRELEASE_MEL_SOURCE_PROJECTION_AUX"
ROLES = dict(zip(ARMS, ("uniform_control", "source_aux002")))
PROTOCOL = ROOT / "docs/mel_source_aux_protocol_20261003.json"
DEFAULT_OUT = ROOT / "results/mel_source_aux_import_20261003"
DEFAULT_APPROVAL = DEFAULT_OUT / "approval.json"
AUDIT_OUT = ROOT / "results/mel_source_aux_audit_20261003"
AUXILIARY = {"lambda": .02, "rms_floor": .0001, "minimum_gram_determinant": .001,
    "domains": ["musdb", "mir1k"], "role": "train", "scored_waveform_only": True,
    "formula": "cv^2+(ca-1)^2", "slot_weights": [1, 1, 1, 1, 1, 1]}


def check_protocol(p):
    exact = {"schema": 1, "purpose": PURPOSE, "exploratory_training_authorized": True,
        "formal_training_authorized": False, "deployment_authorized": False, "teacher": "kim_melband",
        "approved_pseudo_songs": 24, "origin_step": 1500, "additional_common_steps": 500,
        "absolute_limit": 2000, "checkpoint_every": 250, "source_arm": ARMS[0], "auxiliary": AUXILIARY,
        "internal_arm_keys": ROLES, "gpu_concurrency_authorized": True, "minimum_free_mib": 2300,
        "minimum_disk_gib": 12, "mechanism_step_limit": 3, "release_selection": "NONE",
        "independent_acceptance_ready": False}
    if any(p.get(k) != v or type(p.get(k)) is not type(v) for k, v in exact.items()):
        raise ValueError("Require fresh fixed500 equal-slot Mel auxiliary protocol")
    if (any(type(p["auxiliary"][k]) is not float for k in ("lambda", "rms_floor", "minimum_gram_determinant")) or
        any(type(v) is not int for v in p["auxiliary"]["slot_weights"]) or
        p["auxiliary"]["scored_waveform_only"] is not True):
        raise ValueError("Exact auxiliary parameter types required")


def check_approval(doc):
    check_protocol(doc["protocol"])
    if (doc.get("purpose") != PURPOSE or doc.get("deployment_authorized") is not False or
        doc.get("release_selection") != "NONE" or doc.get("arm_roles") != ROLES or
        doc.get("teacher") != "kim_melband" or
        doc.get("source_binding") != source.binding(Path(doc["source_approval"]))):
        raise ValueError("Changed source or release authority")


def verified_approval(path):
    doc = acq.read_sealed(path)
    check_approval(doc)
    for name, digest in doc["bindings_sha256"].items():
        if acq.sha256(name) != digest:
            raise ValueError(f"Auxiliary dependency changed: {name}")
    source.p.verified_approval(Path(doc["source_approval"]))
    origin = inp.verified_approval(Path(doc["origin_approval"]))
    if len(origin["pair_ids"]) != 24:
        raise ValueError("Fixed24 identity differs")
    return doc


def origin_state(doc):
    state = m.load_checked_checkpoint(Path(doc["origin_checkpoint"]), doc["protocol"]["origin_checkpoint_sha256"])
    if (state.get("purpose") != source.PURPOSE or state.get("binding") != doc["source_binding"] or
        state.get("arm_roles") != source.p.ROLES or state.get("teacher") != "kim_melband" or
        state.get("step") != 1500 or state.get("limit") != 1500 or state.get("smoke") is not False or
        state.get("deployment_authorized") is not False or state["sampler"]["cursor"] != 1500 or
        state["schedule"]["step"] != 1500 or state["schedule"]["last_validation"] != 1500 or
        state["schedule"]["stopped_at"] is not None or set(state["arms"]) != set(ARMS) or
        any(a["updates"] != 1500 for a in state["arms"].values()) or not d.finite_state(state)):
        raise ValueError("Require exact complete uniform1500 source state")
    return state


class SourceAuxStream:
    def __init__(self, path=None, preparation_doc=None):
        if (path is None) == (preparation_doc is None):
            raise ValueError("Explicit sealed stream OR preparation-only diagnostic")
        self.path = Path(path) if path is not None else None
        self.doc = verified_approval(self.path) if self.path else copy.deepcopy(preparation_doc)
        check_approval(self.doc)
        self.bound = acq.sha256(self.path) if self.path else "PREPARATION_ONLY_NO_TRAINING"
        self.origin_doc = inp.verified_approval(Path(self.doc["origin_approval"]))
        self.dataset = inp.ApprovedTeacherDataset(self.origin_doc, "kim_melband")
        self.seed, self.config = self.dataset.seed, self.dataset.config
        self.true = m.LockedTruePool(m.bulk.OLD_LOCK, self.config)
        self.source_state = origin_state(self.doc)
        sampler = self.source_state["sampler"]
        if (sampler["seed"] != self.seed or sampler["true_lock_sha256"] != self.true.bound or
            sampler["approval_sha256"] != acq.sha256(self.doc["source_approval"]) or sampler["teacher"] != "kim_melband"):
            raise ValueError("Source sampler/teacher roles changed")
        self.cursor, self.last_metadata = 1500, None

    def state_dict(self):
        return {"approval_sha256": self.bound, "true_lock_sha256": self.true.bound,
                "seed": self.seed, "cursor": self.cursor, "teacher": "kim_melband"}

    def load_state_dict(self, state):
        expected = self.state_dict() | {"cursor": state.get("cursor")}
        if (self.path is None or state != expected or type(state.get("cursor")) is not int or
            not 1500 <= state["cursor"] <= 2000 or acq.sha256(self.path) != self.bound):
            raise ValueError("Changed auxiliary stream authority/cursor")
        self.cursor, self.last_metadata = state["cursor"], None

    def next_batch(self):
        if self.cursor >= 2000:
            raise ValueError("Auxiliary sampler budget exhausted")
        xs, vs, metadata = [], [], []
        for domain in m.DOMAINS[:3]:
            item = self.true.crop(domain, self.seed, self.cursor)
            xs.append(item["x"]); vs.append(item["v"]); metadata.append(item["meta"])
        for index in range(3):
            recipe = m.data.crop_recipe(self.dataset.rows, self.config, self.seed, self.cursor*3+index)
            item = self.dataset.crop(recipe)
            xs.append(item["x"]); vs.append(item["v"])
            metadata.append(item["meta"] | {"domain": "pseudo", "role": "pseudo_label_train_candidate"})
        x, v = torch.stack(xs), torch.stack(vs)
        row = {"x": x, "targets": {arm: v.clone() for arm in ARMS}, "domains": m.DOMAINS,
               "cursor": self.cursor, "metadata": metadata}
        self.cursor += 1
        self.last_metadata = copy.deepcopy(metadata)
        return row


def draft_document():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    check_protocol(p)
    source_approval, source_run, exit_path = (ROOT / p[k] for k in ("source_approval", "origin_run", "origin_exit"))
    for path, digest in ((source_approval, p["source_approval_sha256"]), (exit_path, p["origin_exit_sha256"]),
                         (source_run / "completion.json", p["origin_completion_sha256"])):
        if acq.sha256(path) != digest:
            raise ValueError("Changed completed source evidence")
    source_doc = source.p.verified_approval(source_approval)
    exit_doc = json.loads(exit_path.read_text(encoding="utf-8-sig"))
    launch_path = Path(exit_doc["launch_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    if (exit_doc["exit_code"] != 0 or launch["probe_only"] or
        exit_doc["launcher_process_id"] != launch["worker"]["LauncherProcessId"] or Path(launch["out"]) != source_run):
        raise ValueError("Missing matched actual normal exit")
    completion = acq.read_sealed(source_run / "completion.json")
    receipt = acq.read_sealed(source_run / "checkpoint_1500.json")
    if (completion["binding"] != source.binding(source_approval) or completion["step"] != 1500 or
        completion["additional_steps"] != 500 or completion["final_checkpoint"] != receipt or
        receipt["sha256"] != p["origin_checkpoint_sha256"]):
        raise ValueError("Changed final source receipt")
    evidence_path = ROOT / "results/mel_weak_weight_monitor_20261003/completion_review.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (evidence["trainer_verify"]["exit_code"] != 0 or evidence["score_review"]["exit_code"] != 0 or
        evidence["hashes"]["results/mel_weak_weight_20261003/completion.json"] != p["origin_completion_sha256"]):
        raise ValueError("Missing existing source completion verification")
    original_approval = Path(source_doc["origin_approval"])
    original = inp.verified_approval(original_approval)
    files = [PROTOCOL, ROOT / p["authorization_record"], source_approval, original_approval,
        source_run / "completion.json", source_run / "checkpoint_1500.json", source_run / "NONRELEASE_weak_step_1500.pt",
        source_run / "development_step_1500.json", source_run / "selection_suite.json", source_run / "frozen_scores.json",
        exit_path, launch_path, evidence_path, ROOT / "reports/40_mel_loss_direction_and_source_aux_plan.md"]
    files += [ROOT / "scripts" / name for name in ("159_source_projection_auxiliary.py", "_test_source_projection_auxiliary.py",
        "160_prepare_mel_source_aux.py", "161_train_mel_source_aux.py", "162_start_mel_source_aux.ps1",
        "163_review_mel_source_aux.py", "_test_mel_source_aux.py", "141_process_exit_capture.ps1")]
    doc = {"schema": 1, "purpose": PURPOSE, "protocol": p, "origin_approval": str(original_approval),
        "source_approval": str(source_approval), "origin_checkpoint": str(source_run / "NONRELEASE_weak_step_1500.pt"),
        "source_binding": source.binding(source_approval), "source_protocol": original["source_protocol"],
        "arm_roles": ROLES, "teacher": "kim_melband", "release_selection": "NONE", "deployment_authorized": False,
        "bindings_sha256": {str(path.resolve()): acq.sha256(path) for path in files}}
    check_approval(doc)
    origin_state(doc)
    return doc


def prepare(out):
    old.require_fresh(out)
    doc = draft_document()
    audit_path = AUDIT_OUT / "audit.json"
    audit = acq.read_sealed(audit_path)
    if (audit["purpose"] != PURPOSE or audit["dependencies_sha256"] != doc["bindings_sha256"] or
        audit["model_unchanged"] is not True or audit["model_updates"] != 0 or
        audit["auxiliary_active_count"] < 1 or audit["finite_nonzero_auxiliary_gradient"] is not True):
        raise ValueError("Missing current real TRAIN preseal auxiliary audit")
    doc["audit_dependencies_sha256"] = copy.deepcopy(doc["bindings_sha256"])
    doc["bindings_sha256"][str(audit_path.resolve())] = acq.sha256(audit_path)
    out.mkdir(parents=True)
    acq.write_new_json(out / "approval.json", acq.seal(doc))
    print("MEL_AUX_IMPORT SEALED fixed500; equal Mel targets; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verified_approval(args.out / "approval.json")
        print("MEL_AUX_IMPORT VERIFIED")
    else:
        prepare(args.out)
