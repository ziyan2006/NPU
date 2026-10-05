"""Explicitly authorize a same-Mel, loss-only fork; never change old approvals."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch

import importlib.util
spec = importlib.util.spec_from_file_location("weak_origin", Path(__file__).with_name("150_train_paired_exploration.py"))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
inp, m, d, acq, ROOT = old.inp, old.m, old.d, old.acq, old.ROOT
ARMS = old.ARMS
PURPOSE = "NONRELEASE_MEL_WEAK_WEIGHT_ABLATION"
ROLES = dict(zip(ARMS, ("uniform_control", "weak12_weight2")))
PROTOCOL = ROOT / "docs/mel_weak_weight_protocol_20261003.json"
DEFAULT_OUT = ROOT / "results/mel_weak_weight_import_20261003"
DEFAULT_APPROVAL = DEFAULT_OUT / "approval.json"


def check_protocol(p):
    exact = {"schema": 1, "purpose": PURPOSE, "exploratory_training_authorized": True,
        "formal_training_authorized": False, "deployment_authorized": False, "teacher": "kim_melband",
        "approved_pseudo_songs": 24, "origin_step": 1000, "additional_common_steps": 500,
        "absolute_limit": 1500, "checkpoint_every": 250, "weights": {"uniform_control": 1, "weak12_weight2": 2},
        "internal_arm_keys": ROLES, "gpu_concurrency_authorized": True, "minimum_free_mib": 2300,
        "mechanism_step_limit": 3, "release_selection": "NONE", "independent_acceptance_ready": False}
    if any(p.get(k) != v or type(p.get(k)) is not type(v) for k, v in exact.items()):
        raise ValueError("Require explicitly approved fixed-Mel loss-only500-step fork")
    if any(type(v) is not int for v in p["weights"].values()):
        raise ValueError("Exact integer loss weights required, not booleans")


def check_approval(doc):
    check_protocol(doc["protocol"])
    if (doc.get("purpose") != PURPOSE or doc.get("deployment_authorized") is not False or
        doc.get("release_selection") != "NONE" or doc.get("origin_binding") != old.binding(Path(doc["origin_approval"]))):
        raise ValueError("Changed origin or release authority")


def verified_approval(path):
    doc = acq.read_sealed(path)
    check_approval(doc)
    for name, digest in doc["bindings_sha256"].items():
        if acq.sha256(name) != digest:
            raise ValueError(f"Fork dependency changed: {name}")
    origin = inp.verified_approval(Path(doc["origin_approval"]))
    if len(origin["pair_ids"]) != 24:
        raise ValueError("Fixed24 source identity differs")
    return doc


def origin_state(doc):
    state = m.load_checked_checkpoint(Path(doc["origin_checkpoint"]), doc["protocol"]["origin_checkpoint_sha256"])
    if (state.get("purpose") != old.PURPOSE or state.get("binding") != doc["origin_binding"] or
        state.get("step") != 1000 or state.get("limit") != 1000 or state.get("deployment_authorized") is not False or
        state["sampler"]["cursor"] != 1000 or state["schedule"]["step"] != 1000 or
        state["schedule"]["last_validation"] != 1000 or set(state["arms"]) != set(ARMS) or
        any(a["updates"] != 1000 for a in state["arms"].values()) or not d.finite_state(state)):
        raise ValueError("Require exact complete original1000 paired state")
    return state


class MelForkStream:
    def __init__(self, path):
        self.path = Path(path)
        self.doc = verified_approval(self.path)
        self.bound = acq.sha256(self.path)
        self.origin_doc = inp.verified_approval(Path(self.doc["origin_approval"]))
        self.dataset = inp.ApprovedTeacherDataset(self.origin_doc, "kim_melband")
        self.seed, self.config = self.dataset.seed, self.dataset.config
        self.true = m.LockedTruePool(m.bulk.OLD_LOCK, self.config)
        self.source_state = origin_state(self.doc)
        source_sampler = self.source_state["sampler"]
        if (source_sampler["seed"] != self.seed or source_sampler["true_lock_sha256"] != self.true.bound or
            source_sampler["approval_sha256"] != acq.sha256(self.doc["origin_approval"])):
            raise ValueError("Source sampler/roles changed")
        self.cursor = 1000
        self.last_metadata = None

    def state_dict(self):
        return {"approval_sha256": self.bound, "true_lock_sha256": self.true.bound,
                "seed": self.seed, "cursor": self.cursor, "teacher": "kim_melband"}

    def load_state_dict(self, state):
        expected = self.state_dict() | {"cursor": state.get("cursor")}
        if (state != expected or type(state.get("cursor")) is not int or
            not 1000 <= state["cursor"] <= 1500 or acq.sha256(self.path) != self.bound):
            raise ValueError("Changed fork approval, teacher, roles, seed or cursor")
        self.cursor = state["cursor"]
        self.last_metadata = None

    def next_batch(self):
        if self.cursor >= 1500:
            raise ValueError("Fork sampler budget exhausted")
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


def prepare(out):
    old.require_fresh(out)
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    check_protocol(p)
    source_approval, source_run, exit_path = (ROOT / p[k] for k in ("origin_approval", "origin_run", "origin_exit"))
    if acq.sha256(source_approval) != p["origin_approval_sha256"] or acq.sha256(exit_path) != p["origin_exit_sha256"]:
        raise ValueError("Changed original approval/actual exit evidence")
    exit_doc = json.loads(exit_path.read_text(encoding="utf-8-sig"))
    launch_path = Path(exit_doc["launch_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    if (exit_doc["exit_code"] != 0 or launch["probe_only"] or
        exit_doc["launcher_process_id"] != launch["worker"]["LauncherProcessId"] or Path(launch["out"]) != source_run):
        raise ValueError("Missing matched normal original exit")
    old.verify_run(source_run, source_approval)
    origin = inp.verified_approval(source_approval)
    files = [PROTOCOL, ROOT / p["authorization_record"], source_approval, source_run / "completion.json",
        source_run / "checkpoint_1000.json", source_run / "NONRELEASE_pair_step_1000.pt", exit_path, launch_path,
        ROOT / "reports/35_exploration_1000_completion_and_next_trial.md"]
    files += [ROOT / "scripts" / name for name in ("153_prepare_mel_weak_weight.py", "154_train_mel_weak_weight.py",
        "155_start_mel_weak_weight.ps1", "156_review_mel_weak_weight.py", "_test_mel_weak_weight.py",
        "141_process_exit_capture.ps1")]
    doc = {"schema": 1, "purpose": PURPOSE, "protocol": p, "origin_approval": str(source_approval),
        "origin_checkpoint": str(source_run / "NONRELEASE_pair_step_1000.pt"), "origin_binding": old.binding(source_approval),
        "source_protocol": origin["source_protocol"], "arm_roles": ROLES, "teacher": "kim_melband",
        "release_selection": "NONE", "deployment_authorized": False,
        "bindings_sha256": {str(path.resolve()): acq.sha256(path) for path in files}}
    check_approval(doc)
    origin_state(doc)
    out.mkdir(parents=True)
    acq.write_new_json(out / "approval.json", acq.seal(doc))
    print("MEL_WEAK_IMPORT SEALED;500 extra steps; both targets Mel; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verified_approval(args.out / "approval.json")
        print("MEL_WEAK_IMPORT VERIFIED")
    else:
        prepare(args.out)
