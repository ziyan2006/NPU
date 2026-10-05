"""Fresh .02-vs-.2 source auxiliary fork importer. Never expands old authorities."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import torch

spec = importlib.util.spec_from_file_location("strength_diagnostic_source", Path(__file__).with_name("164_diagnose_source_aux_strength.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
t, p, m, d, acq, ROOT = s.t, s.p, s.m, s.d, s.acq, s.ROOT
old, inp, ARMS = t.old, p.inp, t.ARMS
PURPOSE = "NONRELEASE_MEL_SOURCE_AUX_STRENGTH_FORK"
ROLES = dict(zip(ARMS, ("source_aux002_control", "source_aux020")))
LAMBDAS = dict(zip(ARMS, (.02, .2)))
PROTOCOL = ROOT / "docs/mel_source_strength_protocol_20261003.json"
DEFAULT_OUT = ROOT / "results/mel_source_strength_import_20261003"
DEFAULT_APPROVAL = DEFAULT_OUT / "approval.json"
REQUIRED = ("166_train_mel_source_strength.py", "167_start_mel_source_strength.ps1",
            "168_review_mel_source_strength.py", "_test_mel_source_strength.py")


def check_protocol(doc):
    exact = {"schema": 1, "purpose": PURPOSE, "exploratory_training_authorized": True,
        "formal_training_authorized": False, "deployment_authorized": False, "teacher": "kim_melband",
        "approved_pseudo_songs": 24, "origin_step": 2000, "additional_common_steps": 500,
        "absolute_limit": 2500, "checkpoint_every": 250, "source_arm": ARMS[1], "source_stopped_at": 2000,
        "tranche_stopping": "new_common_fixed500_budget_preserve_legacy_stop_evidence",
        "internal_arm_keys": ROLES, "arm_lambdas": LAMBDAS, "gpu_concurrency_authorized": True,
        "minimum_free_mib": 2300, "minimum_disk_gib": 12, "mechanism_step_limit": 3,
        "release_selection": "NONE", "independent_acceptance_ready": False,
        "slot_weights": [1]*6, "auxiliary_formula": "cv^2+(ca-1)^2",
        "rms_floor": .0001, "minimum_gram_determinant": .001,
        "auxiliary_domains": ["musdb", "mir1k"], "auxiliary_role": "train"}
    if any(doc.get(k) != v or type(doc.get(k)) is not type(v) for k, v in exact.items()):
        raise ValueError("Fresh fixed500 loss-strength-only authority required")
    if (any(type(x) is not float for x in doc["arm_lambdas"].values()) or
        any(type(x) is not int for x in doc["slot_weights"])):
        raise ValueError("Exact strength/slot types required")


def check_approval(doc):
    check_protocol(doc["protocol"])
    if (doc.get("purpose") != PURPOSE or doc.get("deployment_authorized") is not False or
        doc.get("release_selection") != "NONE" or doc.get("arm_roles") != ROLES or
        doc.get("teacher") != "kim_melband" or
        doc.get("source_binding") != t.binding(Path(doc["source_approval"]))):
        raise ValueError("Changed source or release authority")


def verified_approval(path):
    doc = acq.read_sealed(path)
    check_approval(doc)
    for filename, digest in doc["bindings_sha256"].items():
        if acq.sha256(filename) != digest:
            raise ValueError(f"Strength dependency changed: {filename}")
    p.verified_approval(Path(doc["source_approval"]))
    inp.verified_approval(Path(doc["origin_approval"]))
    return doc


def origin_state(doc):
    source_doc, state, _ = s.checked_source()
    if (Path(doc["source_approval"]) != p.DEFAULT_APPROVAL or
        Path(doc["origin_approval"]) != Path(source_doc["origin_approval"]) or
        acq.sha256(doc["origin_checkpoint"]) != doc["protocol"]["origin_checkpoint_sha256"] or
        Path(doc["origin_checkpoint"]) != s.SOURCE / "NONRELEASE_aux_step_2000.pt"):
        raise ValueError("Exact verified complete candidate2000 source required")
    return state


class StrengthStream:
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
            raise ValueError("Changed input identity/seed/teacher")
        self.cursor, self.last_metadata = 2000, None

    def state_dict(self):
        return {"approval_sha256": self.bound, "true_lock_sha256": self.true.bound,
                "seed": self.seed, "cursor": self.cursor, "teacher": "kim_melband"}

    def load_state_dict(self, state):
        if (self.path is None or state != self.state_dict() | {"cursor": state.get("cursor")} or
            type(state.get("cursor")) is not int or not 2000 <= state["cursor"] <= 2500 or
            acq.sha256(self.path) != self.bound):
            raise ValueError("Changed strength stream authority/cursor")
        self.cursor, self.last_metadata = state["cursor"], None

    def next_batch(self):
        if self.cursor >= 2500:
            raise ValueError("Fresh strength sampler500 budget exhausted")
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
    source_doc, _, source_bindings = s.checked_source()
    s.verify(s.DEFAULT_OUT)
    diagnostic = acq.read_sealed(s.DEFAULT_OUT / "diagnostic.json")
    if (acq.sha256(s.DEFAULT_OUT / "diagnostic.json") != protocol["diagnostic_sha256"] or
        diagnostic["model_updates"] != 0 or diagnostic["model_and_modes_unchanged"] is not True):
        raise ValueError("Verified TRAIN-only strength evidence required")
    scripts = [Path(__file__), PROTOCOL, ROOT / "reports/46_source_aux_strength_result_and_next_trial.md",
               s.DEFAULT_OUT / "plan.json", s.DEFAULT_OUT / "diagnostic.json",
               ROOT / "reports/47_source_strength_stopping_authorization.md"]
    scripts += [Path(__file__).with_name(name) for name in REQUIRED]
    # Fail closed while the fresh trainer/reviewer/launcher/tests are unfinished.
    bindings = source_bindings | {str(path): acq.sha256(path) for path in scripts}
    doc = {"schema": 1, "purpose": PURPOSE, "protocol": protocol, "source_protocol": source_doc["source_protocol"],
        "source_approval": str(p.DEFAULT_APPROVAL), "source_binding": t.binding(p.DEFAULT_APPROVAL),
        "origin_approval": source_doc["origin_approval"], "origin_checkpoint": str(s.SOURCE / "NONRELEASE_aux_step_2000.pt"),
        "arm_roles": ROLES, "teacher": "kim_melband", "bindings_sha256": bindings,
        "deployment_authorized": False, "release_selection": "NONE"}
    check_approval(doc)
    origin_state(doc)
    return doc


def prepare(out):
    old.require_fresh(out)
    s.no_active_trainer()
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '(134_generate_teacher_library|139_generate_paired_htdemucs|166_train_mel_source_strength)[.]py[\" ]+(run|train|smoke)' } | Select-Object -ExpandProperty ProcessId"
    active = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    if active.stdout.strip():
        raise ValueError("Active teacher/strength worker: postpone new sealing")
    doc = draft_document()
    out.mkdir(parents=True)
    acq.write_new_json(out / "approval.json", acq.seal(doc))
    verified_approval(out / "approval.json")
    print("SOURCE_STRENGTH IMPORT SEALED; training still requires real CPU/CUDA proofs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out)
    else:
        verified_approval(args.out / "approval.json")
        print("SOURCE_STRENGTH IMPORT VERIFIED; NONRELEASE", flush=True)
