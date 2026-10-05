"""Separately approved, fixed-24 NONRELEASE importer; never edits old snapshots.

Training authority is read from the new exploration record, not inferred from an
audit purpose. Reuses immutable decoding/cropping helpers only after approval.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import json
from pathlib import Path

import soundfile as sf

import importlib.util
spec = importlib.util.spec_from_file_location("exploration_mechanics", Path(__file__).with_name("143_paired_distillation_mechanics.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
data, acq, ROOT = m.data, m.pilot.acq, m.bulk.ROOT
PURPOSE = "NONRELEASE_PAIRED_EXPLORATION"
PROTOCOL = ROOT / "docs/teacher_exploratory_protocol_20261003.json"
DEFAULT_OUT = ROOT / "results/paired_exploratory_import_20261003_r2"
DEFAULT_APPROVAL = DEFAULT_OUT / "approval.json"


def check_protocol(doc):
    if (doc.get("schema") != 1 or doc.get("purpose") != PURPOSE or
        doc.get("exploratory_training_authorized") is not True or
        doc.get("formal_training_authorized") is not False or doc.get("promotion_authorized") is not False or
        doc.get("approved_pseudo_songs") != 24 or type(doc.get("maximum_exploratory_steps")) is not int or
        doc.get("gpu_concurrency_authorized") is not True or
        doc["maximum_exploratory_steps"] != 1000 or doc.get("checkpoint_every") != 250 or
        doc.get("independent_acceptance_ready") is not False or doc.get("release_selection") != "NONE"):
        raise ValueError("Require fixed-24, bounded exploration authority; never release approval")


def check_approval(doc):
    check_protocol(doc["protocol"])
    if (doc.get("schema") != 1 or doc.get("purpose") != PURPOSE or
        doc.get("exploratory_training_authorized") is not True or doc.get("deployment_authorized") is not False or
        doc.get("original_htdemucs_exit_code") is not None or
        doc.get("accepted_completion_basis") != "independent_full_waveform_verify_for_exploration_only" or
        len(doc.get("pair_ids", [])) != 24 or len(set(doc["pair_ids"])) != 24 or
        set(doc.get("snapshots", {})) != {"htdemucs", "kim_melband"} or
        [r["song_id"] for r in doc.get("approved_records", [])] != doc["pair_ids"] or
        any(r.get("exploratory_eligible") is not True or r.get("deployment_eligible") is not False for r in doc["approved_records"])):
        raise ValueError("Missing/broadened per-song exploration approval")


def verified_approval(path):
    doc = acq.read_sealed(path)
    check_approval(doc)
    for name, expected in doc["bindings_sha256"].items():
        if acq.sha256(name) != expected:
            raise ValueError(f"Exploration dependency changed: {name}")
    for snapshot in doc["snapshots"].values():
        if acq.sha256(snapshot["path"]) != snapshot["sha256"]:
            raise ValueError("Historical snapshot changed")
    return doc


class ApprovedTeacherDataset(data.TeacherWaveformDataset):
    """Own authorized constructor; old cpu_audit constructor is NOT a train route."""
    def __init__(self, approval, teacher):
        check_approval(approval)
        if teacher not in approval["snapshots"]:
            raise ValueError("Unapproved teacher")
        snapshot = approval["snapshots"][teacher]
        self.path = Path(snapshot["path"])
        if acq.sha256(self.path) != snapshot["sha256"]:
            raise ValueError("Snapshot mismatch")
        self.doc = acq.read_sealed(self.path)
        self.config, self.seed = self.doc["input"], self.doc["sampler_seed"]
        data.geometry(self.config)
        self.cursor, self.cache_songs = 0, 1
        self.cache, self.signatures = OrderedDict(), {}
        self.bound = snapshot["sha256"]
        self.rows = copy.deepcopy(self.doc["records"])
        if ([r["song_id"] for r in self.rows] != approval["pair_ids"] or
            any(r["role"] != "pseudo_label_train_candidate" or r["training_eligible"] is not False for r in self.rows)):
            raise ValueError("Historical role/order changed or silently promoted")
        self.by_id = {r["song_id"]: r for r in self.rows}
        for row in self.rows:
            if set(row["label_files"]) != {"vocals.wav", "accompaniment.wav"}:
                raise ValueError("Missing waveform targets")
            for info in (row["source"], *row["label_files"].values()):
                if acq.sha256(info["path"]) != info["sha256"]:
                    raise ValueError("Source/target hash changed")
                self.signatures[info["path"]] = data.stat_signature(info["path"])
            for info in row["label_files"].values():
                f = sf.info(info["path"])
                if (f.frames != row["source"]["samples"] or f.samplerate != 44100 or f.channels != 2 or f.subtype != "FLOAT"):
                    raise ValueError("Label geometry changed")
        self.mix_identity = data.digest_json([(r["song_id"], r["source"]["pcm_sha256"], r["source"]["samples"]) for r in self.rows])

    def crop(self, recipe):
        result = super().crop(recipe)
        result["meta"] = result["meta"] | {"historical_training_eligible": False,
            "exploratory_eligible": True, "deployment_eligible": False, "purpose": PURPOSE}
        return result


class ApprovedPairStream(m.SharedBatchStream):
    def __init__(self, approval_path):
        self.path = Path(approval_path)
        self.doc = verified_approval(self.path)
        self.bound = acq.sha256(self.path)
        self.datasets = [ApprovedTeacherDataset(self.doc, name) for name in ("htdemucs", "kim_melband")]
        a, b = self.datasets
        if a.mix_identity != b.mix_identity or a.config != b.config or a.seed != b.seed:
            raise ValueError("Targets must share source, gain recipe and sampler")
        # Decode/hash identical MP3 PCM only once per draw, still bounded to one
        # song. Each arm independently checks its own label signatures/windows.
        b.cache = a.cache
        self.seed, self.config, self.cursor = a.seed, a.config, 0
        self.true = m.LockedTruePool(m.bulk.OLD_LOCK, self.config)

    def state_dict(self):
        return {"approval_sha256": self.bound, "true_lock_sha256": self.true.bound, "seed": self.seed, "cursor": self.cursor}

    def load_state_dict(self, state):
        if (state.get("approval_sha256") != self.bound or state.get("true_lock_sha256") != self.true.bound or
            state.get("seed") != self.seed or type(state.get("cursor")) is not int or
            not 0 <= state["cursor"] <= self.doc["protocol"]["maximum_exploratory_steps"] or acq.sha256(self.path) != self.bound):
            raise ValueError("Changed approval, role lock or input cursor")
        self.cursor = state["cursor"]


def prepare(out):
    m.bulk.guard_output(out)
    if out.exists():
        raise ValueError("Fresh approval directory required; preserve old pending files")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    check_protocol(protocol)
    source_protocol = ROOT / protocol["source_protocol"]
    bundle_path = ROOT / protocol["source_bundle"]
    if (acq.sha256(source_protocol) != protocol["source_protocol_sha256"] or
        acq.sha256(bundle_path) != protocol["source_bundle_sha256"]):
        raise ValueError("Original protocol/bundle mismatch")
    old = json.loads(source_protocol.read_text(encoding="utf-8"))
    m.validate_protocol(old)
    bundle = acq.read_sealed(bundle_path)
    if bundle["technical_pairs_passed"] != 24 or bundle["original_inference_exit_code"] is not None:
        raise ValueError("Unexpected completion evidence; no fabricated historical exit code")
    bindings = copy.deepcopy(bundle["bindings_sha256"])
    paths = [PROTOCOL, bundle_path, ROOT / protocol["authorization_record"], ROOT / protocol["listening_and_rights_record"],
             Path(__file__), ROOT / "scripts/150_train_paired_exploration.py", ROOT / "scripts/151_start_paired_exploration.ps1",
             ROOT / "scripts/147_paired_device_mechanics.py", ROOT / "scripts/146_evaluate_paired_development.py",
             ROOT / "scripts/141_process_exit_capture.ps1",
             ROOT / "results/paired_development_eval_20261002/frozen_scores.json",
             ROOT / "results/paired_development_eval_20261002/development_evaluation.json",
             ROOT / "results/layout_control_microbatch_20261001/selection_suite.json"]
    bindings.update({str(p.resolve()): acq.sha256(p) for p in paths})
    for name, expected in bindings.items():
        if acq.sha256(name) != expected:
            raise ValueError(f"Old preparation/source changed: {name}")
    control_log = ROOT / "results/teacher_pairs_htdemucs_20261002/post_complete_verify_20261002.log"
    if "MATCHED_HT ALL VERIFIED; listening remains pending" not in control_log.read_text(encoding="utf-8-sig"):
        raise ValueError("Independent full waveform verification absent")
    snapshots = copy.deepcopy(bundle["snapshots"])
    records = []
    for song_id in bundle["pair_ids"]:
        records.append({"song_id": song_id, "exploratory_eligible": True, "deployment_eligible": False,
            "listening_basis": "user_confirmed_24_preselected_listening_pack; not exhaustive whole-song review",
            "rights_basis": "user_project_training_statement_in_report27; no upload/release extension"})
    approval = {"schema": 1, "purpose": PURPOSE, "protocol": protocol, "source_protocol": old,
        "pair_ids": bundle["pair_ids"], "snapshots": snapshots, "approved_records": records,
        "bindings_sha256": bindings, "exploratory_training_authorized": True, "deployment_authorized": False,
        "original_htdemucs_exit_code": None, "accepted_completion_basis": "independent_full_waveform_verify_for_exploration_only"}
    check_approval(approval)
    for name in snapshots:
        dataset = ApprovedTeacherDataset(approval, name)
        print(f"EXPLORATORY_IMPORT verified teacher={name} songs={len(dataset.rows)}", flush=True)
    out.mkdir(parents=True)
    acq.write_new_json(out / "approval.json", acq.seal(approval))
    print("EXPLORATORY_IMPORT SEALED; fixed 24; old snapshots untouched; no optimizer", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verified_approval(args.out / "approval.json")
        print("EXPLORATORY_IMPORT VERIFIED", flush=True)
    else:
        prepare(args.out)


if __name__ == "__main__":
    main()
