"""Read-only technical audit and CPU-only snapshots for the fixed teacher pairs.

No labels or eligibility are changed. Missing original exit status is retained as
unknown, even when independent waveform verification succeeds. No training.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("paired_prepare_quality", Path(__file__).with_name("138_audit_teacher_labels.py"))
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)
data, bulk, pilot = quality.data, quality.bulk, quality.pilot
matched_spec = importlib.util.spec_from_file_location("paired_prepare_control", Path(__file__).with_name("139_generate_paired_htdemucs.py"))
matched = importlib.util.module_from_spec(matched_spec)
matched_spec.loader.exec_module(matched)
DEFAULT_OUT = bulk.ROOT/"results/paired_distillation_prepare_20261002"


def prepare(out, control, verify_log):
    bulk.guard_output(out)
    if out.exists():
        raise ValueError("Fresh paired preparation output required")
    plan = matched.verify_plan(control)
    status = json.loads((control/"run_status.json").read_text(encoding="utf-8"))
    if (status["status"] != "complete" or status["verified_completed"] != 24 or status["error"] or
        "MATCHED_HT ALL VERIFIED; listening remains pending" not in verify_log.read_text(encoding="utf-8-sig")):
        raise ValueError("Completed 24-song control and independent verification required")
    protocol_path = data.PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data.validate_protocol(protocol)
    batch = Path(plan["batch_path"])
    rows = plan["records"]
    ids = [r["song_id"] for r in rows]
    if ids != data.choose_pairs(bulk.verify_plan(batch)["records"], 24, protocol["paired_comparison_planned"]["cohort_seed"]):
        raise ValueError("Paired IDs must not be selected by teacher quality")
    paths = [control/"plan.json", batch/"plan.json", verify_log, protocol_path, Path(__file__),
             bulk.ROOT/"scripts/143_paired_distillation_mechanics.py", bulk.OLD_LOCK,
             bulk.ROOT/"scripts/136_prepare_distillation_data.py", bulk.ROOT/"scripts/126_verify_training_baseline.py",
             bulk.ROOT/"scripts/119_model_selection_suite.py", bulk.ROOT/"models/student_bott2_mir1k_candidate.pt"]
    entries, records, pending = [], {"htdemucs": [], "kim_melband": []}, []
    expected = {"htdemucs": matched.binding(control, plan), "kim_melband": bulk.binding(batch, pilot.assets.ASSETS)}
    for row in rows:
        hints = {}
        for name, root in (("htdemucs", control), ("kim_melband", batch)):
            folder = root/row["song_id"]
            entry = pilot.acq.read_sealed(folder/"entry.json")
            if (entry["binding"] != expected[name] or entry["song_id"] != row["song_id"] or
                entry["pcm_sha256"] != row["source"]["pcm_sha256"] or entry["training_eligible"] is not False or
                entry["per_song_listening_review"] != "pending" or set(entry["files_sha256"]) != {"vocals.wav", "accompaniment.wav"}):
                raise ValueError("Changed or prematurely approved teacher entry")
            paths.append(folder/"entry.json")
            for filename, sha in entry["files_sha256"].items():
                if pilot.acq.sha256(folder/filename) != sha:
                    raise ValueError("Changed teacher pair label")
            result, flags = quality.metrics(folder/"vocals.wav", folder/"accompaniment.wav", row["source"]["samples"])
            entries.append({"song_id": row["song_id"], "teacher": name, "metrics": result,
                            "listening_hints": flags, "technical_passed": True, "training_eligible": False})
            hints[name] = flags
            records[name].append({"song_id": row["song_id"], "role": row["role"], "source": row["source"],
                                  "training_eligible": False, "label_files": {filename: {
                                      "path": str((folder/filename).resolve()), "sha256": sha}
                                      for filename, sha in entry["files_sha256"].items()}})
        pending.append({"song_id": row["song_id"], "track_id": row["track_id"], "source_path": row["source"]["path"],
                        "teacher_directories": {"htdemucs": str((control/row["song_id"]).resolve()),
                                                "kim_melband": str((batch/row["song_id"]).resolve())},
                        "same_input_pcm_sha256": row["source"]["pcm_sha256"], "listening_hints": hints,
                        "windows": quality.listening_windows(row["source"]["samples"]),
                        "review_status": "pending", "rights_status": "pending", "training_eligible": False})
        print(f"PAIRED_INPUT_PREPARE checked={len(pending)}/24", flush=True)
    exit_paths = sorted(control.glob("detached_exit_*.json"))
    exit_doc = json.loads(exit_paths[-1].read_text(encoding="utf-8")) if exit_paths else None
    if exit_paths:
        paths.append(exit_paths[-1])
    bindings = {str(path.resolve()): pilot.acq.sha256(path) for path in paths}
    out.mkdir(parents=True)
    snapshot_paths = {}
    for name in records:
        snapshot = {"schema": 1, "input": protocol["input"], "sampler_seed": protocol["paired_comparison_planned"]["cohort_seed"],
                    "records": records[name], "bindings_sha256": bindings, "training_authorized": False,
                    "purpose": "CPU-only paired mechanism checks; no approved training importer", "teacher": name}
        path = out/f"{name}_snapshot.json"
        pilot.acq.write_new_json(path, pilot.acq.seal(snapshot))
        snapshot_paths[name] = {"path": str(path.resolve()), "sha256": pilot.acq.sha256(path)}
    report = {"schema": 1, "pair_ids": ids, "records": entries, "snapshots": snapshot_paths,
              "bindings_sha256": bindings, "technical_pairs_passed": 24, "training_authorized": False,
              "human_reviewed_pairs": 0, "original_inference_exit_code": exit_doc.get("exit_code") if exit_doc else None,
              "original_inference_exit_code_proven": bool(exit_doc and type(exit_doc.get("exit_code")) is int),
              "completion_caveat": "Original null exit code is NOT zero. Independent script139 waveform verification passed separately; formal start gates are unresolved.",
              "formal_training_gate": "CLOSED: human listening/rights, independent truth roles, original numeric exit status/approved completion evidence unresolved"}
    pilot.acq.write_new_json(out/"paired_bundle.json", pilot.acq.seal(report))
    pilot.acq.write_new_json(out/"listening_pending.json", pilot.acq.seal({"schema": 1, "records": pending,
        "bundle_sha256": pilot.acq.sha256(out/"paired_bundle.json"), "reviewed": 0, "training_authorized": False}))
    print("PAIRED_INPUT_PREPARE PASS pairs=24; technical only, formal start CLOSED", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--control", type=Path, default=matched.DEFAULT_OUT)
    ap.add_argument("--verify-log", type=Path, default=matched.DEFAULT_OUT/"post_complete_verify_20261002.log")
    args = ap.parse_args()
    prepare(args.out, args.control, args.verify_log)


if __name__ == "__main__":
    main()
