"""Matched full-FLOAT-input HTDemucs controls for the preselected 24 songs.

Only pseudo-label generation. No student fitting, listening approval or release.
Existing completed/partial outputs are verified/preserved, never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import time

import torch

spec = importlib.util.spec_from_file_location("matched_teacher_inputs", Path(__file__).with_name("136_prepare_distillation_data.py"))
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
bulk, pilot = data.bulk, data.pilot
DEFAULT_OUT = bulk.ROOT/"results/teacher_pairs_htdemucs_20261002"
AUDIT = bulk.ROOT/"results/teacher_label_audit_20261002/technical_audit.json"
SELECTION = data.DEFAULT_OUT/"snapshot.json"


def require_fixed_selection(plan, selection, protocol):
    pair = protocol["paired_comparison_planned"]
    ids = data.choose_pairs(plan["records"], pair["pseudo_source_songs"], pair["cohort_seed"])
    if selection["paired_teacher_24_song_ids"] != ids or selection["training_authorized"] is not False:
        raise ValueError("Pre-scoring 24-song selection changed or prematurely approved")
    records = [r for r in plan["records"] if r["song_id"] in ids]
    if len(records) != 24 or any(r["role"] != "pseudo_label_train_candidate" for r in records):
        raise ValueError("Matched control must retain exactly 24 sealed train candidates")
    return records


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if pilot.acq.sha256(path) != expected:
            raise ValueError(f"Matched-control source/protocol/code/weight changed: {path}")


def prepare(batch, out, audit_path, selection_path, protocol_path):
    bulk.guard_output(out)
    if out.exists():
        raise ValueError("Fresh matched-control directory required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data.validate_protocol(protocol)
    plan = bulk.verify_plan(batch)
    selection, audit = pilot.acq.read_sealed(selection_path), pilot.acq.read_sealed(audit_path)
    verify_bindings(selection["bindings_sha256"])
    verify_bindings(audit["bindings_sha256"])
    if (pilot.acq.sha256(batch/"plan.json") != protocol["batch_plan_sha256"] or
        audit["songs"] != 275 or audit["technical_passed"] != 275 or audit["quality_approved"] is not False):
        raise ValueError("Complete technical audit of the bound batch required; no inferred human approval")
    records = require_fixed_selection(plan, selection, protocol)
    cohort = pilot.acq.read_sealed(pilot.DEFAULT_OUT/"cohort.json")
    teacher = cohort["protocol"]["baseline"]
    if teacher["name"] != "htdemucs" or teacher["shifts"] != 1 or teacher["overlap"] != .25:
        raise ValueError("Baseline differs from validated pilot recipe")
    paths = [batch/"plan.json", audit_path, selection_path, protocol_path, pilot.DEFAULT_OUT/"cohort.json",
             Path(cohort["baseline_checkpoint"]), Path(cohort["frozen_checkpoint"]), bulk.OLD_LOCK,
             bulk.ROOT/"docs/model_training_protocol_v1.json", Path(__file__),
             bulk.ROOT/"scripts/136_prepare_distillation_data.py", bulk.ROOT/"scripts/common.py"]
    paths.extend(bulk.ROOT/"scripts"/name for name in bulk.RECIPE_FILES)
    paths.extend(batch/r["song_id"]/"entry.json" for r in records)
    bindings = {str(p.resolve()): pilot.acq.sha256(p) for p in paths}
    if bindings[str(Path(cohort["baseline_checkpoint"]).resolve())] != cohort["baseline_sha256"]:
        raise ValueError("Local HTDemucs weight differs from pilot")
    seconds = sum(r["source"]["seconds"] for r in records)
    needed = bulk.storage_estimate(seconds, plan["protocol"]["storage"])
    reserve = plan["protocol"]["storage"]["minimum_free_after_bytes"]
    bulk.check_storage(shutil.disk_usage(out.parent).free, needed, reserve)
    out.mkdir(parents=True)
    document = {"schema": 1, "batch_path": str(batch.resolve()), "records": records, "bindings_sha256": bindings,
                "runtime": pilot.runtime_binding(pilot.assets.ASSETS), "teacher": teacher,
                "weight_path": cohort["baseline_checkpoint"], "weight_sha256": cohort["baseline_sha256"],
                "audio_seconds": seconds, "expected_bytes_with_margin": needed, "minimum_free_after_bytes": reserve,
                "normalization": "Existing script11 teacher_vocals Demucs normalization/inversion; no saved-label clipping/gain",
                "precision": "CUDA float32, no autocast/automatic recipe fallback", "seed_per_song": "20261001 + numeric song ID",
                "training_authorized": False, "per_song_listening_review": "pending", "paired_inputs": "Original full decoded FLOAT32 PCM"}
    pilot.acq.write_new_json(out/"plan.json", pilot.acq.seal(document))
    print(f"MATCHED_HT_PLAN songs=24 seconds={seconds:.3f} bytes_with_margin={needed}; no inference/training", flush=True)


def verify_plan(out):
    bulk.guard_output(out)
    plan = pilot.acq.read_sealed(out/"plan.json")
    verify_bindings(plan["bindings_sha256"])
    if plan["runtime"] != pilot.runtime_binding(pilot.assets.ASSETS) or plan["training_authorized"] is not False:
        raise ValueError("Matched teacher runtime/authority changed")
    return plan


def binding(out, plan):
    return {"plan_sha256": pilot.acq.sha256(out/"plan.json"), "teacher": "htdemucs", "runtime": plan["runtime"]}


def verify_song(folder, row, bound, mix=None):
    entry = pilot.acq.read_sealed(folder/"entry.json")
    if (entry["binding"] != bound or entry["song_id"] != row["song_id"] or entry["pcm_sha256"] != row["source"]["pcm_sha256"] or
        entry["training_eligible"] is not False or entry["per_song_listening_review"] != "pending" or
        set(entry["files_sha256"]) != {"vocals.wav", "accompaniment.wav"}):
        raise ValueError("Changed/incomplete matched teacher entry")
    for name, expected in entry["files_sha256"].items():
        if pilot.acq.sha256(folder/name) != expected:
            raise ValueError("Matched teacher stem changed")
    mix = bulk.decode(Path(row["source"]["path"])) if mix is None else mix
    if mix.shape[-1] != row["source"]["samples"] or pilot.wave_digest(mix) != row["source"]["pcm_sha256"]:
        raise ValueError("HTDemucs input differs from Mel-Band full FLOAT32 input")
    v, a = pilot.read_wave(folder/"vocals.wav"), pilot.read_wave(folder/"accompaniment.wav")
    expected, _ = pilot.residual_pair(mix, v)
    if not torch.equal(a, expected):
        raise ValueError("Matched backing is not exact same input minus vocal")
    return entry


def pending_rows(out, plan, bound):
    pending, completed = [], 0
    for row in plan["records"]:
        folder = out/row["song_id"]
        if (folder/"entry.json").exists():
            verify_song(folder, row, bound)
            completed += 1
        elif folder.exists():
            raise ValueError(f"Partial matched output preserved: {folder}")
        else:
            pending.append(row)
    return pending, completed


def run(out, limit=0):
    bulk.guard_output(out)
    with bulk.worker_lock(out):
        completed, active = 0, None
        try:
            plan = verify_plan(out)
            bound = binding(out, plan)
            bulk.state(out, "checking_resume", 24, 0)
            pending, completed = pending_rows(out, plan, bound)
            if not pending:
                bulk.state(out, "complete", 24, completed)
                print("MATCHED_HT COMPLETE verified=24/24; no training", flush=True)
                return
            seconds = sum(r["source"]["seconds"] for r in pending)
            needed = math_storage(seconds)
            bulk.check_storage(shutil.disk_usage(out).free, needed, plan["minimum_free_after_bytes"])
            if not torch.cuda.is_available():
                raise ValueError("CUDA required, no CPU/precision fallback")
            free, _ = torch.cuda.mem_get_info()
            if free < 3*2**30:
                raise ValueError("Need at least 3 GiB currently free GPU memory; do not kill other processes")
            from common import load_demucs
            bulk.state(out, "loading_teacher", 24, completed)
            model = load_demucs("htdemucs").to("cuda").eval()
            written = 0
            for row in pending:
                if limit and written >= limit:
                    bulk.state(out, "paused_at_requested_limit", 24, completed)
                    return
                active = row["song_id"]
                bulk.state(out, "running", 24, completed, active)
                mix = bulk.decode(Path(row["source"]["path"]))
                if mix.shape[-1] != row["source"]["samples"] or pilot.wave_digest(mix) != row["source"]["pcm_sha256"]:
                    raise ValueError("Changed source PCM")
                seed = plan["teacher"]["seed"]+int(active.split("_")[-1])
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                vocal = pilot.core.t11.teacher_vocals(model, mix, "cuda")
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                accompaniment, consistency = pilot.residual_pair(mix, vocal)
                folder = out/active
                folder.mkdir()
                files = {"vocals.wav": pilot.write_wave_new(folder/"vocals.wav", vocal),
                         "accompaniment.wav": pilot.write_wave_new(folder/"accompaniment.wav", accompaniment)}
                if not torch.equal(pilot.read_wave(folder/"vocals.wav"), vocal) or not torch.equal(pilot.read_wave(folder/"accompaniment.wav"), accompaniment):
                    raise ValueError("FLOAT readback changed samples")
                entry = {"schema": 1, "binding": bound, "song_id": active, "track_id": row["track_id"],
                         "pcm_sha256": row["source"]["pcm_sha256"], "samples": row["source"]["samples"],
                         "files_sha256": files, "elapsed_s": elapsed, "rtf": elapsed/row["source"]["seconds"],
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "mixture_consistency_max_error": consistency, "technical_waveform_checks_passed": True,
                         "training_eligible": False, "per_song_listening_review": "pending"}
                pilot.acq.write_new_json(folder/"entry.json", pilot.acq.seal(entry))
                verify_song(folder, row, bound, mix)
                completed += 1
                written += 1
                bulk.state(out, "running", 24, completed)
                print(f"MATCHED_HT saved={completed}/24 song={active} rtf={entry['rtf']:.4f}", flush=True)
                del mix, vocal, accompaniment
            verify_plan(out)
            bulk.state(out, "complete", 24, completed)
            print("MATCHED_HT COMPLETE verified=24/24; no student/SD changes", flush=True)
        except BaseException as error:
            bulk.state(out, "failed", 24, completed, active, f"{type(error).__name__}: {error}")
            raise


def math_storage(seconds):
    # Same two FLOAT stereo labels and 10% margin as sealed bulk plan.
    return bulk.storage_estimate(seconds, {"saved_stems": 2, "estimate_safety_factor": 1.1})


def verify(out):
    plan = verify_plan(out)
    bound = binding(out, plan)
    for index, row in enumerate(plan["records"]):
        verify_song(out/row["song_id"], row, bound)
        print(f"MATCHED_HT_VERIFY {index+1}/24", flush=True)
    print("MATCHED_HT ALL VERIFIED; listening remains pending", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "run", "verify", "status"))
    ap.add_argument("--batch", type=Path, default=bulk.DEFAULT_OUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--audit", type=Path, default=AUDIT)
    ap.add_argument("--selection", type=Path, default=SELECTION)
    ap.add_argument("--protocol", type=Path, default=data.PROTOCOL)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("limit must be nonnegative")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.action == "prepare":
        prepare(args.batch, args.out, args.audit, args.selection, args.protocol)
    elif args.action == "run":
        run(args.out, args.limit)
    elif args.action == "verify":
        verify(args.out)
    else:
        bulk.guard_output(args.out)
        print((args.out/"run_status.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
