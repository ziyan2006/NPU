"""Sealed full-song Mel-Band labels for eligible PRIVATE library songs.

Prepare isolates known validation/regression music BEFORE audio decoding,
fingerprints full eligible PCM, quarantines conflicting versions, and seals a
disk budget. Run resumes verified songs, preserves partial failures, uses an
exclusive worker lock, and never trains or replaces a student or SD image.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("teacher_bulk_pilot", Path(__file__).with_name("131_run_teacher_pilot.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
ROOT = pilot.ROOT
DEFAULT_OUT = ROOT / "results/teacher_library_melband_20261002"
PROTOCOL = ROOT / "docs/teacher_bulk_protocol_20261002.json"
OLD_LOCK = ROOT / "results/training_protocol_20261001/dataset_lock.json"
NEW_ASSIGNMENT = ROOT / "data/datasets/CambridgeMTK-new-candidates-20261001/assignment.json"
RECIPE_FILES = ("134_generate_teacher_library.py", "131_run_teacher_pilot.py", "130_acquire_teacher_pilot.py",
                "128_acquire_cambridge_candidates.py", "110_train_residual_ablation.py", "11_smoke_train.py",
                "13_ab_compare.py", "23_build_true_stem_cache.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def guard_output(out):
    root, target = (ROOT / "results").resolve(), out.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Batch outputs must be a child of ignored workspace results")


def composition_keys(name):
    # Conservative: versions/remixes of reserved songs are not fresh train songs.
    raw = re.sub(r"\.(?:mp3|mp4|wav|flac|ogg|m4a|aiff|aif)$", "", str(name), flags=re.I)
    clean = re.sub(r"\([^)]*\)|\[[^]]*\]", "", raw)
    clean = re.sub(r"\s+(?:[-–]\s*)?(?:original|extended|radio|club|vip)\s+(?:mix|edit|version)\s*$", "", clean, flags=re.I)
    return {pilot.acq.song_key(raw), pilot.acq.song_key(clean)} - {""}


def classify(paths, forbidden):
    banned = set().union(*(composition_keys(name) for name in forbidden)) if forbidden else set()
    eligible, excluded = [], []
    for path in paths:
        if "chills" in path.name.casefold() or composition_keys(path.name) & banned:
            excluded.append(path)
        else:
            eligible.append(path)
    return eligible, excluded


def group_sources(rows):
    """Identity collapse; different decoded versions of same title quarantine."""
    groups = []
    for row in rows:
        keys = composition_keys(Path(row["path"]).name)
        hits = [g for g in groups if keys & g["keys"] or row["pcm_sha256"] in g["pcm"]]
        group = {"keys": set(keys), "pcm": {row["pcm_sha256"]}, "rows": [row]}
        for hit in hits:
            group["keys"].update(hit["keys"])
            group["pcm"].update(hit["pcm"])
            group["rows"].extend(hit["rows"])
            groups.remove(hit)
        groups.append(group)
    accepted, quarantine = [], []
    for group in groups:
        ordered = sorted(group["rows"], key=lambda r: r["path"])
        if len(group["pcm"]) != 1:
            quarantine.append({"reason": "Same conservative composition key but different full decoded PCM", "sources": ordered})
        else:
            accepted.append({"source": ordered[0], "aliases": ordered, "composition_keys": sorted(group["keys"])})
    return sorted(accepted, key=lambda r: r["source"]["path"]), quarantine


def storage_estimate(seconds, settings):
    return math.ceil(seconds * pilot.SR * 2 * 4 * settings["saved_stems"] * settings["estimate_safety_factor"])


def check_storage(free, needed, reserve):
    if needed < 0 or reserve < 0 or free < needed + reserve:
        raise ValueError(f"Insufficient disk budget: free={free} needed={needed} reserve={reserve}")


def probe(path):
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels:format=duration", "-of", "json", str(path)], timeout=30)
    doc = json.loads(raw)
    if len(doc["streams"]) != 1:
        raise ValueError("Missing first audio stream")
    duration = float(doc["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid source duration; no guessed fallback")
    return {"duration_s": duration, "source_sample_rate": int(doc["streams"][0]["sample_rate"]),
            "source_channels": int(doc["streams"][0]["channels"])}


def decode(path):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:a:0",
        "-ar", str(pilot.SR), "-ac", "2", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"], timeout=600)
    if not raw or len(raw) % 8:
        raise ValueError("Empty/incomplete decoded stereo PCM")
    audio = torch.from_numpy(np.frombuffer(raw, dtype="<f4").reshape(-1, 2).T.copy())
    pilot.finite_stereo(audio)
    return audio


def media_versions():
    return {name: subprocess.check_output([name, "-version"], timeout=30).decode("utf-8", "replace").splitlines()[0]
            for name in ("ffmpeg", "ffprobe")}


def protected_inventory(data_lock, assignment):
    old = pilot.acq.read_sealed(data_lock)
    if pilot.acq.sha256(old["checkpoint"]) != old["frozen_sha256"]:
        raise ValueError("Frozen student changed")
    forbidden = old["known_regression_ids"] + old["blind_ids"] + [r["track_id"] for r in old["records"] if r["role"] != "train"]
    bindings = {str(data_lock.resolve()): pilot.acq.sha256(data_lock)}
    new = pilot.acq.read_sealed(assignment)
    bindings[str(assignment.resolve())] = pilot.acq.sha256(assignment)
    for row in new["records"]:
        if row["role"] != "train_candidate":
            forbidden.extend((row["track_id"], f"{row['artist']} - {row['title']}"))
    # All recorded historic library holdouts, not just latest report's list.
    for report in sorted((ROOT / "results").glob("*.json")):
        doc = json.loads(report.read_text(encoding="utf-8"))
        values = doc.get("holdout_tracks", []) if isinstance(doc, dict) else []
        if values:
            forbidden.extend(values)
            bindings[str(report.resolve())] = pilot.acq.sha256(report)
    return old, sorted(set(forbidden)), bindings


def prepare(out, protocol_path, data_lock, assignment, library):
    guard_output(out)
    if out.exists():
        raise ValueError("New batch directory required; no old labels or plans overwritten")
    settings = json.loads(protocol_path.read_text(encoding="utf-8"))
    teacher_path = ROOT / settings["teacher_protocol"]
    if pilot.acq.sha256(teacher_path) != settings["teacher_protocol_sha256"] or not settings["authorization"]["full_song_label_generation"]:
        raise ValueError("Batch teacher/authorization changed")
    old, forbidden, bindings = protected_inventory(data_lock, assignment)
    paths = sorted(p.resolve() for p in library.rglob("*") if p.is_file() and p.suffix.casefold() == ".mp3")
    if not paths:
        raise ValueError("Empty library")
    eligible, excluded_paths = classify(paths, forbidden)
    excluded = [{"path": str(p), "sha256": pilot.acq.sha256(p), "reason": "Reserved composition/version or legacy chills exclusion; audio NOT decoded"}
                for p in excluded_paths]
    banned_bytes = {r["sha256"] for r in excluded}
    inspected, quarantine = [], []
    for index, path in enumerate(eligible):
        digest = pilot.acq.sha256(path)
        if digest in banned_bytes:
            excluded.append({"path": str(path), "sha256": digest, "reason": "Byte-identical alias of reserved audio; audio NOT decoded"})
            continue
        try:
            info = probe(path)
            if info["duration_s"] > settings["maximum_song_seconds"]:
                raise ValueError("Longer than bounded full-song batch maximum; not cropped")
            wave = decode(path)
            if abs(wave.shape[-1]/pilot.SR - info["duration_s"]) > 1.:
                raise ValueError("Decoded duration differs from container by >1 second")
            inspected.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size, **info,
                              "samples": wave.shape[-1], "seconds": wave.shape[-1]/pilot.SR,
                              "pcm_sha256": pilot.wave_digest(wave)})
            del wave
        except (ValueError, subprocess.SubprocessError) as error:
            quarantine.append({"reason": str(error), "sources": [{"path": str(path), "sha256": digest}]})
        if (index + 1) % 10 == 0:
            print(f"TEACHER_BULK_PREPARE inspected={index+1}/{len(eligible)} reserved={len(excluded)} quarantined={len(quarantine)}", flush=True)
    grouped, conflicts = group_sources(inspected)
    quarantine.extend(conflicts)
    if not grouped:
        raise ValueError("No eligible full songs")
    rows = [{"song_id": f"song_{index:04d}", "track_id": Path(group["source"]["path"]).name,
             "role": settings["output_role"], **group} for index, group in enumerate(grouped)]
    seconds = sum(r["source"]["seconds"] for r in rows)
    needed = storage_estimate(seconds, settings["storage"])
    free = shutil.disk_usage(ROOT).free
    check_storage(free, needed, settings["storage"]["minimum_free_after_bytes"])
    bindings.update({str(protocol_path.resolve()): pilot.acq.sha256(protocol_path), str(teacher_path.resolve()): pilot.acq.sha256(teacher_path),
                     str(Path(old["checkpoint"]).resolve()): old["frozen_sha256"]})
    plan = {"schema": 1, "created_utc": now(), "protocol": settings, "protocol_path": str(protocol_path.resolve()),
            "teacher_protocol_path": str(teacher_path.resolve()), "teacher_protocol": json.loads(teacher_path.read_text(encoding="utf-8")),
            "library": str(library.resolve()), "library_file_count": len(paths), "records": rows, "excluded": excluded,
            "quarantined": quarantine, "protected_ids": forbidden, "input_bindings_sha256": bindings,
            "media_versions": media_versions(), "code_sha256": {str((ROOT/"scripts"/n).resolve()): pilot.acq.sha256(ROOT/"scripts"/n) for n in RECIPE_FILES},
            "budget": {"music_seconds": seconds, "estimated_label_bytes_with_margin": needed, "free_bytes_at_prepare": free,
                       "reserve_bytes": settings["storage"]["minimum_free_after_bytes"]},
            "same_pcm_alias_files_collapsed": sum(len(r["aliases"])-1 for r in rows), "student_training_performed": False,
            "new_blind_data_claimed": False, "per_song_listening_review": "pending"}
    out.mkdir(parents=True)
    pilot.acq.write_new_json(out / "plan.json", pilot.acq.seal(plan))
    print(f"TEACHER_BULK_PLAN SEALED songs={len(rows)} music_hours={seconds/3600:.2f} labels_gib={needed/2**30:.2f} "
          f"excluded_files={len(excluded)} quarantine_groups={len(quarantine)} duplicates={plan['same_pcm_alias_files_collapsed']}", flush=True)


def verify_plan(out):
    guard_output(out)
    plan = pilot.acq.read_sealed(out / "plan.json")
    for group in ("input_bindings_sha256", "code_sha256"):
        for path, expected in plan[group].items():
            if pilot.acq.sha256(path) != expected:
                raise ValueError(f"Sealed batch input/recipe changed: {path}")
    if plan["media_versions"] != media_versions():
        raise ValueError("Audio decoder version changed; new recipe/plan required")
    banned = set().union(*(composition_keys(name) for name in plan["protected_ids"]))
    seen_pcm, seen_keys, seen_ids = set(), set(), set()
    for row in plan["records"]:
        source = row["source"]
        keys = set(row["composition_keys"])
        if row["role"] != "pseudo_label_train_candidate" or not re.fullmatch(r"song_\d{4}", row["song_id"]):
            raise ValueError("Invalid batch role/ID")
        if keys & (banned | seen_keys) or source["pcm_sha256"] in seen_pcm or row["song_id"] in seen_ids:
            raise ValueError("Batch composition/PCM identity overlap or reserved contamination")
        seen_pcm.add(source["pcm_sha256"])
        seen_keys.update(keys)
        seen_ids.add(row["song_id"])
        for alias in row["aliases"]:
            path = Path(alias["path"])
            if alias["pcm_sha256"] != source["pcm_sha256"] or pilot.acq.sha256(path) != alias["sha256"]:
                raise ValueError("Batch source alias changed")
    return plan


def binding(out, assets):
    return {"plan_sha256": pilot.acq.sha256(out/"plan.json"), "asset_receipt_sha256": pilot.acq.sha256(assets/"asset_receipt.json"),
            "runtime": pilot.runtime_binding(assets)}


def verify_song(folder, row, expected_binding, mix=None):
    entry = pilot.acq.read_sealed(folder/"entry.json")
    source = row["source"]
    if entry["binding"] != expected_binding or entry["song_id"] != row["song_id"] or entry["pcm_sha256"] != source["pcm_sha256"]:
        raise ValueError("Song result belongs to changed batch inputs/runtime")
    if set(entry["files_sha256"]) != {"vocals.wav", "accompaniment.wav"}:
        raise ValueError("Incomplete song output")
    for name, digest in entry["files_sha256"].items():
        if pilot.acq.sha256(folder/name) != digest:
            raise ValueError("Saved batch stem changed")
    mix = decode(Path(source["path"])) if mix is None else mix
    if mix.shape[-1] != source["samples"] or pilot.wave_digest(mix) != source["pcm_sha256"]:
        raise ValueError("Full decoded input differs from sealed fingerprint")
    v, a = pilot.read_wave(folder/"vocals.wav"), pilot.read_wave(folder/"accompaniment.wav")
    expected_a, _ = pilot.residual_pair(mix, v)
    if not torch.equal(a, expected_a):
        raise ValueError("Saved backing is not exact sealed input minus vocal")
    return entry


@contextmanager
def worker_lock(out):
    # Native advisory byte lock releases automatically on worker exit/crash.
    # File remains for reuse, unlike a stale O_EXCL PID-file lock.
    with (out/"worker.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError("Another batch worker already owns this output") from error
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def state(out, status, total, completed, active=None, error=None):
    doc = {"schema": 1, "updated_utc": now(), "pid": os.getpid(), "status": status, "total": total,
           "verified_completed": completed, "active_song_id": active, "error": error,
           "student_training": False, "student_promotion": False}
    tmp = out / f"run_status.{os.getpid()}.tmp"
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, out/"run_status.json")


def run(out, assets, limit=0):
    guard_output(out)
    with worker_lock(out):
        total, completed, active = 0, 0, None
        try:
            plan = verify_plan(out)
            total = len(plan["records"])
            state(out, "checking_resume", total, 0)
            pilot.assets.verify(assets, Path(plan["teacher_protocol_path"]))
            bound = binding(out, assets)
            pending = []
            for row in plan["records"]:
                folder = out/row["song_id"]
                if (folder/"entry.json").exists():
                    verify_song(folder, row, bound)
                    completed += 1
                elif folder.exists():
                    raise ValueError(f"Partial song preserved, refusing overwrite: {folder}")
                else:
                    pending.append(row)
            if not pending:
                state(out, "complete", total, completed)
                print(f"TEACHER_BULK COMPLETE verified={completed}/{total}", flush=True)
                return
            settings = plan["protocol"]
            check_storage(shutil.disk_usage(out).free, storage_estimate(sum(r["source"]["seconds"] for r in pending), settings["storage"]),
                          settings["storage"]["minimum_free_after_bytes"])
            if not torch.cuda.is_available():
                raise ValueError("CUDA required; no silent device/precision fallback")
            load_doc = {"protocol_path": plan["teacher_protocol_path"], "protocol": plan["teacher_protocol"]}
            state(out, "loading_teacher", total, completed)
            model, config = pilot.load_mel(assets, load_doc)
            model = model.to("cuda").eval()
            if config["inference"]["chunk_size"] != settings["chunk_samples"] or config["inference"]["num_overlap"] != settings["overlap"]:
                raise ValueError("Batch differs from teacher's validated inference configuration")
            # Recheck saved pilot proof and its runtime/asset binding before bulk.
            parity = pilot.acq.read_sealed(pilot.DEFAULT_OUT/"kim_melband_roformer/author_parity.json")
            pilot.verify_cohort(pilot.DEFAULT_OUT, verify_sources=False)
            expected = pilot.entry_binding(pilot.DEFAULT_OUT, "kim_melband_roformer", assets, bound["runtime"])
            if not parity["passed"] or parity["binding"] != expected:
                raise ValueError("Pilot author's inference parity proof is missing/stale")
            written = 0
            for row in pending:
                if limit and written >= limit:
                    state(out, "paused_at_requested_limit", total, completed)
                    return
                active, source = row["song_id"], row["source"]
                state(out, "running", total, completed, active)
                check_storage(shutil.disk_usage(out).free, storage_estimate(source["seconds"], settings["storage"]),
                              settings["storage"]["minimum_free_after_bytes"])
                mix = decode(Path(source["path"]))
                if pilot.wave_digest(mix) != source["pcm_sha256"] or mix.shape[-1] != source["samples"]:
                    raise ValueError("Input PCM changed between prepare and inference")
                torch.manual_seed(20261002)
                torch.cuda.manual_seed_all(20261002)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                vocal = pilot.mel_demix(model, mix, settings["chunk_samples"], settings["overlap"], "cuda")
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                accompaniment, consistency = pilot.residual_pair(mix, vocal)
                folder = out/active
                folder.mkdir()
                files = {"vocals.wav": pilot.write_wave_new(folder/"vocals.wav", vocal),
                         "accompaniment.wav": pilot.write_wave_new(folder/"accompaniment.wav", accompaniment)}
                # Readback before publishing entry.json as completion marker.
                if not torch.equal(pilot.read_wave(folder/"vocals.wav"), vocal) or not torch.equal(pilot.read_wave(folder/"accompaniment.wav"), accompaniment):
                    raise ValueError("FLOAT stem serialization changed samples")
                entry = {"schema": 1, "binding": bound, "song_id": active, "track_id": row["track_id"], "role": row["role"],
                         "source_path": source["path"], "source_sha256": source["sha256"], "pcm_sha256": source["pcm_sha256"],
                         "samples": source["samples"], "seconds": source["seconds"], "files_sha256": files, "elapsed_s": elapsed,
                         "rtf": elapsed/source["seconds"], "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "mixture_consistency_max_error": consistency, "vocal_peak": float(vocal.abs().max()),
                         "technical_waveform_checks_passed": True, "per_song_listening_review": "pending", "training_eligible": False}
                pilot.acq.write_new_json(folder/"entry.json", pilot.acq.seal(entry))
                verify_song(folder, row, bound, mix)
                completed += 1
                written += 1
                state(out, "running", total, completed)
                print(f"TEACHER_BULK saved={completed}/{total} song={active} seconds={source['seconds']:.2f} "
                      f"elapsed={elapsed:.2f} rtf={entry['rtf']:.4f} peak_mib={entry['peak_cuda_allocated_bytes']/2**20:.1f}", flush=True)
                del mix, vocal, accompaniment
            verify_plan(out)
            pilot.assets.verify(assets, Path(plan["teacher_protocol_path"]))
            state(out, "complete", total, completed)
            print(f"TEACHER_BULK COMPLETE verified={completed}/{total}; no student/SD changes", flush=True)
        except BaseException as error:
            state(out, "failed", total, completed, active, f"{type(error).__name__}: {error}")
            raise


def verify(out, assets):
    plan = verify_plan(out)
    pilot.assets.verify(assets, Path(plan["teacher_protocol_path"]))
    bound = binding(out, assets)
    for index, row in enumerate(plan["records"]):
        verify_song(out/row["song_id"], row, bound)
        if (index+1) % 10 == 0:
            print(f"TEACHER_BULK_VERIFY {index+1}/{len(plan['records'])}", flush=True)
    print(f"TEACHER_BULK ALL VERIFIED songs={len(plan['records'])}; per-song listening remains pending", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "run", "verify", "status"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--data-lock", type=Path, default=OLD_LOCK)
    ap.add_argument("--assignment", type=Path, default=NEW_ASSIGNMENT)
    ap.add_argument("--library", type=Path, default=Path(os.environ.get("STEM_AUDIO_LIBRARY", "D:/DJ_Music_Library")))
    ap.add_argument("--assets", type=Path, default=pilot.assets.ASSETS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("limit must be nonnegative")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.action == "prepare":
        prepare(args.out, args.protocol, args.data_lock, args.assignment, args.library)
    elif args.action == "run":
        run(args.out, args.assets, args.limit)
    elif args.action == "verify":
        verify(args.out, args.assets)
    else:
        guard_output(args.out)
        print((args.out/"run_status.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
