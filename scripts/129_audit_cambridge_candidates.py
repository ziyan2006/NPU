"""Technical source QC only: explicit stems, shared gain and private role audit.

No network, separation model, quality ranking, training or acceptance promotion.
Common sample zero is a provider-alignment assumption, not a measured guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location("cambridge_acquisition_qc", Path(__file__).with_name("128_acquire_cambridge_candidates.py"))
acq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acq)
ROOT = acq.ROOT


def partition_stems(basenames, mapping):
    actual = set(basenames)
    vocal = mapping["vocal_basenames"]
    if len(actual) != len(basenames) or not vocal or len(vocal) != len(set(vocal)) or not set(vocal) < actual:
        raise ValueError("Need distinct existing vocal and accompaniment raw WAVs")
    named_vocals = {n for n in actual if re.search(r"vox|vocal|lead.?double|backing", n, re.I)}
    if not named_vocals <= set(vocal):
        raise ValueError("Named vocal/effect/double must not silently enter accompaniment")
    if any(not re.fullmatch(r"[^/\\:]+\.wav", name, re.I) for name in vocal):
        raise ValueError("Mapping must contain WAV basenames only")
    # Filename checks cannot prove that unnamed loops/SFX contain no vocals.
    return sorted(vocal), sorted(actual - set(vocal))


def stereo(raw):
    if raw.ndim != 2 or raw.shape[1] not in (1, 2) or not np.isfinite(raw).all():
        raise ValueError("Only finite mono or stereo raw audio is supported")
    return np.repeat(raw, 2, axis=1) if raw.shape[1] == 1 else raw


def sum_reference(named_audio, vocal_names):
    if not named_audio or set(vocal_names) >= set(named_audio) or not set(vocal_names) <= set(named_audio):
        raise ValueError("Need disjoint nonempty vocal/accompaniment groups")
    length = max(len(raw) for raw in named_audio.values())
    v, a = np.zeros((length, 2), np.float64), np.zeros((length, 2), np.float64)
    for name, raw in sorted(named_audio.items()):
        target = v if name in vocal_names else a
        target[:len(raw)] += stereo(raw)
    mix = a + v
    peak = float(np.max(np.abs(mix)))
    if peak <= 1e-12 or not np.any(v) or not np.any(a):
        raise ValueError("Silent mix/vocal/accompaniment cannot be a vocal positive")
    common_gain = min(1., .95 / peak)
    # Do not crop at the shortest stem or independently normalize components.
    return (mix * common_gain).astype(np.float32), (v * common_gain).astype(np.float32), common_gain


def audio_stats(raw, sample_rate):
    frame_power = np.mean(np.asarray(raw, np.float64) ** 2, axis=1)
    block_samples = sample_rate
    active = 0
    for start in range(0, len(raw), block_samples):
        power = frame_power[start:start + block_samples]
        if power.mean() >= 1e-6:
            active += len(power)
    return {"peak": float(np.abs(raw).max()),
            "rms_dbfs": float(10 * np.log10(max(float(frame_power.mean()), 1e-20))),
            "activity_seconds_at_minus60_dbfs_1s_windows": active / sample_rate,
            "near_full_scale_fraction": float(np.mean(np.abs(raw) >= .9999))}


def validate_quarantine(report):
    if (report.get("schema") != 1 or report.get("formal_training_ready") is not False or
            report.get("blind_status") != "incomplete" or report.get("model_scoring_performed") is not False or
            report.get("training_performed") is not False):
        raise ValueError("Source QC cannot certify training readiness or blind acceptance")
    ids = [r["track_id"] for r in report["records"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Duplicate/empty quarantine song inventory")
    for row in report["records"]:
        if (row["role"] not in acq.ROLES or row["training_eligible"] is not False or
                row["acceptance_ready"] is not False or row["source_listening_review"] != "pending"):
            raise ValueError("Quarantine roles cannot be silently promoted")


def audit_song(folder, row, receipt, mapping, old_hashes):
    if receipt["archive_sha256"] != mapping["archive_sha256"]:
        raise ValueError("Stem mapping is not bound to this downloaded archive")
    audio_paths = sorted(folder.glob("*.wav"))
    vocal, accompaniment = partition_stems([p.name for p in audio_paths], mapping)
    raw_audio, files, flags = {}, {}, []
    for path in audio_paths:
        expected = receipt["extracted_files"].get(f"{row['track_id']}/{path.name}")
        actual_hash = acq.sha256(path)
        if expected != actual_hash:
            raise ValueError("Original stem changed or absent from receipt")
        if actual_hash in old_hashes:
            raise ValueError("Exact original stem hash overlaps locked old data")
        info = sf.info(path)
        if info.samplerate != 44100 or info.channels not in (1, 2) or info.frames < 1 or info.subtype != "PCM_24":
            raise ValueError("Unexpected source rate/channels/subtype/length")
        raw, sr = sf.read(path, dtype="float32", always_2d=True)
        stereo(raw)  # Finiteness and channel check before statistics.
        stats = audio_stats(raw, sr)
        files[path.name] = {"sha256": actual_hash, "bytes": path.stat().st_size,
                           "sample_rate": sr, "channels": info.channels, "frames": info.frames,
                           "subtype": info.subtype, **stats}
        if stats["near_full_scale_fraction"] > 0:
            flags.append(f"Near-full-scale samples in {path.name}: not proof of clipping; listen/review")
        raw_audio[path.name] = raw
    mix, v, gain = sum_reference(raw_audio, set(vocal))
    if len({f["frames"] for f in files.values()}) > 1:
        flags.append("Unequal raw stem lengths; zero-extend short tails, never trim all to shortest")
    if any(re.search(r"loop|sfx|noise|synthsand", n, re.I) for n in accompaniment):
        flags.append("Unnamed vocal content inside loops/SFX remains unreviewed")
    flags.extend(["Source-stem bleed, timing, effects and label purity require listening review",
                  "Common sample zero assumed from provider raw export; no automatic cross-correlation realignment",
                  "Synthetic raw sum, not official mastered mix; unity mono duplication is not pan reconstruction",
                  "Bundled education-only use does not establish commercial training permission"])
    return {"track_id": row["track_id"], "artist": row["artist"], "title": row["title"],
            "style": row["style"], "role": row["role"],
            "same_artist_in_known_ids": row["same_artist_in_known_ids"],
            "archive_sha256": receipt["archive_sha256"], "bytes": receipt["bytes"],
            "unpacked_bytes": receipt["unpacked_bytes"], "raw_files": files,
            "vocal_basenames": vocal, "accompaniment_basenames": accompaniment,
            "sample_rate": 44100, "samples": len(mix), "duration_s": len(mix) / 44100,
            "common_gain": gain, "mix_stats": audio_stats(mix, 44100), "vocal_stats": audio_stats(v, 44100),
            "synthetic_mix_float32_sha256": hashlib.sha256(mix.astype("<f4").tobytes()).hexdigest(),
            "synthetic_vocal_float32_sha256": hashlib.sha256(v.astype("<f4").tobytes()).hexdigest(),
            "technical_qc": "pass", "source_listening_review": "pending",
            "training_eligible": False, "acceptance_ready": False, "flags": flags}


def audit(out, mapping_path, report_path):
    if report_path.exists():
        raise ValueError("Use a new QC report; previous evidence is never overwritten")
    allowed = ((ROOT / "data").resolve(), (ROOT / "results").resolve())
    if not any(report_path.resolve().is_relative_to(p) for p in allowed):
        raise ValueError("Detailed audio/source QC must remain in private ignored data/results")
    assignment = acq.verify_assignment(out)
    mapping_doc = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping_doc.get("schema") != 1:
        raise ValueError("Unexpected source map schema")
    mappings = {m["track_id"]: m for m in mapping_doc["records"]}
    if len(mappings) != len(mapping_doc["records"]) or set(mappings) != {r["track_id"] for r in assignment["records"]}:
        raise ValueError("Source map must cover each assigned composition exactly once")
    rows, receipts = [], {}
    for row in assignment["records"]:
        receipt_path = out / f"{row['track_id']}_receipt.json"
        receipt = acq.read_sealed(receipt_path)
        if (receipt["assignment_sha256"] != acq.sha256(out / "assignment.json") or
                receipt["track_id"] != row["track_id"] or receipt["role"] != row["role"] or
                acq.sha256(out / f"{row['track_id']}.zip") != receipt["archive_sha256"]):
            raise ValueError("Receipt does not match sealed role/archive")
        record = audit_song(out / row["track_id"], row, receipt, mappings[row["track_id"]],
                            set(assignment["old_audio_sha256"]))
        rows.append(record)
        receipts[str(receipt_path.resolve())] = acq.sha256(receipt_path)
        print(f"SOURCE_QC {row['track_id']} wav={len(record['raw_files'])} vocal={len(record['vocal_basenames'])} "
              f"seconds={record['duration_s']:.3f} technical=PASS listening=PENDING", flush=True)
    report = {"schema": 1, "assignment_path": str((out / "assignment.json").resolve()),
              "assignment_sha256": acq.sha256(out / "assignment.json"),
              "mapping_path": str(mapping_path.resolve()), "mapping_sha256": acq.sha256(mapping_path),
              "dependency_sha256": {str(Path(__file__).resolve()): acq.sha256(__file__),
                                    str(Path(acq.__file__).resolve()): acq.sha256(acq.__file__)},
              "receipts_sha256": receipts, "records": rows,
              "download_bytes": sum(r["bytes"] for r in rows),
              "unpacked_bytes": sum(r["unpacked_bytes"] for r in rows),
              "raw_wav_files": sum(len(r["raw_files"]) for r in rows),
              "unique_song_seconds": sum(r["duration_s"] for r in rows),
              "candidate_role_counts": {role: sum(r["role"] == role for r in rows) for role in sorted(acq.ROLES)},
              "blind_status": "incomplete", "blind_target_songs": 10, "blind_artist_aim": 5,
              "formal_training_ready": False, "model_scoring_performed": False, "training_performed": False,
              "status_reason": "Only one new-song acceptance candidate; source listening/rights review incomplete; all three artists already seen; excerpts are not full songs"}
    validate_quarantine(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    acq.write_new_json(report_path, acq.seal(report))
    print(f"SOURCE_QC COMPLETE songs={len(rows)} blind=incomplete formal_training_ready=False", flush=True)


def verify(report_path):
    report = acq.read_sealed(report_path)
    validate_quarantine(report)
    for path_key, sha_key in (("assignment_path", "assignment_sha256"), ("mapping_path", "mapping_sha256")):
        if acq.sha256(report[path_key]) != report[sha_key]:
            raise ValueError("QC input changed")
    assignment = acq.verify_assignment(Path(report["assignment_path"]).parent)
    assigned = {r["track_id"]: r for r in assignment["records"]}
    if len(assigned) != len(report["records"]) or set(assigned) != {r["track_id"] for r in report["records"]}:
        raise ValueError("QC song inventory changed")
    for bindings in (report["receipts_sha256"], report["dependency_sha256"]):
        for path, expected in bindings.items():
            if acq.sha256(path) != expected:
                raise ValueError(f"QC evidence/dependency changed: {path}")
    out = Path(report["assignment_path"]).parent
    for row in report["records"]:
        if row["role"] != assigned[row["track_id"]]["role"] or row["training_eligible"] or row["acceptance_ready"]:
            raise ValueError("Quarantine candidate cannot be silently promoted")
        if acq.sha256(out / f"{row['track_id']}.zip") != row["archive_sha256"]:
            raise ValueError("Archive changed")
        for name, info in row["raw_files"].items():
            if acq.sha256(out / row["track_id"] / name) != info["sha256"]:
                raise ValueError("Raw original audio changed")
    if report["formal_training_ready"] or report["blind_status"] != "incomplete":
        raise ValueError("Technical QC alone cannot authorize formal training or certify blind acceptance")
    print("SOURCE_QC VERIFIED; quarantine remains closed to training", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=acq.DEFAULT_OUT)
    ap.add_argument("--mapping", type=Path, default=ROOT / "docs/cambridge_stem_map_20261001.json")
    ap.add_argument("--report", type=Path, default=ROOT / "results/cambridge_new_sources_20261001/source_qc.json")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args.report)
    else:
        audit(acq.guard_output(args.out), args.mapping, args.report)


if __name__ == "__main__":
    main()
