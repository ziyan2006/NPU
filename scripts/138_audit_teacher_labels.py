"""Read-only streaming technical audit; never approve pseudo-labels for training.

Run only after script134's complete post-export verification. FLOAT overshoot,
estimated vocal energy and transients are listening hints, not separation scores.
The original sealed entries and waveforms are never changed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location("label_quality_inputs", Path(__file__).with_name("136_prepare_distillation_data.py"))
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
bulk, pilot = data.bulk, data.pilot
DEFAULT_OUT = bulk.ROOT/"results/teacher_label_audit_20261002"


class Statistics:
    def __init__(self):
        self.samples = 0
        self.total = np.zeros(2, dtype=np.float64)
        self.square = np.zeros(2, dtype=np.float64)
        self.peak, self.over_one, self.max_jump = 0., 0, 0.
        self.last = None

    def add(self, wave):
        if wave.ndim != 2 or wave.shape[1] != 2 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Nonfinite, empty or non-stereo waveform")
        value = wave.astype(np.float64)
        self.samples += len(value)
        self.total += value.sum(axis=0)
        self.square += (value*value).sum(axis=0)
        self.peak = max(self.peak, float(np.abs(value).max()))
        self.over_one += int((np.abs(value) > 1).sum())
        if self.last is not None:
            self.max_jump = max(self.max_jump, float(np.abs(value[0]-self.last).max()))
        if len(value) > 1:
            self.max_jump = max(self.max_jump, float(np.abs(np.diff(value, axis=0)).max()))
        self.last = value[-1]

    def report(self):
        if self.samples < 1:
            raise ValueError("Empty waveform statistics")
        return {"samples": self.samples, "rms": math.sqrt(float(self.square.sum())/(2*self.samples)),
                "channel_dc": (self.total/self.samples).tolist(), "peak": self.peak,
                "fraction_abs_above_one": self.over_one/(2*self.samples), "max_adjacent_jump": self.max_jump}


def metrics(vocal_path, backing_path, samples):
    stats = {name: Statistics() for name in ("vocals", "backing", "reconstructed_mix")}
    active_windows, quiet_vocal_windows = 0, 0
    for path in (vocal_path, backing_path):
        info = sf.info(path)
        if info.frames != samples or info.samplerate != pilot.SR or info.channels != 2 or info.subtype != "FLOAT":
            raise ValueError("Label length/rate/channels/FLOAT subtype differs from sealed input")
    with sf.SoundFile(vocal_path) as vs, sf.SoundFile(backing_path) as bs:
        while True:
            v = vs.read(pilot.SR, dtype="float32", always_2d=True)
            a = bs.read(pilot.SR, dtype="float32", always_2d=True)
            if v.shape != a.shape:
                raise ValueError("Misaligned stem windows")
            if not len(v):
                break
            # This context is explicitly a FLOAT stem sum, not a new source or truth.
            x = v.astype(np.float64)+a.astype(np.float64)
            for name, wave in (("vocals", v), ("backing", a), ("reconstructed_mix", x)):
                stats[name].add(wave)
            if float(np.mean(x*x)) > 1e-6:
                active_windows += 1
                quiet_vocal_windows += int(float(np.mean(v.astype(np.float64)**2)) < 1e-10)
    result = {name: value.report() for name, value in stats.items()}
    if any(value["samples"] != samples for value in result.values()):
        raise ValueError("Incomplete technical audit")
    result.update(active_one_second_windows=active_windows, quiet_vocal_active_windows=quiet_vocal_windows)
    flags = []
    if active_windows and quiet_vocal_windows/active_windows > .5:
        flags.append("Mostly quiet estimated vocal on active mix: may be instrumental; listen")
    for name in ("vocals", "backing"):
        value = result[name]
        if value["fraction_abs_above_one"] > .001:
            flags.append(f"{name} FLOAT overshoot: playback needs common gain; not proof of clipping")
        if max(abs(v) for v in value["channel_dc"]) > .01:
            flags.append(f"{name} DC offset needs listening/inspection")
        if value["rms"] > 2*max(result["reconstructed_mix"]["rms"], 1e-12):
            flags.append(f"{name} energy exceeds mix: inspect cancellations/label artifacts")
    return result, flags


def listening_windows(samples, seconds=18):
    length = min(samples, round(seconds*pilot.SR))
    return [{"start_sample": round(max(0, min(samples-length, samples*f-length/2))), "samples": length}
            for f in (.1, .5, .9)]


def audit(batch, out, verify_log, protocol_path):
    bulk.guard_output(out)
    if out.exists():
        raise ValueError("Fresh audit output required; existing evidence is preserved")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data.validate_protocol(protocol)
    status = json.loads((batch/"run_status.json").read_text(encoding="utf-8"))
    plan = bulk.verify_plan(batch)
    count = len(plan["records"])
    if (count != 275 or status["status"] != "complete" or status["verified_completed"] != count or status["error"] or
        pilot.acq.sha256(batch/"plan.json") != protocol["batch_plan_sha256"] or
        f"TEACHER_BULK ALL VERIFIED songs={count}; per-song listening remains pending" not in verify_log.read_text(encoding="utf-8-sig")):
        raise ValueError("Complete post-export verification evidence is required")
    bound = bulk.binding(batch, pilot.assets.ASSETS)
    paired = data.choose_pairs(plan["records"], 24, protocol["paired_comparison_planned"]["cohort_seed"])
    records, pending, bindings = [], [], {
        str((batch/"plan.json").resolve()): pilot.acq.sha256(batch/"plan.json"),
        str(protocol_path.resolve()): pilot.acq.sha256(protocol_path),
        str(verify_log.resolve()): pilot.acq.sha256(verify_log),
        str(Path(__file__).resolve()): pilot.acq.sha256(__file__)}
    for index, row in enumerate(plan["records"]):
        folder = batch/row["song_id"]
        entry = pilot.acq.read_sealed(folder/"entry.json")
        if (entry["binding"] != bound or entry["song_id"] != row["song_id"] or entry["pcm_sha256"] != row["source"]["pcm_sha256"] or
            entry["training_eligible"] is not False or entry["per_song_listening_review"] != "pending" or
            set(entry["files_sha256"]) != {"vocals.wav", "accompaniment.wav"}):
            raise ValueError("Changed entry binding or unexpected automatic eligibility")
        for name, expected in entry["files_sha256"].items():
            if pilot.acq.sha256(folder/name) != expected:
                raise ValueError("Label changed since complete verification")
        bindings[str((folder/"entry.json").resolve())] = pilot.acq.sha256(folder/"entry.json")
        result, flags = metrics(folder/"vocals.wav", folder/"accompaniment.wav", row["source"]["samples"])
        records.append({"song_id": row["song_id"], "track_id": row["track_id"], "technical_passed": True,
                        "metrics": result, "listening_hints": flags, "training_eligible": False})
        pending.append({"song_id": row["song_id"], "track_id": row["track_id"], "paired_24": row["song_id"] in paired,
                        "source_path": row["source"]["path"], "vocals_path": str((folder/"vocals.wav").resolve()),
                        "backing_path": str((folder/"accompaniment.wav").resolve()), "listening_hints": flags,
                        "suggested_same_input_windows": listening_windows(row["source"]["samples"]),
                        "review_status": "pending", "source_rights_status": "pending", "training_eligible": False,
                        "review_questions": ["Input/output alignment", "Vocal residual/omissions", "Instrument bleed/damage",
                                             "Transient/boundary artifacts", "Rights and intended training use"]})
        if (index+1) % 10 == 0:
            print(f"LABEL_TECHNICAL_AUDIT {index+1}/{count}", flush=True)
    pending.sort(key=lambda r: (not r["paired_24"], not bool(r["listening_hints"]), r["song_id"]))
    out.mkdir(parents=True)
    report = {"schema": 1, "bindings_sha256": bindings, "songs": count, "technical_passed": count,
              "songs_with_listening_hints": sum(bool(r["listening_hints"]) for r in records), "records": records,
              "paired_24_song_ids": paired, "quality_approved": False, "training_authorized": False,
              "caveat": "Finite/format/identity checks do not measure separation quality. No human review or rights approval implied."}
    pilot.acq.write_new_json(out/"technical_audit.json", pilot.acq.seal(report))
    pilot.acq.write_new_json(out/"listening_pending.json", pilot.acq.seal({"schema": 1, "records": pending,
        "audit_sha256": pilot.acq.sha256(out/"technical_audit.json"), "reviewed": 0, "training_authorized": False}))
    print(f"LABEL_TECHNICAL_AUDIT PASS songs={count}; listening/rights pending; no training", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, default=bulk.DEFAULT_OUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify-log", type=Path, default=bulk.DEFAULT_OUT/"post_complete_verify_20261002.log")
    ap.add_argument("--protocol", type=Path, default=data.PROTOCOL)
    args = ap.parse_args()
    audit(args.batch, args.out, args.verify_log, args.protocol)


if __name__ == "__main__":
    main()
