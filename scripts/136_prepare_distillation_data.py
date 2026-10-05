"""CPU-only full-song label input/crop prototype and locked teacher contrast.

No CUDA API calls, optimizer, teacher inference, worker modifications or student
promotion. The current snapshots are audit-only and reject training usage until
a separately reviewed training import is implemented.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch

spec = importlib.util.spec_from_file_location("distillation_input_bulk", Path(__file__).with_name("134_generate_teacher_library.py"))
bulk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bulk)
pilot, ROOT = bulk.pilot, bulk.ROOT
PROTOCOL = ROOT/"docs/teacher_distillation_protocol_v1.json"
DEFAULT_OUT = ROOT/"results/distillation_input_prepare_20261002"


def digest_json(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_input(config):
    integers = ("sample_rate", "fft", "hop", "crop_frames", "warmup_frames", "start_grid_frames")
    if (any(type(config[name]) is not int for name in integers) or
        config["sample_rate"] != 44100 or config["fft"] != 1024 or config["hop"] != 256 or
        config["crop_frames"] < 32 or config["crop_frames"] % 8 or
        config["warmup_frames"] < 96 or config["warmup_frames"] % 8 or config["start_grid_frames"] != 8 or
        not 0 <= config["source_edge_seconds"] <= 1 or len(config["common_gain_db_range"]) != 2 or
        not all(math.isfinite(v) for v in config["common_gain_db_range"]) or
        config["common_gain_db_range"][0] > config["common_gain_db_range"][1]):
        raise ValueError("Unsupported crop/front-end/context configuration")


def geometry(config):
    validate_input(config)
    length = (config["crop_frames"]+config["warmup_frames"]-1)*config["hop"]
    score = slice((config["warmup_frames"]+2)*config["hop"], length-2*config["hop"])
    edge = round(config["source_edge_seconds"]*config["sample_rate"])
    if score.stop <= score.start or score.start < edge:
        raise ValueError("No edge-safe score region")
    return length, score, max(0, edge-2*config["hop"])


def validate_protocol(protocol):
    expected_graph = {"bands": 128, "layout": "legacy_log", "bottleneck_blocks": 2,
                      "frontend": ["linear", 1.0], "lf_kill_bands": 44}
    baseline = json.loads((ROOT/"docs/model_training_protocol_v1.json").read_text(encoding="utf-8"))
    p = protocol["paired_comparison_planned"]
    if (protocol["schema"] != 1 or protocol["graph"] != expected_graph or protocol["frozen_sha256"] != baseline["frozen_sha256"] or
        protocol["student_training_authorized"] is not False or protocol["student_promotion_authorized"] is not False or
        p["arms"] != ["htdemucs_waveform_control", "kim_melband_waveform_candidate"] or
        p["true_batch_domains"] != ["musdb", "mir1k", "instrumental"] or p["effective_batch"] != 6 or
        p["pseudo_batch_slots"] != 3 or p["microbatch"] != 1 or p["pseudo_source_songs"] != 24 or
        not 2000 <= p["minimum_steps_before_early_stop"] <= p["maximum_steps"] <= 10000):
        raise ValueError("Unsupported graph, unpaired teacher experiment or premature training authority")
    geometry(protocol["input"])


def crop_recipe(rows, config, seed, cursor):
    if not rows or type(cursor) is not int or cursor < 0:
        raise ValueError("Need a valid counter and source pool")
    length, score, tail = geometry(config)
    rng = random.Random(int(hashlib.sha256(f"{seed}:{cursor}".encode()).hexdigest(), 16))
    row = rows[rng.randrange(len(rows))]
    limit = row["source"]["samples"]-length-tail
    if limit < 0:
        raise ValueError("Source too short; never silently pad or shorten context")
    grid = config["start_grid_frames"]*config["hop"]
    return {"song_id": row["song_id"], "cursor": cursor, "start_sample": rng.randrange(limit//grid+1)*grid,
            "samples": length, "common_db": rng.uniform(*config["common_gain_db_range"]),
            "score_start": score.start, "score_end": score.stop}


def common_gain(x, db):
    pilot.finite_stereo(x)
    if not math.isfinite(db):
        raise ValueError("Nonfinite augmentation gain")
    return min(10**(db/20), .95/max(float(x.abs().max()), 1e-12))


def choose_pairs(rows, count, seed):
    ids = sorted(r["song_id"] for r in rows)
    if count < 1 or count > len(ids) or len(ids) != len(set(ids)):
        raise ValueError("Invalid paired teacher source pool")
    return sorted(random.Random(seed).sample(ids, count))


def stat_signature(path):
    item = Path(path).stat()
    return item.st_size, item.st_mtime_ns, item.st_ctime_ns


def read_window(path, start, length):
    with sf.SoundFile(path) as stream:
        if stream.samplerate != pilot.SR or stream.channels != 2 or stream.subtype != "FLOAT" or start < 0 or start+length > len(stream):
            raise ValueError("Need aligned FLOAT32 label window without padding")
        stream.seek(start)
        wave = stream.read(length, dtype="float32", always_2d=True)
    value = torch.from_numpy(np.ascontiguousarray(wave.T))
    pilot.finite_stereo(value)
    return value


class TeacherWaveformDataset:
    def __init__(self, snapshot_path, purpose="training", cache_songs=1):
        self.path = Path(snapshot_path)
        self.doc = pilot.acq.read_sealed(self.path)
        if purpose != "cpu_audit" or self.doc.get("training_authorized") is not False:
            raise ValueError("Current import is CPU-audit-only; per-song approval and training importer are still required")
        if cache_songs < 1:
            raise ValueError("Positive bounded CPU cache required")
        self.config = self.doc["input"]
        geometry(self.config)
        self.cursor, self.seed, self.cache_songs = 0, self.doc["sampler_seed"], cache_songs
        self.cache, self.signatures = OrderedDict(), {}
        self.bound = pilot.acq.sha256(self.path)
        for path, expected in self.doc["bindings_sha256"].items():
            if pilot.acq.sha256(path) != expected:
                raise ValueError("Snapshot recipe/source role binding changed")
        self.rows = self.doc["records"]
        if not self.rows or len({r["song_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Missing/duplicate input songs")
        self.by_id = {row["song_id"]: row for row in self.rows}
        for row in self.rows:
            if row["role"] != "pseudo_label_train_candidate" or row["training_eligible"] is not False:
                raise ValueError("Unexpected role or automatic eligibility promotion")
            for name, info in row["label_files"].items():
                if name not in ("vocals.wav", "accompaniment.wav") or pilot.acq.sha256(info["path"]) != info["sha256"]:
                    raise ValueError("Saved label changed")
                item = sf.info(info["path"])
                if (item.frames != row["source"]["samples"] or item.samplerate != pilot.SR or item.channels != 2 or item.subtype != "FLOAT"):
                    raise ValueError("Label channel/sample-count/subtype differs from source")
                self.signatures[info["path"]] = stat_signature(info["path"])
            if set(row["label_files"]) != {"vocals.wav", "accompaniment.wav"}:
                raise ValueError("Missing paired label")
            source = row["source"]
            if pilot.acq.sha256(source["path"]) != source["sha256"]:
                raise ValueError("Original MP3 changed")
            self.signatures[source["path"]] = stat_signature(source["path"])
        # Target-independent: paired teachers must share this identity and x.
        self.mix_identity = digest_json([(r["song_id"], r["source"]["pcm_sha256"], r["source"]["samples"]) for r in self.rows])

    def _source(self, row):
        for path in (row["source"]["path"], *(info["path"] for info in row["label_files"].values())):
            if stat_signature(path) != self.signatures[path]:
                raise ValueError("Previously verified source/label changed during iteration")
        ident = row["song_id"]
        if ident in self.cache:
            wave = self.cache.pop(ident)
        else:
            wave = bulk.decode(Path(row["source"]["path"]))
            if wave.shape[-1] != row["source"]["samples"] or pilot.wave_digest(wave) != row["source"]["pcm_sha256"]:
                raise ValueError("Full PCM differs from sealed teacher input; no approximate MP3 seek allowed")
        self.cache[ident] = wave
        while len(self.cache) > self.cache_songs:
            self.cache.popitem(last=False)
        return wave

    def crop(self, recipe):
        row = self.by_id[recipe["song_id"]]
        length, region, tail = geometry(self.config)
        start = recipe["start_sample"]
        if (type(start) is not int or start < 0 or start % (self.config["hop"]*self.config["start_grid_frames"]) or
            start+length+tail > row["source"]["samples"] or recipe["samples"] != length):
            raise ValueError("Invalid context/grid/source crop")
        full = self._source(row)
        x = full[:, start:start+length].clone()
        v = read_window(row["label_files"]["vocals.wav"]["path"], start, length)
        a = read_window(row["label_files"]["accompaniment.wav"]["path"], start, length)
        expected, _ = pilot.residual_pair(x, v)
        if not torch.equal(a, expected):
            raise ValueError("Crop labels are not exact original mix-minus-vocal")
        gain = common_gain(x, recipe["common_db"])
        x, v, a = x*gain, v*gain, a*gain
        if float((x-(v+a)).abs().max()) > 2e-6*max(1., float(v.abs().max()), float(a.abs().max())):
            raise ValueError("Common-gain mixture consistency failed")
        meta = {**recipe, "pcm_sha256": row["source"]["pcm_sha256"], "gain": gain,
                "mix_identity": self.mix_identity, "input_pcm_sha256": pilot.wave_digest(x),
                "score_start": region.start, "score_end": region.stop, "training_eligible": False}
        return {"x": x, "v": v, "a": a, "meta": meta}

    def next_crop(self):
        recipe = crop_recipe(self.rows, self.config, self.seed, self.cursor)
        result = self.crop(recipe)
        self.cursor += 1
        return result

    def state_dict(self):
        return {"snapshot_sha256": self.bound, "seed": self.seed, "cursor": self.cursor}

    def load_state_dict(self, state):
        if (state["snapshot_sha256"] != self.bound or state["seed"] != self.seed or
            type(state["cursor"]) is not int or state["cursor"] < 0):
            raise ValueError("Sampler resume binding/seed/counter mismatch")
        self.cursor = state["cursor"]


def assert_paired_inputs(left, right):
    for key in ("mix_identity", "song_id", "pcm_sha256", "start_sample", "samples", "gain", "input_pcm_sha256"):
        if left["meta"][key] != right["meta"][key]:
            raise ValueError("Teacher comparison changed input/crop/gain")
    if not torch.equal(left["x"], right["x"]):
        raise ValueError("Teacher comparison inputs are not sample-identical")


def prepare(batch, out, protocol_path, count):
    bulk.guard_output(out)
    if out.exists():
        raise ValueError("Fresh CPU preparation directory required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    if pilot.acq.sha256(batch/"plan.json") != protocol["batch_plan_sha256"] or protocol["student_training_authorized"]:
        raise ValueError("Unexpected batch or training authority")
    geometry(protocol["input"])
    plan = bulk.verify_plan(batch)
    assets = pilot.assets.ASSETS
    expected_binding = bulk.binding(batch, assets)
    selected = plan["records"][:count]
    if count < 1 or len(selected) != count:
        raise ValueError("Invalid input audit song count")
    records, bindings = [], {str((batch/"plan.json").resolve()): pilot.acq.sha256(batch/"plan.json"),
                            str(protocol_path.resolve()): pilot.acq.sha256(protocol_path), str(Path(__file__).resolve()): pilot.acq.sha256(__file__),
                            str((ROOT/"docs/model_training_protocol_v1.json").resolve()): pilot.acq.sha256(ROOT/"docs/model_training_protocol_v1.json"),
                            str((ROOT/"scripts/119_model_selection_suite.py").resolve()): pilot.acq.sha256(ROOT/"scripts/119_model_selection_suite.py"),
                            str((ROOT/"scripts/126_verify_training_baseline.py").resolve()): pilot.acq.sha256(ROOT/"scripts/126_verify_training_baseline.py")}
    # First N IDs, NOT selected by teacher quality. Reading complete songs only.
    for row in selected:
        folder = batch/row["song_id"]
        entry = bulk.verify_song(folder, row, expected_binding)
        if entry["role"] != row["role"] or not entry["technical_waveform_checks_passed"] or entry["training_eligible"]:
            raise ValueError("Unexpected batch completion/eligibility")
        receipt = folder/"entry.json"
        bindings[str(receipt.resolve())] = pilot.acq.sha256(receipt)
        records.append({"song_id": row["song_id"], "role": row["role"], "source": row["source"],
                        "training_eligible": False, "label_files": {name: {"path": str((folder/name).resolve()), "sha256": sha}
                            for name, sha in entry["files_sha256"].items()}})
    pair = protocol["paired_comparison_planned"]
    pair_ids = choose_pairs(plan["records"], pair["pseudo_source_songs"], pair["cohort_seed"])
    paired_seconds = sum(r["source"]["seconds"] for r in plan["records"] if r["song_id"] in pair_ids)
    snapshot = {"schema": 1, "input": protocol["input"], "sampler_seed": pair["cohort_seed"], "records": records,
                "bindings_sha256": bindings, "training_authorized": False, "purpose": "CPU loader audit, not a train or validation split",
                "paired_teacher_24_song_ids": pair_ids, "paired_teacher_audio_seconds": paired_seconds,
                "matched_baseline_extra_label_bytes_with_margin": bulk.storage_estimate(paired_seconds, plan["protocol"]["storage"]),
                "pair_selection_from_sealed_plan_before_student_scoring": True,
                "blind_acceptance_ready": False}
    out.mkdir(parents=True)
    path = out/"snapshot.json"
    pilot.acq.write_new_json(path, pilot.acq.seal(snapshot))
    audit(path, out/"cpu_audit.json")


def audit(snapshot, report_path):
    if report_path.exists():
        raise ValueError("Fresh audit report required")
    bulk.guard_output(report_path.parent)
    data = TeacherWaveformDataset(snapshot, purpose="cpu_audit")
    crops = []
    for _ in range(8):
        crop = data.next_crop()
        crops.append(crop["meta"])
    saved = data.state_dict()
    expected = data.next_crop()
    data.load_state_dict(saved)
    restored = data.next_crop()
    assert_paired_inputs(expected, restored)
    if not torch.equal(expected["v"], restored["v"]) or not torch.equal(expected["a"], restored["a"]):
        raise ValueError("Sampler resume changed teacher targets")
    spectrum = pilot.core.stft_batch(restored["x"][None])
    frames = data.config["crop_frames"]+data.config["warmup_frames"]
    if spectrum.shape != (1, 2, 513, frames):
        raise ValueError("CPU student frontend shape mismatch")
    rebuilt = pilot.core.t09._istft(spectrum.flatten(0, 1), restored["x"].shape[-1]).reshape_as(restored["x"])
    error = float((rebuilt-restored["x"]).abs().max())
    if error > 2e-6:
        raise ValueError("CPU STFT/iSTFT identity check failed")
    report = {"schema": 1, "snapshot_sha256": pilot.acq.sha256(snapshot), "device": "cpu", "cuda_api_used": False,
              "songs": len(data.rows), "crop_draws": crops, "sample_cursor_resume_identical": True,
              "frontend_shape": list(spectrum.shape), "stft_istft_max_absolute_error": error,
              "student_training_performed": False, "per_song_review_pending": True, "training_import_rejected_until_review": True}
    pilot.acq.write_new_json(report_path, pilot.acq.seal(report))
    print(f"DISTILLATION_CPU_INPUT PASS songs={len(data.rows)} frames={frames} resume=identical; no training/GPU", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, default=bulk.DEFAULT_OUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--songs", type=int, default=2)
    ap.add_argument("--audit-snapshot", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.audit_snapshot:
        audit(args.audit_snapshot, args.out/"cpu_audit_readback.json")
    else:
        prepare(args.batch, args.out, args.protocol, args.songs)


if __name__ == "__main__":
    main()
