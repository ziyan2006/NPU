"""Lock existing train/development sources; explicitly leave new blind data missing.

Hashes original audio, not just caches. No downloads, holdout audio decoding,
checkpoint mutation, SD access or model-quality claims are performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_lock_selection", ROOT / "scripts/119_model_selection_suite.py")
suite = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = suite
spec.loader.exec_module(suite)
core = suite.core
FROZEN_SHA = "cb16d333c372d9558ef563b9b4baa9b9f7f46f0ec72d5b71f9e26557a1abfb55"
QUALITY = {
    "musdb": {"reference": "Decoded isolated AAC stems; synthetic mix = D+B+O+V",
              "flags": ["7-second excerpts, not full MUSDB18-HQ", "AAC bandwidth/codec effects", "check official stem-bleed errata"],
              "rights": "Per-track academic/research restrictions; see official MUSDB and original source licenses",
              "source": "https://sigsep.github.io/datasets/musdb.html"},
    "mir1k": {"reference": "Left accompaniment/right vocal; sum and duplicate to stereo",
              "flags": ["Not true spatial stereo mixtures", "song grouping is singer+song ID, not guaranteed singer isolation"],
              "rights": "Research dataset; consult local bundled terms before reuse",
              "source": "Local MIR-1K original package and manifest"},
    "onair": {"reference": "Aligned raw vocal vs mastered mix; backing proxy = mix - vocal",
              "flags": ["Only two training songs", "Master/stem processing mismatch; not exact isolated backing reference"],
              "rights": "Existing manifest reports CC BY-SA 4.0; verify original package scope before publishing derivatives",
              "source": "Local OnAir v4 package and manifest"},
    "instrumental": {"reference": "Synthetic stereo from aligned instrumental stems, vocal = 0",
                     "flags": ["Negative examples only; does not add electronic-vocal coverage", "Generated panning is not real stereo recording"],
                     "rights": "Existing manifest reports CC BY-NC-SA 4.0; no public redistribution in this workflow",
                     "source": "Local mshoxxDB v1.2 package and manifest"},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def document_digest(document):
    body = {k: v for k, v in document.items() if k != "content_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def validate_roles(document):
    if document.get("schema") != 1 or document.get("frozen_sha256") != FROZEN_SHA:
        raise ValueError("Unexpected data-lock schema or deployed starting checkpoint")
    roles = {name: set() for name in ("train", "development", "blind", "regression")}
    for record in document["records"]:
        role = record["role"]
        key = core.composition_key(record["track_id"])
        if role not in ("train", "development") or record["domain"] not in QUALITY or not key or key in roles[role]:
            raise ValueError("Invalid/duplicate composition or role")
        roles[role].add(key)
    roles["regression"] = {core.composition_key(s) for s in document["known_regression_ids"]}
    roles["blind"] = {core.composition_key(s) for s in document["blind_ids"]}
    names = list(roles)
    if any(roles[a] & roles[b] for i, a in enumerate(names) for b in names[i+1:]):
        raise ValueError("Train/development/regression/blind composition overlap")
    # This lock contains ONLY existing sources. New blind data requires a
    # separately reviewed import; an edited empty list can never become ready.
    if document["blind_ids"] or document["blind_status"] != "missing":
        raise ValueError("Existing seen material cannot be certified as new blind data")


def source_info(path):
    path = Path(path).resolve()
    if path.suffix.lower() == ".mp4":
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels,duration",
            "-of", "json", str(path)]))
        audio = [s for s in probe["streams"] if "sample_rate" in s]
        if len(audio) != 5:
            raise ValueError("MUSDB container does not contain five audio streams")
        info = {"audio_streams": len(audio), "sample_rate": int(audio[0]["sample_rate"]),
                "channels_per_stream": audio[0]["channels"], "duration_s": float(audio[0]["duration"])}
    else:
        item = sf.info(str(path))
        info = {"audio_streams": 1, "sample_rate": item.samplerate,
                "channels_per_stream": item.channels, "duration_s": item.frames/item.samplerate}
    return {"sha256": sha256(path), "bytes": path.stat().st_size, **info}


def inventory(corpus, checkpoint, seed):
    suite.guard_split(corpus)
    if sha256(checkpoint) != FROZEN_SHA:
        raise ValueError("Frozen checkpoint hash mismatch")
    records, files = [], {}
    for domain in sorted(corpus.train):
        for role, pool in (("train", corpus.train[domain]), ("development", corpus.val.get(domain, []))):
            for record in sorted(pool, key=lambda r: r["track_id"]):
                fields = {field: [str(Path(p).resolve()) for p in record.get(field, [])]
                          for field in ("mix_files", "vocal_files", "stem_files")}
                for path in sorted({p for values in fields.values() for p in values}):
                    if path not in files:
                        files[path] = source_info(path)
                records.append({"domain": domain, "track_id": record["track_id"], "role": role,
                                "may_have_been_seen_by_frozen_model": True,
                                "source_container_duration_s": sum(files[p]["duration_s"] for p in fields["mix_files"]),
                                **fields})
                if len(records) % 20 == 0:
                    print(f"DATA_LOCK records={len(records)} files={len(files)}", flush=True)
    doc = {"schema": 1, "seed": seed, "checkpoint": str(checkpoint.resolve()), "frozen_sha256": FROZEN_SHA,
           "records": records, "files": files, "quality": QUALITY,
           "known_regression_ids": corpus.final_ids, "blind_ids": [], "blind_status": "missing",
           "blind_reason": "Old holdouts repeatedly observed; no new electronic-vocal acceptance source has been imported/sealed",
           "teacher": {**corpus.teacher_meta, "used_for_loss": False, "scope": "Six local anchor songs only, not all historical training sources"},
           "manifest_provenance": corpus.manifests, "excluded_before_split": corpus.excluded,
           "dependency_sha256": {name: sha256(ROOT / "scripts" / name) for name in (
               "09_target_model.py", "11_smoke_train.py", "13_ab_compare.py", "23_build_true_stem_cache.py",
               "110_train_residual_ablation.py", "119_model_selection_suite.py", "125_lock_training_data.py")},
           "guard_limit": "NFKC/casefold title alias check, not fingerprints, singer isolation or exhaustive historical exposure tracking",
           "counts": {d: {r: sum(row["domain"] == d and row["role"] == r for row in records)
                          for r in ("train", "development")} for d in sorted(corpus.train)},
           "duration_caveat": "Source container seconds before resampling/alignment/cropping; overlapping views and remixes do not add independent music",
           "source_container_train_seconds": sum(r["source_container_duration_s"] for r in records if r["role"] == "train")}
    validate_roles(doc)
    doc["content_sha256"] = document_digest(doc)
    return doc


def verify_lock(path, verify_files=True):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_roles(document)
    if document_digest(document) != document.get("content_sha256"):
        raise ValueError("Data-lock content was changed")
    if sha256(document["checkpoint"]) != FROZEN_SHA:
        raise ValueError("Frozen checkpoint changed")
    for name, expected in document["dependency_sha256"].items():
        if sha256(ROOT / "scripts" / name) != expected:
            raise ValueError(f"Data recipe changed: {name}")
    for value in document["manifest_provenance"].values():
        if "path" in value and sha256(value["path"]) != value["sha256"]:
            raise ValueError("A source split manifest changed")
    if verify_files:
        for name, value in document["files"].items():
            if Path(name).stat().st_size != value["bytes"] or sha256(name) != value["sha256"]:
                raise ValueError(f"Original audio changed: {name}")
    return document


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/training_protocol_20261001/dataset_lock.json")
    ap.add_argument("--verify", type=Path)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    if args.verify:
        document = verify_lock(args.verify)
        print(f"DATA_LOCK VERIFIED records={len(document['records'])}; blind=missing", flush=True)
        return
    if args.out.exists():
        raise ValueError("Use a new data-lock path; never overwrite prior inventories")
    corpus = core.Corpus(ROOT / "data/datasets/MUSDB18-7-STEMS", ROOT / "results/opt_bott2_5m_cache", args.seed)
    document = inventory(corpus, ROOT / "models/student_bott2_mir1k_candidate.pt", args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"DATA_LOCK COMPLETE {document['counts']}; blind=missing; out={args.out}", flush=True)


if __name__ == "__main__":
    main()
