"""Export the fixed 24 teacher pairs as private, playback-only comparisons.

No teacher inference, training, eligibility changes or original label writes.
Each song has one safe gain shared by all three windows and all five signals.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil

import soundfile as sf
import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = load("paired_listening_inputs", "136_prepare_distillation_data.py")
playback = load("paired_listening_pcm", "133_export_teacher_listening.py")
pilot, bulk = data.pilot, data.bulk
PREVIEW_ROOT = bulk.ROOT / "试听文件"
DEFAULT_OUT = PREVIEW_ROOT / "teacher_pairs_20261002"
DEFAULT_INPUT = bulk.ROOT / "results/paired_distillation_prepare_20261002"
ASSETS = Path(__file__).with_name("paired_listening_assets")
SIGNALS = ("mix", "htdemucs_vocals", "htdemucs_accompaniment", "kim_melband_vocals", "kim_melband_accompaniment")


def guard_output(out):
    out, root = Path(out).resolve(), PREVIEW_ROOT.resolve()
    if out == root or not out.is_relative_to(root):
        raise ValueError("Playback output must be a child of ignored 试听文件")
    return out


def read_inputs(prepared):
    prepared = Path(prepared)
    bundle = pilot.acq.read_sealed(prepared / "paired_bundle.json")
    pending = pilot.acq.read_sealed(prepared / "listening_pending.json")
    if (bundle["training_authorized"] is not False or bundle["human_reviewed_pairs"] != 0 or
            pending["reviewed"] != 0 or pending["training_authorized"] is not False or
            pending["bundle_sha256"] != pilot.acq.sha256(prepared / "paired_bundle.json")):
        raise ValueError("Need the unchanged pending pair preparation; this exporter cannot approve it")
    for path, expected in bundle["bindings_sha256"].items():
        if pilot.acq.sha256(path) != expected:
            raise ValueError(f"Paired preparation binding changed: {path}")
    snapshots = {}
    for teacher in ("htdemucs", "kim_melband"):
        info = bundle["snapshots"][teacher]
        if pilot.acq.sha256(info["path"]) != info["sha256"]:
            raise ValueError("Teacher snapshot changed")
        snapshots[teacher] = pilot.acq.read_sealed(info["path"])
        if snapshots[teacher]["training_authorized"] is not False:
            raise ValueError("Current snapshots must remain audit-only")
    ids = bundle["pair_ids"]
    if (len(ids) != 24 or len(set(ids)) != 24 or any(not re.fullmatch(r"song_\d{4}", ident) for ident in ids) or
            [r["song_id"] for r in pending["records"]] != ids or
            any([r["song_id"] for r in doc["records"]] != ids for doc in snapshots.values())):
        raise ValueError("Exactly the fixed 24 pairs in their original order are required")
    for index, item in enumerate(pending["records"]):
        left, right = (snapshots[t]["records"][index] for t in ("htdemucs", "kim_melband"))
        if (left["source"] != right["source"] or item["same_input_pcm_sha256"] != left["source"]["pcm_sha256"] or
                Path(item["source_path"]).resolve() != Path(left["source"]["path"]).resolve() or
                item["review_status"] != "pending" or item["rights_status"] != "pending" or item["training_eligible"] is not False):
            raise ValueError("Pair input identity, pending review or rights changed")
        for row in (left, right):
            if row["role"] != "pseudo_label_train_candidate" or row["training_eligible"] is not False:
                raise ValueError("Unexpected source role or eligibility")
            if set(row["label_files"]) != {"vocals.wav", "accompaniment.wav"}:
                raise ValueError("Missing teacher stem")
        valid_windows(item["windows"], left["source"]["samples"])
    return bundle, pending, snapshots


def valid_windows(windows, samples):
    if len(windows) != 3:
        raise ValueError("Exactly three preselected listening windows required")
    for item in windows:
        start, length = item["start_sample"], item["samples"]
        if (type(start) is not int or type(length) is not int or start < 0 or length < 1 or
                start + length > samples or length > 18 * pilot.SR):
            raise ValueError("Invalid listening window; no padding or approximate seek")
    if len({item["samples"] for item in windows}) != 1:
        raise ValueError("All three listening windows must have the same length")


def budget(pending):
    # PCM24 stereo, five signals, fixed windows, headers/UI and 1% margin.
    return int(sum(w["samples"] for r in pending["records"] for w in r["windows"]) * 5 * 6 * 1.01) + 1024 * 1024


def song_windows(left, right, windows):
    source = left["source"]
    if source != right["source"] or pilot.acq.sha256(source["path"]) != source["sha256"]:
        raise ValueError("Original paired MP3 identity changed")
    paths = [source["path"]]
    for row in (left, right):
        for info in row["label_files"].values():
            if pilot.acq.sha256(info["path"]) != info["sha256"]:
                raise ValueError("Original paired FLOAT label changed")
            paths.append(info["path"])
    signatures = {path: data.stat_signature(path) for path in paths}
    full = bulk.decode(Path(source["path"]))
    if full.shape[-1] != source["samples"] or pilot.wave_digest(full) != source["pcm_sha256"]:
        raise ValueError("Full original decoded FLOAT input differs; no MP3 time seek fallback")
    valid_windows(windows, full.shape[-1])
    clips = []
    for window in windows:
        start, length = window["start_sample"], window["samples"]
        waves = {"mix": full[:, start:start + length].clone()}
        for teacher, row in (("htdemucs", left), ("kim_melband", right)):
            v = data.read_window(row["label_files"]["vocals.wav"]["path"], start, length)
            a = data.read_window(row["label_files"]["accompaniment.wav"]["path"], start, length)
            expected, _ = pilot.residual_pair(waves["mix"], v)
            if not torch.equal(a, expected):
                raise ValueError("Playback source stems are not exact input-minus-vocal")
            waves[f"{teacher}_vocals"], waves[f"{teacher}_accompaniment"] = v, a
        clips.append(waves)
    if any(data.stat_signature(path) != signature for path, signature in signatures.items()):
        raise ValueError("Source/label changed during playback export")
    return clips


def write_new_text(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def export(prepared, out):
    out = guard_output(out)
    if out.exists():
        raise ValueError("Existing playback pack preserved; use verify, not another export")
    bundle, pending, snapshots = read_inputs(prepared)
    estimate = budget(pending)
    bulk.check_storage(shutil.disk_usage(out.parent).free, estimate, 12 * 1024**3)
    asset_names = ("index.html", "app.js", "style.css")
    asset_sha = {str((ASSETS / name).resolve()): pilot.acq.sha256(ASSETS / name) for name in asset_names}
    out.mkdir(parents=True)
    records, catalog_rows = [], []
    for index, item in enumerate(pending["records"]):
        left, right = (snapshots[t]["records"][index] for t in ("htdemucs", "kim_melband"))
        clips = song_windows(left, right, item["windows"])
        # ONE gain across every signal AND every window of the same song.
        gain = playback.safe_gain({f"{i}_{key}": wave for i, waves in enumerate(clips) for key, wave in waves.items()})
        record = {"song_id": item["song_id"], "track_id": item["track_id"], "common_playback_gain": gain,
                  "original_source": left["source"], "label_sha256": {t: snapshots[t]["records"][index]["label_files"]
                                                                         for t in snapshots}, "windows": []}
        for i, (window, waves) in enumerate(zip(item["windows"], clips), 1):
            folder = out / item["song_id"] / f"w{i}"
            folder.mkdir(parents=True)
            files = {}
            for name in SIGNALS:
                path = folder / f"{name}.wav"
                files[path.relative_to(out).as_posix()] = playback.write_playback_new(path, waves[name], gain)
            record["windows"].append({**window, "files_sha256": files})
        records.append(record)
        catalog_rows.append({"song_id": item["song_id"], "title": item["track_id"], "gain": gain,
                             "hints": item["listening_hints"], "windows": item["windows"]})
        print(f"PAIRED_LISTENING_EXPORT {index + 1}/24 song={item['song_id']} gain={gain:.6f}", flush=True)
    bindings = {str((Path(prepared) / name).resolve()): pilot.acq.sha256(Path(prepared) / name)
                for name in ("paired_bundle.json", "listening_pending.json")}
    bindings.update(asset_sha)
    bindings[str(Path(__file__).resolve())] = pilot.acq.sha256(__file__)
    catalog = {"schema": 1, "bundle_sha256": pilot.acq.sha256(Path(prepared) / "paired_bundle.json"),
               "sample_rate": pilot.SR, "records": catalog_rows, "training_authorized": False}
    inert_json = json.dumps(catalog, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    page = (ASSETS / "index.html").read_text(encoding="utf-8").replace("__CATALOG_DATA__", inert_json)
    write_new_text(out / "index.html", page)
    for name in ("app.js", "style.css"):
        write_new_text(out / name, (ASSETS / name).read_text(encoding="utf-8"))
    manifest = {"schema": 1, "bundle_sha256": catalog["bundle_sha256"], "bindings_sha256": bindings,
                "records": records, "sample_rate": pilot.SR, "subtype": "PCM_24", "estimated_bytes": estimate,
                "ui_files_sha256": {name: pilot.acq.sha256(out / name) for name in asset_names},
                "normalization": "ONE non-amplifying safe gain per song across 3 windows and all 5 signals; peak <=0.95",
                "strict_perceptual_loudness_matched": False, "human_review": "pending", "rights_review": "pending",
                "training_authorized": False, "training_label_copies": False, "music_uploaded": False,
                "drafts": "Browser-only notes, not a signed approval or training input"}
    pilot.acq.write_new_json(out / "manifest.json", pilot.acq.seal(manifest))
    verify(out)


def pack_files(doc):
    files = dict(doc["ui_files_sha256"])
    if set(files) != {"index.html", "app.js", "style.css"}:
        raise ValueError("Unexpected listening UI files")
    for row in doc["records"]:
        if not re.fullmatch(r"song_\d{4}", row["song_id"]) or len(row["windows"]) != 3:
            raise ValueError("Invalid song/window pack")
        for i, window in enumerate(row["windows"], 1):
            expected = {f"{row['song_id']}/w{i}/{name}.wav" for name in SIGNALS}
            if set(window["files_sha256"]) != expected or files.keys() & expected:
                raise ValueError("Incomplete or unsafe listening file paths")
            files.update(window["files_sha256"])
    return files


def verify(out):
    out = guard_output(out)
    doc = pilot.acq.read_sealed(out / "manifest.json")
    if (doc["training_authorized"] is not False or doc["training_label_copies"] is not False or
            doc["music_uploaded"] is not False or doc["human_review"] != "pending" or doc["rights_review"] != "pending" or
            doc["sample_rate"] != pilot.SR or doc["subtype"] != "PCM_24" or len(doc["records"]) != 24 or
            len({r["song_id"] for r in doc["records"]}) != 24):
        raise ValueError("Playback metadata cannot become automatic approval")
    for path, expected in doc["bindings_sha256"].items():
        if pilot.acq.sha256(path) != expected:
            raise ValueError(f"Playback preparation/code binding changed: {path}")
    files = pack_files(doc)
    for name, expected in files.items():
        path = out / name
        if not path.resolve().is_relative_to(out) or pilot.acq.sha256(path) != expected:
            raise ValueError("Playback file changed or escaped pack")
    for row in doc["records"]:
        if not 0 < row["common_playback_gain"] <= 1:
            raise ValueError("Unexpected playback gain")
        for window in row["windows"]:
            for name in window["files_sha256"]:
                path = out / name
                info = sf.info(path)
                saved = pilot.read_wave(path)
                if (info.subtype != "PCM_24" or info.frames != window["samples"] or
                        float(saved.abs().max()) > .950001):
                    raise ValueError("Playback format/length/headroom changed")
    print("PAIRED_LISTENING VERIFIED songs=24 windows=72 audio=360; review=pending; no training", flush=True)
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    verify(args.out) if args.verify else export(args.prepared, args.out)


if __name__ == "__main__":
    main()
