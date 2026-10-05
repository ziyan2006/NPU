"""Export playback-only teacher comparisons, without modifying float labels.

Each song gets ONE common safe gain across mix and both teachers' stems.
This avoids player clipping, not perceptual loudness differences. These copies
are NOT training labels or evidence that human listening has passed.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import soundfile as sf
import torch

spec = importlib.util.spec_from_file_location("teacher_listening_core", Path(__file__).with_name("131_run_teacher_pilot.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
PREVIEW_ROOT = pilot.ROOT / "试听文件"
DEFAULT_OUT = PREVIEW_ROOT / "teacher_pilot_20261002"
# Two ordinary fixed pilot examples, plus one high-disagreement diagnostic.
# Selection is for listening coverage, never for quantitative scoring.
DEFAULT_IDS = ("clip_043", "clip_045", "clip_042")
SELECTION_NOTE = "Two ordinary examples and one disagreement diagnostic; not a random/blind quality sample"
PCM_TOLERANCE = 2 ** -22


def guard_output(out):
    resolved, root = out.resolve(), PREVIEW_ROOT.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Playback copies must remain below ignored 试听文件")


def safe_gain(waves):
    if not waves:
        raise ValueError("Missing comparison waveforms")
    shape = next(iter(waves.values())).shape
    for wave in waves.values():
        pilot.finite_stereo(wave)
        if wave.shape != shape:
            raise ValueError("Playback comparisons must be aligned")
    peak = max(float(wave.abs().max()) for wave in waves.values())
    return min(1., .95 / max(peak, 1e-12))


def write_playback_new(path, wave, gain):
    if path.exists():
        raise ValueError("Refusing playback overwrite")
    pilot.finite_stereo(wave)
    if not 0 < gain <= 1 or float((wave * gain).abs().max()) > .950001:
        raise ValueError("Invalid safe playback gain")
    sf.write(path, (wave * gain).numpy().T, pilot.SR, subtype="PCM_24")
    check = pilot.read_wave(path)
    if check.shape != wave.shape or float((check - wave * gain).abs().max()) > PCM_TOLERANCE:
        raise ValueError("PCM playback readback changed alignment or samples")
    return pilot.acq.sha256(path)


def inputs_for_clip(cohort, assets, runtime, row):
    paths = {"mix.wav": Path(row["mix_file"])}
    receipts = {}
    for teacher in pilot.TEACHERS:
        folder = cohort / teacher / row["clip_id"]
        pilot.verify_entry(folder, pilot.entry_binding(cohort, teacher, assets, runtime), row)
        receipt = folder / "entry.json"
        receipts[str(receipt.resolve())] = pilot.acq.sha256(receipt)
        for stem in ("vocals", "accompaniment"):
            paths[f"{teacher}_{stem}.wav"] = folder / f"{stem}.wav"
    return paths, receipts


def export(cohort, assets, out, clip_ids):
    guard_output(out)
    if out.exists():
        raise ValueError("New playback directory required; existing evidence is preserved")
    if not clip_ids or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Need unique pilot clips")
    doc = pilot.verify_cohort(cohort)
    runtime = pilot.runtime_binding(assets)
    rows = {row["clip_id"]: row for row in doc["records"]}
    if any(ident not in rows or rows[ident]["role"] != "pseudo_label_pilot_only" for ident in clip_ids):
        raise ValueError("This listening export is for sealed MP3 pilot excerpts only")
    records = []
    out.mkdir(parents=True)
    for ident in clip_ids:
        row = rows[ident]
        paths, receipts = inputs_for_clip(cohort, assets, runtime, row)
        waves = {name: pilot.read_wave(path) for name, path in paths.items()}
        gain = safe_gain(waves)
        folder = out / ident
        folder.mkdir()
        files = {name: write_playback_new(folder / name, wave, gain) for name, wave in waves.items()}
        records.append({"clip_id": ident, "track_id": row["track_id"], "samples": row["samples"],
                        "seconds": row["seconds"], "common_playback_gain": gain,
                        "source_paths": {name: str(path.resolve()) for name, path in paths.items()},
                        "source_sha256": {name: pilot.acq.sha256(path) for name, path in paths.items()},
                        "entry_receipts_sha256": receipts, "playback_files_sha256": files})
    manifest = {"schema": 1, "cohort_path": str(cohort.resolve()), "cohort_sha256": pilot.acq.sha256(cohort / "cohort.json"),
                "asset_path": str(assets.resolve()), "script_sha256": pilot.acq.sha256(__file__), "records": records,
                "sample_rate": pilot.SR, "subtype": "PCM_24", "selection_note": SELECTION_NOTE,
                "normalization": "ONE gain per song across all five waveforms; never amplify; peak <= 0.95",
                "strict_perceptual_loudness_matched": False, "listening_review": "pending",
                "training_label_copies": False, "student_training_authorized": False}
    pilot.acq.write_new_json(out / "manifest.json", pilot.acq.seal(manifest))
    verify(out)


def verify(out):
    guard_output(out)
    manifest = pilot.acq.read_sealed(out / "manifest.json")
    cohort, assets = Path(manifest["cohort_path"]), Path(manifest["asset_path"])
    if pilot.acq.sha256(cohort / "cohort.json") != manifest["cohort_sha256"] or pilot.acq.sha256(__file__) != manifest["script_sha256"]:
        raise ValueError("Playback binding changed")
    doc = pilot.verify_cohort(cohort)
    runtime = pilot.runtime_binding(assets)
    rows = {row["clip_id"]: row for row in doc["records"]}
    for item in manifest["records"]:
        row = rows[item["clip_id"]]
        paths, receipts = inputs_for_clip(cohort, assets, runtime, row)
        waves = {name: pilot.read_wave(path) for name, path in paths.items()}
        if receipts != item["entry_receipts_sha256"] or safe_gain(waves) != item["common_playback_gain"]:
            raise ValueError("Playback gain or teacher receipts changed")
        if set(paths) != set(item["playback_files_sha256"]) or row["samples"] != item["samples"]:
            raise ValueError("Playback comparison is incomplete")
        for name, path in paths.items():
            preview = out / item["clip_id"] / name
            if pilot.acq.sha256(path) != item["source_sha256"][name] or pilot.acq.sha256(preview) != item["playback_files_sha256"][name]:
                raise ValueError("Playback or original label changed")
            info = sf.info(preview)
            saved = pilot.read_wave(preview)
            expected = waves[name] * item["common_playback_gain"]
            if info.subtype != "PCM_24" or saved.shape != expected.shape or float((saved - expected).abs().max()) > PCM_TOLERANCE:
                raise ValueError("Playback is not the aligned common-gain PCM copy")
    print(f"TEACHER_LISTENING VERIFIED songs={len(manifest['records'])}; human_review=pending labels_unchanged=True", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, default=pilot.DEFAULT_OUT)
    ap.add_argument("--assets", type=Path, default=pilot.assets.ASSETS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clip-ids", nargs="+", default=DEFAULT_IDS)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.verify:
        verify(args.out)
    else:
        export(args.cohort, args.assets, args.out, args.clip_ids)


if __name__ == "__main__":
    main()
