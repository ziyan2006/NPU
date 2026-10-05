"""Sealed teacher pilot: 10 MP3 excerpts and old DEVELOPMENT true references.

No student updates, new Cambridge data, held-out library MP3s or SD access.
Full waveform labels, not band masks, are saved to a new ignored result folder.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load("teacher_pilot_core", "110_train_residual_ablation.py")
acq = load("teacher_pilot_seals", "128_acquire_cambridge_candidates.py")
assets = load("teacher_pilot_assets", "130_acquire_teacher_pilot.py")
DEFAULT_OUT = ROOT / "results/teacher_pilot_20261001"
SR = 44100
TEACHERS = ("htdemucs", "kim_melband_roformer")


def wave_digest(x):
    raw = x.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(raw.tobytes()).hexdigest()


def finite_stereo(x):
    if x.ndim != 2 or x.shape[0] != 2 or x.shape[-1] < 1 or not torch.isfinite(x).all():
        raise ValueError("Need finite aligned nonempty stereo waveform")


def residual_pair(x, vocal):
    finite_stereo(x)
    finite_stereo(vocal)
    if x.shape != vocal.shape:
        raise ValueError("Teacher output length/channel mismatch")
    # Floating point stems may exceed full scale. Never clip or independently
    # normalize them: exact mixture consistency matters for the student target.
    accompaniment = x - vocal
    error = float((x - (vocal + accompaniment)).abs().max())
    if error > 2e-6 * max(1., float(x.abs().max()), float(vocal.abs().max())):
        raise ValueError("Teacher residual-pair mixture consistency failed")
    return accompaniment, error


def write_wave_new(path, x):
    finite_stereo(x)
    if path.exists():
        raise ValueError(f"Refusing waveform overwrite: {path}")
    sf.write(path, x.detach().cpu().numpy().T, SR, subtype="FLOAT")
    return acq.sha256(path)


def read_wave(path):
    raw, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR or raw.shape[1] != 2:
        raise ValueError("Saved pilot audio representation changed")
    x = torch.from_numpy(np.ascontiguousarray(raw.T))
    finite_stereo(x)
    return x


def guard_mp3(names, known_regression):
    keys = [acq.song_key(name) for name in names]
    forbidden = {acq.song_key(name) for name in known_regression}
    if len(keys) != len(set(keys)) or any(key in forbidden or not key for key in keys):
        raise ValueError("Pilot MP3 duplicate or known regression/holdout composition")


def resolve_duplicate_mp3(matches):
    if not matches or not all(p.is_file() for p in matches):
        raise ValueError("Missing pilot MP3")
    files = {str(p.resolve()): acq.sha256(p) for p in sorted(matches)}
    decoded = {}
    if len(set(files.values())) > 1:
        # MP3 artwork/tags may differ. Only full decoded PCM identity permits
        # an automatic choice; an equal title/duration/5-second probe does not.
        for p in sorted(matches):
            value = subprocess.check_output([
                "ffmpeg", "-v", "error", "-i", str(p), "-map", "0:a:0", "-ar", str(SR), "-ac", "2",
                "-c:a", "pcm_f32le", "-f", "hash", "-hash", "sha256", "pipe:1"], timeout=180).decode().strip()
            if not value.startswith("SHA256="):
                raise ValueError("Unexpected decoded audio fingerprint")
            decoded[str(p.resolve())] = value.split("=", 1)[1]
        if len(set(decoded.values())) != 1:
            raise ValueError("Duplicate MP3 filename contains different decoded audio; review first")
    return sorted(matches)[0].resolve(), {"original_files_sha256": files, "full_decoded_stereo_44100_float32_sha256": decoded,
                                       "counted_as_compositions": 1}


def prepare(out, protocol_path, data_lock, library):
    if out.exists():
        raise ValueError("New pilot directory required; no role or input overwrite")
    if not out.resolve().is_relative_to((ROOT / "results").resolve()) or out.resolve() == (ROOT / "results").resolve():
        raise ValueError("Pilot must remain below ignored results")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    old = acq.read_sealed(data_lock)
    if acq.sha256(old["checkpoint"]) != old["frozen_sha256"]:
        raise ValueError("Frozen student changed")
    names = protocol["mp3"]["files"]
    guard_mp3(names, old["known_regression_ids"])
    paths, duplicates = {}, {}
    for name in names:
        matches = sorted(library.rglob(name))
        paths[name], duplicates[name] = resolve_duplicate_mp3(matches)
    dev = [r for r in old["records"] if r["role"] == "development" and r["domain"] in ("musdb", "instrumental")]
    if sum(r["domain"] == "musdb" for r in dev) != 19 or sum(r["domain"] == "instrumental" for r in dev) != 3:
        raise ValueError("Expected all 19+3 existing reserved development compositions")
    out.mkdir(parents=True)
    audio_folder = out / "inputs"
    audio_folder.mkdir()
    rows, sources = [], {}
    for value in duplicates.values():
        sources.update(value["original_files_sha256"])
    for record in dev:
        for field in ("mix_files", "vocal_files", "stem_files"):
            for name in record.get(field, []):
                path = Path(name)
                value = acq.sha256(path)
                if value != old["files"][name]["sha256"]:
                    raise ValueError("Original development source changed")
                sources[str(path.resolve())] = value
        if record["domain"] == "musdb":
            x, v = core.decode_musdb(Path(record["mix_files"][0]))
            start = 0
        else:
            spec = core.t23.TrackSpec(record["track_id"], "mshoxx", record["mix_files"], [], record["stem_files"])
            x, v = core.t23.load_track(spec)
            length = min(x.shape[-1], round(30 * SR))
            start = (x.shape[-1] - length) // 2
            x, v = x[:, start:start+length], v[:, start:start+length]
        for variant, db in (("native", 0), ("weak_minus12", -12)):
            if record["domain"] == "instrumental" and db:
                continue
            target = v * 10 ** (db / 20)
            mix = x - v + target
            gain = min(1., .95 / max(float(mix.abs().max()), 1e-12))
            mix, target = mix * gain, target * gain
            ident = f"clip_{len(rows):03d}"
            mix_path, ref_path = audio_folder / f"{ident}_mix.wav", audio_folder / f"{ident}_ref.wav"
            mix_sha, ref_sha = write_wave_new(mix_path, mix), write_wave_new(ref_path, target)
            rows.append({"clip_id": ident, "track_id": record["track_id"], "role": "development",
                         "domain": record["domain"] + "/" + variant, "vocal_gain_db": db,
                         "source_start_sample": start, "samples": mix.shape[-1], "seconds": mix.shape[-1]/SR,
                         "mix_file": str(mix_path.resolve()), "mix_sha256": mix_sha,
                         "reference_file": str(ref_path.resolve()), "reference_sha256": ref_sha,
                         "input_samples_sha256": wave_digest(mix), "common_gain": gain})
    for name in names:
        path = paths[name]
        sources[str(path)] = acq.sha256(path)
        duration = core.t11.ffprobe_duration(path)
        start_s = max(0., (duration - protocol["mp3"]["segment_seconds"]) / 2)
        mix = core.t11.decode_excerpt(path, start_s, min(duration, protocol["mp3"]["segment_seconds"]))
        ident = f"clip_{len(rows):03d}"
        mix_path = audio_folder / f"{ident}_mix.wav"
        rows.append({"clip_id": ident, "track_id": name, "role": "pseudo_label_pilot_only", "domain": "mp3/no_reference",
                     "source_file": str(path), "source_start_seconds": start_s,
                     "samples": mix.shape[-1], "seconds": mix.shape[-1]/SR,
                     "mix_file": str(mix_path.resolve()), "mix_sha256": write_wave_new(mix_path, mix),
                     "reference_file": None, "reference_sha256": None,
                     "input_samples_sha256": wave_digest(mix), "true_separation_reference_available": False})
    doc = {"schema": 1, "protocol_path": str(protocol_path.resolve()), "protocol_sha256": acq.sha256(protocol_path),
           "protocol": protocol, "data_lock_path": str(data_lock.resolve()), "data_lock_sha256": acq.sha256(data_lock),
           "frozen_checkpoint": old["checkpoint"], "frozen_sha256": old["frozen_sha256"],
           "baseline_checkpoint": str((ROOT / protocol["baseline"]["weight_path"]).resolve()),
           "baseline_sha256": acq.sha256(ROOT / protocol["baseline"]["weight_path"]),
           "source_sha256": sources, "records": rows, "mp3_duplicate_audit": duplicates,
           "dependency_sha256": {str((ROOT / "scripts" / name).resolve()): acq.sha256(ROOT / "scripts" / name)
                                 for name in ("11_smoke_train.py", "23_build_true_stem_cache.py", "110_train_residual_ablation.py", "131_run_teacher_pilot.py")},
           "no_new_cambridge_audio_used": True, "student_training_performed": False, "blind_evaluation_performed": False,
           "caveat": "Existing development may be in pretrained teachers' training sets; fixed MP3 excerpts are not proven to contain vocals"}
    acq.write_new_json(out / "cohort.json", acq.seal(doc))
    print(f"TEACHER_COHORT SEALED clips={len(rows)} mp3={len(names)} true_ref_dev={len(rows)-len(names)}", flush=True)


def verify_cohort(out, verify_sources=True):
    doc = acq.read_sealed(out / "cohort.json")
    for field, expected in (("protocol_path", "protocol_sha256"), ("data_lock_path", "data_lock_sha256"),
                            ("frozen_checkpoint", "frozen_sha256"), ("baseline_checkpoint", "baseline_sha256")):
        if acq.sha256(doc[field]) != doc[expected]:
            raise ValueError(f"Pilot input binding changed: {field}")
    for path, expected in doc["dependency_sha256"].items():
        if acq.sha256(path) != expected:
            raise ValueError("Pilot source recipe changed")
    if verify_sources:
        for path, expected in doc["source_sha256"].items():
            if acq.sha256(path) != expected:
                raise ValueError("Original pilot audio changed")
    old = acq.read_sealed(doc["data_lock_path"])
    dev_ids = {r["track_id"] for r in old["records"] if r["role"] == "development"}
    guard_mp3([r["track_id"] for r in doc["records"] if r["role"] == "pseudo_label_pilot_only"], old["known_regression_ids"])
    ids = set()
    for row in doc["records"]:
        if row["clip_id"] in ids or row["role"] not in ("development", "pseudo_label_pilot_only"):
            raise ValueError("Invalid/duplicated pilot role")
        ids.add(row["clip_id"])
        if row["role"] == "development" and row["track_id"] not in dev_ids:
            raise ValueError("Reference clip is not reserved DEVELOPMENT")
        for path_key, hash_key in (("mix_file", "mix_sha256"), ("reference_file", "reference_sha256")):
            if row[path_key] and acq.sha256(row[path_key]) != row[hash_key]:
                raise ValueError("Pilot decoded waveform changed")
        x = read_wave(row["mix_file"])
        if x.shape[-1] != row["samples"] or wave_digest(x) != row["input_samples_sha256"]:
            raise ValueError("Pilot samples differ from seal")
    return doc


def mel_demix(model, mix, chunk, overlap, device):
    """Author window/padding recipe with strict finite/coverage/length checks.

    Unlike the original helper, this never replaces NaN output with silence.
    Run the parity check against the pinned author's helper before using labels.
    """
    finite_stereo(mix)
    if chunk < 2048 or overlap < 1 or chunk % overlap:
        raise ValueError("Invalid teacher overlap configuration")
    original_length = mix.shape[-1]
    step, fade = chunk // overlap, chunk // 10
    border = chunk - step
    padded = original_length > 2 * border and border > 0
    audio = F.pad(mix, (border, border), mode="reflect") if padded else mix
    audio = audio.to(device)
    window = torch.ones(chunk, device=device)
    window[:fade] *= torch.linspace(0, 1, fade, device=device)
    window[-fade:] *= torch.linspace(1, 0, fade, device=device)
    result, counter = torch.zeros_like(audio), torch.zeros_like(audio)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        for start in range(0, audio.shape[-1], step):
            part = audio[:, start:start+chunk]
            length = part.shape[-1]
            if length < chunk:
                part = F.pad(part, (0, chunk-length), mode="reflect" if length > chunk//2 + 1 else "constant")
            predicted = model(part[None])[0]
            if predicted.shape != part.shape or not torch.isfinite(predicted).all():
                raise ValueError("Teacher chunk returned mismatched or non-finite samples")
            weights = window.clone()
            if start == 0:
                weights[:fade] = 1
            elif start + chunk >= audio.shape[-1]:
                weights[-fade:] = 1
            result[:, start:start+length] += predicted[:, :length].float() * weights[:length]
            counter[:, start:start+length] += weights[:length]
    if not torch.all(counter > 0):
        raise ValueError("Teacher overlap leaves uncovered samples")
    result = result / counter
    if padded:
        result = result[:, border:-border]
    result = result.cpu()
    if result.shape != mix.shape:
        raise ValueError("Teacher alignment/length changed")
    finite_stereo(result)
    return result


def load_mel(asset_path, protocol):
    assets.verify(asset_path, Path(protocol["protocol_path"]))
    runtime, source = asset_path / "runtime", asset_path / "source"
    sys.path.insert(0, str(runtime.resolve()))
    sys.path.insert(0, str(source.resolve()))
    # Only a tuple of scalar YAML values is supported, never arbitrary Python
    # objects from FullLoader/UnsafeLoader.
    class ConfigLoader(yaml.SafeLoader):
        pass
    ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    config = yaml.load((source / "configs/config_vocals_mel_band_roformer.yaml").read_text(), Loader=ConfigLoader)
    from models.mel_band_roformer import MelBandRoformer
    model = MelBandRoformer(**config["model"])
    state = torch.load(asset_path / protocol["protocol"]["candidate"]["weight_filename"], map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected pure tensor state_dict; unsafe checkpoint loading is prohibited")
    model.load_state_dict(state, strict=True)
    del state
    return model.eval(), config


def runtime_binding(asset_path):
    local_files = sorted(p for p in (asset_path / "runtime").rglob("*")
                         if p.is_file() and (p.suffix == ".py" or p.name == "METADATA"))
    return {"base_versions": {name: importlib.metadata.version(name) for name in (
        "torch", "numpy", "einops", "librosa", "soundfile", "demucs")},
        "isolated_runtime_sha256": {str(p.relative_to(asset_path)).replace("\\", "/"): acq.sha256(p) for p in local_files}}


def entry_binding(out, teacher, asset_path, runtime):
    return {"cohort_sha256": acq.sha256(out / "cohort.json"), "teacher": teacher,
            "script_sha256": acq.sha256(__file__), "runtime": runtime,
            "asset_receipt_sha256": acq.sha256(asset_path / "asset_receipt.json") if teacher != "htdemucs" else None}


def verify_entry(folder, binding, row):
    entry = acq.read_sealed(folder / "entry.json")
    if entry["binding"] != binding or entry["clip_id"] != row["clip_id"] or entry["input_samples_sha256"] != row["input_samples_sha256"]:
        raise ValueError("Teacher result belongs to a changed cohort/runtime")
    for name, expected in entry["files_sha256"].items():
        if name not in ("vocals.wav", "accompaniment.wav") or acq.sha256(folder / name) != expected:
            raise ValueError("Saved teacher result changed")
    if set(entry["files_sha256"]) != {"vocals.wav", "accompaniment.wav"}:
        raise ValueError("Teacher result missing a stem")
    x, v, a = read_wave(row["mix_file"]), read_wave(folder / "vocals.wav"), read_wave(folder / "accompaniment.wav")
    expected_a, error = residual_pair(x, v)
    if not torch.equal(a, expected_a):
        raise ValueError("Saved accompaniment is not exact input-minus-vocal")
    return entry


def run(out, teacher, asset_path, limit):
    if not torch.cuda.is_available():
        raise ValueError("CUDA pilot required; do not silently change precision/device")
    doc = verify_cohort(out)
    device = "cuda"
    runtime = runtime_binding(asset_path)
    binding = entry_binding(out, teacher, asset_path, runtime)
    settings = doc["protocol"]
    pending = []
    for row in doc["records"]:
        folder = out / teacher / row["clip_id"]
        if (folder / "entry.json").exists():
            verify_entry(folder, binding, row)
        else:
            if folder.exists():
                raise ValueError("Incomplete output preserved; cannot overwrite unreceipted stems")
            pending.append(row)
    if not pending:
        print(f"TEACHER_PILOT VERIFIED complete teacher={teacher} clips={len(doc['records'])}", flush=True)
        return
    if teacher == "htdemucs":
        from common import load_demucs
        model = load_demucs("htdemucs").to(device).eval()
    else:
        model, config = load_mel(asset_path, doc)
        model = model.to(device).eval()
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("pinned_mel_author_utils", asset_path / "source/utils.py")
        author = module_from_spec(spec)
        spec.loader.exec_module(author)
        sample = read_wave(pending[0]["mix_file"])
        official_config = SimpleNamespace(inference=SimpleNamespace(**config["inference"]),
                                          training=SimpleNamespace(**config["training"]))
        # Exact author's recipe parity is checked on the first real pilot clip,
        # not assumed from a mock. Duplicate call is excluded from label timing.
        official, _ = author.demix_track(official_config, model, sample.clone(), device)
        strict = mel_demix(model, sample, settings["candidate"]["chunk_samples"], settings["candidate"]["overlap"], device)
        official = torch.from_numpy(official["vocals"])
        if official.shape != strict.shape or not torch.allclose(official, strict, atol=2e-6, rtol=1e-5):
            raise ValueError("Strict teacher wrapper differs from pinned author inference")
        parity_path = out / teacher / "author_parity.json"
        parity_path.parent.mkdir(parents=True, exist_ok=True)
        if not parity_path.exists():
            acq.write_new_json(parity_path, acq.seal({"binding": binding, "clip_id": pending[0]["clip_id"],
                "maximum_absolute_difference": float((strict-official).abs().max()), "passed": True}))
        print("TEACHER_AUTHOR_PARITY PASS", flush=True)
    count = 0
    for row in pending:
        if limit and count >= limit:
            break
        x = read_wave(row["mix_file"])
        seed = settings["baseline"]["seed"] + int(row["clip_id"].split("_")[-1])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if teacher == "htdemucs":
            vocal = core.t11.teacher_vocals(model, x, device)
        else:
            vocal = mel_demix(model, x, settings["candidate"]["chunk_samples"], settings["candidate"]["overlap"], device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        accompaniment, consistency = residual_pair(x, vocal)
        folder = out / teacher / row["clip_id"]
        folder.mkdir(parents=True)
        files = {"vocals.wav": write_wave_new(folder / "vocals.wav", vocal),
                 "accompaniment.wav": write_wave_new(folder / "accompaniment.wav", accompaniment)}
        entry = {"schema": 1, "binding": binding, "clip_id": row["clip_id"], "track_id": row["track_id"],
                 "role": row["role"], "domain": row["domain"], "input_samples_sha256": row["input_samples_sha256"],
                 "files_sha256": files, "seconds": row["seconds"], "elapsed_s": elapsed,
                 "rtf": elapsed/row["seconds"], "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "mixture_consistency_max_error": consistency,
                 "vocal_peak": float(vocal.abs().max()), "source_listening_review": "pending", "student_training_eligible": False}
        acq.write_new_json(folder / "entry.json", acq.seal(entry))
        print(f"TEACHER_PILOT teacher={teacher} clip={row['clip_id']} domain={row['domain']} "
              f"seconds={row['seconds']:.2f} elapsed={elapsed:.2f} rtf={entry['rtf']:.4f} "
              f"peak_mib={entry['peak_cuda_allocated_bytes']/2**20:.1f}", flush=True)
        count += 1
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"TEACHER_PILOT saved={count} remaining={len(pending)-count} teacher={teacher}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "run", "verify"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--protocol", type=Path, default=assets.PROTOCOL)
    ap.add_argument("--data-lock", type=Path, default=ROOT / "results/training_protocol_20261001/dataset_lock.json")
    ap.add_argument("--library", type=Path, default=Path(os.environ.get("STEM_AUDIO_LIBRARY", "D:/DJ_Music_Library")))
    ap.add_argument("--assets", type=Path, default=assets.ASSETS)
    ap.add_argument("--teacher", choices=TEACHERS, default="htdemucs")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N new outputs; later run resumes verified clips")
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("limit must be nonnegative")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.action == "prepare":
        prepare(args.out, args.protocol, args.data_lock, args.library)
    elif args.action == "run":
        run(args.out, args.teacher, args.assets, args.limit)
    else:
        doc = verify_cohort(args.out)
        runtime = runtime_binding(args.assets)
        for teacher in TEACHERS:
            binding = entry_binding(args.out, teacher, args.assets, runtime)
            for row in doc["records"]:
                verify_entry(args.out / teacher / row["clip_id"], binding, row)
        print("TEACHER_PILOT ALL OUTPUTS VERIFIED; student/SD unchanged", flush=True)


if __name__ == "__main__":
    main()
