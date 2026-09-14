"""Build leakage-safe, true-stem caches for the tiny causal separator.

Supported sources:

* OnAir v4: royalty-free full mixes and multitracks at 44.1 kHz.  The three
  ``misc`` tracks with an explicit human vocal mapping are used; one whole song
  is held out before any 30-second crops are made.
* MIR-1K: 1,000 split-stereo karaoke clips (left accompaniment, right vocal).
  Clips are grouped by their original song before splitting, preventing clips
  from one song appearing on both sides of the evaluation.
* mshoxxDB: 18 full-length electronic instrumental mixtures.  These are
  zero-vocal negatives: they teach the separator not to erase synth leads,
  pads, bass, or drums, but they are never presented as positive vocal data.

The emitted ``train/*.pt`` files have exactly the same band-domain schema as
``11_smoke_train.py``'s Demucs pseudo-label cache, so they can be added with
``--extra-cache-dir``.  ``holdout`` is never consumed by that option; it exists
for true-reference evaluation in ``24_true_stem_eval.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import correlate, correlation_lags, resample_poly

from common import RESULTS, SR


_HERE = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t09 = _load_module("t09_true_cache", "09_target_model.py")
t11 = _load_module("t11_true_cache", "11_smoke_train.py")


ONAIR_VOCALS = {
    "Under My Skin - OnAir Music": ["Vox.wav"],
    "Set Me Free - OnAir Music": ["Vox.wav"],
    "Ronin - OnAir Music": ["Ronin ft. KXNE Stems - Acappella.wav"],
}


@dataclass
class TrackSpec:
    track_id: str
    dataset: str
    mix_files: list[str]
    vocal_files: list[str]
    stem_files: list[str] = field(default_factory=list)


def _find_named_dir(root: Path, name: str) -> Path | None:
    if root.name == name and root.is_dir():
        return root
    matches = [p for p in root.rglob(name) if p.is_dir()]
    return matches[0] if matches else None


def discover_onair(root: Path) -> list[TrackSpec]:
    tracks = []
    for track_id, vocal_names in ONAIR_VOCALS.items():
        folder = _find_named_dir(root, track_id)
        if folder is None:
            continue
        mix = folder / f"{track_id}.wav"
        vocals = [folder / name for name in vocal_names]
        if not mix.exists() or not all(p.exists() for p in vocals):
            missing = [str(p) for p in [mix, *vocals] if not p.exists()]
            raise FileNotFoundError(f"incomplete OnAir track {track_id}: {missing}")
        tracks.append(TrackSpec(track_id, "onair", [str(mix.resolve())],
                                [str(p.resolve()) for p in vocals]))
    if not tracks:
        raise FileNotFoundError(
            f"no mapped OnAir vocal tracks below {root}; extract the v4 misc zip first")
    return tracks


def _mir_song_id(path: Path) -> str:
    """Drop the final clip number while preserving singer + original song id."""
    parts = path.stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else path.stem


def discover_mir1k(root: Path) -> list[TrackSpec]:
    wav_dirs = [p for p in root.rglob("Wavfile") if p.is_dir()]
    wav_dir = root if root.name.lower() == "wavfile" else (
        wav_dirs[0] if wav_dirs else None)
    if wav_dir is None:
        raise FileNotFoundError(f"MIR-1K Wavfile directory not found below {root}")
    groups: dict[str, list[Path]] = {}
    for path in sorted(wav_dir.glob("*.wav")):
        groups.setdefault(_mir_song_id(path), []).append(path)
    if not groups:
        raise FileNotFoundError(f"no MIR-1K wav files in {wav_dir}")
    return [TrackSpec(song_id, "mir1k",
                      [str(p.resolve()) for p in paths], [])
            for song_id, paths in sorted(groups.items())]


def discover_mshoxx(root: Path) -> list[TrackSpec]:
    tracks = []
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        # Each piece folder is ``<title>_v1`` and its mastered mixture has the
        # same basename.  The other FLAC files are instrument stems.
        mixture = folder / f"{folder.name}.flac"
        if mixture.exists():
            stems = sorted(p for p in folder.glob("*.flac") if p != mixture)
            tracks.append(TrackSpec(folder.name, "mshoxx",
                                    [str(mixture.resolve())], [],
                                    [str(p.resolve()) for p in stems]))
    if not tracks:
        raise FileNotFoundError(
            f"no mshoxxDB piece mixtures below {root}; extract the archive first")
    return tracks


def discover_tracks(root: Path, dataset: str) -> list[TrackSpec]:
    if dataset == "onair":
        return discover_onair(root)
    if dataset == "mir1k":
        return discover_mir1k(root)
    return discover_mshoxx(root)


def _read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        common = math.gcd(sr, SR)
        audio = resample_poly(audio, SR // common, sr // common, axis=0).astype(
            np.float32, copy=False)
    return np.ascontiguousarray(audio.T), SR


def _stereo(audio: np.ndarray) -> np.ndarray:
    if audio.shape[0] == 1:
        return np.repeat(audio, 2, axis=0)
    if audio.shape[0] >= 2:
        return audio[:2]
    raise ValueError(f"audio has no channels: {audio.shape}")


def _align_vocal(mix: np.ndarray, vocal: np.ndarray,
                 max_lag_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Align a raw OnAir stem to its mastered mix by waveform correlation.

    Most OnAir stems share sample zero with the mix, but Ronin's stem export is
    about 3.71 seconds early.  Matching only lengths silently produced a
    perfectly plausible tensor with a completely false target.  Estimate lag
    cheaply at 2.205 kHz, refine it to the nearest source sample, then trim the
    leading offset from whichever signal starts earlier.
    """
    down = 20
    xm = resample_poly(mix.mean(axis=0), 1, down)
    vm = resample_poly(vocal.mean(axis=0), 1, down)
    xm = xm - xm.mean()
    vm = vm - vm.mean()
    corr = correlate(xm, vm, mode="full", method="fft")
    lags = correlation_lags(len(xm), len(vm), mode="full")
    keep = np.abs(lags) <= round(max_lag_s * SR / down)
    coarse = int(lags[keep][np.argmax(np.abs(corr[keep]))]) * down

    # At lag L, mix sample n corresponds to vocal sample n-L.  Refine around
    # the decimated estimate on several interior excerpts so sub-millisecond
    # phase error does not poison waveform metrics.
    probe_n = min(round(5.0 * SR), mix.shape[-1] // 8, vocal.shape[-1] // 8)
    best_lag, best_score = coarse, -1.0
    for lag in range(coarse - down - 2, coarse + down + 3):
        score = 0.0
        for frac in (0.25, 0.5, 0.75):
            x0 = round(frac * mix.shape[-1])
            v0 = x0 - lag
            if x0 < 0 or v0 < 0:
                continue
            n = min(probe_n, mix.shape[-1] - x0, vocal.shape[-1] - v0)
            if n <= 0:
                continue
            xa = mix[:, x0:x0 + n].reshape(-1)
            va = vocal[:, v0:v0 + n].reshape(-1)
            denom = float(np.linalg.norm(xa) * np.linalg.norm(va)) + 1e-12
            score += abs(float(np.dot(xa, va))) / denom
        if score > best_score:
            best_score, best_lag = score, lag

    if best_lag < 0:
        vocal = vocal[:, -best_lag:]
    elif best_lag > 0:
        mix = mix[:, best_lag:]
    n = min(mix.shape[-1], vocal.shape[-1])
    return (np.ascontiguousarray(mix[:, :n]),
            np.ascontiguousarray(vocal[:, :n]))


def load_track(spec: TrackSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stereo ``(mixture, vocal)`` tensors at the product sample rate."""
    if spec.dataset == "onair":
        mix, _ = _read(Path(spec.mix_files[0]))
        mix = _stereo(mix)
        vocal_parts = [_stereo(_read(Path(p))[0]) for p in spec.vocal_files]
        n = min([mix.shape[-1], *[v.shape[-1] for v in vocal_parts]])
        vocal = np.sum([v[..., :n] for v in vocal_parts], axis=0,
                       dtype=np.float32)
        mix, vocal = _align_vocal(mix, vocal)
        return torch.from_numpy(mix), torch.from_numpy(vocal)

    if spec.dataset == "mshoxx":
        released, _ = _read(Path(spec.mix_files[0]))
        if spec.stem_files:
            # The released mshoxxDB files are mono.  Duplicating them to L/R
            # lets the network identify this dataset from perfect channel
            # correlation instead of learning electronic timbre.  Rebuild a
            # deterministic stereo mix from the aligned instrument stems with
            # equal-power pans and small gain variation to remove that shortcut.
            parts = []
            for filename in spec.stem_files:
                stem, _ = _read(Path(filename))
                mono = stem.mean(axis=0)
                digest = hashlib.sha256(Path(filename).name.encode("utf-8")).digest()
                pan = 0.08 + 0.84 * (int.from_bytes(digest[:4], "little") / 2**32)
                gain_db = -3.0 + 6.0 * (int.from_bytes(digest[4:8], "little") / 2**32)
                angle = pan * math.pi / 2.0
                gain = 10.0 ** (gain_db / 20.0)
                parts.append(np.stack((mono * math.cos(angle) * gain,
                                       mono * math.sin(angle) * gain)))
            n = min(p.shape[-1] for p in parts)
            mix = np.sum([p[..., :n] for p in parts], axis=0, dtype=np.float32)
            # Match the released master's RMS without hard clipping.  The
            # target remains exact because the whole synthetic mix is non-vocal.
            target_rms = float(np.sqrt(np.mean(released.astype(np.float64) ** 2)))
            mix_rms = float(np.sqrt(np.mean(mix.astype(np.float64) ** 2)))
            if mix_rms > 1e-12:
                mix *= target_rms / mix_rms
            peak = float(np.max(np.abs(mix)))
            if peak > 0.98:
                mix *= 0.98 / peak
            mix = np.ascontiguousarray(mix)
        else:
            mix = _stereo(released)
        # mshoxxDB is instrumental.  The exact vocal reference is silence and
        # the exact accompaniment reference is the selected/generated mix.
        mix_tensor = torch.from_numpy(mix)
        return mix_tensor, torch.zeros_like(mix_tensor)

    mixes, vocals = [], []
    for filename in spec.mix_files:
        split, _ = _read(Path(filename))
        if split.shape[0] < 2:
            raise ValueError(f"MIR-1K file is not split stereo: {filename}")
        accompaniment = split[0:1]
        vocal = split[1:2]
        mixes.append(_stereo(accompaniment + vocal))
        vocals.append(_stereo(vocal))
    return (torch.from_numpy(np.ascontiguousarray(np.concatenate(mixes, axis=-1))),
            torch.from_numpy(np.ascontiguousarray(np.concatenate(vocals, axis=-1))))


def _vocal_ratio_db(vocal: torch.Tensor, mix: torch.Tensor) -> float:
    vrms = vocal.float().square().mean().sqrt()
    mrms = mix.float().square().mean().sqrt()
    return float(20 * torch.log10((vrms + 1e-12) / (mrms + 1e-12)))


def _save_track(cache_dir: Path, spec: TrackSpec, track_index: int,
                segment_s: float, min_segment_s: float,
                Wa: torch.Tensor, train_vocal_gains_db: tuple[float, ...] = (0.0,),
                augment_train: bool = False) -> list[dict]:
    mix, vocal = load_track(spec)
    segment_n = round(segment_s * SR)
    min_n = round(min_segment_s * SR)
    records = []
    rank = 0
    for start in range(0, mix.shape[-1], segment_n):
        end = min(start + segment_n, mix.shape[-1])
        if end - start < min_n:
            continue
        x = mix[..., start:end]
        v = vocal[..., start:end]
        gain_db = 0.0
        if augment_train:
            gain_db = train_vocal_gains_db[(track_index * 7 + rank) %
                                            len(train_vocal_gains_db)]
            if gain_db:
                # Exact source-level remix: the original accompaniment is
                # recovered before changing vocal level.  This teaches both the
                # quiet EDM vocal case and the strong karaoke/rap case without
                # the magnitude-domain approximation used by --aug-stem-db.
                accompaniment = x - v
                v = v * (10.0 ** (gain_db / 20.0))
                x = accompaniment + v
        mix_band, mask_band = t11.band_targets(x, v, Wa, "magnitude")
        filename = f"t{track_index:03d}_k{rank:03d}.pt"
        ratio_db = _vocal_ratio_db(v, x)
        torch.save({
            "mix": mix_band.half(),
            "mask": mask_band.half(),
            "src": spec.track_id,
            "dataset": spec.dataset,
            "dur": (end - start) / SR,
            "start_s": start / SR,
            "vocal_db": ratio_db,
            "vocal_gain_db": gain_db,
            "target_mask": "magnitude",
            "band_layout": t09.DEFAULT_BAND_LAYOUT,
            "n_bands": int(Wa.shape[1]),
        }, cache_dir / filename)
        records.append({"file": filename, "start_s": round(start / SR, 3),
                        "dur_s": round((end - start) / SR, 3),
                        "vocal_db": round(ratio_db, 2),
                        "vocal_gain_db": gain_db})
        rank += 1
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=("onair", "mir1k", "mshoxx"))
    ap.add_argument("--root", required=True, help="extracted dataset root")
    ap.add_argument("--output", default=None,
                    help="cache root (default results/<dataset>_true_cache)")
    ap.add_argument("--segment-s", type=float, default=30.0)
    ap.add_argument("--min-segment-s", type=float, default=5.0)
    ap.add_argument("--holdout-tracks", type=int, default=None,
                    help="whole source songs held out; defaults: 1 OnAir, "
                         "20 MIR-1K, 4 mshoxxDB")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--n-bands", type=int, default=t09.N_BANDS)
    ap.add_argument("--train-vocal-gains-db", default="0",
                    help="comma-separated exact vocal gains cycled across train "
                         "segments; holdout audio is always left at 0 dB")
    args = ap.parse_args()
    train_vocal_gains_db = tuple(float(x) for x in
                                 args.train_vocal_gains_db.split(",") if x.strip())
    if not train_vocal_gains_db:
        raise SystemExit("--train-vocal-gains-db must contain at least one value")

    root = Path(args.root).resolve()
    out = (Path(args.output) if args.output else
           RESULTS / f"{args.dataset}_true_cache").resolve()
    tracks = discover_tracks(root, args.dataset)
    n_holdout = args.holdout_tracks
    if n_holdout is None:
        n_holdout = {"onair": 1, "mir1k": 20, "mshoxx": 4}[args.dataset]
    if not 0 < n_holdout < len(tracks):
        raise SystemExit(f"holdout must be in [1, {len(tracks) - 1}]")

    # Keep the most different OnAir vocal (rap/trap-flavoured Ronin) as the
    # untouched test song.  MIR-1K has enough songs for a seeded random split.
    if args.dataset == "onair" and n_holdout == 1:
        holdout_ids = {"Ronin - OnAir Music"}
    else:
        order = [t.track_id for t in tracks]
        random.Random(args.seed).shuffle(order)
        holdout_ids = set(order[-n_holdout:])

    for split in ("train", "holdout"):
        (out / split).mkdir(parents=True, exist_ok=True)
    Wa = torch.from_numpy(t09.make_analysis_matrix(args.n_bands))
    manifest_tracks = []
    counts = {"train": 0, "holdout": 0}
    audio_s = {"train": 0.0, "holdout": 0.0}
    for index, spec in enumerate(tracks):
        split = "holdout" if spec.track_id in holdout_ids else "train"
        records = _save_track(out / split, spec, index, args.segment_s,
                              args.min_segment_s, Wa, train_vocal_gains_db,
                              augment_train=(split == "train"))
        counts[split] += len(records)
        audio_s[split] += sum(r["dur_s"] for r in records)
        manifest_tracks.append({**asdict(spec), "split": split,
                                "cache_items": records})
        print(f"  {split:7s} {spec.track_id}: {len(records)} segments, "
              f"{sum(r['dur_s'] for r in records):.1f}s")

    metadata = {
        "schema": 1,
        "dataset": args.dataset,
        "source_root": str(root),
        "source_license": ({
            "onair": "CC BY-SA 4.0",
            "mir1k": "MIR-1K research dataset; see bundled README/license",
            "mshoxx": "CC BY-NC-SA 4.0",
        }[args.dataset]),
        "sample_rate": SR,
        "n_bands": args.n_bands,
        "band_layout": t09.DEFAULT_BAND_LAYOUT,
        "target_mask": "magnitude",
        "train_vocal_gains_db": list(train_vocal_gains_db),
        "segment_s": args.segment_s,
        "seed": args.seed,
        "holdout_track_ids": sorted(holdout_ids),
        "counts": counts,
        "audio_s": {k: round(v, 1) for k, v in audio_s.items()},
        "tracks": manifest_tracks,
    }
    (out / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    for split in ("train", "holdout"):
        (out / split / "_meta.json").write_text(json.dumps({
            "dataset": args.dataset,
            "split": split,
            "entries": counts[split],
            "audio_s": round(audio_s[split], 1),
            "n_bands": args.n_bands,
            "band_layout": t09.DEFAULT_BAND_LAYOUT,
            "target_mask": "magnitude",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out / 'manifest.json'}")
    print(f"train {counts['train']} crops / {audio_s['train']/60:.1f} min; "
          f"holdout {counts['holdout']} crops / {audio_s['holdout']/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
