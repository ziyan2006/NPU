"""Quality metrics for 2-stem separation WITHOUT ground-truth stems.

You only handed us a finished stereo mix, so classic BSS-Eval (SDR/SIR/SAR)
is not computable -- those all need the isolated reference stems.  Instead we
combine three families of evidence, and we are explicit about what each one
can and cannot prove:

1. Pseudo-reference agreement  (anchor = strongest model, default htdemucs)
   SI-SDR / LSD / correlation of each candidate against the anchor.
   -> answers "how far is this model from the best separator we have".
   -> blind spot: if anchor and candidate share an error, it stays hidden.

2. Reference-free artifact indicators
   - vocal-floor: 10th-percentile frame energy of the vocal stem.  A model
     that leaks the whole backing track into "vocals" cannot have a low floor.
   - spectral flatness of the vocal stem (musical-noise / hiss proxy).
   - energy split: share of mixture energy routed to the vocal bus.
   -> answers "does the vocal stem actually sound like an isolated voice".

3. Reconstruction check
   SI-SDR(mixture, vocals+accompaniment).  For umx this is ~0 dB by
   construction (accompaniment := mixture - vocals) so it only carries
   information for the demucs family.

Usage
-----
python 05_metrics.py --anchor htdemucs
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from common import RESULTS, SR, STEMS

CANDIDATES = ["umx", "htdemucs", "hdemucs_mmi", "mdx", "mdx7ecf8ec1"]


def load(name):
    p = STEMS / name / "vocals.wav"
    if not p.exists():
        return None
    v, _ = sf.read(str(p), dtype="float32", always_2d=True)
    a, _ = sf.read(str(STEMS / name / "accompaniment.wav"), dtype="float32",
                   always_2d=True)
    m, _ = sf.read(str(STEMS / name / "mixture.wav"), dtype="float32",
                   always_2d=True)
    return v.T, a.T, m.T  # (C, T)


# --------------------------------------------------------------------------
def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> float:
    """Scale-invariant SDR in dB, averaged over channels."""
    vals = []
    for c in range(min(est.shape[0], ref.shape[0])):
        e, r = est[c].astype(np.float64), ref[c].astype(np.float64)
        alpha = float((e * r).sum() / ((r * r).sum() + eps))
        target = alpha * r
        noise = e - target
        vals.append(10.0 * np.log10(
            float((target ** 2).sum()) / (float((noise ** 2).sum()) + eps) + eps))
    return float(np.mean(vals))


def lsd_db(est, ref, n_fft=2048, hop=512, fmin=100.0, fmax=8000.0) -> float:
    """Log-spectral distance over the vocal band, averaged over channels."""
    import librosa

    vals = []
    for c in range(est.shape[0]):
        E = np.abs(librosa.stft(est[c].astype(np.float32), n_fft=n_fft,
                                hop_length=hop)) ** 2
        R = np.abs(librosa.stft(ref[c].astype(np.float32), n_fft=n_fft,
                                hop_length=hop)) ** 2
        freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
        band = (freqs >= fmin) & (freqs <= fmax)
        d = 10 * np.log10(E[band] + 1e-10) - 10 * np.log10(R[band] + 1e-10)
        vals.append(float(np.mean(d ** 2) ** 0.5))
    return float(np.mean(vals))


def corr(est, ref) -> float:
    vals = []
    for c in range(est.shape[0]):
        e, r = est[c].astype(np.float64), ref[c].astype(np.float64)
        if e.std() < 1e-9 or r.std() < 1e-9:
            continue
        vals.append(float(np.corrcoef(e, r)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def frame_rms_db(x, win=0.05, hop=0.025):
    n = int(win * SR)
    h = int(hop * SR)
    mono = x.mean(0).astype(np.float64)
    if len(mono) < n:
        return np.array([-120.0])
    idx = range(0, len(mono) - n + 1, h)
    r = np.array([np.sqrt(np.mean(mono[i:i + n] ** 2) + 1e-12) for i in idx])
    return 20 * np.log10(r + 1e-12)


def spectral_flatness(x, n_fft=2048, hop=512) -> float:
    """Mean spectral flatness (0 = tonal, 1 = white noise) of the vocal stem."""
    import librosa

    vals = [float(np.mean(librosa.feature.spectral_flatness(
        y=x[c].astype(np.float32), n_fft=n_fft, hop_length=hop)))
        for c in range(x.shape[0])]
    return float(np.mean(vals))


def band_energy_frac(x, lo: float, hi: float, n_fft=2048, hop=512) -> float:
    """Fraction of a stem's energy that falls inside [lo, hi) Hz."""
    import librosa

    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    m = (freqs >= lo) & (freqs < hi)
    fr = []
    for c in range(x.shape[0]):
        S = np.abs(librosa.stft(x[c].astype(np.float32), n_fft=n_fft,
                                hop_length=hop)) ** 2
        fr.append(float(S[m].sum() / (S.sum() + 1e-12)))
    return float(np.mean(fr))


def rms_db(x) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12)))


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="htdemucs")
    ap.add_argument("--out", default=str(RESULTS / "metrics.json"))
    args = ap.parse_args()

    stems = {}
    for name in CANDIDATES:
        got = load(name)
        if got is not None:
            stems[name] = got
    if not stems:
        print("no stems found; run 03_separate.py first")
        return 1

    mix = next(iter(stems.values()))[2]
    total = float(np.mean(mix.astype(np.float64) ** 2))
    mix_rms = rms_db(mix)

    rows = {}
    for name, (v, a, m) in stems.items():
        row = {
            "vocal_rms_dB": round(rms_db(v), 2),
            "acc_rms_dB": round(rms_db(a), 2),
            "vocal_energy_share": round(
                float(np.mean(v.astype(np.float64) ** 2)) / total, 4),
            "vocal_floor_p10_dB": round(float(np.percentile(frame_rms_db(v), 10)), 2),
            "vocal_floor_p50_dB": round(float(np.percentile(frame_rms_db(v), 50)), 2),
            "vocal_flatness": round(spectral_flatness(v), 4),
            "recon_si_sdr_dB": round(si_sdr(v + a, m), 2),
            # reference-free crosstalk probes: where the energy sits
            "vocal_lowband_frac_lt150": round(band_energy_frac(v, 0, 150), 4),
            "vocal_midband_frac_300_3400": round(band_energy_frac(v, 300, 3400), 4),
            "acc_lowband_frac_lt150": round(band_energy_frac(a, 0, 150), 4),
            "acc_midband_frac_300_3400": round(band_energy_frac(a, 300, 3400), 4),
        }
        rows[name] = row

    # ---- pseudo-reference agreement ----
    anchor = args.anchor if args.anchor in stems else None
    if anchor:
        av, aa, _ = stems[anchor]
        for name, (v, a, _) in stems.items():
            rows[name]["anchor"] = anchor
            if name == anchor:
                rows[name]["vs_anchor_vocal_si_sdr_dB"] = 99.0
                rows[name]["vs_anchor_acc_si_sdr_dB"] = 99.0
                rows[name]["vs_anchor_vocal_lsd_dB"] = 0.0
                rows[name]["vs_anchor_vocal_corr"] = 1.0
                continue
            rows[name]["vs_anchor_vocal_si_sdr_dB"] = round(si_sdr(v, av), 2)
            rows[name]["vs_anchor_acc_si_sdr_dB"] = round(si_sdr(a, aa), 2)
            rows[name]["vs_anchor_vocal_lsd_dB"] = round(lsd_db(v, av), 2)
            rows[name]["vs_anchor_vocal_corr"] = round(corr(v, av), 4)

    # ---- consensus: leave-one-out mean SI-SDR against all other vocal stems ----
    names = sorted(stems)
    for name in names:
        others = [n for n in names if n != name]
        if not others:
            continue
        vals = [si_sdr(stems[name][0], stems[o][0]) for o in others]
        rows[name]["consensus_vocal_si_sdr_dB"] = round(float(np.mean(vals)), 2)
        rows[name]["consensus_std_dB"] = round(float(np.std(vals)), 2)

    # ---- pairwise matrix (vocals) ----
    pair = {}
    for i, j in itertools.combinations(names, 2):
        pair[f"{i}|{j}"] = round(si_sdr(stems[i][0], stems[j][0]), 2)

    payload = {
        "anchor": anchor,
        "mixture_rms_dB": round(mix_rms, 2),
        "per_model": rows,
        "pairwise_vocal_si_sdr_dB": pair,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- console table ----
    hdr = ["model", "voc_rms", "floor_p10", "flatness", "v_lo<150",
           "v_mid", "recon", "vSDR@anc", "vLSD@anc", "consensus"]
    print("".join(f"{h:>11s}" for h in hdr))
    for name in names:
        r = rows[name]
        cells = [
            name,
            f"{r['vocal_rms_dB']:.1f}",
            f"{r['vocal_floor_p10_dB']:.1f}",
            f"{r['vocal_flatness']:.4f}",
            f"{r['vocal_lowband_frac_lt150']:.3f}",
            f"{r['vocal_midband_frac_300_3400']:.3f}",
            f"{r['recon_si_sdr_dB']:.1f}",
            f"{r.get('vs_anchor_vocal_si_sdr_dB', float('nan')):.1f}",
            f"{r.get('vs_anchor_vocal_lsd_dB', float('nan')):.1f}",
            f"{r.get('consensus_vocal_si_sdr_dB', float('nan')):.1f}",
        ]
        print("".join(f"{c:>11s}" for c in cells))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
