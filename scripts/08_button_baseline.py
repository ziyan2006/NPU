"""Button-toggle product: measure the FREE baseline before sizing the NPU.

Product spec: press a button on the board -> vocals disappear, instrumental keeps
playing; press again -> vocals come back.

Before deciding the model, we must know how far out-of-phase stereo cancellation
(center-channel removal, "OOPS") gets us for free. If it is already close enough
for a demo, the NPU has to beat that; if it destroys the track, the NPU is justified.

Metrics are reference-relative (no ground-truth stems available):
  - frame-level L/R correlation, split by vocal-active / vocal-inactive frames
  - how much of the reference vocal still leaks into each instrumental estimate
  - per-band energy change, to expose what each method damages
"""
import json
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SR = 44100
NFFT, HOP = 2048, 1024
BANDS = [(30, 120), (120, 500), (500, 2000), (2000, 8000), (8000, 16000)]
BAND_NAMES = ["30-120 kick/bass", "120-500 lowmid", "500-2k vocal f0",
              "2k-8k presence", "8k-16k air"]


def rd(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR, f"{p}: sr={sr}"
    return x


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12))


def n_frames(n):
    return 1 + (n - NFFT) // HOP


def frame_energy(x):
    mono = np.asarray(x, np.float64).mean(axis=1)
    nf = n_frames(len(mono))
    e = np.empty(nf)
    for i in range(nf):
        s = mono[i * HOP: i * HOP + NFFT]
        e[i] = float(np.mean(s ** 2))
    return e


def frame_corr(L, R, idx):
    out = []
    for i in idx:
        a = L[i * HOP: i * HOP + NFFT]
        b = R[i * HOP: i * HOP + NFFT]
        if a.std() < 1e-7 or b.std() < 1e-7:
            continue
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out)


def si_sdr(est, ref):
    est = np.asarray(est, np.float64).ravel()
    ref = np.asarray(ref, np.float64).ravel()
    a = float(np.dot(est, ref)) / (float(np.dot(ref, ref)) + 1e-12)
    t = a * ref
    e = est - t
    return float(10 * np.log10((np.dot(t, t) + 1e-12) / (np.dot(e, e) + 1e-12)))


def leakage_db(container, target):
    """Fraction of `target` energy still present inside `container`. Lower = cleaner."""
    c = np.asarray(container, np.float64).ravel()
    t = np.asarray(target, np.float64).ravel()
    a = float(np.dot(c, t)) / (float(np.dot(t, t)) + 1e-12)
    proj = a * t
    return float(10 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(t, t) + 1e-12)))


def band_energy_db(x, lo, hi):
    mono = np.asarray(x, np.float64).mean(axis=1)
    w = np.hanning(len(mono))
    f = np.fft.rfft(mono * w)
    freqs = np.fft.rfftfreq(len(mono), 1 / SR)
    sel = (freqs >= lo) & (freqs < hi)
    e = float(np.sum(np.abs(f[sel]) ** 2)) + 1e-20
    return 10 * np.log10(e)


def main():
    mix = rd(ROOT / "data/work/excerpt_60_100.wav")
    ht_v = rd(ROOT / "results/stems/htdemucs/vocals.wav")
    ht_a = rd(ROOT / "results/stems/htdemucs/accompaniment.wav")
    hd_a = rd(ROOT / "results/stems/hdemucs_mmi/accompaniment.wav")
    hd_v = rd(ROOT / "results/stems/hdemucs_mmi/vocals.wav")
    um_v = rd(ROOT / "results/stems/umx/vocals.wav")
    um_a = rd(ROOT / "results/stems/umx/accompaniment.wav")

    L, R = mix[:, 0].astype(np.float64), mix[:, 1].astype(np.float64)

    # ---- 1. stereo structure ----
    overall = float(np.corrcoef(L, R)[0, 1])
    ev = frame_energy(ht_v)
    thr = float(np.percentile(ev, 75))
    active = ev >= thr
    idx_a = np.where(active)[0]
    idx_i = np.where(~active)[0]
    ca, ci = frame_corr(L, R, idx_a), frame_corr(L, R, idx_i)

    # ---- 2. free baseline: out-of-phase stereo ("OOPS" / center cancellation) ----
    # Write L = M + S and R = M - S.  The karaoke output you HEAR is the side
    # signal S = (L-R)/2; the mid M = (L+R)/2 is what gets thrown away.  Vocals
    # are usually mixed centered, so S should be vocal-light and M vocal-heavy.
    # NOTE: output S on BOTH channels (not [S,-S]) -- an anti-phase pair cancels
    # itself in mono downmix, which is why the naive form measures as silence.
    s = (L - R) / 2
    m = (L + R) / 2
    side = np.stack([s, s], axis=1).astype(np.float32)
    mid = np.stack([m, m], axis=1).astype(np.float32)

    free = {
        "side_rms_dB": round(rms_db(side), 2),
        "side_minus_mix_dB": round(rms_db(side) - rms_db(mix), 2),
        "corr_side_vs_reference_accompaniment": round(float(np.corrcoef(
            side.mean(axis=1), ht_a.mean(axis=1))[0, 1]), 3),
        "vocal_survives_in_side_dB": round(leakage_db(side, ht_v), 1),
        "vocal_survives_in_mid_dB": round(leakage_db(mid, ht_v), 1),
    }

    # ---- 3. how much reference vocal survives in each karaoke output ----
    leak = {
        "center_cancel_side": leakage_db(side, ht_v),
        "htdemucs": leakage_db(ht_a, ht_v),
        "hdemucs_mmi": leakage_db(hd_a, hd_v),
        "umx": leakage_db(um_a, ht_v),
    }
    # and how close each vocal estimate is to the Demucs consensus
    voc_sim = {
        "center_cancel_side_vs_htdemucs": si_sdr(side, ht_v),
        "htdemucs_vs_hdemucs": si_sdr(ht_v, hd_v),
        "umx_vs_htdemucs": si_sdr(um_v, ht_v),
    }

    # ---- 4. per-band damage ----
    ref_b = [band_energy_db(mix, lo, hi) for lo, hi in BANDS]
    band = []
    for (lo, hi), r, nm in zip(BANDS, ref_b, BAND_NAMES):
        band.append({
            "band": nm,
            "mix_dB": round(r, 1),
            "free_side_delta": round(band_energy_db(side, lo, hi) - r, 1),
            "htdemucs_delta": round(band_energy_db(ht_a, lo, hi) - r, 1),
            "umx_delta": round(band_energy_db(um_a, lo, hi) - r, 1),
        })

    out = {
        "excerpt": "Le Youth - Chills 60-100s",
        "stereo": {
            "lr_corr_overall": round(overall, 4),
            "vocal_active_frames": int(active.sum()),
            "total_frames": int(len(active)),
            "lr_corr_vocal_active_mean": round(float(ca.mean()), 3),
            "lr_corr_vocal_active_median": round(float(np.median(ca)), 3),
            "lr_corr_vocal_inactive_mean": round(float(ci.mean()), 3),
            "lr_corr_vocal_inactive_median": round(float(np.median(ci)), 3),
        },
        "free_baseline_center_cancel": free,
        "rms_dB": {
            "mix": round(rms_db(mix), 2),
            "free_side_output": round(rms_db(side), 2),
            "htdemucs_instrumental": round(rms_db(ht_a), 2),
            "umx_instrumental": round(rms_db(um_a), 2),
        },
        "vocal_leak_into_instrumental_dB": {k: round(v, 1) for k, v in leak.items()},
        "vocal_estimate_agreement_dB": {k: round(v, 2) for k, v in voc_sim.items()},
        "per_band_delta_dB": band,
    }

    p = ROOT / "results/button_baseline.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    sf.write(str(ROOT / "results/listen/center_cancel_karaoke.mp3"), side, SR)
    sf.write(str(ROOT / "results/listen/center_cancel_mid.mp3"), mid, SR)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwritten: {p}")


if __name__ == "__main__":
    main()
