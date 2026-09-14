"""19_filterbank_audit.py -- how much does the 128-band *layout* cost?

Why this exists
---------------
18_loss_attribution.py found 79 of the 128 non-zero bands in the cached
representation -- the same 49 dead columns as make_analysis_matrix() -- and
band 0..6 entirely zero.  The cause is a units mismatch:

  * band_edges() = np.geomspace(30, 16000, 129) -- 128 *logarithmic* bands,
    so the lowest band is 30-31.5 Hz, i.e. 1.5 Hz wide;
  * the STFT grid is np.linspace(0, 22050, 513) -- one bin every 43.07 Hz.

A triangular band narrower than the bin spacing contains no bin at all, so its
column in the analysis matrix is exactly zero.  Below ~500 Hz almost every
logarithmic band is empty, and the bands that are not empty contain exactly one
bin: the representation samples the bass at one point per band instead of
averaging a region, and the synthesis matrix then smears that single sample
over its whole width.

This also explains an oddity in the band scan of report 03: 96 bands scored
*better* than 128 (13.39 dB vs 12.82 dB).  A count of bands and the quality of
the layout were being varied together -- adding logarithmic bands down there
adds empty channels and splits real content.

The audit re-runs the oracle-mask path from 12_smoke_diag.py (projection mask
-> band-compress -> upsample -> iSTFT -> SI-SDR against the teacher) for several
layouts on identical audio, so the layouts are compared on their own merits
rather than through a trained model.  Layout cost is a ceiling: no amount of
training removes it.

Run:  python 19_filterbank_audit.py --tracks 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common import RESULTS, SR, load_demucs                       # noqa: E402


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t09 = _load("t09", "09_target_model.py")
t11 = _load("t11", "11_smoke_train.py")

N_BINS = t09.N_BINS


# ---------------------------------------------------------------------------
# band matrices for an arbitrary edge set
# ---------------------------------------------------------------------------
def analysis_matrix(edges: np.ndarray) -> np.ndarray:
    """(513, B) triangular weights, each column normalised to 1.

    Same construction as t09.make_analysis_matrix, with one guard: the
    geometric centre of a band whose lower edge is 0 Hz is 0, which the original
    treats as degenerate and leaves the column empty.  Here it falls back to the
    arithmetic centre, so a layout that starts at DC does not silently lose its
    first band.
    """
    freqs = np.linspace(0.0, SR / 2, N_BINS)
    nb = len(edges) - 1
    W = np.zeros((N_BINS, nb), dtype=np.float64)
    for k in range(nb):
        lo, hi = float(edges[k]), float(edges[k + 1])
        c = math.sqrt(lo * hi)
        if c <= lo or c >= hi:
            c = 0.5 * (lo + hi)
        if c <= lo or c >= hi:
            continue
        left = (freqs >= lo) & (freqs < c)
        right = (freqs >= c) & (freqs <= hi)
        W[left, k] = (freqs[left] - lo) / (c - lo)
        W[right, k] = (hi - freqs[right]) / (hi - c)
    col = W.sum(axis=0, keepdims=True)
    col[col == 0] = 1.0
    return (W / col).astype(np.float32)


def synthesis_matrix(edges: np.ndarray) -> np.ndarray:
    """(513, B) linear interpolation from band centres back to bins."""
    centers = np.sqrt(edges[:-1] * edges[1:])
    centers[0] = max(centers[0], 1e-6)
    freqs = np.linspace(0.0, SR / 2, N_BINS)
    nb = len(edges) - 1
    G = np.zeros((N_BINS, nb), dtype=np.float64)
    for j, f in enumerate(freqs):
        if f <= centers[0]:
            G[j, 0] = 1.0
        elif f >= centers[-1]:
            G[j, -1] = 1.0
        else:
            k = int(np.searchsorted(centers, f) - 1)
            k = min(max(k, 0), nb - 2)
            w = (f - centers[k]) / (centers[k + 1] - centers[k])
            G[j, k] = 1.0 - w
            G[j, k + 1] = w
    return G.astype(np.float32)


def snap_to_bins(hz: np.ndarray) -> np.ndarray:
    """Snap band boundaries to STFT bin indices, one band >= 1 bin.

    128 bands over 513 bins averages 4 bins per band, so this is not a
    compromise: it is the only way to make every band contain data while
    keeping a fine grid where the ear (and the vocal) needs it.
    """
    frac = hz / (SR / 2) * (N_BINS - 1)
    idx = np.round(frac).astype(int)
    idx = np.clip(idx, 0, N_BINS - 1)
    for k in range(1, len(idx)):
        idx[k] = max(idx[k], idx[k - 1] + 1)
    if idx[-1] > N_BINS - 1:
        idx[-1] = N_BINS - 1
        for k in range(len(idx) - 2, -1, -1):
            idx[k] = min(idx[k], idx[k + 1] - 1)
    return idx * (SR / 2) / (N_BINS - 1)


def snap_centers_to_unique_bins(hz: np.ndarray) -> np.ndarray:
    """Move requested band centres onto distinct FFT bins.

    Snapping *edges* can leave a one-bin band with no sample at its triangular
    peak.  Snapping centres instead guarantees that every band owns at least
    its centre bin, while retaining the requested mel/log density as closely
    as the 43 Hz FFT grid permits.
    """
    idx = np.rint(hz / (SR / 2) * (N_BINS - 1)).astype(int)
    idx = np.clip(idx, 0, N_BINS - 1)
    for k in range(1, len(idx)):
        idx[k] = max(idx[k], idx[k - 1] + 1)
    if idx[-1] > N_BINS - 1:
        idx[-1] = N_BINS - 1
        for k in range(len(idx) - 2, -1, -1):
            idx[k] = min(idx[k], idx[k + 1] - 1)
    if np.any(np.diff(idx) <= 0):
        raise ValueError("not enough FFT bins for unique band centres")
    return idx * (SR / 2) / (N_BINS - 1)


def matrices_from_centers(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analysis average and synthesis interpolation for explicit centres."""
    freqs = np.linspace(0.0, SR / 2, N_BINS)
    nb = len(centers)
    G = np.zeros((N_BINS, nb), dtype=np.float64)
    for j, f in enumerate(freqs):
        if f <= centers[0]:
            G[j, 0] = 1.0
        elif f >= centers[-1]:
            G[j, -1] = 1.0
        else:
            k = int(np.searchsorted(centers, f) - 1)
            k = min(max(k, 0), nb - 2)
            w = (f - centers[k]) / (centers[k + 1] - centers[k])
            G[j, k] = 1.0 - w
            G[j, k + 1] = w
    col = G.sum(axis=0, keepdims=True)
    if np.any(col == 0):
        raise AssertionError("a snapped centre produced an empty band")
    return (G / col).astype(np.float32), G.astype(np.float32)


def mel_points(n: int, fmin: float, fmax: float) -> np.ndarray:
    m2f = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)      # noqa: E731
    f2m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)         # noqa: E731
    return m2f(np.linspace(f2m(fmin), f2m(fmax), n))


def designs() -> dict[str, np.ndarray]:
    """Candidate layouts, all with exactly 128 bands unless named otherwise."""
    d = {}
    # the layout the cache was built with, and the historical scan points
    for n in (32, 64, 96, 128, 192, 256):
        d[f"log{n} (current)" if n == 128 else f"log{n} (historic)"] = \
            np.geomspace(t09.FMIN, t09.FMAX, n + 1)
    d["uniform128 0-22k"] = np.linspace(0.0, SR / 2, 129)
    d["uniform128 0-16k"] = np.linspace(0.0, t09.FMAX, 129)
    d["mel128 snapped"] = snap_to_bins(mel_points(129, 0.0, SR / 2))
    d["mel128 snapped 30-16k"] = snap_to_bins(mel_points(129, t09.FMIN,
                                                         t09.FMAX))
    d["log128 snapped 30-16k"] = snap_to_bins(
        np.geomspace(t09.FMIN, t09.FMAX, 129))
    d["log128 snapped 30-22k"] = snap_to_bins(
        np.geomspace(t09.FMIN, SR / 2, 129))
    return d


def centre_designs() -> dict[str, np.ndarray]:
    return {
        "mel128 unique centres 0-16k": snap_centers_to_unique_bins(
            mel_points(128, 0.0, t09.FMAX)),
        "mel128 unique centres 30-16k": snap_centers_to_unique_bins(
            mel_points(128, t09.FMIN, t09.FMAX)),
        "log128 unique centres 30-16k": snap_centers_to_unique_bins(
            np.geomspace(t09.FMIN, t09.FMAX, 128)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=8)
    ap.add_argument("--seg", type=float, default=20.0)
    ap.add_argument("--screen-s", type=float, default=0.0,
                    help="if >0, decode this much and keep the best --seg "
                         "sub-window by vocal/mixture energy, the same "
                         "screening 11_smoke_train applies to the training "
                         "pool.  Without it a track's fixed 60 s offset is "
                         "usually instrumental in house/EDM and every ceiling "
                         "below is measured against a vocal that is not there.")
    ap.add_argument("--n-cand", type=int, default=5)
    ap.add_argument("--pool", default="holdout", choices=("holdout", "train"))
    ap.add_argument("--out", default="filterbank_audit.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = json.loads((RESULTS / "v4_report.json").read_text(encoding="utf-8"))
    names = rep["holdout_tracks"][:args.tracks]
    print(f"device {device}   {len(names)} held-out tracks x {args.seg:g} s"
          + (f"  (screened from {args.screen_s:g} s)" if args.screen_s else ""))

    ds = designs()
    mats = {}
    for name, edges in ds.items():
        Wa = analysis_matrix(edges)
        cols = (Wa > 0).sum(axis=0)
        mats[name] = (edges, Wa, synthesis_matrix(edges), int((cols == 0).sum()))
    for name, centers in centre_designs().items():
        Wa, G = matrices_from_centers(centers)
        cols = (Wa > 0).sum(axis=0)
        mats[name] = (centers, Wa, G, int((cols == 0).sum()))
    print("\nlayout                    bands  dead   bins/band (min/med/max)"
          "   Hz/band at 100 Hz / 1 kHz")
    for name, (points, Wa, G, dead) in mats.items():
        cols = (Wa > 0).sum(axis=0)
        w = np.diff(points)
        f100 = np.interp(100.0, points[:-1], w)
        f1k = np.interp(1000.0, points[:-1], w)
        print(f"  {name:<24}{len(cols):>5}{dead:>6}   "
              f"{cols.min():>3}/{int(np.median(cols)):>3}/{cols.max():>3}"
              f"      {f100:>7.1f} / {f1k:>7.1f}")

    teacher = load_demucs("htdemucs").to(device).eval()
    xmat = {n: torch.from_numpy(m[1]) for n, m in mats.items()}
    gmat = {n: torch.from_numpy(m[2]) for n, m in mats.items()}

    rows = []
    for i, nm in enumerate(names):
        path = None
        for p in t11.LIB.rglob("*"):
            if p.name == nm:
                path = p
                break
        if path is None:
            print(f"  [{i}] {nm}: not found, skipped")
            continue
        mix = t11.decode_excerpt(path, 60.0, args.screen_s or args.seg)
        voc = t11.teacher_vocals(teacher, mix, device)
        n = min(mix.shape[-1], voc.shape[-1])
        mix, voc = mix[:, :n], voc[:, :n]
        if args.screen_s:
            W = int(args.seg * SR)
            offs = np.unique(np.linspace(0, max(n - W, 0),
                                         max(1, args.n_cand)).astype(int))
            sc = [t11.window_vocal_db(voc[:, o:o + W], mix[:, o:o + W])
                  for o in offs]
            k = int(np.argmax(sc))
            o = int(offs[k])
            print(f"      screening: best of {len(offs)} windows at {o / SR:.1f} s, "
                  f"vocal/mixture {sc[k]:+.1f} dB")
            mix, voc = mix[:, o:o + W], voc[:, o:o + W]
            n = mix.shape[-1]
        X = t09._stft_dev(mix) if hasattr(t09, "_stft_dev") else t11._stft_dev(mix)
        V = t09._stft_dev(voc) if hasattr(t09, "_stft_dev") else t11._stft_dev(voc)
        Xa = X.abs()
        # the projection mask 12_smoke_diag uses, and the soft mask we train on
        proj = ((V * X.conj()).real / (Xa ** 2 + 1e-6)).clamp(0, 1)
        soft = (V.abs() / (V.abs() + (X - V).abs() + 1e-9)).clamp(0, 1)
        raw = t09.si_sdr((mix - voc).numpy(), voc.numpy())
        row = {"track": nm, "si_sdr_mixture_vs_teacher_vocals": round(float(raw), 2),
               "layouts": {}}
        for name, (points, Wa, G, dead) in mats.items():
            Wa_ = xmat[name]
            G_ = gmat[name]
            out = {}
            for lab, m in (("proj", proj), ("soft", soft)):
                mb = torch.einsum("fb,cft->cbt", Wa_, m)
                mu = torch.einsum("fb,cbt->cft", G_, mb).clamp(0, 1)
                y = t11._istft_dev(X * mu, n)
                out[lab] = round(float(t09.si_sdr(y.numpy(), voc.numpy())), 3)
            row["layouts"][name] = out
        rows.append(row)
        print(f"  [{i}] {nm[:44]:<44} mix~T {raw:6.2f} dB   "
              + "  ".join(f"{k.split(' ')[0]}={v['proj']:.2f}"
                          for k, v in row["layouts"].items()))

    print("\nmean SI-SDR vs the teacher's vocal (dB), higher is better:")
    print(f"  {'layout':<26}{'proj mask':>11}{'soft mask':>11}{'dead':>6}")
    summary = {}
    for name in mats:
        p = float(np.mean([r["layouts"][name]["proj"] for r in rows]))
        s = float(np.mean([r["layouts"][name]["soft"] for r in rows]))
        summary[name] = {"proj": round(p, 3), "soft": round(s, 3),
                         "dead_bands": mats[name][3]}
        print(f"  {name:<26}{p:>11.2f}{s:>11.2f}{mats[name][3]:>6}")

    out = RESULTS / args.out
    out.write_text(json.dumps({"n_tracks": len(rows), "seg_s": args.seg,
                               "summary": summary, "per_track": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
