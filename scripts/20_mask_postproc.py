"""20_mask_postproc.py -- low-cost mask post-processing knobs, measured.

Where this comes from
---------------------
18_loss_attribution.py found the model's worst mask error (1.799 in log units) in
the *loudest* bands, and 19_filterbank_audit.py found why the loudest bands are
also the least trustworthy:

    the geometric band grid has 49 dead columns out of 128, all of them below
    roughly 500 Hz.  The network therefore sees exactly zero in those 49 input
    channels for every frame of every song.

That matters more than "some channels are wasted", because the *synthesis*
matrix still routes low-frequency bins through those bands' centres.  The mask
applied below ~500 Hz is therefore the network's response to a zero input -- a
learned per-band constant with no dependence on the song.  A constant mask in the
bass region does two bad things at once:

  * it lets bass energy into the vocal estimate, which the residual path then
    removes from what the user hears;
  * and it removes whatever that constant says to remove, whether or not there
    is a vocal there.

So the post-processing below is not a cosmetic tweak:

  1. ``--temp`` rescales the last-layer output, ``m = (a * o + 1) / 2``.  Because
     the final layer is a plain convolution, this folds into its weights exactly
     -- **zero extra hardware**, not one multiplier.
  2. ``--lf-kill-hz`` forces the vocal mask to zero below a frequency.  On the PL
     that is a band-index comparator: **zero extra hardware** as well.

Both are evaluated with the teacher loaded once, on the pinned 24-track holdout,
with the same scorable gate the A/B harness uses.

Run:  python 20_mask_postproc.py --ckpt v4_best.pt
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
import torch.nn.functional as F

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

def band_top_hz(k: int, layout: str = t09.DEFAULT_BAND_LAYOUT,
                n_bands: int = t09.N_BANDS) -> float:
    if layout == t09.DEFAULT_BAND_LAYOUT:
        return float(t09.band_edges(n_bands)[k + 1])
    centers = t09.band_centers(n_bands, layout)
    return float(centers[min(max(k, 0), len(centers) - 1)])


def hz_to_band(hz: float, layout: str = t09.DEFAULT_BAND_LAYOUT,
               n_bands: int = t09.N_BANDS) -> int:
    """Highest band index whose top edge is <= hz (0 disables the knob)."""
    return t11.lf_kill_band_for(hz, layout, n_bands)


def _db(x: torch.Tensor) -> float:
    return float(20.0 * torch.log10(x.pow(2).mean().sqrt() + 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v4_best.pt")
    ap.add_argument("--seg", type=float, default=30.0)
    ap.add_argument("--tracks", type=int, default=24)
    ap.add_argument("--temps", default="0.6,0.8,1.0,1.25,1.5,2.0,3.0")
    ap.add_argument("--lf-hz", default="0,80,150,250,400,700")
    ap.add_argument("--smooth-frames", default="1,2,4,8",
                    help="causal trailing-average lengths to test")
    ap.add_argument("--smooth-lf-hz", type=float, default=250.0,
                    help="low-frequency protection combined with smoothing")
    ap.add_argument("--mask-gains", default="1,1.125,1.25,1.5",
                    help="vocal-mask gains combined with --smooth-lf-hz")
    ap.add_argument("--out", default="mask_postproc.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = json.loads((RESULTS / "v4_report.json").read_text(encoding="utf-8"))
    names = rep["holdout_tracks"][:args.tracks]
    temps = [float(x) for x in args.temps.split(",") if x.strip()]
    lfs = [float(x) for x in args.lf_hz.split(",") if x.strip()]
    smooths = [int(x) for x in args.smooth_frames.split(",") if x.strip()]
    gains = [float(x) for x in args.mask_gains.split(",") if x.strip()]
    print(f"device {device}  checkpoint {args.ckpt}  {len(names)} tracks "
          f"x {args.seg:g} s")

    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = RESULTS / ckpt
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = t11.t09.CausalSpectralUNet(
        2, (32, 64, 96, 128), 4,
        bottleneck_blocks=int(blob.get("bottleneck_blocks", 0)),
        temporal_dilations=tuple(blob.get("temporal_dilations", ()))).to(device)
    net.load_state_dict(blob["model"])
    net.frontend = tuple(blob.get("frontend") or ("linear", 1.0))
    net.mask_mode = blob.get("mask_mode", "independent")
    net.band_layout = blob.get("band_layout", t11.t09.DEFAULT_BAND_LAYOUT)
    net.n_bands = int(blob.get("n_bands", t11.t09.N_BANDS))
    net.eval()
    print(f"  step {blob.get('step')}  best val {blob.get('best_val')}")

    teacher = load_demucs("htdemucs").to(device).eval()
    layout = t11.band_layout_for(net)
    n_bands = t11.n_bands_for(net)
    Wa = torch.from_numpy(t09.make_analysis_matrix(n_bands, layout=layout)).to(device)
    Gs = torch.from_numpy(t09.make_synthesis_matrix(n_bands, layout=layout)).to(device)

    cfgs = [("baseline a=1 lf=0", 1.0, 0, 1, 1.0)]
    cfgs += [(f"temp a={t:g}", t, 0, 1, 1.0)
             for t in temps if abs(t - 1.0) > 1e-9]
    cfgs += [(f"lf<{f:g}Hz", 1.0, hz_to_band(f, layout, n_bands), 1, 1.0)
             for f in lfs if f > 0]
    cfgs += [(f"lf<{args.smooth_lf_hz:g}Hz+smooth{s}", 1.0,
              hz_to_band(args.smooth_lf_hz, layout, n_bands), s, 1.0)
             for s in smooths if s > 1]
    cfgs += [(f"lf<{args.smooth_lf_hz:g}Hz+gain{g:g}", 1.0,
              hz_to_band(args.smooth_lf_hz, layout, n_bands), 1, g)
             for g in gains if abs(g - 1.0) > 1e-9]

    recs = []
    low_band_share = []
    for i, nm in enumerate(names):
        p = None
        for cand in t11.LIB.rglob("*"):
            if cand.name == nm:
                p = cand
                break
        if p is None:
            print(f"  [{i}] {nm[:40]}: not found")
            continue
        total = t11.ffprobe_duration(p)
        start = 0.5 * max(0.0, total - args.seg)
        mix = t11.decode_excerpt(p, start, args.seg)
        ref = t11.teacher_vocals(teacher, mix, device)
        n = min(mix.shape[-1], ref.shape[-1])
        mix, ref = mix[:, :n], ref[:, :n]

        X = t11._stft_dev(mix.to(device))
        bands = torch.einsum("fb,cft->cbt", Wa, X.abs())
        T = bands.shape[-1]
        pad = (-T) % 8
        if pad:
            bands = F.pad(bands, (0, pad))
        with torch.no_grad():
            out = net(t11.apply_frontend(bands[None],
                                         t11.frontend_spec(net)))[0]
        if pad:
            out = out[..., :T]
        o = out[0:2].float()

        floor = float(t09.si_sdr(mix.numpy(), ref.numpy()))
        v_db = _db(ref)
        scorable = bool(v_db > -40.0 and floor > -8.0)
        acc_ref = (mix - ref)[..., :n]

        # how much of the vocal mask the dead bands actually decide
        with torch.no_grad():
            m_base = ((o + 1.0) * 0.5).clamp(0, 1)
            low_band_share.append(float(
                m_base[:, :hz_to_band(700, layout, n_bands)].mean()))

        rec = {"track": nm, "scorable": scorable, "floor_dB": round(floor, 2)}
        for label, a, lf, smooth, gain in cfgs:
            with torch.no_grad():
                m = ((a * o + 1.0) * 0.5).clamp(0.0, 1.0)
                m = t11.causal_smooth_mask(m, smooth)
                m = (m * gain).clamp(0.0, 1.0)
                if lf:
                    m = m.clone()
                    m[:, :lf] = 0.0
                mask_bin = torch.einsum("fb,cbt->cft", Gs, m).clamp(0.0, 1.0)
                voc = t11._istft_dev(X * mask_bin, n).cpu()
            acc = mix - voc
            rec[label] = {
                "voc": round(float(t09.si_sdr(voc.numpy(), ref.numpy())), 3),
                "acc": round(float(t09.si_sdr(acc.numpy(), acc_ref.numpy())), 3),
            }
        recs.append(rec)
        print(f"  [{i:>2}] {nm[:36]:<36} floor {floor:6.2f} "
              f"{'score' if scorable else '  ---'}  "
              f"base voc {rec['baseline a=1 lf=0']['voc']:6.2f} "
              f"acc {rec['baseline a=1 lf=0']['acc']:6.2f}", flush=True)

    def med(key, field, only_scorable=True):
        vals = [r[key][field] for r in recs
                if (r["scorable"] or not only_scorable) and key in r]
        return float(np.median(vals)) if vals else float("nan")

    print(f"\nmedian SI-SDR vs teacher over the {len(recs)} excerpts "
          f"({sum(r['scorable'] for r in recs)} scorable):")
    print(f"  {'config':<22}{'vocal(all)':>12}{'vocal(scor)':>13}"
          f"{'accomp(all)':>13}{'accomp(scor)':>14}"
          f"{'wins voc/acc':>14}")
    summary = {}
    base_key = "baseline a=1 lf=0"
    for label, a, lf, smooth, gain in cfgs:
        wins_v = sum(1 for r in recs if r["scorable"] and label in r
                     and r[label]["voc"] > r[base_key]["voc"])
        wins_a = sum(1 for r in recs if r["scorable"] and label in r
                     and r[label]["acc"] > r[base_key]["acc"])
        n_sc = sum(r["scorable"] for r in recs)
        row = {"temp": a, "lf_kill_band": lf, "smooth_frames": smooth,
               "vocal_mask_gain": gain,
               "lf_kill_hz": (round(band_top_hz(lf - 1, layout, n_bands), 1)
                              if lf else 0.0),
               "vocal_median_all": round(med(label, "voc", False), 3),
               "vocal_median_scorable": round(med(label, "voc"), 3),
               "accomp_median_all": round(med(label, "acc", False), 3),
               "accomp_median_scorable": round(med(label, "acc"), 3),
               "wins_vocal": wins_v, "wins_accomp": wins_a, "n_scorable": n_sc}
        summary[label] = row
        print(f"  {label:<22}{row['vocal_median_all']:>12.3f}"
              f"{row['vocal_median_scorable']:>13.3f}"
              f"{row['accomp_median_all']:>13.3f}"
              f"{row['accomp_median_scorable']:>14.3f}"
              f"{f'{wins_v}/{wins_a} of {n_sc}':>14}")

    print(f"\nmean vocal mask on the bands below ~700 Hz (dead input channels): "
          f"{np.mean(low_band_share):.4f}")
    base = summary["baseline a=1 lf=0"]
    best_v = max(summary.items(), key=lambda kv: kv[1]["vocal_median_scorable"])
    best_a = max(summary.items(), key=lambda kv: kv[1]["accomp_median_scorable"])
    print(f"  best vocal(scorable): {best_v[0]} "
          f"({best_v[1]['vocal_median_scorable']:.3f} vs "
          f"{base['vocal_median_scorable']:.3f})")
    print(f"  best accomp(scorable): {best_a[0]} "
          f"({best_a[1]['accomp_median_scorable']:.3f} vs "
          f"{base['accomp_median_scorable']:.3f})")

    out = RESULTS / args.out
    out.write_text(json.dumps(
        {"checkpoint": args.ckpt, "seg_s": args.seg, "n": len(recs),
         "mean_vocal_mask_below_700Hz": round(float(np.mean(low_band_share)), 5),
         "summary": summary, "per_track": recs},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
