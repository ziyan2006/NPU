"""18_loss_attribution.py -- where does the validation log-L1 actually come from?

The training loss is

    L = mean_bins | log(x*m_hat + eps) - log(x*t + eps) |     (vocal)
      + mean_bins | log(x*m_hat + eps) - log(x*t + eps) |     (accompaniment)

and it has not changed since v1.  v1 (15 min of audio, 6000 steps) and v4
(274 min, 88948 steps) both bottom out at a *training* loss of 0.71, which says
the network -- not the data -- is the binding constraint.  Before rewriting the
loss it is worth knowing which bins the 0.9x validation number is made of.

Two specific hypotheses are tested here, both cheap to falsify:

  1. the metric is dominated by near-silent bins, where the soft-mask target is
     a ratio of two near-zero numbers, i.e. noise, and eps=1e-3 turns any small
     output into a large log error.  If so, the number means nothing;
  2. the error lives in the lowest bands.  SI-SDR is energy-weighted, so bass
     leaking into the vocal estimate would dominate it -- and that is a
     frequency-resolution problem, not a "the model cannot sing" problem.  The
     128-band layout is uniform, so band 0 already spans several STFT bins of
     kick drum.

Outputs results/loss_attribution_<tag>.json and prints both tables.

    python 18_loss_attribution.py --ckpt v4_best.pt --cache-dir v4_cache
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common import RESULTS, SR                                 # noqa: E402


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t11 = _load("t11", "11_smoke_train.py")
EPS = t11.EPS

BINS = [(0.0, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 1e9)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v4_best.pt")
    ap.add_argument("--cache-dir", default="v4_cache")
    ap.add_argument("--tag", default=None, help="output label (default: ckpt stem)")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tag = args.tag or Path(args.ckpt).stem
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = RESULTS / args.cache_dir
    items = []
    for f in sorted(cache.glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        items.append((d["mix"].float(), d["mask"].float()))
    n_val = max(1, round(len(items) * 0.1))
    val_items = items[-n_val:]
    print(f"device {device}   cache {cache.name}: {len(items)} excerpts, "
          f"{len(val_items)} used for validation")

    blob = torch.load(RESULTS / args.ckpt, map_location="cpu", weights_only=False)
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
    print(f"checkpoint {args.ckpt}  step {blob.get('step')}  "
          f"best val {blob.get('best_val')}")

    # keep tensors 4-D (N, 2, 128, T): flattening first would interleave
    # channel -> band -> time and the per-band table would be garbage
    ev4, ea4, x4, tv4, ta4 = [], [], [], [], []
    g = torch.Generator().manual_seed(args.seed)
    lp = lambda z: (z + EPS).log()
    with torch.no_grad():
        for _ in range(args.batches):
            x, ym = t11.sample_batch(val_items, g, args.crop, args.batch)
            x, ym = x.to(device), ym.to(device)
            m_v, m_a = t11.forward_masks(net, x)
            ev4.append((lp(x * m_v) - lp(x * ym)).abs().cpu())
            ea4.append((lp(x * m_a) - lp(x * (1.0 - ym))).abs().cpu())
            x4.append(x.cpu())
            tv4.append(ym.cpu())
            ta4.append((1.0 - ym).cpu())
    ev4, ea4 = torch.cat(ev4), torch.cat(ea4)
    x4, tv4, ta4 = torch.cat(x4), torch.cat(tv4), torch.cat(ta4)
    ev, ea = ev4.flatten(), ea4.flatten()
    xv, tv = x4.flatten(), tv4.flatten()

    total = float((ev.mean() + ea.mean()))
    print(f"\ntotal val log-L1 = {total:.4f}  "
          f"(vocal term {float(ev.mean()):.4f}, accompaniment "
          f"{float(ea.mean()):.4f})")
    print(f"bins evaluated: {ev.numel():,}")

    # ---- hypothesis 1: is the metric silent-bin noise? ---------------------
    print("\nby mixture band magnitude (x) -- hypothesis 1:")
    print(f"  {'range':>16}{'bin share':>12}{'vocal':>10}{'accomp':>10}"
          f"{'loss share':>12}")
    rows = []
    tw = float(ev.sum() + ea.sum())
    for lo, hi in BINS:
        sel = (xv >= lo) & (xv < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        mv, ma = float(ev[sel].mean()), float(ea[sel].mean())
        share = float(ev[sel].sum() + ea[sel].sum()) / tw
        rows.append({"lo": lo, "hi": hi if hi < 1e8 else None,
                     "frac_bins": n / ev.numel(), "vocal": mv,
                     "accompaniment": ma, "loss_share": share})
        print(f"  {f'[{lo:g},{hi:g})':>16}{n / ev.numel():>12.1%}{mv:>10.3f}"
              f"{ma:>10.3f}{share:>12.1%}")
    for thr in (0.01, 0.1, 1.0):
        sel = xv < thr
        part = float(ev[sel].sum() + ea[sel].sum()) / tw
        print(f"  bins with x < {thr:<5g}: {float(sel.float().mean()):>5.1%} of bins "
              f"own {part:>5.1%} of the metric")

    print("\nby target vocal mask value -- hypothesis 1:")
    for lo, hi, name in ((0.0, 1e-6, "== 0 (no vocal)"), (1e-6, 0.5, "(0, 0.5]"),
                         (0.5, 2.0, "(0.5, 1]")):
        sel = (tv >= lo) & (tv < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        print(f"  {name:>16}  bins {n / ev.numel():>6.1%}   vocal err "
              f"{float(ev[sel].mean()):>6.3f}   loss share "
              f"{float(ev[sel].sum()) / float(ev.sum()):>6.1%}")

    silent = float(((lp(x4 * 0.0) - lp(x4 * tv4)).abs().mean()
                    + (lp(x4 * 0.0) - lp(x4 * ta4)).abs().mean()))
    print(f"\nconstant-silence predictor would score {silent:.4f} "
          f"(model: {total:.4f})")

    # ---- hypothesis 2: is the error in the low bands? ----------------------
    nb = int(ev4.shape[2])
    evb = ev4.mean(dim=(0, 1, 3))
    eab = ea4.mean(dim=(0, 1, 3))
    xb = x4.mean(dim=(0, 1, 3))
    tot_x = float(xb.sum())
    nyq = SR / 2
    print("\nby band index -- hypothesis 2 (128 uniform bands over "
          f"0-{nyq / 1000:.2f} kHz; {nb // 8} rows of 8 bands):")
    print(f"  {'bands':>10}{'top Hz':>9}{'Hz/band':>9}{'vocal':>9}{'accomp':>9}"
          f"{'x mean':>9}{'energy share':>13}")
    band_rows = []
    for b0 in range(0, nb, 8):
        sl = slice(b0, min(b0 + 8, nb))
        hi_hz = (b0 + 8) / nb * nyq
        es = float(xb[sl].sum()) / tot_x
        band_rows.append({"band0": b0, "top_hz": round(hi_hz, 1),
                          "hz_per_band": round(nyq / nb, 1),
                          "vocal": round(float(evb[sl].mean()), 4),
                          "accompaniment": round(float(eab[sl].mean()), 4),
                          "x_mean": round(float(xb[sl].mean()), 4),
                          "energy_share": round(es, 6)})
        print(f"  {f'{b0}-{b0 + 7}':>10}{hi_hz:>9.0f}{nyq / nb:>9.0f}"
              f"{float(evb[sl].mean()):>9.3f}{float(eab[sl].mean()):>9.3f}"
              f"{float(xb[sl].mean()):>9.3f}{es:>13.1%}")

    print("\n  energy share by octave:")
    for f_lo in (0, 100, 200, 400, 700, 1500, 3000, 6000, 11000):
        b_lo = int(f_lo / nyq * nb)
        b_hi = int(min(f_lo * 2, nyq) / nyq * nb)
        sl = slice(b_lo, max(b_hi, b_lo + 1))
        print(f"    {f_lo:>6}-{min(f_lo * 2, int(nyq)):>6} Hz  bands "
              f"{b_lo:>3}-{min(b_hi, nb) - 1:>3}  energy "
              f"{float(xb[sl].sum()) / tot_x:>6.1%}  vocal err "
              f"{float(evb[sl].mean()):>6.3f}")

    e_low = float(evb[:24].mean())
    e_high = float(evb[24:].mean())
    lo_share = float(xb[:24].sum()) / tot_x
    print(f"\n  vocal error below band 24 (~{24 / nb * nyq / 1000:.1f} kHz): "
          f"{e_low:.3f}   above: {e_high:.3f}")
    print(f"  those bands hold {lo_share:.1%} of the mixture energy")

    out = RESULTS / f"loss_attribution_{tag}.json"
    out.write_text(json.dumps(
        {"checkpoint": args.ckpt, "step": blob.get("step"),
         "total_val_logL1": total, "vocal_term": float(ev.mean()),
         "accompaniment_term": float(ea.mean()),
         "vocal_error_below_band24": e_low, "vocal_error_above_band24": e_high,
         "silence_baseline": silent, "by_mixture_magnitude": rows,
         "by_band": band_rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
