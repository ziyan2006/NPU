"""15_window_audit.py -- how much of the pseudo-label cache actually matters?

The v2 run picked each track's 30 s excerpt uniformly at random, and an A/B on
held-out tracks then showed that several excerpts held no usable vocals at all.
That is a data-quality defect, not a model defect, and it is invisible in the
loss curve -- so it needs its own measurement.

For every cached excerpt this reports the *band-domain* vocal-to-mixture energy
ratio

    r = 10 log10  sum(|mix_band * mask|^2) / sum(|mix_band|^2)

which the cache already stores (mix bands + soft mask), so no teacher re-run is
needed.  Near 0 dB the excerpt is mostly vocals; -30 dB means the excerpt is
teaching the model to output silence.

With two caches it also does the comparison that actually matters: **the same
tracks, paired**, old window vs screened window.  A per-track pairing removes
the "was this just a different song?" objection.

    python 15_window_audit.py results/v2_cache results/v3_cache
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from common import RESULTS


def load_cache(cache: Path) -> dict:
    """track filename -> measurements for one excerpt of that track."""
    out = {}
    for f in sorted(cache.glob("*.pt")):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        mix, mask = d["mix"].float(), d["mask"].float()
        num = float((mix * mask).pow(2).sum())
        den = float(mix.pow(2).sum()) + 1e-12
        ratio = 10.0 * torch.log10(torch.tensor(max(num, 1e-20) / den))
        out[d.get("src", f.name)] = {
            "band_db": float(ratio),
            "wave_db": d.get("vocal_db"),
            "offset_s": d.get("offset_s"),
            "index": int(f.stem) if f.stem.isdigit() else None,
        }
    return out


def summarise(cache: Path, items: dict) -> dict:
    vals = sorted(v["band_db"] for v in items.values())

    def q(p: float) -> float:
        return round(vals[min(len(vals) - 1, int(p * len(vals)))], 1)

    n = len(vals)
    meta = cache / "_meta.json"
    screening = None
    if meta.exists():
        try:
            screening = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "cache": str(cache),
        "entries": n,
        "band_ratio_dB": {"min": q(0.0), "p25": q(0.25), "median": q(0.50),
                          "p75": q(0.75), "max": q(1.0)},
        "frac_above_-6dB": round(sum(v > -6 for v in vals) / n, 3),
        "frac_above_-12dB": round(sum(v > -12 for v in vals) / n, 3),
        "frac_below_-20dB": round(sum(v < -20 for v in vals) / n, 3),
        "frac_below_-30dB": round(sum(v < -30 for v in vals) / n, 3),
        "wave_ratio_dB_median": (
            round(sorted(v["wave_db"] for v in items.values()
                         if v["wave_db"] is not None)[
                sum(1 for v in items.values() if v["wave_db"] is not None) // 2], 1)
            if any(v["wave_db"] is not None for v in items.values()) else None),
        "screening": screening,
        "quietest": [
            {"track": t[:52], "band_ratio_dB": round(v["band_db"], 1)}
            for t, v in sorted(items.items(), key=lambda kv: kv[1]["band_db"])[:6]],
    }


def print_summary(r: dict) -> None:
    b = r["band_ratio_dB"]
    print(f"\n{r['cache']}  ({r['entries']} excerpts)")
    print(f"  band-domain vocal-to-mixture ratio   min {b['min']:+.1f}  "
          f"p25 {b['p25']:+.1f}  median {b['median']:+.1f}  "
          f"p75 {b['p75']:+.1f}  max {b['max']:+.1f} dB")
    print(f"  > -6 dB {r['frac_above_-6dB']*100:5.1f}%   "
          f"> -12 dB {r['frac_above_-12dB']*100:5.1f}%   "
          f"< -20 dB {r['frac_below_-20dB']*100:5.1f}%   "
          f"< -30 dB {r['frac_below_-30dB']*100:5.1f}%")
    if r["screening"]:
        m = r["screening"]
        print(f"  screening: scan {m['scan_s']:.0f}s -> teacher once -> "
              f"best of {m['n_cand']} x {m['seg_s']:.0f}s windows")
    else:
        print("  screening: NONE (excerpt drawn at random)")
    if r["wave_ratio_dB_median"] is not None:
        print(f"  waveform-domain median: {r['wave_ratio_dB_median']:+.1f} dB")
    print("  quietest excerpts:")
    for e in r["quietest"]:
        print(f"    {e['band_ratio_dB']:>+7.1f} dB  {e['track']}")


def main() -> int:
    caches = [Path(a) for a in sys.argv[1:]] or [RESULTS / "v2_cache"]
    loaded = [(c, load_cache(c)) for c in caches]
    rows = [summarise(c, it) for c, it in loaded if it]
    for r in rows:
        print_summary(r)

    if len(loaded) == 2 and all(it for _, it in loaded):
        (ca, a), (cb, b) = loaded
        shared = sorted(set(a) & set(b))
        if shared:
            da = [b[t]["band_db"] - a[t]["band_db"] for t in shared]
            da_sorted = sorted(da)
            med = da_sorted[len(da_sorted) // 2]
            q1 = da_sorted[len(da_sorted) // 4]
            print(f"\n  PAIRED comparison on {len(shared)} identical tracks "
                  f"({ca.name} -> {cb.name})")
            print(f"    per-track gain in vocal-to-mixture ratio: "
                  f"p25 {q1:+.1f}  median {med:+.1f}  "
                  f"max {da_sorted[-1]:+.1f} dB")
            print(f"    improved: {sum(d > 1 for d in da)}/{len(da)}   "
                  f"worse: {sum(d < -1 for d in da)}/{len(da)}   "
                  f"within +/-1 dB: {sum(abs(d) <= 1 for d in da)}/{len(da)}")
            moved = sorted(((b[t]['band_db'] - a[t]['band_db'], t) for t in shared),
                           reverse=True)
            print("    biggest rescues (old window was empty, new one is not):")
            for d, t in moved[:5]:
                print(f"      {a[t]['band_db']:>+7.1f} -> {b[t]['band_db']:>+7.1f} "
                      f"({d:+6.1f} dB)  {t[:46]}")
            rows.append({
                "paired_with": ca.name, "n_paired": len(shared),
                "gain_dB": {"p25": round(q1, 1), "median": round(med, 1),
                            "max": round(da_sorted[-1], 1)},
                "improved": sum(d > 1 for d in da),
                "worse": sum(d < -1 for d in da),
                "within_1dB": sum(abs(d) <= 1 for d in da),
            })

    out = RESULTS / "window_audit.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
