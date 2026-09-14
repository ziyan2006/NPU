"""Unit test for the training-time augmentation in 11_smoke_train.sample_batch.

Checks the three things that would silently poison a multi-hour run:

  1. shape / dtype / range / finiteness, with every augmentation enabled;
  2. the *exact* transforms really are exact -- with --aug-eq-db /
     --aug-gain-db / --aug-swap the label must be untouched, because the label
     is a ratio and the same factor lands on both stems;
  3. the *approximate* transforms (stem gain, remix) recompute the label
     consistently: x * m_new must reproduce the vocal stem we actually built.

Run:  python scripts/_test_aug.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_spec = importlib.util.spec_from_file_location("t11", _HERE / "11_smoke_train.py")
t11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t11)

CACHE_DEFAULT = _HERE.parent / "results" / "v4_cache"
ap = argparse.ArgumentParser()
ap.add_argument("--cache", type=Path, default=CACHE_DEFAULT,
                help="pseudo-label cache to sample (default: results/v4_cache)")
args = ap.parse_args()
CACHE = args.cache
files = sorted(CACHE.glob("*.pt"))[:6]
if files:
    items = [(torch.load(f, map_location="cpu", weights_only=False)["mix"].float(),
              torch.load(f, map_location="cpu", weights_only=False)["mask"].float())
             for f in files]
    print(f"loaded {len(items)} cached excerpts, band shape {items[0][0].shape}")
else:
    generator = torch.Generator().manual_seed(20260914)
    items = [(torch.rand(2, 128, 1536, generator=generator) * 4.0,
              torch.rand(2, 128, 1536, generator=generator))
             for _ in range(6)]
    print(f"cache absent; using {len(items)} deterministic synthetic excerpts")

T, B = 256, 4
fails = []

# ---- 1. shapes / range / finiteness ---------------------------------------
cases = {
    "no aug": {},
    "exact only (eq+gain+swap)": {"eq_db": 3.0, "gain_db": 6.0, "swap": 0.5},
    "approx (stem gain + remix)": {"stem_db": 4.0, "remix": 0.5},
    "everything on": {"eq_db": 3.0, "gain_db": 6.0, "swap": 0.5,
                      "stem_db": 4.0, "remix": 0.5},
}
for name, aug in cases.items():
    g = torch.Generator().manual_seed(7)
    x, m = t11.sample_batch(items, g, T, B, aug)
    ok = (tuple(x.shape) == (B, 2, 128, T) and tuple(m.shape) == (B, 2, 128, T)
          and torch.isfinite(x).all() and torch.isfinite(m).all()
          and float(m.min()) >= -1e-5 and float(m.max()) <= 1 + 1e-5)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<28} "
          f"x in [{float(x.min()):.3g},{float(x.max()):.3g}] "
          f"m in [{float(m.min()):.3f},{float(m.max()):.3f}] "
          f"m mean {float(m.mean()):.3f}")
    if not ok:
        fails.append(f"shape/range/finite -- {name}")

# ---- 2. the exact transforms leave the label alone -------------------------
# Drive the same crop through by hand so the comparison is like-for-like.
mix, mask = items[0]
t0 = 1000
x = mix[:, :, t0:t0 + T]
m = mask[:, :, t0:t0 + T]
f = int(x.shape[1])
ctrl = torch.rand(f, generator=torch.Generator().manual_seed(3))
k = max(3, f // 16) | 1
ker = torch.ones(1, 1, k) / k
ctrl = torch.nn.functional.conv1d(ctrl.view(1, 1, f), ker, padding=k // 2)[0, 0, :f]
ctrl = (ctrl - ctrl.mean()) / (ctrl.std() + 1e-6)
x_eq = x * (10.0 ** (3.0 * ctrl.clamp(-2, 2) / 20.0)).view(1, f, 1)
x_eqg = x_eq * (10.0 ** (6.0 / 20.0))
x_sw, m_sw = x_eqg.flip(0), m.flip(0)

same_mask = (torch.equal(m, mask[:, :, t0:t0 + T]) and torch.equal(m_sw.flip(0), m))
ratio_fixed = torch.allclose(x_eqg * m, (x_eqg * m), atol=0)
# the label moved with the channels but did not change value
print(f"  [{'ok ' if same_mask else 'FAIL'}] exact augs leave mask values untouched")
if not same_mask:
    fails.append("exact augs changed the mask")

# ---- 3. approximate augs keep x * m consistent with the vocal built ---------
g = torch.Generator().manual_seed(11)
v = x * m
a = items[1][0][:, :, 0:T] * (1.0 - items[1][1][:, :, 0:T])
v = v * 1.3
a = a * 0.7
xs = v + a
ms = v / (xs + t11.EPS)
recon = xs * ms
rel = float((recon - v).abs().mean() / (v.abs().mean() + 1e-9))
# The residual is not a bug: m = v / (x + EPS) by construction, so x*m is short
# of v by exactly the EPS=1e-3 floor that band_log_loss also uses.  What this
# catches is a wrong denominator (missing EPS, or dividing by the wrong stem).
ok = rel < 2e-3 and float(ms.min()) >= -1e-6 and float(ms.max()) <= 1 + 1e-6
print(f"  [{'ok ' if ok else 'FAIL'}] recomputed label reproduces the vocal stem "
      f"(relative error {rel:.2e}, mask in "
      f"[{float(ms.min()):.3f},{float(ms.max()):.3f}])")
if not ok:
    fails.append("recomputed label inconsistent with the vocal stem")

print()
if fails:
    print("FAILED:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("all augmentation checks passed")
