"""17_quantize_check.py -- will the trained student survive 8-bit arithmetic?

The FPGA spec has said "weights INT8 / activations INT8 / accumulator INT32"
since day one, but that was a *plan*, never a measurement.  Everything the
architecture was credited with -- 517 KB of weights, 1.226 GMAC/audio-second,
RTF 0.056 -- assumes the network keeps working once every weight and every
activation is rounded to 8 bits.  If it does not, the fix is a different
network (or a different input representation), and that fix has to happen
*before* any RTL is written.

What this measures
------------------
For each variant of the arithmetic, the full chain
    STFT -> 128 bands -> U-Net -> mask -> iSTFT
is run on held-out audio and compared against the float chain:

    float           fp32 reference (what the listening tests used)
    mask8           float net, mask quantised to 8 bits over [0, 1]
    w8              INT8 weights (per output channel), float activations
    w8_tensor       INT8 weights with a single per-tensor scale, for contrast
    a8              float weights, INT8 activations (per-tensor, max-calibrated)
    a8_p999         same, but calibrated on the 99.9th percentile (clipping)
    w8a8            INT8 weights + INT8 activations
    deploy          w8a8 + 8-bit mask: the actual PL datapath

The headline number is SI-SDR(variant, float).  Anything above ~30 dB is
inaudible; 20 dB is a rounding error; below 10 dB the datapath -- not the
model -- is the bottleneck.

Two things the FPGA team actually needs out of this
--------------------------------------------------
1. **Per-layer activation scales.**  These are the numbers the RTL multiplies
   by, and the layer with a huge crest factor is the one that needs either a
   wider scale or a small-norm workaround.
2. **Input headroom.**  The network's input is a *linear* magnitude spectrum,
   which spans many decades.  With one scale for all of it, quiet bins round
   to zero.  The script reports what fraction of bins and what fraction of
   energy survive, which decides whether the front end needs a log stage --
   a training-time decision, so it has to be settled now.

Usage
-----
    python 17_quantize_check.py --ckpt ../results/v4_model.pt \
        --calib-cache ../results/v4_cache --tracks 6 --device cpu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 13_ab_compare.py already loads 09 and 11 by absolute path.  Re-loading them
# here would produce a *second* copy of the module objects, and therefore a
# second, distinct CausalConv2d class -- `isinstance` would then match nothing
# and every quantiser would silently no-op.  Borrow 13's copies instead.
t13 = _load("t13", "13_ab_compare.py")
t09 = t13.t09
t11 = t13.t11

RESULTS = t11.RESULTS
SR = t11.SR
QMAX = 127                       # symmetric INT8
MASK_LEVELS = 255                # (tanh + 1) / 2 stored as UINT8


# ---------------------------------------------------------------------------
# fake quantisation
# ---------------------------------------------------------------------------
def q_sym(x: torch.Tensor, step, qmax: int = QMAX) -> torch.Tensor:
    """Symmetric round-to-nearest quantiser; `step` is the LSB size."""
    return torch.clamp(torch.round(x / step), -qmax, qmax) * step


def qmax_for(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def conv_scales(net: nn.Module, per_channel: bool = True) -> dict[str, torch.Tensor]:
    """Weight step for every conv, from the real trained weights."""
    out = {}
    for name, mod in net.named_modules():
        if not isinstance(mod, t09.CausalConv2d):
            continue
        w = mod.conv.weight.detach()
        flat = w.reshape(w.shape[0], -1)
        amax = flat.abs().amax(dim=1) if per_channel else flat.abs().amax()
        out[name] = (amax / QMAX).clamp_min(1e-12)
    return out


def quantize_weights_(net: nn.Module, per_channel: bool = True,
                      bits: int = 8) -> dict:
    """Round every conv weight in place; returns per-conv statistics."""
    qm = qmax_for(bits)
    stats = {}
    for name, mod in net.named_modules():
        if not isinstance(mod, t09.CausalConv2d):
            continue
        w = mod.conv.weight.data
        flat = w.reshape(w.shape[0], -1)
        amax = flat.abs().amax(dim=1) if per_channel else flat.abs().amax()
        step = (amax / qm).clamp_min(1e-12)
        q = q_sym(flat, step[:, None] if per_channel else step, qm)
        mod.conv.weight.data = q.reshape_as(w)
        err = (q - flat)
        stats[name] = {
            "w_absmax": float(flat.abs().max()),
            "step": float(step.max() if per_channel else step),
            "rel_rmse": float((err.pow(2).mean().sqrt()
                               / (flat.pow(2).mean().sqrt() + 1e-12))),
            "taps_per_output": int(flat.shape[1]),
            "acc_bits": int(math.ceil(math.log2(
                2 * flat.shape[1] * qm * (2 ** (bits - 1) - 1) + 1))),
        }
    return stats


class ActQuant:
    """Calibrates and then applies INT8 to every conv input.

    Inserted as a forward *pre* hook on each CausalConv2d, i.e. exactly the
    tensor the RTL would present to the MAC array.  CausalConv2d pads after the
    hook, and zeros pad to zeros, so the hook position is faithful.

    ``per_channel=True`` gives every *input channel* its own scale.  That is
    nearly free in hardware -- it is a post-scale on the previous layer's
    accumulator -- and it is the standard cure for a tensor whose channels
    differ wildly in level.
    """

    def __init__(self, net: nn.Module, mode: str = "max", pct: float = 99.9,
                 per_channel: bool = False, sample_cap: int = 1 << 20,
                 skip: tuple = (), bits: int = 8):
        self.mode, self.pct = mode, pct
        self.per_channel, self.sample_cap = per_channel, sample_cap
        self.skip = set(skip)
        self.bits = bits
        self.qmax = qmax_for(bits)
        self.spans: dict[str, torch.Tensor] = {}
        self.steps: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._install_probe(net)

    def _reduce(self, a: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) absolute values -> span vector (1) or (1, C, 1, 1)."""
        if self.per_channel:
            if a.dim() != 4:
                return a.max().reshape(1)
            return a.amax(dim=(0, 2, 3)).reshape(1, -1, 1, 1)
        if self.mode == "max":
            return a.max().reshape(1)
        flat = a.flatten()
        if flat.numel() > self.sample_cap:
            flat = flat[:: max(1, flat.numel() // self.sample_cap)]
        return torch.quantile(flat, self.pct / 100.0).reshape(1)

    # -- phase 1: observe ------------------------------------------------
    def _install_probe(self, net: nn.Module):
        def mk(name):
            def hook(_m, inp):
                a = inp[0].detach().float().abs()
                v = self._reduce(a)
                old = self.spans.get(name)
                self.spans[name] = v if old is None else torch.maximum(old, v)
                return None
            return hook
        for name, mod in net.named_modules():
            if isinstance(mod, t09.CausalConv2d):
                self._handles.append(mod.register_forward_pre_hook(mk(name)))

    def probe(self, net: nn.Module, batches, pre=None) -> None:
        """`pre` is the model's own input front end, if it has one: the RTL sees
        the front end's output, so that is what has to be calibrated."""
        net.eval()
        dev = next(net.parameters()).device
        with torch.no_grad():
            for x in batches:
                x = x.to(dev)          # batches come off the cache on the CPU
                net(pre(x) if pre is not None else x)

    # -- phase 2: apply --------------------------------------------------
    def install(self, net: nn.Module) -> None:
        self.remove(net)
        self.steps = {k: v.clamp_min(1e-12) / self.qmax
                      for k, v in self.spans.items()}
        qm = self.qmax

        def mk(name):
            step = self.steps[name]

            def hook(_m, inp):
                return (q_sym(inp[0], step, qm),)

            return hook

        for name, mod in net.named_modules():
            if isinstance(mod, t09.CausalConv2d) and name not in self.skip:
                self._handles.append(mod.register_forward_pre_hook(mk(name)))

    def remove(self, net: nn.Module) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def peak(self, name: str) -> float:
        return float(self.spans[name].max()) if name in self.spans else 0.0

    def mean_span(self, name: str) -> float:
        return float(self.spans[name].mean()) if name in self.spans else 0.0


# ---------------------------------------------------------------------------
# the deployment chain
# ---------------------------------------------------------------------------
@torch.no_grad()
def chain(net, mix: torch.Tensor, Wa, Gs, device: str, mask_levels: int = 0):
    """separate() with an optional uniform quantiser on the final mask."""
    n = mix.shape[-1]
    X = t09._stft(mix.to(device))
    bands = torch.einsum("fb,cft->cbt", Wa.to(device), X.abs())
    T = bands.shape[-1]
    pad = (-T) % 8
    if pad:
        bands = torch.nn.functional.pad(bands, (0, pad))
    out = net(t11.apply_frontend(bands[None], t11.frontend_spec(net)))[0]
    if pad:
        out = out[..., :T]
    m = ((out[0:2].float() + 1.0) * 0.5).clamp(0.0, 1.0)
    if mask_levels:
        m = torch.round(m * mask_levels) / mask_levels
    mask_bin = torch.einsum("fb,cbt->cft", Gs.to(device), m).clamp(0.0, 1.0)
    voc = t09._istft(X * mask_bin, n).cpu()
    return voc, mix.cpu() - voc


def build_batches(cache: Path, limit: int, crop: int, batch: int, seed: int = 0):
    """Calibration crops straight from the teacher cache (training data)."""
    files = sorted(cache.glob("*.pt"))[:limit]
    if not files:
        raise SystemExit(f"no cache entries in {cache}")
    g = torch.Generator().manual_seed(seed)
    # Teacher caches are allowed to use fp16 on disk to keep multi-dataset
    # caches small.  The checkpoint itself is fp32, so calibration inputs must
    # be promoted before they enter the convolution layers.
    items = [torch.load(f, map_location="cpu", weights_only=False)["mix"].float()
             for f in files]
    out = []
    for _ in range(max(1, limit // batch)):
        xs = []
        for _ in range(batch):
            m = items[int(torch.randint(len(items), (1,), generator=g))]
            t = int(torch.randint(m.shape[-1] - crop + 1, (1,), generator=g))
            xs.append(m[:, :, t:t + crop])
        out.append(torch.stack(xs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RESULTS / "v4_model.pt"))
    ap.add_argument("--calib-cache", default=str(RESULTS / "v4_cache"))
    ap.add_argument("--calib-batches", type=int, default=12)
    ap.add_argument("--calib-crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--names-from", default=str(RESULTS / "v4_report.json"))
    ap.add_argument("--tracks", type=int, default=6)
    ap.add_argument("--seg", type=float, default=30.0)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default so a training run can keep the GPU")
    ap.add_argument("--act-bits", default="8,10,12,16",
                    help="activation bit widths to sweep.  DSP48E1 multiplies "
                         "18x25 bits, so 12- or 16-bit activations cost no "
                         "extra multipliers over INT8 -- only memory")
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--with-teacher", action="store_true",
                    help="also score against HTDemucs (slow, needs the GPU free)")
    args = ap.parse_args()

    device = args.device
    act_bits = [int(b) for b in args.act_bits.split(",") if b.strip()]
    print("=" * 78)
    print("INT8 DEPLOYMENT CHECK -- does the arithmetic survive quantisation?")
    print("=" * 78)

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    net, blob = t13.load_student(ckpt, device)
    print(f"\ncheckpoint   : {ckpt.name}  (step {blob.get('step')}, "
          f"best val {blob.get('best_val')})")
    print(f"device       : {device}")

    layout = t11.band_layout_for(net)
    n_bands = t11.n_bands_for(net)
    Wa = torch.from_numpy(t09.make_analysis_matrix(n_bands, layout=layout))
    Gs = torch.from_numpy(t09.make_synthesis_matrix(n_bands, layout=layout))

    # ---- calibration ----------------------------------------------------
    print(f"\ncalibrating on {args.calib_batches} x {args.batch} crops from "
          f"{Path(args.calib_cache).name}")
    batches = build_batches(Path(args.calib_cache), args.calib_batches,
                            args.calib_crop, args.batch)
    fe = t11.frontend_spec(net)
    pre = (lambda x: t11.apply_frontend(x, fe)) if fe[0] != "linear" else None
    print(f"  input front end: {fe[0]}  scale {fe[1]:g}")
    t0 = time.perf_counter()
    cal = ActQuant(net, mode="max")
    cal.probe(net, batches, pre)

    # a second observer over the same crops, reducing with a 99.9th percentile
    # instead of the max: the difference between the two spans *is* the crest
    # factor the per-tensor INT8 scale has to absorb
    pct = ActQuant(net, mode="pct", pct=99.9)
    pct.probe(net, batches, pre)

    # and a third that gives every input channel its own scale
    chn = ActQuant(net, mode="max", per_channel=True)
    chn.probe(net, batches, pre)
    print(f"  three observers over {len(cal.spans)} conv inputs in "
          f"{time.perf_counter() - t0:.1f} s")

    # ---- weight quantisation -------------------------------------------
    fp_state = {k: v.clone() for k, v in net.state_dict().items()}
    w_stats = quantize_weights_(net, per_channel=True)
    print(f"\nweight quantisation (per output channel, symmetric INT8)")
    print(f"  {'layer':<8}{'taps':>7}{'|w|max':>10}{'step':>12}"
          f"{'rel RMSE':>10}{'acc bits':>10}")
    for name, s in w_stats.items():
        print(f"  {name:<8}{s['taps_per_output']:>7}{s['w_absmax']:>10.4f}"
              f"{s['step']:>12.3e}{s['rel_rmse']:>10.4f}"
              f"{s['acc_bits']:>10d}")
    worst_bits = max(s["acc_bits"] for s in w_stats.values())
    print(f"  -> worst-case accumulator needs {worst_bits} bits "
          f"(INT32 is {'enough' if worst_bits <= 31 else 'TOO NARROW'})")

    # per-tensor contrast, on a copy
    net_tensor = t13.load_student(ckpt, device)[0]
    quantize_weights_(net_tensor, per_channel=False)

    # restore int8-per-channel weights
    net.load_state_dict(fp_state)
    quantize_weights_(net, per_channel=True)

    # ---- calibration table ---------------------------------------------
    maxspans = dict(cal.spans)
    pctspans = dict(pct.spans)
    chnspans = dict(chn.spans)
    print(f"\nactivation calibration (symmetric INT8, step = span / 127)")
    print(f"  {'layer':<8}{'per-tensor':>12}{'p99.9':>11}{'per-chan max':>13}"
          f"{'chan spread':>12}{'step':>11}")
    act_rows = []
    for name, mod in net.named_modules():
        if not isinstance(mod, t09.CausalConv2d) or name not in maxspans:
            continue
        a = cal.peak(name)
        b = pct.peak(name)
        c = chn.peak(name)
        spread = c / max(chn.mean_span(name), 1e-12)
        act_rows.append({"layer": name,
                         "in_channels": mod.conv.in_channels,
                         "kernel": list(mod.conv.kernel_size),
                         "stride": list(mod.conv.stride),
                         "span_per_tensor": round(a, 4),
                         "span_p999": round(b, 4),
                         "span_per_channel_max": round(c, 4),
                         "channel_spread": round(spread, 2),
                         "step_per_tensor": round(a / QMAX, 6),
                         "step_per_channel_max": round(c / QMAX, 6)})
        print(f"  {name:<8}{a:>12.4f}{b:>11.4f}{c:>13.4f}"
              f"{spread:>12.2f}{a/QMAX:>11.3e}")

    # ---- input headroom --------------------------------------------------
    # the very first conv's input is whatever the front end emits, so that is
    # the tensor whose dynamic range has to fit in 8 bits
    first = cal.peak("enc0")
    step0 = first / QMAX
    raw = batches[0].flatten()
    probe = pre(batches[0]).flatten() if pre is not None else raw
    zero_frac = float((probe.abs() < step0 / 2).float().mean())
    energy_kept = float(q_sym(probe, step0).pow(2).sum()
                        / (probe.pow(2).sum() + 1e-12))
    pcs = [float(torch.quantile(probe.abs(), q)) for q in
           (0.5, 0.9, 0.99, 0.999)]
    print(f"\nfirst-conv input headroom "
          f"(front end = {fe[0]}, one global scale)")
    print(f"  span {first:.4f}   step = span/127 = {step0:.4e}")
    print(f"  |x| percentiles 50/90/99/99.9 : "
          f"{pcs[0]:.3e} / {pcs[1]:.3e} / {pcs[2]:.3e} / {pcs[3]:.3e}")
    print(f"  median bin sits at {pcs[0]/step0:.2f} LSB")
    print(f"  bins rounding to zero : {zero_frac*100:.2f} %")
    print(f"  energy retained       : {energy_kept*100:.2f} %  "
          f"(energy lives in the few loud bins; the quiet half is where "
          f"vocal detail is)")

    # Candidate front-ends, all measured on the *raw* band magnitudes.  A
    # network whose input is a linear magnitude spectrum has to be
    # scale-sensitive over decades, and 8 bits cannot hold decades.
    # Compressing the input first is a *training-time* decision, so it has to
    # be settled before any RTL exists.  Each transform is quantised to 8 bits
    # over its own range and inverted, and the number reported is the
    # input-domain SNR -- how much of the spectrum survives the round trip.
    cands = {
        "linear": (lambda v: v, lambda y: y),
        "log1p(x/med)": (lambda v: torch.log1p(v / max(pcs[0], 1e-9)),
                         lambda y: torch.expm1(y) * max(pcs[0], 1e-9)),
        "sqrt(x)": (lambda v: v.clamp_min(0).sqrt(),
                    lambda y: y.clamp_min(0).pow(2)),
        "x^0.3": (lambda v: v.clamp_min(0).pow(0.3),
                  lambda y: y.clamp_min(0).pow(1 / 0.3)),
    }
    print(f"\n  candidate front-ends, 8-bit round trip on the raw input:")
    print(f"  {'transform':<16}{'SNR dB':>9}{'bins->0':>10}{'peak LSB':>10}"
          f"{'HW cost':>10}")
    hw = {"linear": "none", "log1p(x/med)": "log LUT",
          "sqrt(x)": "1 sqrt", "x^0.3": "pow LUT"}
    fe_rows = {}
    for label, (fwd, inv) in cands.items():
        y = fwd(raw)
        lo, hi = float(y.min()), float(y.max())
        stp = max(hi - lo, 1e-12) / 255
        yq = q_sym(y - (lo + hi) / 2, stp) + (lo + hi) / 2
        xh = inv(yq)
        err = xh - raw
        snr = 20 * math.log10(
            float(raw.pow(2).sum().sqrt())
            / (float(err.pow(2).sum().sqrt()) + 1e-12) + 1e-12)
        zf = float((y < lo + stp / 2).float().mean())
        pk = float(y.abs().max()) / stp
        fe_rows[label] = {"snr_dB": round(snr, 2),
                          "bins_to_zero_pct": round(zf * 100, 2),
                          "peak_in_LSB": round(pk, 1)}
        mark = "  <- active" if fe[0] != "linear" and label.startswith(fe[0]) \
            else ""
        print(f"  {label:<16}{snr:>9.2f}{zf*100:>10.2f}{pk:>10.1f}"
              f"{hw[label]:>10}{mark}")

    # ---- variants on held-out audio -------------------------------------
    names = t13.held_out_tracks(args.tracks, Path(args.names_from))
    print(f"\nheld-out excerpts: {len(names)} "
          f"(from {Path(args.names_from).name})")
    files = []
    for n in names:
        p = t13.find_file(n)
        if p is not None:
            files.append(p)

    teacher = None
    if args.with_teacher and device == "cuda":
        teacher = t11.load_demucs("htdemucs").to(device).eval()

    variants = ["float", "mask8", "w8", "w8_tensor"]
    variants += [f"a{b}" for b in act_bits]          # activations only
    variants += ["a8_p999", "a8_ch", "a8_noin"]      # 8-bit variants
    variants += [f"deploy{b}" for b in act_bits]     # int8 weights + mask8
    variants += ["deploy_ch", "deploy_noin"]
    rows = []

    width = max(9, max(len(v) for v in variants) + 1)
    print(f"\n  {'track':<30}" + "".join(f"{k:>{width}}" for k in variants[1:]))
    print(f"  {'-'*30}" + "-" * width * len(variants[1:]))

    for i, p in enumerate(files):
        total = t13.ffprobe_duration(p)
        start = 0.5 * max(0.0, total - args.seg)
        mix = t13.decode(p, start, args.seg)
        n = mix.shape[-1]

        # ---- float reference -------------------------------------------
        cal.remove(net)
        net.load_state_dict(fp_state)
        ref_v, ref_a = chain(net, mix, Wa, Gs, device, 0)

        res = {}
        # float net, mask rounded to 8 bits
        vm, _ = chain(net, mix, Wa, Gs, device, MASK_LEVELS)
        res["mask8"] = vm

        # INT8 weights, float activations, 8-bit mask
        net.load_state_dict(fp_state)
        quantize_weights_(net, per_channel=True)
        v, _ = chain(net, mix, Wa, Gs, device, MASK_LEVELS)
        res["w8"] = v
        # single per-tensor weight scale, for contrast
        vt, _ = chain(net_tensor, mix, Wa, Gs, device, MASK_LEVELS)
        res["w8_tensor"] = vt

        # float weights, quantised activations, at each candidate bit width.
        # The 7020's DSP48E1 multiplier is 18x25 bits, so INT8xINT8 wastes most
        # of it -- a 12- or 16-bit activation costs no extra DSP.  That makes
        # the bit width a memory/bandwidth question, not a multiplier one, and
        # it is worth knowing where the accuracy actually comes back.
        net.load_state_dict(fp_state)
        for bits in act_bits:
            obs = ActQuant(net, mode="max", bits=bits)
            obs.spans = dict(maxspans)
            obs.install(net)
            v, _ = chain(net, mix, Wa, Gs, device, MASK_LEVELS)
            obs.remove(net)
            res[f"a{bits}"] = v

        # 8-bit activation variants that would cost nothing extra in the PL
        net.load_state_dict(fp_state)
        for tag, spans, kw in (("a8_p999", pctspans, {}),
                               ("a8_ch", chnspans, {"per_channel": True}),
                               ("a8_noin", maxspans, {"skip": ("enc0",)})):
            obs = ActQuant(net, mode="max", **kw)
            obs.spans = dict(spans)
            obs.install(net)
            v, _ = chain(net, mix, Wa, Gs, device, MASK_LEVELS)
            obs.remove(net)
            res[tag] = v

        # the full PL datapath: int8 weights, quantised activations, uint8 mask,
        # i.e. every rounding the hardware will actually do
        for bits in act_bits:
            net.load_state_dict(fp_state)
            quantize_weights_(net, per_channel=True)
            obs = ActQuant(net, mode="max", bits=bits)
            obs.spans = dict(maxspans)
            obs.install(net)
            v, _ = chain(net, mix, Wa, Gs, device, MASK_LEVELS)
            obs.remove(net)
            res[f"deploy{bits}"] = v

        net.load_state_dict(fp_state)
        quantize_weights_(net, per_channel=True)
        for tag, spans, kw in (("deploy_ch", chnspans, {"per_channel": True}),
                               ("deploy_noin", maxspans, {"skip": ("enc0",)})):
            obs = ActQuant(net, mode="max", **kw)
            obs.spans = dict(spans)
            obs.install(net)
            v, _ = chain(net, mix, Wa, Gs, device, MASK_LEVELS)
            obs.remove(net)
            res[tag] = v

        row = {"track": p.name,
               "mixture_dBFS": round(t13._db(mix), 2),
               "float_vocal_dBFS": round(t13._db(ref_v), 2)}
        for k in variants:
            if k == "float":
                continue
            row[f"{k}_vs_float_dB"] = round(
                float(t09.si_sdr(res[k][..., :n].numpy(),
                                 ref_v[..., :n].numpy())), 2)
        if teacher is not None:
            ref = t11.teacher_vocals(teacher, mix, device)
            L = min(n, ref.shape[-1])
            row["float_vs_teacher_dB"] = round(
                float(t09.si_sdr(ref_v[..., :L].numpy(), ref[..., :L].numpy())), 2)
            row["deploy_vs_teacher_dB"] = round(
                float(t09.si_sdr(res["deploy"][..., :L].numpy(),
                                 ref[..., :L].numpy())), 2)
        rows.append(row)
        print(f"  {p.name[:30]:<30}" + "".join(
            f"{row.get(f'{k}_vs_float_dB', float('nan')):>{width}.2f}"
            for k in variants if k != "float"), flush=True)

    def med(key):
        v = [r[key] for r in rows if key in r]
        return round(float(np.median(v)), 2) if v else None

    summary = {
        "checkpoint": ckpt.name,
        "checkpoint_step": blob.get("step"),
        "device": device,
        "input_frontend": {"mode": fe[0], "scale": fe[1]},
        "calibration": {"cache": Path(args.calib_cache).name,
                        "crops": args.calib_batches * args.batch,
                        "crop_frames": args.calib_crop},
        "weight_quant": w_stats,
        "weight_bits": args.weight_bits,
        "activation_bits_swept": act_bits,
        "accumulator_bits_needed": worst_bits,
        "accumulator_ok_int32": bool(worst_bits <= 31),
        "activation_calibration": act_rows,
        "front_end": {
            "input_span": round(first, 6),
            "input_step": step0,
            "median_bin_in_LSB": round(pcs[0] / step0, 3),
            "bins_rounding_to_zero_pct": round(zero_frac * 100, 3),
            "input_energy_retained_pct": round(energy_kept * 100, 3),
            "abs_percentiles_50_90_99_999": [float(f"{p:.3e}") for p in pcs],
            "front_end_candidates": fe_rows,
        },
        "variants": variants,
        "per_track": rows,
        "median_penalty_dB_vs_float": {k: med(f"{k}_vs_float_dB")
                                       for k in variants if k != "float"},
    }
    out = RESULTS / f"quant_{ckpt.stem}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nmedian penalty vs float chain (dB):")
    for k, v in summary["median_penalty_dB_vs_float"].items():
        print(f"  {k:<12}{v:>8.2f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
