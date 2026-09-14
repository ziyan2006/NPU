"""Translate model complexity into a concrete Zynq-7020 hardware budget, and
cost out a *causal* spectral U-Net that the PL can actually hold.

Why a single "does it fit" question is the wrong question
---------------------------------------------------------
A raw MAC/s count makes HTDemucs look almost affordable on a 7020.  It isn't,
for reasons the MAC count hides.  We therefore test every model against three
independent gates and a design only passes if it clears all of them:

  GATE 1  causality   algorithmic lookahead <= 100 ms   (the streaming spec)
  GATE 2  compute     realistic RTF <= 0.5 on 220 DSP   (need headroom)
  GATE 3  on-chip     activation working set fits in 630 KB BRAM; weights
                      stream from DDR within ~1.2 GB/s

Hardware reference -- XC7Z020-1 (Zynq-7020):
    DSP48E1        220
    LUT            53,200
    FF             106,400
    BRAM           140 x 36 Kb = 4.9 Mb = 630 KB
    PL clock       200 MHz is comfortably reachable for DSP-heavy pipelines
    DDR3 (HP)      ~1.2 GB/s sustained on a typical 7020 board

Usage
-----
python 06_fpga_budget.py
python 06_fpga_budget.py --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from common import RESULTS

# --------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------
PL_FREQ_MHZ = 200.0
N_DSP = 220
BRAM_KB = 630.0
DDR_MBPS = 1200.0
MAC_PER_DSP_CYCLE = 1.0
SUSTAINED_EFFICIENCY = 0.5

DSP_PEAK_GMAC = N_DSP * MAC_PER_DSP_CYCLE * PL_FREQ_MHZ * 1e6 / 1e9
DSP_REAL_GMAC = DSP_PEAK_GMAC * SUSTAINED_EFFICIENCY

LATENCY_BUDGET_MS = 100.0
COMPUTE_HEADROOM = 0.5
BRAM_USE_FRACTION = 0.8          # leave 20% for control/FIFO/scratch

# How much of the theoretical DSP throughput you can actually hold, by operator
# mix.  Dense convs map beautifully onto a systolic array; attention needs
# data-dependent gather + softmax and wrecks array utilisation.
EFFICIENCY = {"conv_only": 0.50, "hybrid_attention": 0.15}


# --------------------------------------------------------------------------
# parametric architecture cost model
# --------------------------------------------------------------------------
@dataclass
class Plan:
    """A causal, spectrogram-domain, band-compressed U-Net.

    Front-end : fixed (non-learned) log/ERB filterbank      -> 0 MAC
    Core      : causal 2D conv U-Net on (bands x time)
    Back-end  : band-wise mask, linearly interpolated to bins -> 0 MAC

    Three deliberate choices keep this inside a 7020:

      * Model 128 bands, not 513 raw bins.  The vocal band is perceptually
        log-spaced, so a fixed filterbank loses almost nothing while cutting
        the dominant term of the cost by 4x.
      * Upsample with linear interpolation (0 MAC) and let the following conv
        do the work, instead of paying for a ConvTranspose per stage.
      * Use (1, kt) time-only kernels on the two widest decoder stages, where
        bands x frames is largest and a (3,3) kernel buys the least.
    """
    name: str
    sr: int = 44100
    n_fft: int = 1024
    hop: int = 256
    n_bands: int = 128
    enc: tuple = (32, 64, 96, 128)
    kf: int = 3
    kt: int = 3
    freq_stride: tuple = (1, 2, 2, 2)
    stems: int = 2
    out_bins: int = 513
    low_res_kernel_1x1: bool = True
    chunk_frames: int = 4          # frames processed per weight-load (~23 ms)
    act_bits: int = 8              # INT8 activations, INT32 accumulators

    @property
    def fps(self) -> float:
        return self.sr / self.hop

    def layers(self) -> list[dict]:
        """Ordered layer list with the shape each layer runs at."""
        L: list[dict] = []
        cin, b, f = 2, self.n_bands, self.fps
        for i, cout in enumerate(self.enc):
            sf = self.freq_stride[i]
            st = 1 if i == 0 else 2
            kf = self.kf if i == 0 else min(self.kf, 3)
            L.append(dict(kind="enc", cin=cin, cout=cout, kf=kf, kt=self.kt,
                          bands=b, fps=f))
            cin = cout
            b = max(1, b // sf)
            f = f / st
        L.append(dict(kind="bott", cin=cin, cout=cin, kf=1, kt=self.kt,
                      bands=b, fps=f))
        cb = cin
        for i in range(len(self.enc) - 1, -1, -1):
            skip_c = self.enc[i]
            out_c = self.enc[i - 1] if i > 0 else self.enc[0]
            b = b * self.freq_stride[i]
            f = f * (1 if i == 0 else 2)
            if i <= 1 and self.low_res_kernel_1x1:
                kf2, kt2 = 1, self.kt
            else:
                kf2, kt2 = self.kf, self.kt
            L.append(dict(kind="dec", cin=cb + skip_c, cout=out_c,
                          kf=kf2, kt=kt2, bands=b, fps=f))
            cb = out_c
        L.append(dict(kind="out", cin=cb, cout=self.stems * 2, kf=1, kt=1,
                      bands=self.n_bands, fps=self.fps))
        return L

    def cost(self) -> dict:
        L = self.layers()
        ab = self.act_bits / 8.0
        params = 0.0
        macs = 0.0
        act_bytes = 0.0
        max_buf = 0.0
        for lay in L:
            w = lay["cin"] * lay["cout"] * lay["kf"] * lay["kt"]
            params += w
            macs += w * lay["bands"] * lay["fps"]
            # history line buffer this layer needs at its own resolution;
            # taps beyond the first still have to be resident while the kernel
            # slides, so the buffer is (cin x bands x kt) activations
            lb = lay["cin"] * lay["bands"] * lay["kt"] * ab
            act_bytes += lb
            max_buf = max(max_buf, lb)
        # skip connections: one chunk of every encoder resolution is retained
        b = self.n_bands
        for i, cout in enumerate(self.enc):
            sb = cout * b * self.chunk_frames * ab
            act_bytes += sb
            max_buf = max(max_buf, sb)
            b = max(1, b // self.freq_stride[i])
        # decoder line buffers at full band resolution
        out_lb = self.enc[0] * self.n_bands * self.kt * ab
        act_bytes += out_lb

        latency_ms = (self.n_fft / self.sr
                      + (3 + self.chunk_frames) * self.hop / self.sr) * 1000
        rtf = macs / 1e9 / (DSP_PEAK_GMAC * EFFICIENCY["conv_only"])
        w_kb = params / 1024
        return {
            "name": self.name,
            "n_fft": self.n_fft, "hop": self.hop, "n_bands": self.n_bands,
            "frames_per_s": round(self.fps, 2),
            "enc": list(self.enc),
            "params_M": round(params / 1e6, 4),
            "weight_int8_KB": round(w_kb, 1),
            "gmac_per_audio_s": round(macs / 1e9, 4),
            "n_layers": len(L),
            "alg_latency_ms": round(latency_ms, 1),
            "realistic_rtf_7020": round(rtf, 4),
            "activation_total_KB": round(act_bytes / 1024, 1),
            "activation_max_buf_KB": round(max_buf / 1024, 1),
            "weight_stream_MBps_perframe": round(w_kb * 1024 * self.fps / 1e6, 1),
            "weight_stream_MBps_perchunk": round(
                w_kb * 1024 * self.fps / self.chunk_frames / 1e6, 1),
        }

    def verdict(self) -> dict:
        c = self.cost()
        gates = {
            "causal_ok": c["alg_latency_ms"] <= LATENCY_BUDGET_MS,
            "compute_ok": c["realistic_rtf_7020"] <= COMPUTE_HEADROOM,
            "memory_ok": c["activation_total_KB"]
            <= BRAM_KB * BRAM_USE_FRACTION,
            "bandwidth_ok": c["weight_stream_MBps_perchunk"] <= DDR_MBPS * 0.5,
        }
        c["gates"] = gates
        c["verdict"] = "GO" if all(gates.values()) else \
            "NO-GO (" + ", ".join(k for k, v in gates.items() if not v) + ")"
        return c


PRESETS = [
    Plan("causal-tiny", n_bands=64, enc=(24, 48, 64, 96)),
    Plan("causal-recommended", n_bands=128, enc=(32, 64, 96, 128)),
    Plan("causal-wide", n_bands=256, enc=(32, 64, 96, 128)),
    Plan("causal-aggressive", n_bands=256, enc=(48, 96, 128, 192)),
    Plan("causal-2d-fullfreq", n_bands=513, enc=(32, 64, 96, 128),
         freq_stride=(1, 1, 2, 2), low_res_kernel_1x1=False),
]


# --------------------------------------------------------------------------
# baselines, judged on the same three gates
# --------------------------------------------------------------------------
BASELINE_FACTS = {
    "umx": {
        "causal": False, "segment_s": None,
        "note": "bidirectional LSTM sweeps the whole track; "
                "`unidirectional=True` is a drop-in causal variant",
    },
    "htdemucs": {"causal": False, "segment_s": 7.8,
                 "note": "cross-attention over a 7.8 s segment"},
    "hdemucs_mmi": {"causal": False, "segment_s": 40.0,
                    "note": "bottleneck self-attention over a 40 s segment"},
    "mdx": {"causal": False, "segment_s": 44.0,
            "note": "complex-spectrogram U-Net, segment-length receptive field"},
    "mdx7ecf8ec1": {"causal": False, "segment_s": 44.0, "note": "MDX-Net"},
}


def judge_baseline(row: dict) -> dict:
    tag = row["name"]
    facts = BASELINE_FACTS.get(tag, {})
    seg = facts.get("segment_s")
    lookahead_ms = seg * 1000 / 2 if seg else None
    eff = EFFICIENCY["hybrid_attention"] if tag in ("htdemucs", "hdemucs_mmi") \
        else EFFICIENCY["conv_only"]
    rtf = row["gmac_per_audio_s"] / (DSP_PEAK_GMAC * eff)
    peak_mb = row.get("peak_activation_MB")
    gates = {
        "causal_ok": bool(lookahead_ms) and lookahead_ms <= LATENCY_BUDGET_MS,
        "compute_ok": rtf <= COMPUTE_HEADROOM,
        "memory_ok": peak_mb is not None and peak_mb <= BRAM_KB / 1024,
    }
    out = dict(row)
    out.update({
        "lookahead_ms": round(lookahead_ms, 1) if lookahead_ms else "unbounded",
        "assumed_dsp_efficiency": eff,
        "realistic_rtf_7020": round(rtf, 3),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else
                   "NO-GO (" + ", ".join(k for k, v in gates.items() if not v) + ")",
        "note": facts.get("note", ""),
    })
    return out


def load_baselines() -> list[dict]:
    p = RESULTS / "complexity.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for r in data:
        g = r.get("mac_per_audio_s_G")
        if g is None:
            continue
        act = r.get("activation") or {}
        out.append(judge_baseline({
            "name": r["tag"],
            "family": r["family"],
            "params_M": r["params_M"],
            "weight_int8_MB": r.get("weight_int8_MB"),
            "gmac_per_audio_s": g,
            "peak_activation_MB": act.get("max_layer_MB"),
            "sum_activation_MB": act.get("sum_all_layers_MB"),
        }))
    return out


def sweep() -> list[dict]:
    rows = []
    for bands in (64, 128, 256):
        for w in (16, 24, 32, 48, 64):
            rows.append(Plan(f"b{bands}_w{w}", n_bands=bands,
                             enc=(w, w * 2, w * 3, w * 4)).cost())
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("Zynq-7020 compute ceiling")
    print(f"  DSP48E1              : {N_DSP}")
    print(f"  PL clock             : {PL_FREQ_MHZ:.0f} MHz")
    print(f"  peak int8            : {DSP_PEAK_GMAC:.1f} GMAC/s"
          f"  ({MAC_PER_DSP_CYCLE:.1f} MAC/DSP/cycle)")
    print(f"  sustained conv (x{SUSTAINED_EFFICIENCY:.2f}): {DSP_REAL_GMAC:.1f} GMAC/s")
    print(f"  BRAM                 : {BRAM_KB:.0f} KB")
    print(f"  DDR3 sustained       : {DDR_MBPS:.0f} MB/s")
    print("=" * 78)

    base = load_baselines()
    if base:
        print("\nBASELINES  --  judged on all gates, not just MACs")
        print(f"{'model':<14}{'params':>8}{'GMAC/s':>9}{'eff':>6}{'RTF':>7}"
              f"{'peakAct':>9}{'lookahead':>11}   verdict")
        for r in sorted(base, key=lambda x: x["gmac_per_audio_s"]):
            print(f"{r['name']:<14}{r['params_M']:>7.2f}M"
                  f"{r['gmac_per_audio_s']:>9.2f}"
                  f"{r['assumed_dsp_efficiency']:>6.2f}"
                  f"{r['realistic_rtf_7020']:>7.2f}"
                  f"{(str(r['peak_activation_MB'])+'MB' if r['peak_activation_MB'] else '-'):>9}"
                  f"{(str(r['lookahead_ms'])+'ms' if isinstance(r['lookahead_ms'], float) else r['lookahead_ms']):>11}"
                  f"   {r['verdict']}")
        print("\n  notes")
        for r in base:
            if r["note"]:
                print(f"    {r['name']:<14} {r['note']}")
    else:
        print("\n(no complexity.json yet -- run 04_complexity.py)")

    designs = [p.verdict() for p in PRESETS]
    print("\nCANDIDATE CAUSAL DESIGNS  --  same cost model, same gates")
    print(f"{'plan':<22}{'bands':>6}{'params':>8}{'wINT8':>8}{'GMAC/s':>9}"
          f"{'RTF':>7}{'lat':>7}{'act':>9}{'bw':>8}   verdict")
    for r in designs:
        print(f"{r['name']:<22}{r['n_bands']:>6}{r['params_M']:>7.3f}M"
              f"{r['weight_int8_KB']:>7.0f}K{r['gmac_per_audio_s']:>9.3f}"
              f"{r['realistic_rtf_7020']:>7.3f}{r['alg_latency_ms']:>6.0f}ms"
              f"{r['activation_total_KB']:>8.0f}K"
              f"{r['weight_stream_MBps_perchunk']:>7.0f}   {r['verdict']}")

    payload = {
        "hardware": {
            "device": "XC7Z020-1", "n_dsp": N_DSP, "pl_freq_MHz": PL_FREQ_MHZ,
            "bram_KB": BRAM_KB, "ddr_MBps": DDR_MBPS,
            "peak_gmac_s": round(DSP_PEAK_GMAC, 2),
            "sustained_gmac_s": round(DSP_REAL_GMAC, 2),
            "sustained_efficiency": SUSTAINED_EFFICIENCY,
            "latency_budget_ms": LATENCY_BUDGET_MS,
            "compute_headroom_rtf": COMPUTE_HEADROOM,
        },
        "baselines": base,
        "designs": designs,
    }
    if args.sweep:
        payload["sweep"] = sweep()
    out = RESULTS / "fpga_budget.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
