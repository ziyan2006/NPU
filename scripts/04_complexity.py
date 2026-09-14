"""Model complexity scans: parameter count, FLOPs per second of audio,
and activation working-set size -- the three numbers that decide whether a
model can live on a Zynq 7020.

Why FLOPs and not just CPU runtime: the FPGA budget is expressed in
multiply-accumulate operations per second of audio, which is architecture
independent.  CPU runtime is a secondary signal (BLAS efficiency, thread
count, memory bandwidth all distort it).

Usage
-----
python 04_complexity.py                # all available models
python 04_complexity.py --only htdemucs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import DEMUCS_SINGLE, MODELS, RESULTS, SR, load_demucs

SEG_SECONDS = 10.0          # probe length; FLOPs are reported per second of audio
BUDGET_MAC_PER_S = None     # filled in by the report script


def dummy_audio(seconds: float = SEG_SECONDS) -> torch.Tensor:
    n = int(seconds * SR)
    t = torch.arange(n, dtype=torch.float32) / SR
    # a broadband test signal: sum of tones + noise so every branch is exercised
    x = 0.1 * torch.sin(2 * np.pi * 220 * t) + 0.02 * torch.randn(n)
    return x[None].repeat(2, 1)  # (2, n)


# --------------------------------------------------------------------------
# FLOPs
# --------------------------------------------------------------------------
def count_flops_demucs(model, seconds: float = SEG_SECONDS):
    """FLOPs of one forward pass, plus the probe duration actually used.

    Demucs models are applied segment-by-segment (default ~7.8 s).  We probe on
    exactly one segment so the per-second-of-audio figure is representative of
    what `apply_model` really executes.
    """
    from torch.utils.flop_counter import FlopCounterMode

    seg_attr = float(getattr(model, "segment", 7.8) or 7.8)
    probe_s = seg_attr if 1.0 <= seg_attr <= 60.0 else 10.0
    probe_s = min(probe_s, seconds)
    n = int(probe_s * SR)
    x = torch.randn(1, model.audio_channels, n)

    counter = FlopCounterMode(display=False)
    model.eval()
    with counter, torch.no_grad():
        try:
            model(x)
        except Exception as exc:  # some models need extra args
            return None, None, f"{type(exc).__name__}: {exc}"
    return counter.get_total_flops(), probe_s, None


def count_flops_umx(seconds: float = SEG_SECONDS):
    from torch.utils.flop_counter import FlopCounterMode
    from openunmix import model as umx_model

    net = umx_model.OpenUnmix(
        nb_bins=2049, nb_channels=2, hidden_size=512, nb_layers=3,
        unidirectional=False, max_bin=1487,
    )
    sd = torch.load(str(MODELS / "umxhq_vocals.pth"), map_location="cpu",
                    weights_only=True)
    net.load_state_dict(sd, strict=False)
    net.eval()

    hop = 1024
    frames = int(seconds * SR / hop) + 1
    nb_bins, nb_ch = 2049, 2
    # model.forward expects (nb_samples, nb_channels, nb_bins, nb_frames)
    x = torch.randn(1, nb_ch, nb_bins, frames)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        net(x)
    return counter.get_total_flops(), float(seconds), None


# --------------------------------------------------------------------------
# activation working set
# --------------------------------------------------------------------------
def activation_stats(model, seconds: float = 1.0):
    """Per-layer output tensor sizes for one forward pass (float32 bytes).

    Returns (max_single_layer_MB, total_all_layers_MB, n_layers, biggest name).
    """
    sizes: list[tuple[str, int]] = []

    def hook(name):
        def fn(module, inp, out):
            n = 0
            for t in out if isinstance(out, (tuple, list)) else (out,):
                if torch.is_tensor(t):
                    n += t.numel() * 4
            if n:
                sizes.append((name, n))
        return fn

    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
          if len(list(m.children())) == 0]
    try:
        if hasattr(model, "audio_channels"):
            seg = int(min(getattr(model, "segment", 7.8), seconds) * SR)
            x = torch.randn(1, model.audio_channels, seg)
        else:
            x = None
        if x is not None:
            with torch.no_grad():
                model(x)
    except Exception:
        pass
    finally:
        for h in hs:
            h.remove()

    if not sizes:
        return None
    biggest = max(sizes, key=lambda kv: kv[1])
    return {
        "max_layer_MB": round(biggest[1] / 1024**2, 3),
        "max_layer_name": biggest[0],
        "sum_all_layers_MB": round(sum(s for _, s in sizes) / 1024**2, 1),
        "n_tensors": len(sizes),
    }


def profile_demucs(tag: str, seconds: float) -> dict:
    t0 = time.perf_counter()
    model = load_demucs(tag)
    load_s = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    flops, probe_s, err = count_flops_demucs(model, seconds)
    out = {
        "tag": tag,
        "family": type(model).__name__,
        "sources": list(getattr(model, "sources", [])),
        "params_M": round(params / 1e6, 3),
        "weight_fp32_MB": round(params * 4 / 1024**2, 2),
        "weight_int8_MB": round(params / 1024**2, 2),
        "segment_s": float(getattr(model, "segment", 7.8) or 7.8),
        "load_s": round(load_s, 2),
    }
    if flops:
        # FLOPs measured on one `probe_s` slice -> normalise to per audio second
        out["probe_s"] = round(probe_s, 3)
        out["flops_per_probe_G"] = round(flops / 1e9, 2)
        out["mac_per_audio_s_G"] = round(flops / 2 / probe_s / 1e9, 3)
    else:
        out["flops_error"] = err
    out["activation"] = activation_stats(model)
    return out


def profile_umx(seconds: float) -> dict:
    from openunmix import model as umx_model

    m = umx_model.OpenUnmix(nb_bins=2049, nb_channels=2, hidden_size=512,
                            nb_layers=3, unidirectional=False, max_bin=1487)
    sd = torch.load(str(MODELS / "umxhq_vocals.pth"), map_location="cpu",
                    weights_only=True)
    m.load_state_dict(sd, strict=False)
    m.eval()
    params = sum(p.numel() for p in m.parameters())
    flops, probe_s, err = count_flops_umx(seconds)
    out = {
        "tag": "umx",
        "family": "OpenUnmix+BiLSTM(umxhq vocals)",
        "sources": ["vocals"],
        "params_M": round(params / 1e6, 3),
        "weight_fp32_MB": round(params * 4 / 1024**2, 2),
        "weight_int8_MB": round(params / 1024**2, 2),
    }
    if flops:
        out["probe_s"] = round(probe_s, 3)
        out["mac_per_audio_s_G"] = round(flops / 2 / probe_s / 1e9, 3)
        out["flops_per_probe_G"] = round(flops / 1e9, 2)
    else:
        out["flops_error"] = err
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--seconds", type=float, default=SEG_SECONDS)
    args = ap.parse_args()

    rows = []
    targets = [args.only] if args.only else list(DEMUCS_SINGLE) + ["umx"]
    for tag in targets:
        if tag == "umx":
            print("[umx] profiling ...", flush=True)
            rows.append(profile_umx(args.seconds))
        else:
            if not DEMUCS_SINGLE[tag].exists():
                print(f"[{tag}] weights missing, skipping", flush=True)
                continue
            print(f"[{tag}] profiling ...", flush=True)
            try:
                rows.append(profile_demucs(tag, args.seconds))
            except Exception as exc:
                print(f"[{tag}] SKIPPED ({type(exc).__name__}: {exc})", flush=True)
                continue
        print(json.dumps(rows[-1], indent=2), flush=True)

    out = RESULTS / "complexity.json"
    old = {}
    if out.exists():
        old = {r["tag"]: r for r in json.loads(out.read_text(encoding="utf-8"))}
    for r in rows:
        old[r["tag"]] = r
    out.write_text(json.dumps(list(old.values()), indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
