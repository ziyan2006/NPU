"""Run 2-stem (vocals / accompaniment) separation on one WAV with one engine.

Engines
-------
umx           Open-Unmix umxhq (BiLSTM, magnitude spectrogram, 1 target)
htdemucs      Demucs v4 Hybrid Transformer  (quality ceiling)
hdemucs_mmi   Demucs v3 HDemucs, waveform+spectrogram hybrid (no transformer)
mdx           MDX-Net, complex-spectrogram U-Net (closest family to the target)

2-stem convention
-----------------
vocals        = model vocal estimate
accompaniment = sum of the model's non-vocal stems  (for umx: mixture - vocals)

Usage
-----
python 03_separate.py --engine htdemucs --in data/work/excerpt.wav --tag htdemucs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    MODELS,
    RESULTS,
    SR,
    peak_rss_mb,
    read_wav,
    write_wav,
)


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------
def _umx_separator():
    from openunmix import model as umx_model

    # umxhq: 4096-pt STFT (2049 bins) but the network only models the first
    # 1487 bins (~16 kHz), 3 BLSTM layers, hidden 512, single target.
    # bidirectional=True  <=>  unidirectional=False
    net = umx_model.OpenUnmix(
        nb_bins=2049, nb_channels=2,
        hidden_size=512, nb_layers=3, unidirectional=False,
        max_bin=1487,
    )
    sd = torch.load(str(MODELS / "umxhq_vocals.pth"), map_location="cpu",
                    weights_only=True)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"  [umx] missing={len(missing)} unexpected={list(unexpected)}", flush=True)
    net.eval()
    sep = umx_model.Separator(
        # single target requires residual=True (EM needs >= 2 sources);
        # residual == mixture - vocals, which is exactly our accompaniment.
        target_models={"vocals": net}, niter=1, residual=True,
        sample_rate=SR, n_fft=4096, n_hop=1024, nb_channels=2,
        wiener_win_len=300, filterbank="torch",
    )
    sep.freeze()
    sep.eval()
    return sep


def run_umx(mix: torch.Tensor):
    sep = _umx_separator()
    with torch.no_grad():
        # Separator.forward expects (batch, channels, samples) and returns
        # (batch, nb_targets, channels, samples)
        est = sep(mix[None])
        d = sep.to_dict(est)
    print(f"  [umx] targets: {sorted(d)}", flush=True)
    voc = d["vocals"][0]
    acc = d["residual"][0]
    return voc, acc


def _demucs_apply(model, mix: torch.Tensor):
    from demucs.apply import apply_model

    ref = mix.mean(0)
    norm = (mix - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(
            model, norm[None], device="cpu", shifts=1, split=True,
            overlap=0.25, progress=False,
        )[0]
    return out * ref.std() + ref.mean()


def run_demucs(mix: torch.Tensor, tag: str):
    from common import load_demucs

    model = load_demucs(tag)
    src = _demucs_apply(model, mix)
    names = list(model.sources)
    iv = names.index("vocals")
    voc = src[iv]
    acc = torch.stack([src[i] for i in range(len(names)) if i != iv]).sum(0)
    return voc, acc


ENGINES = {
    "umx": lambda mix, tag: run_umx(mix),
    "htdemucs": lambda mix, tag: run_demucs(mix, "htdemucs"),
    "hdemucs_mmi": lambda mix, tag: run_demucs(mix, "hdemucs_mmi"),
    "mdx": lambda mix, tag: run_demucs(mix, "mdx"),
    "mdx7ecf8ec1": lambda mix, tag: run_demucs(mix, "mdx7ecf8ec1"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--tag", default=None, help="output sub-directory name")
    ap.add_argument("--out-root", default=str(RESULTS / "stems"))
    args = ap.parse_args()

    tag = args.tag or args.engine
    mix = read_wav(args.inp)
    dur = mix.shape[1] / SR
    print(f"[{tag}] input {args.inp}  {mix.shape[1]} samples  {dur:.2f}s", flush=True)

    rss0 = peak_rss_mb()
    t0 = time.perf_counter()
    voc, acc = ENGINES[args.engine](mix, tag)
    wall = time.perf_counter() - t0
    rss1 = peak_rss_mb()

    outdir = Path(args.out_root) / tag
    write_wav(outdir / "vocals.wav", voc)
    write_wav(outdir / "accompaniment.wav", acc)
    write_wav(outdir / "mixture.wav", mix)

    rec = {
        "tag": tag,
        "engine": args.engine,
        "input": str(args.inp),
        "audio_seconds": round(dur, 3),
        "wall_seconds": round(wall, 3),
        "rtf": round(wall / dur, 4),
        "rss_start_mb": round(rss0, 1),
        "rss_peak_mb": round(rss1, 1),
        "rss_delta_mb": round(rss1 - rss0, 1),
    }
    meta = Path(args.out_root) / f"{tag}.json"
    meta.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
