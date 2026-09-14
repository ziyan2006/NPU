"""Shared helpers for the NPU-stem evaluation pipeline.

All paths are anchored to the project root (the parent of this file).
Audio convention: torch tensors of shape (channels, samples) at 44.1 kHz stereo.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

SR = 44100
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
WORK = DATA / "work"
RESULTS = ROOT / "results"
STEMS = RESULTS / "stems"

for _d in (WORK, RESULTS, STEMS):
    _d.mkdir(parents=True, exist_ok=True)

# Thread count matters more than it looks: on the i7-14700HX (8P+12E, 28 logical
# threads) HTDemucs runs RTF 0.882 at 7 threads, 0.340 at 14, 0.306 at 20 and
# 0.333 at 28.  Opening every logical core is *slower* because of hyperthread
# oversubscription.  20 (the physical core count) measured fastest -- see
# 10_cpu_vs_gpu.py and reports/04_训练算力平台选择.md.
_NT = min(20, os.cpu_count() or 8)
torch.set_num_threads(_NT)

DEMUCS_REPO = MODELS / "demucs_repo"

# tag -> (display name, weight file or None, family)
DEMUCS_SINGLE = {
    "htdemucs": MODELS / "htdemucs.th",
    "hdemucs_mmi": DEMUCS_REPO / "75fc33f5-1941ce65.th",
    "mdx": DEMUCS_REPO / "0d19c1c6-0f06f20e.th",
    "mdx7ecf8ec1": DEMUCS_REPO / "7ecf8ec1-70f50cc9.th",
}


def read_wav(path) -> torch.Tensor:
    """Read a WAV file as float32 (channels, samples) at 44.1 kHz."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"expected {SR} Hz, got {sr} Hz for {path}")
    return torch.from_numpy(np.ascontiguousarray(x.T))


def write_wav(path, wav: torch.Tensor) -> None:
    """Write (channels, samples) float tensor to a 32-bit float WAV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = wav.detach().cpu().numpy().T
    sf.write(str(p), arr, SR, subtype="FLOAT")


def peak_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        return float("nan")


def load_demucs(tag: str):
    """Load one Demucs-family model from the local weight files."""
    from demucs.states import load_model

    if tag not in DEMUCS_SINGLE:
        raise KeyError(f"unknown demucs tag {tag!r}")
    path = DEMUCS_SINGLE[tag]
    if not path.exists():
        raise FileNotFoundError(path)
    model = load_model(str(path))
    model.eval()
    return model


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def n_params_effective(model) -> int:
    """Parameter count weighted by actual storage bits per tensor.

    Quantised checkpoints (e.g. MDX `_q`) store fp16/int8 buffers; a plain
    `numel()` sum over-counts those.  We fold dtype bit-width in here.
    """
    total_bits = 0
    for p in model.parameters():
        total_bits += p.numel() * _dtype_bits(p.dtype)
    return total_bits / _dtype_bits(next(model.parameters()).dtype)


def _dtype_bits(dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 16
    if dtype in (torch.int8, torch.uint8):
        return 8
    if dtype == torch.float64:
        return 64
    return 32
