"""CPU vs GPU benchmark for the two expensive stages of this project.

Stage A -- pseudo-label generation
    Run the pretrained HTDemucs teacher over unlabelled music to produce
    training targets.  Cost is proportional to *audio hours*, not to model size.

Stage B -- training the 0.53 M-parameter causal spectral U-Net
    Train the target network on those labels.  Cost is proportional to
    (audio hours x epochs).

Both are timed on this machine's CPU (i7-14700HX) and, if a CUDA build of
torch is installed, on the RTX 4070 Laptop GPU.  Results are written to
results/cpu_vs_gpu.json so the recommendation is backed by measurements
rather than intuition.

Run:
    python 10_cpu_vs_gpu.py                 # both devices
    python 10_cpu_vs_gpu.py --devices cpu   # CPU only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from common import DEMUCS_SINGLE, RESULTS, SR, n_params  # noqa: E402

N_BANDS = 128
N_FFT, HOP = 1024, 256
FRAME_RATE = SR / HOP               # 172.266 frames per second


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def load_target_module():
    spec = importlib.util.spec_from_file_location(
        "target_model", SCRIPTS / "09_target_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated() / 1024**2


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Stage B: target-model training step
# ---------------------------------------------------------------------------
def bench_train_step(model_fn, device: torch.device, frames: int, batch: int,
                     iters: int, warmup: int, autocast_dtype=None) -> dict:
    model = model_fn().to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch, 2, N_BANDS, frames, device=device)
    y = torch.randn(batch, 4, N_BANDS, frames, device=device)
    crit = nn.MSELoss()
    amp = autocast_dtype is not None and device.type == "cuda"

    def step():
        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast("cuda", dtype=autocast_dtype):
                loss = crit(model(x), y)
        else:
            loss = crit(model(x), y)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    sync(device)
    dt = (time.perf_counter() - t0) / iters

    audio_s_per_sample = frames / FRAME_RATE
    return {
        "task": "train_step",
        "device": str(device),
        "precision": "bf16-autocast" if amp else "fp32",
        "batch": batch,
        "frames": frames,
        "audio_s_per_batch": round(audio_s_per_sample * batch, 3),
        "ms_per_step": round(dt * 1000, 2),
        "audio_s_per_wall_s": round(audio_s_per_sample * batch / dt, 1),
        "params_M": round(n_params(model) / 1e6, 4),
        "peak_mem_MB": round(gpu_mem_mb()) if device.type == "cuda" else None,
    }


def bench_train_only_fwd(model_fn, device: torch.device, frames: int, batch: int,
                         iters: int = 8) -> dict:
    """Forward-only throughput: this is also the inference speed of the product."""
    model = model_fn().to(device).eval()
    x = torch.randn(batch, 2, N_BANDS, frames, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync(device)
        dt = (time.perf_counter() - t0) / iters
    audio_s = frames / FRAME_RATE * batch
    return {
        "task": "forward_only",
        "device": str(device),
        "batch": batch,
        "frames": frames,
        "ms_per_pass": round(dt * 1000, 2),
        "audio_s_per_wall_s": round(audio_s / dt, 1),
    }


# ---------------------------------------------------------------------------
# Stage A: HTDemucs teacher inference
# ---------------------------------------------------------------------------
def bench_htdemucs(device: torch.device, seconds: float, iters: int,
                   warmup: int) -> dict:
    from demucs.states import load_model

    model = load_model(str(DEMUCS_SINGLE["htdemucs"]))
    model = model.to(device).eval()
    target = int(seconds * SR)
    valid = int(model.valid_length(target))
    x = torch.randn(1, model.audio_channels, valid, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync(device)
        dt = (time.perf_counter() - t0) / iters
    audio_s = valid / SR
    return {
        "task": "htdemucs_forward",
        "device": str(device),
        "segment_s": round(audio_s, 3),
        "ms_per_segment": round(dt * 1000, 2),
        "rtf": round(dt / audio_s, 4),
        "audio_s_per_wall_s": round(audio_s / dt, 2),
        "peak_mem_MB": round(gpu_mem_mb()) if device.type == "cuda" else None,
    }


# ---------------------------------------------------------------------------
# data pipeline: STFT analysis (the step that runs between disk and the GPU)
# ---------------------------------------------------------------------------
def bench_stft(device: torch.device, seconds: float, iters: int = 5) -> dict:
    n = int(seconds * SR)
    x = torch.randn(2, n, device=device)
    win = torch.hann_window(N_FFT, device=device)
    with torch.no_grad():
        for _ in range(2):
            torch.stft(x, N_FFT, HOP, window=win, return_complex=True)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.stft(x, N_FFT, HOP, window=win, return_complex=True)
        sync(device)
        dt = (time.perf_counter() - t0) / iters
    return {
        "task": "stft_frontend",
        "device": str(device),
        "audio_s": seconds,
        "ms_per_pass": round(dt * 1000, 2),
        "audio_s_per_wall_s": round(seconds / dt, 1),
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def hours_needed(audio_hours: float, audio_s_per_wall_s: float) -> float:
    if not audio_s_per_wall_s:
        return float("nan")
    return audio_hours * 3600 / audio_s_per_wall_s / 3600


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=7.8)
    ap.add_argument("--threads", type=int, default=20,
                    help="CPU threads. 20 measured fastest on this i7-14700HX "
                         "(8P+12E); 28 (all logical) is *slower* due to "
                         "hyperthread oversubscription.")
    args = ap.parse_args()

    threads = args.threads
    torch.set_num_threads(threads)

    print(f"torch          : {torch.__version__}")
    print(f"cpu threads    : {threads}")
    print(f"cuda available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu            : {p.name}  sm_{p.major}{p.minor}  "
              f"{p.total_memory/1024**3:.2f} GB")
        torch.backends.cudnn.benchmark = True
    print()

    tmod = load_target_module()
    model_fn = lambda: tmod.CausalSpectralUNet(  # noqa: E731
        n_in=2, enc=(32, 64, 96, 128), n_out=4)

    wanted = [d.strip() for d in args.devices.split(",") if d.strip()]
    if "cuda" not in wanted or not torch.cuda.is_available():
        wanted = [d for d in wanted if d != "cuda"]
    devices = [torch.device(d) for d in wanted]

    rows: list[dict] = []
    for dev in devices:
        reset_peak()
        print(f"---------- {dev.type.upper()} ----------")

        print("  [B] target model: training step ...", flush=True)
        r = bench_train_step(model_fn, dev, args.frames, args.batch,
                             args.iters, warmup=2)
        rows.append(r)
        print(f"      {r['precision']:14s} {r['ms_per_step']:8.1f} ms/step  "
              f"-> {r['audio_s_per_wall_s']:8.1f} audio-s per wall-s")

        if dev.type == "cuda":
            print("  [B] target model: training step (bf16) ...", flush=True)
            r = bench_train_step(model_fn, dev, args.frames, args.batch,
                                 args.iters, warmup=2,
                                 autocast_dtype=torch.bfloat16)
            rows.append(r)
            print(f"      {r['precision']:14s} {r['ms_per_step']:8.1f} ms/step  "
                  f"-> {r['audio_s_per_wall_s']:8.1f} audio-s per wall-s")

        print("  [B] target model: forward only ...", flush=True)
        r = bench_train_only_fwd(model_fn, dev, args.frames, args.batch)
        rows.append(r)
        print(f"      {r['ms_per_pass']:8.2f} ms/pass -> "
              f"{r['audio_s_per_wall_s']:8.1f} audio-s per wall-s")

        print("  [A] HTDemucs teacher forward ...", flush=True)
        try:
            r = bench_htdemucs(dev, args.seconds, iters=4, warmup=1)
            rows.append(r)
            print(f"      RTF {r['rtf']:.4f}  ({r['ms_per_segment']:.0f} ms per "
                  f"{r['segment_s']:.2f}s segment) -> "
                  f"{r['audio_s_per_wall_s']:.2f} audio-s per wall-s")
        except Exception as exc:
            rows.append({"task": "htdemucs_forward", "device": str(dev),
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"      FAILED: {type(exc).__name__}: {exc}")

        print("  [pipeline] STFT analysis ...", flush=True)
        r = bench_stft(dev, seconds=10.0)
        rows.append(r)
        print(f"      {r['ms_per_pass']:8.2f} ms per 10s audio -> "
              f"{r['audio_s_per_wall_s']:8.1f} audio-s per wall-s")

        if dev.type == "cuda":
            print(f"  peak GPU memory during bench: {gpu_mem_mb():.0f} MB")
        print()

    # ---- summary -----------------------------------------------------------
    def pick(task, dev, precision=None):
        for r in rows:
            if r.get("task") != task or r.get("device") != dev:
                continue
            if precision and r.get("precision") != precision:
                continue
            return r
        return None

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    label_fmt = f"{'stage':<26}{'CPU':>16}{'GPU':>16}{'speedup':>10}"

    cpu_label = pick("htdemucs_forward", "cpu")
    gpu_label = pick("htdemucs_forward", "cuda")
    cpu_tr = pick("train_step", "cpu")
    gpu_tr = pick("train_step", "cuda", "fp32") or pick("train_step", "cuda")
    cpu_trb = pick("train_step", "cpu")
    gpu_trb = pick("train_step", "cuda", "bf16-autocast")

    print(f"\n{'stage':<26}{'CPU':>16}{'GPU':>16}{'speedup':>10}")
    print("-" * 78)
    for name, a, b in [
        ("A  teacher labels", cpu_label, gpu_label),
        ("B  train step fp32", cpu_tr, gpu_tr),
        ("B  train step bf16", cpu_trb, gpu_trb),
    ]:
        if not a or not b or "audio_s_per_wall_s" not in a or \
                "audio_s_per_wall_s" not in b:
            continue
        ca, cb = a["audio_s_per_wall_s"], b["audio_s_per_wall_s"]
        print(f"{name:<26}{ca:>13.1f} a/s{cb:>13.1f} a/s{cb/ca:>9.1f}x")

    print("\nwall-clock for a dataset of N audio hours")
    print("-" * 78)
    print(f"{'hours':>8}{'CPU labels':>14}{'GPU labels':>14}"
          f"{'CPU train/epoch':>18}{'GPU train/epoch':>18}")
    if cpu_label and gpu_label:
        for h in (1, 3, 10, 30, 100):
            cl = hours_needed(h, cpu_label["audio_s_per_wall_s"])
            gl = hours_needed(h, gpu_label["audio_s_per_wall_s"])
            ct = hours_needed(h, cpu_tr["audio_s_per_wall_s"]) if cpu_tr else float("nan")
            gt = hours_needed(h, (gpu_trb or gpu_tr)["audio_s_per_wall_s"]) \
                if (gpu_trb or gpu_tr) else float("nan")
            print(f"{h:>8}{cl:>13.2f}h{gl:>13.2f}h{ct:>17.2f}h{gt:>17.2f}h")

    out = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads": threads,
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "partition": {
            "task": "benchmark harness",
            "device": "both",
            "frames": args.frames,
            "batch": args.batch,
        },
        "rows": rows,
    }
    p = RESULTS / "cpu_vs_gpu.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
