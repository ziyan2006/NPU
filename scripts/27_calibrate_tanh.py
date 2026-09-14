"""Calibrate the final convolution before tanh for the NPU integer path.

The earlier quantization report observed convolution *inputs* only.  The final
TANH_LUT also needs the output range of ``net.out`` before torch.tanh.  This
script collects that missing range and appends it to the existing report while
preserving all held-out evaluation results.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


t17 = load_module("t17_tanh_calibration", "17_quantize_check.py")
t13 = t17.t13
t11 = t17.t11


class PreTanhObserver:
    def __init__(self, sample_cap_per_batch: int = 1 << 18):
        self.sample_cap_per_batch = sample_cap_per_batch
        self.minimum = math.inf
        self.maximum = -math.inf
        self.max_abs = 0.0
        self.samples: list[torch.Tensor] = []

    def hook(self, _module, _inputs, output) -> None:
        values = output.detach().float().cpu().flatten()
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.max_abs = max(self.max_abs, float(values.abs().max()))
        step = max(1, math.ceil(values.numel() / self.sample_cap_per_batch))
        self.samples.append(values[::step])


def candidate(values: torch.Tensor, span: float, bits: int = 12) -> dict:
    qmax = 2 ** (bits - 1) - 1
    step = max(float(span) / qmax, 1e-12)
    quantized = torch.clamp(torch.round(values / step), -qmax - 1, qmax)
    reconstructed = quantized * step
    ref = torch.tanh(values)
    got = torch.tanh(reconstructed)
    error = got - ref
    snr = 20.0 * math.log10(
        float(ref.square().sum().sqrt()) /
        (float(error.square().sum().sqrt()) + 1e-20))
    return {
        "span": float(span),
        "step_int12": step,
        "input_saturation_pct": float((values.abs() > span).float().mean() * 100.0),
        "tanh_output_snr_db": snr,
        "tanh_output_max_abs_error": float(error.abs().max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=Path,
        default=ROOT / "models" / "student_bott2_mir1k_candidate.pt")
    parser.add_argument(
        "--cache", type=Path,
        default=ROOT / "results" / "mir1k_gainmix_cache" / "train")
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "quant_opt_bott2_mir1k_5m_model.json")
    parser.add_argument("--files", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    if not args.cache.exists():
        raise SystemExit(f"calibration cache not found: {args.cache}")
    if not args.report.exists():
        raise SystemExit(f"quantization report not found: {args.report}")

    net, checkpoint = t13.load_student(args.ckpt, args.device)
    batches = t17.build_batches(args.cache, args.files, args.crop, args.batch)
    frontend = t11.frontend_spec(net)
    pre = (lambda x: t11.apply_frontend(x, frontend))
    observer = PreTanhObserver()
    handle = net.out.register_forward_hook(observer.hook)
    net.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                net(pre(batch.to(args.device)))
    finally:
        handle.remove()

    values = torch.cat(observer.samples)
    abs_values = values.abs()
    p99 = float(torch.quantile(abs_values, 0.99))
    p999 = float(torch.quantile(abs_values, 0.999))
    p9999 = float(torch.quantile(abs_values, 0.9999))
    candidates = {
        "max": candidate(values, observer.max_abs),
        "p99_99": candidate(values, p9999),
        "p99_9": candidate(values, p999),
    }
    result = {
        "observer": "output of CausalSpectralUNet.out before torch.tanh",
        "checkpoint_step": checkpoint.get("step"),
        "cache": args.cache.name,
        "files": args.files,
        "batches": len(batches),
        "batch": args.batch,
        "crop_frames": args.crop,
        "sample_count": int(values.numel()),
        "minimum": observer.minimum,
        "maximum": observer.maximum,
        "max_abs": observer.max_abs,
        "abs_percentiles": {"99": p99, "99.9": p999, "99.99": p9999},
        "candidates": candidates,
        "recommended": "max",
        "recommended_step_int12": candidates["max"]["step_int12"],
        "reason": "Matches the existing max-calibrated activation policy and clips no observed values.",
    }

    report = json.loads(args.report.read_text(encoding="utf-8"))
    report["pre_tanh_calibration"] = result
    with args.report.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False))
        stream.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"updated {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
