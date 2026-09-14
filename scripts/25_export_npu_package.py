"""Export the current causal U-Net into a deterministic NPU reference package.

The package is deliberately independent from PyTorch serialization.  It gives
the PS/RTL boundary a stable layer schedule, 64-byte-aligned INT8 weights,
INT32 biases, per-output-channel weight scales, and per-layer INT12 activation
scales.  The exporter also recomputes every layer's shape and MAC count so a
checkpoint whose graph no longer matches the frozen XC7Z020 design is rejected
before it reaches hardware.  This is the specialized-model reference package;
the programmable NPU task packer described in hardware/spec is a later stage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


t13 = _load("t13_npu_export", "13_ab_compare.py")
t09 = t13.t09
t11 = t13.t11

ALIGN = 64
ACT_BITS = 12
ACT_QMAX = 2 ** (ACT_BITS - 1) - 1
WEIGHT_QMAX = 127
PE_IN = 8
PE_OUT = 8
PL_CLOCK_HZ = 200_000_000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _portable_source(path: Path) -> str:
    """Record a reproducible source name without leaking a workstation path."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _align(blob: bytearray, alignment: int = ALIGN) -> int:
    padding = (-len(blob)) % alignment
    if padding:
        blob.extend(b"\0" * padding)
    return len(blob)


def _shape_schedule(frames: int) -> list[dict]:
    if frames <= 0 or frames % 8:
        raise ValueError("chunk frames must be a positive multiple of 8")
    t1, t2, t3 = frames // 2, frames // 4, frames // 8
    return [
        {"name": "enc0", "input": [2, 128, frames], "output": [32, 128, frames]},
        {"name": "enc1", "input": [32, 128, frames], "output": [64, 64, t1]},
        {"name": "enc2", "input": [64, 64, t1], "output": [96, 32, t2]},
        {"name": "enc3", "input": [96, 32, t2], "output": [128, 16, t3]},
        {"name": "bott", "input": [128, 16, t3], "output": [128, 16, t3]},
        {"name": "bott_blocks.0.conv", "input": [128, 16, t3],
         "output": [128, 16, t3], "residual": True},
        {"name": "bott_blocks.1.conv", "input": [128, 16, t3],
         "output": [128, 16, t3], "residual": True},
        {"name": "dec3", "input": [224, 32, t2], "output": [96, 32, t2],
         "upsample": "bott_x2", "skip": "enc2"},
        {"name": "dec2", "input": [160, 64, t1], "output": [64, 64, t1],
         "upsample": "dec3_x2", "skip": "enc1"},
        {"name": "dec1", "input": [96, 128, frames],
         "output": [32, 128, frames], "upsample": "dec2_x2", "skip": "enc0"},
        {"name": "out", "input": [32, 128, frames],
         "output": [4, 128, frames], "activation": "tanh"},
    ]


def _activation_rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("accumulator_ok_int32") is not True:
        raise ValueError("quantization report does not approve INT32 accumulators")
    return {row["layer"]: row for row in payload["activation_calibration"]}


def _pre_tanh_step(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration = payload.get("pre_tanh_calibration")
    if not calibration:
        raise ValueError(
            "quantization report lacks pre_tanh_calibration; run "
            "scripts/27_calibrate_tanh.py first")
    step = float(calibration["recommended_step_int12"])
    if not math.isfinite(step) or step <= 0:
        raise ValueError(f"invalid pre-tanh INT12 step: {step}")
    return step


def _next_output_steps(rows: dict[str, dict], pre_tanh_step: float) -> dict[str, float]:
    """Map each convolution to the quantizer used by its consumer.

    Residual/skip paths are explicitly requantized at their merge.  The final
    tanh is represented in signed Q1.11 before conversion to a UINT8 mask.
    """
    def step(name: str) -> float:
        return float(rows[name]["span_per_tensor"]) / ACT_QMAX

    return {
        "enc0": step("enc1"),
        "enc1": step("enc2"),
        "enc2": step("enc3"),
        "enc3": step("bott"),
        "bott": step("bott_blocks.0.conv"),
        "bott_blocks.0.conv": step("bott_blocks.1.conv"),
        "bott_blocks.1.conv": step("dec3"),
        "dec3": step("dec2"),
        "dec2": step("dec1"),
        "dec1": step("out"),
        "out": pre_tanh_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(ROOT / "models" /
                                               "student_bott2_mir1k_candidate.pt"))
    parser.add_argument("--quant-report", default=str(ROOT / "results" /
                                                       "quant_opt_bott2_mir1k_5m_model.json"))
    parser.add_argument("--out", default=str(ROOT / "hardware" / "generated" /
                                              "bott2_mir1k_v1"))
    parser.add_argument("--chunk-frames", type=int, default=16)
    args = parser.parse_args()

    ckpt = Path(args.ckpt).resolve()
    quant_report = Path(args.quant_report).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    net, checkpoint = t13.load_student(ckpt, "cpu")
    if int(checkpoint.get("n_bands", 128)) != 128:
        raise ValueError("frozen NPU supports exactly 128 bands")
    if int(checkpoint.get("bottleneck_blocks", 0)) != 2:
        raise ValueError("frozen NPU requires exactly two bottleneck residual blocks")
    if tuple(checkpoint.get("temporal_dilations", ())) != ():
        raise ValueError("frozen NPU does not support temporal dilation blocks")

    schedule = _shape_schedule(args.chunk_frames)
    qrows = _activation_rows(quant_report)
    pre_tanh_step = _pre_tanh_step(quant_report)
    output_steps = _next_output_steps(qrows, pre_tanh_step)
    modules = dict(net.named_modules())

    weight_blob = bytearray()
    bias_blob = bytearray()
    scale_blob = bytearray()
    total_params = 0
    total_macs = 0
    total_scheduled_cycles = 0
    layer_rows = []

    for entry in schedule:
        name = entry["name"]
        module = modules[name]
        if not isinstance(module, t09.CausalConv2d):
            raise TypeError(f"{name} is not a CausalConv2d")
        conv = module.conv
        weight = conv.weight.detach().cpu().numpy().astype(np.float32)
        bias = conv.bias.detach().cpu().numpy().astype(np.float32)
        flat = weight.reshape(weight.shape[0], -1)
        weight_step = np.maximum(np.max(np.abs(flat), axis=1), 1e-12) / WEIGHT_QMAX
        quant_weight = np.clip(np.rint(weight / weight_step[:, None, None, None]),
                               -WEIGHT_QMAX, WEIGHT_QMAX).astype(np.int8)

        input_step = float(qrows[name]["span_per_tensor"]) / ACT_QMAX
        output_step = output_steps[name]
        bias_i32_64 = np.rint(bias / (input_step * weight_step)).astype(np.int64)
        if np.any(bias_i32_64 > np.iinfo(np.int32).max) or np.any(
                bias_i32_64 < np.iinfo(np.int32).min):
            raise OverflowError(f"{name} bias exceeds INT32")
        bias_i32 = bias_i32_64.astype("<i4")
        requant = (input_step * weight_step / output_step).astype("<f4")

        weight_offset = _align(weight_blob)
        weight_blob.extend(quant_weight.tobytes(order="C"))
        bias_offset = _align(bias_blob)
        bias_blob.extend(bias_i32.tobytes(order="C"))
        scale_offset = _align(scale_blob)
        # Two float32 arrays per output channel: weight step then requant ratio.
        scale_blob.extend(weight_step.astype("<f4").tobytes(order="C"))
        scale_blob.extend(requant.tobytes(order="C"))

        out_elems = math.prod(entry["output"])
        taps = int(weight.shape[1] * weight.shape[2] * weight.shape[3])
        macs = out_elems * taps
        spatial = int(entry["output"][1] * entry["output"][2])
        scheduled_cycles = (spatial * math.ceil(conv.out_channels / PE_OUT) *
                            math.ceil(conv.in_channels / PE_IN) *
                            math.prod(conv.kernel_size))
        lane_utilization = macs / (scheduled_cycles * PE_IN * PE_OUT)
        params = int(weight.size + bias.size)
        total_macs += macs
        total_scheduled_cycles += scheduled_cycles
        total_params += params
        dequant = quant_weight.astype(np.float32) * weight_step[:, None, None, None]
        rel_rmse = float(np.sqrt(np.mean((dequant - weight) ** 2)) /
                         (np.sqrt(np.mean(weight ** 2)) + 1e-12))
        layer_row = {
            **entry,
            "op": "conv2d_causal",
            "in_channels": int(conv.in_channels),
            "out_channels": int(conv.out_channels),
            "kernel": list(conv.kernel_size),
            "stride": list(conv.stride),
            "dilation": list(conv.dilation),
            "groups": int(conv.groups),
            "time_pad_left": int(module.pad_t),
            "frequency_pad_each_side": int(module.pad_f),
            "post_op": ("tanh" if name == "out" else
                        "leaky_relu_0.1_then_residual_add" if entry.get("residual")
                        else "leaky_relu_0.1"),
            "params": params,
            "macs_per_chunk": macs,
            "scheduled_cycles_pout8_pin8": scheduled_cycles,
            "mac_lane_utilization": lane_utilization,
            "input_activation_step_int12": input_step,
            "output_activation_step_int12": output_step,
            "weight_offset": weight_offset,
            "weight_bytes": int(quant_weight.nbytes),
            "bias_offset": bias_offset,
            "bias_bytes": int(bias_i32.nbytes),
            "scale_offset": scale_offset,
            "scale_bytes": int(weight_step.nbytes + requant.nbytes),
            "scale_layout": "float32 weight_step[Cout], float32 requant[Cout]",
            "weight_layout": "OIHW int8",
            "weight_relative_rmse": rel_rmse,
        }
        if name == "out":
            layer_row["pre_tanh_activation_step_int12"] = pre_tanh_step
            layer_row["post_tanh_output_step_int12"] = 1.0 / ACT_QMAX
        layer_rows.append(layer_row)

    expected_params = sum(p.numel() for p in net.parameters())
    if total_params != expected_params:
        raise AssertionError(f"exported {total_params} params, model has {expected_params}")

    (out / "weights_int8.bin").write_bytes(weight_blob)
    (out / "bias_int32.bin").write_bytes(bias_blob)
    (out / "scales_f32.bin").write_bytes(scale_blob)

    audio_s = args.chunk_frames * t09.HOP / t09.SR
    lf_band = t11.lf_kill_band_for(250.0, checkpoint.get("band_layout", "legacy_log"),
                                   int(checkpoint.get("n_bands", 128)))
    manifest = {
        "schema": 1,
        "target": "XC7Z020-1",
        "checkpoint": _portable_source(ckpt),
        "checkpoint_sha256": _sha256(ckpt),
        "checkpoint_step": checkpoint.get("step"),
        "graph": {
            "sample_rate_hz": t09.SR,
            "n_fft": t09.N_FFT,
            "hop": t09.HOP,
            "fft_bins": t09.N_BINS,
            "bands": 128,
            "band_layout": checkpoint.get("band_layout", "legacy_log"),
            "chunk_frames": args.chunk_frames,
            "chunk_audio_ms": audio_s * 1000.0,
            "chunk_samples": args.chunk_frames * t09.HOP,
            "low_frequency_protection_hz": 250.0,
            "low_frequency_kill_bands": lf_band,
            "mask_mode": checkpoint.get("mask_mode", "independent"),
            "output_formula": "accompaniment = delayed_mix - button_gain * vocals",
        },
        "quantization": {
            "weights": "signed int8, symmetric, per output channel",
            "activations": "signed int12, symmetric, per tensor",
            "bias": "signed int32 in accumulator units",
            "accumulator": "signed int32",
            "mask": "uint8 after tanh and affine conversion",
            "pre_tanh_activation_step_int12": pre_tanh_step,
            "post_tanh_output_step_int12": 1.0 / ACT_QMAX,
            "calibration_report": _portable_source(quant_report),
        },
        "totals": {
            "parameters": total_params,
            "macs_per_chunk": total_macs,
            "gmac_per_audio_second": total_macs / audio_s / 1e9,
            "estimated_compute_ms_at_22_gmac_s": total_macs / 22e9 * 1000.0,
            "pe_array": {"pout": PE_OUT, "pin": PE_IN,
                         "mac_lanes": PE_OUT * PE_IN,
                         "clock_hz": PL_CLOCK_HZ},
            "scheduled_cycles_pout8_pin8": total_scheduled_cycles,
            "scheduled_ms_at_200mhz_ideal":
                total_scheduled_cycles / PL_CLOCK_HZ * 1000.0,
            "scheduled_ms_at_50pct_control_efficiency":
                total_scheduled_cycles / PL_CLOCK_HZ * 2000.0,
            "weights_blob_bytes": len(weight_blob),
            "bias_blob_bytes": len(bias_blob),
            "scales_blob_bytes": len(scale_blob),
            "ddr_weight_bandwidth_mb_s_reload_each_chunk":
                len(weight_blob) / audio_s / 1e6,
        },
        "layers": layer_rows,
    }
    manifest_path = out / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        stream.write("\n")

    header = f"""// Generated by scripts/25_export_npu_package.py. Do not edit.
#pragma once
#include <cstdint>
namespace stem_npu {{
constexpr std::uint32_t kManifestSchema = 1;
constexpr std::uint32_t kSampleRateHz = {t09.SR};
constexpr std::uint32_t kNfft = {t09.N_FFT};
constexpr std::uint32_t kHop = {t09.HOP};
constexpr std::uint32_t kBands = 128;
constexpr std::uint32_t kChunkFrames = {args.chunk_frames};
constexpr std::uint32_t kChunkSamples = {args.chunk_frames * t09.HOP};
constexpr std::uint32_t kLowFrequencyKillBands = {lf_band};
constexpr std::uint32_t kLayerCount = {len(layer_rows)};
constexpr std::uint32_t kPeInput = {PE_IN};
constexpr std::uint32_t kPeOutput = {PE_OUT};
constexpr std::uint32_t kMacLanes = {PE_IN * PE_OUT};
constexpr std::uint32_t kWeightBlobBytes = {len(weight_blob)};
constexpr std::uint32_t kBiasBlobBytes = {len(bias_blob)};
constexpr std::uint32_t kScaleBlobBytes = {len(scale_blob)};
}}  // namespace stem_npu
"""
    with (out / "npu_config.hpp").open(
            "w", encoding="utf-8", newline="\n") as stream:
        stream.write(header)

    print(f"wrote {manifest_path}")
    print(f"layers={len(layer_rows)} params={total_params:,} "
          f"MAC/chunk={total_macs:,} ({manifest['totals']['gmac_per_audio_second']:.3f} GMAC/s)")
    print(f"PE schedule={total_scheduled_cycles:,} cycles = "
          f"{total_scheduled_cycles / PL_CLOCK_HZ * 1000:.2f} ms ideal, "
          f"{total_scheduled_cycles / PL_CLOCK_HZ * 2000:.2f} ms at 50% control efficiency")
    print(f"weights={len(weight_blob):,} B bias={len(bias_blob):,} B "
          f"scales={len(scale_blob):,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
