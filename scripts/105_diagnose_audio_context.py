"""Offline context-loss experiment; float A/B audio is NOT a board capture.

Uses the deployed checkpoint and song, holds STFT/synthesis fixed, and compares
16-frame reset inference with continuous inference and retained input history.
Optional compiled integer reference checks distinguish context from quantization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import types

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_model(path, device):
    spec = importlib.util.spec_from_file_location("target", ROOT / "scripts/09_target_model.py")
    target = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = target.CausalSpectralUNet(bottleneck_blocks=2).to(device).eval()
    net.load_state_dict(checkpoint["model"])
    return target, net


@torch.no_grad()
def chunked(net, bands, history):
    outputs = []
    for start in range(0, bands.shape[-1], 16):
        left = max(0, start - history)
        outputs.append(net(bands[..., left:start + 16])[..., start-left:])
    return torch.cat(outputs, dim=-1)


def mask_metrics(mask, reference):
    # Exclude the shared cold start and unobserved low bands.
    a = mask[:, 44:, 128:]
    b = reference[:, 44:, 128:]
    delta = (a[..., 1:] - a[..., :-1]).abs()
    boundary = torch.arange(129, mask.shape[-1], device=a.device) % 16 == 0
    return {
        "mae_vs_continuous": (a-b).abs().mean().item(),
        "boundary_jump_mean": delta[..., boundary].mean().item(),
        "interior_jump_mean": delta[..., ~boundary].mean().item(),
        "mean_by_phase16": [a[..., p::16].mean().item() for p in range(16)],
    }


def enable_streaming(net, target):
    """Proof of concept only: retain each convolution's causal input halo.

    Chunk sizes must preserve all stride phases (16 is divisible by 8).
    Hardware must implement equivalent state/quantization, not run this Python.
    """
    def forward(layer, x):
        if layer.pad_t:
            past = getattr(layer, "stream_past", None)
            if past is None:
                past = x.new_zeros(*x.shape[:-1], layer.pad_t)
            combined = torch.cat((past, x), dim=-1)
            layer.stream_past = combined[..., -layer.pad_t:].clone()
            x = combined
        return layer.conv(F.pad(x, (0, 0, layer.pad_f, layer.pad_f)))

    for layer in net.modules():
        if isinstance(layer, target.CausalConv2d):
            layer.forward = types.MethodType(forward, layer)
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", type=Path, default=ROOT / "hardware/build/navigator_audio_media/music.mp3")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/student_bott2_mir1k_candidate.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "hardware/build/audio_context_diagnostic")
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--integer-blocks", type=int, default=0)
    args = ap.parse_args()
    if args.seconds < 2 or args.integer_blocks < 0:
        ap.error("Use at least 2 seconds and a non-negative integer block count")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target, net = load_model(args.checkpoint, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(args.audio),
                                   "-t", str(args.seconds), "-ar", "44100", "-ac", "2",
                                   "-f", "f32le", "pipe:1"])
    pcm = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).copy()
    mix = torch.from_numpy(pcm.T).to(device)
    window = torch.hann_window(1024, device=device)
    # Board starts with zero history, not reflect padding. Keep both paths identical.
    spectrum = torch.stft(mix, 1024, 256, window=window, center=True,
                          pad_mode="constant", return_complex=True)
    wa = torch.from_numpy(target.make_analysis_matrix()).to(device)
    gs = torch.from_numpy(target.make_synthesis_matrix()).to(device)
    bands = torch.einsum("fb,cft->cbt", wa, spectrum.abs())[None]
    count = bands.shape[-1]
    bands = F.pad(bands, (0, (-count) % 16))
    with torch.no_grad():
        full = (net(bands)[0, :2] + 1) / 2
        variants = {"continuous_float": full}
        for history in (0, 16, 32, 64, 96, 128):
            variants[f"history_{history}_float"] = (chunked(net, bands, history)[0, :2]+1)/2
        step = 0.07046897899364925
        quant_bands = (bands / step).round().clamp(-2048, 2047) * step
        np.save(args.out / "input_int12.npy", (bands[0]/step).round().clamp(-2048, 2047)
                .cpu().numpy().transpose(1, 2, 0).astype("<i2"))
        quant_reset = (chunked(net, quant_bands, 0)[0, :2]+1)/2
        variants["history_0_quantized_input_float_net"] = quant_reset
        streaming = enable_streaming(copy.deepcopy(net), target)
        stream_mask = (chunked(streaming, bands, 0)[0, :2]+1)/2
        variants["stateful_float"] = stream_mask
        stream_error = (stream_mask-full).abs().max().item()
        if stream_error > 0.0001:
            raise AssertionError(f"Stateful/continuous max mask error: {stream_error}")
        state_layout = {name: list(layer.stream_past.shape)
                        for name, layer in streaming.named_modules()
                        if hasattr(layer, "stream_past")}
        stationary = bands[..., 128:129].expand(-1, -1, -1, 256).contiguous()
        stable_full = (net(stationary)[0, :2]+1)/2
        stable_reset = (chunked(net, stationary, 0)[0, :2]+1)/2
    report = {
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "seconds": len(pcm)/44100,
        "scope": "Float controlled experiment, not recorded board output; no separation ground truth",
        "variants": {},
        "stationary_reset": mask_metrics(stable_reset, stable_full),
        "stationary_continuous": mask_metrics(stable_full, stable_full),
        "stateful_max_mask_error": stream_error,
        "stateful_history_shapes_nchw": state_layout,
        "stateful_int16_history_bytes_unpadded": sum(int(np.prod(s)) * 2 for s in state_layout.values()),
    }
    sf.write(args.out / "original.wav", pcm, 44100, subtype="FLOAT")
    for name, value in variants.items():
        stats = mask_metrics(value[..., :count], full[..., :count])
        mask = value[..., :count].clone()
        mask[:, :44] = 0
        binmask = torch.einsum("fb,cbt->cft", gs, mask).clamp(0, 1)
        vocal = torch.istft(spectrum * binmask, 1024, 256, window=window,
                             length=mix.shape[-1])
        residual = mix - vocal
        stats["vocal_clip_fraction"] = (vocal.abs() > 1).float().mean().item()
        stats["residual_clip_fraction"] = (residual.abs() > 1).float().mean().item()
        stats["residual_rms"] = residual.square().mean().sqrt().item()
        report["variants"][name] = stats
        if name in ("continuous_float", "history_0_float", "history_96_float"):
            sf.write(args.out / f"{name}_accompaniment.wav", residual.cpu().numpy().T,
                     44100, subtype="FLOAT")
    if args.integer_blocks:
        from npu_task_reference import TaskReference
        program_dir = ROOT / "hardware/generated/bott2_mir1k_v1_program"
        manifest = json.loads((program_dir / "task_image.json").read_text())
        program = json.loads((program_dir / "program.json").read_text())
        descriptor = program["tensors"][program["entry"]["input_tensor"]]
        offset = manifest["sections"]["activation"]["offset"] + descriptor["base_offset"]
        report["integer_blocks"] = []
        for block in range(8, 8 + args.integer_blocks):
            start = block * 16
            values = (bands[0, ..., start:start+16] / step).round().clamp(-2048, 2047)
            if values.shape[-1] != 16:
                raise ValueError("Audio too short for integer reference block")
            packed = np.zeros((128, 16, 8), dtype="<i2")
            packed[:, :, :2] = values.cpu().numpy().transpose(1, 2, 0)
            task = bytearray((program_dir / "task_image.bin").read_bytes())
            task[offset:offset+packed.nbytes] = packed.tobytes()
            ref = TaskReference(program_dir, bytes(task))
            ref.run()
            q = np.frombuffer(ref.output_bytes(), dtype="<i2").reshape(128, 16, 8)[:, :, :2]
            decoded = (np.clip(q.astype(np.float32), -2047, 2047)+2047)/4094
            expected = quant_reset[:, :, start:start+16].cpu().numpy().transpose(1, 2, 0)
            row = {"block": block, "metric_scope": "all 128 bands, vocal channels only",
                   "mask_mae_vs_quantized_input_float": float(np.abs(decoded-expected).mean()),
                   "mask_max_error": float(np.abs(decoded-expected).max())}
            report["integer_blocks"].append(row)
            print(json.dumps(row), flush=True)
        packed = np.zeros((128, 16, 8), dtype="<i2")
        values = (stationary[0, ..., :16]/step).round().clamp(-2048, 2047)
        packed[:, :, :2] = values.cpu().numpy().transpose(1, 2, 0)
        task = bytearray((program_dir / "task_image.bin").read_bytes())
        task[offset:offset+packed.nbytes] = packed.tobytes()
        ref = TaskReference(program_dir, bytes(task))
        ref.run()
        q = np.frombuffer(ref.output_bytes(), dtype="<i2").reshape(128, 16, 8)[44:, :, :2]
        mask = (np.clip(q.astype(np.float32), -2047, 2047)+2047)/4094
        report["integer_stationary_reset"] = {
            "boundary_jump_mean": float(np.abs(mask[:, 0]-mask[:, -1]).mean()),
            "interior_jump_mean": float(np.abs(np.diff(mask, axis=1)).mean()),
            "note": "One constant-spectrum task; compare last/first mask for repeated identical blocks",
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
