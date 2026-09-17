#!/usr/bin/env python3
"""Generate deterministic host and board audio test vectors."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "hardware" / "build" / "audio_vectors"
MP3_FIXTURES = ROOT / "software" / "audio_player" / "tests" / "fixtures"
STEM_OUTPUT = OUTPUT / "stem_frontend"
STEM_BACKEND_OUTPUT = OUTPUT / "stem_backend"
STEM_FFT_SIZE = 1024
STEM_HOP = 256
STEM_FRAMES_PER_BLOCK = 16
STEM_BANDS = 128
STEM_LANES = 8
STEM_INPUT_SCALE = np.float32(0.07046897899364925)


def build_stereo_wav() -> bytes:
    frames = [
        (0, 0),
        (32767, -32768),
        (16384, -16384),
        (-1, 1),
        (1000, -2000),
        (-3000, 4000),
        (8192, 4096),
        (-8192, -4096),
    ]
    pcm = b"".join(struct.pack("<hh", *frame) for frame in frames)
    fmt = struct.pack("<HHIIHH", 1, 2, 44100, 176400, 4, 16)
    junk = b"abc"
    body = (
        b"WAVE"
        + b"JUNK" + struct.pack("<I", len(junk)) + junk + b"\x00"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def expected_outputs() -> dict[Path, bytes]:
    wav = build_stereo_wav()
    manifest = {
        "channels": 2,
        "frames": 8,
        "sample_rate_hz": 44100,
        "sha256": hashlib.sha256(wav).hexdigest(),
    }
    return {
        OUTPUT / "stereo_44100.wav": wav,
        OUTPUT / "manifest.json": (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("ascii"),
    }


def fnv1a32(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def run_ffmpeg(arguments: list[str]) -> None:
    preferred = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
    ffmpeg = (str(preferred) if preferred.is_file()
              else shutil.which("ffmpeg.exe") or shutil.which("ffmpeg"))
    if ffmpeg is None:
        raise RuntimeError("ffmpeg 9.0.1 is required for MP3 fixtures")
    version = subprocess.run(
        [ffmpeg, "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    if "ffmpeg version 9.0.1" not in version:
        raise RuntimeError(f"expected FFmpeg 9.0.1, found: {version}")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", *arguments],
                   check=True)


def generate_mp3_fixtures() -> dict[Path, bytes]:
    tone = (
        "aevalsrc=0.2511886432*sin(2*PI*1000*t)|"
        "0.2511886432*sin(2*PI*2000*t):s=44100:d=3"
    )
    with tempfile.TemporaryDirectory(prefix="stem_mp3_vectors_") as directory:
        temporary = Path(directory)
        outputs: dict[str, bytes] = {}
        encodings = {
            "test_44100_stereo_cbr.mp3": ["-b:a", "128k"],
            "test_44100_stereo_vbr.mp3": ["-q:a", "4"],
        }
        for name, codec_arguments in encodings.items():
            path = temporary / name
            run_ffmpeg([
                "-f", "lavfi", "-i", tone, "-map_metadata", "-1",
                "-ar", "44100", "-ac", "2", "-codec:a", "libmp3lame",
                *codec_arguments, "-id3v2_version", "0", "-write_id3v1", "0",
                "-y", str(path),
            ])
            outputs[name] = path.read_bytes()

        negative_encodings = {
            "invalid_layer2.mp2": [
                "-ar", "44100", "-ac", "2", "-codec:a", "mp2",
                "-b:a", "128k", "-f", "mp2",
            ],
            "invalid_48000_stereo.mp3": [
                "-ar", "48000", "-ac", "2", "-codec:a", "libmp3lame",
                "-b:a", "128k", "-f", "mp3",
            ],
            "invalid_44100_mono.mp3": [
                "-ar", "44100", "-ac", "1", "-codec:a", "libmp3lame",
                "-b:a", "128k", "-f", "mp3",
            ],
            "invalid_mpeg2_stereo.mp3": [
                "-ar", "22050", "-ac", "2", "-codec:a", "libmp3lame",
                "-b:a", "64k", "-f", "mp3",
            ],
        }
        negative_outputs = {}
        for name, codec_arguments in negative_encodings.items():
            path = temporary / name
            run_ffmpeg([
                "-f", "lavfi", "-i", tone, "-map_metadata", "-1",
                *codec_arguments, "-id3v2_version", "0", "-write_id3v1", "0",
                "-y", str(path),
            ])
            negative_outputs[name] = path.read_bytes()

        references = {}
        for name in outputs:
            pcm_path = temporary / f"{name}.s16le"
            run_ffmpeg([
                "-i", str(temporary / name), "-f", "s16le",
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                "-y", str(pcm_path),
            ])
            references[name] = {
                "bytes": pcm_path.stat().st_size,
                "frames": pcm_path.stat().st_size // 4,
                "fnv1a32": f"0x{fnv1a32(pcm_path.read_bytes()):08X}",
            }
        manifest = {
            "ffmpeg": "9.0.1",
            "duration_seconds": 3,
            "sample_rate_hz": 44100,
            "channels": 2,
            "source_amplitude_dbfs": -12.0,
            "left_frequency_hz": 1000,
            "right_frequency_hz": 2000,
            "files": {
                name: {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                for name, content in outputs.items()
            },
            "reference_pcm": references,
            "minimp3_raw_reference": {
                "commit": "ea99364f61c14656440e8d77e9c233ccf3124633",
                "frames": 134784,
                "test_44100_stereo_cbr.mp3_fnv1a32": "0xF0F5EDD2",
                "test_44100_stereo_vbr.mp3_fnv1a32": "0x23DF9715",
            },
            "negative_vectors": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in negative_outputs.items()
            },
        }
        result = {
            MP3_FIXTURES / name: content for name, content in outputs.items()
        }
        result.update({OUTPUT / name: content
                       for name, content in negative_outputs.items()})
        result[MP3_FIXTURES / "mp3_manifest.json"] = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("ascii")
        return result


def load_stem_filterbanks() -> tuple[np.ndarray, np.ndarray]:
    generator_path = ROOT / "scripts" / "98_generate_stem_constants.py"
    spec = importlib.util.spec_from_file_location("stem_constants", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load filterbank generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = module.load_filterbank_namespace()
    analysis = namespace["make_analysis_matrix"](
        STEM_BANDS, layout="legacy_log"
    )
    synthesis = namespace["make_synthesis_matrix"](
        STEM_BANDS, layout="legacy_log"
    )
    return (
        np.asarray(analysis, dtype=np.float32),
        np.asarray(synthesis, dtype=np.float32).T.copy(),
    )


def build_stem_pcm(name: str, blocks: int) -> np.ndarray:
    frame_count = blocks * STEM_FRAMES_PER_BLOCK * STEM_HOP + STEM_HOP
    pcm = np.zeros((frame_count, 2), dtype=np.int16)
    if name == "impulse":
        pcm[0] = (32767, -32768)
        pcm[257] = (-12000, 16000)
    elif name == "dc":
        pcm[:, 0] = 8192
        pcm[:, 1] = -4096
    elif name == "nyquist":
        alternating = np.where(np.arange(frame_count) & 1, -12000, 12000)
        pcm[:, 0] = alternating.astype(np.int16)
        pcm[:, 1] = (-alternating).astype(np.int16)
    elif name == "noise":
        rng = np.random.default_rng(0x5EED)
        pcm[:] = rng.integers(-12000, 12001, size=pcm.shape, dtype=np.int16)
    elif name == "clip":
        position = np.arange(frame_count, dtype=np.float64)
        pcm[:, 0] = np.rint(
            32767.0 * np.sin(2.0 * math.pi * 1000.0 * position / 44100.0)
        ).astype(np.int16)
        pcm[:, 1] = np.rint(
            32767.0 * np.sin(2.0 * math.pi * 2000.0 * position / 44100.0)
        ).astype(np.int16)
    elif name == "chirp20":
        position = np.arange(frame_count, dtype=np.float64)
        duration = frame_count / 44100.0
        time = position / 44100.0
        rate = (15000.0 - 50.0) / duration
        phase = 2.0 * math.pi * (50.0 * time + 0.5 * rate * time * time)
        pcm[:, 0] = np.rint(10000.0 * np.sin(phase)).astype(np.int16)
        pcm[:, 1] = np.rint(9000.0 * np.cos(phase * 0.731)).astype(np.int16)
    else:
        raise ValueError(f"unknown STEM vector: {name}")
    return pcm


def stem_reference(pcm: np.ndarray, blocks: int,
                   analysis: np.ndarray) -> dict[str, bytes]:
    window = (
        np.float32(0.5)
        - np.float32(0.5) * np.cos(
            np.float32(2.0 * math.pi)
            * np.arange(STEM_FFT_SIZE, dtype=np.float32)
            / np.float32(STEM_FFT_SIZE)
        )
    ).astype(np.float32)
    spectrum = np.empty(
        (blocks, 2, STEM_FRAMES_PER_BLOCK, STEM_FFT_SIZE // 2 + 1),
        dtype=np.complex64,
    )
    magnitude = np.empty(spectrum.shape, dtype=np.float32)
    bands = np.empty(
        (blocks, 2, STEM_FRAMES_PER_BLOCK, STEM_BANDS), dtype=np.float32
    )
    packed = np.zeros(
        (blocks, STEM_BANDS, STEM_FRAMES_PER_BLOCK, STEM_LANES),
        dtype=np.int16,
    )
    normalized = pcm.astype(np.float32) / np.float32(32768.0)
    for block in range(blocks):
        for frame in range(STEM_FRAMES_PER_BLOCK):
            center = (block * STEM_FRAMES_PER_BLOCK + frame) * STEM_HOP
            indices = np.arange(
                center - STEM_FFT_SIZE // 2,
                center + STEM_FFT_SIZE // 2,
                dtype=np.int64,
            )
            indices[indices < 0] *= -1
            for channel in range(2):
                samples = normalized[indices, channel] * window
                bins = np.fft.rfft(samples).astype(np.complex64)
                spectrum[block, channel, frame] = bins
                mag = np.abs(bins).astype(np.float32)
                magnitude[block, channel, frame] = mag
                for band in range(STEM_BANDS):
                    bands[block, channel, frame, band] = np.sum(
                        mag * analysis[:, band], dtype=np.float32
                    )
        quantized = np.rint(bands[block] / STEM_INPUT_SCALE)
        quantized = np.clip(quantized, -2048, 2047).astype(np.int16)
        packed[block, :, :, 0] = quantized[0].T
        packed[block, :, :, 1] = quantized[1].T
    return {
        "pcm.s16le": np.asarray(pcm, dtype="<i2").tobytes(order="C"),
        "spectrum.c64le": np.asarray(spectrum, dtype="<c8").tobytes(order="C"),
        "magnitude.f32le": np.asarray(magnitude, dtype="<f4").tobytes(order="C"),
        "bands.f32le": np.asarray(bands, dtype="<f4").tobytes(order="C"),
        "packed.s16le": np.asarray(packed, dtype="<i2").tobytes(order="C"),
    }


def build_backend_mask(name: str, blocks: int) -> np.ndarray:
    packed = np.zeros(
        (blocks, STEM_BANDS, STEM_FRAMES_PER_BLOCK, STEM_LANES),
        dtype=np.int16,
    )
    if name == "full":
        packed[..., :2] = 2047
    elif name == "random":
        rng = np.random.default_rng(0xBACC0E)
        packed[..., :2] = rng.integers(
            -4096, 4097, size=packed[..., :2].shape, dtype=np.int16
        )
        packed[0, 44, 0, 0] = -32768
    elif name != "zero":
        raise ValueError(f"unknown backend mask: {name}")
    return packed


def stem_backend_reference(spectrum: np.ndarray, packed: np.ndarray,
                           synthesis: np.ndarray) -> np.ndarray:
    blocks = spectrum.shape[0]
    total_output = blocks * STEM_FRAMES_PER_BLOCK * STEM_HOP - STEM_FFT_SIZE // 2
    accumulation = np.zeros((total_output + STEM_FFT_SIZE, 2), dtype=np.float64)
    weight = np.zeros(total_output + STEM_FFT_SIZE, dtype=np.float64)
    window = (
        0.5 - 0.5 * np.cos(
            2.0 * math.pi * np.arange(STEM_FFT_SIZE, dtype=np.float64)
            / STEM_FFT_SIZE
        )
    )
    for block in range(blocks):
        for frame in range(STEM_FRAMES_PER_BLOCK):
            center = (block * STEM_FRAMES_PER_BLOCK + frame) * STEM_HOP
            start = center - STEM_FFT_SIZE // 2
            valid_start = max(start, 0)
            source_start = valid_start - start
            valid_end = min(start + STEM_FFT_SIZE, accumulation.shape[0])
            source_end = source_start + valid_end - valid_start
            weight[valid_start:valid_end] += window[source_start:source_end] ** 2
            for channel in range(2):
                quantized = packed[block, :, frame, channel].astype(np.int32)
                band_mask = np.clip(np.abs(quantized) / 2047.0, 0.0, 1.0)
                band_mask[:44] = 0.0
                bin_mask = band_mask @ synthesis
                masked = spectrum[block, channel, frame] * bin_mask
                restored = np.fft.irfft(masked, n=STEM_FFT_SIZE)
                accumulation[valid_start:valid_end, channel] += (
                    restored[source_start:source_end]
                    * window[source_start:source_end]
                )
    output = np.zeros((total_output, 2), dtype=np.float32)
    valid = weight[:total_output] > 1.0e-12
    output[valid] = (
        accumulation[:total_output][valid] / weight[:total_output][valid, None]
    ).astype(np.float32)
    return output


def generate_stem_backend_vectors(analysis: np.ndarray,
                                  synthesis: np.ndarray) -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {}
    manifest: dict[str, object] = {
        "schema": 1,
        "startup_frames": 3584,
        "steady_frames": 4096,
        "low_bands_zeroed": 44,
        "cases": {},
    }
    cases = {
        "chirp20_zero": ("chirp20", "zero", 20),
        "chirp20_full": ("chirp20", "full", 20),
        "chirp20_random": ("chirp20", "random", 20),
        "impulse_full": ("impulse", "full", 1),
    }
    case_manifest = manifest["cases"]
    assert isinstance(case_manifest, dict)
    for case_name, (pcm_name, mask_name, blocks) in cases.items():
        pcm = build_stem_pcm(pcm_name, blocks)
        frontend = stem_reference(pcm, blocks, analysis)
        spectrum = np.frombuffer(frontend["spectrum.c64le"], dtype="<c8").reshape(
            blocks, 2, STEM_FRAMES_PER_BLOCK, STEM_FFT_SIZE // 2 + 1
        )
        packed = build_backend_mask(mask_name, blocks)
        vocal = stem_backend_reference(spectrum, packed, synthesis)
        delayed = pcm[:vocal.shape[0]].astype(np.float32) / np.float32(32768.0)
        payloads = {
            "spectrum.c64le": np.asarray(spectrum, dtype="<c8").tobytes(),
            "mask.s16le": np.asarray(packed, dtype="<i2").tobytes(),
            "vocal.f32le": np.asarray(vocal, dtype="<f4").tobytes(),
            "delay.f32le": np.asarray(delayed, dtype="<f4").tobytes(),
        }
        files = {}
        for suffix, payload in payloads.items():
            filename = f"{case_name}.{suffix}"
            outputs[STEM_BACKEND_OUTPUT / filename] = payload
            files[filename] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        case_manifest[case_name] = {
            "blocks": blocks,
            "output_frames": int(vocal.shape[0]),
            "files": files,
        }
    outputs[STEM_BACKEND_OUTPUT / "manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    return outputs


def generate_stem_vectors() -> dict[Path, bytes]:
    analysis, synthesis = load_stem_filterbanks()
    cases = {
        "impulse": 1,
        "dc": 1,
        "nyquist": 1,
        "noise": 1,
        "chirp20": 20,
        "clip": 1,
    }
    outputs: dict[Path, bytes] = {}
    manifest: dict[str, object] = {
        "schema": 1,
        "sample_rate_hz": 44100,
        "fft_size": STEM_FFT_SIZE,
        "hop_size": STEM_HOP,
        "frames_per_block": STEM_FRAMES_PER_BLOCK,
        "bands": STEM_BANDS,
        "nhwc8_lanes": STEM_LANES,
        "input_scale": float(STEM_INPUT_SCALE),
        "start_padding": "reflect",
        "cases": {},
    }
    case_manifest = manifest["cases"]
    assert isinstance(case_manifest, dict)
    for name, blocks in cases.items():
        pcm = build_stem_pcm(name, blocks)
        payloads = stem_reference(pcm, blocks, analysis)
        files = {}
        for suffix, payload in payloads.items():
            filename = f"{name}.{suffix}"
            outputs[STEM_OUTPUT / filename] = payload
            files[filename] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        case_manifest[name] = {
            "blocks": blocks,
            "pcm_frames": int(pcm.shape[0]),
            "files": files,
        }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    outputs[STEM_OUTPUT / "manifest.json"] = manifest_bytes
    outputs.update(generate_stem_backend_vectors(analysis, synthesis))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mp3", action="store_true")
    parser.add_argument("--stem", action="store_true")
    arguments = parser.parse_args()
    if arguments.mp3 and arguments.stem:
        parser.error("--mp3 and --stem are mutually exclusive")
    if arguments.mp3:
        outputs = generate_mp3_fixtures()
    elif arguments.stem:
        outputs = generate_stem_vectors()
    else:
        outputs = expected_outputs()
    if arguments.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise AssertionError(f"audio vector differs or is missing: {path}")
        print("audio vectors: PASS (deterministic)")
        return
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if arguments.mp3:
        destination = "MP3 fixture/vector paths"
    elif arguments.stem:
        destination = f"{STEM_OUTPUT} and {STEM_BACKEND_OUTPUT}"
    else:
        destination = str(OUTPUT)
    print(f"audio vectors: generated {len(outputs)} files in {destination}")


if __name__ == "__main__":
    main()
