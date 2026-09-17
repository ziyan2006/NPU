#!/usr/bin/env python3
"""Generate deterministic host and board audio test vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "hardware" / "build" / "audio_vectors"
MP3_FIXTURES = ROOT / "software" / "audio_player" / "tests" / "fixtures"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mp3", action="store_true")
    arguments = parser.parse_args()
    outputs = generate_mp3_fixtures() if arguments.mp3 else expected_outputs()
    if arguments.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise AssertionError(f"audio vector differs or is missing: {path}")
        print("audio vectors: PASS (deterministic)")
        return
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    destination = "MP3 fixture/vector paths" if arguments.mp3 else str(OUTPUT)
    print(f"audio vectors: generated {len(outputs)} files in {destination}")


if __name__ == "__main__":
    main()
