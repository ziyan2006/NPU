#!/usr/bin/env python3
"""Generate deterministic host and board audio test vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "hardware" / "build" / "audio_vectors"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    outputs = expected_outputs()
    if arguments.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise AssertionError(f"audio vector differs or is missing: {path}")
        print("audio vectors: PASS (deterministic)")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_bytes(content)
    print(f"audio vectors: generated {len(outputs)} files in {OUTPUT}")


if __name__ == "__main__":
    main()
