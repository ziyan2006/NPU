#!/usr/bin/env python3
"""Describe one task for scripts/67_run_navigator_a9_task.tcl.

This is intentionally metadata-only: it does not copy task tensors or touch a
board.  The printed JSON can be passed to the JTAG command without manually
calculating the output address or FNV-1a checksum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fnv1a32(payload: bytes) -> int:
    value = 2166136261
    for byte in payload:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--program-dir", type=Path, required=True)
    args = parser.parse_args()

    image = args.image.read_bytes()
    golden = args.golden.read_bytes()
    if len(image) < 256 or len(image) % 64:
        raise ValueError("task image must be at least 256 bytes and 64-byte aligned")
    if not golden:
        raise ValueError("golden output must not be empty")
    program = json.loads((args.program_dir / "program.json").read_text(
        encoding="utf-8"))
    image_manifest = json.loads((args.program_dir / "task_image.json").read_text(
        encoding="utf-8"))
    output_tensor = int(program["entry"]["output_tensor"])
    output_offset = (int(image_manifest["sections"]["activation"]["offset"])
                     + int(program["tensors"][output_tensor]["base_offset"]))
    if output_offset < 0 or output_offset + len(golden) > len(image):
        raise ValueError("golden output range lies outside task image")
    result = {
        "task_bytes": len(image),
        "command_count": int(image_manifest["command_count"]),
        "output_offset": output_offset,
        "output_bytes": len(golden),
        "expected_fnv1a": f"0x{fnv1a32(golden):08X}",
    }
    print("NPU_A9_TASK_METADATA " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
