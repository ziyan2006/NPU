"""Build a tiny structurally-valid NPU task whose only command is END.

This is the first hardware smoke image for Navigator Z7020 bring-up.  It keeps
the required task sections (including the 4096-entry tanh LUT) so it exercises
the actual task loader, AXI read path, command fetch, command processor, and
completion CSR without touching accelerator data paths or persistent storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "hardware" / "build" / "npu_end_task"

MAGIC = b"STEMNPU\0"
HEADER_BYTES = 256
ALIGNMENT = 64
NONE_INDEX = 0xFFFF


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_end_task(activation_byte: int = 0) -> tuple[bytes, dict[str, object]]:
    if not 0 <= activation_byte <= 0xFF:
        raise ValueError("activation_byte must fit in one byte")
    # Section sizes obey npu_task_loader's structural checks.  Descriptors and
    # arenas are deliberately inert because END must not dereference them.
    section_sizes = {
        "commands": 16,
        "tensor_desc": 64,
        "operator_desc": 64,
        "quant_desc": 32,
        "segments": 8,
        "activation": 1,
        "weights": 1,
        "bias": 1,
        "quant_params": 16,
        "tanh_lut": 8192,
    }
    cursor = HEADER_BYTES
    sections: dict[str, dict[str, int]] = {}
    for name, size in section_sizes.items():
        cursor = align_up(cursor)
        sections[name] = {"offset": cursor, "bytes": size}
        cursor += size
    total_bytes = align_up(cursor)
    image = bytearray(total_bytes)

    # <BB7H: opcode, flags, tag, dst, src0, src1, opdesc, qdesc, imm
    image[256:272] = struct.pack(
        "<BB7H", 0x03, 0x02, 0xE001,
        NONE_INDEX, NONE_INDEX, NONE_INDEX, NONE_INDEX, NONE_INDEX, 0,
    )
    image[576] = activation_byte

    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    struct.pack_into("<II", header, 0x08, HEADER_BYTES, total_bytes)
    struct.pack_into("<HHHH", header, 0x10, 1, 0, 1, 0)
    struct.pack_into("<II", header, 0x18, 0, 1)
    header_offsets = {
        "commands": 0x20, "tensor_desc": 0x28, "operator_desc": 0x30,
        "quant_desc": 0x38, "segments": 0x40, "activation": 0x48,
        "weights": 0x50, "bias": 0x58, "quant_params": 0x60,
        "tanh_lut": 0x68,
    }
    for name, offset in header_offsets.items():
        struct.pack_into("<II", header, offset,
                         sections[name]["offset"], sections[name]["bytes"])
    struct.pack_into("<HHI", header, 0x70, 0, 0, 0)
    # The RTL deliberately leaves full CRC/SHA verification to trusted host
    # software.  Still populate both CRCs to make the blob diagnostic-friendly.
    struct.pack_into("<I", header, 0x78,
                     zlib.crc32(image[HEADER_BYTES:]) & 0xFFFF_FFFF)
    header_crc_input = bytearray(header)
    struct.pack_into("<I", header_crc_input, 0x7C, 0)
    struct.pack_into("<I", header, 0x7C,
                     zlib.crc32(header_crc_input) & 0xFFFF_FFFF)
    image[:HEADER_BYTES] = header
    metadata: dict[str, object] = {
        "purpose": "Navigator Z7020 JTAG-only END-task hardware smoke",
        "total_bytes": total_bytes,
        "command_count": 1,
        "command": {"opcode": "END", "flags": ["IRQ"], "tag": "0xe001"},
        "activation_byte": activation_byte,
        "sections": sections,
        "sha256": hashlib.sha256(image).hexdigest(),
    }
    return bytes(image), metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--activation-byte", type=lambda value: int(value, 0), default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    image, metadata = build_end_task(args.activation_byte)
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "task_image.bin"
    manifest_path = output / "task_image.json"
    image_path.write_bytes(image)
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {image_path} ({len(image):,} bytes, sha256={metadata['sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
