"""Build the self-contained task image consumed by the RTL task loader.

The image uses one fixed 256-byte little-endian header followed by 64-byte
aligned sections.  The mutable activation arena is zero initialized; software
validates the image CRC first and then writes the input tensor into that arena.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROGRAM_DEFAULT = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"

MAGIC = b"STEMNPU\0"
HEADER_BYTES = 256
FORMAT_MAJOR = 1
FORMAT_MINOR = 0
ALIGNMENT = 64

SECTION_FILES = (
    ("commands", "tile_commands.bin"),
    ("tensor_desc", "tensor_desc.bin"),
    ("operator_desc", "tile_operator_desc.bin"),
    ("quant_desc", "quant_desc.bin"),
    ("segments", "segments.bin"),
    ("activation", None),
    ("weights", "weights_o8i8.bin"),
    ("bias", "bias_int32.bin"),
    ("quant_params", "quant_params.bin"),
    ("tanh_lut", "tanh_lut_int12.bin"),
)


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def u32_pair(buffer: bytearray, offset: int, low: int, high: int) -> None:
    struct.pack_into("<II", buffer, offset, low, high)


def build_task_image(program_dir: Path) -> tuple[bytes, dict]:
    program = json.loads((program_dir / "program.json").read_text(encoding="utf-8"))
    analysis = json.loads((program_dir / "analysis.json").read_text(encoding="utf-8"))

    payloads: dict[str, bytes] = {}
    for name, filename in SECTION_FILES:
        if name == "activation":
            size = int(analysis["memory"]["mutable_arena_bytes_no_reuse"])
            payloads[name] = bytes(size)
        else:
            assert filename is not None
            payloads[name] = (program_dir / filename).read_bytes()

    command_bytes = payloads["commands"]
    if len(command_bytes) == 0 or len(command_bytes) % 16:
        raise ValueError("tile command stream must contain complete 16-byte commands")

    cursor = HEADER_BYTES
    sections: dict[str, dict[str, int | str | bool]] = {}
    for name, filename in SECTION_FILES:
        cursor = align_up(cursor)
        payload = payloads[name]
        sections[name] = {
            "offset": cursor,
            "bytes": len(payload),
            "mutable": name == "activation",
            "source": filename or "zero-initialized activation arena",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        cursor += len(payload)
    total_bytes = align_up(cursor)

    image = bytearray(total_bytes)
    for name, _ in SECTION_FILES:
        section = sections[name]
        start = int(section["offset"])
        payload = payloads[name]
        image[start:start + len(payload)] = payload

    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    u32_pair(header, 0x08, HEADER_BYTES, total_bytes)
    struct.pack_into(
        "<HHHH", header, 0x10, FORMAT_MAJOR, FORMAT_MINOR,
        int(program["isa"]["major"]), int(program["isa"]["minor"]),
    )
    u32_pair(header, 0x18, 1, len(command_bytes) // 16)  # bit 0: build-image CRC

    header_offsets = {
        "commands": 0x20,
        "tensor_desc": 0x28,
        "operator_desc": 0x30,
        "quant_desc": 0x38,
        "segments": 0x40,
        "activation": 0x48,
        "weights": 0x50,
        "bias": 0x58,
        "quant_params": 0x60,
        "tanh_lut": 0x68,
    }
    for name, offset in header_offsets.items():
        section = sections[name]
        u32_pair(header, offset, int(section["offset"]), int(section["bytes"]))

    struct.pack_into(
        "<HHI", header, 0x70,
        int(program["entry"]["input_tensor"]),
        int(program["entry"]["output_tensor"]), 0,
    )
    manifest_digest = bytes.fromhex(program["source"]["manifest_sha256"])
    if len(manifest_digest) != 32:
        raise ValueError("source manifest digest must be SHA-256")
    header[0x80:0xA0] = manifest_digest

    # payload_crc covers the pristine build image after the header.  Software
    # checks it before patching the mutable activation arena.
    payload_crc = zlib.crc32(image[HEADER_BYTES:]) & 0xFFFF_FFFF
    struct.pack_into("<I", header, 0x78, payload_crc)
    header_for_crc = bytearray(header)
    struct.pack_into("<I", header_for_crc, 0x7C, 0)
    header_crc = zlib.crc32(header_for_crc) & 0xFFFF_FFFF
    struct.pack_into("<I", header, 0x7C, header_crc)
    image[:HEADER_BYTES] = header

    image_digest = hashlib.sha256(image).hexdigest()
    metadata = {
        "schema": 1,
        "format": {"major": FORMAT_MAJOR, "minor": FORMAT_MINOR},
        "isa": program["isa"],
        "magic_ascii": MAGIC.rstrip(b"\0").decode("ascii"),
        "header_bytes": HEADER_BYTES,
        "total_bytes": total_bytes,
        "alignment_bytes": ALIGNMENT,
        "command_count": len(command_bytes) // 16,
        "entry": program["entry"],
        "payload_crc32": f"{payload_crc:08x}",
        "header_crc32": f"{header_crc:08x}",
        "sha256": image_digest,
        "source_manifest_sha256": program["source"]["manifest_sha256"],
        "sections": sections,
    }
    return bytes(image), metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=PROGRAM_DEFAULT)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    program_dir = args.program.resolve()
    image_path = args.image.resolve() if args.image else program_dir / "task_image.bin"
    manifest_path = (args.manifest.resolve() if args.manifest
                     else program_dir / "task_image.json")
    image, metadata = build_task_image(program_dir)
    manifest_text = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not image_path.exists() or image_path.read_bytes() != image:
            raise SystemExit(f"out of date: {image_path}")
        if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != manifest_text:
            raise SystemExit(f"out of date: {manifest_path}")
        print(f"task image is reproducible: {image_path}")
        return 0

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image)
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    print(f"wrote {image_path} ({len(image):,} bytes)")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
