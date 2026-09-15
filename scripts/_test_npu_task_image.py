"""Regression tests for the self-contained NPU task image."""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"


def load_builder():
    path = ROOT / "scripts" / "33_build_npu_task_image.py"
    spec = importlib.util.spec_from_file_location("npu_task_image_builder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    builder = load_builder()
    image, manifest = builder.build_task_image(PROGRAM)
    program = json.loads((PROGRAM / "program.json").read_text(encoding="utf-8"))

    assert image[:8] == b"STEMNPU\0"
    header_bytes, total_bytes = struct.unpack_from("<II", image, 0x08)
    assert header_bytes == 256
    assert total_bytes == len(image)
    assert len(image) % 64 == 0
    assert struct.unpack_from("<HHHH", image, 0x10) == (1, 0, 1, 0)
    flags, command_count = struct.unpack_from("<II", image, 0x18)
    assert flags & 1
    assert command_count == 1869
    assert command_count * 16 == len((PROGRAM / "tile_commands.bin").read_bytes())

    header_crc = struct.unpack_from("<I", image, 0x7C)[0]
    header_copy = bytearray(image[:header_bytes])
    struct.pack_into("<I", header_copy, 0x7C, 0)
    assert zlib.crc32(header_copy) & 0xFFFF_FFFF == header_crc
    payload_crc = struct.unpack_from("<I", image, 0x78)[0]
    assert zlib.crc32(image[header_bytes:]) & 0xFFFF_FFFF == payload_crc

    names = [name for name, _ in builder.SECTION_FILES]
    previous_end = header_bytes
    for name in names:
        section = manifest["sections"][name]
        offset = int(section["offset"])
        size = int(section["bytes"])
        assert offset % 64 == 0
        assert offset >= previous_end
        assert offset + size <= len(image)
        previous_end = offset + size
        if name == "activation":
            assert image[offset:offset + size] == bytes(size)
        else:
            source = PROGRAM / str(section["source"])
            assert image[offset:offset + size] == source.read_bytes()

    input_td, output_td = struct.unpack_from("<HH", image, 0x70)
    assert input_td == int(program["entry"]["input_tensor"])
    assert output_td == int(program["entry"]["output_tensor"])

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        image_path = out / "task.bin"
        manifest_path = out / "task.json"
        image_path.write_bytes(image)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        rebuilt, rebuilt_manifest = builder.build_task_image(PROGRAM)
        assert rebuilt == image
        assert rebuilt_manifest == manifest

    print("task image: PASS")
    print(f"  bytes: {len(image):,}")
    print(f"  commands: {command_count:,}")
    print(f"  header CRC32: {header_crc:08x}")
    print(f"  payload CRC32: {payload_crc:08x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
