#!/usr/bin/env python3
"""Generate the resident STEM task payload and its derived C metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from npu_isa import TENSOR_STRUCT


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_DEFAULT = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
OUTPUT_DEFAULT = ROOT / "software" / "audio_player" / "generated"
EXPECTED_LAYOUT = "NHWC8"
EXPECTED_DTYPE = "INT12_IN_INT16"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def descriptor_from_binary(program_dir: Path, tensor_index: int) -> tuple[int, ...]:
    payload = (program_dir / "tensor_desc.bin").read_bytes()
    start = tensor_index * TENSOR_STRUCT.size
    end = start + TENSOR_STRUCT.size
    if start < 0 or end > len(payload):
        raise ValueError(f"tensor descriptor {tensor_index} is outside tensor_desc.bin")
    return TENSOR_STRUCT.unpack(payload[start:end])


def derive_tensor(program_dir: Path, program: dict[str, Any],
                  manifest: dict[str, Any], entry_name: str) -> dict[str, int]:
    tensor_index = int(program["entry"][entry_name])
    tensor = program["tensors"][tensor_index]
    descriptor = descriptor_from_binary(program_dir, tensor_index)
    base_offset = int(tensor["base_offset"])
    allocation_bytes = int(tensor["allocation_bytes"])
    shape_nhwc = tuple(map(int, tensor["shape_nhwc"]))
    strides = tuple(map(int, tensor["strides"]))
    if descriptor[0] != base_offset or descriptor[1] != allocation_bytes:
        raise ValueError(f"{entry_name} JSON and binary descriptor ranges differ")
    if tuple(descriptor[2:6]) != shape_nhwc or tuple(descriptor[6:10]) != strides:
        raise ValueError(f"{entry_name} JSON and binary descriptor geometry differs")
    if tensor["layout"] != EXPECTED_LAYOUT or tensor["dtype"] != EXPECTED_DTYPE:
        raise ValueError(f"{entry_name} must use NHWC8 INT12-in-INT16 storage")
    n, frequency, time, channels = shape_nhwc
    padded_channels = (channels + 7) // 8 * 8
    storage_bytes = n * frequency * time * padded_channels * 2
    if allocation_bytes != storage_bytes:
        raise ValueError(
            f"{entry_name} allocation {allocation_bytes} != NHWC8 extent {storage_bytes}"
        )

    activation = manifest["sections"]["activation"]
    activation_offset = int(activation["offset"])
    activation_bytes = int(activation["bytes"])
    absolute_offset = activation_offset + base_offset
    if base_offset < 0 or base_offset + allocation_bytes > activation_bytes:
        raise ValueError(f"{entry_name} lies outside the activation arena")
    return {
        "tensor": tensor_index,
        "offset": absolute_offset,
        "bytes": allocation_bytes,
        "channels": channels,
        "frequency": frequency,
        "time": time,
        "padded_channels": padded_channels,
    }


def validate_sections(manifest: dict[str, Any], image: bytes) -> None:
    previous_end = int(manifest["header_bytes"])
    for name, section in manifest["sections"].items():
        offset = int(section["offset"])
        byte_count = int(section["bytes"])
        if offset % int(manifest["alignment_bytes"]):
            raise ValueError(f"task section {name} is not aligned")
        if offset < previous_end or offset + byte_count > len(image):
            raise ValueError(f"task section {name} range is invalid")
        actual_sha = sha256(image[offset:offset + byte_count])
        if actual_sha != section["sha256"]:
            raise ValueError(f"task section {name} SHA-256 mismatch")
        previous_end = offset + byte_count


def c_byte_array(symbol: str, payload: bytes) -> str:
    rows = []
    for offset in range(0, len(payload), 12):
        values = ", ".join(f"0x{value:02x}" for value in payload[offset:offset + 12])
        rows.append(f"    {values},")
    return "\n".join((f"const uint8_t {symbol}[{len(payload)}] = {{", *rows, "};"))


def expected_outputs(program_dir: Path, output_dir: Path) -> dict[Path, bytes]:
    program = load_json(program_dir / "program.json")
    manifest = load_json(program_dir / "task_image.json")
    image = (program_dir / "task_image.bin").read_bytes()
    task_sha = sha256(image)
    if len(image) != int(manifest["total_bytes"]):
        raise ValueError("task image length differs from manifest")
    if task_sha != manifest["sha256"]:
        raise ValueError("task image SHA-256 differs from manifest")
    alignment = int(manifest["alignment_bytes"])
    if len(image) % alignment:
        raise ValueError("task image is not aligned")
    validate_sections(manifest, image)

    input_tensor = derive_tensor(program_dir, program, manifest, "input_tensor")
    output_tensor = derive_tensor(program_dir, program, manifest, "output_tensor")
    activation = manifest["sections"]["activation"]
    activation_offset = int(activation["offset"])
    activation_bytes = int(activation["bytes"])
    command_count = int(manifest["command_count"])
    command_bytes = int(manifest["sections"]["commands"]["bytes"])
    if command_count * int(manifest["isa"]["command_bytes"]) != command_bytes:
        raise ValueError("command count and command section size differ")

    metadata = (
        "/* Generated by scripts/99_generate_stem_task_payload.py; do not edit.\n"
        f" * task-image-sha256: {task_sha}\n"
        f" * source-manifest-sha256: {manifest['source_manifest_sha256']}\n"
        " */\n"
        "#ifndef STEM_TASK_METADATA_H\n"
        "#define STEM_TASK_METADATA_H\n\n"
        "#include \"stem_contract.h\"\n\n"
        f"#define STEM_TASK_ALIGNMENT_BYTES {alignment}u\n"
        f"#define STEM_TASK_IMAGE_BYTES {len(image)}u\n"
        f"#define STEM_TASK_COMMAND_COUNT {command_count}u\n"
        f"#define STEM_TASK_ACTIVATION_OFFSET {activation_offset}u\n"
        f"#define STEM_TASK_ACTIVATION_BYTES {activation_bytes}u\n"
        f"#define STEM_TASK_INPUT_TENSOR {input_tensor['tensor']}u\n"
        f"#define STEM_TASK_INPUT_OFFSET {input_tensor['offset']}u\n"
        f"#define STEM_TASK_INPUT_BYTES {input_tensor['bytes']}u\n"
        f"#define STEM_TASK_INPUT_CHANNELS {input_tensor['channels']}u\n"
        f"#define STEM_TASK_INPUT_FREQUENCY_BANDS {input_tensor['frequency']}u\n"
        f"#define STEM_TASK_INPUT_TIME_FRAMES {input_tensor['time']}u\n"
        f"#define STEM_TASK_INPUT_PADDED_CHANNELS {input_tensor['padded_channels']}u\n"
        f"#define STEM_TASK_OUTPUT_TENSOR {output_tensor['tensor']}u\n"
        f"#define STEM_TASK_OUTPUT_OFFSET {output_tensor['offset']}u\n"
        f"#define STEM_TASK_OUTPUT_BYTES {output_tensor['bytes']}u\n"
        f"#define STEM_TASK_OUTPUT_CHANNELS {output_tensor['channels']}u\n"
        f"#define STEM_TASK_OUTPUT_FREQUENCY_BANDS {output_tensor['frequency']}u\n"
        f"#define STEM_TASK_OUTPUT_TIME_FRAMES {output_tensor['time']}u\n"
        f"#define STEM_TASK_OUTPUT_PADDED_CHANNELS {output_tensor['padded_channels']}u\n"
        f"#define STEM_TASK_IMAGE_SHA256 \"{task_sha}\"\n"
        f"#define STEM_TASK_SOURCE_MANIFEST_SHA256 \"{manifest['source_manifest_sha256']}\"\n\n"
        "_Static_assert((STEM_TASK_IMAGE_BYTES % STEM_TASK_ALIGNMENT_BYTES) == 0u,\n"
        "               \"task image must be 64-byte aligned\");\n"
        "_Static_assert(STEM_TASK_INPUT_BYTES == STEM_INPUT_STORAGE_BYTES,\n"
        "               \"input tensor extent differs from the STEM contract\");\n"
        "_Static_assert(STEM_TASK_OUTPUT_BYTES == STEM_OUTPUT_STORAGE_BYTES,\n"
        "               \"output tensor extent differs from the STEM contract\");\n"
        "_Static_assert(STEM_TASK_INPUT_CHANNELS == STEM_INPUT_CHANNELS &&\n"
        "               STEM_TASK_OUTPUT_CHANNELS == STEM_OUTPUT_CHANNELS,\n"
        "               \"task channel counts differ from the STEM contract\");\n"
        "_Static_assert(STEM_TASK_INPUT_FREQUENCY_BANDS == STEM_BAND_COUNT &&\n"
        "               STEM_TASK_OUTPUT_FREQUENCY_BANDS == STEM_BAND_COUNT,\n"
        "               \"task frequency extent differs from the STEM contract\");\n"
        "_Static_assert(STEM_TASK_INPUT_TIME_FRAMES == STEM_BLOCK_FRAMES &&\n"
        "               STEM_TASK_OUTPUT_TIME_FRAMES == STEM_BLOCK_FRAMES,\n"
        "               \"task time extent differs from the STEM contract\");\n"
        "_Static_assert(STEM_TASK_INPUT_PADDED_CHANNELS == STEM_NHWC8_LANES &&\n"
        "               STEM_TASK_OUTPUT_PADDED_CHANNELS == STEM_NHWC8_LANES,\n"
        "               \"task tensors must use NHWC8 storage\");\n"
        "_Static_assert(STEM_TASK_INPUT_OFFSET >= STEM_TASK_ACTIVATION_OFFSET &&\n"
        "               STEM_TASK_INPUT_OFFSET + STEM_TASK_INPUT_BYTES <=\n"
        "               STEM_TASK_ACTIVATION_OFFSET + STEM_TASK_ACTIVATION_BYTES,\n"
        "               \"input tensor lies outside activation arena\");\n"
        "_Static_assert(STEM_TASK_OUTPUT_OFFSET >= STEM_TASK_ACTIVATION_OFFSET &&\n"
        "               STEM_TASK_OUTPUT_OFFSET + STEM_TASK_OUTPUT_BYTES <=\n"
        "               STEM_TASK_ACTIVATION_OFFSET + STEM_TASK_ACTIVATION_BYTES,\n"
        "               \"output tensor lies outside activation arena\");\n"
        "_Static_assert(STEM_TASK_OUTPUT_OFFSET + STEM_TASK_OUTPUT_BYTES <=\n"
        "               STEM_TASK_IMAGE_BYTES,\n"
        "               \"output tensor lies outside task image\");\n\n"
        "#endif /* STEM_TASK_METADATA_H */\n"
    )
    payload_header = (
        "/* Generated by scripts/99_generate_stem_task_payload.py; do not edit. */\n"
        "#ifndef STEM_TASK_PAYLOAD_H\n"
        "#define STEM_TASK_PAYLOAD_H\n\n"
        "#include <stddef.h>\n#include <stdint.h>\n"
        "#include \"stem_task_metadata.h\"\n\n"
        "extern const uint8_t stem_task_payload[STEM_TASK_IMAGE_BYTES];\n"
        "extern const size_t stem_task_payload_bytes;\n\n"
        "#endif /* STEM_TASK_PAYLOAD_H */\n"
    )
    payload_source = (
        "/* Generated by scripts/99_generate_stem_task_payload.py; do not edit.\n"
        f" * task-image-sha256: {task_sha}\n"
        " */\n"
        "#include \"stem_task_payload.h\"\n\n"
        "#if defined(__GNUC__)\n"
        "__attribute__((aligned(STEM_TASK_ALIGNMENT_BYTES)))\n"
        "#endif\n"
        f"{c_byte_array('stem_task_payload', image)}\n\n"
        "const size_t stem_task_payload_bytes = sizeof(stem_task_payload);\n"
    )
    return {
        output_dir / "stem_task_metadata.h": metadata.encode("ascii"),
        output_dir / "stem_task_payload.h": payload_header.encode("ascii"),
        output_dir / "stem_task_payload.c": payload_source.encode("ascii"),
    }


def write_or_check(outputs: dict[Path, bytes], check: bool) -> None:
    for path, expected in outputs.items():
        if check:
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit(f"out of date: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-dir", type=Path, default=PROGRAM_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = expected_outputs(args.program_dir.resolve(), args.output_dir.resolve())
    write_or_check(outputs, args.check)
    action = "verified" if args.check else "generated"
    print(f"stem task payload: {action} {len(outputs)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
