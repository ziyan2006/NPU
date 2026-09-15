"""Real-command AXI regression for nearest-neighbor UPSAMPLE2X RTL."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from npu_isa import COMMAND_STRUCT, Command, Opcode, TENSOR_STRUCT


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"


def write_hex(path: Path, values: list[int], digits: int) -> None:
    path.write_text("".join(f"{value:0{digits}x}\n" for value in values),
                    encoding="ascii")


def beats(payload: bytes) -> list[int]:
    assert len(payload) % 8 == 0
    return [int.from_bytes(payload[index:index + 8], "little")
            for index in range(0, len(payload), 8)]


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def build_case(command: Command, command_raw: bytes, tensor_blob: bytes,
               program: dict, seed: int) -> dict:
    source = program["tensors"][command.src0_td]
    destination = program["tensors"][command.dst_td]
    source_raw = tensor_blob[
        command.src0_td * TENSOR_STRUCT.size:
        (command.src0_td + 1) * TENSOR_STRUCT.size]
    destination_raw = tensor_blob[
        command.dst_td * TENSOR_STRUCT.size:
        (command.dst_td + 1) * TENSOR_STRUCT.size]
    channels, height, width = map(int, source["shape_chw"])
    padded_channels = (channels + 7) & -8
    rng = random.Random(seed)
    source_bytes = bytearray()
    for _ in range(height * width * padded_channels):
        source_bytes += rng.randint(-2048, 2047).to_bytes(
            2, "little", signed=True)

    pixel_bytes = padded_channels * 2
    row_bytes = width * pixel_bytes
    expected_bytes = bytearray()
    for row in range(height):
        row_start = row * row_bytes
        output_row = bytearray()
        for column in range(width):
            start = row_start + column * pixel_bytes
            pixel = source_bytes[start:start + pixel_bytes]
            output_row += pixel
            output_row += pixel
        expected_bytes += output_row
        expected_bytes += output_row
    assert len(source_bytes) == int(source["allocation_bytes"])
    assert len(expected_bytes) == int(destination["allocation_bytes"])
    return {
        "command": int.from_bytes(command_raw, "little"),
        "source_desc": int.from_bytes(source_raw, "little"),
        "destination_desc": int.from_bytes(destination_raw, "little"),
        "source": beats(source_bytes),
        "expected": beats(expected_bytes),
        "source_row_beats": row_bytes // 8,
        "destination_row_beats": row_bytes // 4,
        "source_total_beats": len(source_bytes) // 8,
        "destination_total_beats": len(expected_bytes) // 8,
    }


def main() -> None:
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")
    program = json.loads((PROGRAM / "program.json").read_text())
    command_blob = (PROGRAM / "tile_commands.bin").read_bytes()
    tensor_blob = (PROGRAM / "tensor_desc.bin").read_bytes()
    commands = [Command.unpack(command_blob[index:index + COMMAND_STRUCT.size])
                for index in range(0, len(command_blob), COMMAND_STRUCT.size)]
    cases = [(index, command) for index, command in enumerate(commands)
             if command.opcode == Opcode.UPSAMPLE2X]
    assert len(cases) == 3

    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        image = temp / "tb_npu_upsample2x.vvp"
        run([iverilog, "-g2012", "-Wall", "-s", "tb_npu_upsample2x",
             "-o", str(image), str(RTL / "include" / "npu_isa_pkg.sv"),
             str(RTL / "npu_u32_mul_iter.sv"),
             str(RTL / "npu_upsample2x.sv"),
             str(RTL / "tb" / "tb_npu_upsample2x.sv")])
        for case_index, (command_index, command) in enumerate(cases):
            command_start = command_index * COMMAND_STRUCT.size
            case = build_case(
                command,
                command_blob[command_start:command_start + COMMAND_STRUCT.size],
                tensor_blob, program, 0x5550_3200 + case_index)
            paths = {}
            for name, digits in (("command", 32), ("source_desc", 128),
                                 ("destination_desc", 128), ("source", 16),
                                 ("expected", 16)):
                path = temp / f"case{case_index}.{name}.hex"
                value = case[name]
                write_hex(path, value if isinstance(value, list) else [value],
                          digits)
                paths[name] = path
            output = run([
                vvp, str(image),
                f"+COMMAND_HEX={paths['command'].as_posix()}",
                f"+SOURCE_DESC_HEX={paths['source_desc'].as_posix()}",
                "+DESTINATION_DESC_HEX="
                f"{paths['destination_desc'].as_posix()}",
                f"+SOURCE_HEX={paths['source'].as_posix()}",
                f"+EXPECTED_HEX={paths['expected'].as_posix()}",
                "+ACTIVATION_BASE=0000000010000000",
                f"+SOURCE_ROW_BEATS={case['source_row_beats']}",
                "+DESTINATION_ROW_BEATS="
                f"{case['destination_row_beats']}",
                f"+SOURCE_TOTAL_BEATS={case['source_total_beats']}",
                "+DESTINATION_TOTAL_BEATS="
                f"{case['destination_total_beats']}"])
            if "npu_upsample2x:" not in output or "PASS" not in output:
                raise RuntimeError(output)
            source = program["tensors"][command.src0_td]
            destination = program["tensors"][command.dst_td]
            print(f"command {command_index}: {source['shape_chw']} -> "
                  f"{destination['shape_chw']} PASS")
    print("All real UPSAMPLE2X commands passed AXI RTL regression")


if __name__ == "__main__":
    main()
