"""Real-tile bit-exact regression for the residual VEC_ADD RTL."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from npu_isa import (COMMAND_STRUCT, OPERATOR_STRUCT, QUANT_PARAM_STRUCT,
                     requantize, saturate)


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
OPERATOR_KEYS = (
    "src_td", "dst_td", "weight_td", "bias_td", "kh", "kw",
    "stride_h", "stride_w", "dilation_h", "dilation_w", "pad_top",
    "pad_bottom", "pad_left", "pad_right", "groups", "post_op_id",
    "tile_origin_h", "tile_origin_w", "tile_origin_cout", "tile_h",
    "tile_w", "tile_cout", "input_channel_start", "input_channel_count",
)


def signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def write_hex(path: Path, values: list[int], digits: int) -> None:
    path.write_text("".join(f"{value:0{digits}x}\n" for value in values),
                    encoding="ascii")


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def build_case(tile: dict, command_blob: bytes, descriptor_blob: bytes,
               quant_blob: bytes, program: dict, seed: int) -> dict:
    command_start = int(tile["vector_command"]) * COMMAND_STRUCT.size
    command = command_blob[command_start:command_start + COMMAND_STRUCT.size]
    descriptor_start = int(tile["operator_desc"]) * OPERATOR_STRUCT.size
    descriptor_raw = descriptor_blob[
        descriptor_start:descriptor_start + OPERATOR_STRUCT.size]
    descriptor = dict(zip(OPERATOR_KEYS, map(int, OPERATOR_STRUCT.unpack(
        descriptor_raw)), strict=True))

    quant_desc = program["quantization"][int(tile["vector_quant_desc"])]
    quant_start = int(quant_desc["param_offset"])
    quant_raw = quant_blob[quant_start:quant_start + QUANT_PARAM_STRUCT.size]
    multiplier, shift, clamp_min, clamp_max = QUANT_PARAM_STRUCT.unpack(
        quant_raw)

    rng = random.Random(seed)
    result_count = descriptor["tile_h"] * descriptor["tile_w"]
    activation_rows: list[int] = []
    output_rows: list[int] = []
    expected_rows: list[int] = []
    activation_addresses: list[int] = []
    cin_pad = (descriptor["input_channel_count"] + 7) & -8
    pixel_bytes = cin_pad * 2
    local_w = descriptor["tile_w"] - 1 + descriptor["kw"]
    channel_offset = descriptor["tile_origin_cout"] * 2

    for h in range(descriptor["tile_h"]):
        for w in range(descriptor["tile_w"]):
            activation_addresses.append(
                ((h + descriptor["pad_top"]) * local_w
                 + w + descriptor["pad_left"]) * pixel_bytes
                + channel_offset)
            activation_word = output_word = expected_word = 0
            for lane in range(descriptor["tile_cout"]):
                residual = rng.randint(-2048, 2047)
                conv = rng.randint(-2048, 2047)
                residual_q = requantize(residual, multiplier, shift,
                                        clamp_min, clamp_max)
                expected = saturate(conv + residual_q, 12)
                activation_word |= (residual & 0xffff) << (lane * 16)
                output_word |= (conv & 0xffff) << (lane * 16)
                expected_word |= (expected & 0xffff) << (lane * 16)
            activation_rows.append(activation_word)
            output_rows.append(output_word)
            expected_rows.append(expected_word)

    return {
        "command": int.from_bytes(command, "little"),
        "descriptor": int.from_bytes(descriptor_raw, "little"),
        "activation": activation_rows,
        "output": output_rows,
        "expected": expected_rows,
        "activation_address": activation_addresses,
        "quant": int.from_bytes(quant_raw, "little"),
        "vector_quant_address": int(tile["weight_bank_layout"]["vector_quant"]),
        "result_count": result_count,
    }


def main() -> None:
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")
    schedule = json.loads((PROGRAM / "tile_schedule.json").read_text())
    program = json.loads((PROGRAM / "program.json").read_text())
    command_blob = (PROGRAM / "tile_commands.bin").read_bytes()
    descriptor_blob = (PROGRAM / "tile_operator_desc.bin").read_bytes()
    quant_blob = (PROGRAM / "quant_params.bin").read_bytes()
    residual_tiles = [row for row in schedule["tiles"] if row["residual"]]
    cases = (residual_tiles[0], residual_tiles[1], residual_tiles[16])

    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        image = temp / "tb_npu_vec_add.vvp"
        run([iverilog, "-g2012", "-Wall", "-s", "tb_npu_vec_add",
             "-o", str(image), str(RTL / "include" / "npu_isa_pkg.sv"),
             str(RTL / "include" / "npu_dma_pkg.sv"),
             str(RTL / "npu_u32_mul_iter.sv"),
             str(RTL / "npu_requant_post.sv"),
             str(RTL / "npu_vec_add.sv"),
             str(RTL / "tb" / "tb_npu_vec_add.sv")])
        for index, tile in enumerate(cases):
            case = build_case(tile, command_blob, descriptor_blob, quant_blob,
                              program, 0x5645_4341 + index)
            paths = {}
            for name, digits in (("command", 32), ("descriptor", 128),
                                 ("activation", 32), ("output", 32),
                                 ("expected", 32),
                                 ("activation_address", 8), ("quant", 32)):
                path = temp / f"case{index}.{name}.hex"
                value = case[name]
                write_hex(path, value if isinstance(value, list) else [value],
                          digits)
                paths[name] = path
            output = run([
                vvp, str(image),
                f"+COMMAND_HEX={paths['command'].as_posix()}",
                f"+DESCRIPTOR_HEX={paths['descriptor'].as_posix()}",
                f"+ACTIVATION_HEX={paths['activation'].as_posix()}",
                f"+OUTPUT_HEX={paths['output'].as_posix()}",
                f"+EXPECTED_HEX={paths['expected'].as_posix()}",
                "+ACTIVATION_ADDRESS_HEX="
                f"{paths['activation_address'].as_posix()}",
                f"+QUANT_HEX={paths['quant'].as_posix()}",
                f"+RESULT_COUNT={case['result_count']}",
                "+VECTOR_QUANT_ADDRESS="
                f"{case['vector_quant_address']}"])
            if "npu_vec_add:" not in output or "PASS" not in output:
                raise RuntimeError(output)
            print(f"{tile['layer']} tile {tile['index']}: "
                  f"{case['result_count']} vectors PASS")
    print("Real-tile residual VEC_ADD RTL tests: PASS")


if __name__ == "__main__":
    main()
