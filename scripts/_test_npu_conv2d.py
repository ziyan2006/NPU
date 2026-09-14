"""Bit-exact RTL tests for real tile descriptors and O8I8 weights."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from npu_isa import OPERATOR_STRUCT


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


def signed32(value: int) -> int:
    return signed(value, 32)


def write_hex(path: Path, values: list[int], digits: int) -> None:
    path.write_text("".join(f"{value:0{digits}x}\n" for value in values),
                    encoding="ascii")


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def build_case(tile: dict, descriptor_blob: bytes, weight_blob: bytes,
               program: dict, seed: int) -> dict[str, list[int] | bytes]:
    descriptor_index = int(tile["operator_desc"])
    raw_descriptor = descriptor_blob[
        descriptor_index * 64:(descriptor_index + 1) * 64]
    fields = OPERATOR_STRUCT.unpack(raw_descriptor)
    descriptor = dict(zip(OPERATOR_KEYS, map(int, fields), strict=True))

    kh, kw = descriptor["kh"], descriptor["kw"]
    sh, sw = descriptor["stride_h"], descriptor["stride_w"]
    th, tw = descriptor["tile_h"], descriptor["tile_w"]
    cin, cout = (descriptor["input_channel_count"],
                 descriptor["tile_cout"])
    cin_pad = (cin + 7) // 8 * 8
    cin_blocks = cin_pad // 8
    local_h = (th - 1) * sh + kh
    local_w = (tw - 1) * sw + kw
    region = tile["input_region"]
    assert [local_h, local_w] == [int(region["local_h"]),
                                  int(region["local_w"])]

    rng = random.Random(seed)
    activation = [0] * (local_h * local_w * cin_pad)
    h0, h1 = int(region["pad_top"]), local_h - int(region["pad_bottom"])
    w0, w1 = int(region["pad_left"]), local_w - int(region["pad_right"])
    for h in range(h0, h1):
        for w in range(w0, w1):
            base = (h * local_w + w) * cin_pad
            for channel in range(cin):
                activation[base + channel] = rng.randint(-2048, 2047)

    activation_bytes = b"".join(
        int(value).to_bytes(2, "little", signed=True) for value in activation)
    activation_rows = [
        int.from_bytes(activation_bytes[offset:offset + 16], "little")
        for offset in range(0, len(activation_bytes), 16)
    ]

    layer_operator = program["operators"][int(tile["layer_index"])]
    weight_tensor = program["tensors"][int(tile["weight_td"])]
    output_block_bytes = (cin_pad * kh * kw * 8)
    weight_start = (int(weight_tensor["base_offset"])
                    + (int(tile["origin"]["cout"]) // 8)
                    * output_block_bytes)
    weight_bytes = weight_blob[weight_start:weight_start + output_block_bytes]
    assert len(weight_bytes) == int(tile["bytes"]["weight"])
    assert [kh, kw] == [int(layer_operator["kh"]),
                        int(layer_operator["kw"])]
    weight_rows = [
        int.from_bytes(weight_bytes[offset:offset + 64], "little")
        for offset in range(0, len(weight_bytes), 64)
    ]

    requests: list[int] = []
    results: list[int] = []
    for oh in range(th):
        for ow in range(tw):
            accumulators = [0] * 8
            for cin_block in range(cin_blocks):
                for kernel_h in range(kh):
                    for kernel_w in range(kw):
                        activation_address = (
                            ((oh * sh + kernel_h) * local_w
                             + (ow * sw + kernel_w)) * cin_pad * 2
                            + cin_block * 16)
                        weight_address = (
                            ((cin_block * kh + kernel_h) * kw + kernel_w)
                            * 64)
                        requests.append((weight_address << 32)
                                        | activation_address)
                        weight_row = weight_bytes[
                            weight_address:weight_address + 64]
                        activation_base = activation_address // 2
                        for out_lane in range(cout):
                            dot = 0
                            for in_lane in range(8):
                                channel = cin_block * 8 + in_lane
                                if channel < cin:
                                    dot += (activation[activation_base + in_lane]
                                            * signed(
                                                weight_row[out_lane * 8
                                                           + in_lane], 8))
                            accumulators[out_lane] = signed32(
                                accumulators[out_lane] + dot)
            result = 0
            for lane, value in enumerate(accumulators):
                result |= (value & 0xFFFF_FFFF) << (lane * 32)
            results.append(result)

    assert len(requests) == int(tile["cycles"]["compute"])
    assert len(results) == th * tw
    return {
        "descriptor": raw_descriptor,
        "activation": activation_rows,
        "weight": weight_rows,
        "requests": requests,
        "results": results,
    }


def main() -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise SystemExit("Icarus Verilog is required")

    schedule = json.loads(
        (PROGRAM / "tile_schedule.json").read_text(encoding="utf-8"))
    program = json.loads((PROGRAM / "program.json").read_text(encoding="utf-8"))
    descriptor_blob = (PROGRAM / "tile_operator_desc.bin").read_bytes()
    weight_blob = (PROGRAM / "weights_o8i8.bin").read_bytes()
    cases = []
    for layer in ("out", "bott", "enc0", "enc1"):
        tile = next(row for row in schedule["tiles"] if row["layer"] == layer)
        cases.append((layer, tile))

    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        image = temp / "tb_npu_conv2d_controller.vvp"
        run([
            iverilog, "-g2012", "-Wall", "-s", "tb_npu_conv2d_controller",
            "-o", str(image),
            str(RTL / "include" / "npu_isa_pkg.sv"),
            str(RTL / "npu_u32_mul_iter.sv"),
            str(RTL / "npu_tensor_mac_8x8.sv"),
            str(RTL / "npu_conv2d_controller.sv"),
            str(RTL / "tb" / "tb_npu_conv2d_controller.sv"),
        ])
        for case_index, (name, tile) in enumerate(cases):
            vectors = build_case(tile, descriptor_blob, weight_blob, program,
                                 0xC02D_2026 + case_index)
            prefix = temp / name
            descriptor_path = prefix.with_suffix(".descriptor.hex")
            activation_path = prefix.with_suffix(".activation.hex")
            weight_path = prefix.with_suffix(".weight.hex")
            request_path = prefix.with_suffix(".request.hex")
            result_path = prefix.with_suffix(".result.hex")
            descriptor_path.write_text(
                f"{int.from_bytes(vectors['descriptor'], 'little'):0128x}\n",
                encoding="ascii")
            write_hex(activation_path, vectors["activation"], 32)
            write_hex(weight_path, vectors["weight"], 128)
            write_hex(request_path, vectors["requests"], 16)
            write_hex(result_path, vectors["results"], 64)
            output = run([
                vvp, str(image),
                f"+DESCRIPTOR_HEX={descriptor_path.as_posix()}",
                f"+ACTIVATION_HEX={activation_path.as_posix()}",
                f"+WEIGHT_HEX={weight_path.as_posix()}",
                f"+REQUEST_HEX={request_path.as_posix()}",
                f"+RESULT_HEX={result_path.as_posix()}",
                f"+REQUEST_COUNT={len(vectors['requests'])}",
                f"+RESULT_COUNT={len(vectors['results'])}",
            ])
            assert "npu_conv2d_controller: PASS" in output
            print(f"{name}: {len(vectors['requests'])} requests, "
                  f"{len(vectors['results'])} results PASS")

        error_image = temp / "tb_npu_conv2d_controller_errors.vvp"
        run([
            iverilog, "-g2012", "-Wall", "-s",
            "tb_npu_conv2d_controller_errors", "-o", str(error_image),
            str(RTL / "include" / "npu_isa_pkg.sv"),
            str(RTL / "npu_u32_mul_iter.sv"),
            str(RTL / "npu_tensor_mac_8x8.sv"),
            str(RTL / "npu_conv2d_controller.sv"),
            str(RTL / "tb" / "tb_npu_conv2d_controller_errors.sv"),
        ])
        error_output = run([vvp, str(error_image)])
        assert "npu_conv2d_controller errors/reset: PASS" in error_output
        print("descriptor errors and soft-reset recovery PASS")

    print("Real-tile CONV2D controller RTL tests: PASS")


if __name__ == "__main__":
    main()
