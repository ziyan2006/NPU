"""Real-tile bit-exact regression for CONV2D, parameter fetch, and post."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from npu_isa import (OPERATOR_STRUCT, PostOp, leaky_relu_q, requantize)


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


def align(value: int, amount: int = 64) -> int:
    return (value + amount - 1) & -amount


def write_hex(path: Path, values: list[int], digits: int) -> None:
    path.write_text("".join(f"{value:0{digits}x}\n" for value in values),
                    encoding="ascii")


def run(command: list[str], cwd: Path = ROOT) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def build_case(tile: dict, descriptor_blob: bytes, weight_blob: bytes,
               bias_blob: bytes, quant_blob: bytes, program: dict,
               lut: list[int], seed: int) -> dict:
    start = int(tile["operator_desc"]) * 64
    raw_descriptor = descriptor_blob[start:start + 64]
    descriptor = dict(zip(OPERATOR_KEYS, map(int, OPERATOR_STRUCT.unpack(
        raw_descriptor)), strict=True))
    kh, kw = descriptor["kh"], descriptor["kw"]
    sh, sw = descriptor["stride_h"], descriptor["stride_w"]
    th, tw = descriptor["tile_h"], descriptor["tile_w"]
    cin, cout = descriptor["input_channel_count"], descriptor["tile_cout"]
    cin_pad = align(cin, 8)
    cin_blocks = cin_pad // 8
    local_h = (th - 1) * sh + kh
    local_w = (tw - 1) * sw + kw

    rng = random.Random(seed)
    activation = [0] * (local_h * local_w * cin_pad)
    region = tile["input_region"]
    for h in range(int(region["pad_top"]),
                   local_h - int(region["pad_bottom"])):
        for w in range(int(region["pad_left"]),
                       local_w - int(region["pad_right"])):
            base = (h * local_w + w) * cin_pad
            for channel in range(cin):
                activation[base + channel] = rng.randint(-2048, 2047)
    activation_bytes = b"".join(value.to_bytes(2, "little", signed=True)
                                 for value in activation)
    activation_rows = [int.from_bytes(activation_bytes[i:i + 16], "little")
                       for i in range(0, len(activation_bytes), 16)]

    weight_tensor = program["tensors"][int(tile["weight_td"])]
    weight_bytes_count = cin_pad * kh * kw * 8
    weight_start = (int(weight_tensor["base_offset"])
                    + (int(tile["origin"]["cout"]) // 8)
                    * weight_bytes_count)
    weights = weight_blob[weight_start:weight_start + weight_bytes_count]
    bias_tensor = program["tensors"][int(tile["bias_td"])]
    bias_start = (int(bias_tensor["base_offset"])
                  + int(tile["origin"]["cout"]) * 4)
    biases_raw = bias_blob[bias_start:bias_start + cout * 4]
    biases = [signed(int.from_bytes(biases_raw[i:i + 4], "little"), 32)
              for i in range(0, len(biases_raw), 4)]
    quant_desc = program["quantization"][int(tile["quant_desc"])]
    quant_start = (int(quant_desc["param_offset"])
                   + int(tile["origin"]["cout"]) * 16)
    quant_raw = quant_blob[quant_start:quant_start + cout * 16]
    quant = []
    for i in range(0, len(quant_raw), 16):
        row = quant_raw[i:i + 16]
        quant.append((signed(int.from_bytes(row[0:4], "little"), 32), row[4],
                      signed(int.from_bytes(row[8:12], "little"), 32),
                      signed(int.from_bytes(row[12:16], "little"), 32)))

    bias_address = align(len(weights))
    quant_address = align(bias_address + len(biases_raw))
    bank = bytearray(quant_address + align(len(quant_raw)))
    bank[0:len(weights)] = weights
    bank[bias_address:bias_address + len(biases_raw)] = biases_raw
    bank[quant_address:quant_address + len(quant_raw)] = quant_raw
    weight_rows = [int.from_bytes(bank[i:i + 64], "little")
                   for i in range(0, len(bank), 64)]

    requests: list[int] = []
    results: list[int] = []
    post_op = PostOp(descriptor["post_op_id"])
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
                            ((cin_block * kh + kernel_h) * kw + kernel_w) * 64)
                        requests.append((weight_address << 32)
                                        | activation_address)
                        weight_row = weights[weight_address:weight_address + 64]
                        activation_base = activation_address // 2
                        for out_lane in range(cout):
                            dot = sum(
                                activation[activation_base + in_lane]
                                * signed(weight_row[out_lane * 8 + in_lane], 8)
                                for in_lane in range(8)
                                if cin_block * 8 + in_lane < cin)
                            accumulators[out_lane] = signed32(
                                accumulators[out_lane] + dot)
            result = 0
            for lane in range(cout):
                biased = signed32(accumulators[lane] + biases[lane])
                multiplier, shift, clamp_min, clamp_max = quant[lane]
                value = requantize(biased, multiplier, shift,
                                   clamp_min, clamp_max)
                if post_op == PostOp.RELU:
                    value = max(value, 0)
                elif post_op == PostOp.LEAKY_RELU_0P1:
                    value = leaky_relu_q(value)
                elif post_op == PostOp.TANH_LUT:
                    value = lut[min(max(value, -2048), 2047) + 2048]
                result |= (value & 0xffff) << (lane * 16)
            results.append(result)
    return {"descriptor": raw_descriptor, "activation": activation_rows,
            "weight": weight_rows, "requests": requests, "results": results,
            "bias_address": bias_address,
            "param_request_count": 3 if cout > 4 else 2}


def main() -> None:
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")
    schedule = json.loads((PROGRAM / "tile_schedule.json").read_text())
    program = json.loads((PROGRAM / "program.json").read_text())
    descriptor_blob = (PROGRAM / "tile_operator_desc.bin").read_bytes()
    weight_blob = (PROGRAM / "weights_o8i8.bin").read_bytes()
    bias_blob = (PROGRAM / "bias_int32.bin").read_bytes()
    quant_blob = (PROGRAM / "quant_params.bin").read_bytes()
    lut_blob = (PROGRAM / "tanh_lut_int12.bin").read_bytes()
    lut = [signed(int.from_bytes(lut_blob[i:i + 2], "little"), 16)
           for i in range(0, len(lut_blob), 2)]

    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        image = temp / "tb_npu_conv2d_pipeline.vvp"
        run([iverilog, "-g2012", "-Wall", "-s", "tb_npu_conv2d_pipeline",
             "-o", str(image), str(RTL / "include" / "npu_isa_pkg.sv"),
             str(RTL / "npu_u32_mul_iter.sv"),
             str(RTL / "npu_tensor_mac_8x8.sv"),
             str(RTL / "npu_conv2d_controller.sv"),
             str(RTL / "npu_requant_post.sv"),
             str(RTL / "npu_conv2d_pipeline.sv"),
             str(RTL / "tb" / "tb_npu_conv2d_pipeline.sv")])
        lut_path = temp / "tanh.hex"
        write_hex(lut_path, [value & 0xffff for value in lut], 4)
        for case_index, layer in enumerate(("out", "bott", "enc0", "enc1")):
            tile = next(row for row in schedule["tiles"] if row["layer"] == layer)
            vectors = build_case(tile, descriptor_blob, weight_blob, bias_blob,
                                 quant_blob, program, lut, 0x5049_5045 + case_index)
            paths = {}
            for name, key, digits in (("activation", "activation", 32),
                                      ("weight", "weight", 128),
                                      ("request", "requests", 16),
                                      ("result", "results", 32)):
                paths[name] = temp / f"{layer}.{name}.hex"
                write_hex(paths[name], vectors[key], digits)
            descriptor_path = temp / f"{layer}.descriptor.hex"
            descriptor_path.write_text(
                f"{int.from_bytes(vectors['descriptor'], 'little'):0128x}\n")
            output = run([
                vvp, str(image), f"+DESCRIPTOR_HEX={descriptor_path.as_posix()}",
                f"+ACTIVATION_HEX={paths['activation'].as_posix()}",
                f"+WEIGHT_HEX={paths['weight'].as_posix()}",
                f"+REQUEST_HEX={paths['request'].as_posix()}",
                f"+RESULT_HEX={paths['result'].as_posix()}",
                f"+LUT_HEX={lut_path.as_posix()}",
                f"+BIAS_ADDRESS={vectors['bias_address']}",
                f"+PARAM_REQUEST_COUNT={vectors['param_request_count']}",
                f"+REQUEST_COUNT={len(vectors['requests'])}",
                f"+RESULT_COUNT={len(vectors['results'])}"])
            if "npu_conv2d_pipeline: PASS" not in output:
                raise RuntimeError(output)
            print(f"{layer}: {len(vectors['requests'])} requests, "
                  f"{len(vectors['results'])} post results PASS")
    print("Real-tile CONV2D post pipeline RTL tests: PASS")


if __name__ == "__main__":
    main()
