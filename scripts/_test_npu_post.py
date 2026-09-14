"""Bit-exact RTL regression for bias, requant, activation, and tanh LUT."""
from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from npu_isa import PostOp, leaky_relu_q, round_shift_rne


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"


def signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def wrap32(value: int) -> int:
    return signed(value, 32)


def pack_signed(values: list[int], bits: int) -> int:
    result = 0
    mask = (1 << bits) - 1
    for index, value in enumerate(values):
        result |= (int(value) & mask) << (index * bits)
    return result


def write_hex(path: Path, values: list[int], digits: int) -> None:
    path.write_text("".join(f"{value:0{digits}x}\n" for value in values),
                    encoding="ascii")


def run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def make_case(rng: random.Random, post_op: PostOp,
              accumulator: list[int] | None = None,
              bias: list[int] | None = None,
              shifts: list[int] | None = None) -> tuple[int, int, int, int, int]:
    accumulators = accumulator or [rng.randint(-(1 << 31), (1 << 31) - 1)
                                   for _ in range(8)]
    biases = bias or [rng.randint(-(1 << 31), (1 << 31) - 1)
                      for _ in range(8)]
    shift_values = shifts or [rng.choice([0, 1, 2, 15, 31, 32, 37, 62, 63])
                              for _ in range(8)]
    multipliers = [rng.randint(0, 0x7fffffff) for _ in range(8)]
    clamp_min = [rng.randint(-2048, 0) for _ in range(8)]
    clamp_max = [rng.randint(0, 2047) for _ in range(8)]
    lane_mask = rng.randint(1, 255)

    params = 0
    outputs: list[int] = []
    lut = LUT_VALUES
    for lane in range(8):
        params |= multipliers[lane] << (lane * 128)
        params |= shift_values[lane] << (lane * 128 + 32)
        params |= (clamp_min[lane] & 0xffffffff) << (lane * 128 + 64)
        params |= (clamp_max[lane] & 0xffffffff) << (lane * 128 + 96)
        biased = wrap32(accumulators[lane] + biases[lane])
        value = round_shift_rne(biased * multipliers[lane],
                                shift_values[lane])
        value = min(max(value, clamp_min[lane]), clamp_max[lane])
        if post_op == PostOp.RELU:
            value = max(value, 0)
        elif post_op == PostOp.LEAKY_RELU_0P1:
            value = leaky_relu_q(value)
        elif post_op == PostOp.TANH_LUT:
            value = lut[min(max(value, -2048), 2047) + 2048]
        outputs.append(value if lane_mask & (1 << lane) else 0)

    return (pack_signed(accumulators, 32), pack_signed(biases, 32), params,
            (int(post_op) << 8) | lane_mask, pack_signed(outputs, 16))


LUT_BLOB = (PROGRAM / "tanh_lut_int12.bin").read_bytes()
LUT_VALUES = [signed(int.from_bytes(LUT_BLOB[index:index + 2], "little"), 16)
              for index in range(0, len(LUT_BLOB), 2)]


def main() -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")

    rng = random.Random(0x7020_504F_5354)
    cases = []
    directed_acc = [-5, -3, -1, 1, 3, 5, (1 << 31) - 1, -(1 << 31)]
    directed_bias = [0, 0, 0, 0, 0, 0, 1, -1]
    for post_op in PostOp:
        cases.append(make_case(rng, post_op, directed_acc,
                               directed_bias, [1] * 8))
    for post_op in PostOp:
        for _ in range(24):
            cases.append(make_case(rng, post_op))

    accumulator, bias, quant, meta, expected = map(list, zip(*cases))
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        write_hex(temp / "post_accumulator.hex", accumulator, 64)
        write_hex(temp / "post_bias.hex", bias, 64)
        write_hex(temp / "post_quant.hex", quant, 256)
        write_hex(temp / "post_meta.hex", meta, 6)
        write_hex(temp / "post_expected.hex", expected, 32)
        write_hex(temp / "post_lut.hex", [value & 0xffff for value in LUT_VALUES], 4)
        image = temp / "tb_npu_requant_post.vvp"
        compile_output = run([
            iverilog, "-g2012", "-Wall", "-s", "tb_npu_requant_post",
            "-o", str(image),
            str(RTL / "include" / "npu_isa_pkg.sv"),
            str(RTL / "npu_requant_post.sv"),
            str(RTL / "tb" / "tb_npu_requant_post.sv"),
        ], ROOT)
        output = compile_output + run(
            [vvp, str(image), f"+CASE_COUNT={len(cases)}"], temp)
    if "bit-exact vectors PASS" not in output:
        raise RuntimeError(output)
    print(f"npu_requant_post Python golden: {len(cases)} vectors PASS")


if __name__ == "__main__":
    main()
