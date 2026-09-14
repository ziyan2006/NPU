"""Regression tests for the executable draft NPU ISA and compiler output."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from npu_isa import (  # noqa: E402
    COMMAND_STRUCT,
    NONE_INDEX,
    Command,
    CommandFlag,
    Opcode,
    dequantize_scale,
    leaky_relu_q,
    quantize_scale,
    requantize,
    round_shift_rne,
    saturate,
)


def load_compiler():
    spec = importlib.util.spec_from_file_location(
        "npu_compiler_test", SCRIPTS / "26_compile_npu_program.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_point() -> None:
    tie_cases = {1: 0, 3: 2, 5: 2, 7: 4, -1: 0, -3: -2, -5: -2, -7: -4}
    for value, expected in tie_cases.items():
        assert round_shift_rne(value, 1) == expected
    assert round_shift_rne(-9, 0) == -9
    assert round_shift_rne(3, -2) == 12
    assert saturate(-3000, 12) == -2048
    assert saturate(3000, 12) == 2047
    assert saturate(-1, 8, signed=False) == 0
    assert leaky_relu_q(-25) == -2
    assert leaky_relu_q(-35) == -4
    assert leaky_relu_q(19) == 19

    for scale in (1e-6, 0.001, 0.1, 0.75, 1.0, 1.5, 8.0):
        multiplier, shift = quantize_scale(scale)
        relative_error = abs(dequantize_scale(multiplier, shift) - scale) / scale
        assert relative_error < 1e-8, (scale, multiplier, shift, relative_error)
    multiplier, shift = quantize_scale(0.5)
    assert requantize(7, multiplier, shift, -2048, 2047) == 4
    assert requantize(100_000, multiplier, shift, -2048, 2047) == 2047


def test_command_encoding() -> None:
    command = Command(
        opcode=Opcode.CONV2D,
        flags=int(CommandFlag.SATURATE | CommandFlag.FUSED_POST_OP),
        tag=17,
        dst_td=3,
        src0_td=4,
        src1_td=5,
        op_desc=6,
        quant_desc=7,
        imm=0x1234,
    )
    payload = command.pack()
    assert len(payload) == 16 == COMMAND_STRUCT.size
    assert Command.unpack(payload) == command
    try:
        Command(opcode=Opcode.NOP, tag=0x1_0000).pack()
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range command field was accepted")
    assert Command(opcode=Opcode.END, src0_td=NONE_INDEX).src0_td == NONE_INDEX


def inverse_o8i8(payload: bytes, cout: int, cin: int,
                  kh: int, kw: int) -> bytes:
    cout_pad = (cout + 7) // 8 * 8
    cin_pad = (cin + 7) // 8 * 8
    packed = np.frombuffer(payload, dtype=np.int8).reshape(
        cout_pad // 8, cin_pad // 8, kh, kw, 8, 8)
    padded = packed.transpose(0, 4, 1, 5, 2, 3).reshape(
        cout_pad, cin_pad, kh, kw)
    return padded[:cout, :cin].tobytes(order="C")


def test_compiler_package() -> None:
    compiler = load_compiler()
    source = ROOT / "hardware" / "generated" / "bott2_mir1k_v1"
    output = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
    analysis = compiler.compile_program(source, output)
    program = json.loads((output / "program.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))

    assert analysis["counts"] == {
        "commands": 72,
        "tensor_descriptors": 42,
        "operator_descriptors": 11,
        "quant_descriptors": 13,
        "segment_records": 6,
    }
    assert analysis["cycles"]["compute"] == 1_986_560
    assert analysis["cycles"]["total_no_overlap"] < 4_000_000
    assert analysis["cycles"]["budget_pass"] is True
    assert analysis["tile_checks_pass"] is True
    assert analysis["isa_freeze_blockers"] == []
    assert analysis["tanh_lut"]["generated"] is True
    assert analysis["tanh_lut"]["entries"] == 4096

    commands_blob = (output / "commands.bin").read_bytes()
    commands = [Command.unpack(commands_blob[i:i + 16])
                for i in range(0, len(commands_blob), 16)]
    assert len(commands) == analysis["counts"]["commands"]
    assert commands[-1].opcode == Opcode.END
    assert any(command.opcode == Opcode.VEC_ADD for command in commands)
    assert sum(command.opcode == Opcode.UPSAMPLE2X for command in commands) == 3

    assert (output / "tensor_desc.bin").stat().st_size == (
        analysis["counts"]["tensor_descriptors"] * compiler.TENSOR_STRUCT.size)
    assert (output / "operator_desc.bin").stat().st_size == (
        analysis["counts"]["operator_descriptors"] * compiler.OPERATOR_STRUCT.size)
    assert (output / "quant_desc.bin").stat().st_size == (
        analysis["counts"]["quant_descriptors"] * compiler.QUANT_DESC_STRUCT.size)
    assert (output / "segments.bin").stat().st_size == (
        analysis["counts"]["segment_records"] * compiler.SEGMENT_STRUCT.size)
    tanh_lut = np.fromfile(output / "tanh_lut_int12.bin", dtype="<i2")
    assert tanh_lut.size == 4096
    assert np.all(tanh_lut[1:] >= tanh_lut[:-1])
    assert tanh_lut[2048] == 0
    assert tanh_lut[0] == -2047 and tanh_lut[-1] == 2047

    source_weights = (source / "weights_int8.bin").read_bytes()
    packed_weights = (output / "weights_o8i8.bin").read_bytes()
    for source_layer, compiled_layer in zip(manifest["layers"], analysis["layers"]):
        start = compiled_layer["packed_weight_offset"]
        length = compiled_layer["packed_weight_bytes"]
        restored = inverse_o8i8(
            packed_weights[start:start + length],
            source_layer["out_channels"], source_layer["in_channels"],
            *source_layer["kernel"])
        source_start = source_layer["weight_offset"]
        source_length = source_layer["weight_bytes"]
        assert restored == source_weights[source_start:source_start + source_length]

    for filename, metadata in program["files"].items():
        payload = (output / filename).read_bytes()
        assert len(payload) == metadata["bytes"]
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
    assert "D:\\" not in (output / "program.json").read_text(encoding="utf-8")

    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.iterdir() if path.is_file()
    }
    compiler.compile_program(source, output)
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.iterdir() if path.is_file()
    }
    assert second_hashes == first_hashes


def main() -> None:
    test_fixed_point()
    test_command_encoding()
    test_compiler_package()
    print("NPU ISA/compiler regression tests: PASS")


if __name__ == "__main__":
    main()
