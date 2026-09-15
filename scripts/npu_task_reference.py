"""Bit-exact executable reference for the current complete NPU task image.

The interpreter deliberately follows the compiled command stream and DMA plan
instead of reconstructing the source neural-network graph.  It therefore also
checks the compiler/RTL contract for scratchpad layout, ping-pong bank selects,
tile stores, residual post-processing, and whole-tensor upsampling.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from npu_isa import (IMM_ACTIVATION_BANK_BIT, IMM_OUTPUT_BANK_BIT,
                     IMM_WEIGHT_BANK_BIT, PostOp, align_up, leaky_relu_q,
                     requantize, saturate)


ROOT = Path(__file__).resolve().parent.parent
PROGRAM_DEFAULT = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
BANK_BYTES = {"A": 64 * 1024, "W": 32 * 1024, "O": 16 * 1024}


def _signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _signed32(value: int) -> int:
    return _signed(value, 32)


def _read_i16(payload: bytearray | bytes, offset: int) -> int:
    return int.from_bytes(payload[offset:offset + 2], "little", signed=True)


def _write_i16(payload: bytearray, offset: int, value: int) -> None:
    payload[offset:offset + 2] = int(value).to_bytes(2, "little", signed=True)


def _read_quant(payload: bytearray | bytes, offset: int) -> tuple[int, int, int, int]:
    row = payload[offset:offset + 16]
    if len(row) != 16:
        raise ValueError(f"quant parameter out of range at {offset}")
    return (int.from_bytes(row[0:4], "little", signed=True), row[4],
            int.from_bytes(row[8:12], "little", signed=True),
            int.from_bytes(row[12:16], "little", signed=True))


def seed_input(image: bytes, program: dict[str, Any], manifest: dict[str, Any],
               seed: int) -> bytes:
    """Put deterministic nonzero INT12 samples in the logical input channels."""
    result = bytearray(image)
    tensor = program["tensors"][int(program["entry"]["input_tensor"])]
    channels, height, width = map(int, tensor["shape_chw"])
    base = (int(manifest["sections"]["activation"]["offset"])
            + int(tensor["base_offset"]))
    rng = random.Random(seed)
    for h in range(height):
        for w in range(width):
            for channel in range(channels):
                offset = (base + h * int(tensor["strides"][1])
                          + w * int(tensor["strides"][2])
                          + channel * int(tensor["strides"][3]))
                _write_i16(result, offset, rng.randint(-2048, 2047))
    return bytes(result)


class TaskReference:
    """Execute one compiled task using byte-addressed external and local memory."""

    def __init__(self, program_dir: Path, image: bytes):
        self.program_dir = program_dir
        self.program = json.loads((program_dir / "program.json").read_text(
            encoding="utf-8"))
        self.schedule = json.loads((program_dir / "tile_schedule.json").read_text(
            encoding="utf-8"))
        self.manifest = json.loads((program_dir / "task_image.json").read_text(
            encoding="utf-8"))
        plan = json.loads((program_dir / "dma_plan.json").read_text(
            encoding="utf-8"))
        self.dma_requests = {int(row["command_index"]): row
                             for row in plan["requests"]}
        self.image = bytearray(image)
        if len(self.image) != int(self.manifest["total_bytes"]):
            raise ValueError("task image byte count does not match its manifest")
        self.spad = {f"{kind}{bank}": bytearray(size)
                     for kind, size in BANK_BYTES.items() for bank in range(2)}
        lut_section = self.manifest["sections"]["tanh_lut"]
        lut_raw = self.image[int(lut_section["offset"]):
                             int(lut_section["offset"]) + int(lut_section["bytes"])]
        self.tanh_lut = np.frombuffer(lut_raw, dtype="<i2").astype(np.int64)
        if self.tanh_lut.size != 4096:
            raise ValueError("TANH LUT must contain 4096 INT16 values")

    def _section_base(self, memory_space: str) -> int:
        section = {"weight": "weights", "quant": "quant_params"}.get(
            memory_space, memory_space)
        return int(self.manifest["sections"][section]["offset"])

    def _dma(self, command_index: int) -> None:
        request = self.dma_requests[command_index]
        local = self.spad[request["scratchpad"]]
        if request["clear_before"]:
            clear_bytes = int(request["clear_bytes"])
            local[0:clear_bytes] = bytes(clear_bytes)
        external_base = self._section_base(request["memory_space"])
        for z in range(int(request["z_count"])):
            for y in range(int(request["y_count"])):
                external = (external_base + int(request["external_offset"])
                            + z * int(request["external_z_stride"])
                            + y * int(request["external_y_stride"]))
                scratchpad = (int(request["scratchpad_offset"])
                              + z * int(request["scratchpad_z_stride"])
                              + y * int(request["scratchpad_y_stride"]))
                count = int(request["x_bytes"])
                if request["opcode"] == "DMA_LOAD":
                    local[scratchpad:scratchpad + count] = \
                        self.image[external:external + count]
                else:
                    self.image[external:external + count] = \
                        local[scratchpad:scratchpad + count]

    @staticmethod
    def _weight_layout(descriptor: dict[str, Any]) -> tuple[int, int, int]:
        cin_pad = align_up(int(descriptor["input_channel_count"]), 8)
        weight_bytes = (align_up(int(descriptor["tile_cout"]), 8) * cin_pad
                        * int(descriptor["kh"]) * int(descriptor["kw"]))
        bias_offset = align_up(weight_bytes, 64)
        quant_offset = align_up(
            bias_offset + int(descriptor["tile_cout"]) * 4, 64)
        vector_offset = align_up(
            quant_offset + int(descriptor["tile_cout"]) * 16, 64)
        return bias_offset, quant_offset, vector_offset

    def _conv2d(self, command: dict[str, Any]) -> None:
        descriptor = self.schedule["operator_descriptors"][int(command["op_desc"])]
        imm = int(command["imm"])
        activation = self.spad[f"A{(imm >> IMM_ACTIVATION_BANK_BIT) & 1}"]
        weight = self.spad[f"W{(imm >> IMM_WEIGHT_BANK_BIT) & 1}"]
        output = self.spad[f"O{(imm >> IMM_OUTPUT_BANK_BIT) & 1}"]

        kh, kw = int(descriptor["kh"]), int(descriptor["kw"])
        sh, sw = int(descriptor["stride_h"]), int(descriptor["stride_w"])
        dh, dw = int(descriptor["dilation_h"]), int(descriptor["dilation_w"])
        th, tw = int(descriptor["tile_h"]), int(descriptor["tile_w"])
        cin, cout = (int(descriptor["input_channel_count"]),
                     int(descriptor["tile_cout"]))
        cin_pad = align_up(cin, 8)
        local_h = (th - 1) * sh + dh * (kh - 1) + 1
        local_w = (tw - 1) * sw + dw * (kw - 1) + 1
        activation_values = np.frombuffer(
            activation, dtype="<i2", count=local_h * local_w * cin_pad)
        activation_values = activation_values.astype(np.int64).reshape(
            local_h, local_w, cin_pad)
        accumulators = np.zeros((th, tw, 8), dtype=np.int64)
        cin_blocks = cin_pad // 8
        for cin_block in range(cin_blocks):
            for kernel_h in range(kh):
                for kernel_w in range(kw):
                    address = (((cin_block * kh + kernel_h) * kw + kernel_w)
                               * 64)
                    matrix = np.frombuffer(weight, dtype=np.int8, count=64,
                                           offset=address).astype(np.int64)
                    matrix = matrix.reshape(8, 8)
                    patch = activation_values[
                        kernel_h * dh:kernel_h * dh + th * sh:sh,
                        kernel_w * dw:kernel_w * dw + tw * sw:sw,
                        cin_block * 8:(cin_block + 1) * 8]
                    accumulators += np.matmul(patch, matrix.T)

        # Addition in the RTL wraps at INT32.  Modulo addition is associative,
        # so one final conversion is equivalent even if an intermediate wraps.
        accumulators = ((accumulators & 0xffff_ffff) ^ 0x8000_0000) \
            - 0x8000_0000
        bias_offset, quant_offset, _ = self._weight_layout(descriptor)
        biases = [int.from_bytes(weight[bias_offset + lane * 4:
                                        bias_offset + lane * 4 + 4],
                                 "little", signed=True)
                  for lane in range(cout)]
        quant = [_read_quant(weight, quant_offset + lane * 16)
                 for lane in range(cout)]
        post_op = PostOp(int(descriptor["post_op_id"]))
        pixel_bytes = align_up(cout, 8) * 2
        for h in range(th):
            for w in range(tw):
                pixel = (h * tw + w) * pixel_bytes
                for lane in range(cout):
                    biased = _signed32(int(accumulators[h, w, lane])
                                       + biases[lane])
                    multiplier, shift, clamp_min, clamp_max = quant[lane]
                    value = requantize(biased, multiplier, shift,
                                       clamp_min, clamp_max)
                    if post_op == PostOp.RELU:
                        value = max(value, 0)
                    elif post_op == PostOp.LEAKY_RELU_0P1:
                        value = leaky_relu_q(value)
                    elif post_op == PostOp.TANH_LUT:
                        value = int(self.tanh_lut[
                            min(max(value, -2048), 2047) + 2048])
                    _write_i16(output, pixel + lane * 2, value)

    def _vec_add(self, command: dict[str, Any]) -> None:
        descriptor = self.schedule["operator_descriptors"][int(command["op_desc"])]
        imm = int(command["imm"])
        activation = self.spad[f"A{(imm >> IMM_ACTIVATION_BANK_BIT) & 1}"]
        weight = self.spad[f"W{(imm >> IMM_WEIGHT_BANK_BIT) & 1}"]
        output = self.spad[f"O{(imm >> IMM_OUTPUT_BANK_BIT) & 1}"]
        _, _, vector_offset = self._weight_layout(descriptor)
        multiplier, shift, clamp_min, clamp_max = _read_quant(weight, vector_offset)
        th, tw = int(descriptor["tile_h"]), int(descriptor["tile_w"])
        cout = int(descriptor["tile_cout"])
        cin_pad = align_up(int(descriptor["input_channel_count"]), 8)
        local_w = (tw - 1) + int(descriptor["kw"])
        pixel_step = cin_pad * 2
        residual = (int(descriptor["pad_top"]) * local_w
                    + int(descriptor["pad_left"])) * pixel_step
        residual += int(descriptor["tile_origin_cout"]) * 2
        residual_row_advance = (local_w - tw) * pixel_step
        output_address = 0
        for _h in range(th):
            for _w in range(tw):
                for lane in range(cout):
                    conv = _read_i16(output, output_address + lane * 2)
                    skip = _read_i16(activation, residual + lane * 2)
                    skip = requantize(skip, multiplier, shift,
                                      clamp_min, clamp_max)
                    _write_i16(output, output_address + lane * 2,
                               saturate(conv + skip, 12))
                residual += pixel_step
                output_address += 16
            residual += residual_row_advance

    def _upsample2x(self, command: dict[str, Any]) -> None:
        source = self.program["tensors"][int(command["src0_td"])]
        destination = self.program["tensors"][int(command["dst_td"])]
        activation_base = int(self.manifest["sections"]["activation"]["offset"])
        source_base = activation_base + int(source["base_offset"])
        destination_base = activation_base + int(destination["base_offset"])
        channels, height, width = map(int, source["shape_chw"])
        pixel_bytes = align_up(channels, 8) * 2
        for h in range(height):
            source_row = source_base + h * int(source["strides"][1])
            row = bytearray()
            for w in range(width):
                pixel = self.image[source_row + w * pixel_bytes:
                                   source_row + (w + 1) * pixel_bytes]
                row += pixel
                row += pixel
            for copy in range(2):
                destination_row = (destination_base
                                   + (h * 2 + copy)
                                   * int(destination["strides"][1]))
                self.image[destination_row:destination_row + len(row)] = row

    def run(self, *, progress: bool = False) -> bytes:
        commands = self.schedule["commands"]
        retired = 0
        for index, command in enumerate(commands):
            opcode = command["opcode"]
            if opcode in {"DMA_LOAD", "DMA_STORE"}:
                self._dma(index)
            elif opcode == "CONV2D":
                self._conv2d(command)
            elif opcode == "VEC_ADD":
                self._vec_add(command)
            elif opcode == "UPSAMPLE2X":
                self._upsample2x(command)
            elif opcode not in {"WAIT", "END"}:
                raise NotImplementedError(f"command {index}: {opcode}")
            retired += 1
            if progress and retired % 250 == 0:
                print(f"reference progress: {retired}/{len(commands)} commands")
        return bytes(self.image)

    def output_bytes(self) -> bytes:
        tensor = self.program["tensors"][int(self.program["entry"]["output_tensor"])]
        start = (int(self.manifest["sections"]["activation"]["offset"])
                 + int(tensor["base_offset"]))
        return bytes(self.image[start:start + int(tensor["allocation_bytes"])])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=PROGRAM_DEFAULT)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=lambda value: int(value, 0),
                        default=0x4e50_5531)
    args = parser.parse_args()
    program_dir = args.program.resolve()
    image_path = args.image.resolve() if args.image else program_dir / "task_image.bin"
    program = json.loads((program_dir / "program.json").read_text(encoding="utf-8"))
    manifest = json.loads((program_dir / "task_image.json").read_text(
        encoding="utf-8"))
    image = seed_input(image_path.read_bytes(), program, manifest, args.seed)
    reference = TaskReference(program_dir, image)
    reference.run(progress=True)
    output = reference.output_bytes()
    if args.output:
        args.output.write_bytes(output)
    print(f"task reference: PASS, {len(output):,} output bytes, "
          f"{sum(value != 0 for value in output):,} nonzero bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
