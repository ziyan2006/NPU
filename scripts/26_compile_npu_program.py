"""Compile the current separator manifest into the draft programmable NPU ISA.

The output is an architecture package, not yet a board-loadable task image.  It
validates the 128-bit command encoding, fixed-size descriptors, O8I8 weight
layout, integer requant parameters, graph liveness, tiling, and cycle budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from npu_isa import (
    ISA_MAJOR,
    ISA_MINOR,
    NONE_INDEX,
    Command,
    CommandFlag,
    DType,
    Layout,
    Opcode,
    PostOp,
    QUANT_PARAM_STRUCT,
    align_up,
    dequantize_scale,
    nhwc8_nbytes,
    quantize_scale,
)


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DEFAULT = ROOT / "hardware" / "generated" / "bott2_mir1k_v1"
OUT_DEFAULT = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"

TENSOR_STRUCT = struct.Struct("<QI4H4IBBHHHI16x")
OPERATOR_STRUCT = struct.Struct("<16H4I16x")
QUANT_DESC_STRUCT = struct.Struct("<IIiiHH12x")
SEGMENT_STRUCT = struct.Struct("<4H")

TENSOR_FLAG_CONSTANT = 1 << 0
TENSOR_FLAG_EXTERNAL = 1 << 1
TENSOR_FLAG_VIEW = 1 << 2

ACTIVATION_BYTES = 2
BUS_BYTES = 8
BURST_BYTES = 1024
BURST_OVERHEAD_CYCLES = 16
COMMAND_OVERHEAD_CYCLES = 4
TARGET_CLOCK_HZ = 200_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_aligned(blob: bytearray, payload: bytes, alignment: int = 64) -> int:
    offset = align_up(len(blob), alignment)
    blob.extend(b"\0" * (offset - len(blob)))
    blob.extend(payload)
    return offset


def post_op_for(text: str) -> PostOp:
    if text == "tanh":
        return PostOp.TANH_LUT
    if text.startswith("leaky_relu_0.1"):
        return PostOp.LEAKY_RELU_0P1
    if text == "relu":
        return PostOp.RELU
    return PostOp.NONE


def activation_strides(shape: list[int]) -> list[int]:
    channels, height, width = map(int, shape)
    padded = align_up(channels, 8)
    # Descriptor order is N,H,W,C and strides are bytes.
    return [height * width * padded * ACTIVATION_BYTES,
            width * padded * ACTIVATION_BYTES,
            padded * ACTIVATION_BYTES,
            ACTIVATION_BYTES]


def pack_tensor(row: dict[str, Any]) -> bytes:
    shape = row["shape_nhwc"]
    strides = row["strides"]
    return TENSOR_STRUCT.pack(
        row["base_offset"], row["allocation_bytes"], *shape, *strides,
        row["dtype_id"], row["layout_id"], row.get("quant_desc", NONE_INDEX),
        row.get("segment_count", 0), row["flags"],
        row.get("segment_offset", 0))


def pack_operator(row: dict[str, Any]) -> bytes:
    fields16 = (
        row["src_td"], row["dst_td"], row["weight_td"], row["bias_td"],
        row["kh"], row["kw"], row["stride_h"], row["stride_w"],
        row["dilation_h"], row["dilation_w"], row["pad_top"],
        row["pad_bottom"], row["pad_left"], row["pad_right"],
        row["groups"], row["post_op_id"],
    )
    fields32 = (row["tile_h"], row["tile_w"], row["tile_cout"], 0)
    return OPERATOR_STRUCT.pack(*fields16, *fields32)


def reorder_oihw_to_o8i8(raw: bytes, cout: int, cin: int,
                          kh: int, kw: int) -> bytes:
    source = np.frombuffer(raw, dtype=np.int8)
    expected = cout * cin * kh * kw
    if source.size != expected:
        raise ValueError(f"OIHW payload has {source.size} values, expected {expected}")
    source = source.reshape(cout, cin, kh, kw)
    cout_pad, cin_pad = align_up(cout, 8), align_up(cin, 8)
    padded = np.zeros((cout_pad, cin_pad, kh, kw), dtype=np.int8)
    padded[:cout, :cin] = source
    packed = padded.reshape(cout_pad // 8, 8, cin_pad // 8, 8, kh, kw)
    packed = packed.transpose(0, 2, 4, 5, 1, 3)
    return packed.tobytes(order="C")


def tile_plan(layer: dict[str, Any]) -> dict[str, int | bool]:
    cin, hin, win = map(int, layer["input"])
    cout, hout, wout = map(int, layer["output"])
    kh, kw = map(int, layer["kernel"])
    sh, sw = map(int, layer["stride"])
    tile_h = min(16, hout)
    tile_w = wout
    tile_cout = min(8, align_up(cout, 8))
    input_h = min(hin, tile_h * sh + 2 * int(layer["frequency_pad_each_side"]))
    input_w = min(win, tile_w * sw + int(layer["time_pad_left"]))
    input_bytes = input_h * input_w * align_up(cin, 8) * ACTIVATION_BYTES
    output_bytes = tile_h * tile_w * tile_cout * ACTIVATION_BYTES
    accumulator_bytes = tile_h * tile_w * tile_cout * 4
    weight_bytes = tile_cout * align_up(cin, 8) * kh * kw
    return {
        "tile_h": tile_h,
        "tile_w": tile_w,
        "tile_cout": tile_cout,
        "input_tile_bytes": input_bytes,
        "output_tile_bytes": output_bytes,
        "weight_tile_bytes": weight_bytes,
        "accumulator_tile_bytes": accumulator_bytes,
        "fits_activation_216kib": input_bytes + output_bytes <= 216 * 1024,
        "fits_weight_108kib": weight_bytes <= 108 * 1024,
        "fits_accumulator_54kib": accumulator_bytes <= 54 * 1024,
    }


class ProgramBuilder:
    def __init__(self) -> None:
        self.tensors: list[dict[str, Any]] = []
        self.tensor_by_name: dict[str, int] = {}
        self.operators: list[dict[str, Any]] = []
        self.quant_descs: list[dict[str, Any]] = []
        self.quant_params = bytearray()
        self.segments = bytearray()
        self.commands: list[Command] = []
        self.mutable_cursor = 0

    def tensor(self, name: str, shape_chw: list[int], *,
               dtype: DType = DType.INT12_IN_INT16,
               layout: Layout = Layout.NHWC8,
               base_offset: int | None = None,
               allocation_bytes: int | None = None,
               flags: int = TENSOR_FLAG_EXTERNAL,
               quant_desc: int = NONE_INDEX) -> int:
        if name in self.tensor_by_name:
            raise ValueError(f"duplicate tensor: {name}")
        channels, height, width = map(int, shape_chw)
        size = (nhwc8_nbytes(shape_chw, ACTIVATION_BYTES)
                if allocation_bytes is None else int(allocation_bytes))
        if base_offset is None:
            base_offset = align_up(self.mutable_cursor, 64)
            self.mutable_cursor = base_offset + size
        row = {
            "name": name,
            "base_offset": int(base_offset),
            "allocation_bytes": size,
            "shape_chw": [channels, height, width],
            "shape_nhwc": [1, height, width, channels],
            "strides": activation_strides(shape_chw),
            "dtype": dtype.name,
            "dtype_id": int(dtype),
            "layout": layout.name,
            "layout_id": int(layout),
            "quant_desc": quant_desc,
            "flags": flags,
        }
        index = len(self.tensors)
        self.tensors.append(row)
        self.tensor_by_name[name] = index
        return index

    def constant(self, name: str, *, base_offset: int, size: int,
                 dtype: DType, layout: Layout, shape_chw: list[int]) -> int:
        return self.tensor(
            name, shape_chw, dtype=dtype, layout=layout,
            base_offset=base_offset, allocation_bytes=size,
            flags=TENSOR_FLAG_CONSTANT | TENSOR_FLAG_EXTERNAL)

    def segmented(self, name: str, shape_chw: list[int],
                  parts: list[tuple[int, int]]) -> int:
        segment_offset = len(self.segments)
        dst_channel = 0
        for tensor_index, channel_count in parts:
            self.segments.extend(SEGMENT_STRUCT.pack(
                tensor_index, dst_channel, channel_count, 0))
            dst_channel += channel_count
        if dst_channel != shape_chw[0]:
            raise ValueError(f"segmented channels {dst_channel} != {shape_chw[0]}")
        index = self.tensor(
            name, shape_chw, layout=Layout.SEGMENTED, base_offset=0,
            allocation_bytes=0,
            flags=TENSOR_FLAG_VIEW | TENSOR_FLAG_EXTERNAL)
        self.tensors[index]["segment_offset"] = segment_offset
        self.tensors[index]["segment_count"] = len(parts)
        self.tensors[index]["segments"] = [
            {"tensor": tensor_index, "dst_channel": sum(c for _, c in parts[:i]),
             "channels": count}
            for i, (tensor_index, count) in enumerate(parts)
        ]
        return index

    def quant(self, name: str, real_scales: list[float],
              clamp_min: int = -2048, clamp_max: int = 2047,
              post_op: PostOp = PostOp.NONE) -> int:
        offset = len(self.quant_params)
        errors = []
        encoded = []
        for real_scale in real_scales:
            multiplier, shift = quantize_scale(float(real_scale))
            approx = dequantize_scale(multiplier, shift)
            errors.append(abs(approx - real_scale) / max(abs(real_scale), 1e-30))
            self.quant_params.extend(QUANT_PARAM_STRUCT.pack(
                multiplier, shift, clamp_min, clamp_max))
            encoded.append({
                "real_scale": float(real_scale),
                "multiplier": multiplier,
                "shift": shift,
                "relative_error": errors[-1],
            })
        row = {
            "name": name,
            "param_offset": offset,
            "param_count": len(real_scales),
            "clamp_min": clamp_min,
            "clamp_max": clamp_max,
            "post_op": post_op.name,
            "post_op_id": int(post_op),
            "max_scale_relative_error": max(errors, default=0.0),
            "params": encoded,
        }
        index = len(self.quant_descs)
        self.quant_descs.append(row)
        return index

    def command(self, opcode: Opcode, **kwargs: int) -> None:
        self.commands.append(Command(opcode=opcode, tag=len(self.commands), **kwargs))


def compile_program(source: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_weights = (source / "weights_int8.bin").read_bytes()
    source_bias = (source / "bias_int32.bin").read_bytes()
    source_scales = (source / "scales_f32.bin").read_bytes()
    builder = ProgramBuilder()
    packed_weights = bytearray()
    packed_bias = bytearray()

    input_shape = manifest["layers"][0]["input"]
    builder.tensor("input", input_shape)
    current_name = "input"
    layer_output_name: dict[str, str] = {}
    layer_rows = []
    event_mask = 0x0003

    for layer_index, layer in enumerate(manifest["layers"]):
        name = layer["name"]
        cin, _, _ = map(int, layer["input"])
        cout, _, _ = map(int, layer["output"])
        kh, kw = map(int, layer["kernel"])

        if name.startswith("dec"):
            previous_td = builder.tensor_by_name[current_name]
            previous_shape = builder.tensors[previous_td]["shape_chw"]
            up_shape = [previous_shape[0], previous_shape[1] * 2,
                        previous_shape[2] * 2]
            up_name = f"{name}.upsample"
            up_td = builder.tensor(up_name, up_shape)
            builder.command(Opcode.UPSAMPLE2X, dst_td=up_td,
                            src0_td=previous_td, imm=0x0003)
            skip_name = str(layer["skip"])
            skip_td = builder.tensor_by_name[layer_output_name[skip_name]]
            src_td = builder.segmented(
                f"{name}.concat", layer["input"],
                [(up_td, up_shape[0]), (skip_td, layer["input"][0] - up_shape[0])])
        else:
            src_td = builder.tensor_by_name[current_name]

        raw_weight = source_weights[
            int(layer["weight_offset"]):
            int(layer["weight_offset"]) + int(layer["weight_bytes"])]
        reordered = reorder_oihw_to_o8i8(raw_weight, cout, cin, kh, kw)
        weight_offset = add_aligned(packed_weights, reordered)
        weight_td = builder.constant(
            f"{name}.weight", base_offset=weight_offset, size=len(reordered),
            dtype=DType.INT8, layout=Layout.WEIGHT_O8I8,
            shape_chw=[align_up(cout, 8), align_up(cin, 8), kh * kw])

        raw_bias = source_bias[
            int(layer["bias_offset"]):
            int(layer["bias_offset"]) + int(layer["bias_bytes"])]
        bias_offset = add_aligned(packed_bias, raw_bias)
        bias_td = builder.constant(
            f"{name}.bias", base_offset=bias_offset, size=len(raw_bias),
            dtype=DType.INT32, layout=Layout.LINEAR,
            shape_chw=[cout, 1, 1])

        scale_offset = int(layer["scale_offset"]) + cout * 4
        requant = np.frombuffer(source_scales, dtype="<f4", count=cout,
                                offset=scale_offset).astype(np.float64)
        post_op = post_op_for(str(layer["post_op"]))
        quant_td = builder.quant(f"{name}.requant", requant.tolist(),
                                 post_op=post_op)

        residual = bool(layer.get("residual"))
        output_name = name if not residual else f"{name}.conv"
        output_td = builder.tensor(output_name, layer["output"], quant_desc=quant_td)
        plan = tile_plan(layer)
        op_row = {
            "name": name,
            "src_td": src_td,
            "dst_td": output_td,
            "weight_td": weight_td,
            "bias_td": bias_td,
            "kh": kh,
            "kw": kw,
            "stride_h": int(layer["stride"][0]),
            "stride_w": int(layer["stride"][1]),
            "dilation_h": int(layer["dilation"][0]),
            "dilation_w": int(layer["dilation"][1]),
            "pad_top": int(layer["frequency_pad_each_side"]),
            "pad_bottom": int(layer["frequency_pad_each_side"]),
            "pad_left": int(layer["time_pad_left"]),
            "pad_right": 0,
            "groups": int(layer["groups"]),
            "post_op": post_op.name,
            "post_op_id": int(post_op),
            **{key: int(plan[key]) for key in ("tile_h", "tile_w", "tile_cout")},
        }
        op_index = len(builder.operators)
        builder.operators.append(op_row)

        builder.command(Opcode.DMA_LOAD, flags=int(CommandFlag.ASYNC),
                        dst_td=src_td, src0_td=src_td, imm=0x0001)
        builder.command(Opcode.DMA_LOAD, flags=int(CommandFlag.ASYNC),
                        dst_td=weight_td, src0_td=weight_td, imm=0x0002)
        builder.command(Opcode.WAIT, imm=event_mask)
        builder.command(
            Opcode.CONV2D,
            flags=int(CommandFlag.SATURATE | CommandFlag.FUSED_POST_OP),
            dst_td=output_td, src0_td=src_td, src1_td=weight_td,
            op_desc=op_index, quant_desc=quant_td)

        final_td = output_td
        final_name = output_name
        if residual:
            residual_td = builder.tensor_by_name[current_name]
            final_name = name
            residual_scale = (float(layer["input_activation_step_int12"]) /
                              float(layer["output_activation_step_int12"]))
            add_quant = builder.quant(
                f"{name}.residual_rescale", [residual_scale],
                post_op=PostOp.NONE)
            final_td = builder.tensor(final_name, layer["output"], quant_desc=quant_td)
            builder.command(
                Opcode.VEC_ADD, flags=int(CommandFlag.SATURATE),
                dst_td=final_td, src0_td=output_td, src1_td=residual_td,
                quant_desc=add_quant)

        builder.command(Opcode.DMA_STORE, flags=int(CommandFlag.ASYNC),
                        src0_td=final_td, dst_td=final_td, imm=0x0004)
        builder.command(Opcode.WAIT, imm=0x0004)
        current_name = final_name
        layer_output_name[name] = final_name
        layer_rows.append({
            "name": name,
            "operator_desc": op_index,
            "quant_desc": quant_td,
            "input_tensor": src_td,
            "output_tensor": final_td,
            "packed_weight_offset": weight_offset,
            "packed_weight_bytes": len(reordered),
            "bias_offset": bias_offset,
            "bias_bytes": len(raw_bias),
            "compute_cycles": int(layer["scheduled_cycles_pout8_pin8"]),
            "macs": int(layer["macs_per_chunk"]),
            "tile": plan,
        })

    builder.command(Opcode.END, flags=int(CommandFlag.IRQ))

    pre_tanh_step = manifest.get("quantization", {}).get(
        "pre_tanh_activation_step_int12")
    tanh_lut = b""
    blockers = []
    if pre_tanh_step is None:
        blockers.append({
            "id": "NUM-TANH-001",
            "severity": "blocks_isa_freeze",
            "detail": (
                "The source package has no calibrated pre-tanh activation "
                "scale. Run scripts/27_calibrate_tanh.py and re-export."
            ),
        })
    else:
        qin = np.arange(-2048, 2048, dtype=np.float64)
        qout = np.clip(np.rint(np.tanh(qin * float(pre_tanh_step)) * 2047.0),
                       -2048, 2047).astype("<i2")
        tanh_lut = qout.tobytes(order="C")

    commands_bin = b"".join(command.pack() for command in builder.commands)
    tensor_bin = b"".join(pack_tensor(row) for row in builder.tensors)
    operator_bin = b"".join(pack_operator(row) for row in builder.operators)
    quant_desc_bin = b"".join(QUANT_DESC_STRUCT.pack(
        row["param_offset"], row["param_count"], row["clamp_min"],
        row["clamp_max"], row["post_op_id"], 0)
        for row in builder.quant_descs)

    compute_cycles = sum(row["compute_cycles"] for row in layer_rows)
    vector_cycles = sum(
        math.ceil(nhwc8_nbytes(layer["output"], 2) / 2 / 8)
        for layer in manifest["layers"] if layer.get("residual"))
    upsample_cycles = sum(
        math.ceil(builder.tensors[builder.tensor_by_name[f"{layer['name']}.upsample"]]
                  ["allocation_bytes"] / 2 / 8)
        for layer in manifest["layers"] if layer["name"].startswith("dec"))
    weight_bytes = (len(packed_weights) + len(packed_bias) +
                    len(builder.quant_params) + len(tanh_lut))
    activation_reads = sum(nhwc8_nbytes(layer["input"], 2)
                           for layer in manifest["layers"])
    activation_writes = sum(nhwc8_nbytes(layer["output"], 2)
                            for layer in manifest["layers"])
    descriptor_bytes = (len(commands_bin) + len(tensor_bin) + len(operator_bin) +
                        len(quant_desc_bin) + len(builder.segments))
    ddr_bytes = weight_bytes + activation_reads + activation_writes + descriptor_bytes
    dma_cycles = math.ceil(ddr_bytes / BUS_BYTES) + (
        math.ceil(ddr_bytes / BURST_BYTES) * BURST_OVERHEAD_CYCLES)
    command_cycles = len(builder.commands) * COMMAND_OVERHEAD_CYCLES
    no_overlap_cycles = (compute_cycles + vector_cycles + upsample_cycles +
                         dma_cycles + command_cycles)

    # Full-tensor DDR descriptors are deliberately simple.  This separate
    # liveness number is the lower bound for a future reusing arena allocator.
    producer = {builder.tensor_by_name["input"]: 0}
    last_use: dict[int, int] = {}
    for pc, command in enumerate(builder.commands):
        if command.dst_td != NONE_INDEX and command.opcode in {
                Opcode.CONV2D, Opcode.VEC_ADD, Opcode.UPSAMPLE2X}:
            producer[command.dst_td] = pc
        for tensor_index in (command.src0_td, command.src1_td):
            if tensor_index != NONE_INDEX:
                last_use[tensor_index] = pc
                row = builder.tensors[tensor_index]
                for segment in row.get("segments", []):
                    last_use[segment["tensor"]] = pc
    intervals = []
    for index, row in enumerate(builder.tensors):
        if row["flags"] & (TENSOR_FLAG_CONSTANT | TENSOR_FLAG_VIEW):
            continue
        intervals.append((producer.get(index, 0), last_use.get(index, len(builder.commands)),
                          row["allocation_bytes"], row["name"]))
    peak_live = 0
    peak_names: list[str] = []
    for pc in range(len(builder.commands) + 1):
        live = [(size, name) for start, end, size, name in intervals
                if start <= pc <= end]
        total = sum(size for size, _ in live)
        if total > peak_live:
            peak_live = total
            peak_names = [name for _, name in live]

    analysis = {
        "schema": 1,
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "counts": {
            "commands": len(builder.commands),
            "tensor_descriptors": len(builder.tensors),
            "operator_descriptors": len(builder.operators),
            "quant_descriptors": len(builder.quant_descs),
            "segment_records": len(builder.segments) // SEGMENT_STRUCT.size,
        },
        "memory": {
            "mutable_arena_bytes_no_reuse": align_up(builder.mutable_cursor, 64),
            "peak_live_tensor_bytes": peak_live,
            "peak_live_tensors": peak_names,
            "packed_weight_bytes": len(packed_weights),
            "packed_bias_bytes": len(packed_bias),
            "integer_quant_param_bytes": len(builder.quant_params),
            "tanh_lut_bytes": len(tanh_lut),
            "estimated_ddr_bytes_per_task": ddr_bytes,
        },
        "cycles": {
            "compute": compute_cycles,
            "vector": vector_cycles,
            "upsample": upsample_cycles,
            "dma_no_overlap": dma_cycles,
            "command": command_cycles,
            "total_no_overlap": no_overlap_cycles,
            "budget": 4_000_000,
            "budget_pass": no_overlap_cycles <= 4_000_000,
            "milliseconds_at_200mhz": no_overlap_cycles / TARGET_CLOCK_HZ * 1000.0,
        },
        "tile_checks_pass": all(
            bool(value) for row in layer_rows for key, value in row["tile"].items()
            if key.startswith("fits_")),
        "tanh_lut": {
            "generated": bool(tanh_lut),
            "entries": len(tanh_lut) // 2,
            "input_step_int12": pre_tanh_step,
            "output_step_int12": 1.0 / 2047.0,
        },
        "isa_freeze_blockers": blockers,
        "limitations": [
            "Commands describe layer-level data movement; scratchpad bank assignment "
            "and tile-level DMA expansion remain a P3 task.",
            "The mutable DDR arena uses simple non-overlapping allocations; the "
            "liveness report provides the input to a later reuse allocator.",
        ],
        "layers": layer_rows,
    }

    output.mkdir(parents=True, exist_ok=True)
    files = {
        "commands.bin": commands_bin,
        "tensor_desc.bin": tensor_bin,
        "operator_desc.bin": operator_bin,
        "quant_desc.bin": quant_desc_bin,
        "quant_params.bin": bytes(builder.quant_params),
        "segments.bin": bytes(builder.segments),
        "weights_o8i8.bin": bytes(packed_weights),
        "bias_int32.bin": bytes(packed_bias),
    }
    if tanh_lut:
        files["tanh_lut_int12.bin"] = tanh_lut
    for filename, payload in files.items():
        (output / filename).write_bytes(payload)

    program = {
        "schema": 1,
        "isa": {"major": ISA_MAJOR, "minor": ISA_MINOR,
                "command_bytes": len(Command(opcode=Opcode.NOP).pack())},
        "target": manifest["target"],
        "source": {
            "package": source.relative_to(ROOT).as_posix(),
            "manifest_sha256": analysis["source_manifest_sha256"],
        },
        "entry": {"input_tensor": builder.tensor_by_name["input"],
                  "output_tensor": builder.tensor_by_name[current_name]},
        "commands": [
            {**command.__dict__, "opcode": command.opcode.name,
             "flags": int(command.flags)} for command in builder.commands],
        "tensors": builder.tensors,
        "operators": builder.operators,
        "quantization": builder.quant_descs,
        "files": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                  for name, payload in files.items()},
    }
    with (output / "program.json").open(
            "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(program, indent=2, ensure_ascii=False))
        stream.write("\n")
    with (output / "analysis.json").open(
            "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(analysis, indent=2, ensure_ascii=False))
        stream.write("\n")
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    analysis = compile_program(args.source.resolve(), args.out.resolve())
    print(f"wrote {args.out.resolve()}")
    print(json.dumps({"counts": analysis["counts"], "memory": analysis["memory"],
                      "cycles": analysis["cycles"],
                      "tile_checks_pass": analysis["tile_checks_pass"],
                      "isa_freeze_blockers": analysis["isa_freeze_blockers"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
