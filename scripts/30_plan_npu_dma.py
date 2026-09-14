"""Resolve every tile DMA command into checked DDR/scratchpad geometry.

The JSON output is the executable reference for ``npu_dma_agu.sv``.  Addresses
are offsets within the task's activation/weight/bias/quant sections so a driver
can relocate a task without rewriting descriptors.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from npu_isa import (
    IMM_ACTIVATION_BANK_BIT,
    IMM_DMA_VECTOR_QUANT_BIT,
    IMM_OUTPUT_BANK_BIT,
    IMM_SEGMENT_LSB,
    IMM_SEGMENTED_BIT,
    IMM_WEIGHT_BANK_BIT,
    NONE_INDEX,
    align_up,
)


ROOT = Path(__file__).resolve().parent.parent
PROGRAM_DEFAULT = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
BANK_CAPACITY = {"A0": 64 * 1024, "A1": 64 * 1024,
                 "W0": 32 * 1024, "W1": 32 * 1024,
                 "O0": 16 * 1024, "O1": 16 * 1024}


def _end(request: dict[str, Any], *, scratchpad: bool) -> int:
    prefix = "scratchpad" if scratchpad else "external"
    return (int(request[f"{prefix}_offset"])
            + (int(request["z_count"]) - 1)
            * int(request[f"{prefix}_z_stride"])
            + (int(request["y_count"]) - 1)
            * int(request[f"{prefix}_y_stride"])
            + int(request["x_bytes"]))


def _base_request(command_index: int, command: dict[str, Any],
                  memory_space: str, scratchpad: str, bank: int
                  ) -> dict[str, Any]:
    return {
        "command_index": command_index,
        "tag": int(command["tag"]),
        "opcode": command["opcode"],
        "memory_space": memory_space,
        "scratchpad": f"{scratchpad}{bank}",
        "event_mask": int(command["imm"]) & 0xFF,
        "external_offset": 0,
        "scratchpad_offset": 0,
        "x_bytes": 0,
        "y_count": 1,
        "z_count": 1,
        "external_y_stride": 0,
        "external_z_stride": 0,
        "scratchpad_y_stride": 0,
        "scratchpad_z_stride": 0,
        "clear_before": False,
        "clear_bytes": 0,
        "descriptor_dependencies": [],
    }


def _weight_layout(operator: dict[str, Any]) -> dict[str, int]:
    physical_cout = align_up(int(operator["tile_cout"]), 8)
    cin = align_up(int(operator["input_channel_count"]), 8)
    weight = physical_cout * cin * int(operator["kh"]) * int(operator["kw"])
    bias = int(operator["tile_cout"]) * 4
    quant = int(operator["tile_cout"]) * 16
    bias_offset = align_up(weight, 64)
    quant_offset = align_up(bias_offset + bias, 64)
    return {
        "weight_bytes": weight,
        "bias_bytes": bias,
        "quant_bytes": quant,
        "bias_offset": bias_offset,
        "quant_offset": quant_offset,
        "vector_quant_offset": align_up(quant_offset + quant, 64),
    }


def resolve(command_index: int, command: dict[str, Any],
            program: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    tensors = program["tensors"]
    quant_descs = program["quantization"]
    operator_index = int(command["op_desc"])
    operator = schedule["operator_descriptors"][operator_index]
    tile = schedule["tiles"][operator_index]
    imm = int(command["imm"])
    wlayout = _weight_layout(operator)

    if command["opcode"] == "DMA_STORE":
        tensor_index = int(command["src0_td"])
        tensor = tensors[tensor_index]
        bank = (imm >> IMM_OUTPUT_BANK_BIT) & 1
        request = _base_request(command_index, command, "activation", "O", bank)
        request.update({
            "external_offset": (int(tensor["base_offset"])
                                + int(operator["tile_origin_h"])
                                * int(tensor["strides"][1])
                                + int(operator["tile_origin_w"])
                                * int(tensor["strides"][2])
                                + int(operator["tile_origin_cout"])
                                * int(tensor["strides"][3])),
            "x_bytes": int(operator["tile_cout"]) * int(tensor["strides"][3]),
            "y_count": int(operator["tile_w"]),
            "z_count": int(operator["tile_h"]),
            "external_y_stride": int(tensor["strides"][2]),
            "external_z_stride": int(tensor["strides"][1]),
            "scratchpad_y_stride": (align_up(int(operator["tile_cout"]), 8)
                                     * int(tensor["strides"][3])),
            "scratchpad_z_stride": (int(operator["tile_w"])
                                     * align_up(int(operator["tile_cout"]), 8)
                                     * int(tensor["strides"][3])),
            "descriptor_dependencies": [f"operator:{operator_index}",
                                        f"tensor:{tensor_index}"],
        })
        if tensor_index != int(command["dst_td"]) \
                or tensor_index != int(tile["dst_td"]):
            raise ValueError(f"store {command_index}: tensor mismatch")
        allocation_end = int(tensor["base_offset"]) + int(tensor["allocation_bytes"])
    elif int(command["quant_desc"]) != NONE_INDEX:
        quant_index = int(command["quant_desc"])
        quant = quant_descs[quant_index]
        vector = bool((imm >> IMM_DMA_VECTOR_QUANT_BIT) & 1)
        bank = (imm >> IMM_WEIGHT_BANK_BIT) & 1
        request = _base_request(command_index, command, "quant", "W", bank)
        request.update({
            "external_offset": (int(quant["param_offset"])
                                + (0 if vector else
                                   int(operator["tile_origin_cout"]) * 16)),
            "scratchpad_offset": (wlayout["vector_quant_offset"] if vector
                                  else wlayout["quant_offset"]),
            "x_bytes": 16 if vector else wlayout["quant_bytes"],
            "descriptor_dependencies": [f"operator:{operator_index}",
                                        f"quant:{quant_index}"],
            "quant_slot": "vector" if vector else "convolution",
        })
        required = 1 if vector else (int(operator["tile_origin_cout"])
                                     + int(operator["tile_cout"]))
        if int(quant["param_count"]) < required:
            raise ValueError(f"quant load {command_index}: parameter bounds")
        allocation_end = None
    else:
        tensor_index = int(command["src0_td"])
        tensor = tensors[tensor_index]
        layout = tensor["layout"]
        bank = (imm >> IMM_WEIGHT_BANK_BIT) & 1
        if layout == "WEIGHT_O8I8":
            request = _base_request(command_index, command, "weight", "W", bank)
            block = (align_up(int(operator["input_channel_count"]), 8)
                     * int(operator["kh"]) * int(operator["kw"]) * 8)
            request.update({
                "external_offset": (int(tensor["base_offset"])
                                    + (int(operator["tile_origin_cout"]) // 8)
                                    * block),
                "x_bytes": wlayout["weight_bytes"],
                "descriptor_dependencies": [f"operator:{operator_index}",
                                            f"tensor:{tensor_index}"],
            })
        elif layout == "LINEAR" and tensor["dtype"] == "INT32":
            request = _base_request(command_index, command, "bias", "W", bank)
            request.update({
                "external_offset": (int(tensor["base_offset"])
                                    + int(operator["tile_origin_cout"]) * 4),
                "scratchpad_offset": wlayout["bias_offset"],
                "x_bytes": wlayout["bias_bytes"],
                "descriptor_dependencies": [f"operator:{operator_index}",
                                            f"tensor:{tensor_index}"],
            })
        else:
            logical_index = int(command["dst_td"])
            logical = tensors[logical_index]
            segmented = bool((imm >> IMM_SEGMENTED_BIT) & 1)
            segment_index = (imm >> IMM_SEGMENT_LSB) & 3
            bank = (imm >> IMM_ACTIVATION_BANK_BIT) & 1
            request = _base_request(command_index, command, "activation", "A", bank)
            local_h = ((int(operator["tile_h"]) - 1) * int(operator["stride_h"])
                       + int(operator["dilation_h"]) * (int(operator["kh"]) - 1) + 1)
            local_w = ((int(operator["tile_w"]) - 1) * int(operator["stride_w"])
                       + int(operator["dilation_w"]) * (int(operator["kw"]) - 1) + 1)
            local_c_bytes = align_up(int(operator["input_channel_count"]), 8) * 2
            raw_h0 = (int(operator["tile_origin_h"]) * int(operator["stride_h"])
                      - int(operator["pad_top"]))
            raw_w0 = (int(operator["tile_origin_w"]) * int(operator["stride_w"])
                      - int(operator["pad_left"]))
            h0, w0 = max(raw_h0, 0), max(raw_w0, 0)
            h1 = min(raw_h0 + local_h, int(tensor["shape_chw"][1]))
            w1 = min(raw_w0 + local_w, int(tensor["shape_chw"][2]))
            destination_channel = 0
            channel_start = int(operator["input_channel_start"])
            channel_count = int(operator["input_channel_count"])
            dependencies = [f"operator:{operator_index}",
                            f"tensor:{tensor_index}"]
            if segmented:
                segment = logical["segments"][segment_index]
                segment_record = int(logical["segment_offset"]) // 8 + segment_index
                if int(segment["tensor"]) != tensor_index:
                    raise ValueError(f"activation {command_index}: segment tensor")
                destination_channel = int(segment["dst_channel"])
                channel_start = 0
                channel_count = int(segment["channels"])
                dependencies += [f"tensor:{logical_index}",
                                 f"segment:{segment_record}"]
            request.update({
                "external_offset": (int(tensor["base_offset"])
                                    + h0 * int(tensor["strides"][1])
                                    + w0 * int(tensor["strides"][2])
                                    + channel_start * int(tensor["strides"][3])),
                "scratchpad_offset": ((max(-raw_h0, 0) * local_w
                                       + max(-raw_w0, 0)) * local_c_bytes
                                      + destination_channel * 2),
                "x_bytes": channel_count * 2,
                "y_count": w1 - w0,
                "z_count": h1 - h0,
                "external_y_stride": int(tensor["strides"][2]),
                "external_z_stride": int(tensor["strides"][1]),
                "scratchpad_y_stride": local_c_bytes,
                "scratchpad_z_stride": local_w * local_c_bytes,
                "clear_before": not segmented or segment_index == 0,
                "clear_bytes": local_h * local_w * local_c_bytes,
                "descriptor_dependencies": dependencies,
                "local_shape_hwc": [local_h, local_w,
                                    align_up(int(operator["input_channel_count"]), 8)],
            })
            if logical_index != int(operator["src_td"]):
                raise ValueError(f"activation {command_index}: logical tensor")
        allocation_end = int(tensor["base_offset"]) + int(tensor["allocation_bytes"])

    if request["x_bytes"] <= 0 or request["y_count"] <= 0 or request["z_count"] <= 0:
        raise ValueError(f"DMA {command_index}: empty request")
    request["payload_bytes"] = (int(request["x_bytes"])
                                * int(request["y_count"])
                                * int(request["z_count"]))
    request["external_end"] = _end(request, scratchpad=False)
    request["scratchpad_end"] = _end(request, scratchpad=True)
    if allocation_end is not None and request["external_end"] > allocation_end:
        raise ValueError(f"DMA {command_index}: external allocation bounds")
    if request["scratchpad_end"] > BANK_CAPACITY[request["scratchpad"]]:
        raise ValueError(f"DMA {command_index}: scratchpad bounds")
    if request["clear_bytes"] > BANK_CAPACITY[request["scratchpad"]]:
        raise ValueError(f"DMA {command_index}: clear bounds")
    return request


def build_plan(program_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    program = json.loads((program_dir / "program.json").read_text(encoding="utf-8"))
    schedule = json.loads((program_dir / "tile_schedule.json").read_text(encoding="utf-8"))
    requests = [
        resolve(index, command, program, schedule)
        for index, command in enumerate(schedule["commands"])
        if command["opcode"] in {"DMA_LOAD", "DMA_STORE"}
    ]
    kinds = Counter(request["memory_space"] for request in requests)
    max_scratchpad_end: dict[str, int] = {}
    for request in requests:
        bank = request["scratchpad"]
        max_scratchpad_end[bank] = max(
            max_scratchpad_end.get(bank, 0), int(request["scratchpad_end"]),
            int(request["clear_bytes"]))
    analysis = {
        "schema": 1,
        "status": "P4-proposal",
        "request_count": len(requests),
        "load_count": sum(r["opcode"] == "DMA_LOAD" for r in requests),
        "store_count": sum(r["opcode"] == "DMA_STORE" for r in requests),
        "request_count_by_space": dict(sorted(kinds.items())),
        "payload_bytes": sum(int(r["payload_bytes"]) for r in requests),
        "activation_clear_bytes": sum(int(r["clear_bytes"])
                                      for r in requests if r["clear_before"]),
        "maximum_scratchpad_end": dict(sorted(max_scratchpad_end.items())),
        "bounds_pass": True,
    }
    output = {"schema": 1, "status": "P4-proposal",
              "source_program": program_dir.name,
              "analysis": analysis, "requests": requests}
    output_dir = output_dir or program_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dma_plan.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (output_dir / "dma_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=PROGRAM_DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_plan(
        args.program.resolve(), args.output.resolve() if args.output else None),
        indent=2))


if __name__ == "__main__":
    main()
