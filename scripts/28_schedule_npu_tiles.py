"""Expand the layer program into a deterministic tile-level NPU schedule.

The scheduler is an architecture model, not a board-loadable runtime yet.  It
assigns scratchpad banks, emits tile descriptors and commands, and simulates
DMA/compute overlap while checking bank access intervals for hazards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from npu_isa import (
    NONE_INDEX,
    OPERATOR_STRUCT,
    Command,
    CommandFlag,
    Event,
    Opcode,
    align_up,
    encode_control_imm,
)


ROOT = Path(__file__).resolve().parent.parent
PROGRAM_DEFAULT = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"

ACTIVATION_BYTES = 2
ACCUMULATOR_BYTES = 4
BUS_BYTES = 8
BURST_BYTES = 1024
BURST_OVERHEAD_CYCLES = 16
COMMAND_OVERHEAD_CYCLES = 4
TARGET_CLOCK_HZ = 200_000_000

BANK_CAPACITY = {
    "A0": 64 * 1024,
    "A1": 64 * 1024,
    "W0": 32 * 1024,
    "W1": 32 * 1024,
    "P0": 16 * 1024,
    "P1": 16 * 1024,
    "O0": 16 * 1024,
    "O1": 16 * 1024,
}

EVENT = {event.name: int(event) for event in Event}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dma_cycles(byte_counts: list[int]) -> int:
    return sum(
        math.ceil(size / BUS_BYTES)
        + math.ceil(size / BURST_BYTES) * BURST_OVERHEAD_CYCLES
        for size in byte_counts if size
    )


def pack_operator(row: dict[str, Any]) -> bytes:
    fields16 = (
        row["src_td"], row["dst_td"], row["weight_td"], row["bias_td"],
        row["kh"], row["kw"], row["stride_h"], row["stride_w"],
        row["dilation_h"], row["dilation_w"], row["pad_top"],
        row["pad_bottom"], row["pad_left"], row["pad_right"],
        row["groups"], row["post_op_id"],
    )
    fields32 = (
        row["tile_origin_h"], row["tile_origin_w"],
        row["tile_origin_cout"], row["tile_h"], row["tile_w"],
        row["tile_cout"], row["input_channel_start"],
        row["input_channel_count"],
    )
    return OPERATOR_STRUCT.pack(*fields16, *fields32)


class CommandBuilder:
    def __init__(self) -> None:
        self.commands: list[Command] = []

    def add(self, opcode: Opcode, **kwargs: int) -> int:
        index = len(self.commands)
        self.commands.append(Command(opcode=opcode, tag=index, **kwargs))
        return index


def input_region(operator: dict[str, Any], tile_h0: int,
                 tile_h: int, tile_w: int,
                 source: dict[str, Any]) -> dict[str, int]:
    _, hin, win = map(int, source["shape_chw"])
    raw_h0 = tile_h0 * int(operator["stride_h"]) - int(operator["pad_top"])
    raw_h1 = ((tile_h0 + tile_h - 1) * int(operator["stride_h"])
              - int(operator["pad_top"])
              + int(operator["dilation_h"]) * (int(operator["kh"]) - 1) + 1)
    raw_w0 = -int(operator["pad_left"])
    raw_w1 = ((tile_w - 1) * int(operator["stride_w"])
              - int(operator["pad_left"])
              + int(operator["dilation_w"]) * (int(operator["kw"]) - 1) + 1)
    h0, h1 = max(raw_h0, 0), min(raw_h1, hin)
    w0, w1 = max(raw_w0, 0), min(raw_w1, win)
    return {
        "h_start": h0,
        "h_count": max(h1 - h0, 0),
        "w_start": w0,
        "w_count": max(w1 - w0, 0),
        "pad_top": max(-raw_h0, 0),
        "pad_bottom": max(raw_h1 - hin, 0),
        "pad_left": max(-raw_w0, 0),
        "pad_right": max(raw_w1 - win, 0),
    }


def make_tiles(program: dict[str, Any], manifest: dict[str, Any],
               analysis: dict[str, Any]) -> tuple[list[dict[str, Any]],
                                                  list[dict[str, Any]]]:
    tensors = program["tensors"]
    analysis_layers = {row["name"]: row for row in analysis["layers"]}
    manifest_layers = {row["name"]: row for row in manifest["layers"]}
    descriptors: list[dict[str, Any]] = []
    tiles: list[dict[str, Any]] = []

    for layer_index, operator in enumerate(program["operators"]):
        name = operator["name"]
        layer = manifest_layers[name]
        layer_info = analysis_layers[name]
        source = tensors[int(operator["src_td"])]
        residual = bool(layer.get("residual"))
        vector_quant_desc = NONE_INDEX
        if residual:
            matches = [
                int(command["quant_desc"])
                for command in program["commands"]
                if command["opcode"] == "VEC_ADD"
                and int(command["dst_td"]) == int(layer_info["output_tensor"])
                and int(command["src0_td"]) == int(operator["dst_td"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"layer {name} has {len(matches)} residual quant descriptors")
            vector_quant_desc = matches[0]
        _, hout, wout = map(int, layer["output"])
        cin = int(layer["input"][0])
        cout = int(layer["output"][0])
        tile_h_max = int(operator["tile_h"])
        tile_cout_max = int(operator["tile_cout"])
        spatial_index = 0

        for h0 in range(0, hout, tile_h_max):
            th = min(tile_h_max, hout - h0)
            region = input_region(operator, h0, th, wout, source)
            input_parts = []
            if source["layout"] == "SEGMENTED":
                for segment_index, segment in enumerate(source["segments"]):
                    part_td = int(segment["tensor"])
                    channels = int(segment["channels"])
                    byte_count = (region["h_count"] * region["w_count"]
                                  * align_up(channels, 8) * ACTIVATION_BYTES)
                    input_parts.append({
                        "segment": segment_index,
                        "tensor": part_td,
                        "dst_channel": int(segment["dst_channel"]),
                        "channels": channels,
                        "bytes": byte_count,
                    })
            else:
                input_parts.append({
                    "segment": 0,
                    "tensor": int(operator["src_td"]),
                    "dst_channel": 0,
                    "channels": cin,
                    "bytes": (region["h_count"] * region["w_count"]
                              * align_up(cin, 8) * ACTIVATION_BYTES),
                })

            for c0 in range(0, cout, tile_cout_max):
                tc = min(tile_cout_max, cout - c0)
                physical_cout = align_up(tc, 8)
                weight_bytes = (physical_cout * align_up(cin, 8)
                                * int(operator["kh"]) * int(operator["kw"]))
                bias_bytes = tc * 4
                quant_bytes = tc * 16
                vector_quant_bytes = 16 if residual else 0
                bias_bank_offset = align_up(weight_bytes, 64)
                quant_bank_offset = align_up(bias_bank_offset + bias_bytes, 64)
                vector_quant_bank_offset = align_up(
                    quant_bank_offset + quant_bytes, 64)
                weight_bank_footprint = (
                    vector_quant_bank_offset + vector_quant_bytes
                    if residual else quant_bank_offset + quant_bytes)
                output_bytes = th * wout * physical_cout * ACTIVATION_BYTES
                accumulator_bytes = th * wout * physical_cout * ACCUMULATOR_BYTES
                descriptor = {
                    **operator,
                    "tile_origin_h": h0,
                    "tile_origin_w": 0,
                    "tile_origin_cout": c0,
                    "tile_h": th,
                    "tile_w": wout,
                    "tile_cout": tc,
                    "input_channel_start": 0,
                    "input_channel_count": cin,
                }
                descriptor_index = len(descriptors)
                descriptors.append(descriptor)
                tile_index = len(tiles)
                bank = tile_index & 1
                activation_bank = spatial_index & 1
                tiles.append({
                    "index": tile_index,
                    "layer_index": layer_index,
                    "layer": name,
                    "operator_desc": descriptor_index,
                    "quant_desc": int(layer_info["quant_desc"]),
                    "vector_quant_desc": vector_quant_desc,
                    "src_td": int(operator["src_td"]),
                    "conv_dst_td": int(operator["dst_td"]),
                    "dst_td": int(layer_info["output_tensor"]),
                    "weight_td": int(operator["weight_td"]),
                    "bias_td": int(operator["bias_td"]),
                    "residual": residual,
                    "origin": {"h": h0, "w": 0, "cout": c0},
                    "shape": {"h": th, "w": wout, "cout": tc},
                    "input_region": region,
                    "input_parts": input_parts,
                    "loads_input": c0 == 0,
                    "spatial_index": spatial_index,
                    "banks": {
                        "activation": f"A{activation_bank}",
                        "weight": f"W{bank}",
                        "accumulator": f"P{bank}",
                        "output": f"O{bank}",
                    },
                    "bank_ids": {
                        "activation": activation_bank,
                        "weight": bank,
                        "accumulator": bank,
                        "output": bank,
                    },
                    "bytes": {
                        "input": sum(part["bytes"] for part in input_parts),
                        "weight": weight_bytes,
                        "bias": bias_bytes,
                        "quant": quant_bytes,
                        "vector_quant": vector_quant_bytes,
                        "weight_bank_footprint": weight_bank_footprint,
                        "output": output_bytes,
                        "accumulator": accumulator_bytes,
                    },
                    "weight_bank_layout": {
                        "weight": 0,
                        "bias": bias_bank_offset,
                        "conv_quant": quant_bank_offset,
                        "vector_quant": (vector_quant_bank_offset
                                         if residual else None),
                        "alignment": 64,
                    },
                    "cycles": {
                        "compute": (th * wout * math.ceil(tc / 8)
                                    * math.ceil(cin / 8)
                                    * int(operator["kh"]) * int(operator["kw"])),
                        "vector": (math.ceil(th * wout * physical_cout / 8)
                                   if layer.get("residual") else 0),
                    },
                })
            spatial_index += 1
    return descriptors, tiles


def add_input_loads(builder: CommandBuilder, tile: dict[str, Any]) -> list[int]:
    if not tile["loads_input"]:
        return []
    command_indices = []
    parts = tile["input_parts"]
    aid = tile["bank_ids"]["activation"]
    for index, part in enumerate(parts):
        event = EVENT[f"A{aid}_READY"] if index == len(parts) - 1 else 0
        command_indices.append(builder.add(
            Opcode.DMA_LOAD, flags=int(CommandFlag.ASYNC),
            dst_td=tile["src_td"], src0_td=part["tensor"],
            op_desc=tile["operator_desc"],
            imm=encode_control_imm(event=event, activation=aid,
                                   segmented=len(parts) > 1,
                                   segment=part["segment"])))
    return command_indices


def add_weight_loads(builder: CommandBuilder, tile: dict[str, Any]) -> list[int]:
    wid = tile["bank_ids"]["weight"]
    common = {
        "flags": int(CommandFlag.ASYNC),
        "op_desc": tile["operator_desc"],
    }
    commands = [
        builder.add(Opcode.DMA_LOAD, dst_td=tile["weight_td"],
                    src0_td=tile["weight_td"],
                    imm=encode_control_imm(weight=wid), **common),
        builder.add(Opcode.DMA_LOAD, dst_td=tile["bias_td"],
                    src0_td=tile["bias_td"],
                    imm=encode_control_imm(weight=wid), **common),
        builder.add(Opcode.DMA_LOAD, dst_td=NONE_INDEX, src0_td=NONE_INDEX,
                    quant_desc=tile["quant_desc"],
                    imm=encode_control_imm(
                        event=(0 if tile["residual"] else
                               EVENT[f"W{wid}_READY"]), weight=wid), **common),
    ]
    if tile["residual"]:
        commands.append(builder.add(
            Opcode.DMA_LOAD, dst_td=NONE_INDEX, src0_td=NONE_INDEX,
            quant_desc=tile["vector_quant_desc"],
            imm=encode_control_imm(event=EVENT[f"W{wid}_READY"], weight=wid),
            **common))
    return commands


def emit_commands(program: dict[str, Any], manifest: dict[str, Any],
                  tiles: list[dict[str, Any]]) -> tuple[CommandBuilder,
                                                       list[dict[str, Any]]]:
    builder = CommandBuilder()
    layer_tiles: list[dict[str, Any]] = []
    previous_dst = int(program["entry"]["input_tensor"])

    for layer_index, layer in enumerate(manifest["layers"]):
        current = [tile for tile in tiles if tile["layer_index"] == layer_index]
        if not current:
            raise ValueError(f"layer {layer['name']} has no tiles")
        layer_command_start = len(builder.commands)

        if str(layer["name"]).startswith("dec"):
            up_name = f"{layer['name']}.upsample"
            up_td = next(i for i, row in enumerate(program["tensors"])
                         if row["name"] == up_name)
            builder.add(Opcode.UPSAMPLE2X, dst_td=up_td, src0_td=previous_dst)

        current[0]["input_load_commands"] = add_input_loads(builder, current[0])
        current[0]["weight_load_commands"] = add_weight_loads(builder, current[0])
        output_bank_used = [False, False]

        for local_index, tile in enumerate(current):
            aid = tile["bank_ids"]["activation"]
            wid = tile["bank_ids"]["weight"]
            oid = tile["bank_ids"]["output"]
            wait_mask = EVENT[f"W{wid}_READY"]
            if tile["loads_input"]:
                wait_mask |= EVENT[f"A{aid}_READY"]
            if output_bank_used[oid]:
                wait_mask |= EVENT[f"S{oid}_DONE"]
            tile["wait_command"] = builder.add(Opcode.WAIT, imm=wait_mask)
            tile["compute_command"] = builder.add(
                Opcode.CONV2D,
                flags=int(CommandFlag.ASYNC | CommandFlag.SATURATE
                          | CommandFlag.FUSED_POST_OP),
                dst_td=tile["conv_dst_td"], src0_td=tile["src_td"],
                src1_td=tile["weight_td"], op_desc=tile["operator_desc"],
                quant_desc=tile["quant_desc"],
                imm=encode_control_imm(
                    event=EVENT[f"C{oid}_DONE"], activation=aid, weight=wid,
                    accumulator=tile["bank_ids"]["accumulator"], output=oid,
                    segmented=len(tile["input_parts"]) > 1))

            if local_index + 1 < len(current):
                following = current[local_index + 1]
                following["input_load_commands"] = add_input_loads(builder, following)
                following["weight_load_commands"] = add_weight_loads(builder, following)

            tile["compute_wait_command"] = builder.add(
                Opcode.WAIT, imm=EVENT[f"C{oid}_DONE"])
            if tile["residual"]:
                tile["vector_command"] = builder.add(
                    Opcode.VEC_ADD, flags=int(CommandFlag.SATURATE),
                    dst_td=tile["dst_td"], src0_td=tile["conv_dst_td"],
                    src1_td=tile["src_td"], op_desc=tile["operator_desc"],
                    quant_desc=tile["vector_quant_desc"],
                    imm=encode_control_imm(activation=aid, output=oid))
            tile["store_command"] = builder.add(
                Opcode.DMA_STORE, flags=int(CommandFlag.ASYNC),
                dst_td=tile["dst_td"], src0_td=tile["dst_td"],
                op_desc=tile["operator_desc"],
                imm=encode_control_imm(event=EVENT[f"S{oid}_DONE"],
                                       output=oid))
            output_bank_used[oid] = True

        final_store_mask = sum(EVENT[f"S{i}_DONE"]
                               for i, used in enumerate(output_bank_used) if used)
        builder.add(Opcode.WAIT, imm=final_store_mask)
        previous_dst = current[-1]["dst_td"]
        layer_tiles.append({
            "name": layer["name"],
            "first_tile": current[0]["index"],
            "tile_count": len(current),
            "command_start": layer_command_start,
            "command_count": len(builder.commands) - layer_command_start,
        })

    builder.add(Opcode.END, flags=int(CommandFlag.IRQ))
    return builder, layer_tiles


def simulate(program: dict[str, Any], manifest: dict[str, Any],
             tiles: list[dict[str, Any]], command_count: int
             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tensors = program["tensors"]
    accesses: list[dict[str, Any]] = []
    elapsed = 0
    total_compute = total_vector = total_dma_read = total_dma_write = 0
    total_upsample = 0
    bank_reuse_stall = 0
    layer_rows = []
    previous_dst = int(program["entry"]["input_tensor"])

    for layer_index, layer in enumerate(manifest["layers"]):
        current = [tile for tile in tiles if tile["layer_index"] == layer_index]
        layer_start = elapsed
        upsample_cycles = 0
        if str(layer["name"]).startswith("dec"):
            up_td = next(i for i, row in enumerate(tensors)
                         if row["name"] == f"{layer['name']}.upsample")
            source_bytes = int(tensors[previous_dst]["allocation_bytes"])
            output_bytes = int(tensors[up_td]["allocation_bytes"])
            vector = math.ceil(output_bytes / ACTIVATION_BYTES / 8)
            transfer = dma_cycles([source_bytes]) + dma_cycles([output_bytes])
            upsample_cycles = vector + transfer
            elapsed += upsample_cycles
            total_upsample += upsample_cycles

        read_free = write_free = compute_free = elapsed
        activation_read_until = [elapsed, elapsed]
        weight_read_until = [elapsed, elapsed]
        output_store_until = [elapsed, elapsed]
        input_ready: dict[int, int] = {}
        previous_compute_start = elapsed

        for local_index, tile in enumerate(current):
            aid = tile["bank_ids"]["activation"]
            wid = tile["bank_ids"]["weight"]
            pid = tile["bank_ids"]["accumulator"]
            oid = tile["bank_ids"]["output"]
            release = elapsed if local_index == 0 else previous_compute_start

            if tile["loads_input"]:
                unconstrained = max(read_free, release)
                load_start = max(unconstrained, activation_read_until[aid])
                bank_reuse_stall += load_start - unconstrained
                input_sizes = [part["bytes"] for part in tile["input_parts"]]
                input_end = load_start + dma_cycles(input_sizes)
                accesses.append({
                    "bank": f"A{aid}", "mode": "write", "kind": "dma_load",
                    "start": load_start, "end": input_end,
                    "tile": tile["index"],
                })
                read_free = input_end
                input_ready[tile["spatial_index"]] = input_end
                total_dma_read += input_end - load_start
            else:
                input_end = input_ready[tile["spatial_index"]]

            unconstrained = max(read_free, release)
            weight_start = max(unconstrained, weight_read_until[wid])
            bank_reuse_stall += weight_start - unconstrained
            weight_sizes = [tile["bytes"][key] for key in
                            ("weight", "bias", "quant", "vector_quant")]
            weight_end = weight_start + dma_cycles(weight_sizes)
            accesses.append({
                "bank": f"W{wid}", "mode": "write", "kind": "dma_load",
                "start": weight_start, "end": weight_end,
                "tile": tile["index"],
            })
            read_free = weight_end
            total_dma_read += weight_end - weight_start

            unconstrained_compute = max(compute_free, input_end, weight_end)
            compute_start = max(unconstrained_compute, output_store_until[oid])
            bank_reuse_stall += compute_start - unconstrained_compute
            compute_end = compute_start + tile["cycles"]["compute"]
            vector_end = compute_end + tile["cycles"]["vector"]
            previous_compute_start = compute_start
            compute_free = vector_end
            activation_read_until[aid] = vector_end
            weight_read_until[wid] = compute_end
            total_compute += tile["cycles"]["compute"]
            total_vector += tile["cycles"]["vector"]

            accesses.extend([
                {"bank": f"A{aid}", "mode": "read", "kind": "compute",
                 "start": compute_start, "end": vector_end,
                 "tile": tile["index"]},
                {"bank": f"W{wid}", "mode": "read", "kind": "compute",
                 "start": compute_start, "end": compute_end,
                 "tile": tile["index"]},
                {"bank": f"P{pid}", "mode": "write", "kind": "accumulate",
                 "start": compute_start, "end": compute_end,
                 "tile": tile["index"]},
                {"bank": f"O{oid}", "mode": "write", "kind": "post",
                 "start": compute_start, "end": vector_end,
                 "tile": tile["index"]},
            ])

            store_start = max(write_free, vector_end)
            store_end = store_start + dma_cycles([tile["bytes"]["output"]])
            accesses.append({
                "bank": f"O{oid}", "mode": "read", "kind": "dma_store",
                "start": store_start, "end": store_end,
                "tile": tile["index"],
            })
            write_free = store_end
            output_store_until[oid] = store_end
            total_dma_write += store_end - store_start
            tile["timing"] = {
                "input_ready": input_end,
                "weight_load_start": weight_start,
                "weight_ready": weight_end,
                "compute_start": compute_start,
                "compute_end": compute_end,
                "result_ready": vector_end,
                "store_start": store_start,
                "store_end": store_end,
            }

        elapsed = max(read_free, write_free, compute_free)
        previous_dst = current[-1]["dst_td"]
        layer_rows.append({
            "name": layer["name"],
            "start": layer_start,
            "end": elapsed,
            "cycles": elapsed - layer_start,
            "upsample_cycles": upsample_cycles,
            "tile_count": len(current),
        })

    conflicts = []
    by_bank: dict[str, list[dict[str, Any]]] = {}
    for access in accesses:
        by_bank.setdefault(access["bank"], []).append(access)
    for bank, rows in by_bank.items():
        ordered = sorted(rows, key=lambda row: (row["start"], row["end"]))
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]:
                if second["start"] >= first["end"]:
                    break
                if "write" in (first["mode"], second["mode"]):
                    conflicts.append({"bank": bank, "first": first, "second": second})

    command_cycles = command_count * COMMAND_OVERHEAD_CYCLES
    no_overlap = (total_compute + total_vector + total_dma_read
                  + total_dma_write + total_upsample + command_cycles)
    total = elapsed + command_cycles
    cycle_report = {
        "compute": total_compute,
        "vector": total_vector,
        "upsample_including_ddr": total_upsample,
        "dma_read": total_dma_read,
        "dma_write": total_dma_write,
        "command": command_cycles,
        "no_overlap": no_overlap,
        "hidden_by_overlap": no_overlap - total,
        "bank_reuse_stall": bank_reuse_stall,
        "total": total,
        "budget": 4_000_000,
        "budget_pass": total <= 4_000_000,
        "milliseconds_at_200mhz": total / TARGET_CLOCK_HZ * 1000.0,
        "milliseconds_at_100mhz": total / 100_000_000 * 1000.0,
    }
    return {
        "cycles": cycle_report,
        "layers": layer_rows,
        "bank_conflicts": conflicts,
        "bank_conflict_count": len(conflicts),
    }, accesses


def coverage_report(manifest: dict[str, Any],
                    tiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for layer_index, layer in enumerate(manifest["layers"]):
        _, hout, wout = map(int, layer["output"])
        cout = int(layer["output"][0])
        coverage = [[0] * cout for _ in range(hout)]
        current = [tile for tile in tiles if tile["layer_index"] == layer_index]
        for tile in current:
            h0, c0 = tile["origin"]["h"], tile["origin"]["cout"]
            for h in range(h0, h0 + tile["shape"]["h"]):
                for channel in range(c0, c0 + tile["shape"]["cout"]):
                    coverage[h][channel] += 1
        missing = sum(value == 0 for row in coverage for value in row) * wout
        overlap = sum(value > 1 for row in coverage for value in row) * wout
        rows.append({
            "name": layer["name"], "tiles": len(current),
            "output_elements": hout * wout * cout,
            "missing_elements": missing, "overlap_elements": overlap,
            "pass": missing == 0 and overlap == 0,
        })
    return rows


def compile_schedule(program_dir: Path, source_dir: Path | None,
                     output: Path) -> dict[str, Any]:
    program = json.loads((program_dir / "program.json").read_text(encoding="utf-8"))
    analysis = json.loads((program_dir / "analysis.json").read_text(encoding="utf-8"))
    if source_dir is None:
        source_dir = ROOT / str(program["source"]["package"])
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))

    descriptors, tiles = make_tiles(program, manifest, analysis)
    builder, layer_commands = emit_commands(program, manifest, tiles)
    simulation, accesses = simulate(program, manifest, tiles, len(builder.commands))
    coverage = coverage_report(manifest, tiles)

    max_usage = {bank: 0 for bank in BANK_CAPACITY}
    for tile in tiles:
        if tile["loads_input"]:
            bank = tile["banks"]["activation"]
            max_usage[bank] = max(max_usage[bank], tile["bytes"]["input"])
        for kind, bank_kind in (("weight", "weight"),
                                ("accumulator", "accumulator"),
                                ("output", "output")):
            bank = tile["banks"][bank_kind]
            size = tile["bytes"][kind]
            if kind == "weight":
                size = tile["bytes"]["weight_bank_footprint"]
            max_usage[bank] = max(max_usage[bank], size)
    capacity = {
        bank: {"capacity": BANK_CAPACITY[bank], "maximum_used": max_usage[bank],
               "pass": max_usage[bank] <= BANK_CAPACITY[bank]}
        for bank in BANK_CAPACITY
    }

    command_blob = b"".join(command.pack() for command in builder.commands)
    operator_blob = b"".join(pack_operator(row) for row in descriptors)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tile_commands.bin").write_bytes(command_blob)
    (output / "tile_operator_desc.bin").write_bytes(operator_blob)

    binary_files = {
        name: {"bytes": (output / name).stat().st_size,
               "sha256": sha256(output / name)}
        for name in ("tile_commands.bin", "tile_operator_desc.bin")
    }
    schedule = {
        "schema": 1,
        "status": "P3-proposal",
        "source_program": program_dir.name,
        "immediate_encoding": {
            "event_mask": "bits[7:0]", "activation_bank": "bit[8]",
            "weight_bank": "bit[9]", "accumulator_bank": "bit[10]",
            "output_bank": "bit[11]", "segment_index": "bits[13:12]",
            "segmented": "bit[14]",
        },
        "events": EVENT,
        "banks": capacity,
        "commands": [
            {"opcode": command.opcode.name, "flags": command.flags,
             "tag": command.tag, "dst_td": command.dst_td,
             "src0_td": command.src0_td, "src1_td": command.src1_td,
             "op_desc": command.op_desc, "quant_desc": command.quant_desc,
             "imm": command.imm}
            for command in builder.commands
        ],
        "layers": layer_commands,
        "operator_descriptors": descriptors,
        "tiles": tiles,
        "files": binary_files,
    }
    schedule_path = output / "tile_schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")

    scheduled_ddr_bytes = sum(
        ((tile["bytes"]["input"] if tile["loads_input"] else 0)
         + tile["bytes"]["weight"] + tile["bytes"]["bias"]
         + tile["bytes"]["quant"] + tile["bytes"]["vector_quant"]
         + tile["bytes"]["output"])
        for tile in tiles
    )
    # The simulator already models upsample DDR.  Recompute its byte count from
    # the graph explicitly to keep the report independent of cycle assumptions.
    previous_dst = int(program["entry"]["input_tensor"])
    upsample_ddr_bytes = 0
    for layer_index, layer in enumerate(manifest["layers"]):
        current = [tile for tile in tiles if tile["layer_index"] == layer_index]
        if str(layer["name"]).startswith("dec"):
            up_td = next(i for i, row in enumerate(program["tensors"])
                         if row["name"] == f"{layer['name']}.upsample")
            upsample_ddr_bytes += (int(program["tensors"][previous_dst]["allocation_bytes"])
                                   + int(program["tensors"][up_td]["allocation_bytes"]))
        previous_dst = current[-1]["dst_td"]
    scheduled_ddr_bytes += upsample_ddr_bytes

    tile_analysis = {
        "schema": 1,
        "status": "P3-proposal",
        "counts": {
            "commands": len(builder.commands),
            "operator_descriptors": len(descriptors),
            "tiles": len(tiles),
        },
        "coverage": coverage,
        "coverage_pass": all(row["pass"] for row in coverage),
        "bank_capacity": capacity,
        "bank_capacity_pass": all(row["pass"] for row in capacity.values()),
        "bank_conflict_count": simulation["bank_conflict_count"],
        "bank_conflicts": simulation["bank_conflicts"],
        "cycles": simulation["cycles"],
        "layer_cycles": simulation["layers"],
        "memory": {
            "scheduled_ddr_bytes": scheduled_ddr_bytes,
            "upsample_ddr_bytes": upsample_ddr_bytes,
            "activation_input_reuse": "one load per spatial tile",
        },
        "assumptions": [
            "DMA read and write channels may overlap; transfers within one direction serialize.",
            "Zero padding is synthesized by the address generator and consumes no DMA bytes.",
            "Weights, bias, and quant parameters share the selected W bank.",
            "Layer boundaries wait for all output stores; inter-layer overlap is not modeled.",
            "Upsample is conservatively modeled as vector cycles plus DDR read and write cycles.",
        ],
        "files": binary_files,
        "access_interval_count": len(accesses),
    }
    analysis_path = output / "tile_analysis.json"
    analysis_path.write_text(json.dumps(tile_analysis, indent=2) + "\n",
                             encoding="utf-8")
    return tile_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=PROGRAM_DEFAULT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.program
    report = compile_schedule(args.program.resolve(),
                              args.source.resolve() if args.source else None,
                              output.resolve())
    print(json.dumps({
        "counts": report["counts"],
        "cycles": report["cycles"],
        "coverage_pass": report["coverage_pass"],
        "bank_capacity_pass": report["bank_capacity_pass"],
        "bank_conflict_count": report["bank_conflict_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
