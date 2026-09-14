"""Audit every normalized DMA request against the v1 AXI burst policy."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PROGRAM_DEFAULT = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"


def split_row(address: int, byte_count: int) -> list[dict[str, int]]:
    bursts = []
    remaining = byte_count
    while remaining:
        if remaining >= 8:
            beat_bytes = 8
            beats_by_row = remaining // 8
        elif remaining >= 4:
            beat_bytes, beats_by_row = 4, 1
        elif remaining >= 2:
            beat_bytes, beats_by_row = 2, 1
        else:
            beat_bytes, beats_by_row = 1, 1
        if address % beat_bytes:
            raise ValueError(f"address {address:#x} is not {beat_bytes}-byte aligned")
        beats_by_boundary = (4096 - address % 4096) // beat_bytes
        beats = min(beats_by_row, beats_by_boundary, 256)
        if beats <= 0:
            raise ValueError(f"cannot make progress at address {address:#x}")
        transfer_bytes = beats * beat_bytes
        bursts.append({
            "address": address,
            "beat_bytes": beat_bytes,
            "beats": beats,
            "bytes": transfer_bytes,
        })
        address += transfer_bytes
        remaining -= transfer_bytes
    return bursts


def analyze(program_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    plan = json.loads((program_dir / "dma_plan.json").read_text(encoding="utf-8"))
    request_rows = []
    burst_count_by_direction: Counter[str] = Counter()
    beat_count_by_direction: Counter[str] = Counter()
    narrow_bursts = 0
    boundary_splits = 0
    max_bursts_per_request = 0
    total_rows = 0
    total_bytes = 0

    for request in plan["requests"]:
        if int(request["external_offset"]) & 7:
            raise ValueError(
                f"request {request['command_index']} has unaligned row base")
        if int(request["y_count"]) > 1 and int(request["external_y_stride"]) & 7:
            raise ValueError(
                f"request {request['command_index']} has unaligned y stride")
        if int(request["z_count"]) > 1 and int(request["external_z_stride"]) & 7:
            raise ValueError(
                f"request {request['command_index']} has unaligned z stride")

        direction = "write" if request["opcode"] == "DMA_STORE" else "read"
        request_bursts = 0
        for z_index in range(int(request["z_count"])):
            for y_index in range(int(request["y_count"])):
                address = (int(request["external_offset"])
                           + z_index * int(request["external_z_stride"])
                           + y_index * int(request["external_y_stride"]))
                bursts = split_row(address, int(request["x_bytes"]))
                total_rows += 1
                request_bursts += len(bursts)
                burst_count_by_direction[direction] += len(bursts)
                beat_count_by_direction[direction] += sum(
                    burst["beats"] for burst in bursts)
                narrow_bursts += sum(
                    burst["beat_bytes"] < 8 for burst in bursts)
                boundary_splits += sum(
                    index + 1 < len(bursts)
                    and burst["address"] + burst["bytes"] & 4095 == 0
                    for index, burst in enumerate(bursts))
                total_bytes += sum(burst["bytes"] for burst in bursts)
        max_bursts_per_request = max(max_bursts_per_request, request_bursts)
        request_rows.append(request_bursts)

    if total_bytes != int(plan["analysis"]["payload_bytes"]):
        raise ValueError("AXI burst bytes do not match normalized DMA payload")
    result = {
        "schema": 1,
        "status": "P4-proposal",
        "request_count": len(plan["requests"]),
        "row_count": total_rows,
        "payload_bytes": total_bytes,
        "burst_count": sum(burst_count_by_direction.values()),
        "burst_count_by_direction": dict(sorted(burst_count_by_direction.items())),
        "beat_count": sum(beat_count_by_direction.values()),
        "beat_count_by_direction": dict(sorted(beat_count_by_direction.items())),
        "assumed_burst_overhead_cycles": 16,
        "estimated_axi_cycles": {
            direction: beat_count_by_direction[direction]
            + burst_count_by_direction[direction] * 16
            for direction in ("read", "write")
        },
        "narrow_burst_count": narrow_bursts,
        "four_kib_split_count": boundary_splits,
        "maximum_bursts_per_request": max_bursts_per_request,
        "maximum_axi_beats_per_burst": 256,
        "alignment_pass": True,
        "payload_exact_pass": True,
    }
    result["estimated_axi_cycles"]["total"] = sum(
        result["estimated_axi_cycles"].values())
    output_dir = output_dir or program_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "axi_dma_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=PROGRAM_DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(
        args.program.resolve(), args.output.resolve() if args.output else None),
        indent=2))


if __name__ == "__main__":
    main()
