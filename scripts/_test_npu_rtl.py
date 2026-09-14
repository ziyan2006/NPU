"""Compile and simulate NPU control/DMA RTL, including the full DMA plan."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
NONE_INDEX = 0xFFFF


def descriptor(blob: bytes, index: int, size: int) -> bytes:
    if index == NONE_INDEX:
        return bytes(size)
    return blob[index * size:(index + 1) * size]


def pack_dma_request(row: dict) -> int:
    memory_space = {"activation": 0, "weight": 1, "bias": 2, "quant": 3}
    scratchpad = {"A": 0, "W": 1, "O": 2}
    fields = [
        (int(row["clear_bytes"]), 32),
        (int(row["scratchpad_z_stride"]), 32),
        (int(row["scratchpad_y_stride"]), 32),
        (int(row["external_z_stride"]), 32),
        (int(row["external_y_stride"]), 32),
        (int(row["z_count"]), 16),
        (int(row["y_count"]), 16),
        (int(row["x_bytes"]), 32),
        (int(row["scratchpad_offset"]), 32),
        (int(row["external_offset"]), 64),
        (int(bool(row["clear_before"])), 1),
        (int(row["scratchpad"][-1]), 1),
        (scratchpad[row["scratchpad"][0]], 2),
        (memory_space[row["memory_space"]], 2),
        (int(row["opcode"] == "DMA_STORE"), 1),
        (int(row["event_mask"]), 8),
        (int(row["tag"]), 16),
    ]
    result = shift = 0
    for value, width in fields:
        assert 0 <= value < 1 << width
        result |= value << shift
        shift += width
    assert shift == 351
    return result


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout + result.stderr


iverilog = shutil.which("iverilog")
vvp = shutil.which("vvp")
if not iverilog or not vvp:
    raise SystemExit("Icarus Verilog (iverilog + vvp) is required for RTL tests")

analysis = json.loads((PROGRAM / "tile_analysis.json").read_text(encoding="utf-8"))
command_count = int(analysis["counts"]["commands"])
payload = (PROGRAM / "tile_commands.bin").read_bytes()
assert len(payload) == command_count * 16

with tempfile.TemporaryDirectory() as temporary:
    temp = Path(temporary)
    command_hex = temp / "tile_commands.hex"
    command_hex.write_text("".join(
        f"{int.from_bytes(payload[offset:offset + 16], 'little'):032x}\n"
        for offset in range(0, len(payload), 16)
    ), encoding="ascii")

    common = [
        str(RTL / "include" / "npu_isa_pkg.sv"),
        str(RTL / "npu_command_processor.sv"),
    ]
    directed_image = temp / "tb_npu_command_processor.vvp"
    output = run([
        iverilog, "-g2012", "-s", "tb_npu_command_processor",
        "-o", str(directed_image), *common,
        str(RTL / "tb" / "tb_npu_command_processor.sv"),
    ])
    output += run([vvp, str(directed_image)])
    assert "npu_command_processor: PASS" in output

    stream_image = temp / "tb_npu_command_stream.vvp"
    output = run([
        iverilog, "-g2012", "-s", "tb_npu_command_stream",
        "-o", str(stream_image), *common,
        str(RTL / "tb" / "tb_npu_command_stream.sv"),
    ])
    output += run([
        vvp, str(stream_image),
        f"+COMMAND_HEX={command_hex.as_posix()}",
        f"+COMMAND_COUNT={command_count}",
    ])
    assert f"PASS ({command_count} commands" in output

    dma_common = [
        str(RTL / "include" / "npu_isa_pkg.sv"),
        str(RTL / "include" / "npu_dma_pkg.sv"),
    ]
    cache_image = temp / "tb_npu_descriptor_cache.vvp"
    output = run([
        iverilog, "-g2012", "-Wall", "-s", "tb_npu_descriptor_cache",
        "-o", str(cache_image), *dma_common,
        str(RTL / "npu_descriptor_cache.sv"),
        str(RTL / "tb" / "tb_npu_descriptor_cache.sv"),
    ])
    output += run([vvp, str(cache_image)])
    assert "npu_descriptor_cache: PASS" in output

    agu_image = temp / "tb_npu_dma_agu.vvp"
    output = run([
        iverilog, "-g2012", "-Wall", "-s", "tb_npu_dma_agu",
        "-o", str(agu_image), *dma_common,
        str(RTL / "npu_dma_agu.sv"),
        str(RTL / "tb" / "tb_npu_dma_agu.sv"),
    ])
    output += run([vvp, str(agu_image)])
    assert "npu_dma_agu: PASS" in output

    program = json.loads((PROGRAM / "program.json").read_text(encoding="utf-8"))
    schedule = json.loads(
        (PROGRAM / "tile_schedule.json").read_text(encoding="utf-8"))
    dma_plan = json.loads(
        (PROGRAM / "dma_plan.json").read_text(encoding="utf-8"))
    tensor_blob = (PROGRAM / "tensor_desc.bin").read_bytes()
    operator_blob = (PROGRAM / "tile_operator_desc.bin").read_bytes()
    quant_blob = (PROGRAM / "quant_desc.bin").read_bytes()
    segment_blob = (PROGRAM / "segments.bin").read_bytes()
    command_rows = schedule["commands"]

    vector_rows: dict[str, list[str]] = {
        name: [] for name in (
            "command", "source", "destination", "operator", "quant",
            "segment", "expected",
        )
    }
    for request in dma_plan["requests"]:
        command_index = int(request["command_index"])
        command = command_rows[command_index]
        raw_command = payload[command_index * 16:(command_index + 1) * 16]
        source = descriptor(tensor_blob, int(command["src0_td"]), 64)
        destination = descriptor(tensor_blob, int(command["dst_td"]), 64)
        operator = descriptor(operator_blob, int(command["op_desc"]), 64)
        quant = descriptor(quant_blob, int(command["quant_desc"]), 32)
        segment_dependencies = [
            item for item in request["descriptor_dependencies"]
            if item.startswith("segment:")
        ]
        segment_index = (int(segment_dependencies[0].split(":")[1])
                         if segment_dependencies else NONE_INDEX)
        segment = descriptor(segment_blob, segment_index, 8)
        vector_rows["command"].append(
            f"{int.from_bytes(raw_command, 'little'):032x}")
        vector_rows["source"].append(
            f"{int.from_bytes(source, 'little'):0128x}")
        vector_rows["destination"].append(
            f"{int.from_bytes(destination, 'little'):0128x}")
        vector_rows["operator"].append(
            f"{int.from_bytes(operator, 'little'):0128x}")
        vector_rows["quant"].append(
            f"{int.from_bytes(quant, 'little'):064x}")
        vector_rows["segment"].append(
            f"{int.from_bytes(segment, 'little'):016x}")
        vector_rows["expected"].append(f"{pack_dma_request(request):088x}")

    vector_paths = {}
    for name, rows in vector_rows.items():
        path = temp / f"dma_{name}.hex"
        path.write_text("\n".join(rows) + "\n", encoding="ascii")
        vector_paths[name] = path

    dma_stream_image = temp / "tb_npu_dma_stream.vvp"
    output = run([
        iverilog, "-g2012", "-Wall", "-s", "tb_npu_dma_stream",
        "-o", str(dma_stream_image), *dma_common,
        str(RTL / "npu_dma_agu.sv"),
        str(RTL / "tb" / "tb_npu_dma_stream.sv"),
    ])
    output += run([
        vvp, str(dma_stream_image),
        f"+COMMAND_HEX={vector_paths['command'].as_posix()}",
        f"+SOURCE_HEX={vector_paths['source'].as_posix()}",
        f"+DESTINATION_HEX={vector_paths['destination'].as_posix()}",
        f"+OPERATOR_HEX={vector_paths['operator'].as_posix()}",
        f"+QUANT_HEX={vector_paths['quant'].as_posix()}",
        f"+SEGMENT_HEX={vector_paths['segment'].as_posix()}",
        f"+EXPECTED_HEX={vector_paths['expected'].as_posix()}",
        f"+REQUEST_COUNT={len(dma_plan['requests'])}",
    ])
    assert f"PASS ({len(dma_plan['requests'])} requests)" in output

    run([iverilog, "-g2012", "-Wall", "-tnull", "-f",
         str(RTL / "npu_rtl.f")])

print("NPU command processor, descriptor cache, and DMA AGU RTL tests: PASS")
