"""Compile and simulate NPU control/DMA RTL, including the full DMA plan."""
from __future__ import annotations

import json
import random
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


def signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & (1 << 31) else value


def pack_mac_token(activations: list[int], weights: list[list[int]],
                   input_mask: int, output_mask: int,
                   first: bool, last: bool) -> int:
    activation_bits = sum(
        (value & 0xFFFF) << (lane * 16)
        for lane, value in enumerate(activations))
    weight_bits = sum(
        (weights[out_lane][in_lane] & 0xFF)
        << ((out_lane * 8 + in_lane) * 8)
        for out_lane in range(8) for in_lane in range(8))
    return (activation_bits | (weight_bits << 128)
            | (input_mask << 640) | (output_mask << 648)
            | (int(first) << 656) | (int(last) << 657))


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
        str(RTL / "npu_u32_mul_iter.sv"),
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
        str(RTL / "npu_u32_mul_iter.sv"),
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

    descriptor_tables = {
        "tensor_table": (tensor_blob, 64),
        "operator_table": (operator_blob, 64),
        "quant_table": (quant_blob, 32),
        "segment_table": (segment_blob, 8),
    }
    descriptor_paths = {}
    descriptor_counts = {}
    for name, (blob, record_bytes) in descriptor_tables.items():
        assert len(blob) % record_bytes == 0
        rows = [
            f"{int.from_bytes(blob[offset:offset + record_bytes], 'little'):0{record_bytes * 2}x}"
            for offset in range(0, len(blob), record_bytes)
        ]
        path = temp / f"{name}.hex"
        path.write_text("\n".join(rows) + "\n", encoding="ascii")
        descriptor_paths[name] = path
        descriptor_counts[name] = len(rows)

    frontend_image = temp / "tb_npu_dma_frontend_stream.vvp"
    output = run([
        iverilog, "-g2012", "-Wall", "-s", "tb_npu_dma_frontend_stream",
        "-o", str(frontend_image), *dma_common,
        str(RTL / "npu_descriptor_cache.sv"),
        str(RTL / "npu_u32_mul_iter.sv"),
        str(RTL / "npu_dma_agu.sv"),
        str(RTL / "npu_dma_frontend.sv"),
        str(RTL / "tb" / "tb_npu_dma_frontend_stream.sv"),
    ])
    output += run([
        vvp, str(frontend_image),
        f"+COMMAND_HEX={vector_paths['command'].as_posix()}",
        f"+EXPECTED_HEX={vector_paths['expected'].as_posix()}",
        f"+TENSOR_HEX={descriptor_paths['tensor_table'].as_posix()}",
        f"+OPERATOR_HEX={descriptor_paths['operator_table'].as_posix()}",
        f"+QUANT_HEX={descriptor_paths['quant_table'].as_posix()}",
        f"+SEGMENT_HEX={descriptor_paths['segment_table'].as_posix()}",
        f"+REQUEST_COUNT={len(dma_plan['requests'])}",
        f"+TENSOR_COUNT={descriptor_counts['tensor_table']}",
        f"+OPERATOR_COUNT={descriptor_counts['operator_table']}",
        f"+QUANT_COUNT={descriptor_counts['quant_table']}",
        f"+SEGMENT_COUNT={descriptor_counts['segment_table']}",
    ])
    assert f"PASS ({len(dma_plan['requests'])} requests" in output

    dma_engine_image = temp / "tb_npu_dma_engine.vvp"
    output = run([
        iverilog, "-g2012", "-Wall", "-s", "tb_npu_dma_engine",
        "-o", str(dma_engine_image),
        str(RTL / "include" / "npu_dma_pkg.sv"),
        str(RTL / "npu_dma_engine.sv"),
        str(RTL / "tb" / "tb_npu_dma_engine.sv"),
    ])
    output += run([vvp, str(dma_engine_image)])
    assert "npu_dma_engine: PASS" in output

    scratchpad_image = temp / "tb_npu_scratchpad.vvp"
    output = run([
        iverilog, "-g2012", "-Wall", "-s", "tb_npu_scratchpad",
        "-o", str(scratchpad_image),
        str(RTL / "include" / "npu_dma_pkg.sv"),
        str(RTL / "npu_scratchpad_bank.sv"),
        str(RTL / "npu_scratchpad.sv"),
        str(RTL / "tb" / "tb_npu_scratchpad.sv"),
    ])
    output += run([vvp, str(scratchpad_image)])
    assert "npu_scratchpad: PASS" in output

    subsystem_image = temp / "tb_npu_dma_subsystem.vvp"
    output = run([
        iverilog, "-g2012", "-Wall", "-s", "tb_npu_dma_subsystem",
        "-o", str(subsystem_image), *dma_common,
        str(RTL / "npu_descriptor_cache.sv"),
        str(RTL / "npu_u32_mul_iter.sv"),
        str(RTL / "npu_dma_agu.sv"),
        str(RTL / "npu_dma_frontend.sv"),
        str(RTL / "npu_dma_engine.sv"),
        str(RTL / "npu_scratchpad_bank.sv"),
        str(RTL / "npu_scratchpad.sv"),
        str(RTL / "npu_dma_subsystem.sv"),
        str(RTL / "tb" / "tb_npu_dma_subsystem.sv"),
    ])
    output += run([vvp, str(subsystem_image)])
    assert "npu_dma_subsystem: PASS" in output

    rng = random.Random(0x8A8_2026)
    mac_tokens: list[int] = []
    mac_results: list[int] = []

    def add_mac_group(length: int, input_mask: int, output_mask: int,
                      *, full_int16: bool = False,
                      overflow_stress: bool = False) -> None:
        accumulators = [0] * 8
        for token_index in range(length):
            if overflow_stress:
                activations = [32767] * 8
                weights = [[127] * 8 for _ in range(8)]
            else:
                low, high = (-32768, 32767) if full_int16 else (-2048, 2047)
                activations = [rng.randint(low, high) for _ in range(8)]
                weights = [
                    [rng.randint(-128, 127) for _ in range(8)]
                    for _ in range(8)
                ]
            first = token_index == 0
            last = token_index == length - 1
            mac_tokens.append(pack_mac_token(
                activations, weights, input_mask, output_mask, first, last))
            for output_lane in range(8):
                if not (output_mask >> output_lane) & 1:
                    accumulators[output_lane] = 0
                    continue
                dot = sum(
                    activations[input_lane] * weights[output_lane][input_lane]
                    for input_lane in range(8)
                    if (input_mask >> input_lane) & 1)
                accumulators[output_lane] = signed32(
                    dot if first else accumulators[output_lane] + dot)
        mac_results.append(sum(
            (value & 0xFFFFFFFF) << (lane * 32)
            for lane, value in enumerate(accumulators)))

    add_mac_group(1, 0xFF, 0xFF)
    add_mac_group(3, 0x0F, 0x07)
    add_mac_group(80, 0xFF, 0xFF, overflow_stress=True)
    for group_index in range(40):
        input_lanes = rng.randint(1, 8)
        output_lanes = rng.randint(1, 8)
        add_mac_group(
            rng.randint(1, 13), (1 << input_lanes) - 1,
            (1 << output_lanes) - 1,
            full_int16=group_index % 7 == 0)

    mac_token_path = temp / "mac_tokens.hex"
    mac_expected_path = temp / "mac_expected.hex"
    mac_token_path.write_text(
        "".join(f"{token:0165x}\n" for token in mac_tokens),
        encoding="ascii")
    mac_expected_path.write_text(
        "".join(f"{result:064x}\n" for result in mac_results),
        encoding="ascii")
    mac_image = temp / "tb_npu_tensor_mac_8x8.vvp"
    output = run([
        iverilog, "-g2012", "-Wall", "-s", "tb_npu_tensor_mac_8x8",
        "-o", str(mac_image), str(RTL / "npu_tensor_mac_8x8.sv"),
        str(RTL / "tb" / "tb_npu_tensor_mac_8x8.sv"),
    ])
    output += run([
        vvp, str(mac_image),
        f"+TOKEN_HEX={mac_token_path.as_posix()}",
        f"+EXPECTED_HEX={mac_expected_path.as_posix()}",
        f"+TOKEN_COUNT={len(mac_tokens)}",
        f"+RESULT_COUNT={len(mac_results)}",
    ])
    assert f"PASS ({len(mac_tokens)} tokens, {len(mac_results)} results)" in output

    run([iverilog, "-g2012", "-Wall", "-tnull", "-f",
         str(RTL / "npu_rtl.f")])

print("NPU command, DMA, scratchpad, and Tensor MAC RTL tests: PASS")
