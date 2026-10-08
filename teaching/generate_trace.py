"""Build a tiny task and record real rising-edge RTL traces for the teaching page.

Run from any directory: python teaching/generate_trace.py --iverilog iverilog --vvp vvp
No source RTL is modified. Temporary enum-ternary normalization works around
Icarus's enum assignment limitation; each replacement is an equivalent if/else.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from npu_isa import (Command, Opcode, CommandFlag, Event, NONE_INDEX,
                     TENSOR_STRUCT, OPERATOR_STRUCT, QUANT_DESC_STRUCT,
                     QUANT_PARAM_STRUCT)

C = "bench.dut.core."
D = C + "data_subsystem."
P = D + "conv2d_pipeline."
M = P + "controller.mac."

# Values are sampled BEFORE the rising edge: VALID & READY means transfer AT
# this edge. State registers describe the phase entering this edge.
SIGNALS = {
    "cp_state": C + "command_processor.state_q",
    "pc": C + "cp_pc", "events": C + "cp_events", "event_set": C + "event_set",
    "retired": C + "commands_retired_o", "irq": "bench.irq_o",
    "dma_engine_pc": D + "engine_pc_q", "dma_engine_tag": D + "engine.error_tag_o",
    "dma_frontend_tag": D + "frontend.command_q[31:16]",
    "dispatch_pc": C + "command_processor.dispatch_pc_q",
    "core_done": "bench.dut.core_done", "capture": C + "command_processor.command_capture",
    "fetch_v": C + "fetch_command_valid", "fetch_r": C + "fetch_command_ready",
    "fetch_pc": C + "fetch_command_pc", "command": C + "fetch_command_bits",
    "dma_v": C + "cp_dma_valid", "dma_r": C + "cp_dma_ready",
    "conv_v": C + "cp_compute_valid", "conv_r": C + "cp_compute_ready",
    "vec_v": C + "cp_vector_valid", "vec_r": C + "cp_vector_ready",
    "dma_busy": C + "dma_busy", "conv_busy": C + "conv_busy",
    "dma_frontend_busy": D + "frontend_busy", "dma_engine_busy": D + "engine_busy",
    "vec_busy": C + "vec_busy", "up_busy": C + "up_busy",
    "dma_event": C + "dma_event_set", "conv_event": C + "conv_event_set",
    "vec_done": C + "vec_done", "up_done": C + "up_done",
    "ef_state": C + "execution_frontend.state_q",
    "df_state": D + "frontend.state_q", "agu_state": D + "frontend.address_generator.state_q",
    "de_state": D + "engine.state_q", "param_state": P + "param_state_q",
    "ctrl_state": P + "controller.setup_state_q", "post_state": P + "post.state_q",
    "post_lane": P + "post.lane_q", "v_state": C + "vec_add.state_q",
    "v_post": C + "vec_add.residual_requant.state_q", "u_state": C + "upsample.state_q",
    "v_post_lane": C + "vec_add.residual_requant.lane_q",
    "agu_mul_a": D + "frontend.address_generator.mul_a", "agu_mul_b": D + "frontend.address_generator.mul_b",
    "agu_mul_busy": D + "frontend.address_generator.mul_busy", "agu_mul_done": D + "frontend.address_generator.mul_done",
    "agu_mul_step": D + "frontend.address_generator.shared_address_multiplier.step_q",
    "agu_mul_result": D + "frontend.address_generator.mul_result",
    "ctrl_mul_a": P + "controller.mul_a", "ctrl_mul_b": P + "controller.mul_b",
    "ctrl_mul_busy": P + "controller.mul_busy", "ctrl_mul_done": P + "controller.mul_done",
    "ctrl_mul_step": P + "controller.setup_multiplier.step_q",
    "ctrl_mul_result": P + "controller.mul_result",
    "v_mul_a": C + "vec_add.mul_a", "v_mul_b": C + "vec_add.mul_b",
    "v_mul_busy": C + "vec_add.mul_busy", "v_mul_done": C + "vec_add.mul_done",
    "v_mul_step": C + "vec_add.setup_multiplier.step_q",
    "v_mul_result": C + "vec_add.mul_result",
    "u_mul_a": C + "upsample.row_multiplier.operand_a_i", "u_mul_b": C + "upsample.row_multiplier.operand_b_i",
    "u_mul_busy": C + "upsample.mul_busy", "u_mul_done": C + "upsample.mul_done",
    "u_mul_step": C + "upsample.row_multiplier.step_q",
    "u_mul_result": C + "upsample.mul_result",
    "u_write_beat": C + "upsample.write_beat_q", "u_row_copy": C + "upsample.output_row_copy_q",
    "ar_v": "bench.m_axi_mem_arvalid", "ar_r": "bench.m_axi_mem_arready",
    "ar_addr": "bench.m_axi_mem_araddr", "ar_len": "bench.m_axi_mem_arlen",
    "r_v": "bench.m_axi_mem_rvalid", "r_r": "bench.m_axi_mem_rready",
    "r_data": "bench.m_axi_mem_rdata", "r_last": "bench.m_axi_mem_rlast",
    "aw_v": "bench.m_axi_mem_awvalid", "aw_r": "bench.m_axi_mem_awready",
    "aw_addr": "bench.m_axi_mem_awaddr", "aw_len": "bench.m_axi_mem_awlen",
    "w_v": "bench.m_axi_mem_wvalid", "w_r": "bench.m_axi_mem_wready",
    "w_data": "bench.m_axi_mem_wdata", "w_last": "bench.m_axi_mem_wlast",
    "b_v": "bench.m_axi_mem_bvalid", "b_r": "bench.m_axi_mem_bready",
    "dma_ar": D + "m_axi_arvalid_o", "up_ar": C + "upsample.m_axi_arvalid_o",
    "dma_rdata_v": D + "m_axi_rvalid_i", "dma_rdata_r": D + "m_axi_rready_o",
    "up_rdata_v": C + "upsample.m_axi_rvalid_i", "up_rdata_r": C + "upsample.m_axi_rready_o",
    "dma_aw": D + "m_axi_awvalid_o", "up_aw": C + "upsample.m_axi_awvalid_o",
    "dma_wv": D + "m_axi_wvalid_o", "dma_wr": D + "m_axi_wready_i",
    "up_wv": C + "upsample.m_axi_wvalid_o", "up_wr": C + "upsample.m_axi_wready_i",
    "block_ar": C + "block_reader.m_axi_arvalid_o",
    "block_req_v": C + "block_request_valid", "block_req_r": C + "block_request_ready",
    "block_res_v": C + "block_response_valid", "block_res_r": C + "block_response_ready",
    "fetch_req_v": C + "memory_request_valid[1]", "fetch_req_r": C + "memory_request_ready[1]",
    "fetch_res_v": C + "memory_response_valid[1]", "fetch_res_r": C + "memory_response_ready[1]",
    "df_req_v": D + "descriptor_memory_request_valid_o", "df_req_r": D + "descriptor_memory_request_ready_i",
    "df_res_v": D + "descriptor_memory_response_valid_i", "df_res_r": D + "descriptor_memory_response_ready_o",
    "ef_req_v": C + "execution_frontend.memory_request_valid_o", "ef_req_r": C + "execution_frontend.memory_request_ready_i",
    "ef_res_v": C + "execution_frontend.memory_response_valid_i", "ef_res_r": C + "execution_frontend.memory_response_ready_o",
    "conv_start_v": C + "exec_conv_start_valid", "conv_start_r": C + "exec_conv_start_ready",
    "vec_start_v": C + "exec_vec_start_valid", "vec_start_r": C + "exec_vec_start_ready",
    "up_start_v": C + "exec_up_start_valid", "up_start_r": C + "exec_up_start_ready",
    "sp_wv": D + "spad_write_valid", "sp_wr": D + "spad_write_ready",
    "sp_wkind": D + "spad_write_kind", "sp_waddr": D + "spad_write_address",
    "sp_wdata": D + "spad_write_data", "sp_wstrb": D + "spad_write_strobe",
    "sp_rv": D + "spad_read_request_valid", "sp_rr": D + "spad_read_request_ready",
    "sp_res_v": D + "spad_read_response_valid", "sp_res_r": D + "spad_read_response_ready",
    "a_req_v": D + "conv_activation_request_valid", "a_req_r": D + "conv_activation_request_ready",
    "a_res_v": D + "conv_activation_response_valid", "a_res_r": D + "conv_activation_response_ready",
    "a_data": D + "conv_activation_read_data", "a_addr": D + "conv_activation_read_address",
    "w_req_v": D + "conv_weight_request_valid", "w_req_r": D + "conv_weight_request_ready",
    "w_res_v": D + "conv_weight_response_valid", "w_res_r": D + "conv_weight_response_ready",
    "weight_data": D + "conv_weight_read_data", "weight_addr": D + "conv_weight_read_address",
    "mac_v": M + "input_valid_i", "mac_r": M + "input_ready_o",
    "mac_product": M + "product_valid_q", "mac_l1": M + "sum_l1_valid_q",
    "mac_l2": M + "sum_l2_valid_q", "mac_dot": M + "dot_valid_q",
    "mac_out_v": M + "result_valid_o", "mac_out_r": M + "result_ready_i",
    "mac_data": M + "result_o",
    "post_v": P + "result_buffer_valid_q", "post_r": P + "post_input_ready",
    "out_v": D + "conv_result_valid", "out_r": D + "conv_result_ready",
    "out_data": D + "conv_result_data", "fifo_count": D + "conv_output_count_q",
    "o_v": D + "conv_output_write_valid", "o_r": D + "conv_output_write_ready",
    "o_data": D + "conv_output_write_data", "o_addr": D + "conv_output_write_address",
    "v_read_v": C + "vec_read_request_valid", "v_read_r": C + "vec_read_request_ready",
    "v_kind": C + "vec_read_kind", "v_res_v": C + "vec_read_response_valid",
    "v_res_r": C + "vec_read_response_ready", "v_data": C + "vec_read_data",
    "v_write_v": C + "vec_write_valid", "v_write_r": C + "vec_write_ready",
    "v_write_data": C + "vec_write_data", "bank_stall": C + "bank_collision_stall",
}


def make_fixture(directory: Path):
    def tensor(base, size, shape, strides, dtype=2, layout=1, flags=2):
        return TENSOR_STRUCT.pack(base, size, *shape, *strides, dtype, layout, NONE_INDEX, 0, flags, 0)
    tensors = [tensor(0, 16, (1, 1, 1, 8), (16, 16, 16, 2)),
               tensor(64, 16, (1, 1, 1, 8), (16, 16, 16, 2)),
               tensor(128, 64, (1, 2, 2, 8), (64, 32, 16, 2)),
               tensor(0, 64, (8, 1, 1, 8), (8, 8, 8, 1), 1, 2, 3),
               tensor(0, 32, (1, 1, 1, 8), (32, 32, 32, 4), 4, 0, 3)]
    op = OPERATOR_STRUCT.pack(0, 1, 3, 4, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                              0, 0, 0, 1, 1, 8, 0, 8)
    commands = [Command(Opcode.NOP),
        Command(Opcode.DMA_LOAD, 1, dst_td=0, src0_td=0, op_desc=0, imm=1),
        Command(Opcode.DMA_LOAD, 1, dst_td=3, src0_td=3, op_desc=0),
        Command(Opcode.DMA_LOAD, 1, dst_td=4, src0_td=4, op_desc=0),
        Command(Opcode.DMA_LOAD, 1, op_desc=0, quant_desc=0),
        Command(Opcode.DMA_LOAD, 1, op_desc=0, quant_desc=1, imm=0x404),
        Command(Opcode.WAIT, imm=5),
        Command(Opcode.CONV2D, 13, dst_td=1, src0_td=0, src1_td=3, op_desc=0, quant_desc=0, imm=16),
        Command(Opcode.WAIT, imm=16),
        Command(Opcode.VEC_ADD, 4, dst_td=1, src0_td=1, src1_td=0, op_desc=0, quant_desc=1),
        Command(Opcode.DMA_STORE, 1, dst_td=1, src0_td=1, op_desc=0, imm=64),
        Command(Opcode.WAIT, imm=64),
        Command(Opcode.UPSAMPLE2X, dst_td=2, src0_td=1),
        Command(Opcode.END, 2)]
    for i, cmd in enumerate(commands):
        object.__setattr__(cmd, "tag", i)
    quant = QUANT_PARAM_STRUCT.pack(1 << 30, 30, -2048, 2047)
    files = {"tile_commands.bin": b"".join(c.pack() for c in commands),
             "tensor_desc.bin": b"".join(tensors), "tile_operator_desc.bin": op,
             "quant_desc.bin": QUANT_DESC_STRUCT.pack(0, 8, -2048, 2047, 0, 0)
                               + QUANT_DESC_STRUCT.pack(128, 1, -2048, 2047, 0, 0),
             "segments.bin": bytes(8), "weights_o8i8.bin": bytes([2]) + bytes(63),
             "bias_int32.bin": bytes(32), "quant_params.bin": quant * 9,
             "tanh_lut_int12.bin": bytes(8192)}
    for name, data in files.items():
        (directory / name).write_bytes(data)
    program = {"isa": {"major": 1, "minor": 0}, "entry": {"input_tensor": 0, "output_tensor": 2},
               "source": {"manifest_sha256": hashlib.sha256(b"NPU teaching fixture v1").hexdigest()}}
    (directory / "program.json").write_text(json.dumps(program))
    (directory / "analysis.json").write_text(json.dumps({"memory": {"mutable_arena_bytes_no_reuse": 256}}))
    spec = importlib.util.spec_from_file_location("image_builder", ROOT / "scripts/33_build_npu_task_image.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    raw, manifest = builder.build_task_image(directory)
    image = bytearray(raw)
    activation = manifest["sections"]["activation"]["offset"]
    struct.pack_into("<h", image, activation, 3)
    (directory / "image.hex").write_text("".join(f"{int.from_bytes(image[i:i+8], 'little'):016x}\n"
                                               for i in range(0, len(image), 8)))
    return image, manifest, commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--ivl-base", help="Icarus -B path for an unpacked distribution")
    parser.add_argument("--keep", type=Path)
    args = parser.parse_args()
    directory = args.keep or Path(tempfile.mkdtemp(prefix="npu-teaching-"))
    directory.mkdir(parents=True, exist_ok=True)
    image, manifest, commands = make_fixture(directory)
    sources = []
    enums = {}
    normalized = 0
    for source in (ROOT / "hardware/rtl/npu_rtl.f").read_text().splitlines():
        if not source or source.startswith("#"):
            continue
        original = (ROOT / source).read_text()
        for body, name in re.findall(r"typedef enum logic\s*\[[^]]+\]\s*\{(.*?)\}\s*(\w+)", original, re.S):
            enums[name] = [part.strip().split("=")[0].strip() for part in body.split(",")]
        # Equivalent transformation, limited to enum-valued ternaries.
        regex = r"((?:state_q|kind_q)\s*)<=\s*([^;?]+)\?\s*([A-Z][A-Z0-9_]+)\s*:\s*([A-Z][A-Z0-9_]+)\s*;"
        def replace(match):
            return f"if ({match[2].strip()}) {match[1]}<= {match[3]}; else {match[1]}<= {match[4]};"
        transformed, count = re.subn(regex, replace, original)
        normalized += count
        target = directory / Path(source).name
        target.write_text(transformed)
        sources.append(str(target))
    monitor = directory / "tb_teaching.sv"
    fmt = ",".join(["%0d"] + ["%h"] * len(SIGNALS)) + "\\n"
    monitor.write_text('`timescale 1ns/1ps\nmodule tb_teaching;\n'
        'tb_npu_top_task bench();\ninteger fd; integer cycle=0;\n'
        'initial fd=$fopen("trace.csv", "w");\n'
        'always @(posedge bench.aclk) if (bench.aresetn) begin\n'
        'cycle=cycle+1;\n'
        f'$fwrite(fd,"{fmt}",cycle,' + ",".join(SIGNALS.values()) + ');\nend\nendmodule\n')
    compile_cmd = [args.iverilog]
    if args.ivl_base:
        compile_cmd += ["-B", args.ivl_base]
    compile_cmd += ["-g2012", "-s", "tb_teaching", "-o", str(directory / "sim.vvp"), *sources,
                    str(ROOT / "hardware/rtl/tb/tb_npu_top_task.sv"), str(monitor)]
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
    (directory / "compile.log").write_text(compile_result.stdout + compile_result.stderr)
    if compile_result.returncode:
        raise RuntimeError(f"Compilation failed; inspect {directory}/compile.log")
    output_offset = manifest["sections"]["activation"]["offset"] + 128
    run = subprocess.run([args.vvp, str(directory / "sim.vvp"),
        "+IMAGE=image.hex", f"+IMAGE_BYTES={len(image)}", "+OUTPUT=output.hex",
        f"+OUTPUT_OFFSET={output_offset}", "+OUTPUT_BYTES=64", "+EXPECTED_COMMANDS=14",
        "+MAX_SIM_NS=500000"], cwd=directory, capture_output=True, text=True, timeout=120)
    (directory / "run.log").write_text(run.stdout + run.stderr)
    if run.returncode or "npu_top task: PASS" not in run.stdout:
        raise RuntimeError(f"Simulation failed; inspect {directory}/run.log\n{run.stdout}")
    words = [int(line, 16) for line in (directory / "output.hex").read_text().splitlines()
             if line and not line.startswith(("//", "@"))]
    output = b"".join(w.to_bytes(8, "little") for w in words)
    expected = (struct.pack("<h", 9) + bytes(14)) * 4
    assert output == expected, f"Output mismatch: {output.hex()}"
    raw_rows = (directory / "trace.csv").read_text().splitlines()
    rows = []
    for line in raw_rows:
        fields = line.split(",")
        assert len(fields) == len(SIGNALS) + 1
        row = [int(fields[0])]
        for field in fields[1:]:
            row.append(None if re.search("[xz]", field, re.I) else field.lstrip("0") or "0")
        rows.append(row)
    first = next(i for i, row in enumerate(rows) if row[list(SIGNALS).index("capture") + 1] == "1")
    data = {"schema": 1, "clockNs": 10, "sampling": "pre-rising-edge",
            "provenance": "Icarus Verilog 13 · npu_top · tb_npu_top_task AXI memory model",
            "rtlSha256": hashlib.sha256(b"".join((ROOT / s).read_bytes() for s in
                (ROOT / "hardware/rtl/npu_rtl.f").read_text().splitlines() if s and not s.startswith("#"))).hexdigest(),
            "normalizedEnumTernaries": normalized, "expectedOutput": [9, 0, 0, 0, 0, 0, 0, 0],
            "commands": [{"pc": i*16, "index": i, "opcode": int(c.opcode), "name": c.opcode.name,
                "hex": c.pack().hex(), "flags": c.flags, "imm": c.imm,
                "dst": c.dst_td, "src0": c.src0_td, "src1": c.src1_td,
                "op": c.op_desc, "quant": c.quant_desc} for i, c in enumerate(commands)],
            "enums": enums, "signals": list(SIGNALS), "signalPaths": SIGNALS,
            "rows": rows[first:]}
    (HERE / "trace.js").write_text("window.NPU_TRACE=" + json.dumps(data, separators=(",", ":")) + ";\n")
    print(run.stdout.strip())
    print(f"Verified output: four pixels [9,0,0,0,0,0,0,0]; recorded {len(rows[first:])} cycles.")
    print(f"Temporary simulation files: {directory}")
    if not args.keep:
        shutil.rmtree(directory)


if __name__ == "__main__":
    main()
