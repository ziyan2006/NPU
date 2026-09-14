"""Check generated C/SystemVerilog ISA headers against the Python source."""
from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from npu_isa import (  # noqa: E402
    COMMAND_STRUCT,
    OPERATOR_STRUCT,
    QUANT_DESC_STRUCT,
    QUANT_PARAM_STRUCT,
    SEGMENT_STRUCT,
    TENSOR_STRUCT,
    Command,
    CommandFlag,
    DType,
    ErrorCode,
    Event,
    Layout,
    Opcode,
    PostOp,
    encode_control_imm,
)


SV_HEADER = ROOT / "hardware" / "rtl" / "include" / "npu_isa_pkg.sv"
C_HEADER = ROOT / "software" / "include" / "npu_isa.h"


class CCommand(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("opcode", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("tag", ctypes.c_uint16),
        ("dst_td", ctypes.c_uint16),
        ("src0_td", ctypes.c_uint16),
        ("src1_td", ctypes.c_uint16),
        ("op_desc", ctypes.c_uint16),
        ("quant_desc", ctypes.c_uint16),
        ("imm", ctypes.c_uint16),
    ]


def parse_c_defines(text: str, prefix: str) -> dict[str, int]:
    pattern = re.compile(rf"^#define {prefix}_([A-Z0-9_]+) 0x([0-9a-f]+)u$", re.M)
    return {name: int(value, 16) for name, value in pattern.findall(text)}


def parse_sv_values(text: str, prefix: str) -> dict[str, int]:
    pattern = re.compile(
        rf"^\s*{prefix}_([A-Z0-9_]+) = (?:8|16)'h([0-9a-f]+),?$", re.M)
    return {name: int(value, 16) for name, value in pattern.findall(text)}


subprocess.run([sys.executable, str(SCRIPTS / "29_generate_npu_isa_headers.py"),
                "--check"], cwd=ROOT, check=True)
sv_text = SV_HEADER.read_text(encoding="utf-8")
c_text = C_HEADER.read_text(encoding="utf-8")

for prefix, enum_type in (
    ("NPU_OP", Opcode),
    ("NPU_DTYPE", DType),
    ("NPU_LAYOUT", Layout),
    ("NPU_POST", PostOp),
    ("NPU_ERR", ErrorCode),
):
    expected = {item.name: int(item) for item in enum_type}
    assert parse_c_defines(c_text, prefix) == expected
    assert parse_sv_values(sv_text, prefix) == expected

expected_flags = {item.name: int(item) for item in CommandFlag if int(item)}
expected_events = {item.name: int(item) for item in Event}
assert parse_c_defines(c_text, "NPU_FLAG") == expected_flags
assert parse_c_defines(c_text, "NPU_EVENT") == expected_events

assert ctypes.sizeof(CCommand) == COMMAND_STRUCT.size == 16
assert CCommand.opcode.offset == 0
assert CCommand.flags.offset == 1
assert CCommand.tag.offset == 2
assert CCommand.dst_td.offset == 4
assert CCommand.imm.offset == 14
assert TENSOR_STRUCT.size == 64
assert OPERATOR_STRUCT.size == 64
assert QUANT_DESC_STRUCT.size == 32
assert QUANT_PARAM_STRUCT.size == 16
assert SEGMENT_STRUCT.size == 8

imm = encode_control_imm(event=int(Event.C1_DONE), activation=1, weight=0,
                         accumulator=1, output=1, segmented=True, segment=2)
python_command = Command(
    opcode=Opcode.CONV2D,
    flags=int(CommandFlag.ASYNC | CommandFlag.SATURATE
              | CommandFlag.FUSED_POST_OP),
    tag=0x1234,
    dst_td=1,
    src0_td=2,
    src1_td=3,
    op_desc=4,
    quant_desc=5,
    imm=imm,
)
c_command = CCommand(
    int(python_command.opcode), python_command.flags, python_command.tag,
    python_command.dst_td, python_command.src0_td, python_command.src1_td,
    python_command.op_desc, python_command.quant_desc, python_command.imm,
)
assert bytes(c_command) == python_command.pack()
assert "logic [7:0]  opcode;" in sv_text
assert "logic [15:0] imm;" in sv_text
assert "logic [31:0] input_channel_count;" in sv_text

compiler = shutil.which("gcc") or shutil.which("clang")
if compiler:
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "header_check.c"
        source.write_text('#include "npu_isa.h"\nint main(void) { return 0; }\n',
                          encoding="ascii")
        subprocess.run([compiler, "-std=c11", "-fsyntax-only",
                        "-I", str(C_HEADER.parent), str(source)], check=True)

iverilog = shutil.which("iverilog")
vvp = shutil.which("vvp")
if iverilog and vvp:
    with tempfile.TemporaryDirectory() as temporary:
        image = Path(temporary) / "tb_npu_isa_pkg.vvp"
        subprocess.run([
            iverilog, "-g2012", "-s", "tb_npu_isa_pkg", "-o", str(image),
            str(SV_HEADER),
            str(ROOT / "hardware" / "rtl" / "tb" / "tb_npu_isa_pkg.sv"),
        ], check=True)
        result = subprocess.run([vvp, str(image)], check=True,
                                capture_output=True, text=True)
        assert "npu_isa_pkg RTL layout: PASS" in result.stdout

print("NPU generated ISA header tests: PASS")
