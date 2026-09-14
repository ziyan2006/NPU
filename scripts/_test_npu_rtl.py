"""Compile and simulate the first synthesizable NPU command-processor RTL."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"


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

print("NPU command-processor RTL tests: PASS")
