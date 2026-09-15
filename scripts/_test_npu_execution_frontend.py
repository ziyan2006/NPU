"""Dispatch all real compute/vector commands through the descriptor frontend."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from npu_isa import COMMAND_STRUCT, Command, Opcode


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"


def write_words(path: Path, values: list[int], digits: int) -> None:
    path.write_text("".join(f"{value:0{digits}x}\n" for value in values),
                    encoding="ascii")


def chunks(payload: bytes, size: int) -> list[int]:
    assert len(payload) % size == 0
    return [int.from_bytes(payload[offset:offset + size], "little")
            for offset in range(0, len(payload), size)]


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def main() -> int:
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")
    blob = (PROGRAM / "tile_commands.bin").read_bytes()
    selected: list[tuple[int, bytes]] = []
    supported = {Opcode.CONV2D, Opcode.VEC_ADD, Opcode.UPSAMPLE2X}
    for pc in range(0, len(blob), COMMAND_STRUCT.size):
        raw = blob[pc:pc + COMMAND_STRUCT.size]
        if Command.unpack(raw).opcode in supported:
            selected.append((pc, raw))
    assert len(selected) == 283

    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        sim = temp / "tb_npu_execution_frontend.vvp"
        run([iverilog, "-g2012", "-Wall", "-s", "tb_npu_execution_frontend",
             "-o", str(sim), str(RTL / "include" / "npu_isa_pkg.sv"),
             str(RTL / "include" / "npu_dma_pkg.sv"),
             str(RTL / "npu_descriptor_cache.sv"),
             str(RTL / "npu_execution_frontend.sv"),
             str(RTL / "tb" / "tb_npu_execution_frontend.sv")])
        commands_hex = temp / "commands.hex"
        pcs_hex = temp / "pcs.hex"
        operators_hex = temp / "operators.hex"
        tensors_hex = temp / "tensors.hex"
        write_words(commands_hex, [int.from_bytes(raw, "little")
                                   for _, raw in selected], 32)
        write_words(pcs_hex, [pc for pc, _ in selected], 8)
        write_words(operators_hex,
                    chunks((PROGRAM / "tile_operator_desc.bin").read_bytes(), 64),
                    128)
        write_words(tensors_hex,
                    chunks((PROGRAM / "tensor_desc.bin").read_bytes(), 64), 128)
        output = run([vvp, str(sim), f"+COMMANDS={commands_hex}",
                      f"+PCS={pcs_hex}", f"+OPERATORS={operators_hex}",
                      f"+TENSORS={tensors_hex}",
                      f"+COMMAND_COUNT={len(selected)}"])
        if "execution frontend: PASS" not in output:
            raise RuntimeError(output)
    print("execution frontend RTL: PASS (248 conv, 32 residual, 3 upsample)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
