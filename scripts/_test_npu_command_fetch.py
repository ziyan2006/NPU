"""RTL regression for bounded sequential command fetch."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"


def write_commands(path: Path, payload: bytes) -> None:
    assert len(payload) % 16 == 0
    path.write_text("".join(
        f"{int.from_bytes(payload[offset:offset + 16], 'little'):032x}\n"
        for offset in range(0, len(payload), 16)), encoding="ascii")


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def main() -> int:
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")
    commands = (PROGRAM / "tile_commands.bin").read_bytes()
    count = len(commands) // 16
    assert commands[-16] == 0x03

    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        sim = temp / "tb_npu_command_fetch.vvp"
        run([iverilog, "-g2012", "-Wall", "-s", "tb_npu_command_fetch",
             "-o", str(sim), str(RTL / "npu_command_fetch.sv"),
             str(RTL / "tb" / "tb_npu_command_fetch.sv")])
        cases = []
        valid_path = temp / "valid.hex"
        write_commands(valid_path, commands)
        cases.append((valid_path, 0, 0))

        missing_end = bytearray(commands)
        missing_end[-16] = 0x00
        missing_path = temp / "missing.hex"
        write_commands(missing_path, missing_end)
        cases.append((missing_path, 1, 4))

        early_end = bytearray(commands)
        early_end[10 * 16] = 0x03
        early_path = temp / "early.hex"
        write_commands(early_path, early_end)
        cases.append((early_path, 1, 3))

        for path, expect_error, reason in cases:
            output = run([vvp, str(sim), f"+COMMANDS={path}",
                          f"+COMMAND_COUNT={count}",
                          f"+EXPECT_ERROR={expect_error}",
                          f"+ERROR_REASON={reason}"])
            if "command fetch: PASS" not in output:
                raise RuntimeError(output)

    print("command fetch RTL: PASS (1,869 commands + END-boundary faults)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
