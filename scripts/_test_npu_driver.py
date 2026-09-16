#!/usr/bin/env python3
"""Check that the portable software driver matches and compiles with the RTL."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "software" / "include" / "npu_driver.h"
SOURCE = ROOT / "software" / "src" / "npu_driver.c"
TEST = ROOT / "software" / "tests" / "test_npu_driver.c"
RTL = ROOT / "hardware" / "rtl" / "npu_csr.sv"

EXPECTED_REGISTERS = {
    "IP_ID": 0x000,
    "VERSION": 0x004,
    "ISA_VERSION": 0x008,
    "CAPABILITY0": 0x00C,
    "CAPABILITY1": 0x010,
    "CONTROL": 0x014,
    "STATUS": 0x018,
    "IRQ_STATUS": 0x01C,
    "IRQ_ENABLE": 0x020,
    "TASK_BASE_LO": 0x024,
    "TASK_BASE_HI": 0x028,
    "TASK_BYTES": 0x02C,
    "TASK_TAG": 0x030,
    "DOORBELL": 0x034,
    "COMPLETED_TAG": 0x038,
    "ERROR_CODE": 0x03C,
    "ERROR_PC": 0x040,
    "ERROR_INST_TAG": 0x044,
    "WATCHDOG_LIMIT": 0x048,
    "COMMANDS_RETIRED": 0x04C,
    "CYCLES_TOTAL_LO": 0x050,
    "CYCLES_TOTAL_HI": 0x054,
    "CYCLES_COMPUTE_LO": 0x058,
    "CYCLES_COMPUTE_HI": 0x05C,
    "CYCLES_RD_WAIT_LO": 0x060,
    "CYCLES_RD_WAIT_HI": 0x064,
    "CYCLES_WR_WAIT_LO": 0x068,
    "CYCLES_WR_WAIT_HI": 0x06C,
    "CYCLES_BANK_STALL_LO": 0x070,
    "CYCLES_BANK_STALL_HI": 0x074,
    "BYTES_READ_LO": 0x078,
    "BYTES_READ_HI": 0x07C,
    "BYTES_WRITTEN_LO": 0x080,
    "BYTES_WRITTEN_HI": 0x084,
    "READ_HIGH_WATER_LO": 0x088,
    "READ_HIGH_WATER_HI": 0x08C,
    "ERROR_COUNT": 0x090,
    "WRITE_HIGH_WATER_LO": 0x094,
    "WRITE_HIGH_WATER_HI": 0x098,
}


def parse_driver_registers(text: str) -> dict[str, int]:
    return {
        name: int(value, 16)
        for name, value in re.findall(
            r"^#define\s+NPU_REG_([A-Z0-9_]+)\s+0x([0-9a-fA-F]+)u\s*$",
            text,
            re.MULTILINE,
        )
    }


def compile_driver() -> str:
    include = str(ROOT / "software" / "include")
    native = next(
        (path for name in ("cc", "gcc", "clang") if (path := shutil.which(name))),
        None,
    )
    cross_candidates = [
        Path(r"C:\AMDDesignTools\2026.1\gnu\microblaze\nt\bin\mb-gcc.exe"),
        Path(r"C:\Xilinx\Vitis\2024.2\gnu\microblaze\nt\bin\mb-gcc.exe"),
    ]
    with tempfile.TemporaryDirectory(prefix="npu_driver_") as directory:
        output = Path(directory)
        common = ["-std=c11", "-Wall", "-Wextra", "-Werror", f"-I{include}"]
        if native is not None:
            executable = output / ("test_npu_driver.exe" if Path(native).suffix else "test_npu_driver")
            subprocess.run(
                [native, *common, str(SOURCE), str(TEST), "-o", str(executable)],
                check=True,
            )
            completed = subprocess.run(
                [str(executable)], check=True, capture_output=True, text=True
            )
            if "npu_driver: PASS" not in completed.stdout:
                raise AssertionError("native driver test did not report PASS")
            return f"native compile/run ({native})"

        cross = next((str(path) for path in cross_candidates if path.exists()), None)
        if cross is None:
            raise AssertionError("no native or Vivado cross C compiler found")
        for source in (SOURCE, TEST):
            subprocess.run(
                [
                    cross,
                    *common,
                    "-ffreestanding",
                    "-c",
                    str(source),
                    "-o",
                    str(output / f"{source.stem}.o"),
                ],
                check=True,
            )
        return f"cross compile ({cross})"


def main() -> None:
    header = HEADER.read_text(encoding="utf-8")
    rtl = RTL.read_text(encoding="utf-8")
    actual = parse_driver_registers(header)
    if actual != EXPECTED_REGISTERS:
        raise AssertionError(
            f"driver CSR map mismatch\nexpected={EXPECTED_REGISTERS}\nactual={actual}"
        )

    rtl_addresses = {
        int(value, 16)
        for value in re.findall(r"12'h([0-9a-fA-F]{3}):\s+read_decode_data", rtl)
    }
    if rtl_addresses != set(EXPECTED_REGISTERS.values()):
        raise AssertionError(
            "RTL read-decode CSR map differs from the public driver header"
        )
    rtl_ip_id = re.search(r"IP_ID\s*=\s*32'h([0-9a-fA-F_]+)", rtl)
    header_ip_id = re.search(r"NPU_IP_ID_VALUE\s+0x([0-9a-fA-F]+)u", header)
    if rtl_ip_id is None or header_ip_id is None:
        raise AssertionError("IP ID constants not found")
    if int(rtl_ip_id.group(1).replace("_", ""), 16) != int(
        header_ip_id.group(1), 16
    ):
        raise AssertionError("driver and RTL IP ID differ")

    compiler_result = compile_driver()
    print(f"npu_driver_contract: PASS ({len(actual)} CSRs; {compiler_result})")


if __name__ == "__main__":
    main()
