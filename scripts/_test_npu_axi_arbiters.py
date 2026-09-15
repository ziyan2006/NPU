"""Concurrent-burst regression for shared NPU AXI read/write arbiters."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"


def main() -> int:
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")
    with tempfile.TemporaryDirectory() as directory:
        image = Path(directory) / "tb_npu_axi_arbiters.vvp"
        result = subprocess.run([
            iverilog, "-g2012", "-Wall", "-s", "tb_npu_axi_arbiters",
            "-o", str(image), str(RTL / "npu_axi_read_arbiter3.sv"),
            str(RTL / "npu_axi_write_arbiter2.sv"),
            str(RTL / "tb" / "tb_npu_axi_arbiters.sv")], cwd=ROOT,
            capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        result = subprocess.run([vvp, str(image)], cwd=ROOT,
                                capture_output=True, text=True)
        if result.returncode or "NPU AXI arbiters: PASS" not in result.stdout:
            raise RuntimeError(result.stdout + result.stderr)
    print("shared AXI arbiters RTL: PASS (3 read + 2 write clients)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
