"""AXI4-Lite CSR protocol and task lifecycle regression."""
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
        image = Path(directory) / "tb_npu_csr.vvp"
        compile_result = subprocess.run([
            iverilog, "-g2012", "-Wall", "-s", "tb_npu_csr", "-o", str(image),
            str(RTL / "include" / "npu_isa_pkg.sv"), str(RTL / "npu_csr.sv"),
            str(RTL / "tb" / "tb_npu_csr.sv")], cwd=ROOT,
            capture_output=True, text=True)
        if compile_result.returncode:
            raise RuntimeError(compile_result.stdout + compile_result.stderr)
        result = subprocess.run([vvp, str(image)], cwd=ROOT,
                                capture_output=True, text=True)
        if result.returncode or "npu_csr: PASS" not in result.stdout:
            raise RuntimeError(result.stdout + result.stderr)
    print("AXI4-Lite CSR RTL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
