"""Run the complete 1,869-command image through npu_top and one AXI memory."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from npu_task_reference import TaskReference, seed_input


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"


def write_u64_hex(path: Path, payload: bytes) -> None:
    assert len(payload) % 8 == 0
    path.write_text("".join(
        f"{int.from_bytes(payload[offset:offset + 8], 'little'):016x}\n"
        for offset in range(0, len(payload), 8)), encoding="ascii")


def main() -> int:
    vivado_bin = Path(r"C:\AMDDesignTools\2026.1\Vivado\bin")
    xvlog = shutil.which("xvlog") or str(vivado_bin / "xvlog.bat")
    xelab = shutil.which("xelab") or str(vivado_bin / "xelab.bat")
    xsim = shutil.which("xsim") or str(vivado_bin / "xsim.bat")
    if not all(Path(tool).exists() for tool in (xvlog, xelab, xsim)):
        raise RuntimeError("Vivado XSim is required for the complete RTL task")
    image = (PROGRAM / "task_image.bin").read_bytes()
    manifest = json.loads((PROGRAM / "task_image.json").read_text(encoding="utf-8"))
    program = json.loads((PROGRAM / "program.json").read_text(encoding="utf-8"))
    image = seed_input(image, program, manifest, 0x4e50_5531)
    reference = TaskReference(PROGRAM, image)
    reference.run(progress=True)
    expected_output = reference.output_bytes()
    output_tensor = program["tensors"][int(program["entry"]["output_tensor"])]
    output_offset = (int(manifest["sections"]["activation"]["offset"])
                     + int(output_tensor["base_offset"]))
    output_bytes = int(output_tensor["allocation_bytes"])

    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        image_hex = temp / "task_image.hex"
        output_hex = temp / "output.hex"
        write_u64_hex(image_hex, image)
        file_list = temp / "rtl.f"
        sources = [line.strip() for line in (RTL / "npu_rtl.f").read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
        sources.append("hardware/rtl/tb/tb_npu_top_task.sv")
        file_list.write_text("\n".join((ROOT / source).as_posix()
                                        for source in sources) + "\n")
        compile_result = subprocess.run([xvlog, "-sv", "-f", str(file_list)],
                                        cwd=temp, capture_output=True, text=True,
                                        timeout=240)
        if compile_result.returncode:
            raise RuntimeError(compile_result.stdout + compile_result.stderr)
        elaborate_result = subprocess.run([
            xelab, "tb_npu_top_task", "-s", "npu_top_task_sim"], cwd=temp,
            capture_output=True, text=True, timeout=300)
        if elaborate_result.returncode:
            raise RuntimeError(elaborate_result.stdout + elaborate_result.stderr)
        result = subprocess.run([
            xsim, "npu_top_task_sim", "-runall",
            "-testplusarg", f'"IMAGE={image_hex.name}"',
            "-testplusarg", f'"IMAGE_BYTES={len(image)}"',
            "-testplusarg", f'"OUTPUT={output_hex.name}"',
            "-testplusarg", f'"OUTPUT_OFFSET={output_offset}"',
            "-testplusarg", f'"OUTPUT_BYTES={output_bytes}"'], cwd=temp,
            capture_output=True, text=True, timeout=600)
        if result.returncode or "npu_top task: PASS" not in result.stdout:
            raise RuntimeError(result.stdout + result.stderr)
        output_words = [int(line, 16) for line in output_hex.read_text().splitlines()
                        if line and not line.startswith(("//", "@"))]
        if len(output_words) != output_bytes // 8:
            raise RuntimeError(f"output dump length {len(output_words)}")
        rtl_output = b"".join(word.to_bytes(8, "little")
                              for word in output_words)
        if rtl_output != expected_output:
            mismatch = next(index for index, (rtl, expected)
                            in enumerate(zip(rtl_output, expected_output))
                            if rtl != expected)
            raise RuntimeError(
                f"full-task bit mismatch at output byte {mismatch}: "
                f"RTL=0x{rtl_output[mismatch]:02x}, "
                f"reference=0x{expected_output[mismatch]:02x}")
        print(result.stdout.strip())
        print(f"full task bit-exact output: {output_bytes:,} bytes PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
