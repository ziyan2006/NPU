"""Run the minimal END task through the complete NPU top-level RTL."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"


def load_builder():
    path = ROOT / "scripts" / "52_build_npu_end_task.py"
    spec = importlib.util.spec_from_file_location("npu_end_task", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_u64_hex(path: Path, payload: bytes) -> None:
    assert len(payload) % 8 == 0
    path.write_text("".join(f"{int.from_bytes(payload[i:i + 8], 'little'):016x}\n"
                            for i in range(0, len(payload), 8)), encoding="ascii")


def main() -> int:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("Icarus Verilog is required for the fast smoke test")
    image, metadata = load_builder().build_end_task()
    assert metadata["total_bytes"] == len(image) == 9024
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        image_hex, output_hex = temp / "task.hex", temp / "output.hex"
        write_u64_hex(image_hex, image)
        sources = [line.strip() for line in (RTL / "npu_rtl.f").read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
        sources.append("hardware/rtl/tb/tb_npu_top_task.sv")
        file_list = temp / "rtl.f"
        file_list.write_text("\n".join((ROOT / source).as_posix()
                                         for source in sources) + "\n")
        snapshot = temp / "npu_end_task.vvp"
        for command, timeout in (([iverilog, "-g2012", "-s", "tb_npu_top_task",
                                   "-o", str(snapshot), "-f", str(file_list)], 240),
                                 ([vvp, str(snapshot), f"+IMAGE={image_hex.name}",
                                   f"+IMAGE_BYTES={len(image)}", f"+OUTPUT={output_hex.name}",
                                   "+OUTPUT_OFFSET=576", "+OUTPUT_BYTES=8",
                                   "+EXPECTED_COMMANDS=1", "+MAX_SIM_NS=100000"], 120)):
            result = subprocess.run(command, cwd=temp, capture_output=True, text=True,
                                    timeout=timeout)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        output_words = [line for line in output_hex.read_text(encoding="ascii").splitlines()
                        if line and not line.startswith(("//", "@"))]
        if output_words != ["0000000000000000"]:
            raise RuntimeError("END task unexpectedly modified its inert activation byte")
    print("minimal END task RTL: PASS (task loader through CSR completion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
