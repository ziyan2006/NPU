"""RTL regression for task-header parsing, block reads, and LUT preload."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"


def write_u64_hex(path: Path, payload: bytes) -> None:
    assert len(payload) % 8 == 0
    path.write_text("".join(
        f"{int.from_bytes(payload[offset:offset + 8], 'little'):016x}\n"
        for offset in range(0, len(payload), 8)), encoding="ascii")


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def main() -> int:
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")
    image = (PROGRAM / "task_image.bin").read_bytes()
    manifest = json.loads((PROGRAM / "task_image.json").read_text(encoding="utf-8"))
    lut = (PROGRAM / "tanh_lut_int12.bin").read_bytes()

    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        sim = temp / "tb_npu_task_loader.vvp"
        run([iverilog, "-g2012", "-Wall", "-s", "tb_npu_task_loader",
             "-o", str(sim), str(RTL / "npu_axi_block_reader.sv"),
             str(RTL / "npu_task_loader.sv"),
             str(RTL / "tb" / "tb_npu_task_loader.sv")])
        lut_hex = temp / "lut.hex"
        write_u64_hex(lut_hex, lut)

        # Manifest keys use commands/tensor_desc/..., whereas TB plusargs are
        # intentionally short and map directly to RTL output names.
        offsets = {
            "COMMAND": "commands", "TENSOR": "tensor_desc",
            "OPERATOR": "operator_desc", "QUANT": "quant_desc",
            "SEGMENT": "segments", "ACTIVATION": "activation",
            "WEIGHT": "weights", "BIAS": "bias",
            "QUANT_PARAM": "quant_params", "LUT": "tanh_lut",
        }
        common = [f"+LUT={lut_hex}", f"+TOTAL_BYTES={len(image)}"]
        common += [f"+{argument}_OFFSET={manifest['sections'][section]['offset']}"
                   for argument, section in offsets.items()]

        valid_header = temp / "header_valid.hex"
        write_u64_hex(valid_header, image[:128])
        output = run([vvp, str(sim), f"+HEADER={valid_header}",
                      "+EXPECT_ERROR=0", "+ERROR_REASON=0", *common])
        if "task loader: PASS" not in output:
            raise RuntimeError(output)

        corrupt = bytearray(image[:128])
        corrupt[0] ^= 0x80
        invalid_header = temp / "header_invalid.hex"
        write_u64_hex(invalid_header, corrupt)
        output = run([vvp, str(sim), f"+HEADER={invalid_header}",
                      "+EXPECT_ERROR=1", "+ERROR_REASON=2", *common])
        if "task loader: PASS" not in output:
            raise RuntimeError(output)

    print("task loader RTL: PASS (real 1,869-command image, 4,096 LUT entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
