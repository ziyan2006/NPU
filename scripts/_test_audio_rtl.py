"""Focused RTL regressions for the WM8960 audio output subsystem."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "hardware" / "rtl"

CASES = {
    "csr": (
        "tb_audio_axi_csr",
        [
            RTL / "audio" / "audio_regs_pkg.sv",
            RTL / "audio" / "audio_axi_csr.sv",
            RTL / "tb" / "tb_audio_axi_csr.sv",
        ],
        "audio_axi_csr: PASS",
    ),
    "fifo": (
        "tb_audio_async_fifo",
        [
            RTL / "audio" / "audio_async_fifo.sv",
            RTL / "tb" / "tb_audio_async_fifo.sv",
        ],
        "audio_async_fifo: PASS",
    ),
    "key_mixer": (
        "tb_audio_key_mixer",
        [
            RTL / "audio" / "audio_key_debounce.sv",
            RTL / "audio" / "audio_stem_mixer.sv",
            RTL / "tb" / "tb_audio_key_mixer.sv",
        ],
        "audio_key_mixer: PASS",
    ),
    "i2s": (
        "tb_audio_i2s_tx",
        [
            RTL / "audio" / "audio_i2s_tx.sv",
            RTL / "audio" / "audio_test_tone.sv",
            RTL / "tb" / "tb_audio_i2s_tx.sv",
        ],
        "audio_i2s_tx: PASS",
    ),
}


def run_case(name: str, iverilog: str, vvp: str) -> None:
    top, sources, marker = CASES[name]
    with tempfile.TemporaryDirectory() as directory:
        image = Path(directory) / f"{top}.vvp"
        compile_result = subprocess.run(
            [iverilog, "-g2012", "-Wall", "-s", top, "-o", str(image),
             *map(str, sources)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if compile_result.returncode:
            raise RuntimeError(compile_result.stdout + compile_result.stderr)
        seeds = range(1, 21) if name == "fifo" else (1,)
        for seed in seeds:
            result = subprocess.run(
                [vvp, str(image), f"+SEED={seed}"], cwd=ROOT,
                capture_output=True, text=True,
            )
            if result.returncode or marker not in result.stdout:
                raise RuntimeError(
                    f"{name} seed {seed}\n" + result.stdout + result.stderr
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES))
    args = parser.parse_args()
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        raise RuntimeError("iverilog and vvp are required")
    selected = [args.case] if args.case else list(CASES)
    for name in selected:
        run_case(name, iverilog, vvp)
    print(f"audio RTL: PASS ({', '.join(selected)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
