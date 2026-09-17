#!/usr/bin/env python3
"""Compile and run host-side tests for the staged audio player."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAYER = ROOT / "software" / "audio_player"
INCLUDE = PLAYER / "include"

CASES = {
    "foundation": (
        [
            PLAYER / "src" / "audio_hw.c",
            PLAYER / "src" / "pcm_ring.c",
            PLAYER / "src" / "wav_source.c",
            PLAYER / "tests" / "test_audio_foundation.c",
        ],
        "audio foundation: PASS",
    ),
}


def visual_studio_vcvars() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools")
        / "VC/Auxiliary/Build/vcvars64.bat",
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community")
        / "VC/Auxiliary/Build/vcvars64.bat",
    ]
    return next((path for path in candidates if path.is_file()), None)


def compile_and_run(case: str) -> None:
    sources, marker = CASES[case]
    with tempfile.TemporaryDirectory(prefix=f"audio_{case}_") as directory:
        output = Path(directory)
        executable = output / "test.exe"
        native = next(
            (path for name in ("cc", "gcc", "clang")
             if (path := shutil.which(name))),
            None,
        )
        defines = ["XPAR_AUDIO_OUT_AXI_0_BASEADDR=0x43C10000U"]
        if native is not None:
            command = [
                native,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{INCLUDE}",
                *(f"-D{define}" for define in defines),
                *(str(source) for source in sources),
                "-o",
                str(executable),
            ]
            subprocess.run(command, check=True)
        else:
            vcvars = visual_studio_vcvars()
            if vcvars is None:
                raise AssertionError("no native C compiler found")
            setup = subprocess.run(
                f'call "{vcvars}" >nul && set',
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            setup_environment = dict(
                line.split("=", 1)
                for line in setup.stdout.splitlines()
                if "=" in line
            )
            environment = os.environ.copy()
            environment.update(setup_environment)
            compiler_path = next(
                (value for key, value in setup_environment.items()
                 if key.casefold() == "path".casefold()),
                None,
            )
            compiler = shutil.which("cl", path=compiler_path)
            if compiler is None:
                raise AssertionError("Visual Studio environment has no cl.exe")
            command = [
                compiler,
                "/nologo",
                "/std:c11",
                "/W4",
                "/WX",
                f"/I{INCLUDE}",
                *(f"/D{define}" for define in defines),
                *(str(source) for source in sources),
                f"/Fe:{executable}",
            ]
            subprocess.run(command, check=True, env=environment, cwd=output)
        completed = subprocess.run(
            [str(executable)], check=True, capture_output=True, text=True
        )
        if marker not in completed.stdout:
            raise AssertionError(f"{case} did not report {marker!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES))
    arguments = parser.parse_args()
    selected = [arguments.case] if arguments.case else list(CASES)
    for case in selected:
        compile_and_run(case)
    print(f"audio player: PASS ({', '.join(selected)})")


if __name__ == "__main__":
    main()
