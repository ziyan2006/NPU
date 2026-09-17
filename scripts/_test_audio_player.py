#!/usr/bin/env python3
"""Compile and run host-side tests for the staged audio player."""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
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


def load_workspace_builder():
    path = ROOT / "scripts" / "100_create_audio_vitis_workspace.py"
    spec = importlib.util.spec_from_file_location("audio_vitis_builder", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import workspace builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vitis_layout() -> None:
    builder = load_workspace_builder()
    toolchain = builder.resolve_toolchain()
    for name in ("sdtgen", "create_bsp", "config_bsp", "build_bsp",
                 "create_app", "build_app"):
        if not Path(toolchain[name]).is_file():
            raise AssertionError(f"missing Vitis toolchain entry {name}")
    if builder.to_tool_path(Path(r"D:\audio\build")) != "D:/audio/build":
        raise AssertionError("Windows tool paths must use Tcl-safe slashes")
    pyesw_command = builder.pyesw_script_command(
        toolchain, "create_bsp", ["--help"])
    if pyesw_command != [sys.executable, toolchain["create_bsp"], "--help"]:
        raise AssertionError("pyesw must run in the current configured Python")

    with tempfile.TemporaryDirectory(prefix="audio_artifact_contract_") as directory:
        present = Path(directory) / "present.elf"
        present.write_bytes(b"elf")
        builder.require_artifacts([present], "probe")
        try:
            builder.require_artifacts([Path(directory) / "missing.elf"], "probe")
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing output passed the artifact contract")

    with tempfile.TemporaryDirectory(prefix="audio_clean_contract_") as directory:
        parent = Path(directory)
        workspace = parent / "workspace_tone"
        generated = workspace / "generated.h"
        workspace.mkdir()
        generated.write_text("generated", encoding="ascii")
        generated.chmod(stat.S_IREAD)
        previous_root = os.environ.get("AUDIO_PLAYER_WORKSPACE_ROOT")
        os.environ["AUDIO_PLAYER_WORKSPACE_ROOT"] = str(parent)
        try:
            builder.remove_workspace(workspace)
        finally:
            if generated.exists():
                generated.chmod(stat.S_IWRITE)
            if previous_root is None:
                os.environ.pop("AUDIO_PLAYER_WORKSPACE_ROOT", None)
            else:
                os.environ["AUDIO_PLAYER_WORKSPACE_ROOT"] = previous_root
        if workspace.exists():
            raise AssertionError("clean left a generated workspace behind")

    xsa = (
        ROOT / "hardware" / "build" / "navigator_z7020_audio_export"
        / "stem_npu_audio_navigator_z7020.xsa"
    )
    builder.validate_audio_xsa(xsa)
    description = builder.describe_build("Wav", xsa)
    required_sources = {
        "audio_hw.c", "pcm_ring.c", "wav_source.c",
        "player_platform_vitis.c", "player_main.c",
    }
    if set(description["sources"]) != required_sources:
        raise AssertionError(f"Vitis source imports differ: {description['sources']}")
    if description["libraries"] != ["xilffs"]:
        raise AssertionError("standalone domain must enable xilffs")
    if description["definition"] != "PLAYER_MODE_WAV=1":
        raise AssertionError("WAV application build definition is missing")

    with tempfile.TemporaryDirectory(prefix="bad_audio_xsa_") as directory:
        bad_xsa = Path(directory) / "bad.xsa"
        with zipfile.ZipFile(bad_xsa, "w") as archive:
            archive.writestr(
                "system.hwh",
                '<MEMRANGE INSTANCE="audio_out_0" BASEVALUE="0x43C20000"/>',
            )
        try:
            builder.validate_audio_xsa(bad_xsa)
        except ValueError:
            pass
        else:
            raise AssertionError("workspace builder accepted a wrong audio address")

    builder.inspect_symbol_table(
        "00001000 T player_main\n"
        "00002000 T f_mount\n"
        "00003000 T audio_hw_write_frame\n"
    )
    try:
        builder.inspect_symbol_table(
            "00001000 T player_main\n"
            "         U f_mount\n"
            "00003000 T audio_hw_write_frame\n"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("ELF inspection accepted an unresolved symbol")


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
    choices = sorted([*CASES, "vitis_layout"])
    parser.add_argument("--case", choices=choices)
    arguments = parser.parse_args()
    selected = [arguments.case] if arguments.case else choices
    for case in selected:
        if case == "vitis_layout":
            test_vitis_layout()
        else:
            compile_and_run(case)
    print(f"audio player: PASS ({', '.join(selected)})")


if __name__ == "__main__":
    main()
