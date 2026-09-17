#!/usr/bin/env python3
"""Compile and run host-side tests for the staged audio player."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
NPU_INCLUDE = ROOT / "software" / "include"

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
    "mp3": (
        [
            PLAYER / "src" / "sd_mp3_source.c",
            PLAYER / "tests" / "test_mp3_source.c",
        ],
        "mp3 source: PASS",
    ),
    "frontend": (
        [
            PLAYER / "third_party" / "kissfft" / "kiss_fft.c",
            PLAYER / "third_party" / "kissfft" / "kiss_fftr.c",
            PLAYER / "generated" / "stem_filterbank.c",
            PLAYER / "src" / "stem_frontend.c",
            PLAYER / "tests" / "test_stem_frontend.c",
        ],
        "stem frontend: PASS",
    ),
    "npu_session": (
        [
            ROOT / "software" / "src" / "npu_driver.c",
            PLAYER / "generated" / "stem_task_payload.c",
            PLAYER / "src" / "stem_npu_session.c",
            PLAYER / "tests" / "test_stem_npu_session.c",
        ],
        "stem npu session: PASS",
    ),
}


def verify_kissfft_upstream() -> None:
    directory = PLAYER / "third_party" / "kissfft"
    metadata = json.loads((directory / "UPSTREAM.json").read_text(encoding="utf-8"))
    if metadata["url"] != "https://github.com/mborgerding/kissfft.git":
        raise AssertionError("KissFFT upstream URL changed")
    if metadata["tag"] != "131.2.0":
        raise AssertionError("KissFFT tag changed")
    if metadata["commit"] != "7bce4153c6bc8aba2db0e889e576f9d00505cbe1":
        raise AssertionError("KissFFT commit changed")
    if metadata["license"] != "BSD-3-Clause":
        raise AssertionError("KissFFT license metadata changed")
    for name, expected in metadata["files"].items():
        path = directory / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"KissFFT vendored file differs: {name}")


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
    mp3_description = builder.describe_build("Mp3Bypass", xsa)
    if mp3_description["definition"] != "PLAYER_MODE_MP3_BYPASS=1":
        raise AssertionError("MP3 bypass build definition is missing")
    if "sd_mp3_source.c" not in mp3_description["sources"]:
        raise AssertionError("MP3 source is missing from Vitis imports")
    if "minimp3.h" not in mp3_description["vendor_headers"]:
        raise AssertionError("pinned minimp3 header is missing from Vitis imports")
    with tempfile.TemporaryDirectory(prefix="mp3_linker_contract_") as directory:
        linker = Path(directory) / "UserConfig.cmake"
        linker.write_text(
            "set(USER_LINK_OTHER_FLAGS\n)\n",
            encoding="ascii",
        )
        builder.configure_linker_stack(linker, "Mp3Bypass")
        if "--defsym=_STACK_SIZE=0x10000" not in linker.read_text(encoding="ascii"):
            raise AssertionError("MP3 application stack must be 64 KiB")

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
    if case == "mp3":
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "97_generate_audio_vectors.py"),
             "--mp3", "--check"],
            check=True,
        )
    elif case == "frontend":
        verify_kissfft_upstream()
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "98_generate_stem_constants.py"),
             "--check"],
            check=True,
        )
    elif case == "npu_session":
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "99_generate_stem_task_payload.py"),
             "--check"],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "97_generate_audio_vectors.py"),
             "--stem", "--check"],
            check=True,
        )
    with tempfile.TemporaryDirectory(prefix=f"audio_{case}_") as directory:
        output = Path(directory)
        executable = output / "test.exe"
        native = next(
            (path for name in ("cc", "gcc", "clang")
             if (path := shutil.which(name))),
            None,
        )
        defines = ["XPAR_AUDIO_OUT_AXI_0_BASEADDR=0x43C10000U"]
        include_paths = [INCLUDE]
        if case == "mp3":
            include_paths.append(PLAYER / "third_party" / "minimp3")
        elif case == "frontend":
            include_paths.extend([
                PLAYER / "third_party" / "kissfft",
                PLAYER / "generated",
            ])
        elif case == "npu_session":
            include_paths.extend([NPU_INCLUDE, PLAYER / "generated"])
        if native is not None:
            command = [
                native,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                *(f"-I{path}" for path in include_paths),
                *(f"-D{define}" for define in defines),
                *(str(source) for source in sources),
                *(["-lm"] if case == "frontend" else []),
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
                *(["/wd4244", "/D_CRT_SECURE_NO_WARNINGS"]
                  if case == "mp3" else
                  ["/wd4267", "/D_CRT_SECURE_NO_WARNINGS"]
                  if case == "frontend" else []),
                *(f"/I{path}" for path in include_paths),
                *(f"/D{define}" for define in defines),
                *(str(source) for source in sources),
                f"/Fe:{executable}",
            ]
            subprocess.run(command, check=True, env=environment, cwd=output)
        run_environment = os.environ.copy()
        if case == "mp3":
            run_environment["MP3_FIXTURE_DIR"] = str(
                PLAYER / "tests" / "fixtures"
            )
            run_environment["MP3_VECTOR_DIR"] = str(
                ROOT / "hardware" / "build" / "audio_vectors"
            )
        elif case == "frontend":
            run_environment["STEM_VECTOR_DIR"] = str(
                ROOT / "hardware" / "build" / "audio_vectors" / "stem_frontend"
            )
        completed = subprocess.run(
            [str(executable)], check=False, capture_output=True, text=True,
            env=run_environment,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise subprocess.CalledProcessError(
                completed.returncode, [str(executable)]
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
