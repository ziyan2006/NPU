#!/usr/bin/env python3
"""Create a Zynq standalone Vitis workspace for staged audio playback."""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XSA = (
    ROOT / "hardware" / "build" / "navigator_z7020_audio_export"
    / "stem_npu_audio_navigator_z7020.xsa"
)
BUILD_ROOT = ROOT / "hardware" / "build" / "navigator_audio_player"
PLATFORM_NAME = "navigator_z7020_audio_platform"
DOMAIN_NAME = "standalone_ps7_cortexa9_0"
MODES = {
    "Tone": ("tone_player", "PLAYER_MODE_TONE=1"),
    "Wav": ("wav_player", "PLAYER_MODE_WAV=1"),
    "Mp3Bypass": ("mp3_bypass_player", "PLAYER_MODE_MP3_BYPASS=1"),
    "FullStem": ("stem_player", "PLAYER_MODE_FULL_STEM=1"),
}
MODE_SOURCES = {
    "Tone": ["audio_hw.c", "pcm_ring.c", "wav_source.c",
             "player_platform_vitis.c", "player_main.c"],
    "Wav": ["audio_hw.c", "pcm_ring.c", "wav_source.c",
            "player_platform_vitis.c", "player_main.c"],
    "Mp3Bypass": ["audio_hw.c", "sd_mp3_source.c",
                  "player_platform_vitis.c", "player_main.c"],
    "FullStem": [
        "audio_hw.c", "pcm_ring.c", "sd_mp3_source.c", "stem_frontend.c",
        "stem_npu_session.c", "stem_backend.c", "player.c",
        "player_platform_vitis.c", "player_main.c", "npu_driver.c",
        "kiss_fft.c", "kiss_fftr.c", "stem_filterbank.c",
        "stem_task_payload.c",
    ],
}
MODE_HEADERS = {
    "Tone": ["audio_hw.h", "pcm_ring.h", "wav_source.h", "player_platform.h"],
    "Wav": ["audio_hw.h", "pcm_ring.h", "wav_source.h", "player_platform.h"],
    "Mp3Bypass": ["audio_hw.h", "sd_mp3_source.h", "player_platform.h"],
    "FullStem": [
        "audio_hw.h", "pcm_ring.h", "sd_mp3_source.h", "stem_contract.h",
        "stem_frontend.h", "stem_npu_session.h", "stem_backend.h", "player.h",
        "player_platform.h", "npu_driver.h", "stem_filterbank.h",
        "stem_task_metadata.h", "stem_task_payload.h",
    ],
}
MODE_VENDOR_HEADERS = {
    "Tone": [],
    "Wav": [],
    "Mp3Bypass": ["minimp3.h"],
    "FullStem": ["minimp3.h", "kiss_fft.h", "kiss_fftr.h",
                 "_kiss_fft_guts.h", "kiss_fft_log.h"],
}
REQUIRED_SYMBOLS = {"player_main", "f_mount", "audio_hw_write_frame"}


def resolve_toolchain() -> dict[str, str]:
    vitis_root = Path(
        os.environ.get("XILINX_VITIS", r"C:\AMDDesignTools\2026.1\Vitis")
    )
    vivado_root = Path(
        os.environ.get("XILINX_VIVADO", r"C:\AMDDesignTools\2026.1\Vivado")
    )
    pyesw = vitis_root / "data" / "embeddedsw" / "scripts" / "pyesw"
    return {
        "vitis": str(vitis_root / "bin" / "vitis.bat"),
        "sdtgen": str(vivado_root / "bin" / "sdtgen.bat"),
        "create_bsp": str(pyesw / "create_bsp.py"),
        "config_bsp": str(pyesw / "config_bsp.py"),
        "build_bsp": str(pyesw / "build_bsp.py"),
        "create_app": str(pyesw / "create_app.py"),
        "build_app": str(pyesw / "build_app.py"),
    }


def to_tool_path(path: Path) -> str:
    return path.resolve().as_posix()


def require_artifacts(paths: list[Path], stage: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{stage} did not produce required files: {missing}")


def run_tool(command: list[str], artifacts: list[Path], stage: str) -> None:
    print(f"AUDIO_VITIS_STAGE {stage}", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{stage} exited with status {result.returncode}")
    # AMD 2026.1 vitis.bat can return zero after an embedded Python failure.
    require_artifacts(artifacts, stage)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def source_path(name: str) -> Path:
    candidates = (
        ROOT / "software" / "audio_player" / "src" / name,
        ROOT / "software" / "src" / name,
        ROOT / "software" / "audio_player" / "generated" / name,
        ROOT / "software" / "audio_player" / "third_party" / "kissfft" / name,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def header_path(name: str) -> Path:
    candidates = (
        ROOT / "software" / "audio_player" / "include" / name,
        ROOT / "software" / "include" / name,
        ROOT / "software" / "audio_player" / "generated" / name,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def vendor_header_path(name: str) -> Path:
    candidates = (
        ROOT / "software" / "audio_player" / "third_party" / "minimp3" / name,
        ROOT / "software" / "audio_player" / "third_party" / "kissfft" / name,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def validate_audio_xsa(path: Path) -> None:
    require_file(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not an XSA zip archive: {path}")
    with zipfile.ZipFile(path) as archive:
        descriptions = []
        for name in archive.namelist():
            if name.lower().endswith((".hwh", ".bda")):
                descriptions.append(archive.read(name).decode(errors="replace"))
    text = "\n".join(descriptions)
    if "audio_out_0" not in text or not re.search(r"0x43c10000\b", text, re.I):
        raise ValueError("XSA does not map audio_out_0 at 0x43C10000")


def describe_build(mode: str, xsa: Path) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError(f"unsupported audio player mode: {mode}")
    validate_audio_xsa(xsa)
    for name in MODE_SOURCES[mode]:
        require_file(source_path(name))
    for name in MODE_HEADERS[mode]:
        require_file(header_path(name))
    for name in MODE_VENDOR_HEADERS[mode]:
        require_file(vendor_header_path(name))
    application, definition = MODES[mode]
    workspace_root = Path(
        os.environ.get("AUDIO_PLAYER_WORKSPACE_ROOT", str(BUILD_ROOT))
    ).resolve()
    return {
        "mode": mode,
        "xsa": str(xsa),
        "workspace": str(workspace_root / f"workspace_{mode.lower()}"),
        "platform": PLATFORM_NAME,
        "domain": DOMAIN_NAME,
        "application": application,
        "definition": definition,
        "libraries": ["xilffs"],
        "sources": list(MODE_SOURCES[mode]),
        "headers": list(MODE_HEADERS[mode]),
        "vendor_headers": list(MODE_VENDOR_HEADERS[mode]),
        "elf": str(BUILD_ROOT / f"{application}.elf"),
    }


def inspect_symbol_table(text: str) -> None:
    defined: set[str] = set()
    undefined: set[str] = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2:
            symbol_type, name = fields
        elif len(fields) >= 3:
            symbol_type, name = fields[-2], fields[-1]
        else:
            continue
        if symbol_type.upper() == "U":
            undefined.add(name)
        else:
            defined.add(name)
    missing = REQUIRED_SYMBOLS - defined
    if missing or undefined:
        raise ValueError(
            f"ELF symbol contract failed: missing={sorted(missing)}, "
            f"undefined={sorted(undefined)}"
        )


def inspect_elf(path: Path) -> None:
    require_file(path)
    nm = os.environ.get("ARM_NONE_EABI_NM") or shutil.which("arm-none-eabi-nm")
    if not nm:
        raise RuntimeError("arm-none-eabi-nm was not found")
    result = subprocess.run(
        [nm, "-g", str(path)], check=True, capture_output=True, text=True
    )
    inspect_symbol_table(result.stdout)


def remove_workspace(workspace: Path) -> None:
    expected_parent = Path(
        os.environ.get("AUDIO_PLAYER_WORKSPACE_ROOT", str(BUILD_ROOT))
    ).resolve()
    if workspace.parent.resolve() != expected_parent:
        raise ValueError(f"refusing to clean workspace outside {expected_parent}")
    if workspace.exists():
        def remove_readonly(function: object, path: str, error: object) -> None:
            del error
            os.chmod(path, stat.S_IWRITE)
            function(path)  # type: ignore[operator]

        shutil.rmtree(workspace, onexc=remove_readonly)


def import_files(component: object, directory: Path, names: list[str]) -> None:
    component.import_files(
        from_loc=str(directory), files=names, dest_dir_in_cmp="src"
    )


def generated_domain_has_xilffs(workspace: Path) -> bool:
    for path in workspace.rglob("*"):
        if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            continue
        if path.suffix.lower() not in {".cmake", ".json", ".mss", ".txt", ".yaml", ".yml"}:
            continue
        if "xilffs" in path.read_text(encoding="utf-8", errors="ignore").lower():
            return True
    return False


def configure_generated_app(app_source: Path,
                            description: dict[str, object]) -> None:
    for name in description["sources"]:
        shutil.copy2(source_path(str(name)), app_source / str(name))
    for name in description["headers"]:
        shutil.copy2(header_path(str(name)), app_source / str(name))
    for name in description["vendor_headers"]:
        shutil.copy2(vendor_header_path(str(name)), app_source / str(name))

    user_config = app_source / "UserConfig.cmake"
    text = user_config.read_text(encoding="utf-8")
    replacement = (
        'set(USER_COMPILE_DEFINITIONS\n'
        f'"{description["definition"]}"\n)'
    )
    text, count = re.subn(
        r'set\(USER_COMPILE_DEFINITIONS\s*""\s*\)', replacement, text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("could not configure generated application mode")
    if description["mode"] == "FullStem":
        text, count = re.subn(
            r"set\(USER_COMPILE_OPTIMIZATION_LEVEL\s+-O0\s*\)",
            "set(USER_COMPILE_OPTIMIZATION_LEVEL -O2)",
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError("could not enable FullStem compiler optimization")
    user_config.write_text(text, encoding="utf-8")


def configure_linker_stack(user_config: Path, mode: str) -> None:
    if mode not in ("Mp3Bypass", "FullStem"):
        return
    text = user_config.read_text(encoding="utf-8")
    stack_size = "0x20000" if mode == "FullStem" else "0x10000"
    text, count = re.subn(
        r"set\(USER_LINK_OTHER_FLAGS\s*\)",
        f'set(USER_LINK_OTHER_FLAGS\n"-Wl,--defsym=_STACK_SIZE={stack_size}"\n)',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("could not set audio application stack")
    if mode == "FullStem":
        text, count = re.subn(
            r"set\(USER_LINK_LIBRARIES\s*\)",
            'set(USER_LINK_LIBRARIES\n"m"\n)',
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError("could not link FullStem with libm")
    user_config.write_text(text, encoding="utf-8")


def pyesw_script_command(toolchain: dict[str, str], script: str,
                         arguments: list[str]) -> list[str]:
    return [sys.executable, toolchain[script], *arguments]


def build_with_pyesw(description: dict[str, object]) -> Path:
    toolchain = resolve_toolchain()
    for name, path in toolchain.items():
        require_file(Path(path))

    workspace = Path(str(description["workspace"]))
    sdt_dir = workspace / "sdt"
    domain = workspace / "domain"
    application = workspace / str(description["application"])
    app_source = application / "src"
    system_dts = sdt_dir / "system-top.dts"
    bsp_yaml = domain / "bsp.yaml"
    app_yaml = app_source / "app.yaml"

    run_tool(
        [toolchain["sdtgen"], "-xsa", to_tool_path(Path(str(description["xsa"]))),
         "-dir", to_tool_path(sdt_dir)],
        [system_dts], "generate_sdt",
    )
    run_tool(
        pyesw_script_command(toolchain, "create_bsp", [
            "--proc", "ps7_cortexa9_0", "--sdt", to_tool_path(system_dts),
            "--ws_dir", to_tool_path(domain), "--os", "standalone",
            "--template", "empty_application",
        ]),
        [bsp_yaml], "create_bsp",
    )
    run_tool(
        pyesw_script_command(toolchain, "config_bsp", [
            "--domain_path", to_tool_path(domain), "--addlib", "xilffs",
        ]),
        [bsp_yaml], "add_xilffs",
    )
    if not generated_domain_has_xilffs(domain):
        raise RuntimeError("generated standalone domain does not contain xilffs")
    run_tool(
        pyesw_script_command(toolchain, "build_bsp", [
            "--domain_path", to_tool_path(domain),
        ]),
        [domain / "lib" / "libxil.a", domain / "lib" / "libxilffs.a"],
        "build_bsp",
    )
    run_tool(
        pyesw_script_command(toolchain, "create_app", [
            "--domain_path", to_tool_path(domain),
            "--ws_dir", to_tool_path(application),
            "--name", str(description["application"]),
            "--template", "empty_application", "--no_clangd", "True",
        ]),
        [app_yaml, app_source / "UserConfig.cmake"], "create_app",
    )
    configure_generated_app(app_source, description)
    configure_linker_stack(app_source / "UserConfig.cmake",
                           str(description["mode"]))

    expected_elf = application / "build" / f"{description['application']}.elf"
    run_tool(
        pyesw_script_command(toolchain, "build_app", [
            "--ws_dir", to_tool_path(application),
        ]),
        [expected_elf], "build_app",
    )
    return expected_elf


def build(mode: str, xsa: Path, clean: bool) -> dict[str, object]:
    description = describe_build(mode, xsa)
    workspace = Path(str(description["workspace"]))
    if clean:
        remove_workspace(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError(f"workspace already exists: {workspace}; use -Clean")
    workspace.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)

    built_elf = build_with_pyesw(description)
    destination = Path(str(description["elf"]))
    shutil.copy2(built_elf, destination)
    inspect_elf(destination)
    description["backend"] = "pyesw"
    manifest = BUILD_ROOT / f"{description['application']}_build.json"
    manifest.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
    return description


def main() -> int:
    mode = os.environ.get("AUDIO_PLAYER_MODE", "Wav")
    xsa = Path(os.environ.get("AUDIO_PLAYER_XSA", str(DEFAULT_XSA))).resolve()
    clean = os.environ.get("AUDIO_PLAYER_CLEAN", "0") == "1"
    description = build(mode, xsa, clean)
    print("AUDIO_VITIS_BUILD PASS " + json.dumps(description))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"AUDIO_VITIS_BUILD FAIL: {error}", file=sys.stderr)
        raise
