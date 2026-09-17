#!/usr/bin/env python3
"""Create and build the Navigator Z7020 Vitis standalone application.

Run through ``vitis.bat -s`` (or scripts/65_build_navigator_vitis.ps1).  The
script deliberately creates only an ignored workspace under hardware/build;
it neither contacts a board nor generates boot media.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import vitis


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XSA = (ROOT / "hardware" / "build" / "navigator_z7020_vendor_v37_export"
               / "stem_npu_navigator_z7020_candidate.xsa")
DEFAULT_WORKSPACE = ROOT / "hardware" / "build" / "navigator_vitis_workspace"
PAYLOAD_DIR = ROOT / "software" / "bringup" / "navigator_vitis" / "generated"
VITIS_SOURCES = ROOT / "software" / "bringup" / "navigator_vitis"

PLATFORM_NAME = "navigator_z7020_npu_platform"
DOMAIN_NAME = "standalone_ps7_cortexa9_0"
APPLICATION_NAME = "navigator_z7020_npu_runner"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def remove_workspace(workspace: Path) -> None:
    expected = DEFAULT_WORKSPACE.resolve()
    if workspace.resolve() != expected:
        raise ValueError("refusing to clean any workspace except the default build path")
    if workspace.exists():
        shutil.rmtree(workspace)


def component_import(app: object, directory: Path, files: list[str]) -> None:
    for name in files:
        require_file(directory / name)
    app.import_files(from_loc=str(directory), files=files, dest_dir_in_cmp="src")


def main() -> int:
    xsa = Path(os.environ.get("NPU_VITIS_XSA", str(DEFAULT_XSA))).resolve()
    workspace = Path(os.environ.get("NPU_VITIS_WORKSPACE",
                                    str(DEFAULT_WORKSPACE))).resolve()
    clean = os.environ.get("NPU_VITIS_CLEAN", "0") == "1"
    if clean:
        remove_workspace(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError(
            f"workspace already exists: {workspace}; set NPU_VITIS_CLEAN=1 "
            "to replace the default ignored workspace"
        )
    require_file(xsa)
    require_file(PAYLOAD_DIR / "navigator_npu_task_payload.c")
    require_file(PAYLOAD_DIR / "navigator_npu_task_payload.h")

    workspace.mkdir(parents=True, exist_ok=True)
    client = vitis.create_client()
    try:
        if os.environ.get("NPU_VITIS_DEBUG", "0") == "1":
            client.log_level("DEBUG")
        client.set_workspace(str(workspace))
        # Follow AMD's Zynq standalone example: create the hardware platform
        # first, then add the A9 domain.  This also keeps the project shape
        # aligned with the GUI flow used when diagnosing a Vitis installation.
        platform = client.create_platform_component(
            name=PLATFORM_NAME,
            hw_design=str(xsa),
        )
        platform.add_domain(
            name=DOMAIN_NAME,
            cpu="ps7_cortexa9_0",
            os="standalone",
        )
        platform.build()
        xpfm = client.find_platform_in_repos(PLATFORM_NAME)
        app = client.create_app_component(
            name=APPLICATION_NAME,
            platform=xpfm,
            domain=DOMAIN_NAME,
            template="empty_application",
        )
        component_import(app, ROOT / "software" / "src", ["npu_driver.c"])
        component_import(app, ROOT / "software" / "include",
                         ["npu_driver.h"])
        component_import(app, VITIS_SOURCES,
                         ["navigator_npu_runner.c", "npu_vitis_platform.c",
                          "npu_vitis_platform.h"])
        component_import(app, PAYLOAD_DIR,
                         ["navigator_npu_task_payload.c",
                          "navigator_npu_task_payload.h"])
        app.build()

        elf_candidates = sorted(
            path for path in (workspace / APPLICATION_NAME).rglob("*.elf")
            if path.is_file()
        )
        if len(elf_candidates) != 1:
            raise RuntimeError(f"expected one ELF, found: {elf_candidates}")
        result = {
            "xsa": str(xsa),
            "workspace": str(workspace),
            "platform": PLATFORM_NAME,
            "domain": DOMAIN_NAME,
            "application": APPLICATION_NAME,
            "elf": str(elf_candidates[0]),
        }
        output = workspace / "npu_vitis_build.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("NPU_VITIS_BUILD PASS " + json.dumps(result))
    finally:
        vitis.dispose()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"NPU_VITIS_BUILD FAIL: {error}", file=sys.stderr)
        raise
