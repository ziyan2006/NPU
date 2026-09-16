#!/usr/bin/env python3
"""Fail-fast pre-board evidence gate for the XC7Z020 reference integration."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "hardware" / "reports" / "vivado_2026_1" / "reference_zynq_soc_impl"


def read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"required evidence is missing: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8", errors="replace")


def run_python(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-rtl",
        action="store_true",
        help="also rerun the 1,869-command XSim bit-exact task (about 90 seconds)",
    )
    args = parser.parse_args()

    summary = dict(
        line.split("=", 1)
        for line in read(IMPL / "implementation_summary.txt").splitlines()
        if "=" in line
    )
    if summary.get("part") != "xc7z020clg400-1":
        raise AssertionError(f"unexpected reference part: {summary.get('part')}")
    if not summary.get("status", "").endswith("Complete!"):
        raise AssertionError(f"implementation is incomplete: {summary.get('status')}")
    for key in ("wns_ns", "whs_ns"):
        if float(summary[key]) < 0.0:
            raise AssertionError(f"negative {key}: {summary[key]}")
    for key in (
        "failing_setup_paths",
        "failing_hold_paths",
        "unrouted_nets",
        "critical_drc",
    ):
        if int(summary[key]) != 0:
            raise AssertionError(f"{key} is not zero: {summary[key]}")

    timing = read(IMPL / "timing_summary.rpt")
    if "All user specified timing constraints are met." not in timing:
        raise AssertionError("Vivado timing summary does not report MET")
    route = read(IMPL / "route_status.rpt")
    if not re.search(r"# of nets with routing errors\.*\s*:\s*0\s*:", route):
        raise AssertionError("route report contains routing errors")
    if "All paths are Safely Timed." not in read(IMPL / "cdc.rpt"):
        raise AssertionError("CDC report is not clean")

    utilization = read(IMPL / "utilization_hierarchical.rpt")
    top = re.search(
        r"\| npu_soc_wrapper\s+\|\s+\(top\)\s+\|\s+(\d+)\s+\|"
        r"\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|"
        r"\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|",
        utilization,
    )
    if top is None:
        raise AssertionError("cannot parse post-route top utilization")
    total_luts, _, _, _, ffs, bram36, bram18, dsps = map(int, top.groups())
    if (total_luts, ffs, bram36, bram18, dsps) != (25634, 25293, 61, 0, 72):
        raise AssertionError("post-route utilization changed without evidence update")

    run_python("_test_npu_driver.py")
    if args.full_rtl:
        run_python("_test_npu_top_task.py")

    print(
        "preboard_gate: PASS "
        f"(WNS={summary['wns_ns']} ns, WHS={summary['whs_ns']} ns, "
        f"LUT={total_luts}, FF={ffs}, BRAM36={bram36}, DSP={dsps})"
    )
    print("board-only blockers: PS/DDR/MIO preset, OS DMA adapter, bitstream, hardware test")


if __name__ == "__main__":
    main()
