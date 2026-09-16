#!/usr/bin/env python3
"""Validate the provisional Navigator Z7020 profile and offline evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "hardware" / "boards" / "alientek_navigator_z7020"
IMPL = (
    ROOT
    / "hardware"
    / "reports"
    / "vivado_2026_1"
    / "navigator_z7020_candidate_impl"
)


def read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"required file is missing: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    manifest = json.loads(read(BOARD / "board_manifest.json"))
    assert manifest["status"] == "provisional_pending_silkscreen_confirmation"
    assert manifest["fpga_part"] == "xc7z020clg400-2"
    assert manifest["ddr"]["total_bytes"] == 1024**3
    assert manifest["ddr"]["bus_width_bits"] == 32
    assert manifest["ddr"]["trace_delays_confirmed"] is False
    assert manifest["uart"] == {
        "instance": 0,
        "rx_mio": 14,
        "tx_mio": 15,
        "baud": 115200,
    }

    profile = read(BOARD / "minimal_jtag_profile.tcl")
    required = (
        "xc7z020clg400-2",
        "MT41J256M16 RE-125",
        "MIO 14 .. 15",
        "0x3FFFFFFF",
        "PCW_USE_M_AXI_GP0 1",
        "PCW_USE_S_AXI_HP0 1",
        "PCW_S_AXI_HP0_DATA_WIDTH 64",
        "PCW_FPGA0_PERIPHERAL_FREQMHZ 100.000000",
    )
    for token in required:
        if token not in profile:
            raise AssertionError(f"board profile lost required token: {token}")

    summary_text = read(IMPL / "implementation_summary.txt")
    summary = dict(
        line.split("=", 1)
        for line in summary_text.splitlines()
        if "=" in line
    )
    if summary.get("part") != "xc7z020clg400-2":
        raise AssertionError(f"wrong implemented part: {summary.get('part')}")
    if not summary.get("status", "").endswith("Complete!"):
        raise AssertionError(f"implementation incomplete: {summary.get('status')}")
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
        raise AssertionError("target timing report does not say MET")
    route = read(IMPL / "route_status.rpt")
    if not re.search(r"# of nets with routing errors\.*\s*:\s*0\s*:", route):
        raise AssertionError("target route report contains routing errors")
    if "All paths are Safely Timed." not in read(IMPL / "cdc.rpt"):
        raise AssertionError("target CDC report is not clean")

    utilization = read(IMPL / "utilization_hierarchical.rpt")
    top = re.search(
        r"\| npu_soc_wrapper\s+\|\s+\(top\)\s+\|\s+(\d+)\s+\|"
        r"\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|"
        r"\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|",
        utilization,
    )
    if top is None:
        raise AssertionError("cannot parse target post-route utilization")
    total_luts, _, _, _, ffs, bram36, bram18, dsps = map(int, top.groups())
    expected_resources = (25436, 25293, 61, 0, 72)
    if (total_luts, ffs, bram36, bram18, dsps) != expected_resources:
        raise AssertionError("target utilization changed without evidence update")

    local = BOARD / "board_confirmation.local.json"
    confirmation = "pending"
    if local.is_file():
        values = json.loads(read(local))
        required_confirmation = (
            "fpga_marking",
            "core_board_revision",
            "base_board_revision",
            "ddr_marking",
        )
        missing = [name for name in required_confirmation if not values.get(name)]
        if not missing and "XC7Z020CLG400-2" not in values["fpga_marking"].upper():
            raise AssertionError("confirmed FPGA marking is not XC7Z020CLG400-2")
        confirmation = "complete" if not missing else f"incomplete:{','.join(missing)}"

    print(
        "navigator_z7020_offline_gate: PASS "
        f"(part={summary['part']}, WNS={summary['wns_ns']} ns, "
        f"WHS={summary['whs_ns']} ns, LUT={total_luts}, FF={ffs}, "
        f"BRAM36={bram36}, DSP={dsps}, board_confirmation={confirmation})"
    )
    print(
        "hardware gate remains CLOSED until silkscreen/revision and vendor PS7 "
        "preset are confirmed"
    )


if __name__ == "__main__":
    main()
