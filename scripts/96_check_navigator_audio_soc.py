#!/usr/bin/env python3
"""Audit the Navigator Z7020 NPU plus WM8960 implementation artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "hardware" / "build" / "navigator_z7020_audio"
BD = (
    PROJECT / "navigator_audio_soc.srcs" / "sources_1" / "bd"
    / "audio_soc" / "audio_soc.bd"
)
REPORT = (
    ROOT / "hardware" / "reports" / "vivado_2026_1"
    / "navigator_z7020_audio_impl"
)
XDC = (
    ROOT / "hardware" / "boards" / "alientek_navigator_z7020"
    / "audio_out_v37.xdc"
)

EXPECTED_PINS = {
    "sys_clk": "U18",
    "key_n": "L14",
    "aud_scl": "E18",
    "aud_sda": "F17",
    "aud_mclk": "E19",
    "aud_bclk": "M18",
    "aud_dac_lrclk": "G18",
    "aud_dacdat": "G17",
}


def read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    xdc = read(XDC)
    for port, pin in EXPECTED_PINS.items():
        pattern = rf"PACKAGE_PIN\s+{pin}\b.*get_ports\s+{port}\b"
        if not re.search(pattern, xdc):
            raise AssertionError(f"missing exact pin constraint: {port}={pin}")
    if not re.search(r"create_clock\s+-period\s+20\.000.*get_ports\s+sys_clk", xdc):
        raise AssertionError("50 MHz sys_clk constraint is missing")

    design = json.loads(read(BD))["design"]
    if design["design_info"]["device"] != "xc7z020clg400-2":
        raise AssertionError("wrong FPGA part")
    components = design["components"]
    if "stem_npu_0" not in components or "audio_out_0" not in components:
        raise AssertionError("NPU/audio IP is missing from block design")
    ps = components.get("processing_system7_0", {}).get("parameters", {})
    expected_sd = {
        "PCW_SD0_PERIPHERAL_ENABLE": "1",
        "PCW_SD0_SD0_IO": "MIO 40 .. 45",
        "PCW_SD0_GRP_CD_ENABLE": "1",
        "PCW_SD0_GRP_CD_IO": "MIO 10",
    }
    for key, expected in expected_sd.items():
        actual = ps.get(key, {}).get("value")
        if actual != expected:
            raise AssertionError(
                f"audio runtime SD0 mismatch: {key}={actual!r}, "
                f"expected {expected!r}"
            )
    clock_component = components.get("audio_clock", {})
    clock_xci_path = clock_component.get("xci_path")
    if not clock_xci_path:
        raise AssertionError("audio clock XCI reference is missing")
    clock_xci = json.loads(read(BD.parent / Path(clock_xci_path.replace("\\", "/"))))
    clock = clock_xci["ip_inst"]["parameters"]["component_parameters"]
    expected_clock = {
        "MMCM_DIVCLK_DIVIDE": "3",
        "MMCM_CLKFBOUT_MULT_F": "63.250",
        "MMCM_CLKOUT0_DIVIDE_F": "93.375",
    }
    for key, expected in expected_clock.items():
        values = clock.get(key, [])
        actual = values[0].get("value") if values else None
        if actual != expected:
            raise AssertionError(f"audio clock {key}={actual!r}, expected {expected}")

    addresses = read(REPORT / "address_map.txt")
    for owner, address in (("stem_npu_0", "0x43C00000"),
                           ("audio_out_0", "0x43C10000")):
        if owner not in addresses or address not in addresses:
            raise AssertionError(f"missing address mapping {owner} at {address}")

    summary = dict(
        line.split("=", 1)
        for line in read(REPORT / "implementation_summary.txt").splitlines()
        if "=" in line
    )
    if summary.get("part") != "xc7z020clg400-2":
        raise AssertionError("implementation summary has wrong part")
    if not summary.get("status", "").endswith("Complete!"):
        raise AssertionError("implementation is incomplete")
    for name in ("wns_ns", "whs_ns"):
        if float(summary[name]) < 0:
            raise AssertionError(f"negative {name}: {summary[name]}")
    for name in ("failing_setup_paths", "failing_hold_paths",
                 "unrouted_nets", "critical_drc"):
        if int(summary[name]):
            raise AssertionError(f"{name} is nonzero: {summary[name]}")
    if int(summary.get("iobuf_count", "0")) != 2:
        raise AssertionError(
            f"WM8960 I2C requires exactly 2 IOBUFs, got "
            f"{summary.get('iobuf_count', 'missing')}"
        )

    utilization = read(REPORT / "utilization_hierarchical.rpt")
    audio_rows = []
    for line in utilization.splitlines():
        fields = [field.strip() for field in line.split("|")[1:-1]]
        if fields and fields[0] == "audio_out_0":
            audio_rows.append(fields)
    if len(audio_rows) != 1 or len(audio_rows[0]) < 10:
        raise AssertionError("audio hierarchy is absent from utilization")
    audio_ramb36 = int(audio_rows[0][7])
    if audio_ramb36 != 16:
        raise AssertionError(
            f"8192x64 audio FIFO requires 16 RAMB36, got {audio_ramb36}"
        )
    cdc = read(REPORT / "cdc.rpt")
    cdc_rows = [
        line for line in cdc.splitlines()
        if re.match(r"^\s*\d+\s+CDC-", line)
    ]
    missing_async_reg = [line for line in cdc_rows if "missing ASYNC_REG" in line]
    if missing_async_reg:
        raise AssertionError(
            f"CDC synchronizers missing ASYNC_REG: {len(missing_async_reg)}"
        )
    comb_cdc = [line for line in cdc_rows if "CDC-10" in line]
    reset_cdc_source = (
        "audio_soc_i/rst_ps7_0_100M/U0/"
        "ACTIVE_LOW_PR_OUT_DFF[0].FDRE_PER_N/C"
    )
    reset_cdc_destination = (
        "audio_soc_i/audio_out_0/inst/audio_reset_sync_q_reg[0]/CLR"
    )
    unexpected_comb_cdc = [
        line for line in comb_cdc
        if reset_cdc_source not in line
        or not line.rstrip().endswith(reset_cdc_destination)
    ]
    if len(comb_cdc) > 1 or unexpected_comb_cdc:
        raise AssertionError(
            "unexpected combinational CDC paths: "
            f"total={len(comb_cdc)}, unexpected={len(unexpected_comb_cdc)}"
        )
    route = read(REPORT / "route_status.rpt")
    if not re.search(r"# of nets with routing errors\.*\s*:\s*0\s*:", route):
        raise AssertionError("routing errors found")
    print(
        "navigator_audio_soc: PASS "
        f"(WNS={summary['wns_ns']} ns, WHS={summary['whs_ns']} ns, "
        "NPU=0x43C00000, AUDIO=0x43C10000)"
    )


if __name__ == "__main__":
    main()
