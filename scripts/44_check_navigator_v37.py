#!/usr/bin/env python3
"""Check the vendor-V3.7-derived Navigator Z7020 offline build."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "hardware" / "boards" / "alientek_navigator_z7020"
VENDOR_XCI = (
    BOARD
    / "vendor_reference_local"
    / "ps7"
    / "system_processing_system7_0_0.xci"
)
VENDOR_XCI_SHA256 = "591462696860ad9e4bdc03c9f5b58e95097c329cd2d308b8e806bb9bcedc7f68"
PROJECT = ROOT / "hardware" / "build" / "navigator_z7020_vendor_v37"
BD = (
    PROJECT
    / "reference_zynq_soc.srcs"
    / "sources_1"
    / "bd"
    / "npu_soc"
    / "npu_soc.bd"
)
GENERATED_PS7_XCI = (
    PROJECT
    / "reference_zynq_soc.srcs"
    / "sources_1"
    / "bd"
    / "npu_soc"
    / "ip"
    / "npu_soc_processing_system7_0_0"
    / "npu_soc_processing_system7_0_0.xci"
)
REPORT = ROOT / "hardware" / "reports" / "vivado_2026_1" / "navigator_z7020_vendor_v37_impl"

EXPECTED_PS = {
    "PCW_CRYSTAL_PERIPHERAL_FREQMHZ": "33.333333",
    "PCW_UIPARAM_DDR_MEMORY_TYPE": "DDR 3 (Low Voltage)",
    "PCW_UIPARAM_DDR_PARTNO": "MT41K256M16 RE-125",
    "PCW_UIPARAM_DDR_BUS_WIDTH": "32 Bit",
    "PCW_UIPARAM_DDR_FREQ_MHZ": "533.333333",
    "PCW_DDR_RAM_HIGHADDR": "0x3FFFFFFF",
    "PCW_UART0_PERIPHERAL_ENABLE": "1",
    "PCW_UART0_UART0_IO": "MIO 14 .. 15",
    "PCW_USE_M_AXI_GP0": "1",
    "PCW_USE_S_AXI_HP0": "1",
    "PCW_S_AXI_HP0_DATA_WIDTH": "64",
    "PCW_FPGA0_PERIPHERAL_FREQMHZ": "100.000000",
    "PCW_QSPI_PERIPHERAL_ENABLE": "0",
    "PCW_SD0_PERIPHERAL_ENABLE": "0",
    "PCW_ENET0_PERIPHERAL_ENABLE": "0",
    "PCW_ENET1_PERIPHERAL_ENABLE": "0",
    **{f"PCW_UIPARAM_DDR_BOARD_DELAY{i}": "0.25" for i in range(4)},
}


def read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    manifest = json.loads(read(BOARD / "board_manifest.json"))
    if manifest["status"] != "provisional_pending_silkscreen_confirmation":
        raise AssertionError("board status must remain provisional before physical checks")
    if manifest["board_origin"] != "third_party_clone":
        raise AssertionError("board must be recorded as a third-party clone")
    if manifest["reference_basis"] != "seller statement, not a verified official PCB revision":
        raise AssertionError("V3.7 reference basis has changed")
    if manifest["revision_sensitive"]["base_board_revision"] is not None:
        raise AssertionError("unmarked clone cannot have a confirmed official PCB revision")
    ddr_manifest = manifest["ddr"]
    if ddr_manifest["physical_marking_confirmed"]:
        if ddr_manifest["user_reported_device_count"] != 2:
            raise AssertionError("two DDR markings must be confirmed together")
        if ddr_manifest["user_reported_marking"] != "NT5CC256M16EP-EK":
            raise AssertionError("user-confirmed DDR marking changed")
    if ddr_manifest["vivado_compatible_part"] != EXPECTED_PS["PCW_UIPARAM_DDR_PARTNO"]:
        raise AssertionError("board manifest and PS7 DDR catalog part disagree")
    if ddr_manifest["vendor_v37_ps7_reference_sha256"] != VENDOR_XCI_SHA256:
        raise AssertionError("board manifest vendor PS7 digest disagrees")
    if not VENDOR_XCI.is_file():
        raise AssertionError("local vendor V3.7 PS7 XCI is missing")
    digest = hashlib.sha256(VENDOR_XCI.read_bytes()).hexdigest()
    if digest != VENDOR_XCI_SHA256:
        raise AssertionError("vendor PS7 XCI SHA-256 changed; review source revision")

    namespace = "{http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009}"
    vendor = {
        node.attrib[f"{namespace}referenceId"].removeprefix("PARAM_VALUE."): node.text
        for node in ET.parse(VENDOR_XCI).iter()
        if node.tag == f"{namespace}configurableElementValue"
    }
    for key in (
        "PCW_CRYSTAL_PERIPHERAL_FREQMHZ",
        "PCW_UIPARAM_DDR_MEMORY_TYPE",
        "PCW_UIPARAM_DDR_PARTNO",
        "PCW_UIPARAM_DDR_BUS_WIDTH",
        "PCW_UIPARAM_DDR_FREQ_MHZ",
        "PCW_UART0_PERIPHERAL_ENABLE",
        "PCW_UART0_UART0_IO",
    ):
        if vendor.get(key) != EXPECTED_PS[key]:
            raise AssertionError(f"vendor reference changed: {key}")

    design = json.loads(read(BD))["design"]
    if design["design_info"]["device"] != "xc7z020clg400-2":
        raise AssertionError("wrong implemented FPGA part")
    ps = design["components"]["processing_system7_0"]["parameters"]
    for key, expected in EXPECTED_PS.items():
        actual = ps.get(key, {}).get("value")
        if actual != expected:
            raise AssertionError(f"built PS7 mismatch: {key}={actual!r}, expected {expected!r}")
    ddr_keys = [key for key in vendor if key.startswith("PCW_UIPARAM_DDR_")]
    if len(ddr_keys) != 72:
        raise AssertionError("unexpected vendor DDR field count")
    generated_ps = json.loads(read(GENERATED_PS7_XCI))["ip_inst"]["parameters"][
        "component_parameters"
    ]
    for key in ddr_keys:
        actual = generated_ps.get(key, [{}])[0].get("value")
        if actual != vendor[key]:
            raise AssertionError(f"generated PS7 DDR field differs from V3.7 reference: {key}")
    recorded_ddr_keys = [key for key in ddr_keys if key in ps]
    if len(recorded_ddr_keys) < 58:
        raise AssertionError("too few vendor DDR fields are recorded in the NPU block design")
    for key in recorded_ddr_keys:
        if ps[key]["value"] != vendor[key]:
            raise AssertionError(f"DDR field differs from V3.7 reference: {key}")

    summary = dict(
        line.split("=", 1)
        for line in read(REPORT / "implementation_summary.txt").splitlines()
        if "=" in line
    )
    if summary.get("part") != "xc7z020clg400-2":
        raise AssertionError("implementation summary has wrong part")
    if not summary.get("status", "").endswith("Complete!"):
        raise AssertionError("vendor-V3.7 implementation is incomplete")
    for key in ("wns_ns", "whs_ns"):
        if float(summary[key]) < 0.0:
            raise AssertionError(f"negative {key}: {summary[key]}")
    for key in ("failing_setup_paths", "failing_hold_paths", "unrouted_nets", "critical_drc"):
        if int(summary[key]) != 0:
            raise AssertionError(f"{key} is nonzero: {summary[key]}")

    timing = read(REPORT / "timing_summary.rpt")
    if "All user specified timing constraints are met." not in timing:
        raise AssertionError("100 MHz timing constraints are not met")
    for check in ("no_clock", "unconstrained_internal_endpoints"):
        if f"checking {check} (0)" not in timing:
            raise AssertionError(f"timing report has unresolved {check} paths")
    route = read(REPORT / "route_status.rpt")
    if not re.search(r"# of nets with routing errors\.*\s*:\s*0\s*:", route):
        raise AssertionError("routing errors found")
    if "All paths are Safely Timed." not in read(REPORT / "cdc.rpt"):
        raise AssertionError("CDC report is not clean")

    print(
        "navigator_v37_preboard: PASS "
        f"(WNS={summary['wns_ns']} ns, WHS={summary['whs_ns']} ns, "
        f"generated DDR fields={len(ddr_keys)}/{len(ddr_keys)}, "
        f"BD fields={len(recorded_ddr_keys)}/{len(ddr_keys)}, MIO matched)"
    )
    print("Seller-reported -2 and offline checks do not establish board acceptance; JTAG DDR/NPU tests remain pending")


if __name__ == "__main__":
    main()
