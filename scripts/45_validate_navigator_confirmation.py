#!/usr/bin/env python3
"""Validate a physical-board record before exporting a JTAG-only candidate.

The V3.7 seller claim is a reference profile, not a verified PCB revision.
This script does not certify DDR timing or authorize flash programming.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path


REFERENCE_PROFILE = "alientek_navigator_v3.7_wm8960"
DDR_MARKING = "NT5CC256M16EP-EK"


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def validate(values: object) -> str:
    if not isinstance(values, dict):
        raise ValueError("board confirmation must be a JSON object")
    _nonempty(values.get("confirmed_by"), "confirmed_by")
    confirmed_date = _nonempty(values.get("confirmed_date"), "confirmed_date")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", confirmed_date):
        raise ValueError("confirmed_date must be YYYY-MM-DD")
    try:
        date.fromisoformat(confirmed_date)
    except ValueError as exc:
        raise ValueError("confirmed_date must be YYYY-MM-DD") from exc

    marking = _nonempty(values.get("fpga_marking"), "fpga_marking")
    normalized = re.sub(r"[^A-Z0-9]", "", marking.upper())
    if not normalized.startswith("XC7Z020CLG400"):
        raise ValueError("chip-top marking must identify XC7Z020 in CLG400")
    if values.get("fpga_marking_basis") != "chip_top":
        raise ValueError("fpga_marking_basis must be chip_top, not a PCB legend")
    if str(values.get("fpga_speed_grade")) != "2":
        raise ValueError("the implemented candidate requires reported speed grade -2")
    speed_grade_basis = values.get("fpga_speed_grade_basis")
    if speed_grade_basis not in {
        "amd_device_lookup", "original_package_label", "seller_statement"
    }:
        raise ValueError("speed grade needs AMD device lookup, original package label, or seller_statement")

    markings = values.get("ddr_markings")
    if not isinstance(markings, list) or markings != [DDR_MARKING, DDR_MARKING]:
        raise ValueError("both physical DDR markings must be NT5CC256M16EP-EK")
    if values.get("boot_mode_for_first_test") != "JTAG":
        raise ValueError("first test must use JTAG boot mode")
    if values.get("jtag_only_prototype") is not True:
        raise ValueError("first export must be acknowledged as JTAG-only prototype")
    if values.get("no_flash_write") is not True:
        raise ValueError("flash writes must remain disabled during first test")
    if values.get("reference_profile") != REFERENCE_PROFILE:
        raise ValueError("reference_profile must name the V3.7 WM8960 material")

    origin = values.get("board_origin")
    if origin == "third_party_clone":
        if values.get("base_board_revision") not in (None, ""):
            raise ValueError("unmarked clone must not be recorded as an official PCB revision")
        if values.get("base_board_revision_marking") != "absent":
            raise ValueError("clone must explicitly record the absent baseboard revision marking")
        if values.get("reference_basis") != "seller_statement":
            raise ValueError("clone V3.7 compatibility must be attributed to seller_statement")
    elif origin == "official_alientek":
        if speed_grade_basis == "seller_statement":
            raise ValueError("seller_statement speed grade is limited to clone JTAG-only prototypes")
        _nonempty(values.get("core_board_revision"), "core_board_revision")
        _nonempty(values.get("base_board_revision"), "base_board_revision")
        if values.get("reference_basis") != "official_material":
            raise ValueError("official board must identify its official reference material")
    else:
        raise ValueError("board_origin must be third_party_clone or official_alientek")
    return (f"board_confirmation: PASS origin={origin} profile={REFERENCE_PROFILE} "
            f"speed_grade_basis={speed_grade_basis} scope=JTAG-only; not board acceptance")


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"duplicate JSON key: {key}")
        values[key] = value
    return values


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: 45_validate_navigator_confirmation.py board_confirmation.local.json", file=sys.stderr)
        return 2
    try:
        values = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"),
                            object_pairs_hook=_no_duplicate_keys)
        print(validate(values))
        return 0
    except (OSError, ValueError) as exc:
        print(f"board_confirmation: FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
