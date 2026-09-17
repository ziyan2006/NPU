#!/usr/bin/env python3
"""Regression tests for the clone/official JTAG export confirmation gate."""

from __future__ import annotations

import copy
import runpy
import unittest
from pathlib import Path


validate = runpy.run_path(
    str(Path(__file__).with_name("45_validate_navigator_confirmation.py"))
)["validate"]


def clone_confirmation() -> dict[str, object]:
    return {
        "confirmed_by": "board owner",
        "confirmed_date": "2026-09-16",
        "board_origin": "third_party_clone",
        "reference_profile": "alientek_navigator_v3.7_wm8960",
        "reference_basis": "seller_statement",
        "fpga_marking": "XC7Z020 CLG400ABX2317",
        "fpga_marking_basis": "chip_top",
        "fpga_speed_grade": "2",
        "fpga_speed_grade_basis": "amd_device_lookup",
        "core_board_revision": None,
        "base_board_revision": None,
        "base_board_revision_marking": "absent",
        "ddr_markings": ["NT5CC256M16EP-EK", "NT5CC256M16EP-EK"],
        "boot_mode_for_first_test": "JTAG",
        "jtag_only_prototype": True,
        "no_flash_write": True,
    }


class ConfirmationTests(unittest.TestCase):
    def test_unmarked_clone_can_be_recorded_without_invented_revision(self) -> None:
        self.assertIn("PASS", validate(clone_confirmation()))

    def test_seller_v37_claim_cannot_become_pcb_revision(self) -> None:
        values = clone_confirmation()
        values["base_board_revision"] = "3.7"
        with self.assertRaisesRegex(ValueError, "official PCB revision"):
            validate(values)

    def test_board_legend_does_not_confirm_fpga_speed_grade(self) -> None:
        values = clone_confirmation()
        values["fpga_marking_basis"] = "pcb_legend"
        with self.assertRaisesRegex(ValueError, "chip_top"):
            validate(values)

    def test_package_line_does_not_encode_speed_grade(self) -> None:
        values = clone_confirmation()
        values["fpga_speed_grade_basis"] = "chip_top_package_line"
        with self.assertRaisesRegex(ValueError, "AMD device lookup"):
            validate(values)

    def test_seller_speed_grade_only_releases_jtag_only_candidate(self) -> None:
        values = clone_confirmation()
        values["fpga_speed_grade_basis"] = "seller_statement"
        result = validate(values)
        self.assertIn("speed_grade_basis=seller_statement", result)
        self.assertIn("not board acceptance", result)
        values["jtag_only_prototype"] = False
        with self.assertRaisesRegex(ValueError, "JTAG-only"):
            validate(values)

    def test_seller_speed_grade_cannot_release_flash_write(self) -> None:
        values = clone_confirmation()
        values["fpga_speed_grade_basis"] = "seller_statement"
        values["no_flash_write"] = False
        with self.assertRaisesRegex(ValueError, "flash writes"):
            validate(values)

    def test_clone_needs_explicit_seller_reference(self) -> None:
        values = clone_confirmation()
        values["reference_basis"] = None
        with self.assertRaisesRegex(ValueError, "seller_statement"):
            validate(values)

    def test_both_ddr_markings_are_required(self) -> None:
        values = clone_confirmation()
        values["ddr_markings"] = ["NT5CC256M16EP-EK"]
        with self.assertRaisesRegex(ValueError, "both physical DDR"):
            validate(values)

    def test_no_flash_write_is_required(self) -> None:
        values = copy.deepcopy(clone_confirmation())
        values["no_flash_write"] = False
        with self.assertRaisesRegex(ValueError, "flash writes"):
            validate(values)


if __name__ == "__main__":
    unittest.main()
