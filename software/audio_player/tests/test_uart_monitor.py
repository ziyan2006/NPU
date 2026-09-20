#!/usr/bin/env python3
"""Regression tests for the FullStem UART acceptance monitor."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MONITOR_PATH = ROOT / "scripts" / "104_monitor_audio_uart.py"


def load_monitor():
    spec = importlib.util.spec_from_file_location("audio_uart_monitor", MONITOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {MONITOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def telemetry_line(second: int, **overrides: int | str) -> str:
    values: dict[str, int | str] = {
        "sec": second,
        "state": "PLAY",
        "stem": 0 if second < 600 or second >= 1200 else 1,
        "ramp": 0,
        "played": second * 44100,
        "fifo": 4096,
        "fifo_min": 2048,
        "uf": 0,
        "of": 0,
        "dec_err": 0,
        "npu_err": 0,
        "codec_err": 0,
        "deadline_miss": 0,
        "blk_us_avg": 80000,
        "blk_us_max": 90000,
        "decfe_us_avg": 22000,
        "decfe_us_max": 25000,
        "npu_us_avg": 36000,
        "npu_us_max": 40000,
        "sink_us_avg": 18000,
        "sink_us_max": 20000,
    }
    values.update(overrides)
    return "[AUDIO] " + " ".join(f"{key}={value}" for key, value in values.items())


def success_log(seconds: int = 1800) -> list[str]:
    return [telemetry_line(second) for second in range(1, seconds + 1)]


class AudioUartMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.monitor = load_monitor()

    def evaluate(self, lines: list[str], seconds: int = 1800) -> dict:
        return self.monitor.evaluate_lines(
            lines,
            required_seconds=seconds,
            required_stem_transitions=2,
        )

    def test_accepts_exactly_1800_clean_play_seconds(self) -> None:
        summary = self.evaluate(success_log())
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["play_seconds"], 1800)
        self.assertEqual(summary["stem_transitions"], 2)
        self.assertEqual(summary["first_play_second"], 1)
        self.assertEqual(summary["last_play_second"], 1800)
        self.assertEqual(summary["failures"], [])
        self.assertEqual(summary["maxima"]["blk_us_max"], 90000)
        self.assertEqual(summary["maxima"]["npu_us_max"], 40000)

    def test_rejects_heartbeat_gap(self) -> None:
        lines = success_log()
        lines[900] = telemetry_line(902)
        summary = self.evaluate(lines)
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("heartbeat gap" in item for item in summary["failures"]))

    def test_rejects_each_error_counter(self) -> None:
        for field in ("uf", "of", "dec_err", "npu_err", "codec_err"):
            with self.subTest(field=field):
                lines = success_log()
                lines[400] = telemetry_line(401, **{field: 1})
                summary = self.evaluate(lines)
                self.assertEqual(summary["status"], "FAIL")
                self.assertTrue(any(field in item for item in summary["failures"]))

    def test_rejects_deadline_counter_or_over_budget_measurement(self) -> None:
        lines = success_log()
        lines[99] = telemetry_line(100, deadline_miss=1)
        self.assertEqual(self.evaluate(lines)["status"], "FAIL")

        lines = success_log()
        lines[99] = telemetry_line(100, blk_us_max=92881)
        summary = self.evaluate(lines)
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("deadline" in item for item in summary["failures"]))

    def test_rejects_missing_key_transitions(self) -> None:
        lines = [telemetry_line(second, stem=0) for second in range(1, 1801)]
        summary = self.evaluate(lines)
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("STEM target transitions" in item
                            for item in summary["failures"]))

    def test_rejects_early_eof(self) -> None:
        lines = success_log(900)
        lines.append(telemetry_line(901, state="DONE"))
        summary = self.evaluate(lines)
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("left PLAY" in item for item in summary["failures"]))

    def test_ignores_boot_chatter_but_rejects_malformed_audio_lines(self) -> None:
        lines = ["BOOT", "CODEC_READY", "FULL_STEM"] + success_log()
        self.assertEqual(self.evaluate(lines)["status"], "PASS")

        lines = success_log()
        lines.insert(20, "[AUDIO] sec=not-a-number state=PLAY")
        summary = self.evaluate(lines)
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("malformed telemetry" in item
                            for item in summary["failures"]))

    def test_requires_stage_timing_fields(self) -> None:
        lines = success_log()
        lines[0] = lines[0].replace(" decfe_us_avg=22000", "")
        summary = self.evaluate(lines)
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("decfe_us_avg" in item for item in summary["failures"]))


if __name__ == "__main__":
    unittest.main()
