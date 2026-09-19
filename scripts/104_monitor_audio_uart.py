#!/usr/bin/env python3
"""Capture and qualify FullStem UART telemetry from the Navigator board.

The monitor is intentionally independent of Vivado/Vitis and imports pyserial
only for a live run.  ``evaluate_lines`` is the deterministic parser used by
the host regression tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


AUDIO_PREFIX = "[AUDIO] "
BLOCK_DEADLINE_US = 92_880
STAGE_TIMING_FIELDS = (
    "decfe_us_avg",
    "decfe_us_max",
    "npu_us_avg",
    "npu_us_max",
    "sink_us_avg",
    "sink_us_max",
)
REQUIRED_FIELDS = (
    "sec",
    "state",
    "stem",
    "ramp",
    "fifo",
    "fifo_min",
    "uf",
    "of",
    "dec_err",
    "npu_err",
    "codec_err",
    "deadline_miss",
    "blk_us_avg",
    "blk_us_max",
) + STAGE_TIMING_FIELDS
INTEGER_FIELDS = tuple(field for field in REQUIRED_FIELDS if field != "state")
ZERO_COUNTER_FIELDS = ("uf", "of", "dec_err", "npu_err", "codec_err", "deadline_miss")
MAXIMUM_FIELDS = (
    "fifo",
    "fifo_min",
    "blk_us_avg",
    "blk_us_max",
) + STAGE_TIMING_FIELDS


class TelemetryError(ValueError):
    """Raised when an AUDIO-prefixed line does not satisfy the contract."""


def parse_telemetry_line(line: str) -> dict[str, int | str] | None:
    """Parse one player telemetry line; return ``None`` for ordinary chatter."""
    stripped = line.strip()
    if not stripped.startswith(AUDIO_PREFIX):
        return None
    fields: dict[str, str] = {}
    for token in stripped[len(AUDIO_PREFIX):].split():
        if "=" not in token:
            raise TelemetryError(f"token has no '=': {token!r}")
        key, value = token.split("=", 1)
        if not key or not value:
            raise TelemetryError(f"empty key/value in token: {token!r}")
        if key in fields:
            raise TelemetryError(f"duplicate field: {key}")
        fields[key] = value
    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if missing:
        raise TelemetryError("missing fields: " + ", ".join(missing))
    parsed: dict[str, int | str] = {"state": fields["state"]}
    for field in INTEGER_FIELDS:
        try:
            value = int(fields[field], 0)
        except ValueError as exc:
            raise TelemetryError(f"{field} is not an integer: {fields[field]!r}") from exc
        if value < 0:
            raise TelemetryError(f"{field} is negative: {value}")
        parsed[field] = value
    if parsed["stem"] not in (0, 1) or parsed["ramp"] not in (0, 1):
        raise TelemetryError("stem and ramp must be 0 or 1")
    return parsed


class AudioAcceptance:
    """Stateful acceptance policy for one uninterrupted FullStem run."""

    def __init__(self, required_seconds: int = 1800,
                 required_stem_transitions: int = 2) -> None:
        if required_seconds <= 0:
            raise ValueError("required_seconds must be positive")
        if required_stem_transitions < 0:
            raise ValueError("required_stem_transitions cannot be negative")
        self.required_seconds = required_seconds
        self.required_stem_transitions = required_stem_transitions
        self.telemetry_lines = 0
        self.ignored_lines = 0
        self.malformed_lines = 0
        self.play_seconds = 0
        self.first_play_second: int | None = None
        self.last_play_second: int | None = None
        self.last_stem: int | None = None
        self.stem_transitions = 0
        self.failures: list[str] = []
        self.maxima = {field: 0 for field in MAXIMUM_FIELDS}
        self._target_checked = False

    def _fail(self, message: str) -> None:
        if message not in self.failures:
            self.failures.append(message)

    @property
    def target_reached(self) -> bool:
        return self.play_seconds >= self.required_seconds

    @property
    def fatal_failure(self) -> bool:
        return bool(self.failures)

    def feed_line(self, line: str) -> dict[str, object]:
        """Consume a line and return its structured capture record."""
        record: dict[str, object] = {"raw": line.rstrip("\r\n")}
        try:
            telemetry = parse_telemetry_line(line)
        except TelemetryError as exc:
            self.malformed_lines += 1
            message = f"malformed telemetry: {exc}"
            self._fail(message)
            record["error"] = message
            return record
        if telemetry is None:
            self.ignored_lines += 1
            record["kind"] = "chatter"
            return record

        self.telemetry_lines += 1
        record["kind"] = "telemetry"
        record["telemetry"] = telemetry
        second = int(telemetry["sec"])
        state = str(telemetry["state"])

        for field in MAXIMUM_FIELDS:
            if field in telemetry:
                self.maxima[field] = max(self.maxima[field], int(telemetry[field]))
        for field in ZERO_COUNTER_FIELDS:
            if int(telemetry[field]) != 0:
                self._fail(f"{field} became {telemetry[field]} at sec={second}")
        if int(telemetry["blk_us_max"]) > BLOCK_DEADLINE_US:
            self._fail(
                f"deadline exceeded at sec={second}: "
                f"blk_us_max={telemetry['blk_us_max']} > {BLOCK_DEADLINE_US}"
            )

        if self.first_play_second is None:
            if state != "PLAY":
                return record
            self.first_play_second = second
        elif state != "PLAY" and not self.target_reached:
            self._fail(
                f"left PLAY before acceptance at sec={second}: state={state}"
            )
            return record

        if state != "PLAY" or self.target_reached:
            return record
        if self.last_play_second is not None and second != self.last_play_second + 1:
            self._fail(
                f"heartbeat gap in PLAY: {self.last_play_second} -> {second}"
            )
        stem = int(telemetry["stem"])
        if self.last_stem is not None and stem != self.last_stem:
            self.stem_transitions += 1
        self.last_stem = stem
        self.last_play_second = second
        self.play_seconds += 1
        if self.target_reached and not self._target_checked:
            self._target_checked = True
            if self.stem_transitions < self.required_stem_transitions:
                self._fail(
                    "STEM target transitions insufficient: "
                    f"{self.stem_transitions} < {self.required_stem_transitions}"
                )
        return record

    def summary(self, final: bool = True) -> dict[str, object]:
        failures = list(self.failures)
        if final and self.play_seconds < self.required_seconds:
            failures.append(
                "insufficient consecutive PLAY seconds: "
                f"{self.play_seconds} < {self.required_seconds}"
            )
        if final and self.play_seconds >= self.required_seconds \
                and self.stem_transitions < self.required_stem_transitions:
            message = (
                "STEM target transitions insufficient: "
                f"{self.stem_transitions} < {self.required_stem_transitions}"
            )
            if message not in failures:
                failures.append(message)
        return {
            "status": "PASS" if not failures and self.target_reached else "FAIL",
            "required_seconds": self.required_seconds,
            "required_stem_transitions": self.required_stem_transitions,
            "play_seconds": self.play_seconds,
            "first_play_second": self.first_play_second,
            "last_play_second": self.last_play_second,
            "stem_transitions": self.stem_transitions,
            "telemetry_lines": self.telemetry_lines,
            "ignored_lines": self.ignored_lines,
            "malformed_lines": self.malformed_lines,
            "maxima": dict(self.maxima),
            "failures": failures,
        }


def evaluate_lines(lines: Iterable[str], required_seconds: int = 1800,
                   required_stem_transitions: int = 2) -> dict[str, object]:
    """Evaluate a finite synthetic or captured log without serial hardware."""
    acceptance = AudioAcceptance(required_seconds, required_stem_transitions)
    for line in lines:
        acceptance.feed_line(line)
        if acceptance.target_reached or acceptance.fatal_failure:
            break
    return acceptance.summary(final=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def run_live(args: argparse.Namespace) -> int:
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError:
        print("pyserial is required for live capture: python -m pip install pyserial",
              file=sys.stderr)
        return 2

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    capture_path = output_dir / f"audio_uart_{stamp}.jsonl"
    summary_path = output_dir / f"audio_uart_{stamp}.summary.json"
    acceptance = AudioAcceptance(args.seconds, args.stem_transitions)
    started_monotonic = time.monotonic()
    last_telemetry_monotonic: float | None = None
    runtime_failure: str | None = None

    print(f"Capturing {args.port} at {args.baud} baud")
    print(f"JSONL: {capture_path}")
    try:
        with serial.Serial(args.port, args.baud, timeout=0.25) as uart, \
                capture_path.open("w", encoding="utf-8", newline="\n") as capture:
            while not acceptance.target_reached and not acceptance.fatal_failure:
                payload = uart.readline()
                now = time.monotonic()
                if not payload:
                    if last_telemetry_monotonic is not None \
                            and now - last_telemetry_monotonic > args.idle_timeout:
                        runtime_failure = (
                            "UART telemetry timeout: "
                            f"{now - last_telemetry_monotonic:.1f}s"
                        )
                        break
                    if last_telemetry_monotonic is None \
                            and now - started_monotonic > args.startup_timeout:
                        runtime_failure = (
                            f"no telemetry within {args.startup_timeout:.1f}s"
                        )
                        break
                    continue
                line = payload.decode("utf-8", errors="replace").rstrip("\r\n")
                record = acceptance.feed_line(line)
                record["observed_at"] = utc_now()
                capture.write(json.dumps(record, sort_keys=True) + "\n")
                capture.flush()
                if record.get("kind") == "telemetry":
                    last_telemetry_monotonic = now
                print(line)
    except (OSError, serial.SerialException) as exc:
        runtime_failure = f"serial capture failed: {exc}"
    except KeyboardInterrupt:
        runtime_failure = "capture interrupted by user"

    if runtime_failure is not None:
        acceptance._fail(runtime_failure)
    summary = acceptance.summary(final=True)
    summary.update({
        "port": args.port,
        "baud": args.baud,
        "block_deadline_us": BLOCK_DEADLINE_US,
        "capture": str(capture_path),
        "started_at": datetime.fromtimestamp(
            time.time() - (time.monotonic() - started_monotonic), timezone.utc
        ).isoformat(timespec="seconds"),
        "finished_at": utc_now(),
    })
    write_summary(summary_path, summary)
    print(f"Summary: {summary_path}")
    print(f"AUDIO_BOARD_ACCEPTANCE: {summary['status']}")
    for failure in summary["failures"]:
        print(f"  - {failure}")
    return 0 if summary["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify 30 minutes of Navigator FullStem UART telemetry")
    parser.add_argument("--port", default=os.environ.get("STEM_UART_PORT"),
                        help="UART port, or set STEM_UART_PORT")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=int, default=1800,
                        help="required consecutive PLAY telemetry seconds")
    parser.add_argument("--stem-transitions", type=int, default=2,
                        help="minimum observed STEM target transitions")
    parser.add_argument("--output", default="hardware/build/navigator_audio_acceptance")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--idle-timeout", type=float, default=3.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.port:
        parser.error("--port is required (or set STEM_UART_PORT)")
    if args.baud <= 0 or args.seconds <= 0 or args.stem_transitions < 0:
        parser.error("baud/seconds must be positive and transitions nonnegative")
    if args.startup_timeout <= 0 or args.idle_timeout <= 0:
        parser.error("timeouts must be positive")
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
