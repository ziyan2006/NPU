"""Structural regression tests for the proposed P3 tile scheduler."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

from npu_isa import COMMAND_STRUCT, OPERATOR_STRUCT, Command, Opcode


ROOT = Path(__file__).resolve().parent.parent
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
SOURCE = ROOT / "hardware" / "generated" / "bott2_mir1k_v1"
SCRIPT = ROOT / "scripts" / "28_schedule_npu_tiles.py"


def load_scheduler():
    spec = importlib.util.spec_from_file_location("npu_tile_scheduler", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tile scheduler")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


scheduler = load_scheduler()
analysis = json.loads((PROGRAM / "tile_analysis.json").read_text(encoding="utf-8"))
schedule = json.loads((PROGRAM / "tile_schedule.json").read_text(encoding="utf-8"))

assert OPERATOR_STRUCT.size == 64
assert COMMAND_STRUCT.size == 16
assert analysis["counts"] == {
    "commands": 1837,
    "operator_descriptors": 248,
    "tiles": 248,
}
assert analysis["coverage_pass"]
assert all(row["missing_elements"] == 0 for row in analysis["coverage"])
assert all(row["overlap_elements"] == 0 for row in analysis["coverage"])
assert analysis["bank_capacity_pass"]
assert analysis["bank_conflict_count"] == 0
assert analysis["cycles"]["compute"] == 1_986_560
assert analysis["cycles"]["total"] < analysis["cycles"]["budget"]
assert analysis["cycles"]["hidden_by_overlap"] > 0
assert analysis["memory"]["scheduled_ddr_bytes"] == 3_555_136

commands_blob = (PROGRAM / "tile_commands.bin").read_bytes()
commands = [Command.unpack(commands_blob[offset:offset + COMMAND_STRUCT.size])
            for offset in range(0, len(commands_blob), COMMAND_STRUCT.size)]
assert len(commands) == analysis["counts"]["commands"]
assert [command.tag for command in commands] == list(range(len(commands)))
assert commands[-1].opcode == Opcode.END
assert sum(command.opcode == Opcode.CONV2D for command in commands) == 248
assert sum(command.opcode == Opcode.UPSAMPLE2X for command in commands) == 3
assert sum(command.opcode == Opcode.VEC_ADD for command in commands) == 32

descriptor_blob = (PROGRAM / "tile_operator_desc.bin").read_bytes()
assert len(descriptor_blob) == analysis["counts"]["operator_descriptors"] * 64
for tile in schedule["tiles"]:
    offset = tile["operator_desc"] * OPERATOR_STRUCT.size
    fields = OPERATOR_STRUCT.unpack(descriptor_blob[offset:offset + 64])
    assert list(fields[16:24]) == [
        tile["origin"]["h"], tile["origin"]["w"],
        tile["origin"]["cout"], tile["shape"]["h"],
        tile["shape"]["w"], tile["shape"]["cout"], 0,
        schedule["operator_descriptors"][tile["operator_desc"]]
        ["input_channel_count"],
    ]
    expected_tile_bank = tile["index"] & 1
    assert tile["bank_ids"]["weight"] == expected_tile_bank
    assert tile["bank_ids"]["accumulator"] == expected_tile_bank
    assert tile["bank_ids"]["output"] == expected_tile_bank
    assert tile["bank_ids"]["activation"] == tile["spatial_index"] & 1
    assert tile["loads_input"] == (tile["origin"]["cout"] == 0)

decoder_tiles = [tile for tile in schedule["tiles"]
                 if tile["layer"].startswith("dec")]
assert decoder_tiles
assert all(len(tile["input_parts"]) == 2 for tile in decoder_tiles)

for filename, metadata in schedule["files"].items():
    path = PROGRAM / filename
    assert path.stat().st_size == metadata["bytes"]
    assert digest(path) == metadata["sha256"]

with tempfile.TemporaryDirectory() as temporary:
    first = Path(temporary) / "first"
    second = Path(temporary) / "second"
    scheduler.compile_schedule(PROGRAM, SOURCE, first)
    scheduler.compile_schedule(PROGRAM, SOURCE, second)
    for filename in (
        "tile_commands.bin", "tile_operator_desc.bin",
        "tile_schedule.json", "tile_analysis.json",
    ):
        assert digest(first / filename) == digest(second / filename)
        assert digest(first / filename) == digest(PROGRAM / filename)

print("npu tile schedule tests: PASS")
