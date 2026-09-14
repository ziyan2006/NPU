"""Regression tests for the P4 DMA reference plan and address contract."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
SCRIPT = ROOT / "scripts" / "30_plan_npu_dma.py"


spec = importlib.util.spec_from_file_location("npu_dma_planner", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load DMA planner")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)

analysis = json.loads((PROGRAM / "dma_analysis.json").read_text(encoding="utf-8"))
plan = json.loads((PROGRAM / "dma_plan.json").read_text(encoding="utf-8"))
assert analysis == plan["analysis"]
assert analysis == {
    "schema": 1,
    "status": "P4-proposal",
    "request_count": 1078,
    "load_count": 830,
    "store_count": 248,
    "request_count_by_space": {
        "activation": 302, "bias": 248, "quant": 280, "weight": 248,
    },
    "payload_bytes": 3_020_480,
    "activation_clear_bytes": 1_246_656,
    "maximum_scratchpad_end": {
        "A0": 57_600, "A1": 57_600,
        "O0": 4_096, "O1": 4_096,
        "W0": 16_320, "W1": 16_320,
    },
    "bounds_pass": True,
}

by_command = {row["command_index"]: row for row in plan["requests"]}
enc0 = by_command[0]
assert enc0["memory_space"] == "activation"
assert enc0["scratchpad"] == "A0"
assert enc0["clear_bytes"] == 5_184
assert enc0["scratchpad_offset"] == 320
assert (enc0["x_bytes"], enc0["y_count"], enc0["z_count"]) == (4, 16, 17)

segments = [row for row in plan["requests"]
            if any(dep.startswith("segment:")
                   for dep in row["descriptor_dependencies"])]
assert len(segments) == 28
assert all(row["scratchpad_y_stride"] > row["x_bytes"] for row in segments)
assert any(not row["clear_before"] for row in segments)

vector_quant = [row for row in plan["requests"]
                if row.get("quant_slot") == "vector"]
assert len(vector_quant) == 32
assert all(row["x_bytes"] == 16 for row in vector_quant)

tail_stores = [row for row in plan["requests"]
               if row["opcode"] == "DMA_STORE" and row["x_bytes"] == 8]
assert len(tail_stores) == 8

tile_analysis = json.loads(
    (PROGRAM / "tile_analysis.json").read_text(encoding="utf-8"))
assert (analysis["payload_bytes"] + tile_analysis["memory"]["upsample_ddr_bytes"]
        == tile_analysis["memory"]["scheduled_ddr_bytes"])

with tempfile.TemporaryDirectory() as temporary:
    first = Path(temporary) / "first"
    second = Path(temporary) / "second"
    planner.build_plan(PROGRAM, first)
    planner.build_plan(PROGRAM, second)
    for filename in ("dma_plan.json", "dma_analysis.json"):
        expected = hashlib.sha256((PROGRAM / filename).read_bytes()).digest()
        assert hashlib.sha256((first / filename).read_bytes()).digest() == expected
        assert hashlib.sha256((second / filename).read_bytes()).digest() == expected

print("NPU DMA plan tests: PASS")
