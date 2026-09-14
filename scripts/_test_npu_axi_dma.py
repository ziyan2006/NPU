"""Regression tests for full-program AXI DMA burst feasibility."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
SCRIPT = ROOT / "scripts" / "31_plan_axi_dma.py"

spec = importlib.util.spec_from_file_location("npu_axi_dma_planner", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load AXI DMA planner")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)

analysis = json.loads(
    (PROGRAM / "axi_dma_analysis.json").read_text(encoding="utf-8"))
assert analysis["request_count"] == 1078
assert analysis["row_count"] == 44_940
assert analysis["payload_bytes"] == 3_020_480
assert analysis["burst_count"] == 45_664
assert analysis["burst_count_by_direction"] == {"read": 13_920, "write": 31_744}
assert analysis["beat_count"] == 378_696
assert analysis["narrow_burst_count"] == 2_272
assert analysis["four_kib_split_count"] == 374
assert analysis["maximum_bursts_per_request"] == 495
assert analysis["estimated_axi_cycles"] == {
    "read": 539_976, "write": 569_344, "total": 1_109_320,
}
assert analysis["alignment_pass"] and analysis["payload_exact_pass"]

crossing = planner.split_row(4088, 40)
assert crossing == [
    {"address": 4088, "beat_bytes": 8, "beats": 1, "bytes": 8},
    {"address": 4096, "beat_bytes": 8, "beats": 4, "bytes": 32},
]
long_row = planner.split_row(0, 3000)
assert [row["beats"] for row in long_row] == [256, 119]
assert planner.split_row(8192, 6)[-1]["beat_bytes"] == 2

tile_analysis = json.loads(
    (PROGRAM / "tile_analysis.json").read_text(encoding="utf-8"))
assert tile_analysis["cycles"]["dma_read"] == 539_976
assert tile_analysis["cycles"]["dma_write"] == 569_344

with tempfile.TemporaryDirectory() as temporary:
    output = Path(temporary)
    planner.analyze(PROGRAM, output)
    expected = hashlib.sha256(
        (PROGRAM / "axi_dma_analysis.json").read_bytes()).digest()
    actual = hashlib.sha256((output / "axi_dma_analysis.json").read_bytes()).digest()
    assert actual == expected

print("NPU AXI DMA plan tests: PASS")
