"""Fast structural checks for the generated XC7Z020 model package."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "hardware" / "generated" / "bott2_mir1k_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
assert manifest["schema"] == 1
assert manifest["target"] == "XC7Z020-1"
assert manifest["graph"]["bands"] == 128
assert manifest["graph"]["chunk_frames"] % 8 == 0
assert manifest["graph"]["low_frequency_kill_bands"] == 44
assert len(manifest["layers"]) == 11
assert manifest["totals"]["parameters"] == 824_900
assert manifest["totals"]["macs_per_chunk"] == 123_338_752
assert manifest["totals"]["scheduled_cycles_pout8_pin8"] == 1_986_560
assert manifest["totals"]["pe_array"]["mac_lanes"] == 64
assert manifest["quantization"]["pre_tanh_activation_step_int12"] > 0
assert manifest["layers"][-1]["pre_tanh_activation_step_int12"] > 0
assert manifest["layers"][-1]["post_tanh_output_step_int12"] == 1 / 2047

checkpoint = ROOT / manifest["checkpoint"]
assert len(manifest["checkpoint_sha256"]) == 64
if checkpoint.exists():
    assert sha256(checkpoint) == manifest["checkpoint_sha256"]
else:
    print(f"checkpoint not distributed; skipped source SHA check: {checkpoint}")

blob_specs = (
    ("weights_int8.bin", "weight", 1),
    ("bias_int32.bin", "bias", 4),
    ("scales_f32.bin", "scale", 8),
)
for filename, field, bytes_per_element in blob_specs:
    size = (PACKAGE / filename).stat().st_size
    for layer in manifest["layers"]:
        offset = layer[f"{field}_offset"]
        length = layer[f"{field}_bytes"]
        assert offset % 64 == 0
        assert offset + length <= size
        cout = layer["out_channels"]
        if field == "weight":
            cin = layer["in_channels"] // layer["groups"]
            expected = cout * cin * math.prod(layer["kernel"])
        else:
            expected = cout * bytes_per_element
        assert length == expected, (layer["name"], field, length, expected)

print("NPU package structural tests: PASS")
