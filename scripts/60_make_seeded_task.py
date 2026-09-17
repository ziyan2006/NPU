"""Prepare one deterministic nonzero-input task image and its exact output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from npu_task_reference import PROGRAM_DEFAULT, TaskReference, seed_input


ROOT = Path(__file__).resolve().parent.parent
SEED = 0x4E50_5531
OUTPUT = ROOT / "hardware" / "build" / "navigator_seeded_task"


def main() -> int:
    program = json.loads((PROGRAM_DEFAULT / "program.json").read_text(encoding="utf-8"))
    manifest = json.loads((PROGRAM_DEFAULT / "task_image.json").read_text(encoding="utf-8"))
    image = seed_input((PROGRAM_DEFAULT / "task_image.bin").read_bytes(),
                       program, manifest, SEED)
    reference = TaskReference(PROGRAM_DEFAULT, image)
    reference.run(progress=True)
    expected = reference.output_bytes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "task_image.bin").write_bytes(image)
    (OUTPUT / "expected_output.bin").write_bytes(expected)
    metadata = {"seed": f"0x{SEED:08x}", "task_sha256": hashlib.sha256(image).hexdigest(),
                "output_bytes": len(expected), "output_sha256": hashlib.sha256(expected).hexdigest()}
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
