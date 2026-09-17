"""Produce the exact full-task output expected from the pristine zeroed image."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from npu_task_reference import PROGRAM_DEFAULT, TaskReference


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "hardware" / "build" / "navigator_zero_input_reference.bin"


def main() -> int:
    image = (PROGRAM_DEFAULT / "task_image.bin").read_bytes()
    reference = TaskReference(PROGRAM_DEFAULT, image)
    reference.run(progress=True)
    output = reference.output_bytes()
    OUTPUT.write_bytes(output)
    metadata = {
        "input": "pristine task image; activation arena is all zero",
        "bytes": len(output),
        "sha256": hashlib.sha256(output).hexdigest(),
        "nonzero_bytes": sum(value != 0 for value in output),
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
