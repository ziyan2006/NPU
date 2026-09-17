"""Split a binary into ordered JTAG-friendly chunks without changing its bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-bytes", type=int, default=128 * 1024)
    args = parser.parse_args()
    if args.chunk_bytes <= 0 or args.chunk_bytes % 64:
        raise SystemExit("--chunk-bytes must be a positive multiple of 64")
    payload = args.input.resolve().read_bytes()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    chunks = []
    for offset in range(0, len(payload), args.chunk_bytes):
        name = f"chunk_{offset:010d}.bin"
        chunk = payload[offset:offset + args.chunk_bytes]
        (output / name).write_bytes(chunk)
        chunks.append({"offset": offset, "bytes": len(chunk), "file": name})
    manifest = {"total_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                "chunk_bytes": args.chunk_bytes, "chunks": chunks}
    (output / "chunks.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"wrote {len(chunks)} chunks totaling {len(payload):,} bytes to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
