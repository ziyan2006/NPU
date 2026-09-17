"""Compare two binary files exactly and report their first mismatch."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    expected, actual = args.expected.read_bytes(), args.actual.read_bytes()
    if expected != actual:
        first = next(i for i, values in enumerate(zip(expected, actual)) if values[0] != values[1])
        raise SystemExit(f"mismatch byte {first}: expected=0x{expected[first]:02x} actual=0x{actual[first]:02x}; "
                         f"sizes={len(expected)}/{len(actual)}")
    print(f"binary compare: PASS bytes={len(actual)} sha256={hashlib.sha256(actual).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
