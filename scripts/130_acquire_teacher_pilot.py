"""Acquire pinned public Mel-Band teacher assets, without changing the base env.

The public publisher checksum is checked before safe tensor-only weight loading
in the separate pilot. No music is uploaded or decoded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/teachers/kim_melband_20261001"
PROTOCOL = ROOT / "docs/teacher_pilot_protocol_20261001.json"
SOURCE_FILES = ("README.md", "requirements.txt", "inference.py", "utils.py",
                "configs/config_vocals_mel_band_roformer.yaml",
                "models/mel_band_roformer/__init__.py", "models/mel_band_roformer/attend.py",
                "models/mel_band_roformer/mel_band_roformer.py")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire(out, protocol):
    if not out.resolve().is_relative_to((ROOT / "data/teachers").resolve()) or out.resolve() == (ROOT / "data/teachers").resolve():
        raise ValueError("Teacher assets must remain below ignored workspace data/teachers")
    settings = json.loads(protocol.read_text(encoding="utf-8"))["candidate"]
    if (settings["source_repository"] != "KimberleyJensen/Mel-Band-Roformer-Vocal-Model" or
            settings["weight_repository"] != "KimberleyJSN/melbandroformer" or
            settings["weight_bytes"] > 1500000000):
        raise ValueError("Unexpected public teacher source or download budget")
    receipt = out / "asset_receipt.json"
    if receipt.exists():
        verify(out, protocol)
        return
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source"
    files = {}
    base = f"https://raw.githubusercontent.com/{settings['source_repository']}/{settings['source_commit']}/"
    for name in SOURCE_FILES:
        path = source / name
        response = requests.get(base + name, timeout=(15, 30))
        response.raise_for_status()
        expected = hashlib.sha256(response.content).hexdigest()
        if path.exists():
            if sha256(path) != expected:
                raise ValueError("Interrupted source download differs; never overwrite it")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(response.content)
        files[str(path.relative_to(out)).replace("\\", "/")] = expected
    weight = out / settings["weight_filename"]
    url = (f"https://huggingface.co/{settings['weight_repository']}/resolve/"
           f"{settings['weight_revision']}/{settings['weight_filename']}")
    partial = weight.with_suffix(weight.suffix + ".partial")
    if not weight.exists():
        if partial.exists():
            raise ValueError("Partial weight preserved; explicitly review before a new acquisition")
        size, last = 0, time.monotonic()
        with requests.get(url, stream=True, timeout=(15, 45)) as response:
            response.raise_for_status()
            with partial.open("xb") as stream:
                for block in response.iter_content(4 * 1024 * 1024):
                    size += len(block)
                    if size > settings["weight_bytes"]:
                        raise ValueError("Download larger than publisher metadata")
                    stream.write(block)
                    if time.monotonic() - last > 20:
                        print(f"TEACHER_DOWNLOAD bytes={size}/{settings['weight_bytes']}", flush=True)
                        last = time.monotonic()
        if size != settings["weight_bytes"] or sha256(partial) != settings["weight_sha256"]:
            raise ValueError("Publisher weight size/checksum failed; partial preserved")
        partial.rename(weight)
    if weight.stat().st_size != settings["weight_bytes"] or sha256(weight) != settings["weight_sha256"]:
        raise ValueError("Existing weight differs from pinned publisher checksum")
    files[weight.name] = settings["weight_sha256"]
    doc = {"schema": 1, "protocol_sha256": sha256(protocol), "source_commit": settings["source_commit"],
           "weight_revision": settings["weight_revision"], "weight_url": url, "files_sha256": files,
           "weight_checksum_provenance": "Public Hugging Face API LFS SHA-256 metadata, checked against downloaded bytes",
           "licensing_scope": settings["license_metadata"], "music_uploaded": False,
           "base_python_environment_changed": False}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2)
    print(f"TEACHER_ASSETS COMPLETE checksum={settings['weight_sha256']}", flush=True)


def verify(out=ASSETS, protocol=PROTOCOL):
    doc = json.loads((out / "asset_receipt.json").read_text(encoding="utf-8"))
    settings = json.loads(protocol.read_text(encoding="utf-8"))["candidate"]
    if doc["protocol_sha256"] != sha256(protocol) or doc["source_commit"] != settings["source_commit"]:
        raise ValueError("Teacher protocol/source binding changed")
    expected_names = {"source/" + n for n in SOURCE_FILES} | {settings["weight_filename"]}
    if set(doc["files_sha256"]) != expected_names:
        raise ValueError("Incomplete/unexpected teacher assets")
    for name, expected in doc["files_sha256"].items():
        if sha256(out / name) != expected:
            raise ValueError(f"Teacher asset changed: {name}")
    if doc["files_sha256"][settings["weight_filename"]] != settings["weight_sha256"]:
        raise ValueError("Teacher weight is not the pinned publisher checkpoint")
    print("TEACHER_ASSETS VERIFIED", flush=True)
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ASSETS)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args.out, args.protocol)
    else:
        acquire(args.out, args.protocol)


if __name__ == "__main__":
    main()
