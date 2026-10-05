"""Download public educational excerpts into quarantine, never into training.

Prepare seals whole-song candidate roles against the old data lock and filename
inventories BEFORE audio download. Download follows only author-published URLs;
no accounts, access requests, TLS disabling, model inference or SD writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import unicodedata
from urllib.parse import urlparse
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
HOST = "www.mtkdata.cambridgemusictechnology.co.uk"
DEFAULT_PLAN = ROOT / "docs/cambridge_acquisition_plan_20261001.json"
DEFAULT_OUT = ROOT / "data/datasets/CambridgeMTK-new-candidates-20261001"
ROLES = {"train_candidate", "development_candidate", "acceptance_candidate"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_digest(doc):
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def seal(doc):
    doc["content_sha256"] = content_digest(doc)
    return doc


def write_new_json(path, doc):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, ensure_ascii=False, allow_nan=False)


def read_sealed(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if content_digest(doc) != doc.get("content_sha256"):
        raise ValueError(f"Changed sealed document: {path}")
    return doc


def song_key(value):
    value = unicodedata.normalize("NFKC", str(value).replace("\\", "/").rsplit("/", 1)[-1]).casefold()
    value = re.sub(r"\.(?:mp3|mp4|wav|flac|ogg|m4a|aiff|aif)$", "", value)
    value = re.sub(r"\.stem$", "", value)
    return "".join(c for c in value if c.isalnum())


def record_keys(row):
    return {song_key(row["track_id"]), song_key(f"{row['artist']} - {row['title']}")}


def check_roles(plan, known_ids):
    if plan.get("schema") != 1 or not re.fullmatch(r"[0-9a-f]{40}", plan["metadata_commit"]):
        raise ValueError("Invalid candidate plan schema or unpinned metadata")
    if not plan["records"] or sum(r["expected_bytes"] for r in plan["records"]) > plan["maximum_download_bytes"]:
        raise ValueError("Invalid/over-budget acquisition")
    known = {song_key(name) for name in known_ids}
    used, artists = set(), {}
    for row in plan["records"]:
        ident, role = row["track_id"], row["role"]
        if not re.fullmatch(r"[A-Za-z0-9_]+", ident) or role not in ROLES or row["expected_bytes"] <= 0:
            raise ValueError("Invalid song ID/role/size")
        url = urlparse(row["url"])
        if (url.scheme != "http" or url.netloc != HOST or url.query or url.fragment or
                not re.fullmatch(r"/MTK\d{3}/" + re.escape(ident) + r"\.zip", url.path)):
            raise ValueError("Only exact public excerpt URLs on the named publisher host are permitted")
        keys = record_keys(row)
        if not all(keys) or keys & (known | used):
            raise ValueError(f"Duplicate/previously observed song: {ident}")
        used.update(keys)
        artist = song_key(row["artist"])
        if artist in artists and artists[artist] != role:
            raise ValueError("One newly acquired artist cannot span candidate roles")
        artists[artist] = role


def guard_output(out):
    out = out.resolve()
    allowed = (ROOT / "data/datasets").resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("Acquisition output must be a child of workspace data/datasets")
    return out


def prepare(plan_path, out, data_lock, library):
    out = guard_output(out)
    if out.exists():
        raise ValueError("Use a new quarantine directory; never overwrite a previous assignment")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    old = read_sealed(data_lock)
    if sha256(old["checkpoint"]) != old["frozen_sha256"]:
        raise ValueError("Frozen weights changed")
    musdb = sorted((ROOT / "data/datasets/MUSDB18-7-STEMS").rglob("*.stem.mp4"))
    if len(musdb) != 144 or not library.is_dir():
        raise ValueError("Old MUSDB/private-library filename inventories are required for leakage checks")
    private = sorted(p for p in library.rglob("*") if p.suffix.lower() in (
        ".mp3", ".wav", ".flac", ".m4a", ".aiff", ".aif"))
    if not private:
        raise ValueError("Empty private-library inventory")
    known = sorted(set([r["track_id"] for r in old["records"]] + old["known_regression_ids"] +
                       old["teacher"]["source_ids"] + [p.name for p in musdb + private]))
    check_roles(plan, known)
    url = ("https://raw.githubusercontent.com/SiddGururani/mixing_secrets/" +
           plan["metadata_commit"] + "/" + plan["catalogue_path"])
    response = requests.get(url, timeout=(15, 30))
    response.raise_for_status()
    catalogue = response.content
    links = {html.unescape(u) for u in re.findall(r'href="([^"]+\.zip)"', response.text)}
    if any(r["url"] not in links for r in plan["records"]):
        raise ValueError("An excerpt URL is not explicitly linked in the pinned author catalogue")
    out.mkdir(parents=True)
    with (out / "catalogue_snapshot.html").open("xb") as stream:
        stream.write(catalogue)
    assignment = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path.resolve()), "plan_sha256": sha256(plan_path), "plan": plan,
        "data_lock_path": str(data_lock.resolve()), "data_lock_sha256": sha256(data_lock),
        "frozen_checkpoint": old["checkpoint"], "frozen_sha256": old["frozen_sha256"],
        "catalogue_url": url, "catalogue_sha256": sha256(out / "catalogue_snapshot.html"),
        "known_ids": known, "inventory_counts": {"musdb": len(musdb), "private_library_files": len(private)},
        "old_audio_sha256": sorted({row["sha256"] for row in old["files"].values()}),
        "leakage_limit": "Names and manually supplied canonical aliases, plus exact original-file hashes; not exhaustive acoustic fingerprints or missing historical archives",
        "model_scoring_performed": False, "training_performed": False, "blind_status": "incomplete",
        "records": [{**row, "same_artist_in_known_ids": any(
            song_key(name).startswith(song_key(row["artist"])) for name in known)} for row in plan["records"]],
    }
    write_new_json(out / "assignment.json", seal(assignment))
    print(f"CANDIDATE_ASSIGNMENT sealed songs={len(plan['records'])}; blind=incomplete", flush=True)


def verify_assignment(out):
    doc = read_sealed(out / "assignment.json")
    for key, expected in (("plan_path", "plan_sha256"), ("data_lock_path", "data_lock_sha256"),
                          ("frozen_checkpoint", "frozen_sha256")):
        if sha256(doc[key]) != doc[expected]:
            raise ValueError(f"Changed assignment dependency: {key}")
    if sha256(out / "catalogue_snapshot.html") != doc["catalogue_sha256"]:
        raise ValueError("Catalogue snapshot changed")
    if doc["records"] != [{**r, "same_artist_in_known_ids": any(
            song_key(n).startswith(song_key(r["artist"])) for n in doc["known_ids"])}
            for r in doc["plan"]["records"]]:
        raise ValueError("Candidate roles differ from the acquisition plan")
    check_roles(doc["plan"], doc["known_ids"])
    return doc


def safe_members(archive, track_id, max_unpacked):
    infos, seen = archive.infolist(), set()
    if len(infos) > 512 or sum(i.file_size for i in infos) > max_unpacked:
        raise ValueError("Archive entry/unpacked-size cap exceeded")
    for info in infos:
        path = PurePosixPath(info.filename)
        if (info.orig_filename != info.filename or "\\" in info.orig_filename or
                path.is_absolute() or ".." in path.parts or
                not path.parts or path.parts[0] != track_id or
                any(":" in part or part.endswith((".", " ")) for part in path.parts) or
                any(re.search(r'[<>"|?*\x00-\x1f]', part) or
                    re.fullmatch(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part, re.I)
                    for part in path.parts) or
                stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError(f"Unsafe archive path: {info.filename}")
        key = info.filename.casefold().rstrip("/")
        if key in seen:
            raise ValueError("Duplicate/case-colliding archive paths")
        seen.add(key)
        if not info.is_dir() and (len(path.parts) != 2 or path.suffix.casefold() not in (".wav", ".txt")):
            raise ValueError("Unexpected nested or non-audio/text archive member")
        if not info.is_dir() and info.file_size / max(info.compress_size, 1) > 1000:
            raise ValueError("Suspicious decompression ratio")
    return infos


def validate_archive(path, row, max_unpacked):
    with zipfile.ZipFile(path) as archive:
        infos = safe_members(archive, row["track_id"], max_unpacked)
        readmes = [i for i in infos if PurePosixPath(i.filename).name.casefold() == "readme.txt"]
        if len(readmes) != 1 or readmes[0].file_size > 65536:
            raise ValueError("One small bundled Readme.txt is required")
        terms = archive.read(readmes[0]).decode("utf-8", errors="replace")
        if "educational purposes only" not in terms.casefold() or "commercial purpose" not in terms.casefold():
            raise ValueError("Bundled educational/non-commercial restrictions require manual review")
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failed")
        return infos, terms


def download_all(out):
    doc = verify_assignment(out)
    max_unpacked = doc["plan"]["maximum_unpacked_bytes_per_archive"]
    for row in doc["records"]:
        ident = row["track_id"]
        receipt_path = out / f"{ident}_receipt.json"
        if receipt_path.exists():
            receipt = read_sealed(receipt_path)
            if receipt["assignment_sha256"] != sha256(out / "assignment.json"):
                raise ValueError("Receipt belongs to a different role assignment")
            if sha256(out / f"{ident}.zip") != receipt["archive_sha256"]:
                raise ValueError("Existing archive changed")
            for name, expected in receipt["extracted_files"].items():
                if sha256(out / name) != expected:
                    raise ValueError("Extracted original file changed")
            print(f"CANDIDATE_DOWNLOAD verified existing {ident}", flush=True)
            continue
        archive_path, partial = out / f"{ident}.zip", out / f"{ident}.zip.partial"
        if not archive_path.exists():
            if partial.exists():
                raise ValueError("Interrupted partial preserved; use a reviewed new output instead of silent overwrite")
            received = 0
            with requests.get(row["url"], stream=True, timeout=(15, 30), allow_redirects=False) as response:
                if response.status_code != 200 or int(response.headers.get("Content-Length", -1)) != row["expected_bytes"]:
                    raise ValueError("Publisher response/size differs; review before downloading")
                with partial.open("xb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        received += len(block)
                        if received > row["expected_bytes"]:
                            raise ValueError("Download exceeded predeclared size")
                        stream.write(block)
                if received != row["expected_bytes"]:
                    raise ValueError("Truncated download; partial preserved")
            partial.rename(archive_path)
        if archive_path.stat().st_size != row["expected_bytes"]:
            raise ValueError("Existing archive size mismatch")
        infos, terms = validate_archive(archive_path, row, max_unpacked)
        target = out / ident
        if target.exists():
            raise ValueError("Unreceipted extraction preserved; do not overwrite it")
        # Extract only validated names, without executing bundled files.
        target.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for info in infos:
                if not info.is_dir():
                    with archive.open(info) as source, (out / info.filename).open("xb") as destination:
                        shutil.copyfileobj(source, destination)
        files = {i.filename: sha256(out / i.filename) for i in infos if not i.is_dir()}
        write_new_json(receipt_path, seal({
            "schema": 1, "track_id": ident, "role": row["role"], "source_url": row["url"],
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "assignment_sha256": sha256(out / "assignment.json"),
            "transport": "Publisher exposes HTTP only; SHA-256 is a local integrity record, not an authenticated publisher checksum",
            "bytes": archive_path.stat().st_size, "archive_sha256": sha256(archive_path),
            "unpacked_bytes": sum(i.file_size for i in infos), "extracted_files": files,
            "bundled_terms": terms, "commercial_training_authorized": False,
            "quality_status": "quarantine_pending_stem_and_listening_review"}))
        print(f"CANDIDATE_DOWNLOAD complete {ident} bytes={row['expected_bytes']} role={row['role']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "download", "verify"))
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data-lock", type=Path, default=ROOT / "results/training_protocol_20261001/dataset_lock.json")
    ap.add_argument("--library", type=Path, default=Path(os.environ.get("STEM_AUDIO_LIBRARY", "D:/DJ_Music_Library")))
    args = ap.parse_args()
    out = guard_output(args.out)
    if args.action == "prepare":
        prepare(args.plan, out, args.data_lock, args.library)
    else:
        verify_assignment(out)
        if args.action == "download":
            download_all(out)
        else:
            # Also verifies every existing receipt and original file, but never
            # downloads missing records during a read-only verification.
            for row in read_sealed(out / "assignment.json")["records"]:
                receipt = read_sealed(out / f"{row['track_id']}_receipt.json")
                if receipt["assignment_sha256"] != sha256(out / "assignment.json"):
                    raise ValueError("Changed receipt assignment")
                if sha256(out / f"{row['track_id']}.zip") != receipt["archive_sha256"]:
                    raise ValueError("Archive changed")
                for name, expected in receipt["extracted_files"].items():
                    if sha256(out / name) != expected:
                        raise ValueError("Extracted original changed")
            print("CANDIDATE_ACQUISITION VERIFIED; no training/acceptance certification", flush=True)


if __name__ == "__main__":
    main()
