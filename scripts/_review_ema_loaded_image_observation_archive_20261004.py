"""One new read-only211 archive review; no observer import or child launch.

Event observations stay historical. Current file rehashes do not manufacture
pre-execution certification, loader causality, or a full audio-runtime gate.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/mel_ema_loaded_image_observation_20261004"
REVIEW = OUT / "independent_archive_review.json"
PINS = {
    "scripts/211_ema_decoder_loaded_image_observation.py": "4d700a762fa998450571eb993e0702c430dfed4a15d2cd5e22df200597513eeb",
    "scripts/_test_ema_decoder_loaded_image_observation.py": "3c037442ebc7020081a18b53ba867f31c01a06a19b68e8eeede406b76d33397b",
    "docs/ema_loaded_image_observation_scope_20261004.json": "c7a5c38d523455bbac4c6334d81718fdcdc0d63dc292b1e80ff7f75be8dd4a5a",
    "results/ema211_observer_unit_tests_attempt03.log": "4bc2c9d41655c337b2a9a5811ea4ec63d0a4749f42ae77cc82860338efc44cbc",
    "results/ema211_observation_attempt02.log": "0d4d8a6c29945a24b0cb9057ba3e0f76d001e11a81dec39df003b140ea2c4a61",
    "results/mel_ema_loaded_image_observation_20261004/loaded_image_evidence.json": "5dd237dfeea7861bdddc519a6979997ebe5b460a8528d2ab1b605d9c818dbb75",
}
TARGETS = {
    "ffmpeg": (r"C:\ffmpeg\bin\ffmpeg.exe", "72a489eccd008c2ec2c0a5856c5c75bc3d8bbfa90166c4566865c246445e6aa3", 87964),
    "ffprobe": (r"C:\ffmpeg\bin\ffprobe.exe", "19202b23c0043f15ad1b7bce2344f406fd52bd6efd8f995ce02e7392a1cec52f", 97548),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def typed(value):
    # Independent implementation of the sealed209 schema, including dict order.
    if type(value) is dict:
        return ["dict", [[typed(k), typed(v)] for k, v in value.items()]]
    if type(value) in (list, tuple):
        return [type(value).__name__, [typed(v) for v in value]]
    if type(value) in (str, int, bool, type(None)):
        return [type(value).__name__, value]
    raise ValueError("Not exact209 integer/string metadata")


def digest(value):
    return hashlib.sha256(json.dumps(typed(value), ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(value):
    require(type(value) is dict and "content_sha256" not in value, "No resealing")
    return {**value, "content_sha256": digest(value)}


def check_seal(value):
    require(type(value) is dict and type(value.get("content_sha256")) is str, "Own seal required")
    require(value["content_sha256"] == digest({k: v for k, v in value.items() if k != "content_sha256"}), "Own seal mismatch")


def equal(left, right, message):
    require(typed(left) == typed(right), message)


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def log_read(path):
    data = Path(path).read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig").replace("\r\n", "\n")


def verify(bindings):
    for path, expected in bindings.items():
        require(type(path) is str and type(expected) is str and re.fullmatch(r"[a-f0-9]{64}", expected), "Binding schema")
        require(sha(path) == expected, "Changed archived file: " + path)


def review():
    require(not REVIEW.exists(), "Existing review retained; no repeat/overwrite")
    rng, environment = random.getstate(), dict(os.environ)
    require(not any(k in sys.modules for k in ("torch", "numpy", "scipy", "soundfile")), "No heavy runtime imports")
    bindings = {str(ROOT / p): h for p, h in PINS.items()}
    verify(bindings)  # SHA before parsing any saved evidence.
    proof = json_read(OUT / "loaded_image_evidence.json")
    check_seal(proof)
    equal(proof["purpose"], "NONRELEASE_EMA211_VERSION_ONLY_ACTUAL_LOADED_FILE_OBSERVATION", "Purpose")
    for field, expected in {
        "actual_version_children": 2, "actual_audio_draws": 0, "student_pt_model_adam_cuda": 0,
        "closed_component_reruns": 0, "parent_python_rng_environment_unchanged": True,
        "observed_version_load_event_file_paths_and_bytes": True,
        "observed_files_authenticated_before_original_loading": False,
        "API_set_forwarder_SxS_causal_mapping_verified": False,
        "nondebug_launch_equivalence_verified": False, "conditional_audio_decode_runtime_verified": False,
        "Python_native_audio_runtime_verified": False, "historical208_actual_childpath_verified": False,
        "full_audio_backend_gate": "PENDING", "training_authorized": False, "release_selection": "NONE",
    }.items():
        equal(proof[field], expected, "Full scope/authority: " + field)
    equal([r["label"] for r in proof["rows"]], ["ffmpeg", "ffprobe"], "Complete ordered rows")
    journal_path = OUT / "actual_event_journal.jsonl"
    journal = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    cursor, unique_files = 0, {}
    for row in proof["rows"]:
        check_seal(row)
        label, (target, target_hash, pid) = row["label"], TARGETS[row["label"]]
        equal(row["argv"], [target, "-version"], "Fixed version argv")
        equal(row["pid"], pid, "Original owned PID")
        for key, expected in {"actual_child_created": True, "debug_exit_code": 0, "popen_exit_code": 0,
                "detached_on_failure": False, "error": None, "observed_file_event_count": 36,
                "held_file_handles_post_exit_equal": True}.items():
            equal(row[key], expected, "Complete successful row: " + key)
        equal(journal[cursor], {"kind": "version_request", "label": label, "argv": row["argv"], "utc": row["started_utc"]}, "Original version request journal")
        child = journal[cursor + 1]
        require(set(child) == {"kind", "label", "pid", "utc"} and child["kind"] == "child_created" and child["label"] == label, "Child-created journal schema")
        equal(child["pid"], pid, "Journal owned PID")
        cursor += 2
        events = row["events"]
        require(type(events) is list and len(events) == 44, "Entire44 event sequence")
        equal([e["code"] for e in events if e["code"] in (3, 5)], [3, 5], "One create and one exit")
        require(events[0]["code"] == 3 and events[-1]["code"] == 5, "Complete event boundaries")
        files = [e for e in events if e["code"] in (3, 6)]
        equal(len(files), 36, "Complete file events")
        require(len([e for e in events if e["code"] == 6]) == 35, "35 DLL events")
        for index, event in enumerate(events):
            equal(event["sequence"], index, "Exact event order")
            equal(event["pid"], pid, "No foreign child")
            require(type(event["tid"]) is int and event["tid"] > 0 and type(event["code"]) is int and 1 <= event["code"] <= 8, "Valid native metadata")
            equal(journal[cursor], {"kind": "debug_event", "label": label, "event": event}, "Full typed journal/file event")
            cursor += 1
            equal(event["continuation"], 0x10002, "Actual successful event continuation")
            if event["code"] in (3, 6):
                equal(event["measured_from_actual_event_handle"], True, "Actual event handle measurement")
                require(type(event["base_address"]) is int and event["base_address"] > 0, "Actual image base metadata")
                image = event["file"]
                require(set(image) == {"final_path", "bytes", "sha256", "volume_serial", "file_index", "last_write_ticks"}, "Complete actual file metadata")
                require(type(image["final_path"]) is str and image["final_path"].startswith("\\\\?\\"), "Final native file path")
                for key in ("bytes", "volume_serial", "file_index", "last_write_ticks"):
                    require(type(image[key]) is int and image[key] >= 0, "Exact file integer metadata")
                require(0 < image["bytes"] <= 256 * 1024 * 1024 and re.fullmatch(r"[a-f0-9]{64}", image["sha256"]), "Bounded actual file bytes")
                path = image["final_path"]
                if path in unique_files:
                    equal(unique_files[path], image, "Shared files full typed equality")
                unique_files[path] = image
            if event["code"] == 1:
                equal(event["exception_code"], 0x80000003, "Only initial loader breakpoint")
                equal(event["first_chance"], 1, "Only first chance")
        require(len([e for e in events if e["code"] == 1]) == 1, "Exactly one initial break")
        equal(os.path.normcase(files[0]["file"]["final_path"].removeprefix("\\\\?\\")), os.path.normcase(target), "Actual process EXE path")
        equal(files[0]["file"]["sha256"], target_hash, "Actual process EXE bytes")
        final = journal[cursor]
        require(final["kind"] == "row_final" and final["label"] == label, "Original final journal")
        final_row = seal(final["row"])
        check_seal(final_row)
        equal(final_row, row, "Both independently sealed complete rows, not single-side stripping")
        cursor += 1
        started, completed = (datetime.fromisoformat(row[k]) for k in ("started_utc", "completed_utc"))
        require(started.utcoffset().total_seconds() == completed.utcoffset().total_seconds() == 0 and 0 < (completed-started).total_seconds() < 30, "Bounded actual UTC interval")
        for stream in ("stdout", "stderr"):
            info = row[stream]
            path = OUT / (label + "_version_" + stream + ".bin")
            equal(info, {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}, "Complete output metadata")
            bindings[str(path)] = info["sha256"]
            if stream == "stderr":
                equal(info["bytes"], 0, "Empty actual stderr")
            else:
                require(path.read_bytes().startswith((label + " version 9.0.1").encode()), "Actual measured version header, not supplychain trust")
    equal(cursor, len(journal), "No omitted/additional/failure journal entries")
    require(not (OUT / "failure.json").exists(), "No successful-run failure artifact")
    require(len(unique_files) == 37, "37 unique observed physical file paths")
    for path, image in unique_files.items():
        require(Path(path).stat().st_size == image["bytes"] and sha(path) == image["sha256"], "Current post-hoc file bytes match historical event; not retro-authentication")
        bindings[path] = image["sha256"]
    for path, value in proof["bindings_sha256"].items():
        require(path not in bindings or bindings[path] == value, "No conflicting historical binding")
        bindings[path] = value
    receipt_path = OUT / "execution_receipt.json"
    receipt = json_read(receipt_path)
    for kind, chunk in (("final_unit", "edef25"), ("actual_observation", "32efd8")):
        record = receipt[kind]
        equal(record["exec_chunk"], chunk, "Actual foreground chunk")
        equal(record["actual_native_exit_code"], 0, "Actual native exit")
        equal(record["output_fully_consumed"], True, "Complete exit consumption")
        equal(record["native_exit_marker"], "NATIVE_EXIT_CODE=0", "Native exit marker")
        equal(sha(ROOT / record["log"]), record["log_sha256"], "Real archived log")
    equal(receipt["final_unit"]["tests"], 24, "Actual final unit count")
    equal(receipt["actual_observation"]["owned_child_pids"], [87964, 97548], "Only two actual own children")
    equal(receipt["actual_observation"]["actual_version_children"], 2, "Version-only exposure")
    equal(receipt["training_authorized"], False, "Not training authority")
    unit_log = log_read(ROOT / receipt["final_unit"]["log"])
    require(len(re.findall(r"^test_\d+.* \.\.\. ok$", unit_log, re.M)) == 24 and "Ran 24 tests in 0.027s" in unit_log and "\nOK\n" in unit_log, "Complete archived final24 log")
    for record in receipt["retained_attempts"]:
        bindings[str(ROOT / record["log"])] = record["log_sha256"]
    scope_path = OUT / "diagnostic_scope.json"
    scope = json_read(scope_path)
    equal(scope["bindings_sha256"], proof["bindings_sha256"], "Full original diagnostic bindings")
    equal(scope["music_inputs"], 0, "No music inputs")
    equal(scope["training_authorized"], False, "Diagnostic only")
    for path in (receipt_path, scope_path, journal_path, Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    verify(bindings)
    equal(random.getstate(), rng, "Reader Python RNG unchanged")
    equal(dict(os.environ), environment, "Reader environment unchanged")
    require(not any(k in sys.modules for k in ("torch", "numpy", "scipy", "soundfile")), "No audio/model runtime import")
    result = seal({"purpose": "NONRELEASE_EMA211_INDEPENDENT_READONLY_ARCHIVE_REVIEW",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "bindings_sha256": bindings,
        "binding_count": len(bindings), "full_evidence": copy.deepcopy(proof),
        "event_journal_full_typed_symmetric_equal": True, "own_and_nested_row_seals_checked": True,
        "observed_file_event_count": 72, "unique_observed_file_paths": 37,
        "current_posthoc_file_hashes_equal": True, "historical_preexecution_runtime_authenticated": False,
        "additional_version_children": 0, "additional_audio_draws": 0, "model_pt_adam_cuda": 0,
        "full_audio_backend_gate": "PENDING", "training_authorized": False, "release_selection": "NONE"})
    with REVIEW.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    saved = json_read(REVIEW)
    check_seal(saved)
    check_seal(saved["full_evidence"])
    for row in saved["full_evidence"]["rows"]:
        check_seal(row)
    equal(saved["full_evidence"], proof, "Complete file/embedded sealed proof")
    equal(saved, result, "Complete sealed saved review")
    verify(bindings)
    print("NEW211_READONLY_ARCHIVE_REVIEW bindings=" + str(len(bindings)) + " unique_files=37 file_events=72 children=0 audio_model_cuda=0")
    print("REVIEW_SHA256=" + sha(REVIEW))


if __name__ == "__main__":
    require(sys.argv[1:] == [], "Read-only fixed archive review")
    review()
