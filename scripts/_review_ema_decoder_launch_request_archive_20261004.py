"""New210 read-only archive review. No Popen/request capture/audio/unit rerun."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
from unittest.mock import patch
import subprocess

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/mel_ema_decoder_launch_request_monitor_20261004"
PINS = {
    "scripts/209_ema_decoder_target_guard.py": "6b904f9f1b8497d16f3ecca85d9122b000604f4e357456eef0a0e9ddf28fe7d2",
    "scripts/210_ema_decoder_launch_request.py": "47bc072538ca057e73c80c0b83d86cbebcabd42227eda4b4095e47de158acd65",
    "scripts/_test_ema_decoder_launch_request.py": "62480358d843ddeb37c10136a5d52d28accecc76ffe65bae739b2f80063732c3",
    "docs/ema_decoder_launch_request_unit_scope_20261004.json": "8b00796c7b0c2b0d6d2bf67636f9b73a467d2db1bacc74f4774a447a1a1867d4",
    "results/ema210_launch_request_unit_tests_attempt01.log": "e53238e60a1c0391b2f1ba2ef1a9f9dcb09cac466ff810ae485f0511aa68b96c",
    "results/mel_ema_decoder_launch_request_monitor_20261004/request_interception_evidence.json": "fd1937ada6f393c824133be7668dc72a0502701cfcc46fe6b8b0b1aab52dc89a",
    "results/mel_ema_decoder_launch_request_monitor_20261004/unit_execution_receipt.json": "e60a53e9302c8e7dd7682a347f8c503a73845b2bd549499b66937b67466e534e",
}


def sha(path):
    with open(path, "rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected: raise ValueError("Changed archive binding: " + str(path))


def review():
    before_rng = random.getstate()
    bindings = {str(ROOT / p): h for p, h in PINS.items()}
    verify(bindings)  # SHA before any saved JSON interpretation.
    spec = importlib.util.spec_from_file_location("ema210_archive_typed209", ROOT / "scripts/209_ema_decoder_target_guard.py")
    meta = importlib.util.module_from_spec(spec); spec.loader.exec_module(meta)
    proof = json.loads((OUT / "request_interception_evidence.json").read_text(encoding="utf-8-sig"))
    meta.check_seal(proof)
    expected_labels = ["musdb0", "musdb1", "pseudo_probe0", "pseudo_decode0", "version0ffmpeg", "version0ffprobe"]
    meta.require(meta.typed([v["original"]["label"] for v in proof["rows"]]) == meta.typed(expected_labels), "Entire original row order")
    for row in proof["rows"]:
        meta.check_seal(row)
        original, policy, old, new = (row[k] for k in ("original", "prospective_policy", "intercepted_original", "intercepted_explicit"))
        target = meta.EXPECTED[original["argv"][0]]["target"]
        expected_policy = {"label": original["label"], "source_line": original["source_line"], "argv": [target, *original["argv"][1:]],
            "kwargs": copy.deepcopy(original["kwargs"]), "executable": target, "shell": False, "close_fds": True,
            "creationflags": subprocess.CREATE_NO_WINDOW,
            "startupinfo": {"dwFlags": subprocess.STARTF_USESHOWWINDOW, "wShowWindow": subprocess.SW_HIDE},
            "env": None, "cwd": None, "actual_spawn_authorized": False}
        meta.require(meta.typed(policy) == meta.typed(expected_policy), "Full typed prospective policy")
        for request, argv, app, flags, hide in ((old, original["argv"], None, 0, 0), (new, policy["argv"], target, subprocess.CREATE_NO_WINDOW, subprocess.STARTF_USESHOWWINDOW)):
            meta.require(set(request) == {"application_name", "command_line", "process_security", "thread_security", "inherit_handles", "creationflags", "environment", "cwd", "startupinfo"}, "Complete request schema")
            meta.require(meta.typed(request["application_name"]) == meta.typed(app) and request["command_line"] == subprocess.list2cmdline(argv), "Exact application and serialization")
            meta.require(type(request["creationflags"]) is int and request["creationflags"] == flags, "Exact creation flags")
            for key in ("environment", "cwd", "process_security", "thread_security"): meta.require(request[key] is None, "Inherited no overrides")
            startup = request["startupinfo"]
            meta.require(type(startup["dwFlags"]) is int and startup["dwFlags"] == (subprocess.STARTF_USESTDHANDLES | hide), "Exact pipe/visibility flags")
            meta.require(type(startup["wShowWindow"]) is int and startup["wShowWindow"] == 0, "Window metadata")
            meta.require(type(request["inherit_handles"]) is int and request["inherit_handles"] == 1, "Restricted pipe handle inheritance")
            meta.require(startup["hStdOutput"] in startup["lpAttributeList"]["handle_list"], "Actual stdout handle record")
    for field, expected in {"request_interceptions": 12, "native_CreateProcess_calls": 0, "actual_child_processes": 0,
                           "actual_audio_draws": 0, "model_or_optimizer_or_pt_load": 0, "cuda_initialized": False,
                           "full_audio_backend_gate": "PENDING", "training_authorized": False, "release_selection": "NONE",
                           "historical208_actual_image_path_observed": False, "explicit_policy_actual_spawn_verified": False,
                           "runtime_transitive_loaded_modules_verified": False}.items():
        meta.require(meta.typed(proof[field]) == meta.typed(expected), "Scope/authority type: " + field)
    log = (ROOT / "results/ema210_launch_request_unit_tests_attempt01.log").read_text(encoding="utf-8-sig")
    meta.require(len(re.findall(r"^test_\d+.* \.\.\. ok$", log, re.M)) == 25 and "Ran 25 tests in 0.786s" in log, "Full new25 log")
    receipt = json.loads((OUT / "unit_execution_receipt.json").read_text(encoding="utf-8-sig"))
    meta.require(receipt["actual_native_exit_code"] == 0 and receipt["exec_chunk"] == "ed8bd2" and receipt["output_fully_consumed"] is True and receipt["native_exit_marker"] == "NATIVE_EXIT_CODE=0", "Actual foreground receipt, not inferred from OK")
    for path, digest in proof["bindings_sha256"].items():
        meta.require(path not in bindings or bindings[path] == digest, "No conflicting archive binding")
        bindings[path] = digest
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    verify(bindings)
    meta.require(before_rng == random.getstate() and not any(v in sys.modules for v in ("torch", "numpy", "soundfile", "scipy")), "No numerical/audio imports or RNG changes")
    return meta.seal({"purpose": "NONRELEASE_EMA210_INDEPENDENT_ARCHIVE_REVIEW_NOT_REQUEST_OR_UNIT_RERUN",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "bindings_sha256": bindings,
        "binding_count": len(bindings), "full_evidence": copy.deepcopy(proof), "own_and_nested_seals_checked": True,
        "full_typed_policy_and_original_serialization_checked": True, "additional_request_interceptions": 0,
        "actual_child_processes": 0, "actual_audio_draws": 0, "model_pt_adam_cuda": 0, "release_selection": "NONE"})


if __name__ == "__main__":
    with patch.object(subprocess, "Popen", side_effect=AssertionError("ARCHIVE_ONLY_NO_REQUEST")):
        print(json.dumps(review(), ensure_ascii=True, allow_nan=False))
