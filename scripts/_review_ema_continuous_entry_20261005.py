"""New independent stdlib-only archive review, no imported219/220/221/222.

No units, worker, decoder, audio, PT, Module, autograd, Adam or CUDA calls.
Each214/209/plain manifest keeps and verifies its OWN seal before full typed
file/embedded comparison. No one-sided seal removal or weakened thresholds.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/ema222_development_input_preflight_attempt01"
OUT = ROOT / "results/mel_ema_continuous_entry_monitor_20261005"
PINS = {
    "scripts/219_ema_continuous_training_core.py": "cfd19e0b4059164621c190eda4f74819e0c8b73b358396b94e8f1bbb7f3619ca",
    "scripts/_test_ema_continuous_training_core.py": "8beae43107242d96f4996ee4ba1c98e9aa24ec7ed2ccb0c8ab15817501dbbc60",
    "results/ema219_continuous_core_tests_attempt01.log": "516c9039474fd60e6eda6a14871e5509e43c71b761c8b676c27aca1f63ffff60",
    "scripts/220_ema_original_development_adapter.py": "1eb6be95980268c2d1ef239712073cbafbd8d20b11af14f82b9b1da10369c857",
    "scripts/_test_ema_original_development_adapter.py": "89b352dbadc805dae8913251f2df8a14169173f9d41b233185ece95f004ebf59",
    "results/ema220_original_DEV_adapter_tests_attempt01.log": "50430bf721d5829981f0f4c770e9e622efdce8f7c74ac3082be14f02df4b45f9",
    "scripts/221_ema_supervised_training.py": "f1eb83d4abe087a182c761aa6e2e736a07bac0084d71822b2a70816ad6f3cc65",
    "scripts/_test_ema_supervised_training.py": "a17a3a4e24a88f3e09f8b14a9e7e882111cc0c3fac09fb707a400d8f71e542b3",
    "results/ema221_supervised_training_tests_attempt01.log": "5b4758a50b932d041552dfcf19ca5b22def16f811eac31a29191f24ed9a64fcf",
    "results/ema221_supervised_training_tests_attempt02.log": "8e26e971db4c9fabd96952dbc63cd9cea28a01073a8d30a97dbed67bf51c6dac",
    "scripts/222_ema_development_input_preflight.py": "4c06f93c665003aa41935225e169109fb399c91e5eed74c964bbd3aa4e6f5f81",
    "scripts/_test_ema_development_input_preflight.py": "c2e89817d836441d1d88b849fa0ca8f998dbd2cfffb5b179c3f675be5e620128",
    "results/ema222_DEV_input_tests_attempt01.log": "233fa48d9348c97a59c200dc2b4116c3f4db5b9801e7c27965bdf93260fe8e07",
    "results/ema222_DEV_input_run_attempt01.log": "90347018c8b15bce2c765de84fc43ba7a9b1cc273ec81615141ad8908bc69043",
    "results/ema222_development_input_preflight_attempt01/supervised_preflight_result.json": "dc6110ab62bfe4ebb8e9ae94c4069eebebe687b976b33de97dbd11d901c40907",
    "results/ema222_development_input_preflight_attempt01/input_preflight.json": "570150010ff11548e4d0b8943f3f78bd8e6716865f7096141e2142f18eea3a30",
    "results/ema222_development_input_preflight_attempt01/native_event_journal.jsonl": "d44e0e234c9e50c846e1bfbc05f50bfaa6aabe7357a13e10f8bbeeb1e5a2aa06",
    "results/ema222_development_input_preflight_attempt01/decoder_native_journal.jsonl": "76bf7dd9318fb34a54ebc342d5ca2f5416028351c545f2a4fd055ef34fa4b766",
    "results/mel_lr_scale_20261004/selection_suite.json": "54acd248571d7aa00137b28b1b8e615772437f278eeea52ccbc11c103946a9d2",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plain_seal(doc):
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(doc["content_sha256"] == actual, "Own plain-acq manifest seal")


def review():
    require(not OUT.exists(), "Fresh independent review output required; do not repeat")
    for name, expected in PINS.items():
        require(sha(ROOT / name) == expected, "Entry archive changed: " + name)
    source = ROOT / "scripts/217_ema_actual_update_mechanism.py"
    require(sha(source) == "468734d23c353c805a87fa10f3c6d134a14bf789adabb2659fb63237e4bc71fd", "Closed native primitive SHA")
    spec = importlib.util.spec_from_file_location("ema_entry_reader_metadata_only", source)
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    require(not any(s in sys.modules for s in ("torch", "numpy", "scipy", "soundfile")), "Independent stdlib reader only")
    p, meta = q.p, q.g.meta
    result, file_report = read(RUN / "supervised_preflight_result.json"), read(RUN / "input_preflight.json")
    p.check_metadata(result); p.check_metadata(file_report)
    embedded = result["mechanism_report"]; p.check_metadata(embedded)
    require(p.typed_metadata(embedded) == p.typed_metadata(file_report), "Complete own-sealed file/embedded report")
    require(result["error"] is None and type(result["launcher_exit_code"]) is int and result["launcher_exit_code"] == 0
            and len(result["processes"]) == 3 and all(v["exit_code"] == 0 for v in result["processes"].values())
            and result["all_event_handles_post_exit_equal"] is True and result["training_authorized"] is False
            and result["formal_training_updates"] == 0, "Actual complete native EXIT/scope")
    bindings = {str(ROOT / k): v for k, v in PINS.items()}
    for collection in (result["preheld_source_bindings_sha256"], file_report["development_source_bindings_sha256"]):
        for name, expected in collection.items():
            require(name not in bindings or bindings[name] == expected, "Binding conflicts")
            bindings[name] = expected
    for name, expected in bindings.items():
        require(sha(name) == expected, "Full archive/source byte binding: " + name)
    original = read(ROOT / "results/mel_lr_scale_20261004/selection_suite.json"); plain_seal(original)
    candidate = copy.deepcopy(file_report["development_manifest"])
    candidate["content_sha256"] = original["content_sha256"]; plain_seal(candidate)
    require(p.typed_metadata(original) == p.typed_metadata(candidate), "Whole original31/177/93 own-sealed manifest")
    require(file_report["CPU_threads"] == 4 and file_report["CPU_Python_NumPy_RNG_unchanged"] is True
            and file_report["cuda_initialized"] is False and file_report["model_PT_Adam_forward_backward_calls"] == 0
            and file_report["actual_DEV_predictions"] == file_report["actual_Adam_steps"] == 0
            and file_report["complete_input_target_descriptor_PCM_guard_passed"] is True, "Input-only actual scope")
    lease = file_report["actual_windows_lease"]
    require(lease["write_and_delete_denied_while_held"] is True and lease["wrong_sha_refused_and_handles_closed"] is True
            and lease["refused_access_requests"] == [{"desired_access": 0x40000000, "error": 32},
                {"desired_access": 0x10000, "error": 32}]
            and sha(RUN / lease["fixture"]) == lease["sha256"] == lease["actual_held_identity"]["sha256"], "Actual lease evidence")
    events = [json.loads(line) for line in (RUN / "native_event_journal.jsonl").read_text().splitlines()]
    actual = [v for v in events if "sequence" in v]
    images = [v for v in actual if "file" in v]
    exits = {str(v["pid"]): v for v in actual if v["code"] == 5}
    require(len(images) == result["native_file_events"] == 245 and set(exits) == set(result["processes"])
            and all(v["matched_preheld_physical_identity"] is True for v in images), "Whole actual native-event accounting")
    for row in images:
        q.a.validate_native(row["file"], result["preheld_native_physical_files"])
    require(len(result["preheld_native_physical_files"]) == 246, "Real preheld runtime union")
    decoder_rows = [json.loads(line) for line in (RUN / "decoder_native_journal.jsonl").read_text().splitlines()]
    require(len(decoder_rows) == file_report["decoder_request_count"] == 38
            and sha(RUN / "decoder_native_journal.jsonl") == file_report["decoder_native_journal_sha256"], "Whole38 original decoder calls")
    manifest = q.g.read_manifest()
    for row in decoder_rows:
        meta.check_seal(row)
        require(row["error"] is None and row["exit_code"] == row["popen_exit_code"] == 0
                and row["accepted_debug_call"] is True and row["held_files_post_exit_equal"] is True
                and row["pre_spawn_runtime_files_held"] == 37 and not row["detached_on_failure"], "Actual accepted per-child native scope")
        for item in row["events"]:
            if "file" in item:
                q.g.validate_loaded(item["file"], manifest,
                    initial_target=row["request"]["executable"] if item["code"] == 3 else None)
    for name, expected in bindings.items():
        require(sha(name) == expected, "Archive/source changed through reader")
    report = p.seal_metadata({"purpose": "NONRELEASE_EMA219_222_INDEPENDENT_ENTRY_ARCHIVE_REVIEW",
        "completed_utc": q.obs.now(), "file_bindings_sha256": bindings, "bindings": len(bindings),
        "original_manifest31_177_93_complete_typed_equal": True, "actual_windows_selected_source_lease_passed": True,
        "actual_preheld_native_files": 246, "native_image_events": 245, "decoder_calls": 38,
        "actual_owned_process_exit_codes": {pid: row["exit_code"] for pid, row in result["processes"].items()},
        "new_reader_audio_PT_Model_forward_Adam_CUDA_calls": 0, "formal_training_updates": 0,
        "new_full_DEV_predictions": 0, "full_training_ready": False,
        "remaining": ["Complete remaining training-only semantic dependency review",
                      "Complete bound formal protocol/continued authorization/activation and fresh launch checks"],
        "release_selection": "NONE", "old_failed_evidence_preserved": True})
    OUT.mkdir()
    q.obs.write_new(OUT / "independent_entry_review.json", report)
    print("ENTRY_ARCHIVE_REVIEW_OK "+json.dumps({"bindings": len(bindings), "native_events": 245,
        "decoder_calls": 38, "formal_training_updates": 0, "reader_heavy_calls": 0}), flush=True)


if __name__ == "__main__":
    review()
