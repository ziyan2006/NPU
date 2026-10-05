"""New component receipt archive, never a training approval or test rerun."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/mel_ema_continuous_entry_monitor_20261005/unit_gate.json"
PINS = {
    "scripts/_review_ema_continuous_entry_20261005.py": "7c0f12afe5bb7dd45dcaff1b109df460e71db1cf6f3ce221c05f1830cefd36d8",
    "results/mel_ema_continuous_entry_monitor_20261005/independent_entry_review.json": "7eb80476919b7badd299e55a600de9f0d6b54a7e4e9b5f2a58a2b4e9276eeda9",
    "results/ema_entry_archive_review_attempt01.log": "f2ef79a791451215ac791382deb20ae5519d301fe1936d10d816deb1f1878d1e",
    "scripts/217_ema_actual_update_mechanism.py": "468734d23c353c805a87fa10f3c6d134a14bf789adabb2659fb63237e4bc71fd",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def archive():
    assert not OUT.exists(), "Preserve existing archive; no repeat"
    for name, expected in PINS.items():
        assert sha(ROOT / name) == expected, name
    spec = importlib.util.spec_from_file_location("ema_entry_archive_metadata", ROOT / "scripts/217_ema_actual_update_mechanism.py")
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    assert not any(name in sys.modules for name in ("torch", "numpy", "soundfile", "scipy"))
    review = json.loads((ROOT / "results/mel_ema_continuous_entry_monitor_20261005/independent_entry_review.json").read_text())
    q.p.check_metadata(review)
    assert review["bindings"] == 2874 and review["full_training_ready"] is False
    scope_path = ROOT / "docs/ema_continuous_entry_scope_20261005.json"
    scope = json.loads(scope_path.read_text())
    bindings = {str(ROOT / name): expected for name, expected in PINS.items()}
    bindings.update({str(path): sha(path) for path in (scope_path, ROOT / "reports/113_mel_ema_continuous_entry_recovery.md", Path(__file__).resolve())})
    logs = [("219", "ema219_continuous_core_tests_attempt01.log", 20),
            ("220", "ema220_original_DEV_adapter_tests_attempt01.log", 12),
            ("221", "ema221_supervised_training_tests_attempt02.log", 12),
            ("222", "ema222_DEV_input_tests_attempt01.log", 6)]
    for component, filename, count in logs:
        path = ROOT / "results" / filename
        text = path.read_text(encoding="utf-8-sig")
        assert f"Ran {count} tests" in text and "\nOK" in text and "FAILED" not in text
        assert scope["components"][component]["units"] == count
        bindings[str(path)] = sha(path)
    row = q.p.seal_metadata({"purpose": "NONRELEASE_EMA219_222_COMPONENT_ARCHIVE_NOT_TRAINING_ACTIVATION",
        "completed_utc": q.obs.now(), "bindings_sha256": bindings,
        "scope": scope, "independent_review_sha256": PINS["results/mel_ema_continuous_entry_monitor_20261005/independent_entry_review.json"],
        "real_tool_receipts": {"219": {"start_chunk": "05591a", "session": 66541, "completion_chunk": "95813a", "native_exit_code": 0},
            "220": {"exec_chunk": "8e1702", "native_exit_code": 0},
            "221": {"exec_chunk": "9489a6", "native_exit_code": 0, "retained_attempt01_chunk": "fc513a", "retained_attempt01_native_exit_code": 1},
            "222_units": {"exec_chunk": "5aec89", "native_exit_code": 0},
            "222_actual": {"start_chunk": "14e072", "session": 2487, "completion_chunk": "538cf3", "native_exit_code": 0},
            "independent_reader": {"exec_chunk": "ba48e2", "native_exit_code": 0}},
        "full_training_ready": False, "formal_updates": 0, "training_authorized": False, "release_selection": "NONE"})
    q.obs.write_new(OUT, row)
    q.p.check_metadata(json.loads(OUT.read_text()))
    for path, digest in bindings.items():
        assert sha(path) == digest
    print("COMPONENT_ENTRY_ARCHIVE_OK bindings="+str(len(bindings))+" training_authorized=false formal_updates=0", flush=True)


if __name__ == "__main__":
    archive()
