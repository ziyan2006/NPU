"""One NEW read-only evidence review, never constructs209 or draws208.

Reads already-written metadata/hashes, verifies separate own typed seals,
retains every import/symbol, and checks full file/embedded symmetric identity.
Only writes its own fresh isolated certificate after all checks pass.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/mel_ema_decoder_target_monitor_20261004/independent_archive_review.json"
PINS = {
    "scripts/209_ema_decoder_target_guard.py": "6b904f9f1b8497d16f3ecca85d9122b000604f4e357456eef0a0e9ddf28fe7d2",
    "scripts/_test_ema_decoder_target_guard.py": "6ffadcb649ed1d594f821b083cded6d9cf7193bf01856f74eae51dd57a9c02d2",
    "docs/ema_decoder_target_unit_scope_20261004.json": "767c96fff50cf8f6bea58f754a65ed6d7edc6e7e69a4a5096953d011d85f6699",
    "results/ema209_target_unit_tests_attempt01.log": "5d290a90e59f3dbb9453848960a65830525de19279345d7ab50e960ee57481a7",
    "results/ema209_actual_target_hold_attempt01.json": "f8c0b1c199e4f77fed650a833bf1281d3e4b795dd693cfd9148a5505592c6454",
    "results/mel_ema_decoder_target_monitor_20261004/unit_execution_receipt.json": "6866afa242685bf93bd9f192c0acbb3537766d0f81d4210815228ac2c92b0fc9",
    "scripts/203_ema_shadow_state.py": "7fb5fd1b3c5f66f3e9a2ebf0b020997c3454c8cf334e90de03282ebe78b939ae",
    "scripts/204_mel_ema_source_contract.py": "e36f6045ea3eec18bc28dd89c5b521bbc887a55a900425b7b55d3dec4ebf2ea9",
    "scripts/205_ema_cpu_state_transaction.py": "8a8f2da566b664abfb19e6c24fe5fb4bf78bd61c13408ebf385b5cf425cd90f5",
    "scripts/206_ema_live_input_schedule.py": "eaf1112263df6f74f97ea3e9f9a262db401dd24a819197ce0dbabdc6333d61ba",
    "scripts/207_ema_live_cpu_state.py": "7a1ab6a2dbcf9b2af99030784b34bcf6c2a73732227e0abadbb0cf2f88f18d90",
    "scripts/208_ema_authenticated_audio_input.py": "8d208e34ed1ce6b18997c64d75ad1aa03d67401b7752fa8d995da23d64982825",
    "scripts/_test_ema_authenticated_audio_input.py": "43f3fe3fdf6c389472d0f652be28a4ddbd262fc4e40cd927f533ef9a85051c4c",
    "docs/ema_authenticated_audio_unit_scope_20261004.json": "df91a1517a124a606255848d257c1b6e1ac7a4dd25788a68e62ab88fcfc3d573",
    "results/ema208_unit_tests_attempt01.log": "51777a1731a86d431db20398aa9a9c8477aaba9d1b4cf1910a0e61c09e2ac81c",
    "reports/108_mel_ema_actual_audio_input_progress.md": "dfea10a5c121c2ca68ac29e33bf8c0d11f49abf2f7449980dc71fa26ab8f68a2",
    "results/mel_ema_audio_backend_monitor_20261004/unit_gate.json": "3fb6fd6610f8290a9ed5479d2ce1177e9445798902b4e8256309b20b908c0fc1",
    "results/mel_ema_audio_backend_monitor_20261004/progress.json": "53857dc1ba02a43543e730272668293748c8cca63b81d0b1eee31b3941432bbd",
    "results/mel_ema_audio_backend_monitor_20261004/actual_input_replay_evidence.json": "d198b7f32b0b208645496de0005a929af11174d757e89ee5f366af4b01387a3a",
    "results/mel_ema_audio_backend_monitor_20261004/decoder_chain_scope_review.json": "9408cfad301350237863cbe5a58e74ae45b8d156e40349cc5ba9a0456bea226d",
    "results/mel_ema_audio_backend_monitor_20261004/independent_archive_review.json": "d1fe276c759dfbbcc4b1c0425426f50c20d8d0151cb974a5541ee9a0c3a73bc5"
}


def require(value, message):
    if not value:
        raise ValueError(message)


def exact_tree(value):
    # Independent implementation, preserving dict order and bool/int/list/tuple.
    kind = type(value)
    if kind is dict:
        pairs = []
        for key in value:
            pairs.append([exact_tree(key), exact_tree(value[key])])
        return ["dict", pairs]
    if kind is list or kind is tuple:
        return [kind.__name__, list(map(exact_tree, value))]
    if kind is str or kind is int or kind is bool or value is None:
        return [kind.__name__, value]
    raise ValueError("Unsupported exact209 metadata type")


def own_hash(doc):
    body = {key: value for key, value in doc.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(exact_tree(body), ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def own_seal(doc):
    require(type(doc) is dict and type(doc.get("content_sha256")) is str, "209 own seal required")
    require(own_hash(doc) == doc["content_sha256"], "209 independent own seal mismatch")


def main():
    require(not OUT.exists() and OUT.parent.is_dir(), "Fresh review in new isolated monitor only")
    before = {key: hashlib.sha256((ROOT / key).read_bytes()).hexdigest() for key in PINS}
    require(before == PINS, "New209 or sealed prior binding mismatch; do not rewrite")
    receipt = json.loads((ROOT / "results/mel_ema_decoder_target_monitor_20261004/unit_execution_receipt.json").read_text(encoding="utf8"))
    native = receipt["exec_command_result"]
    require(native["chunk_id"] == "be6ebb" and native["exit_code"] == 0 and
            "NATIVE_EXIT_CODE=0" in native["output"] and "Ran 35 tests in 1.832s" in native["output"] and
            "\nOK\n" in native["output"].replace("\r\n", "\n"), "Real consumed native0/35-test receipt")
    require(receipt["wmi_launch"] is False and receipt["detached_exit_file"] is None,
            "Foreground receipt, no fabricated detached exit")
    evidence = json.loads((ROOT / "results/ema209_actual_target_hold_attempt01.json").read_text(encoding="utf8"))
    own_seal(evidence)
    contract = evidence["contract"]
    own_seal(contract)
    require(contract["complete_audio_runtime_authenticated"] is False and
            contract["retroactive208_target_authentication"] is False and
            contract["training_authorized"] is False, "No partial evidence may grant full authority")
    require(evidence["constructor_actual_read_only"] == 1 and evidence["actual_empty_held_scopes"] == 1,
            "Exactly one actual metadata-only guard/hold")
    for key in ("actual_audio_draws", "decoder_launches", "student_pt_loads", "model_forwards",
                "adam_construction_or_steps", "student_updates"):
        require(type(evidence[key]) is int and evidence[key] == 0, "No audio/model/training operation")
    require(evidence["cuda_initialization"] is False and evidence["full_audio_runtime_gate"] == "PENDING",
            "CUDA/full runtime remains pending")
    require(len(contract["bindings_sha256"]) == 5 and len(contract["pe_import_inventory"]) == 3,
            "Complete direct five-file/three-image inventory")
    inventory = {}
    for name, row in contract["pe_import_inventory"].items():
        require(row["machine"] == 0x8664 and row["loader_resolution_authenticated"] is False,
                "Only actual x64 static inventory, not loader resolution")
        inventory[name] = {
            "normal_dll_count": len(row["normal"]), "delay_dll_count": len(row["delay"]),
            "normal_symbol_count": sum(len(dll["symbols"]) for dll in row["normal"]),
            "delay_symbol_count": sum(len(dll["symbols"]) for dll in row["delay"]),
            "dynamic_loader_symbols": copy.deepcopy(row["dynamic_loader_symbols"])
        }
    # Review complete trees, not a subset of summary fields. Both sides retain own seals.
    embedded = copy.deepcopy(evidence)
    own_seal(embedded)
    own_seal(embedded["contract"])
    require(exact_tree(evidence) == exact_tree(embedded), "Full file/embedded typed symmetry")
    after = {key: hashlib.sha256((ROOT / key).read_bytes()).hexdigest() for key in PINS}
    require(after == before, "Evidence or sealed dependency mutated during review")
    require(not any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile")),
            "Pure metadata review must not import numerical/audio runtime")
    result = {"schema": 1, "purpose": "NONRELEASE_EMA209_TARGET_ONLY_ARCHIVE_REVIEW",
        "reviewed_utc": datetime.now(timezone.utc).isoformat(), "bindings_sha256": before,
        "all_bindings_unchanged_before_after": True, "binding_count": len(before),
        "complete_original_evidence": embedded, "static_import_inventory_summary": inventory,
        "own_and_nested_contract_seals_independently_checked": True,
        "complete_file_embedded_typed_symmetric_comparison": True,
        "new_guard_constructions": 0, "decoder_launches": 0, "actual_audio_draws": 0,
        "student_pt_loads": 0, "model_forwards": 0, "adam_construction_or_steps": 0,
        "student_updates": 0, "cuda_initialization": False,
        "full_audio_backend_gate": "PENDING", "formal_training_authorized": False,
        "release_selection": "NONE"}
    result["content_sha256"] = own_hash(result)
    own_seal(result)
    with OUT.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    # Read the actual saved review, validate each own seal, then compare full trees.
    saved = json.loads(OUT.read_text(encoding="utf8"))
    own_seal(saved)
    own_seal(saved["complete_original_evidence"])
    own_seal(saved["complete_original_evidence"]["contract"])
    require(exact_tree(result) == exact_tree(saved), "Saved full review typed round trip")
    require(exact_tree(evidence) == exact_tree(saved["complete_original_evidence"]),
            "Original file and saved embedded evidence differ")
    print(json.dumps({"binding_count": len(before), "review": str(OUT),
        "review_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "content_sha256": result["content_sha256"], "inventory": inventory,
        "audio_draws": 0, "new_target_guards": 0, "full_audio_backend_gate": "PENDING"}))


if __name__ == "__main__":
    main()
