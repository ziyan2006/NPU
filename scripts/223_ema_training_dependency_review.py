"""One new stdlib-only semantic/whole structured dependency review.

No historical CLI, decoder, PT, Torch, teacher, optimizer or CUDA operation.
This reads every JSON field/record/cache entry, not truncated text excerpts.
Historical cache PTs are not used by this waveform training route. Their
metadata is preserved; this does not claim revalidation of unused cache PTs.
Source bytes are guarded per actual draw by219; native images by221/213.
"""
from __future__ import annotations
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/ema223_dependency_review_attempt01"
PINS = {
    "scripts/214_ema_audio_backend_preparation.py": "d16edc51c036597b6ac1fc2966c3063c19ce90304f110723557471133a71a713",
    "results/mel_ema_continuous_entry_monitor_20261005/independent_entry_review.json": "7eb80476919b7badd299e55a600de9f0d6b54a7e4e9b5f2a58a2b4e9276eeda9",
    "results/ema222_development_input_preflight_attempt01/input_preflight.json": "570150010ff11548e4d0b8943f3f78bd8e6716865f7096141e2142f18eea3a30",
    "results/training_protocol_20261001/dataset_lock.json": "36a9be8ade7822969a7fed661c8f226f49b89e8ece90e1e6192ae0ae1b4b2c38",
    "results/paired_exploratory_import_20261003_r2/approval.json": "aa2eb50b82635cefb46fd4fc2d3cae611154eb064b26f2568968b756b139b7c6",
}
# Read to EOF and reviewed by the main agent: original graph/matrices/PCM,
# scalar loss, sampler, role policy, state/RNG and new integration call sites.
SEMANTIC_SOURCES = (
    "09_target_model.py", "11_smoke_train.py", "13_ab_compare.py", "23_build_true_stem_cache.py",
    "110_train_residual_ablation.py", "119_model_selection_suite.py", "120_train_layout_control.py",
    "125_lock_training_data.py", "126_verify_training_baseline.py", "128_acquire_cambridge_candidates.py",
    "130_acquire_teacher_pilot.py", "131_run_teacher_pilot.py", "134_generate_teacher_library.py",
    "136_prepare_distillation_data.py", "143_paired_distillation_mechanics.py", "146_evaluate_paired_development.py",
    "147_paired_device_mechanics.py", "149_prepare_exploratory_import.py", "150_train_paired_exploration.py",
    "159_source_projection_auxiliary.py", "170_instrumental_protection_loss.py", "176_accompaniment_component_loss.py",
    "194_train_mel_lr_scale.py", "203_ema_shadow_state.py",
    "204_mel_ema_source_contract.py", "205_ema_cpu_state_transaction.py", "206_ema_live_input_schedule.py",
    "207_ema_live_cpu_state.py", "208_ema_authenticated_audio_input.py", "209_ema_decoder_target_guard.py",
    "210_ema_decoder_launch_request.py", "211_ema_decoder_loaded_image_observation.py",
    "212_ema_single_trajectory_engine.py", "213_ema_direct_decoder_runtime.py", "214_ema_audio_backend_preparation.py",
    "215_ema_model_audio_audit.py", "216_ema_cuda_runtime_observation.py", "217_ema_actual_update_mechanism.py",
    "218_ema_mechanism_runtime_recovery.py", "219_ema_continuous_training_core.py",
    "220_ema_original_development_adapter.py", "221_ema_supervised_training.py",
)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def plain(doc):
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    require(hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest() == doc["content_sha256"], "Own historical plain seal")

def walk(value, counts):
    counts[type(value).__name__] += 1
    if type(value) is dict:
        for key, item in value.items():
            require(type(key) is str, "JSON keys")
            walk(key, counts); walk(item, counts)
    elif type(value) is list:
        for item in value:
            walk(item, counts)
    elif type(value) is float:
        require(math.isfinite(value), "Finite complete tree")
    else:
        require(type(value) in (str, int, bool, type(None)), "Exact JSON primitives")

def review():
    require(not OUT.exists(), "Fresh dependency review; preserve existing evidence")
    # Validate214 metadata implementation from the already accepted parent,
    # before import. No native object or worker is constructed here.
    parent = read(ROOT / next(k for k in PINS if k.endswith("independent_entry_review.json")))
    authority = parent["file_bindings_sha256"]
    relative = "scripts/214_ema_audio_backend_preparation.py"
    expected = authority[str((ROOT / relative).resolve())]
    require(sha(ROOT / relative) == expected == PINS[relative], "Accepted214 metadata bytes")
    #214's hash is parent-authoritative, not a guessed version/path label.
    bindings = {str((ROOT / k).resolve()): v for k, v in PINS.items() if k != relative}
    bindings[str((ROOT / relative).resolve())] = expected
    for name, digest in bindings.items():
        require(sha(name) == digest, "Fixed review parent/metadata: " + name)
    spec = importlib.util.spec_from_file_location("ema223_metadata_only", ROOT / relative)
    p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
    p.check_metadata(parent)
    actual = read(ROOT / "results/ema222_development_input_preflight_attempt01/input_preflight.json")
    p.check_metadata(actual)
    lock = read(ROOT / "results/training_protocol_20261001/dataset_lock.json"); plain(lock)
    authority = dict(authority)
    for name, digest in lock["dependency_sha256"].items():
        authority[str((ROOT / "scripts" / name).resolve())] = digest
    # Accepted integration sources have explicit SHA pin edges to their state,
    # loss and historical helpers. Follow those DATA dictionaries by AST;
    # never import/execute the helpers to discover a dependency.
    visited = set()
    while True:
        todo = [name for name in authority if name not in visited and
                ((name.endswith(".py") and Path(name).is_relative_to(ROOT / "scripts")) or name.endswith("approval.json"))]
        if not todo:
            break
        for name in todo:
            visited.add(name)
            require(sha(name) == authority[name], "Parent/source pin edge bytes")
            if name.endswith("approval.json"):
                previous = read(name); plain(previous)
                for path, digest in previous["bindings_sha256"].items():
                    require(path not in authority or authority[path] == digest, "Conflicting original approval byte binding")
                    authority[path] = digest
                continue
            tree = ast.parse(Path(name).read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and type(k.value) is str and k.value.startswith(("scripts/", "results/")) and isinstance(v, ast.Constant) and type(v.value) is str and re.fullmatch("[0-9a-f]{64}", v.value):
                        path = str((ROOT / k.value).resolve())
                        require(path not in authority or authority[path] == v.value, "Conflicting immutable source pin edges")
                        authority[path] = v.value
    source_review = {}
    for name in SEMANTIC_SOURCES:
        path = (ROOT / "scripts" / name).resolve()
        require(str(path) in authority, "Source absent from accepted native/source parent: " + name)
        digest = authority[str(path)]
        require(sha(path) == digest, "Reviewed source changed: " + name)
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text)
        source_review[name] = {"sha256": digest, "complete_lines": len(text.splitlines()),
            "definitions": [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]}
        bindings[str(path)] = digest
    require(len(lock["records"]) == 198 and len(lock["files"]) == 1025 and lock["blind_status"] == "missing" and lock["blind_ids"] == [], "Whole existing lock, no blind promotion")
    namespace = {"Path": Path, "re": re, "unicodedata": unicodedata}
    tree = ast.parse((ROOT / "scripts/110_train_residual_ablation.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "composition_key")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "original110[223title-key]", "exec"), namespace)
    key = namespace["composition_key"]
    groups, roles, files, inventory = Counter(), {r: set() for r in ("train", "development", "regression")}, {r: set() for r in ("train", "development")}, []
    for row in lock["records"]:
        require(row["domain"] in ("musdb", "mir1k", "instrumental", "onair") and row["role"] in files, "Locked role/domain")
        composition = key(row["track_id"])
        require(composition and composition not in roles[row["role"]], "Unique conservative composition key")
        roles[row["role"]].add(composition)
        groups[(row["domain"], row["role"])] += 1
        selected = {name for field in ("mix_files", "vocal_files", "stem_files") for name in row[field]}
        require(selected and selected <= lock["files"].keys(), "All complete source references present")
        files[row["role"]].update(selected)
        inventory.append({"domain": row["domain"], "id": row["track_id"], "role": row["role"], "source_files": len(selected)})
    roles["regression"] = {key(v) for v in lock["known_regression_ids"]}
    require(not any(roles[a] & roles[b] for i, a in enumerate(roles) for b in list(roles)[i+1:]) and not files["train"] & files["development"], "TRAIN/DEV/regression roles and physical file lists disjoint")
    expected_groups = {("musdb", "train"): 73, ("mir1k", "train"): 81, ("instrumental", "train"): 11,
        ("onair", "train"): 2, ("musdb", "development"): 19, ("mir1k", "development"): 9, ("instrumental", "development"): 3}
    require(dict(groups) == expected_groups, "Complete original split counts")
    require(files["train"] | files["development"] == lock["files"].keys(), "No omitted lock file")
    for name, info in lock["files"].items():
        require(Path(name).is_absolute() and re.fullmatch("[0-9a-f]{64}", info["sha256"]) and type(info["bytes"]) is int and info["bytes"] > 0
            and info["duration_s"] > 0 and info["sample_rate"] > 0 and info["channels_per_stream"] > 0, "Complete locked audio metadata")
    documents, cache_review = [lock], {}
    for domain, reference in lock["manifest_provenance"].items():
        if "path" not in reference:
            continue
        require(sha(reference["path"]) == reference["sha256"], "Historical cache manifest bytes")
        bindings[reference["path"]] = reference["sha256"]
        cache = read(reference["path"]); documents.append(cache)
        require(type(cache["tracks"]) is list, "Whole actual cache track LIST, not an assumed mapping")
        by_id = {r["track_id"]: r for r in cache["tracks"]}
        require(len(by_id) == len(cache["tracks"]), "Unique cache track IDs")
        entries = Counter(); cache_files = set()
        for row in cache["tracks"]:
            require(row["split"] in ("train", "holdout"), "Original cache split")
            for item in row["cache_items"]:
                require(item["file"] not in cache_files and Path(item["file"]).name == item["file"]
                    and item["start_s"] >= 0 and 0 < item["dur_s"] <= cache["segment_s"], "Every unused historical cache descriptor")
                cache_files.add(item["file"]); entries[row["split"]] += 1
        require(dict(entries) == cache["counts"], "All cache descriptors/counts consumed")
        selected = [r for r in lock["records"] if r["domain"] == domain]
        for row in selected:
            prior = by_id[row["track_id"]]
            for field in ("mix_files", "vocal_files", "stem_files"):
                require(row[field] == prior.get(field, []), "Exact source arrays, no cached remixes used")
        cache_review[domain] = {"tracks": len(by_id), "cache_descriptors": sum(entries.values()), "locked_tracks": len(selected), "cache_PT_loaded": 0}
    approval = read(ROOT / "results/paired_exploratory_import_20261003_r2/approval.json"); plain(approval); documents.append(approval)
    require(len(approval["pair_ids"]) == len(set(approval["pair_ids"])) == 24 and approval["deployment_authorized"] is False, "Fixed24 input approval, not this training authorization")
    require([r["song_id"] for r in approval["approved_records"]] == approval["pair_ids"] and all(r["exploratory_eligible"] is True and r["deployment_eligible"] is False for r in approval["approved_records"]), "Complete approved IDs")
    snapshot_identities = []
    for teacher, reference in approval["snapshots"].items():
        require(sha(reference["path"]) == reference["sha256"], "Whole preserved snapshot bytes")
        bindings[reference["path"]] = reference["sha256"]
        snapshot = read(reference["path"]); plain(snapshot); documents.append(snapshot)
        require([r["song_id"] for r in snapshot["records"]] == approval["pair_ids"] and snapshot["training_authorized"] is False, "Original pseudo role/order not promoted")
        identity = []
        for row in snapshot["records"]:
            require(row["role"] == "pseudo_label_train_candidate" and row["training_eligible"] is False and set(row["label_files"]) == {"vocals.wav", "accompaniment.wav"}, "Pseudo is not final truth")
            require(row["source"]["source_channels"] == 2 and row["source"]["samples"] >= 89856, "Full original pseudo source geometry")
            require(key(row["source"]["path"]) not in roles["train"] | roles["development"] | roles["regression"], "Pseudo/source composition disjoint")
            identity.append((row["song_id"], row["source"]))
        snapshot_identities.append(p.typed_metadata(identity))
    require(len(snapshot_identities) == 2 and snapshot_identities[0] == snapshot_identities[1], "Preserved paired source identities; training uses ONLY kim target")
    counts = Counter()
    for document in documents:
        walk(document, counts)
    require(not any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile")), "No heavy imports or runtime operations")
    for name, digest in bindings.items():
        require(sha(name) == digest, "Changed through new review")
    report = p.seal_metadata({"purpose": "NONRELEASE_EMA223_COMPLETE_CALLED_DEPENDENCY_REVIEW", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "complete_dependency_review_passed": True, "bindings_sha256": bindings, "semantic_source_review": source_review,
        "full_original_record_inventory": inventory, "full_tree_node_counts": dict(counts), "records": 198, "original_file_metadata": 1025,
        "whole_legacy_cache_manifests": cache_review, "pseudo_songs": 24, "active_teacher": "kim_melband", "OnAir_used": False,
        "original_evaluator_binding": actual["original_evaluator_binding"], "input_backend_actual_evidence": str(ROOT / "results/ema222_development_input_preflight_attempt01/input_preflight.json"),
        "semantic_conclusions": ["Original CPU/CUDA FP32 graph/matrices and original194 scalar/six-microbatch order unchanged",
            "Original143 local seed/cursor +136 crop/common-gain;3true+3pseudo;LR1/kill32/aux.2/instrumental4/denominator6",
            "Single raw/original Adam;203 shadow no feedback;206 original raw cumulative qualification;old stop/legacy preserved",
            "Original119/194 DEV policy/31songs177views;frozen44 archive;2newstage pairs;no ranking by loss",
            "Imports of teacher/acquisition modules do not call asset/network/model generation CLIs",
            "Complete JSON read is structured machine inventory;not a claim of manually reading each SHA string",
            "Unused legacy cache PTs not loaded;selected actual waveform/source/native bytes are revalidated and held per future draw"],
        "new_audio_PT_model_forward_backward_Adam_CUDA_calls": 0, "formal_training_updates": 0,
        "training_authorized": False, "release_selection": "NONE"})
    OUT.mkdir()
    with (OUT / "dependency_review.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
    print("COMPLETE_DEPENDENCY_REVIEW_OK "+json.dumps({"sources": len(source_review), "records": 198, "file_metadata": 1025, "tree_nodes": sum(counts.values()), "heavy_calls": 0}), flush=True)

if __name__ == "__main__":
    review()
