"""Build/check one complete fixed500 activation, stdlib/metadata only.

No Torch/PT, decoder, units, historical CLI or training subprocess. Build does
not launch. Check performs the necessary fresh resource/duplicate preflight.
The worker independently checks activation, source runtime/RNG and all guards.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / "results/ema_training_activation_20261005"
OUT = ROOT / "results/mel_ema_single_trajectory_20261005"
PINS = {
    "scripts/221_ema_supervised_training.py": "f1eb83d4abe087a182c761aa6e2e736a07bac0084d71822b2a70816ad6f3cc65",
    "scripts/217_ema_actual_update_mechanism.py": "468734d23c353c805a87fa10f3c6d134a14bf789adabb2659fb63237e4bc71fd",
    "scripts/223_ema_training_dependency_review.py": "125c478b3d252bc92828612bbf5900017da0147ff415b9e2f5ca7233ced9d82b",
    "results/ema223_dependency_review_attempt01/dependency_review.json": "f52df70ba45328ca610aa8aa4d6532a7d65ef82623f592c9a17b9a663affd42c",
    "results/ema223_dependency_review_attempt03.log": "026929b5db2f5b655b9494b546369bc303ca25d605a5648f361400c53b7f39b6",
    "results/mel_ema_continuous_entry_monitor_20261005/independent_entry_review.json": "7eb80476919b7badd299e55a600de9f0d6b54a7e4e9b5f2a58a2b4e9276eeda9",
    "results/mel_ema_continuous_entry_monitor_20261005/unit_gate.json": "efe4ae18437033758f0ef978e59b6f43257c6b818a03bfac5fd5161a2675f877",
    "results/ema222_development_input_preflight_attempt01/input_preflight.json": "570150010ff11548e4d0b8943f3f78bd8e6716865f7096141e2142f18eea3a30",
    "results/ema218_actual_cpu_mechanism_attempt01/mechanism_result.json": "35973a31bc88b6c591a93e9beb8a96a1d7a2e18a509e67366575df632e4a81fb",
    "results/ema218_actual_cuda_mechanism_attempt01/mechanism_result.json": "6cc0734a3b23d7b46182e7739d18cb7b4135ccdd029db6394f94e83d6404e8ad",
    "results/ema218_actual_cpu_mechanism_attempt01/independent_readonly_review.json": "ef7d20c28d7a4da6410f7aada441916deccdc7559ba5fb80ee936f3cf758c7e2",
    "results/ema218_actual_cuda_mechanism_attempt01/independent_readonly_review.json": "52bc9186e893900a4a926bc124f0f4696e4a10c0cda55fd4d8c78df97362f320",
    "results/ema215_real_model_audio_audit_attempt01/independent_readonly_review.json": "57d98b07712eeaec630b24397205b142b5cea42d2f93f74e7726b6013ea81dd4",
    "results/ema216_cuda_runtime_observation_attempt01/independent_readonly_review.json": "bc3954c9f0ce9f6bbbd8d3d68c436d6dddf1874b30dc794589bb9c7cc1cff89f",
    "reports/102_mel_ema_shadow_trial_plan.md": "39256e70d7e8ce4675102711c3b4595b0b81f315410bc3602df5677c16f784c8",
    "reports/112_mel_ema_actual_cpu_cuda_mechanism_recovery.md": "f0a293e2c2125adebd85e6e0394e0c9798cc95829b9436f1ea2c2706587625b9",
    "reports/113_mel_ema_continuous_entry_recovery.md": "8dfd468be50e267f0bfcaa50f6c69d2fe8820536a9735f3f7106d841fa0622f5",
    "scripts/141_process_exit_capture.ps1": "fdd6a7bf690623472f0d270dc6762807a3ca52f08c0a99bb7e08b151f409fde3",
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load(name, relative):
    require(relative in PINS and sha(ROOT / relative) == PINS[relative], "Exact metadata source SHA")
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def sources():
    for name, digest in PINS.items():
        require(sha(ROOT / name) == digest, "Changed accepted evidence/source: " + name)
    q = load("ema225_native_metadata", "scripts/217_ema_actual_update_mechanism.py")
    t = load("ema225_supervisor_metadata", "scripts/221_ema_supervised_training.py")
    require(not any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile")), "No heavy imports")
    return q, t

def accepted(q):
    audit, runtime = q.gate()  # Read-only actual parent gates, NOT their workers/verify CLIs.
    parent = read(ROOT / "results/mel_ema_continuous_entry_monitor_20261005/independent_entry_review.json")
    units = read(ROOT / "results/mel_ema_continuous_entry_monitor_20261005/unit_gate.json")
    dependency = read(ROOT / "results/ema223_dependency_review_attempt01/dependency_review.json")
    inputs = read(ROOT / "results/ema222_development_input_preflight_attempt01/input_preflight.json")
    for doc in (parent, units, dependency, inputs):
        q.p.check_metadata(doc)
    require(dependency["complete_dependency_review_passed"] is True and dependency["records"] == 198
        and dependency["original_file_metadata"] == 1025 and dependency["new_audio_PT_model_forward_backward_Adam_CUDA_calls"] == 0,
        "Complete42-source semantic/whole structured review")
    require(parent["original_manifest31_177_93_complete_typed_equal"] is True and parent["actual_windows_selected_source_lease_passed"] is True
        and parent["actual_preheld_native_files"] == 246 and parent["decoder_calls"] == 38, "Actual original DEV corpus and per-source/native guards")
    require(inputs["complete_input_target_descriptor_PCM_guard_passed"] is True and inputs["CPU_threads"] == 4, "Actual complete DEV input preparation")
    packet = {}
    for device in ("cpu", "cuda"):
        doc = read(ROOT / f"results/ema218_actual_{device}_mechanism_attempt01/mechanism_result.json")
        review = read(ROOT / f"results/ema218_actual_{device}_mechanism_attempt01/independent_readonly_review.json")
        q.p.check_metadata(doc); q.p.check_metadata(doc["mechanism_report"]); q.p.check_metadata(review)
        report = doc["mechanism_report"]
        require(doc["error"] is None and doc["launcher_exit_code"] == 0 and all(row["exit_code"] == 0 for row in doc["processes"].values())
            and doc["all_event_handles_post_exit_equal"] is True and report["source_step"] == 4500 and report["unique_input_counters"] == [4500]
            and report["formal_training_updates"] == 0 and all(report[k] is True for k in
                ("original194_first_raw_Adam_bit_equal", "real_raw_and_EMA_nonzero_independent", "whole_disk_replay_bit_type_equal", "whole_joint_rollback_bit_type_equal")), "Actual218 first raw/EMA/disk/whole transaction")
        require(review["result_sha256"] == sha(ROOT / f"results/ema218_actual_{device}_mechanism_attempt01/mechanism_result.json")
            and review["verified_explicit_device_input_AND_target_PCM_roundtrip_in_exact_AST"] is True, "Actual per-device PCM/target proof")
        if device == "cuda":
            require(review["verified_original_CUDA_runtime_and_rng"] is True and report["CUDA_initialized"] is True, "Strict source CUDA proof")
        packet[device] = report
    require(packet["cpu"]["input_packet_digest"] == packet["cuda"]["input_packet_digest"]
        and packet["cpu"]["input_metadata_digest"] == packet["cuda"]["input_metadata_digest"], "Whole cross-device same PCM/metadata")
    return audit, runtime, parent, units, dependency, inputs

def write(path, doc):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2, ensure_ascii=True, allow_nan=False)

def build():
    require(not DIRECTORY.exists() and not OUT.exists(), "Fresh complete activation and formal output required; preserve existing")
    q, t = sources()
    audit, runtime, parent, units, dependency, inputs = accepted(q)
    # Originals and completed units remain closed. Bind sources/code/documents,
    # not mutable progress summaries; the output is intentionally fresh.
    bindings = dict(parent["file_bindings_sha256"])
    for collection in (dependency["bindings_sha256"], units["bindings_sha256"]):
        for name, digest in collection.items():
            require(name not in bindings or bindings[name] == digest, "Conflicting archive binding")
            bindings[name] = digest
    for relative, digest in PINS.items():
        bindings[str((ROOT / relative).resolve())] = digest
    new_files = ("scripts/224_start_mel_ema_single_trajectory.ps1", "scripts/225_prepare_mel_ema_activation.py",
        "docs/mel_ema_single_trajectory_protocol_20261005.json", "reports/114_mel_ema_fixed500_training_activation.md")
    for relative in new_files:
        bindings[str((ROOT / relative).resolve())] = sha(ROOT / relative)
    protocol = read(ROOT / "docs/mel_ema_single_trajectory_protocol_20261005.json")
    require(protocol["source_sha256"] == "b3a0450ecd9bba259981bd47f8a49c0fee1a997d7a087a72570f589cc14ed5b3"
        and protocol["additional_updates"] == 500 and protocol["development_steps"] == [4750, 5000]
        and protocol["single_original_Adam"] is True and protocol["EMA"]["feedback"] is False, "Fixed protocol matches direct continuation")
    for name, digest in bindings.items():
        require(sha(name) == digest, "Complete plan source binding: " + name)
    DIRECTORY.mkdir()
    human_path = DIRECTORY / "continued_human_authorization.json"
    human = q.p.seal_metadata({"purpose": "NONRELEASE_FIXED500_CONTINUED_HUMAN_TRAINING_AUTHORIZATION",
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "basis": "Direct human messages in this same chat, not an agent or scheduler message",
        "human_requests_verbatim": ["目前训练进度如何？继续训练", "启动训练", "那就继续恢复到可训练状态", "继续"],
        "scope": "Report102 fixed source4500 control ARMS0;500 new updates4501..5000;raw versus EMA evaluation only;shared GPU authorized",
        "training_authorized_after_complete_actual_gates": True, "new_rights_or_data_authorized": False, "release_or_deployment_authorized": False,
        "old_input_approval_not_reused_as_new_training_approval": True, "protocol_path": str((ROOT / new_files[2]).resolve()),
        "protocol_sha256": sha(ROOT / new_files[2]), "no_historical_reruns": True, "automation_resumed": False})
    write(human_path, human); bindings[str(human_path)] = sha(human_path)
    path_map = {
        "actual_zero_update_audio_audit": "results/ema215_real_model_audio_audit_attempt01/independent_readonly_review.json",
        "strict_source_CUDA_runtime_RNG": "results/ema216_cuda_runtime_observation_attempt01/independent_readonly_review.json",
        "actual_CPU_CUDA_first_raw_original_Adam_bit_identity": "results/ema218_actual_cuda_mechanism_attempt01/independent_readonly_review.json",
        "actual_EMA_noalias": "results/ema218_actual_cuda_mechanism_attempt01/independent_readonly_review.json",
        "complete_disk_replay_and_joint_CPU_CUDA_RNG_rollback": "results/ema218_actual_cuda_mechanism_attempt01/independent_readonly_review.json",
        "cross_device_input_target_PCM_metadata": "results/ema218_actual_cpu_mechanism_attempt01/independent_readonly_review.json",
        "continuous_engine_units": "results/mel_ema_continuous_entry_monitor_20261005/unit_gate.json",
        "original_DEVELOPMENT_evaluator_verified": "results/mel_ema_continuous_entry_monitor_20261005/independent_entry_review.json",
        "per_draw_source_and_native_runtime_guard": "results/mel_ema_continuous_entry_monitor_20261005/independent_entry_review.json",
        "complete_dependency_review": "results/ema223_dependency_review_attempt01/dependency_review.json",
    }
    gates = {name: {"passed": True, "path": str((ROOT / relative).resolve()), "sha256": bindings[str((ROOT / relative).resolve())]} for name, relative in path_map.items()}
    gates["human_training_authorization_record"] = {"passed": True, "path": str(human_path), "sha256": bindings[str(human_path)]}
    plan_path = DIRECTORY / "training_plan.json"
    plan = q.p.seal_metadata({"purpose": "NONRELEASE_EMA_FIXED500_COMPLETE_TRAINING_PLAN", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "continued_human_authorization": human, "gates": gates, "bindings_sha256": bindings,
        "formal_out": str(OUT), "formal_updates_started": 0, "source_actual_step": 4500, "old_failed_evidence_preserved": True,
        "original_DEV_verification_scope": "Exact source AST units plus actual whole31/177/93 input preparation;new predictions occur only at new complete stages",
        "resources": "Fresh separate check before one launch, then worker disk/native/parent lifetime guards", "release_selection": "NONE"})
    write(plan_path, plan)
    bindings = dict(bindings); bindings[str(plan_path)] = sha(plan_path)
    activation = q.p.seal_metadata({"purpose": "NONRELEASE_EMA219_CONTINUOUS_SINGLE_TRAJECTORY", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_step": 4500, "source_sha256": protocol["source_sha256"], "limit": 5000, "additional_updates": 500,
        "stages": [4750, 5000], "single_raw_original_Adam": True, "EMA_feedback": False, "training_authorized": True,
        "release_selection": "NONE", "deployment": False, "gates": gates, "evaluator_binding": inputs["original_evaluator_binding"],
        "bindings_sha256": bindings, "formal_out": str(OUT), "training_plan_path": str(plan_path), "protocol_path": str((ROOT / new_files[2]).resolve())})
    activation_path = DIRECTORY / "activation.json"
    write(activation_path, activation)
    t.authority(activation_path, q)
    print("EMA_ACTIVATION_BUILT "+json.dumps({"activation": str(activation_path), "sha256": sha(activation_path), "bindings": len(bindings), "actual_gate_count": len(gates), "formal_updates_started": 0}), flush=True)

def check(path):
    q, t = sources()
    require(Path(path).resolve() == DIRECTORY / "activation.json" and not OUT.exists(), "Once-only intended activation and fresh formal output")
    doc = t.authority(path, q)
    q.p.check_metadata(read(doc["training_plan_path"]))
    accepted(q)
    resource = q.r.resource(); resource["no_duplicate"] = t.duplicate_preflight()
    require(not any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile")), "Preflight metadata-only")
    # Read-only preflight output is a console receipt. The WMI launcher has its
    # own exclusive actual receipts; a successful check is not an actual launch.
    print("EMA_ACTIVATION_PREFLIGHT_OK "+json.dumps({"activation_sha256": sha(path), "gates": len(doc["gates"]), "bindings": len(doc["bindings_sha256"]), "resource": resource,
        "worker_spawned": False, "formal_updates": 0}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "check")); parser.add_argument("--activation", type=Path)
    args = parser.parse_args()
    if args.operation == "build":
        build()
    else:
        require(args.activation is not None, "Explicit activation required"); check(args.activation)
