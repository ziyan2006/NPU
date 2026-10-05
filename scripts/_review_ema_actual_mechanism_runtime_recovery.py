"""Independent218 archive reader: CPU deserialize only, no models/updates."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
CODE = "scripts/218_ema_mechanism_runtime_recovery.py"
CODE_SHA = "2652de693f6300e274feb102379e095f4d0a747755a38c2a2c4908d3a76eba4a"
spec = importlib.util.spec_from_file_location("review218_native_primitives", ROOT / CODE)
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
q, p = k.q, k.q.p
require, sha = q.require, q.sha


def review(out):
    require(sha(ROOT / CODE) == CODE_SHA, "Exact executed recovery source")
    out = Path(out).resolve()
    device = "cuda" if out.name == "ema218_actual_cuda_mechanism_attempt01" else "cpu"
    require(out.parent == (ROOT / "results").resolve()
            and out.name == "ema218_actual_" + device + "_mechanism_attempt01"
            and not (out / "independent_readonly_review.json").exists(), "One new fixed218 review, no overwrite/repeat")
    path = out / "mechanism_result.json"
    result_sha = sha(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    p.check_metadata(result); report = result["mechanism_report"]; p.check_metadata(report)
    require(result["purpose"] == report["purpose"] == k.PURPOSE and result["device"] == report["device"] == device
            and result["error"] is None and result["launcher_exit_code"] == 0
            and all(row["exit_code"] == 0 for row in result["processes"].values())
            and result["all_event_handles_post_exit_equal"] is True
            and result["actual_preheld_native_mechanism_verified"] is True
            and result["full_training_ready"] is False and result["training_authorized"] is False, "Full actual mechanism exit/scope gate")
    require(result["preheld_source_bindings_sha256"][str(ROOT / CODE)] == CODE_SHA, "Actual supervisor held recovery source")
    for name, expected in result["preheld_source_bindings_sha256"].items():
        require(sha(name) == expected, "Changed held mechanism binding: " + name)
    known = result["preheld_native_physical_files"]
    require(len(known) == 246, "Full preheld actual native inventory")
    for row in known.values():
        require(sha(row["final_path"]) == row["sha256"], "Changed current native file")
    entries = [json.loads(line) for line in (out / "native_event_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [row for row in entries if "sequence" in row]
    require([row["sequence"] for row in events] == list(range(1, len(events)+1)), "Complete native sequence")
    image_count, exits = 0, {}
    for event in events:
        require(str(event["pid"]) in result["processes"], "Only actual owned PID")
        if "file" in event:
            image_count += 1
            require(event["matched_preheld_physical_identity"] is True, "Preheld acceptance for every actual native load")
            q.a.validate_native(event["file"], known)
        if event["code"] == 5:
            require(event["pid"] not in exits and event["exit_code"] == 0, "One actual native EXIT0 per process")
            exits[event["pid"]] = event["exit_code"]
    require(image_count == result["native_file_events"] and set(map(str, exits)) == set(result["processes"]), "Full image/exit coverage")
    q.validate_exposure(report["exposure"])
    # Pure AST validator reconstructs exact runtime repair; never executes worker.
    _, _, insert_count = k.build()
    require(insert_count == 8 and report["unique_input_counters"] == [4500] and report["unique_slots"] == 6
            and report["logical_diagnostic_step"] == 4501 and report["formal_training_updates"] == 0
            and report["new_audio_draws"] == report["original_training_PT_reads"] == 0, "Exactly one audited input; no formal training")
    e = q.load("review218_portable_state_primitives", "scripts/212_ema_single_trajectory_engine.py")
    import torch
    from unittest.mock import patch
    c = e.c
    reader_rng = e.capture_rng("cpu")
    artifacts = {}
    with c.source.zero_execution_guard(), patch.object(q.subprocess, "Popen", side_effect=AssertionError("Reader forbids children")):
        states = {}
        for key, filename in (("source4500_disk_sha256", "source4500_complete.pt"),
                              ("diagnostic4501_disk_sha256", "diagnostic4501_complete.pt"),
                              ("rolled_back4500_disk_sha256", "rolled_back4500_complete.pt")):
            file = out / filename; expected = report[key]
            require(sha(file) == expected, "State SHA BEFORE deserialize")
            state = torch.load(file, map_location="cpu", weights_only=True)
            require(sha(file) == expected, "State bytes changed during read")
            c.check_seal(state); c.check_seal(state["live_context"]); c.check_seal(state["input_state"])
            require(state["device"] == device and state["scope"] == q.device_scope(device)
                    and state["training_authorized"] is False and state["release_selection"] == "NONE", "Actual portable mechanism state scope")
            states[filename] = state; artifacts[str(file)] = expected
        before, after, rolled = [states[f] for f in ("source4500_complete.pt", "diagnostic4501_complete.pt", "rolled_back4500_complete.pt")]
        require(c.equal(before, rolled) and c.digest(before) == report["initial_state_digest"]
                and c.digest(after) == report["after_state_digest"], "Full disk source/rollback/post-update types and bits")
        for key in ("schema", "purpose", "scope", "device", "source_sha256", "parent", "parent_digest", "runtime", "layout", "loss_identity",
                    "training_authorized", "actual_audio_backend_verified", "cuda_transaction_verified", "release_selection"):
            require(c.equal(before[key], after[key]), "Changed immutable full mechanism field " + key)
        parent = before["parent"]
        c.source.check_arm(parent["raw_arm"], parent["raw_arm"]["optimizer"]["param_groups"][0]["lr"])
        source_context = before["live_context"]["source_context"]
        require(c.digest(parent) == before["parent_digest"]
                and c.digest(parent["parent_metadata"]["rng"]) == report["source_parent_CUDA_rng_digest"], "Full immutable parent and original CUDA RNG")
        require(c.equal(before["runtime"], report["actual_source_runtime"])
                and before["runtime"]["threads"] == (4 if device == "cuda" else 2), "Post-import explicit source4/CPU2 threads")
        if device == "cuda":
            require(c.equal(before["runtime"], parent["parent_metadata"]["runtime"])
                    and c.equal(before["rng"], parent["parent_metadata"]["rng"]), "WHOLE original CUDA runtime/RNG bits and types")
        else:
            require(before["rng"]["torch_cuda"] == [] and report["CPU_explicit_migration_not_CUDA_resume"] is True, "Explicit CPU migration not CUDA resume")
        require(c.equal(before["rng"], after["rng"]), "Deterministic graph/update unexpectedly consumed global RNG")
        probe = SimpleNamespace(_layout=before["layout"], _names=before["raw"]["parameter_names"],
                                _group=before["raw"]["optimizer"]["param_groups"][0], _defaults=before["raw"]["optimizer_defaults"],
                                _context_template=source_context)
        probe._lr = lambda step: c.CpuStateOwner._lr(probe, step)
        for step, state in ((4500, before), (4501, after)):
            live = e.live.LiveContext(source_context, e.j.provenance(parent)); live.load_state_dict(state["live_context"])
            require(live.step == state["raw"]["updates"] == state["shadow"]["step"] == step
                    and state["shadow"]["updates"] == step-4500
                    and state["input_state"]["sampler"]["cursor"] == step
                    and state["live_context"]["context"]["schedule"]["stopped_at"] == 4500,
                    "All raw/Adam/EMA count/input/cumulative schedule exposures, old stop preserved")
            c.CpuStateOwner._validate_raw(probe, state["raw"], step)
            e.validate_rng(state["rng"], device)
            shadow = state["shadow"]
            require(type(shadow["tensors"]) is OrderedDict and list(shadow["tensors"]) == list(state["raw"]["tensors"])
                    and c.equal(shadow["modes_at_copy"], state["raw"]["modes"]), "Whole shadow tensor names/modes")
        require(c.equal(before["raw"]["tensors"], before["shadow"]["tensors"]) and before["input_state"]["last_metadata"] is None, "Exact E0 and initial semantic input")
        metadata = after["input_state"]["last_metadata"]; e.live.validate_metadata(metadata)
        require(c.digest(metadata) == report["input_metadata_digest"], "Complete typed consumed metadata")
        for desc in after["layout"]["tensors"]:
            name = desc["name"]
            expected = before["shadow"]["tensors"][name]*.99 + after["raw"]["tensors"][name]*.01 if desc["trainable"] else after["raw"]["tensors"][name].clone()
            require(c.equal(expected, after["shadow"]["tensors"][name]), "Independent separate FP32 EMA exact bits " + name)
        require(any(not c.equal(before["raw"]["tensors"][n], after["raw"]["tensors"][n]) for n in probe._names)
                and any(not c.equal(after["shadow"]["tensors"][n], after["raw"]["tensors"][n]) for n in probe._names), "Real nonzero update and independent EMA")
        c.noalias(states, ())
        require(all(report[key] is True for key in ("original194_first_raw_Adam_bit_equal", "real_raw_and_EMA_nonzero_independent",
                    "whole_disk_replay_bit_type_equal", "fault_after_real_Adam_and_EMA_count_and_CPU_CUDA_RNG_mutation", "whole_joint_rollback_bit_type_equal")), "All actual mechanism checks")
        require(c.equal(reader_rng, e.capture_rng("cpu")) and not torch.cuda.is_initialized(), "Independent reader did not initialize CUDA or change RNG")
    require(sha(path) == result_sha and sha(ROOT / CODE) == CODE_SHA, "Archive/implementation unchanged after reader")
    saved = p.seal_metadata({"purpose": "NONRELEASE_EMA218_INDEPENDENT_PORTABLE_STATE_AND_NATIVE_ARCHIVE_REVIEW",
        "device": device, "result_sha256": result_sha, "implementation_sha256": CODE_SHA, "artifact_bindings_sha256": artifacts,
        "verified_original194_first_raw_Adam_bit_equal_worker_reference": True,
        "verified_independent_EMA_separate_FP32_bits": True, "verified_whole_disk_rollback_state_bit_type_equal": True,
        "verified_original_CUDA_runtime_and_rng": device == "cuda", "verified_postimport_threads": 4 if device == "cuda" else 2,
        "verified_preheld_native_files": 246, "verified_native_file_events": image_count,
        "verified_unique_input_counters": [4500], "verified_input_packet_digest": report["input_packet_digest"],
        "verified_input_metadata_digest": report["input_metadata_digest"], "verified_explicit_device_input_AND_target_PCM_roundtrip_in_exact_AST": True,
        "reader_state_artifact_reads": 3, "reader_source_PT_CPU_storage_reads": 0, "reader_model_forward_Adam_audio_calls": 0,
        "reader_CUDA_initialized": False, "reader_CPU_Python_NumPy_rng_unchanged": True,
        "formal_training_updates": 0, "full_training_ready": False, "release_selection": "NONE"})
    q.obs.write_new(out / "independent_readonly_review.json", saved)
    print(json.dumps(saved, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", type=Path, required=True)
    review(parser.parse_args().out)
