"""Independent read-only215 archive review. No model/draw/Adam/CUDA execution."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/ema215_real_model_audio_audit_attempt01"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    import torch
    import torch._dynamo
    from unittest.mock import patch
    from contextlib import ExitStack
    a = load("ema215_readonly_metadata_helpers", "scripts/215_ema_model_audio_audit.py")
    e = load("ema215_readonly_typed_state_helpers", "scripts/212_ema_single_trajectory_engine.py")
    require, c = e.require, e.c
    def forbidden(*args, **kwargs):
        raise RuntimeError("Independent reader cannot create/call model/Adam/autograd/CUDA/child")
    import subprocess
    paths = [OUT / "supervised_audit_result.json", OUT / "native_event_journal.jsonl", OUT / "audit_result.pt"]
    before_hashes = {str(path): sha(path) for path in paths}
    with ExitStack() as stack:
        stack.enter_context(c.source.zero_execution_guard())
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        rng = e.capture_rng("cpu")
        doc = json.loads(paths[0].read_text(encoding="utf-8"))
        a.p.check_metadata(doc); report = doc["audit_report"]; a.p.check_metadata(report)
        require(doc["error"] is None and doc["launcher_exit_code"] == 0
                and all(row["exit_code"] == 0 for row in doc["processes"].values()), "Whole owned process EXIT0")
        require(doc["all_native_and_source_holds_before_worker_spawn"] is True
                and doc["all_event_handles_post_exit_equal"] is True
                and doc["all_preheld_source_bytes_post_exit_equal"] is True,
                "Actual preheld native/source interval checks")
        require(sha(ROOT / "scripts/215_ema_model_audio_audit.py") == doc["worker_source_sha256"], "Changed executed worker source")
        for relative, expected in doc["source_hashes"].items():
            require(sha(ROOT / relative) == expected, "Changed integration source")
        for name, expected in doc["preheld_source_bindings_sha256"].items():
            require(sha(name) == expected, "Changed bound Python/input source")
        events = [json.loads(line) for line in paths[1].read_text(encoding="utf-8").splitlines()]
        events = [item for item in events if "sequence" in item]
        require([event["sequence"] for event in events] == list(range(1, len(events) + 1)), "Complete native event sequence")
        images = [event for event in events if event["code"] in (3, 6)]
        require(len(images) == doc["native_file_event_count"], "Every actual native file event")
        for event in images:
            require(event["matched_preheld_physical_identity"] is True, "Unaccepted native image")
            a.validate_native(event["file"], doc["preheld_native_physical_files"])
        for pid, process in doc["processes"].items():
            rows = [event for event in events if str(event["pid"]) == pid]
            require(len(rows) == process["events"] and sum(row["code"] in (3, 6) for row in rows) == process["image_file_events"]
                    and sum(row["code"] == 3 for row in rows) == sum(row["code"] == 5 for row in rows) == 1,
                    "Whole per-process event/EXIT coverage")
        require(report["artifact_path"] == str(paths[2]) and report["artifact_sha256"] == before_hashes[str(paths[2])],
                "Exact new audit artifact SHA before deserialize")
        proof = torch.load(paths[2], map_location="cpu", weights_only=True)
        c.check_seal(proof)
        require(proof["content_sha256"] == report["artifact_own_typed205_seal"]
                and proof["purpose"] == doc["purpose"] == a.PURPOSE, "New proof own seal/scope")
        for packet in (proof["full_before"], proof["full_after"]):
            c.check_seal(packet)
            c.check_seal(packet["live_context"]); c.check_seal(packet["input_state"])
            require(packet["live_context"]["context"]["step"] == 4500
                    and packet["input_state"]["sampler"]["cursor"] == 4500
                    and packet["input_state"]["last_metadata"] is None
                    and packet["shadow"]["updates"] == 0
                    and packet["rng"]["torch_cuda"] == []
                    and len(packet["parent"]["parent_metadata"]["rng"]["torch_cuda"]) == 1,
                    "Full source exposure/CPU migration and immutable parent CUDA RNG")
        require(c.equal(proof["full_before"], proof["full_after"])
                and c.digest(proof["full_before"]) == report["full_state_before_sha256"] == report["full_state_after_sha256"],
                "Own-sealed full state bit/type symmetric equality")
        batch = proof["input_packet"]
        require(c.digest(batch) == proof["input_packet_digest"] and batch["cursor"] == 4500
                and batch["x"].shape == batch["v"].shape == (6, 2, 89856), "Whole real input packet identity")
        e.live.validate_metadata(batch["metadata"])
        for name, key in (("x", "input_sha256"), ("v", "target_sha256")):
            tensor = batch[name]
            require(tensor.dtype == torch.float32 and tensor.device.type == "cpu" and bool(torch.isfinite(tensor).all())
                    and not tensor.requires_grad, "Finite detached FP32 real PCM")
            require([hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest() for w in tensor] == proof["input_identity"][key],
                    "Actual PCM bits versus metadata")
        require(c.equal(proof["input_identity"]["metadata"], batch["metadata"])
                and a.p.typed_metadata(proof["input_identity"]) == a.p.typed_metadata(report["input_identity"])
                and a.p.typed_metadata(proof["losses"]) == a.p.typed_metadata(report["losses"]),
                "Full file/embedded input and complete losses symmetric typed comparison")
        exposure = proof["exposure"]
        require(exposure["audio_draws_started"] == exposure["audio_draws_completed"] == 1
                and exposure["forward_started"] == exposure["forward_completed"] == 6
                and exposure["source_cpu_storage_reads"] == exposure["actual_raw_models"] == exposure["actual_CPU_Adam_constructions"] == 1
                and exposure["adam_steps"] == exposure["autograd_engine_calls"] == exposure["source_training_PT_reads"] == 0
                and a.p.typed_metadata(exposure) == a.p.typed_metadata(report["exposure"]), "Exact actual exposure counts")
        require(a.p.typed_metadata(proof["decoder_rows"]) == a.p.typed_metadata(report["decoder_rows"])
                and len(proof["decoder_rows"]) == report["decoder_calls"], "Complete decoder proofs, no one-side seal stripping")
        decoder_manifest = a.g.read_manifest()
        for row in proof["decoder_rows"]:
            a.g.meta.check_seal(row)
            require(row["exit_code"] == row["popen_exit_code"] == 0 and row["accepted_debug_call"] is True
                    and row["pre_spawn_runtime_files_held"] == 37 and row["held_files_post_exit_equal"] is True,
                    "Actual conditional music decoder call accepted only after EXIT/held bytes")
            for event in row["events"]:
                if event["code"] in (3, 6):
                    a.g.validate_loaded(event["file"], decoder_manifest)
        require(proof["updates"] == 0 and proof["CUDA_initialized"] is False
                and proof["training_authorized"] is False and report["full_training_ready"] is False
                and e.equal(rng, e.capture_rng("cpu")), "Read-only scope/RNG")
        require(all(sha(path) == expected for path, expected in before_hashes.items()), "Archive changed during read-only review")
        result = a.p.seal_metadata({"purpose": "NONRELEASE_EMA215_INDEPENDENT_READONLY_ARCHIVE_REVIEW",
            "reader_source_sha256": sha(__file__), "bound_artifacts_sha256": before_hashes,
            "preheld_source_files": len(doc["preheld_source_bindings_sha256"]),
            "preheld_native_files": len(doc["preheld_native_physical_files"]), "native_file_events": len(images),
            "verified_actual_audio_draws": 1, "verified_actual_microbatch_forwards": 6,
            "verified_decoder_calls": report["decoder_calls"], "new_reader_audio_draws": 0,
            "new_reader_model_forwards": 0, "new_reader_Adam_constructions_or_steps": 0,
            "new_reader_source_PT_reads": 0, "new_reader_audit_artifact_reads": 1,
            "CUDA_initialized": False, "full_training_ready": False, "release_selection": "NONE"})
        a.obs.write_new(OUT / "independent_readonly_review.json", result)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
