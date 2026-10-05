"""Read-only216 native/runtime/source-RNG evidence review; no CUDA/model."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/ema216_cuda_runtime_observation_attempt01"


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
    from contextlib import ExitStack
    from unittest.mock import patch
    import subprocess
    r = load("ema216_review_metadata_only", "scripts/216_ema_cuda_runtime_observation.py")
    e = load("ema216_review_typed_CPU_helpers", "scripts/212_ema_single_trajectory_engine.py")
    require = e.require
    def forbidden(*args, **kwargs):
        raise RuntimeError("Independent reader forbids model/Adam/autograd/CUDA/child")
    files = [OUT / "runtime_observation.json", OUT / "native_event_journal.jsonl",
             ROOT / "results/ema215_real_model_audio_audit_attempt01/audit_result.pt"]
    hashes = {str(path): sha(path) for path in files}
    with ExitStack() as stack:
        stack.enter_context(e.c.source.zero_execution_guard())
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        rng = e.capture_rng("cpu")
        doc = json.loads(files[0].read_text(encoding="utf-8")); r.p.check_metadata(doc)
        report = doc["runtime_report"]; r.p.check_metadata(report)
        require(doc["error"] is None and doc["launcher_exit_code"] == 0
                and all(row["exit_code"] == 0 for row in doc["processes"].values())
                and doc["all_event_handles_post_exit_equal"] is True, "Actual runtime process exits/held event bytes")
        require(sha(ROOT / "scripts/216_ema_cuda_runtime_observation.py") == doc["worker_source_sha256"], "Executed runtime source changed")
        for name, expected in doc["preheld_source_bindings_sha256"].items():
            require(sha(name) == expected, "Bound runtime source changed")
        events = [json.loads(line) for line in files[1].read_text(encoding="utf-8").splitlines()]
        events = [event for event in events if "sequence" in event]
        require([event["sequence"] for event in events] == list(range(1, len(events) + 1)), "Complete native event sequence")
        images = [event for event in events if event["code"] in (3, 6)]
        require(len(images) == doc["native_file_events"], "All actual native image events")
        for event in images:
            name = r.g.canonical_path(event["file"]["final_path"])
            require(r.obs.meta.typed(event["file"]) == r.obs.meta.typed(doc["observed_native_physical_files"][name]), "Full physical observed identity")
            if event["matched_preheld_known_identity"]:
                r.a.validate_native(event["file"], doc["preheld_known_native_files"])
            else:
                require(event["new_cuda_native_observation_only_not_preimport_certified"] is True
                        and r.obs.meta.typed(doc["newly_observed_native_files"][name]) == r.obs.meta.typed(event["file"]),
                        "New actual native files remain observation-only")
        prior = r.gate()["audit_report"]
        require(hashes[str(files[2])] == prior["artifact_sha256"], "Exact previously reviewed audit artifact before deserialize")
        proof = torch.load(files[2], map_location="cpu", weights_only=True)
        e.c.check_seal(proof); e.c.check_seal(proof["full_before"])
        source = proof["full_before"]["parent"]["parent_metadata"]
        require(r.p.typed_metadata(source["runtime"]) == r.p.typed_metadata(report["actual_source_runtime"]),
                "Full actual CUDA runtime versus immutable parent, no normalized types")
        e.validate_rng(source["rng"], "cuda")  # Shape/type-only, no CUDA call.
        require(e.c.digest(source["rng"]) == report["original_RNG_typed205_digest"] == report["restored_RNG_typed205_digest"]
                and report["original_complete_RNG_restore_bit_type_equal"] is True,
                "Source complete CPU/Python/NumPy/CUDA RNG identity")
        require(report["source_cpu_storage_reads"] == 1 and report["source_training_PT_reads"] == 0
                and report["actual_model_constructions"] == report["actual_Adam_constructions_or_steps"] == 0
                and report["actual_forward_backward_audio_draws"] == 0
                and report["CUDA_explicitly_initialized"] is True and report["CUDA_update_mechanism_verified"] is False
                and doc["new_native_preimport_authenticated"] is False and doc["training_authorized"] is False,
                "Actual initialization only, no numerical update certification")
        require(e.equal(rng, e.capture_rng("cpu")) and not torch.cuda.is_initialized()
                and all(sha(path) == expected for path, expected in hashes.items()), "Reader RNG/archive bytes unchanged")
        result = r.p.seal_metadata({"purpose": "NONRELEASE_EMA216_INDEPENDENT_READONLY_RUNTIME_REVIEW",
            "reader_source_sha256": sha(__file__), "bindings_sha256": hashes,
            "runtime_equals_immutable4500_source": True, "original_complete_RNG_identity_verified": True,
            "native_file_events": len(images), "unique_native_files": len(doc["observed_native_physical_files"]),
            "new_native_files_observed": len(doc["newly_observed_native_files"]),
            "new_reader_audit_artifact_reads": 1, "new_reader_model_audio_Adam_calls": 0,
            "reader_CUDA_initialized": False, "new_native_preimport_authenticated": False,
            "CUDA_update_mechanism_verified": False, "training_authorized": False, "release_selection": "NONE"})
        r.obs.write_new(OUT / "independent_readonly_review.json", result)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
