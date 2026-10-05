"""Independent208 archive-only review. No unit rerun or actual audio draw.

Reads complete JSON documents, each document's own seal, exact typed original
input identities and all input/file bindings. No file writes or deserialization
of student/optimizer containers. Console output is retained by the caller.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
from unittest.mock import patch

import torch
import torch._dynamo

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema208_archive_only", ROOT / "scripts/208_ema_authenticated_audio_input.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
PINS = {
    "scripts/208_ema_authenticated_audio_input.py": "8d208e34ed1ce6b18997c64d75ad1aa03d67401b7752fa8d995da23d64982825",
    "scripts/_test_ema_authenticated_audio_input.py": "43f3fe3fdf6c389472d0f652be28a4ddbd262fc4e40cd927f533ef9a85051c4c",
    "docs/ema_authenticated_audio_unit_scope_20261004.json": "df91a1517a124a606255848d257c1b6e1ac7a4dd25788a68e62ab88fcfc3d573",
    "results/ema208_unit_tests_attempt01.log": "51777a1731a86d431db20398aa9a9c8477aaba9d1b4cf1910a0e61c09e2ac81c",
    "results/mel_ema_audio_backend_monitor_20261004/actual_input_replay_evidence.json": "d198b7f32b0b208645496de0005a929af11174d757e89ee5f366af4b01387a3a",
    "reports/107_mel_ema_joint_live_cpu_state_progress.md": "3f9d3ed84d2f0c80074c66ccc83cd0efaad5ea9c08beb1721945fa82be8174d7",
    "results/mel_ema_live_cpu_state_monitor_20261004/unit_gate.json": "8733d739b1b57ce8adeb8eb4e296795ff826bb79c75de4d84c7f537159742b93",
    "results/mel_ema_live_cpu_state_monitor_20261004/progress.json": "664c9463edde3657e93e753b4a5bc9ca26427d8d742b086a56f4690a9d08a7d9",
    "results/train_adam_memory_20261004/inputs.json": "19caf93b84357d46e37e5614cc517fe2a3d8c75bfc27c24e2f22d63e5992cc0f",
}


def blocked(*args, **kwargs):
    raise RuntimeError("ARCHIVE_ONLY_BEFORE_AUDIO_MODEL_OR_OPTIMIZER_EXECUTION")


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def review():
    require, equal = api.require, api.equal
    rng = api.storage.capture_cpu_rng()
    bindings = {str(ROOT / p): h for p, h in PINS.items()}
    api.verify_files(bindings)  # All file SHA checks precede JSON interpretation.
    proof = read_json("results/mel_ema_audio_backend_monitor_20261004/actual_input_replay_evidence.json")
    api.storage.check_seal(proof)
    contract = proof["contract"]
    api.storage.check_seal(contract)
    reference_doc = api.acq.read_sealed(ROOT / "results/train_adam_memory_20261004/inputs.json")
    reference = reference_doc["inputs"][0]
    expected_identity = {key: reference[key] for key in ("counter", "input_sha256", "target_sha256", "metadata")}
    require(equal(proof["actual_first_identity"], expected_identity), "Full typed actual/original input identity")
    require(equal(proof["actual_replay_identity"], expected_identity), "Full typed actual/original replay identity")
    require(equal(proof["actual_first_identity"], proof["actual_replay_identity"]), "Symmetric full typed replay identity")
    api.live.validate_metadata(proof["actual_first_identity"]["metadata"])
    require(equal(contract["source_sampler"], api.ORIGINAL_SAMPLER), "Original full typed source sampler")
    require(equal(contract["true_train_counts"], {"musdb": 73, "mir1k": 81, "instrumental": 11}), "Original TRAIN pools")
    approval = api.acq.read_sealed(api.inp.DEFAULT_APPROVAL)
    require(equal(contract["pseudo_ids"], approval["pair_ids"]) and len(contract["pseudo_ids"]) == 24, "Original approved24 order")
    for key, expected in {"actual_input_unique_counters": [4500], "actual_successful_draws": 2,
                          "unique_slots": 6, "repeated_slots": 6, "cuda_initialized": False,
                          "cpu_python_numpy_rng_unchanged": True, "release_selection": "NONE"}.items():
        require(equal(proof[key], expected), "Actual input scope: " + key)
    for key in ("source_pt_deserialization", "student_modules", "forward", "autograd_engine",
                "adam_construction", "adam_steps", "training_updates", "output_audio"):
        require(equal(proof[key], 0), "No actual training API execution: " + key)
    for key in ("training_authorized", "cuda_resume_verified", "full_training_transaction_verified"):
        require(equal(contract[key], False), "Not a formal trainer gate: " + key)
    for key in ("full_zero_update_model_audit", "cpu_cuda_update_mechanism", "formal_training_authorization"):
        require(equal(proof[key], "PENDING"), "Future full gate remains pending")
    require(equal(contract["single_mel_target"], True), "Single Mel target")
    plan = api.acq.read_sealed(ROOT / "results/teacher_library_melband_20261002/plan.json")
    require(equal(contract["decoder_versions"], plan["media_versions"]), "Original decoder versions")
    require(equal(contract["input_packages"], {p: importlib.metadata.version(p) for p in ("torch", "numpy", "soundfile", "scipy")}), "Recorded input package versions")
    for path, digest in (contract["input_bindings_sha256"] | proof["selected_true_original_files_sha256"]).items():
        require(path not in bindings or bindings[path] == digest, "Conflicting archive binding")
        bindings[path] = digest
    for executable in contract["decoder_executables"].values():
        require(executable in contract["input_bindings_sha256"], "Actual executable bound")
    bindings[str(Path(__file__).resolve())] = api.file_sha(__file__)
    api.verify_files(bindings)
    log = (ROOT / "results/ema208_unit_tests_attempt01.log").read_text(encoding="utf-8-sig")
    require(len(re.findall(r"^test_\d+.* \.\.\. ok$", log, re.M)) == 35 and "Ran 35 tests in 44.997s" in log and
            log.rstrip().endswith("OK"), "Actual full35 log, no inferred native exit")
    scope = read_json("docs/ema_authenticated_audio_unit_scope_20261004.json")
    require(equal(scope["actual_audio_unique_counters"], [4500]) and equal(scope["training_updates"], 0), "Recorded scope")
    require(equal(rng, api.storage.capture_cpu_rng()) and not torch.cuda.is_initialized(), "Read-only RNG/CUDA scope")
    return {"purpose": "NONRELEASE_EMA208_INDEPENDENT_ARCHIVE_REVIEW_NOT_UNIT_RERUN",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(), "bindings_sha256": bindings,
            "binding_count": len(bindings), "input_contract_binding_count": len(contract["input_bindings_sha256"]),
            "selected_true_original_file_count": len(proof["selected_true_original_files_sha256"]),
            "proof_content_seal": proof["content_sha256"], "contract_content_seal": contract["content_sha256"],
            "own_nested_seals_checked": True, "full_typed_reference_and_replay_equal": True,
            "actual_audio_draws_added": 0, "source_pt_deserialization": 0, "model_computation": 0,
            "adam_construction_or_step": 0, "cuda_initialized": False,
            "cpu_python_numpy_rng_unchanged": True, "release_selection": "NONE"}


if __name__ == "__main__":
    with ExitStack() as blocks:
        for obj, attr in ((api.AuthenticatedAudioStream, "__init__"), (torch, "load"),
                          (torch.nn.Module, "__init__"), (torch.nn.Module, "_call_impl"),
                          (torch.autograd, "grad"), (torch.autograd, "backward"),
                          (torch.optim.Adam, "__init__"), (torch.optim.Adam, "step"),
                          (torch.cuda, "init"), (torch.cuda, "_lazy_init"),
                          (api.data.sf, "SoundFile"), (api.inp.m.core.t23.sf, "read"),
                          (api.inp.m.bulk, "decode"), (api.inp.m.core, "decode_musdb"),
                          (api.inp.m.core.t23, "load_track")):
            blocks.enter_context(patch.object(obj, attr, side_effect=blocked))
        print(json.dumps(review(), ensure_ascii=True, allow_nan=False))
