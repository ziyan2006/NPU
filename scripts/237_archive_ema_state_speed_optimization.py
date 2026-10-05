"""Fresh stdlib-only archive review; no benchmark/model/decoder replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/ema237_state_speed_archive_20261005"
PINS = {
    "scripts/194_train_mel_lr_scale.py": "56455bab403f5ffbab941980dde4cfc2d9fe88934d35f7dd5e841f34b5279f48",
    "scripts/205_ema_cpu_state_transaction.py": "8a8f2da566b664abfb19e6c24fe5fb4bf78bd61c13408ebf385b5cf425cd90f5",
    "scripts/207_ema_live_cpu_state.py": "7a1ab6a2dbcf9b2af99030784b34bcf6c2a73732227e0abadbb0cf2f88f18d90",
    "scripts/208_ema_authenticated_audio_input.py": "8d208e34ed1ce6b18997c64d75ad1aa03d67401b7752fa8d995da23d64982825",
    "scripts/212_ema_single_trajectory_engine.py": "8d10c304f3c3c2b4d2d6509dcec719e454b6afdabe651a329d1bc99b4a85d666",
    "scripts/219_ema_continuous_training_core.py": "cfd19e0b4059164621c190eda4f74819e0c8b73b358396b94e8f1bbb7f3619ca",
    "scripts/221_ema_supervised_training.py": "f1eb83d4abe087a182c761aa6e2e736a07bac0084d71822b2a70816ad6f3cc65",
    "scripts/228_ema_fast_hotpath.py": "fb1f674052e0a2dd2601d99f9f89e950b17f98d8876712c722beab7d6c066cce",
    "scripts/232_ema_fast_state_commitment.py": "674b73366eeb0094721fe37d307e44713952e4162bc0e4506bf8a422014dd47e",
    "scripts/_test_ema_fast_state_commitment.py": "bc540c9ffb47083d9dc8965e60e2457805f7ae43f65984e4d7509178d16bb1e5",
    "scripts/233_ema_batched_loss_reporting.py": "e73c965561dad2572f0423f47d50a719c81f078c048bd91ac01a94b3f1dd30a0",
    "scripts/_test_ema_batched_loss_reporting.py": "d9429b14e80fb1451af5f6bde90a602e51f1265a2018e59a404da39ea8f1f389",
    "scripts/234_benchmark_ema_state_and_sync.py": "d8bf5d74543e4eab550437e13a1ce399dfa6b46d425590eda3ba3ff05290e512",
    "scripts/235_ema_private_fast_storage_core.py": "ef875746528ef99ab5d831d402667810ef408958c0f61134cd42bce640505e02",
    "scripts/_test_ema_private_fast_storage_core.py": "05fdd0040070b781833238900945f4fc1923eb55c69be27dec19719a291d1de4",
    "scripts/236_benchmark_ema_private_core.py": "13f47805514c77ce1dc3b4384f3405e3eba672544ca2e3fe8ab8a435b7508422",
    "reports/116_mel_ema_training_speed_optimization.md": "afa671801468f2711856956d8b838e0f7840377ae5a72f21ffa72c481dbe6c59",
    "reports/117_mel_ema_state_speed_optimization.md": "b3038a83c1d66466a97ec0eeb1cc9ddb64f308e13ea1d85c79ca55efffa509c2",
    "docs/ema_state_speed_scope_20261005.json": "770e7e01beb02ca8ab16fad17ad0b0907cdce06b03387841fc770b838639683c",
    "results/ema231_speed_optimization_archive_20261005/review.json": "53b229b727036d60986f96374a7d3ed4ec94a9e606257870f09323074ce08041",
    "results/ema232_units_attempt01.log": "e58e796fbb351e0946669bad6502f29f6a6302e5fe46b5af82c5543b50e1b1f7",
    "results/ema233_units_attempt01.log": "ad891f4e0e5e76afd9f1eae0216f61b5d91b71a47563fc308219b40941cbb5f8",
    "results/ema234_benchmark_attempt01.log": "dcbd318353d072edda57b99a64fdc4587754b6991ca36c1445c3add79041b7dd",
    "results/ema235_units_attempt01.log": "2851da4dfe43cad6c2698a8735ddf6ee2960fdd29021446ff5ab36fcb23704db",
    "results/ema235_units_attempt02.log": "48c64d09ffb2aa7cf19f23fa35d1f5b0d3d098b080d175b00eec920fb15d6c23",
    "results/ema235_units_attempt03.log": "80940e275626204f83497a3742dceaa499ffb6471e7d0845e358bb2b969082b6",
    "results/ema236_benchmark_attempt01.log": "b8cd8656eedcbaabb18f5c426a022a8364ed1054370552e4493d2651897cc5b2",
    "results/ema236_benchmark_attempt02.log": "53f63f8db87906f50249a10bc0d137c4e44ae1042dcfec8f67ca6615260c9847",
    "results/ema234_state_and_sync_benchmark_attempt01/benchmark.json": "086509e0a9d5deb6f0e07b66c51e9e4131695c92b8aeff3c3952698982318fa2",
    "results/ema234_state_and_sync_benchmark_attempt01/cpu_benchmark.json": "0c9ac51ca7f352a11faf496a3dc7d925fad972e07ff0e58ced246b61627ced81",
    "results/ema234_state_and_sync_benchmark_attempt01/cuda_benchmark.json": "2bb5c95476fba64ec8ee7416fb7df70cdc10859f204ca6b95d63fc66b6ed8788",
    "results/ema236_private_core_benchmark_attempt02/benchmark.json": "9fa92c5f3e6621598925cdb31c6dfe6a653fbc0cc093ab927a6babd290930b2d",
    "results/ema236_initial_fixture_failure_review.json": "183404f07a4c5ae27140d1f76ec9d15872725c52611aeddd984ebf596d79f089",
    "results/ema235_initial_bytecode_test_failure_review.json": "444d634d5136eab564043867a1f81cf247a23178a1d6c3b785ed5809aea0990d",
    "results/ema235_property_test_failure_review.json": "88714c555ddd86953621d623846f2ae1a84899104f3eb6c12452239a284786b8",
    "results/mel_ema_single_trajectory_20261005/completion.json": "2cc020104fc53198197acb62a6f925368012882888204080cef65764feb2ac85",
    "results/mel_ema_single_trajectory_20261005/NONRELEASE_EMA_step_5000.pt": "d2be133da5bdd778fb615ef28167bf263e130a819596c6630e8c3a8726c1ab91",
}


def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()

def read(relative): return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))

def commitment(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def check_plain(doc):
    require(type(doc) is dict and type(doc.get("content_sha256")) is str, "Own plain performance seal required")
    content = {key: value for key, value in doc.items() if key != "content_sha256"}
    require(commitment(content) == doc["content_sha256"], "Own plain performance seal mismatch")

def typed(value):
    # Independent JSON tree comparison; not typed205 tensor seal substitution.
    if type(value) is dict: return ["dict", [[typed(k), typed(v)] for k, v in value.items()]]
    if type(value) is list: return ["list", [typed(v) for v in value]]
    if type(value) is float: return ["float", value.hex()]
    return [type(value).__name__, value]

def summarize(samples):
    require(type(samples) is dict and set(samples) == {"baseline", "optimized"}, "Two performance variants")
    for rows in samples.values():
        require(type(rows) is list and len(rows) == 3 and all(type(v) is float and v > 0 for v in rows), "Three real positive timings")
    med = {key: statistics.median(rows) for key, rows in samples.items()}
    return {"median_seconds": med, "speedup": med["baseline"]/med["optimized"],
            "reduction_percent": 100*(1-med["optimized"]/med["baseline"])}

def review():
    require(not OUT.exists(), "Fresh archive only; no overwrite or duplicate review")
    bindings = {relative: sha(ROOT / relative) for relative in PINS}
    require(bindings == PINS, "Fixed source/evidence byte binding changed")
    main = read("results/ema234_state_and_sync_benchmark_attempt01/benchmark.json")
    full = read("results/ema236_private_core_benchmark_attempt02/benchmark.json")
    check_plain(main); check_plain(full)
    for document in (main, full):
        for key in ("source_bindings", "new_source_bindings"):
            for filename, expected in document.get(key, {}).items():
                path = Path(filename).resolve(); require(path.is_relative_to(ROOT), "Repository source binding")
                require(PINS.get(path.relative_to(ROOT).as_posix()) == expected, "Independent source binding mismatch")
    require(len(main["benchmarks"]) == 2 and [r["device"] for r in main["benchmarks"]] == ["cpu", "cuda"], "Both devices")
    stats = {}
    for row in main["benchmarks"]:
        device = row["device"]
        saved = read("results/ema234_state_and_sync_benchmark_attempt01/" + device + "_benchmark.json")
        require(typed(saved) == typed(row), "Complete embedded/file device evidence mismatch")
        measured = summarize(row["samples_seconds"])
        require(typed(row["median_seconds"]) == typed(measured["median_seconds"])
                and row["maintenance_path_speedup"] == measured["speedup"], "Recomputed full maintenance medians")
        state = {key: [v["before_snapshot_and_seal"] + v["after_snapshot_and_seal"] for v in rows]
                 for key, rows in row["stage_seconds"].items()}
        scalar = summarize(row["scalar_report_samples_seconds"])
        require(typed(scalar["median_seconds"]) == typed(row["scalar_report_median_seconds"])
                and scalar["speedup"] == row["scalar_report_speedup"], "Recomputed scalar medians")
        require(row["full_typed_seals_equal"] is True and row["noalias_and_restoration_verified"] is True
                and type(row["real_model_Adam_updates"]) is int and row["real_model_Adam_updates"] == 0, "Synthetic maintenance scope")
        stats[device] = {"combined_maintenance": measured, "state_snapshots_only": summarize(state),
                         "scalar_batching": scalar, "scalar_adopt": False}
    measured = summarize(full["samples_seconds"])
    require(typed(measured["median_seconds"]) == typed(full["median_seconds"])
            and full["speedup"] == measured["speedup"], "Recomputed entire CPU fixture medians")
    require(typed(full["exposure"]) == typed({"six_microbatch_forward_backward": 42, "synthetic_Adam_steps": 7, "formal_updates": 0}), "Actual synthetic exposure counts")
    require(all(full[key] is True for key in ("complete_actual_synthetic_Adam_rollback", "noalias", "CPU_Python_NumPy_RNG_unchanged", "original_modules_unmodified")), "Full fixture invariants")
    require(full["CUDA"] is False and full["training_authorized"] is False and main["training_authorized"] is False
            and main["CUDA_initialized"] is True and main["actual_training_step_speedup"] == "UNMEASURED", "No formal speed claim/authority")
    for name, count in (("232_units_attempt01", 17), ("233_units_attempt01", 6), ("235_units_attempt03", 5)):
        log = (ROOT / ("results/ema" + name + ".log")).read_text(encoding="utf-8-sig")
        require("Ran " + str(count) + " tests" in log and "\nOK\n" in log.replace("\r\n", "\n"), "Final new units complete")
    for relative in ("results/ema236_initial_fixture_failure_review.json", "results/ema235_initial_bytecode_test_failure_review.json", "results/ema235_property_test_failure_review.json"):
        doc = read(relative); require(doc["native_exit_code"] == 1 and sha(ROOT / doc["log_path"]) == doc["log_sha256"], "Failure evidence retained")
    scope = read("docs/ema_state_speed_scope_20261005.json")
    require(scope["formal_221_worker_changed_or_launched"] is False and scope["actual_student_CUDA_complete_step_speedup"] == "UNMEASURED"
            and scope["training_authorized"] is False, "Complete scope limits")
    completion = read("results/mel_ema_single_trajectory_20261005/completion.json")
    require(type(completion["step"]) is int and completion["step"] == 5000 and completion["formal_training_updates"] == 500
            and completion["final_checkpoint"]["sha256"] == PINS["results/mel_ema_single_trajectory_20261005/NONRELEASE_EMA_step_5000.pt"], "Closed formal budget unchanged")
    require(all(sha(ROOT / relative) == value for relative, value in bindings.items()), "Sources/evidence changed during review")
    result = {"purpose": "NONRELEASE_EMA237_STD_READ_ONLY_PERFORMANCE_ARCHIVE", "utc": datetime.now(timezone.utc).isoformat(),
              "bindings_sha256": bindings | {"scripts/237_archive_ema_state_speed_optimization.py": sha(__file__)},
              "binding_count": len(bindings)+1, "maintenance_statistics": stats, "whole_CPU_synthetic_fixture_statistics": measured,
              "new_final_unit_count": 28, "benchmark236_exposure": full["exposure"],
              "scalar_batching_adopt": False, "full_GPU_student_training_speedup": "UNMEASURED", "formal_worker_modified_or_launched": False,
              "source_PT_deserialization_decoder_student_forward_Adam_or_benchmark_replay_in_this_reader": 0,
              "CUDA_in_this_reader": False, "failure_logs_preserved": True, "release_selection": "NONE", "training_authorized": False,
              "limits": "This reader checks SHA,seals,complete JSON embedding,statistics and scope;not new tensor computation,actual runtime acceptance or model quality verification."}
    result["content_sha256"] = commitment(result)
    OUT.mkdir()
    with (OUT / "review.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({"bindings": result["binding_count"], "CPU_fixture": measured, "content_sha256": result["content_sha256"]}, indent=2))


if __name__ == "__main__": review()
