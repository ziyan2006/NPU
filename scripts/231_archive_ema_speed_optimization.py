"""Archive only new performance evidence; no tests/decoder/heavy imports."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINS = {
    "scripts/230_ema_fast_worker_integration.py": "6d516baeef6ed4fe5057a66cb6c65cfcb4a832c55e7d1a5f0408233a4a83d31f",
    "scripts/_test_ema_fast_worker_integration.py": "181023bd6cf934a9f9320f07de18af594a0d72fbbf44390d64f528e50e021a4a",
    "scripts/227_benchmark_ema_decoder_hotpath.py": "dafd11e9dd7dbdde46b976659933112d7db675fc7058d0d900fc75443cf7d8d4",
    "scripts/229_benchmark_ema_pcm_commitment.py": "96801c8da0404c9df75a019dc8212e76c29c57960d068698d22746ef7b4d27d7",
    "scripts/_test_ema_decoder_hotpath.py": "2e5387d2eaad8d04db5a1ec34c5b96e0712980652d6115dc725ab9b2e18e7939",
    "scripts/_test_ema_fast_hotpath.py": "4471c91dc205e0b817ef22652d2e0d43fa46eafef1b50259f76bebe2c952153e",
    "scripts/208_ema_authenticated_audio_input.py": "8d208e34ed1ce6b18997c64d75ad1aa03d67401b7752fa8d995da23d64982825",
    "scripts/213_ema_direct_decoder_runtime.py": "c699c0f54b5817da6fc187ff065df757823857c852e8c9b3bdb0144a5d81af84",
}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_plain_sealed(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    require(doc.get("content_sha256") == digest(body), "New plain benchmark seal: " + str(path))
    for name, expected in doc["source_bindings"].items():
        require(sha(name) == expected, "Benchmark source changed: " + name)
    return doc


def archive(out):
    require(out.parent == ROOT / "results" and not out.exists(), "Fresh exclusive archive only")
    for name, expected in PINS.items(): require(sha(ROOT / name) == expected, "Pinned source changed: " + name)
    spec = importlib.util.spec_from_file_location("ema231_static_recipe", ROOT / "scripts/230_ema_fast_worker_integration.py")
    integration = importlib.util.module_from_spec(spec); spec.loader.exec_module(integration)
    decoder_path = ROOT / "results/ema227_decoder_benchmark_attempt01/benchmark.json"
    pcm_path = ROOT / "results/ema229_pcm_benchmark_attempt01/benchmark.json"
    decoder, pcm = read_plain_sealed(decoder_path), read_plain_sealed(pcm_path)
    require(decoder["probe_and_PCM_bit_identical"] is True and decoder["analytic_PCM_bit_identical"] is True
            and decoder["new_synthetic_child_requests"] == 4 and decoder["real_music_draws"] == 0, "Decoder evidence scope")
    require(pcm["all_SHA_bit_identical"] is True and pcm["all_elements_finite_checked_every_call"] is True
            and pcm["RNG_unchanged"] is True and pcm["actual_audio_draws"] == 0, "CPU evidence scope")
    require(decoder["CUDA"] is pcm["CUDA"] is decoder["training_authorized"] is pcm["training_authorized"] is False,
            "No benchmark training authority")
    for log, tests in (("ema226_units_attempt01.log", 15), ("ema228_units_attempt02.log", 13), ("ema230_units_attempt01.log", 7)):
        text = (ROOT / "results" / log).read_text(encoding="utf-8")
        require("Ran " + str(tests) + " tests" in text and "\nOK" in text.replace("\r", ""), "Complete new test log: " + log)
    paths = list(PINS) + [str(p.relative_to(ROOT)) for p in integration.PINS]
    paths += [str(decoder_path.relative_to(ROOT)), str(pcm_path.relative_to(ROOT)),
              "results/ema227_decoder_benchmark_attempt01/baseline_events.json",
              "results/ema227_decoder_benchmark_attempt01/optimized_events.json",
              "reports/116_mel_ema_training_speed_optimization.md", "docs/ema_hotpath_optimization_scope_20261005.json",
              "scripts/231_archive_ema_speed_optimization.py"]
    paths += ["results/" + name for name in ("ema226_units_attempt01.log", "ema227_benchmark_attempt01.log",
              "ema228_units_attempt01.log", "ema228_units_attempt02.log", "ema229_benchmark_attempt01.log", "ema230_units_attempt01.log")]
    before = {str(ROOT / name): sha(ROOT / name) for name in dict.fromkeys(paths)}
    result = {"purpose": "NONRELEASE_EMA231_NEW_SPEED_COMPONENT_ARCHIVE_REVIEW",
              "bindings_sha256": before, "decoder_benchmark": decoder, "PCM_benchmark": pcm,
              "integration_static_audit": integration.audit(), "new_final_tests": 35,
              "new_child_decoder_requests": 0, "new_music_model_PT_Adam_updates": 0, "CUDA": False,
              "training_authorized": False, "end_to_end_training_speedup": "UNMEASURED", "release_selection": "NONE"}
    require(before == {path: sha(path) for path in before}, "Parents changed during archive read")
    result["content_sha256"] = digest(result)
    out.mkdir()
    with (out / "review.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({"bound_files": len(before), "new_final_tests": 35, "archive_sha256": sha(out / "review.json"),
                      "content_sha256": result["content_sha256"], "new_compute_audio_PT_Adam_CUDA": 0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    archive(Path(parser.parse_args().out).resolve())
