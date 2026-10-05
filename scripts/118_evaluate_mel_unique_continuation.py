"""Score the continued mel_unique checkpoint on the identical cached holdouts."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = load("audit_eval_mel_cont", "109_audit_model_limits.py")
training = load("training_eval_mel_cont", "110_train_residual_ablation.py")
helpers = load("eval_helpers_mel_cont", "111_evaluate_residual_ablation.py")


def key(row):
    return row["dataset"], row["track"], round(float(row["start_s"]), 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--continuation", type=Path, default=ROOT / "results/mel_unique_finetune_cont_20261001")
    ap.add_argument("--initial-experiment", type=Path, default=ROOT / "results/mel_unique_finetune_20261001")
    ap.add_argument("--evaluation", type=Path, default=ROOT / "results/mel_unique_finetune_20261001/holdout_evaluation.json")
    ap.add_argument("--musdb-root", type=Path, default=ROOT / "data/datasets/MUSDB18-7-STEMS")
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "试听文件/mel_unique续训模型对照_20261001")
    args = ap.parse_args()

    continuation_path = args.continuation / "experiment.json"
    continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
    meta = continuation["metadata"]
    if not continuation.get("source_candidate_unchanged") or not continuation.get("frozen_deployed_checkpoint_unchanged"):
        raise ValueError("Continuation experiment is incomplete or changed its frozen inputs")
    if meta["script_sha256"] != audit.sha256(ROOT / "scripts/117_continue_mel_unique_probe.py"):
        raise ValueError("Continuation training source no longer matches provenance")
    if meta["core_script_sha256"] != audit.sha256(training.__file__):
        raise ValueError("Training core changed after continuation")

    previous = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if previous["script_sha256"] != audit.sha256(ROOT / "scripts/115_evaluate_mel_unique_probe.py"):
        raise ValueError("Cached initial evaluation source does not match")
    initial_experiment = json.loads((args.initial_experiment / "experiment.json").read_text(encoding="utf-8"))
    source_path = args.initial_experiment / "mel_unique_best.pt"
    if audit.sha256(source_path) != meta["sha256"]:
        raise ValueError("Continuation source candidate hash mismatch")
    if meta["continued_from_sha256"] != meta["sha256"]:
        raise ValueError("Continuation provenance does not name its starting candidate")
    if audit.sha256(ROOT / "models/student_bott2_mir1k_candidate.pt") != meta["frozen_deployed_checkpoint_sha256"]:
        raise ValueError("Frozen deployed checkpoint changed")
    if previous["models"]["baseline"]["sha256"] != meta["frozen_deployed_checkpoint_sha256"]:
        raise ValueError("Cached evaluation used a different baseline checkpoint")
    if previous["models"]["mel_unique"]["sha256"] != meta["sha256"]:
        raise ValueError("Cached evaluation used a different initial candidate")

    output_path = args.continuation / "holdout_evaluation.json"
    if output_path.exists() or args.audio_dir.exists():
        raise ValueError("Evaluation output already exists; preserve previous artifacts")
    rows = previous["rows"]
    row_map = {key(row): row for row in rows}
    if len(rows) != 143 or len(row_map) != len(rows):
        raise ValueError("Expected 143 unique rows in the fixed holdout evaluation")

    candidate_path = args.continuation / "mel_unique_cont_best.pt"
    expected_hash = continuation["arms"]["mel_unique_cont"]["best_sha256"]
    if audit.sha256(candidate_path) != expected_hash:
        raise ValueError("Continued checkpoint hash mismatch")
    net, blob = audit.t13.load_student(candidate_path, "cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if net.frontend != ("linear", 1.0) or net.band_layout != "mel_unique" or net.n_bands != 128:
        raise ValueError("Continued model has incompatible representation metadata")
    if blob.get("finetune", {}).get("continued_from_sha256") != meta["sha256"]:
        raise ValueError("Checkpoint does not descend from the evaluated initial candidate")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wa = torch.from_numpy(audit.t09.make_analysis_matrix(128, layout="mel_unique")).to(device)
    gs = torch.from_numpy(audit.t09.make_synthesis_matrix(128, layout="mel_unique")).to(device)
    kill = audit.t11.lf_kill_band_for(250., "mel_unique", 128)
    baseline_net, _ = audit.t13.load_student(ROOT / "models/student_bott2_mir1k_candidate.pt", device)
    initial_net, _ = audit.t13.load_student(source_path, device)
    base_wa = torch.from_numpy(audit.t09.make_analysis_matrix(128, layout="legacy_log")).to(device)
    base_gs = torch.from_numpy(audit.t09.make_synthesis_matrix(128, layout="legacy_log")).to(device)
    base_kill = audit.t11.lf_kill_band_for(250., "legacy_log", 128)

    old_sources = {entry["dataset"]: entry for entry in previous["sources"]}
    manifests = []
    for folder in ("mir1k_gainmix_cache", "onair_true_cache", "mshoxx_stereo_instrumental_cache"):
        path = ROOT / "results" / folder / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        domain = "instrumental" if document["dataset"] == "mshoxx" else document["dataset"]
        if audit.sha256(path) != meta["manifest_provenance"][domain]["sha256"]:
            raise ValueError(f"{domain} data manifest changed after continuation")
        if old_sources[document["dataset"]]["sha256"] != audit.sha256(path):
            raise ValueError(f"{domain} cache evaluation used a different manifest")
        manifests.append(document)

    cambridge = audit.cambridge_manifest(ROOT / "data/datasets/CambridgeMTK-electronic-excerpts")
    if old_sources["cambridge_multitrack"]["source_sha256"] != cambridge["source_sha256"]:
        raise ValueError("Cambridge source audio changed since the first evaluation")
    manifests.append(cambridge)
    proxy_mix = ROOT / "results/demons_mir1k_candidate_lf/01_mixture.wav"
    proxy_vocal = ROOT / "results/demons_mir1k_candidate_lf/04_teacher_vocals.wav"
    if (audit.sha256(proxy_mix) != previous["teacher_proxy"]["mixture_sha256"] or
            audit.sha256(proxy_vocal) != previous["teacher_proxy"]["teacher_sha256"]):
        raise ValueError("Demons proxy changed since the first evaluation")
    manifests.append({"dataset": "dj_teacher_proxy", "tracks": [{
        "track_id": "Jerro, Sophia Bel - Demons.mp3", "dataset": "dj_teacher_proxy", "split": "holdout",
        "mix_files": [str(proxy_mix)], "vocal_files": [str(proxy_vocal)]}]})

    musdb = sorted((args.musdb_root / "test").glob("*.stem.mp4"))
    if len(musdb) != 50:
        raise ValueError("Expected original MUSDB 50-song test inventory")
    musdb_hashes = {str(p): audit.sha256(p) for p in musdb}
    if old_sources["musdb"]["source_sha256"] != musdb_hashes:
        raise ValueError("MUSDB test audio changed since the first evaluation")
    manifests.append({"dataset": "musdb", "tracks": [{
        "track_id": p.stem, "dataset": "musdb", "split": "holdout",
        "mix_files": [str(p)], "vocal_files": []} for p in musdb]})

    model_meta = {"baseline": previous["models"]["baseline"],
                  "mel_unique": previous["models"]["mel_unique"],
                  "mel_unique_cont": {"path": str(candidate_path), "sha256": expected_hash,
                    "step": blob.get("step"), "params": blob.get("params"),
                    "band_layout": "mel_unique", "n_bands": 128, "frontend": net.frontend,
                    "kill_bands_lf250": kill, "finetune": blob.get("finetune", {})}}
    seen = set()
    for manifest in manifests:
        for record in manifest["tracks"]:
            if record["split"] != "holdout":
                continue
            helpers.reject_overlap(blob.get("finetune", {}), manifest["dataset"], record["track_id"])
            spec = audit.t23.TrackSpec(record["track_id"], record["dataset"], record["mix_files"],
                                       record["vocal_files"], record.get("stem_files", []))
            if record["dataset"] == "musdb":
                mix, vocal = training.decode_musdb(Path(record["mix_files"][0]))
            elif record["dataset"] == "dj_teacher_proxy":
                mix = torch.from_numpy(audit.t23._stereo(audit.t23._read(Path(spec.mix_files[0]))[0]))
                vocal = torch.from_numpy(audit.t23._stereo(audit.t23._read(Path(spec.vocal_files[0]))[0]))
                if mix.shape != vocal.shape:
                    raise ValueError("Pinned teacher proxy is not aligned")
            elif record["dataset"] == "cambridge_multitrack":
                def read(paths):
                    return [torch.from_numpy(audit.t23._stereo(audit.t23._read(Path(p))[0])) for p in paths]
                mix, vocal = audit.mix_multitrack(read(spec.vocal_files), read(spec.stem_files))
            else:
                mix, vocal = audit.t23.load_track(spec)

            for start in range(0, mix.shape[-1], 30 * audit.SR):
                end = min(start + 30 * audit.SR, mix.shape[-1])
                if end - start < 5 * audit.SR:
                    continue
                x, v = mix[:, start:end].to(device), vocal[:, start:end].to(device)
                row_key = (manifest["dataset"], record["track_id"], round(start / audit.SR, 6))
                if row_key not in row_map or row_key in seen:
                    raise ValueError(f"Holdout segmentation differs from cached evaluation: {row_key}")
                seen.add(row_key)
                spectrum = audit.t09._stft(x)
                bands = torch.einsum("fb,cft->cbt", wa, spectrum.abs())
                predicted_masks = audit.masks(net, bands)
                estimate = audit.render(spectrum, predicted_masks[:2], gs, x.shape[-1], kill)
                row_map[row_key]["models"]["mel_unique_cont"] = audit.waveform_metrics(estimate, x, v)
                if manifest["dataset"] == "cambridge_multitrack":
                    name = record["track_id"].replace(" - ", "_").replace(",", "_")
                    folder = args.audio_dir / name
                    folder.mkdir(parents=True, exist_ok=True)
                    sf.write(str(folder / "00_mixture.wav"), x.cpu().numpy().T, audit.SR, subtype="FLOAT")
                    sf.write(str(folder / "01_reference_accompaniment.wav"), (x - v).cpu().numpy().T,
                             audit.SR, subtype="FLOAT")
                    base_masks = audit.masks(baseline_net,
                        torch.einsum("fb,cft->cbt", base_wa, spectrum.abs()))
                    initial_masks = audit.masks(initial_net,
                        torch.einsum("fb,cft->cbt", wa, spectrum.abs()))
                    base_estimate = audit.render(spectrum, base_masks[:2], base_gs, x.shape[-1], base_kill)
                    initial_estimate = audit.render(spectrum, initial_masks[:2], gs, x.shape[-1], kill)
                    sf.write(str(folder / "02_baseline_accompaniment.wav"),
                             (x - base_estimate).cpu().numpy().T, audit.SR, subtype="FLOAT")
                    sf.write(str(folder / "03_mel_unique_accompaniment.wav"),
                             (x - initial_estimate).cpu().numpy().T, audit.SR, subtype="FLOAT")
                    sf.write(str(folder / "04_mel_unique_cont_accompaniment.wav"),
                             (x - estimate).cpu().numpy().T, audit.SR, subtype="FLOAT")
            print(f"EVAL_CONT dataset={manifest['dataset']} track={record['track_id']} "
                  f"matched={len(seen)}/{len(rows)}", flush=True)

    if seen != set(row_map):
        raise ValueError(f"Evaluation coverage mismatch: {len(seen)} of {len(row_map)} rows")
    merged_rows = list(row_map.values())
    labels = ["baseline", "mel_unique", "mel_unique_cont"]
    summary = helpers.grouped_summary(merged_rows, labels)
    payload = {"schema": 1,
               "scope": "Offline FP32; continuation scored on identical hash-verified holdouts; no board/SD write",
               "script_sha256": audit.sha256(__file__),
               "continuation_experiment_sha256": audit.sha256(continuation_path),
               "initial_evaluation_sha256": audit.sha256(args.evaluation),
               "models": model_meta, "rows": merged_rows, "summary": summary,
               "provisional_float_gate": {name: helpers.provisional_gate(merged_rows, summary, name)
                                           for name in ("mel_unique", "mel_unique_cont")},
               "sources": previous["sources"],
               "teacher_proxy": previous["teacher_proxy"],
               "audio_dir": str(args.audio_dir),
               "audio_scope": "FLOAT WAV renders for both Cambridge excerpts; offline only",
               "aggregation": previous["aggregation"],
               "caveat": "Reuses the previously seen 143-segment/78-song suite; not a fresh blind test. Candidate step selected on reserved training-pool validation."}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"MEL_UNIQUE_CONT_EVAL PASS segments={len(merged_rows)} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
