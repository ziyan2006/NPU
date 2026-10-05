"""Fixed-step weak-vocal objective regression vs frozen and cached control.

Recompute the frozen waveform scores to verify the cached 145-clip inputs;
reuse control metrics only after hashes, scripts, sources and budget checks.
No regression-based checkpoint picking and no deployment/SD writes.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


training = load("weak_eval_training", "122_train_weak_vocal_objective.py")
regression = load("weak_eval_regression", "121_evaluate_layout_control.py")
suite, core, audit = training.suite, training.core, training.suite.audit
helpers = regression.helpers


def row_key(row):
    return row["dataset"], row["track"], round(row["start_s"], 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=ROOT / "results/weak_vocal_objective_20261001")
    args = ap.parse_args()
    experiment_path = args.experiment / "experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    meta = experiment["metadata"]
    if (experiment.get("status") != "complete" or not experiment.get("paired_batches_identical") or
            not experiment.get("deployed_checkpoint_unchanged") or not experiment.get("control_unchanged")):
        raise ValueError("New training is not a completed paired experiment")
    if core.digest(training.__file__) != meta["script_sha256"] or meta["objective_policy"] != training.LOSS_POLICY:
        raise ValueError("Objective implementation/policy changed after training")
    parent_dir = Path(meta["control_directory"])
    parent = training.load_control(parent_dir)
    if core.digest(parent_dir / "experiment.json") != meta["control_experiment_sha256"]:
        raise ValueError("Matched-control metadata changed")
    if experiment["arms"]["weak_vocal"]["batch_sha256"] != parent["arms"]["legacy_log"]["batch_sha256"]:
        raise ValueError("Not the same training waveforms/budget")
    for filename, expected in (("selection_suite.json", meta["selection_suite_sha256"]),
                               ("frozen_selection_scores.json", meta["frozen_selection_scores_sha256"])):
        if core.digest(args.experiment / filename) != expected:
            raise ValueError(f"Selection evidence changed: {filename}")
    cached_path = parent_dir / "regression_evaluation.json"
    if core.digest(cached_path) != meta["control_regression_sha256"]:
        raise ValueError("Control regression changed")
    cached = json.loads(cached_path.read_text(encoding="utf-8"))
    if cached["script_sha256"] != core.digest(regression.__file__):
        raise ValueError("Control evaluation implementation changed")
    if (cached["experiment_sha256"] != meta["control_experiment_sha256"] or
            cached["models"]["baseline"]["sha256"] != meta["sha256"] or
            cached["models"]["legacy_log"]["sha256"] != meta["control_final_sha256"]):
        raise ValueError("Cached control model/provenance mismatch")
    row_map = {row_key(row): row for row in cached["rows"]}
    if len(row_map) != 145 or len(row_map) != len(cached["rows"]):
        raise ValueError("Expected 145 unique previously inspected regression clips")
    output = args.experiment / "regression_evaluation.json"
    if output.exists():
        raise ValueError("Keep historical evaluations; output already exists")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = {"baseline": Path(meta["checkpoint"]), "weak_vocal": args.experiment / "weak_vocal_final.pt"}
    nets, models = {}, {}
    for label, path in paths.items():
        expected = meta["sha256"] if label == "baseline" else experiment["arms"][label]["final_sha256"]
        if core.digest(path) != expected:
            raise ValueError("Frozen/candidate model hash changed")
        net, blob = core.t13.load_student(path, device)
        if (net.band_layout != "legacy_log" or net.frontend != ("linear", 1.) or net.n_bands != 128 or
                blob.get("bottleneck_blocks") != 2 or blob.get("temporal_dilations")):
            raise ValueError("Candidate changed the frozen graph/layout")
        if label != "baseline" and (blob["step"] != 200 or blob.get("finetune", {}).get("objective_policy") != meta["objective_policy"]):
            raise ValueError("Evaluation must use the fixed final-step objective candidate")
        nets[label] = net
        models[label] = {"path": str(path), "sha256": expected, "step": blob["step"], "band_layout": "legacy_log",
                         "internally_selected": label == "weak_vocal" and experiment["arms"][label]["best_step"] == 200}
    models["control"] = {**cached["models"]["legacy_log"], "scores_reused_from_sha256": meta["control_regression_sha256"]}
    wa_array = core.t09.make_analysis_matrix(128, layout="legacy_log")
    gs_array = core.t09.make_synthesis_matrix(128, layout="legacy_log")
    import hashlib
    if (hashlib.sha256(wa_array.tobytes()).hexdigest() != meta["filterbanks"]["legacy_log"]["analysis_sha256"] or
            hashlib.sha256(gs_array.tobytes()).hexdigest() != meta["filterbanks"]["legacy_log"]["synthesis_sha256"]):
        raise ValueError("Fixed legacy filterbanks changed")
    wa, gs = torch.from_numpy(wa_array).to(device), torch.from_numpy(gs_array).to(device)
    kill = core.t11.lf_kill_band_for(250., "legacy_log", 128)
    sources = {row["dataset"]: row for row in cached["sources"]}
    documents = []
    for folder in ("mir1k_gainmix_cache", "onair_true_cache", "mshoxx_stereo_instrumental_cache"):
        path = ROOT / "results" / folder / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        domain = "instrumental" if document["dataset"] == "mshoxx" else document["dataset"]
        if (core.digest(path) != meta["manifest_provenance"][domain]["sha256"] or
                core.digest(path) != sources[document["dataset"]]["sha256"]):
            raise ValueError("Reference manifest changed")
        documents.append(document)
    cambridge = audit.cambridge_manifest(ROOT / "data/datasets/CambridgeMTK-electronic-excerpts")
    if cambridge["source_sha256"] != sources["cambridge_multitrack"]["source_sha256"]:
        raise ValueError("Cambridge source stems changed")
    documents.append(cambridge)
    proxy_mix = ROOT / "results/demons_mir1k_candidate_lf/01_mixture.wav"
    proxy_vocal = ROOT / "results/demons_mir1k_candidate_lf/04_teacher_vocals.wav"
    if (core.digest(proxy_mix) != cached["teacher_proxy"]["mixture_sha256"] or
            core.digest(proxy_vocal) != cached["teacher_proxy"]["teacher_sha256"]):
        raise ValueError("Teacher proxy changed")
    documents.append({"dataset": "dj_teacher_proxy", "tracks": [{"track_id": "Jerro, Sophia Bel - Demons.mp3",
                      "dataset": "dj_teacher_proxy", "split": "holdout", "mix_files": [str(proxy_mix)],
                      "vocal_files": [str(proxy_vocal)]}]})
    musdb = sorted((Path(meta["manifest_provenance"]["musdb"]["root"]) / "test").glob("*.stem.mp4"))
    if len(musdb) != 50 or {str(p): core.digest(p) for p in musdb} != sources["musdb"]["source_sha256"]:
        raise ValueError("MUSDB known regression audio changed")
    documents.append({"dataset": "musdb", "tracks": [{"track_id": p.stem, "dataset": "musdb", "split": "holdout",
                      "mix_files": [str(p)], "vocal_files": []} for p in musdb]})
    rows, seen, baseline_max_delta = [], set(), 0.
    with torch.no_grad():
        for document in documents:
            for record in document["tracks"]:
                if record["split"] != "holdout":
                    continue
                helpers.reject_overlap(meta, document["dataset"], record["track_id"])
                if record["dataset"] == "musdb":
                    mix, vocal = core.decode_musdb(Path(record["mix_files"][0]))
                elif record["dataset"] == "dj_teacher_proxy":
                    mix, vocal = [torch.from_numpy(core.t23._stereo(core.t23._read(Path(p))[0]))
                                  for p in (record["mix_files"][0], record["vocal_files"][0])]
                elif record["dataset"] == "cambridge_multitrack":
                    read = lambda files: [torch.from_numpy(core.t23._stereo(core.t23._read(Path(p))[0])) for p in files]
                    mix, vocal = audit.mix_multitrack(read(record["vocal_files"]), read(record["stem_files"]))
                else:
                    spec = core.t23.TrackSpec(record["track_id"], record["dataset"], record["mix_files"],
                                             record["vocal_files"], record.get("stem_files", []))
                    mix, vocal = core.t23.load_track(spec)
                if mix.shape != vocal.shape:
                    raise ValueError("Reference sample alignment changed")
                for start in range(0, mix.shape[-1], 30*core.SR):
                    end = min(start+30*core.SR, mix.shape[-1])
                    if end-start < 5*core.SR:
                        continue
                    for db in ((0, -12) if record["dataset"] == "cambridge_multitrack" else (0,)):
                        dataset = document["dataset"]+("_weak_minus12" if db else "")
                        key = (dataset, record["track_id"], round(start/core.SR, 6))
                        if key not in row_map or key in seen:
                            raise ValueError("Known regression segmentation changed")
                        seen.add(key)
                        v = vocal[:, start:end]*10**(db/20)
                        x = mix[:, start:end] if not db else mix[:, start:end]-vocal[:, start:end]+v
                        waveform_hash = suite.tensor_digest(x, v)
                        x, v = x.to(device), v.to(device)
                        spectrum = core.t09._stft(x)
                        bands = torch.einsum("fb,cft->cbt", wa, spectrum.abs())
                        metrics = {}
                        for label, net in nets.items():
                            masks = regression.bounded_masks(net, bands)
                            pred = audit.render(spectrum, masks[:2], gs, x.shape[-1], kill)
                            metrics[label] = suite.separation_metrics(pred.cpu(), x.cpu(), v.cpu())
                        for name in ("vocal_si_sdr_db", "accompaniment_si_sdr_db", "vocal_error_snr_db", "accompaniment_error_snr_db"):
                            old, new = row_map[key]["models"]["baseline"][name], metrics["baseline"][name]
                            if (old is None) != (new is None):
                                raise ValueError("Frozen score availability differs")
                            if old is not None:
                                baseline_max_delta = max(baseline_max_delta, abs(old-new))
                        if baseline_max_delta > .005:
                            raise ValueError("Recomputed frozen reference differs from cached regression")
                        row = {k: copy.deepcopy(value) for k, value in row_map[key].items() if k != "models"}
                        row.update(waveforms_sha256=waveform_hash, models={**metrics, "control": row_map[key]["models"]["legacy_log"]})
                        rows.append(row)
                print(f"WEAK_EVAL dataset={document['dataset']} track={record['track_id']} clips={len(seen)}/145", flush=True)
    if seen != set(row_map):
        raise ValueError("Missing known regression clips")
    vs_control = [{**row, "models": {"baseline": row["models"]["control"], "weak_vocal": row["models"]["weak_vocal"]}} for row in rows]
    payload = {"schema": 1, "script_sha256": core.digest(__file__), "training_script_sha256": meta["script_sha256"],
               "regression_helpers_sha256": core.digest(regression.__file__), "experiment_sha256": core.digest(experiment_path),
               "control_regression_sha256": meta["control_regression_sha256"], "models": models, "rows": rows,
               "baseline_vs_cached_max_metric_delta_db": baseline_max_delta,
               "summary": helpers.grouped_summary(rows, ["baseline", "control", "weak_vocal"]),
               "paired_vs_frozen": regression.paired_intervals(rows, ["control", "weak_vocal"]),
               "paired_vs_same_budget_control": regression.paired_intervals(vs_control, ["weak_vocal"]),
               "provisional_float_gate": {name: helpers.provisional_gate(rows, helpers.grouped_summary(rows, ["baseline", "control", "weak_vocal"]), name)
                                           for name in ("control", "weak_vocal")},
               "sources": cached["sources"], "teacher_proxy": cached["teacher_proxy"], "mask_execution": cached["mask_execution"],
               "scope": "Offline same-graph FP32 final-step diagnostic; 145 known clips, no regression checkpoint picking",
               "caveats": cached["caveats"]+ ["Cached same-budget control reused after integrity checks and frozen-score replay",
                                            "Uniform average sample weight is not an equal-gradient-norm guarantee",
                                            "No integer, speaker, or board acceptance; deployed weights remain frozen"]}
    training.control.write_json(output, payload)
    print(f"WEAK_VOCAL_EVAL PASS rows={len(rows)} output={output}", flush=True)


if __name__ == "__main__":
    main()
