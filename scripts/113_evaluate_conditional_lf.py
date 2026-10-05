"""Offline probe: release 120-250 Hz masks only when upper bands predict vocals.

This tests a post-model spectral gate on fixed weights. It is not trained or
board-validated; the activity thresholds are exploratory operating points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("audit_conditional_lf", "109_audit_model_limits.py")
evaluation = load("eval_conditional_lf", "111_evaluate_residual_ablation.py")
cutoff = load("fixed_lf_comparison", "112_compare_lf_cutoffs.py")


def causal_gate(score, threshold, ramp=0.18, attack=0.45, release=0.93):
    """Soft voice-activity gate with an immediate-ish attack and slower release."""
    target = ((score - threshold) / ramp).clamp(0, 1)
    out = torch.empty_like(target)
    state = target.new_zeros(())
    for i in range(target.numel()):
        coefficient = attack if target[i] > state else release
        state = coefficient * state + (1 - coefficient) * target[i]
        out[i] = state
    return out


def grouped_gate_delta(rows, dataset, metric, reference, candidate):
    by_track = {}
    for row in rows:
        if row["dataset"] != dataset or (metric.startswith("vocal_") and not row["vocals_present"]):
            continue
        before, after = row["models"][reference][metric], row["models"][candidate][metric]
        if before is not None and after is not None:
            by_track.setdefault(row["track"], []).append(after - before)
    tracks = {track: sum(vals) / len(vals) for track, vals in by_track.items()}
    values = list(tracks.values())
    return {"n_tracks": len(values), "mean": sum(values) / len(values) if values else None,
            "median": sorted(values)[len(values) // 2] if values else None,
            "improved_tracks": sum(v > 0 for v in values), "per_track": tracks}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=ROOT / "results/residual_finetune_clean_20261001")
    ap.add_argument("--baseline", type=Path, default=ROOT / "models/student_bott2_mir1k_candidate.pt")
    ap.add_argument("--musdb-root", type=Path, default=ROOT / "data/datasets/MUSDB18-7-STEMS")
    ap.add_argument("--output", type=Path, default=ROOT / "results/conditional_lf_probe_20261001.json")
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "试听文件/低频条件释放对照_20261001")
    ap.add_argument("--thresholds", default="0.05,0.10,0.20",
                    help="exploratory high-band mask activity thresholds, comma-separated")
    args = ap.parse_args()
    if args.output.exists() or args.audio_dir.exists():
        raise ValueError("Refusing to overwrite an existing probe result")
    thresholds = [float(v) for v in args.thresholds.split(",")]
    if not thresholds or len(thresholds) != len(set(thresholds)) or any(not 0 <= v < 1 for v in thresholds):
        raise ValueError("Thresholds must be unique values in [0, 1)")

    experiment = json.loads((args.experiment / "experiment.json").read_text(encoding="utf-8"))
    if not experiment.get("deployed_checkpoint_unchanged"):
        raise ValueError("Training experiment is incomplete")
    if experiment["metadata"]["script_sha256"] != audit.sha256(evaluation.training.__file__):
        raise ValueError("Training source no longer matches its receipt")
    if audit.sha256(args.baseline) != experiment["metadata"]["sha256"]:
        raise ValueError("Frozen model hash changed")
    residual_path = args.experiment / "residual_best.pt"
    if audit.sha256(residual_path) != experiment["arms"]["residual"]["best_sha256"]:
        raise ValueError("Residual candidate hash changed")
    baseline, baseline_blob = audit.t13.load_student(args.baseline, "cpu")
    residual, residual_blob = audit.t13.load_student(residual_path, "cpu")
    for net in (baseline, residual):
        if net.frontend != ("linear", 1.) or net.band_layout != "legacy_log" or net.n_bands != 128:
            raise ValueError("Probe requires the fixed linear legacy_log/128 graph")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline, residual = baseline.to(device).eval(), residual.to(device).eval()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wa = torch.from_numpy(audit.t09.make_analysis_matrix()).to(device)
    gs = torch.from_numpy(audit.t09.make_synthesis_matrix()).to(device)
    kill_250 = audit.t11.lf_kill_band_for(250, "legacy_log", 128)
    kill_120 = audit.t11.lf_kill_band_for(120, "legacy_log", 128)
    if (kill_250, kill_120) != (44, 29):
        raise ValueError("Unexpected low-frequency band indices")

    documents, sources = [], []
    for folder in ("mir1k_gainmix_cache", "onair_true_cache", "mshoxx_stereo_instrumental_cache"):
        path = ROOT / "results" / folder / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        domain = "instrumental" if manifest["dataset"] == "mshoxx" else manifest["dataset"]
        if audit.sha256(path) != experiment["metadata"]["manifest_provenance"][domain]["sha256"]:
            raise ValueError(f"Dataset manifest changed: {domain}")
        documents.append(manifest)
        sources.append({"dataset": manifest["dataset"], "path": str(path), "sha256": audit.sha256(path)})
    cambridge = audit.cambridge_manifest(ROOT / "data/datasets/CambridgeMTK-electronic-excerpts")
    documents.append(cambridge)
    sources.append({"dataset": cambridge["dataset"], "recipe": cambridge["recipe"],
                    "source_sha256": cambridge["source_sha256"]})
    proxy_folder = ROOT / "results/demons_mir1k_candidate_lf"
    proxy_mix, proxy_vocal = proxy_folder / "01_mixture.wav", proxy_folder / "04_teacher_vocals.wav"
    if not proxy_mix.exists() or not proxy_vocal.exists():
        raise FileNotFoundError("Pinned Demons teacher-reference excerpt not found")
    documents.append({"dataset": "dj_teacher_proxy", "tracks": [{
        "track_id": "Jerro, Sophia Bel - Demons.mp3", "dataset": "dj_teacher_proxy", "split": "holdout",
        "mix_files": [str(proxy_mix)], "vocal_files": [str(proxy_vocal)]}]})
    sources.append({"dataset": "dj_teacher_proxy", "mixture_sha256": audit.sha256(proxy_mix),
                    "teacher_sha256": audit.sha256(proxy_vocal), "ground_truth": False})
    musdb = sorted((args.musdb_root / "test").glob("*.stem.mp4"))
    if len(musdb) != 50:
        raise ValueError("Expected the pinned 50-song MUSDB test excerpt pool")
    documents.append({"dataset": "musdb", "tracks": [{"track_id": p.stem, "dataset": "musdb",
        "mix_files": [str(p)], "vocal_files": [], "split": "holdout"} for p in musdb]})
    sources.append({"dataset": "musdb", "source_sha256": {str(p): audit.sha256(p) for p in musdb}})

    labels = ["baseline_lf250", "residual_lf250", "residual_lf120"] + [f"conditional_{x:g}" for x in thresholds]
    model_info = {
        "baseline": {"path": str(args.baseline), "sha256": audit.sha256(args.baseline), "step": baseline_blob.get("step")},
        "residual": {"path": str(residual_path), "sha256": audit.sha256(residual_path), "step": residual_blob.get("step")}}
    rows = []
    for document in documents:
        for record in document["tracks"]:
            if record["split"] != "holdout":
                continue
            evaluation.reject_overlap(experiment["metadata"], document["dataset"], record["track_id"])
            spec = audit.t23.TrackSpec(record["track_id"], record["dataset"], record.get("mix_files", []),
                                      record.get("vocal_files", []), record.get("stem_files", []))
            if document["dataset"] == "musdb":
                mix, vocal = evaluation.training.decode_musdb(Path(record["mix_files"][0]))
            elif document["dataset"] == "dj_teacher_proxy":
                mix = torch.from_numpy(audit.t23._stereo(audit.t23._read(Path(spec.mix_files[0]))[0]))
                vocal = torch.from_numpy(audit.t23._stereo(audit.t23._read(Path(spec.vocal_files[0]))[0]))
            elif document["dataset"] == "cambridge_multitrack":
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
                spectrum = audit.t09._stft(x)
                bands = torch.einsum("fb,cft->cbt", wa, spectrum.abs())
                xrms, vrms = x.square().mean().sqrt(), v.square().mean().sqrt()
                vocals_present = bool(vrms > 1e-5 and 20 * torch.log10((vrms + 1e-12)/(xrms + 1e-12)) >= -30)
                base_mask = audit.masks(baseline, bands)[:2]
                mask = audit.masks(residual, bands)[:2]
                models = {}
                models["baseline_lf250"] = audit.waveform_metrics(
                    audit.render(spectrum, base_mask, gs, x.shape[-1], kill_250), x, v)
                models["residual_lf250"] = audit.waveform_metrics(
                    audit.render(spectrum, mask, gs, x.shape[-1], kill_250), x, v)
                models["residual_lf120"] = audit.waveform_metrics(
                    audit.render(spectrum, mask, gs, x.shape[-1], kill_120), x, v)

                upper_energy = bands[:, kill_250:, :].clamp_min(0)
                upper_mask = mask[:, kill_250:, :].clamp(0, 1)
                activity = (upper_energy * upper_mask).sum(dim=(0, 1)) / (
                    upper_energy.sum(dim=(0, 1)) + 1e-8)
                gate_stats = {}
                conditional_audio = {}
                for threshold in thresholds:
                    gate = causal_gate(activity, threshold)
                    gated = mask.clone()
                    gated[:, :kill_120] = 0
                    gated[:, kill_120:kill_250] *= gate[None, None, :]
                    label = f"conditional_{threshold:g}"
                    pv = audit.render(spectrum, gated, gs, x.shape[-1])
                    models[label] = audit.waveform_metrics(pv, x, v)
                    gate_stats[label] = {"mean": float(gate.mean()),
                                         "active_gt_0_5_pct": float((gate > 0.5).float().mean() * 100),
                                         "p95": float(torch.quantile(gate, .95))}
                    if document["dataset"] == "cambridge_multitrack":
                        conditional_audio[label] = (x - pv).cpu()

                rows.append({"dataset": document["dataset"], "track": record["track_id"],
                             "start_s": start / audit.SR, "seconds": x.shape[-1] / audit.SR,
                             "vocals_present": vocals_present, "models": models, "gate_stats": gate_stats})
                if conditional_audio:
                    folder = args.audio_dir / record["track_id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    if start == 0:
                        sf.write(str(folder / "00_mixture.wav"), x.cpu().numpy().T, audit.SR, subtype="FLOAT")
                        sf.write(str(folder / "01_reference_accompaniment.wav"),
                                 (x - v).cpu().numpy().T, audit.SR, subtype="FLOAT")
                        sf.write(str(folder / "02_residual_lf250_accompaniment.wav"),
                                 (x - audit.render(spectrum, mask, gs, x.shape[-1], kill_250)).cpu().numpy().T,
                                 audit.SR, subtype="FLOAT")
                        sf.write(str(folder / "03_residual_lf120_accompaniment.wav"),
                                 (x - audit.render(spectrum, mask, gs, x.shape[-1], kill_120)).cpu().numpy().T,
                                 audit.SR, subtype="FLOAT")
                        for label, audio in conditional_audio.items():
                            sf.write(str(folder / f"{label}_accompaniment.wav"), audio.numpy().T,
                                     audit.SR, subtype="FLOAT")
            print(f"COND_LF dataset={document['dataset']} track={record['track_id']} segments={len(rows)}",
                  flush=True)

    summary_rows = [{**row, "models": {
        "baseline": row["models"]["baseline_lf250"],
        **{label: row["models"][label] for label in labels if label != "baseline_lf250"}}} for row in rows]
    summary = evaluation.grouped_summary(
        summary_rows, ["baseline", *[label for label in labels if label != "baseline_lf250"]])
    changes = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        changes[dataset] = {label: {
            metric: grouped_gate_delta(rows, dataset, metric, "residual_lf250", label)
            for metric in ("vocal_si_sdr_db", "vocal_error_snr_db", "accompaniment_si_sdr_db",
                           "accompaniment_error_snr_db", "joint_fit_residual_vocal_gain",
                           "joint_fit_accompaniment_gain")}
            for label in labels if label not in ("baseline_lf250", "residual_lf250")}
    report = {"script_sha256": audit.sha256(__file__),
              "fixed_cutoff_script_sha256": audit.sha256(cutoff.__file__),
              "evaluation_script_sha256": audit.sha256(evaluation.__file__),
              "training_script_sha256": audit.sha256(evaluation.training.__file__),
              "experiment_sha256": audit.sha256(args.experiment / "experiment.json"),
              "scope": "Exploratory fixed-weight software gate on the same previously viewed holdouts; not trained or board-validated",
              "gate": {"activity": "magnitude-band-weighted mean predicted vocal mask over bands >= LF250 index 44",
                       "thresholds": thresholds, "ramp_width": 0.18, "attack_coefficient": 0.45,
                       "release_coefficient": 0.93,
                       "rule": "zero masks below LF120; multiply LF120-LF250 mask by causal smoothed upper-band activity gate"},
              "cutoff_band_indices": {"lf250": kill_250, "lf120": kill_120},
              "sources": sources, "models": model_info, "rows": rows,
              "summary_vs_baseline_lf250": summary,
              "paired_changes_vs_residual_lf250": changes,
              "aggregation": "within track mean then equal track mean; all deltas paired by track",
              "audio_dir": str(args.audio_dir)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"CONDITIONAL_LF_PROBE PASS segments={len(rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
