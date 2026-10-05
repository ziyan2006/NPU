"""Compare LF250 vs LF120 on the same fixed-weight holdouts.

This is an offline diagnosis only. It never trains, quantizes, changes RTL,
packages a board image, or writes removable media.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import soundfile as sf
import torch

import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("audit_lf_cutoff", "109_audit_model_limits.py")
evaluation = load("eval_lf_cutoff", "111_evaluate_residual_ablation.py")


def grouped_cutoff_delta(rows, dataset, metric, before, after):
    by_track = {}
    for row in rows:
        if row["dataset"] != dataset:
            continue
        if metric.startswith("vocal_") and not row["vocals_present"]:
            continue
        a = row["models"][before][metric]
        b = row["models"][after][metric]
        if a is None or b is None:
            continue
        by_track.setdefault(row["track"], []).append(b - a)
    per_track = {name: statistics.fmean(values) for name, values in by_track.items()}
    vals = list(per_track.values())
    return {"n_tracks": len(vals), "mean": statistics.fmean(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "improved_tracks": sum(v > 0 for v in vals),
            "per_track": per_track}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=ROOT / "results/residual_finetune_clean_20261001")
    ap.add_argument("--baseline", type=Path, default=ROOT / "models/student_bott2_mir1k_candidate.pt")
    ap.add_argument("--musdb-root", type=Path, default=ROOT / "data/datasets/MUSDB18-7-STEMS")
    ap.add_argument("--output", type=Path, default=ROOT / "results/lf_cutoff_comparison_20261001.json")
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "试听文件/低频保护对照_20261001")
    args = ap.parse_args()

    for path in (args.output, args.audio_dir):
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing output: {path}")
    experiment = json.loads((args.experiment / "experiment.json").read_text(encoding="utf-8"))
    if not experiment.get("deployed_checkpoint_unchanged"):
        raise ValueError("Fine-tuning experiment is incomplete")
    if experiment["metadata"]["script_sha256"] != audit.sha256(evaluation.training.__file__):
        raise ValueError("Training source no longer matches its manifest")
    baseline, base_blob = audit.t13.load_student(args.baseline, "cpu")
    candidate_path = args.experiment / "residual_best.pt"
    candidate, candidate_blob = audit.t13.load_student(candidate_path, "cpu")
    if audit.sha256(args.baseline) != experiment["metadata"]["sha256"]:
        raise ValueError("Frozen baseline hash changed")
    if audit.sha256(candidate_path) != experiment["arms"]["residual"]["best_sha256"]:
        raise ValueError("Residual candidate hash changed")
    for net in (baseline, candidate):
        if net.frontend != ("linear", 1.) or net.band_layout != "legacy_log" or net.n_bands != 128:
            raise ValueError("Low-frequency comparison requires the frozen legacy_log/128 graph")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline, candidate = baseline.to(device).eval(), candidate.to(device).eval()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wa = torch.from_numpy(audit.t09.make_analysis_matrix()).to(device)
    gs = torch.from_numpy(audit.t09.make_synthesis_matrix()).to(device)
    cutoffs = {hz: audit.t11.lf_kill_band_for(hz, "legacy_log", 128) for hz in (250, 120)}
    labels = ("baseline_lf250", "baseline_lf120", "residual_lf250", "residual_lf120")

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
        raise FileNotFoundError("Pinned Demons teacher reference not found")
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

    model_info = {
        "baseline": {"path": str(args.baseline), "sha256": audit.sha256(args.baseline), "step": base_blob.get("step")},
        "residual": {"path": str(candidate_path), "sha256": audit.sha256(candidate_path),
                     "step": candidate_blob.get("step")}}
    rows = []
    for document in documents:
        for record in document["tracks"]:
            if record["split"] != "holdout":
                continue
            for model in ("baseline", "residual"):
                metadata = experiment["metadata"] if model == "residual" else {
                    "splits": {"mir1k": {"train": [], "validation": []}}, "teacher": {"source_ids": []}}
                evaluation.reject_overlap(metadata, document["dataset"], record["track_id"])
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
                outputs = {}
                for name, net in (("baseline", baseline), ("residual", candidate)):
                    mask = audit.masks(net, bands)[:2]
                    for hz, kill in cutoffs.items():
                        label = f"{name}_lf{hz}"
                        pv = audit.render(spectrum, mask, gs, x.shape[-1], kill)
                        outputs[label] = audit.waveform_metrics(pv, x, v)
                        if document["dataset"] == "cambridge_multitrack":
                            folder = args.audio_dir / record["track_id"]
                            folder.mkdir(parents=True, exist_ok=True)
                            if start == 0 and name == "baseline" and hz == 250:
                                sf.write(str(folder / "00_mixture.wav"), x.cpu().numpy().T, audit.SR, subtype="FLOAT")
                                sf.write(str(folder / "01_reference_accompaniment.wav"),
                                         (x - v).cpu().numpy().T, audit.SR, subtype="FLOAT")
                            sf.write(str(folder / f"{label}_accompaniment.wav"),
                                     (x - pv).cpu().numpy().T, audit.SR, subtype="FLOAT")
                rows.append({"dataset": document["dataset"], "track": record["track_id"],
                             "start_s": start / audit.SR, "seconds": x.shape[-1] / audit.SR,
                             "vocals_present": vocals_present, "models": outputs})
            print(f"LF_EVAL dataset={document['dataset']} track={record['track_id']} segments={len(rows)}",
                  flush=True)

    summary_rows = [{**row, "models": {
        "baseline": row["models"]["baseline_lf250"],
        **{label: row["models"][label] for label in labels if label != "baseline_lf250"}}}
        for row in rows]
    summary = evaluation.grouped_summary(
        summary_rows, ["baseline", *[label for label in labels if label != "baseline_lf250"]])
    cutoff = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        cutoff[dataset] = {}
        for model in ("baseline", "residual"):
            before, after = f"{model}_lf250", f"{model}_lf120"
            cutoff[dataset][model] = {
                metric: grouped_cutoff_delta(rows, dataset, metric, before, after)
                for metric in ("vocal_si_sdr_db", "vocal_error_snr_db", "accompaniment_si_sdr_db",
                               "accompaniment_error_snr_db", "joint_fit_residual_vocal_gain",
                               "joint_fit_accompaniment_gain")}
    report = {"script_sha256": audit.sha256(__file__),
              "evaluation_script_sha256": audit.sha256(evaluation.__file__),
              "training_script_sha256": audit.sha256(evaluation.training.__file__),
              "experiment_sha256": audit.sha256(args.experiment / "experiment.json"),
              "scope": "Fixed weights; same 143 holdout segments; FP32 offline; no board capture; not a fresh blind test",
              "source_audio_sha256": sources, "models": model_info,
              "cutoffs_hz": cutoffs, "rows": rows, "summary_vs_baseline_lf250": summary,
              "paired_lf120_minus_lf250": cutoff,
              "aggregation": "segment means within each track, then equal mean across tracks; changes paired by track",
              "audio_dir": str(args.audio_dir)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"LF_CUTOFF_COMPARE PASS segments={len(rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
