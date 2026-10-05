"""Evaluate internally selected candidates, without fitting on final holdouts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import soundfile as sf
import torch

from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = spec_from_file_location(name, ROOT / "scripts" / file)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = load("audit_eval_residual", "109_audit_model_limits.py")
training = load("training_eval_residual", "110_train_residual_ablation.py")


def reject_overlap(metadata, dataset, track):
    key = training.composition_key(track)
    for domain, used in metadata.get("splits", {}).items():
        for split in ("train", "validation"):
            if key in {training.composition_key(name) for name in used.get(split, [])}:
                raise ValueError(f"Final holdout composition was used in {domain}/{split}: {dataset}/{track}")
    if key in {training.composition_key(name) for name in metadata.get("teacher", {}).get("source_ids", [])}:
        raise ValueError("Holdout overlaps teacher anchors")


def grouped_summary(rows, labels):
    result = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == dataset]
        result[dataset] = {"tracks": len({r["track"] for r in group}), "segments": len(group),
                           "scorable_vocal_segments": sum(r["vocals_present"] for r in group),
                           "models": {}, "paired_vs_baseline": {}}
        for label in labels:
            values = {}
            for metric in group[0]["models"][label]:
                by_track = {}
                for row in group:
                    v = row["models"][label][metric]
                    if v is not None and (not metric.startswith("vocal_") or row["vocals_present"]):
                        by_track.setdefault(row["track"], []).append(v)
                means = [statistics.fmean(v) for v in by_track.values()]
                values[metric] = {"n_tracks": len(means), "mean": statistics.fmean(means) if means else None,
                                  "median": statistics.median(means) if means else None}
            result[dataset]["models"][label] = values
            if label == "baseline":
                continue
            gains = {}
            for metric in ("vocal_si_sdr_db", "accompaniment_si_sdr_db", "accompaniment_error_snr_db"):
                by_track = {}
                for row in group:
                    a, b = row["models"][label][metric], row["models"]["baseline"][metric]
                    if a is not None and b is not None and (not metric.startswith("vocal_") or row["vocals_present"]):
                        by_track.setdefault(row["track"], []).append(a - b)
                means = [statistics.fmean(v) for v in by_track.values()]
                gains[metric] = {"n_tracks": len(means), "mean": statistics.fmean(means) if means else None,
                                 "median": statistics.median(means) if means else None,
                                 "improved_tracks": sum(v > 0 for v in means)}
            result[dataset]["paired_vs_baseline"][label] = gains
    return result


def provisional_gate(rows, summary, label):
    """Predeclared smoke criteria, not a production-quality confidence claim."""
    reasons = []
    for row in rows:
        for metric in ("vocal_si_sdr_db", "accompaniment_si_sdr_db"):
            if metric.startswith("vocal_") and not row.get("vocals_present", True):
                continue
            value = row["models"][label][metric]
            if value is None or not math.isfinite(value):
                reasons.append(f"Unavailable score on scorable holdout {row['dataset']}/{row['track']}/{metric}")
    for domain in ("mir1k", "musdb", "onair"):
        delta = summary[domain]["paired_vs_baseline"][label]
        for metric in ("vocal_si_sdr_db", "accompaniment_si_sdr_db"):
            if delta[metric]["mean"] is None or delta[metric]["mean"] < -0.15:
                reasons.append(f"{domain}/{metric}: mean regression worse than 0.15 dB")
    delta = summary["mshoxx"]["paired_vs_baseline"][label]
    if delta["accompaniment_error_snr_db"]["mean"] < -0.25:
        reasons.append("instrumental: error SNR mean regresses by more than 0.25 dB")
    for row in rows:
        if row["dataset"] not in ("cambridge_multitrack", "dj_teacher_proxy"):
            continue
        for metric in ("vocal_si_sdr_db", "accompaniment_si_sdr_db"):
            a, b = row["models"][label][metric], row["models"]["baseline"][metric]
            if a is not None and b is not None and a - b < -0.25:
                reasons.append(f"counterexample {row['track']}/{metric}: regression worse than 0.25 dB")
    improvement = any(summary[d]["paired_vs_baseline"][label]["accompaniment_si_sdr_db"]["mean"] >= 0.15
                      for d in ("mir1k", "musdb", "onair"))
    if not improvement:
        reasons.append("No real vocal dataset improves mean accompaniment SI-SDR by at least 0.15 dB")
    return {"passed": not reasons, "reasons": reasons,
            "scope": "Provisional float smoke only; full integer and board acceptance are still required"}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=ROOT / "results/residual_finetune_clean_20261001")
    ap.add_argument("--musdb-root", type=Path, default=ROOT / "data/datasets/MUSDB18-7-STEMS")
    ap.add_argument("--baseline", type=Path, default=ROOT / "models/student_bott2_mir1k_candidate.pt")
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "试听文件/离线目标对照_干净留出_20261001")
    args = ap.parse_args()
    experiment = json.loads((args.experiment / "experiment.json").read_text(encoding="utf-8"))
    if not experiment.get("deployed_checkpoint_unchanged"):
        raise ValueError("Experiment is not complete")
    if experiment["metadata"]["script_sha256"] != audit.sha256(training.__file__):
        raise ValueError("Training source no longer matches experiment provenance")
    paths = {"baseline": args.baseline, **{arm: args.experiment / f"{arm}_best.pt" for arm in experiment["arms"]}}
    output = args.experiment / "holdout_evaluation.json"
    if output.exists():
        raise ValueError("Evaluation already exists; preserve it")
    if args.audio_dir.exists():
        raise ValueError("Listening folder already exists; preserve it")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nets, model_meta = {}, {}
    for label, path in paths.items():
        net, blob = audit.t13.load_student(path, device)
        if label != "baseline" and audit.sha256(path) != experiment["arms"][label]["best_sha256"]:
            raise ValueError("Candidate hash no longer matches experiment")
        if net.frontend != ("linear", 1.) or net.band_layout != "legacy_log" or net.n_bands != 128:
            raise ValueError("Mismatched inference graph")
        nets[label] = net
        model_meta[label] = {"sha256": audit.sha256(path), "step": blob["step"], "path": str(path),
                             "finetune": blob.get("finetune", {})}
    if model_meta["baseline"]["sha256"] != experiment["metadata"]["sha256"]:
        raise ValueError("Wrong baseline checkpoint")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wa = torch.from_numpy(audit.t09.make_analysis_matrix()).to(device)
    gs = torch.from_numpy(audit.t09.make_synthesis_matrix()).to(device)
    documents, sources = [], []
    for folder in ("mir1k_gainmix_cache", "onair_true_cache", "mshoxx_stereo_instrumental_cache"):
        path = ROOT / "results" / folder / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        domain = "instrumental" if manifest["dataset"] == "mshoxx" else manifest["dataset"]
        if audit.sha256(path) != experiment["metadata"]["manifest_provenance"][domain]["sha256"]:
            raise ValueError("Dataset manifest changed after training")
        documents.append(manifest)
        sources.append({"dataset": manifest["dataset"], "path": str(path), "sha256": audit.sha256(path)})
    cambridge = audit.cambridge_manifest(ROOT / "data/datasets/CambridgeMTK-electronic-excerpts")
    documents.append(cambridge)
    sources.append({"dataset": cambridge["dataset"], "source_sha256": cambridge["source_sha256"],
                    "recipe": cambridge["recipe"], "caveat": cambridge["caveat"]})
    proxy_folder = ROOT / "results/demons_mir1k_candidate_lf"
    proxy_mix, proxy_vocal = proxy_folder / "01_mixture.wav", proxy_folder / "04_teacher_vocals.wav"
    if not proxy_mix.exists() or not proxy_vocal.exists():
        raise ValueError("Pinned Demons teacher-reference WAVs are required")
    documents.append({"dataset": "dj_teacher_proxy", "tracks": [{
        "track_id": "Jerro, Sophia Bel - Demons.mp3", "dataset": "dj_teacher_proxy", "split": "holdout",
        "mix_files": [str(proxy_mix)], "vocal_files": [str(proxy_vocal)]}]})
    musdb = sorted((args.musdb_root / "test").glob("*.stem.mp4"))
    if len(musdb) != 50:
        raise ValueError("Expected original 50-song MUSDB sample test pool")
    documents.append({"dataset": "musdb", "tracks": [{"track_id": p.stem, "dataset": "musdb",
        "mix_files": [str(p)], "vocal_files": [], "split": "holdout"} for p in musdb]})
    sources.append({"dataset": "musdb", "source_sha256": {str(p): audit.sha256(p) for p in musdb}})
    rows = []
    for manifest in documents:
        for record in manifest["tracks"]:
            if record["split"] != "holdout":
                continue
            for meta in model_meta.values():
                reject_overlap(meta.get("finetune", {}), manifest["dataset"], record["track_id"])
            spec = audit.t23.TrackSpec(record["track_id"], record["dataset"], record["mix_files"],
                                      record["vocal_files"], record.get("stem_files", []))
            if record["dataset"] == "musdb":
                mix, vocal = training.decode_musdb(Path(record["mix_files"][0]))
            elif record["dataset"] == "dj_teacher_proxy":
                mix = torch.from_numpy(audit.t23._stereo(audit.t23._read(Path(spec.mix_files[0]))[0]))
                vocal = torch.from_numpy(audit.t23._stereo(audit.t23._read(Path(spec.vocal_files[0]))[0]))
                if mix.shape != vocal.shape:
                    raise ValueError("Pinned teacher excerpt is not aligned")
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
                spectrum = audit.t09._stft(x)
                bands = torch.einsum("fb,cft->cbt", wa, spectrum.abs())
                vrms, xrms = v.square().mean().sqrt(), x.square().mean().sqrt()
                present = bool(vrms > 1e-5 and 20 * torch.log10((vrms+1e-12)/(xrms+1e-12)) >= -30)
                metrics, rendered = {}, {}
                for label, net in nets.items():
                    masks = audit.masks(net, bands)
                    pv = audit.render(spectrum, masks[:2], gs, x.shape[-1], 44)
                    metrics[label] = audit.waveform_metrics(pv, x, v)
                    if manifest["dataset"] in ("cambridge_multitrack", "dj_teacher_proxy"):
                        rendered[label] = (x - pv).cpu()
                row = {"dataset": manifest["dataset"], "track": record["track_id"],
                       "start_s": start / audit.SR, "seconds": x.shape[-1] / audit.SR,
                       "vocals_present": present, "models": metrics}
                rows.append(row)
                if rendered:
                    name = (record["track_id"].replace(" - ", "_").replace(",", "_").replace(".mp3", ""))
                    folder = args.audio_dir / name
                    folder.mkdir(parents=True, exist_ok=True)
                    sf.write(str(folder / "00_mixture.wav"), x.cpu().numpy().T, audit.SR, subtype="FLOAT")
                    sf.write(str(folder / "01_reference_accompaniment.wav"), (x-v).cpu().numpy().T, audit.SR, subtype="FLOAT")
                    for i, (label, audio) in enumerate(rendered.items(), start=2):
                        sf.write(str(folder / f"{i:02d}_{label}_accompaniment.wav"), audio.numpy().T, audit.SR, subtype="FLOAT")
            print(f"EVAL dataset={manifest['dataset']} track={record['track_id']} segments={len(rows)}", flush=True)
    summary = grouped_summary(rows, list(nets))
    report = {"script_sha256": audit.sha256(__file__), "experiment_sha256": audit.sha256(args.experiment / "experiment.json"),
              "models": model_meta, "rows": rows, "summary": summary,
              "sources": sources,
              "provisional_float_gate": {name: provisional_gate(rows, summary, name) for name in nets if name != "baseline"},
              "scope": "Continuous FP32, LF250, no board capture; candidates chosen on internal validation only",
              "teacher_proxy": {"track": "Demons", "ground_truth": False,
                                "mixture_sha256": audit.sha256(proxy_mix), "teacher_sha256": audit.sha256(proxy_vocal)},
              "audio_dir": str(args.audio_dir),
              "audio_scope": "Offline float reconstructions, not board recordings; FLOAT WAV avoids PCM clipping",
              "aggregation": "Mean within each track, then mean/median of tracks; vocal scoring excludes silence",
              "musdb_mix": "Decoded stems sum, not separately AAC-coded mixture stream"}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"RESIDUAL_EVAL PASS segments={len(rows)} out={output}", flush=True)


if __name__ == "__main__":
    main()
