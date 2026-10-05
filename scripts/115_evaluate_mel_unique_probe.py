"""Compare the mel_unique short-finetune candidate to the frozen legacy model.

Both checkpoints use the same convolution graph, but each is evaluated with
its matching analysis/synthesis filterbank and 250-Hz output protection.
Holdouts are inference-only and outputs are written to a fresh directory.
"""
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


audit = load("audit_eval_mel_unique", "109_audit_model_limits.py")
training = load("training_eval_mel_unique", "110_train_residual_ablation.py")
helpers = load("eval_helpers_mel_unique", "111_evaluate_residual_ablation.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=ROOT / "results/mel_unique_finetune_20261001")
    ap.add_argument("--musdb-root", type=Path, default=ROOT / "data/datasets/MUSDB18-7-STEMS")
    ap.add_argument("--baseline", type=Path, default=ROOT / "models/student_bott2_mir1k_candidate.pt")
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "试听文件/mel_unique模型对照_20261001")
    args = ap.parse_args()
    experiment_path = args.experiment / "experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    metadata = experiment["metadata"]
    if not experiment.get("deployed_checkpoint_unchanged"):
        raise ValueError("Training experiment is not complete")
    if metadata["script_sha256"] != audit.sha256(args.experiment.parent.parent / "scripts/114_train_mel_unique_probe.py"):
        raise ValueError("Training source no longer matches experiment provenance")
    if metadata["core_script_sha256"] != audit.sha256(training.__file__):
        raise ValueError("Training core changed after the experiment")

    candidate_path = args.experiment / "mel_unique_best.pt"
    output_path = args.experiment / "holdout_evaluation.json"
    if output_path.exists() or args.audio_dir.exists():
        raise ValueError("Evaluation output already exists; choose a fresh path and preserve prior data")
    paths = {"baseline": args.baseline, "mel_unique": candidate_path}
    expected_sha = {"baseline": metadata["sha256"],
                    "mel_unique": experiment["arms"]["mel_unique"]["best_sha256"]}
    layouts = {"baseline": "legacy_log", "mel_unique": "mel_unique"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    nets, model_meta, banks, kills = {}, {}, {}, {}
    for label, path in paths.items():
        if audit.sha256(path) != expected_sha[label]:
            raise ValueError(f"{label} checkpoint does not match recorded hash")
        net, blob = audit.t13.load_student(path, device)
        if net.frontend != ("linear", 1.0) or net.band_layout != layouts[label] or net.n_bands != 128:
            raise ValueError(f"{label} checkpoint representation metadata is inconsistent")
        nets[label] = net
        banks[label] = tuple(torch.from_numpy(fn(128, layout=layouts[label])).to(device)
                             for fn in (audit.t09.make_analysis_matrix,
                                        audit.t09.make_synthesis_matrix))
        kills[label] = audit.t11.lf_kill_band_for(250., layouts[label], 128)
        model_meta[label] = {"path": str(path), "sha256": audit.sha256(path),
                             "step": blob.get("step"), "params": blob.get("params"),
                             "band_layout": layouts[label], "n_bands": 128,
                             "frontend": net.frontend,
                             "kill_bands_lf250": kills[label],
                             "finetune": blob.get("finetune", {})}

    documents, sources = [], []
    for folder in ("mir1k_gainmix_cache", "onair_true_cache", "mshoxx_stereo_instrumental_cache"):
        path = ROOT / "results" / folder / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        domain = "instrumental" if manifest["dataset"] == "mshoxx" else manifest["dataset"]
        if audit.sha256(path) != metadata["manifest_provenance"][domain]["sha256"]:
            raise ValueError(f"{domain} manifest changed after training")
        documents.append(manifest)
        sources.append({"dataset": manifest["dataset"], "path": str(path),
                        "sha256": audit.sha256(path)})

    cambridge = audit.cambridge_manifest(ROOT / "data/datasets/CambridgeMTK-electronic-excerpts")
    documents.append(cambridge)
    sources.append({"dataset": cambridge["dataset"], "source_sha256": cambridge["source_sha256"],
                    "recipe": cambridge["recipe"], "caveat": cambridge["caveat"]})
    proxy_folder = ROOT / "results/demons_mir1k_candidate_lf"
    proxy_mix, proxy_vocal = proxy_folder / "01_mixture.wav", proxy_folder / "04_teacher_vocals.wav"
    if not proxy_mix.exists() or not proxy_vocal.exists():
        raise ValueError("Pinned Demons teacher-reference WAVs are required")
    documents.append({"dataset": "dj_teacher_proxy", "tracks": [{
        "track_id": "Jerro, Sophia Bel - Demons.mp3", "dataset": "dj_teacher_proxy",
        "split": "holdout", "mix_files": [str(proxy_mix)], "vocal_files": [str(proxy_vocal)]}]})
    musdb = sorted((args.musdb_root / "test").glob("*.stem.mp4"))
    if len(musdb) != 50:
        raise ValueError("Expected the original 50-song MUSDB test inventory")
    documents.append({"dataset": "musdb", "tracks": [{
        "track_id": p.stem, "dataset": "musdb", "split": "holdout",
        "mix_files": [str(p)], "vocal_files": []} for p in musdb]})
    sources.append({"dataset": "musdb", "source_sha256": {str(p): audit.sha256(p) for p in musdb}})

    rows = []
    for manifest in documents:
        for record in manifest["tracks"]:
            if record["split"] != "holdout":
                continue
            for meta in model_meta.values():
                helpers.reject_overlap(meta.get("finetune", {}), manifest["dataset"], record["track_id"])
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
                spectrum = audit.t09._stft(x)
                vrms, xrms = v.square().mean().sqrt(), x.square().mean().sqrt()
                present = bool(vrms > 1e-5 and 20 * torch.log10((vrms + 1e-12) / (xrms + 1e-12)) >= -30)
                metrics, rendered = {}, {}
                for label, net in nets.items():
                    wa, gs = banks[label]
                    bands = torch.einsum("fb,cft->cbt", wa, spectrum.abs())
                    predicted_masks = audit.masks(net, bands)
                    estimate = audit.render(spectrum, predicted_masks[:2], gs, x.shape[-1], kills[label])
                    metrics[label] = audit.waveform_metrics(estimate, x, v)
                    if manifest["dataset"] == "cambridge_multitrack":
                        rendered[label] = (x - estimate).cpu()
                rows.append({"dataset": manifest["dataset"], "track": record["track_id"],
                             "start_s": start / audit.SR, "seconds": x.shape[-1] / audit.SR,
                             "vocals_present": present, "models": metrics})
                if rendered:
                    name = record["track_id"].replace(" - ", "_").replace(",", "_")
                    folder = args.audio_dir / name
                    folder.mkdir(parents=True, exist_ok=True)
                    sf.write(str(folder / "00_mixture.wav"), x.cpu().numpy().T, audit.SR, subtype="FLOAT")
                    sf.write(str(folder / "01_reference_accompaniment.wav"), (x - v).cpu().numpy().T,
                             audit.SR, subtype="FLOAT")
                    for i, (label, audio) in enumerate(rendered.items(), start=2):
                        sf.write(str(folder / f"{i:02d}_{label}_accompaniment.wav"),
                                 audio.numpy().T, audit.SR, subtype="FLOAT")
            print(f"EVAL dataset={manifest['dataset']} track={record['track_id']} total_segments={len(rows)}",
                  flush=True)

    if len(rows) != 143:
        raise ValueError(f"Expected the fixed 143-segment evaluation set, got {len(rows)}")
    summary = helpers.grouped_summary(rows, ["baseline", "mel_unique"])
    gate = helpers.provisional_gate(rows, summary, "mel_unique")
    payload = {"schema": 1,
               "scope": "Offline FP32 comparison; matching filterbank per checkpoint; LF250; no board/SD write",
               "script_sha256": audit.sha256(__file__),
               "experiment_sha256": audit.sha256(experiment_path),
               "models": model_meta, "rows": rows, "summary": summary,
               "provisional_float_gate": gate, "sources": sources,
               "teacher_proxy": {"track": "Demons", "ground_truth": False,
                                 "mixture_sha256": audit.sha256(proxy_mix),
                                 "teacher_sha256": audit.sha256(proxy_vocal)},
               "audio_dir": str(args.audio_dir),
               "audio_scope": "Cambridge FLOAT WAV renders only; offline model outputs, not board captures",
               "aggregation": "Mean within track, then mean/median over tracks; vocal scoring excludes silent segments",
               "caveat": "Same holdout suite used by previous reports, not a fresh blind set; the mel_unique candidate was selected on reserved training-pool validation."}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"MEL_UNIQUE_EVAL PASS segments={len(rows)} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
