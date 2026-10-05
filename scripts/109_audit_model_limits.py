"""Offline model attribution on existing holdouts; never trains or writes SD.

All learned models run with continuous FP32 context. Known-reference masks are
comparison constructions, NOT optimised band-mask ceilings or deployable models.
Input-only INT12 experiments deliberately exclude weight/activation quantisation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


t13 = load_module("ab_model_audit", "13_ab_compare.py")
t23 = load_module("truth_model_audit", "23_build_true_stem_cache.py")
t09, t11 = t13.t09, t13.t11
SR = 44100


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mix_multitrack(vocals, instruments):
    """Common sample zero, zero-extend shorter tracks; shared peak gain only."""
    parts = vocals + instruments
    if not vocals or not instruments:
        raise ValueError("Need explicitly mapped vocal and instrument stems")
    length = max(p.shape[-1] for p in parts)
    def add(stems):
        return torch.stack([F.pad(p.expand(2, -1) if p.shape[0] == 1 else p,
                                  (0, length - p.shape[-1])) for p in stems]).sum(0)
    vocal, accompaniment = add(vocals), add(instruments)
    mix = vocal + accompaniment
    gain = 0.95 / mix.abs().max().clamp_min(1e-12)
    return mix * gain, vocal * gain


def cambridge_manifest(root):
    mappings = {"Skelpolu_HumanMistakes": ["12_LeadVox.wav", "13_BackingVox1.wav", "14_BackingVox2.wav"],
                "Triviul_Angelsaint": ["10_LeadVox.wav", "11_LeadDouble.wav", "12_BackingVox.wav"]}
    records, hashes = [], {}
    for song, names in mappings.items():
        folder = root / song
        waves = sorted(folder.glob("*.wav"))
        if len(waves) != (14 if song.startswith("Skelpolu") else 12):
            raise ValueError(f"Incomplete multitrack: {song}")
        vocal = [folder / name for name in names]
        if not all(p.exists() for p in vocal):
            raise ValueError("Explicit vocal mapping is missing")
        instruments = [p for p in waves if p not in vocal]
        for path in [*waves, folder / "Readme.txt"]:
            hashes[str(path)] = sha256(path)
        records.append({"track_id": song, "dataset": "cambridge_multitrack", "split": "holdout",
                        "mix_files": [], "vocal_files": [str(p) for p in vocal],
                        "stem_files": [str(p) for p in instruments]})
    return {"dataset": "cambridge_multitrack", "tracks": records,
            "recipe": "Raw stems at common sample zero; shorter stems zero-extended; mix=V+A; shared gain sets mixture peak to 0.95",
            "caveat": "Unmastered synthetic mix, two short educational excerpts only, not a representative commercial EDM benchmark",
            "source_sha256": hashes}


def energy_db_ratio(numerator, denominator):
    return float(10 * torch.log10((numerator + 1e-12) / (denominator + 1e-12)))


def waveform_metrics(pred_vocal, mix, ref_vocal):
    """Legacy non-demeaned SI-SDR plus scale-dependent residual error.

    A stereo joint least-squares leakage coefficient is only a diagnostic, not
    an independently isolated residual-vocal waveform or perceptual score.
    """
    pred_a = mix - pred_vocal
    ref_a = mix - ref_vocal
    out = {}
    for name, estimate, reference in (("vocal", pred_vocal, ref_vocal),
                                      ("accompaniment", pred_a, ref_a)):
        est, ref = estimate.double().flatten(), reference.double().flatten()
        power = ref.square().sum()
        if power < 1e-10:
            out[f"{name}_si_sdr_db"] = None
            out[f"{name}_error_snr_db"] = None
        else:
            target = (est @ ref) / power * ref
            out[f"{name}_si_sdr_db"] = (energy_db_ratio(
                target.square().sum(), (est - target).square().sum())
                if est.square().sum() > 1e-10 else None)
            out[f"{name}_error_snr_db"] = energy_db_ratio(
                power, (est - ref).square().sum())
    out["removed_energy_relative_mix_db"] = energy_db_ratio(
        pred_vocal.double().square().sum(), mix.double().square().sum())
    a, v, y = ref_a.double().flatten(), ref_vocal.double().flatten(), pred_a.double().flatten()
    aa, vv, av = a @ a, v @ v, a @ v
    determinant = aa * vv - av * av
    if vv > 1e-10 and determinant > 1e-8 * aa * vv:
        ya, yv = y @ a, y @ v
        ca = (ya * vv - yv * av) / determinant
        cv = (yv * aa - ya * av) / determinant
        out["joint_fit_accompaniment_gain"] = float(ca)
        out["joint_fit_residual_vocal_gain"] = float(cv)
        out["joint_fit_residual_vocal_gain_db"] = float(
            20 * torch.log10(cv.abs().clamp_min(1e-12)))
    else:
        out["joint_fit_accompaniment_gain"] = None
        out["joint_fit_residual_vocal_gain"] = None
        out["joint_fit_residual_vocal_gain_db"] = None
    return out


def mask_input_stats(bands, wa, step):
    active = wa.sum(dim=0) > 0
    values = bands[:, active].flatten()
    q = (values / step).round().clamp(-2048, 2047)
    return {
        "zero_rounding_pct_active_bands": float((q == 0).float().mean() * 100),
        "saturation_pct_active_bands": float((values > 2047 * step).float().mean() * 100),
        "input_quantization_snr_db_active_bands": energy_db_ratio(
            values.double().square().sum(), (values.double() - q.double() * step).square().sum()),
    }


@torch.no_grad()
def masks(net, bands):
    count = bands.shape[-1]
    x = F.pad(bands, (0, (-count) % 8))[None]
    output = net(t11.apply_frontend(x, net.frontend))[0, ..., :count]
    return (output + 1.0) * 0.5


def render(spectrum, band_mask, gs, length, kill=0):
    mask = band_mask.clone()
    mask[:, :kill] = 0
    full_mask = torch.einsum("fb,cbt->cft", gs, mask).clamp(0, 1)
    return t09._istft(spectrum * full_mask, length)


def summary(rows):
    grouped = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == dataset]
        data = {"tracks": len({r["track"] for r in group}), "segments": len(group),
                "seconds": sum(r["seconds"] for r in group), "variants": {}}
        for variant in group[0]["variants"]:
            metrics = {}
            for metric in group[0]["variants"][variant]:
                vals = [r["variants"][variant][metric] for r in group
                        if r["variants"][variant][metric] is not None]
                metrics[metric] = {"n": len(vals),
                                   "mean": statistics.fmean(vals) if vals else None,
                                   "median": statistics.median(vals) if vals else None}
            data["variants"][variant] = metrics
        for field in ("vocal_energy_below_250hz_pct", "vocal_energy_below_500hz_pct"):
            vals = [r[field] for r in group if r[field] is not None]
            data[field] = statistics.fmean(vals) if vals else None
        grouped[dataset] = data
    return grouped


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, action="append")
    ap.add_argument("--cambridge-only", action="store_true", help="Use the two local electronic vocal multitracks only")
    ap.add_argument("--cambridge-root", type=Path, default=ROOT / "data/datasets/CambridgeMTK-electronic-excerpts")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/student_bott2_mir1k_candidate.pt")
    ap.add_argument("--baseline", type=Path, default=ROOT / "models/student_k3s_model.pt")
    ap.add_argument("--input-step", type=float, default=0.07046897899364925)
    ap.add_argument("--segment-s", type=float, default=30.0)
    ap.add_argument("--max-segments-per-track", type=int, default=0, help="0 keeps all segments")
    ap.add_argument("--output", type=Path, default=ROOT / "results/model_audit_20261001.json")
    args = ap.parse_args()
    if args.segment_s < 5 or args.input_step <= 0 or args.max_segments_per_track < 0:
        ap.error("Invalid segment length, input step or segment limit")
    if args.cambridge_only and args.manifest:
        ap.error("Choose manifests OR --cambridge-only")
    manifests = args.manifest or [ROOT / "results" / p / "manifest.json" for p in (
        "mir1k_gainmix_cache", "onair_true_cache", "mshoxx_stereo_instrumental_cache")]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, blob = t13.load_student(args.checkpoint, device)
    baseline, base_blob = t13.load_student(args.baseline, device)
    if net.frontend != ("linear", 1.0) or net.band_layout != "legacy_log" or net.n_bands != 128:
        raise ValueError("This attribution currently targets the deployed linear 128-band graph")
    if baseline.frontend != net.frontend or baseline.band_layout != net.band_layout or baseline.n_bands != 128:
        raise ValueError("Baseline must use identical input representation")
    banks = {layout: tuple(torch.from_numpy(fn(128, layout=layout)).to(device) for fn in (
        t09.make_analysis_matrix, t09.make_synthesis_matrix))
        for layout in ("legacy_log", "mel_unique")}
    wa, gs = banks["legacy_log"]
    kill = t11.lf_kill_band_for(250, "legacy_log", 128)
    print(f"MODEL_AUDIT device={device} checkpoint_step={blob.get('step')} kill_bands={kill}", flush=True)
    rows, sources = [], []
    documents = ([(args.cambridge_root, cambridge_manifest(args.cambridge_root))] if args.cambridge_only else
                 [(path, json.loads(path.read_text(encoding="utf-8"))) for path in manifests])
    for manifest_path, manifest in documents:
        ids = {split: {r["track_id"] for r in manifest["tracks"] if r["split"] == split}
               for split in ("train", "holdout")}
        if ids["train"] & ids["holdout"]:
            raise ValueError("Track overlap in supplied manifest")
        records = [r for r in manifest["tracks"] if r["split"] == "holdout"]
        if not records:
            raise ValueError("Manifest has no holdouts")
        sources.append({"path": str(manifest_path), "sha256": (sha256(manifest_path) if manifest_path.is_file() else
                        hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()),
                        "dataset": manifest["dataset"], "holdout_tracks": sorted(ids["holdout"]),
                        "recipe": manifest.get("recipe"), "source_sha256": manifest.get("source_sha256"),
                        "scope": "song IDs disjoint within manifest; not a universal singer/composition split"})
        for record in records:
            spec = t23.TrackSpec(record["track_id"], record["dataset"], record["mix_files"],
                                 record["vocal_files"], record.get("stem_files", []))
            if spec.dataset == "cambridge_multitrack":
                def read_stems(paths):
                    return [torch.from_numpy(t23._stereo(t23._read(Path(p))[0])) for p in paths]
                mix, vocal = mix_multitrack(read_stems(spec.vocal_files), read_stems(spec.stem_files))
            else:
                mix, vocal = t23.load_track(spec)
            segment_n = round(args.segment_s * SR)
            selected = 0
            for start in range(0, mix.shape[-1], segment_n):
                end = min(start + segment_n, mix.shape[-1])
                if end - start < 5 * SR:
                    continue
                x, v = mix[:, start:end].to(device), vocal[:, start:end].to(device)
                spectrum, v_spec = t09._stft(x), t09._stft(v)
                length = x.shape[-1]
                bands = torch.einsum("fb,cft->cbt", wa, spectrum.abs())
                output = masks(net, bands)
                base_output = masks(baseline, bands)
                quant_input = (bands / args.input_step).round().clamp(-2048, 2047) * args.input_step
                quant_output = masks(net, quant_input)
                projection = ((v_spec * spectrum.conj()).real / (spectrum.abs().square() + 1e-6)).clamp(0, 1)
                versions = {"mixture": torch.zeros_like(x),
                            "baseline_lf250": render(spectrum, base_output[:2], gs, length, kill),
                            "current_lf250": render(spectrum, output[:2], gs, length, kill),
                            "current_lf120": render(spectrum, output[:2], gs, length,
                                                     t11.lf_kill_band_for(120, "legacy_log", 128)),
                            "current_no_lf": render(spectrum, output[:2], gs, length),
                            "current_input_int12_lf250": render(spectrum, quant_output[:2], gs, length, kill),
                            "ref_mask_full513": t09._istft(spectrum * projection, length)}
                for layout, (analysis, synthesis) in banks.items():
                    band_projection = torch.einsum("fb,cft->cbt", analysis, projection)
                    for cutoff in (0, 250):
                        k = t11.lf_kill_band_for(cutoff, layout, 128)
                        versions[f"ref_mask_{layout}_lf{cutoff}"] = render(
                            spectrum, band_projection, synthesis, length, k)
                variants = {name: waveform_metrics(pv, x, v) for name, pv in versions.items()}
                # Inspect independent accompaniment head without treating it as product output.
                pred_head_a = render(spectrum, output[2:4], gs, length)
                variants["current_independent_head_a"] = waveform_metrics(x - pred_head_a, x, v)
                variants["current_independent_head_a_lf250"] = waveform_metrics(
                    render(spectrum, 1 - output[2:4], gs, length, kill), x, v)
                ref_energy = v_spec.abs().double().square()
                frequencies = torch.arange(513, device=device) * SR / 1024
                total_v = ref_energy.sum()
                row = {"dataset": manifest["dataset"], "track": record["track_id"],
                       "start_s": start / SR, "seconds": length / SR,
                       "variants": variants, "input": mask_input_stats(bands, wa, args.input_step),
                       "active_band_head_sum_mae": float(
                           (output[:2, wa.sum(0) > 0, 128:] + output[2:, wa.sum(0) > 0, 128:] - 1).abs().mean()),
                       "active_band_input_quant_mask_mae": float(
                           (quant_output[:2, wa.sum(0) > 0, 128:] - output[:2, wa.sum(0) > 0, 128:]).abs().mean()),
                       "vocal_energy_below_250hz_pct": float(ref_energy[:, frequencies < 250].sum() / total_v * 100) if total_v > 1e-10 else None,
                       "vocal_energy_below_500hz_pct": float(ref_energy[:, frequencies < 500].sum() / total_v * 100) if total_v > 1e-10 else None}
                rows.append(row)
                print(f"  {manifest['dataset']} {record['track_id']} t={start/SR:.1f} "
                      f"A_SDR={variants['current_lf250']['accompaniment_si_sdr_db']:.3f}", flush=True)
                selected += 1
                if args.max_segments_per_track and selected >= args.max_segments_per_track:
                    break
    payload = {"schema": 1, "scope": "Offline, no training, no board audio capture, no SD writes",
               "script_sha256": sha256(__file__), "device": device,
               "torch_version": torch.__version__, "continuous_context": True,
               "stft": {"n_fft": 1024, "hop": 256, "center": True, "pad_mode": "reflect",
                        "note": "Same established truth-eval STFT; not sample-exact board frontend"},
               "metric_notes": {"si_sdr": "Legacy flattened stereo non-demeaned metric; higher is better",
                                "error_snr": "Scale-dependent reference/error energy ratio; higher is better",
                                "ref_mask": "Uses test reference; diagnostic construction, not exact constrained optimum or achievable model score",
                                "input_int12": "Only input quantised; not full integer NPU arithmetic",
                                "joint_fit": "Two-source linear regression diagnostic; not perceptual vocal removal",
                                "aggregation": "Unweighted segment means; no assumption of independent songs or singers",
                                "truth": "MIR-1K true split channels; OnAir aligned raw vocal plus residual backing proxy; mshoxx synthetic stereo instrumental negatives; Cambridge explicitly mapped stems summed into an unmastered mix"},
               "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint),
                              "step": blob.get("step"), "params": blob.get("params"),
                              "config": blob.get("config", blob.get("args")), "frontend": net.frontend},
               "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline),
                            "step": base_blob.get("step"), "params": base_blob.get("params")},
               "input_int12_step": args.input_step, "low_frequency_kill_bands": kill,
               "band_dead_columns": {name: int((bank[0].sum(0) == 0).sum()) for name, bank in banks.items()},
               "manifests": sources, "summary": summary(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"MODEL_AUDIT PASS segments={len(rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
