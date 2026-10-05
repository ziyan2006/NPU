"""CPU-only frozen-model whole-source diagnostics on OLD DEVELOPMENT records.

Preselect one alphabetical source per domain before decoding/scoring. MUSDB is
an excerpt; MIR-1K is a locked concatenation, not an original continuous song.
Compare retained context/stateful16/reset16 and LF protection; no model changes,
new acceptance scoring, optimizer, teacher, full integer NPU claim or SD writes.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time

import soundfile as sf
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("whole_source_development", Path(__file__).with_name("146_evaluate_paired_development.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
ctx = d.m.load("whole_source_context", "105_diagnose_audio_context.py")
m, core, acq, ROOT = d.m, d.core, d.acq, d.ROOT
DEFAULT_OUT = ROOT / "results/full_source_diagnostic_20261002"
HISTORY, BLOCK, SR = 128, 16, 44100
EXPORTS = ("continuous_lf44", "stateful16_lf44", "reset16_lf44")


def selected_records(corpus):
    rows = []
    for domain in ("musdb", "mir1k", "instrumental"):
        pool = sorted(corpus.val[domain], key=lambda row: row["track_id"])
        if not pool or any(row["role"] != "development" for row in pool):
            raise ValueError("Only explicitly locked DEVELOPMENT pools")
        row = pool[0]
        if not 2 <= row["source_container_duration_s"] <= 240:
            raise ValueError("Preselected full source outside bounded duration; do not quality-select a replacement")
        rows.append(row)
    return rows


@torch.no_grad()
def context_masks(net, bands, history, block):
    if (bands.device.type != "cpu" or bands.ndim != 4 or bands.shape[0:3] != (1, 2, 128) or
        bands.shape[-1] % 16 or type(history) is not int or history < 0 or history % 8 or
        type(block) is not int or block < 8 or block % 8):
        raise ValueError("Finite CPU bands, grid-aligned history/block required")
    outputs = []
    for start in range(0, bands.shape[-1], block):
        left = max(0, start - history)
        outputs.append(net(bands[..., left:start + block])[0, :2, ..., start-left:])
    return (torch.cat(outputs, dim=-1) + 1) / 2


def valid_samples(length, joins=()):
    if type(length) is not int or length <= (HISTORY + 4) * 256:
        raise ValueError("Source too short for stable context scoring")
    valid = torch.ones(length, dtype=torch.bool)
    valid[:HISTORY * 256] = False
    valid[-512:] = False
    for sample in joins:
        if type(sample) is not int or not 0 < sample < length:
            raise ValueError("Invalid source concatenation seam")
        valid[max(0, sample - 512):min(length, sample + HISTORY * 256)] = False
    if int(valid.sum()) < SR:
        raise ValueError("Insufficient usable scoring audio; no empty metrics")
    return valid


def boundary_metrics(mask, continuous):
    if mask.shape != continuous.shape or mask.shape[-1] <= HISTORY + 2:
        raise ValueError("Aligned long masks required")
    a, b = mask[:, 44:, HISTORY:], continuous[:, 44:, HISTORY:]
    delta = (a[..., 1:] - a[..., :-1]).abs()
    phase = torch.arange(HISTORY + 1, mask.shape[-1]) % BLOCK == 0
    return {"mask_mae_vs_continuous": float((a - b).abs().mean()),
            "mask_max_abs_vs_continuous": float((a - b).abs().max()),
            "boundary_jump_mean": float(delta[..., phase].mean()),
            "interior_jump_mean": float(delta[..., ~phase].mean())}


@torch.no_grad()
def diagnose(net, x, v, joins=()):
    if (x.shape != v.shape or x.ndim != 2 or x.shape[0] != 2 or x.device.type != "cpu" or v.device.type != "cpu" or
        x.dtype != torch.float32 or v.dtype != torch.float32 or not torch.isfinite(x).all() or not torch.isfinite(v).all()):
        raise ValueError("Aligned finite float32 CPU stereo required")
    original, mode, rng = copy.deepcopy(net.state_dict()), net.training, m.capture_rng("cpu")
    try:
        net.eval()
        wa = torch.from_numpy(core.t09.make_analysis_matrix())
        gs = torch.from_numpy(core.t09.make_synthesis_matrix())
        spectrum = core.stft_batch(x[None])
        bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
        count = bands.shape[-1]
        bands = F.pad(bands, (0, (-count) % BLOCK))
        started = time.perf_counter()
        continuous = context_masks(net, bands, HISTORY, 256)
        continuous_seconds = time.perf_counter() - started
        prefix = min(512, bands.shape[-1])
        full_prefix = (net(bands[..., :prefix])[0, :2] + 1) / 2
        prefix_error = float((continuous[..., :prefix] - full_prefix).abs().max())
        if prefix_error > 1e-4:
            raise ValueError("Bounded retained history differs from continuous prefix")
        print("FULL_SOURCE context reference complete", flush=True)
        stateful = ctx.enable_streaming(copy.deepcopy(net), core.t09)
        start = time.perf_counter()
        stateful_mask = context_masks(stateful, bands, 0, BLOCK)
        stateful_seconds = time.perf_counter() - start
        stateful_error = float((stateful_mask - continuous).abs().max())
        if stateful_error > 1e-4:
            raise ValueError("Layer state retention does not reproduce continuous masks")
        print("FULL_SOURCE stateful16 complete", flush=True)
        start = time.perf_counter()
        reset = context_masks(net, bands, 0, BLOCK)
        reset_seconds = time.perf_counter() - start
        masks = {"continuous_lf44": (continuous, 44), "stateful16_lf44": (stateful_mask, 44),
                 "reset16_lf44": (reset, 44), "continuous_no_lf": (continuous, 0)}
        waves = {name: core.product_vocal(spectrum, mask[None, ..., :count], gs, x.shape[-1], kill)[0]
                 for name, (mask, kill) in masks.items()}
        waves["bypass"] = torch.zeros_like(x)
        truth_spectrum = core.stft_batch(v[None])
        # Diagnostic known-answer constructions; not a learnt/optimal band mask.
        projection = ((truth_spectrum * spectrum.conj()).real / (spectrum.abs().square() + 1e-6)).clamp(0, 1)
        waves["known_reference_full513"] = core.t09._istft((spectrum * projection)[0], x.shape[-1])
        band_projection = torch.einsum("fk,bcft->bckt", wa, projection)
        waves["known_reference_bands_lf44"] = core.product_vocal(spectrum, band_projection, gs, x.shape[-1], 44)[0]
        valid = valid_samples(x.shape[-1], joins)
        scored = {name: d.suite.separation_metrics(pv[:, valid], x[:, valid], v[:, valid]) for name, pv in waves.items()}
        mask_scores = {name: boundary_metrics(mask[..., :count], continuous[..., :count]) for name, (mask, _) in masks.items()}
        blocked = gs[:, 44:].sum(1) <= 1e-12
        power = truth_spectrum.abs().double().square()
        stable_frames = F.pad(valid, (0, (-valid.numel()) % 256)).reshape(-1, 256).all(1)
        stable_frames = F.pad(stable_frames, (0, max(0, power.shape[-1] - stable_frames.numel())))[:power.shape[-1]]
        total = power[..., stable_frames].sum()
        blocked_percent = float(power[:, :, blocked, :][..., stable_frames].sum() / total * 100) if total > 1e-10 else None
        difference = {name: {"max_abs": float((pv - waves["continuous_lf44"]).abs().max()),
                      "rms": float((pv[:, valid] - waves["continuous_lf44"][:, valid]).square().mean().sqrt())}
                      for name, pv in waves.items()}
        result = {"samples": x.shape[-1], "seconds": x.shape[-1] / SR, "scored_samples": int(valid.sum()),
                  "source_join_samples": list(joins), "continuous_prefix_max_mask_error": prefix_error,
                  "stateful16_max_mask_error": stateful_error, "mask_diagnostics": mask_scores,
                  "waveform_metrics": scored, "waveform_difference_vs_continuous": difference,
                  "vocal_stft_energy_in_fully_blocked_lf44_bins_percent": blocked_percent,
                  "processing_seconds_cpu": {"retained_history": continuous_seconds, "stateful16": stateful_seconds, "reset16": reset_seconds},
                  "peak_amplitudes": {name: {"vocal": float(pv.abs().max()), "backing": float((x - pv).abs().max())} for name, pv in waves.items()}}
        if not m.equal_state(original, net.state_dict()):
            raise ValueError("Read-only diagnosis changed frozen model")
        return result, waves
    finally:
        net.load_state_dict(original, strict=True)
        net.train(mode)
        m.restore_rng(rng, "cpu")


def source_joins(record, samples):
    if record["domain"] != "mir1k":
        return []
    # Reuse exactly the locked loader's resampling lengths; source file seams
    # are an existing corpus recipe, not real-time playback dropouts.
    lengths = [core.t23._read(Path(name))[0].shape[-1] for name in record["mix_files"]]
    if sum(lengths) != samples:
        raise ValueError("MIR concatenation boundaries differ from locked loader")
    return [sum(lengths[:i]) for i in range(1, len(lengths))]


def save_wave(path, wave):
    if wave.device.type != "cpu" or not torch.isfinite(wave).all():
        raise ValueError("Finite CPU output required")
    with path.open("xb") as stream:
        sf.write(stream, wave.numpy().T, SR, format="WAV", subtype="FLOAT")


def plan(corpus, records):
    files = [Path(__file__), Path(d.__file__), Path(ctx.__file__), Path(m.__file__), m.data.PROTOCOL,
             m.bulk.OLD_LOCK, ROOT / "models/student_bott2_mir1k_candidate.pt",
             *(ROOT / "scripts" / name for name in ("09_target_model.py", "13_ab_compare.py", "11_smoke_train.py",
                "23_build_true_stem_cache.py", "110_train_residual_ablation.py", "109_audit_model_limits.py", "119_model_selection_suite.py", "125_lock_training_data.py", "126_verify_training_baseline.py"))]
    return acq.seal({"schema": 1, "scope": "Preselected old DEVELOPMENT diagnostics; not new blind acceptance or teacher selection",
        "selection": "First alphabetical locked record in each old domain, chosen before audio scoring",
        "records": records, "bindings": {str(p): acq.sha256(p) for p in files},
        "sample_rate": SR, "history_frames": HISTORY, "block_frames": BLOCK,
        "formal_training": False, "checkpoint_selected": "NONE", "deployment": False})


def verify(out):
    sealed_plan = acq.read_sealed(out / "plan.json")
    report = acq.read_sealed(out / "diagnostic.json")
    if (report["plan_sha256"] != acq.sha256(out / "plan.json") or report["formal_training"] is not False or
        report["checkpoint_selected"] != "NONE" or report["deployment"] is not False or
        len(report["records"]) != len(sealed_plan["records"])):
        raise ValueError("Changed diagnostic scope/plan/coverage")
    for path, digest in (sealed_plan["bindings"] | report["source_files"]).items():
        if acq.sha256(path) != digest:
            raise ValueError("Diagnostic source/role/code binding changed")
    for name, digest in report["audio_files"].items():
        if Path(name).name != name or acq.sha256(out / name) != digest:
            raise ValueError("Diagnostic listening output changed")
    if any(r["metrics"]["stateful16_max_mask_error"] > 1e-4 for r in report["records"]):
        raise ValueError("Context equivalence not demonstrated")
    print(f"FULL_SOURCE_VERIFY PASS sources={len(report['records'])} CPU-only, not acceptance", flush=True)


def run(out):
    m.bulk.guard_output(out)
    if out.exists():
        raise ValueError("Fresh diagnostic directory required; do not overwrite previous evidence")
    corpus = d.LockedDevelopmentCorpus(m.bulk.OLD_LOCK, cache_songs=1)
    records = selected_records(corpus)
    needed = math.ceil(sum(r["source_container_duration_s"] for r in records) * SR * 2 * 4 * 6 * 1.2)
    if shutil.disk_usage(ROOT / "results").free < needed + 12 * 1024**3:
        raise ValueError("Insufficient diagnostic audio budget plus 12 GiB reserve")
    sealed_plan = plan(corpus, records)
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", sealed_plan)  # BEFORE new audio scores.
    protocol = json.loads(m.data.PROTOCOL.read_text(encoding="utf-8"))
    net = m.frozen_factory(protocol)().eval()
    rows, audio_files = [], {}
    for index, record in enumerate(records):
        print(f"FULL_SOURCE {index + 1}/{len(records)} {record['domain']} {record['track_id']}", flush=True)
        x, v = corpus.audio(record)
        x, v = x.clone(), v.clone()
        gain = min(1., .95 / max(float(x.abs().max()), 1e-12))
        x, v = x * gain, v * gain
        joins = source_joins(record, x.shape[-1])
        metrics, waves = diagnose(net, x, v, joins)
        export_waves = {"mix": x, "reference_backing": x - v, "reference_vocal": v}
        export_waves.update({name: x - waves[name] for name in EXPORTS})
        for name, wave in export_waves.items():
            path = out / f"source_{index + 1:02d}_{name}.wav"
            save_wave(path, wave)
            audio_files[path.name] = acq.sha256(path)
        rows.append({"domain": record["domain"], "track_id": record["track_id"], "role": "development",
                     "common_gain": gain, "input_pcm_sha256": m.pilot.wave_digest(x), "metrics": metrics})
        print(f"FULL_SOURCE completed seconds={metrics['seconds']:.3f} stateful_mask_error={metrics['stateful16_max_mask_error']:.3g}", flush=True)
    report = {"schema": 1, "plan_sha256": acq.sha256(out / "plan.json"), "records": rows,
              "source_files": corpus.file_hashes, "audio_files": audio_files, "device": "cpu",
              "formal_training": False, "checkpoint_selected": "NONE", "deployment": False,
              "notes": ["Full decoded locked sources; MUSDB is a 6.8s excerpt, MIR consists of joined karaoke clips",
                        "Exclude initial 128 frames, final 512 samples and MIR splice transient regions from metrics",
                        "Fixed FP32/legacy_log/LF44; reflect STFT boundaries, not sample-exact board frontend",
                        "Known-reference masks use answers; not attainable scores or an optimized band-mask ceiling",
                        "Mask jumps/waveform difference are diagnostics, not perceived stutter or analogue board recordings",
                        "FLOAT32 WAVs retain one common input gain, no per-variant normalization/clipping",
                        "CPU timings are not FPGA real-time latency or GPU throughput; no independent new-song acceptance"]}
    acq.write_new_json(out / "diagnostic.json", acq.seal(report))
    verify(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.verify:
        verify(args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
