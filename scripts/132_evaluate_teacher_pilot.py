"""Score sealed teacher outputs against old DEVELOPMENT references only.

MP3 disagreement/removal levels are diagnostics, never ground-truth quality.
No screening result permits automatic full relabeling or student promotion.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import statistics

import torch

spec = importlib.util.spec_from_file_location("teacher_pilot_evaluation_core", Path(__file__).with_name("131_run_teacher_pilot.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
metrics_core = pilot.load("teacher_pilot_waveform_metrics", "109_audit_model_limits.py")


def summarize(rows):
    result = {}
    for domain in sorted({r["domain"] for r in rows if r["role"] == "development"}):
        group = [r for r in rows if r["domain"] == domain]
        result[domain] = {}
        for teacher in pilot.TEACHERS:
            result[domain][teacher] = {}
            for key in group[0]["teachers"][teacher]["metrics"]:
                by_song = {}
                for row in group:
                    value = row["teachers"][teacher]["metrics"][key]
                    if value is not None:
                        if not math.isfinite(value):
                            raise ValueError("Nonfinite teacher evaluation metric")
                        by_song.setdefault(row["track_id"], []).append(value)
                values = {song: statistics.fmean(v) for song, v in by_song.items()}
                result[domain][teacher][key] = {"songs": len(values), "mean": statistics.fmean(values.values()) if values else None,
                                               "by_song": values}
    return result


def assess(summary, policy):
    reasons, deltas, snr = [], {}, []
    base, candidate = pilot.TEACHERS
    for domain in ("musdb/native", "musdb/weak_minus12"):
        if domain not in summary:
            return {"screening_passed": False, "reasons": ["Missing required vocal development domain"]}
        b, c = summary[domain][base], summary[domain][candidate]
        required = ("vocal_error_snr_db", "joint_fit_accompaniment_gain", "residual_vocal_abs_gain")
        if any(not b[k]["by_song"] or set(b[k]["by_song"]) != set(c[k]["by_song"]) for k in required):
            return {"screening_passed": False, "reasons": ["Unpaired/unavailable vocal metrics"]}
        gain = c["vocal_error_snr_db"]["mean"] - b["vocal_error_snr_db"]["mean"]
        preservation = c["joint_fit_accompaniment_gain"]["mean"] - b["joint_fit_accompaniment_gain"]["mean"]
        residue = 20 * math.log10(max(c["residual_vocal_abs_gain"]["mean"], 1e-12) / max(b["residual_vocal_abs_gain"]["mean"], 1e-12))
        deltas[domain] = {"vocal_error_snr_gain_db": gain, "accompaniment_gain_delta": preservation,
                          "projected_residual_vocal_change_db": residue}
        snr.append(gain)
        if gain < -policy["maximum_any_vocal_domain_regression_db"]:
            reasons.append(f"Vocal domain regressed: {domain}")
        if preservation < -policy["maximum_mean_accompaniment_gain_drop"]:
            reasons.append(f"Accompaniment preservation regressed: {domain}")
        if residue > policy["maximum_residual_vocal_increase_db_per_domain"]:
            reasons.append(f"Projected remaining vocal increased: {domain}")
    mean_gain = statistics.fmean(snr)
    if mean_gain < policy["minimum_mean_vocal_error_snr_gain_db"]:
        reasons.append("Insufficient mean vocal reconstruction gain")
    domain = "instrumental/native"
    if domain not in summary:
        return {"screening_passed": False, "reasons": ["Missing pure instrumental controls"]}
    b = summary[domain][base]["removed_energy_relative_mix_db"]
    c = summary[domain][candidate]["removed_energy_relative_mix_db"]
    if not b["by_song"] or set(b["by_song"]) != set(c["by_song"]):
        return {"screening_passed": False, "reasons": ["Unpaired instrumental controls"]}
    delta = c["mean"] - b["mean"]
    deltas[domain] = {"false_removal_energy_change_db": delta}
    if delta > policy["maximum_instrumental_removed_energy_increase_db"]:
        reasons.append("Pure instrumental false removal increased")
    return {"screening_passed": not reasons, "mean_vocal_error_snr_gain_db": mean_gain,
            "deltas": deltas, "reasons": reasons, "whole_library_relabel_authorized": False,
            "student_training_or_promotion_authorized": False, "source_listening_review": "pending"}


def evaluate(out, asset_path, report_path):
    if report_path.exists():
        raise ValueError("Use a new evaluation report; never overwrite evidence")
    if not report_path.resolve().is_relative_to((pilot.ROOT / "results").resolve()):
        raise ValueError("Detailed private-song evaluation remains in ignored results")
    doc = pilot.verify_cohort(out)
    runtime = pilot.runtime_binding(asset_path)
    rows, receipts = [], {}
    edge = round(doc["protocol"]["development"]["score_edge_seconds"] * pilot.SR)
    for row in doc["records"]:
        x = pilot.read_wave(row["mix_file"])
        if x.shape[-1] <= 2 * edge:
            raise ValueError("No edge-safe teacher scoring region")
        interval = slice(edge, x.shape[-1]-edge)
        reference = pilot.read_wave(row["reference_file"]) if row["reference_file"] else None
        scored = {"clip_id": row["clip_id"], "track_id": row["track_id"], "role": row["role"], "domain": row["domain"],
                  "seconds": row["seconds"], "teachers": {}}
        predictions = {}
        for teacher in pilot.TEACHERS:
            folder = out / teacher / row["clip_id"]
            binding = pilot.entry_binding(out, teacher, asset_path, runtime)
            entry = pilot.verify_entry(folder, binding, row)
            receipts[str((folder / "entry.json").resolve())] = pilot.acq.sha256(folder / "entry.json")
            prediction = pilot.read_wave(folder / "vocals.wav")
            predictions[teacher] = prediction
            if reference is not None:
                metrics = metrics_core.waveform_metrics(prediction[:, interval], x[:, interval], reference[:, interval])
                value = metrics["joint_fit_residual_vocal_gain"]
                metrics["residual_vocal_abs_gain"] = abs(value) if value is not None else None
                scored["teachers"][teacher] = {"metrics": metrics, "elapsed_s": entry["elapsed_s"], "rtf": entry["rtf"],
                                                "peak_cuda_allocated_bytes": entry["peak_cuda_allocated_bytes"]}
            else:
                power_ratio = float(10 * torch.log10((prediction.double().square().sum()+1e-12) / (x.double().square().sum()+1e-12)))
                scored["teachers"][teacher] = {"removed_energy_relative_mix_db_diagnostic_only": power_ratio,
                                                "true_reference_metrics": None, "elapsed_s": entry["elapsed_s"], "rtf": entry["rtf"],
                                                "peak_cuda_allocated_bytes": entry["peak_cuda_allocated_bytes"]}
        disagreement = (predictions[pilot.TEACHERS[1]] - predictions[pilot.TEACHERS[0]]).double()
        scored["teacher_disagreement_relative_mix_db_diagnostic_only"] = float(10 * torch.log10(
            (disagreement.square().sum()+1e-12) / (x.double().square().sum()+1e-12)))
        rows.append(scored)
    summary = summarize(rows)
    gate = assess(summary, doc["protocol"]["screening_gate"])
    performance = {}
    for teacher in pilot.TEACHERS:
        entries = [row["teachers"][teacher] for row in rows]
        elapsed = sum(entry["elapsed_s"] for entry in entries)
        seconds = sum(row["seconds"] for row in rows)
        performance[teacher] = {"inference_elapsed_s": elapsed, "audio_seconds_including_remixes": seconds,
                               "aggregate_rtf": elapsed/seconds,
                               "max_cuda_allocated_bytes": max(entry["peak_cuda_allocated_bytes"] for entry in entries),
                               "timing_scope": "Individual label inference; excludes loading, parity and disk serialization"}
    report = {"schema": 1, "cohort_sha256": pilot.acq.sha256(out / "cohort.json"),
              "protocol_sha256": doc["protocol_sha256"], "script_sha256": pilot.acq.sha256(__file__),
              "receipts_sha256": receipts, "rows": rows, "development_summary": summary,
              "performance": performance, "screening": gate, "blind_evaluation_performed": False,
              "caveats": ["Old DEVELOPMENT, not unseen benchmark; teacher pretraining data incomplete",
                          "MP3 disagreement and removal levels do not show which teacher is correct",
                          "Joint projection residual coefficient is diagnostic, not an isolated vocal waveform",
                          "No human source listening completed; no teacher/student automatically approved"]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pilot.acq.write_new_json(report_path, pilot.acq.seal(report))
    print(f"TEACHER_SCREENING passed={gate['screening_passed']} mean_vocal_gain={gate.get('mean_vocal_error_snr_gain_db')}; "
          "whole_library_relabel=False student_promotion=False", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=pilot.DEFAULT_OUT)
    ap.add_argument("--assets", type=Path, default=pilot.assets.ASSETS)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    evaluate(args.out, args.assets, args.report or args.out / "evaluation.json")


if __name__ == "__main__":
    main()
