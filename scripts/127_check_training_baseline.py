"""Independent reload/scoring audit for a completed TRAIN-only small-fit run.

Optional --compare-models checks real resume parity against an uninterrupted
run, comparing tensor values rather than checkpoint container metadata hashes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("small_fit_audit", ROOT / "scripts/126_verify_training_baseline.py")
fit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fit
spec.loader.exec_module(fit)
lock, core = fit.lock, fit.core


def compare_scores(a, b):
    if len(a["clips"]) != len(b["clips"]):
        raise ValueError("Different fixed-cohort coverage")
    largest = abs(a["loss"]-b["loss"])
    for ar, br in zip(a["clips"], b["clips"]):
        if any(ar[key] != br[key] for key in ("track", "samples_sha256", "vocal_gain_db")):
            raise ValueError("Different fixed-cohort waveform")
        for key, value in ar.items():
            if key.endswith("_db") or key.endswith("_gain") or key == "loss":
                expected = br[key]
                if value is None or expected is None:
                    if value != expected:
                        raise ValueError("A projection diagnostic became unavailable")
                elif not (math.isfinite(value) and math.isfinite(expected)):
                    raise ValueError("Non-finite audit score")
                else:
                    largest = max(largest, abs(value-expected))
    if largest > 1e-5:
        raise ValueError(f"Saved-weight score mismatch: {largest}")
    return largest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--data-lock", type=Path, default=ROOT / "results/training_protocol_20261001/dataset_lock.json")
    ap.add_argument("--protocol", type=Path, default=ROOT / "docs/model_training_protocol_v1.json")
    ap.add_argument("--compare-models", type=Path, help="Uninterrupted historical run with the same cohort/config; no deployment comparison")
    args = ap.parse_args()
    report = json.loads((args.experiment / "experiment.json").read_text(encoding="utf-8"))
    if report["status"] != "complete" or report["deployed_model_promoted"] or report["formal_training_started"]:
        raise ValueError("Expected a completed, unpromoted small-fit run")
    if lock.document_digest(report["inputs"]) != report["binding"]:
        raise ValueError("Run input binding is corrupt")
    for path, name in ((args.data_lock, "data_lock_sha256"), (args.protocol, "protocol_sha256")):
        if lock.sha256(path) != report["inputs"][name]:
            raise ValueError("Current protocol/data lock differs from experiment")
    for name, expected in report["inputs"]["code_sha256"].items():
        if lock.sha256(ROOT / "scripts" / name) != expected:
            raise ValueError(f"Training recipe changed: {name}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    fit.validate_protocol(protocol)
    locked = lock.verify_lock(args.data_lock)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = report["inputs"]["device"]
    if str(torch.__version__) != report["inputs"]["torch_version"] or (
        device == "cuda" and (not torch.cuda.is_available() or torch.cuda.get_device_name(0) != report["inputs"]["gpu"])):
        raise ValueError("Audit runtime differs from experiment")
    corpus = core.Corpus(ROOT / "data/datasets/MUSDB18-7-STEMS", ROOT / "results/opt_bott2_5m_cache", protocol["seed"])
    rows = fit.build_cohort(corpus, protocol["small_fit"])
    if [{k: v for k, v in row.items() if k not in ("x", "v")} for row in rows] != report["inputs"]["cohort"]:
        raise ValueError("Rebuilt audio cohort changed")
    wa = torch.from_numpy(core.t09.make_analysis_matrix()).to(device)
    gs = torch.from_numpy(core.t09.make_synthesis_matrix()).to(device)
    frozen, _ = core.t13.load_student(Path(locked["checkpoint"]), device)
    audit = {"status": "verified", "run_binding": report["binding"],
             "checker_sha256": lock.sha256(__file__), "arms": {}, "scope": "TRAIN-only reloaded-weight audit"}
    if args.compare_models:
        old = json.loads((args.compare_models / "experiment.json").read_text(encoding="utf-8"))
        if old["status"] != "complete" or old["inputs"]["cohort"] != report["inputs"]["cohort"] or old["protocol"] != protocol:
            raise ValueError("Comparison is not the same completed cohort/protocol")
    for arm in protocol["small_fit"]["arms"]:
        run = report["arms"][arm]
        path = args.experiment / f"{arm}_final.pt"
        if run["status"] != "complete" or lock.sha256(path) != run["final_sha256"]:
            raise ValueError("Final research checkpoint changed")
        net, blob = core.t13.load_student(path, device)
        if (blob["small_fit"]["binding"] != report["binding"] or blob["small_fit"]["arm"] != arm or
            blob["step"] != protocol["small_fit"]["steps"] or blob["steps"] != protocol["small_fit"]["steps"]):
            raise ValueError("Research checkpoint step/arm/protocol mismatch")
        baseline = fit.evaluate(frozen, rows, wa, gs, device, protocol["small_fit"], arm)
        score = fit.evaluate(net, rows, wa, gs, device, protocol["small_fit"], arm)
        largest = max(compare_scores(baseline, run["baseline"]), compare_scores(score, run["curve"][-1]["scores"]))
        assessment = fit.assess_fit(score, baseline, protocol["small_fit"]["gate"])
        if assessment != run["final_assessment"]:
            raise ValueError("Final TRAIN gate was not reproducible")
        audit["arms"][arm] = {"score_max_abs_difference": largest, "assessment": assessment,
                              "final_sha256": run["final_sha256"]}
        if args.compare_models:
            old_path = args.compare_models / f"{arm}_final.pt"
            if lock.sha256(old_path) != old["arms"][arm]["final_sha256"]:
                raise ValueError("Uninterrupted comparison weights changed")
            old_blob = torch.load(old_path, map_location="cpu", weights_only=False)
            if set(old_blob["model"]) != set(blob["model"]):
                raise ValueError("Resume comparison model graph changed")
            largest_weight = max(float((blob["model"][k].cpu()-old_blob["model"][k].cpu()).abs().max())
                                 for k in blob["model"])
            if largest_weight != 0.:
                raise ValueError(f"Resume changed final learned weights: {largest_weight}")
            audit["arms"][arm]["uninterrupted_model_max_difference"] = largest_weight
    if lock.sha256(locked["checkpoint"]) != lock.FROZEN_SHA:
        raise ValueError("Frozen model changed")
    audit["deployed_checkpoint_unchanged"] = True
    fit.write_json(args.experiment / "reload_audit.json", audit)
    print(f"SMALL_FIT_AUDIT VERIFIED saved scores/model binding; resume_parity={bool(args.compare_models)}; TRAIN only", flush=True)


if __name__ == "__main__":
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
