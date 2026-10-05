"""Read saved paired development scores and export aligned NONRELEASE listening.

Never connects to a live model, constructs an optimizer, initializes CUDA,
changes training inputs or chooses a deployment checkpoint. Only complete
250-step checkpoint/score pairs are read. Audio uses the previously selected
three OLD DEVELOPMENT sources, first 20 seconds (MUSDB is only 6.8 seconds).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import soundfile as sf
import torch
import torch.nn.functional as F


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a = load("exploration_review_audio", "148_diagnose_full_source_audio.py")
inp = load("exploration_review_import", "149_prepare_exploratory_import.py")
dev, m, core, acq, ROOT = a.d, a.m, a.core, a.acq, a.ROOT
ARMS, PURPOSE = m.ARMS, inp.PURPOSE
DEFAULT_RUN = ROOT / "results/paired_exploration_20261003"
DEFAULT_OUT = ROOT / "results/paired_exploration_review_20261003"
SOURCE_PLAN_SHA = "6ed9734fa5f90477fa8f8f1c3004c6abf3c65cf8eb99b58b2f004697f870feac"
SOURCE_REPORT_SHA = "3fc21f827d8f21b8477777c230df7df9eedb326cb119b2e1fd12751fd7b0aded"
LISTEN_SECONDS = 20
SCOPE = "Old true-reference development only; not independent acceptance or release approval"


def expected_binding(approval):
    return {"approval_sha256": acq.sha256(approval),
            "trainer_sha256": acq.sha256(Path(__file__).with_name("150_train_paired_exploration.py")),
            "importer_sha256": acq.sha256(inp.__file__),
            "device_helpers_sha256": acq.sha256(Path(__file__).with_name("147_paired_device_mechanics.py")),
            "development_adapter_sha256": acq.sha256(dev.__file__)}


def plain(doc):
    return {k: v for k, v in doc.items() if k != "content_sha256"}


def check_scores(result, manifest, metric_names=None):
    rows = result["rows"]
    if not rows or [{k: v for k, v in r.items() if k != "metrics"} for r in rows] != manifest["rows"]:
        raise ValueError("Missing/reordered/unpaired development views")
    names = set(rows[0]["metrics"]) if metric_names is None else set(metric_names)
    if any(set(r["metrics"]) != names for r in rows):
        raise ValueError("Changed/missing metric coverage")
    calculated = dev.suite.aggregate(rows)  # rejects nonfinite scalar scores
    if result["summary"] != calculated:
        raise ValueError("Stored summary differs from original per-track aggregation")
    return calculated


def check_pair(receipt, state, packet, bound, manifest, baseline):
    step = receipt.get("step")
    if (type(step) is not int or not 250 <= step <= 1000 or step % 250 or
        receipt.get("purpose") != PURPOSE or receipt.get("deployment_authorized") is not False or
        receipt.get("binding") != bound or receipt.get("checkpoint") != f"NONRELEASE_pair_step_{step:04d}.pt"):
        raise ValueError("Require a complete NONRELEASE quarter-step receipt")
    if (state.get("schema") != 1 or state.get("purpose") != PURPOSE or state.get("step") != step or
        state.get("limit") != 1000 or state.get("binding") != bound or state.get("deployment_authorized") is not False or
        state["sampler"]["cursor"] != step or state["sampler"]["approval_sha256"] != bound["approval_sha256"] or
        state["schedule"]["step"] != step or state["schedule"]["last_validation"] != step or
        set(state.get("arms", {})) != set(ARMS) or any(v["updates"] != step for v in state["arms"].values())):
        raise ValueError("Incomplete/wrong-step paired model state")
    if (packet.get("step") != step or packet.get("suite_sha256") != manifest["sha256"] or
        packet.get("policy_sha256") != acq.content_digest(dev.suite.POLICY) or
        packet.get("baseline_scores_sha256") != acq.content_digest(baseline) or packet.get("scope") != SCOPE or
        any(set(packet.get(k, {})) != set(ARMS) for k in ("scores", "evaluations", "model_state_sha256"))):
        raise ValueError("Changed development policy/baseline/paired coverage")
    reviewed = {}
    for arm in ARMS:
        if dev.state_digest(state["arms"][arm]["model"]) != packet["model_state_sha256"][arm]:
            raise ValueError("Scores were not calculated from this saved model")
        summary = check_scores(packet["evaluations"][arm], manifest, baseline["rows"][0]["metrics"])
        assessment = dev.suite.assess(summary, baseline["summary"])
        if assessment != packet["scores"][arm]:
            raise ValueError("Stored assessment differs from unchanged original policy")
        reviewed[arm] = {"eligible_on_old_development": assessment["eligible"],
            "mean_vocal_error_snr_gain_db": assessment["rank_gain_db"],
            "mean_projected_remaining_vocal_change_db": assessment.get("mean_residual_vocal_change_db"),
            "reasons": assessment["reasons"], "domain_deltas": assessment.get("deltas", {})}
    return reviewed


def inspect_run(run, approval_path):
    approval = inp.verified_approval(approval_path)  # metadata/code bindings only; no label decoding
    bound = expected_binding(approval_path)
    manifest = acq.read_sealed(run / "selection_suite.json")
    historical = json.loads(dev.OLD_MANIFEST.read_text(encoding="utf-8"))
    if (plain(manifest) != historical or manifest["sha256"] != dev.EXPECTED_SUITE or
        (manifest["tracks"], manifest["clips"]) != (31, 177)):
        raise ValueError("Not the unchanged preselected 31-song/177-view development suite")
    baseline = acq.read_sealed(run / "frozen_scores.json")
    check_scores(baseline, manifest)
    files = {str(p.resolve()): acq.sha256(p) for p in
             (approval_path, run / "selection_suite.json", run / "frozen_scores.json", Path(__file__))}
    rows, states, pending = [], {}, []
    for step in (250, 500, 750, 1000):
        paths = [run / f"checkpoint_{step:04d}.json", run / f"development_step_{step:04d}.json",
                 run / f"NONRELEASE_pair_step_{step:04d}.pt"]
        if not all(p.exists() for p in paths):
            if any(p.exists() for p in paths):
                pending.append(step)  # score can precede the full-state commit; no quality claim
            continue
        try:
            receipt = acq.read_sealed(paths[0])
        except json.JSONDecodeError:
            pending.append(step)  # writer is still creating the final receipt
            continue
        if receipt.get("step") != step:
            raise ValueError("Receipt filename does not match its logical step")
        state = m.load_checked_checkpoint(paths[2], receipt["sha256"])
        packet = acq.read_sealed(paths[1])
        reviewed = check_pair(receipt, state, packet, bound, manifest, baseline)
        rows.append({"step": step, "arms": reviewed})
        states[step] = state
        files.update({str(p.resolve()): acq.sha256(p) for p in paths})
    report = {"schema": 1, "purpose": PURPOSE, "binding": bound, "approval": str(approval_path.resolve()),
        "committed_evaluations": rows,
        "pending_partial_steps": pending, "tracks": 31, "views": 177, "bindings_sha256": files,
        "cuda_used": False, "optimizer_constructed": False, "training_started": False,
        "release_selection": "NONE", "deployment": False, "independent_acceptance_scored": False,
        "scope": SCOPE,
        "interpretation": ["Vocal reconstruction gain: higher is better; projected vocal change: lower is better",
            "Also inspect every domain's accompaniment preservation and instrumental false removal",
            "Different teacher-target training losses cannot rank perceived student quality",
            "Development eligibility is NOT blind acceptance or permission to replace the board model"]}
    return report, states, approval


def fresh_output(out, audio=False):
    m.bulk.guard_output(out)
    if out.exists():
        raise ValueError("Fresh ignored result directory required; never overwrite evidence/audio")
    reserve = 12 * 1024**3 + (256 * 1024**2 if audio else 1024**2)
    if shutil.disk_usage(ROOT / "results").free < reserve:
        raise ValueError("Keep the running trainer's 12 GiB disk reserve")


@torch.no_grad()
def vocal_wave(net, mix):
    if (mix.device.type != "cpu" or mix.dtype != torch.float32 or mix.ndim != 2 or mix.shape[0] != 2 or
        mix.shape[-1] < 44100 or not torch.isfinite(mix).all() or
        any(v.device.type != "cpu" for v in (*net.parameters(), *net.buffers()))):
        raise ValueError("Finite CPU stereo/model required; no CUDA inference")
    original, modes, rng = copy.deepcopy(net.state_dict()), [v.training for v in net.modules()], m.capture_rng("cpu")
    try:
        net.eval()
        wa = torch.from_numpy(core.t09.make_analysis_matrix())
        gs = torch.from_numpy(core.t09.make_synthesis_matrix())
        spectrum = core.stft_batch(mix[None])
        bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
        count = bands.shape[-1]
        bands = F.pad(bands, (0, (-count) % 16))
        masks = a.context_masks(net, bands, 128, 256)
        vocal = core.product_vocal(spectrum, masks[None, ..., :count], gs, mix.shape[-1], 44)[0]
        if not torch.isfinite(vocal).all() or not m.equal_state(original, net.state_dict()):
            raise ValueError("Nonfinite prediction or changed read-only model")
        return vocal
    finally:
        net.load_state_dict(original, strict=True)
        for module, mode in zip(net.modules(), modes):
            module.training = mode
        m.restore_rng(rng, "cpu")


def listening_waves(models, mix, reference_vocal):
    if (set(models) != {"frozen", *ARMS} or reference_vocal.shape != mix.shape or
        reference_vocal.device.type != "cpu" or reference_vocal.dtype != torch.float32 or
        not torch.isfinite(reference_vocal).all()):
        raise ValueError("All three models and aligned real reference required")
    waves = {"mix": mix.clone(), "reference_vocal": reference_vocal.clone(),
             "reference_backing": mix - reference_vocal}
    for name, net in models.items():
        pv = vocal_wave(net, mix)
        waves[f"{name}_vocal"] = pv
        waves[f"{name}_backing"] = mix - pv
    # One additional gain for EVERY output prevents playback clipping without
    # normalizing student accompaniment separately (which would bias listening).
    peak = max(float(wave.abs().max()) for wave in waves.values())
    gain = min(1., .95 / max(peak, 1e-12))
    return {name: wave * gain for name, wave in waves.items()}, gain


def checked_sources():
    folder = a.DEFAULT_OUT
    if acq.sha256(folder / "plan.json") != SOURCE_PLAN_SHA or acq.sha256(folder / "diagnostic.json") != SOURCE_REPORT_SHA:
        raise ValueError("Pre-training listening source plan/report changed")
    plan, report = acq.read_sealed(folder / "plan.json"), acq.read_sealed(folder / "diagnostic.json")
    if (report["plan_sha256"] != SOURCE_PLAN_SHA or plan["formal_training"] is not False or
        plan["checkpoint_selected"] != "NONE" or plan["deployment"] is not False or
        [r["domain"] for r in plan["records"]] != ["musdb", "mir1k", "instrumental"] or
        len(report["records"]) != 3):
        raise ValueError("Listening inputs cannot become new acceptance or training audio")
    for path, digest in plan["bindings"].items():
        if acq.sha256(path) != digest:
            raise ValueError("Old source/model/role/code plan changed")
    inputs = {str((folder / name).resolve()): acq.sha256(folder / name) for name in ("plan.json", "diagnostic.json")}
    sources = []
    for index, (record, old) in enumerate(zip(plan["records"], report["records"]), 1):
        if record["role"] != "development" or old["role"] != "development" or any(record[k] != old[k] for k in ("domain", "track_id")):
            raise ValueError("Listening record not identical to preselected DEVELOPMENT record")
        waves = []
        for kind in ("mix", "reference_vocal"):
            name = f"source_{index:02d}_{kind}.wav"
            path = folder / name
            digest = acq.sha256(path)
            if digest != report["audio_files"][name]:
                raise ValueError("Preselected input/reference audio changed")
            info = sf.info(path)
            if (info.samplerate, info.channels, info.subtype, info.frames) != (44100, 2, "FLOAT", old["metrics"]["samples"]):
                raise ValueError("Listening audio geometry changed")
            # Hash the whole existing file, decode only a fixed prefix; no new
            # source/crop choice based on the student scores.
            array, sr = sf.read(path, frames=min(info.frames, LISTEN_SECONDS * 44100), dtype="float32", always_2d=True)
            wave = torch.from_numpy(array.T.copy())
            if sr != 44100 or not torch.isfinite(wave).all():
                raise ValueError("Nonfinite listening input/reference")
            waves.append(wave)
            inputs[str(path.resolve())] = digest
        if waves[0].shape != waves[1].shape:
            raise ValueError("Listening source/reference lengths differ")
        sources.append((record, old, *waves))
    return sources, inputs


def export_listening(report, state, approval, out):
    fresh_output(out, audio=True)
    sources, source_files = checked_sources()
    factory = m.frozen_factory(approval["source_protocol"])
    models = {name: factory() for name in ("frozen", *ARMS)}
    for arm in ARMS:
        models[arm].load_state_dict(state["arms"][arm]["model"], strict=True)
    plan = {"schema": 1, "step": state["step"], "purpose": PURPOSE,
        "selection": "Previously chosen old DEVELOPMENT sources, fixed first min(20s, length), not quality-selected",
        "source_plan_sha256": SOURCE_PLAN_SHA, "sources_sha256": source_files,
        "models_sha256": {name: dev.state_digest(net.state_dict()) for name, net in models.items()},
        "frontend": "FP32 / legacy_log / retained history128 / LF44; not integer board frontend",
        "sample_rate": 44100, "maximum_seconds_per_source": LISTEN_SECONDS,
        "release_selection": "NONE", "deployment": False, "independent_acceptance_scored": False}
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    rows, outputs = [], {}
    for index, (record, old, mix, ref) in enumerate(sources, 1):
        print(f"EXPLORATION_LISTEN source={index}/3 step={state['step']} {record['domain']}", flush=True)
        waves, gain = listening_waves(models, mix, ref)
        files = {}
        for name, wave in waves.items():
            path = out / f"source_{index:02d}_{name}.wav"
            a.save_wave(path, wave)
            files[name] = path.name
            outputs[path.name] = acq.sha256(path)
        rows.append({"domain": record["domain"], "track_id": record["track_id"], "role": "development",
            "samples": mix.shape[-1], "seconds": mix.shape[-1] / 44100, "start_sample": 0,
            "source_common_gain": old["common_gain"], "additional_common_playback_gain": gain,
            "input_pcm_sha256": m.pilot.wave_digest(mix), "audio": files,
            "source_join_samples_in_prefix": [j for j in old["metrics"]["source_join_samples"] if j < mix.shape[-1]]})
    evidence = copy.deepcopy(report)
    evidence.update({"listening_step": state["step"], "listening_records": rows,
        "plan_sha256": acq.sha256(out / "plan.json"), "audio_sha256": outputs,
        "notes": ["MUSDB is a 6.8s excerpt; MIR is joined karaoke clips with known seams",
            "Pure instrumental source checks false removal, not vocal suppression",
            "All nine variants share one playback gain; no independent normalization or clipping",
            "CPU floating-point reference only: not proof of real-board continuity or release quality"]})
    acq.write_new_json(out / "review.json", acq.seal(evidence))
    verify_output(out)


def verify_output(out):
    doc = acq.read_sealed(out / "review.json")
    if (doc["purpose"] != PURPOSE or doc["release_selection"] != "NONE" or doc["deployment"] is not False or
        doc["cuda_used"] is not False or doc["optimizer_constructed"] is not False or doc["training_started"] is not False or
        doc["independent_acceptance_scored"] is not False):
        raise ValueError("Review cannot become training/deployment approval")
    inp.verified_approval(Path(doc["approval"]))
    if doc["binding"] != expected_binding(Path(doc["approval"])):
        raise ValueError("Training/import/evaluation recipe changed since this review")
    for path, digest in doc["bindings_sha256"].items():
        if acq.sha256(path) != digest:
            raise ValueError("Read-only review input changed")
    if "listening_step" in doc:
        plan = acq.read_sealed(out / "plan.json")
        if acq.sha256(out / "plan.json") != doc["plan_sha256"]:
            raise ValueError("Listening plan changed")
        for path, digest in plan["sources_sha256"].items():
            if acq.sha256(path) != digest:
                raise ValueError("Listening source changed")
        for name, digest in doc["audio_sha256"].items():
            if Path(name).name != name or acq.sha256(out / name) != digest:
                raise ValueError("Listening output changed")
    print("EXPLORATION_REVIEW VERIFIED CPU-only; release_selection=NONE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("summary", "listen", "verify"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--approval", type=Path, default=inp.DEFAULT_APPROVAL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--step", type=int, choices=(250, 500, 750, 1000))
    args = parser.parse_args()
    torch.set_num_threads(2)  # leave CPU capacity for the live trainer; never query CUDA
    torch.use_deterministic_algorithms(True)
    if args.operation == "verify":
        if args.out is None:
            parser.error("verify requires --out")
        verify_output(args.out)
        return 0
    report, states, approval = inspect_run(args.run, args.approval)
    if not states or (args.step is not None and args.step not in states):
        print("EXPLORATION_REVIEW PENDING no requested complete checkpoint/score pair; no output or restart", flush=True)
        return 2
    step = args.step if args.step is not None else max(states)
    out = args.out or DEFAULT_OUT / f"{args.operation}_step_{step:04d}"
    if args.operation == "listen":
        export_listening(report, states[step], approval, out)
    else:
        fresh_output(out)
        out.mkdir(parents=True)
        acq.write_new_json(out / "review.json", acq.seal(report))
        verify_output(out)
    print(json.dumps({"out": str(out.resolve()), "latest_step": step,
                      "committed_evaluations": report["committed_evaluations"]}, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
