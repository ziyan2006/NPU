"""Independent CPU LF-boundary review; own zero-update baselines, frozen44 and source full-control."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
import importlib.util


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t = load("aux_review_trainer", "194_train_mel_lr_scale.py")
r = load("aux_review_helpers", "152_review_paired_exploration.py")
p, m, acq, ROOT, ARMS, PURPOSE = t.p, t.m, t.acq, t.ROOT, t.ARMS, t.PURPOSE
dev = r.dev
DEFAULT_OUT = ROOT / "results/mel_lr_scale_review_20261004"


def check_fork_baseline(state, packet, origin, original):
    chosen = origin["arms"][p.SOURCE_ARM]
    if any(not m.equal_state(state["arms"][arm], chosen) for arm in ARMS):
        raise ValueError("Both LR forks must copy source ARMS[1] LF32 including Adam/modes")
    if any(packet["evaluations"][a] != original["evaluations"][p.SOURCE_ARM] for a in ARMS):
        raise ValueError("Both LF32 baselines must reproduce source candidate4000")
    p.require_kills(packet.get("arm_kill_bands"))
    # Both arms share same source model/Adam/LF32 baseline.


def checked_pair(receipt, state, packet, bound, manifest, baseline, origin_sha):
    for value in (receipt, state, packet):
        p.require_weights(value.get("arm_instrumental_weights"))
        p.require_ca_weights(value.get("arm_accompaniment_weights"))
        p.require_kills(value.get("arm_kill_bands"))
        p.require_lr_scales(value.get("arm_lr_scales"))
    step = receipt.get("step")
    if (type(step) is not int or step not in (4000, 4250, 4500) or
        receipt.get("purpose") != PURPOSE or receipt.get("binding") != bound or
        receipt.get("checkpoint") != f"NONRELEASE_lr_scale_step_{step:04d}.pt" or
        receipt.get("additional_step") != step-4000 or receipt.get("smoke") is not False or
        receipt.get("deployment_authorized") is not False or
        receipt.get("arm_roles") != p.ROLES or receipt.get("arm_lambdas") != p.LAMBDAS or receipt.get("arm_instrumental_weights") != p.WEIGHTS or receipt.get("teacher") != "kim_melband"):
        raise ValueError("Wrong loss-only checkpoint receipt")
    if (state.get("purpose") != PURPOSE or state.get("binding") != bound or state.get("schema") != 1 or
        state.get("step") != step or state.get("limit") != 4500 or state.get("smoke") is not False or
        state.get("arm_roles") != p.ROLES or state.get("arm_lambdas") != p.LAMBDAS or state.get("arm_instrumental_weights") != p.WEIGHTS or state.get("arm_accompaniment_weights") != p.CA_WEIGHTS or state.get("teacher") != "kim_melband" or
        state.get("origin_sha256") != origin_sha or state.get("deployment_authorized") is not False or
        state.get("source_stopped_at") != 4000 or state.get("source_legacy_stop_events") != [3750, 4000] or
        state.get("tranche_stopping") != "new_common_fixed500_budget_preserve_legacy_stop_evidence" or
        type(state.get("legacy_stop_events")) is not list or
        state["legacy_stop_events"] != sorted(set(state["legacy_stop_events"])) or
        any(type(event) is not int or event not in (4250, 4500) or event > step for event in state["legacy_stop_events"]) or
        state["schedule"]["stopped_at"] != (4500 if step == 4500 else None) or
        state["sampler"]["cursor"] != step or state["sampler"].get("teacher") != "kim_melband" or
        state["sampler"]["approval_sha256"] != bound["approval_sha256"] or
        state["schedule"]["step"] != step or state["schedule"]["last_validation"] != step or
        set(state["arms"]) != set(ARMS) or any(s["updates"] != step for s in state["arms"].values()) or
        not t.d.finite_state(state)):
        raise ValueError("Partial state or changed loss roles/teacher/origin")
    for saved in state["arms"].values():
        groups = saved["optimizer"]["param_groups"]
        if (len(groups) != 1 or set(saved["optimizer"]["state"]) != set(groups[0]["params"]) or
            any(float(v["step"]) != step for v in saved["optimizer"]["state"].values())):
            raise ValueError("Incomplete Adam exposure")
    if (packet.get("purpose") != PURPOSE or packet.get("arm_roles") != p.ROLES or packet.get("arm_lambdas") != p.LAMBDAS or packet.get("arm_instrumental_weights") != p.WEIGHTS or packet.get("arm_accompaniment_weights") != p.CA_WEIGHTS or
        packet.get("teacher") != "kim_melband" or packet.get("additional_step") != step-4000 or
        packet.get("source_stopped_at") != 4000 or packet.get("source_legacy_stop_events") != [3750, 4000] or packet.get("tranche_stopping") != state["tranche_stopping"] or
        packet.get("legacy_stop_events") != state["legacy_stop_events"] or
        packet.get("step") != step or packet.get("suite_sha256") != manifest["sha256"] or
        packet.get("policy_sha256") != acq.content_digest(dev.suite.POLICY) or
        packet.get("baseline_scores_sha256") != acq.content_digest(baseline) or packet.get("scope") != r.SCOPE or
        packet.get("frozen_kill_bands") != 44 or
        any(set(packet.get(k, {})) != set(ARMS) for k in ("scores", "evaluations", "model_state_sha256", "low_frequency_backing"))):
        raise ValueError("Changed original scoring policy or paired coverage")
    result = {}
    for arm in ARMS:
        if dev.state_digest(state["arms"][arm]["model"]) != packet["model_state_sha256"][arm]:
            raise ValueError("Score/model mismatch")
        summary = r.check_scores(packet["evaluations"][arm], manifest, baseline["rows"][0]["metrics"])
        low = packet["low_frequency_backing"][arm]
        if low is None or set(low["rows"][0]["metrics"]) != {"lf_backing_error_power", "lf_backing_reference_power", "lf_backing_error_snr_db"}:
            raise ValueError("Missing descriptive low-frequency accompaniment evidence")
        low_summary = r.check_scores(low, manifest)
        assessment = dev.suite.assess(summary, baseline["summary"])
        if assessment != packet["scores"][arm]:
            raise ValueError("Score aggregation/policy mismatch")
        result[p.ROLES[arm]] = {"internal_key": arm, "teacher": "kim_melband", "kill_bands": p.KILLS[arm],
            "versus_frozen": assessment, "summary": summary, "low_frequency_backing_summary": low_summary}
    return result


def check_layout_and_adam(state, origin):
    chosen = origin["arms"][p.SOURCE_ARM]
    p.require_lr_scales(state.get("arm_lr_scales"))
    if not m.equal_state(state["schedule"]["config"], origin["schedule"]["config"]):
        raise ValueError("Original learning rate/schedule config changed")
    expected_group = chosen["optimizer"]["param_groups"][0]
    for arm,saved in state["arms"].items():
        group = saved["optimizer"]["param_groups"][0]
        if (saved["parameter_names"] != chosen["parameter_names"] or
            set(saved["model"]) != set(chosen["model"]) or len(saved["modes"]) != len(chosen["modes"]) or
            any(type(value) is not bool for value in saved["modes"]) or
            any(group.get(key) != value for key,value in expected_group.items() if key != "lr") or
            group["lr"] != m.learning_rate(state["step"], state["schedule"]["config"])*(1.0 if state["step"] == p.START else p.LR_SCALES[arm])):
            raise ValueError("Parameter layout/order, modes or Adam settings changed")
        for name, value in saved["model"].items():
            if value.shape != chosen["model"][name].shape or value.dtype != chosen["model"][name].dtype:
                raise ValueError("Model graph tensor layout changed")
        for index, values in saved["optimizer"]["state"].items():
            parameter = saved["model"][saved["parameter_names"][index]]
            if (set(values) != {"step", "exp_avg", "exp_avg_sq"} or float(values["step"]) != state["step"] or
                any(values[key].shape != parameter.shape or values[key].dtype != parameter.dtype for key in ("exp_avg", "exp_avg_sq")) or
                bool((values["exp_avg_sq"] < 0).any())):
                raise ValueError("Invalid Adam tensor exposure/moments")


def inspect_run(run, approval_path):
    approval = p.verified_approval(approval_path)
    bound = t.binding(approval_path)
    manifest = acq.read_sealed(run / "selection_suite.json")
    historical = json.loads(dev.OLD_MANIFEST.read_text(encoding="utf-8"))
    if r.plain(manifest) != historical or manifest["sha256"] != dev.EXPECTED_SUITE or (manifest["tracks"], manifest["clips"]) != (31, 177):
        raise ValueError("Not unchanged old31 songs/177 views")
    baseline = acq.read_sealed(run / "frozen_scores.json")
    r.check_scores(baseline, manifest)
    paths = [approval_path, run / "selection_suite.json", run / "frozen_scores.json", Path(__file__)]
    files = {str(path.resolve()): acq.sha256(path) for path in paths}
    stages, states, pending = [], {}, []
    origin = p.origin_state(approval)
    start_summaries = None
    for step in (4000, 4250, 4500):
        paths = [run / f"checkpoint_{step:04d}.json", run / f"NONRELEASE_lr_scale_step_{step:04d}.pt", run / f"development_step_{step:04d}.json"]
        if not all(path.exists() for path in paths):
            if any(path.exists() for path in paths):
                pending.append(step)
            continue
        try:
            receipt = acq.read_sealed(paths[0])
        except json.JSONDecodeError:
            pending.append(step)
            continue
        if receipt["step"] != step:
            raise ValueError("Receipt filename changed")
        state = m.load_checked_checkpoint(paths[1], receipt["sha256"])
        check_layout_and_adam(state, origin)
        packet = acq.read_sealed(paths[2])
        arms = checked_pair(receipt, state, packet, bound, manifest, baseline, approval["protocol"]["origin_checkpoint_sha256"])
        if step == 4000:
            original_path = p.SOURCE / "development_step_4000.json"
            check_fork_baseline(state, packet, origin, acq.read_sealed(original_path))
            files[str(original_path.resolve())] = acq.sha256(original_path)
            start_summaries = {role: reviewed["summary"] for role, reviewed in arms.items()}
        if start_summaries is None:
            raise ValueError("Missing complete fork-start evidence")
        for role, reviewed in arms.items():
            reviewed["versus_own_lf32_fork_start_4000"] = dev.suite.assess(reviewed["summary"], start_summaries[role])
            reviewed["versus_source_lf32_4000"] = dev.suite.assess(reviewed["summary"], start_summaries[p.ROLES[ARMS[0]]])
            reviewed["versus_parallel_lr1_control"] = dev.suite.assess(reviewed["summary"], arms[p.ROLES[ARMS[0]]]["summary"])
        stages.append({"step": step, "additional_step": step-4000, "arms": arms})
        states[step] = state
        files.update({str(path.resolve()): acq.sha256(path) for path in paths})
    report = {"schema": 1, "purpose": PURPOSE, "binding": bound, "teacher": "kim_melband", "arm_roles": p.ROLES, "arm_instrumental_weights": p.WEIGHTS, "arm_accompaniment_weights": p.CA_WEIGHTS, "arm_kill_bands": p.KILLS, "arm_lr_scales": p.LR_SCALES,
        "approval": str(approval_path.resolve()), "committed_evaluations": stages, "pending_partial_steps": pending,
        "tracks": 31, "views": 177, "bindings_sha256": files, "cuda_used": False, "optimizer_constructed": False,
        "training_started": False, "release_selection": "NONE", "deployment": False, "independent_acceptance_scored": False,
        "scope": r.SCOPE, "interpretation": ["Both arms use the SAME Mel targets/full auxiliary; internal keys are compatibility identifiers only",
            "Inspect projected vocal residue (lower), vocal reconstruction (higher), accompaniment preservation and instrumental false removal per domain",
            "Compare against each own common LF32 source4000 and parallel LR1 control; baseline is not training gain",
            "Baseline4000 is not a new trained stage. Old development is not blind acceptance"]}
    return report, states, approval, origin


@torch.no_grad()
def vocal_wave(net, mix, kill):
    if type(kill) is not int or kill not in (32, 44):
        raise ValueError("Listening requires declared LF32/44")
    if (mix.device.type != "cpu" or mix.dtype != torch.float32 or mix.ndim != 2 or mix.shape[0] != 2 or
        mix.shape[-1] < 44100 or not torch.isfinite(mix).all() or
        any(value.device.type != "cpu" for value in (*net.parameters(), *net.buffers()))):
        raise ValueError("Finite CPU stereo/model")
    original, modes, rng = t.d.portable(net.state_dict()), [mod.training for mod in net.modules()], m.capture_rng("cpu")
    try:
        net.eval()
        wa = torch.from_numpy(m.core.t09.make_analysis_matrix())
        gs = torch.from_numpy(m.core.t09.make_synthesis_matrix())
        spectrum = m.core.stft_batch(mix[None])
        bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
        count = bands.shape[-1]
        bands = r.F.pad(bands, (0, (-count) % 16))
        masks = r.a.context_masks(net, bands, 128, 256)
        vocal = m.core.product_vocal(spectrum, masks[None, ..., :count], gs, mix.shape[-1], kill)[0]
        if not torch.isfinite(vocal).all() or not m.equal_state(original, net.state_dict()):
            raise ValueError("Nonfinite/mutating read-only listening")
        return vocal
    finally:
        net.load_state_dict(original, strict=True)
        for module, mode in zip(net.modules(), modes):
            module.training = mode
        m.restore_rng(rng, "cpu")


def listening_waves(models, mix, reference):
    if set(models) != {"frozen", "origin_4000_lf32", *p.ROLES.values()} or reference.shape != mix.shape or not torch.isfinite(reference).all():
        raise ValueError("Four declared CPU/LF models and matched real reference required")
    waves = {"mix": mix.clone(), "reference_vocal": reference.clone(), "reference_backing": mix-reference}
    for role, net in models.items():
        vocal = vocal_wave(net, mix, 44 if role == "frozen" else 32)
        waves[f"{role}_vocal"], waves[f"{role}_backing"] = vocal, mix-vocal
    gain = min(1., .95/max(max(float(w.abs().max()) for w in waves.values()), 1e-12))
    return {name: wave*gain for name, wave in waves.items()}, gain


def export_listening(report, state, approval, origin, out):
    if state["step"] == 4000:
        raise ValueError("No duplicate source4000 listening; new trained stages only")
    r.fresh_output(out, audio=True)
    sources, source_files = r.checked_sources()
    factory = m.frozen_factory(approval["source_protocol"])
    models = {role: factory() for role in ("frozen", "origin_4000_lf32", *p.ROLES.values())}
    models["origin_4000_lf32"].load_state_dict(origin["arms"][p.SOURCE_ARM]["model"], strict=True)
    for arm, role in p.ROLES.items():
        models[role].load_state_dict(state["arms"][arm]["model"], strict=True)
    plan = {"schema": 1, "purpose": PURPOSE, "teacher": "kim_melband", "step": state["step"], "arm_roles": p.ROLES, "arm_instrumental_weights": p.WEIGHTS, "arm_accompaniment_weights": p.CA_WEIGHTS, "arm_kill_bands": p.KILLS, "arm_lr_scales": p.LR_SCALES,
        "sources_sha256": source_files, "selection": "Preselected three old development sources; first min20s, not score-selected",
        "models_sha256": {role: dev.state_digest(net.state_dict()) for role, net in models.items()},
        "frontend": "FP32 legacy_log/history128; each declared LF44/32; not integer board inference", "sample_rate": 44100,
        "release_selection": "NONE", "deployment": False}
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    rows, outputs = [], {}
    for index, (record, old, mix, ref) in enumerate(sources, 1):
        print(f"MEL_LR_SCALE_LISTEN source={index}/3 step={state['step']}", flush=True)
        waves, gain = listening_waves(models, mix, ref)
        files = {}
        for name, wave in waves.items():
            path = out / f"source_{index:02d}_{name}.wav"
            r.a.save_wave(path, wave)
            files[name], outputs[path.name] = path.name, acq.sha256(path)
        rows.append({"domain": record["domain"], "track_id": record["track_id"], "role": "development",
            "samples": mix.shape[-1], "source_common_gain": old["common_gain"], "additional_common_playback_gain": gain,
            "input_pcm_sha256": m.pilot.wave_digest(mix), "audio": files,
            "source_join_samples_in_prefix": [j for j in old["metrics"]["source_join_samples"] if j < mix.shape[-1]]})
    report = copy.deepcopy(report) | {"listening_step": state["step"], "listening_records": rows,
        "plan_sha256": acq.sha256(out / "plan.json"), "audio_sha256": outputs,
        "notes": ["All11 variants per source use one common gain, not separate normalization",
            "MUSDB6.8s; MIR20s known joined excerpts; instrumental checks false removal",
            "CPU reference only; no independent acceptance or real-board continuity claim"]}
    acq.write_new_json(out / "review.json", acq.seal(report))
    verify_output(out)


def verify_output(out):
    doc = acq.read_sealed(out / "review.json")
    if (doc["purpose"] != PURPOSE or doc["teacher"] != "kim_melband" or doc["arm_roles"] != p.ROLES or doc["arm_instrumental_weights"] != p.WEIGHTS or doc.get("arm_accompaniment_weights") != p.CA_WEIGHTS or
        doc["release_selection"] != "NONE" or any(doc[k] is not False for k in
            ("deployment", "cuda_used", "optimizer_constructed", "training_started", "independent_acceptance_scored"))):
        raise ValueError("Review is not deployment or training approval")
    p.verified_approval(Path(doc["approval"]))
    p.require_kills(doc.get("arm_kill_bands"))
    p.require_lr_scales(doc.get("arm_lr_scales"))
    if doc["binding"] != t.binding(Path(doc["approval"])):
        raise ValueError("Review recipe changed")
    for path, digest in doc["bindings_sha256"].items():
        if acq.sha256(path) != digest:
            raise ValueError("Review input changed")
    if "listening_step" in doc:
        plan = acq.read_sealed(out / "plan.json")
        if acq.sha256(out / "plan.json") != doc["plan_sha256"] or plan["teacher"] != "kim_melband":
            raise ValueError("Listening plan changed")
        for path, digest in plan["sources_sha256"].items():
            if acq.sha256(path) != digest:
                raise ValueError("Listening source changed")
        for name, digest in doc["audio_sha256"].items():
            if Path(name).name != name or acq.sha256(out / name) != digest:
                raise ValueError("Listening output changed")
    print("MEL_LR_SCALE_REVIEW VERIFIED CPU-only; release_selection=NONE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("summary", "listen", "verify"))
    parser.add_argument("--run", type=Path, default=t.DEFAULT_OUT)
    parser.add_argument("--approval", type=Path, default=p.DEFAULT_APPROVAL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--step", type=int, choices=(4000, 4250, 4500))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.operation == "verify":
        if args.out is None:
            parser.error("verify requires --out")
        verify_output(args.out)
        return 0
    report, states, approval, origin = inspect_run(args.run, args.approval)
    if not states or args.step is not None and args.step not in states:
        print("MEL_LR_SCALE_REVIEW PENDING incomplete requested stage; do not restart", flush=True)
        return 2
    step = args.step if args.step is not None else max(states)
    out = args.out or DEFAULT_OUT / f"{args.operation}_step_{step:04d}"
    if args.operation == "listen":
        export_listening(report, states[step], approval, origin, out)
    else:
        r.fresh_output(out)
        out.mkdir(parents=True)
        acq.write_new_json(out / "review.json", acq.seal(report))
        verify_output(out)
    print(json.dumps({"out": str(out), "latest_step": step, "stages": report["committed_evaluations"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
