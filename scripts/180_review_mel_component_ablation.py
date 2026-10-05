"""CPU-only independent score/checkpoint review of the same-Mel loss ablation.

Compatibility arm keys do NOT identify different teachers in this experiment.
Every quarter-stage is checked against both frozen and fork-start3000 models.
No optimizers, CUDA inference, training, deployment or score-selected songs.
"""
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


t = load("aux_review_trainer", "178_train_mel_component_ablation.py")
r = load("aux_review_helpers", "152_review_paired_exploration.py")
p, m, acq, ROOT, ARMS, PURPOSE = t.p, t.m, t.acq, t.ROOT, t.ARMS, t.PURPOSE
dev = r.dev
DEFAULT_OUT = ROOT / "results/mel_component_ablation_review_20261003"


def check_fork_baseline(state, packet, origin, original):
    chosen = origin["arms"][ARMS[1]]
    if any(not m.equal_state(state["arms"][arm], chosen) for arm in ARMS):
        raise ValueError("Fork initial model/Adam/modes differ from candidate3000")
    if packet["evaluations"][ARMS[0]] != packet["evaluations"][ARMS[1]]:
        raise ValueError("Fork baseline must be identical")
    if packet["evaluations"][ARMS[0]] != original["evaluations"][ARMS[1]]:
        raise ValueError("Fork baseline does not reproduce source candidate3000")


def checked_pair(receipt, state, packet, bound, manifest, baseline, origin_sha):
    for value in (receipt, state, packet):
        p.require_weights(value.get("arm_instrumental_weights"))
        p.require_ca_weights(value.get("arm_accompaniment_weights"))
    step = receipt.get("step")
    if (type(step) is not int or step not in (3000, 3250, 3500) or
        receipt.get("purpose") != PURPOSE or receipt.get("binding") != bound or
        receipt.get("checkpoint") != f"NONRELEASE_component_step_{step:04d}.pt" or
        receipt.get("additional_step") != step-3000 or receipt.get("smoke") is not False or
        receipt.get("deployment_authorized") is not False or
        receipt.get("arm_roles") != p.ROLES or receipt.get("arm_lambdas") != p.LAMBDAS or receipt.get("arm_instrumental_weights") != p.WEIGHTS or receipt.get("teacher") != "kim_melband"):
        raise ValueError("Wrong loss-only checkpoint receipt")
    if (state.get("purpose") != PURPOSE or state.get("binding") != bound or state.get("schema") != 1 or
        state.get("step") != step or state.get("limit") != 3500 or state.get("smoke") is not False or
        state.get("arm_roles") != p.ROLES or state.get("arm_lambdas") != p.LAMBDAS or state.get("arm_instrumental_weights") != p.WEIGHTS or state.get("arm_accompaniment_weights") != p.CA_WEIGHTS or state.get("teacher") != "kim_melband" or
        state.get("origin_sha256") != origin_sha or state.get("deployment_authorized") is not False or
        state.get("source_stopped_at") != 3000 or state.get("source_legacy_stop_events") != [2750, 3000] or
        state.get("tranche_stopping") != "new_common_fixed500_budget_preserve_legacy_stop_evidence" or
        type(state.get("legacy_stop_events")) is not list or
        state["legacy_stop_events"] != sorted(set(state["legacy_stop_events"])) or
        any(type(event) is not int or event not in (3250, 3500) or event > step for event in state["legacy_stop_events"]) or
        state["schedule"]["stopped_at"] != (3500 if step == 3500 else None) or
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
        packet.get("teacher") != "kim_melband" or packet.get("additional_step") != step-3000 or
        packet.get("source_stopped_at") != 3000 or packet.get("source_legacy_stop_events") != [2750, 3000] or packet.get("tranche_stopping") != state["tranche_stopping"] or
        packet.get("legacy_stop_events") != state["legacy_stop_events"] or
        packet.get("step") != step or packet.get("suite_sha256") != manifest["sha256"] or
        packet.get("policy_sha256") != acq.content_digest(dev.suite.POLICY) or
        packet.get("baseline_scores_sha256") != acq.content_digest(baseline) or packet.get("scope") != r.SCOPE or
        any(set(packet.get(k, {})) != set(ARMS) for k in ("scores", "evaluations", "model_state_sha256"))):
        raise ValueError("Changed original scoring policy or paired coverage")
    result = {}
    for arm in ARMS:
        if dev.state_digest(state["arms"][arm]["model"]) != packet["model_state_sha256"][arm]:
            raise ValueError("Score/model mismatch")
        summary = r.check_scores(packet["evaluations"][arm], manifest, baseline["rows"][0]["metrics"])
        assessment = dev.suite.assess(summary, baseline["summary"])
        if assessment != packet["scores"][arm]:
            raise ValueError("Score aggregation/policy mismatch")
        result[p.ROLES[arm]] = {"internal_key": arm, "teacher": "kim_melband",
            "versus_frozen": assessment, "summary": summary}
    return result


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
    start_summary = None
    for step in (3000, 3250, 3500):
        paths = [run / f"checkpoint_{step:04d}.json", run / f"NONRELEASE_component_step_{step:04d}.pt", run / f"development_step_{step:04d}.json"]
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
        packet = acq.read_sealed(paths[2])
        arms = checked_pair(receipt, state, packet, bound, manifest, baseline, approval["protocol"]["origin_checkpoint_sha256"])
        if step == 3000:
            original_path = p.SOURCE / "development_step_3000.json"
            check_fork_baseline(state, packet, origin, acq.read_sealed(original_path))
            files[str(original_path.resolve())] = acq.sha256(original_path)
            start_summary = arms["aux_full_control"]["summary"]
        if start_summary is None:
            raise ValueError("Missing complete fork-start evidence")
        for role, reviewed in arms.items():
            reviewed["versus_fork_start_3000"] = dev.suite.assess(reviewed["summary"], start_summary)
        stages.append({"step": step, "additional_step": step-3000, "arms": arms})
        states[step] = state
        files.update({str(path.resolve()): acq.sha256(path) for path in paths})
    report = {"schema": 1, "purpose": PURPOSE, "binding": bound, "teacher": "kim_melband", "arm_roles": p.ROLES, "arm_instrumental_weights": p.WEIGHTS, "arm_accompaniment_weights": p.CA_WEIGHTS,
        "approval": str(approval_path.resolve()), "committed_evaluations": stages, "pending_partial_steps": pending,
        "tracks": 31, "views": 177, "bindings_sha256": files, "cuda_used": False, "optimizer_constructed": False,
        "training_started": False, "release_selection": "NONE", "deployment": False, "independent_acceptance_scored": False,
        "scope": r.SCOPE, "interpretation": ["Both arms use the SAME Mel targets; internal keys are compatibility identifiers only",
            "Inspect projected vocal residue (lower), vocal reconstruction (higher), accompaniment preservation and instrumental false removal per domain",
            "Compare against fork-start3000 AND original frozen student; changed loss scale cannot rank sound quality",
            "Baseline3000 is not a new trained stage. Old development is not blind acceptance"]}
    return report, states, approval, origin


def listening_waves(models, mix, reference):
    if set(models) != {"frozen", "origin_instrumental4_3000", *p.ROLES.values()} or reference.shape != mix.shape or not torch.isfinite(reference).all():
        raise ValueError("Four CPU models and matched real reference required")
    waves = {"mix": mix.clone(), "reference_vocal": reference.clone(), "reference_backing": mix-reference}
    for role, net in models.items():
        vocal = r.vocal_wave(net, mix)
        waves[f"{role}_vocal"], waves[f"{role}_backing"] = vocal, mix-vocal
    gain = min(1., .95/max(max(float(w.abs().max()) for w in waves.values()), 1e-12))
    return {name: wave*gain for name, wave in waves.items()}, gain


def export_listening(report, state, approval, origin, out):
    r.fresh_output(out, audio=True)
    sources, source_files = r.checked_sources()
    factory = m.frozen_factory(approval["source_protocol"])
    models = {role: factory() for role in ("frozen", "origin_instrumental4_3000", *p.ROLES.values())}
    models["origin_instrumental4_3000"].load_state_dict(origin["arms"][ARMS[1]]["model"], strict=True)
    for arm, role in p.ROLES.items():
        models[role].load_state_dict(state["arms"][arm]["model"], strict=True)
    plan = {"schema": 1, "purpose": PURPOSE, "teacher": "kim_melband", "step": state["step"], "arm_roles": p.ROLES, "arm_instrumental_weights": p.WEIGHTS, "arm_accompaniment_weights": p.CA_WEIGHTS,
        "sources_sha256": source_files, "selection": "Preselected three old development sources; first min20s, not score-selected",
        "models_sha256": {role: dev.state_digest(net.state_dict()) for role, net in models.items()},
        "frontend": "FP32 legacy_log/history128/LF44; not integer board inference", "sample_rate": 44100,
        "release_selection": "NONE", "deployment": False}
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    rows, outputs = [], {}
    for index, (record, old, mix, ref) in enumerate(sources, 1):
        print(f"MEL_COMPONENT_ABLATION_LISTEN source={index}/3 step={state['step']}", flush=True)
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
    print("MEL_COMPONENT_ABLATION_REVIEW VERIFIED CPU-only; release_selection=NONE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("summary", "listen", "verify"))
    parser.add_argument("--run", type=Path, default=t.DEFAULT_OUT)
    parser.add_argument("--approval", type=Path, default=p.DEFAULT_APPROVAL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--step", type=int, choices=(3000, 3250, 3500))
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
        print("MEL_COMPONENT_ABLATION_REVIEW PENDING incomplete requested stage; do not restart", flush=True)
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
