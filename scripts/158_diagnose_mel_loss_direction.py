"""Bounded CPU TRAIN-only attribution after the completed Mel weight ablation.

No optimizer, network updates, CUDA, development/acceptance decode, audio export,
or production mask modification. Scalar gain probes are local directions, not
full parameter gradients. Answer-assisted masks are diagnostics, not attainable
student performance or an optimized waveform/theoretical quality ceiling.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import torch

spec = importlib.util.spec_from_file_location("mel_direction_review", Path(__file__).with_name("156_review_mel_weak_weight.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
t, m, acq, ROOT = r.t, r.m, r.acq, r.ROOT
PURPOSE = "NONRELEASE_CPU_TRAIN_LOSS_DIRECTION_DIAGNOSTIC"
DEFAULT_OUT = ROOT / "results/mel_loss_direction_20261003"
BATCHES, MASK_ITERATIONS = 12, 96
GAINS = (.8, 1., 1.2)


def decompose(predicted, mix, vocal):
    if predicted.shape != mix.shape or vocal.shape != mix.shape:
        raise ValueError("Aligned source projection required")
    p, x, v = [a.detach().cpu().double().flatten() for a in (predicted, mix, vocal)]
    if not all(torch.isfinite(a).all() for a in (p, x, v)):
        raise ValueError("Finite source projection required")
    a, y = x-v, x-p
    aa, vv, av = a@a, v@v, a@v
    denominator = aa*vv-av*av
    if aa <= 1e-10 or vv <= 1e-10 or denominator <= 1e-8*aa*vv:
        return None
    ca = ((y@a)*vv-(y@v)*av)/denominator
    cv = ((y@v)*aa-(y@a)*av)/denominator
    residual = y-ca*a-cv*v
    energy = (p-v).square().sum()
    terms = {"accompaniment_damage": (ca-1).square()*aa,
             "remaining_vocal": cv.square()*vv,
             "correlated_cross_term": 2*(ca-1)*cv*av,
             "unmodeled_fit_residual": residual.square().sum()}
    closure = abs(float(sum(terms.values())-energy))/max(float(energy), 1e-12)
    if closure > 1e-8:
        raise ValueError("Source energy identity does not close")
    return {"accompaniment_gain": float(ca), "remaining_vocal_gain": float(cv),
            "relative_closure_error": closure,
            "error_energy_fractions": {key: float(value/energy.clamp_min(1e-12)) for key, value in terms.items()}}


def bounded_spectral_mask(spectrum, truth, synthesis, kill, iterations=MASK_ITERATIONS):
    """Projected gradient for bounded band masks, with a per-frame safe Hessian bound.

    Solve complex spectral squared error only, not waveform L1. The nonnegative
    synthesis permits max Hessian row sum as a Lipschitz upper bound. Fixed
    iterations and projected-gradient residual are reported; no optimum claim.
    """
    if (spectrum.device.type != "cpu" or truth.device.type != "cpu" or synthesis.device.type != "cpu" or
        spectrum.shape != truth.shape or spectrum.ndim != 4 or not spectrum.is_complex() or
        not truth.is_complex() or synthesis.ndim != 2 or synthesis.shape[0] != spectrum.shape[2] or
        type(kill) is not int or not 0 <= kill < synthesis.shape[1] or
        type(iterations) is not int or not 1 <= iterations <= MASK_ITERATIONS or
        not all(torch.isfinite(value).all() for value in (spectrum, truth, synthesis)) or
        (synthesis < 0).any()):
        raise ValueError("Finite CPU complex spectra/nonnegative synthesis and bounded recipe required")
    with torch.no_grad():
        x, v, gs = spectrum.to(torch.complex128), truth.to(torch.complex128), synthesis.double().clone()
        gs[:, :kill] = 0
        power, target = x.abs().square(), (v*x.conj()).real
        row_sum = gs.sum(1)
        lipschitz = torch.einsum("fk,bcft->bckt", gs, power*row_sum[None, None, :, None]).amax(2, keepdim=True).clamp_min(1e-20)
        mask = torch.zeros((*x.shape[:2], gs.shape[1], x.shape[-1]), dtype=torch.float64)
        def error(value):
            return float((x*torch.einsum("fk,bckt->bcft", gs, value)-v).abs().square().mean())
        errors = [error(mask)]
        for _ in range(iterations):
            full = torch.einsum("fk,bckt->bcft", gs, mask)
            gradient = torch.einsum("fk,bcft->bckt", gs, power*full-target)
            mask = (mask-gradient/lipschitz).clamp(0, 1)
            mask[:, :, :kill] = 0
            errors.append(error(mask))
        if any(b > a+max(1., abs(a))*1e-10 for a, b in zip(errors, errors[1:])):
            raise ValueError("Projected spectral objective increased")
        gradient = torch.einsum("fk,bcft->bckt", gs, power*torch.einsum("fk,bckt->bcft", gs, mask)-target)
        projected = (mask-gradient/lipschitz).clamp(0, 1)
        projected[:, :, :kill] = 0
        return mask.float(), {"iterations": iterations, "spectral_mse_trace": errors,
            "projected_step_max_abs": float((mask-projected).abs().max()),
            "optimized_variable": "Answer-assisted per-frame128 real band mask only; no network",
            "objective": "Complex spectral squared error, not original wave+complex L1",
            "optimum_or_theoretical_ceiling_claimed": False}


def slot_probe(net, x, v, wa, gs):
    if x.device.type != "cpu" or v.device.type != "cpu" or x.shape != v.shape or x.shape[:2] != (1, 2):
        raise ValueError("CPU aligned single stereo training crop required")
    region = m.fit.suite.scoring_slice(x.shape[-1], 96)
    spectrum = m.core.stft_batch(x)
    with torch.no_grad():
        mask = (net(torch.einsum("fk,bcft->bckt", wa, spectrum.abs()))[:, :2]+1)/2
    gain = torch.ones((), requires_grad=True)
    loss, parts, prediction = m.fit.reconstruction_loss((mask*gain).clamp(0, 1), spectrum, x, v, gs, 96, 44)
    wave_direction = float(torch.autograd.grad(parts["wave_l1"], gain, retain_graph=True)[0])
    complex_direction = float(torch.autograd.grad(parts["complex_l1"], gain)[0])
    base = {"loss": float(loss.detach()), "wave_l1": float(parts["wave_l1"].detach()),
        "complex_l1": float(parts["complex_l1"].detach()), "d_wave_d_gain": wave_direction,
        "d_complex_d_gain": complex_direction, "d_total_d_gain": wave_direction+complex_direction,
        "decomposition": decompose(prediction[..., region], x[..., region], v[..., region])}
    with torch.no_grad():
        base["gain_sweep"] = []
        for scalar in GAINS:
            value, _, predicted = m.fit.reconstruction_loss((mask*scalar).clamp(0, 1), spectrum, x, v, gs, 96, 44)
            base["gain_sweep"].append({"gain": scalar, "loss": float(value),
                "metrics": r.dev.suite.separation_metrics(predicted[0, ..., region], x[0, ..., region], v[0, ..., region])})
    return base


def summarize(rows):
    result = {}
    for role in sorted({row["model_role"] for row in rows}):
        result[role] = {}
        for bucket in sorted({row["bucket"] for row in rows}):
            group = [row for row in rows if row["model_role"] == role and row["bucket"] == bucket]
            if not group:
                continue
            fits = [row["probe"]["decomposition"] for row in group if row["probe"]["decomposition"] is not None]
            result[role][bucket] = {"crops": len(group), "identifiable_fits": len(fits),
                "loss_locally_prefers_less_mask": sum(row["probe"]["d_total_d_gain"] > 0 for row in group),
                "wave_locally_prefers_less_mask": sum(row["probe"]["d_wave_d_gain"] > 0 for row in group),
                "complex_locally_prefers_less_mask": sum(row["probe"]["d_complex_d_gain"] > 0 for row in group),
                "components_disagree": sum(row["probe"]["d_wave_d_gain"]*row["probe"]["d_complex_d_gain"] < 0 for row in group),
                "mean_vocal_to_mix_power_ratio": statistics.fmean(row["vocal_to_mix_power_ratio"] for row in group),
                "mean_projected_remaining_vocal_gain": statistics.fmean(fit["remaining_vocal_gain"] for fit in fits) if fits else None,
                "mean_projected_accompaniment_gain": statistics.fmean(fit["accompaniment_gain"] for fit in fits) if fits else None,
                "mean_error_energy_fractions": {key: statistics.fmean(fit["error_energy_fractions"][key] for fit in fits)
                    for key in fits[0]["error_energy_fractions"]} if fits else None}
    return result


def verify(out):
    doc = acq.read_sealed(out / "diagnostic.json")
    plan = acq.read_sealed(out / "plan.json")
    if (doc["purpose"] != PURPOSE or doc["plan_sha256"] != acq.sha256(out / "plan.json") or
        doc["model_updates"] != 0 or doc["optimizer_constructed"] is not False or doc["cuda_used"] is not False or
        doc["release_selection"] != "NONE" or doc["deployment"] is not False or
        len(doc["rows"]) != BATCHES*6*3 or doc["summary"] != summarize(doc["rows"])):
        raise ValueError("Diagnostic coverage/authority changed")
    for name, digest in plan["bindings_sha256"].items():
        if acq.sha256(name) != digest:
            raise ValueError(f"Diagnostic binding changed: {name}")
    if any(row["metadata"]["role"] not in ("train", "pseudo_label_train_candidate") for row in doc["rows"]):
        raise ValueError("Not exclusively TRAIN/pseudo TRAIN crops")
    print("MEL_DIRECTION VERIFIED CPU-only TRAIN; model_updates=0; release_selection=NONE", flush=True)


def run(out):
    t.old.require_fresh(out)
    t.verify(t.DEFAULT_OUT, t.p.DEFAULT_APPROVAL)
    report, states, approval, origin = r.inspect_run(t.DEFAULT_OUT, t.p.DEFAULT_APPROVAL)
    if set(states) != {1000, 1250, 1500} or report["pending_partial_steps"]:
        raise ValueError("Completed fully committed ablation required")
    launch_root = ROOT / "results/mel_weak_weight_launch_20261003"
    exit_path = launch_root / "detached_exit_20261002_222032_3823062.json"
    exit_doc = json.loads(exit_path.read_text(encoding="utf-8-sig"))
    launch_path = Path(exit_doc["launch_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    if (exit_doc["exit_code"] != 0 or launch["probe_only"] or Path(launch["out"]) != t.DEFAULT_OUT or
        exit_doc["launcher_process_id"] != launch["worker"]["LauncherProcessId"]):
        raise ValueError("Matched real exit required")
    paths = [Path(__file__), Path(__file__).with_name("_test_mel_loss_direction.py"), Path(r.__file__), Path(t.__file__),
        Path(t.p.__file__), t.p.DEFAULT_APPROVAL, t.DEFAULT_OUT / "completion.json", exit_path, launch_path,
        Path(m.fit.__file__), Path(r.dev.suite.__file__), ROOT / "models/student_bott2_mir1k_candidate.pt"]
    paths += [t.DEFAULT_OUT / name for name in ("NONRELEASE_weak_step_1000.pt", "NONRELEASE_weak_step_1500.pt")]
    plan = {"schema": 1, "purpose": PURPOSE, "bindings_sha256": {str(path.resolve()): acq.sha256(path) for path in paths},
        "draw_cursors": list(range(1000, 1000+BATCHES)), "shared_slots_per_batch": 6,
        "model_roles": ["origin_mel1000", "uniform1500", "weighted1500"], "gain_sweep": list(GAINS),
        "band_mask_probe_selection": "First metadata vocal_db=-12 true TRAIN slot in each of musdb/mir1k, among these fixed12 draws; not score-selected",
        "mask_iterations": MASK_ITERATIONS, "input_only_roles": ["train", "pseudo_label_train_candidate"],
        "model_updates": 0, "optimizer_constructed": False, "cuda_used": False, "deployment": False,
        "release_selection": "NONE", "formal_training_started": False,
        "scope": "Bounded reused TRAIN draws, not whole-library causal proof, development selection or acceptance"}
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    rows, probes = [], []
    rng = m.capture_rng("cpu")
    try:
        factory = m.frozen_factory(approval["source_protocol"])
        models = {role: factory().eval() for role in plan["model_roles"]}
        models["origin_mel1000"].load_state_dict(states[1000]["arms"][t.ARMS[0]]["model"], strict=True)
        for arm, role in zip(t.ARMS, ("uniform1500", "weighted1500")):
            models[role].load_state_dict(states[1500]["arms"][arm]["model"], strict=True)
        digests = {role: r.dev.state_digest(net.state_dict()) for role, net in models.items()}
        wa, gs = (torch.from_numpy(fn()) for fn in (m.core.t09.make_analysis_matrix, m.core.t09.make_synthesis_matrix))
        stream = t.p.MelForkStream(t.p.DEFAULT_APPROVAL)
        seen = set()
        for draw in range(BATCHES):
            batch = stream.next_batch()
            if batch["cursor"] != 1000+draw or not torch.equal(batch["targets"][t.ARMS[0]], batch["targets"][t.ARMS[1]]):
                raise ValueError("Fixed same-Mel draws changed")
            for slot, meta in enumerate(batch["metadata"]):
                if meta["role"] not in plan["input_only_roles"]:
                    raise ValueError("Do not diagnose development/acceptance sources")
                x, v = batch["x"][slot:slot+1], batch["targets"][t.ARMS[0]][slot:slot+1]
                region = m.fit.suite.scoring_slice(x.shape[-1], 96)
                power = float(x[..., region].square().mean())
                vocal_power = float(v[..., region].square().mean())
                ratio = vocal_power/max(power, 1e-12)
                bucket = "inactive_or_instrumental" if vocal_power <= 1e-8 else ("weak_ratio_le_0.1" if ratio <= .1 else "normal_ratio_gt_0.1")
                spectrum, truth = m.core.stft_batch(x), m.core.stft_batch(v)
                blocked = gs[:, 44:].sum(1) <= 1e-12
                scored_power = truth[..., 98:-2].abs().double().square()
                blocked_percent = float(scored_power[:, :, blocked].sum()/scored_power.sum()*100) if scored_power.sum() > 1e-10 else None
                for role, net in models.items():
                    rows.append({"model_role": role, "cursor": batch["cursor"], "slot": slot,
                        "metadata": copy.deepcopy(meta), "input_sha256": m.pilot.wave_digest(x), "target_sha256": m.pilot.wave_digest(v),
                        "bucket": bucket, "vocal_to_mix_power_ratio": ratio, "vocal_rms": vocal_power**.5,
                        "vocal_energy_in_fully_blocked_lf44_bins_percent": blocked_percent,
                        "probe": slot_probe(net, x, v, wa, gs)})
                domain = meta["domain"]
                if slot < 2 and meta["vocal_db"] == -12 and domain not in seen:
                    seen.add(domain)
                    variants = {}
                    for kill in (44, 0):
                        mask, details = bounded_spectral_mask(spectrum, truth, gs, kill)
                        predicted = m.core.product_vocal(spectrum, mask, gs, x.shape[-1], kill)
                        variants[f"bounded128_lf{kill}"] = details | {"metrics": r.dev.suite.separation_metrics(predicted[0, ..., region], x[0, ..., region], v[0, ..., region])}
                    full = ((truth*spectrum.conj()).real/spectrum.abs().square().clamp_min(1e-12)).clamp(0, 1)
                    predicted = m.core.t09._istft((spectrum*full).flatten(0, 1), x.shape[-1]).reshape_as(x)
                    variants["answer_full513_no_lf"] = {"metrics": r.dev.suite.separation_metrics(predicted[0, ..., region], x[0, ..., region], v[0, ..., region]),
                        "note": "Per-bin analytic spectral MSE mask using true answers; not waveform L1 optimum or student performance"}
                    probes.append({"cursor": batch["cursor"], "slot": slot, "metadata": copy.deepcopy(meta),
                        "input_sha256": m.pilot.wave_digest(x), "target_sha256": m.pilot.wave_digest(v), "variants": variants})
            print(f"MEL_DIRECTION TRAIN draws={draw+1}/{BATCHES}; models unchanged", flush=True)
        if seen != {"musdb", "mir1k"}:
            raise ValueError("Predeclared first12 draws did not contain both weak metadata domains")
        if any(r.dev.state_digest(net.state_dict()) != digests[role] or any(param.grad is not None for param in net.parameters()) for role, net in models.items()):
            raise ValueError("Read-only diagnostic modified network/gradients")
        doc = {"schema": 1, "purpose": PURPOSE, "plan_sha256": acq.sha256(out / "plan.json"),
            "rows": rows, "summary": summarize(rows), "answer_assisted_mask_probes": probes,
            "models_sha256": digests, "model_updates": 0, "optimizer_constructed": False, "cuda_used": False,
            "release_selection": "NONE", "deployment": False, "model_parameters_unchanged": True,
            "interpretation": ["Positive dL/dg prefers reducing all estimated-vocal masks in one local direction only",
                "All diagnostic directions are the original unweighted per-slot loss, allowing comparisons across models",
                "Positive slot weights cannot reverse an individual loss direction; not a full parameter-gradient claim",
                "Gain sweeps are diagnostics, not a deployable global gain recommendation",
                "TRUTH-assisted finite-iteration masks cannot be called achieved or theoretical quality ceilings",
                "Instrumental/collinear joint source fits are not identifiable; null is not zero residue",
                "Fixed12 TRAIN draws are not independent dev/test evidence or whole-library causal proof"]}
        acq.write_new_json(out / "diagnostic.json", acq.seal(doc))
        verify(out)
    finally:
        m.restore_rng(rng, "cpu")


if __name__ == "__main__":
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
