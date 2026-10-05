"""Sealed CPU TRAIN-only frontend attribution; no gradients, updates or CUDA.

Four fixed context paths on the SAME spectra and common interval. LF44
statistics are spectral diagnostics, not waveform quality floors or an oracle.
This tool cannot start a training tranche or export/deploy audio/checkpoints.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
import types
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("train_frontend_review", Path(__file__).with_name("180_review_mel_component_ablation.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
t, p, m, d, acq, ROOT = r.t, r.p, r.m, r.t.d, r.acq, r.ROOT
s, core, suite = p.q.s, m.core, m.fit.suite
PURPOSE = "NONRELEASE_CPU_TRAIN_CONTEXT_RECONSTRUCTION_DIAGNOSTIC"
CURSORS = tuple(range(3000, 3012))
MODELS = ("origin_instrumental4_3000", "aux_full_control_3500", "aux_residual_only_3500")
PATHS = ("whole352", "history128_block256", "history128_block16", "reset16_negative_control")
THREADS, SAMPLES, FRAMES, HOP, KILL = 2, 89856, 352, 256, 44
SOURCE = t.DEFAULT_OUT
DEFAULT_OUT = ROOT / "results/train_context_reconstruction_20261003"
PROOF = ROOT / "results/mel_component_ablation_monitor_20261003/completion_review.json"
PROOF_SHA = "561ab6cdf7177d75e49cc07025b7392c3d65cb3d22564efaeb6db20d9340c3d8"
REPORT = ROOT / "reports/65_train_context_reconstruction_plan.md"
TEST = Path(__file__).with_name("_test_train_context_reconstruction.py")


def fixed_scope():
    return {"schema": 1, "purpose": PURPOSE, "diagnostic_only": True,
        "training_authorized": False, "cuda_used": False, "model_updates": 0,
        "backward_used": False, "optimizer_constructed": False,
        "cursors": list(CURSORS), "models": list(MODELS), "paths": list(PATHS),
        "unique_input_batches": 12, "model_batches": 36, "slot_checks": 216,
        "samples": SAMPLES, "frames": FRAMES, "threads": THREADS,
        "native_sample_interval": [98*HOP, SAMPLES-2*HOP],
        "common_sample_interval": [130*HOP, SAMPLES-2*HOP],
        "native_complex_interval": [98, FRAMES-2],
        "common_mask_interval": [130, FRAMES-2], "lf_kill_bands": KILL,
        "forward_reconstruction_dtype": "float32", "detached_statistics_dtype": "float64",
        "teacher": "kim_melband", "release_selection": "NONE", "deployment": False}


def check_plan(doc):
    # JSON equality alone accepts bool==int. Canonical representation does not.
    if any(json.dumps(doc.get(k), sort_keys=True) != json.dumps(v, sort_keys=True)
           for k, v in fixed_scope().items()):
        raise ValueError("Changed fixed zero-update frontend diagnostic scope")


def check_bindings(bindings):
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Missing immutable dependency bindings")
    for name, digest in bindings.items():
        if acq.sha256(name) != digest:
            raise ValueError(f"Changed diagnostic dependency: {name}")


def module_files():
    """Bind the actual import graph, not just the top-level prototype."""
    seen, files, queue = set(), set(), [r]
    while queue:
        mod = queue.pop()
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        filename = getattr(mod, "__file__", None)
        if not filename:
            continue
        filename = Path(filename).resolve()
        if not filename.is_relative_to(ROOT / "scripts"):
            continue
        files.add(filename)
        queue.extend(v for v in vars(mod).values() if isinstance(v, types.ModuleType))
    return files


def no_active_worker():
    names = "134_generate_teacher_library|139_generate_paired_htdemucs|150_train_paired_exploration|154_train_mel_weak_weight|161_train_mel_source_aux|166_train_mel_source_strength|172_train_mel_instrumental_protection|178_train_mel_component_ablation|158_diagnose_mel_loss_direction|164_diagnose_source_aux_strength|169_diagnose_gradient_contributions|175_diagnose_auxiliary_components|181_diagnose_train_context_reconstruction"
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '(" + names + ")[.]py[\" ]+(train|run|smoke)' } | Select-Object -ExpandProperty ProcessId"
    reply = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], check=True, capture_output=True, text=True)
    own = {os.getpid(), os.getppid()}
    if any(int(line.strip()) not in own for line in reply.stdout.splitlines() if line.strip()):
        raise ValueError("Active training/diagnostic: preserve and postpone")


def check_disk():
    if shutil.disk_usage(ROOT / "results").free < 12*(1 << 30):
        raise ValueError("Preserve at least12GiB free disk")


def same_exit_record(actual, reviewed):
    """PowerShell removes fractional trailing zeros when reserializing UTC.

    Normalize ONLY that lossless spelling, not times, IDs, types or paths.
    The raw exit file is separately SHA-bound to the reviewed completion.
    """
    if not isinstance(actual, dict) or not isinstance(reviewed, dict) or actual.keys() != reviewed.keys():
        return False
    def normalize(doc):
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,7}))?Z", doc.get("utc", ""))
        if not match:
            raise ValueError("Exact UTC exit time required")
        fraction = (match[2] or "").rstrip("0")
        return doc | {"utc": match[1]+("."+fraction if fraction else "")+"Z"}
    return json.dumps(normalize(actual), sort_keys=True) == json.dumps(normalize(reviewed), sort_keys=True)


def checked_sources():
    if acq.sha256(PROOF) != PROOF_SHA:
        raise ValueError("Previously independently reviewed completion changed")
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    approval = p.verified_approval(p.DEFAULT_APPROVAL)
    completion_path = SOURCE / "completion.json"
    completion = acq.read_sealed(completion_path)
    launch_path, exit_path = Path(proof["launch_receipt"]), Path(proof["detached_exit_path"])
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    exit_doc = json.loads(exit_path.read_text(encoding="utf-8-sig"))
    terminal = proof["terminal_status"]
    if (proof["purpose"] != "NONRELEASE_MEL_COMPONENT_ABLATION_COMPLETION_REVIEW" or
        terminal["status"] != "complete" or terminal["step"] != 3500 or terminal["additional_step"] != 500 or terminal["error"] is not None or
        proof["matched_launch_exit"] is not True or proof["no_active_worker_or_shim"] is not True or proof["processes"] or
        not same_exit_record(exit_doc, proof["detached_exit"]) or exit_doc["exit_code"] != 0 or
        Path(exit_doc["launch_receipt"]) != launch_path or launch["probe_only"] or Path(launch["out"]) != SOURCE or
        exit_doc["launcher_process_id"] != launch["worker"]["LauncherProcessId"] or
        completion["step"] != 3500 or completion["additional_steps"] != 500 or completion["limit"] != 3500 or
        completion["binding"] != t.binding(p.DEFAULT_APPROVAL) or
        proof["training_verify"]["actual_exit_code"] != 0 or proof["summary_verify"]["actual_exit_code"] != 0 or
        proof["summary_verify"]["tracks"] != 31 or proof["summary_verify"]["views"] != 177 or proof["summary_verify"]["pending"] or
        proof["release_selection"] != "NONE" or proof["deployment"] is not False):
        raise ValueError("Exact previously reviewed normal500 completion required")
    recorded = {completion_path: proof["completion_manifest_sha256"], launch_path: proof["launch_sha256"],
        exit_path: proof["detached_exit_sha256"], ROOT / proof["training_verify"]["log"]: proof["training_verify"]["sha256"],
        ROOT / proof["summary_verify"]["review"]: proof["summary_verify"]["sha256"]}
    check_bindings({str(path): digest for path, digest in recorded.items()})
    files = set(recorded) | module_files() | {PROOF, REPORT, TEST, Path(__file__).resolve(), p.DEFAULT_APPROVAL,
        ROOT / "reports/36_weak_weight_authorization.md", ROOT / "reports/64_mel_component_ablation_completion.md",
        Path(approval["origin_approval"]), m.bulk.OLD_LOCK}
    models, sampler = {}, None
    for step in (3000, 3500):
        receipt_path = SOURCE / f"checkpoint_{step}.json"
        receipt = acq.read_sealed(receipt_path)
        checkpoint = SOURCE / receipt["checkpoint"]
        if (acq.sha256(receipt_path) != completion["outputs_sha256"].get(receipt_path.name) or
            acq.sha256(checkpoint) != completion["outputs_sha256"].get(checkpoint.name)):
            raise ValueError("Source receipt/checkpoint differs from completed manifest")
        state = m.load_checked_checkpoint(checkpoint, receipt["sha256"])
        if (receipt["step"] != step or receipt["checkpoint"] != f"NONRELEASE_component_step_{step}.pt" or
            receipt["binding"] != completion["binding"] or receipt["smoke"] is not False or
            state["purpose"] != t.PURPOSE or state["binding"] != completion["binding"] or state["step"] != step or
            state["limit"] != 3500 or state["sampler"]["cursor"] != step or state["schedule"]["step"] != step or
            state["schedule"]["last_validation"] != step or state["schedule"]["stopped_at"] != (3500 if step == 3500 else None) or
            state["source_stopped_at"] != 3000 or state["source_legacy_stop_events"] != [2750, 3000] or
            state["legacy_stop_events"] != ([] if step == 3000 else [3250, 3500]) or
            state["teacher"] != "kim_melband" or state["arm_roles"] != p.ROLES or state["arm_lambdas"] != p.LAMBDAS or state["arm_instrumental_weights"] != p.WEIGHTS or
            state["arm_accompaniment_weights"] != p.CA_WEIGHTS or state["smoke"] is not False or state["deployment_authorized"] is not False or
            set(state["arms"]) != set(t.ARMS) or any(a["updates"] != step for a in state["arms"].values()) or not d.finite_state(state)):
            raise ValueError("Changed source full-state/teacher/roles/stopping")
        for saved in state["arms"].values():
            groups = saved["optimizer"]["param_groups"]
            if (len(groups) != 1 or set(saved["optimizer"]["state"]) != set(groups[0]["params"]) or
                any(float(v["step"]) != step for v in saved["optimizer"]["state"].values())):
                raise ValueError("Missing original Adam exposure")
        if step == 3000:
            if not m.equal_state(state["arms"][t.ARMS[0]], state["arms"][t.ARMS[1]]):
                raise ValueError("Origin model/Adam/modes not identical")
            models[MODELS[0]], sampler = state["arms"][t.ARMS[1]], state["sampler"]
        else:
            if completion["final_checkpoint"] != receipt or receipt["sha256"] != proof["final_checkpoint_sha256"]:
                raise ValueError("Final source does not match reviewed checkpoint")
            models[MODELS[1]], models[MODELS[2]] = (state["arms"][arm] for arm in t.ARMS)
        files |= {receipt_path, checkpoint}
    bindings = approval["bindings_sha256"] | {str(path.resolve()): acq.sha256(path) for path in files}
    check_bindings(bindings)
    return approval, models, sampler, bindings


def cpu_float(value, shape=None):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32 or
        (shape is not None and tuple(value.shape) != shape) or not torch.isfinite(value).all()):
        raise ValueError("Finite aligned CPU FP32 tensor required")


def check_net(net):
    if torch.cuda.is_initialized():
        raise ValueError("CUDA initialization prohibited")
    for value in (*net.parameters(), *net.buffers()):
        if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32) or not torch.isfinite(value).all():
            raise ValueError("CPU FP32 model required")
    if any(isinstance(mod, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)) for mod in net.modules()):
        raise ValueError("Deterministic batch-independent frozen graph required")


@contextmanager
def readonly_model(net):
    check_net(net)
    state, modes, rng = copy.deepcopy(net.state_dict()), [mod.training for mod in net.modules()], m.capture_rng("cpu")
    params = list(net.parameters())
    grads = [(param.grad, None if param.grad is None else param.grad.detach().clone()) for param in params]
    try:
        yield
        changed = (not m.equal_state(state, net.state_dict()) or modes != [mod.training for mod in net.modules()] or
            not m.equal_state(rng, m.capture_rng("cpu")) or torch.cuda.is_initialized() or
            any(param.grad is not old or not m.equal_state(snapshot, param.grad) for param, (old, snapshot) in zip(params, grads)))
        if changed:
            raise ValueError("Read-only model/state/modes/RNG/grad mutated")
    finally:
        # On a failed assertion preserve the isolated diagnostic model too;
        # restoration never turns the failure into a passing result.
        net.load_state_dict(state, strict=True)
        for mod, mode in zip(net.modules(), modes):
            mod.training = mode
        for param, (old, snapshot) in zip(params, grads):
            param.grad = old
            if old is not None:
                old.copy_(snapshot)
        m.restore_rng(rng, "cpu")


def output_mask(net, bands):
    output = net(bands)
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] < 2 or output.shape[2:] != bands.shape[2:]:
        raise ValueError("Changed model mask layout")
    mask = (output[:, :2]+1)/2
    cpu_float(mask, tuple(bands.shape))
    if (mask < 0).any() or (mask > 1).any():
        raise ValueError("Original bounded real mask required")
    return mask


@torch.no_grad()
def context_mask(net, bands, history, block):
    cpu_float(bands)
    if (bands.ndim != 4 or tuple(bands.shape[:3]) != (1, 2, 128) or bands.shape[-1] < 8 or bands.shape[-1] % 8 or
        type(history) is not int or history < 0 or history % 8 or type(block) is not int or block < 8 or block % 8):
        raise ValueError("Stride8-aligned CPU bands/history/block required")
    outputs = []
    for start in range(0, bands.shape[-1], block):
        left = max(0, start-history)
        outputs.append(output_mask(net, bands[..., left:start+block])[..., start-left:])
    return torch.cat(outputs, dim=-1)


def delta_stats(a, b):
    if a.shape != b.shape or a.numel() == 0:
        raise ValueError("Nonempty aligned diagnostic region required")
    delta = (a.detach().double()-b.detach().double()).abs()
    return {"mean_abs": float(delta.mean()), "max_abs": float(delta.max())}


def mask_stats(mask, whole):
    cpu_float(mask); cpu_float(whole)
    if mask.shape != whole.shape or mask.shape[-1] != FRAMES:
        raise ValueError("Original352-frame diagnostic mask required")
    target = mask[..., KILL:, 130:FRAMES-2]
    reference = whole[..., KILL:, 130:FRAMES-2]
    jumps = (target[..., 1:]-target[..., :-1]).detach().double().abs()
    phase = torch.arange(131, FRAMES-2) % 16 == 0
    return {**delta_stats(target, reference), "boundary_jump_mean": float(jumps[..., phase].mean()),
        "interior_jump_mean": float(jumps[..., ~phase].mean()), "frame_interval": [130, FRAMES-2], "band_interval": [KILL, 128]}


def lf_geometry(gs):
    cpu_float(gs, (513, 128))
    if (gs < 0).any() or not torch.allclose(gs.sum(1), torch.ones(513), atol=1e-6, rtol=0):
        raise ValueError("Nonnegative row-normalized original synthesis required")
    forced = gs[:, KILL:].sum(1) == 0
    bins = forced.nonzero().flatten().tolist()
    return forced, {"kill_bands": KILL, "forced_zero_fft_bins": bins,
        "forced_zero_bin_hz": [index*44100/1024 for index in bins],
        "one_sided_energy_weighting": "2 for interior bins,1 for DC/Nyquist", "layout": "legacy_log"}


def lf_statistics(spectrum, truth, mask, gs):
    if (spectrum.device.type != "cpu" or truth.device.type != "cpu" or spectrum.dtype != torch.complex64 or
        truth.dtype != torch.complex64 or spectrum.shape != (1, 2, 513, FRAMES) or truth.shape != spectrum.shape or
        not torch.isfinite(spectrum).all() or not torch.isfinite(truth).all()):
        raise ValueError("Original finite CPU complex64 spectra required")
    cpu_float(mask, (1, 2, 128, FRAMES))
    forced, geometry = lf_geometry(gs)
    region = slice(98, FRAMES-2)
    weights = torch.full((513,), 2., dtype=torch.float64)
    weights[[0, -1]] = 1
    def power(value):
        return value[..., region].detach().to(torch.complex128).abs().square()*weights[None, None, :, None]
    truth_power = power(truth)
    total = float(truth_power.sum())
    before = torch.einsum("fk,bckt->bcft", gs, mask).clamp(0, 1)
    after = m.fit.protected_full_mask(mask, gs, KILL)
    discarded = power(spectrum*(before-after))
    return {"geometry": geometry, "complex_frame_interval": [98, FRAMES-2], "reference_spectral_energy": total,
        "reference_forced_zero_energy_fraction": float(truth_power[:, :, forced].sum())/total if total > 1e-20 else None,
        "reference_fraction_skip": None if total > 1e-20 else "zero_or_near_zero_reference",
        "discarded_prediction_spectral_energy": float(discarded.sum()),
        "waveform_quality_floor_claimed": False}


def reconstruction_stats(predicted, x, v, region, physical):
    pv, mix, ref = [a[..., region].detach() for a in (predicted, x, v)]
    mse = float((pv.double()-ref.double()).square().mean())
    mae = float((pv.double()-ref.double()).abs().mean())
    metrics = suite.separation_metrics(pv[0], mix[0], ref[0]) if physical else None
    return {"sample_interval": [region.start, region.stop], "target_error_mse": mse, "target_error_mae": mae,
        "physical_true_reference_metrics": metrics, "pseudo_target_is_ground_truth": False}


@torch.no_grad()
def slot_probe(net, x, v, wa, gs, meta):
    cpu_float(x, (1, 2, SAMPLES)); cpu_float(v, (1, 2, SAMPLES))
    cpu_float(wa, (513, 128)); cpu_float(gs, (513, 128))
    if meta.get("role") not in ("train", "pseudo_label_train_candidate") or meta.get("domain") not in m.DOMAINS:
        raise ValueError("TRAIN roles only")
    with readonly_model(net):
        spectrum, truth = core.stft_batch(x), core.stft_batch(v)
        bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
        count = bands.shape[-1]
        if count != FRAMES:
            raise ValueError("Changed352-frame original crop")
        padded = F.pad(bands, (0, (-count) % 16))
        masks = {PATHS[0]: output_mask(net, padded)[..., :count]}
        for name, history, block in ((PATHS[1], 128, 256), (PATHS[2], 128, 16), (PATHS[3], 0, 16)):
            masks[name] = context_mask(net, padded, history, block)[..., :count]
        native, common = suite.scoring_slice(SAMPLES, 96), suite.scoring_slice(SAMPLES, 128)
        physical = meta["role"] == "train"
        predictions = {name: core.product_vocal(spectrum, mask, gs, SAMPLES, KILL) for name, mask in masks.items()}
        base, parts, fitted = m.fit.reconstruction_loss(masks[PATHS[0]], spectrum, x, v, gs, 96, KILL)
        same = delta_stats(fitted, predictions[PATHS[0]])
        if same["max_abs"] > 5e-6:
            raise ValueError("126/110 equivalent waveform reconstruction differs")
        roundtrip = core.t09._istft(spectrum.flatten(0, 1), SAMPLES).reshape_as(x)
        if delta_stats(roundtrip, x)["max_abs"] > 5e-6:
            raise ValueError("Original STFT/ISTFT input roundtrip fails")
        result = {"paths": {name: {"mask": mask_stats(mask, masks[PATHS[0]]),
            "wave_vs_whole_common": delta_stats(predictions[name][..., common], predictions[PATHS[0]][..., common]),
            "common": reconstruction_stats(predictions[name], x, v, common, physical),
            "native_descriptive_only": reconstruction_stats(predictions[name], x, v, native, physical)} for name, mask in masks.items()},
            "lf44": lf_statistics(spectrum, truth, masks[PATHS[0]], gs),
            "same_mask_126_vs_110": {**same, "bit_equal": torch.equal(fitted, predictions[PATHS[0]])},
            "original_unweighted_base_loss_descriptive_only": {"total": float(base), **{key: float(value) for key, value in parts.items()}},
            "input_roundtrip": {"all": delta_stats(roundtrip, x), "common": delta_stats(roundtrip[..., common], x[..., common]),
                "left_edge512": delta_stats(roundtrip[..., :512], x[..., :512]), "right_edge512": delta_stats(roundtrip[..., -512:], x[..., -512:])},
            "model_state_modes_rng_grad_unchanged": True}
        if not d.finite_state(result):
            raise ValueError("Nonfinite diagnostic result")
    return result


def bucket(meta):
    return f"{meta['domain']}/{meta['role']}/vocal_db={meta.get('vocal_db')}"


def summarize(rows):
    summary = {}
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        slots = [slot for row in selected for slot in row["slots"]]
        groups = {}
        for name in sorted({bucket(slot["metadata"]) for slot in slots}):
            group = [slot for slot in slots if bucket(slot["metadata"]) == name]
            fractions = [slot["probe"]["lf44"]["reference_forced_zero_energy_fraction"] for slot in group
                if slot["probe"]["lf44"]["reference_forced_zero_energy_fraction"] is not None]
            groups[name] = {"slots": len(group), "lf_fraction_count": len(fractions),
                "lf_fraction_mean": statistics.fmean(fractions) if fractions else None,
                "lf_fraction_skip_count": len(group)-len(fractions),
                "paths": {path: {"mask_mean_abs_vs_whole": statistics.fmean(slot["probe"]["paths"][path]["mask"]["mean_abs"] for slot in group),
                    "mask_max_abs_vs_whole": max(slot["probe"]["paths"][path]["mask"]["max_abs"] for slot in group),
                    "wave_common_mean_abs_vs_whole": statistics.fmean(slot["probe"]["paths"][path]["wave_vs_whole_common"]["mean_abs"] for slot in group)} for path in PATHS}}
        summary[model] = {"model_batches": len(selected), "slot_checks": len(slots), "groups": groups,
            "musdb_weak_minus12_slots": sum(slot["metadata"]["domain"] == "musdb" and slot["metadata"].get("vocal_db") == -12 for slot in slots),
            "mir_weak_minus12_slots": sum(slot["metadata"]["domain"] == "mir1k" and slot["metadata"].get("vocal_db") == -12 for slot in slots)}
    return summary


def fresh_output(out):
    out = out.resolve()
    allowed = (ROOT / "results").resolve()
    if out == allowed or not out.is_relative_to(allowed) or out.exists():
        raise ValueError("Fresh child result directory only; do not overwrite evidence")
    return out


def prepare(out):
    out = fresh_output(out)
    no_active_worker(); check_disk()
    _, models, _, bindings = checked_sources()
    plan = fixed_scope() | {"bindings_sha256": bindings,
        "model_digests": {name: r.dev.state_digest(saved["model"]) for name, saved in models.items()}}
    check_plan(plan)
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    print("TRAIN_FRONTEND PLAN SEALED;36 CPU model-batches216 slots; updates0", flush=True)


def verified_plan(out):
    plan = acq.read_sealed(out / "plan.json")
    check_plan(plan)
    check_bindings(plan["bindings_sha256"])
    _, models, _, bindings = checked_sources()
    if bindings != plan["bindings_sha256"] or plan["model_digests"] != {name: r.dev.state_digest(saved["model"]) for name, saved in models.items()}:
        raise ValueError("Changed source/code/test/plan binding")
    return plan


def status(out, values):
    temporary = out / f"status_{os.getpid()}.tmp"
    acq.write_new_json(temporary, values | {"pid": os.getpid(), "updated_utc": m.bulk.now(), "purpose": PURPOSE,
        "model_updates": 0, "cuda_used": False, "release_selection": "NONE"})
    os.replace(temporary, out / "run_status.json")


def reject_existing_run(out):
    if any((out / name).exists() for name in ("run_status.json", "inputs.json", "diagnostic.json")) or list(out.glob("row_*.json")):
        raise ValueError("Existing run/evidence: inspect/verify, never automatically repeat")


def run(out):
    reject_existing_run(out)
    no_active_worker(); check_disk()
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only fresh process required")
    plan = verified_plan(out)
    approval, models, sampler, _ = checked_sources()
    torch.set_num_threads(THREADS)
    outer_rng = m.capture_rng("cpu")
    rows, inputs, row_hashes = [], [], {}
    with m.bulk.worker_lock(out), d.deterministic_runtime("cpu"):
        status(out, {"status": "running", "phase": "collect_fixed_train", "completed_model_batches": 0, "limit": 36, "error": None})
        try:
            dataset = p.inp.ApprovedTeacherDataset(p.inp.verified_approval(Path(approval["origin_approval"])), "kim_melband")
            true = m.LockedTruePool(m.bulk.OLD_LOCK, dataset.config)
            if sampler["seed"] != dataset.seed or sampler["true_lock_sha256"] != true.bound:
                raise ValueError("Changed original input seed/true lock")
            batches = []
            for cursor in CURSORS:
                batch = s.collect(dataset, true, cursor)
                t.validate_metadata(batch["metadata"])
                cpu_float(batch["x"], (6, 2, SAMPLES)); cpu_float(batch["v"], (6, 2, SAMPLES))
                hashes = [m.pilot.wave_digest(value) for value in batch["x"]]
                if hashes != [meta["input_pcm_sha256"] for meta in batch["metadata"]]:
                    raise ValueError("Actual TRAIN input PCM differs")
                inputs.append({"counter": cursor, "input_sha256": hashes, "target_sha256": [m.pilot.wave_digest(value) for value in batch["v"]], "metadata": batch["metadata"]})
                batches.append(batch)
            acq.write_new_json(out / "inputs.json", acq.seal({"purpose": PURPOSE, "plan_sha256": acq.sha256(out / "plan.json"), "inputs": inputs}))
            wa, gs = (torch.from_numpy(fn()) for fn in (core.t09.make_analysis_matrix, core.t09.make_synthesis_matrix))
            for model in MODELS:
                net = m.frozen_factory(approval["source_protocol"])()
                saved = models[model]
                net.load_state_dict(saved["model"], strict=True)
                if saved["parameter_names"] != [key for key, _ in net.named_parameters()] or len(saved["modes"]) != len(list(net.modules())):
                    raise ValueError("Changed model parameter/mode layout")
                for mod, mode in zip(net.modules(), saved["modes"]):
                    mod.training = mode
                check_net(net)
                if r.dev.state_digest(net.state_dict()) != plan["model_digests"][model]:
                    raise ValueError("Changed model identity")
                for index, batch in enumerate(batches):
                    check_disk()
                    started = time.perf_counter()
                    slots = [{"slot": j, "metadata": meta,
                        "probe": slot_probe(net, batch["x"][j:j+1], batch["v"][j:j+1], wa, gs, meta)}
                        for j, meta in enumerate(batch["metadata"])]
                    row = {"model": model, "counter": CURSORS[index], "slots": slots, "seconds": time.perf_counter()-started}
                    name = f"row_{len(rows):02d}.json"
                    acq.write_new_json(out / name, acq.seal(row))
                    row_hashes[name] = acq.sha256(out / name)
                    rows.append(row)
                    status(out, {"status": "running", "phase": "cpu_frontend_diagnostic", "model": model, "counter": CURSORS[index],
                        "completed_model_batches": len(rows), "limit": 36, "error": None})
                    print(f"TRAIN_FRONTEND model={model} counter={CURSORS[index]} completed={len(rows)}/36 seconds={row['seconds']:.3f}", flush=True)
                if r.dev.state_digest(net.state_dict()) != plan["model_digests"][model] or [mod.training for mod in net.modules()] != saved["modes"] or any(param.grad is not None for param in net.parameters()):
                    raise ValueError("Diagnostic changed model/state/modes/grad")
            result = fixed_scope() | {"plan_sha256": acq.sha256(out / "plan.json"), "bindings_sha256": plan["bindings_sha256"],
                "inputs_sha256": acq.sha256(out / "inputs.json"), "row_sha256": row_hashes,
                "model_digests": plan["model_digests"], "rows": rows, "inputs": inputs, "summary": summarize(rows),
                "runtime": d.runtime_identity("cpu"), "model_state_modes_rng_grad_unchanged": True,
                "scope": "Fixed TRAIN frontend attribution only; not old DEVELOPMENT replacement, board stream, quality floor, full-corpus causality or model capacity proof"}
            if not d.finite_state(result) or torch.cuda.is_initialized():
                raise ValueError("Nonfinite diagnostic or unexpected CUDA")
            acq.write_new_json(out / "diagnostic.json", acq.seal(result))
            validate_result(plan, result)
            status(out, {"status": "complete", "phase": "cpu_frontend_diagnostic_complete", "completed_model_batches": 36, "limit": 36, "error": None})
            print("TRAIN_FRONTEND COMPLETE36; CPU-only updates0; NONRELEASE", flush=True)
        except BaseException as error:
            status(out, {"status": "failed", "phase": "cpu_frontend_diagnostic", "completed_model_batches": len(rows), "limit": 36, "error": repr(error)})
            raise
        finally:
            m.restore_rng(outer_rng, "cpu")


def validate_result(plan, result):
    check_plan(result)
    if (result["bindings_sha256"] != plan["bindings_sha256"] or result["model_digests"] != plan["model_digests"] or
        result["model_state_modes_rng_grad_unchanged"] is not True or len(result["rows"]) != 36 or len(result["inputs"]) != 12 or
        result["runtime"]["device"] != "cpu" or result["runtime"]["threads"] != THREADS or result["runtime"]["deterministic"] is not True or
        result["summary"] != summarize(result["rows"]) or not d.finite_state(result)):
        raise ValueError("Changed diagnostic coverage/identity/runtime/summary")
    for index, entry in enumerate(result["inputs"]):
        if entry["counter"] != CURSORS[index] or len(entry["target_sha256"]) != 6 or entry["input_sha256"] != [meta["input_pcm_sha256"] for meta in entry["metadata"]]:
            raise ValueError("Changed input accounting")
        t.validate_metadata(entry["metadata"])
    geometry = lf_geometry(torch.from_numpy(core.t09.make_synthesis_matrix()))[1]
    for i, row in enumerate(result["rows"]):
        index = i % 12
        if row["model"] != MODELS[i//12] or row["counter"] != CURSORS[index] or len(row["slots"]) != 6 or row["seconds"] < 0:
            raise ValueError("Changed model/counter/slot ordering")
        for j, slot in enumerate(row["slots"]):
            probe = slot["probe"]
            if slot["slot"] != j or slot["metadata"] != result["inputs"][index]["metadata"][j] or set(probe["paths"]) != set(PATHS) or probe["model_state_modes_rng_grad_unchanged"] is not True:
                raise ValueError("Changed role/path or mutated model")
            if probe["same_mask_126_vs_110"]["max_abs"] > 5e-6 or probe["input_roundtrip"]["all"]["max_abs"] > 5e-6:
                raise ValueError("Reconstruction identity failed")
            lf = probe["lf44"]
            fraction = lf["reference_forced_zero_energy_fraction"]
            if lf["geometry"] != geometry or lf["complex_frame_interval"] != [98, FRAMES-2] or lf["waveform_quality_floor_claimed"] is not False or not (
                (fraction is None and lf["reference_fraction_skip"] == "zero_or_near_zero_reference" and lf["reference_spectral_energy"] <= 1e-20) or
                (fraction is not None and 0 <= fraction <= 1 and lf["reference_fraction_skip"] is None and lf["reference_spectral_energy"] > 1e-20)):
                raise ValueError("LF geometry/fraction/skip scope changed")
            for path, values in probe["paths"].items():
                if (values["mask"]["frame_interval"] != [130, FRAMES-2] or values["mask"]["band_interval"] != [44, 128] or
                    values["common"]["sample_interval"] != [130*HOP, SAMPLES-2*HOP] or
                    values["native_descriptive_only"]["sample_interval"] != [98*HOP, SAMPLES-2*HOP] or
                    any(values[part]["pseudo_target_is_ground_truth"] is not False for part in ("common", "native_descriptive_only"))):
                    raise ValueError("Changed scoring support or pseudo authority")
                pseudo = slot["metadata"]["role"] == "pseudo_label_train_candidate"
                if any((values[part]["physical_true_reference_metrics"] is None) != pseudo for part in ("common", "native_descriptive_only")):
                    raise ValueError("Pseudo target promoted to physical true metrics")
                if path == PATHS[0] and any(values[key]["max_abs"] != 0 for key in ("mask", "wave_vs_whole_common")):
                    raise ValueError("Whole reference differs from itself")


def verify(out):
    plan = verified_plan(out)
    result = acq.read_sealed(out / "diagnostic.json")
    if result["plan_sha256"] != acq.sha256(out / "plan.json") or result["inputs_sha256"] != acq.sha256(out / "inputs.json"):
        raise ValueError("Changed plan/inputs artifact binding")
    inputs = acq.read_sealed(out / "inputs.json")
    if inputs["inputs"] != result["inputs"] or inputs["purpose"] != PURPOSE or inputs["plan_sha256"] != result["plan_sha256"]:
        raise ValueError("Changed independent input evidence")
    expected_names = [f"row_{index:02d}.json" for index in range(36)]
    if set(result["row_sha256"]) != set(expected_names):
        raise ValueError("Incomplete committed rows")
    for index, name in enumerate(expected_names):
        if acq.sha256(out / name) != result["row_sha256"][name] or r.r.plain(acq.read_sealed(out / name)) != result["rows"][index]:
            raise ValueError("Changed independent committed row")
    validate_result(plan, result)
    print("TRAIN_FRONTEND VERIFIED36 model-batches216 slots; CPU-only updates0; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    globals()[args.operation](args.out)
