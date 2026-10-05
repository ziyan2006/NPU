"""Fixed TRAIN-only stored Adam direction alignment; never an optimizer update."""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import types
import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = load("adam_memory_source_review", "196_review_mel_lr_scale.py")
q = load("adam_memory_split_helpers", "175_diagnose_auxiliary_components.py")
t, p, m, d, acq, ROOT = r.t, r.p, r.m, r.t.d, r.acq, r.ROOT
PURPOSE = "NONRELEASE_TRAIN_ADAM_MEMORY_ALIGNMENT"
MODELS = ("common_lf32_4000", "lr1_4500", "lr_half_4500")
CURSORS = (4500, 4501, 4502)
GROUPS = q.GROUPS
COEFFICIENTS = (1., 4., 1., .2, .2)
SOURCE = t.DEFAULT_OUT
DEFAULT_OUT = ROOT / "results/train_adam_memory_20261004"
MONITOR = ROOT / "results/train_adam_memory_monitor_20261004"
TEST = Path(__file__).with_name("_test_train_adam_memory.py")
PROTOCOL = ROOT / "docs/train_adam_memory_protocol_20261004.json"
PROOF = ROOT / "results/mel_lr_scale_monitor_20261004/completion_review.json"
PINS = {
    "results/mel_lr_scale_monitor_20261004/completion_review.json": "817ff28fa4c25cf0319a326899fd2ac062fbd95688b2ccce56647f1c78e2a715",
    "reports/89_mel_lr_scale_completion.md": "4bac168c5e42d56b8b024452ab891b409e95361a0088c1001296a24016aabeb0",
    "reports/90_train_adam_memory_alignment_plan.md": "f6ca742d09c4b34d749b81ac592234242cad4245c5d60e0336d1d36c7ae22548",
}
CHECKPOINT_SHA = {4000: "927806c04e71729b1a1d7a5ebb3fccafecb85790979222bdad6966b6e0729e18",
                  4500: "b3a0450ecd9bba259981bd47f8a49c0fee1a997d7a087a72570f589cc14ed5b3"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def exact(a, b):
    return canonical(a) == canonical(b)


def tree_digest(value):
    """Include recursive types, tensor dtype/shape and bytes; no optimizer object."""
    h = hashlib.sha256()
    def visit(v):
        h.update((type(v).__module__ + "." + type(v).__name__).encode())
        if isinstance(v, torch.Tensor):
            if v.device.type != "cpu":
                raise ValueError("CPU copies only")
            h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(v[key])
        elif isinstance(v, (tuple, list)):
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def scope():
    return {"schema": 1, "purpose": PURPOSE, "models": list(MODELS), "cursors": list(CURSORS),
        "unique_input_batches": 3, "unique_input_slots": 18, "model_batches": 9,
        "primary_slot_forwards": 54, "reference_slot_forwards": 18, "total_slot_forwards": 72,
        "groups": list(GROUPS), "coefficients": list(COEFFICIENTS), "normalizer": 6,
        "samples": 89856, "frames": 352, "warmup": 96, "native_support": [25088, 89344],
        "kill_bands": 32, "forced_zero_fft_bins": [0, 1, 2, 3], "threads": 2,
        "forward_backward_dtype": "float32", "spectrum_dtype": "complex64", "statistics_dtype": "float64",
        "gradient_rtol": .0002, "gradient_atol": .0000002,
        "dot_arithmetic_rtol": .000001, "lr_dot_norm_atol_factor": .0000002, "combined_dot_norm_atol_factor": .0000005,
        "direction_formula": "u=(m/(1-beta1^t))/(sqrt(v/(1-beta2^t))+eps); d=-actual_group_lr*u",
        "interpretation": "Stored-moments last-direction proxy, not rounded parameter delta or next Adam update",
        "input_reader": "independent_readonly_original193_recipe_cursor4500_to4502",
        "training_authorized": False, "optimizer_constructed": False, "model_updates": 0,
        "cuda_used": False, "deployment": False, "release_selection": "NONE"}


def check_scope(doc):
    if any(not exact(doc.get(key), value) for key, value in scope().items()):
        raise ValueError("Changed fixed zero-update scope/types")


def cpu_float(v, shape=None):
    if (not isinstance(v, torch.Tensor) or v.dtype != torch.float32 or v.device.type != "cpu" or
        (shape is not None and tuple(v.shape) != tuple(shape)) or not torch.isfinite(v).all()):
        raise ValueError("Finite CPU FP32 and exact shape required")


def matrix_identity():
    wa = torch.from_numpy(m.core.t09.make_analysis_matrix())
    gs = torch.from_numpy(m.core.t09.make_synthesis_matrix())
    cpu_float(wa, (513, 128)); cpu_float(gs, (513, 128))
    if torch.nonzero(gs[:, 32:].eq(0).all(dim=1)).flatten().tolist() != list(range(4)):
        raise ValueError("Original literal LF32 synthesis identity required")
    return wa, gs, {"analysis_sha256": tree_digest(wa), "synthesis_sha256": tree_digest(gs),
                    "kill_bands": 32, "forced_zero_fft_bins": list(range(4))}


def group_for_slot(i, meta):
    return q.group_for_slot(i, meta)


def bucket(meta):
    if meta["domain"] == "pseudo":
        if meta["role"] != "pseudo_label_train_candidate":
            raise ValueError("Changed pseudo role")
        return "pseudo_not_final_truth"
    if meta["role"] != "train" or type(meta.get("vocal_db")) is not int:
        raise ValueError("True TRAIN literal gain required")
    if meta["domain"] == "instrumental" and meta["vocal_db"] == 0:
        return "instrumental_zero_reference"
    gain = {0: "native", -12: "weak_minus12", -6: "attenuated_minus6", 6: "boosted_plus6"}
    if meta["domain"] not in ("musdb", "mir1k") or meta["vocal_db"] not in gain:
        raise ValueError("All and only original legitimate gains")
    return meta["domain"] + "_" + gain[meta["vocal_db"]]


def input_entry(batch):
    cursor = batch["diagnostic_counter"]
    if type(cursor) is not int or cursor not in CURSORS:
        raise ValueError("Only next three fixed TRAIN counters")
    x, v = batch["x"], batch["v"]
    cpu_float(x, (6, 2, 89856)); cpu_float(v, x.shape)
    t.validate_metadata(batch["metadata"])
    for meta in batch["metadata"]:
        bucket(meta)
        if (meta["score_start"], meta["score_end"]) != (25088, 89344):
            raise ValueError("Original native support changed")
    hashes = [m.pilot.wave_digest(w) for w in x]
    if hashes != [meta["input_pcm_sha256"] for meta in batch["metadata"]]:
        raise ValueError("Actual PCM/metadata differs")
    if torch.count_nonzero(v[2]):
        raise ValueError("Instrumental reference must remain zero")
    return {"counter": cursor, "input_sha256": hashes, "target_sha256": [m.pilot.wave_digest(w) for w in v],
            "metadata": copy.deepcopy(batch["metadata"]), "identity_sha256": tree_digest(batch)}


def collect_next(dataset, true, cursor):
    """Exact193 next_batch crop algorithm in a bounded stateless diagnostic copy.

    No old sampler/limit loaded or overridden. All crop sources verify their
    own immutable PCM/teacher bindings. No sampling based on activity/scores.
    """
    if type(cursor) is not int or cursor not in CURSORS:
        raise ValueError("Fixed diagnostic next-input bounds")
    xs, vs, metas = [], [], []
    for domain in m.DOMAINS[:3]:
        item = true.crop(domain, dataset.seed, cursor)
        xs.append(item["x"]); vs.append(item["v"]); metas.append(item["meta"])
    for i in range(3):
        recipe = m.data.crop_recipe(dataset.rows, dataset.config, dataset.seed, cursor*3+i)
        item = dataset.crop(recipe)
        xs.append(item["x"]); vs.append(item["v"])
        metas.append(item["meta"] | {"domain": "pseudo", "role": "pseudo_label_train_candidate"})
    batch = {"x": torch.stack(xs), "v": torch.stack(vs), "metadata": metas, "diagnostic_counter": cursor}
    input_entry(batch)
    return batch


def stored_direction(saved):
    names, model, optimizer = saved["parameter_names"], saved["model"], saved["optimizer"]
    if (type(names) is not list or not names or len(set(names)) != len(names) or
        any(type(n) is not str or n not in model for n in names) or type(saved["updates"]) is not int):
        raise ValueError("Actual unique parameter names/exposure required")
    if set(optimizer) != {"state", "param_groups"} or type(optimizer["param_groups"]) is not list or len(optimizer["param_groups"]) != 1:
        raise ValueError("Original single complete Adam group required")
    group = optimizer["param_groups"][0]
    fixed = {"betas": (.9, .999), "eps": 1e-8, "weight_decay": 0,
        "amsgrad": False, "maximize": False, "foreach": False, "capturable": False,
        "differentiable": False, "fused": False, "decoupled_weight_decay": False}
    if set(group) != set(fixed) | {"lr", "params"} or any(tree_digest(group[k]) != tree_digest(v) for k, v in fixed.items()):
        raise ValueError("Actual Adam flags/beta/epsilon/types differ from original")
    ids = group["params"]
    if (type(ids) is not list or any(type(i) is not int for i in ids) or ids != list(range(len(names))) or
        set(optimizer["state"]) != set(ids) or any(type(i) is not int for i in optimizer["state"]) or
        type(group["lr"]) is not float or not math.isfinite(group["lr"]) or group["lr"] <= 0):
        raise ValueError("Exact original ID/name ordering and actual LR required")
    us, ds, mapping = [], [], []
    for i, name in zip(ids, names):
        param, values = model[name], optimizer["state"][i]
        cpu_float(param)
        if set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Complete stored Adam tensor fields required")
        step, moment, variance = values["step"], values["exp_avg"], values["exp_avg_sq"]
        cpu_float(step, ()); cpu_float(moment, param.shape); cpu_float(variance, param.shape)
        count = float(step)
        if count != saved["updates"] or count <= 0 or not count.is_integer() or (variance < 0).any():
            raise ValueError("Actual finite nonnegative moments/positive exposure required")
        beta1, beta2 = group["betas"]
        u = (moment.detach()/(1-beta1**count))/(torch.sqrt(variance.detach()/(1-beta2**count))+group["eps"])
        delta_proxy = -group["lr"]*u
        cpu_float(u, param.shape); cpu_float(delta_proxy, param.shape)
        if u.data_ptr() == moment.data_ptr() or delta_proxy.data_ptr() == u.data_ptr():
            raise ValueError("Direction must use independent copies")
        us.append(u.reshape(-1)); ds.append(delta_proxy.reshape(-1))
        mapping.append({"parameter_id": i, "parameter_name": name, "shape": list(param.shape),
            "parameter_dtype": str(param.dtype), "step": count, "step_dtype": str(step.dtype),
            "step_shape": list(step.shape), "moment_sha256": tree_digest(moment), "variance_sha256": tree_digest(variance)})
    u, descent = torch.cat(us), torch.cat(ds)
    return u, descent, {"param_groups": copy.deepcopy(group), "parameter_mapping": mapping,
        "optimizer_sha256": tree_digest(optimizer), "u_sha256": tree_digest(u), "d_sha256": tree_digest(descent),
        "u_l2": float(u.detach().double().norm()), "d_l2": float(descent.detach().double().norm()),
        "formula_dtype": "float32", "proxy_not_actual_update": True}


def alignment(g, u, descent):
    for value in (g, u, descent):
        cpu_float(value, g.shape)
    if g.ndim != 1 or not g.numel():
        raise ValueError("Nonempty aligned flattened vectors")
    gv, uv, dv = (v.detach().double() for v in (g, u, descent))
    ng, nu, nd = (float(v.norm()) for v in (gv, uv, dv))
    gu, gd = float(gv@uv), float(gv@dv)
    def cosine(dot, a, b):
        if not a or not b:
            return None
        result = dot/(a*b)
        if not math.isfinite(result) or abs(result) > 1+1e-12:
            raise ValueError("Invalid detached cosine")
        return max(-1., min(1., result))
    result = {"g_l2": ng, "u_l2": nu, "d_l2": nd, "g_dot_u": gu, "g_dot_d": gd,
        "cos_g_u": cosine(gu, ng, nu), "cos_g_d": cosine(gd, ng, nd),
        "first_order_descent": gd < 0 if ng and nd else None,
        "gradient_sha256": tree_digest(g), "statistics_dtype": "float64"}
    validate_alignment(result)
    return result


def validate_alignment(doc):
    for key in ("g_l2", "u_l2", "d_l2", "g_dot_u", "g_dot_d"):
        if type(doc[key]) is not float or not math.isfinite(doc[key]):
            raise ValueError("Literal finite detached statistic required")
    if min(doc[key] for key in ("g_l2", "u_l2", "d_l2")) < 0 or doc["statistics_dtype"] != "float64":
        raise ValueError("Invalid norm/precision")
    for key, dot, norm in (("cos_g_u", "g_dot_u", "u_l2"), ("cos_g_d", "g_dot_d", "d_l2")):
        expected = doc[dot]/(doc["g_l2"]*doc[norm]) if doc["g_l2"] and doc[norm] else None
        if expected is None:
            if doc[key] is not None or doc[dot] != 0.:
                raise ValueError("Zero norm cosine unavailable")
        elif (type(doc[key]) is not float or abs(expected) > 1+1e-12 or
              not math.isclose(doc[key], max(-1., min(1., expected)), abs_tol=1e-12)):
            raise ValueError("Cosine/dot mismatch")
    available = bool(doc["g_l2"] and doc["d_l2"])
    if not exact(doc["first_order_descent"], doc["g_dot_d"] < 0 if available else None):
        raise ValueError("Local descent sign changed")


@contextlib.contextmanager
def readonly(net, saved, batch):
    before = tree_digest({"net": net.state_dict(), "modes": [mod.training for mod in net.modules()],
        "grad": [param.grad for param in net.parameters()], "source": saved, "batch": batch,
        "rng": m.capture_rng("cpu")})
    try:
        yield
    finally:
        after = tree_digest({"net": net.state_dict(), "modes": [mod.training for mod in net.modules()],
            "grad": [param.grad for param in net.parameters()], "source": saved, "batch": batch,
            "rng": m.capture_rng("cpu")})
        if before != after or torch.cuda.is_initialized():
            raise ValueError("Mutated weights/modes/grad/moments/inputs/RNG or initialized CUDA")


def slot_losses(net, xb, vb, wa, gs, meta):
    spectrum = m.core.stft_batch(xb)
    if spectrum.dtype != torch.complex64 or spectrum.shape != (1, 2, 513, 352):
        raise ValueError("Original352 complex64 support required")
    bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
    output = net(bands)
    cpu_float(output, (1, 4, 128, 352))
    mask = (output[:, :2]+1)/2
    base, parts, wave = m.fit.reconstruction_loss(mask, spectrum, xb, vb, gs, 96, 32)
    cpu_float(base, ()); cpu_float(wave, xb.shape)
    region = m.fit.suite.scoring_slice(xb.shape[-1], 96)
    if (region.start, region.stop) != (25088, 89344):
        raise ValueError("Changed original scoring edges")
    residual, accompaniment, original, info = q.split_auxiliary(wave[0, :, region], xb[0, :, region], vb[0, :, region], meta)
    actual, actual_info = t.k.source_projection_component_loss(wave[0, :, region], xb[0, :, region], vb[0, :, region], meta, 1)
    if not torch.equal(original, actual) or not exact(info, actual_info):
        raise ValueError("Split scalar is not original194 full auxiliary")
    return base, residual, accompaniment, original, info, {"wave_sha256": tree_digest(wave),
        "mask_sha256": tree_digest(mask), "base": float(base.detach()), "parts": {key: float(value.detach()) for key, value in parts.items()}}


def parameter_probe(net, saved, batch, wa, gs, independent):
    input_identity = input_entry(batch)
    params = tuple(net.parameters())
    if (saved["parameter_names"] != [name for name, _ in net.named_parameters()] or
        any(not param.requires_grad for param in params) or any(isinstance(mod, (
            torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)) for mod in net.modules())):
        raise ValueError("Original deterministic graph and parameter ordering required")
    for param in params:
        cpu_float(param)
    u, descent, direction = stored_direction(saved)
    vectors = {key: torch.zeros_like(u) for key in GROUPS}
    slots = []
    def grad(loss, retain=False):
        return q.s.flatten_gradients(torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True), params)
    with readonly(net, saved, batch):
        for i, meta in enumerate(batch["metadata"]):
            base, residual, accompaniment, original, info, identity = slot_losses(net, batch["x"][i:i+1], batch["v"][i:i+1], wa, gs, meta)
            active = info["active"]
            if type(active) is not bool or (active and i >= 2):
                raise ValueError("Original auxiliary role/activity required")
            gb = grad(base*(1/6), active)
            gr = grad(residual*(1/6), True) if active else torch.zeros_like(u)
            ga = grad(accompaniment*(1/6), True) if active else torch.zeros_like(u)
            go = grad(original*(1/6)) if active else torch.zeros_like(u)
            equality = q.compare_gradients(go, gr+ga)
            if not active and any(float(v.detach()) != 0. for v in (residual, accompaniment, original)):
                raise ValueError("Skipped auxiliary nonzero")
            group = group_for_slot(i, meta)
            vectors[group] += gb; vectors[GROUPS[3]] += gr; vectors[GROUPS[4]] += ga
            slots.append({"slot": i, "group": group, "bucket": bucket(meta), "metadata": meta,
                "auxiliary": info, "base_div6": float(base.detach())/6, "cv2_div6": float(residual.detach())/6,
                "ca1_sq_div6": float(accompaniment.detach())/6, "full_aux_div6": float(original.detach())/6,
                "scalar_split_bit_exact": True, "split_gradient_equivalence": equality,
                "identity": identity, "alignment": {"base": alignment(gb, u, descent),
                    "cv2": alignment(gr, u, descent), "ca1_sq": alignment(ga, u, descent)}})
        combined = vectors[GROUPS[0]]+4*vectors[GROUPS[1]]+vectors[GROUPS[2]]+.2*(vectors[GROUPS[3]]+vectors[GROUPS[4]])
        reference = None
        if independent:
            actual = torch.zeros_like(u)
            checks = []
            for i, meta in enumerate(batch["metadata"]):
                base, _, _, original, info, identity = slot_losses(net, batch["x"][i:i+1], batch["v"][i:i+1], wa, gs, meta)
                loss, _ = t.w.combine_slot_loss(base, original, meta, i, 4, info)
                if not exact(identity, slots[i]["identity"]):
                    raise ValueError("Repeated original194 loss/wave identity differs")
                actual += grad(loss)
                checks.append({"slot": i, "loss": float(loss.detach()), "original194_scalar_wave_bit_exact": True})
            reference = q.compare_gradients(actual, combined) | {"slot_checks": checks,
                "reference_forwards": 6, "operation": "original194 combine_slot_loss autograd.grad, not backward/Adam",
                "rtol": .0002, "atol": .0000002, "bit_exact_claimed": False,
                "actual_gradient_sha256": tree_digest(actual), "split_combined_gradient_sha256": tree_digest(combined)}
        result = {"direction": direction, "group_alignment": {key: alignment(value, u, descent) for key, value in vectors.items()},
            "weighted_alignment": {key: alignment(value*coefficient, u, descent) for (key, value), coefficient in zip(vectors.items(), COEFFICIENTS)},
            "combined_alignment": alignment(combined, u, descent), "slots": slots, "input": input_identity,
            "independent_combination": reference, "primary_slot_forwards": 6, "reference_slot_forwards": 6 if independent else 0,
            "weights_modes_grad_rng_moments_inputs_unchanged": True, "existing_grad_storage_written": False}
    return result


def no_active_worker():
    # Ignore only this foreground worker and its venv shim, never other tasks.
    escaped = str(ROOT).replace("'", "''")
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and ($_.CommandLine -like '*"+escaped+"*' -or $_.CommandLine -match '(197_diagnose_train_adam_memory|_test_train_adam_memory)[.]py') } | Select-Object -ExpandProperty ProcessId"
    reply = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], check=True, capture_output=True, text=True)
    own = {os.getpid(), os.getppid()}
    if any(int(line.strip()) not in own for line in reply.stdout.splitlines() if line.strip()):
        raise ValueError("Active repository Python task; preserve and postpone")
    p.no_active_trainer()


def resources():
    if shutil.disk_usage(ROOT).free/2**30 < 12 or torch.cuda.is_initialized():
        raise ValueError("Disk>=12GiB and fresh CPU/no CUDA required")


def binding_stamps(paths):
    """Cheap per-batch guard after full SHA verification; no cache/data writes."""
    result = {}
    for filename in paths:
        path = Path(filename)
        if not path.is_file():
            raise ValueError("Bound file missing")
        stat = path.stat()
        result[str(path)] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return result


def unchanged_bindings(stamps):
    if binding_stamps(stamps) != stamps:
        raise ValueError("Bound file changed during readonly diagnostic")


def module_files():
    seen, files, queue = set(), set(), [r, q]
    while queue:
        mod = queue.pop()
        if id(mod) in seen:
            continue
        seen.add(id(mod)); filename = getattr(mod, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(ROOT / "scripts"):
            files.add(Path(filename).resolve())
            queue.extend(value for value in vars(mod).values() if isinstance(value, types.ModuleType))
    return files


def source_evidence():
    """Read existing completed proof/bindings; never execute old verify/summary."""
    authority = acq.read_sealed(p.DEFAULT_APPROVAL)
    if acq.sha256(p.DEFAULT_APPROVAL) != "fe9d1e930a84430922ef8cfb23446895a5d3313b0bea46112ac974a4e2956e1b":
        raise ValueError("Changed existing193 authority")
    p.check_approval(authority)
    bindings = dict(authority["bindings_sha256"])
    bindings |= {str(ROOT / path): sha for path, sha in PINS.items()}
    p.q.check_bindings(bindings)
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    if (proof["actual_training_exit_code"] != 0 or type(proof["actual_training_exit_code"]) is not int or
        proof["actual_worker_or_shim"] or proof["formal_round_completed"] is not True or
        proof["final_verify"]["actual_native_exit_code"] != 0 or proof["summary"]["actual_native_exit_code"] != 0 or
        proof["summary"]["built_in_verify_output_passed"] is not True or proof["pending_steps"] or
        proof["state"]["status"] != "complete" or proof["state"]["step"] != 4500 or
        proof["state"]["additional_step"] != 500 or proof["state"]["error"] is not None):
        raise ValueError("Real complete500/native exit0/independent closed proof required")
    launch, ending = proof["actual_launch"], proof["matched_actual_exit"]
    for entry in (launch, ending):
        bindings[entry["path"]] = entry["sha256"]
        if not exact(json.loads(Path(entry["path"]).read_text(encoding="utf-8-sig")), entry["document"]):
            raise ValueError("Retained actual launch/exit differs")
    if (ending["document"]["exit_code"] != 0 or launch["document"]["probe_only"] or
        Path(ending["document"]["launch_receipt"]) != Path(launch["path"]) or
        ending["document"]["launcher_process_id"] != launch["document"]["worker"]["LauncherProcessId"] or
        Path(launch["document"]["out"]) != SOURCE):
        raise ValueError("Launch/exit pairing mismatch")
    bindings |= {str((ROOT / path).resolve()): sha for path, sha in proof["immutable_evidence_sha256"].items()}
    models, descriptors, states = {}, {}, {}
    for step in (4000, 4500):
        receipt_path = SOURCE / f"checkpoint_{step}.json"
        receipt = acq.read_sealed(receipt_path)
        checkpoint = SOURCE / receipt["checkpoint"]
        if receipt["sha256"] != CHECKPOINT_SHA[step]:
            raise ValueError("Predeclared checkpoint identity differs")
        state = m.load_checked_checkpoint(checkpoint, receipt["sha256"])
        if (state["step"] != step or state["schedule"]["step"] != step or state["schedule"]["last_validation"] != step or
            state["sampler"]["cursor"] != step or state["schedule"]["stopped_at"] != (4500 if step == 4500 else None) or
            state["legacy_stop_events"] != ([4250, 4500] if step == 4500 else []) or
            state["source_stopped_at"] != 4000 or state["source_legacy_stop_events"] != [3750, 4000] or
            state["limit"] != 4500 or state["smoke"] is not False or state["deployment_authorized"] is not False or
            not exact(state["binding"], t.binding(p.DEFAULT_APPROVAL)) or not exact(receipt["binding"], state["binding"])):
            raise ValueError("Full checkpoint source/sampler/stops/authority mismatch")
        for key, expected in (("arm_roles", p.ROLES), ("arm_kill_bands", p.KILLS), ("arm_lr_scales", p.LR_SCALES),
                              ("arm_lambdas", p.LAMBDAS), ("arm_instrumental_weights", p.WEIGHTS), ("arm_accompaniment_weights", p.CA_WEIGHTS)):
            if not exact(state[key], expected):
                raise ValueError("Changed original branch/loss identity")
        states[step] = state
        bindings.update({str(path.resolve()): acq.sha256(path) for path in (receipt_path, checkpoint)})
    if tree_digest(states[4000]["arms"][t.ARMS[0]]) != tree_digest(states[4000]["arms"][t.ARMS[1]]):
        raise ValueError("Common baseline full states differ")
    for name, step, arm in ((MODELS[0], 4000, t.ARMS[0]), (MODELS[1], 4500, t.ARMS[0]), (MODELS[2], 4500, t.ARMS[1])):
        state, saved = states[step], states[step]["arms"][arm]
        if saved["updates"] != step or saved["optimizer"]["param_groups"][0]["lr"] != m.learning_rate(step, state["schedule"]["config"])*(1. if step == 4000 else p.LR_SCALES[arm]):
            raise ValueError("Actual original exposure/group LR identity differs")
        _, _, direction = stored_direction(saved)
        models[name] = saved
        descriptors[name] = {"step": step, "arm": arm, "checkpoint_sha256": CHECKPOINT_SHA[step],
            "source_saved_sha256": tree_digest(saved), "state_sha256": tree_digest(state), "source_modes": saved["modes"],
            "source_runtime": state["runtime"], "sampler": state["sampler"], "schedule_sha256": tree_digest(state["schedule"]),
            "rng_sha256": tree_digest(state["rng"]), "direction": direction}
    files = module_files() | {Path(__file__).resolve(), TEST, PROTOCOL, p.DEFAULT_APPROVAL,
        ROOT / "reports/36_weak_weight_authorization.md", ROOT / "reports/85_mel_lr_scale_trial_plan.md",
        ROOT / "reports/86_mel_lr_scale_authorization.md", ROOT / "reports/84_train_lf_cross_boundary_result.md"}
    bindings |= {str(path.resolve()): acq.sha256(path) for path in files}
    p.q.check_bindings(bindings)
    return authority, models, descriptors, states[4500]["sampler"], bindings


def check_gate(path):
    gate = acq.read_sealed(path)
    log = Path(gate["log"]).read_text(encoding="utf-8-sig")
    if (type(gate["actual_exit_code"]) is not int or gate["actual_exit_code"] != 0 or gate["draft_reviewed"] is not True or
        type(gate["tests_passed"]) is not int or gate["tests_passed"] < 30 or gate["tool_sha256"] != acq.sha256(__file__) or gate["test_sha256"] != acq.sha256(TEST) or
        gate["protocol_sha256"] != acq.sha256(PROTOCOL) or gate["log_sha256"] != acq.sha256(gate["log"]) or
        not re.search(r"Ran "+str(gate["tests_passed"])+r" tests? in", log) or not re.search(r"(?m)^OK\s*$", log)):
        raise ValueError("All actual units/draft review before prepare required")
    return gate


def fresh(out):
    if out.exists():
        raise ValueError("Existing output/evidence: preserve, never overwrite/restart")


def prepare(out, unit_evidence):
    fresh(out); no_active_worker(); resources()
    gate = check_gate(unit_evidence)
    if not exact(json.loads(PROTOCOL.read_text(encoding="utf-8")), scope()):
        raise ValueError("Exact predeclared new diagnostic protocol required")
    _, _, descriptors, sampler, bindings = source_evidence()
    _, _, matrix = matrix_identity()
    torch.set_num_threads(2)
    with d.deterministic_runtime("cpu"):
        runtime = d.runtime_identity("cpu")
    bindings |= {str(Path(unit_evidence).resolve()): acq.sha256(unit_evidence), gate["log"]: gate["log_sha256"]}
    plan = scope() | {"bindings_sha256": bindings, "model_descriptors": descriptors, "source_sampler": sampler,
        "matrix": matrix, "expected_runtime": runtime, "unit_gate": str(Path(unit_evidence).resolve()),
        "source_mode_policy": "preserve_all_saved_modes_in_cpu_diagnostic_copy_no_BN_dropout"}
    check_scope(plan)
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    print("ADAM_MEMORY PLAN SEALED;3 unique TRAIN inputs9 model-batches; updates0", flush=True)


def verified_plan(out):
    plan = acq.read_sealed(out / "plan.json")
    check_scope(plan); check_gate(Path(plan["unit_gate"]))
    p.q.check_bindings(plan["bindings_sha256"])
    _, _, matrix = matrix_identity()
    if not exact(matrix, plan["matrix"]):
        raise ValueError("Matrix identity changed")
    return plan


def status(out, values):
    temp = out / f"adam_status_{os.getpid()}.tmp"
    acq.write_new_json(temp, values | {"purpose": PURPOSE, "pid": os.getpid(), "updated_utc": m.bulk.now(),
        "primary_slot_forwards": values["completed_model_batches"]*6, "reference_slot_forwards": values.get("completed_reference_slots", 0),
        "model_updates": 0, "cuda_used": False, "limit": 9, "deployment": False, "release_selection": "NONE"})
    os.replace(temp, out / "run_status.json")


def summary(rows, inputs):
    buckets = {}
    for entry in inputs:
        for meta in entry["metadata"]:
            key = bucket(meta); buckets[key] = buckets.get(key, 0)+1
    per_model = {}
    for name in MODELS:
        selected = [row for row in rows if row["model"] == name]
        per_model[name] = {}
        for group in GROUPS:
            docs = [row["group_alignment"][group] for row in selected]
            available = [doc for doc in docs if doc["cos_g_u"] is not None]
            per_model[name][group] = {"model_batches": len(docs), "available": len(available), "unavailable": len(docs)-len(available),
                "positive_g_dot_u": sum(doc["g_dot_u"] > 0 for doc in available),
                "negative_g_dot_u": sum(doc["g_dot_u"] < 0 for doc in available),
                "mean_cos_g_u": sum(doc["cos_g_u"] for doc in available)/len(available) if available else None,
                "mean_g_dot_d": sum(doc["g_dot_d"] for doc in docs)/len(docs) if docs else None}
    return {"unique_slot_buckets": buckets, "per_model": per_model,
        "not_independent_model_samples": True, "no_quality_or_beta_reset_inference": True}


def run(out):
    if any((out / name).exists() for name in ("run_status.json", "rows", "inputs.json", "diagnostic.json")):
        raise ValueError("Existing run/evidence: diagnose or verify, never repeat")
    no_active_worker(); resources(); plan = verified_plan(out)
    authority, models, descriptors, sampler, _ = source_evidence()
    if not exact(descriptors, plan["model_descriptors"]) or not exact(sampler, plan["source_sampler"]):
        raise ValueError("Changed source full state")
    stamps = binding_stamps(list(plan["bindings_sha256"])+[out / "plan.json"])
    torch.set_num_threads(2)
    rows, inputs, reference_count = [], [], 0
    with m.bulk.worker_lock(out), d.deterministic_runtime("cpu"):
        if not exact(d.runtime_identity("cpu"), plan["expected_runtime"]):
            raise ValueError("Strict CPU FP32 runtime changed")
        status(out, {"status": "running", "phase": "initialization_and_collect", "completed_model_batches": 0, "error": None})
        try:
            dataset = p.inp.ApprovedTeacherDataset(p.inp.verified_approval(Path(authority["origin_approval"])), "kim_melband")
            true = m.LockedTruePool(m.bulk.OLD_LOCK, dataset.config)
            if sampler["cursor"] != 4500 or sampler["seed"] != dataset.seed or sampler["true_lock_sha256"] != true.bound:
                raise ValueError("Actual next cursor/seed/lock differs")
            rng_before = tree_digest(m.capture_rng("cpu"))
            batches = [collect_next(dataset, true, cursor) for cursor in CURSORS]
            unchanged_bindings(stamps)
            inputs = [input_entry(batch) for batch in batches]
            if rng_before != tree_digest(m.capture_rng("cpu")):
                raise ValueError("Input collection altered RNG")
            acq.write_new_json(out / "inputs.json", acq.seal({"purpose": PURPOSE, "inputs": inputs, "rng_unchanged": True}))
            wa, gs, _ = matrix_identity()
            (out / "rows").mkdir()
            for name in MODELS:
                saved = models[name]
                factory_rng = d.portable(m.capture_rng("cpu"))
                try:
                    net = m.frozen_factory(authority["source_protocol"])()
                finally:
                    m.restore_rng(factory_rng, "cpu")
                net.load_state_dict(saved["model"], strict=True)
                if len(saved["modes"]) != len(list(net.modules())) or any(type(flag) is not bool for flag in saved["modes"]):
                    raise ValueError("Complete source modes required")
                for module, flag in zip(net.modules(), saved["modes"]):
                    module.training = flag
                if tree_digest(dict(net.state_dict())) != tree_digest(dict(saved["model"])):
                    raise ValueError("Copied model identity differs")
                for index, batch in enumerate(batches):
                    resources(); unchanged_bindings(stamps)
                    probe = parameter_probe(net, saved, batch, wa, gs, index == 0)
                    unchanged_bindings(stamps)
                    row = acq.seal({"model": name, "counter": CURSORS[index], "model_descriptor": descriptors[name], **probe})
                    acq.write_new_json(out / "rows" / f"row_{len(rows):02d}.json", row)
                    rows.append(row); reference_count += probe["reference_slot_forwards"]
                    status(out, {"status": "running", "phase": "cpu_autograd_alignment", "model": name, "counter": CURSORS[index],
                        "completed_model_batches": len(rows), "completed_reference_slots": reference_count, "error": None})
                    print(f"ADAM_MEMORY model={name} counter={CURSORS[index]} completed={len(rows)}/9 reference_slots={reference_count}", flush=True)
                del net
            unchanged_bindings(stamps)
            p.q.check_bindings(plan["bindings_sha256"])
            result = scope() | {"plan_sha256": acq.sha256(out / "plan.json"), "inputs_sha256": acq.sha256(out / "inputs.json"),
                "rows": rows, "inputs": inputs, "model_descriptors": descriptors,
                "runtime": d.runtime_identity("cpu"), "summary": summary(rows, inputs), "error": None,
                "weights_modes_grad_rng_moments_inputs_unchanged": True, "existing_grad_storage_written": False}
            validate_result(result, plan)
            acq.write_new_json(out / "diagnostic.json", acq.seal(result))
            status(out, {"status": "complete", "phase": "complete_requires_external_verify", "completed_model_batches": 9,
                "completed_reference_slots": 18, "error": None})
            print("ADAM_MEMORY COMPLETE9 model-batches54 primary+18 reference slots; updates0 CUDAfalse", flush=True)
        except BaseException as error:
            status(out, {"status": "failed", "phase": "cpu_autograd_alignment", "completed_model_batches": len(rows),
                "completed_reference_slots": reference_count, "error": repr(error)})
            raise


def validate_result(result, plan):
    check_scope(result)
    if (len(result["rows"]) != 9 or len(result["inputs"]) != 3 or result["error"] is not None or
        result["weights_modes_grad_rng_moments_inputs_unchanged"] is not True or result["existing_grad_storage_written"] is not False or
        not exact(result["model_descriptors"], plan["model_descriptors"]) or not exact(result["runtime"], plan["expected_runtime"]) or
        not d.finite_state(result) or torch.cuda.is_initialized()):
        raise ValueError("Incomplete scope/state/runtime/result")
    for index, entry in enumerate(result["inputs"]):
        if (type(entry["counter"]) is not int or entry["counter"] != CURSORS[index] or
            len(entry["input_sha256"]) != 6 or len(entry["target_sha256"]) != 6 or
            entry["input_sha256"] != [meta["input_pcm_sha256"] for meta in entry["metadata"]]):
            raise ValueError("Changed fixed input/target/metadata coverage")
        for meta in entry["metadata"]:
            bucket(meta)
            if (meta["score_start"], meta["score_end"]) != (25088, 89344):
                raise ValueError("Changed support")
    primary, reference = 0, 0
    for i, row in enumerate(result["rows"]):
        name, index = MODELS[i//3], i % 3
        if (row["model"] != name or type(row["counter"]) is not int or row["counter"] != CURSORS[index] or
            not exact(row["input"], result["inputs"][index]) or len(row["slots"]) != 6 or
            not exact(row["model_descriptor"], plan["model_descriptors"][name]) or
            not exact(row["direction"], plan["model_descriptors"][name]["direction"]) or
            row["weights_modes_grad_rng_moments_inputs_unchanged"] is not True or row["existing_grad_storage_written"] is not False):
            raise ValueError("Changed model/shared input/direction/slot identity")
        t.validate_metadata(row["input"]["metadata"])
        if set(row["group_alignment"]) != set(GROUPS) or set(row["weighted_alignment"]) != set(GROUPS):
            raise ValueError("Missing original components")
        for doc in list(row["group_alignment"].values())+list(row["weighted_alignment"].values())+[row["combined_alignment"]]:
            validate_alignment(doc)
            if (doc["u_l2"], doc["d_l2"]) != (row["direction"]["u_l2"], row["direction"]["d_l2"]):
                raise ValueError("Stored direction norm mismatch")
            # FP32 d=-LR*u rounds elementwise, hence an explicitly bounded
            # detached arithmetic check, not a claim of FP64 bit identity.
            expected = -row["direction"]["param_groups"]["lr"]*doc["g_dot_u"]
            if not math.isclose(doc["g_dot_d"], expected, rel_tol=1e-6,
                                abs_tol=2e-7*doc["g_l2"]*doc["d_l2"]+1e-15):
                raise ValueError("Stored LR/sign dot arithmetic differs")
        combined_dot = sum(c*row["group_alignment"][g]["g_dot_d"] for g,c in zip(GROUPS, COEFFICIENTS))
        error_scale = sum(c*row["group_alignment"][g]["g_l2"] for g,c in zip(GROUPS, COEFFICIENTS))*row["direction"]["d_l2"]
        if not math.isclose(row["combined_alignment"]["g_dot_d"], combined_dot, rel_tol=1e-6, abs_tol=5e-7*error_scale+1e-15):
            raise ValueError("Original coefficient/combined dot arithmetic differs")
        for j, slot in enumerate(row["slots"]):
            meta = row["input"]["metadata"][j]
            if (slot["slot"] != j or slot["group"] != group_for_slot(j, meta) or slot["bucket"] != bucket(meta) or
                not exact(slot["metadata"], meta) or slot["scalar_split_bit_exact"] is not True):
                raise ValueError("Changed original role/gain/split")
            for doc in slot["alignment"].values():
                validate_alignment(doc)
            equal = slot["split_gradient_equivalence"]
            if not 0 <= equal["max_error"] <= equal["error_bound"]:
                raise ValueError("Split gradient tolerance missing")
            active = slot["auxiliary"]["active"]
            if type(active) is not bool or (active and j >= 2) or (not active and any(slot[key] != 0. for key in ("cv2_div6", "ca1_sq_div6", "full_aux_div6"))):
                raise ValueError("Activity/zero-reference skip mismatch")
            if not math.isclose(slot["cv2_div6"]+slot["ca1_sq_div6"], slot["full_aux_div6"], rel_tol=2e-7, abs_tol=1e-10):
                raise ValueError("Split scalar arithmetic differs")
            if not active and any(slot["alignment"][g]["g_l2"] != 0. for g in ("cv2", "ca1_sq")):
                raise ValueError("Skipped auxiliary gradient nonzero")
        expected_ref = 6 if index == 0 else 0
        if not exact(row["primary_slot_forwards"], 6) or not exact(row["reference_slot_forwards"], expected_ref):
            raise ValueError("True forward accounting differs")
        primary += row["primary_slot_forwards"]; reference += expected_ref
        combined = row["independent_combination"]
        if expected_ref:
            if (combined is None or not 0 <= combined["max_error"] <= combined["error_bound"] or
                combined["reference_forwards"] != 6 or combined["rtol"] != .0002 or combined["atol"] != .0000002 or combined["bit_exact_claimed"] is not False or
                len(combined["slot_checks"]) != 6 or any(v["original194_scalar_wave_bit_exact"] is not True for v in combined["slot_checks"])):
                raise ValueError("First-batch original194 gradient/scalar/wave check missing")
        elif combined is not None:
            raise ValueError("Unexpected extra reference forward")
    if (primary, reference) != (54, 18) or not exact(summary(result["rows"], result["inputs"]), result["summary"]):
        raise ValueError("Counting/aggregation changed")


def symmetric_rows(out, embedded):
    paths = sorted((out / "rows").glob("row_*.json"))
    if len(paths) != 9 or len(embedded) != 9:
        raise ValueError("Exactly9 separately committed rows required")
    for index, (path, inside) in enumerate(zip(paths, embedded)):
        outside = acq.read_sealed(path)
        if acq.content_digest(inside) != inside.get("content_sha256"):
            raise ValueError("Embedded row seal differs")
        if path.name != f"row_{index:02d}.json" or not exact(outside, inside):
            raise ValueError("Changed committed row/full symmetric typed comparison")


def verify(out):
    no_active_worker(); resources()
    plan = verified_plan(out)
    ending = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    expected = {"status": "complete", "completed_model_batches": 9, "primary_slot_forwards": 54,
                "reference_slot_forwards": 18, "model_updates": 0, "cuda_used": False, "error": None}
    if any(not exact(ending.get(key), value) for key, value in expected.items()):
        raise ValueError("Complete readonly run status required before independent verification")
    result = acq.read_sealed(out / "diagnostic.json")
    if result["plan_sha256"] != acq.sha256(out / "plan.json") or result["inputs_sha256"] != acq.sha256(out / "inputs.json"):
        raise ValueError("Changed plan/input hashes")
    input_doc = acq.read_sealed(out / "inputs.json")
    if not exact(input_doc["inputs"], result["inputs"]) or input_doc["rng_unchanged"] is not True:
        raise ValueError("Committed input differs")
    symmetric_rows(out, result["rows"])
    validate_result(result, plan)
    path = out / "verification.json"
    if path.exists():
        raise ValueError("Existing independent verification; do not repeat")
    acq.write_new_json(path, acq.seal({"purpose": PURPOSE, "diagnostic_sha256": acq.sha256(out / "diagnostic.json"),
        "plan_sha256": acq.sha256(out / "plan.json"), "verified_model_batches": 9, "primary_slots": 54, "reference_slots": 18,
        "file_and_embedded_rows_sealed_symmetric_type_sensitive": True, "model_forward_count": 0,
        "optimizer_constructed": False, "cuda_used": False, "model_updates": 0, "independently_verified": True}))
    print("ADAM_MEMORY VERIFIED9/54+18; no forward/optimizer/update; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--unit-evidence", type=Path, default=MONITOR / "unit_gate.json")
    args = parser.parse_args()
    prepare(args.out, args.unit_evidence) if args.operation == "prepare" else globals()[args.operation](args.out)
