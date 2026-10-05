"""Fixed TRAIN reference-only LF/DC decomposition. No model load or forward."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch

spec = importlib.util.spec_from_file_location("sealed_context183", Path(__file__).with_name("183_verify_train_context_reconstruction.py"))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
k, q, acq, ROOT = v.k, v.q, v.acq, v.ROOT
m, p, t = k.m, k.p, k.t
PURPOSE = "NONRELEASE_TRAIN_LOW_FREQUENCY_REFERENCE_DIAGNOSTIC"
DEFAULT_OUT = ROOT / "results/train_low_frequency_reference_20261003"
MONITOR = ROOT / "results/train_low_frequency_reference_monitor_20261003"
TEST = Path(__file__).with_name("_test_train_low_frequency_reference.py")
REPORT = ROOT / "reports/70_train_low_frequency_reference_plan.md"
CURSORS = tuple(range(3000, 3012))
FRAMES, SAMPLES, THREADS = 352, 89856, 2
REGIONS = {"native": (98, 350), "common_descriptive_only": (130, 350)}
PARTITIONS = {"windowed_dc_bin0": (0, 1), "other_forced_zero_bins1_5": (1, 6), "retained_bins6_512": (6, 513)}
ENERGY_RTOL, ENERGY_ATOL, FRACTION_ATOL = 1e-12, 1e-10, 5e-12
PINNED = {
    "reports/69_train_context_reconstruction_result.md": "38b78491145494e1742bbc4f5159dba2ad8e6ea1a15ad5a4ae459db5646a5385",
    "reports/70_train_low_frequency_reference_plan.md": "af899ab6daed389692f71c35d58e3accb31d7aab1023979ed6d85d4b1c42e9ff",
    "results/train_context_reconstruction_recovery_monitor_20261003/completion_review.json": "e92bb74d09a8a2fa4c851578669591a2bf01e0b61abef8f0a91c05bf98388922",
    "results/train_context_reconstruction_recovery_monitor_20261003/aggregation.json": "969777bd172f68402a36c5a03ba4f869e57dd75d76f6ea35dd6816d9c9bc70cd",
    "results/train_context_reconstruction_recovery_monitor_20261003/input_equivalence_audit.json": "67fdffa9e015d91512ad0d517430de6f73da9e7ac522d5909ee56c0a2711bf8f",
    "results/train_context_reconstruction_recovery_20261003/plan.json": "0867a9c646fe1d5e83fc8d8146dcd40d195530829dc8f148fe33b4cc8df73f8b",
    "results/train_context_reconstruction_recovery_20261003/inputs.json": "43fb3d239534f285695cd5acdff8230393f965e8a81f7efba6c0e646c4ae6b84",
    "results/train_context_reconstruction_recovery_20261003/diagnostic.json": "75d70fabe35061871d05a91b233c4aaa20951b45a7cf2b2e2c05e61529d01e1f",
    "results/train_context_reconstruction_review_20261003/plan.json": "034e0c358acd4dc30ee8f5b13dcbb3609923262e89d615455d082bfaca900c31",
    "results/train_context_reconstruction_review_20261003/verification.json": "39e401f08e72508fb1282beeeef4a7996aa44a0d5151bdcc6662cdb9049289c7",
}


def canonical(doc):
    return v.canonical(doc)


def scope():
    return {"schema": 1, "purpose": PURPOSE, "cursors": list(CURSORS), "reference_batches": 12, "slot_checks": 72,
        "samples": SAMPLES, "frames": FRAMES, "threads": THREADS, "fft": 1024, "hop": 256, "sample_rate": 44100,
        "stft": "original09_periodic_Hann_centerTrue_reflect_onesided_unnormalized",
        "regions": {name: list(region) for name, region in REGIONS.items()},
        "partitions": {name: list(region) for name, region in PARTITIONS.items()},
        "forced_zero_fft_bins": list(range(6)), "lf_kill_bands": 44, "activity_rms_threshold": .0001,
        "demeaning": "isolated_target_copy_channelwise_whole_crop_FP32_mean",
        "energy_rtol": ENERGY_RTOL, "energy_atol": ENERGY_ATOL, "old_fraction_atol": FRACTION_ATOL,
        "reference_processing_dtype": "float32", "spectrum_dtype": "complex64", "detached_statistics_dtype": "float64",
        "model_loaded": False, "model_forward_used": False, "backward_used": False, "optimizer_constructed": False,
        "model_updates": 0, "cuda_used": False, "training_authorized": False, "deployment": False, "release_selection": "NONE"}


def check_scope(doc):
    if any(canonical(doc.get(key)) != canonical(value) for key, value in scope().items()):
        raise ValueError("Changed fixed reference-only diagnostic scope")


def no_active_worker():
    k.no_active_worker()
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '(183_verify_train_context_reconstruction|184_diagnose_train_low_frequency_reference)[.]py[\" ]+(prepare|run|verify)' } | Select-Object -ExpandProperty ProcessId"
    reply = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], check=True, capture_output=True, text=True)
    if any(int(line.strip()) not in {os.getpid(), os.getppid()} for line in reply.stdout.splitlines() if line.strip()):
        raise ValueError("Active diagnostic/verifier; preserve and postpone")


def source_evidence():
    """Use completed, independently verified evidence, never rerun182/183 or load PT."""
    if torch.cuda.is_initialized():
        raise ValueError("Fresh CPU-only process required")
    pinned = {str(ROOT / name): digest for name, digest in PINNED.items()}
    q.check_bindings(pinned)
    proof = json.loads((ROOT / "results/train_context_reconstruction_recovery_monitor_20261003/completion_review.json").read_text(encoding="utf-8"))
    if (canonical(proof["actual_run_exit_code"]) != "0" or proof["foreground_run_session"] != 14585 or
        proof["independently_verified"] is not True or proof["no_active_worker_or_shim"] is not True or proof["processes"] or
        proof["terminal_status"]["status"] != "complete" or proof["terminal_status"]["completed_model_batches"] != 36 or
        proof["terminal_status"]["error"] is not None or proof["model_updates"] != 0 or proof["cuda_used"] is not False or
        proof["initial182_verify"]["actual_exit_code"] != 1 or proof["independent183_verifier"]["verify_actual_exit_code"] != 0):
        raise ValueError("Real complete run and independently recovered verification required")
    review_plan = acq.read_sealed(v.DEFAULT_OUT / "plan.json")
    review = acq.read_sealed(v.DEFAULT_OUT / "verification.json")
    v.check_plan(review_plan); v.check_plan(review)
    if (review["plan_sha256"] != acq.sha256(v.DEFAULT_OUT / "plan.json") or review["independently_verified"] is not True or
        review["symmetric_rows_verified"] != 36 or review["original_all_scope_identity_accounting_and_summary_gates_passed"] is not True):
        raise ValueError("Exact independently verified36/216 reference provenance required")
    source_plan = acq.read_sealed(k.DEFAULT_OUT / "plan.json")
    k.check_plan(source_plan)
    inputs = acq.read_sealed(k.DEFAULT_OUT / "inputs.json")
    audit = acq.read_sealed(k.AUDIT)
    result = acq.read_sealed(k.DEFAULT_OUT / "diagnostic.json")
    if (canonical(inputs["inputs"]) != canonical(audit["inputs"]) or canonical(result["inputs"]) != canonical(inputs["inputs"]) or
        [entry["counter"] for entry in inputs["inputs"]] != list(CURSORS) or len(result["rows"]) != 36 or
        result["plan_sha256"] != acq.sha256(k.DEFAULT_OUT / "plan.json") or
        result["inputs_sha256"] != acq.sha256(k.DEFAULT_OUT / "inputs.json") or
        review["source_diagnostic_sha256"] != acq.sha256(k.DEFAULT_OUT / "diagnostic.json")):
        raise ValueError("Changed fixed canonical input evidence")
    # These bindings already include all historical approvals, full states,
    # locks and actual imported code. Hash verification does not instantiate them.
    bindings = source_plan["bindings_sha256"] | review_plan["bindings_sha256"] | pinned
    bindings |= {str(path.resolve()): acq.sha256(path) for path in (Path(__file__), TEST, Path(v.__file__))}
    q.check_bindings(bindings)
    approval = acq.read_sealed(p.DEFAULT_APPROVAL)
    p.check_approval(approval)
    prior = result["rows"][:12]
    if any(row["counter"] != cursor or row["model"] != q.MODELS[0] for row, cursor in zip(prior, CURSORS)):
        raise ValueError("Original reference rows must follow fixed counters")
    return approval, inputs["inputs"], prior, bindings


def original_geometry():
    gs = torch.from_numpy(q.core.t09.make_synthesis_matrix())
    forced, geometry = q.lf_geometry(gs)
    if forced.nonzero().flatten().tolist() != list(range(6)):
        raise ValueError("LF44 forced zero geometry changed")
    return geometry


def checked_reference(value):
    q.cpu_float(value, (2, SAMPLES))
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError("Detached reference required; no backward scope")


def demean_copy(value):
    checked_reference(value)
    means = value.mean(-1, keepdim=True)
    copy = value.clone() - means
    q.cpu_float(copy, (2, SAMPLES))
    return copy, means


def energy_close(a, b):
    return math.isclose(a, b, rel_tol=ENERGY_RTOL, abs_tol=ENERGY_ATOL)


def validate_energy(doc):
    channels = doc["per_channel_bin_energy"]
    if len(channels) != 2 or any(len(row) != 513 for row in channels) or len(doc["per_bin_energy"]) != 513:
        raise ValueError("Two channels513 bins required")
    values = [x for row in channels for x in row] + doc["per_bin_energy"] + [doc["total_energy"]]
    if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Finite nonnegative energy required")
    if any(not energy_close(a+b, c) for a, b, c in zip(*channels, doc["per_bin_energy"])) or not energy_close(math.fsum(doc["per_bin_energy"]), doc["total_energy"]):
        raise ValueError("Bin/channel energy conservation failed")
    expected = {name: math.fsum(doc["per_bin_energy"][start:end]) for name, (start, end) in PARTITIONS.items()}
    if set(doc["partition_energy"]) != set(expected) or any(not energy_close(value, doc["partition_energy"][name]) for name, value in expected.items()) or not energy_close(math.fsum(expected.values()), doc["total_energy"]):
        raise ValueError("Partition conservation failed")
    total = doc["total_energy"]
    fractions = doc["partition_fraction"]
    if set(fractions) != set(expected) or len(doc["per_bin_fraction"]) != 513:
        raise ValueError("Exact fraction layout required")
    if total == 0:
        if doc["fraction_skip"] != "exact_zero_spectral_reference" or any(x is not None for x in [*fractions.values(), *doc["per_bin_fraction"], doc["forced_zero_fraction"]]):
            raise ValueError("Zero reference must skip fractions, no epsilon")
    else:
        if doc["fraction_skip"] is not None:
            raise ValueError("Positive reference must retain fraction")
        for name, value in expected.items():
            actual = fractions[name]
            if type(actual) not in (float, int) or not math.isfinite(actual) or not 0 <= actual <= 1 or abs(actual-value/total) > FRACTION_ATOL:
                raise ValueError("Changed partition fraction")
        if any(type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1 or abs(x-e/total) > FRACTION_ATOL for x, e in zip(doc["per_bin_fraction"], doc["per_bin_energy"])):
            raise ValueError("Changed per-bin fraction")
        force = (expected["windowed_dc_bin0"]+expected["other_forced_zero_bins1_5"])/total
        if not math.isclose(doc["forced_zero_fraction"], force, abs_tol=FRACTION_ATOL, rel_tol=0):
            raise ValueError("Changed forced zero fraction")


def energy(spectrum, region):
    if (not isinstance(spectrum, torch.Tensor) or spectrum.device.type != "cpu" or spectrum.dtype != torch.complex64 or
        tuple(spectrum.shape) != (2, 513, FRAMES) or spectrum.requires_grad or not torch.isfinite(spectrum).all()):
        raise ValueError("Finite detached CPU complex64 original spectrum required")
    if type(region) is not tuple or region not in tuple(REGIONS.values()) or any(type(x) is not int for x in region):
        raise ValueError("Only exact predeclared frame supports")
    weights = torch.full((513,), 2., dtype=torch.float64); weights[[0, -1]] = 1
    power = spectrum[..., region[0]:region[1]].detach().to(torch.complex128).abs().square()*weights[None, :, None]
    channels = power.sum(-1).tolist()
    bins = power.sum((0, 2)).tolist()
    total = float(power.sum())
    partitions = {name: math.fsum(bins[start:end]) for name, (start, end) in PARTITIONS.items()}
    doc = {"frame_interval": list(region), "per_channel_bin_energy": channels, "per_bin_energy": bins,
        "total_energy": total, "partition_energy": partitions,
        "partition_fraction": {name: value/total if total > 0 else None for name, value in partitions.items()},
        "per_bin_fraction": [value/total if total > 0 else None for value in bins],
        "forced_zero_fraction": (partitions["windowed_dc_bin0"]+partitions["other_forced_zero_bins1_5"])/total if total > 0 else None,
        "fraction_skip": None if total > 0 else "exact_zero_spectral_reference"}
    validate_energy(doc)
    return doc


def moments(value):
    checked_reference(value)
    result = {}
    for name, region in {"whole": (0, SAMPLES), "native": (98*256, SAMPLES-512), "common_descriptive_only": (130*256, SAMPLES-512)}.items():
        selected = value[:, region[0]:region[1]].detach().double()
        rms = float(selected.square().mean().sqrt())
        result[name] = {"sample_interval": list(region), "mean_per_channel": selected.mean(-1).tolist(),
            "mean_combined": float(selected.mean()), "rms_per_channel": selected.square().mean(-1).sqrt().tolist(),
            "rms_combined": rms, "exact_zero": bool(torch.count_nonzero(selected) == 0), "active_rms_gt_1e_4": rms > .0001}
    return result


def validate_moments(doc):
    intervals = {"whole": [0, SAMPLES], "native": [98*256, SAMPLES-512], "common_descriptive_only": [130*256, SAMPLES-512]}
    if set(doc) != set(intervals):
        raise ValueError("Exact moment supports required")
    for name, region in intervals.items():
        part = doc[name]
        if (canonical(part["sample_interval"]) != canonical(region) or
            len(part["mean_per_channel"]) != 2 or len(part["rms_per_channel"]) != 2 or
            type(part["exact_zero"]) is not bool or type(part["active_rms_gt_1e_4"]) is not bool):
            raise ValueError("Changed moment support/channel layout")
        means, rms = part["mean_per_channel"], part["rms_per_channel"]
        if any(type(x) not in (float, int) or not math.isfinite(x) for x in [*means, *rms, part["mean_combined"], part["rms_combined"]]):
            raise ValueError("Finite moments required")
        if (any(x < 0 for x in rms) or part["rms_combined"] < 0 or
            any(abs(a) > b+1e-12 for a,b in zip(means,rms)) or
            not math.isclose(part["mean_combined"], math.fsum(means)/2, rel_tol=1e-12, abs_tol=1e-12) or
            not math.isclose(part["rms_combined"]**2, math.fsum(x*x for x in rms)/2, rel_tol=1e-12, abs_tol=1e-15) or
            part["active_rms_gt_1e_4"] != (part["rms_combined"] > .0001) or
            part["exact_zero"] != (part["rms_combined"] == 0)):
            raise ValueError("Changed moment accounting/activity")


def check_old_native(doc, previous):
    if not energy_close(doc["total_energy"], previous["reference_spectral_energy"]):
        raise ValueError("Original reference total energy changed")
    actual, old = doc["forced_zero_fraction"], previous["reference_forced_zero_energy_fraction"]
    if old is None:
        if actual is not None or doc["total_energy"] != 0 or previous["reference_fraction_skip"] != "zero_or_near_zero_reference":
            raise ValueError("Old skipped reference is not exact zero")
    elif actual is None or abs(actual-old) > FRACTION_ATOL:
        raise ValueError("Native LF fraction differs from original reviewed reference")


@torch.no_grad()
def reference_probe(value, previous):
    checked_reference(value)
    before = value.clone()
    raw = q.core.stft_batch(value)
    altered, means = demean_copy(value)
    changed = q.core.stft_batch(altered)
    result = {"original": {name: energy(raw, region) for name, region in REGIONS.items()},
        "demeaned_copy_descriptive_only": {name: energy(changed, region) for name, region in REGIONS.items()},
        "original_moments": moments(value), "demeaned_copy_moments": moments(altered),
        "subtracted_channel_means_FP32": means.flatten().tolist(),
        "original_reference_unchanged": torch.equal(value, before),
        "demeaned_copy_is_new_truth": False, "waveform_quality_floor_claimed": False}
    check_old_native(result["original"]["native"], previous)
    if not result["original_reference_unchanged"] or not q.d.finite_state(result):
        raise ValueError("Nonfinite or changed original reference")
    return result


def group_summary(rows):
    slots = [slot for row in rows for slot in row["slots"]]
    grouped = {}
    for slot in slots:
        meta = slot["metadata"]
        active = slot["probe"]["original_moments"]["native"]["active_rms_gt_1e_4"]
        key = q.bucket(meta)+("/active" if active else "/low_activity")
        grouped.setdefault(key, []).append(slot)
    summary = {}
    for key, selected in sorted(grouped.items()):
        parts = {}
        for version in ("original", "demeaned_copy_descriptive_only"):
            parts[version] = {}
            for region in REGIONS:
                docs = [slot["probe"][version][region] for slot in selected]
                fractions = [doc["forced_zero_fraction"] for doc in docs if doc["forced_zero_fraction"] is not None]
                parts[version][region] = {"count": len(fractions), "skip_count": len(docs)-len(fractions),
                    "forced_zero_fraction_mean": statistics.fmean(fractions) if fractions else None,
                    "forced_zero_fraction_min": min(fractions) if fractions else None,
                    "forced_zero_fraction_max": max(fractions) if fractions else None,
                    "partition_fraction_means": {part: statistics.fmean(doc["partition_fraction"][part] for doc in docs if doc["fraction_skip"] is None) if fractions else None for part in PARTITIONS}}
        summary[key] = {"slots": len(selected), "pseudo_reference_is_final_truth": False, "statistics": parts}
    return summary


def validate_result(plan, result, expected_inputs, prior):
    check_scope(result)
    if any(canonical(result[name]) != canonical(plan[name]) for name in ("geometry", "expected_inputs", "expected_runtime", "input_reader")):
        raise ValueError("Changed reference identity/support/runtime")
    if canonical(result["bindings_sha256"]) != canonical(plan["bindings_sha256"]) or canonical(result["inputs"]) != canonical(expected_inputs):
        raise ValueError("Changed plan bindings or canonical TRAIN inputs")
    if (len(result["rows"]) != 12 or set(result["row_sha256"]) != {f"row_{i:02d}.json" for i in range(12)} or
        result["cpu_rng_unchanged"] is not True or result["input_reference_unchanged"] is not True or
        canonical(result["runtime"]) != canonical(plan["expected_runtime"])):
        raise ValueError("Complete CPU-only12 rows with unchanged inputs/RNG required")
    for index, row in enumerate(result["rows"]):
        if type(row["counter"]) is not int or row["counter"] != CURSORS[index] or len(row["slots"]) != 6 or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
            raise ValueError("Changed row order/accounting")
        for j, slot in enumerate(row["slots"]):
            if type(slot["slot"]) is not int or slot["slot"] != j or canonical(slot["metadata"]) != canonical(expected_inputs[index]["metadata"][j]):
                raise ValueError("Changed original TRAIN role/slot/metadata")
            probe = slot["probe"]
            if probe["original_reference_unchanged"] is not True or probe["demeaned_copy_is_new_truth"] is not False or probe["waveform_quality_floor_claimed"] is not False:
                raise ValueError("Reference-only interpretation required")
            validate_moments(probe["original_moments"]); validate_moments(probe["demeaned_copy_moments"])
            if len(probe["subtracted_channel_means_FP32"]) != 2 or any(type(x) is not float or not math.isfinite(x) for x in probe["subtracted_channel_means_FP32"]):
                raise ValueError("Finite original FP32 channel means required")
            for version in ("original", "demeaned_copy_descriptive_only"):
                for name, region in REGIONS.items():
                    doc = probe[version][name]
                    if canonical(doc["frame_interval"]) != canonical(list(region)):
                        raise ValueError("Changed energy support")
                    validate_energy(doc)
            check_old_native(probe["original"]["native"], prior[index]["slots"][j]["probe"]["lf44"])
            if j == 2 and (probe["original"]["native"]["total_energy"] != 0 or not probe["original_moments"]["whole"]["exact_zero"]):
                raise ValueError("Instrumental TRAIN reference must be exactly zero")
    if canonical(result["summary"]) != canonical(group_summary(result["rows"])) or not q.d.finite_state(result):
        raise ValueError("Changed/nonfinite recomputed summary")


def prepare(out):
    out = q.fresh_output(out)
    no_active_worker(); q.check_disk()
    _, inputs, _, bindings = source_evidence()
    plan = scope() | {"bindings_sha256": bindings, "input_reader": "immutable182_explicit_fixed3000..3011",
        "geometry": original_geometry(), "expected_inputs": inputs,
        "expected_runtime": acq.read_sealed(k.DEFAULT_OUT / "diagnostic.json")["runtime"]}
    check_scope(plan)
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    print("REFERENCE PLAN SEALED12 batches72 slots; no model load/forward; CPU-only", flush=True)


def verified_plan(out):
    plan = acq.read_sealed(out / "plan.json")
    check_scope(plan); q.check_bindings(plan["bindings_sha256"])
    approval, inputs, prior, bindings = source_evidence()
    if (canonical(plan["bindings_sha256"]) != canonical(bindings) or canonical(plan["expected_inputs"]) != canonical(inputs) or
        canonical(plan["geometry"]) != canonical(original_geometry()) or plan["input_reader"] != "immutable182_explicit_fixed3000..3011" or
        canonical(plan["expected_runtime"]) != canonical(acq.read_sealed(k.DEFAULT_OUT / "diagnostic.json")["runtime"])):
        raise ValueError("Changed bound reference scope/input/geometry")
    return plan, approval, inputs, prior


def status(out, values):
    temporary = out / f"status_{os.getpid()}.tmp"
    acq.write_new_json(temporary, values | {"pid": os.getpid(), "updated_utc": m.bulk.now(), "purpose": PURPOSE,
        "model_loaded": False, "model_forward_used": False, "model_updates": 0, "cuda_used": False, "release_selection": "NONE"})
    os.replace(temporary, out / "run_status.json")


def run(out):
    q.reject_existing_run(out)
    no_active_worker(); q.check_disk()
    plan, approval, expected_inputs, prior = verified_plan(out)
    torch.set_num_threads(THREADS)
    before_rng = m.capture_rng("cpu")
    rows, inputs, hashes = [], [], {}
    with m.bulk.worker_lock(out), q.d.deterministic_runtime("cpu"), torch.no_grad():
        status(out, {"status": "running", "phase": "collect_reference_and_stft", "completed_reference_batches": 0, "limit": 12, "error": None})
        try:
            dataset = p.inp.ApprovedTeacherDataset(p.inp.verified_approval(Path(approval["origin_approval"])), "kim_melband")
            true = m.LockedTruePool(m.bulk.OLD_LOCK, dataset.config)
            for index, cursor in enumerate(CURSORS):
                q.check_disk(); started = time.perf_counter()
                batch = k.collect_fixed_train(dataset, true, cursor)
                entry = k.input_entry(batch, cursor)
                if canonical(entry) != canonical(expected_inputs[index]):
                    raise ValueError("Actual PCM/target/metadata differs from originally audited TRAIN")
                slots = [{"slot": j, "metadata": meta,
                    "probe": reference_probe(batch["v"][j], prior[index]["slots"][j]["probe"]["lf44"])} for j, meta in enumerate(batch["metadata"])]
                if canonical(k.input_entry(batch, cursor)) != canonical(entry) or not m.equal_state(before_rng, m.capture_rng("cpu")) or torch.cuda.is_initialized():
                    raise ValueError("Input/reference/RNG changed or CUDA initialized")
                row = {"counter": cursor, "slots": slots, "seconds": time.perf_counter()-started}
                name = f"row_{index:02d}.json"
                acq.write_new_json(out / name, acq.seal(row))
                hashes[name] = acq.sha256(out / name); rows.append(row); inputs.append(entry)
                status(out, {"status": "running", "phase": "reference_low_frequency_statistics", "completed_reference_batches": len(rows), "limit": 12, "error": None})
                print(f"REFERENCE counter={cursor} completed={len(rows)}/12 seconds={row['seconds']:.3f}", flush=True)
            acq.write_new_json(out / "inputs.json", acq.seal({"purpose": PURPOSE, "plan_sha256": acq.sha256(out / "plan.json"), "inputs": inputs}))
            result = {key: value for key, value in plan.items() if key != "content_sha256"} | {
                "plan_sha256": acq.sha256(out / "plan.json"), "inputs_sha256": acq.sha256(out / "inputs.json"),
                "inputs": inputs, "rows": rows, "row_sha256": hashes, "summary": group_summary(rows),
                "runtime": q.d.runtime_identity("cpu"), "cpu_rng_unchanged": m.equal_state(before_rng, m.capture_rng("cpu")),
                "input_reference_unchanged": True}
            validate_result(plan, result, expected_inputs, prior)
            if torch.cuda.is_initialized():
                raise ValueError("Unexpected CUDA")
            acq.write_new_json(out / "diagnostic.json", acq.seal(result))
            status(out, {"status": "complete", "phase": "reference_diagnostic_complete", "completed_reference_batches": 12, "limit": 12, "error": None})
            print("REFERENCE COMPLETE12 batches72 slots; no model/CUDA/updates; NONRELEASE", flush=True)
        except BaseException as error:
            status(out, {"status": "failed", "phase": "reference_low_frequency_statistics", "completed_reference_batches": len(rows), "limit": 12, "error": repr(error)})
            raise
        finally:
            m.restore_rng(before_rng, "cpu")


def verify(out):
    if (out / "verification.json").exists():
        raise ValueError("Existing verification: inspect, never repeat or overwrite")
    no_active_worker(); q.check_disk()
    plan, _, inputs, prior = verified_plan(out)
    terminal = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    if terminal["status"] != "complete" or terminal["completed_reference_batches"] != 12 or terminal["limit"] != 12 or terminal["error"] is not None:
        raise ValueError("Exact complete12 terminal required")
    result = acq.read_sealed(out / "diagnostic.json")
    evidence = acq.read_sealed(out / "inputs.json")
    if (result["plan_sha256"] != acq.sha256(out / "plan.json") or result["inputs_sha256"] != acq.sha256(out / "inputs.json") or
        evidence["purpose"] != PURPOSE or evidence["plan_sha256"] != result["plan_sha256"] or canonical(evidence["inputs"]) != canonical(inputs)):
        raise ValueError("Changed plan/input identity")
    validate_result(plan, result, inputs, prior)
    for index in range(12):
        name = f"row_{index:02d}.json"
        v.checked_row(acq.read_sealed(out / name), result["rows"][index], acq.sha256(out / name), result["row_sha256"][name])
    review = scope() | {"independently_verified": True, "symmetric_rows_verified": 12,
        "plan_sha256": result["plan_sha256"], "diagnostic_sha256": acq.sha256(out / "diagnostic.json"),
        "inputs_sha256": result["inputs_sha256"], "bindings_sha256": plan["bindings_sha256"], "summary": result["summary"]}
    acq.write_new_json(out / "verification.json", acq.seal(review))
    print("REFERENCE INDEPENDENTLY VERIFIED12 batches72 slots; symmetric rows; no new STFT/model/CUDA", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    globals()[args.operation](args.out)

