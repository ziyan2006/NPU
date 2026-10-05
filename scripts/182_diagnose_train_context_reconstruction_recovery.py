"""Explicit new zero-update recovery of181's fixed TRAIN reader scope failure.

Uses immutable181 computational/source checks, never181.run or patched globals.
New plan/directory bind the failed attempt and a real canonical input audit.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import torch

spec = importlib.util.spec_from_file_location("immutable_frontend181", Path(__file__).with_name("181_diagnose_train_context_reconstruction.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
m, p, t, acq, ROOT = q.m, q.p, q.t, q.acq, q.ROOT
TOOL = "182_fixed_train_reader_recovery"
DEFAULT_OUT = ROOT / "results/train_context_reconstruction_recovery_20261003"
MONITOR = ROOT / "results/train_context_reconstruction_recovery_monitor_20261003"
AUDIT = MONITOR / "input_equivalence_audit.json"
REPORT = ROOT / "reports/66_train_context_reconstruction_recovery.md"
TEST = Path(__file__).with_name("_test_train_context_reconstruction_recovery.py")
FAILURE = ROOT / "results/train_context_reconstruction_monitor_20261003/failure_review.json"


def collect_fixed_train(dataset, true, cursor):
    if type(cursor) is not int or cursor not in q.CURSORS:
        raise ValueError("Only declared TRAIN3000..3011, not a resume stream")
    xs, vs, metadata = [], [], []
    for domain in m.DOMAINS[:3]:
        item = true.crop(domain, dataset.seed, cursor)
        xs.append(item["x"]); vs.append(item["v"]); metadata.append(item["meta"])
    for index in range(3):
        recipe = m.data.crop_recipe(dataset.rows, dataset.config, dataset.seed, cursor*3+index)
        item = dataset.crop(recipe)
        xs.append(item["x"]); vs.append(item["v"])
        metadata.append(item["meta"] | {"domain": "pseudo", "role": "pseudo_label_train_candidate"})
    t.validate_metadata(metadata)
    x, v = torch.stack(xs), torch.stack(vs)
    q.cpu_float(x, (6, 2, q.SAMPLES)); q.cpu_float(v, (6, 2, q.SAMPLES))
    return {"x": x, "v": v, "metadata": metadata, "diagnostic_counter": cursor}


def input_entry(batch, cursor):
    hashes = [m.pilot.wave_digest(value) for value in batch["x"]]
    if hashes != [meta["input_pcm_sha256"] for meta in batch["metadata"]]:
        raise ValueError("Actual TRAIN PCM differs from metadata")
    return {"counter": cursor, "input_sha256": hashes,
        "target_sha256": [m.pilot.wave_digest(value) for value in batch["v"]], "metadata": batch["metadata"]}


def no_active_worker():
    q.no_active_worker()
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '182_diagnose_train_context_reconstruction_recovery[.]py[\" ]+(audit|run)' } | Select-Object -ExpandProperty ProcessId"
    reply = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], check=True, capture_output=True, text=True)
    if any(int(line.strip()) not in {os.getpid(), os.getppid()} for line in reply.stdout.splitlines() if line.strip()):
        raise ValueError("Active recovery worker: preserve and postpone")


def check_failure():
    doc = json.loads(FAILURE.read_text(encoding="utf-8"))
    if (doc["purpose"] != "NONRELEASE_TRAIN_CONTEXT_RECONSTRUCTION_FAILED_ATTEMPT_REVIEW" or
        type(doc["actual_exit_code"]) is not int or doc["actual_exit_code"] != 1 or
        doc["foreground_exec_session"] != 88459 or doc["terminal_status"]["status"] != "failed" or
        doc["terminal_status"]["completed_model_batches"] != 0 or doc["committed_row_count"] != 0 or
        doc["inputs_file_exists"] is not False or doc["diagnostic_file_exists"] is not False or
        doc["model_updates"] != 0 or doc["cuda_used"] is not False or doc["no_active_python_181_or_178"] is not True):
        raise ValueError("Exact preserved failed181 attempt required")
    files = {str((ROOT / name).resolve()): digest for name, digest in doc["raw_evidence_sha256"].items()}
    q.check_bindings(files)
    status = json.loads((ROOT / doc["task_dir"] / "run_status.json").read_text(encoding="utf-8"))
    if status != doc["terminal_status"]:
        raise ValueError("Failed181 terminal evidence changed")
    return files | {str(FAILURE): acq.sha256(FAILURE)}


def sources(require_audit=True):
    approval, models, sampler, bindings = q.checked_sources()
    bindings |= check_failure() | {str(path.resolve()): acq.sha256(path) for path in (Path(__file__), TEST, REPORT)}
    if require_audit:
        audit_doc = acq.read_sealed(AUDIT)
        if (audit_doc["tool"] != TOOL or audit_doc["bindings_sha256"] != bindings or
            audit_doc["canonical_reader"] != "177 ComponentStream.next_batch" or
            audit_doc["cursors"] != list(q.CURSORS) or len(audit_doc["inputs"]) != 12 or
            audit_doc["slot_checks"] != 72 or audit_doc["bit_equal_x_targets_metadata"] is not True or
            audit_doc["cpu_rng_unchanged"] is not True or audit_doc["cuda_initialized"] is not False or
            audit_doc["model_updates"] != 0):
            raise ValueError("Real fixed TRAIN reader equivalence audit required")
        bindings[str(AUDIT)] = acq.sha256(AUDIT)
    q.check_bindings(bindings)
    return approval, models, sampler, bindings


def dataset_pair(approval, sampler):
    dataset = p.inp.ApprovedTeacherDataset(p.inp.verified_approval(Path(approval["origin_approval"])), "kim_melband")
    true = m.LockedTruePool(m.bulk.OLD_LOCK, dataset.config)
    if sampler["seed"] != dataset.seed or sampler["true_lock_sha256"] != true.bound:
        raise ValueError("Original seed/true lock changed")
    return dataset, true


def audit(_out):
    if AUDIT.exists():
        raise ValueError("Existing real input audit: never overwrite/repeat")
    no_active_worker(); q.check_disk()
    approval, _, sampler, bindings = sources(False)
    torch.set_num_threads(q.THREADS)
    before = m.capture_rng("cpu")
    inputs = []
    with q.d.deterministic_runtime("cpu"):
        dataset, true = dataset_pair(approval, sampler)
        # Isolated stateless crop-reader instance ONLY, no optimizer or trainer.
        canonical = object.__new__(p.ComponentStream)
        canonical.dataset, canonical.true = dataset, true
        canonical.seed, canonical.config, canonical.cursor = dataset.seed, dataset.config, 3000
        for cursor in q.CURSORS:
            batch = collect_fixed_train(dataset, true, cursor)
            original = canonical.next_batch()
            if (original["cursor"] != cursor or original["metadata"] != batch["metadata"] or
                not torch.equal(original["x"], batch["x"]) or
                any(not torch.equal(original["targets"][arm], batch["v"]) for arm in t.ARMS)):
                raise ValueError("New reader differs from canonical TRAIN input/target/role")
            inputs.append(input_entry(batch, cursor))
        if not m.equal_state(before, m.capture_rng("cpu")) or torch.cuda.is_initialized():
            raise ValueError("Input audit changed CPU RNG or initialized CUDA")
    MONITOR.mkdir(parents=True, exist_ok=True)
    acq.write_new_json(AUDIT, acq.seal({"tool": TOOL, "bindings_sha256": bindings,
        "canonical_reader": "177 ComponentStream.next_batch", "cursors": list(q.CURSORS),
        "inputs": inputs, "slot_checks": 72, "bit_equal_x_targets_metadata": True,
        "cpu_rng_unchanged": True, "cuda_initialized": False, "model_updates": 0,
        "model_forward_used": False, "backward_used": False, "release_selection": "NONE"}))
    print("RECOVERY REAL INPUT AUDIT PASS12 batches72 slots; original177 bit equality; updates0", flush=True)


def check_plan(doc):
    q.check_plan(doc)
    if doc.get("tool") != TOOL or doc.get("recovery_of") != str(q.DEFAULT_OUT) or doc.get("input_reader") != "explicit_fixed3000..3011_canonical177":
        raise ValueError("Recovery identity/reader/provenance changed")


def prepare(out):
    out = q.fresh_output(out)
    no_active_worker(); q.check_disk()
    _, models, _, bindings = sources()
    plan = q.fixed_scope() | {"tool": TOOL, "recovery_of": str(q.DEFAULT_OUT),
        "input_reader": "explicit_fixed3000..3011_canonical177", "bindings_sha256": bindings,
        "model_digests": {name: q.r.dev.state_digest(saved["model"]) for name, saved in models.items()}}
    check_plan(plan)
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal(plan))
    print("RECOVERY PLAN SEALED; same36 CPU model-batches216 slots; updates0", flush=True)


def verified_plan(out):
    plan = acq.read_sealed(out / "plan.json")
    check_plan(plan); q.check_bindings(plan["bindings_sha256"])
    _, models, _, bindings = sources()
    if bindings != plan["bindings_sha256"] or plan["model_digests"] != {name: q.r.dev.state_digest(saved["model"]) for name, saved in models.items()}:
        raise ValueError("Changed recovery source/code/audit/binding")
    return plan


def status(out, values):
    q.status(out, values | {"tool": TOOL, "recovery_of": str(q.DEFAULT_OUT)})


def run(out):
    q.reject_existing_run(out)
    no_active_worker(); q.check_disk()
    if torch.cuda.is_initialized():
        raise ValueError("Fresh CPU-only process required")
    plan = verified_plan(out)
    approval, models, sampler, _ = sources()
    torch.set_num_threads(q.THREADS)
    outer_rng = m.capture_rng("cpu")
    rows, inputs, row_hashes = [], [], {}
    with m.bulk.worker_lock(out), q.d.deterministic_runtime("cpu"):
        status(out, {"status": "running", "phase": "collect_fixed_train", "completed_model_batches": 0, "limit": 36, "error": None})
        try:
            dataset, true = dataset_pair(approval, sampler)
            batches = []
            for cursor in q.CURSORS:
                batch = collect_fixed_train(dataset, true, cursor)
                inputs.append(input_entry(batch, cursor)); batches.append(batch)
            if inputs != acq.read_sealed(AUDIT)["inputs"]:
                raise ValueError("Actual inputs differ from sealed canonical equivalence audit")
            acq.write_new_json(out / "inputs.json", acq.seal({"purpose": q.PURPOSE, "tool": TOOL,
                "plan_sha256": acq.sha256(out / "plan.json"), "inputs": inputs}))
            wa, gs = (torch.from_numpy(fn()) for fn in (q.core.t09.make_analysis_matrix, q.core.t09.make_synthesis_matrix))
            for model in q.MODELS:
                net = m.frozen_factory(approval["source_protocol"])()
                saved = models[model]
                net.load_state_dict(saved["model"], strict=True)
                if saved["parameter_names"] != [key for key, _ in net.named_parameters()] or len(saved["modes"]) != len(list(net.modules())):
                    raise ValueError("Changed parameter/mode layout")
                for mod, mode in zip(net.modules(), saved["modes"]):
                    mod.training = mode
                q.check_net(net)
                if q.r.dev.state_digest(net.state_dict()) != plan["model_digests"][model]:
                    raise ValueError("Changed model identity")
                for index, batch in enumerate(batches):
                    q.check_disk(); started = time.perf_counter()
                    slots = [{"slot": j, "metadata": meta, "probe": q.slot_probe(net, batch["x"][j:j+1], batch["v"][j:j+1], wa, gs, meta)} for j, meta in enumerate(batch["metadata"])]
                    row = {"model": model, "counter": q.CURSORS[index], "slots": slots, "seconds": time.perf_counter()-started}
                    name = f"row_{len(rows):02d}.json"
                    acq.write_new_json(out / name, acq.seal(row))
                    row_hashes[name] = acq.sha256(out / name); rows.append(row)
                    status(out, {"status": "running", "phase": "cpu_frontend_diagnostic", "model": model, "counter": q.CURSORS[index], "completed_model_batches": len(rows), "limit": 36, "error": None})
                    print(f"RECOVERY model={model} counter={q.CURSORS[index]} completed={len(rows)}/36 seconds={row['seconds']:.3f}", flush=True)
                if q.r.dev.state_digest(net.state_dict()) != plan["model_digests"][model] or [mod.training for mod in net.modules()] != saved["modes"] or any(param.grad is not None for param in net.parameters()):
                    raise ValueError("Diagnostic changed model/modes/grad")
            result = {key: value for key, value in plan.items() if key != "content_sha256"} | {
                "plan_sha256": acq.sha256(out / "plan.json"), "inputs_sha256": acq.sha256(out / "inputs.json"),
                "row_sha256": row_hashes, "rows": rows, "inputs": inputs, "summary": q.summarize(rows),
                "runtime": q.d.runtime_identity("cpu"), "model_state_modes_rng_grad_unchanged": True,
                "scope": "Fixed TRAIN frontend attribution only, not board acceptance/full-corpus/quality floor/capacity proof"}
            if torch.cuda.is_initialized():
                raise ValueError("Unexpected CUDA initialization")
            validate_result(plan, result)
            acq.write_new_json(out / "diagnostic.json", acq.seal(result))
            status(out, {"status": "complete", "phase": "cpu_frontend_diagnostic_complete", "completed_model_batches": 36, "limit": 36, "error": None})
            print("RECOVERY COMPLETE36 model-batches216 slots; CPU-only updates0; NONRELEASE", flush=True)
        except BaseException as error:
            status(out, {"status": "failed", "phase": "cpu_frontend_diagnostic", "completed_model_batches": len(rows), "limit": 36, "error": repr(error)})
            raise
        finally:
            m.restore_rng(outer_rng, "cpu")


def validate_result(plan, result):
    check_plan(result)
    q.validate_result(plan, result)
    if result["inputs"] != acq.read_sealed(AUDIT)["inputs"]:
        raise ValueError("Changed originally audited TRAIN input/target/metadata")


def verify(out):
    plan = verified_plan(out)
    result = acq.read_sealed(out / "diagnostic.json")
    if result["plan_sha256"] != acq.sha256(out / "plan.json") or result["inputs_sha256"] != acq.sha256(out / "inputs.json"):
        raise ValueError("Changed plan/input artifact")
    inputs = acq.read_sealed(out / "inputs.json")
    if inputs["inputs"] != result["inputs"] or inputs["purpose"] != q.PURPOSE or inputs["tool"] != TOOL or inputs["plan_sha256"] != result["plan_sha256"]:
        raise ValueError("Changed input evidence")
    names = [f"row_{index:02d}.json" for index in range(36)]
    if set(result["row_sha256"]) != set(names):
        raise ValueError("Incomplete committed rows")
    for index, name in enumerate(names):
        if acq.sha256(out / name) != result["row_sha256"][name] or q.r.r.plain(acq.read_sealed(out / name)) != result["rows"][index]:
            raise ValueError("Changed committed row")
    validate_result(plan, result)
    print("RECOVERY VERIFIED36 model-batches216 slots; CPU-only updates0; NONRELEASE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("audit", "prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    globals()[args.operation](args.out)
