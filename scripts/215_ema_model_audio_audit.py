"""One real six-slot CPU model audit, zero Adam updates, under preheld images.

New integration, not an old208 input replay/test or formal training approval.
The supervisor holds previously observed native files BEFORE Python starts,
checks every actual load event, and holds source/Python/input bytes through EXIT.
The worker uses original input functions through213's supervised direct decoder.
Original194 forward/loss is unchanged except removing its one backward call.
No CUDA, source PT deserialize, DEV, output WAV or teacher inference.
"""
from __future__ import annotations

import argparse
import ast
import copy
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA215_REAL_MODEL_ZERO_UPDATE_AUDIO_AUDIT"
PREPARATION = "results/ema214_audio_backend_preparation_attempt04/preparation_result.json"
PINS = {
    "scripts/214_ema_audio_backend_preparation.py": "d16edc51c036597b6ac1fc2966c3063c19ce90304f110723557471133a71a713",
    "scripts/213_ema_direct_decoder_runtime.py": "c699c0f54b5817da6fc187ff065df757823857c852e8c9b3bdb0144a5d81af84",
    "scripts/212_ema_single_trajectory_engine.py": "8d10c304f3c3c2b4d2d6509dcec719e454b6afdabe651a329d1bc99b4a85d666",
    PREPARATION: "747a5de5bf4d3408e14f7c6d383ba6ade3b165239520450de1817042a486c6e4",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_files():
    for path, expected in PINS.items():
        require(sha(ROOT / path) == expected, "Changed215 dependency: " + path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_files()
p = load("ema215_metadata_and_native_primitives", "scripts/214_ema_audio_backend_preparation.py")
g = load("ema215_direct_decoder_only", "scripts/213_ema_direct_decoder_runtime.py")
obs = p.obs


def preparation():
    doc = json.loads((ROOT / PREPARATION).read_text(encoding="utf-8"))
    p.check_metadata(doc)
    p.check_metadata(doc["prepared_backend"])
    require(doc["error"] is None and doc["launcher_exit_code"] == 0
            and all(row["exit_code"] == 0 for row in doc["processes"].values())
            and doc["all_event_handles_post_exit_equal"] is True
            and len(doc["native_physical_files"]) == 218, "Complete prior native observation")
    require(doc["prepared_backend"]["actual_pcm_draws"] == 0
            and doc["prepared_backend"]["training_authorized"] is False,
            "Preparation is inventory, never training approval")
    return doc


def selected_true_files(lock, sampler):
    """Read-only original first local-Random selection; never calls crop/audio."""
    require(type(sampler["seed"]) is int and sampler["cursor"] == 4500,
            "Only predetermined zero-audit counter4500")
    selected, bindings = [], {}
    for domain in ("musdb", "mir1k", "instrumental"):
        rows = sorted((row for row in lock["records"] if row["domain"] == domain
                       and row["role"] == "train"), key=lambda row: row["track_id"])
        rng = random.Random(int(hashlib.sha256(
            f"{sampler['seed']}:true:{domain}:4500".encode()).hexdigest(), 16))
        row = rows[rng.randrange(len(rows))]
        selected.append({"domain": domain, "track_id": row["track_id"]})
        for field in ("mix_files", "vocal_files", "stem_files"):
            for path in row.get(field, []):
                expected = lock["files"][path]["sha256"]
                require(path not in bindings or bindings[path] == expected, "Conflicting selected true file")
                bindings[path] = expected
    return selected, bindings


def native_manifest(prior):
    files = copy.deepcopy(prior["native_physical_files"])
    for name, info in g.read_manifest()["files"].items():
        require(name not in files or obs.meta.typed(files[name]) == obs.meta.typed(info),
                "Conflicting actual Python/decoder physical file identity")
        files[name] = copy.deepcopy(info)
    return files


def validate_native(actual, files):
    """Exact physical ID/bytes, allowing only observed hard-link spelling."""
    fields = ("bytes", "sha256", "volume_serial", "file_index", "last_write_ticks")
    matches = [info for info in files.values()
               if all(obs.meta.typed(actual[name]) == obs.meta.typed(info[name]) for name in fields)]
    require(matches, "Actual native image was not preheld: " + actual["final_path"])


def forward_only_ast(tree):
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("validate_metadata", "boundary_backward")]
    require([n.name for n in nodes] == ["validate_metadata", "boundary_backward"], "Two exact original194 functions")
    nodes = copy.deepcopy(nodes)
    backward = nodes[1]
    removed = []
    for node in ast.walk(backward):
        for field, value in ast.iter_fields(node):
            if type(value) is list:
                for item in list(value):
                    if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                        call = item.value
                        if isinstance(call.func, ast.Attribute) and call.func.attr == "backward":
                            require(isinstance(call.func.value, ast.Name) and call.func.value.id == "scaled"
                                    and not call.args and not call.keywords, "Only exact scaled.backward() can be removed")
                            value.remove(item); removed.append(ast.dump(item, include_attributes=False))
    require(len(removed) == 1, "Exactly one original backward expression removed, no loss rewrite")
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("backward", "step") for n in ast.walk(backward)), "No hidden backward/Adam")
    return ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), removed


class ZeroUpdateForward:
    def __init__(self, loss):
        self.loss = loss
        source = ROOT / "scripts/194_train_mel_lr_scale.py"
        tree, self.removed = forward_only_ast(ast.parse(source.read_text(encoding="utf-8-sig")))
        namespace = dict(loss.function.__globals__)
        exec(compile(tree, str(source), "exec"), namespace)
        self.function = namespace["boundary_backward"]

    def __call__(self, model, batch):
        # No new reduction or per-slot loss implementation. This is194's
        # complete loss path under no_grad, NOT an update/reference gradient.
        return self.function(model, batch["x"], batch["v"], self.loss.wa,
                             self.loss.gs, "cpu", batch["metadata"], 32)


def worker_audit(out):
    """One new integrated real model audit; actual source restore, no updates."""
    a = load("ema215_original_input_components", "scripts/208_ema_authenticated_audio_input.py")
    e = load("ema215_new_trajectory", "scripts/212_ema_single_trajectory_engine.py")
    import torch
    import torch._dynamo
    from contextlib import ExitStack
    from unittest.mock import patch
    from types import MethodType
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(not torch.cuda.is_initialized(), "CPU zero audit only")
    prior = preparation()["prepared_backend"]
    out = Path(out).resolve()
    require(out.parent == (ROOT / "results").resolve() and out.is_dir()
            and not (out / "audit_result.pt").exists(), "Fresh owned audit artifacts")
    exposure = {"audio_draws_started": 0, "audio_draws_completed": 0,
                "forward_started": 0, "forward_completed": 0,
                "source_cpu_storage_reads": 0, "actual_raw_models": 0, "actual_CPU_Adam_constructions": 0,
                "adam_steps": 0, "autograd_engine_calls": 0, "source_training_PT_reads": 0}
    decoder = g.DirectDecoder(g.read_manifest())
    def progress(phase):
        require(not (out / "supervisor_refused.json").exists(), "Supervisor refused unknown/changed runtime")
        print("AUDIT_PHASE " + json.dumps({"phase": phase, "exposure": exposure}, allow_nan=False), flush=True)
    def forbidden(*args, **kwargs):
        raise RuntimeError("Zero audit forbids autograd/Adam.step/CUDA initialization")
    original_load = torch.load
    def only_storage(path, *args, **kwargs):
        require(Path(path).resolve() == (ROOT / e.j.STORAGE_REL).resolve()
                and exposure["source_cpu_storage_reads"] == 0, "One pinned205 CPU storage, not original training PT")
        exposure["source_cpu_storage_reads"] += 1
        return original_load(path, *args, **kwargs)
    try:
        with ExitStack() as stack:
            for obj, name in ((torch.autograd, "backward"), (torch.autograd, "grad"),
                              (torch.optim.Adam, "step"), (torch.cuda, "_lazy_init")):
                stack.enter_context(patch.object(obj, name, forbidden))
            stack.enter_context(patch.object(torch, "load", only_storage))
            stack.enter_context(g.original_decoder_functions(a.inp.m.core, a.inp.m.bulk, decoder))
            approval = a.inp.verified_approval(a.inp.DEFAULT_APPROVAL)
            dataset = a.inp.ApprovedTeacherDataset(approval, "kim_melband")
            true = a.inp.m.LockedTruePool(a.inp.m.bulk.OLD_LOCK, dataset.config)
            loss = e.OriginalBoundaryLoss("cpu")
            target = loss.function.__globals__["m"].core.t09
            def factory():
                exposure["actual_raw_models"] += 1
                return target.CausalSpectralUNet(bottleneck_blocks=2)
            owner = e.SingleTrajectoryEngine.from_fixed_cpu_storage(factory,
                lambda sampler: e.DeviceInputStream(sampler, true, dataset, a.data.crop_recipe, device="cpu"), loss)
            exposure["actual_CPU_Adam_constructions"] = 1
            forward = ZeroUpdateForward(loss)
            # Reuse208's exact backend/function/cache identities WITHOUT its
            # closed constructor, media_versions, PATH selection or old draw.
            class Backend:
                _backend_identity = a.AuthenticatedAudioStream._backend_identity
                _function_identity = a.AuthenticatedAudioStream._function_identity
                _cache_identity = a.AuthenticatedAudioStream._cache_identity
            backend = Backend()
            backend.true, backend.dataset, backend.route = true, dataset, owner.stream
            identity, functions, cache = backend._backend_identity(), backend._function_identity(), backend._cache_identity()
            bindings = dict(prior["bindings_sha256"])
            _, true_bindings = selected_true_files(true.lock, a.ORIGINAL_SAMPLER)
            bindings.update(true_bindings)
            a.verify_files(bindings)
            signatures = {name: a.data.stat_signature(name) for name in bindings}
            def backend_guard(expected_cache):
                require(e.equal(backend._backend_identity(), identity) and backend._function_identity() == functions,
                        "Original complete pool/target/function identity changed")
                require(e.equal(backend._cache_identity(), expected_cache), "Foreign LRU bytes/signatures changed")
                require(all(a.data.stat_signature(name) == sig for name, sig in signatures.items()), "Input/source signature changed")
                owner.stream._guard(); decoder.guard()
            before = owner.state_dict()
            backend_guard(cache)
            progress("source4500_restored_before_real_input")
            exposure["audio_draws_started"] += 1
            batch = owner.stream.next_batch()
            exposure["audio_draws_completed"] += 1
            packet_identity = a.packet_identity(batch)
            require(batch["cursor"] == 4500 and len(batch["metadata"]) == 6,
                    "One predetermined six-slot original TRAIN batch")
            require(all(batch["metadata"][i]["track_id"] == r["track_id"] for i, r in enumerate(selected_true_files(true.lock, a.ORIGINAL_SAMPLER)[0])),
                    "Pure selection versus actual original crop differs")
            cache = backend._cache_identity()
            require(len(true.cache) == 3 and len(dataset.cache) <= 1, "Original bounded LRUs")
            backend_guard(cache)
            input_digest = e.c.digest(batch)
            e.c.noalias(batch, owner._external())
            original_forward = owner.model.forward
            def counted_forward(instance, *args, **kwargs):
                progress("real_microbatch_forward")
                exposure["forward_started"] += 1
                result = original_forward(*args, **kwargs)
                require(result.dtype == torch.float32 and bool(torch.isfinite(result).all())
                        and not result.requires_grad and result.grad_fn is None, "Finite no-grad FP32 model output")
                exposure["forward_completed"] += 1
                return result
            require("forward" not in owner.model.__dict__, "No pre-existing forward override")
            owner.model.forward = MethodType(counted_forward, owner.model)
            try:
                owner.model.train()  # Same deterministic graph/mode as194.
                with torch.no_grad():
                    losses = forward(owner.model, batch)
            finally:
                del owner.model.forward
            require(exposure["forward_started"] == exposure["forward_completed"] == 6,
                    "Exactly six completed original microbatch forwards")
            e.c.typed_tree(losses)
            require(e.c.digest(batch) == input_digest, "Full loss path mutated real PCM/metadata")
            backend_guard(cache); a.verify_files(bindings)
            raw_now = owner._raw_packet(4500)
            require(all(e.equal(raw_now[name], before["raw"][name]) for name in raw_now if name != "modes")
                    and e.equal(owner.shadow.state_dict(owner.model, raw_step=4500), before["shadow"])
                    and e.equal(owner.live.state_dict(), before["live_context"])
                    and e.equal(e.capture_rng("cpu"), before["rng"]),
                    "Forward changed raw/grad/Adam/EMA/schedule/RNG before restoration")
            # Restore whole semantic input/modes/RNG as well as raw/Adam/shadow.
            # Foreign decode LRUs intentionally remain populated; not rolled back.
            owner._apply(before)
            after = owner.state_dict()
            require(e.equal(before, after), "Whole zero-update raw/Adam/EMA/input/schedule/RNG differs")
            require(not torch.cuda.is_initialized() and owner.exposure["adam_started"] == 0,
                    "No source Adam step/CUDA")
            backend_guard(cache)
            for name, expected in prior["imported_python_files_sha256"].items():
                require(sha(name) == expected, "Prepared Python/native source bytes changed")
            proof = e.c.seal({"schema": 1, "purpose": PURPOSE,
                "source_storage_sha256": e.j.STORAGE_SHA, "source_training_PT_sha256": e.c.source.SOURCE_SHA,
                "full_before": before, "full_after": after, "input_packet": e.portable(batch),
                "input_identity": packet_identity, "input_packet_digest": input_digest,
                "losses": e.portable(losses), "removed_AST_expressions": forward.removed,
                "exposure": dict(exposure), "source_step": 4500, "updates": 0,
                "CPU_explicit_migration_not_CUDA_resume": True, "CUDA_initialized": False,
                "foreign_decode_LRU_rollback_claimed": False,
                "decoder_rows": e.portable(decoder.rows), "training_authorized": False, "release_selection": "NONE"})
            with (out / "audit_result.pt").open("xb") as handle:
                torch.save(proof, handle)
                handle.flush(); os.fsync(handle.fileno())
            progress("whole_zero_update_state_equal_saved")
            report = p.seal_metadata({"purpose": PURPOSE, "artifact_path": str(out / "audit_result.pt"),
                "artifact_sha256": sha(out / "audit_result.pt"), "artifact_own_typed205_seal": proof["content_sha256"],
                "exposure": exposure, "input_identity": packet_identity, "losses": losses,
                "full_state_before_sha256": e.c.digest(before), "full_state_after_sha256": e.c.digest(after),
                "decoder_calls": len(decoder.rows), "decoder_rows": decoder.rows,
                "updates": 0, "source_step": 4500, "CUDA_initialized": False,
                "full_training_ready": False, "training_authorized": False, "release_selection": "NONE"})
            print("MODEL_AUDIO_AUDITED " + json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)
    except BaseException as error:
        obs.write_new(out / "worker_failure.json", p.seal_metadata({"purpose": PURPOSE,
            "error": {"type": type(error).__name__, "message": str(error)}, "exposure": exposure,
            "decoder_rows": decoder.rows, "training_authorized": False}))
        raise


def collect(out):
    check_files()
    out = Path(out).resolve()
    require(out.parent == (ROOT / "results").resolve() and not out.exists(), "Fresh direct result directory; refuse repeat/overwrite")
    require(not any(name in __import__("sys").modules for name in ("torch", "numpy", "scipy", "soundfile")), "Light supervisor only")
    require(__import__("shutil").disk_usage(ROOT).free >= 12 * 1024**3, "Audit disk>=12GiB")
    prior = preparation()
    files = native_manifest(prior)
    prepared = prior["prepared_backend"]
    lock_path = ROOT / "results/training_protocol_20261001/dataset_lock.json"
    require(sha(lock_path) == prepared["source_sampler"]["true_lock_sha256"], "Whole input lock changed")
    selected, true_bindings = selected_true_files(json.loads(lock_path.read_text(encoding="utf-8")), prepared["source_sampler"])
    bindings = dict(prepared["bindings_sha256"])
    bindings.update(prepared["imported_python_files_sha256"])
    bindings.update(true_bindings)
    bindings.update({str(ROOT / name): expected for name, expected in PINS.items()})
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    storage = ROOT / "results/mel_ema_cpu_state_monitor_20261004/source_cpu_container_4500_complete_defaults.pt"
    bindings[str(storage)] = "b003b48e46e1e573491c7359251899a600e3daa542462204d38140aa7a11254a"
    native = p.LargeNative()
    native.dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    native.dll.CreateFileW.restype = wintypes.HANDLE
    environment, started = dict(os.environ), time.monotonic()
    out.mkdir()
    row = {"purpose": PURPOSE, "started_utc": obs.now(), "source_hashes": dict(PINS),
           "worker_source_sha256": sha(__file__), "processes": {}, "error": None,
           "selected_true_records": selected, "training_authorized": False}
    process, pending, active, event_handles, source_handles, breaks = None, None, set(), [], [], set()
    buffers, readers, pipe_errors = {"stdout": bytearray(), "stderr": bytearray()}, [], []
    journal_path = out / "native_event_journal.jsonl"
    with journal_path.open("x", encoding="utf-8") as journal:
        def record(item):
            journal.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + "\n"); journal.flush()
        try:
            # Python/input source READ locks cover every prepared import file,
            # all approved labels, and the predetermined true source files.
            for filename, expected in bindings.items():
                handle = native.dll.CreateFileW(filename, 0x80000000, 1, None, 3, 0x80, None)
                require(handle not in (None, 0, ctypes.c_void_p(-1).value), "Unable to prehold source/input: " + filename)
                source_handles.append(handle)
                require(sha(filename) == expected, "Prepared source/input bytes changed before worker: " + filename)
            row["preheld_source_bindings_sha256"] = bindings
            with g.HeldFiles(files, native) as held:
                row["preheld_native_physical_files"] = copy.deepcopy(files)
                row["all_native_and_source_holds_before_worker_spawn"] = True
                import sys
                startup = subprocess.STARTUPINFO()
                startup.dwFlags, startup.wShowWindow = subprocess.STARTF_USESHOWWINDOW, 0
                argv = [sys.executable, "-B", "-X", "pycache_prefix=" + str(out / "unused_pycache"),
                        str(Path(__file__).resolve()), "worker-audit", "--out", str(out)]
                # Fresh unused prefix + -B prevents both reading old default
                # pyc caches and writing new bytecode, no installed/PATH edits.
                require(not (out / "unused_pycache").exists(), "Fresh diagnostic bytecode prefix")
                row["request"] = {"argv": argv, "executable": sys.executable, "shell": False,
                                  "DEBUG_PROCESS": True, "env": None, "cwd": None}
                process = subprocess.Popen(argv, executable=sys.executable, shell=False, close_fds=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startup,
                    creationflags=subprocess.CREATE_NO_WINDOW | 1, env=None, cwd=None)
                row["launcher_pid"] = process.pid
                native.checked("DebugSetProcessKillOnExit", False)
                def consume(name):
                    try:
                        while True:
                            block = getattr(process, name).read(1024 * 1024)
                            if not block:
                                break
                            require(len(buffers[name]) + len(block) <= 32 * 1024**2, "Bounded diagnostic metadata pipes")
                            buffers[name].extend(block)
                    except BaseException as error:
                        pipe_errors.append(error)
                for name in buffers:
                    thread = threading.Thread(target=consume, args=(name,), daemon=True)
                    thread.start(); readers.append(thread)
                created, events = 0, 0
                while not created or active:
                    require(time.monotonic() - started < 480 and not pipe_errors, "Bounded480s audit/no pipe errors")
                    event = obs.DebugEvent()
                    if not native.dll.WaitForDebugEventEx(ctypes.byref(event), 100):
                        require(ctypes.get_last_error() == 121, "Native audit wait failed")
                        continue
                    pending, events = event, events + 1
                    require(events <= 20000 and 1 <= event.code <= 8, "Audit native event scope/budget")
                    item = {"sequence": events, "pid": event.pid, "tid": event.tid, "code": event.code, "observed_utc": obs.now()}
                    if event.code == 3:
                        require(event.pid not in active and created < 4, "Bounded owned Python/console chain only")
                        created += 1; active.add(event.pid)
                        row["processes"][str(event.pid)] = {"events": 0, "image_file_events": 0, "exit_code": None}
                    require(event.pid in active, "Only newly owned audit process tree")
                    current = row["processes"][str(event.pid)]; current["events"] += 1
                    if event.code in (3, 6):
                        info = event.data.create if event.code == 3 else event.data.load
                        require(info.file and info.file not in event_handles, "Missing/duplicate native file event handle")
                        event_handles.append(info.file)
                        actual = native.measure(info.file); item["file"] = actual
                        current["image_file_events"] += 1
                        record({"kind": "prevalidation_image", "event": item})
                        validate_native(actual, files)
                        item["matched_preheld_physical_identity"] = True
                        if event.code == 3:
                            p.check_executable_identity(actual, prior["pre_bound_python_executables"])
                            current["actual_executable_identity"] = actual
                    elif event.code == 1:
                        item.update(exception_code=int(event.data.exception.record.code), first_chance=int(event.data.exception.first_chance))
                    elif event.code == 5:
                        current["exit_code"] = int(event.data.exit_code); active.remove(event.pid)
                    continuation, seen = obs.continuation(event, event.pid in breaks)
                    if seen:
                        breaks.add(event.pid)
                    item["continuation"] = continuation
                    record(item)
                    native.checked("ContinueDebugEvent", event.pid, event.tid, continuation); pending = None
                row["launcher_exit_code"] = process.wait(timeout=2)
                for reader in readers:
                    reader.join(timeout=2)
                require(all(not thread.is_alive() for thread in readers) and not pipe_errors, "Full audit pipe consumption")
                require(row["launcher_exit_code"] == 0 and all(r["exit_code"] == 0 for r in row["processes"].values()), "Actual audit process exit not0")
                lines = buffers["stdout"].decode("utf-8").splitlines()
                reports = [json.loads(line.removeprefix("MODEL_AUDIO_AUDITED ")) for line in lines if line.startswith("MODEL_AUDIO_AUDITED ")]
                require(len(reports) == 1, "Exactly one complete integrated audit report")
                report = reports[0]; p.check_metadata(report)
                require(report["exposure"]["forward_completed"] == 6 and report["updates"] == 0
                        and report["CUDA_initialized"] is False and report["training_authorized"] is False
                        and report["full_state_before_sha256"] == report["full_state_after_sha256"], "Integrated zero-update audit scope/state")
                require(report["artifact_path"] == str(out / "audit_result.pt") and sha(out / "audit_result.pt") == report["artifact_sha256"], "Whole audit artifact bytes")
                entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
                image_events = [item for item in entries if "sequence" in item and "file" in item]
                require(len(image_events) == len(event_handles), "Full actual native event accounting")
                for handle, item in zip(event_handles, image_events):
                    require(obs.meta.typed(native.measure(handle)) == obs.meta.typed(item["file"]), "Actual native image changed through EXIT")
                held.check()
                for filename, expected in bindings.items():
                    require(sha(filename) == expected, "Held source/input changed through EXIT")
                require(environment == dict(os.environ), "Supervisor environment changed")
                check_files()
                row.update(audit_report=report, native_file_event_count=len(image_events),
                    all_event_handles_post_exit_equal=True, all_preheld_source_bytes_post_exit_equal=True,
                    actual_debug_supervised_native_audio_scope_passed=True,
                    nondebug_runtime_equivalence_claimed=False, full_training_ready=False, completed_utc=obs.now())
        except BaseException as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
            record({"kind": "failure", "row": row})
            obs.write_new(out / "supervisor_refused.json", p.seal_metadata(row["error"]))
            if pending is not None:
                continuation, _ = obs.continuation(pending, pending.pid in breaks)
                native.checked("ContinueDebugEvent", pending.pid, pending.tid, continuation)
            for pid in list(active):
                native.checked("DebugActiveProcessStop", pid)
            row["detached_new_owned_pids_on_failure"] = list(active)
            raise
        finally:
            for handle in reversed(event_handles):
                native.checked("CloseHandle", handle)
            for handle in reversed(source_handles):
                native.checked("CloseHandle", handle)
            for name in buffers:
                (out / (name + ".bin")).write_bytes(buffers[name])
            obs.write_new(out / "supervised_audit_result.json", p.seal_metadata(row))
    print("REAL_MODEL_ZERO_UPDATE_AUDIO_AUDIT " + str(out / "supervised_audit_result.json"), flush=True)
    print(json.dumps({"actual_audio_draws": report["exposure"]["audio_draws_completed"],
        "actual_microbatch_forwards": report["exposure"]["forward_completed"], "student_updates": 0,
        "source_step": 4500, "CUDA_initialized": False, "full_training_ready": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("collect", "worker-audit"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "collect":
        collect(args.out)
    else:
        worker_audit(args.out)
