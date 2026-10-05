"""Activation-gated, native-supervised fixed500 EMA exploratory training CLI.

One raw model / one original training Adam. Raw and EMA are CPU DEV copies,
not duplicate training arms. No automatic restart, resume, promotion, deployment,
or release. A missing complete activation record fails BEFORE any worker spawn.
This implementation does not create its own approval or a premature plan.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA221_NATIVE_SUPERVISED_SINGLE_TRAJECTORY_FIXED500"
SOURCE = "scripts/217_ema_actual_update_mechanism.py"
PINS = {
    SOURCE: "468734d23c353c805a87fa10f3c6d134a14bf789adabb2659fb63237e4bc71fd",
    "scripts/219_ema_continuous_training_core.py": "cfd19e0b4059164621c190eda4f74819e0c8b73b358396b94e8f1bbb7f3619ca",
    "scripts/220_ema_original_development_adapter.py": "1eb6be95980268c2d1ef239712073cbafbd8d20b11af14f82b9b1da10369c857",
    "results/ema218_actual_cpu_mechanism_attempt01/mechanism_result.json": "35973a31bc88b6c591a93e9beb8a96a1d7a2e18a509e67366575df632e4a81fb",
    "results/ema218_actual_cuda_mechanism_attempt01/mechanism_result.json": "6cc0734a3b23d7b46182e7739d18cb7b4135ccdd029db6394f94e83d6404e8ad",
    "results/ema218_actual_cpu_mechanism_attempt01/independent_readonly_review.json": "ef7d20c28d7a4da6410f7aada441916deccdc7559ba5fb80ee936f3cf758c7e2",
    "results/ema218_actual_cuda_mechanism_attempt01/independent_readonly_review.json": "52bc9186e893900a4a926bc124f0f4696e4a10c0cda55fd4d8c78df97362f320",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def check_files():
    for relative, expected in PINS.items():
        require(sha(ROOT / relative) == expected, "221 implementation/actual gate changed: " + relative)


def authority(path, q):
    """Lightweight existing plan check; no Torch, PT, CUDA or subprocess."""
    check_files()
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / "results") and path.is_file(), "Complete activation not yet present; do not start")
    doc = json.loads(path.read_text(encoding="utf-8")); q.p.check_metadata(doc)
    require(doc.get("purpose") == "NONRELEASE_EMA219_CONTINUOUS_SINGLE_TRAJECTORY"
            and doc.get("training_authorized") is True and doc.get("source_step") == 4500
            and doc.get("limit") == 5000 and doc.get("additional_updates") == 500
            and doc.get("stages") == [4750, 5000] and doc.get("release_selection") == "NONE"
            and doc.get("deployment") is False and doc.get("EMA_feedback") is False
            and doc.get("single_raw_original_Adam") is True, "Bounded single-trajectory activation")
    bindings = doc.get("bindings_sha256")
    require(type(bindings) is dict and bindings.get(str(Path(__file__).resolve())) == sha(__file__), "Executed CLI must be plan-bound")
    for name, expected in bindings.items():
        require(sha(name) == expected, "Activation source changed: " + name)
    #219's complete own gate checker must pass in the worker as well. No
    #boolean shortcut here may invoke a worker with missing/unbound gates.
    gates = doc.get("gates")
    required = {"actual_zero_update_audio_audit", "strict_source_CUDA_runtime_RNG",
                "actual_CPU_CUDA_first_raw_original_Adam_bit_identity", "actual_EMA_noalias",
                "complete_disk_replay_and_joint_CPU_CUDA_RNG_rollback", "cross_device_input_target_PCM_metadata",
                "continuous_engine_units", "original_DEVELOPMENT_evaluator_verified",
                "per_draw_source_and_native_runtime_guard", "complete_dependency_review", "human_training_authorization_record"}
    require(type(gates) is dict and set(gates) == required, "Missing full actual training gates")
    for key, row in gates.items():
        require(type(row) is dict and row.get("passed") is True and bindings.get(row.get("path")) == row.get("sha256")
                and type(row.get("path")) is str, "Missing or unbound gate: " + key)
    require(type(doc.get("evaluator_binding")) is dict, "Original evaluator binding required")
    return doc


def validate_exposure(exposure):
    require(type(exposure) is dict and all(type(v) is int and v >= 0 for v in exposure.values()), "Actual exposure counter types")
    require(exposure["forward_started"] == exposure["forward_completed"] == exposure["backward_completed"] == 3000
            and exposure["adam_started"] == exposure["adam_completed"] == 500, "Exactly500 actual raw Adam /3000 microbatch forwards/backwards")


def valid_training_report(report):
    validate_exposure(report["exposure"])
    require(report["purpose"] == PURPOSE and report["training_authorized"] is True
            and report["formal_training_updates"] == 500 and report["step"] == 5000
            and report["development_steps"] == [4750, 5000] and report["single_training_Adam"] is True
            and report["release_selection"] == "NONE", "Complete worker budget/DEV checkpoints; not a release")
    return True


def duplicate_preflight():
    """Dynamic command-line/PID inventory, not historical saved running flags."""
    excluded = [os.getpid(), os.getppid()]  # One actual venv-shim/current-host chain.
    script = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' } | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 4 -Compress"
    ps = str(Path(os.environ["SYSTEMROOT"]) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    result = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True,
                            timeout=30, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    if type(rows) is dict:
        rows = [rows]
    import re
    repo = str(ROOT).lower()
    relevant = [row for row in rows if row["ProcessId"] not in excluded and
                (repo in (row.get("CommandLine") or "").lower() or
                 re.search(r"(?:scripts[\\/])(?:_test_ema|_review_ema|(?:19[3-9]|20[0-9]|21[0-9]|22[0-9])_)", row.get("CommandLine") or "", re.I))]
    require(not relevant, "Another repository training/diagnostic/reader task is actually live; do not start a duplicate")
    return {"excluded_current_host_shim_pids": excluded, "actual_relevant_other_processes": relevant}


def collector_ast():
    """Reuse only SHA-bound217's actual native-event/pipe/handle supervisor.

    Declared changes: fixed500 runtime/event budgets;221 worker/activation;
    actual training report scope; preserve an existing refusal marker. File
    event verification/continuation/post-EXIT SHA/owned-PID logic is unchanged.
    """
    check_files()
    original = ast.parse((ROOT / SOURCE).read_text(encoding="utf-8"))
    nodes = [copy.deepcopy(n) for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "collect"]
    require(len(nodes) == 1, "One original native supervisor")
    class Change(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 900:
                return ast.copy_location(ast.Constant(36000), node)
            if type(node.value) is int and node.value == 40000:
                return ast.copy_location(ast.Constant(250000), node)
            mapping = {"ACTUAL_MECHANISM_VERIFIED ": "EMA_TRAIN_VERIFIED ", "mechanism_result.json": "supervised_training_result.json",
                       "ACTUAL_MECHANISM_ARCHIVE ": "EMA_TRAIN_ARCHIVE ", "Bounded900s mechanism": "Bounded36000s fixed500 training",
                       "Fresh217 output, refuse overwrite/repeat": "Fresh221 formal output, refuse overwrite/repeat"}
            if type(node.value) is str and node.value in mapping:
                return ast.copy_location(ast.Constant(mapping[node.value]), node)
            return node

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "argv":
                require(isinstance(node.value, ast.List), "Exact worker argv list")
                node.value.elts.extend(ast.parse('["--activation", str(ACTIVATION_PATH)]', mode="eval").body.elts)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "row":
                require(isinstance(node.value, ast.Dict), "Exact supervisor result schema")
                for i, key in enumerate(node.value.keys):
                    if isinstance(key, ast.Constant) and key.value == "formal_training_updates":
                        node.value.values[i] = ast.Constant(None)  # Unknown until genuine worker completion.
                    if isinstance(key, ast.Constant) and key.value == "training_authorized":
                        node.value.values[i] = ast.Constant(True)
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "require" and len(node.args) == 2:
                if isinstance(node.args[1], ast.Constant) and node.args[1].value == "Actual mechanism scope":
                    node.args[0] = ast.parse("valid_training_report(report)", mode="eval").body
                    node.args[1] = ast.Constant("Complete actual fixed500 training scope")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "row" and node.func.attr == "update":
                for keyword in node.keywords:
                    if keyword.arg == "full_training_ready":
                        keyword.arg, keyword.value = "actual_training_completed", ast.Constant(True)
                    if keyword.arg == "mechanism_report":
                        keyword.arg = "training_report"
                node.keywords.append(ast.keyword(arg="formal_training_updates", value=ast.parse('report["formal_training_updates"]', mode="eval").body))
            return node
    node = Change().visit(nodes[0])
    guards, markers = 0, 0
    for block in ast.walk(node):
        for field, children in ast.iter_fields(block):
            if type(children) is not list:
                continue
            for i, child in enumerate(list(children)):
                if isinstance(child, ast.Assign) and ast.unparse(child).startswith("process = subprocess.Popen("):
                    children.insert(i, ast.parse('''
obs.write_new(out / "supervisor_guard.json", p.seal_metadata({"purpose": PURPOSE, "supervisor_pid": os.getpid(),
    "activation_sha256": sha(ACTIVATION_PATH), "preheld_native_files": len(known), "started_utc": obs.now()}))
''').body[0]); guards += 1
                if isinstance(child, ast.Expr) and ast.unparse(child).startswith("obs.write_new(out / 'supervisor_refused.json',"):
                    children[i] = ast.If(test=ast.parse("not (out / 'supervisor_refused.json').exists()", mode="eval").body,
                                         body=[child], orelse=[]); markers += 1
    require(guards == markers == 1, "One new pre-spawn guard and exclusive refusal marker")
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def train(activation_path, out):
    # Metadata validation happens before Popen or any heavy module import.
    q = load("ema221_native_algorithms_only", SOURCE)
    activation = authority(activation_path, q)
    prior, runtime = q.gate()
    bindings = activation["bindings_sha256"]
    def gate():
        authority(activation_path, q)
        return prior, runtime
    def device_scope(device):
        require(device == "cuda", "Formal fixed500 uses strict source CUDA, no fallback")
    def resource():
        result = q.r.resource()
        result["no_duplicate_task_preflight"] = duplicate_preflight()
        return result
    namespace = dict(q.__dict__)
    namespace.update(__file__=__file__, PURPOSE=PURPOSE, ACTIVATION_PATH=Path(activation_path).resolve(),
                     gate=gate, device_scope=device_scope, validate_exposure=validate_exposure,
                     valid_training_report=valid_training_report,
                     r=type("ResourceBoundary", (), {"resource": staticmethod(resource)})(),
                     PINS=q.PINS | PINS | {str(Path(activation_path).resolve()): sha(activation_path)} | bindings)
    exec(compile(collector_ast(), "217native[221fixed500-scope]", "exec"), namespace)
    namespace["collect"](out, "cuda")


def worker(activation_path, out):
    check_files()
    out = Path(out).resolve()
    require(out.parent == ROOT / "results" and out.is_dir(), "Fresh supervisor-owned worker output")
    q = load("ema221_worker_native_gate", SOURCE)
    activation = authority(activation_path, q)
    guard = json.loads((out / "supervisor_guard.json").read_text(encoding="utf-8")); q.p.check_metadata(guard)
    require(guard["purpose"] == PURPOSE and guard["activation_sha256"] == sha(activation_path)
            and guard["preheld_native_files"] == 246, "Actual preheld native supervisor required")
    # OS process handle, not merely a saved running status, guards supervisor
    # lifetime. No user process is attached, modified, terminated or killed.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]; kernel.GetExitCodeProcess.restype = wintypes.BOOL
    supervisor = kernel.OpenProcess(0x1000, False, guard["supervisor_pid"])
    require(supervisor, "Actual supervisor process is unavailable")
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]; kernel.CloseHandle.restype = wintypes.BOOL
    owner, phase, latest = None, "initializing", None
    exposure = {"forward_started": 0, "forward_completed": 0, "backward_completed": 0, "adam_started": 0, "adam_completed": 0}
    def allowed():
        code = wintypes.DWORD()
        require(kernel.GetExitCodeProcess(supervisor, ctypes.byref(code)) and code.value == 259
                and not (out / "supervisor_refused.json").exists(), "Actual native supervisor stopped/refused; abort safely")
        require(shutil.disk_usage(out).free >= 12*1024**3, "Disk reserve reached; no cleanup or automatic retry")
    def status(state, error=None):
        payload = q.p.seal_metadata({"purpose": PURPOSE, "status": state, "phase": phase,
            "step": 4500 if owner is None else owner.live.step, "limit": 5000,
            "additional_steps": 0 if owner is None else owner.live.step-4500,
            "latest_checkpoint": latest, "actual_training_exposure": dict(exposure), "error": error,
            "release_selection": "NONE", "updated_utc": q.obs.now()})
        with (out / "run_status.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
    try:
        allowed()
        t = load("ema221_continuous_core", "scripts/219_ema_continuous_training_core.py")
        d = load("ema221_original_DEV", "scripts/220_ema_original_development_adapter.py")
        a = t.load("ema221_actual_backend_components", "scripts/208_ema_authenticated_audio_input.py")
        import torch
        import torch._dynamo
        from contextlib import ExitStack
        from types import MethodType
        e, c = t.e, t.c
        checked = t.checked_activation(activation_path)
        prior = q.a.preparation()["prepared_backend"]
        decoder = q.g.DirectDecoder(q.g.read_manifest())
        native = q.p.LargeNative()
        with ExitStack() as stack:
            decoder_journal = stack.enter_context((out / "decoder_native_journal.jsonl").open("x", encoding="utf-8", buffering=1))
            original_decode = decoder.check_output
            def journaled_decode(instance, argv, **kwargs):
                previous = len(instance.rows)
                try:
                    return original_decode(argv, **kwargs)
                finally:
                    for row in instance.rows[previous:]:
                        decoder_journal.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+"\n")
                    decoder_journal.flush()
            decoder.check_output = MethodType(journaled_decode, decoder)
            stack.enter_context(q.g.original_decoder_functions(a.inp.m.core, a.inp.m.bulk, decoder))
            approval = a.inp.verified_approval(a.inp.DEFAULT_APPROVAL)
            dataset = a.inp.ApprovedTeacherDataset(approval, "kim_melband")
            true = a.inp.m.LockedTruePool(a.inp.m.bulk.OLD_LOCK, dataset.config)
            loss = e.OriginalBoundaryLoss("cuda")
            original_m = loss.function.__globals__["m"]
            stack.enter_context(q.g.original_decoder_functions(original_m.core, original_m.bulk, decoder))
            torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            require(not torch.cuda.is_initialized() or torch.cuda.device_count() == 1, "Exact source device")
            # CUDA may have initialized when matrices explicitly moved; no
            # comparison is made until ALL original modules finished importing.
            torch.cuda.init()
            require(sha(ROOT / q.ARTIFACT) == q.PINS[q.ARTIFACT], "SHA before actual zero-update source audit read")
            proof = torch.load(ROOT / q.ARTIFACT, map_location="cpu", weights_only=True)
            c.check_seal(proof); c.check_seal(proof["full_before"]); c.check_seal(proof["full_after"])
            require(c.equal(proof["full_before"], proof["full_after"]) and proof["updates"] == 0
                    and sha(ROOT / q.ARTIFACT) == q.PINS[q.ARTIFACT], "Accepted immutable actual source audit")
            source = proof["full_before"]; parent, raw = source["parent"], source["raw"]
            context = source["live_context"]["source_context"]
            require(c.equal(e.runtime_identity("cuda"), parent["parent_metadata"]["runtime"]), "Post-import exact source CUDA4 FP32 runtime")
            rng = c.portable(parent["parent_metadata"]["rng"])
            phase = "development_input_preparation"; status("running")
            teacher_ids = [Path(row["source"]["path"]).name for row in dataset.rows]
            evaluator = d.OriginalDevelopmentEvaluator.prepare(original_m, teacher_ids,
                lambda bindings: t.SourceReadLease(bindings, native), allowed)
            require(c.equal(evaluator.binding, checked["evaluator_binding"]), "Actual old DEV evaluator binding differs")
            torch.set_num_threads(4)
            require(c.equal(e.runtime_identity("cuda"), parent["parent_metadata"]["runtime"]), "DEV preparation changed source runtime")
            target = original_m.core.t09
            model = target.CausalSpectralUNet(bottleneck_blocks=2).to("cuda")
            model.load_state_dict(raw["tensors"], strict=True)
            for module, mode in zip(model.modules(), raw["modes"]):
                module.training = mode
            for name, parameter in model.named_parameters():
                parameter.grad = None if raw["gradients"][name] is None else raw["gradients"][name].to("cuda").clone()
            group = c.portable(raw["optimizer"]["param_groups"][0]); group.pop("params")
            optimizer = torch.optim.Adam(model.parameters(), **group)
            optimizer.load_state_dict(c.portable(raw["optimizer"]))
            optimizer.defaults.clear(); optimizer.defaults.update(c.portable(raw["optimizer_defaults"]))
            route = e.DeviceInputStream(context["sampler"], true, dataset, a.data.crop_recipe, device="cuda")
            class Backend:
                _backend_identity = a.AuthenticatedAudioStream._backend_identity
                _function_identity = a.AuthenticatedAudioStream._function_identity
                _cache_identity = a.AuthenticatedAudioStream._cache_identity
            backend = Backend(); backend.true, backend.dataset, backend.route = true, dataset, route
            stream = t.GuardedAudioStream(route, backend, decoder, native, allowed)
            e.restore_rng(rng, "cuda")
            owner = t.ContinuousTrajectory(model, optimizer, parent, context, stream, loss, activation=checked)
            restored = owner.state_dict()["trajectory"]
            require(c.equal(restored["raw"], raw) and c.equal(restored["shadow"], source["shadow"])
                    and c.equal(restored["rng"], rng), "Complete actual4500 raw/Adam/E0/source CUDA RNG restore")
            original_forward = model.forward
            def counted(instance, *args, **kwargs):
                allowed(); exposure["forward_started"] += 1
                output = original_forward(*args, **kwargs)
                exposure["forward_completed"] += 1; allowed()
                def backward_arrived(gradient):
                    exposure["backward_completed"] += 1
                    return gradient
                output.register_hook(backward_arrived)
                return output
            model.forward = MethodType(counted, model)
            def checkpoint(label):
                name = "NONRELEASE_EMA_"+label+".pt"
                digest = owner.save_new(out / name)
                receipt = q.p.seal_metadata({"purpose": PURPOSE, "step": owner.live.step, "checkpoint": name,
                    "sha256": digest, "activation_sha256": sha(activation_path), "pending_DEV": owner.live.validation_due,
                    "release_selection": "NONE"})
                q.obs.write_new(out / ("checkpoint_"+label+".json"), receipt)
                return receipt
            latest = checkpoint("step_4500")
            phase = "training"; status("running")
            with (out / "updates.jsonl").open("x", encoding="utf-8", buffering=1) as log:
                while owner.live.step < 5000:
                    allowed()
                    result = owner.update_next(allowed)
                    exposure["adam_started"], exposure["adam_completed"] = owner.exposure["adam_started"], owner.exposure["adam_completed"]
                    log.write(json.dumps(result, ensure_ascii=True, allow_nan=False)+"\n")
                    if owner.live.step % 10 == 0 or owner.live.step == 4501:
                        status("running")
                        print("EMA_TRAIN step="+str(owner.live.step)+"/5000", flush=True)
                    if owner.live.validation_due:
                        phase = "development_validation"; latest = checkpoint("pre_DEV_"+str(owner.live.step)); status("running")
                        packet = evaluator.evaluate(owner, lambda: target.CausalSpectralUNet(bottleneck_blocks=2))
                        path = out / ("development_step_"+str(owner.live.step)+".json")
                        q.obs.write_new(path, packet)
                        saved = json.loads(path.read_text(encoding="utf-8")); c.check_seal(saved)
                        require(c.equal(saved, packet), "Whole saved DEV packet own-seal/types")
                        owner.observe_stage(saved)
                        latest = checkpoint("step_"+str(owner.live.step)); phase = "training"; status("running")
            validate_exposure(exposure)
            require(owner.live.step == stream.cursor == 5000 and owner.shadow.updates == 500
                    and [r["step"] for r in owner.development_receipts] == [4750, 5000], "Actual full500 /all stages /independent EMA")
            allowed(); del model.forward
            decoder_journal.flush()
            q.obs.write_new(out / "audio_draw_receipts.json", q.p.seal_metadata({"purpose": PURPOSE, "draws": stream.draw_receipts}))
            outputs = {path.name: sha(path) for path in out.iterdir()
                       if path.name.startswith(("NONRELEASE_EMA_", "checkpoint_", "development_step_"))
                       or path.name in ("updates.jsonl", "decoder_native_journal.jsonl", "audio_draw_receipts.json")}
            report = q.p.seal_metadata({"purpose": PURPOSE, "device": "cuda", "step": 5000,
                "formal_training_updates": 500, "training_authorized": True, "single_training_Adam": True,
                "development_steps": [4750, 5000], "exposure": exposure, "actual_DEV_exposure": evaluator.exposure,
                "activation_sha256": sha(activation_path), "final_checkpoint": latest, "outputs_sha256": outputs,
                "actual_runtime": e.runtime_identity("cuda"), "decoder_request_count": len(decoder.rows),
                "decoder_native_journal_sha256": sha(out / "decoder_native_journal.jsonl"),
                "accepted_audio_draws": sum(row["completed"] for row in stream.draw_receipts),
                "release_selection": "NONE", "independent_acceptance_ready": False, "deployment": False})
            require(report["accepted_audio_draws"] == 500, "Exactly500 actual six-slot draws, no second training arm")
            q.obs.write_new(out / "completion.json", report)
            phase = "complete_fixed500"; status("complete")
            print("EMA_TRAIN_VERIFIED "+json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)
    except BaseException as error:
        if owner is not None:
            exposure["adam_started"], exposure["adam_completed"] = owner.exposure["adam_started"], owner.exposure["adam_completed"]
        status("failed", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        require(kernel.CloseHandle(supervisor), "Close read-only supervisor process handle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "worker", "status"))
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.operation == "status":
        print(json.dumps({"purpose": PURPOSE, "training_started": False, "requires_complete_activation": True}))
    else:
        require(args.activation is not None and args.out is not None, "Explicit complete activation and fresh output required")
        (train if args.operation == "train" else worker)(args.activation, args.out)
