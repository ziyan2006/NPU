"""Decoder launch REQUEST inspection only, never native process creation.

Extracts only reviewed AST argument expressions without importing old audio
modules. Captures the installed Windows CPython request BEFORE CreateProcess.
An explicit target policy is prospective; neither capture authenticates loaded
images/DLLs or upgrades208/209 into a complete runtime/training gate.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA_DECODER_LAUNCH_REQUEST_NOT_NATIVE_LAUNCH"
PINS = {
    "scripts/209_ema_decoder_target_guard.py": "6b904f9f1b8497d16f3ecca85d9122b000604f4e357456eef0a0e9ddf28fe7d2",
    "scripts/110_train_residual_ablation.py": "5476a5b6e38ed7b90044604e8f0e841ef46cc3ebcd6daa4249226eabce21bc07",
    "scripts/134_generate_teacher_library.py": "fc136fc8250f3f788bdaf4c8d2501a13c2ecef213d38d5f1748b5da95cfa38f3",
    "scripts/131_run_teacher_pilot.py": "99a2f17a7787a7d89d259b5532d5cbee635adfdb92c000352f086b795fad48bb",
    "reports/109_mel_ema_decoder_target_guard_progress.md": "d2985d414ff00f53473f5ae08fdc8026c58c559d935eb6ae4fc784beb1072e14",
    "results/mel_ema_decoder_target_monitor_20261004/unit_gate.json": "56f6564c1246281d014e455829185c957e145342af9349807022324bff0c6776",
    "results/mel_ema_decoder_target_monitor_20261004/progress.json": "73643400b07e6cbc8dfdd2d2f1d7ed1ffb63966371cd2ce79a236ea29e6bebdb",
}
RUNTIME_PINS = {
    r"C:\Users\30519\.workbuddy\binaries\python\versions\3.13.12\Lib\subprocess.py": "e4049ed91ed101b09b0dc96bbf234f646c1728090d643a474b024276cc2a028c",
    r"C:\Users\30519\.workbuddy\binaries\python\versions\3.13.12\python.exe": "ec8139feb5012b12a531196f257fddf53668f0109918d5d44ffa195cf352fdd6",
    r"C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe": "8c1ccc97417ad2a8cd7e6106d9f35192c685b1b035366590ff0d0ba525703caa",
}


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError("Changed launch-request dependency: " + str(path))


verify_bindings({str(ROOT / p): h for p, h in PINS.items()})
spec = importlib.util.spec_from_file_location("ema210_typed209_only", ROOT / "scripts/209_ema_decoder_target_guard.py")
meta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta)  # Pure stdlib helpers, NOT a guard construction.
require, typed, seal, check_seal = meta.require, meta.typed, meta.seal, meta.check_seal


def _value(node, names):
    """Small non-executing AST evaluator, not eval/exec of source functions."""
    if isinstance(node, ast.Constant) and type(node.value) in (str, int):
        return node.value
    if isinstance(node, ast.List):
        return [_value(v, names) for v in node.elts]
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "pilot" and node.attr == "SR":
        return names["pilot_SR"]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1 and not node.keywords:
        value = _value(node.args[0], names)
        require(type(value) in (str, int), "Only primitive str conversion")
        return str(value)
    raise ValueError("Unreviewed launch argument AST: " + type(node).__name__)


def _function(source, name):
    matches = [v for v in ast.parse(source).body if isinstance(v, ast.FunctionDef) and v.name == name]
    require(len(matches) == 1 and not matches[0].decorator_list, "Unique undecorated original function")
    return matches[0]


def _calls(function):
    return sorted((v for v in ast.walk(function) if isinstance(v, ast.Call) and
                   isinstance(v.func, ast.Attribute) and isinstance(v.func.value, ast.Name) and
                   v.func.value.id == "subprocess" and v.func.attr == "check_output"), key=lambda v: v.lineno)


def original_requests(path):
    require(type(path) is str and path and "\0" not in path, "Metadata path string")
    verify_bindings({str(ROOT / p): h for p, h in PINS.items()})
    core = (ROOT / "scripts/110_train_residual_ablation.py").read_text(encoding="utf-8-sig")
    bulk = (ROOT / "scripts/134_generate_teacher_library.py").read_text(encoding="utf-8-sig")
    pilot = ast.parse((ROOT / "scripts/131_run_teacher_pilot.py").read_text(encoding="utf-8-sig"))
    sample_rates = [ast.literal_eval(v.value) for v in pilot.body if isinstance(v, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == "SR" for t in v.targets)]
    require(typed(sample_rates) == typed([44100]), "Original pilot sample rate")
    musdb = _function(core, "decode_musdb")
    graphs = [v.value for v in musdb.body if isinstance(v, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id == "graph" for t in v.targets)]
    require(len(graphs) == 1, "Original one graph expression")
    names = {"path": path, "SR": 44100, "pilot_SR": 44100, "graph": _value(graphs[0], {})}
    groups = [("musdb", musdb, 2), ("pseudo_probe", _function(bulk, "probe"), 1),
              ("pseudo_decode", _function(bulk, "decode"), 1), ("version", _function(bulk, "media_versions"), 1)]
    result = []
    for label, function, count in groups:
        calls = _calls(function)
        require(len(calls) == count, "Exact original call count")
        for index, call in enumerate(calls):
            require(len(call.args) == 1 and all(k.arg == "timeout" for k in call.keywords), "Original no shell/executable overrides")
            for version_name in (("ffmpeg", "ffprobe") if label == "version" else (None,)):
                env = names | {"name": version_name}
                argv = _value(call.args[0], env)
                kwargs = {k.arg: _value(k.value, env) for k in call.keywords}
                require(type(argv) is list and all(type(v) is str and "\0" not in v for v in argv), "Exact string argv")
                require(argv[0] in meta.EXPECTED, "Reviewed decoder only")
                result.append({"label": label + str(index) + (version_name or ""), "source_line": call.lineno,
                               "argv": argv, "kwargs": kwargs})
    require(len(result) == 6, "Six distinct command shapes, not six audio draws")
    return result


def explicit_request(original):
    require(type(original) is dict and set(original) == {"label", "source_line", "argv", "kwargs"}, "Original request schema")
    argv, kwargs = original["argv"], original["kwargs"]
    require(type(argv) is list and argv and all(type(v) is str and "\0" not in v for v in argv), "Exact argv strings")
    require(argv[0] in meta.EXPECTED and type(kwargs) is dict and set(kwargs) <= {"timeout"}, "No unknown command/options")
    require(all(type(v) is int and v > 0 for v in kwargs.values()), "Positive integer timeout")
    require(type(original["label"]) is str and type(original["source_line"]) is int, "Typed provenance")
    target = meta.EXPECTED[argv[0]]["target"]
    require(Path(target).is_absolute() and Path(target).suffix.lower() == ".exe", "Explicit absolute EXE")
    return {"label": original["label"], "source_line": original["source_line"],
            "argv": [target, *argv[1:]], "kwargs": copy.deepcopy(kwargs), "executable": target,
            "shell": False, "close_fds": True, "creationflags": subprocess.CREATE_NO_WINDOW,
            "startupinfo": {"dwFlags": subprocess.STARTF_USESHOWWINDOW, "wShowWindow": subprocess.SW_HIDE},
            "env": None, "cwd": None, "actual_spawn_authorized": False}


class BeforeNativeCreate(BaseException):
    """Raised by replacement function; original native CreateProcess NOT called."""


def capture_request(argv, kwargs, *, policy=None):
    require(os.name == "nt" and subprocess._mswindows, "Installed Windows CPython only")
    require(type(argv) is list and argv and all(type(v) is str and "\0" not in v for v in argv), "Exact argv")
    require(type(kwargs) is dict and set(kwargs) <= {"timeout"}, "Only original check_output timeout")
    require(all(type(v) is int and v > 0 for v in kwargs.values()), "Typed timeout")
    require(policy is None or typed(policy) == typed(explicit_request({"label": policy["label"],
            "source_line": policy["source_line"], "argv": [Path(policy["executable"]).stem, *argv[1:]],
            "kwargs": kwargs})), "Unchanged prospective policy")
    if policy is not None:
        require(typed(argv) == typed(policy["argv"]), "Explicit argv identity")
    import _winapi
    native_before = _winapi.CreateProcess
    captured = []

    def stop(application, command, psec, tsec, inherit, flags, env, cwd, startup):
        captured.append({"application_name": application, "command_line": command,
                         "process_security": psec, "thread_security": tsec,
                         "inherit_handles": inherit, "creationflags": flags, "environment": env,
                         "cwd": cwd, "startupinfo": {"dwFlags": startup.dwFlags,
                         "wShowWindow": startup.wShowWindow,
                         "hStdInput": None if startup.hStdInput is None else int(startup.hStdInput),
                         "hStdOutput": None if startup.hStdOutput is None else int(startup.hStdOutput),
                         "hStdError": None if startup.hStdError is None else int(startup.hStdError),
                         "lpAttributeList": copy.deepcopy(startup.lpAttributeList)}})
        raise BeforeNativeCreate("REQUEST_CAPTURE_ONLY_NO_NATIVE_CHILD")

    call_kwargs = copy.deepcopy(kwargs)
    if policy is not None:
        startup = subprocess.STARTUPINFO()
        startup.dwFlags, startup.wShowWindow = policy["startupinfo"]["dwFlags"], policy["startupinfo"]["wShowWindow"]
        call_kwargs.update(executable=policy["executable"], shell=False, close_fds=True,
                           creationflags=policy["creationflags"], startupinfo=startup, env=None, cwd=None)
    with patch.object(_winapi, "CreateProcess", stop), patch.object(os, "system", side_effect=AssertionError("No os.system")):
        try:
            subprocess.check_output(copy.deepcopy(argv), **call_kwargs)
        except BeforeNativeCreate:
            pass
        else:
            raise ValueError("Missing pre-native interruption")
    require(_winapi.CreateProcess is native_before and len(captured) == 1, "Single intercepted request and restored native binding")
    typed(captured[0])
    return captured[0]


def inspect_installed_requests():
    """Twelve intercepted requests; child spawn/audio/loaded-image counts ZERO."""
    bindings = {str(ROOT / p): h for p, h in PINS.items()} | RUNTIME_PINS
    bindings.update({v["target"]: v["target_sha256"] for v in meta.EXPECTED.values()})
    for relative in ("scripts/210_ema_decoder_launch_request.py", "scripts/_test_ema_decoder_launch_request.py",
                     "docs/ema_decoder_launch_request_unit_scope_20261004.json"):
        bindings[str(ROOT / relative)] = sha(ROOT / relative)
    require(str(Path(subprocess.__file__)) in RUNTIME_PINS, "Actual subprocess path pinned")
    require(sys.executable == r"C:\Users\30519\.workbuddy\binaries\python\envs\stem-npu\Scripts\python.exe", "Actual entry Python")
    verify_bindings(bindings)
    before_rng, before_env = random.getstate(), dict(os.environ)
    requests = original_requests(r"D:\EMA_METADATA_ONLY\NONEXISTENT 音频 & %!.mp3")
    rows = []
    for original in requests:
        policy = explicit_request(original)
        old = capture_request(original["argv"], original["kwargs"])
        new = capture_request(policy["argv"], policy["kwargs"], policy=policy)
        require(old["application_name"] is None and old["command_line"] == subprocess.list2cmdline(original["argv"]), "Actual original NULL application request")
        require(new["application_name"] == policy["executable"] and new["command_line"] == subprocess.list2cmdline(policy["argv"]), "Actual explicit application request")
        require(typed(policy["argv"][1:]) == typed(original["argv"][1:]) and typed(policy["kwargs"]) == typed(original["kwargs"]), "All original non-program tokens/timeouts preserved")
        rows.append(seal({"original": original, "prospective_policy": policy,
                          "intercepted_original": old, "intercepted_explicit": new}))
    verify_bindings(bindings)
    require(before_rng == random.getstate() and before_env == dict(os.environ), "Python RNG/environment unchanged")
    require(not any(p in sys.modules for p in ("torch", "numpy", "scipy", "soundfile")), "No audio/model/native numerical imports")
    return seal({"schema": 1, "purpose": PURPOSE, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings_sha256": bindings, "python_version": sys.version, "python_executable": sys.executable,
        "base_executable": sys._base_executable, "subprocess_file": subprocess.__file__, "rows": rows,
        "request_interceptions": 12, "native_CreateProcess_calls": 0, "actual_child_processes": 0,
        "actual_audio_draws": 0, "model_or_optimizer_or_pt_load": 0, "cuda_initialized": False,
        "python_rng_and_environment_unchanged": True, "target_bytes_checked_before_and_after_requests_only": True,
        "targets_held_during_requests": False, "historical208_actual_image_path_observed": False,
        "explicit_policy_actual_spawn_verified": False, "runtime_transitive_loaded_modules_verified": False,
        "full_audio_backend_gate": "PENDING", "training_authorized": False, "release_selection": "NONE"})


def require_audio_runtime_authority():
    raise ValueError("REQUEST_ONLY: actual loaded-image/native dependency certification still PENDING")
