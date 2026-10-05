"""Fresh representative maintenance-path A/B, NOT a real training step.

Two complete snapshots/seals, four bounded-LRU commitments, six microbatch
scalar reports, and synthetic raw/moment/shadow mutation. No music, source PT,
Module, forward/backward, optimizer construction/step, DEV or formal updates.
"""
from __future__ import annotations
import argparse
import ast
from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema234_state232", ROOT / "scripts/232_ema_fast_state_commitment.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
spec = importlib.util.spec_from_file_location("ema234_pcm228", ROOT / "scripts/228_ema_fast_hotpath.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def cache_functions(old, fast):
    f.require(p.sha(p.INPUT_PATH) == p.INPUT_SHA, "Exact original PCM source")
    tree = ast.parse(p.INPUT_PATH.read_text(encoding="utf-8"))
    pcm = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pcm_sha")
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AuthenticatedAudioStream")
    cache = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_cache_identity")
    ns = dict(require=f.require, torch=torch, hashlib=hashlib, OrderedDict=OrderedDict, portable=old.portable)
    exec(compile(ast.Module(body=[pcm, cache], type_ignores=[]), str(p.INPUT_PATH), "exec"), ns)
    baseline = ns["_cache_identity"]
    optimized = dict(ns, pcm_sha=p.fast_pcm_sha, portable=fast.portable)
    exec(compile(ast.Module(body=[cache], type_ignores=[]), "234[unchanged-cache-traversal]", "exec"), optimized)
    return baseline, optimized["_cache_identity"]


def fixture(device, cache_mib):
    # Original parameter count824900, synthetic22 equal-sized partitions.
    n = 824900
    values = torch.arange(n, dtype=torch.float32) / 1048576
    widths = [n // 22 + (i < n % 22) for i in range(22)]
    parts = torch.split(values, widths)
    def group(offset, target=device):
        return OrderedDict(("synthetic_parameter_" + str(i), (v + offset).to(target, copy=True)) for i, v in enumerate(parts))
    parent = {"arms": [{"raw": group(.125, "cpu"), "m": group(.25, "cpu"), "v": group(1., "cpu")}
                       for _ in range(2)], "selected_raw": group(.125, "cpu"),
              "source_is_synthetic": True, "source_TRAIN_step": None}
    state = {"raw": group(.125), "m": group(.25), "v": group(1.), "grad": group(.01), "shadow": group(.125),
             "parent": parent, "metadata": {"cursor": 0, "shadow_count": 0, "modes": [True] * 27,
                                            "LR": 0.00006281416799501188},
             "CPU_RNG": torch.get_rng_state().clone(), "CUDA_RNG": torch.cuda.get_rng_state().clone() if device == "cuda" else []}
    count = cache_mib * 1024**2 // 4
    template = torch.arange(count, dtype=torch.float32) / 1048576
    from types import SimpleNamespace
    cache = [template.clone() for _ in range(7)]
    backend = SimpleNamespace(true=SimpleNamespace(cache=OrderedDict(("true" + str(i), tuple(cache[2*i:2*i+2])) for i in range(3)),
                                                   signatures={"synthetic": ("only", 1, 2)}),
                              dataset=SimpleNamespace(cache=OrderedDict(pseudo=cache[-1])))
    scalars = [torch.tensor((i - 3) / 16384, dtype=torch.float32, device=device) for i in range(54)]
    scalars[0] = torch.tensor(-0., device=device)
    return state, backend, scalars


def mutate(state):
    # Manual storage mutation only; no optimizer or autograd execution.
    for name in state["raw"]:
        state["raw"][name].add_(.000001)
        state["m"][name].mul_(.9)
        state["v"][name].mul_(.999)
        state["shadow"][name].copy_(state["shadow"][name] * .99 + state["raw"][name] * .01)
    state["metadata"]["cursor"] = state["metadata"]["shadow_count"] = 1


def restore(state, before):
    for group in ("raw", "m", "v", "grad", "shadow"):
        for name in state[group]: state[group][name].copy_(before[group][name])
    state["metadata"] = dict(before["metadata"])


def path(functions, cache_fn, scalar_fn, state, backend, scalars, device):
    if device == "cuda": torch.cuda.synchronize()
    start = time.perf_counter(); breakdown = {}; caches = []
    tick = time.perf_counter(); before = functions.portable(state)
    first = functions.seal(before); functions.check_seal(first)
    breakdown["before_snapshot_and_seal"] = time.perf_counter() - tick
    tick = time.perf_counter(); caches += [cache_fn(backend), cache_fn(backend)]
    breakdown["before_cache_checks"] = time.perf_counter() - tick
    tick = time.perf_counter(); log = {str(i): 0. for i in range(9)}
    for slot in range(6):
        numbers = scalar_fn(scalars[slot*9:(slot+1)*9])
        for i, value in enumerate(numbers): log[str(i)] += value * (1/6)
    breakdown["six_microbatch_scalar_reports"] = time.perf_counter() - tick
    tick = time.perf_counter(); mutate(state)
    if device == "cuda": torch.cuda.synchronize()
    breakdown["synthetic_tensor_mutation"] = time.perf_counter() - tick
    tick = time.perf_counter(); after = functions.portable(state)
    second = functions.seal(after); functions.check_seal(second)
    breakdown["after_snapshot_and_seal"] = time.perf_counter() - tick
    tick = time.perf_counter(); caches += [cache_fn(backend), cache_fn(backend)]
    breakdown["after_cache_checks"] = time.perf_counter() - tick
    elapsed = time.perf_counter() - start
    return elapsed, breakdown, (first, second, caches, [value.hex() for value in log.values()]), before


def run_device(device, out, cache_mib):
    old, fast = f.baseline_functions(), f.optimized_functions()
    original_cache, optimized_cache = cache_functions(old, fast)
    state, backend, scalars = fixture(device, cache_mib)
    origin = old.portable(state)
    samples = {"baseline": [], "optimized": []}; stages = {"baseline": [], "optimized": []}; expected = None
    scalar_samples = {"baseline": [], "optimized": []}
    # Small isolated logging benchmark: 54 scalars, 6reports, 10cycles/sample.
    for repeat in range(3):
        order = ("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")
        for name in order:
            fn = (lambda xs: [float(v.detach()) for v in xs]) if name == "baseline" else f.read_scalars
            if device == "cuda": torch.cuda.synchronize()
            tick = time.perf_counter()
            for _ in range(10):
                result = [fn(scalars[slot*9:(slot+1)*9]) for slot in range(6)]
            if device == "cuda": torch.cuda.synchronize()
            scalar_samples[name].append((time.perf_counter()-tick)/10)
            f.require([[v.hex() for v in row] for row in result] == [[float(v).hex() for v in scalars[slot*9:(slot+1)*9]] for slot in range(6)],
                      "Exact scalar values including signed zero")
    for repeat in range(3):
        order = ("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")
        for name in order:
            restore(state, origin)
            funcs, cache_fn = (old, original_cache) if name == "baseline" else (fast, optimized_cache)
            scalar_fn = (lambda xs: [float(v.detach()) for v in xs]) if name == "baseline" else f.read_scalars
            elapsed, breakdown, result, before = path(funcs, cache_fn, scalar_fn, state, backend, scalars, device)
            commitment = old.digest(result)
            if expected is None: expected = commitment
            f.require(commitment == expected, "Complete before/after states,own seals,cache/log metadata differ")
            restore(state, before)
            f.require(old.equal(old.portable(state), origin), "Whole synthetic state restoration differs")
            # Each snapshot remains independent of current live raw storage.
            for name2 in state["raw"]:
                f.require(before["raw"][name2].untyped_storage().data_ptr() != state["raw"][name2].untyped_storage().data_ptr(), "Snapshot aliases live raw")
            samples[name].append(elapsed); stages[name].append(breakdown)
            print("NEW234 " + device + " round=" + str(repeat) + " " + name + " seconds=" + str(elapsed), flush=True)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    scalar_med = {name: statistics.median(values) for name, values in scalar_samples.items()}
    row = {"device": device, "cache_leaf_MiB": cache_mib, "cache_total_MiB": cache_mib*7,
           "synthetic_parameters": 824900, "synthetic_parameter_leaves": 22, "snapshots_per_sample": 2,
           "full_cache_checks_per_sample": 4, "samples_seconds": samples, "stage_seconds": stages,
           "median_seconds": medians, "maintenance_path_speedup": medians["baseline"]/medians["optimized"],
           "scalar_report_samples_seconds": scalar_samples, "scalar_report_median_seconds": scalar_med,
           "scalar_report_speedup": scalar_med["baseline"]/scalar_med["optimized"],
           "whole_before_after_packet_commitment": expected, "full_typed_seals_equal": True,
           "noalias_and_restoration_verified": True, "real_model_Adam_updates": 0}
    with (out / (device + "_benchmark.json")).open("x", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=True, indent=2, allow_nan=False)
    return row


def benchmark(out, use_cuda, cache_mib):
    f.require(out.parent == ROOT / "results" and not out.exists(), "Fresh output,no repeat/overwrite")
    f.require(shutil.disk_usage(ROOT).free >= 12*1024**3 and 1 <= cache_mib <= 64, "Disk/cache resource limits")
    out.mkdir()
    threads = torch.get_num_threads(); torch.set_num_threads(4)
    cpu_rng = torch.get_rng_state().clone(); py_rng = random.getstate(); np_rng = np.random.get_state()
    rows = []; cuda_rng = None; free = None
    try:
        rows.append(run_device("cpu", out, cache_mib))
        if use_cuda:
            torch.cuda.init(); free, _ = torch.cuda.mem_get_info()
            f.require(free >= 2300*1024**2, "Insufficient free VRAM;do not kill/shared-program/fallback")
            cuda_rng = torch.cuda.get_rng_state_all()
            rows.append(run_device("cuda", out, cache_mib))
            f.require(all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())), "CUDA benchmark RNG changed")
        now = np.random.get_state()
        f.require(torch.equal(cpu_rng, torch.get_rng_state()) and py_rng == random.getstate()
                  and now[0] == np_rng[0] and np.array_equal(now[1], np_rng[1]) and now[2:] == np_rng[2:], "CPU/Python/NumPy RNG changed")
        result = {"purpose": "NONRELEASE_EMA234_SYNTHETIC_MAINTENANCE_PATH_PERFORMANCE",
                  "source_bindings": {str(f.SOURCE): f.SOURCE_SHA, str(p.INPUT_PATH): p.INPUT_SHA,
                        str(ROOT / "scripts/232_ema_fast_state_commitment.py"): f.sha(ROOT / "scripts/232_ema_fast_state_commitment.py"),
                        str(ROOT / "scripts/228_ema_fast_hotpath.py"): f.sha(ROOT / "scripts/228_ema_fast_hotpath.py"),
                        str(Path(__file__).resolve()): f.sha(__file__)},
                  "benchmarks": rows, "initial_free_CUDA_MiB": None if free is None else free/1024**2,
                  "CUDA_initialized": use_cuda, "RNG_unchanged": True,
                  "real_music_source_PT_Module_forward_backward_Adam_updates": 0,
                  "synthetic_manual_mutation_samples": len(rows)*6,
                  "training_authorized": False, "release_selection": "NONE",
                  "actual_training_step_speedup": "UNMEASURED",
                  "limits": "Synthetic maintenance path only: no decoder/source-file lease/network/loss/DEV/checkpointIO;not whole-training speedup"}
        result["content_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=True,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with (out / "benchmark.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=True, indent=2, allow_nan=False)
        print(json.dumps({"benchmarks": [{k: r[k] for k in ("device", "median_seconds", "maintenance_path_speedup",
                              "scalar_report_median_seconds", "scalar_report_speedup")} for r in rows],
                              "content_sha256": result["content_sha256"], "actual_training_step_speedup": "UNMEASURED"}, indent=2), flush=True)
    finally:
        torch.set_num_threads(threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    parser.add_argument("--cuda", action="store_true"); parser.add_argument("--cache-mib", type=int, default=32)
    args = parser.parse_args(); benchmark(Path(args.out).resolve(), args.cuda, args.cache_mib)
