"""Export one fresh CPU-only raw5000/EMA5000 listening pack.

Reuses exact SHA-bound original model/frontend/context definitions through AST.
4500 and frozen comparison audio is read from the existing listening archive,
not inferred again. No trainer, optimizer, decoder subprocess or CUDA entry.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/mel_ema_single_trajectory_20261005"
OLD = ROOT / "results/mel_lr_scale_review_20261004/listen_step_4500"
OUT = ROOT / "results/mel_ema_listening_20261005/step_5000"
PURPOSE = "NONRELEASE_EMA238_CPU_LISTENING_ONLY"
SR = 44100
PINS = {
    "scripts/09_target_model.py": "e7fd2833a89b97b2355260beb125fd3ab72b06e040c82aa4c962945b7bb87b81",
    "scripts/110_train_residual_ablation.py": "5476a5b6e38ed7b90044604e8f0e841ef46cc3ebcd6daa4249226eabce21bc07",
    "scripts/128_acquire_cambridge_candidates.py": "4d82316fe483c87e21fae654a87fb90f79e986c633f1698f143272dee237e037",
    "scripts/131_run_teacher_pilot.py": "99a2f17a7787a7d89d259b5532d5cbee635adfdb92c000352f086b795fad48bb",
    "scripts/143_paired_distillation_mechanics.py": "bb3ad26dcf5c94e3909f8c4f05c9fb4f249e1a64b03f8b3d3155a26619ebddab",
    "scripts/148_diagnose_full_source_audio.py": "0925eeda3217ac6926d328fa16e638c9ee74051b6e3f0d6fff23cf598f531b69",
    "scripts/152_review_paired_exploration.py": "95bd39e58fefb5aae1761314cafafd09367591a22bfc2afd12ffed4cecabc6ec",
    "scripts/196_review_mel_lr_scale.py": "8ad6d934074da19ebf42883bd29463f07ccee9424e9a454f61a32492633df5ba",
    "scripts/205_ema_cpu_state_transaction.py": "8a8f2da566b664abfb19e6c24fe5fb4bf78bd61c13408ebf385b5cf425cd90f5",
    "results/full_source_diagnostic_20261002/plan.json": "6ed9734fa5f90477fa8f8f1c3004c6abf3c65cf8eb99b58b2f004697f870feac",
    "results/full_source_diagnostic_20261002/diagnostic.json": "3fc21f827d8f21b8477777c230df7df9eedb326cb119b2e1fd12751fd7b0aded",
    "results/mel_lr_scale_review_20261004/listen_step_4500/plan.json": "9ff050d2eefd5e975e48c2c13ab91d8677a4397b51adaebf5039af9d8fd15e2d",
    "results/mel_lr_scale_review_20261004/listen_step_4500/review.json": "0a7ccdd9e7eca1900802d8bd3ea63b990440dfc3eb9f14062695022716952755",
    "results/mel_ema_single_trajectory_20261005/completion.json": "2cc020104fc53198197acb62a6f925368012882888204080cef65764feb2ac85",
    "results/mel_ema_single_trajectory_20261005/development_step_5000.json": "e1989f55619e152e5f4259d5adcd1b24311941e39c7b00873d5cd18cd6eda5e2",
    "results/mel_ema_single_trajectory_20261005/NONRELEASE_EMA_step_5000.pt": "d2be133da5bdd778fb615ef28167bf263e130a819596c6630e8c3a8726c1ab91",
}
TRACKS = ["A Classic Education - NightOwl.stem", "abjones_2", "Beneath_v1"]
LABELS = {
    "mix": "原混音", "reference_backing": "真实参考伴奏", "reference_vocal": "真实参考人声",
    "source4500_backing": "4500 起点伴奏", "source4500_vocal": "4500 起点人声",
    "frozen_backing": "冻结基线伴奏", "frozen_vocal": "冻结基线人声",
    "raw5000_backing": "5000 raw 伴奏", "raw5000_vocal": "5000 raw 人声",
    "ema5000_backing": "5000 EMA 伴奏", "ema5000_vocal": "5000 EMA 人声",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def original(relative, names, namespace, constants=()):
    """Compile only unchanged selected definitions, never original top-level code."""
    path = ROOT / relative
    require(sha(path) == PINS[relative], "Changed original source: " + relative)
    selected, found = [], []
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(copy.deepcopy(node)); found.append(node.name)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants
            or isinstance(target, ast.Tuple) and any(isinstance(x, ast.Name) and x.id in constants for x in target.elts)
            for target in node.targets
        ):
            selected.append(copy.deepcopy(node))
    require(set(found) == set(names) and len(found) == len(names), "All exact requested definitions required")
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def primitives():
    base = dict(torch=torch, np=np, nn=nn, F=F, math=math, hashlib=hashlib,
                json=json, Path=Path, OrderedDict=OrderedDict, require=require)
    c = original("scripts/205_ema_cpu_state_transaction.py",
                 ("portable", "typed_tree", "digest", "equal", "seal", "check_seal"), dict(base))
    acq = original("scripts/128_acquire_cambridge_candidates.py",
                   ("sha256", "content_digest", "read_sealed"), dict(base))
    pilot = original("scripts/131_run_teacher_pilot.py", ("wave_digest",), dict(base))
    model = original("scripts/09_target_model.py",
        ("band_edges", "_mel_points", "band_centers", "_center_interpolation", "make_analysis_matrix",
         "make_synthesis_matrix", "CausalConv2d", "CausalResidualBlock", "CausalDepthwiseTemporalBlock",
         "CausalSpectralUNet", "_stft", "_istft"), dict(base, SR=SR),
        ("N_FFT", "HOP", "N_BINS", "N_BANDS", "FMIN", "FMAX", "DEFAULT_BAND_LAYOUT"))
    core = original("scripts/110_train_residual_ablation.py", ("stft_batch", "product_vocal"), dict(base, t09=model))
    core.t09 = model
    m = original("scripts/143_paired_distillation_mechanics.py",
                 ("capture_rng", "restore_rng", "equal_state"), dict(base, random=random))
    m.core = core
    context = original("scripts/148_diagnose_full_source_audio.py", ("context_masks", "save_wave"), dict(base, sf=sf, SR=SR))
    r = SimpleNamespace(F=F, a=context)
    infer = original("scripts/196_review_mel_lr_scale.py", ("vocal_wave",),
                     dict(base, m=m, r=r, t=SimpleNamespace(d=SimpleNamespace(portable=c.portable))))
    sources = original("scripts/152_review_paired_exploration.py", ("checked_sources",),
        dict(base, sf=sf, acq=acq, a=SimpleNamespace(DEFAULT_OUT=ROOT / "results/full_source_diagnostic_20261002"),
             SOURCE_PLAN_SHA=PINS["results/full_source_diagnostic_20261002/plan.json"],
             SOURCE_REPORT_SHA=PINS["results/full_source_diagnostic_20261002/diagnostic.json"], LISTEN_SECONDS=20))
    return SimpleNamespace(c=c, acq=acq, pilot=pilot, model=model, m=m, infer=infer, context=context, sources=sources)


def check_metadata214(doc, c):
    """Exact214 metadata types match205 for JSON primitives; ASCII serialization differs."""
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    digest = hashlib.sha256(json.dumps(c.typed_tree(body), ensure_ascii=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(digest == doc.get("content_sha256"), "Own214 metadata seal changed")


def checked_wav(path, expected, frames=None):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "Changed archived audio: " + str(path))
    info = sf.info(io.BytesIO(data))
    require((info.samplerate, info.channels, info.subtype) == (SR, 2, "FLOAT"), "Original FLOAT32 stereo44100 WAV")
    require(frames is None or info.frames == frames, "Changed prefix duration")
    array, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wave = torch.from_numpy(array.T.copy())
    require(rate == SR and bool(torch.isfinite(wave).all()), "Finite archived PCM")
    return wave


def common_playback(waves):
    require(set(waves) == set(LABELS), "All11 aligned listening variants")
    shape = waves["mix"].shape
    require(len(shape) == 2 and shape[0] == 2, "Stereo outputs")
    require(all(w.shape == shape and w.dtype == torch.float32 and w.device.type == "cpu"
                and bool(torch.isfinite(w).all()) for w in waves.values()), "Matched finite FP32 CPU waves")
    peak = max(float(w.abs().max()) for w in waves.values())
    gain = min(1., .95 / max(peak, 1e-12))
    return {name: value * gain for name, value in waves.items()}, gain


def fresh_output(path):
    path = Path(path).resolve()
    require(path == OUT.resolve(), "Only the new fixed5000 listening directory is writable")
    require(not path.exists(), "Existing export/evidence must not be overwritten or rerun")
    require(shutil.disk_usage(ROOT / "results").free >= 12 * 1024**3 + 256 * 1024**2, "12GiB reserve plus audio budget")


def write_json(path, doc, c):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(c.seal(doc), stream, ensure_ascii=False, allow_nan=False, indent=2)


def page(records):
    sections = []
    order = ("mix", "source4500_backing", "raw5000_backing", "ema5000_backing", "reference_backing",
             "frozen_backing", "raw5000_vocal", "ema5000_vocal", "reference_vocal", "source4500_vocal", "frozen_vocal")
    for row in records:
        players = "".join(f'<div><b>{html.escape(LABELS[name])}</b><audio controls preload="none" src="{html.escape(row["audio"][name])}"></audio></div>' for name in order)
        sections.append(f'<section><h2>{row["index"]}. {html.escape(row["track_id"])}</h2><p>{row["seconds"]:.2f} 秒 · {html.escape(row["domain"])}</p><div class="grid">{players}</div></section>')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5000 步 raw / EMA 试听</title>
<style>body{font:16px system-ui;background:#f5f5f3;color:#202322;max-width:1080px;margin:32px auto;padding:0 20px}section{background:white;border-radius:12px;padding:20px;margin:22px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}audio{display:block;width:100%;margin-top:8px}p{line-height:1.6;color:#555}</style>
<h1>5000 步 raw / EMA 试听对照</h1><p>全部版本每首使用相同输入和共同增益；不单独归一化。4500 与冻结基线直接复用旧导出，未重新推理。先听伴奏中的残留、人声被误删的乐器及接缝。MIR 是拼接片段，不是完整连续歌曲。</p><p>本地 CPU FP32 参考 · NONRELEASE · 非盲测/板端验收。请只同时播放一个版本。</p>''' + "".join(sections) + '''<script>document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'''


def export(out):
    fresh_output(out)
    bindings = {str((ROOT / rel).resolve()): expected for rel, expected in PINS.items()}
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    for path, expected in bindings.items():
        require(sha(path) == expected, "Pinned export input changed: " + path)
    q = primitives(); c = q.c
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(not torch.cuda.is_initialized(), "Fresh CPU-only listening process")
    outer_rng = q.m.capture_rng("cpu")
    completion = json.loads((RUN / "completion.json").read_text(encoding="utf-8"))
    check_metadata214(completion, c)
    require(completion["step"] == 5000 and completion["formal_training_updates"] == 500
            and completion["release_selection"] == "NONE", "Only finished fixed5000 stage")
    score = json.loads((RUN / "development_step_5000.json").read_text(encoding="utf-8")); c.check_seal(score)
    receipt = completion["final_checkpoint"]
    check_metadata214(receipt, c)
    require(receipt["step"] == 5000 and receipt["pending_DEV"] is False
            and receipt["checkpoint"] == "NONRELEASE_EMA_step_5000.pt"
            and receipt["sha256"] == PINS["results/mel_ema_single_trajectory_20261005/NONRELEASE_EMA_step_5000.pt"],
            "Complete nested final checkpoint receipt")
    raw_data = (RUN / receipt["checkpoint"]).read_bytes()
    require(hashlib.sha256(raw_data).hexdigest() == PINS["results/mel_ema_single_trajectory_20261005/NONRELEASE_EMA_step_5000.pt"], "SHA before PT deserialize")
    saved = torch.load(io.BytesIO(raw_data), map_location="cpu", weights_only=True)
    c.check_seal(saved); trajectory = saved["trajectory"]; c.check_seal(trajectory)
    saved_digest = c.digest(saved)
    require(saved["purpose"] == "NONRELEASE_EMA219_CONTINUOUS_SINGLE_TRAJECTORY"
            and saved["scope"] == "activated_single_raw_original_Adam_CUDA_fixed500"
            and trajectory["raw"]["updates"] == 5000 and trajectory["live_context"]["context"]["step"] == 5000
            and saved["release_selection"] == "NONE", "Complete5000 snapshot, not a trainer resume")
    require(c.equal(saved["development_receipts"][-1], score), "Whole committed DEV receipt matches file")
    require(score["model_digest"] == {"raw": c.digest(trajectory["raw"]["tensors"]),
                                      "ema": c.digest(trajectory["shadow"]["tensors"])}, "Raw/EMA weights bound to saved score")
    sources, source_files = q.sources.checked_sources()
    require([record["track_id"] for record, _, _, _ in sources] == TRACKS, "Original preselected songs only")
    bindings.update(source_files)
    old = q.acq.read_sealed(OLD / "review.json"); old_plan = q.acq.read_sealed(OLD / "plan.json")
    require(old["plan_sha256"] == sha(OLD / "plan.json") and old["listening_step"] == 4500
            and old_plan["step"] == 4500, "Original4500 listening archive")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", {"purpose": PURPOSE, "step": 5000, "checkpoint_sha256": hashlib.sha256(raw_data).hexdigest(),
        "models_digest": score["model_digest"], "sources": TRACKS, "bindings_sha256": bindings,
        "frontend": "original196 FP32 legacy_log LF32/history128/block256; original frozen LF44 reused",
        "selection": "original3 DEVELOPMENT sources, unchanged first min20s; no score-based selection",
        "training_started": False, "cuda_used": False, "release_selection": "NONE"}, c)
    rows, output_hashes, counts = [], {}, {"raw": 0, "ema": 0}
    def forbidden(*args, **kwargs):
        raise RuntimeError("Listening must not initialize CUDA, construct Adam or invoke autograd")
    try:
        with patch.object(torch.cuda, "_lazy_init", forbidden), patch.object(torch.optim.Adam, "__init__", forbidden), \
             patch.object(torch.autograd, "backward", forbidden), patch.object(torch.autograd, "grad", forbidden), torch.no_grad():
            models = {}
            for role, source in (("raw", trajectory["raw"]), ("ema", trajectory["shadow"])):
                model = q.model.CausalSpectralUNet(2, (32, 64, 96, 128), 4, bottleneck_blocks=2, temporal_dilations=())
                model.frontend, model.mask_mode, model.band_layout, model.n_bands = ("linear", 1.), "independent", "legacy_log", 128
                require(len(list(model.parameters())) == 22 and len(list(model.modules())) == 27 and not list(model.buffers()), "Exact original graph")
                model.load_state_dict(c.portable(source["tensors"]), strict=True)
                require(c.digest(model.state_dict()) == score["model_digest"][role], "Copied inference weights differ")
                modes = source["modes"] if role == "raw" else source["modes_at_copy"]
                require(len(modes) == 27 and all(type(x) is bool for x in modes), "Full original mode snapshot")
                for module, mode in zip(model.modules(), modes): module.training = mode
                def count(instance, inputs, key=role): counts[key] += 1
                model.register_forward_pre_hook(count)
                models[role] = model
            for index, (record, old_source, mix, ref) in enumerate(sources, 1):
                print(f"EMA_LISTEN source={index}/3 {record['track_id']}", flush=True)
                prior = old["listening_records"][index - 1]
                require(prior["track_id"] == record["track_id"] and prior["input_pcm_sha256"] == q.pilot.wave_digest(mix), "Exact original listening PCM/prefix")
                old_gain = prior["additional_common_playback_gain"]
                require(type(old_gain) is float and 0 < old_gain <= 1, "Original common playback gain")
                waves = {"mix": mix * old_gain, "reference_vocal": ref * old_gain, "reference_backing": (mix - ref) * old_gain}
                for name in ("mix", "reference_vocal", "reference_backing", "lr1_control_vocal", "lr1_control_backing", "frozen_vocal", "frozen_backing"):
                    filename = prior["audio"][name]; path = OLD / filename; expected = old["audio_sha256"][filename]
                    previous = checked_wav(path, expected, mix.shape[-1]); bindings[str(path.resolve())] = expected
                    if name in waves:
                        require(c.equal(previous, waves[name]), "Existing reference PCM differs after original gain")
                    else: waves[name.replace("lr1_control", "source4500")] = previous
                for role, model in models.items():
                    before = c.digest(model.state_dict()); modes = [mod.training for mod in model.modules()]
                    vocal = q.infer.vocal_wave(model, mix, 32)
                    require(c.digest(model.state_dict()) == before and [mod.training for mod in model.modules()] == modes
                            and all(p.grad is None for p in model.parameters()), "Listening changed inference model/modes/grad")
                    waves[role + "5000_vocal"] = vocal * old_gain
                    waves[role + "5000_backing"] = (mix - vocal) * old_gain
                waves, extra_gain = common_playback(waves)
                files = {}
                for name, wave in waves.items():
                    path = out / f"source_{index:02d}_{name}.wav"
                    q.context.save_wave(path, wave)
                    expected = sha(path); reloaded = checked_wav(path, expected, wave.shape[-1])
                    require(c.equal(reloaded, wave) and float(wave.abs().max()) <= .950001, "WAV roundtrip/no-clipping verification")
                    files[name] = path.name; output_hashes[path.name] = expected
                rows.append({"index": index, "track_id": record["track_id"], "domain": record["domain"], "samples": mix.shape[-1],
                    "seconds": mix.shape[-1] / SR, "input_pcm_sha256": q.pilot.wave_digest(mix), "original_common_gain": old_gain,
                    "additional_all_variants_gain": extra_gain, "audio": files,
                    "source_join_samples_in_prefix": prior["source_join_samples_in_prefix"]})
                print(f"EMA_LISTEN source={index} saved11 common_gain={old_gain * extra_gain:.9g}", flush=True)
        require(c.digest(saved) == saved_digest and not torch.cuda.is_initialized(), "Source checkpoint changed or CUDA initialized")
        for path, expected in bindings.items(): require(sha(path) == expected, "Post-export source changed: " + path)
        with (out / "index.html").open("x", encoding="utf-8") as stream: stream.write(page(rows))
        output_hashes["index.html"] = sha(out / "index.html")
        write_json(out / "review.json", {"purpose": PURPOSE, "status": "complete", "step": 5000, "records": rows,
            "audio_files": 33, "output_sha256": output_hashes, "bindings_sha256": bindings, "plan_sha256": sha(out / "plan.json"),
            "forward_calls_completed": counts, "models_digest": score["model_digest"], "checkpoint_unchanged": True,
            "cpu_rng_restored": True, "cuda_used": False, "optimizer_constructed": False, "training_updates": 0,
            "historical_model_forward_calls": 0, "decoder_subprocess_calls": 0, "release_selection": "NONE",
            "independent_acceptance_scored": False, "human_listening_completed": False,
            "scope": "CPU FP32 listening reference, not full training/Adam/native runtime final verification or board acceptance",
            "completed_utc": datetime.now(timezone.utc).isoformat()}, c)
    except BaseException as error:
        write_json(out / "failure.json", {"purpose": PURPOSE, "status": "failed", "error": repr(error),
            "forward_calls_completed": counts, "partial_audio_sha256": output_hashes, "no_automatic_retry": True}, c)
        raise
    finally:
        q.m.restore_rng(outer_rng, "cpu")
        require(c.equal(q.m.capture_rng("cpu"), outer_rng), "Outer CPU/Python/NumPy RNG restore")
    print(json.dumps({"out": str(out), "audio_files": 33, "forward_calls_completed": counts,
                      "cuda_used": False, "training_updates": 0}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.monotonic()
    export(args.out.resolve())
    print(f"EMA_LISTEN_COMPLETE seconds={time.monotonic() - started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
