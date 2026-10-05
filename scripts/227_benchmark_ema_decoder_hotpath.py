"""Fresh synthetic RIFF A/B benchmark, NOT a TRAIN draw or old unit replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import struct
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema227_candidate226", ROOT / "scripts/226_ema_decoder_hotpath.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def history():
    folder = ROOT / "results/mel_ema_single_trajectory_20261005"
    complete = json.loads((folder / "completion.json").read_text(encoding="utf-8"))
    f.require(complete["step"] == 5000 and complete["formal_training_updates"] == 500, "Ended tranche only")
    for name in ("updates.jsonl", "decoder_native_journal.jsonl", "checkpoint_step_4500.json"):
        f.require(f.sha(folder / name) == complete["outputs_sha256"][name], "Changed profiling archive")
    rows = [json.loads(x) for x in (folder / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    f.require([r["step"] for r in rows] == list(range(4501, 5001)), "Exact existing500 updates")
    cut = datetime.fromtimestamp((folder / "checkpoint_step_4500.json").stat().st_mtime).astimezone()
    values, groups = [], defaultdict(list)
    for line in (folder / "decoder_native_journal.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        start, end = datetime.fromisoformat(r["started_utc"]), datetime.fromisoformat(r["completed_utc"])
        if start < cut:
            continue
        sec = (end-start).total_seconds()
        key = "probe" if "ffprobe" in r["request"]["argv"][0].lower() else "decode"
        groups[key].append(sec); values.append(sec)
    total = sum(r["seconds"] for r in rows)
    return {"scope": "Existing completed500 log only;not independent training acceptance",
            "existing_updates": len(rows), "mean_recorded_seconds": statistics.mean(r["seconds"] for r in rows),
            "recorded_update_total_seconds": total, "training_decoder_count": len(values),
            "training_decoder_seconds": sum(values), "decoder_fraction_of_recorded_update": sum(values)/total,
            "groups": {k: {"count":len(v), "total_seconds":sum(v), "mean_seconds":statistics.mean(v)} for k,v in groups.items()}}


def benchmark(out):
    f.require(out.parent == ROOT / "results" and not out.exists(), "Fresh isolated benchmark output required")
    out.mkdir()
    rng = random.getstate()
    path = out / "new_synthetic_226_测试 & % !.wav"
    # Algorithmic new PCM16 data, not any old music/input reference.
    pcm = [(-16000+(i*17)%32000, 15000-(i*31)%30000) for i in range(4096)]
    raw = b"".join(struct.pack("<hh", *p) for p in pcm)
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF",36+len(raw),b"WAVE",b"fmt ",
                         16,1,2,44100,44100*4,4,16,b"data",len(raw))
    path.write_bytes(header+raw)
    expected = b"".join(struct.pack("<ff", l/32768,r/32768) for l,r in pcm)
    requests = f.g.request.original_requests(str(path))
    requests = [next(r for r in requests if r["label"] == name) for name in ("pseudo_probe0","pseudo_decode0")]
    adapters = {"baseline": f.g.DirectDecoder(f.g.read_manifest()), "optimized": f.make_decoder()}
    original = adapters["baseline"].native.measure
    baseline_reads = []
    def counted(instance, handle):
        value = original(handle); baseline_reads.append(value["bytes"])
        return value
    adapters["baseline"].native.measure = MethodType(counted, adapters["baseline"].native)
    outputs, timings = {}, {}
    # Four new child requests, only this synthetic file. No version launches,
    # old reference decoding, original unit suites, PT/model/Adam/CUDA.
    for name, adapter in adapters.items():
        outputs[name], timings[name] = [], []
        for request in requests:
            tick = time.perf_counter()
            value = adapter.check_output(request["argv"], **request["kwargs"])
            timings[name].append(time.perf_counter()-tick)
            outputs[name].append(value)
        if name == "optimized":
            f.require(adapter.native.custody is None, "No cross-call custody/cache")
        with (out / (name+"_events.json")).open("x",encoding="utf-8") as handle:
            json.dump(adapter.rows, handle, ensure_ascii=True, indent=2)
    f.require(outputs["baseline"] == outputs["optimized"], "Probe metadata or PCM bytes differ")
    f.require(outputs["optimized"][1] == expected, "Analytic PCM16->FP32 bytes differ")
    probe = json.loads(outputs["optimized"][0])
    f.require(probe["streams"][0]["sample_rate"] == "44100" and probe["streams"][0]["channels"] == 2,
              "Original stereo44100 metadata")
    for name, adapter in adapters.items():
        f.require(len(adapter.rows) == 2 and not adapter.poisoned and not adapter.active, "Complete child pair")
        for row in adapter.rows:
            f.g.meta.check_seal(row)
            f.require(row["exit_code"] == row["popen_exit_code"] == 0
                      and row["accepted_debug_call"] and row["held_files_post_exit_equal"], "Actual EXIT/pipe/hold failure")
    f.require(rng == random.getstate() and not any(x in sys.modules for x in ("torch","numpy","scipy","soundfile")),
              "No hidden ML/import/RNG execution")
    result = {"purpose": "NONRELEASE_EMA226_NEW_SYNTHETIC_DECODER_PERFORMANCE",
              "source_bindings": {str(f.BASE):f.BASE_SHA, str(ROOT/"scripts/226_ema_decoder_hotpath.py"):f.sha(ROOT/"scripts/226_ema_decoder_hotpath.py")},
              "timings_seconds": timings, "baseline_seconds":sum(timings["baseline"]),
              "optimized_seconds":sum(timings["optimized"]),
              "pair_speedup":sum(timings["baseline"])/sum(timings["optimized"]),
              "baseline_full_reads":len(baseline_reads), "baseline_full_bytes":sum(baseline_reads),
              "optimized_full_reads":adapters["optimized"].native.full_reads,
              "optimized_full_bytes":adapters["optimized"].native.full_bytes,
              "optimized_identity_reads":adapters["optimized"].native.identity_reads,
              "probe_and_PCM_bit_identical":True, "analytic_PCM_bit_identical":True,
              "PCM_sha256":hashlib.sha256(expected).hexdigest(),
              "new_synthetic_child_requests":4, "real_music_draws":0,"training_updates":0,"CUDA":False,
              "training_authorized":False,"release_selection":"NONE", "historical_profile":history(),
              "limits":"One synthetic A/B pair only;not overall training speedup or fresh full-runtime gate;no warmed repeated timing claim"}
    # General performance floats use plain JSON commitment, not typed209.
    result["content_sha256"] = hashlib.sha256(json.dumps(result,sort_keys=True,ensure_ascii=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    with (out/"benchmark.json").open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=True,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out",required=True)
    args=parser.parse_args(); benchmark(Path(args.out).resolve())
