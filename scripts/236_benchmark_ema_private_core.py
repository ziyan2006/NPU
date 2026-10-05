"""New CPU full-engine fixture steps; not real audio/student/CUDA training.

Original219 transaction, six sequential synthetic microbatch backwards,
original CPU Adam primitive/EMA/sampler/context/allRNG snapshots and rollback.
Synthetic22-parameter824900-element model, NOT the audio student's graph.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time
from unittest.mock import patch
import torch
import torch._dynamo

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema236_factory235", ROOT / "scripts/235_ema_private_fast_storage_core.py")
factory = importlib.util.module_from_spec(spec); spec.loader.exec_module(factory)
spec = importlib.util.spec_from_file_location("ema236_original219", ROOT / "scripts/219_ema_continuous_training_core.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
c, e, f = t.c, t.e, factory.f


def context():
    path = ROOT / "results/mel_lr_scale_import_20261004/approval.json"
    f.require(c.sha256(path) == c.source.PINS[str(path.relative_to(ROOT)).replace('\\','/')], "Context config bytes")
    doc = json.loads(path.read_text(encoding="utf-8-sig")); c.source.check_seal(doc)
    return {"step": 4500, "sampler": {"approval_sha256": c.source.PINS["results/mel_lr_scale_import_20261004/approval.json"],
               "true_lock_sha256": "36a9be8ade7822969a7fed661c8f226f49b89e8ece90e1e6192ae0ae1b4b2c38",
               "seed": 20261002, "cursor": 4500, "teacher": "kim_melband"},
            "schedule": {"step": 4500, "last_validation": 4500, "stopped_at": 4500,
                         "best": dict.fromkeys(c.source.ARMS), "stale": dict.fromkeys(c.source.ARMS, 18),
                         "patience_anchor": dict.fromkeys(c.source.ARMS), "config": doc["source_protocol"]["paired_comparison_planned"]}}


class TrueFixture:
    def __init__(self, sampler):
        self.bound, self.config = sampler["true_lock_sha256"], {"synthetic236": True}
    def crop(self, domain, seed, cursor):
        x = torch.full((2, 89856), .1875)
        return {"x": x, "v": torch.zeros_like(x) if domain == "instrumental" else x*.375,
                "meta": {"domain": domain, "role": "train", "vocal_db": 0, "score_start": 25088, "score_end": 89344,
                         "input_pcm_sha256": t.hashlib.sha256(x.numpy().tobytes()).hexdigest(), "synthetic236": True}}


class PseudoFixture:
    def __init__(self):
        self.bound, self.seed, self.config, self.rows = "synthetic236", 20261002, {"synthetic236": True}, [{"synthetic236": True}]
    def crop(self, recipe):
        x = torch.full((2, 89856), .3125)
        return {"x": x, "v": x*.4375, "meta": {"domain": "pseudo", "role": "pseudo_label_train_candidate",
                "purpose": "NONRELEASE_PAIRED_EXPLORATION", "exploratory_eligible": True,
                "deployment_eligible": False, "training_eligible": False, "score_start": 25088, "score_end": 89344,
                "input_pcm_sha256": t.hashlib.sha256(x.numpy().tobytes()).hexdigest(), "synthetic236": True}}


def recipe(rows, config, seed, cursor): return {"synthetic_cursor": cursor}


class Representative(torch.nn.Module):
    def __init__(self):
        super().__init__(); n = 824900
        self.values = torch.nn.ParameterDict(OrderedDict(("p" + str(i), torch.nn.Parameter(
            torch.full((n//22 + (i < n%22),), (i+1)/4096))) for i in range(22)))
    def forward(self, x):
        return sum(value.square().mean() for value in self.values.values()) + x.mean()*.001


class SyntheticLoss:
    identity = {"kind": "NEW236_representative_synthetic_loss_NOT_audio_student"}
    def __call__(self, model, batch, device):
        total = 0.
        for slot in range(6):
            loss = (model(batch["x"][slot])-batch["v"][slot].mean()*.001).square()/6
            loss.backward(); total += float(loss.detach())
        return {"synthetic_six_sequential_loss": total}


def fixture(core):
    ctx = context(); model = Representative(); config = ctx["schedule"]["config"]
    lr = config["cosine_min_learning_rate"]+(config["learning_rate"]-config["cosine_min_learning_rate"])*(1+math.cos(math.pi*4400/9900))/2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False, fused=False)
    for parameter in model.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(4500.), "exp_avg": torch.full_like(parameter, .125),
                                      "exp_avg_sq": torch.full_like(parameter, .25)}
        parameter.grad = torch.full_like(parameter, .0625)
    stream = core.e.DeviceInputStream(ctx["sampler"], TrueFixture(ctx["sampler"]), PseudoFixture(), recipe, device="cpu")
    parent = {"synthetic_fixture": True, "synthetic236": True, "source_sha256": c.source.SOURCE_SHA, "old_stop": 4500,
              "preserved_synthetic_CUDA_bytes": [torch.arange(16, dtype=torch.uint8)]}
    provenance = {"source_sha256": c.source.SOURCE_SHA, "source_limit": 4500,
                  "source_legacy_stop_events": [3750, 4000], "legacy_stop_events": [4250, 4500]}
    return core.ContinuousTrajectory(model, optimizer, parent, ctx, stream, SyntheticLoss(), provenance=provenance)


def benchmark(out):
    f.require(out.parent == ROOT / "results" and not out.exists() and shutil.disk_usage(ROOT).free >= 12*1024**3,
              "Fresh output and disk reserve")
    out.mkdir(); settings = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    outer = e.capture_rng("cpu"); originals = (t.c.portable, t.e.portable, t.ContinuousTrajectory)
    try:
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU fixture only")), \
             patch.object(torch, "load", side_effect=AssertionError("No source PT in synthetic benchmark")):
            fast = factory.make_private_core(t)
            samples = {"baseline": [], "optimized": []}; expected_before = expected_after = expected_loss = None
            exposure = {"six_microbatch_forward_backward": 0, "synthetic_Adam_steps": 0, "formal_updates": 0}
            for repeat in range(3):
                order = ("baseline", "optimized") if repeat%2 == 0 else ("optimized", "baseline")
                for name in order:
                    owner = fixture(t if name == "baseline" else fast)
                    before = owner.state_dict(); before_commitment = c.digest(before)
                    if expected_before is None: expected_before = before_commitment
                    f.require(expected_before == before_commitment, "Whole source fixture raw/Adam/EMA/context/RNG differs")
                    tick = time.perf_counter(); row = owner.update_next(); elapsed = time.perf_counter()-tick
                    after = owner.state_dict(); commitment = c.digest(after)
                    if expected_after is None: expected_after, expected_loss = commitment, row["losses"]
                    f.require(commitment == expected_after and c.equal(row["losses"], expected_loss), "Full result numerical/state bits differ")
                    f.require(owner.exposure["adam_completed"] == 1 and owner.shadow.updates == 1
                              and row["formal_student_update"] is False, "One synthetic raw Adam/EMA,not formal")
                    c.noalias(after, owner._external())
                    samples[name].append(elapsed); exposure["six_microbatch_forward_backward"] += 6; exposure["synthetic_Adam_steps"] += 1
                    print("NEW236 round=" + str(repeat) + " " + name + " complete_fixture_step_seconds=" + str(elapsed), flush=True)
            # A new actual synthetic Adam fault rollback, not old units/replay.
            owner = fixture(fast); before = owner.state_dict()
            def fault(instance):
                random.random(); e.np.random.random(); torch.rand(1)
                raise KeyboardInterrupt("new236_after_actual_synthetic_Adam_and_EMA")
            try:
                owner.update_next(after_joint_mutation=fault)
                raise AssertionError("Expected fault")
            except KeyboardInterrupt: pass
            exposure["synthetic_Adam_steps"] += 1; exposure["six_microbatch_forward_backward"] += 6
            f.require(c.equal(owner.state_dict(), before), "Full actual synthetic Adam/EMA/count/cursor/allCPU RNG rollback")
            f.require(originals == (t.c.portable, t.e.portable, t.ContinuousTrajectory), "Original module globals changed")
            f.require(e.equal(e.capture_rng("cpu"), outer) and not torch.cuda.is_initialized(), "Whole RNG/CUDA differs")
            medians = {k: statistics.median(v) for k, v in samples.items()}
            result = {"purpose": "NONRELEASE_EMA236_FULL_ORIGINAL_ENGINE_CPU_SYNTHETIC_BENCHMARK",
                      "source_bindings": {str(ROOT / "scripts" / name): digest for name, digest in factory.PINS.items()},
                      "new_source_bindings": {str(ROOT / "scripts" / name): f.sha(ROOT / "scripts" / name) for name in
                           ("232_ema_fast_state_commitment.py", "235_ema_private_fast_storage_core.py", "236_benchmark_ema_private_core.py")},
                      "samples_seconds": samples, "median_seconds": medians, "speedup": medians["baseline"]/medians["optimized"],
                      "full_typed_before_bit_identity": expected_before, "full_typed_after_bit_identity": expected_after,
                      "complete_actual_synthetic_Adam_rollback": True, "noalias": True, "CPU_Python_NumPy_RNG_unchanged": True,
                      "original_modules_unmodified": True, "exposure": exposure, "CUDA": False,
                      "actual_music_source_PT_student_forward_updates": 0, "training_authorized": False, "release_selection": "NONE",
                      "limits": "Full219 CPU synthetic fixture transaction path,not real input/student graph/CUDA throughput;original six-slot toy loss;no native decoder/cache guard"}
            result["content_sha256"] = t.hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=True,
                                          separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            with (out / "benchmark.json").open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=True, indent=2, allow_nan=False)
            print(json.dumps({"median_seconds": medians, "speedup": result["speedup"], "exposure": exposure,
                              "content_sha256": result["content_sha256"]}, indent=2), flush=True)
    finally:
        e.restore_rng(outer, "cpu")
        torch.set_num_threads(settings[0]); torch.use_deterministic_algorithms(settings[1], warn_only=settings[2])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = settings[3:5]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = settings[5:7]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    benchmark(Path(parser.parse_args().out).resolve())
