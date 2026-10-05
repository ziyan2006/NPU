"""Paired optimization mechanics, shared inputs/budget and full-state resume.

CLI is hard-limited to three CPU mechanism updates. It cannot start formal
training or export a release model. Unreviewed snapshots remain audit-only.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = load("paired_mechanics_inputs", "136_prepare_distillation_data.py")
fit = load("paired_mechanics_loss", "126_verify_training_baseline.py")
pilot, bulk, core = data.pilot, data.bulk, fit.core
ARMS = ("htdemucs_waveform_control", "kim_melband_waveform_candidate")
DOMAINS = ("musdb", "mir1k", "instrumental", "pseudo", "pseudo", "pseudo")
CPU_LIMIT = 3
DEFAULT_BUNDLE = bulk.ROOT/"results/paired_distillation_prepare_20261002/paired_bundle.json"
DEFAULT_OUT = bulk.ROOT/"results/paired_distillation_cpu_20261002"


def validate_protocol(protocol):
    data.validate_protocol(protocol)
    c = protocol["paired_comparison_planned"]
    exact = {"maximum_steps": 10000, "minimum_steps_before_early_stop": 2000, "effective_batch": 6,
             "microbatch": 1, "optimizer": "Adam", "learning_rate": .0001, "warmup_steps": 100,
             "cosine_min_learning_rate": .00001, "gradient_clip": 3, "validation_every": 250,
             "patience_validation_checks": 8, "minimum_rank_gain_for_patience_reset_db": .02}
    if any(c[k] != v for k, v in exact.items()) or tuple(c["arms"]) != ARMS:
        raise ValueError("Paired budget/optimizer/loss protocol must not be silently changed")
    if "reconstruction_l1" not in c["loss"] or protocol["input"]["crop_frames"] != 256 or protocol["input"]["warmup_frames"] != 96:
        raise ValueError("Fixed product reconstruction and crop context required")
    return c


def learning_rate(step, config):
    if type(step) is not int or not 1 <= step <= config["maximum_steps"]:
        raise ValueError("Learning-rate step outside paired budget")
    if step <= config["warmup_steps"]:
        return config["learning_rate"]*step/config["warmup_steps"]
    progress = (step-config["warmup_steps"])/(config["maximum_steps"]-config["warmup_steps"])
    return config["cosine_min_learning_rate"]+(config["learning_rate"]-config["cosine_min_learning_rate"])*(1+math.cos(math.pi*progress))/2


class SharedSchedule:
    """Stop both arms together; improvement in either arm keeps both running."""
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.step, self.last_validation, self.stopped_at = 0, 0, None
        self.best = {arm: None for arm in ARMS}
        self.stale = {arm: 0 for arm in ARMS}
        self.patience_anchor = {arm: None for arm in ARMS}

    def complete_step(self):
        if self.stopped_at is not None:
            raise ValueError("Shared stopping step already reached")
        self.step += 1
        if self.step == self.config["maximum_steps"]:
            self.stopped_at = self.step

    def observe(self, scores):
        if (self.step < 1 or self.step % self.config["validation_every"] or self.last_validation == self.step or
            set(scores) != set(ARMS)):
            raise ValueError("Need one complete paired validation at the committed common step")
        for arm in ARMS:
            score = scores[arm]
            rank = score["rank_gain_db"]
            if type(score["eligible"]) is not bool or (rank is not None and not math.isfinite(rank)) or (score["eligible"] and rank is None):
                raise ValueError("Invalid eligible/rank result from unchanged script119 policy")
        for arm in ARMS:
            score = scores[arm]
            best = self.best[arm]
            anchor = self.patience_anchor[arm]
            improved = score["eligible"] and (anchor is None or score["rank_gain_db"] >= anchor+self.config["minimum_rank_gain_for_patience_reset_db"])
            if score["eligible"] and (best is None or score["rank_gain_db"] > best["rank_gain_db"]):
                self.best[arm] = {"step": self.step, "rank_gain_db": score["rank_gain_db"]}
            if improved:
                self.patience_anchor[arm] = score["rank_gain_db"]
                self.stale[arm] = 0
            else:
                self.stale[arm] += 1
        self.last_validation = self.step
        if (self.step >= self.config["minimum_steps_before_early_stop"] and
            all(v >= self.config["patience_validation_checks"] for v in self.stale.values())):
            self.stopped_at = self.step

    def selection(self):
        return {arm: self.best[arm] if self.best[arm] is not None else "NONE" for arm in ARMS}

    def state_dict(self):
        return copy.deepcopy({"config": self.config, "step": self.step, "last_validation": self.last_validation,
                              "stopped_at": self.stopped_at, "best": self.best, "stale": self.stale,
                              "patience_anchor": self.patience_anchor})

    def load_state_dict(self, state):
        if (state["config"] != self.config or type(state["step"]) is not int or not 0 <= state["step"] <= self.config["maximum_steps"] or
            state["last_validation"] > state["step"] or set(state["best"]) != set(ARMS) or set(state["stale"]) != set(ARMS)):
            raise ValueError("Shared schedule resume differs from locked budget")
        for name in ("step", "last_validation", "stopped_at", "best", "stale", "patience_anchor"):
            setattr(self, name, copy.deepcopy(state[name]))


class LockedTruePool:
    def __init__(self, lock_path, config, cache_songs=3):
        self.path, self.config = Path(lock_path), config
        self.lock = fit.lock.verify_lock(self.path, verify_files=False)
        self.bound = pilot.acq.sha256(self.path)
        self.pools = {domain: sorted([r for r in self.lock["records"] if r["domain"] == domain and r["role"] == "train"],
                                    key=lambda r: r["track_id"]) for domain in DOMAINS[:3]}
        if any(not rows for rows in self.pools.values()) or cache_songs < 1:
            raise ValueError("All three locked true TRAIN domains are required")
        self.cache, self.signatures, self.cache_songs = OrderedDict(), {}, cache_songs

    def audio(self, row):
        paths = {p for field in ("mix_files", "vocal_files", "stem_files") for p in row.get(field, [])}
        for path in paths:
            signature = data.stat_signature(path)
            if path not in self.signatures:
                if pilot.acq.sha256(path) != self.lock["files"][path]["sha256"]:
                    raise ValueError("Locked original true audio changed")
                self.signatures[path] = signature
            elif signature != self.signatures[path]:
                raise ValueError("True source changed during iteration")
        key = row["domain"], row["track_id"]
        if key not in self.cache:
            if row["domain"] == "musdb":
                value = core.decode_musdb(Path(row["mix_files"][0]))
            else:
                dataset = "mshoxx" if row["domain"] == "instrumental" else "mir1k"
                spec = core.t23.TrackSpec(row["track_id"], dataset, row["mix_files"], row["vocal_files"], row.get("stem_files", []))
                value = core.t23.load_track(spec)
            self.cache[key] = value
            while len(self.cache) > self.cache_songs:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def crop(self, domain, seed, cursor):
        if domain not in self.pools or pilot.acq.sha256(self.path) != self.bound:
            raise ValueError("Unexpected true domain/role binding")
        rng = random.Random(int(hashlib.sha256(f"{seed}:true:{domain}:{cursor}".encode()).hexdigest(), 16))
        row = self.pools[domain][rng.randrange(len(self.pools[domain]))]
        x, v = self.audio(row)
        pilot.finite_stereo(x)
        pilot.finite_stereo(v)
        if x.shape != v.shape:
            raise ValueError("True source alignment changed")
        length, region, tail = data.geometry(self.config)
        grid = self.config["hop"]*self.config["start_grid_frames"]
        limit = x.shape[-1]-length-tail
        if limit < 0:
            raise ValueError("True source too short; no silent padding")
        start = rng.randrange(limit//grid+1)*grid
        x, v = x[:, start:start+length].clone(), v[:, start:start+length].clone()
        db = 0 if domain == "instrumental" else rng.choice((-12, -6, 0, 6))
        a = x-v
        v = v*10**(db/20)
        x = a+v
        gain = data.common_gain(x, rng.uniform(*self.config["common_gain_db_range"]))
        x, v = x*gain, v*gain
        return {"x": x, "v": v, "meta": {"domain": domain, "role": "train", "track_id": row["track_id"],
                "start_sample": start, "vocal_db": db, "gain": gain, "input_pcm_sha256": pilot.wave_digest(x),
                "score_start": region.start, "score_end": region.stop}}


class SharedBatchStream:
    def __init__(self, bundle_path, purpose="training"):
        if purpose != "cpu_mechanism":
            raise ValueError("No approved training import; human/rights/truth/exit gates remain closed")
        self.path = Path(bundle_path)
        self.doc = pilot.acq.read_sealed(self.path)
        if self.doc["training_authorized"] is not False or self.doc["human_reviewed_pairs"] != 0:
            raise ValueError("Current bundle is CPU mechanism evidence, not an approval")
        self.bound = pilot.acq.sha256(self.path)
        for path, expected in self.doc["bindings_sha256"].items():
            if pilot.acq.sha256(path) != expected:
                raise ValueError("Paired preparation source/role/code binding changed")
        self.datasets = []
        for name in ("htdemucs", "kim_melband"):
            snapshot = self.doc["snapshots"][name]
            if pilot.acq.sha256(snapshot["path"]) != snapshot["sha256"]:
                raise ValueError("Paired snapshot changed")
            self.datasets.append(data.TeacherWaveformDataset(snapshot["path"], purpose="cpu_audit"))
        a, b = self.datasets
        if (a.mix_identity != b.mix_identity or a.seed != b.seed or a.config != b.config or
            [r["song_id"] for r in a.rows] != self.doc["pair_ids"] or [r["song_id"] for r in b.rows] != self.doc["pair_ids"]):
            raise ValueError("Matched teacher source identities/config/order differ")
        self.seed, self.config, self.cursor = a.seed, a.config, 0
        self.true = LockedTruePool(bulk.OLD_LOCK, self.config)

    def next_batch(self):
        xs, targets, metadata = [], [[], []], []
        for domain in DOMAINS[:3]:
            item = self.true.crop(domain, self.seed, self.cursor)
            xs.append(item["x"])
            for arm in range(2):
                targets[arm].append(item["v"])
            metadata.append(item["meta"])
        for i in range(3):
            recipe = data.crop_recipe(self.datasets[0].rows, self.config, self.seed, self.cursor*3+i)
            a, b = (dataset.crop(recipe) for dataset in self.datasets)
            data.assert_paired_inputs(a, b)
            xs.append(a["x"])
            targets[0].append(a["v"])
            targets[1].append(b["v"])
            metadata.append(a["meta"] | {"domain": "pseudo", "role": "pseudo_label_train_candidate"})
        batch = {"x": torch.stack(xs), "targets": {ARMS[i]: torch.stack(v) for i, v in enumerate(targets)},
                 "domains": DOMAINS, "metadata": metadata, "cursor": self.cursor}
        self.cursor += 1
        return batch

    def state_dict(self):
        return {"bundle_sha256": self.bound, "true_lock_sha256": self.true.bound, "seed": self.seed, "cursor": self.cursor}

    def load_state_dict(self, state):
        if (state["bundle_sha256"] != self.bound or state["true_lock_sha256"] != self.true.bound or state["seed"] != self.seed or
            type(state["cursor"]) is not int or state["cursor"] < 0 or pilot.acq.sha256(self.path) != self.bound):
            raise ValueError("Paired input sampler resume binding/counter mismatch")
        self.cursor = state["cursor"]


def capture_rng(device):
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], torch.from_numpy(n[1].astype(np.int64)), n[2], n[3], n[4]],
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if device == "cuda" else []}


def restore_rng(state, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device == "cuda":
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device-count resume mismatch")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    elif state["torch_cuda"]:
        raise ValueError("Cannot substitute CPU for a CUDA checkpoint")


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


class PairedEngine:
    def __init__(self, factory, protocol, binding, device="cpu", purpose="training"):
        if device != "cpu" or purpose != "cpu_mechanism":
            raise ValueError("Only bounded CPU mechanism tests authorized; no formal training or CUDA entry")
        self.config = validate_protocol(protocol)
        self.models = {arm: factory() for arm in ARMS}
        if not equal_state(self.models[ARMS[0]].state_dict(), self.models[ARMS[1]].state_dict()):
            raise ValueError("Both arms must start from identical frozen initialization")
        self.optimizers = {arm: torch.optim.Adam(net.parameters(), lr=self.config["learning_rate"]) for arm, net in self.models.items()}
        self.binding, self.device, self.step = binding, device, 0
        self.schedule = SharedSchedule(self.config)
        self.wa = torch.from_numpy(core.t09.make_analysis_matrix())
        self.gs = torch.from_numpy(core.t09.make_synthesis_matrix())

    def state_dict(self, stream):
        if stream.state_dict()["cursor"] != self.step:
            raise ValueError("Cannot save a half-updated pair or uncommitted sampler")
        return {"schema": 1, "binding": self.binding, "device": self.device, "step": self.step,
                "purpose": "NONRELEASE_CPU_MECHANISM", "cpu_update_limit": CPU_LIMIT,
                "arms": {arm: {"model": copy.deepcopy(net.state_dict()), "optimizer": copy.deepcopy(self.optimizers[arm].state_dict()),
                                "updates": self.step} for arm, net in self.models.items()},
                "sampler": copy.deepcopy(stream.state_dict()), "schedule": self.schedule.state_dict(), "rng": capture_rng(self.device)}

    def load_state_dict(self, state, stream):
        if (state["schema"] != 1 or state["binding"] != self.binding or state["device"] != self.device or
            state["purpose"] != "NONRELEASE_CPU_MECHANISM" or state["cpu_update_limit"] != CPU_LIMIT or
            type(state["step"]) is not int or not 0 <= state["step"] <= CPU_LIMIT or set(state["arms"]) != set(ARMS) or
            any(arm["updates"] != state["step"] for arm in state["arms"].values()) or
            state["sampler"]["cursor"] != state["step"] or state["schedule"]["step"] != state["step"]):
            raise ValueError("Changed binding, backend, arm exposure or incomplete paired resume")
        # Validate sampler and schedule before loading either optimizer/model.
        stream.load_state_dict(state["sampler"])
        self.schedule.load_state_dict(state["schedule"])
        for arm in ARMS:
            self.models[arm].load_state_dict(state["arms"][arm]["model"], strict=True)
            self.optimizers[arm].load_state_dict(state["arms"][arm]["optimizer"])
            self.optimizers[arm].zero_grad(set_to_none=True)
        self.step = state["step"]
        restore_rng(state["rng"], self.device)

    def update_next(self, stream):
        if self.step >= CPU_LIMIT:
            raise ValueError("CPU smoke update cap reached; never silently start long training")
        previous = self.state_dict(stream)
        try:
            batch = stream.next_batch()
            x = batch["x"]
            if (tuple(batch["domains"]) != DOMAINS or batch["cursor"] != self.step or x.shape != (6, 2, 89856) or
                set(batch["targets"]) != set(ARMS) or any(v.shape != x.shape for v in batch["targets"].values()) or
                not torch.isfinite(x).all() or any(not torch.isfinite(v).all() for v in batch["targets"].values())):
                raise ValueError("Need aligned 3 true + 3 pseudo same-input batch; no OnAir or silent context padding")
            if not torch.equal(batch["targets"][ARMS[0]][:3], batch["targets"][ARMS[1]][:3]):
                raise ValueError("True-reference targets must be identical across arms")
            lr = learning_rate(self.step+1, self.config)
            losses = {}
            for arm in ARMS:
                net, opt = self.models[arm], self.optimizers[arm]
                net.train()
                for group in opt.param_groups:
                    group["lr"] = lr
                opt.zero_grad(set_to_none=True)
                losses[arm] = fit.backward_batch(net, x, batch["targets"][arm], self.wa, self.gs, "cpu", 96, 44, "reconstruction_l1", 1)
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.config["gradient_clip"], error_if_nonfinite=True)
                opt.step()
            self.step += 1
            self.schedule.complete_step()
            return {"step": self.step, "lr": lr, "losses": losses, "input_sha256": [pilot.wave_digest(v) for v in x],
                    "domains": list(DOMAINS), "metadata": batch.get("metadata", [])}
        except BaseException:
            # If the second arm fails, rewind BOTH arms, RNG, budget and inputs.
            self.load_state_dict(previous, stream)
            raise


def frozen_factory(protocol):
    checkpoint = bulk.ROOT/"models/student_bott2_mir1k_candidate.pt"
    if pilot.acq.sha256(checkpoint) != protocol["frozen_sha256"]:
        raise ValueError("Frozen student changed")
    template, blob = core.t13.load_student(checkpoint, "cpu")
    if (blob.get("bottleneck_blocks") != 2 or blob.get("temporal_dilations") or template.frontend != ("linear", 1.) or
        template.band_layout != "legacy_log" or template.n_bands != 128):
        raise ValueError("Frozen graph differs from paired protocol")
    return lambda: copy.deepcopy(template)


def save_new(path, state):
    with Path(path).open("xb") as stream:
        torch.save(state, stream)


def load_checked_checkpoint(path, expected_sha256):
    if pilot.acq.sha256(path) != expected_sha256:
        raise ValueError("Saved paired checkpoint changed")
    return torch.load(path, map_location="cpu", weights_only=True)


def smoke(bundle, out):
    bulk.guard_output(out)
    if out.exists():
        raise ValueError("Fresh CPU smoke directory required; preserve existing checkpoints")
    protocol = json.loads(data.PROTOCOL.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    factory = frozen_factory(protocol)
    binding = {"bundle_sha256": pilot.acq.sha256(bundle), "protocol_sha256": pilot.acq.sha256(data.PROTOCOL),
               "trainer_sha256": pilot.acq.sha256(__file__), "torch_version": str(torch.__version__),
               "numpy_version": str(np.__version__), "device": "cpu", "frozen_sha256": protocol["frozen_sha256"]}
    stream = SharedBatchStream(bundle, purpose="cpu_mechanism")
    engine = PairedEngine(factory, protocol, binding, purpose="cpu_mechanism")
    out.mkdir(parents=True)
    draws = []
    checkpoint_hashes = {}
    initial_path = out/"NONRELEASE_pair_step_0000.pt"
    save_new(initial_path, engine.state_dict(stream))
    checkpoint_hashes[initial_path.name] = pilot.acq.sha256(initial_path)
    for _ in range(CPU_LIMIT):
        result = engine.update_next(stream)
        draws.append(result)
        path = out/f"NONRELEASE_pair_step_{engine.step:04d}.pt"
        save_new(path, engine.state_dict(stream))
        checkpoint_hashes[path.name] = pilot.acq.sha256(path)
        print(f"PAIRED_CPU_MECHANISM committed_step={engine.step}/{CPU_LIMIT}", flush=True)
    expected = engine.state_dict(stream)
    restored_stream = SharedBatchStream(bundle, purpose="cpu_mechanism")
    restored_engine = PairedEngine(factory, protocol, binding, purpose="cpu_mechanism")
    restored_engine.load_state_dict(load_checked_checkpoint(out/"NONRELEASE_pair_step_0001.pt", checkpoint_hashes["NONRELEASE_pair_step_0001.pt"]), restored_stream)
    resumed = [restored_engine.update_next(restored_stream) for _ in range(2)]
    actual = restored_engine.state_dict(restored_stream)
    if resumed != draws[1:] or not equal_state(actual, expected):
        raise ValueError("Saved paired resume changed inputs/model/Adam/RNG/shared budget")
    report = {"schema": 1, "binding": binding, "updates_per_arm": 3, "effective_batch": 6, "microbatch": 1,
              "draws": draws, "disk_resume_from_step_1_identical": True, "checkpoint_selected": "NONE", "checkpoint_sha256": checkpoint_hashes,
              "formal_student_training": False, "deployment": False, "cuda_used": False,
              "scope": "Bounded CPU optimization/resume mechanism test, not quality, convergence or independent acceptance"}
    pilot.acq.write_new_json(out/"cpu_mechanism.json", pilot.acq.seal(report))
    print("PAIRED_CPU_MECHANISM PASS updates_per_arm=3 disk_resume=identical; selection=NONE; no formal training", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--formal-train", action="store_true", help="Always rejected until a separately approved importer is implemented")
    args = ap.parse_args()
    if args.formal_train:
        raise ValueError("Formal training CLOSED: no reviewed approval/import, independent truth gate or proven original numeric exit status")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    smoke(args.bundle, args.out)


if __name__ == "__main__":
    main()
