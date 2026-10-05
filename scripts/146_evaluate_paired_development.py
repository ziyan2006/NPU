"""Read-only, CPU paired development evaluation using the unchanged script119 policy.

Evaluates existing NONRELEASE mechanism checkpoints, never constructs an optimizer,
starts training, scores new acceptance audio or promotes a checkpoint. A reusable
adapter commits shared selection only after BOTH real development evaluations pass.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("paired_dev_mechanics", Path(__file__).with_name("143_paired_distillation_mechanics.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
suite, core, acq = m.fit.suite, m.core, m.pilot.acq
ROOT = m.bulk.ROOT
DEFAULT_OUT = ROOT / "results/paired_development_eval_20261002"
OLD_MANIFEST = ROOT / "results/layout_control_microbatch_20261001/selection_suite.json"
EXPECTED_SUITE = "e5990cdd2faf71e83469cbef7f94dea6b80d0b376ba49d5b81bca69fe442fb07"


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("Need a finite tensor-only model state")
        h.update(name.encode())
        h.update(suite.tensor_digest(value).encode())
    return h.hexdigest()


class LockedDevelopmentCorpus:
    """Decode only old, explicitly locked DEVELOPMENT sources, not train/holdout."""
    def __init__(self, path, teacher_ids=(), cache_songs=3):
        if cache_songs < 1:
            raise ValueError("Positive bounded development cache required")
        self.path = Path(path)
        self.lock = m.fit.lock.verify_lock(self.path, verify_files=False)
        self.bound = acq.sha256(self.path)
        self.train, self.val = {}, {}
        for domain in ("musdb", "mir1k", "instrumental"):
            self.train[domain] = [r for r in self.lock["records"] if r["domain"] == domain and r["role"] == "train"]
            self.val[domain] = [r | {"dataset": "mshoxx" if domain == "instrumental" else domain}
                                for r in self.lock["records"] if r["domain"] == domain and r["role"] == "development"]
        self.final_ids = self.lock["known_regression_ids"] + self.lock["blind_ids"]
        self.teacher_meta = {"source_ids": list(self.lock["teacher"]["source_ids"]) + list(teacher_ids)}
        self.allowed = {(r["domain"], r["track_id"]): r for rows in self.val.values() for r in rows}
        self.cache, self.signatures, self.file_hashes = OrderedDict(), {}, {}
        self.cache_songs = cache_songs
        suite.guard_split(self)

    def audio(self, row):
        key = row.get("domain"), row.get("track_id")
        if key not in self.allowed or row != self.allowed[key]:
            raise ValueError("Only exact locked DEVELOPMENT records may be decoded")
        if acq.sha256(self.path) != self.bound:
            raise ValueError("Development role lock changed")
        for name in sorted({p for field in ("mix_files", "vocal_files", "stem_files") for p in row.get(field, [])}):
            signature = m.data.stat_signature(name)
            if name not in self.signatures:
                expected = self.lock["files"][name]
                if Path(name).stat().st_size != expected["bytes"] or acq.sha256(name) != expected["sha256"]:
                    raise ValueError("Original development audio changed")
                self.signatures[name] = signature
                self.file_hashes[name] = expected["sha256"]
            elif self.signatures[name] != signature:
                raise ValueError("Development audio changed during evaluation")
        if key not in self.cache:
            if row["domain"] == "musdb":
                x, v = core.decode_musdb(Path(row["mix_files"][0]))
            else:
                track = core.t23.TrackSpec(row["track_id"], row["dataset"], row["mix_files"], row["vocal_files"], row.get("stem_files", []))
                x, v = core.t23.load_track(track)
            if x.shape != v.shape or x.ndim != 2 or x.shape[0] != 2 or not torch.isfinite(x).all() or not torch.isfinite(v).all():
                raise ValueError("Development source must be finite aligned stereo")
            self.cache[key] = x, v
            while len(self.cache) > self.cache_songs:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def checked_suite(corpus, historical):
    rows = suite.build_suite(corpus, 89856, 96)
    manifest = suite.suite_manifest(rows, 89856, 96)
    if manifest != historical or manifest["sha256"] != EXPECTED_SUITE or (manifest["tracks"], manifest["clips"]) != (31, 177):
        raise ValueError("Development inputs/windows changed from the existing 31-song/177-view suite")
    return rows, manifest


class PairedDevelopmentValidator:
    def __init__(self, rows, frozen, evaluator=None):
        if not rows or any(r["x"].device.type != "cpu" or r["v"].device.type != "cpu" for r in rows):
            raise ValueError("Nonempty CPU-only development waveforms required")
        self.rows = rows
        self.manifest = suite.suite_manifest(rows, 89856, 96)
        self.policy = copy.deepcopy(suite.POLICY)
        self.policy_digest = acq.content_digest(self.policy)
        self.evaluator = evaluator or suite.evaluate
        self.wa = torch.from_numpy(core.t09.make_analysis_matrix())
        self.gs = torch.from_numpy(core.t09.make_synthesis_matrix())
        self.baseline = self._score(frozen)
        self.baseline_digest = acq.content_digest(self.baseline)

    def _check_inputs(self):
        if suite.POLICY != self.policy:
            raise ValueError("Locked selection policy changed")
        for row in self.rows:
            if suite.tensor_digest(row["x"], row["v"]) != row["samples_sha256"]:
                raise ValueError("Development input/target waveform changed")
        if suite.suite_manifest(self.rows, 89856, 96) != self.manifest:
            raise ValueError("Development windows or coverage changed")

    def _score(self, net):
        self._check_inputs()
        if any(p.device.type != "cpu" for p in net.parameters()) or any(b.device.type != "cpu" for b in net.buffers()):
            raise ValueError("Read-only mechanism evaluation is CPU-only")
        original = copy.deepcopy(net.state_dict())
        digest, mode, rng = state_digest(original), net.training, m.capture_rng("cpu")
        try:
            with torch.no_grad():
                result = self.evaluator(net, self.rows, self.wa, self.gs, "cpu", 96, 44)
            if state_digest(net.state_dict()) != digest:
                raise ValueError("Evaluation mutated model parameters/buffers")
            self._check_inputs()
            descriptors = [{k: v for k, v in r.items() if k != "metrics"} for r in result["rows"]]
            if descriptors != self.manifest["rows"] or result["summary"] != suite.aggregate(result["rows"]):
                raise ValueError("Unpaired/missing/changed development scores or aggregation")
            return result
        finally:
            net.load_state_dict(original, strict=True)
            net.train(mode)
            m.restore_rng(rng, "cpu")

    def evaluate_pair(self, models, step):
        if set(models) != set(m.ARMS) or type(step) is not int or step < 0:
            raise ValueError("Both arms at the same committed logical step are required")
        if acq.content_digest(self.baseline) != self.baseline_digest:
            raise ValueError("Frozen baseline scores changed")
        evaluations = {arm: self._score(models[arm]) for arm in m.ARMS}
        scores = {arm: suite.assess(evaluations[arm]["summary"], self.baseline["summary"]) for arm in m.ARMS}
        return {"step": step, "suite_sha256": self.manifest["sha256"], "policy_sha256": self.policy_digest,
                "baseline_scores_sha256": self.baseline_digest, "scores": scores, "evaluations": evaluations,
                "model_state_sha256": {arm: state_digest(models[arm].state_dict()) for arm in m.ARMS},
                "scope": "Old true-reference development only; not independent acceptance or release approval"}

    def observe_pair(self, schedule, models, step):
        if schedule.step != step or step < 1 or step % schedule.config["validation_every"] or schedule.last_validation == step:
            raise ValueError("Validation must occur once at the committed common schedule boundary")
        previous = schedule.state_dict()
        try:
            packet = self.evaluate_pair(models, step)
            schedule.observe(packet["scores"])
            return packet
        except BaseException:
            schedule.load_state_dict(previous)
            raise


def checked_cpu_state(path, expected_sha, binding, step):
    state = m.load_checked_checkpoint(path, expected_sha)
    if (state.get("purpose") != "NONRELEASE_CPU_MECHANISM" or state.get("device") != "cpu" or
        state.get("binding") != binding or state.get("cpu_update_limit") != 3 or type(state.get("step")) is not int or
        type(step) is not int or not 0 <= step <= 3 or state["step"] != step or set(state.get("arms", {})) != set(m.ARMS) or
        any(a["updates"] != step for a in state["arms"].values()) or state["sampler"]["cursor"] != step or state["schedule"]["step"] != step):
        raise ValueError("Require hash-bound, complete existing CPU mechanism state; not a training checkpoint")
    return state


def run(out):
    out = out.resolve()
    allowed = (ROOT / "results").resolve()
    if out == allowed or not out.is_relative_to(allowed) or out.exists():
        raise ValueError("Fresh child of ignored results required; never overwrite old evidence")
    protocol = json.loads(m.data.PROTOCOL.read_text(encoding="utf-8"))
    m.validate_protocol(protocol)
    mechanism_path = m.DEFAULT_OUT / "cpu_mechanism.json"
    mechanism = acq.read_sealed(mechanism_path)
    binding = mechanism["binding"]
    if (mechanism["formal_student_training"] is not False or mechanism["cuda_used"] is not False or
        mechanism["checkpoint_selected"] != "NONE" or mechanism["updates_per_arm"] != 3 or
        binding["protocol_sha256"] != acq.sha256(m.data.PROTOCOL) or binding["bundle_sha256"] != acq.sha256(m.DEFAULT_BUNDLE) or
        binding["trainer_sha256"] != acq.sha256(m.__file__) or binding["frozen_sha256"] != protocol["frozen_sha256"] or
        binding["device"] != "cpu" or binding["torch_version"] != str(torch.__version__) or binding["numpy_version"] != str(m.np.__version__)):
        raise ValueError("Existing mechanism report/backend/bindings changed")
    bundle = acq.read_sealed(m.DEFAULT_BUNDLE)
    bindings = dict(bundle["bindings_sha256"])
    for name, expected in bindings.items():
        if acq.sha256(name) != expected:
            raise ValueError(f"Original paired binding changed: {name}")
    for snapshot in bundle["snapshots"].values():
        if acq.sha256(snapshot["path"]) != snapshot["sha256"]:
            raise ValueError("Original paired snapshot changed")
        bindings[snapshot["path"]] = snapshot["sha256"]
    candidate_snapshot = acq.read_sealed(bundle["snapshots"]["kim_melband"]["path"])
    teacher_ids = [Path(r["source"]["path"]).name for r in candidate_snapshot["records"]]
    corpus = LockedDevelopmentCorpus(m.bulk.OLD_LOCK, teacher_ids)
    historical = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    print("PAIRED_DEV loading only locked DEVELOPMENT sources; no train/acceptance decode", flush=True)
    rows, manifest = checked_suite(corpus, historical)
    print(f"PAIRED_DEV suite={manifest['tracks']} songs/{manifest['clips']} views; evaluating frozen baseline", flush=True)
    factory = m.frozen_factory(protocol)
    frozen = factory()
    validator = PairedDevelopmentValidator(rows, frozen)
    models = {arm: factory() for arm in m.ARMS}
    checkpoint_paths = {}
    for step in (0, 3):
        path = m.DEFAULT_OUT / f"NONRELEASE_pair_step_{step:04d}.pt"
        expected = mechanism["checkpoint_sha256"][path.name]
        state = checked_cpu_state(path, expected, binding, step)
        checkpoint_paths[str(path.resolve())] = expected
        for arm in m.ARMS:
            models[arm].load_state_dict(state["arms"][arm]["model"], strict=True)
            if step == 0 and state_digest(models[arm].state_dict()) != state_digest(frozen.state_dict()):
                raise ValueError("Initial paired models differ from the frozen baseline")
    print("PAIRED_DEV initial pair = frozen; evaluating existing step-3 NONRELEASE arms", flush=True)
    packet = validator.evaluate_pair(models, 3)
    for path in (Path(__file__), Path(m.__file__), Path(suite.__file__), Path(suite.audit.__file__), OLD_MANIFEST,
                 mechanism_path, m.DEFAULT_BUNDLE, Path(corpus.path)):
        bindings[str(path.resolve())] = acq.sha256(path)
    bindings.update(corpus.file_hashes)
    bindings.update(checkpoint_paths)
    # New acceptance candidates remain unscored and unpromoted.
    qc_path = ROOT / "results/cambridge_new_sources_20261001/source_qc.json"
    qc = acq.read_sealed(qc_path)
    acceptance = [r for r in qc["records"] if r["role"] == "acceptance_candidate"]
    bindings[str(qc_path.resolve())] = acq.sha256(qc_path)
    evidence = {"schema": 1, "device": "cpu", "torch_version": str(torch.__version__), "policy": suite.POLICY,
                "bindings_sha256": bindings, "suite_sha256": manifest["sha256"],
                "initial_pair_matches_frozen": True, "existing_mechanism_step": 3,
                "development_assessment": packet["scores"], "checkpoint_selected": "NONE",
                "formal_student_training": False, "optimizer_constructed": False, "cuda_used": False,
                "deployment": False, "schedule_advanced": False, "independent_acceptance_scored": False,
                "new_true_acceptance": {"candidate_songs": len(acceptance), "candidate_artists": len({r['artist'] for r in acceptance}),
                    "reviewed_ready_songs": sum(r["acceptance_ready"] is True for r in acceptance), "target_songs": 10,
                    "additional_candidates_needed_at_least": max(0, 10-len(acceptance)), "status": "incomplete"},
                "scope": "Real old-development assessment adapter and existing CPU mechanism checkpoint check; not a teacher quality experiment"}
    out.mkdir(parents=True)
    for name, doc in (("selection_suite.json", manifest), ("frozen_scores.json", validator.baseline), ("paired_scores_step_0003.json", packet)):
        acq.write_new_json(out / name, acq.seal(copy.deepcopy(doc)))
    evidence["outputs_sha256"] = {str((out / name).resolve()): acq.sha256(out / name)
                                 for name in ("selection_suite.json", "frozen_scores.json", "paired_scores_step_0003.json")}
    acq.write_new_json(out / "development_evaluation.json", acq.seal(evidence))
    print("PAIRED_DEV PASS real script119 assessment; selection=NONE; no training/acceptance/deployment", flush=True)


def verify(out):
    doc = acq.read_sealed(out / "development_evaluation.json")
    for paths in (doc["bindings_sha256"], doc["outputs_sha256"]):
        for name, expected in paths.items():
            if acq.sha256(name) != expected:
                raise ValueError(f"Development evaluation binding changed: {name}")
    if (doc["policy"] != suite.POLICY or doc["checkpoint_selected"] != "NONE" or doc["formal_student_training"] is not False or
        doc["independent_acceptance_scored"] is not False or doc["optimizer_constructed"] is not False or doc["cuda_used"] is not False or
        doc["schedule_advanced"] is not False or doc["deployment"] is not False):
        raise ValueError("Development mechanism evidence cannot become training or release approval")
    print("PAIRED_DEV VERIFIED; old development only; formal training remains CLOSED", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--formal-train", action="store_true", help="Always refused; this tool is evaluation only")
    args = ap.parse_args()
    if args.formal_train:
        raise ValueError("Read-only evaluation cannot authorize/start formal training")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.verify:
        verify(args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
