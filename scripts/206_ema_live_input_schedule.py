"""Bounded single-target routing and replay-checked live schedule, NOT a trainer.

No model, optimizer, audio constructor, CUDA setup or training CLI. Supplied
crop backends must be separately authenticated by the future importer. Tests
use synthetic tensors. This state cannot be substituted for a full raw/Adam/
shadow/RNG container, and the closed205 context validator is NOT reused.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
START, LIMIT = 4500, 5000
STAGES = (4750, 5000)
DOMAINS = ("musdb", "mir1k", "instrumental", "pseudo", "pseudo", "pseudo")
PURPOSE = "NONRELEASE_EMA_LIVE_CONTEXT_NOT_TRAINER"
PINS = {
    "scripts/205_ema_cpu_state_transaction.py": "8a8f2da566b664abfb19e6c24fe5fb4bf78bd61c13408ebf385b5cf425cd90f5",
    "scripts/143_paired_distillation_mechanics.py": "bb3ad26dcf5c94e3909f8c4f05c9fb4f249e1a64b03f8b3d3155a26619ebddab",
    "scripts/136_prepare_distillation_data.py": "86b672b965b00c92bc47715dfc9284d19dfed52e11479b7d668e1d340ed5ef2b",
    "scripts/193_prepare_mel_lr_scale.py": "1c358793ce93d8ea7f54fa019610582f1a6427ac3666bef08de1a4c0fc537f89",
}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_dependencies():
    for path, expected in PINS.items():
        if file_sha(ROOT / path) != expected:
            raise ValueError("Frozen live-context dependency changed: " + path)


check_dependencies()
spec = importlib.util.spec_from_file_location("ema206_storage_primitives", ROOT / "scripts/205_ema_cpu_state_transaction.py")
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)
require, equal, portable = storage.require, storage.equal, storage.portable
RAW_ARM, OTHER_ARM = storage.source.ARMS


def validate_sampler(state, original):
    require(type(state) is dict and list(state) == list(original), "Full ordered sampler fields")
    require(type(state["cursor"]) is int and START <= state["cursor"] <= LIMIT, "Exact bounded cursor4500..5000")
    require(equal({k: v for k, v in state.items() if k != "cursor"},
                  {k: v for k, v in original.items() if k != "cursor"}), "Original input binding/seed/teacher changed")


def validate_metadata(rows):
    require(type(rows) is list and len(rows) == 6 and
            [r.get("domain") for r in rows] == list(DOMAINS), "Original six domain slots required")
    for i, row in enumerate(rows):
        require(type(row) is dict and row.get("score_start") == 25088 and row.get("score_end") == 89344,
                "Original native scoring support")
        if i < 3:
            require(row.get("role") == "train" and type(row.get("vocal_db")) is int and
                    row["vocal_db"] in ((0,) if i == 2 else (-12, -6, 0, 6)), "Original true TRAIN roles/remix gains")
        else:
            require(row.get("role") == "pseudo_label_train_candidate" and
                    row.get("purpose") == "NONRELEASE_PAIRED_EXPLORATION" and
                    row.get("exploratory_eligible") is True and row.get("deployment_eligible") is False and
                    row.get("training_eligible") is False, "Pseudo role is not final truth or release authority")
        require(type(row.get("input_pcm_sha256")) is str and len(row["input_pcm_sha256"]) == 64,
                "Input PCM identity missing")
    storage.typed_tree(rows)  # Reject nonfinite/type-unsupported metadata.


class BoundedSingleTargetStream:
    """Original193 crop order; ONE Mel target, cursor boundary is new and local.

    No dataset is constructed here. The future importer must bind actual true
    and teacher pools and all source/cache files before allowing real crops.
    LRU decode caches are not random sampling state. A failed draw restores
    semantic cursor/metadata and CPU RNG, not foreign backend mutations.
    """
    def __init__(self, sampler, true_pool, teacher_dataset, crop_recipe):
        require(type(sampler) is dict and sampler.get("cursor") == START and
                type(sampler.get("cursor")) is int and sampler.get("teacher") == "kim_melband",
                "Explicit original4500 sampler required")
        require(sampler.get("true_lock_sha256") == true_pool.bound and sampler.get("seed") == teacher_dataset.seed,
                "Actual input seed/lock mismatch")
        require(equal(true_pool.config, teacher_dataset.config) and callable(crop_recipe), "Original input configuration/recipe")
        self._source = portable(sampler)
        self.true, self.dataset, self.recipe = true_pool, teacher_dataset, crop_recipe
        self._callables = (id(true_pool), id(teacher_dataset), crop_recipe,
                           true_pool.crop.__func__, teacher_dataset.crop.__func__)
        self._backend_identity = (true_pool.bound, teacher_dataset.bound, storage.digest(teacher_dataset.rows),
                                  storage.digest(teacher_dataset.config), teacher_dataset.seed)
        self.cursor, self.last_metadata, self.poisoned = START, None, False

    def _guard(self):
        check_dependencies()
        require(not self.poisoned, "Poisoned stream cannot continue")
        require((id(self.true), id(self.dataset), self.recipe, self.true.crop.__func__, self.dataset.crop.__func__) ==
                self._callables, "Input routing implementation changed")
        identity = (self.true.bound, self.dataset.bound, storage.digest(self.dataset.rows),
                    storage.digest(self.dataset.config), self.dataset.seed)
        require(equal(identity, self._backend_identity) and equal(self.true.config, self.dataset.config),
                "Input backend recipe/config/order changed")

    def state_dict(self):
        self._guard()
        state = self._source | {"cursor": self.cursor}
        validate_sampler(state, self._source)
        return portable(state)

    def load_state_dict(self, state, *, last_metadata=None):
        self._guard()
        validate_sampler(state, self._source)
        if last_metadata is not None:
            validate_metadata(last_metadata)
        self.cursor, self.last_metadata = state["cursor"], portable(last_metadata)

    def next_batch(self):
        before, metadata_before, rng = self.state_dict(), portable(self.last_metadata), storage.capture_cpu_rng()
        require(self.cursor < LIMIT, "New hard5000 input budget exhausted")
        try:
            xs, vs, metadata = [], [], []
            for domain in DOMAINS[:3]:
                row = self.true.crop(domain, self._source["seed"], self.cursor)
                xs.append(row["x"]); vs.append(row["v"]); metadata.append(portable(row["meta"]))
            for index in range(3):
                recipe = self.recipe(self.dataset.rows, self.dataset.config, self._source["seed"], self.cursor*3+index)
                row = self.dataset.crop(recipe)
                xs.append(row["x"]); vs.append(row["v"])
                metadata.append(portable(row["meta"]) | {"domain": "pseudo", "role": "pseudo_label_train_candidate"})
            validate_metadata(metadata)
            require(all(type(v) is torch.Tensor and v.device.type == "cpu" and v.dtype == torch.float32 and
                        v.shape == (2, 89856) and not v.requires_grad and v.grad_fn is None and
                        bool(torch.isfinite(v).all()) for v in xs+vs), "Finite detached original FP32 crop geometry")
            x, v = torch.stack(xs), torch.stack(vs)
            actual_hashes = [hashlib.sha256(w.detach().contiguous().numpy().tobytes()).hexdigest() for w in x]
            require(actual_hashes == [r["input_pcm_sha256"] for r in metadata], "Actual input PCM differs from metadata")
            require(equal(rng, storage.capture_cpu_rng()), "Original local-counter sampler must not consume global RNG")
            result = {"x": x, "v": v, "domains": DOMAINS, "cursor": self.cursor, "metadata": metadata}
            self.cursor += 1
            self.last_metadata = portable(metadata)
            return result
        except BaseException:
            try:
                self.load_state_dict(before, last_metadata=metadata_before)
                storage.restore_cpu_rng(rng)
            except BaseException as error:
                self.poisoned = True
                raise RuntimeError("Input rollback failed; poisoned") from error
            raise


def checked_score(score):
    require(type(score) is dict and type(score.get("eligible")) is bool and "rank_gain_db" in score,
            "Original policy eligible/rank fields required")
    rank = score["rank_gain_db"]
    require(rank is None or type(rank) is float and math.isfinite(rank), "Typed finite rank, not loss or boolean")
    require(not score["eligible"] or rank is not None, "Eligible score needs rank")
    storage.typed_tree(score)


def apply_raw_observation(schedule, step, score):
    """Original143 best/patience semantics for the ONE active raw trajectory."""
    checked_score(score)
    c, arm = schedule["config"], RAW_ARM
    best, anchor, rank = schedule["best"][arm], schedule["patience_anchor"][arm], score["rank_gain_db"]
    improved = score["eligible"] and (anchor is None or rank >= anchor+c["minimum_rank_gain_for_patience_reset_db"])
    if score["eligible"] and (best is None or rank > best["rank_gain_db"]):
        schedule["best"][arm] = {"step": step, "rank_gain_db": rank}
    if improved:
        schedule["patience_anchor"][arm], schedule["stale"][arm] = rank, 0
    else:
        schedule["stale"][arm] += 1
    schedule["last_validation"] = step
    return (step >= c["minimum_steps_before_early_stop"] and
            schedule["stale"][arm] >= c["patience_validation_checks"])


class LiveContext:
    """Replay-authenticated metadata. Step integers are NOT Adam update proof.

    Old stopped_at4500 remains immutable. Independent fixed500 budget and
    original raw patience events are separate; EMA scores never feed them.
    Full trainer must jointly commit this with raw/Adam/shadow/CPU-CUDA RNG.
    """
    def __init__(self, source_context, provenance):
        require(type(source_context) is dict and list(source_context) == ["step", "sampler", "schedule"] and
                type(source_context["step"]) is int and source_context["step"] == START,
                "Full initial source context4500")
        sampler, sch = source_context["sampler"], source_context["schedule"]
        require(type(sampler["cursor"]) is int and sampler["cursor"] == START and
                type(sch["step"]) is int and sch["step"] == START and type(sch["last_validation"]) is int and
                sch["last_validation"] == START and type(sch["stopped_at"]) is int and sch["stopped_at"] == START,
                "Do not erase original closed4500 stop")
        require(equal(sch["best"], dict.fromkeys((RAW_ARM, OTHER_ARM))) and
                equal(sch["stale"], dict.fromkeys((RAW_ARM, OTHER_ARM), 18)) and
                equal(sch["patience_anchor"], dict.fromkeys((RAW_ARM, OTHER_ARM))), "Actual4500 cumulative fields, not4000")
        expected_sampler = {"approval_sha256": storage.source.PINS["results/mel_lr_scale_import_20261004/approval.json"],
                            "true_lock_sha256": "36a9be8ade7822969a7fed661c8f226f49b89e8ece90e1e6192ae0ae1b4b2c38",
                            "seed": 20261002, "cursor": START, "teacher": "kim_melband"}
        require(equal(sampler, expected_sampler), "Actual complete source sampler identity")
        expected_config = {"maximum_steps": 10000, "minimum_steps_before_early_stop": 2000,
                           "learning_rate": .0001, "warmup_steps": 100, "cosine_min_learning_rate": .00001,
                           "validation_every": 250, "patience_validation_checks": 8,
                           "minimum_rank_gain_for_patience_reset_db": .02}
        require(all(equal(sch["config"][k], v) for k, v in expected_config.items()), "Original LR/patience configuration")
        require(equal(provenance, {"source_sha256": storage.source.SOURCE_SHA,
                                  "source_limit": 4500, "source_legacy_stop_events": [3750, 4000],
                                  "legacy_stop_events": [4250, 4500]}), "Full immutable stop provenance")
        self._source, self._provenance = portable(source_context), portable(provenance)
        self._immutable_digest = storage.digest([self._source, self._provenance])
        self.context, self.journal, self.patience_events = portable(source_context), [], []
        self._poisoned, self._in_transaction = False, False

    @property
    def step(self):
        return self.context["step"]

    @property
    def validation_due(self):
        return self.step in STAGES and self.context["schedule"]["last_validation"] != self.step

    def learning_rate_next(self):
        require(self.step < LIMIT and not self.validation_due, "Full validation/hard budget gate")
        c, step = self.context["schedule"]["config"], self.step+1
        if step <= c["warmup_steps"]:
            return c["learning_rate"]*step/c["warmup_steps"]
        progress = (step-c["warmup_steps"])/(c["maximum_steps"]-c["warmup_steps"])
        return c["cosine_min_learning_rate"]+(c["learning_rate"]-c["cosine_min_learning_rate"])*(1+math.cos(math.pi*progress))/2

    def complete_step(self, sampler):
        self.state_dict()
        self.learning_rate_next()
        validate_sampler(sampler, self._source["sampler"])
        require(sampler["cursor"] == self.step+1, "One contiguous input per committed step")
        self.context["step"] += 1
        self.context["schedule"]["step"] = self.step
        self.context["sampler"] = portable(sampler)

    def observe_raw(self, score):
        self.state_dict()
        require(self.validation_due, "Only one complete new4750/5000 observation")
        score = portable(score)
        checked_score(score)
        # Validate before mutation; no loss-only or EMA comparison is used.
        if apply_raw_observation(self.context["schedule"], self.step, score):
            self.patience_events.append(self.step)
        self.journal.append({"step": self.step, "raw_policy_score": score})

    def _replay(self, step, journal):
        require(type(step) is int and START <= step <= LIMIT and type(journal) is list, "Bounded replay exposure")
        expected_stages = [s for s in STAGES if s < step]
        if journal and journal[-1]["step"] == step and step in STAGES:
            expected_stages.append(step)
        require([r["step"] for r in journal] == expected_stages, "Missing/duplicated/out-of-order full observation")
        context, events = portable(self._source), []
        for row in journal:
            require(type(row) is dict and list(row) == ["step", "raw_policy_score"] and type(row["step"]) is int,
                    "Exact observation journal schema")
            if apply_raw_observation(context["schedule"], row["step"], row["raw_policy_score"]):
                events.append(row["step"])
        context["step"] = context["sampler"]["cursor"] = context["schedule"]["step"] = step
        return context, events

    def state_dict(self):
        require(not self._poisoned, "Poisoned live context")
        require(storage.digest([self._source, self._provenance]) == self._immutable_digest, "Source context/provenance changed")
        expected, events = self._replay(self.step, self.journal)
        require(equal(self.context, expected) and equal(self.patience_events, events), "Live metadata differs from full source+policy replay")
        return storage.seal({"schema": 1, "purpose": PURPOSE, "source_context": portable(self._source),
                             "provenance": portable(self._provenance), "context": portable(self.context),
                             "journal": portable(self.journal), "patience_events": list(events),
                             "new_budget_limit": LIMIT,
                             "new_budget_stopped_at": LIMIT if self.step == LIMIT and not self.validation_due else None,
                             "training_authorized": False, "complete_training_container": False,
                             "release_selection": "NONE"})

    def load_state_dict(self, packet):
        require(not self._poisoned, "Poisoned live context")
        storage.check_seal(packet)
        current = self.state_dict()
        require(list(packet) == list(current), "Full live-context schema")
        variable = {"context", "journal", "patience_events", "new_budget_stopped_at", "content_sha256"}
        require(all(equal(packet[k], current[k]) for k in current if k not in variable), "Changed live-context identity/provenance")
        expected, events = self._replay(packet["context"]["step"], packet["journal"])
        stopped = LIMIT if expected["step"] == LIMIT and expected["schedule"]["last_validation"] == LIMIT else None
        require(equal(packet["context"], expected) and equal(packet["patience_events"], events) and
                equal(packet["new_budget_stopped_at"], stopped), "Tampered cumulative schedule/budget")
        self.context, self.journal, self.patience_events = portable(expected), portable(packet["journal"]), list(events)

    @contextmanager
    def input_schedule_transaction(self, stream):
        """CPU input/schedule rollback ONLY, no raw/Adam/shadow/CUDA claim."""
        require(not self._in_transaction, "Nested context transaction forbidden")
        previous, sampler, metadata, rng = self.state_dict(), stream.state_dict(), portable(stream.last_metadata), storage.capture_cpu_rng()
        require(equal(sampler, previous["context"]["sampler"]), "Input/schedule exposure mismatch before transaction")
        self._in_transaction = True
        try:
            yield self
            require(equal(stream.state_dict(), self.state_dict()["context"]["sampler"]), "Uncommitted input/schedule exposure")
        except BaseException:
            try:
                # Restore before validating current live state, which may be partial.
                self.context, self.journal, self.patience_events = portable(previous["context"]), portable(previous["journal"]), list(previous["patience_events"])
                stream.load_state_dict(sampler, last_metadata=metadata)
                storage.restore_cpu_rng(rng)
                require(equal(self.state_dict(), previous) and equal(stream.state_dict(), sampler), "Context rollback differs")
            except BaseException as error:
                self._poisoned = True
                raise RuntimeError("Input/schedule rollback failed; poisoned") from error
            raise
        finally:
            self._in_transaction = False
