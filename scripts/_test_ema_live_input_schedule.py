"""New206 metadata/routing units, no closed suite, real audio or model update.

Read the sealed205 zero-update storage artifact ONCE for real initial metadata;
never reread the old source PT, instantiate raw/Adam, or decode music. Synthetic
cropping and counter transitions are NOT next-input mechanisms or training.
"""
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


l = load("live206", "206_ema_live_input_schedule.py")
c = l.storage
# Original recipe/schedule implementations, imported without invoking their CLI.
original = load("live206_original143", "143_paired_distillation_mechanics.py")
OUT = l.ROOT / "results/mel_ema_live_context_monitor_20261004"
SOURCE_STORAGE = l.ROOT / "results/mel_ema_cpu_state_monitor_20261004/source_cpu_container_4500_complete_defaults.pt"
SOURCE_STORAGE_SHA = "b003b48e46e1e573491c7359251899a600e3daa542462204d38140aa7a11254a"


def score(eligible=False, rank=None):
    return {"eligible": eligible, "rank_gain_db": rank}


class TrueFixture:
    """Synthetic backend; uses no true corpus even though domain strings match."""
    def __init__(self, sampler, config):
        self.bound, self.config, self.calls, self.fail = sampler["true_lock_sha256"], copy.deepcopy(config), [], False

    def crop(self, domain, seed, cursor):
        self.calls.append((domain, seed, cursor))
        if self.fail:
            random.random(); np.random.random(); torch.rand(1)
            raise KeyboardInterrupt("synthetic draw failure")
        value = torch.full((2, 89856), (l.DOMAINS.index(domain)+1)*.03125)
        meta = {"domain": domain, "role": "train", "vocal_db": 0, "score_start": 25088,
                "score_end": 89344, "input_pcm_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}
        return {"x": value, "v": value*.25 if domain != "instrumental" else torch.zeros_like(value), "meta": meta}


class TeacherFixture:
    def __init__(self, config):
        self.seed, self.config, self.bound = 20261002, copy.deepcopy(config), "synthetic_teacher_snapshot_not_actual_audio"
        self.rows = [{"song_id": "synthetic_0", "source": {"samples": 176400}},
                     {"song_id": "synthetic_1", "source": {"samples": 200000}}]
        self.calls, self.bad = [], None

    def crop(self, recipe):
        self.calls.append(copy.deepcopy(recipe))
        x = torch.full((2, 89856), .125)
        meta = {**recipe, "input_pcm_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                "purpose": "NONRELEASE_PAIRED_EXPLORATION", "exploratory_eligible": True,
                "deployment_eligible": False, "training_eligible": False}
        if self.bad == "nan": x[0, 0] = float("nan")
        if self.bad == "role": meta["deployment_eligible"] = True
        if self.bad == "pcm": meta["input_pcm_sha256"] = "0"*64
        if self.bad == "dtype": x = x.double()
        return {"x": x, "v": x*.5, "meta": meta}


class LiveContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outer_rng = c.capture_cpu_rng()
        cls.guards = ExitStack()
        def block(label):
            def forbidden(*args, **kwargs):
                raise AssertionError("Forbidden206 execution: " + label)
            return forbidden
        for owner, name in ((torch.nn.Module, "__init__"), (torch.nn.Module, "_call_impl"),
                            (torch.autograd, "grad"), (torch.autograd, "backward"),
                            (torch.Tensor, "backward"), (torch.optim.Adam, "__init__"),
                            (torch.optim.Adam, "step"), (torch.cuda, "_lazy_init")):
            cls.guards.enter_context(patch.object(owner, name, block(name)))
        cls.addClassCleanup(cls.guards.close)
        cls.addClassCleanup(c.restore_cpu_rng, cls.outer_rng)
        assert c.sha256(SOURCE_STORAGE) == SOURCE_STORAGE_SHA
        cls.saved = torch.load(SOURCE_STORAGE, weights_only=True, map_location="cpu")
        c.check_seal(cls.saved)
        assert c.sha256(SOURCE_STORAGE) == SOURCE_STORAGE_SHA
        cls.source = copy.deepcopy(cls.saved["context"])
        parent = cls.saved["parent"]["parent_metadata"]
        cls.provenance = {"source_sha256": cls.saved["source_sha256"], "source_limit": parent["limit"],
                          "source_legacy_stop_events": parent["source_legacy_stop_events"],
                          "legacy_stop_events": parent["legacy_stop_events"]}
        cls.input_config = json.loads((l.ROOT / "docs/teacher_distillation_protocol_v1.json").read_text(encoding="utf-8"))["input"]
        cls.saved_digest = c.digest(cls.saved)

    @classmethod
    def tearDownClass(cls):
        assert c.sha256(SOURCE_STORAGE) == SOURCE_STORAGE_SHA
        assert c.digest(cls.saved) == cls.saved_digest
        assert not torch.cuda.is_initialized()
        c.restore_cpu_rng(cls.outer_rng)
        assert c.equal(c.capture_cpu_rng(), cls.outer_rng)
        print("LIVE_CONTEXT_EVIDENCE " + json.dumps({
            "metadata_source_storage_reads": 1, "original_training_source_pt_reads": 0,
            "live_student_modules": 0, "model_forwards": 0, "autograd_engine": 0,
            "adam_constructions": 0, "adam_steps": 0, "student_training_updates": 0,
            "real_audio_crops": 0, "cuda_initialized": False,
            "actual_initial_cursor": 4500, "actual_initial_stale": 18,
            "source_storage_unchanged": True, "outer_rng_restored": True,
            "synthetic_only": True, "full_training_integration": False}, sort_keys=True))

    def setUp(self):
        self.live = l.LiveContext(self.source, self.provenance)
        self.true, self.teacher = TrueFixture(self.source["sampler"], self.input_config), TeacherFixture(self.input_config)
        self.stream = l.BoundedSingleTargetStream(self.source["sampler"], self.true, self.teacher, original.data.crop_recipe)

    def advance(self, step):
        # Metadata counter transitions ONLY, no tensor/moment/model mutation.
        while self.live.step < step:
            self.live.complete_step(self.live.context["sampler"] | {"cursor": self.live.step+1})

    def reject(self, mutation):
        before = self.live.state_dict()
        changed = copy.deepcopy(before)
        changed.pop("content_sha256")
        mutation(changed)
        changed = c.seal(changed)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.live.load_state_dict(changed)
        self.assertTrue(c.equal(before, self.live.state_dict()))

    def test_actual_source_metadata_and_old_stops_preserved(self):
        self.assertTrue(c.equal(self.live.context, self.source))
        self.assertEqual(self.live.context["schedule"]["stale"][l.RAW_ARM], 18)
        self.assertEqual(self.live.context["schedule"]["stopped_at"], 4500)
        self.assertIsNone(self.live.state_dict()["new_budget_stopped_at"])

    def test_initial_stale_reset_rejected(self):
        changed = copy.deepcopy(self.source)
        changed["schedule"]["stale"][l.RAW_ARM] = 0
        with self.assertRaises(ValueError): l.LiveContext(changed, self.provenance)

    def test_lr_500_metadata_transitions_match_original143(self):
        for step in range(4501, 5001):
            self.assertEqual(self.live.learning_rate_next(), original.learning_rate(step, self.source["schedule"]["config"]))
            self.advance(step)
            if step in l.STAGES: self.live.observe_raw(score())
        self.assertEqual(self.live.state_dict()["new_budget_stopped_at"], 5000)
        self.assertEqual(self.live.context["schedule"]["stopped_at"], 4500)
        self.assertEqual(self.live.context["schedule"]["stale"][l.RAW_ARM], 20)
        self.assertEqual(self.live.patience_events, [4750, 5000])
        with self.assertRaises(ValueError): self.live.learning_rate_next()

    def test_pending_complete_observation_blocks_next_step(self):
        self.advance(4750)
        self.assertTrue(self.live.validation_due)
        with self.assertRaises(ValueError): self.live.learning_rate_next()
        self.live.observe_raw(score())
        self.assertFalse(self.live.validation_due)

    def test_valid_eligible_improvement_resets_patience_not_initial_state(self):
        self.advance(4750)
        self.live.observe_raw(score(True, .5))
        sch = self.live.context["schedule"]
        self.assertEqual(sch["best"][l.RAW_ARM], {"step": 4750, "rank_gain_db": .5})
        self.assertEqual(sch["stale"][l.RAW_ARM], 0)
        self.assertEqual(sch["patience_anchor"][l.RAW_ARM], .5)
        self.assertEqual(sch["stale"][l.OTHER_ARM], 18)
        self.live.state_dict()

    def test_original143_observation_algorithm_agrees(self):
        for first, second in ((score(), score()), (score(True, .5), score(True, .51)),
                              (score(True, .5), score(True, .53))):
            live = l.LiveContext(self.source, self.provenance)
            ref = original.SharedSchedule(self.source["schedule"]["config"])
            ref.load_state_dict(copy.deepcopy(self.source["schedule"]))
            for step, sc in ((4750, first), (5000, second)):
                while live.step < step: live.complete_step(live.context["sampler"] | {"cursor": live.step+1})
                ref.step, ref.stopped_at = step, None  # separate synthetic original formula reference
                ref.observe({l.RAW_ARM: sc, l.OTHER_ARM: score()})
                live.observe_raw(sc)
                for key in ("best", "stale", "patience_anchor"):
                    self.assertTrue(c.equal(live.context["schedule"][key][l.RAW_ARM], getattr(ref, key)[l.RAW_ARM]))

    def test_205_stale_floor_is_not_live_schedule_compatibility(self):
        # No205 unit or Module/Adam instance: call its pure validator only.
        old = object.__new__(c.CpuStateOwner)
        old._context_template = copy.deepcopy(self.source)
        self.advance(4750); self.live.observe_raw(score(True, .5))
        altered = copy.deepcopy(self.live.context)
        altered["schedule"]["best"][l.RAW_ARM] = None
        with self.assertRaisesRegex(ValueError, "stale"):
            old._validate_context(altered)

    def test_205_best_shape_is_not_live_schedule_compatibility(self):
        old = object.__new__(c.CpuStateOwner)
        old._context_template = copy.deepcopy(self.source)
        self.advance(4750); self.live.observe_raw(score(True, .5))
        altered = copy.deepcopy(self.live.context)
        altered["schedule"]["stale"][l.RAW_ARM] = 18
        with self.assertRaisesRegex(ValueError, "metric"):
            old._validate_context(altered)

    def test_raw_only_observe_rejects_pair_or_ema_substitution(self):
        self.advance(4750)
        with self.assertRaises(ValueError): self.live.observe_raw({"raw": score(), "ema": score(True, 100.)})
        self.assertTrue(self.live.validation_due)

    def test_invalid_score_rejected_before_mutation(self):
        self.advance(4750)
        for sc in (score(1, .5), score(True, None), score(True, float("nan")), score(True, 1), {"loss": .1}):
            before = self.live.state_dict()
            with self.assertRaises((ValueError, KeyError)): self.live.observe_raw(sc)
            self.assertTrue(c.equal(before, self.live.state_dict()))

    def test_duplicate_observation_rejected(self):
        self.advance(4750); self.live.observe_raw(score())
        with self.assertRaises(ValueError): self.live.observe_raw(score())

    def test_wrong_step_type_and_skip_rejected(self):
        for cursor in (4500, 4502, 4501., True, 5001):
            with self.assertRaises(ValueError): self.live.complete_step(self.source["sampler"] | {"cursor": cursor})

    def test_live_context_full_typed_metadata_disk_roundtrip(self):
        self.advance(4750); self.live.observe_raw(score(True, .5))
        packet = self.live.state_dict()
        with tempfile.TemporaryDirectory(prefix="ema206_metadata_") as directory:
            path = Path(directory) / "metadata_only.pt"
            with path.open("xb") as handle: torch.save(packet, handle)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        other = l.LiveContext(self.source, self.provenance)
        other.load_state_dict(loaded)
        self.assertTrue(c.equal(packet, other.state_dict()))
        other.journal[0]["raw_policy_score"]["rank_gain_db"] = .8
        self.assertTrue(c.equal(packet, self.live.state_dict()))

    def test_journal_cannot_be_dropped_or_schedule_forged(self):
        self.advance(4750); self.live.observe_raw(score(True, .5))
        self.reject(lambda p: p["journal"].clear())
        self.reject(lambda p: p["context"]["schedule"]["stale"].__setitem__(l.RAW_ARM, 18))

    def test_full_schema_seal_and_type_tamper_rejected(self):
        self.reject(lambda p: p.__setitem__("schema", True))
        self.reject(lambda p: p["provenance"].__setitem__("source_limit", 5000))
        self.reject(lambda p: p["context"]["sampler"].__setitem__("seed", 20261003))
        packet = self.live.state_dict(); packet.pop("content_sha256")
        with self.assertRaises(ValueError): self.live.load_state_dict(packet)

    def test_inactive_historical_arm_immutable(self):
        self.reject(lambda p: p["context"]["schedule"]["stale"].__setitem__(l.OTHER_ARM, 19))

    def test_source_metadata_not_aliased(self):
        before = copy.deepcopy(self.source)
        self.advance(4501)
        self.assertTrue(c.equal(before, self.source))
        packet = self.live.state_dict(); packet["source_context"]["step"] = 0
        self.live.state_dict()

    def test_single_target_route_original_order_and_counter_recipe(self):
        batch = self.stream.next_batch()
        self.assertEqual(list(batch), ["x", "v", "domains", "cursor", "metadata"])
        self.assertNotIn("targets", batch)  # no duplicate training arm
        self.assertEqual(batch["cursor"], 4500)
        self.assertEqual(self.true.calls, [(d, 20261002, 4500) for d in l.DOMAINS[:3]])
        expected = [original.data.crop_recipe(self.teacher.rows, self.input_config, 20261002, 13500+i) for i in range(3)]
        self.assertTrue(c.equal(expected, self.teacher.calls))
        self.assertEqual(self.stream.cursor, 4501)
        self.assertNotEqual(batch["x"].untyped_storage().data_ptr(), batch["v"].untyped_storage().data_ptr())

    def test_successful_cpu_input_schedule_transaction(self):
        with self.live.input_schedule_transaction(self.stream):
            self.stream.next_batch()
            self.live.complete_step(self.stream.state_dict())
        self.assertEqual(self.live.step, 4501)

    def test_uncommitted_input_is_rolled_back(self):
        before = self.live.state_dict()
        with self.assertRaises(ValueError):
            with self.live.input_schedule_transaction(self.stream): self.stream.next_batch()
        self.assertTrue(c.equal(before, self.live.state_dict()))
        self.assertEqual(self.stream.cursor, 4500)
        self.assertIsNone(self.stream.last_metadata)

    def test_baseexception_input_failure_restores_cursor_metadata_and_rng(self):
        before = c.capture_cpu_rng(); self.true.fail = True
        with self.assertRaises(KeyboardInterrupt): self.stream.next_batch()
        self.assertTrue(c.equal(before, c.capture_cpu_rng()))
        self.assertEqual(self.stream.cursor, 4500)
        self.assertIsNone(self.stream.last_metadata)

    def test_failure_after_full_context_transition_restores_both(self):
        before, rng = self.live.state_dict(), c.capture_cpu_rng()
        with self.assertRaises(KeyboardInterrupt):
            with self.live.input_schedule_transaction(self.stream):
                self.stream.next_batch(); self.live.complete_step(self.stream.state_dict())
                random.random(); torch.rand(1); np.random.random()
                raise KeyboardInterrupt()
        self.assertTrue(c.equal(before, self.live.state_dict()))
        self.assertTrue(c.equal(rng, c.capture_cpu_rng()))

    def test_nonfinite_role_pcm_and_dtype_draws_fail_without_cursor_commit(self):
        for bad in ("nan", "role", "pcm", "dtype"):
            self.teacher.bad = bad
            with self.assertRaises(ValueError): self.stream.next_batch()
            self.assertEqual(self.stream.cursor, 4500)

    def test_changed_backend_identity_recipe_or_seed_rejected(self):
        self.teacher.seed = 1
        with self.assertRaises(ValueError): self.stream.state_dict()
        self.teacher.seed = 20261002
        self.stream.recipe = lambda *a: {}
        with self.assertRaises(ValueError): self.stream.state_dict()

    def test_hard5000_no_backend_call(self):
        self.stream.load_state_dict(self.source["sampler"] | {"cursor": 5000})
        with self.assertRaises(ValueError): self.stream.next_batch()
        self.assertFalse(self.true.calls or self.teacher.calls)

    def test_cpu_context_rollback_failure_poison(self):
        with patch.object(self.stream, "load_state_dict", side_effect=RuntimeError("restore failure")):
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                with self.live.input_schedule_transaction(self.stream): raise KeyboardInterrupt()
        with self.assertRaisesRegex(ValueError, "Poisoned"): self.live.state_dict()

    def test_nested_context_transaction_rejected(self):
        with self.assertRaises(ValueError):
            with self.live.input_schedule_transaction(self.stream):
                with self.live.input_schedule_transaction(self.stream): pass

    def test_forbidden_calls_blocked_before_any_execution(self):
        for fn in (lambda: torch.nn.Module(), lambda: torch.optim.Adam([]),
                   lambda: torch.autograd.grad(None, None), lambda: torch.cuda._lazy_init()):
            with self.assertRaises(AssertionError): fn()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
