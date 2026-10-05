"""New joint207 CPU integration units; no closed suite or real training.

One sealed205 storage read/actual raw Module/matching CPU Adam construction.
All later states are manually mutated tiny synthetic fixtures. No Adam.step,
audio, forward, autograd engine or CUDA. Joint disk/rollback is CPU-only.
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

spec = importlib.util.spec_from_file_location("joint207", Path(__file__).with_name("207_ema_live_cpu_state.py"))
j = importlib.util.module_from_spec(spec); spec.loader.exec_module(j)
c, l = j.c, j.live
OUT = j.ROOT / "results/mel_ema_live_cpu_state_monitor_20261004"


def reseal(packet):
    packet.pop("content_sha256", None)
    return c.seal(packet)


def metadata():
    result = []
    for index, domain in enumerate(l.DOMAINS):
        row = {"domain": domain, "role": "train" if index < 3 else "pseudo_label_train_candidate",
               "vocal_db": 0, "score_start": 25088, "score_end": 89344, "input_pcm_sha256": "0"*64}
        if index >= 3:
            row.update(purpose="NONRELEASE_PAIRED_EXPLORATION", exploratory_eligible=True,
                       deployment_eligible=False, training_eligible=False)
        result.append(row)
    return result


class TrueFixture:
    def __init__(self, sampler):
        self.bound, self.config = sampler["true_lock_sha256"], {"synthetic_cpu_route_only": True}
        self.fail = False

    def crop(self, domain, seed, cursor):
        if self.fail:
            random.random(); np.random.random(); torch.rand(1)
            raise KeyboardInterrupt("synthetic backend failure")
        x = torch.full((2, 89856), .125)
        row = metadata()[l.DOMAINS.index(domain)]
        row["input_pcm_sha256"] = hashlib.sha256(x.numpy().tobytes()).hexdigest()
        return {"x": x, "v": torch.zeros_like(x), "meta": row}


class TeacherFixture:
    def __init__(self):
        self.config, self.seed = {"synthetic_cpu_route_only": True}, 20261002
        self.bound, self.rows = "synthetic_not_audio_cache", [{"synthetic_song": 1}]

    def crop(self, recipe):
        x = torch.full((2, 89856), .25)
        row = metadata()[3]
        row.update(recipe)
        row["input_pcm_sha256"] = hashlib.sha256(x.numpy().tobytes()).hexdigest()
        return {"x": x, "v": torch.zeros_like(x), "meta": row}


def synthetic_recipe(rows, config, seed, counter):
    return {"synthetic_counter": counter}


def stream_factory(sampler):
    return l.BoundedSingleTargetStream(sampler, TrueFixture(sampler), TeacherFixture(), synthetic_recipe)


def fixture(context, stop_provenance):
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor([1., -0., -2.])))
    model.register_parameter("fixed", torch.nn.Parameter(torch.tensor([3., -0.]), requires_grad=False))
    model.branch = torch.nn.Module()
    model.branch.register_parameter("bias", torch.nn.Parameter(torch.tensor([.125])))
    model.branch.register_buffer("count", torch.tensor(4500, dtype=torch.int64))
    model.branch.register_buffer("scratch", torch.tensor([-0., 7.]), persistent=False)
    model.branch.training = False
    model.weight.grad = torch.tensor([.1, -.2, -0.])
    config = context["schedule"]["config"]
    # math.cos is the original computation path, including scalar rounding.
    import math
    lr = config["cosine_min_learning_rate"] + (config["learning_rate"]-config["cosine_min_learning_rate"])*(1+math.cos(math.pi*(4500-100)/9900))/2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False, fused=False)
    for parameter in model.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(4500.), "exp_avg": torch.full_like(parameter, .125),
                                      "exp_avg_sq": torch.full_like(parameter, .25)}
    parent = {"synthetic_fixture": True, "source_sha256": c.source.SOURCE_SHA,
              "saved_cuda_rng": [torch.arange(16, dtype=torch.uint8)], "old_stop": 4500}
    return j.LiveCpuStateOwner(model, optimizer, c.ema.EmaShadow(model, source_sha256=c.source.SOURCE_SHA),
                              parent, context, stream_factory(context["sampler"]),
                              evidence_scope="synthetic_cpu_live_state_fixture", stop_provenance=stop_provenance)


def manual_step(owner, *, draw=False):
    """Manually set tensor/moment exposure. Never optimizer.step or real input."""
    step, lr = owner.live.step+1, owner.live.learning_rate_next()
    if draw:
        owner.stream.next_batch()
    else:
        owner.stream.load_state_dict(owner.stream.state_dict() | {"cursor": step}, last_metadata=metadata())
    with torch.no_grad():
        owner.model.weight.add_(.015625)
        owner.model.branch.bias.sub_(.0078125)
        owner.model.branch.count.add_(1)
        owner.model.branch.scratch.add_(.03125)
        for parameter in owner.model.parameters():
            state = owner.optimizer.state[parameter]
            state["step"].fill_(step); state["exp_avg"].add_(.001953125); state["exp_avg_sq"].add_(.00390625)
            parameter.grad = None
    owner.optimizer.param_groups[0]["lr"] = lr
    owner.complete_raw_storage_step()


class JointCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch._dynamo  # Register lazy optimizer rules before distinct guards; no compile.
        cls.outer_rng = c.capture_cpu_rng()
        cls.guards = ExitStack()
        cls.addClassCleanup(cls.guards.close)
        cls.addClassCleanup(c.restore_cpu_rng, cls.outer_rng)
        for target, name in ((torch.nn.Module, "_call_impl"), (torch.autograd, "grad"), (torch.autograd, "backward"),
                             (torch.Tensor, "backward"), (torch.optim.Adam, "step"), (torch.cuda, "_lazy_init")):
            def block(label):
                def forbidden(*args, **kwargs):
                    raise RuntimeError("Forbidden207 execution: " + label)
                return forbidden
            cls.guards.enter_context(patch.object(target, name, block(name)))
        cls.actual_calls = {"storage_source_reads": 0, "actual_raw_models": 0, "actual_matching_cpu_adam_constructions": 0}
        target = c._module("joint207_actual09", "scripts/09_target_model.py")
        torch.set_num_threads(2)
        original_load, original_ctor = torch.load, torch.optim.Adam.__init__
        def read(path, *args, **kwargs):
            if Path(path) == j.ROOT / j.STORAGE_REL:
                cls.actual_calls["storage_source_reads"] += 1
            return original_load(path, *args, **kwargs)
        def factory():
            cls.actual_calls["actual_raw_models"] += 1
            return target.CausalSpectralUNet(bottleneck_blocks=2)
        def ctor(instance, *args, **kwargs):
            cls.actual_calls["actual_matching_cpu_adam_constructions"] += 1
            return original_ctor(instance, *args, **kwargs)
        with c.preserve_cpu_rng(), patch.object(torch, "load", read), patch.object(torch.optim.Adam, "__init__", ctor):
            cls.actual = j.LiveCpuStateOwner.from_fixed_cpu_storage(factory, stream_factory)
        cls.source_context = c.portable(cls.actual._context_template)
        cls.stop_provenance = c.portable(cls.actual._provenance)
        # One synthetic combined trajectory builds late-stage packets. It is
        # manually mutated storage, not500 inputs/Adam updates/training steps.
        stage = fixture(cls.source_context, cls.stop_provenance)
        for step in range(4501, 5001):
            with stage.transaction():
                manual_step(stage)
                if step in (4750, 5000):
                    if step == 4750:
                        cls.pending4750 = stage.state_dict()
                    stage.observe_raw_policy_metadata({"eligible": True, "rank_gain_db": .1 if step == 4750 else .11})
            if step == 4750:
                cls.stage4750 = stage.state_dict()
        cls.stage5000 = stage.state_dict()

    @classmethod
    def tearDownClass(cls):
        j.check_dependencies()
        assert c.sha256(j.ROOT / j.STORAGE_REL) == j.STORAGE_SHA
        for path, expected in c.source.PINS.items():
            assert c.sha256(j.ROOT / path) == expected
        assert not torch.cuda.is_initialized()
        c.restore_cpu_rng(cls.outer_rng)
        assert c.equal(c.capture_cpu_rng(), cls.outer_rng)
        print("JOINT_CPU_EVIDENCE " + json.dumps({**cls.actual_calls, "original_training_source_pt_reads": 0,
              "synthetic_manual_storage_exposures": 500, "model_forwards": 0, "autograd_engine": 0,
              "adam_steps": 0, "student_training_updates": 0, "real_audio_inputs": 0,
              "cuda_initialized": False, "outer_rng_restored": True, "source_files_unchanged": True,
              "whole_cpu_storage_scope": True, "actual_audio_backend_verified": False,
              "cuda_transaction_verified": False, "new_training_authorized": False}, sort_keys=True))

    def setUp(self):
        self.rng = c.capture_cpu_rng()
        self.addCleanup(c.restore_cpu_rng, self.rng)
        self.owner = fixture(self.source_context, self.stop_provenance)

    def reject(self, mutation, *, nested=True, outer=True):
        before = self.owner.state_dict(); changed = copy.deepcopy(before)
        mutation(changed)
        if nested:
            changed["live_context"] = reseal(changed["live_context"])
            changed["input_state"] = reseal(changed["input_state"])
        if outer: changed = reseal(changed)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.owner.load_state_dict(changed)
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_actual_zero_update_full_disk_bit_restore(self):
        before = self.actual.state_dict()
        self.assertEqual(self.actual_calls, {"storage_source_reads": 1, "actual_raw_models": 1,
                                           "actual_matching_cpu_adam_constructions": 1})
        path = OUT / "source_live_cpu_container_4500.pt"
        sha = self.actual.save_new(path)
        disk = self.actual.read_checked(path, sha)
        self.actual.load_state_dict(disk)
        self.assertTrue(c.equal(before, self.actual.state_dict()))
        c.noalias(disk, self.actual._external())
        self.assertEqual(len(before["raw"]["parameter_names"]), 22)
        self.assertEqual(before["raw"]["modes"], [True]*27)
        self.assertEqual(before["parent"]["parent_metadata"]["limit"], 4500)
        self.assertEqual(before["rng"]["torch_cuda"], [])
        print("ACTUAL_JOINT_STORAGE " + json.dumps({"path": str(path.relative_to(j.ROOT)), "sha256": sha,
              "step": 4500, "source_storage_sha256": j.STORAGE_SHA, "full_typed_bit_restore": True,
              "source_cuda_rng_retained_not_initialized": True, "zero_update_not_training_checkpoint": True}, sort_keys=True))

    def test_source_cumulative_stop_and_e0(self):
        packet = self.owner.state_dict()
        sch = packet["live_context"]["context"]["schedule"]
        self.assertEqual((sch["stale"][l.RAW_ARM], sch["stopped_at"]), (18, 4500))
        self.assertTrue(c.equal(packet["raw"]["tensors"], packet["shadow"]["tensors"]))

    def test_modes_gradients_nonpersistent_buffers_and_defaults(self):
        raw = self.owner.state_dict()["raw"]
        self.assertEqual(raw["modes"], [True, False])
        self.assertIn("branch.scratch", raw["tensors"])
        self.assertIsNotNone(raw["gradients"]["weight"])
        self.assertIsNone(raw["gradients"]["fixed"])
        self.assertTrue(c.equal(raw["optimizer_defaults"], self.owner.optimizer.defaults))

    def test_synthetic_route_raw_shadow_live_one_commit(self):
        with self.owner.transaction(): manual_step(self.owner, draw=True)
        packet = self.owner.state_dict()
        self.assertEqual(packet["live_context"]["context"]["step"], 4501)
        self.assertEqual(packet["input_state"]["sampler"]["cursor"], 4501)
        self.assertEqual(packet["shadow"]["updates"], 1)
        self.assertFalse(c.equal(packet["shadow"]["tensors"]["weight"], packet["raw"]["tensors"]["weight"]))

    def test_full_late_disk_restore_with_legitimate_stale_reset_best_dictionary(self):
        self.owner.load_state_dict(self.stage4750)
        sch = self.owner.live.context["schedule"]
        self.assertEqual(sch["stale"][l.RAW_ARM], 0)
        self.assertEqual(sch["best"][l.RAW_ARM], {"step": 4750, "rank_gain_db": .1})
        with tempfile.TemporaryDirectory(prefix="ema207_", dir=OUT) as temp:
            path = Path(temp) / "late.pt"; sha = self.owner.save_new(path)
            packet = self.owner.read_checked(path, sha)
            self.owner.load_state_dict(self.stage5000)
            self.owner.load_state_dict(packet)
            self.assertTrue(c.equal(packet, self.owner.state_dict()))

    def test_5000_best_anchor_patience_and_hard_budget(self):
        self.owner.load_state_dict(self.stage5000)
        sch = self.owner.live.context["schedule"]
        self.assertEqual((sch["stale"][l.RAW_ARM], sch["best"][l.RAW_ARM], sch["patience_anchor"][l.RAW_ARM]),
                         (1, {"step": 5000, "rank_gain_db": .11}, .1))
        self.assertEqual((sch["stale"][l.OTHER_ARM], sch["stopped_at"]), (18, 4500))
        self.assertEqual(self.owner.state_dict()["live_context"]["new_budget_stopped_at"], 5000)
        before = self.owner.state_dict()
        with self.assertRaises(ValueError):
            with self.owner.transaction(): manual_step(self.owner)
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_stage_validation_pending_then_joint_observation(self):
        self.owner.load_state_dict(self.pending4750)
        self.assertTrue(self.owner.live.validation_due)
        with self.assertRaises(ValueError):
            with self.owner.transaction(): manual_step(self.owner)
        with self.owner.transaction():
            self.owner.observe_raw_policy_metadata({"eligible": True, "rank_gain_db": .1})
        self.assertTrue(c.equal(self.owner.state_dict(), self.stage4750))

    def test_noalias_and_no_feedback_export(self):
        before = self.owner.state_dict()
        self.owner.load_state_dict(before)
        c.noalias(before, self.owner._external())
        before["raw"]["tensors"]["weight"].add_(2)
        self.assertTrue(c.equal(self.owner.model.weight.detach(), self.owner.shadow._values["weight"]))

    def test_forward_autograd_step_cuda_guards_execute_before_engine(self):
        for fn in (lambda: self.owner.model(None), lambda: torch.autograd.grad(None, None),
                   lambda: self.owner.optimizer.step(), lambda: torch.cuda._lazy_init()):
            with self.assertRaises(RuntimeError): fn()

    def test_base_context_validator_not_called(self):
        with patch.object(c.CpuStateOwner, "_validate_context", side_effect=AssertionError("closed205 validator")):
            self.owner.load_state_dict(self.stage4750)
            with self.owner.transaction():
                manual_step(self.owner)
            self.owner.state_dict()

    def test_raw_only_optimizer_ownership(self):
        self.owner.optimizer.param_groups[0]["params"][0] = self.owner.shadow._values["weight"]
        with self.assertRaises(ValueError): self.owner.state_dict()

    def test_poison_after_rollback_failure(self):
        with patch.object(self.owner.stream, "load_state_dict", side_effect=RuntimeError("cannot restore")):
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                with self.owner.transaction(): raise KeyboardInterrupt("trigger")
        with self.assertRaisesRegex(ValueError, "Poisoned"): self.owner.state_dict()

    def test_nested_joint_transaction_rejected(self):
        before = self.owner.state_dict()
        with self.assertRaises(ValueError):
            with self.owner.transaction():
                with self.owner.transaction(): pass
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_partial_load_late_stream_failure_full_rollback(self):
        before = self.owner.state_dict()
        original, calls = self.owner.stream.load_state_dict, []
        def fail_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1: raise KeyboardInterrupt("after raw/Adam/shadow/live apply")
            return original(*args, **kwargs)
        with patch.object(self.owner.stream, "load_state_dict", side_effect=fail_once):
            with self.assertRaises(KeyboardInterrupt): self.owner.load_state_dict(self.stage4750)
        self.assertEqual(len(calls), 2)
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_rng_load_failure_full_state_rollback(self):
        before = self.owner.state_dict(); changed = copy.deepcopy(self.stage4750)
        changed["rng"]["torch_cpu"].zero_(); changed = reseal(changed)
        with self.assertRaises(RuntimeError): self.owner.load_state_dict(changed)
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_component_transaction_cannot_nest_inside_joint_boundary(self):
        # A foreign206 transaction is not a substitute for the joint boundary.
        # Partial metadata still fails the joint close and is fully restored.
        before = self.owner.state_dict()
        with self.assertRaises(ValueError):
            with self.owner.transaction():
                self.owner.stream.cursor += 1
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_partial_input_draw_rolls_back_whole_state(self):
        before = self.owner.state_dict()
        with self.assertRaises(ValueError):
            with self.owner.transaction(): self.owner.stream.next_batch()
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_backend_failure_rolls_back_rng_and_state(self):
        before = self.owner.state_dict(); self.owner.stream.true.fail = True
        with self.assertRaises(KeyboardInterrupt):
            with self.owner.transaction(): self.owner.stream.next_batch()
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_joint_storage_step_requires_transaction(self):
        with self.assertRaises(ValueError): self.owner.complete_raw_storage_step()
        with self.assertRaises(ValueError): self.owner.observe_raw_policy_metadata({"eligible": False, "rank_gain_db": None})

    def test_no_overwrite_and_hash_before_deserialize(self):
        with tempfile.TemporaryDirectory(prefix="ema207_", dir=OUT) as temp:
            path = Path(temp) / "state.pt"; sha = self.owner.save_new(path)
            with self.assertRaises(FileExistsError): self.owner.save_new(path)
            self.assertEqual(c.sha256(path), sha)
            with patch.object(torch, "load") as load:
                with self.assertRaises(ValueError): self.owner.read_checked(path, "0"*64)
                load.assert_not_called()

    def test_full_input_schema_types_and_support(self):
        self.owner.load_state_dict(self.stage4750)
        for field, value in (("score_start", 25088.), ("score_end", True), ("input_pcm_sha256", "X"*64)):
            self.reject(lambda p, field=field, value=value: p["input_state"]["last_metadata"][0].__setitem__(field, value))

    def test_missing_journal_duplicate_or_cumulative_edit_rejected(self):
        self.owner.load_state_dict(self.stage4750)
        self.reject(lambda p: p["live_context"]["journal"].clear())
        self.reject(lambda p: p["live_context"]["journal"].append(copy.deepcopy(p["live_context"]["journal"][0])))
        self.reject(lambda p: p["live_context"]["context"]["schedule"]["stale"].__setitem__(l.RAW_ARM, 1))

    def test_initial_reset_and_old_stop_or_historical_arm_edit_rejected(self):
        for field, value in (("stale", 0), ("best", {"step": 4500, "rank_gain_db": .1}), ("patience_anchor", .1)):
            self.reject(lambda p, field=field, value=value: p["live_context"]["context"]["schedule"][field].__setitem__(l.RAW_ARM, value))
        self.reject(lambda p: p["live_context"]["context"]["schedule"].__setitem__("stopped_at", 5000))
        self.reject(lambda p: p["live_context"]["context"]["schedule"]["stale"].__setitem__(l.OTHER_ARM, 0))

    def test_parent_cuda_rng_and_stop_provenance_immutable(self):
        self.reject(lambda p: p["parent"]["saved_cuda_rng"][0].zero_())
        self.reject(lambda p: p["live_context"]["provenance"].__setitem__("legacy_stop_events", []))

    def test_raw_moment_defaults_flags_partial_exposure_rejected(self):
        self.reject(lambda p: p["raw"]["optimizer"]["state"][0]["step"].fill_(4501))
        self.reject(lambda p: p["raw"]["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1))
        self.reject(lambda p: p["raw"]["optimizer_defaults"].__setitem__("eps", 1.0))
        self.reject(lambda p: p["raw"]["optimizer"]["param_groups"][0].__setitem__("fused", True))
        self.reject(lambda p: p["shadow"].__setitem__("updates", 1))

    def test_nested_seal_not_stripped_or_only_root_checked(self):
        self.reject(lambda p: p["input_state"].pop("content_sha256"), nested=False)
        self.reject(lambda p: p["live_context"].pop("content_sha256"), nested=False)
        self.reject(lambda p: p["input_state"].__setitem__("extra", 1), nested=False)

    def test_root_types_and_training_authority_rejected(self):
        for field, value in (("schema", True), ("training_authorized", True), ("actual_audio_backend_verified", True),
                             ("cuda_transaction_verified", True), ("release_selection", "EMA")):
            self.reject(lambda p, field=field, value=value: p.__setitem__(field, value))
        self.reject(lambda p: p.pop("rng"))

    def test_partial_sampler_and_metadata_exposure_rejected(self):
        self.reject(lambda p: p["input_state"]["sampler"].__setitem__("cursor", 4501))
        self.reject(lambda p: p["input_state"].__setitem__("last_metadata", metadata()))
        self.owner.load_state_dict(self.stage4750)
        self.reject(lambda p: p["input_state"].__setitem__("last_metadata", None))

    def test_incoming_and_internal_alias_rejected(self):
        self.reject(lambda p: p["raw"]["gradients"].__setitem__("weight", p["raw"]["tensors"]["weight"]))
        self.reject(lambda p: p["raw"]["tensors"].__setitem__("weight", self.owner.model.weight.detach()))

    def test_signed_zero_modes_dtype_and_rng_types_preserved(self):
        self.reject(lambda p: p["shadow"]["tensors"]["fixed"].__setitem__(1, 0.))
        self.reject(lambda p: p["raw"]["modes"].__setitem__(0, 1))
        self.reject(lambda p: p["raw"]["tensors"].__setitem__("weight", p["raw"]["tensors"]["weight"].double()))
        self.reject(lambda p: p["rng"].__setitem__("torch_cuda", [torch.zeros(16, dtype=torch.uint8)]))


def failure_test(point):
    def test(self):
        if point == "after_observation": self.owner.load_state_dict(self.pending4750)
        before = self.owner.state_dict()
        with self.assertRaises(KeyboardInterrupt):
            with self.owner.transaction():
                if point == "after_observation":
                    self.owner.observe_raw_policy_metadata({"eligible": True, "rank_gain_db": .1})
                elif point == "after_joint_commit": manual_step(self.owner)
                elif point == "partial_live":
                    self.owner.live.context["step"] = 4501; self.owner.live.journal.append({"incomplete": True})
                elif point == "existing_grad": self.owner.model.weight.grad.fill_(99)
                elif point == "defaults": self.owner.optimizer.defaults["weight_decay"] = 0.0
                elif point == "modes_buffer":
                    self.owner.model.branch.training = True; self.owner.model.branch.scratch.fill_(9)
                elif point == "raw_moments":
                    with torch.no_grad(): self.owner.model.weight.add_(1)
                    self.owner.optimizer.state[self.owner.model.weight]["exp_avg"].fill_(9)
                random.random(); np.random.random(); torch.rand(1)
                raise KeyboardInterrupt("failure after " + point)
        self.assertTrue(c.equal(before, self.owner.state_dict()))
    return test


for point in ("after_observation", "after_joint_commit", "partial_live", "existing_grad", "defaults", "modes_buffer", "raw_moments"):
    setattr(JointCpuTests, "test_whole_cpu_rollback_" + point, failure_test(point))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    c.require(not (OUT / "unit_gate.json").exists() and not (OUT / "source_live_cpu_container_4500.pt").exists(),
              "Refuse repeated sealed units or existing actual storage output")
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
