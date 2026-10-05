"""New205 CPU storage integration. Never run the closed203/204 unit suites.

Actual source: one204 read, one fresh raw model and one matching CPU Adam, zero
forward/backward/Adam steps. Synthetic transaction mutations are NOT training.
Actual disk artifact is source-state storage, not an updated student checkpoint.
"""
from collections import OrderedDict
from contextlib import ExitStack
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("ema_cpu205", Path(__file__).with_name("205_ema_cpu_state_transaction.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
OUT = c.ROOT / "results/mel_ema_cpu_state_monitor_20261004"


def forbidden(*args, **kwargs):
    raise RuntimeError("No forward/autograd/Adam step/CUDA in205 state units")


def fixture():
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor([1., -0., -2.])))
    model.register_parameter("fixed", torch.nn.Parameter(torch.tensor([3., -0.]), requires_grad=False))
    model.branch = torch.nn.Module()
    model.branch.register_parameter("bias", torch.nn.Parameter(torch.tensor([.125])))
    model.branch.register_buffer("count", torch.tensor(4500, dtype=torch.int64))
    model.branch.register_buffer("scratch", torch.tensor([-0., 7.]), persistent=False)
    model.branch.training = False
    model.weight.grad = torch.tensor([.1, -.2, -0.])
    model.branch.bias.grad = torch.tensor([.125])
    cfg = {"warmup_steps": 100, "maximum_steps": 10000, "learning_rate": .0001, "cosine_min_learning_rate": .00001}
    step = 4500
    lr = .00001 + (.0001-.00001)*(1+math.cos(math.pi*(step-100)/9900))/2
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), foreach=False, fused=False)
    for parameter in model.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(4500.), "exp_avg": torch.full_like(parameter, .125),
                                      "exp_avg_sq": torch.full_like(parameter, .25)}
    sampler = {"approval_sha256": "1"*64, "true_lock_sha256": "2"*64, "seed": 20261002, "cursor": step, "teacher": "kim_melband"}
    schedule = {"step": step, "last_validation": step, "stopped_at": step,
                "best": dict.fromkeys(c.source.ARMS), "stale": dict.fromkeys(c.source.ARMS, 18),
                "patience_anchor": dict.fromkeys(c.source.ARMS), "config": cfg}
    parent = {"synthetic_fixture": True, "source_sha256": c.source.SOURCE_SHA,
              "runtime": {"device": "cuda", "strict_source_marker": True},
              "saved_cuda_rng": [torch.arange(16, dtype=torch.uint8)],
              "parent_limit": 4500, "parent_stop": 4500, "source_legacy": [3750, 4000], "legacy": [4250, 4500]}
    shadow = c.ema.EmaShadow(model, source_sha256=c.source.SOURCE_SHA)
    return c.CpuStateOwner(model, optimizer, shadow, parent,
                           {"step": step, "sampler": sampler, "schedule": schedule},
                           evidence_scope="synthetic_cpu_state_fixture")


def synthetic_commit(owner):
    """Manual state mutation for rollback tests; optimizer.step NEVER runs."""
    step = owner.context["step"] + 1
    with torch.no_grad():
        owner.model.weight.add_(.25)
        owner.model.branch.bias.sub_(.125)
        owner.model.branch.count.add_(1)
        owner.model.branch.scratch.add_(.5)
        for param in owner.model.parameters():
            values = owner.optimizer.state[param]
            values["step"].fill_(step)
            values["exp_avg"].add_(.03125)
            values["exp_avg_sq"].add_(.0625)
            param.grad = None
    owner.optimizer.param_groups[0]["lr"] = owner._lr(step)
    owner.context["step"] = owner.context["sampler"]["cursor"] = owner.context["schedule"]["step"] = step
    owner.shadow.update_after_raw_step(owner.model, raw_step=step)


def reseal(packet):
    packet.pop("content_sha256")
    return c.seal(packet)


class CpuStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the installed optimizer's lazy decorator registry BEFORE mocks.
        # No graph is compiled and no optimizer/model computation executes.
        # Distinct blockers also keep mocked torch objects distinct in registries.
        import torch._dynamo
        cls.guards = ExitStack()
        for target, name in ((torch.nn.Module, "_call_impl"), (torch.autograd, "grad"),
                             (torch.autograd, "backward"), (torch.Tensor, "backward"),
                             (torch.optim.Adam, "step"), (torch.cuda, "_lazy_init")):
            def blocker(*args, **kwargs):
                return forbidden(*args, **kwargs)
            cls.guards.enter_context(patch.object(target, name, blocker))
        cls.addClassCleanup(cls.guards.close)
        cls.initial_rng = c.capture_cpu_rng()
        cls.addClassCleanup(c.restore_cpu_rng, cls.initial_rng)

    @classmethod
    def tearDownClass(cls):
        assert not torch.cuda.is_initialized()
        c.check_dependencies()

    def setUp(self):
        self.owner = fixture()
        self.addCleanup(c.restore_cpu_rng, c.capture_cpu_rng())

    def reject(self, change, *, fresh_seal=True):
        before = self.owner.state_dict()
        changed = copy.deepcopy(before)
        change(changed)
        if fresh_seal:
            changed = reseal(changed)
        with self.assertRaises((ValueError, TypeError, KeyError)):
            self.owner.load_state_dict(changed)
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_z_full_source4500_disk_restore_cpu_explicit_migration(self):
        before_rng = c.capture_cpu_rng()
        with c.preserve_cpu_rng():
            target = c._module("ema205_actual09", "scripts/09_target_model.py")
            # common.py sets threads at import; choose the existing unit CPU2
            # runtime BEFORE the owner captures its runtime identity.
            torch.set_num_threads(2)
            calls = {"source_reads": 0, "raw_models": 0, "adam_constructions": 0}
            original_read, original_ctor = c.source.load_fixed_source, torch.optim.Adam.__init__
            def read(*args, **kwargs):
                calls["source_reads"] += 1
                return original_read(*args, **kwargs)
            def factory():
                calls["raw_models"] += 1
                return target.CausalSpectralUNet(bottleneck_blocks=2)
            def ctor(instance, *args, **kwargs):
                calls["adam_constructions"] += 1
                original_ctor(instance, *args, **kwargs)
            with patch.object(c.source, "load_fixed_source", side_effect=read), patch.object(torch.optim.Adam, "__init__", ctor):
                owner, evidence = c.CpuStateOwner.from_fixed_source(factory)
            original = owner.state_dict()
            self.assertEqual(calls, {"source_reads": 1, "raw_models": 1, "adam_constructions": 1})
            self.assertEqual(original["context"]["schedule"]["stale"][c.source.RAW_ARM], 18)
            self.assertEqual(original["parent"]["parent_metadata"]["schedule"]["stopped_at"], 4500)
            self.assertEqual(original["parent"]["parent_metadata"]["limit"], 4500)
            self.assertEqual(len(original["parent"]["parent_metadata"]["rng"]["torch_cuda"]), 1)
            self.assertEqual(original["rng"]["torch_cuda"], [])
            self.assertEqual(original["raw"]["optimizer"]["param_groups"][0]["lr"], 6.281416799501188e-05)
            self.assertEqual(c.source.state_digest(owner.model.state_dict()), evidence["model_sha256"])
            path = OUT / "source_cpu_container_4500_complete_defaults.pt"
            disk_sha = owner.save_new(path)
            packet = owner.read_checked(path, disk_sha)
            owner.load_state_dict(packet)
            self.assertTrue(c.equal(original, owner.state_dict()))
            c.noalias(packet, [*owner.model.parameters(), *c.source.tensor_leaves(owner.optimizer.state_dict()), *owner.shadow._values.values()])
            for relative, expected in c.source.PINS.items():
                self.assertEqual(c.sha256(c.ROOT / relative), expected)
            print("ACTUAL_CPU_STORAGE_EVIDENCE " + json.dumps({**calls, "source_sha256": c.source.SOURCE_SHA,
                  "selected_arm": c.source.RAW_ARM, "step": 4500, "stale": 18,
                  "disk_path": str(path.relative_to(c.ROOT)), "disk_sha256": disk_sha,
                  "full_typed_bit_restore": True, "noalias": True, "cuda_initialized": False,
                  "source_cuda_rng_preserved_in_parent": True, "active_device": "cpu_explicit_migration",
                  "model_forwards": 0, "autograd_engine": 0, "adam_steps": 0, "student_updates": 0,
                  "real_audio_inputs": 0, "new_training_authorized": False, "cuda_transaction_verified": False}, sort_keys=True))
        self.assertTrue(c.equal(before_rng, c.capture_cpu_rng()))

    def test_existing_grad_none_modes_and_nonpersistent_buffers_saved(self):
        packet = self.owner.state_dict()
        self.assertIsNone(packet["raw"]["gradients"]["fixed"])
        self.assertTrue(c.equal(packet["raw"]["gradients"]["weight"], self.owner.model.weight.grad))
        self.assertEqual(packet["raw"]["modes"], [True, False])
        self.assertIn("branch.scratch", packet["raw"]["tensors"])
        self.assertNotIn("branch.scratch", self.owner.model.state_dict())

    def test_noalias_export_and_restore_inputs(self):
        packet = self.owner.state_dict()
        untouched = copy.deepcopy(packet)
        self.owner.load_state_dict(packet)
        c.noalias(packet, [*self.owner.model.parameters(), *self.owner.model.buffers(),
                          *c.source.tensor_leaves(self.owner.optimizer.state_dict()), *self.owner.shadow._values.values()])
        self.assertTrue(c.equal(packet, untouched))
        packet["raw"]["tensors"]["weight"].add_(1)
        self.assertTrue(c.equal(self.owner.state_dict(), untouched))

    def test_snapshot_save_load_do_not_advance_rng(self):
        before = c.capture_cpu_rng()
        packet = self.owner.state_dict()
        self.owner.load_state_dict(packet)
        self.assertTrue(c.equal(before, c.capture_cpu_rng()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_disk_restore_with_existing_grads_and_synthetic_later_state(self):
        with tempfile.TemporaryDirectory(prefix="ema205_", dir=OUT) as temp:
            path = Path(temp) / "synthetic4500.pt"
            initial = self.owner.state_dict()
            file_sha = self.owner.save_new(path)
            with self.owner.transaction():
                synthetic_commit(self.owner)
                random.random(); np.random.random(); torch.rand(2)
            later = self.owner.state_dict()
            later_path = Path(temp) / "synthetic4501.pt"
            later_sha = self.owner.save_new(later_path)
            self.owner.load_state_dict(self.owner.read_checked(path, file_sha))
            self.assertTrue(c.equal(initial, self.owner.state_dict()))
            self.owner.load_state_dict(self.owner.read_checked(later_path, later_sha))
            self.assertTrue(c.equal(later, self.owner.state_dict()))

    def test_whole_transaction_failure_after_all_state_and_rng_changes(self):
        before = self.owner.state_dict()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with self.owner.transaction():
                synthetic_commit(self.owner)
                self.owner.context["schedule"]["stale"][c.source.RAW_ARM] += 1
                random.random(); np.random.random(); torch.rand(3)
                self.owner.model.branch.training = True
                self.owner.model.weight.grad = torch.full_like(self.owner.model.weight, 999)
                self.owner.optimizer.defaults["eps"] = 1.0
                raise RuntimeError("injected after raw-Adam-shadow-context-RNG")
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_nonfinite_after_raw_mutation_rolls_back_complete_state(self):
        before = self.owner.state_dict()
        with self.assertRaises(ValueError):
            with self.owner.transaction():
                self.owner.model.weight.data[0] = float("nan")
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_partial_shadow_after_raw_mutation_rolls_back(self):
        before = self.owner.state_dict()
        with self.assertRaises(ValueError):
            with self.owner.transaction():
                self.owner.model.weight.data.add_(10)
                self.owner.context["step"] += 1
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_bad_existing_grad_rolls_back(self):
        before = self.owner.state_dict()
        with self.assertRaises(ValueError):
            with self.owner.transaction():
                self.owner.model.weight.grad = torch.full_like(self.owner.model.weight, float("inf"))
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_baseexception_systemexit_also_rolls_back(self):
        before = self.owner.state_dict()
        with self.assertRaises(SystemExit):
            with self.owner.transaction():
                synthetic_commit(self.owner)
                raise SystemExit(1)
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_nested_transaction_refused(self):
        before = self.owner.state_dict()
        with self.assertRaises(ValueError):
            with self.owner.transaction():
                with self.owner.transaction():
                    pass
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_failed_apply_restores_original_state(self):
        before = self.owner.state_dict()
        with self.owner.transaction():
            synthetic_commit(self.owner)
        later = self.owner.state_dict()
        self.owner.load_state_dict(before)
        original_apply = self.owner._apply
        fired = [False]
        def fail_once(packet):
            original_apply(packet)
            if not fired[0]:
                fired[0] = True
                raise RuntimeError("injected restore failure")
        with patch.object(self.owner, "_apply", side_effect=fail_once):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.owner.load_state_dict(later)
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_rollback_failure_poisoned_owner(self):
        with patch.object(self.owner, "_apply", side_effect=RuntimeError("unrecoverable")):
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                with self.owner.transaction():
                    raise RuntimeError("trigger")
        with self.assertRaisesRegex(ValueError, "Poisoned"):
            self.owner.state_dict()

    def test_existing_output_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="ema205_", dir=OUT) as temp:
            path = Path(temp) / "state.pt"
            expected = self.owner.save_new(path)
            with self.assertRaises(FileExistsError):
                self.owner.save_new(path)
            self.assertEqual(c.sha256(path), expected)

    def test_bad_external_disk_hash_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory(prefix="ema205_", dir=OUT) as temp:
            path = Path(temp) / "state.pt"
            self.owner.save_new(path)
            with patch.object(torch, "load") as load:
                with self.assertRaises(ValueError):
                    self.owner.read_checked(path, "0"*64)
                load.assert_not_called()

    def test_half_counts_and_hard5000_rejected(self):
        for key, value in (("step", True), ("step", 4500.), ("step", 5001)):
            self.reject(lambda p, key=key, value=value: p["context"].__setitem__(key, value))
        self.reject(lambda p: p["context"]["sampler"].__setitem__("cursor", 4501))
        self.reject(lambda p: p["shadow"].__setitem__("updates", 1))

    def test_parent_stops_rng_or_float_type_change_rejected_even_resealed(self):
        self.reject(lambda p: p["parent"].__setitem__("parent_stop", None))
        self.reject(lambda p: p["parent"]["saved_cuda_rng"][0].fill_(0))
        self.reject(lambda p: p["context"]["schedule"].__setitem__("stopped_at", None))
        self.reject(lambda p: p["context"]["schedule"]["stale"].__setitem__(c.source.RAW_ARM, 16))
        self.reject(lambda p: p["raw"]["optimizer"]["param_groups"][0].__setitem__("weight_decay", 0.0))

    def test_root_and_nested_type_commitment_not_single_side_strip(self):
        before = self.owner.state_dict()
        no_seal = copy.deepcopy(before); no_seal.pop("content_sha256")
        with self.assertRaises(ValueError):
            self.owner.load_state_dict(no_seal)
        self.reject(lambda p: p.__setitem__("schema", True))
        self.reject(lambda p: p["raw"].__setitem__("updates", 4500.))
        self.assertFalse(c.equal({"totals": {"value": 1}}, {"totals": {"value": 1.0}}))
        self.assertFalse(c.equal(torch.tensor([0.]), torch.tensor([-0.])))

    def test_bad_seal_rejected_without_resealing(self):
        self.reject(lambda p: p["raw"]["tensors"]["weight"].add_(.25), fresh_seal=False)

    def test_missing_extra_order_parameter_dtype_rejected(self):
        self.reject(lambda p: p.pop("rng"))
        self.reject(lambda p: p.__setitem__("extra", 1))
        self.reject(lambda p: p["raw"]["parameter_names"].reverse())
        self.reject(lambda p: p["raw"]["tensors"].__setitem__("weight", p["raw"]["tensors"]["weight"].double()))

    def test_internal_alias_and_live_alias_rejected_before_apply(self):
        self.reject(lambda p: p["raw"]["gradients"].__setitem__("weight", p["raw"]["tensors"]["weight"]))
        self.reject(lambda p: p["raw"]["tensors"].__setitem__("weight", self.owner.model.weight.detach()))

    def test_wrong_modes_buffer_frozen_or_e0_shadow_rejected(self):
        self.reject(lambda p: p["raw"]["modes"].__setitem__(0, 1))
        self.reject(lambda p: p["shadow"]["tensors"]["weight"].add_(1))
        self.reject(lambda p: p["shadow"]["tensors"]["branch.scratch"].add_(1))

    def test_optimizer_full_id_step_flag_moment_checks(self):
        self.reject(lambda p: p["raw"]["optimizer"]["param_groups"][0]["params"].reverse())
        self.reject(lambda p: p["raw"]["optimizer"]["state"].pop(2))
        self.reject(lambda p: p["raw"]["optimizer"]["state"][0]["step"].fill_(4499))
        self.reject(lambda p: p["raw"]["optimizer"]["param_groups"][0].__setitem__("foreach", True))
        self.reject(lambda p: p["raw"]["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1))

    def test_all_adam_group_options_preserved_not_defaults(self):
        for key, value in (("betas", [.9, .999]), ("eps", 1e-7), ("weight_decay", True),
                           ("amsgrad", True), ("maximize", True), ("capturable", True),
                           ("differentiable", True), ("fused", True), ("decoupled_weight_decay", True)):
            with self.subTest(key=key):
                self.reject(lambda p, key=key, value=value: p["raw"]["optimizer"]["param_groups"][0].__setitem__(key, value))

    def test_optimizer_defaults_typed_and_rolled_back(self):
        self.reject(lambda p: p["raw"]["optimizer_defaults"].__setitem__("eps", 1.0))
        before = self.owner.state_dict()
        with self.assertRaises(ValueError):
            with self.owner.transaction():
                self.owner.optimizer.defaults["weight_decay"] = 0.0
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_custom_optimizer_hook_rejected_before_save(self):
        handle = self.owner.optimizer.register_state_dict_post_hook(lambda opt, state: state)
        try:
            with self.assertRaisesRegex(ValueError, "hooks"):
                self.owner.state_dict()
        finally:
            handle.remove()

    def test_existing_grad_ownership_not_silently_stripped(self):
        self.owner.model.weight.grad.requires_grad_(True)
        with self.assertRaises(ValueError):
            self.owner.state_dict()

    def test_shadow_nonzero_synthetic_copy_typed_buffers_and_no_feedback(self):
        with self.owner.transaction():
            synthetic_commit(self.owner)
        before = self.owner.state_dict()
        self.assertFalse(c.equal(before["raw"]["tensors"]["weight"], before["shadow"]["tensors"]["weight"]))
        for name in ("fixed", "branch.count", "branch.scratch"):
            self.assertTrue(c.equal(before["raw"]["tensors"][name], before["shadow"]["tensors"][name]))
        self.owner.shadow.model_state_dict(self.owner.model, raw_step=4501)
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_later_cumulative_metrics_types_and_counts_preserved(self):
        with self.owner.transaction():
            synthetic_commit(self.owner)
            self.owner.context["schedule"]["stale"][c.source.RAW_ARM] = 19
            self.owner.context["schedule"]["best"][c.source.RAW_ARM] = .125
            self.owner.context["schedule"]["patience_anchor"][c.source.RAW_ARM] = .25
        packet = self.owner.state_dict()
        self.owner.load_state_dict(packet)
        self.assertTrue(c.equal(packet, self.owner.state_dict()))
        self.reject(lambda p: p["context"]["schedule"]["stale"].__setitem__(c.source.RAW_ARM, True))

    def test_live_runtime_change_rejected_without_rng_access(self):
        with patch.object(c, "runtime_identity", return_value={"device": "cuda"}):
            with self.assertRaisesRegex(ValueError, "runtime"):
                self.owner.state_dict()

    def test_failure_during_rng_restore_rolls_back(self):
        before = self.owner.state_dict()
        changed = copy.deepcopy(before)
        changed["rng"]["torch_cpu"].zero_()
        changed = reseal(changed)
        with self.assertRaises(RuntimeError):
            self.owner.load_state_dict(changed)
        self.assertTrue(c.equal(before, self.owner.state_dict()))

    def test_wrong_rng_runtime_and_training_claim_rejected(self):
        self.reject(lambda p: p["rng"].__setitem__("torch_cuda", [torch.zeros(16, dtype=torch.uint8)]))
        self.reject(lambda p: p["rng"]["numpy"].__setitem__(2, 625))
        self.reject(lambda p: p["runtime"].__setitem__("device", "cuda"))
        self.reject(lambda p: p.__setitem__("training_authorized", True))
        self.reject(lambda p: p.__setitem__("cuda_transaction_verified", True))

    def test_optimizer_cannot_own_shadow_parameters(self):
        self.owner.optimizer.param_groups[0]["params"][0] = self.owner.shadow._values["weight"]
        with self.assertRaises(ValueError):
            self.owner.state_dict()

    def test_forbidden_apis_rejected_before_execution(self):
        with self.assertRaises(RuntimeError):
            self.owner.model(torch.ones(1))
        with self.assertRaises(RuntimeError):
            self.owner.optimizer.step()
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(None, None)
        with self.assertRaises(RuntimeError):
            torch.cuda._lazy_init()


if __name__ == "__main__":
    if not OUT.is_dir() or (OUT / "source_cpu_container_4500_complete_defaults.pt").exists():
        raise SystemExit("Fresh205 unit output with absent actual source artifact required")
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
