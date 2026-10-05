"""212 integration: tiny synthetic real Adam steps, NOT student training.

No audio, teacher, original source PT, historical suite, or CUDA execution.
One additional pinned CPU-storage migration checks the real22-parameter source
interface without forward/backward/step. All disk fixtures are new temporary
files; failure evidence and logs are preserved by the outer invocation.
"""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("single212", Path(__file__).with_name("212_ema_single_trajectory_engine.py"))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
c, live = e.c, e.live


def context():
    path = e.ROOT / "results/mel_lr_scale_import_20261004/approval.json"
    assert c.sha256(path) == c.source.PINS["results/mel_lr_scale_import_20261004/approval.json"]
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    c.source.check_seal(doc)
    sampler = {"approval_sha256": c.source.PINS["results/mel_lr_scale_import_20261004/approval.json"],
               "true_lock_sha256": "36a9be8ade7822969a7fed661c8f226f49b89e8ece90e1e6192ae0ae1b4b2c38",
               "seed": 20261002, "cursor": 4500, "teacher": "kim_melband"}
    return {"step": 4500, "sampler": sampler,
            "schedule": {"step": 4500, "last_validation": 4500, "stopped_at": 4500,
                         "best": dict.fromkeys(c.source.ARMS), "stale": dict.fromkeys(c.source.ARMS, 18),
                         "patience_anchor": dict.fromkeys(c.source.ARMS),
                         "config": doc["source_protocol"]["paired_comparison_planned"]}}


PROVENANCE = {"source_sha256": c.source.SOURCE_SHA, "source_limit": 4500,
              "source_legacy_stop_events": [3750, 4000], "legacy_stop_events": [4250, 4500]}


class TrueFixture:
    def __init__(self, sampler):
        self.bound, self.config = sampler["true_lock_sha256"], {"synthetic_only": True}
        self.fail = False

    def crop(self, domain, seed, cursor):
        if self.fail:
            random.random(); np.random.random(); torch.rand(1)
            raise KeyboardInterrupt("fixture input failure")
        index = live.DOMAINS.index(domain)
        x = torch.full((2, 89856), .125 + (cursor-4500)/128)
        row = {"domain": domain, "role": "train", "vocal_db": 0,
               "score_start": 25088, "score_end": 89344, "synthetic_fixture": True,
               "input_pcm_sha256": e.hashlib.sha256(x.numpy().tobytes()).hexdigest()}
        return {"x": x, "v": torch.zeros_like(x) if index == 2 else x*.25, "meta": row}


class TeacherFixture:
    def __init__(self):
        self.config, self.bound, self.seed = {"synthetic_only": True}, "synthetic_not_audio", 20261002
        self.rows = [{"synthetic_song": True}]

    def crop(self, recipe):
        x = torch.full((2, 89856), .25)
        row = {"domain": "pseudo", "role": "pseudo_label_train_candidate", "synthetic_fixture": True,
               "purpose": "NONRELEASE_PAIRED_EXPLORATION", "exploratory_eligible": True,
               "deployment_eligible": False, "training_eligible": False,
               "score_start": 25088, "score_end": 89344,
               "input_pcm_sha256": e.hashlib.sha256(x.numpy().tobytes()).hexdigest(), **recipe}
        return {"x": x, "v": x*.5, "meta": row}


def recipe(rows, config, seed, counter):
    return {"synthetic_counter": counter}


def stream_factory(sampler):
    return e.DeviceInputStream(sampler, TrueFixture(sampler), TeacherFixture(), recipe, device="cpu")


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., -2.]))
        self.branch = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.branch.weight.copy_(torch.tensor([[.25, -.125]])); self.branch.bias.fill_(.5)
        self.register_buffer("scratch", torch.tensor([-0., 1.]), persistent=False)
        self.register_buffer("counter", torch.tensor(4500, dtype=torch.int64))
        self.branch.training = False

    def forward(self, x):
        return self.branch(x*self.weight)


class TinyLoss:
    identity = {"kind": "synthetic_tiny_mse_not_original_student_loss", "version": 1}
    def __call__(self, model, batch, device):
        # Real autograd on ONE tiny-model call, not original audio/model graph.
        output = model(batch["x"][:, :, 0])
        loss = (output-batch["v"][:, :, 0].mean(dim=1, keepdim=True)).square().mean()
        loss.backward()
        return {"loss": float(loss.detach())}


def fixture(backward=None):
    ctx = context()
    model = Tiny()
    config = ctx["schedule"]["config"]
    import math
    lr = config["cosine_min_learning_rate"]+(config["learning_rate"]-config["cosine_min_learning_rate"])*(1+math.cos(math.pi*4400/9900))/2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False, fused=False)
    for p in model.parameters():
        optimizer.state[p] = {"step": torch.tensor(4500.), "exp_avg": torch.full_like(p, .125), "exp_avg_sq": torch.full_like(p, .25)}
        p.grad = torch.full_like(p, .0625)
    parent = {"synthetic_fixture": True, "source_sha256": c.source.SOURCE_SHA,
              "saved_cuda_rng": [torch.arange(16, dtype=torch.uint8)], "old_stop": 4500}
    return e.SingleTrajectoryEngine(model, optimizer, parent, ctx, stream_factory(ctx["sampler"]), backward or TinyLoss(),
                                    stop_provenance=PROVENANCE)


def reseal(packet):
    packet.pop("content_sha256", None)
    return c.seal(packet)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch._dynamo  # Register optimizer lazy rules, no compilation.
        cls.outer_rng = e.capture_rng("cpu")
        cls.old_threads = torch.get_num_threads()
        cls.old_runtime = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        cls.guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden in212 CPU units"))
        cls.guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop()
        e.restore_rng(cls.outer_rng, "cpu")
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_runtime[0], warn_only=cls.old_runtime[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cls.old_runtime[2:4]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = cls.old_runtime[4:6]
        print("EVIDENCE: synthetic tiny-model Adam/autograd only; student updates=0; audio=0; original source PT loads=0; CUDA initialized=false", flush=True)

    def test_e0_exact_independent(self):
        owner = fixture(); p = owner.state_dict()
        self.assertTrue(c.equal(p["raw"]["tensors"], p["shadow"]["tensors"]))
        c.noalias(p, owner._external())
        self.assertFalse(p["training_authorized"])

    def test_first_actual_adam_matches_independent_raw_reference(self):
        owner = fixture(); before = owner.state_dict()
        raw = Tiny(); raw.load_state_dict(owner.model.state_dict())
        optimizer = torch.optim.Adam(raw.parameters(), lr=owner._group["lr"], foreach=False, fused=False)
        optimizer.load_state_dict(c.portable(before["raw"]["optimizer"]))
        sampler = stream_factory(context()["sampler"])
        batch = sampler.next_batch()
        raw.train(); optimizer.param_groups[0]["lr"] = owner.live.learning_rate_next()
        optimizer.zero_grad(set_to_none=True); TinyLoss()(raw, batch, "cpu")
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 3, error_if_nonfinite=True)
        # Independent direct original functional kernel; not212's CPU-health
        # adapter and no default accelerator query. Same original parameters,
        # gradients, moments, flags and operation order.
        from torch.optim import _functional
        params = list(raw.parameters()); group = optimizer.param_groups[0]
        with torch.no_grad():
            _functional.adam(params, [p.grad for p in params],
                             [optimizer.state[p]["exp_avg"] for p in params],
                             [optimizer.state[p]["exp_avg_sq"] for p in params], [],
                             [optimizer.state[p]["step"] for p in params],
                             foreach=False, capturable=False, differentiable=False, fused=False,
                             amsgrad=False, beta1=group["betas"][0], beta2=group["betas"][1],
                             lr=group["lr"], weight_decay=group["weight_decay"], eps=group["eps"],
                             maximize=False, has_complex=False, decoupled_weight_decay=False)
        optimizer.zero_grad(set_to_none=True)
        result = owner.update_next()
        self.assertEqual(result["step"], 4501)
        self.assertTrue(c.equal(c.portable(optimizer.state_dict()), owner.state_dict()["raw"]["optimizer"]))
        for name, parameter in raw.named_parameters():
            self.assertTrue(c.equal(c.portable(parameter), owner.state_dict()["raw"]["tensors"][name]))
        self.assertEqual(owner.exposure["adam_completed"], 1)

    def test_ema_separate_fp32_nonzero(self):
        owner = fixture(); before = owner.state_dict(); owner.update_next(); after = owner.state_dict()
        for name in owner._names:
            expected = before["shadow"]["tensors"][name]*.99 + after["raw"]["tensors"][name]*.01
            self.assertTrue(c.equal(expected, after["shadow"]["tensors"][name]))
        self.assertFalse(c.equal(before["raw"]["tensors"], after["raw"]["tensors"]))
        self.assertFalse(c.equal(after["raw"]["tensors"], after["shadow"]["tensors"]))
        c.noalias(after, owner._external())

    def test_buffers_modes_gradients_copied(self):
        owner = fixture(); owner.update_next(); p = owner.state_dict()
        self.assertTrue(all(p["raw"]["modes"]))
        self.assertTrue(all(v is None for v in p["raw"]["gradients"].values()))
        for name in ("scratch", "counter"):
            self.assertTrue(c.equal(p["raw"]["tensors"][name], p["shadow"]["tensors"][name]))

    def test_three_updates_and_cap_before_draw(self):
        owner = fixture()
        for step in (4501, 4502, 4503):
            self.assertEqual(owner.update_next()["step"], step)
        before = owner.state_dict()
        with self.assertRaises(ValueError): owner.update_next()
        self.assertTrue(c.equal(before, owner.state_dict()))
        self.assertEqual(owner.exposure["adam_completed"], 3)

    def test_disk_restore_next_update_bits(self):
        owner = fixture(); owner.update_next()
        with tempfile.TemporaryDirectory(prefix="ema212_") as folder:
            path = Path(folder)/"trajectory.pt"
            sha = owner.save_new(path)
            saved = owner.read_checked(path, sha)
            owner.update_next(); next_state = owner.state_dict()
            owner.load_state_dict(saved); owner.update_next()
            self.assertTrue(c.equal(next_state, owner.state_dict()))

    def test_overwrite_refused(self):
        owner = fixture()
        with tempfile.TemporaryDirectory(prefix="ema212_") as folder:
            path = Path(folder)/"trajectory.pt"; sha = owner.save_new(path)
            with self.assertRaises(FileExistsError): owner.save_new(path)
            self.assertEqual(c.sha256(path), sha)

    def test_sha_before_deserialize(self):
        owner = fixture()
        with patch.object(torch, "load", side_effect=AssertionError("Must reject before torch.load")):
            with self.assertRaises(ValueError): owner.read_checked(Path(__file__), "0"*64)

    def test_failure_after_real_adam_rolls_back_everything(self):
        owner = fixture(); before = owner.state_dict()
        def failed():
            random.random(); np.random.random(); torch.rand(3)
            raise KeyboardInterrupt("failure after actual Adam, before shadow/context commit")
        with patch.object(owner, "complete_raw_storage_step", side_effect=failed):
            with self.assertRaises(KeyboardInterrupt): owner.update_next()
        self.assertTrue(c.equal(before, owner.state_dict()))
        self.assertEqual(owner.exposure["adam_completed"], 1)
        self.assertEqual(owner.exposure["updates_committed"], 0)
        self.assertEqual(owner.exposure["updates_failed"], 1)

    def test_failure_after_shadow_commit_rolls_back(self):
        owner = fixture(); before = owner.state_dict(); complete = owner.complete_raw_storage_step
        def failed():
            complete(); raise SystemExit("failure after complete shadow/live exposure")
        with patch.object(owner, "complete_raw_storage_step", side_effect=failed):
            with self.assertRaises(SystemExit): owner.update_next()
        self.assertTrue(c.equal(before, owner.state_dict()))

    def test_backward_failure_restores_existing_gradients_modes_rng(self):
        class Fail(TinyLoss):
            def __call__(self, model, batch, device):
                super().__call__(model, batch, device)
                random.random(); np.random.random(); torch.rand(3)
                raise KeyboardInterrupt("backward failure")
        owner = fixture(Fail()); before = owner.state_dict()
        with self.assertRaises(KeyboardInterrupt): owner.update_next()
        self.assertTrue(c.equal(before, owner.state_dict()))
        self.assertEqual(owner.exposure["adam_started"], 0)

    def test_nonfinite_gradient_never_adam(self):
        class Bad(TinyLoss):
            def __call__(self, model, batch, device):
                result = super().__call__(model, batch, device)
                model.weight.grad.fill_(float("nan")); return result
        owner = fixture(Bad()); before = owner.state_dict()
        with self.assertRaises(ValueError): owner.update_next()
        self.assertTrue(c.equal(before, owner.state_dict()))
        self.assertEqual(owner.exposure["adam_started"], 0)

    def test_missing_gradient_never_adam(self):
        class Bad(TinyLoss):
            def __call__(self, model, batch, device):
                result = super().__call__(model, batch, device)
                model.weight.grad = None; return result
        owner = fixture(Bad()); before = owner.state_dict()
        with self.assertRaises(ValueError): owner.update_next()
        self.assertTrue(c.equal(before, owner.state_dict()))

    def test_input_mutation_rejected(self):
        class Bad(TinyLoss):
            def __call__(self, model, batch, device):
                result = super().__call__(model, batch, device)
                batch["x"].add_(1); return result
        owner = fixture(Bad()); before = owner.state_dict()
        with self.assertRaises(ValueError): owner.update_next()
        self.assertTrue(c.equal(before, owner.state_dict()))

    def test_input_failure_poison_not_silent_retry(self):
        owner = fixture(); before = owner.state_dict(); owner.stream.true.fail = True
        with self.assertRaises(KeyboardInterrupt): owner.update_next()
        self.assertTrue(owner._poisoned and owner.stream.poisoned)
        self.assertEqual(owner.stream.cursor, 4500)
        self.assertTrue(c.equal(owner._raw_packet(4500), before["raw"]))
        self.assertTrue(c.equal(e.capture_rng("cpu"), before["rng"]))
        with self.assertRaises(ValueError): owner.update_next()

    def test_rollback_failure_poison(self):
        owner = fixture()
        with patch.object(owner, "complete_raw_storage_step", side_effect=KeyboardInterrupt), patch.object(owner, "_apply", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError): owner.update_next()
        self.assertTrue(owner._poisoned)
        with self.assertRaises(ValueError): owner.state_dict()

    def test_partial_restore_rolls_back(self):
        owner = fixture(); original = owner.state_dict(); owner.update_next(); previous = owner.state_dict()
        apply = owner._apply; calls = []
        def failed(packet):
            calls.append(1)
            apply(packet)
            if len(calls) == 1: raise KeyboardInterrupt("partially completed load")
        with patch.object(owner, "_apply", side_effect=failed):
            with self.assertRaises(KeyboardInterrupt): owner.load_state_dict(original)
        self.assertTrue(c.equal(previous, owner.state_dict()))

    def test_single_adam_shadow_not_optimizer_owned(self):
        owner = fixture()
        c.ema._unaliased(owner.shadow._values.values(), owner.model.parameters())
        self.assertEqual(len(owner.optimizer.param_groups), 1)

    def test_original_parent_cuda_rng_and_stop_unchanged(self):
        owner = fixture(); before = owner.state_dict()["parent"]; owner.update_next()
        self.assertTrue(c.equal(before, owner.state_dict()["parent"]))

    def test_nested_transaction_refused(self):
        owner = fixture(); previous = owner.state_dict()
        with self.assertRaises(ValueError):
            with owner.transaction():
                with owner.transaction(): pass
        self.assertTrue(c.equal(previous, owner.state_dict()))

    def test_type_tampering_own_seal_cannot_hide(self):
        owner = fixture(); p = owner.state_dict(); p["training_authorized"] = 0; p = reseal(p)
        with self.assertRaises(ValueError): owner.load_state_dict(p)

    def test_shadow_count_tampering(self):
        owner = fixture(); owner.update_next(); p = owner.state_dict(); p["shadow"]["updates"] = 2; p = reseal(p)
        with self.assertRaises(ValueError): owner.load_state_dict(p)

    def test_partial_adam_step_rejected(self):
        owner = fixture(); p = owner.state_dict(); p["raw"]["optimizer"]["state"][0]["step"].fill_(4499); p = reseal(p)
        with self.assertRaises(ValueError): owner.load_state_dict(p)

    def test_original_lr_not_mutable(self):
        owner = fixture(); p = owner.state_dict(); p["raw"]["optimizer"]["param_groups"][0]["lr"] *= .5; p = reseal(p)
        with self.assertRaises(ValueError): owner.load_state_dict(p)

    def test_source_mode_cannot_enable_fixture_updates(self):
        owner = fixture()
        with self.assertRaises(ValueError):
            e.SingleTrajectoryEngine(owner.model, owner.optimizer, owner._parent, context(), owner.stream, TinyLoss(),
                                      scope="formal_training", stop_provenance=PROVENANCE)

    def test_cuda_refused_before_runtime_init(self):
        owner = fixture()
        with self.assertRaises(ValueError):
            e.SingleTrajectoryEngine(owner.model, owner.optimizer, owner._parent, context(), owner.stream, TinyLoss(),
                                      device="cuda", stop_provenance=PROVENANCE)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_health_adapter_removed_after_actual_step(self):
        owner = fixture(); original = torch.optim.Adam.step
        owner.update_next()
        self.assertIs(torch.optim.Adam.step, original)
        self.assertNotIn("_accelerator_graph_capture_health_check", owner.optimizer.__dict__)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_health_adapter_refuses_capture(self):
        owner = fixture(); owner.optimizer.param_groups[0]["capturable"] = True
        with self.assertRaises(ValueError): e.cpu_adam_step(owner.optimizer)
        self.assertNotIn("_accelerator_graph_capture_health_check", owner.optimizer.__dict__)

    def test_cpu_health_adapter_refuses_preexisting_override(self):
        owner = fixture(); owner.optimizer._accelerator_graph_capture_health_check = lambda: None
        with self.assertRaises(ValueError): e.cpu_adam_step(owner.optimizer)

    def test_rng_types_and_cpu_cuda_migration(self):
        p = e.capture_rng("cpu"); p["torch_cuda"] = [torch.zeros(16, dtype=torch.uint8)]
        with self.assertRaises(ValueError): e.restore_rng(p, "cpu")
        e.validate_rng(p, "cuda")  # Pure shape/type test, NOT live CUDA evidence.
        p["torch_cuda"][0] = torch.zeros(15, dtype=torch.uint8)
        with self.assertRaises(ValueError): e.validate_rng(p, "cuda")

    def test_loss_identity_change_refused(self):
        owner = fixture(); owner.backward.identity = {"kind": "wrong"}
        with self.assertRaises(ValueError): owner.update_next()

    def test_backend_identity_change_refused(self):
        owner = fixture(); owner.stream.dataset.rows.append({"changed": 1})
        with self.assertRaises(ValueError): owner.update_next()

    def test_save_load_does_not_alias_input(self):
        owner = fixture(); original = owner.state_dict(); digest = c.digest(original)
        owner.update_next(); owner.load_state_dict(original); owner.update_next()
        self.assertEqual(c.digest(original), digest)

    def test_sealed_nested_context_not_stripped(self):
        owner = fixture(); p = owner.state_dict(); p["live_context"].pop("content_sha256"); p = reseal(p)
        with self.assertRaises(ValueError): owner.load_state_dict(p)

    def test_adam_options_types_preserved(self):
        owner = fixture(); p = owner.state_dict(); p["raw"]["optimizer"]["param_groups"][0]["weight_decay"] = 0.; p = reseal(p)
        with self.assertRaises(ValueError): owner.load_state_dict(p)

    def test_fixed_cpu_source_new_engine_restore_only(self):
        class BlockedLoss:
            identity = {"kind": "restoration_only_no_forward_or_step"}
            def __call__(self, *args): raise AssertionError("No actual source update authorized")
        target = c._module("ema212_actual09", "scripts/09_target_model.py")
        with c.preserve_cpu_rng():
            owner = e.SingleTrajectoryEngine.from_fixed_cpu_storage(
                lambda: target.CausalSpectralUNet(bottleneck_blocks=2), stream_factory, BlockedLoss())
            packet = owner.state_dict()
            self.assertEqual(len(packet["raw"]["parameter_names"]), 22)
            self.assertEqual(packet["live_context"]["context"]["step"], 4500)
            self.assertTrue(c.equal(packet["rng"]["torch_cuda"], []))
            self.assertEqual(len(packet["parent"]["parent_metadata"]["rng"]["torch_cuda"]), 1)
            with self.assertRaises(ValueError): owner.update_next()
            print("SOURCE_RESTORE_EVIDENCE: pinned205 CPU storage reads=1; actual raw Module=1; matching CPU Adam construction=1; original training PT reads=0; actual source forward/autograd/Adam.step/audio/CUDA=0", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
