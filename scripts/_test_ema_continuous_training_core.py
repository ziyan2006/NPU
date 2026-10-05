"""New219 CPU fixture integration; no student, audio, source PT or CUDA.

Two-step tiny-model Adam cases exercise the continuous transaction. The
fixed500 stage case is MANUAL tiny tensor/moment exposure, not500 Adam steps.
No historical test module, preparation, mechanism or reference is executed.
"""
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("continuous219", Path(__file__).with_name("219_ema_continuous_training_core.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
e, c, live, torch = t.e, t.c, t.live, t.torch


def context():
    path = t.ROOT / "results/mel_lr_scale_import_20261004/approval.json"
    assert c.sha256(path) == c.source.PINS["results/mel_lr_scale_import_20261004/approval.json"]
    doc = json.loads(path.read_text(encoding="utf-8-sig")); c.source.check_seal(doc)
    return {"step": 4500,
            "sampler": {"approval_sha256": c.source.PINS["results/mel_lr_scale_import_20261004/approval.json"],
                        "true_lock_sha256": "36a9be8ade7822969a7fed661c8f226f49b89e8ece90e1e6192ae0ae1b4b2c38",
                        "seed": 20261002, "cursor": 4500, "teacher": "kim_melband"},
            "schedule": {"step": 4500, "last_validation": 4500, "stopped_at": 4500,
                         "best": dict.fromkeys(c.source.ARMS), "stale": dict.fromkeys(c.source.ARMS, 18),
                         "patience_anchor": dict.fromkeys(c.source.ARMS),
                         "config": doc["source_protocol"]["paired_comparison_planned"]}}


PROVENANCE = {"source_sha256": c.source.SOURCE_SHA, "source_limit": 4500,
              "source_legacy_stop_events": [3750, 4000], "legacy_stop_events": [4250, 4500]}


class TrueFixture:
    def __init__(self, sampler):
        self.bound, self.config = sampler["true_lock_sha256"], {"synthetic": True}
        self.fail = False

    def crop(self, domain, seed, cursor):
        if self.fail:
            random.random(); e.np.random.random(); torch.rand(1)
            raise KeyboardInterrupt("NEW219 synthetic input fault")
        x = torch.full((2, 89856), .125)
        metadata = {"domain": domain, "role": "train", "vocal_db": 0, "score_start": 25088, "score_end": 89344,
                    "input_pcm_sha256": t.hashlib.sha256(x.numpy().tobytes()).hexdigest(), "synthetic": True}
        return {"x": x, "v": torch.zeros_like(x) if domain == "instrumental" else x*.25, "meta": metadata}


class TeacherFixture:
    def __init__(self):
        self.bound, self.seed, self.config = "synthetic", 20261002, {"synthetic": True}
        self.rows = [{"synthetic": True}]

    def crop(self, recipe):
        x = torch.full((2, 89856), .25)
        metadata = {"domain": "pseudo", "role": "pseudo_label_train_candidate", "purpose": "NONRELEASE_PAIRED_EXPLORATION",
                    "exploratory_eligible": True, "deployment_eligible": False, "training_eligible": False,
                    "score_start": 25088, "score_end": 89344,
                    "input_pcm_sha256": t.hashlib.sha256(x.numpy().tobytes()).hexdigest(), "synthetic": True}
        return {"x": x, "v": x*.5, "meta": metadata}


def recipe(rows, config, seed, cursor):
    return {"cursor": cursor}


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.register_buffer("signed_zero", torch.tensor([-0.]), persistent=False)
        self.layer.training = False

    def forward(self, x):
        return self.layer(x)


class TinyLoss:
    identity = {"kind": "NEW219_synthetic_not_student_loss"}
    def __call__(self, model, batch, device):
        loss = (model(batch["x"][:, :, 0])-batch["v"][:, :, 0].mean(1, keepdim=True)).square().mean()
        loss.backward()
        return {"synthetic_loss": float(loss.detach())}


def fixture():
    ctx = context(); model = Tiny()
    config = ctx["schedule"]["config"]
    import math
    lr = config["cosine_min_learning_rate"]+(config["learning_rate"]-config["cosine_min_learning_rate"])*(1+math.cos(math.pi*4400/9900))/2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False, fused=False)
    for param in model.parameters():
        optimizer.state[param] = {"step": torch.tensor(4500.), "exp_avg": torch.full_like(param, .125), "exp_avg_sq": torch.full_like(param, .25)}
        param.grad = torch.full_like(param, .0625)
    stream = e.DeviceInputStream(ctx["sampler"], TrueFixture(ctx["sampler"]), TeacherFixture(), recipe, device="cpu")
    parent = {"synthetic_fixture": True, "source_sha256": c.source.SOURCE_SHA,
              "old_stop": 4500, "preserved_parent_CUDA_bytes": [torch.arange(16, dtype=torch.uint8)]}
    return t.ContinuousTrajectory(model, optimizer, parent, ctx, stream, TinyLoss(), provenance=PROVENANCE)


def score(owner, raw_rank=.1, ema_rank=.9):
    packet = owner.state_dict()["trajectory"]
    return c.seal({"purpose": "NONRELEASE_EMA_RAW_SHADOW_OLD_DEVELOPMENT", "step": owner.live.step,
                   "scope": "synthetic_only_NOT_real_DEV", "scores": {"raw": {"eligible": True, "rank_gain_db": raw_rank},
                                                                         "ema": {"eligible": True, "rank_gain_db": ema_rank}},
                   "model_digest": {"raw": c.digest(packet["raw"]["tensors"]), "ema": c.digest(packet["shadow"]["tensors"])},
                   "release_selection": "NONE"})


def reseal(packet):
    packet.pop("content_sha256", None)
    return c.seal(packet)


def manual_storage_to(owner, target):
    # NO forward/autograd/Adam here: manual tiny tensor/moment/count fixture.
    if owner.stream.last_metadata is None:
        batch = owner.stream.next_batch()
        metadata = c.portable(batch["metadata"])
        owner.stream.cursor = 4500; owner.stream.last_metadata = None
    else:
        metadata = c.portable(owner.stream.last_metadata)
    while owner.live.step < target:
        with owner.transaction(), torch.no_grad():
            lr = owner.live.learning_rate_next()
            owner.stream.cursor += 1; owner.stream.last_metadata = c.portable(metadata)
            owner.optimizer.param_groups[0]["lr"] = lr
            for parameter in owner.model.parameters():
                parameter.add_(.0001)
                owner.optimizer.state[parameter]["step"].add_(1)
            owner.complete_raw_storage_step()


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch._dynamo
        cls.outer = e.capture_rng("cpu")
        cls.settings = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                        torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark,
                        torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        cls.guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden in NEW219 CPU fixtures"))
        cls.guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop(); e.restore_rng(cls.outer, "cpu")
        torch.set_num_threads(cls.settings[0]); torch.use_deterministic_algorithms(cls.settings[1], warn_only=cls.settings[2])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cls.settings[3:5]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = cls.settings[5:7]
        print("NEW219: synthetic tiny Adam and manual500 storage only; student/sourcePT/audio/DEV/CUDA=0", flush=True)

    def test_01_new_envelope_e0_noalias_no_authority(self):
        owner = fixture(); packet = owner.state_dict()
        self.assertEqual(packet["purpose"], t.PURPOSE)
        self.assertFalse(packet["training_authorized"])
        self.assertTrue(c.equal(packet["trajectory"]["raw"]["tensors"], packet["trajectory"]["shadow"]["tensors"]))
        c.noalias(packet, owner._external())

    def test_02_two_contiguous_real_tiny_Adam_updates(self):
        owner = fixture(); first = owner.update_next(); second = owner.update_next()
        self.assertEqual([first["step"], second["step"]], [4501, 4502])
        self.assertEqual(owner.exposure["adam_completed"], 2)
        self.assertEqual(owner.shadow.updates, 2)
        self.assertFalse(second["formal_student_update"])
        self.assertEqual(owner.stream.cursor, 4502)

    def test_03_whole_disk_restore_and_next_update_bits(self):
        owner = fixture(); owner.update_next()
        with tempfile.TemporaryDirectory(prefix="ema219-unit-") as directory:
            path = Path(directory) / "state.pt"; digest = owner.save_new(path)
            saved = owner.read_checked(path, digest)
            owner.update_next(); after = owner.state_dict()
            owner.load_state_dict(saved); owner.update_next()
            self.assertTrue(c.equal(owner.state_dict(), after))

    def test_04_post_Adam_EMA_and_all_CPU_RNG_fault_rolls_back(self):
        owner = fixture(); before = owner.state_dict()
        def fault(current):
            self.assertEqual(current.shadow.updates, 1)
            random.random(); e.np.random.random(); torch.rand(1)
            raise KeyboardInterrupt("NEW219 after joint mutation")
        with self.assertRaises(KeyboardInterrupt):
            owner.update_next(after_joint_mutation=fault)
        self.assertTrue(c.equal(owner.state_dict(), before))
        self.assertEqual(owner.exposure["adam_completed"], 1)
        self.assertEqual(owner.exposure["updates_committed"], 0)

    def test_05_input_failure_retains_poison_no_retry(self):
        owner = fixture(); owner.stream.true.fail = True
        with self.assertRaises(KeyboardInterrupt):
            owner.update_next()
        self.assertTrue(owner._poisoned and owner.stream.poisoned)
        self.assertEqual(owner.live.step, 4500)
        self.assertEqual(owner.exposure["adam_started"], 0)

    def test_06_complete500_storage_DEV_stages_old_stop_and_no_EMA_feedback(self):
        owner = fixture(); manual_storage_to(owner, 4750)
        with self.assertRaises(ValueError):
            owner.update_next()
        owner.observe_stage(score(owner, .1, .9))
        with tempfile.TemporaryDirectory(prefix="ema219-stage-") as directory:
            path = Path(directory) / "4750.pt"; digest = owner.save_new(path)
            saved = owner.read_checked(path, digest)
            manual_storage_to(owner, 5000)
            owner.observe_stage(score(owner, .11, 9.))
            final = owner.state_dict()
            self.assertEqual(owner.live.context["schedule"]["patience_anchor"][live.RAW_ARM], .1)
            self.assertEqual(owner.live.context["schedule"]["stale"][live.RAW_ARM], 1)
            self.assertEqual(owner.live.context["schedule"]["stale"][live.OTHER_ARM], 18)
            self.assertEqual(owner.live.context["schedule"]["stopped_at"], 4500)
            self.assertEqual([r["step"] for r in final["development_receipts"]], [4750, 5000])
            self.assertEqual(owner.shadow.updates, 500)
            self.assertEqual(owner.exposure["adam_completed"], 0)
            with self.assertRaises(ValueError):
                owner.update_next()
            owner.load_state_dict(saved)
            self.assertEqual(owner.live.step, 4750)
            self.assertEqual(len(owner.development_receipts), 1)

    def test_07_incomplete_pair_rejected_before_schedule_mutation(self):
        owner = fixture(); manual_storage_to(owner, 4750); before = owner.state_dict()
        packet = score(owner); del packet["scores"]["ema"]; reseal(packet)
        with self.assertRaises(ValueError):
            owner.observe_stage(packet)
        self.assertTrue(c.equal(owner.state_dict(), before))

    def test_08_wrong_evaluated_snapshot_rejected(self):
        owner = fixture(); manual_storage_to(owner, 4750)
        packet = score(owner); packet["model_digest"]["ema"] = "0"*64; reseal(packet)
        with self.assertRaises(ValueError):
            owner.observe_stage(packet)

    def test_09_receipt_and_journal_tampering_rejected(self):
        owner = fixture(); manual_storage_to(owner, 4750); owner.observe_stage(score(owner))
        packet = owner.state_dict(); packet["development_receipts"][0]["scores"]["raw"]["rank_gain_db"] = .7
        reseal(packet["development_receipts"][0]); reseal(packet)
        with self.assertRaises(ValueError):
            owner.load_state_dict(packet)

    def test_10_component_is_not_full219_checkpoint(self):
        owner = fixture()
        with self.assertRaises(ValueError):
            owner.load_state_dict(owner.state_dict()["trajectory"])

    def test_11_authority_boolean_cannot_promote_fixture(self):
        owner = fixture(); packet = owner.state_dict(); packet["training_authorized"] = True; reseal(packet)
        with self.assertRaises(ValueError):
            owner.load_state_dict(packet)

    def test_12_existing_disk_artifact_and_wrong_SHA_refused(self):
        owner = fixture()
        with tempfile.TemporaryDirectory(prefix="ema219-exclusivity-") as directory:
            path = Path(directory) / "state.pt"; owner.save_new(path)
            with self.assertRaises(FileExistsError):
                owner.save_new(path)
            with self.assertRaises(ValueError):
                owner.read_checked(path, "0"*64)

    def test_13_missing_activation_never_initializes_CUDA(self):
        with self.assertRaises(ValueError):
            t.checked_activation(t.ROOT / "results/ema219_NONEXISTENT_activation.json")
        self.assertFalse(torch.cuda.is_initialized())


class SelectionLeaseTests(unittest.TestCase):
    def pools(self):
        pools, files = {}, {}
        for domain in live.DOMAINS[:3]:
            rows = []
            for index in range(4):
                name = domain + str(index)
                rows.append({"domain": domain, "role": "train", "track_id": name, "mix_files": [name]})
                files[name] = {"sha256": t.hashlib.sha256(name.encode()).hexdigest()}
            pools[domain] = rows
        rows = [{"song_id": "p"+str(i), "role": "pseudo_label_train_candidate", "training_eligible": False,
                 "source": {"path": "source"+str(i), "sha256": str(i)*64},
                 "label_files": {"v": {"path": "v"+str(i), "sha256": str(i)*64}, "a": {"path": "a"+str(i), "sha256": str(i)*64}}}
                for i in range(3)]
        dataset = SimpleNamespace(rows=rows, config={}, by_id={r["song_id"]: r for r in rows})
        return SimpleNamespace(pools=pools, lock={"files": files}), dataset

    def test_14_future_counters_use_original_local_selection_and_all_targets(self):
        true, dataset = self.pools()
        select = lambda rows, config, seed, cursor: {"song_id": rows[cursor % 3]["song_id"]}
        outer = random.getstate()
        selected, bindings = t.selected_draw_files(true, dataset, select, {"seed": 20261002}, 4999)
        self.assertEqual(random.getstate(), outer)
        self.assertEqual(len(selected), 6); self.assertEqual(len(bindings), 12)

    def test_15_hard5000_selection_rejects(self):
        true, dataset = self.pools()
        for cursor in (4499, 5000, True, 4500.):
            with self.assertRaises(ValueError):
                t.selected_draw_files(true, dataset, lambda *args: {}, {"seed": 1}, cursor)

    def test_16_pseudo_promoted_to_truth_rejected(self):
        true, dataset = self.pools(); dataset.rows[0]["training_eligible"] = True
        with self.assertRaises(ValueError):
            t.selected_draw_files(true, dataset, lambda *args: {"song_id": "p0"}, {"seed": 1}, 4500)

    def native(self, infos):
        return SimpleNamespace(dll=SimpleNamespace(CreateFileW=Mock(side_effect=list(range(1, len(infos)+1)))),
                               measure=Mock(side_effect=infos), checked=Mock())

    def test_17_lease_pre_post_physical_bytes_and_deny_write_delete(self):
        info = {"sha256": "a"*64, "file_index": 2**60, "bytes": 6}
        native = self.native([info, dict(info)])
        with t.SourceReadLease({"x": "a"*64}, native):
            pass
        self.assertEqual(native.dll.CreateFileW.call_args.args[1:3], (0x80000000, 1))
        native.checked.assert_called_once_with("CloseHandle", 1)

    def test_18_pre_hash_failure_closes_handle_and_never_draws(self):
        native = self.native([{"sha256": "b"*64}])
        with self.assertRaises(ValueError):
            with t.SourceReadLease({"x": "a"*64}, native):
                self.fail("draw was allowed")
        native.checked.assert_called_once_with("CloseHandle", 1)

    def test_19_post_identity_failure_closes_handle_and_refuses_acceptance(self):
        native = self.native([{"sha256": "a"*64, "file_index": 2**60}, {"sha256": "a"*64, "file_index": 2**60+1}])
        with self.assertRaises(ValueError):
            with t.SourceReadLease({"x": "a"*64}, native):
                pass
        native.checked.assert_called_once_with("CloseHandle", 1)

    def test_20_BaseException_lease_closes_without_losing_original_failure(self):
        native = self.native([{"sha256": "a"*64}])
        with self.assertRaises(KeyboardInterrupt):
            with t.SourceReadLease({"x": "a"*64}, native):
                raise KeyboardInterrupt()
        native.checked.assert_called_once_with("CloseHandle", 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
