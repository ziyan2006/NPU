"""New208 real INPUT-only tests, one unique batch4500 and one replay.

No old suite, student checkpoint deserialization, model or optimizer execution.
API blocks are installed BEFORE construction/draws. Actual failures retain all
output and native nonzero exit; no inferred CUDA/formal-training gate.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch._dynamo  # Load lazy registry before independent API mocks.

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema208_input_tests", ROOT / "scripts/208_ema_authenticated_audio_input.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
MONITOR = ROOT / "results/mel_ema_audio_backend_monitor_20261004"
REFERENCE = ROOT / "results/train_adam_memory_20261004/inputs.json"
REFERENCE_SHA = "19caf93b84357d46e37e5614cc517fe2a3d8c75bfc27c24e2f22d63e5992cc0f"
proof = {}


def forbidden(name):
    def blocked(*args, **kwargs):
        raise RuntimeError("FORBIDDEN208_BEFORE_EXECUTION:" + name)
    return blocked


class InputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before_rng = api.storage.capture_cpu_rng()
        api.require(api.file_sha(REFERENCE) == REFERENCE_SHA, "Original input reference changed")
        # Original acq seal, not typed205 seal; read data only, do not call197.
        cls.reference = api.acq.read_sealed(REFERENCE)["inputs"][0]
        cls.stream = api.AuthenticatedAudioStream(copy.deepcopy(api.ORIGINAL_SAMPLER))
        cls.initial = cls.stream.state_dict()
        cls.contract = cls.stream.contract()
        cls.first = cls.stream.next_batch()
        cls.first_id = api.packet_identity(cls.first)
        cls.stream.load_state_dict(cls.initial, last_metadata=None)
        cls.replay = cls.stream.next_batch()
        cls.replay_id = api.packet_identity(cls.replay)
        cls.after_rng = api.storage.capture_cpu_rng()
        cls.stream.verify_all_bindings()
        proof.update({"purpose": api.PURPOSE, "contract": cls.contract,
            "original_input_reference_sha256": REFERENCE_SHA,
            "actual_input_unique_counters": [4500], "actual_successful_draws": 2,
            "unique_slots": 6, "repeated_slots": 6,
            "actual_first_identity": cls.first_id, "actual_replay_identity": cls.replay_id,
            "source_pt_deserialization": 0, "student_modules": 0, "forward": 0,
            "autograd_engine": 0, "adam_construction": 0, "adam_steps": 0,
            "training_updates": 0, "cuda_initialized": torch.cuda.is_initialized(),
            "output_audio": 0, "cpu_python_numpy_rng_unchanged": api.equal(cls.before_rng, cls.after_rng),
            "selected_true_original_files_sha256": {p: cls.stream.true.lock["files"][p]["sha256"]
                                                      for p in cls.stream.true.signatures},
            "backend_cache_scope": "Byte-guarded LRUs; failed draw poisons backend, not full training transaction",
            "full_zero_update_model_audit": "PENDING", "cpu_cuda_update_mechanism": "PENDING",
            "formal_training_authorization": "PENDING", "release_selection": "NONE"})

    @classmethod
    def tearDownClass(cls):
        api.require(api.equal(cls.before_rng, api.storage.capture_cpu_rng()), "Unit scope consumed global RNG")
        api.require(api.file_sha(REFERENCE) == REFERENCE_SHA, "Read-only reference changed")
        # Writes NEW evidence only, never changes historical input files.
        with (MONITOR / "actual_input_replay_evidence.json").open("x", encoding="utf-8") as handle:
            json.dump(api.storage.seal(proof), handle, ensure_ascii=False, indent=2, allow_nan=False)

    def test_01_reference_counter(self):
        self.assertEqual(self.first_id["counter"], self.reference["counter"])

    def test_02_reference_input_all_slots(self):
        self.assertEqual(self.first_id["input_sha256"], self.reference["input_sha256"])

    def test_03_reference_targets_all_slots(self):
        self.assertEqual(self.first_id["target_sha256"], self.reference["target_sha256"])

    def test_04_reference_metadata_full_typed(self):
        self.assertTrue(api.equal(self.first_id["metadata"], self.reference["metadata"]))

    def test_05_replay_full_typed(self):
        self.assertTrue(api.equal(self.first_id, self.replay_id))

    def test_06_replay_actual_pcm_bits(self):
        self.assertTrue(api.equal(self.first["x"], self.replay["x"]))
        self.assertTrue(api.equal(self.first["v"], self.replay["v"]))

    def test_07_replay_noalias(self):
        api.storage.noalias(self.first, api.storage.source.tensor_leaves(self.replay))

    def test_08_one_target(self):
        self.assertEqual(list(self.first), ["x", "v", "domains", "cursor", "metadata"])
        self.assertFalse(hasattr(self.stream, "datasets"))

    def test_09_source_rng(self):
        self.assertTrue(api.equal(self.before_rng, self.after_rng))

    def test_10_cpu_finite_fp32(self):
        for value in (self.first["x"], self.first["v"]):
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_11_original_roles_gain_support(self):
        api.live.validate_metadata(self.first["metadata"])
        self.assertEqual([r["vocal_db"] for r in self.first["metadata"][:3]], [6, 6, 0])

    def test_12_instrument_exact_zero(self):
        self.assertEqual(int(torch.count_nonzero(self.first["v"][2])), 0)

    def test_13_original_pseudo_not_truth(self):
        for row in self.first["metadata"][3:]:
            self.assertIs(row["training_eligible"], False)
            self.assertIs(row["deployment_eligible"], False)
            self.assertIs(row["exploratory_eligible"], True)

    def test_14_approved_ids_order(self):
        self.assertEqual(self.contract["pseudo_ids"], api.acq.read_sealed(api.inp.DEFAULT_APPROVAL)["pair_ids"])
        self.assertEqual(len(self.contract["pseudo_ids"]), 24)

    def test_15_true_counts(self):
        self.assertEqual(self.contract["true_train_counts"], {"musdb": 73, "mir1k": 81, "instrumental": 11})

    def test_16_lru_bounds(self):
        self.assertLessEqual(len(self.stream.true.cache), 3)
        self.assertLessEqual(len(self.stream.dataset.cache), 1)

    def test_17_contract_seal(self):
        api.storage.check_seal(self.contract)
        api.storage.check_seal(self.stream.contract())

    def test_18_contract_not_training(self):
        self.assertIs(self.contract["training_authorized"], False)
        self.assertIs(self.contract["full_training_transaction_verified"], False)
        self.assertIs(self.contract["cuda_resume_verified"], False)

    def test_19_restore_rejects_old_wrong_bound(self):
        before = self.stream.state_dict()
        for cursor in (4499, 5001, True, 4501.0):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                self.stream.load_state_dict(before | {"cursor": cursor})
            self.assertTrue(api.equal(before, self.stream.state_dict()))

    def test_20_initial_rejects_wrong_full_sampler_before_constructor(self):
        for key, value in (("seed", True), ("cursor", 4500.0), ("teacher", "htdemucs"), ("approval_sha256", "0"*64)):
            with self.subTest(key=key), patch.object(api.inp, "ApprovedTeacherDataset", side_effect=AssertionError("must not construct")):
                with self.assertRaises(ValueError):
                    api.AuthenticatedAudioStream(api.ORIGINAL_SAMPLER | {key: value})

    def assert_guard_mutation(self, obj, attr, mutated):
        original = getattr(obj, attr)
        try:
            setattr(obj, attr, mutated)
            with self.assertRaises(ValueError):
                self.stream.guard()
        finally:
            setattr(obj, attr, original)
        self.stream.guard()

    def test_21_true_pool_mapping_guard(self):
        pools = copy.deepcopy(self.stream.true.pools)
        pools["musdb"][0]["role"] = "development"
        self.assert_guard_mutation(self.stream.true, "pools", pools)

    def test_22_teacher_by_id_guard(self):
        rows = copy.deepcopy(self.stream.dataset.by_id)
        rows[next(iter(rows))]["source"]["samples"] += 1
        self.assert_guard_mutation(self.stream.dataset, "by_id", rows)

    def test_23_teacher_config_guard(self):
        self.assert_guard_mutation(self.stream.dataset, "config", self.stream.dataset.config | {"warmup_frames": 95})

    def test_24_teacher_signature_guard(self):
        self.assert_guard_mutation(self.stream.dataset, "signatures", {})

    def test_25_true_cache_bytes_guard(self):
        key = next(iter(self.stream.true.cache))
        original = self.stream.true.cache[key]
        try:
            self.stream.true.cache[key] = (original[0]+.0001, original[1])
            with self.assertRaises(ValueError):
                self.stream.guard()
        finally:
            self.stream.true.cache[key] = original
        self.stream.guard()

    def test_26_teacher_cache_bytes_guard(self):
        cache = self.stream.dataset.cache
        key = next(iter(cache))
        original = cache[key]
        try:
            cache[key] = original+.0001
            with self.assertRaises(ValueError):
                self.stream.guard()
        finally:
            cache[key] = original
        self.stream.guard()

    def test_27_input_binding_guard(self):
        self.assert_guard_mutation(self.stream, "_bindings", {})

    def test_28_contract_guard(self):
        self.assert_guard_mutation(self.stream, "_contract", self.stream._contract | {"training_authorized": True})

    def test_29_no_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_30_original_hard_limit_no_crop(self):
        before = self.stream.state_dict()
        self.stream.load_state_dict(before | {"cursor": 5000})
        try:
            with patch.object(self.stream.true, "crop", side_effect=AssertionError("must not draw")):
                #206 guard intentionally rejects replaced crop before execution.
                with self.assertRaises((ValueError, AttributeError)):
                    self.stream.route.next_batch()
            with self.assertRaises(ValueError):
                self.stream.route.next_batch()
        finally:
            self.stream.load_state_dict(before, last_metadata=self.first["metadata"])

    def test_31_draw_failure_poison_without_audio(self):
        # Separate tiny synthetic route; never replace an actual decoder/crop.
        synthetic = object.__new__(api.AuthenticatedAudioStream)
        synthetic.poisoned = False
        synthetic.guard = lambda: None
        class FailedRoute:
            def next_batch(self):
                raise KeyboardInterrupt("synthetic before audio")
        synthetic.route = FailedRoute()
        with self.assertRaises(KeyboardInterrupt):
            synthetic.next_batch()
        self.assertTrue(synthetic.poisoned)

    def test_33_contract_copy_noalias(self):
        packet = self.stream.contract()
        packet["source_sampler"]["seed"] = 0
        self.stream.guard()
        self.assertEqual(self.stream.contract()["source_sampler"]["seed"], 20261002)

    def test_34_decoder_versions_bound(self):
        plan = api.acq.read_sealed(ROOT / "results/teacher_library_melband_20261002/plan.json")
        self.assertTrue(api.equal(self.contract["decoder_versions"], plan["media_versions"]))
        for path in self.contract["decoder_executables"].values():
            self.assertEqual(api.file_sha(path), self.contract["input_bindings_sha256"][path])

    def test_35_selected_files_bound_after_replay(self):
        for path in self.stream.true.signatures:
            self.assertEqual(api.file_sha(path), self.stream.true.lock["files"][path]["sha256"])

    def test_32_forbidden_apis_pre_execution(self):
        for name, call in (("load", lambda: torch.load("no-file")),
                           ("Module_init", lambda: torch.nn.Module()),
                           ("grad", lambda: torch.autograd.grad(None, None)),
                           ("backward", lambda: torch.tensor(1.).backward()),
                           ("Adam_init", lambda: torch.optim.Adam([])),
                           ("CUDA_init", lambda: torch.cuda.init())):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "FORBIDDEN208_BEFORE_EXECUTION"):
                call()


if __name__ == "__main__":
    # Outer resources/duplicate guard is also checked by the caller dynamically.
    if MONITOR.exists():
        raise ValueError("Fresh208 evidence directory required; never repeat successful input suite")
    if api.shutil.disk_usage(ROOT).free < 12*2**30:
        raise ValueError("Less than12GiB free; no actual input execution")
    MONITOR.mkdir()
    torch.set_num_threads(2)
    previous = api.storage.capture_cpu_rng()
    try:
        with ExitStack() as blocks:
            for obj, attr in ((torch, "load"), (torch.nn.Module, "__init__"), (torch.nn.Module, "_call_impl"),
                              (torch.autograd, "grad"), (torch.autograd, "backward"), (torch.Tensor, "backward"),
                              (torch.optim.Adam, "__init__"), (torch.optim.Adam, "step"),
                              (torch.cuda, "init"), (torch.cuda, "_lazy_init")):
                blocks.enter_context(patch.object(obj, attr, side_effect=forbidden(attr)))
            result = unittest.main(exit=False, verbosity=2).result
            if not result.wasSuccessful():
                sys.exit(1)
            if result.testsRun != 35:
                raise ValueError("Expected all35 NEW208 tests")
    finally:
        api.storage.restore_cpu_rng(previous)
