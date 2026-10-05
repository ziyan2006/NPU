"""Fresh CPU helper equivalence/ownership/rollback tests, no old unit replay."""
from collections import OrderedDict
import importlib.util
from pathlib import Path
import random
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema232_new_units", ROOT / "scripts/232_ema_fast_state_commitment.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


class FastStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        cls.threads = torch.get_num_threads(); torch.set_num_threads(4)
        cls.no_cuda = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU units only")); cls.no_cuda.start()
        cls.no_child = patch.object(subprocess, "Popen", side_effect=AssertionError("No decoder")); cls.no_child.start()
        cls.old, cls.fast = f.baseline_functions(), f.optimized_functions()
        cls.source_sha = f.sha(f.SOURCE)

    @classmethod
    def tearDownClass(cls):
        cls.no_child.stop(); cls.no_cuda.stop()
        assert cls.source_sha == f.sha(f.SOURCE)
        assert cls.rng[0] == random.getstate() and torch.equal(cls.rng[2], torch.get_rng_state())
        now, before = np.random.get_state(), cls.rng[1]
        assert now[0] == before[0] and np.array_equal(now[1], before[1]) and now[2:] == before[2:]
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(cls.threads)
        print("NEW232_CPU_UNITS source/Python/NumPy/CPU RNG unchanged; decoder/PT/model/Adam/CUDA=0")

    def compare(self, value):
        self.assertEqual(self.old.typed_tree(value), self.fast.typed_tree(value))
        self.assertEqual(self.old.digest(value), self.fast.digest(value))

    def test_all_state_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64,
                      torch.complex128, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
            self.compare(torch.tensor([0, 1, -1], dtype=dtype))

    def test_signed_zero_and_subnormal(self):
        self.compare(torch.tensor([-0., 0., 1e-44, -1e-44]))
        self.assertNotEqual(self.fast.digest(torch.tensor([-0.])), self.fast.digest(torch.tensor([0.])))

    def test_empty_scalar_strided(self):
        for value in (torch.empty(0), torch.tensor(3.), torch.arange(48.).reshape(6, 8).T,
                      torch.arange(48.).reshape(6, 8)[:, ::3]): self.compare(value)

    def test_full_recursive_metadata_types_and_order(self):
        self.compare(OrderedDict([(1, (True, 1., -0.)), ("two", [None, "字", torch.arange(9.)])]))
        for a, b in (([1], (1,)), (True, 1), (1, 1.), (0., -0.),
                     ({"x": 1, "y": 2}, {"y": 2, "x": 1}), ({"x": 1}, OrderedDict(x=1))):
            self.assertFalse(self.fast.equal(a, b))

    def test_chunk_edge_and_tail(self):
        self.compare(torch.arange(f.CHUNK + 1, dtype=torch.float32))

    def test_nonfinite_every_dtype_rejected(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64):
            for bad in (float("nan"), float("inf"), -float("inf")):
                value = torch.tensor([0., bad], dtype=dtype)
                self.assertRaises(ValueError, self.fast.typed_tree, value)
                self.assertRaises(ValueError, self.old.typed_tree, value)

    def test_nonfinite_tail_rejected(self):
        value = torch.zeros(f.CHUNK + 1); value[-1] = float("nan")
        self.assertRaises(ValueError, self.fast.digest, value)

    def test_trainable_and_grad_graph_rejected(self):
        for value in (torch.ones(2, requires_grad=True), torch.ones(2, requires_grad=True) * 2):
            self.assertRaises(ValueError, self.fast.typed_tree, value)

    def test_portable_complete_tree_type_bits_noalias(self):
        source = OrderedDict(raw=torch.arange(12.).reshape(3, 4).T, grad=(torch.tensor([-0., 1.]),), metadata=[True, 1, 1.])
        a, b = self.old.portable(source), self.fast.portable(source)
        self.assertEqual(self.old.typed_tree(a), self.old.typed_tree(b))
        self.assertNotEqual(source["raw"].untyped_storage().data_ptr(), b["raw"].untyped_storage().data_ptr())
        source["raw"][0, 0] = 99.
        self.assertEqual(a["raw"][0, 0], b["raw"][0, 0])

    def test_portable_detaches_graph_and_owns_each_leaf(self):
        value = torch.arange(12., requires_grad=True).reshape(3, 4)
        result = self.fast.portable([value, value])
        self.assertFalse(result[0].requires_grad)
        self.assertIsNone(result[0].grad_fn)
        self.assertNotEqual(result[0].untyped_storage().data_ptr(), result[1].untyped_storage().data_ptr())

    def test_no_digest_memoization_after_external_mutation(self):
        value = torch.tensor([0., 1.]); before = self.fast.digest(value)
        value.numpy()[1] = 7.
        self.assertNotEqual(before, self.fast.digest(value))

    def test_complete_seals_compatible_and_tamper_rejected(self):
        value = {"x": torch.arange(16.), "list": [1, 1., True]}
        a, b = self.old.seal(value), self.fast.seal(value)
        self.assertEqual(self.old.typed_tree(a), self.old.typed_tree(b))
        self.old.check_seal(b); self.fast.check_seal(a)
        b["list"][0] = 2
        self.assertRaises(ValueError, self.fast.check_seal, b)

    def test_reseal_and_unknown_metadata_rejected(self):
        self.assertRaises(ValueError, self.fast.seal, {"content_sha256": "x"})
        for value in ({1, 2}, object(), float("nan")):
            self.assertRaises(ValueError, self.fast.typed_tree, value)

    def test_synthetic_full_before_after_and_rollback(self):
        state = dict(raw=torch.arange(16.), Adam={"m": torch.ones(16), "v": torch.ones(16) * 2},
                     shadow=torch.arange(16.), cursor=0, shadow_count=0, rng=torch.get_rng_state())
        before = self.fast.portable(state); sealed = self.fast.seal(before)
        state["raw"].add_(.125); state["Adam"]["m"].mul_(.9); state["shadow"].add_(.01)
        state["cursor"] = state["shadow_count"] = 1
        self.fast.check_seal(self.fast.seal(self.fast.portable(state)))
        restored = self.fast.portable(before); self.fast.check_seal(sealed)
        self.assertTrue(self.old.equal(restored, before))

    def test_cpu_scalar_values_mixed_dtype_and_order(self):
        values = [torch.tensor(-0.), torch.tensor(1.5, dtype=torch.float64), torch.tensor(2., dtype=torch.float16)]
        self.assertEqual([number.hex() for number in f.read_scalars(values)], [float(value).hex() for value in values])

    def test_scalar_reads_unsupported_values_rejected(self):
        for value in (torch.ones(2), torch.tensor(1), torch.tensor(1j), 1.):
            self.assertRaises(ValueError, f.read_scalars, [value])

    def test_scalar_read_no_cache_and_empty(self):
        value = torch.tensor(1.); self.assertEqual(f.read_scalars([]), [])
        self.assertEqual(f.read_scalars([value]), [1.]); value.fill_(2.)
        self.assertEqual(f.read_scalars([value]), [2.])


if __name__ == "__main__": unittest.main(verbosity=2)
