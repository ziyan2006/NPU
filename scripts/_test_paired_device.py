"""Synthetic FP32 backend/resume contracts; CPU tests never init/use CUDA.

PyTorch 2.14 Adam may itself query accelerator availability on a CPU step.
That read-only framework check is not CUDA initialization or GPU work.
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("paired_device_test", Path(__file__).with_name("147_paired_device_mechanics.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
torch.set_num_threads(2)
PROTOCOL = json.loads(d.m.data.PROTOCOL.read_text(encoding="utf-8"))


def factory():
    net = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(.03)
    return net


def engine(**kwargs):
    return d.DevicePairedEngine(factory, PROTOCOL, {"fixture": "synthetic"}, purpose=d.PURPOSE, **kwargs)


class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.runtime = d.deterministic_runtime("cpu")
        self.runtime.__enter__()
        self.addCleanup(self.runtime.__exit__, None, None, None)

    def test_formal_default_and_unknown_backend_reject_before_cuda_factory_adam(self):
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("Adam constructed")):
            for device, purpose in (("cpu", "training"), ("cuda", "training"), ("cpu", "cpu_mechanism"), ("auto", d.PURPOSE)):
                with self.assertRaises(ValueError):
                    d.DevicePairedEngine(lambda: self.fail("factory used"), PROTOCOL, {}, device, purpose)
            with self.assertRaises(ValueError):
                d.formal_entry()

    def test_cpu_update_never_initializes_or_uses_cuda(self):
        with patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CUDA RNG used")), \
             patch.object(torch.cuda, "set_rng_state_all", side_effect=AssertionError("CUDA RNG used")), \
             patch.object(torch.cuda, "synchronize", side_effect=AssertionError("GPU work")), \
             patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")):
            e, s = engine(), d.SyntheticBatchStream()
            row = e.update_next(s)
            self.assertEqual(row["step"], 1)
            self.assertNotEqual(row["losses"][d.ARMS[0]], row["losses"][d.ARMS[1]])
            self.assertEqual(len(row["input_sha256"]), 6)

    def test_disk_resume_matches_model_adam_python_numpy_cpu_rng_cursor_budget(self):
        e, s = engine(), d.SyntheticBatchStream()
        e.update_next(s)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "NONRELEASE.pt"
            d.m.save_new(path, e.state_dict(s))
            digest = d.acq.sha256(path)
            row = e.update_next(s)
            expected = e.state_dict(s)
            other, stream = engine(), d.SyntheticBatchStream()
            other.load_state_dict(d.m.load_checked_checkpoint(path, digest), stream)
            replay = other.update_next(stream)
            self.assertEqual({k: v for k, v in row.items() if k != "seconds"}, {k: v for k, v in replay.items() if k != "seconds"})
            self.assertTrue(d.m.equal_state(expected, other.state_dict(stream)))
            with self.assertRaises(FileExistsError):
                d.m.save_new(path, expected)
            with self.assertRaises(ValueError):
                d.m.load_checked_checkpoint(path, "0" * 64)

    def test_second_arm_failure_restores_both_arms_modes_rng_and_input(self):
        e, s = engine(), d.SyntheticBatchStream()
        e.update_next(s)
        for model in e.models.values():
            model.eval()
        before = e.state_dict(s)
        original = e.backward
        def fail(net, *args):
            d.m.random.random(); d.m.np.random.random(); torch.rand(7)
            if net is e.models[d.ARMS[1]]:
                raise RuntimeError("second arm failed")
            return original(net, *args)
        e.backward = fail
        with self.assertRaises(RuntimeError):
            e.update_next(s)
        self.assertTrue(d.m.equal_state(before, e.state_dict(s)))

    def test_partial_bad_runtime_binding_or_extended_budget_rejected(self):
        e, s = engine(), d.SyntheticBatchStream()
        e.update_next(s)
        saved = e.state_dict(s)
        for key, value in (("binding", {}), ("runtime", {"device": "cuda"}), ("limit", 10000), ("purpose", "TRAIN"), ("step", True)):
            with self.assertRaises(ValueError):
                e.load_state_dict(saved | {key: value}, s)
            self.assertTrue(d.m.equal_state(saved, e.state_dict(s)))
        bad = copy.deepcopy(saved)
        bad["arms"][d.ARMS[1]]["updates"] = 0
        with self.assertRaises(ValueError):
            e.load_state_dict(bad, s)

    def test_adam_names_missing_moments_wrong_lr_and_nonfinite_rejected(self):
        e, s = engine(), d.SyntheticBatchStream()
        e.update_next(s)
        saved = e.state_dict(s)
        for kind in ("names", "moments", "lr", "nan", "shape", "config"):
            bad = copy.deepcopy(saved)
            arm = bad["arms"][d.ARMS[0]]
            if kind == "names":
                arm["parameter_names"].reverse()
            elif kind == "moments":
                arm["optimizer"]["state"].clear()
            elif kind == "lr":
                arm["optimizer"]["param_groups"][0]["lr"] = 2e-4
            elif kind == "config":
                arm["optimizer"]["param_groups"][0]["betas"] = (.5, .9)
            elif kind == "nan":
                next(iter(arm["model"].values())).flatten()[0] = float("nan")
            else:
                arm["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
            with self.assertRaises(ValueError):
                e.load_state_dict(bad, s)
            self.assertTrue(d.m.equal_state(saved, e.state_dict(s)))

    def test_mid_load_failure_rolls_back_state_not_just_validates_metadata(self):
        e, s = engine(), d.SyntheticBatchStream()
        e.update_next(s)
        saved = e.state_dict(s)
        e.update_next(s)
        before = e.state_dict(s)
        optimizer = e.optimizers[d.ARMS[1]]
        original, calls = optimizer.load_state_dict, 0
        def fail_once(state):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("load failed after first arm")
            return original(state)
        with patch.object(optimizer, "load_state_dict", side_effect=fail_once):
            with self.assertRaises(RuntimeError):
                e.load_state_dict(saved, s)
        self.assertTrue(d.m.equal_state(before, e.state_dict(s)))

    def test_no_real_stream_or_nonfinite_wrong_truth_batch(self):
        e, s = engine(), d.SyntheticBatchStream()
        with self.assertRaises(ValueError):
            e.update_next(SimpleNamespace(cursor=0))
        before, original = e.state_dict(s), s.next_batch
        for kind in ("synthetic", "truth", "finite"):
            def changed():
                batch = original()
                if kind == "synthetic":
                    batch["synthetic_only"] = False
                elif kind == "truth":
                    batch["targets"][d.ARMS[1]][0] += .01
                else:
                    batch["x"][0, 0, 0] = float("nan")
                return batch
            s.next_batch = changed
            with self.assertRaises(ValueError):
                e.update_next(s)
            self.assertTrue(d.m.equal_state(before, e.state_dict(s)))

    def test_cap_and_no_backend_auto_fallback(self):
        e, s = engine(), d.SyntheticBatchStream()
        for _ in range(3):
            e.update_next(s)
        with self.assertRaises(ValueError):
            e.update_next(s)
        self.assertEqual(s.cursor, 3)
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(ValueError):
                d.runtime_identity("cuda")

    def test_checkpoints_are_independent_cpu_clones_and_modes_roundtrip(self):
        e, s = engine(), d.SyntheticBatchStream()
        for model in e.models.values():
            model.eval()
        state = e.state_dict(s)
        with torch.no_grad():
            next(e.models[d.ARMS[0]].parameters()).add_(1)
        self.assertFalse(d.m.equal_state(state, e.state_dict(s)))
        e.load_state_dict(state, s)
        self.assertTrue(d.m.equal_state(state, e.state_dict(s)))
        self.assertFalse(e.models[d.ARMS[0]].training)

    def test_external_state_cannot_alias_parameter_order_or_engine_binding(self):
        e, s = engine(), d.SyntheticBatchStream()
        state = e.state_dict(s)
        state["arms"][d.ARMS[0]]["parameter_names"].reverse()
        state["binding"]["fixture"] = "changed"
        self.assertEqual(e.names[d.ARMS[0]], ["0.weight", "0.bias"])
        self.assertEqual(e.binding, {"fixture": "synthetic"})
        with self.assertRaises(ValueError):
            e.load_state_dict(state, s)

    def test_unrecoverable_restore_poisoned_engine_cannot_continue(self):
        e, s = engine(), d.SyntheticBatchStream()
        saved = e.state_dict(s)
        with patch.object(e.optimizers[d.ARMS[1]], "load_state_dict", side_effect=RuntimeError("permanent load failure")):
            with self.assertRaises(RuntimeError):
                e.load_state_dict(saved, s)
        self.assertTrue(e.poisoned)
        with self.assertRaises(ValueError):
            e.update_next(s)

    def test_changed_runtime_rejected_and_deterministic_flags_required(self):
        e, s = engine(), d.SyntheticBatchStream()
        torch.use_deterministic_algorithms(False)
        try:
            with self.assertRaises(ValueError):
                e.state_dict(s)
            with self.assertRaises(ValueError):
                engine()
        finally:
            torch.use_deterministic_algorithms(True)

    def test_synthetic_cursor_identity_and_pair_inputs(self):
        s = d.SyntheticBatchStream()
        row = s.next_batch()
        saved = s.state_dict()
        next_row = s.next_batch()
        other = d.SyntheticBatchStream()
        other.load_state_dict(saved)
        self.assertTrue(d.m.equal_state(next_row, other.next_batch()))
        self.assertTrue(torch.equal(row["targets"][d.ARMS[0]][:3], row["targets"][d.ARMS[1]][:3]))
        self.assertFalse(torch.equal(row["targets"][d.ARMS[0]][3:], row["targets"][d.ARMS[1]][3:]))
        for value in (-1, 10000, True):
            with self.assertRaises(ValueError):
                s.load_state_dict(saved | {"cursor": value})

    def test_gpu_preflight_defers_busy_low_memory_unreadable_multiple_devices(self):
        for output, allowed in (("0, 5000, 0\n", True), ("0, 5000, 33\n", False), ("0, 1000, 0\n", False)):
            with patch.object(d.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
                self.assertEqual(d.gpu_preflight()["allowed"], allowed)
        for output in ("0, N/A, 0\n", "0, 5000, 0\n1, 5000, 0\n", ""):
            with patch.object(d.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
                with self.assertRaises(ValueError):
                    d.gpu_preflight()

    def test_busy_cli_has_no_cuda_model_or_optimizer_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=d.ROOT / "results") as parent:
            out = Path(parent) / "deferred"
            with patch.object(d, "gpu_preflight", return_value={"allowed": False}), \
                 patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
                 patch.object(d.m, "frozen_factory", side_effect=AssertionError("factory used")):
                self.assertEqual(d.smoke(out, "cuda"), 2)
                self.assertEqual([p.name for p in out.iterdir()], ["gpu_preflight.json"])
                with self.assertRaises(ValueError):
                    d.smoke(out, "cuda")


if __name__ == "__main__":
    unittest.main()
