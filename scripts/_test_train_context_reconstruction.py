"""Synthetic safety/geometry tests; NOT real TRAIN, migration or audio evidence."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("frontend_test", Path(__file__).with_name("181_diagnose_train_context_reconstruction.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.001))

    def forward(self, bands):
        value = F.avg_pool2d(F.pad(bands, (2, 0)), (1, 3), stride=1)*self.scale
        return torch.cat((torch.tanh(value), torch.zeros_like(value)), dim=1)


def metadata():
    return [{"domain": domain, "role": "train" if j < 3 else "pseudo_label_train_candidate",
        "vocal_db": 0 if j < 3 else None, "input_pcm_sha256": "0"*64}
        for j, domain in enumerate(q.m.DOMAINS)]


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.wa = torch.from_numpy(q.core.t09.make_analysis_matrix())
        cls.gs = torch.from_numpy(q.core.t09.make_synthesis_matrix())
        generator = torch.Generator().manual_seed(181)
        cls.x = torch.randn((1, 2, q.SAMPLES), generator=generator)*.02
        cls.v = cls.x*.2
        cls.bands = torch.randn((1, 2, 128, q.FRAMES), generator=generator).abs()

    def test_01_original_intervals(self):
        scope = q.fixed_scope()
        self.assertEqual(scope["native_sample_interval"], [25088, 89344])
        self.assertEqual(scope["common_sample_interval"], [33280, 89344])
        self.assertEqual(scope["slot_checks"], 216)
        self.assertFalse(scope["training_authorized"])

    def test_02_changed_scope_rejected(self):
        for key, bad in (("model_updates", 1), ("threads", 4), ("cursors", list(range(12))), ("lf_kill_bands", 0)):
            value = q.fixed_scope(); value[key] = bad
            with self.assertRaises(ValueError): q.check_plan(value)

    def test_03_bool_not_integer_scope(self):
        value = q.fixed_scope(); value["model_updates"] = False
        with self.assertRaises(ValueError): q.check_plan(value)

    def test_04_changed_binding_rejected(self):
        with mock.patch.object(q.acq, "sha256", return_value="real"):
            with self.assertRaises(ValueError): q.check_bindings({"file": "different"})
            q.check_bindings({"file": "real"})
        with self.assertRaises(ValueError): q.check_bindings({})

    def test_05_old_output_rejected(self):
        with mock.patch.object(Path, "exists", return_value=True):
            with self.assertRaises(ValueError): q.fresh_output(q.ROOT / "results/existing")
            with self.assertRaises(ValueError): q.reject_existing_run(q.ROOT / "results/existing")

    def test_06_output_root_rejected(self):
        with self.assertRaises(ValueError): q.fresh_output(q.ROOT / "results")
        with self.assertRaises(ValueError): q.fresh_output(q.ROOT / "reports/new")

    def test_07_cpu_fp32_finite_contract(self):
        for tensor in (torch.zeros(2, dtype=torch.float64), torch.full((2,), float("nan")), torch.zeros(2, device="meta")):
            with self.assertRaises(ValueError): q.cpu_float(tensor)

    def test_08_wrong_shape(self):
        with self.assertRaises(ValueError): q.cpu_float(self.x, (1, 2, 5))
        with self.assertRaises(ValueError): q.slot_probe(TinyCausal(), self.x[..., :-1], self.v[..., :-1], self.wa, self.gs, metadata()[0])

    def test_09_stride_history_block_contract(self):
        for history, block in ((True, 16), (7, 16), (128, 15), (-8, 16), (128, False)):
            with self.assertRaises(ValueError): q.context_mask(TinyCausal(), self.bands, history, block)

    def test_10_context_and_negative_control(self):
        net = TinyCausal()
        with torch.no_grad(): whole = q.output_mask(net, self.bands)
        for block in (16, 256):
            retained = q.context_mask(net, self.bands, 128, block)
            self.assertTrue(torch.equal(whole, retained))
        reset = q.context_mask(net, self.bands, 0, 16)
        self.assertGreater(q.mask_stats(reset, whole)["max_abs"], 0)

    def test_11_tail_padding_and_phase(self):
        value = F.pad(self.bands[..., :345], (0, 7))
        self.assertEqual(value.shape[-1], 352)
        with torch.no_grad(): whole = q.output_mask(TinyCausal(), value)
        self.assertEqual(q.context_mask(TinyCausal(), value, 128, 256).shape, whole.shape)
        with self.assertRaises(ValueError): q.context_mask(TinyCausal(), self.bands[..., :345], 128, 16)

    def test_12_lf_bins_come_from_synthesis(self):
        forced, info = q.lf_geometry(self.gs)
        self.assertEqual(info["forced_zero_fft_bins"], (self.gs[:, 44:].sum(1) == 0).nonzero().flatten().tolist())
        self.assertGreater(int(forced.sum()), 0)
        self.assertEqual(info["forced_zero_bin_hz"], [j*44100/1024 for j in info["forced_zero_fft_bins"]])

    def test_13_invalid_synthesis_rejected(self):
        for matrix in (self.gs[:, :127], self.gs*2, -self.gs):
            with self.assertRaises(ValueError): q.lf_geometry(matrix)

    def test_14_zero_reference_skips_ratio(self):
        spectrum = torch.ones((1, 2, 513, 352), dtype=torch.complex64)
        mask = torch.full((1, 2, 128, 352), .5)
        info = q.lf_statistics(spectrum, torch.zeros_like(spectrum), mask, self.gs)
        self.assertIsNone(info["reference_forced_zero_energy_fraction"])
        self.assertEqual(info["reference_fraction_skip"], "zero_or_near_zero_reference")
        self.assertFalse(info["waveform_quality_floor_claimed"])

    def test_15_forced_only_reference_fraction_one(self):
        spectrum = torch.ones((1, 2, 513, 352), dtype=torch.complex64)
        truth = torch.zeros_like(spectrum)
        forced, _ = q.lf_geometry(self.gs); truth[:, :, forced] = 1
        info = q.lf_statistics(spectrum, truth, torch.full((1, 2, 128, 352), .5), self.gs)
        self.assertEqual(info["reference_forced_zero_energy_fraction"], 1.)

    def test_16_spectral_dtype_rejected(self):
        spectrum = torch.ones((1, 2, 513, 352), dtype=torch.complex128)
        with self.assertRaises(ValueError): q.lf_statistics(spectrum, spectrum, torch.zeros((1, 2, 128, 352)), self.gs)

    def test_17_complete_probe_reconstruction(self):
        value = q.slot_probe(TinyCausal(), self.x, self.v, self.wa, self.gs, metadata()[0])
        self.assertTrue(value["same_mask_126_vs_110"]["bit_equal"])
        self.assertEqual(value["same_mask_126_vs_110"]["max_abs"], 0)
        self.assertLess(value["input_roundtrip"]["all"]["max_abs"], 5e-6)
        self.assertEqual(value["paths"]["whole352"]["common"]["sample_interval"], [33280, 89344])
        self.assertEqual(value["paths"]["whole352"]["native_descriptive_only"]["sample_interval"], [25088, 89344])

    def test_18_pseudo_is_not_physical_truth(self):
        value = q.slot_probe(TinyCausal(), self.x, self.v, self.wa, self.gs, metadata()[3])
        for path in value["paths"].values():
            self.assertIsNone(path["common"]["physical_true_reference_metrics"])
            self.assertFalse(path["common"]["pseudo_target_is_ground_truth"])

    def test_19_all_model_modes_grad_rng_preserved(self):
        net = TinyCausal(); net.training = False
        old_grad = torch.tensor(.25); net.scale.grad = old_grad
        state, rng = copy.deepcopy(net.state_dict()), q.m.capture_rng("cpu")
        q.slot_probe(net, self.x, self.v, self.wa, self.gs, metadata()[0])
        self.assertTrue(q.m.equal_state(state, net.state_dict()))
        self.assertTrue(q.m.equal_state(rng, q.m.capture_rng("cpu")))
        self.assertIs(net.scale.grad, old_grad)
        self.assertEqual(float(net.scale.grad), .25)
        self.assertFalse(net.training)
        self.assertFalse(torch.cuda.is_initialized())

    def test_20_state_mutation_fails_and_restores(self):
        net = TinyCausal(); state = copy.deepcopy(net.state_dict())
        with self.assertRaises(ValueError):
            with q.readonly_model(net), torch.no_grad(): net.scale.add_(1)
        self.assertTrue(q.m.equal_state(state, net.state_dict()))

    def test_21_mode_mutation_fails_and_restores(self):
        net = TinyCausal()
        with self.assertRaises(ValueError):
            with q.readonly_model(net): net.eval()
        self.assertTrue(net.training)

    def test_22_rng_mutation_fails_and_restores(self):
        net = TinyCausal(); rng = q.m.capture_rng("cpu")
        with self.assertRaises(ValueError):
            with q.readonly_model(net): torch.rand(1)
        self.assertTrue(q.m.equal_state(rng, q.m.capture_rng("cpu")))

    def test_23_grad_mutation_fails_and_restores(self):
        net = TinyCausal()
        with self.assertRaises(ValueError):
            with q.readonly_model(net): net.scale.grad = torch.tensor(.5)
        self.assertIsNone(net.scale.grad)

    def test_24_dropout_forbidden(self):
        with self.assertRaises(ValueError): q.check_net(torch.nn.Dropout())

    def test_25_active_other_worker_rejected(self):
        reply = type("Reply", (), {"stdout": "999999\n"})()
        with mock.patch.object(q.subprocess, "run", return_value=reply):
            with self.assertRaises(ValueError): q.no_active_worker()
        reply.stdout = f"{os.getpid()}\n{os.getppid()}\n"
        with mock.patch.object(q.subprocess, "run", return_value=reply): q.no_active_worker()

    def test_26_disk_reserve(self):
        with mock.patch.object(q.shutil, "disk_usage", return_value=type("Usage", (), {"free": 11*(1 << 30)})()):
            with self.assertRaises(ValueError): q.check_disk()

    def test_27_empty_delta_and_wrong_masks_rejected(self):
        with self.assertRaises(ValueError): q.delta_stats(torch.empty(0), torch.empty(0))
        with self.assertRaises(ValueError): q.mask_stats(torch.zeros(1, 2, 128, 16), torch.zeros(1, 2, 128, 16))

    def test_35_instrumental_zero_reference(self):
        value = q.slot_probe(TinyCausal(), self.x, torch.zeros_like(self.x), self.wa, self.gs, metadata()[2])
        self.assertIsNone(value["lf44"]["reference_forced_zero_energy_fraction"])
        self.assertEqual(value["lf44"]["reference_fraction_skip"], "zero_or_near_zero_reference")
        self.assertIsNotNone(value["paths"]["whole352"]["common"]["physical_true_reference_metrics"])

    def test_36_development_role_rejected(self):
        meta = metadata()[0] | {"role": "development"}
        with self.assertRaises(ValueError): q.slot_probe(TinyCausal(), self.x, self.v, self.wa, self.gs, meta)

    def test_37_exit_fraction_trailing_zero_only(self):
        actual = {"utc": "2026-10-03T11:12:29.0600950Z", "exit_code": 0, "launcher_process_id": 58036}
        reviewed = actual | {"utc": "2026-10-03T11:12:29.060095Z"}
        self.assertTrue(q.same_exit_record(actual, reviewed))
        self.assertFalse(q.same_exit_record(actual | {"utc": "2026-10-03T11:12:29.0600951Z"}, reviewed))
        self.assertFalse(q.same_exit_record(actual | {"launcher_process_id": 58037}, reviewed))
        self.assertFalse(q.same_exit_record(actual | {"exit_code": False}, reviewed))
        self.assertFalse(q.same_exit_record(actual | {"extra": 1}, reviewed))

    def test_38_invalid_exit_timestamp_rejected(self):
        with self.assertRaises(ValueError): q.same_exit_record({"utc": "no-time"}, {"utc": "no-time"})


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        wa, gs = (torch.from_numpy(fn()) for fn in (q.core.t09.make_analysis_matrix, q.core.t09.make_synthesis_matrix))
        x = torch.full((1, 2, q.SAMPLES), .02)
        probe = q.slot_probe(TinyCausal(), x, x*.2, wa, gs, metadata()[0])
        inputs = [{"counter": counter, "input_sha256": ["0"*64]*6, "target_sha256": ["1"*64]*6, "metadata": metadata()} for counter in q.CURSORS]
        rows = []
        for model in q.MODELS:
            for index, counter in enumerate(q.CURSORS):
                slots = []
                for j, meta in enumerate(inputs[index]["metadata"]):
                    value = copy.deepcopy(probe)
                    if j >= 3:
                        for path in value["paths"].values():
                            for region in ("common", "native_descriptive_only"):
                                path[region]["physical_true_reference_metrics"] = None
                    slots.append({"slot": j, "metadata": meta, "probe": value})
                rows.append({"model": model, "counter": counter, "slots": slots, "seconds": 0.})
        cls.plan = q.fixed_scope() | {"bindings_sha256": {"unit": "fake"}, "model_digests": {name: "unit_model" for name in q.MODELS}}
        cls.result = cls.plan | {"rows": rows, "inputs": inputs, "model_state_modes_rng_grad_unchanged": True,
            "runtime": {"device": "cpu", "threads": 2, "deterministic": True}, "summary": q.summarize(rows)}

    def test_28_full_synthetic_accounting(self):
        q.validate_result(self.plan, self.result)
        for model in q.MODELS:
            self.assertEqual(self.result["summary"][model]["model_batches"], 12)
            self.assertEqual(self.result["summary"][model]["slot_checks"], 72)

    def test_29_missing_row_rejected(self):
        value = copy.deepcopy(self.result); value["rows"].pop()
        with self.assertRaises(ValueError): q.validate_result(self.plan, value)

    def test_30_reordered_counter_rejected(self):
        value = copy.deepcopy(self.result); value["rows"][0]["counter"] += 1
        with self.assertRaises(ValueError): q.validate_result(self.plan, value)

    def test_31_changed_summary_rejected(self):
        value = copy.deepcopy(self.result); value["summary"][q.MODELS[0]]["slot_checks"] = 71
        with self.assertRaises(ValueError): q.validate_result(self.plan, value)

    def test_32_changed_common_interval_rejected(self):
        value = copy.deepcopy(self.result); value["rows"][0]["slots"][0]["probe"]["paths"]["whole352"]["common"]["sample_interval"][0] += 1
        with self.assertRaises(ValueError): q.validate_result(self.plan, value)

    def test_33_pseudo_promoted_to_truth_rejected(self):
        value = copy.deepcopy(self.result); value["rows"][0]["slots"][3]["probe"]["paths"]["whole352"]["common"]["physical_true_reference_metrics"] = {}
        with self.assertRaises(ValueError): q.validate_result(self.plan, value)

    def test_34_wrong_lf_fraction_rejected(self):
        value = copy.deepcopy(self.result); value["rows"][0]["slots"][0]["probe"]["lf44"]["reference_forced_zero_energy_fraction"] = 2.
        with self.assertRaises(ValueError): q.validate_result(self.plan, value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
