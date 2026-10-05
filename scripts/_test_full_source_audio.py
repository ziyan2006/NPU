"""Whole-source context/metrics contracts, synthetic CPU waveforms only."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import soundfile as sf
import torch

spec = importlib.util.spec_from_file_location("full_source_test", Path(__file__).with_name("148_diagnose_full_source_audio.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
torch.set_num_threads(2)


def model():
    net = a.core.t09.CausalSpectralUNet(enc=(4, 8, 12, 16), bottleneck_blocks=2)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.fill_(.01)
    return net


def waves():
    t = torch.arange(99856, dtype=torch.float32)
    vocal = .05 * torch.sin(t * .023).repeat(2, 1)
    backing = .08 * torch.sin(t * .011).repeat(2, 1)
    return vocal + backing, vocal


class AudioTests(unittest.TestCase):
    def test_role_preselection_alphabetical_before_quality_and_bounded_source(self):
        pools = {domain: [{"track_id": "z", "role": "development", "source_container_duration_s": 5.},
                          {"track_id": "a", "role": "development", "source_container_duration_s": 6.}]
                 for domain in ("musdb", "mir1k", "instrumental")}
        self.assertEqual([r["track_id"] for r in a.selected_records(SimpleNamespace(val=pools))], ["a"] * 3)
        pools["musdb"][0]["role"] = "train"
        with self.assertRaises(ValueError):
            a.selected_records(SimpleNamespace(val=pools))
        pools["musdb"][0]["role"] = "development"
        pools["mir1k"][1]["source_container_duration_s"] = 500
        with self.assertRaises(ValueError):
            a.selected_records(SimpleNamespace(val=pools))

    def test_retained_history_and_layer_state_match_continuous_mask(self):
        net = model().eval()
        gen = torch.Generator(device="cpu").manual_seed(90)
        bands = torch.randn(1, 2, 128, 640, generator=gen).abs() * .1
        with torch.no_grad():
            full = (net(bands)[0, :2] + 1) / 2
        retained = a.context_masks(net, bands, 128, 256)
        streamed = a.context_masks(a.ctx.enable_streaming(copy.deepcopy(net), a.core.t09), bands, 0, 16)
        self.assertLess(float((retained - full).abs().max()), 1e-4)
        self.assertLess(float((streamed - full).abs().max()), 1e-4)
        reset = a.context_masks(net, bands, 0, 16)
        self.assertGreater(float((reset - full).abs().max()), 1e-4)

    def test_bad_shape_or_phase_never_silently_padded(self):
        net = model().eval()
        for frames, history, block in ((15, 128, 16), (32, 7, 16), (32, 128, 15), (32, -8, 16)):
            with self.assertRaises(ValueError):
                a.context_masks(net, torch.ones(1, 2, 128, frames), history, block)

    def test_splice_and_cold_start_edges_excluded_from_scoring(self):
        valid = a.valid_samples(200000, (90000,))
        self.assertFalse(bool(valid[:128 * 256].any()))
        self.assertFalse(bool(valid[-512:].any()))
        self.assertFalse(bool(valid[90000 - 512:90000 + 128 * 256].any()))
        self.assertTrue(bool(valid[60000:70000].all()))
        for length, joins in ((1000, ()), (100000, (0,)), (100000, (100000,)), (100000, (True,))):
            with self.assertRaises(ValueError):
                a.valid_samples(length, joins)

    def test_real_frontend_metrics_readonly_no_optimizer_or_cuda(self):
        net, (x, v) = model().train(), waves()
        original, rng = copy.deepcopy(net.state_dict()), a.m.capture_rng("cpu")
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer started")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
             patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA init")):
            result, outputs = a.diagnose(net, x, v)
        self.assertEqual(result["samples"], x.shape[-1])
        self.assertEqual(set(result["waveform_metrics"]), set(outputs))
        self.assertLess(result["stateful16_max_mask_error"], 1e-4)
        self.assertGreater(result["mask_diagnostics"]["reset16_lf44"]["mask_mae_vs_continuous"], 1e-4)
        self.assertTrue(a.m.equal_state(original, net.state_dict()))
        self.assertTrue(a.m.equal_state(rng, a.m.capture_rng("cpu")))
        self.assertTrue(net.training)

    def test_bad_audio_and_failed_evaluation_preserve_weights_rng_mode(self):
        net, (x, v) = model().train(), waves()
        original, rng = copy.deepcopy(net.state_dict()), a.m.capture_rng("cpu")
        with self.assertRaises(ValueError):
            a.diagnose(net, x[:, :-1], v)
        x[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            a.diagnose(net, x, v)
        x, v = waves()
        def fail(*args):
            torch.rand(7)
            with torch.no_grad():
                next(net.parameters()).add_(1)
            raise RuntimeError("synthetic metric error")
        with patch.object(a.d.suite, "separation_metrics", side_effect=fail):
            with self.assertRaises(RuntimeError):
                a.diagnose(net, x, v)
        self.assertTrue(a.m.equal_state(original, net.state_dict()))
        self.assertTrue(a.m.equal_state(rng, a.m.capture_rng("cpu")))
        self.assertTrue(net.training)

    def test_float_wav_preserves_gain_overpeak_and_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.wav"
            wave = torch.tensor([[1.2, -.8, .5], [1.2, -.8, .5]], dtype=torch.float32)
            a.save_wave(path, wave)
            loaded, sr = sf.read(path, dtype="float32", always_2d=True)
            self.assertEqual(sr, 44100)
            self.assertTrue(torch.equal(torch.from_numpy(loaded.T.copy()), wave))
            self.assertEqual(sf.info(path).subtype, "FLOAT")
            with self.assertRaises(FileExistsError):
                a.save_wave(path, wave)

    def test_source_splices_follow_exact_decoder_lengths(self):
        record = {"domain": "mir1k", "mix_files": ["a", "b", "c"]}
        with patch.object(a.core.t23, "_read", side_effect=lambda path: (torch.zeros(2, {"a": 10, "b": 20, "c": 30}[str(path)]), 44100)):
            self.assertEqual(a.source_joins(record, 60), [10, 30])
            with self.assertRaises(ValueError):
                a.source_joins(record, 50)
        self.assertEqual(a.source_joins({"domain": "musdb"}, 60), [])

    def test_boundary_stats_are_mask_not_audio_dropout_claim(self):
        reference = torch.zeros(2, 128, 256)
        alternate = reference.clone()
        alternate[:, :, 128::16] = 1
        result = a.boundary_metrics(alternate, reference)
        self.assertGreater(result["mask_mae_vs_continuous"], 0)
        self.assertEqual(result["mask_max_abs_vs_continuous"], 1)
        with self.assertRaises(ValueError):
            a.boundary_metrics(alternate[..., :-1], reference)


if __name__ == "__main__":
    unittest.main()
