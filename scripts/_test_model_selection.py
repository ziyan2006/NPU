"""Synthetic split, reconstruction, selection and reproducibility contracts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

spec = importlib.util.spec_from_file_location("selection_under_test", Path(__file__).with_name("119_model_selection_suite.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(4)
regression_spec = importlib.util.spec_from_file_location("regression_under_test", Path(__file__).with_name("121_evaluate_layout_control.py"))
regression = importlib.util.module_from_spec(regression_spec)
regression_spec.loader.exec_module(regression)
control_spec = importlib.util.spec_from_file_location("control_under_test", Path(__file__).with_name("120_train_layout_control.py"))
control = importlib.util.module_from_spec(control_spec)
control_spec.loader.exec_module(control)


def summary(vocal_snr=0., instrumental_snr=20., preserved=1., residual=.5):
    values = {"vocal_error_snr_db": vocal_snr, "accompaniment_error_snr_db": instrumental_snr,
              "joint_fit_accompaniment_gain": preserved, "residual_vocal_abs_gain": residual}
    return {domain: {"metrics": {k: {"mean": v, "by_track": {"song": v}, "n_tracks": 1}
                                for k, v in values.items()}}
            for domain in (*module.VOCAL_DOMAINS, "instrumental/native")}


class SelectionTests(unittest.TestCase):
    def test_projection_distinguishes_vocal_residue_and_instrument_removal(self):
        t = torch.arange(4096, dtype=torch.float64)
        a = torch.sin(2*torch.pi*17*t/4096).repeat(2, 1)
        v = torch.sin(2*torch.pi*53*t/4096).repeat(2, 1) * .1
        mix = a + v
        for retained_a, remaining_v in ((1., .2), (.7, .1), (0., 0.)):
            predicted_v = mix - (retained_a*a + remaining_v*v)
            scores = module.separation_metrics(predicted_v, mix, v)
            self.assertAlmostEqual(scores["joint_fit_accompaniment_gain"], retained_a, places=8)
            self.assertAlmostEqual(scores["residual_vocal_abs_gain"], remaining_v, places=8)
        perfect = module.separation_metrics(v, mix, v)
        self.assertGreater(perfect["vocal_error_snr_db"], 100.)

    def test_good_candidate_passes_but_instrumental_only_gain_cannot_win(self):
        baseline = summary()
        self.assertTrue(module.assess(summary(.3, 20.1, .99, .45), baseline)["eligible"])
        self.assertFalse(module.assess(summary(0., 40., 1., .5), baseline)["eligible"])

    def test_correlated_or_absent_vocal_does_not_fabricate_leakage_coefficient(self):
        a = torch.linspace(-1, 1, 4096).repeat(2, 1)
        for v in (a*.1, torch.zeros_like(a)):
            scores = module.separation_metrics(v*.5, a+v, v)
            self.assertIsNone(scores["joint_fit_accompaniment_gain"])
            self.assertIsNone(scores["residual_vocal_abs_gain"])

    def test_remaining_voice_increase_is_not_hidden_by_reconstruction_gain(self):
        result = module.assess(summary(.4, 22., 1., .55), summary())
        self.assertFalse(result["eligible"])
        self.assertTrue(any("remaining vocal" in r for r in result["reasons"]))

    def test_muted_accompaniment_and_instrument_damage_are_rejected(self):
        self.assertFalse(module.assess(summary(.4, 21., 0., 0.), summary())["eligible"])
        self.assertFalse(module.assess(summary(.4, 19., .99, .4), summary())["eligible"])

    def test_missing_weak_domain_or_unpaired_tracks_fail_closed(self):
        candidate = summary(.4, 21., .99, .4)
        del candidate["musdb/weak_minus12"]
        self.assertFalse(module.assess(candidate, summary())["eligible"])
        candidate = summary(.4, 21., .99, .4)
        candidate["mir1k/native"]["metrics"]["vocal_error_snr_db"]["by_track"] = {"other": .4}
        self.assertFalse(module.assess(candidate, summary())["eligible"])

    def test_aggregate_song_weight_not_window_count(self):
        rows = [{"domain": "test", "track": "long", "metrics": {"score": 1.}} for _ in range(9)]
        rows.append({"domain": "test", "track": "short", "metrics": {"score": 3.}})
        self.assertEqual(module.aggregate(rows)["test"]["metrics"]["score"]["mean"], 2.)
        rows[0]["metrics"]["score"] = float("nan")
        with self.assertRaises(ValueError):
            module.aggregate(rows)

    def test_expansion_is_deterministic_remixes_are_counted_as_same_sources(self):
        length = (352-1)*256
        torch.manual_seed(99)
        vocal = torch.randn(2, length+10000)*.03
        accompaniment = torch.randn_like(vocal)*.02
        corpus = SimpleNamespace(
            train={"train": [{"track_id": "train"}]}, final_ids=["never-test"],
            teacher_meta={"source_ids": ["teacher"]},
            val={d: [{"track_id": d}] for d in ("musdb", "mir1k", "instrumental")},
            audio=lambda r: (accompaniment + vocal, vocal) if r["track_id"] != "instrumental" else
                            (accompaniment, torch.zeros_like(vocal)),
        )
        rows = module.build_suite(corpus, length, 96)
        manifest = module.suite_manifest(rows, length, 96)
        again = module.suite_manifest(module.build_suite(corpus, length, 96), length, 96)
        self.assertEqual(manifest, again)
        self.assertEqual((manifest["clips"], manifest["source_windows"], manifest["tracks"]), (15, 9, 3))
        for native, weak in zip(rows[:6:2], rows[1:6:2]):
            self.assertTrue(torch.allclose(native["x"]-native["v"], weak["x"]-weak["v"], atol=1e-7))
            self.assertTrue(torch.allclose(weak["v"], native["v"]*10**(-12/20), atol=1e-7))
        corpus.val["musdb"][0]["track_id"] = "never-test"
        with self.assertRaises(ValueError):
            module.build_suite(corpus, length, 96)

    def test_title_alias_and_teacher_overlap_guard(self):
        corpus = SimpleNamespace(train={"x": [{"track_id": "Skelpolu - Human Mistakes.stem"}]},
                                 val={"x": [{"track_id": "val"}]}, final_ids=["Skelpolu_HumanMistakes"],
                                 teacher_meta={"source_ids": []})
        with self.assertRaises(ValueError):
            module.guard_split(corpus)
        corpus.final_ids = []
        corpus.teacher_meta["source_ids"] = ["val.wav"]
        with self.assertRaises(ValueError):
            module.guard_split(corpus)

    def test_paired_batch_draws_are_waveform_identical_after_rng_reset(self):
        corpus = object.__new__(module.core.Corpus)
        corpus.train = {d: [{"track_id": d, "dataset": d}] for d in ("musdb", "mir1k", "onair", "instrumental")}
        torch.manual_seed(31)
        mix = torch.randn(2, 90100)*.01
        corpus.audio = lambda r: (mix, mix*.2)
        a = corpus.batch(torch.Generator().manual_seed(47), 89856)
        b = corpus.batch(torch.Generator().manual_seed(47), 89856)
        c = corpus.batch(torch.Generator().manual_seed(48), 89856)
        self.assertEqual(module.tensor_digest(*a), module.tensor_digest(*b))
        self.assertNotEqual(module.tensor_digest(*a), module.tensor_digest(*c))

    def test_song_bootstrap_does_not_treat_windows_as_independent_samples(self):
        scores = {m: 1. for m in ("vocal_error_snr_db", "vocal_si_sdr_db", "accompaniment_si_sdr_db",
                                 "accompaniment_error_snr_db", "residual_vocal_abs_gain", "joint_fit_accompaniment_gain")}
        rows = [{"dataset": "one", "track": "song", "vocals_present": True,
                 "models": {"baseline": scores, "candidate": {k: v+.5 for k, v in scores.items()}}}
                for _ in range(8)]
        result = regression.paired_intervals(rows, ["candidate"])["one"]["candidate"]["vocal_error_snr_db"]
        self.assertEqual((result["n_tracks"], result["mean"], result["bootstrap_95pct"]), (1, .5, None))
        for i, row in enumerate(rows):
            row["track"] = str(i)
        result = regression.paired_intervals(rows, ["candidate"])["one"]["candidate"]["vocal_error_snr_db"]
        self.assertEqual(result["bootstrap_95pct"], [.5, .5])

    def test_microbatch_preserves_effective_loss_and_gradient(self):
        torch.manual_seed(17)
        x = torch.randn(6, 2, 89856)*.01
        v = torch.randn_like(x)*.002
        wa = torch.from_numpy(module.core.t09.make_analysis_matrix())
        gs = torch.from_numpy(module.core.t09.make_synthesis_matrix())
        full = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
        accum = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
        accum.load_state_dict(full.state_dict())
        loss_full = control.backward_batch(full, x, v, wa, gs, "cpu", 96, 44, 1., 6)
        loss_accum = control.backward_batch(accum, x, v, wa, gs, "cpu", 96, 44, 1., 2)
        for a, b in zip(loss_full, loss_accum):
            self.assertAlmostEqual(a, b, places=6)
        for a, b in zip(full.parameters(), accum.parameters()):
            self.assertTrue(torch.allclose(a.grad, b.grad, atol=2e-6, rtol=2e-5))
        with self.assertRaises(ValueError):
            control.backward_batch(torch.nn.Sequential(full, torch.nn.BatchNorm2d(4)), x, v, wa, gs,
                                   "cpu", 96, 44, 1., 2)

    def test_bounded_long_inference_matches_continuous_pinned_graph(self):
        torch.manual_seed(88)
        net = module.core.t09.CausalSpectralUNet(bottleneck_blocks=2).eval()
        # Residual blocks start as identity; activate them to test their full
        # temporal receptive field, not just the shorter encoder/decoder path.
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.add_(torch.randn_like(parameter)*.02)
        net.frontend = ("linear", 1.)
        bands = torch.rand(2, 128, 517)*2
        whole = module.audit.masks(net, bands)
        bounded = regression.bounded_masks(net, bands)
        self.assertEqual(whole.shape, bounded.shape)
        self.assertTrue(torch.allclose(whole, bounded, atol=2e-6, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
