"""Contracts for the new data lock and TRAIN-only learning-mechanics gate."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lock = load("training_lock_test", "125_lock_training_data.py")
fit = load("small_fit_test", "126_verify_training_baseline.py")
audit = load("small_fit_reload_test", "127_check_training_baseline.py")
torch.set_num_threads(4)


class TrainingBaselineTests(unittest.TestCase):
    def test_data_lock_rejects_role_overlap_and_fake_blind(self):
        doc = {"schema": 1, "frozen_sha256": lock.FROZEN_SHA,
               "records": [{"domain": "musdb", "track_id": "artist-song", "role": "train"},
                           {"domain": "musdb", "track_id": "artist-val", "role": "development"}],
               "known_regression_ids": ["old-test"], "blind_ids": [],
               "blind_status": "missing", "files": {}}
        lock.validate_roles(doc)
        bad = copy.deepcopy(doc)
        bad["records"][1]["track_id"] = "Artist Song.stem.mp4"
        with self.assertRaises(ValueError):
            lock.validate_roles(bad)
        bad = copy.deepcopy(doc)
        bad["blind_ids"] = ["old-test.wav"]
        with self.assertRaises(ValueError):
            lock.validate_roles(bad)
        bad = copy.deepcopy(doc)
        bad["blind_status"] = "ready"
        with self.assertRaises(ValueError):
            lock.validate_roles(bad)

    def test_lock_hash_detects_metadata_tampering(self):
        doc = {"seed": 17, "tracks": ["a", "b"]}
        doc["content_sha256"] = lock.document_digest(doc)
        self.assertEqual(lock.document_digest(doc), doc["content_sha256"])
        doc["seed"] = 18
        self.assertNotEqual(lock.document_digest(doc), doc["content_sha256"])

    def test_reconstruction_loss_silence_identity_and_scale(self):
        torch.manual_seed(31)
        x = torch.randn(2, 2, 57088)*.04
        spectrum = fit.core.stft_batch(x)
        gs = torch.from_numpy(fit.core.t09.make_synthesis_matrix())
        identity = torch.ones(2, 2, 128, 224)
        loss, parts, pv = fit.reconstruction_loss(identity, spectrum, x, x, gs, 96, 0)
        self.assertLess(float(loss), 2e-6)
        self.assertTrue(torch.allclose(pv, x, atol=2e-6))
        zero = torch.zeros_like(x)
        loss, parts, _ = fit.reconstruction_loss(identity, fit.core.stft_batch(zero), zero, zero, gs, 96, 44)
        self.assertEqual(float(loss), 0.)
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))

    def test_reconstruction_supervises_empty_input_bands_but_not_protected_outputs(self):
        torch.manual_seed(42)
        x = torch.randn(1, 2, 57088)*.04
        spectrum = fit.core.stft_batch(x)
        wa = torch.from_numpy(fit.core.t09.make_analysis_matrix())
        gs = torch.from_numpy(fit.core.t09.make_synthesis_matrix())
        dead = (wa.sum(0) == 0) & (gs.sum(0) > 0)
        dead[:44] = False
        self.assertEqual(torch.where(dead)[0].tolist(), [44, 46, 50, 51, 53, 55, 57, 60, 64])
        mask = torch.full((1, 2, 128, 224), .5, requires_grad=True)
        loss, _, _ = fit.reconstruction_loss(mask, spectrum, x, x*.2, gs, 96, 44)
        loss.backward()
        self.assertEqual(float(mask.grad[:, :, :44].abs().max()), 0.)
        self.assertTrue((mask.grad[:, :, dead].abs().sum(dim=(0, 1, 3)) > 0).all())
        self.assertEqual(float(mask.grad[..., :94].abs().max()), 0.)

    def test_microbatch_loss_gradients_and_adam_update_match(self):
        torch.manual_seed(77)
        x = torch.randn(4, 2, 57088)*.02
        v = x*.3
        wa = torch.from_numpy(fit.core.t09.make_analysis_matrix())
        gs = torch.from_numpy(fit.core.t09.make_synthesis_matrix())
        for arm in ("band_wave_control", "reconstruction_l1"):
            a = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
            b = copy.deepcopy(a)
            oa, ob = torch.optim.Adam(a.parameters(), lr=1e-4), torch.optim.Adam(b.parameters(), lr=1e-4)
            la = fit.backward_batch(a, x, v, wa, gs, "cpu", 96, 44, arm, 4)
            lb = fit.backward_batch(b, x, v, wa, gs, "cpu", 96, 44, arm, 1)
            for key in la:
                self.assertAlmostEqual(la[key], lb[key], places=5)
            for pa, pb in zip(a.parameters(), b.parameters()):
                self.assertTrue(torch.allclose(pa.grad, pb.grad, atol=2e-6, rtol=2e-5))
            oa.step(); ob.step()
            for pa, pb in zip(a.parameters(), b.parameters()):
                self.assertTrue(torch.allclose(pa, pb, atol=1e-7, rtol=2e-5))

    def test_cohort_uses_train_only_and_deterministic_fixed_remixes(self):
        n = 62000
        t = torch.arange(n)/44100
        a = torch.sin(t*2*torch.pi*731).repeat(2, 1)*.04
        v = torch.sin(t*2*torch.pi*423).repeat(2, 1)*.025
        records = [{"track_id": s, "dataset": "musdb"} for s in ("a", "b", "c")]
        corpus = SimpleNamespace(train={"musdb": records}, val={"musdb": [{"track_id": "never-val"}]},
                                 audio=lambda r: (a+v, v))
        config = {"source_songs": 2, "variants_db": [0, -12], "crop_frames": 128, "warmup_frames": 96}
        rows = fit.build_cohort(corpus, config)
        self.assertEqual([r["track"] for r in rows], ["a", "a", "b", "b"])
        self.assertEqual([r["samples_sha256"] for r in rows],
                         [r["samples_sha256"] for r in fit.build_cohort(corpus, config)])
        self.assertTrue(torch.allclose(rows[0]["x"]-rows[0]["v"], rows[1]["x"]-rows[1]["v"], atol=1e-7))

    def test_learning_gate_cannot_pass_from_native_only_or_fit_loss_only(self):
        baseline = {"loss": 1., "clips": [
            {"vocal_gain_db": db, "vocal_error_snr_db": 2.} for db in (0, -12)]}
        good = {"loss": .8, "clips": [
            {"vocal_gain_db": db, "vocal_error_snr_db": 3.} for db in (0, -12)]}
        policy = {"minimum_loss_reduction_fraction": .1, "minimum_native_error_snr_gain_db": .5,
                  "minimum_weak_error_snr_gain_db": .5, "maximum_any_clip_regression_db": .25}
        self.assertTrue(fit.assess_fit(good, baseline, policy)["passed"])
        good["clips"][1]["vocal_error_snr_db"] = 1.
        self.assertFalse(fit.assess_fit(good, baseline, policy)["passed"])
        good["clips"][1]["vocal_error_snr_db"] = 3.
        good["loss"] = .95
        self.assertFalse(fit.assess_fit(good, baseline, policy)["passed"])
        with self.assertRaises(ValueError):
            fit.assess_fit({"loss": .7, "clips": []}, baseline, policy)

    def test_resume_preserves_optimizer_and_rng(self):
        torch.manual_seed(44)
        a = torch.nn.Linear(3, 2)
        opt = torch.optim.Adam(a.parameters(), lr=.001)
        x = torch.randn(4, 3)
        a(x).square().mean().backward(); opt.step(); opt.zero_grad()
        state = fit.capture_state(a, opt, 1, "bound", "test")
        expected_draw = torch.rand(5)
        b = copy.deepcopy(a)
        new_opt = torch.optim.Adam(b.parameters(), lr=.001)
        fit.restore_state(state, b, new_opt, "bound", "test")
        self.assertTrue(torch.equal(torch.rand(5), expected_draw))
        a(x).square().mean().backward(); opt.step()
        b(x).square().mean().backward(); new_opt.step()
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))
        with self.assertRaises(ValueError):
            fit.restore_state(state, b, new_opt, "different-protocol", "test")

    def test_invalid_protocol_cannot_change_protected_graph_or_drop_weak_views(self):
        import json
        protocol = json.loads((Path(__file__).resolve().parents[1] /
                               "docs/model_training_protocol_v1.json").read_text(encoding="utf-8"))
        fit.validate_protocol(protocol)
        bad = copy.deepcopy(protocol)
        bad["graph"]["lf_kill_bands"] = 10
        with self.assertRaises(ValueError):
            fit.validate_protocol(bad)
        bad = copy.deepcopy(protocol)
        bad["small_fit"]["variants_db"] = [0]
        with self.assertRaises(ValueError):
            fit.validate_protocol(bad)

    def test_reload_audit_rejects_changed_waveforms_and_changed_scores(self):
        scores = {"loss": .3, "clips": [{"track": "train", "samples_sha256": "locked",
                  "vocal_gain_db": 0, "vocal_error_snr_db": 3., "residual_vocal_abs_gain": .2}]}
        self.assertEqual(audit.compare_scores(scores, copy.deepcopy(scores)), 0.)
        bad = copy.deepcopy(scores)
        bad["clips"][0]["samples_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            audit.compare_scores(scores, bad)
        bad = copy.deepcopy(scores)
        bad["clips"][0]["vocal_error_snr_db"] += .01
        with self.assertRaises(ValueError):
            audit.compare_scores(scores, bad)


if __name__ == "__main__":
    unittest.main()
