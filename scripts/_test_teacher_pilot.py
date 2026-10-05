"""Offline contracts for teacher waveform labels and preliminary DEV screening."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load("teacher_pilot_test_core", "131_run_teacher_pilot.py")
evaluation = load("teacher_pilot_test_evaluation", "132_evaluate_teacher_pilot.py")
listening = load("teacher_pilot_test_listening", "133_export_teacher_listening.py")
torch.set_num_threads(4)


class TeacherPilotTests(unittest.TestCase):
    def test_duplicate_and_old_holdout_mp3_rejected(self):
        pilot.guard_mp3(["Artist - New.mp3"], ["Artist - Old.mp3"])
        with self.assertRaises(ValueError):
            pilot.guard_mp3(["Artist - Old.mp3"], ["Artist_Old.wav"])
        with self.assertRaises(ValueError):
            pilot.guard_mp3(["Artist - New.mp3", "Artist_New.wav"], [])

    def test_tag_differences_require_full_pcm_identity_not_just_same_title(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"a.mp3", Path(folder)/"b.mp3"
            a.write_bytes(b"different tags A")
            b.write_bytes(b"different tags B")
            with patch.object(pilot.subprocess, "check_output", return_value=b"SHA256=same-audio"):
                path, audit = pilot.resolve_duplicate_mp3([b, a])
                self.assertEqual(path, a.resolve())
                self.assertEqual(audit["counted_as_compositions"], 1)
            with patch.object(pilot.subprocess, "check_output", side_effect=[b"SHA256=audio-A", b"SHA256=audio-B"]):
                with self.assertRaises(ValueError):
                    pilot.resolve_duplicate_mp3([a, b])

    def test_mel_overlap_identity_short_equal_and_long_lengths(self):
        for n in (100, 3000, 4096, 4097, 13001):
            torch.manual_seed(n)
            x = torch.randn(2, n) * .1
            result = pilot.mel_demix(torch.nn.Identity(), x, 4096, 2, "cpu")
            self.assertEqual(result.shape, x.shape)
            torch.testing.assert_close(result, x, atol=1e-7, rtol=1e-6)

    def test_overlap_never_silently_replaces_nan_or_loses_output_length(self):
        class Nonfinite(torch.nn.Module):
            def forward(self, x):
                return x * float("nan")
        class Short(torch.nn.Module):
            def forward(self, x):
                return x[..., :-1]
        for model in (Nonfinite(), Short()):
            with self.assertRaises(ValueError):
                pilot.mel_demix(model, torch.ones(2, 1000), 4096, 2, "cpu")
        with self.assertRaises(ValueError):
            pilot.mel_demix(torch.nn.Identity(), torch.ones(2, 1000), 4096, 3, "cpu")

    def test_residual_pair_does_not_clip_or_normalize_stems(self):
        x = torch.tensor([[.4, -.3], [.3, -.1]], dtype=torch.float32)
        vocal = torch.tensor([[1.8, -.5], [1.3, -.7]], dtype=torch.float32)
        a, error = pilot.residual_pair(x, vocal)
        torch.testing.assert_close(vocal + a, x)
        self.assertGreater(float(vocal.abs().max()), 1.)
        self.assertGreater(float(a.abs().max()), 1.)
        self.assertLess(error, 1e-6)
        with self.assertRaises(ValueError):
            pilot.residual_pair(x, vocal[:, :-1])

    def test_saved_float_waves_preserve_samples_and_no_overwrite(self):
        x = torch.tensor([[1.8, -.3], [.3, -.1]], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vocal.wav"
            digest = pilot.write_wave_new(path, x)
            self.assertTrue(torch.equal(pilot.read_wave(path), x))
            self.assertEqual(pilot.acq.sha256(path), digest)
            with self.assertRaises(ValueError):
                pilot.write_wave_new(path, x)

    def summary(self):
        domains = {}
        for domain in ("musdb/native", "musdb/weak_minus12"):
            domains[domain] = {}
            for teacher, gain in zip(pilot.TEACHERS, (0., 1.)):
                vals = {"vocal_error_snr_db": 5.+gain, "joint_fit_accompaniment_gain": .99,
                        "residual_vocal_abs_gain": .5-.1*gain}
                domains[domain][teacher] = {k: {"mean": v, "by_song": {"song": v}} for k, v in vals.items()}
        domains["instrumental/native"] = {teacher: {"removed_energy_relative_mix_db": {"mean": -30., "by_song": {"neg": -30.}}}
                                            for teacher in pilot.TEACHERS}
        return domains

    def policy(self):
        return {"minimum_mean_vocal_error_snr_gain_db": .25, "maximum_any_vocal_domain_regression_db": .25,
                "maximum_mean_accompaniment_gain_drop": .03, "maximum_residual_vocal_increase_db_per_domain": .25,
                "maximum_instrumental_removed_energy_increase_db": .5}

    def test_screening_pass_does_not_authorize_training_or_batch_relabel(self):
        result = evaluation.assess(self.summary(), self.policy())
        self.assertTrue(result["screening_passed"])
        self.assertFalse(result["whole_library_relabel_authorized"])
        self.assertFalse(result["student_training_or_promotion_authorized"])

    def test_instrumental_false_removal_and_weak_domain_regression_block_teacher(self):
        data = self.summary()
        data["instrumental/native"][pilot.TEACHERS[1]]["removed_energy_relative_mix_db"]["mean"] = -20.
        self.assertFalse(evaluation.assess(data, self.policy())["screening_passed"])
        data = self.summary()
        data["musdb/weak_minus12"][pilot.TEACHERS[1]]["vocal_error_snr_db"]["mean"] = 4.
        self.assertFalse(evaluation.assess(data, self.policy())["screening_passed"])

    def test_missing_domain_or_unpaired_tracks_cannot_pass(self):
        data = self.summary()
        del data["instrumental/native"]
        self.assertFalse(evaluation.assess(data, self.policy())["screening_passed"])
        data = self.summary()
        data["musdb/native"][pilot.TEACHERS[1]]["vocal_error_snr_db"]["by_song"] = {"different": 6.}
        self.assertFalse(evaluation.assess(data, self.policy())["screening_passed"])

    def test_mp3_diagnostics_are_excluded_from_reference_aggregation(self):
        rows = [{"role": "pseudo_label_pilot_only", "domain": "mp3/no_reference", "track_id": "unlabelled",
                 "teachers": {teacher: {"removed_energy_relative_mix_db_diagnostic_only": 5.} for teacher in pilot.TEACHERS}}]
        self.assertEqual(evaluation.summarize(rows), {})

    def test_playback_uses_common_gain_without_amplifying_quiet_stems(self):
        waves = {"mix": torch.ones(2, 32)*.4, "loud_vocal": torch.ones(2, 32)*1.9,
                 "quiet_vocal": torch.ones(2, 32)*.01}
        gain = listening.safe_gain(waves)
        self.assertAlmostEqual(gain, .5, places=6)
        self.assertLess(float((waves["quiet_vocal"]*gain).abs().max()), .01)
        self.assertEqual(listening.safe_gain({"silent": torch.zeros(2, 32)}), 1.)
        with self.assertRaises(ValueError):
            listening.safe_gain({"a": torch.ones(2, 4), "b": torch.ones(2, 5)})

    def test_playback_pcm_readback_overwrite_and_directory_scope(self):
        torch.manual_seed(1)
        wave = torch.randn(2, 4096)
        original = wave.clone()
        gain = listening.safe_gain({"source": wave})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"preview.wav"
            listening.write_playback_new(path, wave, gain)
            self.assertTrue(torch.equal(wave, original))
            self.assertLessEqual(float(pilot.read_wave(path).abs().max()), .950001)
            with self.assertRaises(ValueError):
                listening.write_playback_new(path, wave, gain)
            with self.assertRaises(ValueError):
                listening.guard_output(Path(folder))
        with self.assertRaises(ValueError):
            listening.guard_output(listening.PREVIEW_ROOT)
        listening.guard_output(listening.PREVIEW_ROOT/"safe-child")


if __name__ == "__main__":
    unittest.main()
