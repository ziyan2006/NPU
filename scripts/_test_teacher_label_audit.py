"""CPU-only technical metrics, fail-closed formats and review boundaries."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location("quality_audit_test", Path(__file__).with_name("138_audit_teacher_labels.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_metrics_include_float_overshoots_without_clipping(self):
        with tempfile.TemporaryDirectory() as folder:
            v, a = Path(folder)/"v.wav", Path(folder)/"a.wav"
            n = 88210
            wave = np.full((n, 2), 1.25, dtype=np.float32)
            sf.write(v, wave, 44100, subtype="FLOAT")
            sf.write(a, -.75*wave, 44100, subtype="FLOAT")
            result, flags = audit.metrics(v, a, n)
            self.assertEqual(result["vocals"]["peak"], 1.25)
            self.assertEqual(result["vocals"]["fraction_abs_above_one"], 1)
            self.assertAlmostEqual(result["reconstructed_mix"]["rms"], .3125)
            self.assertTrue(any("not proof of clipping" in value for value in flags))

    def test_silence_on_active_mix_is_hint_not_quality_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            v, a = Path(folder)/"v.wav", Path(folder)/"a.wav"
            wave = np.zeros((44100, 2), dtype=np.float32)
            sf.write(v, wave, 44100, subtype="FLOAT")
            sf.write(a, wave+.1, 44100, subtype="FLOAT")
            result, flags = audit.metrics(v, a, len(wave))
            self.assertEqual(result["quiet_vocal_active_windows"], 1)
            self.assertTrue(any("may be instrumental" in value for value in flags))

    def test_wrong_subtype_rate_channels_length_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            v, a = Path(folder)/"v.wav", Path(folder)/"a.wav"
            wave = np.zeros((1000, 2), dtype=np.float32)
            sf.write(a, wave, 44100, subtype="FLOAT")
            for rate, x, subtype in ((48000, wave, "FLOAT"), (44100, wave[:, 0], "FLOAT"),
                                      (44100, wave, "PCM_16"), (44100, wave[:500], "FLOAT")):
                sf.write(v, x, rate, subtype=subtype)
                with self.assertRaises(ValueError):
                    audit.metrics(v, a, 1000)

    def test_nonfinite_stats_rejected(self):
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                audit.Statistics().add(np.full((10, 2), value))

    def test_stream_boundary_jump_counted_and_metrics_partition_independent(self):
        x = np.zeros((200, 2), dtype=np.float32)
        x[100:] = .5
        whole, split = audit.Statistics(), audit.Statistics()
        whole.add(x)
        split.add(x[:100])
        split.add(x[100:])
        self.assertEqual(whole.report(), split.report())
        self.assertEqual(split.report()["max_adjacent_jump"], .5)

    def test_listening_windows_are_sample_aligned_and_never_padded(self):
        for samples in (100, 1000000):
            for item in audit.listening_windows(samples):
                self.assertGreaterEqual(item["start_sample"], 0)
                self.assertLessEqual(item["start_sample"]+item["samples"], samples)
                self.assertIsInstance(item["start_sample"], int)


if __name__ == "__main__":
    unittest.main()
