"""New238 helper tests only; no archived suites, real model/audio/PT or CUDA."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("ema238_listen", Path(__file__).with_name("238_export_ema_listening.py"))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


class ListeningHelpers(unittest.TestCase):
    def waves(self):
        return {k: torch.full((2, 32), .2, dtype=torch.float32) for k in e.LABELS}

    def test_common_gain_preserves_unclipped(self):
        before = self.waves(); after, gain = e.common_playback(before)
        self.assertEqual(gain, 1.)
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))

    def test_one_variant_peak_controls_all(self):
        before = self.waves(); before['ema5000_backing'].fill_(2)
        after, gain = e.common_playback(before)
        self.assertEqual(gain, .475)
        self.assertTrue(all(torch.equal(after[k], before[k] * gain) for k in before))
        self.assertLessEqual(float(after['ema5000_backing'].max()), .950001)

    def test_reject_missing_variant(self):
        waves = self.waves(); waves.pop('raw5000_backing')
        with self.assertRaises(ValueError): e.common_playback(waves)

    def test_reject_nan(self):
        waves = self.waves(); waves['mix'][0, 0] = float('nan')
        with self.assertRaises(ValueError): e.common_playback(waves)

    def test_reject_wrong_shape(self):
        waves = self.waves(); waves['mix'] = torch.zeros(1, 32)
        with self.assertRaises(ValueError): e.common_playback(waves)

    def test_reject_fp64(self):
        waves = self.waves(); waves['mix'] = waves['mix'].double()
        with self.assertRaises(ValueError): e.common_playback(waves)

    def test_reject_unmatched_duration(self):
        waves = self.waves(); waves['raw5000_vocal'] = torch.zeros(2, 33)
        with self.assertRaises(ValueError): e.common_playback(waves)

    def test_no_input_mutation(self):
        waves = self.waves(); copies = {k: v.clone() for k,v in waves.items()}
        after, _ = e.common_playback(waves); after['mix'].zero_()
        self.assertTrue(all(torch.equal(waves[k], copies[k]) for k in waves))

    def test_refuse_foreign_output(self):
        with self.assertRaises(ValueError): e.fresh_output(e.ROOT / 'results/not_this_export')

    def test_refuse_existing_output(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(e, 'OUT', Path(folder)):
            with self.assertRaises(ValueError): e.fresh_output(Path(folder))

    def test_page_has_all_controls_and_escaped_name(self):
        row = {'index':1,'track_id':'a&<b>','seconds':20.,'domain':'musdb',
               'audio':{k:k+'.wav' for k in e.LABELS}}
        rendered=e.page([row])
        self.assertEqual(rendered.count('<audio controls'), 11)
        self.assertIn('a&amp;&lt;b&gt;',rendered)
        self.assertIn('a.pause()',rendered)

    def test_ast_rejects_missing_original_function(self):
        with self.assertRaises(ValueError): e.original('scripts/09_target_model.py', ('missing_function',), {})


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
