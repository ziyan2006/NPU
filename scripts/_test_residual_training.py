"""Synthetic tests before any objective-ablation training."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("residual_training", Path(__file__).with_name("110_train_residual_ablation.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(4)


class ResidualTrainingTests(unittest.TestCase):
    def test_same_composition_is_excluded_before_train_validation_split(self):
        records = [{"track_id": name, "split": "train"} for name in (
            "Skelpolu - Human Mistakes.stem", "Triviul - Angelsaint.stem", "Other - Song.stem")]
        eligible, excluded = module.exclude_final_compositions(records, module.CAMBRIDGE_HOLDOUTS)
        self.assertEqual([r["track_id"] for r in eligible], ["Other - Song.stem"])
        self.assertEqual(len(excluded), 2)
        self.assertEqual(module.composition_key("Skelpolu - Human Mistakes.stem.mp4"),
                         module.composition_key("Skelpolu_HumanMistakes"))
        self.assertNotEqual(module.composition_key("Triviul - Dorothy.stem"),
                            module.composition_key("Triviul_Angelsaint"))

    def test_split_never_uses_final_holdout(self):
        rows = [{"track_id": str(i), "split": "train" if i < 8 else "holdout"} for i in range(10)]
        train, validation = module.split_training_records(rows, 4, 2)
        self.assertEqual(len(train), 6)
        self.assertEqual(len(validation), 2)
        self.assertFalse({r['track_id'] for r in train} & {r['track_id'] for r in validation})
        self.assertNotIn("8", [r['track_id'] for r in train + validation])
        self.assertEqual((train, validation), module.split_training_records(rows, 4, 2))

    def test_identity_stft_waveform_and_gradient(self):
        torch.manual_seed(4)
        x = torch.randn(2, 2, (352 - 1) * 256) * 0.1
        spectrum = module.stft_batch(x)
        gs = torch.from_numpy(module.t09.make_synthesis_matrix())
        mask = torch.ones(2, 2, 128, 352)
        pv = module.product_vocal(spectrum, mask, gs, x.shape[-1], 0)
        self.assertLess(float((pv - x).abs().max()), 2e-6)
        # Exercise gradients away from clamp endpoints.
        mask = torch.full((2, 2, 128, 352), 0.5, requires_grad=True)
        pv = module.product_vocal(spectrum, mask, gs, x.shape[-1], 0)
        loss = module.waveform_loss(pv, x, x * 0.25, 96)
        loss.backward()
        self.assertTrue(torch.isfinite(mask.grad).all())
        self.assertGreater(float(mask.grad.abs().sum()), 0)

    def test_warmup_is_excluded(self):
        x = torch.ones(1, 2, 89856)
        pv = x.clone(); pv[..., :25000] = 100
        self.assertEqual(float(module.waveform_loss(pv, x, x, 96)), 0)
        pv[..., 27000] = 100
        self.assertGreater(float(module.waveform_loss(pv, x, x, 96)), 0)

    def test_protected_bands_have_no_product_gradient(self):
        x = torch.randn(1, 2, 89856)
        spectrum = module.stft_batch(x)
        gs = torch.from_numpy(module.t09.make_synthesis_matrix())
        mask = torch.full((1, 2, 128, 352), 0.5, requires_grad=True)
        pv = module.product_vocal(spectrum, mask, gs, x.shape[-1], 44)
        module.waveform_loss(pv, x, x * 0.25, 96).backward()
        self.assertEqual(float(mask.grad[:, :, :44].abs().max()), 0)
        self.assertGreater(float(mask.grad[:, :, 44:].abs().sum()), 0)

    def test_complement_band_objective_ignores_independent_head(self):
        mv, ma, target = torch.full((1, 2, 128, 8), .4), torch.full((1, 2, 128, 8), .2), torch.full((1, 2, 128, 8), .3)
        bands = torch.ones_like(mv)
        self.assertEqual(float(module.objective('complement', mv, ma, target, bands)),
                         float(module.objective('complement', mv, ma * 2, target, bands)))
        self.assertNotEqual(float(module.objective('control', mv, ma, target, bands)),
                            float(module.objective('control', mv, ma * 2, target, bands)))


if __name__ == "__main__":
    unittest.main()
