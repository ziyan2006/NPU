"""CPU-only matched teacher label selection, identity, preservation and binding."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("matched_label_test", Path(__file__).with_name("139_generate_paired_htdemucs.py"))
matched = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matched)


class MatchedTests(unittest.TestCase):
    def test_24_song_selection_matches_pre_scoring_snapshot(self):
        protocol = json.loads(matched.data.PROTOCOL.read_text())
        records = [{"song_id": f"song_{i:04d}", "role": "pseudo_label_train_candidate"} for i in range(275)]
        ids = matched.data.choose_pairs(records, 24, 20261002)
        selection = {"paired_teacher_24_song_ids": ids, "training_authorized": False}
        self.assertEqual([r["song_id"] for r in matched.require_fixed_selection({"records": records}, selection, protocol)], ids)
        with self.assertRaises(ValueError):
            matched.require_fixed_selection({"records": records}, selection | {"paired_teacher_24_song_ids": ids[:-1]}, protocol)
        with self.assertRaises(ValueError):
            matched.require_fixed_selection({"records": records}, selection | {"training_authorized": True}, protocol)

    def test_changed_code_weight_protocol_binding_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bound"
            path.write_bytes(b"original")
            bindings = {str(path): matched.pilot.acq.sha256(path)}
            matched.verify_bindings(bindings)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                matched.verify_bindings(bindings)

    def fixture(self, root):
        x = torch.randn(2, 1024)*.1
        v = x*.3
        a, _ = matched.pilot.residual_pair(x, v)
        row = {"song_id": "song_0006", "source": {"path": "original.mp3", "samples": 1024,
                "pcm_sha256": matched.pilot.wave_digest(x)}}
        bound = {"plan_sha256": "bound"}
        files = {"vocals.wav": matched.pilot.write_wave_new(root/"vocals.wav", v),
                 "accompaniment.wav": matched.pilot.write_wave_new(root/"accompaniment.wav", a)}
        entry = {"binding": bound, "song_id": row["song_id"], "pcm_sha256": row["source"]["pcm_sha256"],
                 "files_sha256": files, "training_eligible": False, "per_song_listening_review": "pending"}
        matched.pilot.acq.write_new_json(root/"entry.json", matched.pilot.acq.seal(entry))
        return x, row, bound

    def test_same_float_input_and_exact_residual_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            x, row, bound = self.fixture(root)
            self.assertFalse(matched.verify_song(root, row, bound, x)["training_eligible"])
            with self.assertRaises(ValueError):
                matched.verify_song(root, row, bound, x*.9999)

    def test_wrong_binding_changed_label_or_length_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            x, row, bound = self.fixture(root)
            with self.assertRaises(ValueError):
                matched.verify_song(root, row, {"plan_sha256": "other"}, x)
            with (root/"vocals.wav").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                matched.verify_song(root, row, bound, x)

    def test_partial_directory_is_preserved_not_reinferred(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"song_0006").mkdir()
            with self.assertRaisesRegex(ValueError, "Partial"):
                matched.pending_rows(root, {"records": [{"song_id": "song_0006"}]}, {})
            self.assertTrue((root/"song_0006").exists())

    def test_existing_completion_is_verified_and_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"song_0006").mkdir()
            (root/"song_0006/entry.json").write_text("fixture")
            rows = [{"song_id": "song_0006"}, {"song_id": "song_0008"}]
            with patch.object(matched, "verify_song", return_value={}) as verify:
                pending, completed = matched.pending_rows(root, {"records": rows}, {})
                self.assertEqual(completed, 1)
                self.assertEqual(pending, rows[1:])
                verify.assert_called_once()

    def test_storage_is_only_24_subset_with_reserve(self):
        needed = matched.math_storage(5417.2174376)
        self.assertLess(needed, 4205000000)
        with self.assertRaises(ValueError):
            matched.bulk.check_storage(needed+12*2**30-1, needed, 12*2**30)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
