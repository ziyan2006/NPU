"""Contracts for full-song batch isolation, storage and verified resume."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("teacher_library_test_core", Path(__file__).with_name("134_generate_teacher_library.py"))
bulk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bulk)
torch.set_num_threads(4)


class LibraryTests(unittest.TestCase):
    def source(self, name, pcm):
        return {"path": name, "pcm_sha256": pcm, "samples": 4, "seconds": 4/bulk.pilot.SR}

    def test_reserved_variants_and_chills_excluded_before_decode(self):
        paths = [Path("Artist - Old (Extended Mix).mp3"), Path("Artist - New.mp3"), Path("Another - Chills.mp3")]
        good, bad = bulk.classify(paths, ["Artist - Old.mp3"])
        self.assertEqual(good, [paths[1]])
        self.assertEqual(bad, [paths[0], paths[2]])
        self.assertTrue(bulk.composition_keys("ARTIST_Old.wav") & bulk.composition_keys("Artist - Old.mp3"))

    def test_tag_or_name_differences_with_identical_full_pcm_count_once(self):
        rows = [self.source("a/Artist - New.mp3", "same"), self.source("b/Artist - New.mp3", "same"),
                self.source("c/Different Filename.mp3", "same")]
        good, bad = bulk.group_sources(rows)
        self.assertEqual(len(good), 1)
        self.assertEqual(len(good[0]["aliases"]), 3)
        self.assertFalse(bad)

    def test_different_pcm_for_same_composition_quarantined_not_extra_songs(self):
        rows = [self.source("Artist - Old.mp3", "one"), self.source("Artist - Old (Extended Mix).mp3", "two"),
                self.source("Other - New.mp3", "three")]
        good, bad = bulk.group_sources(rows)
        self.assertEqual(len(good), 1)
        self.assertEqual(len(bad), 1)
        self.assertEqual(len(bad[0]["sources"]), 2)

    def test_transitive_pcm_and_title_aliases_share_one_quarantine(self):
        rows = [self.source("a/A - One.mp3", "x"), self.source("b/B - Two.mp3", "y"),
                self.source("c/A - One.mp3", "y")]
        good, bad = bulk.group_sources(rows)
        self.assertFalse(good)
        self.assertEqual(len(bad), 1)
        self.assertEqual(len(bad[0]["sources"]), 3)

    def test_storage_reserves_space_for_old_data_and_never_saves_mix(self):
        settings = {"saved_stems": 2, "estimate_safety_factor": 1.1}
        expected = bulk.storage_estimate(3600, settings)
        self.assertEqual(expected, 2794176000)
        bulk.check_storage(expected+100, expected, 100)
        with self.assertRaises(ValueError):
            bulk.check_storage(expected+99, expected, 100)

    def test_output_cannot_escape_or_replace_results_root(self):
        bulk.guard_output(bulk.ROOT/"results"/"new-batch")
        for path in (bulk.ROOT/"results", bulk.ROOT, bulk.ROOT/"results"/".."/"models"):
            with self.assertRaises(ValueError):
                bulk.guard_output(path)

    def test_existing_prepare_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(bulk, "guard_output"):
                with self.assertRaisesRegex(ValueError, "New batch directory"):
                    bulk.prepare(Path(directory), Path("unused"), Path("unused"), Path("unused"), Path("unused"))

    def test_worker_exclusive_lock_reusable_after_release(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with bulk.worker_lock(out):
                with self.assertRaises((ValueError, OSError)):
                    with bulk.worker_lock(out):
                        self.fail("Second worker acquired active batch")
            with bulk.worker_lock(out):
                pass

    def test_state_publication_not_a_student_or_deployment_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            bulk.state(out, "running", 10, 2, "song_0002")
            bulk.state(out, "failed", 10, 2, "song_0002", "failure preserved")
            doc = bulk.json.loads((out/"run_status.json").read_text())
            self.assertEqual(doc["verified_completed"], 2)
            self.assertEqual(doc["status"], "failed")
            self.assertFalse(doc["student_training"])
            self.assertFalse(doc["student_promotion"])
            self.assertFalse(list(out.glob("*.tmp")))

    def test_saved_song_is_not_a_normalized_or_misaligned_label(self):
        mix = torch.tensor([[.4, -.3, .2, .1], [.3, -.1, .2, .1]])
        vocal = mix*3
        backing, error = bulk.pilot.residual_pair(mix, vocal)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            row = {"song_id": "song_0001", "source": {"samples": 4, "pcm_sha256": bulk.pilot.wave_digest(mix)}}
            bound = {"plan": "example"}
            hashes = {"vocals.wav": bulk.pilot.write_wave_new(folder/"vocals.wav", vocal),
                      "accompaniment.wav": bulk.pilot.write_wave_new(folder/"accompaniment.wav", backing)}
            entry = {"song_id": row["song_id"], "pcm_sha256": row["source"]["pcm_sha256"], "binding": bound,
                     "files_sha256": hashes}
            bulk.pilot.acq.write_new_json(folder/"entry.json", bulk.pilot.acq.seal(entry))
            bulk.verify_song(folder, row, bound, mix)
            with self.assertRaises(ValueError):
                bulk.verify_song(folder, row, {"plan": "changed"}, mix)
            with self.assertRaises(ValueError):
                bulk.verify_song(folder, row, bound, mix*.9)
            with (folder/"vocals.wav").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                bulk.verify_song(folder, row, bound, mix)


if __name__ == "__main__":
    unittest.main()
