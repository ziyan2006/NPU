"""Quarantine acquisition contracts: no models, no network and no private audio."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

import numpy as np


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acq = load("cambridge_acquisition_test", "128_acquire_cambridge_candidates.py")
qc = load("cambridge_source_qc_test", "129_audit_cambridge_candidates.py")


class CambridgeCandidateTests(unittest.TestCase):
    def plan(self):
        return {"schema": 1, "metadata_commit": "a" * 40, "maximum_download_bytes": 1000,
                "records": [{"track_id": "Artist_NewSong", "artist": "Artist", "title": "New Song",
                             "role": "train_candidate", "expected_bytes": 100,
                             "url": f"http://{acq.HOST}/MTK001/Artist_NewSong.zip"}]}

    def test_canonical_aliases_block_misleading_archive_codes(self):
        plan = self.plan()
        row = plan["records"][0]
        row.update(track_id="MR0907_Punkdisco", artist="Punkdisco", title="Oral Hygiene",
                   url=f"http://{acq.HOST}/MTK002/MR0907_Punkdisco.zip")
        with self.assertRaisesRegex(ValueError, "observed"):
            acq.check_roles(plan, ["Punkdisco - Oral Hygiene.stem.mp4"])
        self.assertEqual(acq.song_key("Triviul - Alright?.stem.mp4"), acq.song_key("Triviul_Alright"))

    def test_new_artist_cannot_cross_roles_or_duplicate_titles(self):
        plan = self.plan()
        other = copy.deepcopy(plan["records"][0])
        other.update(track_id="Artist_Second", title="Second", role="development_candidate",
                     url=f"http://{acq.HOST}/MTK001/Artist_Second.zip")
        plan["records"].append(other)
        with self.assertRaisesRegex(ValueError, "artist"):
            acq.check_roles(plan, [])
        other["role"] = "train_candidate"
        acq.check_roles(plan, [])
        other["title"] = "New Song"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            acq.check_roles(plan, [])

    def test_unknown_roles_hosts_and_budget_are_rejected(self):
        acq.check_roles(self.plan(), [])
        for key, value in (("role", "blind_ready"), ("url", "http://evil.example/Artist_NewSong.zip"),
                           ("expected_bytes", 1001), ("track_id", "../Artist_NewSong")):
            plan = self.plan()
            plan["records"][0][key] = value
            with self.assertRaises(ValueError):
                acq.check_roles(plan, [])

    def archive(self, names):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            for name in names:
                info = zipfile.ZipInfo(name)
                # Windows ZipInfo's constructor normalizes backslashes. Keep
                # the adversarial raw ZIP header instead of testing a safe,
                # already-normalized fixture.
                info.filename = info.orig_filename = name
                archive.writestr(info, b"example")
        buf.seek(0)
        return zipfile.ZipFile(buf)

    def test_zip_traversal_absolute_and_windows_paths_are_rejected(self):
        for name in ("../escaped.wav", "/Artist/file.wav", "Artist/../escaped.wav",
                     "Artist\\file.wav", "Artist/C:escaped.wav", "Other/file.wav",
                     "Artist/NUL.wav", "Artist/file?.wav", "Artist/nested/file.wav", "Artist/code.py"):
            with self.archive([name]) as archive, self.assertRaises(ValueError):
                acq.safe_members(archive, "Artist", 1000)

    def test_zip_case_collisions_symlinks_and_size_caps(self):
        with self.archive(["Artist/A.wav", "Artist/a.wav"]) as archive, self.assertRaises(ValueError):
            acq.safe_members(archive, "Artist", 1000)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            info = zipfile.ZipInfo("Artist/link.wav")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "outside")
        buf.seek(0)
        with zipfile.ZipFile(buf) as archive, self.assertRaises(ValueError):
            acq.safe_members(archive, "Artist", 1000)
        with self.archive(["Artist/file.wav"]) as archive, self.assertRaises(ValueError):
            acq.safe_members(archive, "Artist", 1)

    def test_bundled_terms_required_and_local_seals_detect_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "archive.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("Artist/Readme.txt", "Provided for educational purposes only. No commercial purpose permitted.")
                archive.writestr("Artist/voice.wav", b"mock")
            infos, terms = acq.validate_archive(path, {"track_id": "Artist"}, 1000)
            self.assertEqual(len(infos), 2)
            self.assertIn("educational", terms)
            doc_path = Path(folder) / "sealed.json"
            acq.write_new_json(doc_path, acq.seal({"role": "acceptance_candidate"}))
            self.assertEqual(acq.read_sealed(doc_path)["role"], "acceptance_candidate")
            with self.assertRaises(FileExistsError):
                acq.write_new_json(doc_path, {})
            changed = json.loads(doc_path.read_text())
            changed["role"] = "train_candidate"
            doc_path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                acq.read_sealed(doc_path)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("Artist/Readme.txt", "Unreviewed terms.")
            with self.assertRaisesRegex(ValueError, "restrictions"):
                acq.validate_archive(path, {"track_id": "Artist"}, 1000)

    def test_output_stays_inside_private_datasets(self):
        self.assertEqual(acq.guard_output(acq.DEFAULT_OUT), acq.DEFAULT_OUT.resolve())
        for path in (acq.ROOT, acq.ROOT / "data/datasets", acq.ROOT / "models/new"):
            with self.assertRaises(ValueError):
                acq.guard_output(path)

    def test_named_vocal_effects_and_doubles_cannot_enter_accompaniment(self):
        names = ["01_Loop.wav", "02_LeadVox.wav", "03_LeadVoxDT.wav", "04_SFXVox.wav"]
        with self.assertRaisesRegex(ValueError, "silently"):
            qc.partition_stems(names, {"vocal_basenames": ["02_LeadVox.wav"]})
        v, a = qc.partition_stems(names, {"vocal_basenames": names[1:]})
        self.assertEqual(v, names[1:])
        self.assertEqual(a, names[:1])
        for vocal in ([], names, ["missing.wav"], [names[1], names[1]]):
            with self.assertRaises(ValueError):
                qc.partition_stems(names, {"vocal_basenames": vocal})

    def test_shared_gain_zero_extension_and_channel_handling(self):
        instrumental = np.array([[1., .5], [.5, .25], [.25, .125]], np.float32)
        vocal = np.array([[1.], [.2]], np.float32)
        mix, v, gain = qc.sum_reference({"drum.wav": instrumental, "vox.wav": vocal}, {"vox.wav"})
        self.assertEqual(mix.shape, (3, 2))
        self.assertAlmostEqual(gain, .475)
        np.testing.assert_allclose(mix - v, instrumental * gain, atol=1e-7)
        np.testing.assert_allclose(v[:2], np.repeat(vocal, 2, axis=1) * gain)
        np.testing.assert_array_equal(v[2], [0, 0])
        self.assertLessEqual(float(np.abs(mix).max()), .950001)

    def test_nonfinite_multichannel_and_silent_positives_rejected(self):
        for raw in (np.zeros((2, 3)), np.array([[np.nan]], np.float32)):
            with self.assertRaises(ValueError):
                qc.stereo(raw)
        with self.assertRaises(ValueError):
            qc.sum_reference({"vox": np.zeros((2, 1)), "drum": np.ones((2, 1))}, {"vox"})

    def test_technical_stats_are_not_model_scores_or_training_approval(self):
        raw = np.ones((44100, 2), np.float32) * .01
        stats = qc.audio_stats(raw, 44100)
        self.assertAlmostEqual(stats["rms_dbfs"], -40., places=5)
        self.assertEqual(stats["activity_seconds_at_minus60_dbfs_1s_windows"], 1.)
        report = {"schema": 1, "formal_training_ready": False, "blind_status": "incomplete",
                  "model_scoring_performed": False, "training_performed": False,
                  "records": [{"track_id": "new", "role": "acceptance_candidate", "training_eligible": False,
                               "acceptance_ready": False, "source_listening_review": "pending"}]}
        qc.validate_quarantine(report)
        for key, value in (("formal_training_ready", True), ("blind_status", "ready"),
                           ("model_scoring_performed", True), ("training_performed", True)):
            bad = copy.deepcopy(report)
            bad[key] = value
            with self.assertRaises(ValueError):
                qc.validate_quarantine(bad)
        bad = copy.deepcopy(report)
        bad["records"][0]["training_eligible"] = True
        with self.assertRaises(ValueError):
            qc.validate_quarantine(bad)


if __name__ == "__main__":
    unittest.main()
