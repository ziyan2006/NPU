"""CPU contracts: label alignment, contexts, fair teacher pairs and sampler resume."""
from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import soundfile as sf
import torch

spec = importlib.util.spec_from_file_location("distillation_dataset_test", Path(__file__).with_name("136_prepare_distillation_data.py"))
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
torch.set_num_threads(2)
CONFIG = json.loads(data.PROTOCOL.read_text(encoding="utf-8"))["input"]


@contextmanager
def fixture(songs=2, samples=120000):
    with tempfile.TemporaryDirectory() as folder:
        root, rows, decoded = Path(folder), [], {}
        for i in range(songs):
            source = root/f"original_{i}.mp3"
            source.write_bytes(f"encoded audio fixture {i}".encode())
            gen = torch.Generator(device="cpu").manual_seed(700+i)
            mix = torch.randn(2, samples, generator=gen)*.15
            v = mix*3
            a, _ = data.pilot.residual_pair(mix, v)
            paths = {"vocals.wav": root/f"v_{i}.wav", "accompaniment.wav": root/f"a_{i}.wav"}
            values = {"vocals.wav": v, "accompaniment.wav": a}
            labels = {name: {"path": str(path), "sha256": data.pilot.write_wave_new(path, values[name])} for name, path in paths.items()}
            rows.append({"song_id": f"song_{i:04d}", "source": {"path": str(source), "samples": samples,
                         "sha256": data.pilot.acq.sha256(source), "pcm_sha256": data.pilot.wave_digest(mix)},
                         "role": "pseudo_label_train_candidate", "training_eligible": False, "label_files": labels})
            decoded[str(source)] = mix
        doc = {"schema": 1, "input": CONFIG, "sampler_seed": 20261002, "records": rows,
               "bindings_sha256": {}, "training_authorized": False}
        path = root/"snapshot.json"
        data.pilot.acq.write_new_json(path, data.pilot.acq.seal(doc))
        with patch.object(data.bulk, "decode", side_effect=lambda p: decoded[str(p)].clone()) as decoder:
            yield path, doc, decoded, decoder


class InputTests(unittest.TestCase):
    def recipe(self, ident="song_0000", start=2048, db=0):
        length, score, _ = data.geometry(CONFIG)
        return {"song_id": ident, "start_sample": start, "samples": length, "common_db": db,
                "score_start": score.start, "score_end": score.stop, "cursor": 0}

    def test_geometry_is_context_and_teacher_source_edge_safe(self):
        length, score, tail = data.geometry(CONFIG)
        self.assertEqual(length, 89856)
        self.assertEqual((score.start, score.stop), (25088, 89344))
        self.assertEqual(tail, 10513)
        self.assertEqual(length//256+1, 352)

    def test_protocol_cannot_simultaneously_change_graph_or_start_training(self):
        protocol = json.loads(data.PROTOCOL.read_text(encoding="utf-8"))
        data.validate_protocol(protocol)
        for field, value in (("graph", protocol["graph"] | {"layout": "mel_unique"}),
                             ("student_training_authorized", True), ("student_promotion_authorized", True)):
            with self.assertRaises(ValueError):
                data.validate_protocol(protocol | {field: value})

    def test_protocol_cannot_drop_pair_or_change_input_exposure_budget(self):
        protocol = json.loads(data.PROTOCOL.read_text(encoding="utf-8"))
        for field, value in (("arms", ["candidate_only"]), ("pseudo_batch_slots", 6),
                             ("maximum_steps", 200), ("effective_batch", 3)):
            changed = copy.deepcopy(protocol)
            changed["paired_comparison_planned"][field] = value
            with self.assertRaises(ValueError):
                data.validate_protocol(changed)

    def test_invalid_frontend_context_or_noninteger_frames_rejected(self):
        for field, value in (("hop", 128), ("warmup_frames", 88), ("crop_frames", 255),
                             ("crop_frames", 256.), ("start_grid_frames", 1), ("sample_rate", 48000)):
            config = copy.deepcopy(CONFIG)
            config[field] = value
            with self.assertRaises(ValueError):
                data.geometry(config)

    def test_deterministic_song_uniform_draws_use_sample_grid_and_safe_tail(self):
        rows = [{"song_id": f"song_{i:04d}", "source": {"samples": 120000+i*10000}} for i in range(4)]
        a = [data.crop_recipe(rows, CONFIG, 31, index) for index in range(200)]
        self.assertEqual(a, [data.crop_recipe(rows, CONFIG, 31, index) for index in range(200)])
        self.assertEqual({r["song_id"] for r in a}, {r["song_id"] for r in rows})
        for item in a:
            source = next(r for r in rows if r["song_id"] == item["song_id"])
            self.assertEqual(item["start_sample"] % 2048, 0)
            self.assertLessEqual(item["start_sample"]+item["score_end"], source["source"]["samples"]-11025)

    def test_paired_24_selection_is_fixed_and_not_teacher_quality_based(self):
        rows = [{"song_id": f"song_{i:04d}"} for i in range(275)]
        a = data.choose_pairs(rows, 24, 20261002)
        self.assertEqual(a, data.choose_pairs(list(reversed(rows)), 24, 20261002))
        self.assertEqual(len(set(a)), 24)
        with self.assertRaises(ValueError):
            data.choose_pairs(rows, 276, 1)

    def test_original_mix_and_stem_windows_remain_sample_exact(self):
        with fixture() as (path, doc, raw, decoder):
            loader = data.TeacherWaveformDataset(path, purpose="cpu_audit")
            item = loader.crop(self.recipe())
            original = raw[doc["records"][0]["source"]["path"]][:, 2048:2048+89856]
            self.assertTrue(torch.equal(item["x"], original))
            self.assertTrue(torch.equal(item["v"], original*3))
            self.assertGreater(float(item["v"].abs().max()), 1.)
            self.assertFalse(item["meta"]["training_eligible"])
            self.assertEqual(item["x"].device.type, "cpu")

    def test_gain_uses_mix_only_and_is_shared_without_label_clipping(self):
        with fixture() as (path, doc, raw, decoder):
            loader = data.TeacherWaveformDataset(path, purpose="cpu_audit")
            native, louder = loader.crop(self.recipe()), loader.crop(self.recipe(db=3))
            gain = louder["meta"]["gain"]
            for key in ("x", "v", "a"):
                self.assertTrue(torch.equal(louder[key], native[key]*gain))
            self.assertLessEqual(float(louder["x"].abs().max()), .950001)
            self.assertGreater(float(louder["v"].abs().max()), 1.)
            self.assertLess(float((louder["x"]-louder["v"]-louder["a"]).abs().max()), 1e-6)

    def test_no_silent_context_padding_or_invalid_grid(self):
        with fixture() as (path, doc, raw, decoder):
            loader = data.TeacherWaveformDataset(path, purpose="cpu_audit")
            for start in (-2048, 1, 40000):
                with self.assertRaises(ValueError):
                    loader.crop(self.recipe(start=start))
        rows = [{"song_id": "short", "source": {"samples": 50000}}]
        with self.assertRaises(ValueError):
            data.crop_recipe(rows, CONFIG, 1, 0)

    def test_sampler_resume_preserves_crops_and_rejects_changed_bindings(self):
        with fixture() as (path, doc, raw, decoder):
            loader = data.TeacherWaveformDataset(path, purpose="cpu_audit")
            loader.next_crop()
            state = loader.state_dict()
            expected = loader.next_crop()
            loader.load_state_dict(state)
            restored = loader.next_crop()
            data.assert_paired_inputs(expected, restored)
            self.assertTrue(torch.equal(expected["v"], restored["v"]))
            for key, value in (("seed", 0), ("snapshot_sha256", "changed"), ("cursor", -1)):
                with self.assertRaises(ValueError):
                    loader.load_state_dict(state | {key: value})

    def test_paired_teacher_inputs_must_match_but_targets_can_differ(self):
        with fixture() as (path, doc, raw, decoder):
            loader = data.TeacherWaveformDataset(path, purpose="cpu_audit")
            left = loader.crop(self.recipe())
            right = copy.deepcopy(left)
            right["v"] *= .5
            data.assert_paired_inputs(left, right)
            right["x"] *= .9
            with self.assertRaises(ValueError):
                data.assert_paired_inputs(left, right)

    def test_cpu_cache_is_bounded_and_no_augmentation_mutates_full_source(self):
        with fixture() as (path, doc, raw, decoder):
            loader = data.TeacherWaveformDataset(path, purpose="cpu_audit", cache_songs=1)
            before = {name: wave.clone() for name, wave in raw.items()}
            loader.crop(self.recipe(db=3))
            loader.crop(self.recipe(start=4096))
            self.assertEqual(decoder.call_count, 1)
            loader.crop(self.recipe(ident="song_0001"))
            self.assertEqual(len(loader.cache), 1)
            for name in raw:
                self.assertTrue(torch.equal(before[name], raw[name]))

    def test_changed_source_pcm_or_label_file_cannot_enter_iteration(self):
        with fixture() as (path, doc, raw, decoder):
            loader = data.TeacherWaveformDataset(path, purpose="cpu_audit")
            source = doc["records"][0]["source"]["path"]
            raw[source] *= .9
            with self.assertRaisesRegex(ValueError, "Full PCM differs"):
                loader.crop(self.recipe())
        with fixture() as (path, doc, raw, decoder):
            loader = data.TeacherWaveformDataset(path, purpose="cpu_audit")
            loader.crop(self.recipe())
            with Path(doc["records"][0]["label_files"]["vocals.wav"]["path"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                loader.crop(self.recipe())

    def test_unreviewed_import_never_starts_training_or_accepts_eval_role(self):
        with fixture() as (path, doc, raw, decoder):
            with self.assertRaises(ValueError):
                data.TeacherWaveformDataset(path)
            doc["records"][0]["role"] = "development"
            new = path.parent/"wrong_role.json"
            data.pilot.acq.write_new_json(new, data.pilot.acq.seal(doc))
            with self.assertRaises(ValueError):
                data.TeacherWaveformDataset(new, purpose="cpu_audit")

    def test_wrong_subtype_rejected_even_with_new_file_hash(self):
        with fixture() as (path, doc, raw, decoder):
            label = doc["records"][0]["label_files"]["vocals.wav"]
            wav, sr = sf.read(label["path"], dtype="float32")
            sf.write(label["path"], wav, sr, subtype="PCM_16")
            label["sha256"] = data.pilot.acq.sha256(label["path"])
            new = path.parent/"wrong_subtype.json"
            data.pilot.acq.write_new_json(new, data.pilot.acq.seal(doc))
            with self.assertRaises(ValueError):
                data.TeacherWaveformDataset(new, purpose="cpu_audit")

    def test_loader_does_not_query_or_initialize_cuda(self):
        with fixture() as (path, doc, raw, decoder):
            with patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
                 patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")), \
                 patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CUDA RNG")):
                loader = data.TeacherWaveformDataset(path, purpose="cpu_audit")
                loader.next_crop()


if __name__ == "__main__":
    unittest.main()
