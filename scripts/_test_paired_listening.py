"""CPU-only private playback contracts; UI drafts never authorize training."""
import copy
from contextlib import redirect_stdout
import http.client
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pack = load("paired_listening_test_pack", "144_export_paired_listening.py")
serve = load("paired_listening_test_server", "145_serve_paired_listening.py")
torch.set_num_threads(2)


def sealed(path, body):
    pack.pilot.acq.write_new_json(path, pack.pilot.acq.seal(body))


def fixture(root):
    root = Path(root)
    prepared = root / "prepared"
    prepared.mkdir()
    source = root / "source.mp3"
    source.write_bytes(b"Mock decoder input, not real MP3")
    x = torch.full((2, 256), .4)
    vocal_ht = torch.full_like(x, .8)
    vocal_ht[:, 128] = 1.9
    vocal_mel = x * .3
    source_info = {"path": str(source), "sha256": pack.pilot.acq.sha256(source), "samples": 256,
                   "pcm_sha256": pack.pilot.wave_digest(x)}
    snapshot_info = {}
    ids = [f"song_{i:04d}" for i in range(24)]
    for teacher, v in (("htdemucs", vocal_ht), ("kim_melband", vocal_mel)):
        directory = root / teacher
        directory.mkdir()
        labels = {}
        for name, wave in (("vocals.wav", v), ("accompaniment.wav", x - v)):
            path = directory / name
            pack.pilot.write_wave_new(path, wave)
            labels[name] = {"path": str(path), "sha256": pack.pilot.acq.sha256(path)}
        rows = [{"song_id": ident, "role": "pseudo_label_train_candidate", "training_eligible": False,
                 "source": source_info, "label_files": labels} for ident in ids]
        path = prepared / f"{teacher}_snapshot.json"
        sealed(path, {"training_authorized": False, "records": rows})
        snapshot_info[teacher] = {"path": str(path), "sha256": pack.pilot.acq.sha256(path)}
    bundle_path = prepared / "paired_bundle.json"
    sealed(bundle_path, {"training_authorized": False, "human_reviewed_pairs": 0, "pair_ids": ids,
                         "snapshots": snapshot_info, "bindings_sha256": {str(source): source_info["sha256"]}})
    records = [{"song_id": ident, "track_id": "Unsafe </script><script>title</script>" if i == 0 else ident,
                "source_path": str(source), "same_input_pcm_sha256": source_info["pcm_sha256"],
                "review_status": "pending", "rights_status": "pending", "training_eligible": False,
                "windows": [{"start_sample": 0, "samples": 256} for _ in range(3)],
                "listening_hints": {"htdemucs": [], "kim_melband": []}} for i, ident in enumerate(ids)]
    sealed(prepared / "listening_pending.json", {"bundle_sha256": pack.pilot.acq.sha256(bundle_path),
                                                 "reviewed": 0, "training_authorized": False, "records": records})
    return prepared, x


class PairedListeningTests(unittest.TestCase):
    def test_exact_window_contract_no_padding_or_fast_seek(self):
        pack.valid_windows([{"start_sample": 5, "samples": 10}] * 3, 20)
        for windows in ([{"start_sample": 5, "samples": 10}],
                        [{"start_sample": 11, "samples": 10}] * 3,
                        [{"start_sample": True, "samples": 10}] * 3,
                        [{"start_sample": 0, "samples": 0}] * 3):
            with self.assertRaises(ValueError):
                pack.valid_windows(windows, 20)
        with self.assertRaises(ValueError):
            pack.valid_windows([{"start_sample": 0, "samples": 19 * pack.pilot.SR}] * 3, 20 * pack.pilot.SR)
        with self.assertRaises(ValueError):
            pack.valid_windows([{"start_sample": 0, "samples": n} for n in (10, 11, 10)], 20)

    def test_output_guard_keeps_playback_in_ignored_child(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pack, "PREVIEW_ROOT", Path(folder)):
            self.assertEqual(pack.guard_output(Path(folder) / "child"), Path(folder) / "child")
            for path in (Path(folder), Path(folder).parent / "outside"):
                with self.assertRaises(ValueError):
                    pack.guard_output(path)

    def test_pending_identity_cannot_become_training_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            prepared, _ = fixture(folder)
            bundle, pending, _ = pack.read_inputs(prepared)
            self.assertFalse(bundle["training_authorized"])
            self.assertEqual(len(pending["records"]), 24)
            path = prepared / "listening_pending.json"
            body = copy.deepcopy(pending)
            body["records"][0]["review_status"] = "approved"
            path.unlink()
            sealed(path, body)
            with self.assertRaises(ValueError):
                pack.read_inputs(prepared)

    def test_changed_source_snapshot_or_pair_order_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            prepared, _ = fixture(folder)
            bundle, _, _ = pack.read_inputs(prepared)
            info = bundle["snapshots"]["htdemucs"]
            Path(info["path"]).write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                pack.read_inputs(prepared)
        with tempfile.TemporaryDirectory() as folder:
            prepared, _ = fixture(folder)
            Path(folder, "source.mp3").write_bytes(b"changed original")
            with self.assertRaises(ValueError):
                pack.read_inputs(prepared)
        with tempfile.TemporaryDirectory() as folder:
            prepared, _ = fixture(folder)
            path = prepared / "listening_pending.json"
            body = pack.pilot.acq.read_sealed(path)
            body["records"].reverse()
            path.unlink()
            sealed(path, body)
            with self.assertRaises(ValueError):
                pack.read_inputs(prepared)

    def test_full_pcm_mismatch_and_original_label_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            prepared, x = fixture(folder)
            _, pending, snapshots = pack.read_inputs(prepared)
            a, b = (snapshots[t]["records"][0] for t in ("htdemucs", "kim_melband"))
            with patch.object(pack.bulk, "decode", return_value=x * .9):
                with self.assertRaises(ValueError):
                    pack.song_windows(a, b, pending["records"][0]["windows"])
            Path(a["label_files"]["vocals.wav"]["path"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                pack.song_windows(a, b, pending["records"][0]["windows"])

    def test_non_exact_residual_rejected_even_if_label_hash_updated(self):
        with tempfile.TemporaryDirectory() as folder:
            prepared, x = fixture(folder)
            _, pending, snapshots = pack.read_inputs(prepared)
            a, b = (snapshots[t]["records"][0] for t in ("htdemucs", "kim_melband"))
            bad = a["label_files"]["accompaniment.wav"]
            Path(bad["path"]).unlink()
            pack.pilot.write_wave_new(Path(bad["path"]), x)
            bad["sha256"] = pack.pilot.acq.sha256(bad["path"])
            with patch.object(pack.bulk, "decode", return_value=x):
                with self.assertRaises(ValueError):
                    pack.song_windows(a, b, pending["records"][0]["windows"])

    def test_disk_estimate_and_reserve_are_enforced(self):
        pending = {"records": [{"windows": [{"samples": 100}] * 3}] * 24}
        estimate = pack.budget(pending)
        self.assertGreater(estimate, 24 * 3 * 100 * 5 * 6)
        with self.assertRaises(ValueError):
            pack.bulk.check_storage(estimate + 10, estimate, 12 * 1024**3)

    def test_export_common_gain_readback_no_gpu_no_original_writes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pack, "PREVIEW_ROOT", Path(folder)):
            prepared, x = fixture(folder)
            before = {path: pack.pilot.acq.sha256(path) for path in Path(folder).rglob("*") if path.is_file()}
            out = Path(folder) / "preview"
            with patch.object(pack.bulk, "decode", return_value=x) as decode, \
                 patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
                 patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CUDA RNG queried")), \
                 redirect_stdout(io.StringIO()):
                pack.export(prepared, out)
            self.assertEqual(decode.call_count, 24)
            doc = pack.pilot.acq.read_sealed(out / "manifest.json")
            self.assertEqual(len(pack.pack_files(doc)), 363)
            self.assertFalse(doc["training_authorized"])
            self.assertFalse(doc["training_label_copies"])
            self.assertEqual(doc["human_review"], "pending")
            for row in doc["records"]:
                self.assertAlmostEqual(row["common_playback_gain"], .5, places=6)
                for window in row["windows"]:
                    saved = pack.pilot.read_wave(out / next(name for name in window["files_sha256"] if name.endswith("/mix.wav")))
                    self.assertLess(float((saved - x * row["common_playback_gain"]).abs().max()), pack.playback.PCM_TOLERANCE)
            for path, expected in before.items():
                self.assertEqual(pack.pilot.acq.sha256(path), expected)
            page = (out / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("Unsafe </script><script>", page)
            self.assertIn("\\u003c/script>", page)
            with self.assertRaises(ValueError):
                pack.export(prepared, out)
            (out / "song_0000/w1/mix.wav").write_bytes(b"tampered preview")
            with self.assertRaises(ValueError):
                pack.verify(out)

    def test_whitelist_rejects_missing_signal_and_path_escape(self):
        doc = {"ui_files_sha256": {name: "sha" for name in ("index.html", "app.js", "style.css")},
               "records": [{"song_id": "../source", "windows": [{"files_sha256": {}}] * 3}]}
        with self.assertRaises(ValueError):
            pack.pack_files(doc)
        doc["records"][0]["song_id"] = "song_0000"
        with self.assertRaises(ValueError):
            pack.pack_files(doc)

    def test_range_handles_head_seek_suffix_and_bounds(self):
        self.assertEqual(serve.byte_range(None, 100), (0, 99, False))
        self.assertEqual(serve.byte_range("bytes=5-10", 100), (5, 10, True))
        self.assertEqual(serve.byte_range("bytes=90-", 100), (90, 99, True))
        self.assertEqual(serve.byte_range("bytes=-7", 100), (93, 99, True))
        self.assertEqual(serve.byte_range("bytes=90-999", 100), (90, 99, True))
        for value in ("bytes=100-", "bytes=20-5", "bytes=-0", "bytes=-", "bytes=0-1,3-4", "invalid"):
            with self.assertRaises(ValueError):
                serve.byte_range(value, 100)

    def test_loopback_service_token_whitelist_range_and_no_writes(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(pack, "PREVIEW_ROOT", Path(folder)), \
                patch.object(serve.pack, "PREVIEW_ROOT", Path(folder)):
            prepared, x = fixture(folder)
            out = Path(folder) / "preview"
            with patch.object(pack.bulk, "decode", return_value=x), redirect_stdout(io.StringIO()):
                pack.export(prepared, out)
            with redirect_stdout(io.StringIO()):
                server = serve.create_server(out, 0, token="testlocalonly123456")
            self.assertEqual(server.server_address[0], "127.0.0.1")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                def request(path, method="GET", headers=None):
                    conn = http.client.HTTPConnection(*server.server_address, timeout=5)
                    conn.request(method, path, headers=headers or {})
                    response = conn.getresponse()
                    result = response.status, dict(response.getheaders()), response.read()
                    conn.close()
                    return result
                base = "/testlocalonly123456/"
                status, headers, body = request(base + "index.html")
                self.assertEqual(status, 200)
                self.assertIn(b"<!doctype html>", body)
                self.assertEqual(headers["Referrer-Policy"], "no-referrer")
                self.assertNotIn("Access-Control-Allow-Origin", headers)
                self.assertEqual(request(base + "style.css", "HEAD")[2], b"")
                status, headers, body = request(base + "song_0000/w1/mix.wav", headers={"Range": "bytes=0-15"})
                self.assertEqual(status, 206)
                self.assertEqual(len(body), 16)
                self.assertTrue(body.startswith(b"RIFF"))
                self.assertEqual(request(base + "song_0000/w1/mix.wav", headers={"Range": "bytes=999999-"})[0], 416)
                for path in ("/", "/wrong/index.html", base + "manifest.json", base + "song_0000/", base + "../source.mp3", base + "%2e%2e/source.mp3"):
                    self.assertEqual(request(path)[0], 404)
                self.assertEqual(request(base + "index.html", "POST")[0], 501)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(5)

    def test_ui_is_offline_and_draft_only_not_an_approval(self):
        code = (pack.ASSETS / "app.js").read_text(encoding="utf-8")
        page = (pack.ASSETS / "index.html").read_text(encoding="utf-8")
        self.assertIn("training_authorized: false", code)
        self.assertIn("original_label_entries_modified: false", code)
        self.assertIn("localStorage.setItem", code)
        self.assertIn("other.pause()", code)
        self.assertNotIn("fetch(", code)
        self.assertNotIn("https://", code + page)
        self.assertIn('value="pending"', page)


if __name__ == "__main__":
    unittest.main()
