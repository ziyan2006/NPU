"""New213 tests, not old211 units/versions or any208 fixed real-input replay.

One fresh TEMP synthetic PCM16 RIFF only: exact original pseudo probe and
decode, each once. No music, Torch/NumPy/SciPy/soundfile/PT/model/Adam/CUDA.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema213_new_runtime", ROOT / "scripts/213_ema_direct_decoder_runtime.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def reseal(doc):
    doc.pop("content_sha256", None)
    return g.meta.seal(doc)


class NewDecoderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = random.getstate()
        cls.manifest = g.read_manifest()
        cls.evidence = json.loads((ROOT / "results/mel_ema_loaded_image_observation_20261004/loaded_image_evidence.json").read_text())
        cls.path = r"D:\NONEXISTENT\new Unicode \u6d4b\u8bd5 & % !.wav"
        cls.requests = g.request.original_requests(cls.path)
        cls.native_result = None

    @classmethod
    def tearDownClass(cls):
        assert random.getstate() == cls.rng
        assert not any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile"))
        if cls.native_result is not None:
            print("NEW_CONDITIONAL_DECODER_EVIDENCE " + json.dumps(cls.native_result, ensure_ascii=True))
        print("EVIDENCE new213 synthetic RIFF only; old version/208 draws=0; student/PT/model/Adam/CUDA=0")

    def test_all_six_original_shapes_no_requests_spawned(self):
        with unittest.mock.patch.object(subprocess, "Popen", side_effect=AssertionError("Metadata-only")):
            for row in self.requests:
                policy = g.checked_request(row["argv"], row["kwargs"])
                self.assertEqual(policy["argv"][1:], row["argv"][1:])
                self.assertEqual(policy["kwargs"], row["kwargs"])
                self.assertFalse(policy["shell"])

    def test_original_graph_not_shortened(self):
        row = next(r for r in self.requests if r["label"] == "musdb1")
        self.assertIn("amix=inputs=3:normalize=0:duration=shortest", " ".join(row["argv"]))
        bad = row["argv"].copy(); bad[bad.index("-filter_complex") + 1] += " "
        self.assertRaises(ValueError, g.checked_request, bad, row["kwargs"])

    def test_timeout_bool_rejected(self):
        row = next(r for r in self.requests if r["label"] == "pseudo_probe0")
        self.assertRaises(ValueError, g.checked_request, row["argv"], {"timeout": True})

    def test_timeout_float_rejected(self):
        row = next(r for r in self.requests if r["label"] == "pseudo_decode0")
        self.assertRaises(ValueError, g.checked_request, row["argv"], {"timeout": 600.0})

    def test_extra_shell_env_executable_rejected(self):
        row = self.requests[0]
        for name in ("shell", "env", "executable", "cwd"):
            self.assertRaises(ValueError, g.checked_request, row["argv"], {name: False})

    def test_missing_timeout_rejected(self):
        row = next(r for r in self.requests if r["label"] == "pseudo_decode0")
        self.assertRaises(ValueError, g.checked_request, row["argv"], {})

    def test_other_program_rejected(self):
        self.assertRaises(ValueError, g.checked_request, ["cmd.exe", "/c", "dir"], {})

    def test_preselected_executable_caller_rejected(self):
        row = self.requests[0]; argv = row["argv"].copy()
        argv[0] = g.meta.EXPECTED[argv[0]]["target"]
        self.assertRaises(ValueError, g.checked_request, argv, row["kwargs"])

    def test_path_shell_characters_only_literal_tokens(self):
        for row in self.requests[:4]:
            self.assertIn(self.path, g.checked_request(row["argv"], row["kwargs"])["argv"])

    def test_rate_pipe_or_xerror_change_rejected(self):
        row = next(r for r in self.requests if r["label"] == "pseudo_decode0")
        for old, new in (("44100", "48000"), ("pipe:1", "out.wav"), ("-xerror", "-y")):
            bad = [new if t == old else t for t in row["argv"]]
            self.assertRaises(ValueError, g.checked_request, bad, row["kwargs"])

    def test_tuple_argv_nul_nonstring_rejected(self):
        self.assertRaises(ValueError, g.checked_request, tuple(self.requests[0]["argv"]), {})
        self.assertRaises(ValueError, g.checked_request, ["ffmpeg", "a\0b"], {})
        self.assertRaises(ValueError, g.checked_request, ["ffmpeg", 1], {})

    def test_physical_manifest_full_unique37(self):
        self.assertEqual(len(self.manifest["files"]), 37)
        self.assertFalse(self.manifest["Python_native_runtime_verified"])
        self.assertFalse(self.manifest["historical_audio_runtime_verified"])
        self.assertFalse(self.manifest["training_authorized"])

    def test_win_sxs_actual_path_retained(self):
        self.assertTrue(any("winsxs" in p for p in self.manifest["files"]))

    def test_copy_noalias_and_nested_ownseal(self):
        manifest = g.manifest_from_evidence(self.evidence)
        next(iter(manifest["files"].values()))["bytes"] = 1
        self.assertNotEqual(manifest, self.manifest)
        bad = copy.deepcopy(self.evidence)
        bad["rows"][0]["popen_exit_code"] = 1
        reseal(bad)  # Do NOT reseal nested row.
        self.assertRaises(ValueError, g.manifest_from_evidence, bad)

    def test_missing_row_seal_rejected(self):
        bad = copy.deepcopy(self.evidence)
        bad["rows"][0].pop("content_sha256")
        reseal(bad)
        self.assertRaises(ValueError, g.manifest_from_evidence, bad)

    def test_unlisted_actual_path_rejected(self):
        info = copy.deepcopy(next(iter(self.manifest["files"].values())))
        info["final_path"] = r"C:\unlisted.dll"
        self.assertRaises(ValueError, g.validate_loaded, info, self.manifest)

    def test_any_actual_file_byte_or_identity_difference_rejected(self):
        original = next(iter(self.manifest["files"].values()))
        for name in ("bytes", "volume_serial", "file_index", "last_write_ticks", "sha256"):
            bad = copy.deepcopy(original)
            bad[name] = "0" * 64 if name == "sha256" else bad[name] + 1
            self.assertRaises(ValueError, g.validate_loaded, bad, self.manifest)

    def test_initial_actual_exe_not_requested_rejected(self):
        file = self.manifest["files"][g.canonical_path(g.meta.EXPECTED["ffprobe"]["target"])]
        self.assertRaises(ValueError, g.validate_loaded, file, self.manifest,
                          initial_target=g.meta.EXPECTED["ffmpeg"]["target"])

    def test_resealed_manifest_substitution_rejected_before_native(self):
        bad = copy.deepcopy(self.manifest)
        next(iter(bad["files"].values()))["bytes"] += 1
        reseal(bad)
        with unittest.mock.patch.object(g.obs, "Native", side_effect=AssertionError("Do not call native")):
            self.assertRaises(ValueError, g.DirectDecoder, bad)

    def test_real_temporary_file_lock_denies_write_delete(self):
        with tempfile.TemporaryDirectory(prefix="ema213_lock_") as directory:
            path = Path(directory) / "new.tmp"
            path.write_bytes(b"synthetic metadata fixture")  # Fixture data, not repo edit.
            native = g.obs.Native()
            native.dll.CreateFileW.argtypes = [g.wintypes.LPCWSTR, g.wintypes.DWORD, g.wintypes.DWORD,
                                              g.wintypes.LPVOID, g.wintypes.DWORD, g.wintypes.DWORD, g.wintypes.HANDLE]
            native.dll.CreateFileW.restype = g.wintypes.HANDLE
            handle = native.dll.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
            try:
                info = native.measure(handle)
            finally:
                native.checked("CloseHandle", handle)
            with g.HeldFiles({g.canonical_path(info["final_path"]): info}, native) as held:
                self.assertRaises(PermissionError, path.write_bytes, b"mutated")
                self.assertRaises(PermissionError, path.unlink)
                held.check()
            self.assertEqual(path.read_bytes(), b"synthetic metadata fixture")

    def test_new_synthetic_probe_and_decode_exact_original_shapes_once(self):
        with tempfile.TemporaryDirectory(prefix="ema213_native_") as directory:
            path = Path(directory) / "new \u6d4b\u8bd5 & % !.wav"
            pcm = [(-8192 + i % 16384, 4096 - i % 8192) for i in range(1024)]
            raw = b"".join(struct.pack("<hh", *frame) for frame in pcm)
            header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(raw), b"WAVE", b"fmt ",
                                 16, 1, 2, 44100, 44100 * 4, 4, 16, b"data", len(raw))
            path.write_bytes(header + raw)  # New synthetic input only, no music or student export.
            rows = g.request.original_requests(str(path))
            adapter = g.DirectDecoder(self.manifest)
            probe = next(r for r in rows if r["label"] == "pseudo_probe0")
            decoded = next(r for r in rows if r["label"] == "pseudo_decode0")
            metadata = json.loads(adapter.check_output(probe["argv"], **probe["kwargs"]))
            self.assertEqual(metadata["streams"][0]["sample_rate"], "44100")
            self.assertEqual(metadata["streams"][0]["channels"], 2)
            actual = adapter.check_output(decoded["argv"], **decoded["kwargs"])
            expected = b"".join(struct.pack("<ff", left / 32768, right / 32768) for left, right in pcm)
            self.assertEqual(actual, expected)
            self.assertEqual(len(adapter.rows), 2)
            for row in adapter.rows:
                g.meta.check_seal(row)
                self.assertTrue(row["accepted_debug_call"])
                self.assertEqual(row["exit_code"], 0)
                self.assertEqual(row["pre_spawn_runtime_files_held"], 37)
                self.assertTrue(all(e["matched_preheld_physical_identity"] for e in row["events"] if e["code"] in (3, 6)))
            type(self).native_result = {"rows": adapter.rows, "synthetic_pcm_bytes": len(actual),
                "synthetic_pcm_sha256": hashlib.sha256(actual).hexdigest(), "exact_analytic_PCM16_to_FP32": True,
                "children": 2, "actual_music_draws": 0, "Python_native_runtime_verified": False,
                "training_authorized": False, "release_selection": "NONE"}


if __name__ == "__main__":
    unittest.main(verbosity=2)
