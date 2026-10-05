"""New210 request-only checks. No actual child/audio/old unit execution."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("ema210_new_units", Path(__file__).with_name("210_ema_decoder_launch_request.py"))
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)
OUT = j.ROOT / "results/mel_ema_decoder_launch_request_monitor_20261004"


class LaunchRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng, cls.env = random.getstate(), dict(os.environ)
        cls.proof = j.inspect_installed_requests()  # One12-request inspection; no native child.
        cls.rows = cls.proof["rows"]

    @classmethod
    def tearDownClass(cls):
        assert cls.rng == random.getstate() and cls.env == dict(os.environ)
        j.verify_bindings(cls.proof["bindings_sha256"])
        print("REQUEST_ONLY_EVIDENCE " + json.dumps({"interceptions": 12, "native_CreateProcess_calls": 0,
              "child_processes": 0, "actual_audio_draws": 0, "model_pt_adam_cuda": 0,
              "full_audio_backend_gate": "PENDING", "old_unit_reruns": 0}, sort_keys=True))

    def test_01_six_original_shapes(self):
        self.assertEqual([v["original"]["label"] for v in self.rows],
                         ["musdb0", "musdb1", "pseudo_probe0", "pseudo_decode0", "version0ffmpeg", "version0ffprobe"])

    def test_02_all_original_null_application(self):
        for row in self.rows:
            self.assertIsNone(row["intercepted_original"]["application_name"])

    def test_03_all_original_serialization(self):
        for row in self.rows:
            self.assertEqual(row["intercepted_original"]["command_line"], subprocess.list2cmdline(row["original"]["argv"]))

    def test_04_explicit_exe_application_and_argv(self):
        for row in self.rows:
            policy, observed = row["prospective_policy"], row["intercepted_explicit"]
            self.assertEqual(observed["application_name"], policy["argv"][0])
            self.assertEqual(observed["command_line"], subprocess.list2cmdline(policy["argv"]))

    def test_05_preserve_every_non_program_token(self):
        for row in self.rows:
            self.assertEqual(j.typed(row["original"]["argv"][1:]), j.typed(row["prospective_policy"]["argv"][1:]))

    def test_06_exact_original_timeout_scope(self):
        self.assertEqual([row["original"]["kwargs"] for row in self.rows], [{}, {}, {"timeout": 30}, {"timeout": 600}, {"timeout": 30}, {"timeout": 30}])

    def test_07_original_filter_graph(self):
        argv = self.rows[1]["original"]["argv"]
        self.assertEqual(argv[argv.index("-filter_complex")+1], "[0:a:1][0:a:2][0:a:3]amix=inputs=3:normalize=0:duration=shortest[a];[a][0:a:4]join=inputs=2:channel_layout=quad:map=0.0-FL|0.1-FR|1.0-BL|1.1-BR[out]")

    def test_08_original_pcm_shapes(self):
        for row in (self.rows[1], self.rows[3]):
            self.assertEqual(row["original"]["argv"][-5:], ["-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
        argv = self.rows[3]["original"]["argv"]
        self.assertEqual(argv[argv.index("-ar")+1], "44100")
        self.assertEqual(argv[argv.index("-ac")+1], "2")

    def test_09_metadata_path_no_file_read(self):
        self.assertIn("NONEXISTENT", self.rows[0]["original"]["argv"][-1])
        self.assertIn("& %!", self.rows[0]["original"]["argv"][-1])
        self.assertFalse(Path(self.rows[0]["original"]["argv"][-1]).exists())

    def test_10_hidden_direct_request(self):
        for row in self.rows:
            observed = row["intercepted_explicit"]
            self.assertEqual(observed["creationflags"], subprocess.CREATE_NO_WINDOW)
            self.assertTrue(observed["startupinfo"]["dwFlags"] & subprocess.STARTF_USESHOWWINDOW)
            self.assertEqual(observed["startupinfo"]["wShowWindow"], subprocess.SW_HIDE)

    def test_11_environment_and_cwd_inherited_not_reconfigured(self):
        for row in self.rows:
            for key in ("intercepted_original", "intercepted_explicit"):
                self.assertIsNone(row[key]["environment"])
                self.assertIsNone(row[key]["cwd"])

    def test_12_handle_metadata_and_security(self):
        for row in self.rows:
            for key in ("intercepted_original", "intercepted_explicit"):
                request = row[key]
                self.assertIsNone(request["process_security"])
                self.assertIsNone(request["thread_security"])
                self.assertEqual(request["inherit_handles"], 1)
                self.assertIs(type(request["startupinfo"]["hStdOutput"]), int)
                self.assertIn(request["startupinfo"]["hStdOutput"], request["startupinfo"]["lpAttributeList"]["handle_list"])

    def test_13_own_and_nested_seals(self):
        j.check_seal(self.proof)
        for row in self.rows: j.check_seal(row)

    def test_14_missing_nested_seal_rejected(self):
        row = copy.deepcopy(self.rows[0]); row.pop("content_sha256")
        with self.assertRaises(ValueError): j.check_seal(row)

    def test_15_typed_bool_int_and_list_tuple_distinct(self):
        self.assertNotEqual(j.typed(True), j.typed(1))
        self.assertNotEqual(j.typed([1]), j.typed((1,)))

    def test_16_no_policy_alias(self):
        original = copy.deepcopy(self.rows[0]["original"])
        policy = j.explicit_request(original)
        policy["argv"][1] = "changed"
        self.assertEqual(original["argv"][1], "-v")

    def test_17_unknown_options_rejected_before_popen(self):
        with patch.object(subprocess, "Popen") as popen:
            for bad in ({"shell": True}, {"executable": "ffmpeg"}, {"timeout": True}, {"timeout": 0}):
                with self.assertRaises(ValueError): j.capture_request(["ffmpeg", "-version"], bad)
            popen.assert_not_called()

    def test_18_invalid_argv_rejected_before_popen(self):
        with patch.object(subprocess, "Popen") as popen:
            for bad in ("ffmpeg -version", [], ["ffmpeg", True], ["ffmpeg", "a\0b"]):
                with self.assertRaises(ValueError): j.capture_request(bad, {})
            popen.assert_not_called()

    def test_19_changed_policy_rejected_before_popen(self):
        for key, value in (("shell", True), ("executable", "ffmpeg"), ("creationflags", 0), ("env", {}), ("actual_spawn_authorized", True)):
            policy = copy.deepcopy(self.rows[0]["prospective_policy"]); policy[key] = value
            with patch.object(subprocess, "Popen") as popen:
                with self.assertRaises(ValueError): j.capture_request(policy["argv"], {}, policy=policy)
                popen.assert_not_called()

    def test_20_ast_arbitrary_code_not_executed(self):
        for code in ("open('x')", "__import__('os').system('x')", "[x for x in ()]", "path.read_bytes()", "True"):
            with self.assertRaises(ValueError): j._value(ast.parse(code, mode="eval").body, {"path": "x"})

    def test_21_unknown_decoder_and_type_rejected(self):
        for argv in (["cmd", "/c"], ["ffmpeg", 1]):
            original = copy.deepcopy(self.rows[0]["original"]); original["argv"] = argv
            with self.assertRaises(ValueError): j.explicit_request(original)

    def test_22_runtime_no_training_authority(self):
        for field in ("historical208_actual_image_path_observed", "explicit_policy_actual_spawn_verified", "runtime_transitive_loaded_modules_verified", "training_authorized"):
            self.assertIs(self.proof[field], False)
        with self.assertRaisesRegex(ValueError, "PENDING"): j.require_audio_runtime_authority()

    def test_23_no_heavy_native_imports(self):
        for name in ("torch", "numpy", "scipy", "soundfile"):
            self.assertNotIn(name, sys.modules)

    def test_24_actual_counts_and_rng(self):
        self.assertEqual(self.proof["request_interceptions"], 12)
        for key in ("native_CreateProcess_calls", "actual_child_processes", "actual_audio_draws", "model_or_optimizer_or_pt_load"):
            self.assertEqual(self.proof[key], 0)
        self.assertEqual(self.rng, random.getstate())
        self.assertEqual(self.env, dict(os.environ))

    def test_25_source_and_runtime_bytes_unchanged(self):
        j.verify_bindings(self.proof["bindings_sha256"])


if __name__ == "__main__":
    evidence = OUT / "request_interception_evidence.json"
    j.require(not OUT.exists(), "Refuse repeated new units/output")
    program = unittest.main(verbosity=2, exit=False)
    if not program.result.wasSuccessful():
        sys.exit(1)
    OUT.mkdir()  # Exclusive new monitor. No failed run is relabelled success.
    with evidence.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(LaunchRequestTests.proof, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    print("REQUEST_EVIDENCE_FILE " + str(evidence) + " SHA256=" + j.sha(evidence))
