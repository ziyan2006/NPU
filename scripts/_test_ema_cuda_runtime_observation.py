"""New216 preflight/scope checks, no CUDA initialization/model or child."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema216_runtime_checks", ROOT / "scripts/216_ema_cuda_runtime_observation.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RuntimeGateTests(unittest.TestCase):
    def test_stdlib_host_has_no_cuda_or_heavy_import(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "scipy", "soundfile")))

    def test_exact_completed_model_audit_gate(self):
        doc = r.gate()
        self.assertTrue(doc["actual_debug_supervised_native_audio_scope_passed"])
        self.assertEqual(doc["audit_report"]["exposure"]["forward_completed"], 6)
        self.assertFalse(doc["audit_report"]["CUDA_initialized"])

    def test_changed_prior_source_refused(self):
        with patch.dict(r.PINS, {"scripts/215_ema_model_audio_audit.py": "0" * 64}):
            self.assertRaises(ValueError, r.check_files)

    def test_tampered_prior_review_seal_refused(self):
        doc = r.gate(); doc["launcher_exit_code"] = 1
        self.assertRaises(ValueError, r.p.check_metadata, doc)

    def test_shared_high_gpu_utilization_not_a_blocker(self):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout="0, 2400, 99\n")), \
             patch.object(r.shutil, "disk_usage", return_value=SimpleNamespace(free=13 * 1024**3)):
            info = r.resource()
        self.assertEqual(info["utilization_percent"], 99)
        self.assertTrue(info["shared_GPU_utilization_authorized"])

    def test_minimum_vram_boundary(self):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout="0, 2300, 99\n")), \
             patch.object(r.shutil, "disk_usage", return_value=SimpleNamespace(free=12 * 1024**3)):
            self.assertEqual(r.resource()["free_mib"], 2300)

    def test_low_vram_refused_before_init(self):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout="0, 2299, 0\n")):
            self.assertRaises(ValueError, r.resource)

    def test_low_disk_refused_before_init(self):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout="0, 2400, 0\n")), \
             patch.object(r.shutil, "disk_usage", return_value=SimpleNamespace(free=12 * 1024**3 - 1)):
            self.assertRaises(ValueError, r.resource)

    def test_multi_gpu_refused(self):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout="0, 3000, 0\n1, 3000, 0\n")):
            self.assertRaises(ValueError, r.resource)

    def test_unexpected_gpu_index_refused(self):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout="1, 3000, 0\n")):
            self.assertRaises(ValueError, r.resource)

    def test_missing_resource_fields_refused(self):
        for result in ("", "0, 2400", "0, 2400, 0, 1"):
            with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout=result)):
                self.assertRaises(ValueError, r.resource)

    def test_unknown_resource_value_refused(self):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout="0, N/A, 0\n")):
            self.assertRaises(ValueError, r.resource)

    def test_existing_output_refused_before_resource_or_child(self):
        with patch.object(r, "resource", side_effect=AssertionError("No preflight for existing output")), \
             patch.object(subprocess, "Popen", side_effect=AssertionError("No child")):
            self.assertRaises(ValueError, r.collect, ROOT / "results/ema215_real_model_audio_audit_attempt01")

    def test_outside_results_refused_before_resource_or_child(self):
        with patch.object(r, "resource", side_effect=AssertionError("No preflight")):
            self.assertRaises(ValueError, r.collect, ROOT / "scripts/ema216_unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
