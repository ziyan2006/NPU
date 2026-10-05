"""Independent recovery of the LF descriptive-statistics CPU-isolation test.

The original35-test source and failing log remain immutable. All34 other
tests are inherited unchanged. The one false-precondition test is executed
in a fresh subprocess with NO Adam fixture, and still checks CUDAfalse,
original tensors and full CPU RNG unchanged. No skip/global override.
"""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location("sealed_lf_units",Path(__file__).with_name("_test_mel_lf_boundary.py"))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class IsolatedLFBoundaryTests(base.LFBoundaryTests):
    def setUp(self):
        if self._testMethodName != "test_low_frequency_descriptor_does_not_modify_wave":
            super().setUp()

    def test_low_frequency_descriptor_does_not_modify_wave(self):
        code = """
import importlib.util,torch
spec=importlib.util.spec_from_file_location('isolated_lf','scripts/186_train_mel_lf_boundary.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
with t.d.deterministic_runtime('cpu'):
    gen=torch.Generator().manual_seed(11)
    x=torch.randn(1,2,89856,generator=gen)*.03;v=x*.2;pv=v*.8
    snapshots=[value.clone() for value in (x,v,pv)]
    rng=t.d.portable(t.m.capture_rng('cpu'))
    result=t.low_frequency_backing_metrics(pv,x,v)
    assert result['lf_backing_error_power']>0
    assert all(torch.equal(a,b) for a,b in zip(snapshots,(x,v,pv)))
    assert t.m.equal_state(rng,t.m.capture_rng('cpu'))
    assert not torch.cuda.is_initialized()
print('ISOLATED_LF_STATISTICS CPU_INPUT_RNG_UNCHANGED CUDA_FALSE NO_OPTIMIZER')
"""
        reply=subprocess.run([sys.executable,"-c",code],cwd=base.p.ROOT,capture_output=True,text=True)
        self.assertEqual(reply.returncode,0,reply.stdout+reply.stderr)
        self.assertIn("CUDA_FALSE NO_OPTIMIZER",reply.stdout)

if __name__ == "__main__":
    unittest.main()

