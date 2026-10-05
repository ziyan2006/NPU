"""Fresh CPU synthetic tests; no actual music, PT, model, Adam or CUDA."""
import hashlib
import importlib.util
from collections import OrderedDict
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location("ema228_units",ROOT/"scripts/228_ema_fast_hotpath.py")
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class FastCommitmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        cls.threads=torch.get_num_threads();torch.set_num_threads(4)
        cls.cuda=patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("No CUDA"));cls.cuda.start()
        cls.child=patch.object(subprocess,"Popen",side_effect=AssertionError("No decoder in228 CPU units"));cls.child.start()
        require_file=f.sha(f.INPUT_PATH)==f.INPUT_SHA
        assert require_file
        spec=importlib.util.spec_from_file_location("ema228_baseline_input",f.INPUT_PATH)
        cls.old=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.old)
        cls.fast=staticmethod(f.fast_cache_identity(cls.old))

    @classmethod
    def tearDownClass(cls):
        cls.child.stop();cls.cuda.stop()
        assert not torch.cuda.is_initialized()
        assert cls.rng[0]==random.getstate()
        now=np.random.get_state(); before=cls.rng[1]
        assert now[0]==before[0] and np.array_equal(now[1],before[1]) and now[2:]==before[2:]
        assert torch.equal(cls.rng[2],torch.get_rng_state())
        torch.set_num_threads(cls.threads)
        print("NEW228_UNITS CPU/Python/NumPy RNG unchanged; real_audio/PT/model/Adam/CUDA=0")

    def compare(self,value):
        before=value.numpy().tobytes()
        self.assertEqual(f.fast_pcm_sha(value),self.old.pcm_sha(value))
        self.assertEqual(before,value.numpy().tobytes())

    def test_contiguous_float32_exact(self):
        self.compare(torch.arange(4096,dtype=torch.float32).reshape(2,2048)/16384)

    def test_strided_transpose_exact(self):
        self.compare(torch.arange(8192,dtype=torch.float32).reshape(4096,2).T)

    def test_sliced_strides_exact(self):
        self.compare(torch.arange(8192,dtype=torch.float32).reshape(2,4096)[:,::3])

    def test_zero_scalar_empty_exact(self):
        for value in (torch.tensor([-0.,0.]),torch.tensor(-0.),torch.empty(0,dtype=torch.float32)):
            self.compare(value)

    def test_subnormal_and_extreme_finite_exact(self):
        self.compare(torch.tensor([torch.finfo(torch.float32).tiny,torch.finfo(torch.float32).max,1e-44,-1e-44]))

    def test_chunk_boundary_exact(self):
        self.compare(torch.arange(f.FINITE_CHUNK_ELEMENTS+17,dtype=torch.float32)/1048576)

    def test_all_nonfinite_rejected(self):
        for bad in (float('nan'),float('inf'),-float('inf')):
            self.assertRaises(ValueError,f.fast_pcm_sha,torch.tensor([0.,bad]))

    def test_tail_nonfinite_rejected(self):
        value=torch.zeros(f.FINITE_CHUNK_ELEMENTS+1)
        value[-1]=float('nan')
        self.assertRaises(ValueError,f.fast_pcm_sha,value)

    def test_wrong_dtype_and_trainable_rejected(self):
        for value in (torch.zeros(2,dtype=torch.float64),torch.zeros(2,dtype=torch.int32),
                      torch.zeros(2,requires_grad=True),np.zeros(2,dtype=np.float32)):
            self.assertRaises(ValueError,f.fast_pcm_sha,value)

    def test_inplace_and_data_mutations_are_not_memoized(self):
        value=torch.tensor([0.,1.]);before=f.fast_pcm_sha(value)
        value.data[0]=-0.
        self.assertNotEqual(before,f.fast_pcm_sha(value))
        before=f.fast_pcm_sha(value);value.numpy()[1]=2.
        self.assertNotEqual(before,f.fast_pcm_sha(value))

    def test_original_cache_order_and_complete_signatures(self):
        a=torch.arange(32,dtype=torch.float32).reshape(2,16)
        backend=SimpleNamespace(true=SimpleNamespace(cache=OrderedDict([('a',(a,a.T)),('b',(a*.25,))]),
                                      signatures={'a':('file',8,9)}),
                                dataset=SimpleNamespace(cache=OrderedDict([('p',a[:,::2])])))
        self.assertEqual(self.fast(backend),self.old.AuthenticatedAudioStream._cache_identity(backend))
        before=self.fast(backend);backend.true.cache.move_to_end('a')
        self.assertNotEqual(before,self.fast(backend))

    def test_original_cache_mutation_detected(self):
        a=torch.ones((2,16))
        backend=SimpleNamespace(true=SimpleNamespace(cache=OrderedDict([('a',(a,))]),signatures={}),
                                dataset=SimpleNamespace(cache=OrderedDict()))
        before=self.fast(backend);a.numpy()[0,0]=.25
        self.assertNotEqual(before,self.fast(backend))

    def test_original_module_helper_unchanged(self):
        self.assertIs(self.old.AuthenticatedAudioStream._cache_identity.__globals__['pcm_sha'],self.old.pcm_sha)
        self.assertIs(self.fast.__globals__['pcm_sha'],f.fast_pcm_sha)


if __name__=='__main__':
    unittest.main(verbosity=2)
