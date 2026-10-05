"""CPU synthetic cache hash benchmark; no audio/model/PT/Adam/CUDA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location("ema229_fast228",ROOT/"scripts/228_ema_fast_hotpath.py")
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def benchmark(out):
    f.require(out.parent==ROOT/"results" and not out.exists(),"Fresh benchmark output required")
    out.mkdir()
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    threads=torch.get_num_threads();torch.set_num_threads(4)
    try:
        with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden')):
            spec=importlib.util.spec_from_file_location('ema229_original208',f.INPUT_PATH)
            old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
            #128 MiB represented by two synthetic64 MiB tensors, one strided.
            array=np.arange(16*1024*1024,dtype=np.float32)/1048576
            values=[torch.from_numpy(array.reshape(2,-1)),torch.from_numpy(array.reshape(-1,2).T)]
            timings={'baseline':[],'optimized':[]};expected=None
            # Alternate order across three rounds; report all samples/medians.
            for repeat in range(3):
                order=('baseline','optimized') if repeat%2==0 else ('optimized','baseline')
                for name in order:
                    fn=old.pcm_sha if name=='baseline' else f.fast_pcm_sha
                    start=time.perf_counter();digests=[fn(v) for v in values]
                    timings[name].append(time.perf_counter()-start)
                    if expected is None: expected=digests
                    f.require(digests==expected,'Contiguous/strided PCM commitments differ')
            f.require(expected==[hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for v in values],
                      'Independent C-order raw SHA differs')
            now=np.random.get_state();before=rng[1]
            f.require(rng[0]==random.getstate() and np.array_equal(now[1],before[1])
                      and now[0]==before[0] and now[2:]==before[2:] and torch.equal(rng[2],torch.get_rng_state())
                      and not torch.cuda.is_initialized(),'Whole CPU/Python/NumPy RNG and CUDA status')
            med={k:statistics.median(v) for k,v in timings.items()}
            result={'purpose':'NONRELEASE_EMA228_CPU_SYNTHETIC_CACHE_COMMITMENT_PERFORMANCE',
                    'source_bindings':{str(f.INPUT_PATH):f.INPUT_SHA,str(ROOT/'scripts/228_ema_fast_hotpath.py'):f.sha(ROOT/'scripts/228_ema_fast_hotpath.py')},
                    'synthetic_bytes_hashed_per_sample':sum(v.numel()*v.element_size() for v in values),
                    'CPU_threads':4,'samples_seconds':timings,'median_seconds':med,
                    'median_speedup':med['baseline']/med['optimized'],'all_SHA_bit_identical':True,
                    'PCM_sha256':expected,'all_elements_finite_checked_every_call':True,
                    'RNG_unchanged':True,'actual_audio_draws':0,'student_forward_Adam_updates':0,'CUDA':False,
                    'training_authorized':False,'release_selection':'NONE',
                    'limits':'Synthetic128MiB hash operation only,not whole step or end-to-end training throughput'}
            result['content_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            with (out/'benchmark.json').open('x',encoding='utf-8') as handle:
                json.dump(result,handle,ensure_ascii=True,indent=2,allow_nan=False)
            print(json.dumps(result,ensure_ascii=True,indent=2))
    finally:
        torch.set_num_threads(threads)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();benchmark(Path(args.out).resolve())
