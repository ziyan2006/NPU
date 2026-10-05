"""New200 zero-compute serialization regressions; no old unit/forward reruns."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

spec=importlib.util.spec_from_file_location('evidence200',Path(__file__).with_name('200_verify_train_adam_bridge_evidence.py'))
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def fixture():
    # Read-only JSON fixture. No model loading, frontend, forward or gradients.
    return tuple(json.loads((v.b.OUT/name).read_text(encoding='utf-8')) for name in
        ('plan.json','inputs.json','totals.json','diagnostic.json','run_status.json'))


class EvidenceTests(unittest.TestCase):
    def test_scope(self): v.check_scope(v.scope())
    def test_protocol(self): self.assertTrue(v.k.exact(json.loads(v.PROTOCOL.read_text()),v.scope()))
    def test_scope_float_counter(self):
        with self.assertRaises(ValueError): v.check_scope(v.scope()|{'source_counter':4500.})
    def test_scope_bool_count(self):
        with self.assertRaises(ValueError): v.check_scope(v.scope()|{'model_forwards':False})
    def test_scope_no_threshold_relaxation(self):
        with self.assertRaises(ValueError): v.check_scope(v.scope()|{'atol':2e-6})
    def test_mutating_seal_regression(self):
        d={'total':1}; embedded=d; outside=copy.deepcopy(v.acq.seal(d))
        self.assertIn('content_sha256',embedded)
        self.assertFalse(v.k.exact({k:x for k,x in outside.items() if k!='content_sha256'},embedded))
        v.symmetric_totals(outside,embedded)
    def test_both_valid_same_payload(self):
        a=v.acq.seal({'total':1,'unicode':'证明'}); v.symmetric_totals(a,copy.deepcopy(a))
    def test_both_valid_different_payload(self):
        with self.assertRaises(ValueError): v.symmetric_totals(v.acq.seal({'x':1}),v.acq.seal({'x':2}))
    def test_invalid_outside_seal(self):
        a=v.acq.seal({'x':1}); c=copy.deepcopy(a); c['x']=2
        with self.assertRaises(ValueError): v.symmetric_totals(c,a)
    def test_invalid_inside_seal(self):
        a=v.acq.seal({'x':1}); c=copy.deepcopy(a); c['x']=2
        with self.assertRaises(ValueError): v.symmetric_totals(a,c)
    def test_missing_inside_seal(self):
        with self.assertRaises(ValueError): v.symmetric_totals(v.acq.seal({'x':1}),{'x':1})
    def test_bool_int_different(self):
        with self.assertRaises(ValueError): v.symmetric_totals(v.acq.seal({'x':1}),v.acq.seal({'x':True}))
    def test_float_int_different(self):
        with self.assertRaises(ValueError): v.symmetric_totals(v.acq.seal({'x':1}),v.acq.seal({'x':1.}))
    def test_full_end_to_end_serialized_fixture(self):
        args=fixture()
        with v.no_compute(): v.validate_payload(*args)
    def test_original_one_sided_failure_fixture(self):
        _,_,t,r,_=fixture(); self.assertFalse(v.k.exact({k:x for k,x in t.items() if k!='content_sha256'},r['totals']))
    def test_old_split_failure_remains(self):
        args=fixture(); v.validate_payload(*args)
        self.assertFalse(args[3]['rows'][1]['probe']['paths']['parameter_full_vs_parts']['old_strict_pass'])
    def test_status_incomplete(self):
        p,i,t,r,s=fixture(); s['committed_reference_rows']=5
        with self.assertRaises(ValueError): v.validate_payload(p,i,t,r,s)
    def test_status_count_type(self):
        p,i,t,r,s=fixture(); s['counts']['primary_started']=6.
        with self.assertRaises(ValueError): v.validate_payload(p,i,t,r,s)
    def test_rng_guard_false(self):
        p,i,t,r,s=fixture(); i['rng_unchanged']=False
        with self.assertRaises(ValueError): v.validate_payload(p,i,t,r,s)
    def test_input_changed(self):
        p,i,t,r,s=fixture(); i['input']['counter']=4501
        with self.assertRaises(ValueError): v.validate_payload(p,i,t,r,s)
    def test_totals_extra_valid_field_rejected(self):
        p,i,t,r,s=fixture(); v.acq.seal(t|{'unexpected':True})
        with self.assertRaises(ValueError): v.validate_payload(p,i,v.acq.seal(t|{'unexpected':True}),r,s)
    def test_full_vjp_bit_exact_gate_not_relaxed(self):
        p,i,t,r,s=fixture(); r['rows'][0]['probe']['paths']['parameter_full_vs_vjp_full']['max_abs_error']=1e-10
        with self.assertRaises(ValueError): v.validate_payload(p,i,t,r,s)
    def test_primary_reference_wave_identity(self):
        p,i,t,r,s=fixture(); r['reference_rows'][0]['identity']['wave_sha256']='0'*64
        with self.assertRaises(ValueError): v.validate_payload(p,i,t,r,s)
    def test_symmetric_row_seals_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); (out/'rows').mkdir(); rows=[]
            for i in range(6):
                row=v.acq.seal({'slot':i,'value':1}); rows.append(row)
                v.acq.write_new_json(out/'rows'/f'row_{i:02d}.json',row)
            v.b.symmetric_rows(out,rows)
            changed=copy.deepcopy(rows); changed[0],changed[1]=changed[1],changed[0]
            with self.assertRaises(ValueError): v.b.symmetric_rows(out,changed)
            changed=copy.deepcopy(rows); changed[0]=v.acq.seal({'slot':0,'value':True})
            with self.assertRaises(ValueError): v.b.symmetric_rows(out,changed)
    def test_zero_compute_interceptions(self):
        net=torch.nn.Linear(1,1)
        for call in (lambda:net(torch.ones(1)),lambda:torch.autograd.grad(None,None),
                     lambda:torch.optim.Adam([]),lambda:torch.cuda.init()):
            with self.assertRaises(AssertionError):
                with v.no_compute(): call()
    def test_existing_proof_rejected_before_compute(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); v.acq.write_new_json(out/'verification.json',{})
            with self.assertRaises(ValueError): v.verify(out)
    def test_failed_binding_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'doc.json'; v.acq.write_new_json(p,{'x':1})
            with self.assertRaises(ValueError): v.k.p.q.check_bindings({str(p):'0'*64})
    def test_fresh_cpu_no_optimizer_no_cuda(self):
        code="import importlib.util,torch; from pathlib import Path; s=importlib.util.spec_from_file_location('fresh',Path(r'"+str(v.__file__)+"')); v=importlib.util.module_from_spec(s); s.loader.exec_module(v); assert not torch.cuda.is_initialized();\nwith v.no_compute(): v.validate_payload(*"+"tuple(__import__('json').loads((v.b.OUT/n).read_text()) for n in ('plan.json','inputs.json','totals.json','diagnostic.json','run_status.json')))\nassert not torch.cuda.is_initialized(); print('ZERO_COMPUTE_CPU_NO_CUDA')"
        proc=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
        self.assertIn('ZERO_COMPUTE_CPU_NO_CUDA',proc.stdout)


if __name__=='__main__': unittest.main(verbosity=2)
