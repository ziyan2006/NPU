"""New198 synthetic numerical-path tests; never run historical test suites."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch

spec=importlib.util.spec_from_file_location('split198',Path(__file__).with_name('198_diagnose_train_adam_split_numerics.py'))
n=importlib.util.module_from_spec(spec); spec.loader.exec_module(n)
torch.set_num_threads(2)


def toy_trace():
    param=torch.nn.Parameter(torch.tensor([.3,-.2]))
    wave=param[:,None].expand(2,32)*torch.linspace(.1,.8,32)
    residual=wave.sum().square(); accompaniment=(wave.square().sum()-1).square()
    base=wave.square().mean()
    before=param.detach().clone()
    doc=n.trace_paths((param,),wave,(residual,accompaniment,residual+accompaniment),base,(torch.ones(2),-torch.ones(2)*.1))
    assert torch.equal(param,before) and param.grad is None
    return doc


def skipped_rows():
    meta=[{'domain':domain,'role':'train' if i<3 else 'pseudo_label_train_candidate','vocal_db':0} for i,domain in enumerate(n.m.DOMAINS)]
    rows=[{'slot':i,'metadata':m,'bucket':n.k.bucket(m),'auxiliary':{'active':False},'identity':{'residual':0.,'accompaniment':0.,'full':0.},
           'numerics':None,'slot_forwards':1,'scalar_split_bit_exact':True,'unchanged_weights_modes_grad_rng_moments_inputs':True} for i,m in enumerate(meta)]
    return rows,{'expected_input':{'metadata':meta}}


class NumericalTests(unittest.TestCase):
    def test_scope(self): n.check_scope(n.scope())
    def test_protocol(self): self.assertTrue(n.k.exact(json.loads(n.PROTOCOL.read_text()),n.scope()))
    def test_count(self): self.assertEqual((n.scope()['unique_input_batches'],n.scope()['slot_forwards']),(1,6))
    def test_counter_type(self):
        with self.assertRaises(ValueError): n.check_scope(n.scope()|{'counter':4500.})
    def test_boolean_count(self):
        with self.assertRaises(ValueError): n.check_scope(n.scope()|{'unique_input_batches':True})
    def test_threshold(self):
        with self.assertRaises(ValueError): n.check_scope(n.scope()|{'atol':2e-6})
    def test_fp64_rejected(self):
        with self.assertRaises(ValueError): n.vector_comparison(torch.ones(2).double(),torch.ones(2).double())
    def test_shape_rejected(self):
        with self.assertRaises(ValueError): n.vector_comparison(torch.ones(2),torch.ones(3))
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): n.vector_comparison(torch.zeros(0),torch.zeros(0))
    def test_matrix_shape_rejected(self):
        with self.assertRaises(ValueError): n.vector_comparison(torch.ones(2,2),torch.ones(2,2))
    def test_nonfinite(self):
        with self.assertRaises(ValueError): n.vector_comparison(torch.tensor([float('nan')]),torch.zeros(1))
    def test_exact(self):
        doc=n.vector_comparison(torch.ones(2),torch.ones(2)); self.assertTrue(doc['old_strict_pass']); self.assertEqual(doc['max_abs_error'],0.)
    def test_old_threshold_failure_retained(self):
        a=torch.tensor([8.8e-5+4.33e-7]); e=torch.tensor([8.8e-5]); doc=n.vector_comparison(a,e)
        self.assertFalse(doc['old_strict_pass']); self.assertEqual(doc['mismatched_elements'],1)
    def test_zero_norm(self): self.assertIsNone(n.vector_comparison(torch.zeros(2),torch.zeros(2))['cosine'])
    def test_zero_actual(self): self.assertIsNone(n.vector_comparison(torch.zeros(2),torch.ones(2))['cosine'])
    def test_all_mismatches_saved(self):
        doc=n.vector_comparison(torch.ones(12),torch.zeros(12)); self.assertEqual([x['index'] for x in doc['all_mismatched_elements']],list(range(12)))
    def test_cancellation(self):
        doc=n.vector_comparison(torch.ones(2),torch.zeros(2),(torch.ones(2),-torch.ones(2)))
        self.assertEqual(doc['all_mismatched_elements'][0]['cancellation_ratio'],0.)
    def test_zero_terms(self):
        doc=n.vector_comparison(torch.ones(2),torch.zeros(2),(torch.zeros(2),torch.zeros(2)))
        self.assertIsNone(doc['all_mismatched_elements'][0]['cancellation_ratio'])
    def test_dot_error(self):
        doc=n.vector_comparison(torch.ones(2),torch.zeros(2),direction=(torch.ones(2),-torch.ones(2)))
        self.assertEqual(doc['stored_direction_dot']['error_dot_u'],2.); self.assertEqual(doc['stored_direction_dot']['error_dot_d'],-2.)
    def test_changed_dot(self):
        doc=n.vector_comparison(torch.ones(2),torch.zeros(2),direction=(torch.ones(2),-torch.ones(2))); doc['stored_direction_dot']['error_dot_d']=2.
        with self.assertRaises(ValueError): n.validate_comparison(doc)
    def test_changed_pass(self):
        doc=n.vector_comparison(torch.ones(2),torch.zeros(2)); doc['old_strict_pass']=True
        with self.assertRaises(ValueError): n.validate_comparison(doc)
    def test_changed_count_type(self):
        doc=n.vector_comparison(torch.ones(2),torch.zeros(2)); doc['mismatched_elements']=2.
        with self.assertRaises(ValueError): n.validate_comparison(doc)
    def test_vector_hash_required(self):
        doc=n.vector_comparison(torch.ones(2),torch.ones(2)); doc['actual_sha256']='bad'
        with self.assertRaises(ValueError): n.validate_comparison(doc)
    def test_changed_cancellation(self):
        doc=n.vector_comparison(torch.ones(2),torch.zeros(2),(torch.ones(2),-torch.ones(2))); doc['all_mismatched_elements'][0]['cancellation_ratio']=.5
        with self.assertRaises(ValueError): n.validate_comparison(doc)
    def test_original_scalar_wave_new_exposure(self):
        # Reuse only sealed synthetic fixture functions, not any old tests.
        sp=importlib.util.spec_from_file_location('readonly_fixture',Path(__file__).with_name('_test_train_adam_memory.py'))
        fixture=importlib.util.module_from_spec(sp); sp.loader.exec_module(fixture)
        batch=fixture.fixture(); net=fixture.Tiny(); wa,gs,_=n.k.matrix_identity(); meta=batch['metadata'][0]
        base,wave,losses,info,identity=n.slot_wave(net,batch['x'][:1],batch['v'][:1],wa,gs,meta)
        self.assertEqual(net.calls,1)
        old=n.k.slot_losses(net,batch['x'][:1],batch['v'][:1],wa,gs,meta)
        self.assertTrue(torch.equal(base,old[0])); self.assertEqual(identity['wave_sha256'],old[5]['wave_sha256']); self.assertEqual(info,old[4])
        self.assertEqual(wave.dtype,torch.float32); self.assertTrue(torch.equal(losses[2],losses[0]+losses[1]))
    def test_toy_same_forward_vjps(self):
        doc=toy_trace(); self.assertEqual(set(doc['comparisons']),set(n.COMPARISONS)); self.assertEqual(doc['forward_count_in_trace'],0)
        self.assertTrue(doc['comparisons']['parameter_full_vs_vjp_full']['old_strict_pass'])
    def test_toy_no_grad_storage(self): self.assertFalse(toy_trace()['parameter_grad_storage_written'])
    def test_legal_gains(self):
        for domain in ('musdb','mir1k'):
            for gain in (-12,-6,0,6): self.assertTrue(n.k.bucket({'domain':domain,'role':'train','vocal_db':gain}))
    def test_boolean_gain_rejected(self):
        with self.assertRaises(ValueError): n.k.bucket({'domain':'musdb','role':'train','vocal_db':True})
    def test_pseudo_not_truth(self): self.assertEqual(n.k.bucket({'domain':'pseudo','role':'pseudo_label_train_candidate'}),'pseudo_not_final_truth')
    def test_instrumental_skip(self): self.assertEqual(n.k.bucket({'domain':'instrumental','role':'train','vocal_db':0}),'instrumental_zero_reference')
    def test_wrong_pseudo_role(self):
        with self.assertRaises(ValueError): n.k.bucket({'domain':'pseudo','role':'train'})
    def test_same_literal_lf32(self): self.assertEqual(n.k.matrix_identity()[2]['forced_zero_fft_bins'],[0,1,2,3])
    def test_existing_output_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'rows').mkdir()
            with self.assertRaises(ValueError): n.run(out)
    def test_symmetric_rows_types_and_seals(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'rows').mkdir(); rows=[]
            for i in range(6):
                doc=n.acq.seal({'slot':i,'value':1}); n.acq.write_new_json(out/'rows'/f'row_{i:02d}.json',doc); rows.append(doc)
            n.symmetric_rows(out,rows)
            changed=copy.deepcopy(rows); changed[0]['value']=True; changed[0]=n.acq.seal(changed[0])
            with self.assertRaises(ValueError): n.symmetric_rows(out,changed)
            changed=copy.deepcopy(rows); changed[0]['value']=2
            with self.assertRaises(ValueError): n.symmetric_rows(out,changed)
    def test_partial_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'rows').mkdir()
            with self.assertRaises(ValueError): n.symmetric_rows(out,[])
    def test_fresh_cpu_no_adam_cuda(self):
        code='import importlib.util,torch; s=importlib.util.spec_from_file_location("fresh198",'+repr(str(Path(n.__file__)))+'); n=importlib.util.module_from_spec(s); s.loader.exec_module(n); assert not torch.cuda.is_initialized(); n.vector_comparison(torch.ones(2),torch.ones(2)); assert not torch.cuda.is_initialized()'
        done=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True); self.assertEqual(done.returncode,0,done.stderr)
    def test_complete_six_skips(self):
        rows,plan=skipped_rows(); n.validate_rows(rows,plan)
    def test_partial_six_rejected(self):
        rows,plan=skipped_rows(); rows.pop()
        with self.assertRaises(ValueError): n.validate_rows(rows,plan)
    def test_slot_count_type(self):
        rows,plan=skipped_rows(); rows[0]['slot_forwards']=True
        with self.assertRaises(ValueError): n.validate_rows(rows,plan)
    def test_skip_nonzero_rejected(self):
        rows,plan=skipped_rows(); rows[2]['identity']['full']=.1
        with self.assertRaises(ValueError): n.validate_rows(rows,plan)
    def test_active_paths_required(self):
        rows,plan=skipped_rows(); rows[0]['auxiliary']['active']=True
        with self.assertRaises(ValueError): n.validate_rows(rows,plan)
    def test_instrumental_not_active(self):
        rows,plan=skipped_rows(); rows[2]['auxiliary']['active']=True
        with self.assertRaises(ValueError): n.validate_rows(rows,plan)
    def test_metadata_change_rejected(self):
        rows,plan=skipped_rows(); rows=copy.deepcopy(rows); rows[0]['metadata']['vocal_db']=6
        with self.assertRaises(ValueError): n.validate_rows(rows,plan)
    def test_binding_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bound.json'; n.acq.write_new_json(path,{'x':1}); stamps=n.k.binding_stamps([path])
            os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1000000000))
            with self.assertRaises(ValueError): n.k.unchanged_bindings(stamps)
    def test_all_state_existing_grad_preserved(self):
        net=torch.nn.Linear(2,2,bias=False); params=tuple(net.parameters()); params[0].grad=torch.ones_like(params[0])
        saved={'moments':torch.ones(2),'cursor':4500,'schedule':{'stopped_at':4500}}; batch={'x':torch.ones(2)}
        with n.k.readonly(net,saved,batch):
            wave=net(torch.ones(1,2)).repeat(2,16); base=wave.square().mean(); a=wave.sum().square(); b=(wave.square().sum()-1).square()
            doc=n.trace_paths(params,wave,(a,b,a+b),base,(torch.ones(4),-torch.ones(4)))
        self.assertTrue(torch.equal(params[0].grad,torch.ones_like(params[0]))); self.assertFalse(doc['parameter_grad_storage_written'])
    def test_source_state_change_rejected(self):
        net=torch.nn.Linear(2,2); saved={'cursor':4500}
        with self.assertRaises(ValueError):
            with n.k.readonly(net,saved,{}): saved['cursor']=4501


if __name__=='__main__': unittest.main(verbosity=2)
