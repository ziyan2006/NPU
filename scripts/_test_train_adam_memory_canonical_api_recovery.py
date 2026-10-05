"""New202 isolated API recovery synthetic gates only; never rerun old tests or real TRAIN probes."""
import copy
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

spec=importlib.util.spec_from_file_location('new202',Path(__file__).with_name('202_diagnose_train_adam_memory_canonical_api_recovery.py'))
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
torch.set_num_threads(2)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor([.1,-.1,.05,-.05])); self.calls=0
    def forward(self,bands):
        self.calls+=1; return self.weight.tanh().reshape(1,4,1,1).expand(1,4,128,352)


def saved(net=None):
    net=net or Tiny()
    return {'model':copy.deepcopy(net.state_dict()),'parameter_names':['weight'],'modes':[mod.training for mod in net.modules()],
        'updates':10,'optimizer':{'param_groups':[{'lr':.01,'betas':(.9,.999),'eps':1e-8,'weight_decay':0,
        'amsgrad':False,'maximize':False,'foreach':False,'capturable':False,'differentiable':False,'fused':False,
        'decoupled_weight_decay':False,'params':[0]}], 'state':{0:{'step':torch.tensor(10.),
        'exp_avg':torch.tensor([.1,-.2,.3,-.4]),'exp_avg_sq':torch.tensor([.01,.03,.02,.04])}}}}


def batch(counter=4500):
    time=torch.arange(89856,dtype=torch.float32)
    a=.15*torch.sin(time*.07).repeat(2,1); v=.04*torch.cos(time*.031).repeat(2,1)
    xs=[]; vs=[]; metas=[]
    for i,domain in enumerate(c.m.DOMAINS):
        target=torch.zeros_like(v) if i==2 else v*(i+1)/2; mix=a+target; xs.append(mix); vs.append(target)
        metas.append({'domain':domain,'role':'train' if i<3 else 'pseudo_label_train_candidate','vocal_db':0,
            'track_id':'SYNTHETIC_UNIT_NOT_REAL_TRAIN','input_pcm_sha256':c.m.pilot.wave_digest(mix),'score_start':25088,'score_end':89344})
    return {'x':torch.stack(xs),'v':torch.stack(vs),'metadata':metas,'diagnostic_counter':counter}


@lru_cache(maxsize=1)
def fixture():
    counts=c.new_counts(); rows=[]; descriptors={}; inputs=[c.k.input_entry(batch(v)) for v in c.k.CURSORS]
    for name in c.k.MODELS:
        net=Tiny(); s=saved(net); net.weight.grad=torch.ones_like(net.weight)*.17
        direction=c.k.stored_direction(s)[2]; descriptors[name]={'direction':direction}
        for index,counter in enumerate(c.k.CURSORS):
            x=batch(counter); wa,gs,_=c.k.matrix_identity()
            probe=c.probe_batch(net,s,x,wa,gs,index==0,counts)
            row=c.acq.seal({'model':name,'counter':counter,**probe}); c.validate_batch(row,descriptors[name],inputs[index],index==0); rows.append(row)
        assert net.calls==24 and torch.equal(net.weight.grad,torch.ones(4)*.17)
    plan=c.acq.seal(c.scope()|{'model_descriptors':descriptors,'expected_inputs':inputs,'expected_runtime':{'device':'cpu'}})
    result=c.acq.seal(c.scope()|{'rows':rows,'counts':counts,'runtime':plan['expected_runtime'],'source_state_unchanged':True,'error':None})
    c.validate_result(result,plan)
    return plan,result


def serialize(out):
    plan,result=copy.deepcopy(fixture())
    for name in ('slots','references','totals','rows'): (out/name).mkdir()
    c.acq.write_new_json(out/'plan.json',plan)
    inp=c.acq.seal({'purpose':c.PURPOSE,'inputs':plan['expected_inputs'],'rng_unchanged':True}); c.acq.write_new_json(out/'inputs.json',inp)
    result.update(plan_sha256=c.acq.sha256(out/'plan.json'),inputs_sha256=c.acq.sha256(out/'inputs.json'))
    for i,row in enumerate(result['rows']):
        c.acq.write_new_json(out/'rows'/f'row_{i:02d}.json',row)
        c.acq.write_new_json(out/'totals'/f'batch_{i:02d}.json',row['totals'])
        for directory,key in (('slots','slots'),('references','reference_rows')):
            for j,doc in enumerate(row[key]): c.acq.write_new_json(out/directory/f'batch_{i:02d}_slot_{j:02d}.json',doc)
    c.acq.seal(result); c.acq.write_new_json(out/'diagnostic.json',result)
    return plan,result


class Gates(unittest.TestCase):
    def test_scope(self): c.check_scope(c.scope())
    def test_retained197_actual_default_out(self):
        self.assertFalse(hasattr(c.k,'OUT')); self.assertTrue(hasattr(c.k,'DEFAULT_OUT'))
        self.assertTrue((c.k.DEFAULT_OUT/'inputs.json').is_file())
    def test_actual_retained_input_source_interface_no_forward(self):
        before=c.m.capture_rng('cpu')
        with patch.object(c.k,'source_evidence',return_value=({}, {}, {}, {'cursor':4500}, {})),patch.object(c.k.p.q,'check_bindings'):
            _,_,_,sampler,expected,bindings=c.source()
        self.assertEqual(sampler['cursor'],4500); self.assertEqual([x['counter'] for x in expected],[4500,4501,4502])
        self.assertEqual(expected,c.acq.read_sealed(c.k.DEFAULT_OUT/'inputs.json')['inputs'])
        self.assertEqual(c.k.tree_digest(before),c.k.tree_digest(c.m.capture_rng('cpu')))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertIn(str((c.ROOT/'reports/99_train_adam_canonical_recovery_prepare_failure.md').resolve()),bindings)
    def test_original201_untouched(self):
        failure=c.acq.read_sealed(c.ROOT/'results/train_adam_memory_canonical_recovery_monitor_20261004/preparation_failure_review.json')
        for name,digest in failure['immutable_bindings'].items(): self.assertEqual(c.acq.sha256(name),digest)
    def test_protocol(self): self.assertTrue(c.k.exact(c.scope(),json.loads(c.PROTOCOL.read_text())))
    def test_budget(self): self.assertEqual((c.scope()['model_batches'],c.scope()['primary_slot_forwards'],c.scope()['reference_slot_forwards']),(9,54,18))
    def test_unique(self): self.assertEqual((c.scope()['unique_input_batches'],c.scope()['unique_input_slots']),(3,18))
    def test_scope_bool(self):
        with self.assertRaises(ValueError): c.check_scope(c.scope()|{'model_batches':True})
    def test_scope_float(self):
        with self.assertRaises(ValueError): c.check_scope(c.scope()|{'counters':[4500.,4501,4502]})
    def test_no_relaxation(self):
        with self.assertRaises(ValueError): c.check_scope(c.scope()|{'atol':2e-6})
    def test_fp64(self):
        with self.assertRaises(ValueError): c.k.cpu_float(torch.ones(2).double())
    def test_shape(self):
        with self.assertRaises(ValueError): c.k.cpu_float(torch.ones(2),(3,))
    def test_finite(self):
        with self.assertRaises(ValueError): c.k.cpu_float(torch.tensor([float('nan')]))
    def test_matrix(self): self.assertEqual(c.k.matrix_identity()[2]['forced_zero_fft_bins'],[0,1,2,3])
    def test_gains(self):
        for domain in ('musdb','mir1k'):
            for gain in (-12,-6,0,6): self.assertTrue(c.k.bucket({'domain':domain,'role':'train','vocal_db':gain}))
    def test_gain_bool(self):
        with self.assertRaises(ValueError): c.k.bucket({'domain':'musdb','role':'train','vocal_db':True})
    def test_gain_bad(self):
        with self.assertRaises(ValueError): c.k.bucket({'domain':'mir1k','role':'train','vocal_db':12})
    def test_pseudo(self): self.assertEqual(c.k.bucket({'domain':'pseudo','role':'pseudo_label_train_candidate'}),'pseudo_not_final_truth')
    def test_pseudo_truth(self):
        with self.assertRaises(ValueError): c.k.bucket({'domain':'pseudo','role':'train'})
    def test_metadata_order(self):
        x=batch(); x['metadata'].reverse()
        with self.assertRaises(ValueError): c.k.input_entry(x)
    def test_zero_reference(self): self.assertEqual(c.k.bucket(batch()['metadata'][2]),'instrumental_zero_reference')
    def test_id_name(self): self.assertEqual(c.k.stored_direction(saved())[2]['parameter_mapping'][0]['parameter_name'],'weight')
    def test_id_bool(self):
        s=saved(); s['optimizer']['param_groups'][0]['params']=[False]
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_id_order(self):
        s=saved(); s['optimizer']['param_groups'][0]['params']=[1]
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_name_missing(self):
        s=saved(); s['parameter_names']=['missing']
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_moment_shape(self):
        s=saved(); s['optimizer']['state'][0]['exp_avg']=torch.ones(3)
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_negative_variance(self):
        s=saved(); s['optimizer']['state'][0]['exp_avg_sq'][0]=-1
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_step(self):
        s=saved(); s['optimizer']['state'][0]['step']=torch.tensor(9.)
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_step_dtype(self):
        s=saved(); s['optimizer']['state'][0]['step']=torch.tensor(10)
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_beta(self):
        s=saved(); s['optimizer']['param_groups'][0]['betas']=[.9,.999]
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_flags(self):
        for flag,value in (('foreach',True),('fused',True),('maximize',True),('amsgrad',True),('weight_decay',0.)):
            s=saved(); s['optimizer']['param_groups'][0][flag]=value
            with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_eps(self):
        s=saved(); s['optimizer']['param_groups'][0]['eps']=1e-7
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_lr(self):
        s=saved(); s['optimizer']['param_groups'][0]['lr']=True
        with self.assertRaises(ValueError): c.k.stored_direction(s)
    def test_scalar_bias_oracle(self):
        s=saved(); u,d,_=c.k.stored_direction(s); st=s['optimizer']['state'][0]
        oracle=[(float(a)/(1-.9**10))/(math.sqrt(float(v)/(1-.999**10))+1e-8) for a,v in zip(st['exp_avg'],st['exp_avg_sq'])]
        torch.testing.assert_close(u,torch.tensor(oracle),rtol=2e-7,atol=1e-8); self.assertTrue(torch.equal(d,-.01*u))
    def test_no_alias(self):
        s=saved(); before=c.k.tree_digest(s); u,d,_=c.k.stored_direction(s); u.zero_(); d.zero_(); self.assertEqual(before,c.k.tree_digest(s))
    def test_zero_norm(self): self.assertIsNone(c.k.alignment(torch.zeros(2),torch.ones(2),-torch.ones(2))['cos_g_u'])
    def test_sign(self): self.assertTrue(c.k.alignment(torch.ones(2),torch.ones(2),-torch.ones(2))['first_order_descent'])
    def test_ascent(self): self.assertFalse(c.k.alignment(torch.ones(2),-torch.ones(2),torch.ones(2))['first_order_descent'])
    def test_readonly_existing_grad(self):
        net=Tiny(); net.weight.grad=torch.ones(4); s=saved(net)
        with c.k.readonly(net,s,{}): torch.autograd.grad(net.weight.square().sum(),tuple(net.parameters()))
        self.assertTrue(torch.equal(net.weight.grad,torch.ones(4)))
    def test_readonly_weights(self):
        net=Tiny()
        with self.assertRaises(ValueError):
            with c.k.readonly(net,saved(net),{}),torch.no_grad(): net.weight.add_(1)
    def test_readonly_grad_mutation(self):
        net=Tiny(); net.weight.grad=torch.ones(4)
        with self.assertRaises(ValueError):
            with c.k.readonly(net,saved(net),{}): net.weight.grad.add_(1)
    def test_readonly_modes(self):
        net=Tiny()
        with self.assertRaises(ValueError):
            with c.k.readonly(net,saved(net),{}): net.eval()
    def test_readonly_moments(self):
        net=Tiny(); s=saved(net)
        with self.assertRaises(ValueError):
            with c.k.readonly(net,s,{}): s['optimizer']['state'][0]['exp_avg'].add_(1)
    def test_readonly_input(self):
        net=Tiny(); x={'x':torch.zeros(2)}
        with self.assertRaises(ValueError):
            with c.k.readonly(net,saved(net),x): x['x'].add_(1)
    def test_readonly_rng(self):
        net=Tiny(); rng=c.m.capture_rng('cpu')
        try:
            with self.assertRaises(ValueError):
                with c.k.readonly(net,saved(net),{}): torch.rand(1)
        finally: c.m.restore_rng(rng,'cpu')
    def test_readonly_schedule(self):
        net=Tiny(); s=saved(net)|{'cursor':4500,'schedule':{'step':4500,'stopped_at':4500}}
        with self.assertRaises(ValueError):
            with c.k.readonly(net,s,{}): s['schedule']['step']=4501
    def test_optimizer_prohibited(self):
        with self.assertRaises(AssertionError):
            with c.no_optimizer_cuda(): torch.optim.Adam(Tiny().parameters())
    def test_cuda_prohibited(self):
        with self.assertRaises(AssertionError):
            with c.no_optimizer_cuda(): torch.cuda.init()
    def test_fresh_subprocess_no_cuda(self):
        code="import importlib.util,torch; s=importlib.util.spec_from_file_location('c',r'"+str(c.__file__)+"'); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); assert not torch.cuda.is_initialized(); c.k.stored_direction({'model':{'p':torch.ones(1)},'parameter_names':['p'],'updates':1,'optimizer':{'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.ones(1),'exp_avg_sq':torch.ones(1)}},'param_groups':[{'lr':.01,'betas':(.9,.999),'eps':1e-8,'weight_decay':0,'amsgrad':False,'maximize':False,'foreach':False,'capturable':False,'differentiable':False,'fused':False,'decoupled_weight_decay':False,'params':[0]}]}}); assert not torch.cuda.is_initialized()"
        self.assertEqual(subprocess.run([sys.executable,'-c',code],capture_output=True).returncode,0)
    def test_actual_counts(self): self.assertEqual(fixture()[1]['counts'],{'primary_started':54,'primary_completed':54,'reference_started':18,'reference_completed':18})
    def test_original_reference(self):
        for row in fixture()[1]['rows'][::3]: c.b.require_comparison(row['totals']['authority_vs_reference'],exact=True)
    def test_vjp(self):
        for row in fixture()[1]['rows']:
            for slot in row['slots'][:2]: c.b.require_comparison(slot['probe']['paths']['parameter_full_vs_vjp_full'],exact=True)
    def test_skips(self):
        for slot in fixture()[1]['rows'][0]['slots'][2:]: self.assertFalse(slot['auxiliary']['active']); self.assertIsNone(slot['probe']['paths'])
    def test_old_split_failure_retained_not_gate(self):
        doc=copy.deepcopy(fixture()[1]['rows'][0]['slots'][0]['probe'])
        doc['paths']['parameter_full_vs_parts']=c.n.vector_comparison(torch.ones(4),torch.zeros(4))
        c.b.check_primary(doc,True); self.assertEqual(doc['paths']['parameter_full_vs_parts']['mismatched_elements'],4)
    def test_canonical_gate(self):
        doc=copy.deepcopy(fixture()[1]['rows'][0]['slots'][0]['probe']); doc['canonical_full_vs_direct']=c.n.vector_comparison(torch.ones(4),torch.zeros(4))
        with self.assertRaises(ValueError): c.b.check_primary(doc,True)
    def test_exact_not_strict(self):
        with self.assertRaises(ValueError): c.b.require_comparison(c.n.vector_comparison(torch.tensor([1.+1e-7]),torch.ones(1)),exact=True)
    def test_forward_missing(self):
        with self.assertRaises(ValueError):
            with c.counted_forward(Tiny(),c.new_counts(),'primary'): pass
    def test_forward_extra(self):
        net=Tiny(); counts=c.new_counts()
        with self.assertRaises(ValueError):
            with c.counted_forward(net,counts,'primary'): net(None); net(None)
        self.assertEqual((counts['primary_started'],counts['primary_completed']),(2,1))
    def test_forward_partial(self):
        net=Tiny(); counts=c.new_counts()
        with patch.object(net,'forward',side_effect=RuntimeError('unit')):
            with self.assertRaises(RuntimeError):
                with c.counted_forward(net,counts,'reference'): net(None)
        self.assertEqual((counts['reference_started'],counts['reference_completed']),(1,0))
    def test_forward_limit(self):
        with self.assertRaises(ValueError):
            with c.counted_forward(Tiny(),c.new_counts()|{'reference_started':18,'reference_completed':18},'reference'): pass
    def test_fp32_order(self):
        total=torch.zeros(1)
        for v in (1e8,1.,-1e8,3.,4.,5.): total+=torch.tensor([v])
        self.assertEqual(float(total),12.)
    def test_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'slots').mkdir()
            with self.assertRaises(ValueError): c.new_run_required(out)
    def test_existing_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): c.k.fresh(Path(tmp))
    def test_full_payload_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); plan,result=serialize(out); c.check_payload(out,plan,result)
    def test_parent_invalid_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); plan,result=serialize(out); result['model_batches']=8
            with self.assertRaises(ValueError): c.check_payload(out,plan,result)
    def test_totals_symmetric_seal_regression(self):
        doc=c.acq.seal({'values':{'number':1},'counter':4500}); inside=copy.deepcopy(doc)
        with self.assertRaises(ValueError): c.symmetric_document({key:v for key,v in doc.items() if key!='content_sha256'},inside)
    def test_totals_type_sensitive(self):
        doc=c.acq.seal({'number':1}); other=c.acq.seal({'number':True})
        with self.assertRaises(ValueError): c.symmetric_document(doc,other)
    def test_embedded_totals_resealed_different(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); plan,result=serialize(out); result['rows'][0]['totals']['group_additivity_not_assumed']=False
            c.acq.seal(result['rows'][0]['totals']); c.acq.seal(result['rows'][0]); c.acq.seal(result)
            with self.assertRaises(ValueError): c.check_payload(out,plan,result)
    def test_embedded_slot_bad_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); plan,result=serialize(out); result['rows'][0]['slots'][0]['slot']=1
            with self.assertRaises(ValueError): c.check_payload(out,plan,result)
    def test_totals_structural_failure(self):
        plan,result=copy.deepcopy(fixture()); result['rows'][0]['totals']['canonical_full_vs_authority']=c.n.vector_comparison(torch.ones(4),torch.zeros(4))
        with self.assertRaises(ValueError): c.validate_result(result,plan)
    def test_reference_wrong_batch(self):
        plan,result=copy.deepcopy(fixture()); result['rows'][1]['reference_rows']=result['rows'][0]['reference_rows']
        with self.assertRaises(ValueError): c.validate_result(result,plan)
    def test_counts_type_sensitive(self):
        plan,result=copy.deepcopy(fixture()); result['counts']['primary_started']=54.
        with self.assertRaises(ValueError): c.validate_result(result,plan)
    def test_model_order(self):
        plan,result=copy.deepcopy(fixture()); result['rows'][0]['model']='lr1_4500'
        with self.assertRaises(ValueError): c.validate_result(result,plan)
    def test_zero_compute_guard(self):
        with self.assertRaises(AssertionError):
            with c.e.no_compute(): Tiny()(None)


if __name__=='__main__': unittest.main(verbosity=2)
