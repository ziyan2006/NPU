"""New199 synthetic unit gates; no historical tests or real TRAIN probes."""
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

spec = importlib.util.spec_from_file_location("canonical199", Path(__file__).with_name("199_verify_train_adam_canonical_bridge.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
torch.set_num_threads(2)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.1, -.1, .05, -.05]))
        self.calls = 0
    def forward(self, bands):
        self.calls += 1
        return self.weight.tanh().reshape(1,4,1,1).expand(1,4,128,352)


def saved(net=None):
    net = net or Tiny()
    return {"model": copy.deepcopy(net.state_dict()), "parameter_names": ["weight"],
        "modes": [mod.training for mod in net.modules()], "updates": 10,
        "optimizer": {"param_groups": [{"lr": .01, "betas": (.9,.999), "eps": 1e-8,
            "weight_decay": 0, "amsgrad": False, "maximize": False, "foreach": False,
            "capturable": False, "differentiable": False, "fused": False, "decoupled_weight_decay": False, "params": [0]}],
            "state": {0: {"step": torch.tensor(10.), "exp_avg": torch.tensor([.1,-.2,.3,-.4]), "exp_avg_sq": torch.tensor([.01,.03,.02,.04])}}}}


def fixture():
    index = torch.arange(89856, dtype=torch.float32)
    a = .15*torch.sin(index*.07).repeat(2,1); v = .04*torch.cos(index*.031).repeat(2,1)
    xs, vs, metas = [], [], []
    for i, domain in enumerate(b.m.DOMAINS):
        target = torch.zeros_like(v) if i == 2 else v*(i+1)/2
        mix = a+target; xs.append(mix); vs.append(target)
        metas.append({"domain": domain, "role": "train" if i < 3 else "pseudo_label_train_candidate", "vocal_db": 0,
            "track_id": "synthetic_unit_not_REAL_TRAIN", "input_pcm_sha256": b.m.pilot.wave_digest(mix), "score_start": 25088, "score_end": 89344})
    return {"x": torch.stack(xs), "v": torch.stack(vs), "metadata": metas, "diagnostic_counter": 4500}


@lru_cache(maxsize=1)
def synthetic_bridge():
    net = Tiny(); state = saved(net); batch = fixture(); wa,gs,_ = b.k.matrix_identity()
    # Populate existing grad deliberately: autograd.grad must leave it unchanged.
    net.weight.grad = torch.ones_like(net.weight)*.17
    u,d,_ = b.k.stored_direction(state); counts = b.new_counts(); primaries = []; vectors = []; refs = []
    authority, reference = torch.zeros_like(u), torch.zeros_like(u)
    for i,meta in enumerate(batch['metadata']):
        with b.k.readonly(net,state,batch), b.counted_forward(net,counts,'primary'):
            base,wave,losses,info,identity = b.n.slot_wave(net,batch['x'][i:i+1],batch['v'][i:i+1],wa,gs,meta)
            doc,vec = b.primary_probe(tuple(net.parameters()),base,wave,losses,info,meta,i,(u,d))
        b.check_primary(doc,info['active']); primaries.append((doc,info,identity)); vectors.append(vec['authority']); authority += vec['authority']
    for i,meta in enumerate(batch['metadata']):
        with b.k.readonly(net,state,batch), b.counted_forward(net,counts,'reference'):
            vec,identity,info,loss,composition = b.reference_slot(net,batch['x'][i:i+1],batch['v'][i:i+1],wa,gs,meta,i)
        doc,pinfo,pid = primaries[i]
        assert b.k.exact(identity,{key:pid[key] for key in identity}) and b.k.exact(pinfo,info)
        assert b.k.exact(doc['loss'],loss) and b.k.exact(doc['composition'],composition)
        comparison = b.n.vector_comparison(vectors[i],vec); b.require_comparison(comparison,exact=True)
        refs.append(comparison); reference += vec
    assert torch.equal(authority,reference) and net.calls == 12
    return primaries, refs, counts, net.weight.grad.clone()


class BridgeTests(unittest.TestCase):
    def test_scope(self): b.check_scope(b.scope())
    def test_protocol(self): self.assertTrue(b.k.exact(json.loads(b.PROTOCOL.read_text()),b.scope()))
    def test_budget_unique(self): self.assertEqual((b.scope()['unique_input_batches'],b.scope()['unique_input_slots'],b.scope()['total_slot_forwards']),(1,6,12))
    def test_scope_float(self):
        with self.assertRaises(ValueError): b.check_scope(b.scope()|{'counter':4500.})
    def test_scope_bool_count(self):
        with self.assertRaises(ValueError): b.check_scope(b.scope()|{'primary_slot_forwards':True})
    def test_scope_no_relaxation(self):
        with self.assertRaises(ValueError): b.check_scope(b.scope()|{'atol':2e-6})
    def test_fp64_rejected(self):
        with self.assertRaises(ValueError): b.n.vector_comparison(torch.ones(2).double(),torch.ones(2).double())
    def test_shape_rejected(self):
        with self.assertRaises(ValueError): b.n.vector_comparison(torch.ones(2),torch.ones(3))
    def test_nonfinite(self):
        with self.assertRaises(ValueError): b.n.vector_comparison(torch.tensor([float('nan')]),torch.zeros(1))
    def test_lf32(self): self.assertEqual(b.k.matrix_identity()[2]['forced_zero_fft_bins'],[0,1,2,3])
    def test_all_legal_gains(self):
        for domain in ('musdb','mir1k'):
            for gain in (-12,-6,0,6): self.assertTrue(b.k.bucket({'domain':domain,'role':'train','vocal_db':gain}))
    def test_bad_gain(self):
        with self.assertRaises(ValueError): b.k.bucket({'domain':'musdb','role':'train','vocal_db':True})
    def test_pseudo_not_truth(self): self.assertEqual(b.k.bucket({'domain':'pseudo','role':'pseudo_label_train_candidate'}),'pseudo_not_final_truth')
    def test_pseudo_bad_role(self):
        with self.assertRaises(ValueError): b.k.bucket({'domain':'pseudo','role':'train'})
    def test_original_metadata_order(self):
        batch=fixture(); batch['metadata'][0],batch['metadata'][1]=batch['metadata'][1],batch['metadata'][0]
        with self.assertRaises(ValueError): b.k.input_entry(batch)
    def test_old_failure_retained(self):
        doc=b.n.vector_comparison(torch.tensor([8.8e-5+4.33e-7]),torch.tensor([8.8e-5]))
        self.assertFalse(doc['old_strict_pass']); self.assertEqual(len(doc['all_mismatched_elements']),1)
    def test_changed_failure_not_accepted(self):
        doc=b.n.vector_comparison(torch.ones(2),torch.zeros(2)); doc['old_strict_pass']=True
        with self.assertRaises(ValueError): b.n.validate_comparison(doc)
    def test_exact_gate_rejects_small_error(self):
        doc=b.n.vector_comparison(torch.tensor([1.+1e-7]),torch.ones(1)); self.assertTrue(doc['old_strict_pass'])
        with self.assertRaises(ValueError): b.require_comparison(doc,exact=True)
    def test_strict_gate_rejects(self):
        with self.assertRaises(ValueError): b.require_comparison(b.n.vector_comparison(torch.ones(2),torch.zeros(2)))
    def test_zero_cosine_unavailable(self): self.assertIsNone(b.k.alignment(torch.zeros(2),torch.ones(2),-torch.ones(2))['cos_g_u'])
    def test_descent_sign(self): self.assertTrue(b.k.alignment(torch.ones(2),torch.ones(2),-torch.ones(2))['first_order_descent'])
    def test_ascent_sign(self): self.assertFalse(b.k.alignment(torch.ones(2),-torch.ones(2),torch.ones(2))['first_order_descent'])
    def test_bias_correction_oracle(self):
        s=saved(); u,d,_=b.k.stored_direction(s); state=s['optimizer']['state'][0]
        oracle=[(float(m)/(1-.9**10))/(math.sqrt(float(v)/(1-.999**10))+1e-8) for m,v in zip(state['exp_avg'],state['exp_avg_sq'])]
        torch.testing.assert_close(u,torch.tensor(oracle),rtol=2e-7,atol=1e-8); self.assertTrue(torch.equal(d,-.01*u))
    def test_direction_no_alias(self):
        s=saved(); old=b.k.tree_digest(s); u,d,_=b.k.stored_direction(s); u.add_(1); d.add_(1); self.assertEqual(old,b.k.tree_digest(s))
    def test_existing_grad_preserved(self): self.assertTrue(torch.equal(synthetic_bridge()[3],torch.ones(4)*.17))
    def test_exact_6_plus_6(self): self.assertEqual(synthetic_bridge()[2],{key:6 for key in b.new_counts()})
    def test_same_graph_vjp(self):
        for doc,info,_ in synthetic_bridge()[0][:2]: b.require_comparison(doc['paths']['parameter_full_vs_vjp_full'],exact=True)
    def test_complete_original_composition(self):
        for doc,info,_ in synthetic_bridge()[0]: b.require_comparison(doc['canonical_full_vs_direct'])
    def test_old_split_failure_is_not_hidden_or_bridge_gate(self):
        doc,info,_=copy.deepcopy(synthetic_bridge()[0][0])
        doc['paths']['parameter_full_vs_parts']=b.n.vector_comparison(torch.ones(4),torch.zeros(4))
        b.check_primary(doc,True)
        self.assertFalse(doc['paths']['parameter_full_vs_parts']['old_strict_pass'])
        self.assertEqual(len(doc['paths']['parameter_full_vs_parts']['all_mismatched_elements']),4)
    def test_full_vjp_failure_is_bridge_gate(self):
        doc,info,_=copy.deepcopy(synthetic_bridge()[0][0])
        doc['paths']['parameter_full_vs_vjp_full']=b.n.vector_comparison(torch.ones(4),torch.zeros(4))
        with self.assertRaises(ValueError): b.check_primary(doc,True)
    def test_combined_wave_vjp_failure_is_bridge_gate(self):
        doc,info,_=copy.deepcopy(synthetic_bridge()[0][0])
        doc['paths']['parameter_full_vs_vjp_parts']=b.n.vector_comparison(torch.ones(4),torch.zeros(4))
        with self.assertRaises(ValueError): b.check_primary(doc,True)
    def test_wave_cotangent_failure_is_bridge_gate(self):
        doc,info,_=copy.deepcopy(synthetic_bridge()[0][0])
        doc['paths']['wave_full_vs_parts']=b.n.vector_comparison(torch.ones(4),torch.zeros(4))
        with self.assertRaises(ValueError): b.check_primary(doc,True)
    def test_reference_bit_exact(self):
        for doc in synthetic_bridge()[1]: b.require_comparison(doc,exact=True)
    def test_skip_inst_pseudo(self):
        for doc,info,_ in synthetic_bridge()[0][2:]: self.assertFalse(info['active']); self.assertIsNone(doc['paths'])
    def test_instrumental_four(self):
        base=torch.tensor(.3); aux=base*0; meta={'domain':'instrumental','role':'train'}
        loss,_=b.k.t.w.combine_slot_loss(base,aux,meta,2,4,{'active':False}); self.assertTrue(torch.equal(loss,(4*base+.2*aux)*(1/6)))
    def test_six_slot_fp32_sequence(self):
        values=[torch.tensor([1e8]),torch.tensor([1.]),torch.tensor([-1e8]),torch.tensor([3.]),torch.tensor([4.]),torch.tensor([5.])]
        sequential=torch.zeros(1)
        for value in values: sequential+=value
        self.assertEqual(float(sequential),12.); self.assertNotEqual(float(sum((v.double() for v in values))),float(sequential))
    def test_forward_missing(self):
        with self.assertRaises(ValueError):
            with b.counted_forward(Tiny(),b.new_counts(),'primary'): pass
    def test_forward_extra(self):
        net=Tiny(); counts=b.new_counts()
        with self.assertRaises(ValueError):
            with b.counted_forward(net,counts,'primary'): net(None); net(None)
        self.assertEqual(counts['primary_started'],2); self.assertEqual(counts['primary_completed'],1)
    def test_forward_partial_failure_retained(self):
        net=Tiny(); counts=b.new_counts()
        with patch.object(net,'forward',side_effect=RuntimeError('unit injected')):
            with self.assertRaises(RuntimeError):
                with b.counted_forward(net,counts,'reference'): net(None)
        self.assertEqual((counts['reference_started'],counts['reference_completed']),(1,0))
    def test_forward_limit(self):
        counts=b.new_counts()|{'primary_started':6,'primary_completed':6}
        with self.assertRaises(ValueError):
            with b.counted_forward(Tiny(),counts,'primary'): pass
    def test_rng_guard(self):
        net=Tiny(); s=saved(net); batch=fixture(); rng=b.m.capture_rng('cpu')
        try:
            with self.assertRaises(ValueError):
                with b.k.readonly(net,s,batch): torch.rand(1)
        finally: b.m.restore_rng(rng,'cpu')
    def test_moments_guard(self):
        net=Tiny(); s=saved(net)
        with self.assertRaises(ValueError):
            with b.k.readonly(net,s,{}): s['optimizer']['state'][0]['exp_avg'].add_(1)
    def test_modes_guard(self):
        net=Tiny()
        with self.assertRaises(ValueError):
            with b.k.readonly(net,saved(net),{}): net.eval()
    def test_existing_grad_mutation_guard(self):
        net=Tiny(); net.weight.grad=torch.ones_like(net.weight)
        with self.assertRaises(ValueError):
            with b.k.readonly(net,saved(net),{}): net.weight.grad.add_(1)
    def test_inputs_guard(self):
        net=Tiny(); batch={'x':torch.zeros(2)}
        with self.assertRaises(ValueError):
            with b.k.readonly(net,saved(net),batch): batch['x'].add_(1)
    def test_weight_guard(self):
        net=Tiny()
        with self.assertRaises(ValueError):
            with b.k.readonly(net,saved(net),{}),torch.no_grad(): net.weight.add_(1)
    def test_cursor_schedule_guard(self):
        net=Tiny(); state=saved(net)|{'cursor':4500,'schedule':{'step':4500,'stopped_at':4500}}
        with self.assertRaises(ValueError):
            with b.k.readonly(net,state,{}): state['schedule']['step']=4501
    def test_existing_output_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'reference_rows').mkdir()
            with self.assertRaises(ValueError): b.run(out)
    def test_separate_row_seals_symmetric(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'rows').mkdir(); rows=[]
            for i in range(6):
                doc=b.acq.seal({'slot':i,'value':1}); b.acq.write_new_json(out/'rows'/f'row_{i:02d}.json',doc); rows.append(doc)
            b.symmetric_rows(out,rows)
            changed=copy.deepcopy(rows); changed[0]=b.acq.seal({'slot':0,'value':True})
            with self.assertRaises(ValueError): b.symmetric_rows(out,changed)
            changed=copy.deepcopy(rows); changed[0]['value']=2
            with self.assertRaises(ValueError): b.symmetric_rows(out,changed)
    def test_partial_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): b.symmetric_rows(Path(tmp),[])
    def test_no_adam_fresh_cpu_cuda_uninitialized(self):
        code="import importlib.util,torch; from pathlib import Path; from unittest.mock import patch; s=importlib.util.spec_from_file_location('fresh',Path(r'"+str(b.__file__)+"')); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        code+="\nwith patch.object(torch.optim,'Adam',side_effect=AssertionError('NO_OPTIMIZER_ALLOWED')):\n assert not torch.cuda.is_initialized(); p=torch.nn.Parameter(torch.ones(2)); g=m.grad((p,),p.square().sum(),False); assert p.grad is None and not torch.cuda.is_initialized(); print('NO_ADAM_CPU_CUDA_UNINITIALIZED')"
        proc=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
        self.assertIn('NO_ADAM_CPU_CUDA_UNINITIALIZED',proc.stdout)


def add_mapping_test(name, change):
    def test(self):
        state=saved(); change(state)
        with self.assertRaises(ValueError): b.k.stored_direction(state)
    setattr(BridgeTests,'test_mapping_'+name,test)


for key,value in [('foreach',True),('fused',True),('amsgrad',True),('maximize',True),('weight_decay',0.),('capturable',True),('differentiable',True),('decoupled_weight_decay',True),('eps',1e-7),('betas',[.9,.999]),('lr',True),('params',[False])]:
    add_mapping_test(key,lambda s,key=key,value=value:s['optimizer']['param_groups'][0].__setitem__(key,value))
add_mapping_test('id_order',lambda s:s['optimizer']['param_groups'][0].__setitem__('params',[1]))
add_mapping_test('name',lambda s:s.__setitem__('parameter_names',['missing']))
add_mapping_test('moment_shape',lambda s:s['optimizer']['state'][0].__setitem__('exp_avg',torch.ones(3)))
add_mapping_test('variance_negative',lambda s:s['optimizer']['state'][0]['exp_avg_sq'].__setitem__(0,-1))
add_mapping_test('step',lambda s:s['optimizer']['state'][0].__setitem__('step',torch.tensor(9.)))
add_mapping_test('step_dtype',lambda s:s['optimizer']['state'][0].__setitem__('step',torch.tensor(10)))
add_mapping_test('missing_variance',lambda s:s['optimizer']['state'][0].pop('exp_avg_sq'))


if __name__ == '__main__': unittest.main(verbosity=2)
