"""New197 synthetic CPU gates. No Adam instance and no real TRAIN probe."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("memory197_tested", Path(__file__).with_name("197_diagnose_train_adam_memory.py"))
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
torch.set_num_threads(2)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.1, -.1, .05, -.05]))
        self.calls = 0
    def forward(self, bands):
        self.calls += 1
        return self.weight.tanh().reshape(1, 4, 1, 1).expand(1, 4, 128, 352)


def saved(net=None):
    net = net or Tiny()
    return {"model": copy.deepcopy(net.state_dict()), "parameter_names": ["weight"],
        "modes": [mod.training for mod in net.modules()], "updates": 10,
        "optimizer": {"param_groups": [{"lr": .01, "betas": (.9, .999), "eps": 1e-8,
            "weight_decay": 0, "amsgrad": False, "maximize": False, "foreach": False,
            "capturable": False, "differentiable": False, "fused": False, "decoupled_weight_decay": False, "params": [0]}],
            "state": {0: {"step": torch.tensor(10.), "exp_avg": torch.tensor([.1, -.2, .3, -.4]),
                "exp_avg_sq": torch.tensor([.01, .03, .02, .04])}}}}


def fixture():
    index = torch.arange(89856, dtype=torch.float32)
    a = .15*torch.sin(index*.07).repeat(2, 1)
    v = .04*torch.cos(index*.031).repeat(2, 1)
    xs, vs, metas = [], [], []
    for slot, domain in enumerate(k.m.DOMAINS):
        target = torch.zeros_like(v) if slot == 2 else v*(slot+1)/2
        mix = a+target
        xs.append(mix); vs.append(target)
        metas.append({"domain": domain, "role": "train" if slot < 3 else "pseudo_label_train_candidate",
            "vocal_db": 0, "track_id": "synthetic_no_REAL_TRAIN", "input_pcm_sha256": k.m.pilot.wave_digest(mix),
            "score_start": 25088, "score_end": 89344})
    return {"x": torch.stack(xs), "v": torch.stack(vs), "metadata": metas, "diagnostic_counter": 4500}


class MemoryTests(unittest.TestCase):
    def test_scope(self): k.check_scope(k.scope())
    def test_scope_bool(self):
        with self.assertRaises(ValueError): k.check_scope(k.scope() | {"model_batches": True})
    def test_scope_counter_float(self):
        with self.assertRaises(ValueError): k.check_scope(k.scope() | {"cursors": [4500., 4501, 4502]})
    def test_scope_counts(self):
        s = k.scope(); self.assertEqual((s["model_batches"], s["primary_slot_forwards"], s["reference_slot_forwards"], s["total_slot_forwards"]), (9,54,18,72))
    def test_scope_coefficients(self): self.assertEqual(k.scope()["coefficients"], [1.,4.,1.,.2,.2])
    def test_matrix(self):
        _, gs, identity = k.matrix_identity(); self.assertEqual(identity["forced_zero_fft_bins"], [0,1,2,3]); self.assertEqual(gs.shape,(513,128))
    def test_fp32(self):
        with self.assertRaises(ValueError): k.cpu_float(torch.ones(2).double())
    def test_shape(self):
        with self.assertRaises(ValueError): k.cpu_float(torch.ones(2), (3,))
    def test_nonfinite(self):
        with self.assertRaises(ValueError): k.cpu_float(torch.tensor([float('nan')]))
    def test_all_gains(self):
        for domain in ('musdb','mir1k'):
            for gain in (-12,-6,0,6):
                self.assertTrue(k.bucket({'domain':domain,'role':'train','vocal_db':gain}).startswith(domain))
    def test_gain_bool(self):
        with self.assertRaises(ValueError): k.bucket({'domain':'musdb','role':'train','vocal_db':True})
    def test_gain_unsupported(self):
        with self.assertRaises(ValueError): k.bucket({'domain':'musdb','role':'train','vocal_db':12})
    def test_pseudo(self): self.assertEqual(k.bucket({'domain':'pseudo','role':'pseudo_label_train_candidate'}),'pseudo_not_final_truth')
    def test_pseudo_not_truth(self):
        with self.assertRaises(ValueError): k.bucket({'domain':'pseudo','role':'train'})
    def test_roles(self):
        for i,domain in enumerate(k.m.DOMAINS):
            self.assertEqual(k.group_for_slot(i, {'domain':domain, 'role':'train' if i<3 else 'pseudo_label_train_candidate'}),k.GROUPS[0 if i<2 else 1 if i==2 else 2])
    def test_dev_reject(self):
        with self.assertRaises(ValueError): k.group_for_slot(0, {'domain':'musdb','role':'development'})
    def test_id_name_mapping(self):
        _,_,info=k.stored_direction(saved()); self.assertEqual(info['parameter_mapping'][0]['parameter_name'],'weight')
    def test_id_type(self):
        s=saved(); s['optimizer']['param_groups'][0]['params']=[False]
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_id_order(self):
        s=saved(); s['optimizer']['param_groups'][0]['params']=[1]
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_missing_name(self):
        s=saved(); s['parameter_names']=['missing']
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_moment_shape(self):
        s=saved(); s['optimizer']['state'][0]['exp_avg']=torch.ones(3)
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_variance_negative(self):
        s=saved(); s['optimizer']['state'][0]['exp_avg_sq'][0]=-1
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_step_exposure(self):
        s=saved(); s['optimizer']['state'][0]['step']=torch.tensor(9.)
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_step_dtype(self):
        s=saved(); s['optimizer']['state'][0]['step']=torch.tensor(10)
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_beta_type(self):
        s=saved(); s['optimizer']['param_groups'][0]['betas']=[.9,.999]
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_flags(self):
        for flag,value in (('foreach',True),('fused',True),('maximize',True),('amsgrad',True),('weight_decay',0.)):
            s=saved(); s['optimizer']['param_groups'][0][flag]=value
            with self.assertRaises(ValueError): k.stored_direction(s)
    def test_epsilon(self):
        s=saved(); s['optimizer']['param_groups'][0]['eps']=1e-7
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_lr_type(self):
        s=saved(); s['optimizer']['param_groups'][0]['lr']=True
        with self.assertRaises(ValueError): k.stored_direction(s)
    def test_bias_correction_scalar_oracle(self):
        s=saved(); u,d,_=k.stored_direction(s); values=s['optimizer']['state'][0]
        oracle=[(float(a)/(1-.9**10))/(math.sqrt(float(b)/(1-.999**10))+1e-8) for a,b in zip(values['exp_avg'],values['exp_avg_sq'])]
        torch.testing.assert_close(u,torch.tensor(oracle),rtol=2e-7,atol=1e-8)
        torch.testing.assert_close(d,-.01*u,rtol=0,atol=0)
    def test_no_alias_mutation(self):
        s=saved(); before=k.tree_digest(s); u,d,_=k.stored_direction(s); u.zero_(); d.zero_(); self.assertEqual(k.tree_digest(s),before)
    def test_correct_descent_sign(self):
        a=k.alignment(torch.tensor([1.,2.]),torch.tensor([1.,2.]),torch.tensor([-.1,-.2])); self.assertTrue(a['first_order_descent']); self.assertGreater(a['cos_g_u'],0); self.assertLess(a['cos_g_d'],0)
    def test_opposition(self):
        a=k.alignment(torch.tensor([-1.,-2.]),torch.tensor([1.,2.]),torch.tensor([-.1,-.2])); self.assertFalse(a['first_order_descent'])
    def test_zero_gradient(self):
        a=k.alignment(torch.zeros(2),torch.ones(2),-torch.ones(2)); self.assertIsNone(a['cos_g_u']); self.assertIsNone(a['first_order_descent'])
    def test_zero_direction(self):
        a=k.alignment(torch.ones(2),torch.zeros(2),torch.zeros(2)); self.assertIsNone(a['cos_g_u']); self.assertIsNone(a['cos_g_d'])
    def test_existing_grad_preserved(self):
        net=Tiny(); net.weight.grad=torch.ones_like(net.weight); s=saved(net)
        with k.readonly(net,s,{}): torch.autograd.grad(net.weight.square().sum(),tuple(net.parameters()))
        self.assertTrue(torch.equal(net.weight.grad,torch.ones(4)))
    def test_readonly_detects_change(self):
        net=Tiny(); s=saved(net)
        with self.assertRaises(ValueError):
            with k.readonly(net,s,{}): s['updates']=11
    def test_rng_detects_change(self):
        net=Tiny(); s=saved(net); rng=k.m.capture_rng('cpu')
        try:
            with self.assertRaises(ValueError):
                with k.readonly(net,s,{}): torch.rand(2)
        finally: k.m.restore_rng(rng,'cpu')
    def test_full_original194_equivalence_and_counts(self):
        batch=fixture(); net=Tiny(); s=saved(net); wa,gs,_=k.matrix_identity()
        before=k.tree_digest(batch); probe=k.parameter_probe(net,s,batch,wa,gs,True)
        self.assertEqual(net.calls,12); self.assertEqual((probe['primary_slot_forwards'],probe['reference_slot_forwards']),(6,6))
        self.assertEqual(before,k.tree_digest(batch)); self.assertIsNone(net.weight.grad)
        actual=torch.zeros(4)
        wave_checks=[]
        original_reconstruction=k.m.fit.reconstruction_loss
        def recorded_reconstruction(*args,**kwargs):
            answer=original_reconstruction(*args,**kwargs)
            wave_checks.append({'wave_sha256':k.tree_digest(answer[2]),'base':float(answer[0].detach())})
            return answer
        def record_backward(loss,*args,**kwargs):
            nonlocal actual
            actual += torch.autograd.grad(loss,tuple(net.parameters()))[0].detach()
        with patch.object(torch.Tensor,'backward',record_backward), patch.object(k.m.fit,'reconstruction_loss',recorded_reconstruction):
            original=k.t.boundary_backward(net,batch['x'],batch['v'],wa,gs,'cpu',batch['metadata'],32)
        self.assertAlmostEqual(original['base_loss'],sum(slot['base_div6'] for slot in probe['slots']),places=6)
        self.assertEqual(k.tree_digest(actual),probe['independent_combination']['actual_gradient_sha256'])
        for check,slot in zip(wave_checks,probe['slots']):
            self.assertEqual(check['wave_sha256'],slot['identity']['wave_sha256']); self.assertEqual(check['base'],slot['identity']['base'])
        self.assertIsNone(net.weight.grad)
    def test_input_hash_change_reject(self):
        b=fixture(); b['x'][0,0,0]+=1
        with self.assertRaises(ValueError): k.input_entry(b)
    def test_counter_reject(self):
        b=fixture(); b['diagnostic_counter']=4503
        with self.assertRaises(ValueError): k.input_entry(b)
    def test_instrumental_zero_reference(self):
        b=fixture(); b['v'][2,0,0]=.01
        with self.assertRaises(ValueError): k.input_entry(b)
    def test_support_edges_reject(self):
        b=fixture(); b['metadata'][0]['score_start']=25089
        with self.assertRaises(ValueError): k.input_entry(b)
    def test_tree_full_types(self): self.assertNotEqual(k.tree_digest({'x':1}),k.tree_digest({'x':True})); self.assertNotEqual(k.tree_digest([1]),k.tree_digest((1,)))
    def test_existing_output_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): k.fresh(Path(tmp))
    def test_symmetric_row_seals(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'rows').mkdir(); rows=[]
            for i in range(9):
                doc=k.acq.seal({'counter':i,'nested':{'value':1}}); k.acq.write_new_json(out/'rows'/f'row_{i:02d}.json',doc); rows.append(doc)
            k.symmetric_rows(out,rows)
            changed=copy.deepcopy(rows); changed[0]['nested']['value']=True; changed[0]=k.acq.seal(changed[0])
            with self.assertRaises(ValueError): k.symmetric_rows(out,changed)
            changed=copy.deepcopy(rows); changed[0]['nested']['value']=2
            with self.assertRaises(ValueError): k.symmetric_rows(out,changed)
    def test_fresh_no_adam_cuda(self):
        code="import importlib.util; s=importlib.util.spec_from_file_location('fresh197',"+repr(str(Path(k.__file__)))+"); k=importlib.util.module_from_spec(s); s.loader.exec_module(k); import torch; assert not torch.cuda.is_initialized(); k.alignment(torch.ones(2),torch.ones(2),-torch.ones(2)); assert not torch.cuda.is_initialized()"
        done=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True); self.assertEqual(done.returncode,0,done.stderr)
    def test_stateless_input_adapter_recipe(self):
        b=fixture(); calls=[]
        class Dataset:
            seed=20261002; config={}; rows=[]
            def crop(self,recipe):
                i=recipe-4500*3; calls.append(('pseudo',recipe))
                return {'x':b['x'][i+3].clone(),'v':b['v'][i+3].clone(),'meta':b['metadata'][i+3].copy()}
        class TruePool:
            def crop(self,domain,seed,cursor):
                i=k.m.DOMAINS.index(domain); calls.append((domain,seed,cursor))
                return {'x':b['x'][i].clone(),'v':b['v'][i].clone(),'meta':b['metadata'][i].copy()}
        with patch.object(k.m.data,'crop_recipe',lambda rows,config,seed,cursor:cursor):
            got=k.collect_next(Dataset(),TruePool(),4500)
        self.assertEqual(k.tree_digest(got),k.tree_digest(b))
        self.assertEqual(calls,[('musdb',20261002,4500),('mir1k',20261002,4500),('instrumental',20261002,4500),('pseudo',13500),('pseudo',13501),('pseudo',13502)])
    def test_existing_run_reject_before_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'rows').mkdir()
            with self.assertRaises(ValueError): k.run(out)
    def test_binding_stamps_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bound.json'; k.acq.write_new_json(path,{'a':1})
            k.unchanged_bindings(k.binding_stamps([path]))
    def test_binding_stamps_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bound.json'; k.acq.write_new_json(path,{'a':1})
            stamps=k.binding_stamps([path])
            os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1000000000))
            with self.assertRaises(ValueError): k.unchanged_bindings(stamps)
    def test_missing_binding_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): k.binding_stamps([Path(tmp)/'missing'])


class FullResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        batch=fixture(); net=Tiny(); state=saved(net); wa,gs,_=k.matrix_identity()
        probe=k.parameter_probe(net,state,batch,wa,gs,True)
        descriptors={name:{'direction':probe['direction']} for name in k.MODELS}
        inputs=[]
        for cursor in k.CURSORS:
            b=copy.deepcopy(batch); b['diagnostic_counter']=cursor; inputs.append(k.input_entry(b))
        rows=[]
        for name in k.MODELS:
            for i,cursor in enumerate(k.CURSORS):
                row=copy.deepcopy(probe) | {'model':name,'counter':cursor,'model_descriptor':descriptors[name],'input':inputs[i]}
                if i:
                    row['reference_slot_forwards']=0; row['independent_combination']=None
                rows.append(row)
        runtime=k.d.runtime_identity('cpu')
        cls.plan={'model_descriptors':descriptors,'expected_runtime':runtime}
        cls.result=k.scope() | {'rows':rows,'inputs':inputs,'model_descriptors':descriptors,'runtime':runtime,
            'error':None,'weights_modes_grad_rng_moments_inputs_unchanged':True,'existing_grad_storage_written':False,
            'summary':k.summary(rows,inputs)}
    def test_complete_accounting_and_aggregation(self): k.validate_result(self.result,self.plan)
    def test_partial_rejected(self):
        x=copy.deepcopy(self.result); x['rows'].pop()
        with self.assertRaises(ValueError): k.validate_result(x,self.plan)
    def test_count_type_rejected(self):
        x=copy.deepcopy(self.result); x['rows'][0]['primary_slot_forwards']=6.
        with self.assertRaises(ValueError): k.validate_result(x,self.plan)
    def test_input_metadata_hash_rejected(self):
        x=copy.deepcopy(self.result); x['inputs'][0]['input_sha256'][0]='a'*64
        with self.assertRaises(ValueError): k.validate_result(x,self.plan)
    def test_summary_changed_rejected(self):
        x=copy.deepcopy(self.result); x['summary']['unique_slot_buckets']['instrumental_zero_reference']=99
        with self.assertRaises(ValueError): k.validate_result(x,self.plan)
    def test_missing_combination_rejected(self):
        x=copy.deepcopy(self.result); x['rows'][0]['independent_combination']=None
        with self.assertRaises(ValueError): k.validate_result(x,self.plan)
    def test_extra_reference_rejected(self):
        x=copy.deepcopy(self.result); x['rows'][1]['reference_slot_forwards']=6
        with self.assertRaises(ValueError): k.validate_result(x,self.plan)
    def test_recorded_grad_mutation_rejected(self):
        x=copy.deepcopy(self.result); x['rows'][0]['existing_grad_storage_written']=True
        with self.assertRaises(ValueError): k.validate_result(x,self.plan)


if __name__=='__main__': unittest.main(verbosity=2)
