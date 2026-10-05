"""New fixed9 CPU canonical Adam-memory diagnosis; immutable197 remains failed.

Direct original194 scalar gradients are authoritative. Independent split
failures are retained measurements, never corrected or used as authority.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location('canonical_recovery_bridge', Path(__file__).with_name('200_verify_train_adam_bridge_evidence.py'))
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
b, k, acq, ROOT = e.b, e.k, e.acq, e.ROOT
n, m = b.n, b.m
PURPOSE = 'NONRELEASE_TRAIN_ADAM_MEMORY_CANONICAL_RECOVERY'
OUT = ROOT/'results/train_adam_memory_canonical_recovery_20261004'
MONITOR = ROOT/'results/train_adam_memory_canonical_recovery_monitor_20261004'
TEST = Path(__file__).with_name('_test_train_adam_memory_canonical_recovery.py')
PROTOCOL = ROOT/'docs/train_adam_memory_canonical_recovery_protocol_20261004.json'
LIMITS = {'primary':54, 'reference':18}
TOTAL_COMPARISONS = ('canonical_full_vs_authority', 'independent_slot_parts_vs_authority_measurement',
    'independent_group_parts_vs_authority_measurement', 'full_group_sum_vs_authority_measurement')


def scope():
    return {'schema':1,'purpose':PURPOSE,'models':list(k.MODELS),'counters':list(k.CURSORS),
        'unique_input_batches':3,'unique_input_slots':18,'model_batches':9,
        'primary_slot_forwards':54,'reference_slot_forwards':18,'total_slot_forwards':72,
        'samples':89856,'frames':352,'warmup':96,'native_support':[25088,89344],
        'kill_bands':32,'normalizer':6,'instrumental_weight':4,'auxiliary_lambda':.2,
        'groups':list(k.GROUPS),'coefficients':list(k.COEFFICIENTS),
        'rtol':.0002,'atol':.0000002,'threads':2,'forward_backward_dtype':'float32',
        'spectrum_dtype':'complex64','statistics_dtype':'float64',
        'authority':'direct_original194_combine_slot_loss_six_slot_FP32_order',
        'reference_requirement':'each_model_first_batch_bit_exact_each_slot_and_total',
        'old_split_failures_retained':True,'original197_rewritten_or_merged':False,
        'optimizer_constructed':False,'parameter_grad_written':False,'cuda_used':False,
        'model_updates':0,'training_authorized':False,'deployment':False,'release_selection':'NONE'}


def check_scope(doc):
    if any(not k.exact(doc.get(key),value) for key,value in scope().items()):
        raise ValueError('Changed fixed canonical recovery scope/types')


def no_active_task():
    e.no_active_task()
    pattern=re.escape(str(ROOT))+r'|201_diagnose_train_adam_memory_canonical_recovery|_test_train_adam_memory_canonical_recovery'
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '"+pattern+"' } | Select-Object -ExpandProperty ProcessId"
    proc=k.subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    if any(int(line.strip()) not in {os.getpid(),os.getppid()} for line in proc.stdout.splitlines() if line.strip()):
        raise ValueError('Active repository task; postpone without terminating it')


def source():
    # Read already successful200 proof, not its CLI or repeated old verification.
    review_path=e.MONITOR/'completion_review.json'
    pins={review_path:'f1d65c53b09e9551a3f63ccb778afd48b2c3d065f9384b0338973fdff24b2408',
        e.OUT/'plan.json':'c053486a6c0011bddf3caebe02f24b235fe26d22c37f618194d6ca292eaa9dcc',
        e.OUT/'verification.json':'1565f22e1eab4813ba2e4edc4019cf3189b08cbcb79c1b4c5993a5d9cfb689ed',
        ROOT/'reports/98_train_adam_memory_canonical_recovery_plan.md':'d19e2220d77c41cbfcbffc02863952fef5a858016565ff602f08f53725a98151'}
    for path,digest in pins.items():
        if acq.sha256(path)!=digest: raise ValueError('Changed recovered bridge authority')
    review=acq.read_sealed(review_path); proof=acq.read_sealed(e.OUT/'verification.json')
    if (not k.exact(review['real_new_verify']['actual_native_exit_code'],0) or
        not k.exact(review['old199_failed_verify']['actual_native_exit_code'],1) or
        review['bridge_evidence_recovered'] is not True or review['original197_recovered'] is not False or
        proof['independently_verified'] is not True or proof['original_structural_gates_passed'] is not True or
        proof['totals_both_seals_full_symmetric_type_sensitive'] is not True or
        not k.exact(proof['model_forwards'],0) or (b.OUT/'verification.json').exists()):
        raise ValueError('Real new200 success/old199 verify1/old197 failure separation required')
    inherited=acq.read_sealed(e.OUT/'plan.json')['bindings_sha256']
    if len(inherited)!=379: raise ValueError('Original379 source bindings required')
    authority,models,descriptors,sampler,bindings=k.source_evidence()
    bindings |= inherited
    for item in review['immutable_bindings']:
        path=(ROOT/item['path']).resolve()
        if acq.sha256(path)!=item['sha256']: raise ValueError('Changed actual200 unit/prepare/verify evidence')
        bindings[str(path)]=item['sha256']
    prior=acq.read_sealed(k.OUT/'inputs.json')
    expected=prior['inputs']
    if len(expected)!=3 or [entry['counter'] for entry in expected]!=list(k.CURSORS):
        raise ValueError('Three retained197 original inputs required')
    for path in (*pins.keys(), e.MONITOR/'aggregation.json', Path(__file__),TEST,PROTOCOL,
        ROOT/'reports/97_train_adam_bridge_evidence_recovery_result.md'):
        bindings[str(path.resolve())]=acq.sha256(path)
    k.p.q.check_bindings(bindings)
    return authority,models,descriptors,sampler,expected,bindings


def unit_gate(path):
    gate=acq.read_sealed(path)
    if (not k.exact(gate['actual_exit_code'],0) or gate['draft_reviewed'] is not True or
        type(gate['tests_passed']) is not int or gate['tests_passed']<50):
        raise ValueError('Actual new complete units and draft review required before prepare')
    for key,file in (('tool_sha256',Path(__file__)),('test_sha256',TEST),
        ('protocol_sha256',PROTOCOL),('log_sha256',Path(gate['log']))):
        if gate[key]!=acq.sha256(file): raise ValueError('Changed new unit binding')
    log=Path(gate['log']).read_text(encoding='utf-8-sig')
    if not re.search(r'Ran '+str(gate['tests_passed'])+r' tests? in',log) or not re.search(r'(?m)^OK\s*$',log):
        raise ValueError('Complete native unit output required')
    return gate


def prepare(out,unit):
    k.fresh(out); no_active_task(); k.resources(); gate=unit_gate(unit)
    if not k.exact(json.loads(PROTOCOL.read_text(encoding='utf-8')),scope()): raise ValueError('Changed protocol')
    _,_,descriptors,sampler,inputs,bindings=source()
    torch.set_num_threads(2)
    with k.d.deterministic_runtime('cpu'): runtime=k.d.runtime_identity('cpu')
    bindings[str(unit.resolve())]=acq.sha256(unit); bindings[gate['log']]=gate['log_sha256']
    out.mkdir(parents=True)
    plan=scope()|{'bindings_sha256':bindings,'model_descriptors':descriptors,'sampler':sampler,
        'expected_inputs':inputs,'matrix':k.matrix_identity()[2],'expected_runtime':runtime,'unit_gate':str(unit.resolve())}
    acq.write_new_json(out/'plan.json',acq.seal(plan))
    print('CANONICAL_MEMORY PLAN SEALED;9 batches54+18 forwards;no training',flush=True)


def read_plan(out):
    plan=acq.read_sealed(out/'plan.json'); check_scope(plan); unit_gate(Path(plan['unit_gate']))
    k.p.q.check_bindings(plan['bindings_sha256'])
    if not k.exact(plan['matrix'],k.matrix_identity()[2]): raise ValueError('Changed LF32 matrices')
    return plan


def new_counts(): return b.new_counts()


@contextlib.contextmanager
def counted_forward(net,counts,kind,changed=lambda:None):
    if kind not in LIMITS or counts[kind+'_started']>=LIMITS[kind]: raise ValueError('Bounded actual forward budget exhausted')
    before=counts[kind+'_started']
    def start(_net,_args):
        counts[kind+'_started']+=1; changed()
        if counts[kind+'_started']>LIMITS[kind] or counts[kind+'_started']!=before+1: raise ValueError('Unexpected extra model forward')
    def end(_net,_args,_result): counts[kind+'_completed']+=1; changed()
    left=net.register_forward_pre_hook(start); right=net.register_forward_hook(end)
    try:
        yield
        if counts[kind+'_completed']!=counts[kind+'_started'] or counts[kind+'_started']!=before+1:
            raise ValueError('Exactly one completed forward per slot required')
    finally: left.remove(); right.remove()


@contextlib.contextmanager
def no_optimizer_cuda():
    k.resources()
    with contextlib.ExitStack() as stack:
        for obj,name in ((torch.optim,'Adam'),(torch.cuda,'init'),(torch.cuda,'_lazy_init')):
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('NO_OPTIMIZER_OR_CUDA')))
        yield
    k.resources()


def new_run_required(out):
    if any((out/name).exists() for name in ('run_status.json','inputs.json','rows','slots','references','totals','diagnostic.json','verification.json')):
        raise ValueError('Existing run/evidence; never overwrite or restart')


def make_totals(total,groups,full,direction,reference):
    u,d=direction[:2]; desc=direction[2]
    parts=groups[k.GROUPS[0]]+4*groups[k.GROUPS[1]]+groups[k.GROUPS[2]]+.2*(groups[k.GROUPS[3]]+groups[k.GROUPS[4]])
    complete=groups[k.GROUPS[0]]+4*groups[k.GROUPS[1]]+groups[k.GROUPS[2]]+.2*full
    doc={'canonical_full_vs_authority':n.vector_comparison(total['canonical'],total['authority'],direction=(u,d)),
        'independent_slot_parts_vs_authority_measurement':n.vector_comparison(total['parts'],total['authority'],direction=(u,d)),
        'independent_group_parts_vs_authority_measurement':n.vector_comparison(parts,total['authority'],direction=(u,d)),
        'full_group_sum_vs_authority_measurement':n.vector_comparison(complete,total['authority'],direction=(u,d)),
        'authority_vs_reference':n.vector_comparison(total['authority'],total['reference'],direction=(u,d)) if reference else None,
        'alignment':{name:k.alignment(v,u,d) for name,v in total.items() if reference or name!='reference'},
        'group_alignment':{name:k.alignment(v,u,d) for name,v in groups.items()},
        'weighted_group_alignment':{name:k.alignment(groups[name]*c,u,d) for name,c in zip(k.GROUPS,k.COEFFICIENTS)},
        'full_auxiliary_group_alignment':k.alignment(full,u,d),'direction':desc,'group_additivity_not_assumed':True}
    return acq.seal(doc)


def probe_batch(net,saved,batch,wa,gs,reference,counts,emit=lambda *args:None,changed=lambda:None):
    params=tuple(net.parameters()); u,d,direction=k.stored_direction(saved)
    total={name:torch.zeros_like(u) for name in ('authority','canonical','parts','reference')}
    groups={name:torch.zeros_like(u) for name in k.GROUPS}; full=torch.zeros_like(u)
    slots=[]; refs=[]; vectors=[]
    for index,meta in enumerate(batch['metadata']):
        with k.readonly(net,saved,batch),counted_forward(net,counts,'primary',changed):
            base,wave,losses,info,identity=n.slot_wave(net,batch['x'][index:index+1],batch['v'][index:index+1],wa,gs,meta)
            doc,vec=b.primary_probe(params,base,wave,losses,info,meta,index,(u,d))
        slot=acq.seal({'slot':index,'metadata':copy.deepcopy(meta),'bucket':k.bucket(meta),'auxiliary':info,'identity':identity,
            'probe':doc,'primary_slot_forwards':1,'scalar_split_bit_exact':True,
            'unchanged_weights_modes_grad_rng_moments_inputs':True})
        emit('slots',index,slot); slots.append(slot)  # Measurement before gate, not a passed batch commit.
        b.check_primary(doc,info['active'])
        for name in ('authority','canonical','parts'): total[name]+=vec[name]
        groups[k.group_for_slot(index,meta)]+=vec['base']; groups[k.GROUPS[3]]+=vec['cv2']; groups[k.GROUPS[4]]+=vec['ca1_squared']
        full+=vec['full']; vectors.append(vec['authority'])
    if reference:
        expected=k.input_entry(batch)
        for index,meta in enumerate(batch['metadata']):
            with k.readonly(net,saved,batch),counted_forward(net,counts,'reference',changed):
                vec,identity,info,loss,composition=b.reference_slot(net,batch['x'][index:index+1],batch['v'][index:index+1],wa,gs,meta,index)
            row=slots[index]
            comparison=n.vector_comparison(vectors[index],vec,direction=(u,d))
            ref=acq.seal({'slot':index,'metadata':copy.deepcopy(meta),'input_sha256':expected['input_sha256'][index],
                'target_sha256':expected['target_sha256'][index],'identity':identity,'auxiliary':info,'loss':loss,'composition':composition,
                'authority_vs_reference':comparison,'scalar_wave_mask_identity_bit_exact':
                    k.exact(identity,{key:row['identity'][key] for key in identity}) and k.exact(info,row['auxiliary']) and
                    k.exact(loss,row['probe']['loss']) and k.exact(composition,row['probe']['composition']),
                'reference_slot_forwards':1,'unchanged_weights_modes_grad_rng_moments_inputs':True})
            emit('references',index,ref); refs.append(ref)
            if not ref['scalar_wave_mask_identity_bit_exact']: raise ValueError('Independent original194 scalar/wave identity differs')
            b.require_comparison(comparison,exact=True); total['reference']+=vec
    totals=make_totals(total,groups,full,(u,d,direction),reference)
    emit('totals',0,totals)
    b.require_comparison(totals['canonical_full_vs_authority'])
    if reference: b.require_comparison(totals['authority_vs_reference'],exact=True)
    return {'slots':slots,'reference_rows':refs,'totals':totals,'source_and_inputs_unchanged':True}


def validate_batch(row,descriptor,expected,reference):
    elements=sum(math.prod(v['shape']) for v in descriptor['direction']['parameter_mapping'])
    if len(row['slots'])!=6 or len(row['reference_rows'])!=(6 if reference else 0) or row['source_and_inputs_unchanged'] is not True:
        raise ValueError('Complete bounded model batch required')
    for index,slot in enumerate(row['slots']):
        meta=expected['metadata'][index]
        if (not k.exact(slot['slot'],index) or not k.exact(slot['metadata'],meta) or slot['bucket']!=k.bucket(meta) or
            not k.exact(slot['primary_slot_forwards'],1) or slot['scalar_split_bit_exact'] is not True or
            slot['unchanged_weights_modes_grad_rng_moments_inputs'] is not True or
            set(slot['identity'])!={'wave_sha256','mask_sha256','base','residual','accompaniment','full'}):
            raise ValueError('Changed fixed slot/role/source identity')
        active=slot['auxiliary']['active']; b.check_primary(slot['probe'],active)
        if type(active) is not bool or (active and index>=2) or (not active and any(slot['identity'][x]!=0. for x in ('residual','accompaniment','full'))):
            raise ValueError('Original activity/skip required')
        for key in ('canonical_full_vs_direct','independent_parts_vs_direct_measurement'):
            doc=slot['probe'][key]
            if doc['elements']!=elements or doc['expected_sha256']!=slot['probe']['direct_gradient_sha256']: raise ValueError('Changed authority shape/hash')
        for key,doc in (slot['probe']['paths'] or {}).items():
            if doc['elements']!=(179712 if key=='wave_full_vs_parts' else elements): raise ValueError('Changed numerical support')
        if reference:
            ref=row['reference_rows'][index]
            if (not k.exact(ref['slot'],index) or not k.exact(ref['metadata'],meta) or
                ref['input_sha256']!=expected['input_sha256'][index] or ref['target_sha256']!=expected['target_sha256'][index] or
                ref['scalar_wave_mask_identity_bit_exact'] is not True or not k.exact(ref['reference_slot_forwards'],1) or
                ref['unchanged_weights_modes_grad_rng_moments_inputs'] is not True or
                set(ref['identity'])!={'wave_sha256','mask_sha256','base','full'} or
                not k.exact(ref['identity'],{key:slot['identity'][key] for key in ref['identity']}) or
                not k.exact(ref['auxiliary'],slot['auxiliary']) or not k.exact(ref['loss'],slot['probe']['loss']) or
                not k.exact(ref['composition'],slot['probe']['composition'])): raise ValueError('Changed independent reference identity')
            b.require_comparison(ref['authority_vs_reference'],exact=True)
            if ref['authority_vs_reference']['actual_sha256']!=slot['probe']['direct_gradient_sha256'] or ref['authority_vs_reference']['elements']!=elements:
                raise ValueError('Changed reference authority')
    totals=row['totals']
    if totals['group_additivity_not_assumed'] is not True or not k.exact(totals['direction'],descriptor['direction']): raise ValueError('Changed stored direction/groups')
    for key in TOTAL_COMPARISONS:
        n.validate_comparison(totals[key])
        if totals[key]['elements']!=elements or totals[key]['expected_sha256']!=totals['alignment']['authority']['gradient_sha256']:
            raise ValueError('Changed total authority shape/hash')
    b.require_comparison(totals['canonical_full_vs_authority'])
    if set(totals['alignment'])!=({'authority','canonical','parts','reference'} if reference else {'authority','canonical','parts'}): raise ValueError('Missing named totals')
    if totals['canonical_full_vs_authority']['actual_sha256']!=totals['alignment']['canonical']['gradient_sha256'] or totals['independent_slot_parts_vs_authority_measurement']['actual_sha256']!=totals['alignment']['parts']['gradient_sha256']:
        raise ValueError('Changed named total gradient identity')
    if reference:
        b.require_comparison(totals['authority_vs_reference'],exact=True)
        if (totals['authority_vs_reference']['elements']!=elements or
            totals['authority_vs_reference']['actual_sha256']!=totals['alignment']['authority']['gradient_sha256'] or
            totals['authority_vs_reference']['expected_sha256']!=totals['alignment']['reference']['gradient_sha256']): raise ValueError('Changed total reference')
    elif totals['authority_vs_reference'] is not None: raise ValueError('Reference only first batch per model')
    for name in ('group_alignment','weighted_group_alignment'):
        if set(totals[name])!=set(k.GROUPS): raise ValueError('All five independently measured groups required')
    for collection in (totals['alignment'],totals['group_alignment'],totals['weighted_group_alignment']):
        for value in collection.values(): k.validate_alignment(value)
    k.validate_alignment(totals['full_auxiliary_group_alignment'])
    if not k.d.finite_state(row): raise ValueError('Nonfinite canonical batch')


def status(out,state,phase,counts,completed,error=None):
    tmp=out/f'status201_{os.getpid()}.tmp'
    acq.write_new_json(tmp,{'purpose':PURPOSE,'status':state,'phase':phase,'counts':copy.deepcopy(counts),
        'completed_model_batches':completed,'limit':9,'error':error,'pid':os.getpid(),'updated_utc':m.bulk.now(),
        'cuda_used':False,'optimizer_constructed':False,'model_updates':0,'release_selection':'NONE'})
    os.replace(tmp,out/'run_status.json')


def run(out):
    new_run_required(out); no_active_task(); k.resources(); plan=read_plan(out)
    authority,models,descriptors,sampler,expected,_=source()
    if not k.exact((descriptors,sampler,expected),(plan['model_descriptors'],plan['sampler'],plan['expected_inputs'])): raise ValueError('Changed sources/inputs')
    stamps=k.binding_stamps(list(plan['bindings_sha256'])+[out/'plan.json'])
    source_digest=k.tree_digest((authority,models,descriptors,sampler)); counts=new_counts(); rows=[]; phase='initialization'
    torch.set_num_threads(2)
    with m.bulk.worker_lock(out),k.d.deterministic_runtime('cpu'),no_optimizer_cuda():
        if not k.exact(k.d.runtime_identity('cpu'),plan['expected_runtime']): raise ValueError('Changed strict CPU runtime')
        status(out,'running',phase,counts,0)
        try:
            dataset=k.p.inp.ApprovedTeacherDataset(k.p.inp.verified_approval(Path(authority['origin_approval'])),'kim_melband')
            true=m.LockedTruePool(m.bulk.OLD_LOCK,dataset.config)
            if sampler['cursor']!=4500 or sampler['seed']!=dataset.seed or sampler['true_lock_sha256']!=true.bound: raise ValueError('Changed original4500 stream identity')
            before_rng=k.tree_digest(m.capture_rng('cpu')); batches=[k.collect_next(dataset,true,c) for c in k.CURSORS]
            if before_rng!=k.tree_digest(m.capture_rng('cpu')) or not k.exact([k.input_entry(x) for x in batches],expected): raise ValueError('Retained197 input hashes/RNG required; no resampling')
            acq.write_new_json(out/'inputs.json',acq.seal({'purpose':PURPOSE,'inputs':expected,'rng_unchanged':True}))
            for name in ('rows','slots','references','totals'): (out/name).mkdir()
            wa,gs,_=k.matrix_identity()
            for name in k.MODELS:
                rng=k.d.portable(m.capture_rng('cpu'))
                try: net=m.frozen_factory(authority['source_protocol'])()
                finally: m.restore_rng(rng,'cpu')
                saved=models[name]; net.load_state_dict(saved['model'],strict=True)
                if len(saved['modes'])!=len(list(net.modules())) or any(type(flag) is not bool for flag in saved['modes']): raise ValueError('All source modes required')
                for mod,flag in zip(net.modules(),saved['modes']): mod.training=flag
                if (k.tree_digest(dict(net.state_dict()))!=k.tree_digest(dict(saved['model'])) or
                    saved['parameter_names']!=[key for key,_ in net.named_parameters()] or any(not v.requires_grad for v in net.parameters()) or
                    any(isinstance(mod,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout,torch.nn.Dropout2d,torch.nn.Dropout3d)) for mod in net.modules())):
                    raise ValueError('Complete original deterministic model/order required')
                for index,batch in enumerate(batches):
                    phase=f'{name}/counter{batch["diagnostic_counter"]}'; row_index=len(rows)
                    def changed(): status(out,'running',phase,counts,len(rows))
                    def emit(directory,slot,doc):
                        filename=f'batch_{row_index:02d}.json' if directory=='totals' else f'batch_{row_index:02d}_slot_{slot:02d}.json'
                        acq.write_new_json(out/directory/filename,doc)
                    k.resources(); k.unchanged_bindings(stamps)
                    with k.readonly(net,saved,batch): probe=probe_batch(net,saved,batch,wa,gs,index==0,counts,emit,changed)
                    row=acq.seal({'model':name,'counter':batch['diagnostic_counter'],**probe})
                    validate_batch(row,descriptors[name],expected[index],index==0)
                    if source_digest!=k.tree_digest((authority,models,descriptors,sampler)): raise ValueError('Global source/schedule changed')
                    k.unchanged_bindings(stamps); acq.write_new_json(out/'rows'/f'row_{row_index:02d}.json',row); rows.append(row)
                    status(out,'running',phase,counts,len(rows)); print(f'CANONICAL_MEMORY model={name} counter={batch["diagnostic_counter"]} completed={len(rows)}/9 counts={counts}',flush=True)
            k.p.q.check_bindings(plan['bindings_sha256']); k.unchanged_bindings(stamps)
            result=scope()|{'plan_sha256':acq.sha256(out/'plan.json'),'inputs_sha256':acq.sha256(out/'inputs.json'),
                'rows':rows,'counts':counts,'runtime':k.d.runtime_identity('cpu'),'source_state_unchanged':True,'error':None}
            validate_result(result,plan); acq.write_new_json(out/'diagnostic.json',acq.seal(result))
            status(out,'complete','canonical9_requires_independent_verify',counts,9)
            print('CANONICAL_MEMORY COMPLETE9/54+18;new canonical recovery,old197failed retained;NONRELEASE',flush=True)
        except BaseException as error:
            status(out,'failed',phase,counts,len(rows),repr(error)); raise


def validate_result(result,plan):
    check_scope(result)
    if (len(result['rows'])!=9 or not k.exact(result['counts'],{'primary_started':54,'primary_completed':54,'reference_started':18,'reference_completed':18}) or
        result['error'] is not None or result['source_state_unchanged'] is not True or not k.exact(result['runtime'],plan['expected_runtime'])):
        raise ValueError('Complete9/54+18 readonly result required')
    for i,row in enumerate(result['rows']):
        model=k.MODELS[i//3]; counter=k.CURSORS[i%3]
        if row['model']!=model or not k.exact(row['counter'],counter): raise ValueError('Changed model/counter order')
        validate_batch(row,plan['model_descriptors'][model],plan['expected_inputs'][i%3],i%3==0)


def symmetric_document(outside,inside): e.symmetric_totals(outside,inside)


def check_payload(out,plan,result):
    # Parent seals and full nested seals are checked separately, including totals.
    symmetric_document(acq.read_sealed(out/'plan.json'),plan)
    symmetric_document(acq.read_sealed(out/'diagnostic.json'),result)
    inp=acq.read_sealed(out/'inputs.json')
    if (result['plan_sha256']!=acq.sha256(out/'plan.json') or result['inputs_sha256']!=acq.sha256(out/'inputs.json') or
        inp['rng_unchanged'] is not True or not k.exact(inp['inputs'],plan['expected_inputs'])): raise ValueError('Changed parent/input hashes')
    expected={'rows':9,'slots':54,'references':18,'totals':9}
    for directory,count in expected.items():
        if len(list((out/directory).glob('*.json')))!=count: raise ValueError('Missing/extra committed evidence')
    for i,row in enumerate(result['rows']):
        symmetric_document(acq.read_sealed(out/'rows'/f'row_{i:02d}.json'),row)
        symmetric_document(acq.read_sealed(out/'totals'/f'batch_{i:02d}.json'),row['totals'])
        for directory,key in (('slots','slots'),('references','reference_rows')):
            for j,doc in enumerate(row[key]): symmetric_document(acq.read_sealed(out/directory/f'batch_{i:02d}_slot_{j:02d}.json'),doc)
    validate_result(result,plan)


def verify(out):
    if (out/'verification.json').exists(): raise ValueError('Existing successful verification; do not repeat')
    no_active_task(); k.resources()
    with e.no_compute():
        plan=read_plan(out); result=acq.read_sealed(out/'diagnostic.json')
        ending=json.loads((out/'run_status.json').read_text(encoding='utf-8'))
        expected={'status':'complete','completed_model_batches':9,'limit':9,'error':None,'cuda_used':False,'model_updates':0,
            'optimizer_constructed':False,'counts':{'primary_started':54,'primary_completed':54,'reference_started':18,'reference_completed':18}}
        if any(not k.exact(ending.get(key),value) for key,value in expected.items()): raise ValueError('Complete actual native run evidence required')
        check_payload(out,plan,result)
    acq.write_new_json(out/'verification.json',acq.seal({'purpose':PURPOSE,'plan_sha256':acq.sha256(out/'plan.json'),
        'diagnostic_sha256':acq.sha256(out/'diagnostic.json'),'verified_model_batches':9,'primary_slots':54,'reference_slots':18,
        'model_forward_count':0,'autograd_calls':0,'optimizer_constructed':False,'cuda_used':False,'model_updates':0,
        'independently_verified':True,'all_parent_rows_totals_both_seals_full_symmetric_typed':True,
        'canonical_and_VJP_strict_gates_passed':True,'first_batch_reference_bit_exact_each_model':True,
        'old_split_strict_failures_retained':True,'original197_rewritten_or_merged':False,'training_authorized':False,'release_selection':'NONE'}))
    print('CANONICAL_MEMORY VERIFIED9;0 forward;original197 remains failed;NONRELEASE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('operation',choices=('prepare','run','verify'))
    parser.add_argument('--out',type=Path,default=OUT); parser.add_argument('--unit-evidence',type=Path,default=MONITOR/'unit_gate.json')
    args=parser.parse_args()
    prepare(args.out,args.unit_evidence) if args.operation=='prepare' else globals()[args.operation](args.out)
