"""Isolated zero-compute proof of immutable199 evidence; never rerun its verify."""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import re
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location('immutable199', Path(__file__).with_name('199_verify_train_adam_canonical_bridge.py'))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
k, acq, ROOT = b.k, b.acq, b.ROOT
PURPOSE = 'NONRELEASE_ZERO_FORWARD_BRIDGE_EVIDENCE_RECOVERY'
OUT = ROOT/'results/train_adam_bridge_evidence_recovery_20261004'
MONITOR = ROOT/'results/train_adam_bridge_evidence_recovery_monitor_20261004'
TEST = Path(__file__).with_name('_test_train_adam_bridge_evidence.py')
PROTOCOL = ROOT/'docs/train_adam_bridge_evidence_protocol_20261004.json'
FAILURE = b.MONITOR/'failure_review.json'


def scope():
    return {'schema':1,'purpose':PURPOSE,'source_out':str(b.OUT.resolve()),
        'model_forwards':0,'autograd_calls':0,'optimizer_constructed':False,'model_updates':0,
        'cuda_used':False,'source_primary_slots':6,'source_reference_slots':6,
        'source_unique_batches':1,'source_unique_slots':6,'source_counter':4500,
        'rtol':.0002,'atol':.0000002,'totals_comparison':'both_seals_then_full_type_sensitive_symmetric',
        'old199_verify_actual_exit':1,'original197_recovered':False,
        'training_authorized':False,'release_selection':'NONE'}


def check_scope(doc):
    if any(not k.exact(doc.get(key),value) for key,value in scope().items()):
        raise ValueError('Changed zero-compute recovery scope/types')


def symmetric_totals(outside, inside):
    # Validate BOTH seals BEFORE comparing FULL documents. No side drops fields.
    for doc in (outside,inside):
        if acq.content_digest(doc) != doc.get('content_sha256'):
            raise ValueError('Invalid independently sealed totals')
    if not k.exact(outside,inside):
        raise ValueError('Full symmetric type-sensitive totals mismatch')


def validate_payload(source_plan, inp, totals, result, ending):
    expected={'status':'complete','counts':{key:6 for key in b.new_counts()},
        'committed_primary_rows':6,'committed_reference_rows':6,'model_updates':0,
        'cuda_used':False,'optimizer_constructed':False,'error':None}
    if any(not k.exact(ending.get(key),value) for key,value in expected.items()):
        raise ValueError('Actual complete6+6 original status required')
    if inp['rng_unchanged'] is not True or not k.exact(inp['input'],source_plan['expected_input']):
        raise ValueError('Original input/RNG identity required')
    symmetric_totals(totals,result['totals'])
    b.validate_result(result,source_plan)  # Original strict/bit-exact gates unchanged.


@contextlib.contextmanager
def no_compute():
    if torch.cuda.is_initialized(): raise ValueError('Unexpected initialized CUDA')
    before=k.tree_digest(b.m.capture_rng('cpu'))
    try:
        with contextlib.ExitStack() as stack:
            for obj,name in ((torch.nn.Module,'__call__'),(torch.autograd,'grad'),
                (torch.autograd,'backward'),(torch.Tensor,'backward'),(torch.optim,'Adam'),
                (torch.cuda,'init'),(torch.cuda,'_lazy_init')):
                stack.enter_context(patch.object(obj,name,side_effect=AssertionError('ZERO_COMPUTE_REQUIRED')))
            yield
    finally:
        if torch.cuda.is_initialized() or before != k.tree_digest(b.m.capture_rng('cpu')):
            raise ValueError('CUDA/RNG changed during zero-compute verification')


def no_active_task():
    b.no_active_task()
    pattern=re.escape(str(ROOT))+r'|200_verify_train_adam_bridge_evidence|_test_train_adam_bridge_evidence'
    cmd="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '"+pattern+"' } | Select-Object -ExpandProperty ProcessId"
    proc=k.subprocess.run(['powershell.exe','-NoProfile','-Command',cmd],capture_output=True,text=True,check=True)
    if any(int(line.strip()) not in {os.getpid(),os.getppid()} for line in proc.stdout.splitlines() if line.strip()):
        raise ValueError('Active repository evidence task; postpone without killing')


def evidence():
    if (b.OUT/'verification.json').exists():
        raise ValueError('Original failed verify unexpectedly replaced')
    source_plan=b.read_plan(b.OUT)
    result=acq.read_sealed(b.OUT/'diagnostic.json')
    inp=acq.read_sealed(b.OUT/'inputs.json'); totals=acq.read_sealed(b.OUT/'totals.json')
    ending=json.loads((b.OUT/'run_status.json').read_text(encoding='utf-8'))
    for name in ('plan','inputs','totals'):
        if result[name+'_sha256'] != acq.sha256(b.OUT/(name+'.json')):
            raise ValueError('Changed original parent file hash')
    b.symmetric_rows(b.OUT,result['rows'])
    b.symmetric_rows(b.OUT,result['reference_rows'],'reference_rows','reference')
    validate_payload(source_plan,inp,totals,result,ending)
    failure=acq.read_sealed(FAILURE)
    if (not k.exact(failure['real_run']['actual_native_exit_code'],0) or
        not k.exact(failure['real_verify']['actual_native_exit_code'],1) or
        failure['real_verify']['independently_verified'] is not False or
        failure['bridge_external_proof_complete'] is not False or
        failure['original197_recovered'] is not False):
        raise ValueError('Original real run0/verify1 separation required')
    bindings=dict(source_plan['bindings_sha256'])
    for item in failure['file_bindings']:
        path=ROOT/item['path']
        if acq.sha256(path) != item['sha256']: raise ValueError('Changed old failure evidence')
        bindings[str(path.resolve())]=item['sha256']
    for path in (Path(__file__),TEST,PROTOCOL,FAILURE,
        ROOT/'reports/95_train_adam_canonical_bridge_verify_failure.md',
        ROOT/'reports/96_train_adam_bridge_evidence_recovery_plan.md'):
        bindings[str(path.resolve())]=acq.sha256(path)
    return bindings,result


def unit_gate(path):
    gate=acq.read_sealed(path)
    if (not k.exact(gate['actual_exit_code'],0) or gate['draft_reviewed'] is not True or
        type(gate['tests_passed']) is not int or gate['tests_passed']<20):
        raise ValueError('Actual new serialization units and reviewed draft required')
    for key,file in (('tool_sha256',Path(__file__)),('test_sha256',TEST),
        ('protocol_sha256',PROTOCOL),('log_sha256',Path(gate['log']))):
        if gate[key] != acq.sha256(file): raise ValueError('Changed actual new unit binding')
    log=Path(gate['log']).read_text(encoding='utf-8-sig')
    if not re.search(r'Ran '+str(gate['tests_passed'])+r' tests? in',log) or not re.search(r'(?m)^OK\s*$',log):
        raise ValueError('Complete successful native unit output required')
    return gate


def prepare(out,unit):
    k.fresh(out); no_active_task(); k.resources(); gate=unit_gate(unit)
    if not k.exact(json.loads(PROTOCOL.read_text(encoding='utf-8')),scope()):
        raise ValueError('Changed actual protocol')
    with no_compute(): bindings,_=evidence()
    bindings[str(unit.resolve())]=acq.sha256(unit); bindings[gate['log']]=gate['log_sha256']
    out.mkdir(parents=True)
    acq.write_new_json(out/'plan.json',acq.seal(scope()|{'bindings_sha256':bindings,'unit_gate':str(unit.resolve())}))
    print('ZERO_FORWARD_EVIDENCE PLAN SEALED; old199verify1 retained; no recovery9/training',flush=True)


def verify(out):
    if any((out/name).exists() for name in ('verification.json','run_status.json')):
        raise ValueError('Existing proof/status; no repeat or overwrite')
    no_active_task(); k.resources(); plan=acq.read_sealed(out/'plan.json'); check_scope(plan)
    unit_gate(Path(plan['unit_gate'])); k.p.q.check_bindings(plan['bindings_sha256'])
    try:
        with no_compute():
            bindings,result=evidence()
            if any(plan['bindings_sha256'].get(path)!=digest for path,digest in bindings.items()):
                raise ValueError('Changed recovery source binding set')
        k.p.q.check_bindings(plan['bindings_sha256'])
        proof=scope()|{'plan_sha256':acq.sha256(out/'plan.json'),'source_diagnostic_sha256':acq.sha256(b.OUT/'diagnostic.json'),
            'source_totals_sha256':acq.sha256(b.OUT/'totals.json'),'source_failure_review_sha256':acq.sha256(FAILURE),
            'both_row_sets_sealed_symmetric_type_sensitive':True,'totals_both_seals_full_symmetric_type_sensitive':True,
            'original_structural_gates_passed':True,'source_bindings_unchanged':True,
            'cpu_rng_unchanged':True,'independently_verified':True,'bridge_evidence_recovered':True,
            'source_old_split_strict_failures_retained':True,'checked_utc':b.m.bulk.now()}
        acq.write_new_json(out/'verification.json',acq.seal(proof))
        acq.write_new_json(out/'run_status.json',scope()|{'status':'complete','error':None,'pid':os.getpid(),'updated_utc':b.m.bulk.now()})
        print('ZERO_FORWARD_EVIDENCE VERIFIED; original structural bridge gates passed; old199verify1/197failed retained; NONRELEASE',flush=True)
    except BaseException as error:
        acq.write_new_json(out/'run_status.json',scope()|{'status':'failed','error':repr(error),'pid':os.getpid(),'updated_utc':b.m.bulk.now()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('operation',choices=('prepare','verify'))
    parser.add_argument('--out',type=Path,default=OUT); parser.add_argument('--unit-evidence',type=Path,default=MONITOR/'unit_gate.json')
    args=parser.parse_args()
    prepare(args.out,args.unit_evidence) if args.operation=='prepare' else verify(args.out)
