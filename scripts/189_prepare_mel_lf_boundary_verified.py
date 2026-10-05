"""Fresh authority wrapper after independently corrected unit precondition.

No historical code/plan/test is edited; no real TRAIN audit is repeated.
Both original failure and zero-update audit are retained and hash-bound.
"""
import argparse
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location("sealed_lf_preparer",Path(__file__).with_name("185_prepare_mel_lf_boundary.py"))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
ROOT,acq=p.ROOT,p.acq
DEFAULT_OUT=ROOT/"results/mel_lf_boundary_gate_recovery_import_20261003"
GATE=ROOT/"docs/mel_lf_boundary_gate_recovery_20261003.json"
MONITOR=ROOT/"results/mel_lf_boundary_monitor_20261003"

def check_gate(doc):
    expected={"schema":1,"purpose":"NONRELEASE_LF_BOUNDARY_UNIT_PRECONDITION_RECOVERY",
        "old_unit_count":35,"old_actual_exit_code":1,"failed_test":"test_low_frequency_descriptor_does_not_modify_wave",
        "recovery_unit_count":35,"unchanged_inherited_tests":34,"fresh_cpu_isolation_cases":1,
        "original_code_test_audit_preserved":True,"rerun_real_train_audit":False,
        "new_training_limit":4000,"additional_common_steps":500,"source_arm":p.SOURCE_ARM,
        "model_recipe_changed":False,"release_selection":"NONE","deployment":False}
    if not p.exact(doc,expected):
        raise ValueError("Exact unit-precondition recovery, NOT model/recipe recovery")

def prepare(out):
    p.old.require_fresh(out);p.no_active_trainer()
    gate=json.loads(GATE.read_text(encoding="utf-8"));check_gate(gate)
    failure=(MONITOR/"unit_tests.log").read_text(encoding="utf-8-sig")
    passed=(MONITOR/"recovery_unit_tests.log").read_text(encoding="utf-8-sig")
    if ("FAILED (failures=1)" not in failure or "Ran 35 tests" not in failure or
        "Ran 35 tests" not in passed or "\nOK" not in passed or "FAILED" in passed):
        raise ValueError("Original real failure and all35 recovered tests required")
    evidence=json.loads((MONITOR/"gate_recovery_execution.json").read_text(encoding="utf-8"))
    if evidence["original_actual_exit_code"]!=1 or evidence["recovery_actual_exit_code"]!=0 or evidence["count"]!=35:
        raise ValueError("Real exec exit evidence required")
    doc=p.draft_document();p.verify_audit()
    files=[Path(__file__),GATE,Path(__file__).with_name("_test_mel_lf_boundary_gate_recovery.py"),
        Path(__file__).with_name("190_start_mel_lf_boundary_verified.ps1"),
        ROOT/"reports/75_mel_lf_boundary_gate_recovery.md",MONITOR/"recovery_unit_tests.log",
        MONITOR/"gate_recovery_execution.json",p.AUDIT_OUT/"plan.json",p.AUDIT_OUT/"audit.json"]
    doc["bindings_sha256"] |= {str(path.resolve()):acq.sha256(path) for path in files}
    doc["unit_gate_recovery"]=gate
    out.mkdir(parents=True)
    acq.write_new_json(out/"approval.json",acq.seal(doc))
    p.verified_approval(out/"approval.json")
    print("LF_BOUNDARY RECOVERY_IMPORT SEALED; real CPU/CUDA proofs still required",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("prepare","verify"))
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.operation=="prepare":prepare(args.out)
    else:
        p.verified_approval(args.out/"approval.json")
        print("LF_BOUNDARY RECOVERY_IMPORT VERIFIED",flush=True)

