"""Independent LR1/half fixed500 fork, from LF32 ARMS[1]4000."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import types
import re
import torch

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

q = load("lf_historical_source_checks", "181_diagnose_train_context_reconstruction.py")
t, p, m, d, acq, ROOT = q.t, q.p, q.m, q.d, q.acq, q.ROOT
old, inp, ARMS = t.old, p.inp, t.ARMS
PURPOSE = "NONRELEASE_MEL_LR_SCALE_FORK"
ROLES = dict(zip(ARMS, ("lr1_control", "lr_half_candidate")))
KILLS = dict(zip(ARMS, (32, 32)))
LR_SCALES = dict(zip(ARMS, (1.0, .5)))
LAMBDAS = dict(zip(ARMS, (.2, .2)))
WEIGHTS = dict(zip(ARMS, (4, 4)))
CA_WEIGHTS = dict(zip(ARMS, (1, 1)))
SLOT_WEIGHTS = {arm: [1, 1, 4, 1, 1, 1] for arm in ARMS}
SOURCE_ARM, START, LIMIT = ARMS[1], 4000, 4500
reference = load("lr_source186", "186_train_mel_lf_boundary.py")
SOURCE = reference.DEFAULT_OUT
SOURCE_APPROVAL = ROOT / "results/mel_lf_boundary_gate_recovery_import_20261003/approval.json"
UNIT_GATE = ROOT / "results/mel_lr_scale_monitor_20261004/unit_gate.json"
SOURCE_SHA = "94323f8a30fb5e5decb65d4c110fd5252d0916de68507f42af555b3f539a8f6d"
PROTOCOL = ROOT / "docs/mel_lr_scale_protocol_20261004.json"
DEFAULT_OUT = ROOT / "results/mel_lr_scale_import_20261004"
DEFAULT_APPROVAL = DEFAULT_OUT / "approval.json"
AUDIT_OUT = ROOT / "results/mel_lr_scale_audit_20261004"
REQUIRED = ("194_train_mel_lr_scale.py", "195_start_mel_lr_scale.ps1",
            "196_review_mel_lr_scale.py", "_test_mel_lr_scale.py")
HARDWARE = ROOT / "hardware/generated/bott2_mir1k_v1_program"

def exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)

def require_kills(value):
    if not exact(value, KILLS):
        raise ValueError("Both arms exact LF32, not bool/float/other threshold")

def require_lr_scales(value):
    if not exact(value, LR_SCALES):
        raise ValueError("Exact declared LR1/half multipliers required")

def require_weights(value):
    if not exact(value, WEIGHTS):
        raise ValueError("Exact fixed instrumental4")

def require_ca_weights(value):
    if not exact(value, CA_WEIGHTS):
        raise ValueError("Both arms retain full auxiliary CA weight1")

def fixed_protocol():
    return {"schema": 1, "purpose": PURPOSE, "exploratory_training_authorized": True,
        "formal_training_authorized": False, "deployment_authorized": False, "teacher": "kim_melband",
        "approved_pseudo_songs": 24, "origin_step": START, "additional_common_steps": 500,
        "absolute_limit": LIMIT, "checkpoint_every": 250, "source_arm": SOURCE_ARM,
        "source_stopped_at": START, "source_legacy_stop_events": [3750, 4000],
        "source_stale": 16, "source_best": None, "source_patience_anchor": None,
        "tranche_stopping": "new_common_fixed500_budget_preserve_legacy_stop_evidence",
        "internal_arm_keys": ROLES, "arm_lambdas": LAMBDAS, "arm_instrumental_weights": WEIGHTS,
        "arm_slot_weights": SLOT_WEIGHTS, "arm_accompaniment_weights": CA_WEIGHTS, "arm_kill_bands": KILLS,
        "base_normalizer": 6, "gpu_concurrency_authorized": True, "minimum_free_mib": 2300,
        "minimum_disk_gib": 12, "mechanism_step_limit": 3, "release_selection": "NONE",
        "independent_acceptance_ready": False, "auxiliary_formula": "cv^2+(ca-1)^2",
        "rms_floor": .0001, "minimum_gram_determinant": .001, "auxiliary_domains": ["musdb", "mir1k"],
        "auxiliary_role": "train", "origin_checkpoint_sha256": SOURCE_SHA,
        "reference_diagnostic_sha256": "d4c63ecb5ce5c07689410e4193228935259edc1c455b8d57b0f5fb615c26f147",
        "arm_lr_scales": LR_SCALES, "lr_application": "preserve_source_group_lr_at4000_apply_multiplier_from4001",
        "lf_application": ["training_reconstruction", "development", "listening"],
        "baseline_interpretation": "Same model/Adam and LF32; common4000 baseline, not training gain",
        "board_changes_authorized": False}

def check_protocol(doc):
    if not exact(doc, fixed_protocol()):
        raise ValueError("Exact independent fixed500 LF-boundary authority required")

def no_active_trainer():
    names = "186_train_mel_lf_boundary|188_review_mel_lf_boundary|191_diagnose_train_lf_cross_boundary|192_diagnose_train_lf_cross_boundary_recovery|196_review_mel_lr_scale|134_generate_teacher_library|139_generate_paired_htdemucs|150_train_paired_exploration|154_train_mel_weak_weight|161_train_mel_source_aux|166_train_mel_source_strength|172_train_mel_instrumental_protection|178_train_mel_component_ablation|158_diagnose_mel_loss_direction|164_diagnose_source_aux_strength|169_diagnose_gradient_contributions|175_diagnose_auxiliary_components|181_diagnose_train_context_reconstruction|182_diagnose_train_context_reconstruction_recovery|184_diagnose_train_low_frequency_reference|194_train_mel_lr_scale"
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?[.]exe$' -and $_.CommandLine -match '(" + names + ")[.]py[\" ]+(train|run|smoke|summary|listen)' } | Select-Object -ExpandProperty ProcessId"
    reply = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                           check=True, capture_output=True, text=True)
    own = {os.getpid(), os.getppid()}
    if any(int(line.strip()) not in own for line in reply.stdout.splitlines() if line.strip()):
        raise ValueError("Active task; preserve and postpone")

def checked_source():
    """Metadata and sealed existing evidence only; never rerun historical verification."""
    authority = reference.p.verified_approval(SOURCE_APPROVAL)
    bindings = dict(authority["bindings_sha256"])
    pins = {
        "reports/85_mel_lr_scale_trial_plan.md": "48b752ccf5421fe9b26b81da475d614f22821e6b974e5a877f1d504d2447be11",
        "reports/84_train_lf_cross_boundary_result.md": "bef09dd1ae68e07e174afdff06ecc1cbbdca918c7738159759266f731575d365",
        "reports/79_mel_lf_boundary_completion.md": "dbf3d3afbe2a8c15c09eca9e936c1c94f2de4f1563f4a6d77e649777cfdf933e",
        "results/mel_lf_boundary_monitor_20261003/completion_review.json": "b863b1c4793c92d36b3d5aebfbd824f33f56289fc3624e83054791e1318c5d63",
        "results/train_lf_cross_boundary_recovery_monitor_20261004/completion_review.json": "e7e1d4699c15cdaca8d891c60b4795e16d758477f550086f928dde6b37c1e9bd",
        "results/train_lf_cross_boundary_recovery_monitor_20261004/aggregation.json": "83930bf24c1048b0b89d969b682052881af16a966230145e794868f0023331ba",
        "results/train_lf_cross_boundary_monitor_20261004/failure_review.json": "e177fe599844648c182e84718bbfadfd066a72c8253890c62c00111133047bda",
    }
    bindings |= {str(ROOT / path): sha for path, sha in pins.items()}
    q.check_bindings(bindings)
    proof = json.loads((ROOT / "results/mel_lf_boundary_monitor_20261003/completion_review.json").read_text(encoding="utf-8"))
    terminal, ending = proof["terminal"], proof["actual_launch_exit"]
    launch_path, exit_path = ROOT / ending["launch_path"], ROOT / ending["exit_path"]
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    exit_doc = json.loads(exit_path.read_text(encoding="utf-8-sig"))
    if (terminal["status"] != "complete" or terminal["step"] != START or terminal["additional_steps"] != 500 or
        terminal["limit"] != START or terminal["error"] is not None or terminal["formal_round_completed"] is not True or
        ending["match_confirmed"] is not True or not q.same_exit_record(exit_doc, ending["receipt"]) or
        type(exit_doc["exit_code"]) is not int or exit_doc["exit_code"] != 0 or
        Path(exit_doc["launch_receipt"]) != launch_path or launch["probe_only"] or Path(launch["out"]) != SOURCE or
        exit_doc["launcher_process_id"] != launch["worker"]["LauncherProcessId"] or
        proof["worker_evidence"]["alive"] is not False or proof["worker_evidence"]["active_repo_python"] or
        proof["final_verify"]["actual_native_exit_code"] != 0 or proof["summary"]["actual_native_exit_code"] != 0 or
        proof["summary"]["built_in_verify_passed"] is not True or
        (proof["summary"]["tracks"], proof["summary"]["views"]) != (31,177)):
        raise ValueError("Existing source complete500/matching actual exit0/review identity required")
    bindings |= {str(ROOT / path): sha for path,sha in proof["bindings"].items()}
    cp = json.loads((ROOT / "results/train_lf_cross_boundary_recovery_monitor_20261004/completion_review.json").read_text(encoding="utf-8"))
    if (cp["actual_run_native_exit_code"] != 0 or cp["independent_verify"]["actual_native_exit_code"] != 0 or
        cp["no_active_worker_or_shim"] is not True or cp["processes"] or
        (cp["completed_model_batches"],cp["completed_model_input_slots"],cp["completed_slot_boundary_reconstructions"]) != (36,216,432) or
        cp["model_updates"] != 0 or cp["cuda_used"] is not False or cp["failed191_preserved"] is not True):
        raise ValueError("Existing independently verified192 plus preserved191 failure required")
    bindings |= {str(ROOT / path): sha for path,sha in cp["bindings"].items()}
    prior = acq.read_sealed(ROOT / "results/train_lf_cross_boundary_recovery_20261004/plan.json")
    bindings |= prior["bindings_sha256"]
    q.check_bindings(bindings)
    receipt = acq.read_sealed(SOURCE / "checkpoint_4000.json")
    state = m.load_checked_checkpoint(SOURCE / receipt["checkpoint"], SOURCE_SHA)
    if (receipt["sha256"] != SOURCE_SHA or receipt["binding"] != reference.binding(SOURCE_APPROVAL) or
        state["binding"] != receipt["binding"] or state["step"] != START or state["limit"] != START or
        state["sampler"]["cursor"] != START or state["schedule"]["step"] != START or state["schedule"]["last_validation"] != START or
        state["schedule"]["stopped_at"] != START or state["legacy_stop_events"] != [3750,4000] or
        state["arm_roles"][SOURCE_ARM] != "lf32_candidate" or state["arm_kill_bands"][SOURCE_ARM] != 32 or
        state["arm_accompaniment_weights"][SOURCE_ARM] != 1 or state["arm_instrumental_weights"][SOURCE_ARM] != 4 or state["smoke"]):
        raise ValueError("Exact completed4000 LF32 ARMS[1] source required")
    for field, value in (("stale",16),("best",None),("patience_anchor",None)):
        if not exact(state["schedule"][field][SOURCE_ARM], value):
            raise ValueError("Actual source cumulative schedule differs")
    packet = acq.read_sealed(SOURCE / "development_step_4000.json")
    for arm,saved in state["arms"].items():
        if saved["updates"] != START or old.dev.state_digest(saved["model"]) != packet["model_state_sha256"][arm]:
            raise ValueError("Source model/score exposure mismatch")
        groups = saved["optimizer"]["param_groups"]
        if (len(groups) != 1 or groups[0]["lr"] != m.learning_rate(START,state["schedule"]["config"]) or
            set(saved["optimizer"]["state"]) != set(groups[0]["params"]) or
            any(float(v["step"]) != START for v in saved["optimizer"]["state"].values())):
            raise ValueError("Source Adam exposure/LR incomplete")
    files = [SOURCE / "checkpoint_4000.json", SOURCE / receipt["checkpoint"], SOURCE / "development_step_4000.json",
             SOURCE / "completion.json", SOURCE_APPROVAL]
    bindings |= {str(path.resolve()): acq.sha256(path) for path in files}
    return authority, state, bindings

def check_approval(doc):
    check_protocol(doc["protocol"])
    for key, value in (("purpose", PURPOSE), ("deployment_authorized", False), ("release_selection", "NONE"),
            ("arm_roles", ROLES), ("arm_lambdas", LAMBDAS), ("arm_instrumental_weights", WEIGHTS),
            ("arm_accompaniment_weights", CA_WEIGHTS), ("arm_kill_bands", KILLS), ("arm_lr_scales", LR_SCALES), ("teacher", "kim_melband"),
            ("source_arm", SOURCE_ARM), ("source_approval", str(SOURCE_APPROVAL)),
            ("source_binding", reference.binding(SOURCE_APPROVAL))):
        if not exact(doc.get(key), value):
            raise ValueError("Changed LF authority/source/roles")

def verified_approval(path):
    doc = acq.read_sealed(path)
    check_approval(doc)
    q.check_bindings(doc["bindings_sha256"])
    reference.p.verified_approval(Path(doc["source_approval"]))
    inp.verified_approval(Path(doc["origin_approval"]))
    return doc

def origin_state(doc):
    authority, state, _ = checked_source()
    if (Path(doc["origin_checkpoint"]) != SOURCE / "NONRELEASE_lf_boundary_step_4000.pt" or
        acq.sha256(doc["origin_checkpoint"]) != SOURCE_SHA or
        Path(doc["origin_approval"]) != Path(authority["origin_approval"])):
        raise ValueError("Exact source4000 full-control checkpoint required")
    return state

class BoundaryStream:
    def __init__(self, path=None, preparation_doc=None):
        if (path is None) == (preparation_doc is None):
            raise ValueError("Explicit sealed OR preparation-only stream")
        self.path = Path(path) if path is not None else None
        self.doc = verified_approval(self.path) if self.path else copy.deepcopy(preparation_doc)
        check_approval(self.doc)
        self.bound = acq.sha256(self.path) if self.path else "PREPARATION_ONLY_NO_TRAINING"
        self.dataset = inp.ApprovedTeacherDataset(inp.verified_approval(Path(self.doc["origin_approval"])), "kim_melband")
        self.seed, self.config = self.dataset.seed, self.dataset.config
        self.true = m.LockedTruePool(m.bulk.OLD_LOCK, self.config)
        self.source_state = origin_state(self.doc)
        sampler = self.source_state["sampler"]
        if (sampler["seed"] != self.seed or sampler["true_lock_sha256"] != self.true.bound or
            sampler["approval_sha256"] != acq.sha256(self.doc["source_approval"]) or
            sampler["teacher"] != "kim_melband"):
            raise ValueError("Original input/seed/teacher lock changed")
        self.cursor, self.last_metadata = START, None

    def state_dict(self):
        return {"approval_sha256": self.bound, "true_lock_sha256": self.true.bound,
                "seed": self.seed, "cursor": self.cursor, "teacher": "kim_melband"}

    def load_state_dict(self, state):
        if (self.path is None or state != self.state_dict() | {"cursor": state.get("cursor")} or
            type(state.get("cursor")) is not int or not START <= state["cursor"] <= LIMIT or
            acq.sha256(self.path) != self.bound):
            raise ValueError("Changed boundary stream/cursor")
        self.cursor, self.last_metadata = state["cursor"], None

    def next_batch(self):
        if self.cursor >= LIMIT:
            raise ValueError("New fixed500 sampler exhausted")
        # Original 177 algorithm; only independent cursor bounds differ.
        xs, vs, metadata = [], [], []
        for domain in m.DOMAINS[:3]:
            item = self.true.crop(domain, self.seed, self.cursor)
            xs.append(item["x"]); vs.append(item["v"]); metadata.append(item["meta"])
        for index in range(3):
            recipe = m.data.crop_recipe(self.dataset.rows, self.config, self.seed, self.cursor*3+index)
            item = self.dataset.crop(recipe)
            xs.append(item["x"]); vs.append(item["v"])
            metadata.append(item["meta"] | {"domain": "pseudo", "role": "pseudo_label_train_candidate"})
        t.validate_metadata(metadata)
        x, v = torch.stack(xs), torch.stack(vs)
        row = {"x": x, "targets": {arm: v.clone() for arm in ARMS}, "domains": m.DOMAINS,
               "cursor": self.cursor, "metadata": metadata}
        self.cursor += 1
        self.last_metadata = copy.deepcopy(metadata)
        return row

def actual_module_files():
    seen, files, queue = set(), set(), [q, reference]
    while queue:
        mod = queue.pop()
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        filename = getattr(mod, "__file__", None)
        if not filename or not Path(filename).resolve().is_relative_to(ROOT / "scripts"):
            continue
        files.add(Path(filename).resolve())
        queue.extend(value for value in vars(mod).values() if isinstance(value, types.ModuleType))
    return files

def hardware_identity(state, source_protocol):
    rng = d.portable(m.capture_rng("cpu"))
    try:
        nets = [m.frozen_factory(source_protocol)() for _ in range(3)]
        saved = state["arms"][SOURCE_ARM]
        for net in nets:
            net.load_state_dict(saved["model"], strict=True)
        def layout(net):
            return {"modules": [(name, type(mod).__module__+"."+type(mod).__name__, repr(mod))
                                for name, mod in net.named_modules()],
                "parameters": [(name, list(v.shape), str(v.dtype)) for name, v in net.named_parameters()],
                "buffers": [(name, list(v.shape), str(v.dtype)) for name, v in net.named_buffers()]}
        identity = layout(nets[0])
        if not all(exact(layout(net), identity) for net in nets[1:]) or list(dict(nets[0].named_parameters())) != saved["parameter_names"]:
            raise ValueError("Graph/parameter layout differs; independent Z7020 audit required")
        gs = torch.from_numpy(m.core.t09.make_synthesis_matrix())
        zeros = {str(kill): torch.nonzero(gs[:, kill:].eq(0).all(dim=1)).flatten().tolist() for kill in (44, 32)}
        if zeros != {"44": list(range(6)), "32": list(range(4))}:
            raise ValueError("Unexpected matrix/LF boundary geometry")
        program = json.loads((HARDWARE / "program.json").read_text(encoding="utf-8"))
        analysis = json.loads((HARDWARE / "analysis.json").read_text(encoding="utf-8"))
        tile = json.loads((HARDWARE / "tile_analysis.json").read_text(encoding="utf-8"))
        return {"graph_layout": identity, "graph_layout_sha256": acq.content_digest(identity),
            "source_and_both_arms_layout_equal": True, "source_parameter_order_equal": True,
            "synthesis_shape": list(gs.shape), "forced_zero_fft_bins": zeros,
            "reference_program": program, "reference_analysis": analysis, "reference_tile_analysis": tile,
            "npu_program_schedule_budget_identity": "Same unchanged reference artifacts for source/LF44/LF32",
            "added_model_mac": 0, "added_model_dsp": 0, "added_tensor_dma_bytes": 0,
            "added_activation_or_weight_bram": 0, "reference_pl_ramb36": 61, "reference_pl_dsp": 72,
            "new_board_implementation_verified": False, "board_changed": False}
    finally:
        m.restore_rng(rng, "cpu")

def draft_document():
    if any(not Path(__file__).with_name(name).is_file() for name in REQUIRED):
        raise ValueError("New tools/tests must be complete before sealing")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    check_protocol(protocol)
    check_unit_gate()
    authority, _, bindings = checked_source()
    files = actual_module_files() | {Path(__file__), PROTOCOL,
        ROOT / "reports/85_mel_lr_scale_trial_plan.md", ROOT / "reports/84_train_lf_cross_boundary_result.md", ROOT / "reports/86_mel_lr_scale_authorization.md"}
    files |= {Path(__file__).with_name(name) for name in REQUIRED}
    files |= {HARDWARE / name for name in ("program.json", "analysis.json", "tile_analysis.json", "tile_schedule.json",
        "dma_plan.json", "tensor_desc.bin", "operator_desc.bin", "tile_commands.bin", "tile_operator_desc.bin")}
    files |= {ROOT / "hardware/spec/22_operator_instruction_contract.md", ROOT / "hardware/spec/30_microarchitecture_budget.md",
              ROOT / "hardware/reports/README.md", ROOT / "results/mel_lr_scale_monitor_20261004/unit_tests.log", UNIT_GATE}
    bindings |= {str(path.resolve()): acq.sha256(path) for path in files}
    doc = {"schema": 1, "purpose": PURPOSE, "protocol": protocol, "source_protocol": authority["source_protocol"],
        "source_approval": str(SOURCE_APPROVAL), "source_binding": reference.binding(SOURCE_APPROVAL),
        "origin_approval": authority["origin_approval"], "origin_checkpoint": str(SOURCE / "NONRELEASE_lf_boundary_step_4000.pt"),
        "source_arm": SOURCE_ARM, "arm_roles": ROLES, "arm_lambdas": LAMBDAS, "arm_instrumental_weights": WEIGHTS,
        "arm_accompaniment_weights": CA_WEIGHTS, "arm_kill_bands": KILLS, "arm_lr_scales": LR_SCALES, "teacher": "kim_melband",
        "bindings_sha256": bindings, "deployment_authorized": False, "release_selection": "NONE"}
    check_approval(doc)
    origin_state(doc)
    return doc

def check_unit_gate():
    gate = acq.read_sealed(UNIT_GATE)
    log = Path(gate["log"]).read_text(encoding="utf-8-sig")
    if (type(gate["actual_exit_code"]) is not int or gate["actual_exit_code"] != 0 or
        gate["tool_sha256"] != acq.sha256(__file__) or gate["trainer_sha256"] != acq.sha256(Path(__file__).with_name(REQUIRED[0])) or
        gate["test_sha256"] != acq.sha256(Path(__file__).with_name(REQUIRED[-1])) or gate["draft_reviewed"] is not True or
        gate["tests_passed"] < 30 or acq.sha256(gate["log"]) != gate["log_sha256"] or
        not re.search(r"Ran "+str(gate["tests_passed"])+r" tests? in",log) or not re.search(r"(?m)^OK\s*$",log)):
        raise ValueError("All new native unit tests exit0 and draft review BEFORE real audit/prepare")
    return gate

def verify_audit():
    doc = draft_document()
    plan = acq.read_sealed(AUDIT_OUT / "plan.json")
    result = acq.read_sealed(AUDIT_OUT / "audit.json")
    if (not exact(plan["bindings_sha256"], doc["bindings_sha256"]) or result["plan_sha256"] != acq.sha256(AUDIT_OUT / "plan.json") or
        result["counter"] != START or result["model_updates"] != 0 or result["cuda_used"] is not False or
        result["optimizer_constructed"] is not False or not all(result[name] is True for name in
        ("control_bit_exact","candidate_gradient_bit_exact","control_wave_bit_exact","model_and_modes_unchanged","input_rng_unchanged")) or
        result["candidate_gradient_l2"] <= 0 or result["auxiliary_active_count"] < 1 or
        result["hardware"]["source_and_both_arms_layout_equal"] is not True or
        result["source_model_digest"] != old.dev.state_digest(origin_state(doc)["arms"][SOURCE_ARM]["model"])):
        raise ValueError("Incomplete real TRAIN4000 LR-only gradient / hardware audit")
    reference.validate_metadata(result["metadata"])
    if result["input_sha256"] != [meta["input_pcm_sha256"] for meta in result["metadata"]]:
        raise ValueError("Actual input PCM mismatch")
    print("LR_SCALE TRAIN_AUDIT VERIFIED; CPU updates0; actual Adam branch proof still required", flush=True)
    return result

def audit(out):
    if out != AUDIT_OUT:
        raise ValueError("Fixed predeclared audit directory")
    old.require_fresh(out); no_active_trainer()
    doc = draft_document()
    out.mkdir(parents=True)
    acq.write_new_json(out / "plan.json", acq.seal({"schema": 1, "purpose": PURPOSE, "counter": START,
        "model_updates": 0, "cuda_used": False, "bindings_sha256": doc["bindings_sha256"]}))
    fresh = load("audit_lf_trainer", REQUIRED[0])
    torch.set_num_threads(2)
    with d.deterministic_runtime("cpu"):
        rng = d.portable(m.capture_rng("cpu"))
        stream = BoundaryStream(preparation_doc=doc)
        batch = stream.next_batch()
        x, v = batch["x"], batch["targets"][ARMS[0]]
        wa, gs = torch.from_numpy(m.core.t09.make_analysis_matrix()), torch.from_numpy(m.core.t09.make_synthesis_matrix())
        saved = stream.source_state["arms"][SOURCE_ARM]
        nets = []
        for _ in range(3):
            net = m.frozen_factory(doc["source_protocol"])()
            net.load_state_dict(saved["model"], strict=True)
            for module, mode in zip(net.modules(), saved["modes"]):
                module.training = mode
            nets.append(net)
        def gradients(net):
            return torch.cat([parameter.grad.detach().flatten() for parameter in net.parameters()])
        source = reference.boundary_backward(nets[0], x, v, wa, gs, "cpu", batch["metadata"], 32)
        control = fresh.boundary_backward(nets[1], x, v, wa, gs, "cpu", batch["metadata"], 32)
        candidate = fresh.boundary_backward(nets[2], x, v, wa, gs, "cpu", batch["metadata"], 32)
        if control != source or candidate != control or not torch.equal(gradients(nets[0]), gradients(nets[1])) or not torch.equal(gradients(nets[1]),gradients(nets[2])):
            raise ValueError("LF44 control loss/gradient not bit-exact old178 full control")
        with torch.no_grad():
            spectrum = m.core.stft_batch(x)
            masks = (nets[0](torch.einsum("fk,bcft->bckt", wa, spectrum.abs()))[:, :2]+1)/2
            old_wave = m.core.product_vocal(spectrum, masks, gs, x.shape[-1], 32)
            pv44 = m.fit.reconstruction_loss(masks, spectrum, x, v, gs, 96, 32)[2]
            pv32 = m.fit.reconstruction_loss(masks, spectrum, x, v, gs, 96, 32)[2]
            if not torch.equal(old_wave, pv44):
                raise ValueError("LF44 old scoring waveform differs")
        unchanged = all(m.equal_state(net.state_dict(), saved["model"]) and
            [module.training for module in net.modules()] == saved["modes"] for net in nets)
        hardware = hardware_identity(stream.source_state, doc["source_protocol"])
        m.restore_rng(rng, "cpu")
        result = {"schema": 1, "purpose": PURPOSE, "counter": START, "plan_sha256": acq.sha256(out / "plan.json"),
            "input_sha256": [m.pilot.wave_digest(wave) for wave in x], "metadata": batch["metadata"],
            "target_sha256": [m.pilot.wave_digest(wave) for wave in v], "source_arm": SOURCE_ARM,
            "source_model_digest": old.dev.state_digest(saved["model"]), "model_and_modes_unchanged": unchanged,
            "control_bit_exact": True, "candidate_gradient_bit_exact": True, "control_wave_bit_exact": True, "input_rng_unchanged": m.equal_state(rng, m.capture_rng("cpu")),
            "lr_does_not_change_raw_gradient": True,
            "candidate_gradient_l2": float(gradients(nets[2]).norm()),
            "same_kill_wave_delta_max": float((pv32-pv44).abs().max()), "auxiliary_active_count": candidate["auxiliary_active_count"],
            "control_losses": control, "candidate_losses": candidate, "hardware": hardware,
            "model_updates": 0, "optimizer_constructed": False, "cuda_used": False,
            "scope": "One fixed next TRAIN4000 batch; zero-update raw gradient audit, Adam LR difference deferred to bounded mechanism",
            "release_selection": "NONE", "deployment": False}
        if not unchanged or not d.finite_state(result) or torch.cuda.is_initialized():
            raise ValueError("Nonfinite/mutating/CUDA audit")
        acq.write_new_json(out / "audit.json", acq.seal(result))
    verify_audit()

def prepare(out):
    old.require_fresh(out); no_active_trainer()
    doc = draft_document(); verify_audit()
    for path in (AUDIT_OUT / "plan.json", AUDIT_OUT / "audit.json"):
        doc["bindings_sha256"][str(path.resolve())] = acq.sha256(path)
    out.mkdir(parents=True)
    acq.write_new_json(out / "approval.json", acq.seal(doc))
    verified_approval(out / "approval.json")
    print("LF_BOUNDARY IMPORT SEALED; real CPU/CUDA proofs still required", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "verify_audit", "prepare", "verify"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "audit": audit(args.out or AUDIT_OUT)
    elif args.command == "verify_audit": verify_audit()
    elif args.command == "prepare": prepare(args.out or DEFAULT_OUT)
    else:
        verified_approval((args.out or DEFAULT_OUT) / "approval.json")
        print("LF_BOUNDARY IMPORT VERIFIED; NONRELEASE", flush=True)
