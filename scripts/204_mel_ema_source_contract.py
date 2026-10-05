"""Pinned LR4500 CPU source reader. NOT an EMA trainer or resume container.

No live Module, optimizer, input sampler, audio, or CUDA state is constructed.
The old stopped-at4500 state stays intact; this reader grants no new updates.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("htdemucs_waveform_control", "kim_melband_waveform_candidate")
RAW_ARM = ARMS[0]
SOURCE_SHA = "b3a0450ecd9bba259981bd47f8a49c0fee1a997d7a087a72570f589cc14ed5b3"
SOURCE_REL = "results/mel_lr_scale_20261004/NONRELEASE_lr_scale_step_4500.pt"
PURPOSE = "NONRELEASE_MEL_LR_SCALE_FORK"
MAPS = {
    "arm_roles": dict(zip(ARMS, ("lr1_control", "lr_half_candidate"))),
    "arm_lambdas": dict.fromkeys(ARMS, .2),
    "arm_instrumental_weights": dict.fromkeys(ARMS, 4),
    "arm_accompaniment_weights": dict.fromkeys(ARMS, 1),
    "arm_kill_bands": dict.fromkeys(ARMS, 32),
    "arm_lr_scales": dict(zip(ARMS, (1.0, .5))),
}
PINS = {
    SOURCE_REL: SOURCE_SHA,
    "results/mel_lr_scale_import_20261004/approval.json": "fe9d1e930a84430922ef8cfb23446895a5d3313b0bea46112ac974a4e2956e1b",
    "results/mel_lr_scale_20261004/checkpoint_4500.json": "829f65b6d101c5b298f374ea1864dbae850f9e1323c77920a4ad1eff037a60e6",
    "results/mel_lr_scale_20261004/completion.json": "def6529b76af98d6b2e037b29c65f7a6c7b31010b3a9e230fc101d21a1eeead7",
    "results/mel_lr_scale_20261004/development_step_4500.json": "06cec2f938b53c854c3e971e7dcef636c3a47e5103046cfa2cd96442c0b046c6",
    "results/mel_lr_scale_monitor_20261004/completion_review.json": "817ff28fa4c25cf0319a326899fd2ac062fbd95688b2ccce56647f1c78e2a715",
    "results/mel_lr_scale_launch_20261004/detached_launch_20261003_193445_5324751.json": "f638d7613590782808f694cfb5acca50a2dc1987726d62a5c43048598fb5e164",
    "results/mel_lr_scale_launch_20261004/detached_exit_20261003_193445_5324751.json": "ae305542f4c5e7287ddde867b3ccafde8b7907b685ac694c645777233c47ded3",
    "scripts/193_prepare_mel_lr_scale.py": "1c358793ce93d8ea7f54fa019610582f1a6427ac3666bef08de1a4c0fc537f89",
    "scripts/194_train_mel_lr_scale.py": "56455bab403f5ffbab941980dde4cfc2d9fe88934d35f7dd5e841f34b5279f48",
    "scripts/170_instrumental_protection_loss.py": "bd2fdeeae3cd511056d2aa179feb3ba67e1b91619412926bf52bda8203137e82",
    "scripts/176_accompaniment_component_loss.py": "5036d4d23fe1c96d45e330b6d3e2bcf84220955d1f1ffda651c1c084c047db1b",
    "reports/102_mel_ema_shadow_trial_plan.md": "39256e70d7e8ce4675102711c3b4595b0b81f315410bc3602df5677c16f784c8",
    "reports/103_mel_ema_shadow_component_progress.md": "c34f5fbb4ed18dbdaad1e11e288f8c5e97254c30feaf37ea5d57f70751bf30d5",
    "scripts/203_ema_shadow_state.py": "7fb5fd1b3c5f66f3e9a2ebf0b020997c3454c8cf334e90de03282ebe78b939ae",
}
LAYOUT = OrderedDict((name, tuple(shape)) for name, shape in (
    ("enc0.conv.weight", (32, 2, 3, 3)), ("enc0.conv.bias", (32,)),
    ("enc1.conv.weight", (64, 32, 3, 3)), ("enc1.conv.bias", (64,)),
    ("enc2.conv.weight", (96, 64, 3, 3)), ("enc2.conv.bias", (96,)),
    ("enc3.conv.weight", (128, 96, 3, 3)), ("enc3.conv.bias", (128,)),
    ("bott.conv.weight", (128, 128, 1, 3)), ("bott.conv.bias", (128,)),
    ("bott_blocks.0.conv.conv.weight", (128, 128, 3, 3)),
    ("bott_blocks.0.conv.conv.bias", (128,)),
    ("bott_blocks.1.conv.conv.weight", (128, 128, 3, 3)),
    ("bott_blocks.1.conv.conv.bias", (128,)),
    ("dec3.conv.weight", (96, 224, 3, 3)), ("dec3.conv.bias", (96,)),
    ("dec2.conv.weight", (64, 160, 3, 3)), ("dec2.conv.bias", (64,)),
    ("dec1.conv.weight", (32, 96, 1, 3)), ("dec1.conv.bias", (32,)),
    ("out.conv.weight", (4, 32, 1, 1)), ("out.conv.bias", (4,)),
))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def typed_equal(a, b):
    """JSON/Python type sensitive; dictionary order is not metadata identity."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and tensor_bytes(a) == tensor_bytes(b)
    if isinstance(a, float):
        return math.isfinite(a) and math.isfinite(b) and a.hex() == b.hex()
    return a == b


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_digest(doc):
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def check_seal(doc):
    require(type(doc) is dict and type(doc.get("content_sha256")) is str,
            "Missing own document seal")
    require(content_digest(doc) == doc["content_sha256"], "Changed own document seal")


def compare_sealed_documents(file_doc, embedded_doc):
    check_seal(file_doc)
    check_seal(embedded_doc)
    # Neither side is stripped or coerced. Full typed symmetric comparison.
    require(typed_equal(file_doc, embedded_doc), "Full parent document differs")


def tensor_bytes(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


def state_digest(state):
    """Original146/119 tensor digest definition, without importing a trainer."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        part = hashlib.sha256(str((array.shape, str(array.dtype))).encode() + array.tobytes())
        digest.update(name.encode())
        digest.update(part.hexdigest().encode())
    return digest.hexdigest()


def tensor_leaves(value):
    if type(value) is torch.Tensor:
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensor_leaves(child)


def storage_ids(value):
    return [v.untyped_storage().data_ptr() for v in tensor_leaves(value) if v.numel()]


def check_tensor(value, dtype, shape, label):
    require(type(value) is torch.Tensor and value.device.type == "cpu" and
            value.dtype == dtype and tuple(value.shape) == tuple(shape) and
            value.layout == torch.strided and value.is_contiguous() and
            not value.requires_grad and value.grad is None, "Tensor identity: " + label)
    require(bool(torch.isfinite(value).all()), "Nonfinite tensor: " + label)


def check_rng(rng):
    require(type(rng) is dict and set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"},
            "Complete source CPU/CUDA RNG required")
    py = rng["python"]
    require(type(py) is tuple and len(py) == 3 and type(py[0]) is int and py[0] == 3,
            "Python RNG version/type")
    require(type(py[1]) is tuple and len(py[1]) == 625 and all(type(v) is int for v in py[1])
            and all(0 <= v < 2**32 for v in py[1][:-1]) and 0 <= py[1][-1] <= 624
            and py[2] is None, "Python RNG state")
    nr = rng["numpy"]
    require(type(nr) is list and len(nr) == 5 and nr[0] == "MT19937" and
            type(nr[2]) is int and 0 <= nr[2] <= 624 and type(nr[3]) is int and
            nr[3] in (0, 1) and type(nr[4]) is float and math.isfinite(nr[4]), "NumPy RNG metadata")
    check_tensor(nr[1], torch.int64, (624,), "numpy RNG")
    require(bool(((nr[1] >= 0) & (nr[1] < 2**32)).all()), "NumPy RNG words")
    cpu = rng["torch_cpu"]
    require(type(cpu) is torch.Tensor and cpu.ndim == 1 and cpu.numel() == 5056,
            "CPU RNG shape")
    check_tensor(cpu, torch.uint8, (5056,), "CPU RNG")
    require(type(rng["torch_cuda"]) is list and len(rng["torch_cuda"]) == 1,
            "Source saved one CUDA device RNG; do not erase or migrate here")
    check_tensor(rng["torch_cuda"][0], torch.uint8, (16,), "Saved CUDA RNG")


def check_arm(arm, lr):
    require(type(arm) is dict and set(arm) == {"model", "optimizer", "parameter_names", "modes", "updates"},
            "Complete raw arm fields")
    require(typed_equal(arm["parameter_names"], list(LAYOUT)), "Original parameter name order")
    require(type(arm["model"]) is dict and list(arm["model"]) == list(LAYOUT),
            "Original ordered model state")
    require(typed_equal(arm["modes"], [True] * 27), "All original module modes")
    require(type(arm["updates"]) is int and arm["updates"] == 4500, "Source update exposure")
    optimizer = arm["optimizer"]
    require(type(optimizer) is dict and set(optimizer) == {"state", "param_groups"}, "Adam fields")
    expected = {"lr": lr, "betas": (.9, .999), "eps": 1e-8, "weight_decay": 0,
                "amsgrad": False, "maximize": False, "foreach": False, "capturable": False,
                "differentiable": False, "fused": False, "decoupled_weight_decay": False,
                "params": list(range(len(LAYOUT)))}
    require(typed_equal(optimizer["param_groups"], [expected]), "Complete typed Adam group/options/ID order")
    require(type(optimizer["state"]) is dict and list(optimizer["state"]) == list(range(len(LAYOUT)))
            and all(type(key) is int for key in optimizer["state"]), "Adam exact ID mapping")
    for index, (name, shape) in enumerate(LAYOUT.items()):
        check_tensor(arm["model"][name], torch.float32, shape, name)
        moments = optimizer["state"][index]
        require(type(moments) is dict and set(moments) == {"step", "exp_avg", "exp_avg_sq"},
                "Complete Adam moments")
        check_tensor(moments["step"], torch.float32, (), name + "/step")
        require(float(moments["step"]) == 4500.0, "Every Adam parameter step must match source")
        check_tensor(moments["exp_avg"], torch.float32, shape, name + "/m")
        check_tensor(moments["exp_avg_sq"], torch.float32, shape, name + "/v")
        require(bool((moments["exp_avg_sq"] >= 0).all()), "Negative Adam second moment")


def validate_source_state(state, documents):
    """Structural check only; file authenticity is enforced by load_fixed_source."""
    require(type(state) is dict, "Portable source dictionary required")
    expected_keys = {"schema", "purpose", "limit", "binding", "runtime", "step", "sampler", "schedule",
                     "rng", "arms", "deployment_authorized", "teacher", "smoke", "origin_sha256",
                     "source_stopped_at", "source_legacy_stop_events", "tranche_stopping", "legacy_stop_events", *MAPS}
    require(set(state) == expected_keys, "Complete source parent fields")
    for key, value in {"schema": 1, "purpose": PURPOSE, "step": 4500, "limit": 4500,
                       "deployment_authorized": False, "smoke": False, "teacher": "kim_melband",
                       "source_stopped_at": 4000, "source_legacy_stop_events": [3750, 4000],
                       "legacy_stop_events": [4250, 4500],
                       "tranche_stopping": "new_common_fixed500_budget_preserve_legacy_stop_evidence",
                       "origin_sha256": "94323f8a30fb5e5decb65d4c110fd5252d0916de68507f42af555b3f539a8f6d", **MAPS}.items():
        require(typed_equal(state[key], value), "Source field/type changed: " + key)
    approval, receipt, completion, dev = (documents[k] for k in ("approval", "receipt", "completion", "development"))
    for document in (approval, receipt, completion, dev):
        check_seal(document)
    compare_sealed_documents(receipt, completion["final_checkpoint"])
    require(typed_equal(state["binding"], receipt["binding"]) and
            typed_equal(state["binding"], completion["binding"]), "Full source code/approval binding")
    expected_binding = {"approval_sha256": PINS["results/mel_lr_scale_import_20261004/approval.json"],
                        "trainer_sha256": PINS["scripts/194_train_mel_lr_scale.py"],
                        "importer_sha256": PINS["scripts/193_prepare_mel_lr_scale.py"],
                        "protection_kernel_sha256": PINS["scripts/170_instrumental_protection_loss.py"],
                        "auxiliary_kernel_sha256": PINS["scripts/176_accompaniment_component_loss.py"],
                        **MAPS, "teacher": "kim_melband"}
    require(typed_equal(state["binding"], expected_binding), "Pinned source binding, not borrowed training approval")
    for key, value in MAPS.items():
        require(all(typed_equal(doc[key], value) for doc in (approval, receipt, completion, dev)),
                "Source document arm identity: " + key)
    require(typed_equal(receipt["sha256"], SOURCE_SHA) and receipt["checkpoint"] == Path(SOURCE_REL).name
            and typed_equal(receipt["step"], 4500) and typed_equal(receipt["additional_step"], 500), "Source PT receipt")
    require(typed_equal(completion["step"], 4500) and typed_equal(completion["limit"], 4500)
            and typed_equal(completion["additional_steps"], 500), "Full closed500 budget")
    require(completion["release_selection"] == "NONE" and completion["deployment"] is False and
            completion["independent_acceptance_ready"] is False, "Source NONRELEASE status")
    for key in ("source_stopped_at", "source_legacy_stop_events", "tranche_stopping", "legacy_stop_events"):
        require(typed_equal(state[key], completion[key]) and typed_equal(state[key], dev[key]), "Closed stop evidence")
    require(typed_equal(state["runtime"], completion["runtime"]), "Original full runtime identity")
    runtime = state["runtime"]
    for key, value in {"device": "cuda", "deterministic": True, "warn_only": False,
                       "cudnn_benchmark": False, "cudnn_deterministic": True,
                       "matmul_tf32": False, "cudnn_tf32": False, "workspace": ":4096:8"}.items():
        require(typed_equal(runtime[key], value), "Strict source FP32 runtime")
    expected_sampler = {"approval_sha256": state["binding"]["approval_sha256"],
                        "true_lock_sha256": "36a9be8ade7822969a7fed661c8f226f49b89e8ece90e1e6192ae0ae1b4b2c38",
                        "seed": 20261002, "cursor": 4500, "teacher": "kim_melband"}
    require(typed_equal(state["sampler"], expected_sampler), "Original sampler cursor/seed/lock")
    config = approval["source_protocol"]["paired_comparison_planned"]
    schedule = {"step": 4500, "last_validation": 4500, "stopped_at": 4500,
                "best": dict.fromkeys(ARMS), "stale": dict.fromkeys(ARMS, 18),
                "patience_anchor": dict.fromkeys(ARMS), "config": config}
    require(typed_equal(state["schedule"], schedule), "Actual cumulative source schedule (stale18, not16)")
    require(type(state["arms"]) is dict and tuple(state["arms"]) == ARMS, "Fixed source arms/order")
    progress = (4500 - config["warmup_steps"]) / (config["maximum_steps"] - config["warmup_steps"])
    lr = config["cosine_min_learning_rate"] + (config["learning_rate"] - config["cosine_min_learning_rate"]) * (1 + math.cos(math.pi * progress)) / 2
    for arm in ARMS:
        check_arm(state["arms"][arm], lr * MAPS["arm_lr_scales"][arm])
        require(state_digest(state["arms"][arm]["model"]) == dev["model_state_sha256"][arm],
                "Actual model does not match original DEV digest")
    check_rng(state["rng"])
    pointers = storage_ids(state)
    require(len(pointers) == len(set(pointers)), "Source tensors share storage")


@contextmanager
def zero_execution_guard():
    def forbidden(*args, **kwargs):
        raise RuntimeError("Source reader forbids live Module/autograd/Adam/CUDA execution")
    with ExitStack() as stack:
        for owner, name in ((torch.nn.Module, "__init__"), (torch.nn.Module, "_call_impl"),
                            (torch.autograd, "grad"), (torch.autograd, "backward"),
                            (torch.Tensor, "backward"), (torch.optim.Adam, "__init__"),
                            (torch.optim.Adam, "step"), (torch.cuda, "_lazy_init")):
            stack.enter_context(patch.object(owner, name, forbidden))
        yield


def make_source_packet(state, documents):
    validate_source_state(state, documents)
    packet = {
        "schema": 1, "purpose": "NONRELEASE_EMA_SOURCE_ONLY_NOT_TRAINING_CONTAINER",
        "source_sha256": SOURCE_SHA, "selected_arm": RAW_ARM,
        "raw_arm": copy.deepcopy(state["arms"][RAW_ARM]),
        "parent_metadata": copy.deepcopy({k: v for k, v in state.items() if k != "arms"}),
        "new_training_authorized": False, "complete_resume_container": False,
    }
    require(typed_equal(packet["raw_arm"], state["arms"][RAW_ARM]), "Raw packet copy changed source bits/types")
    require(typed_equal(packet["parent_metadata"], {k: v for k, v in state.items() if k != "arms"}),
            "Parent metadata/RNG/legacy copy changed")
    require(not (set(storage_ids(packet)) & set(storage_ids(state))), "Packet aliases source storage")
    require(len(storage_ids(packet)) == len(set(storage_ids(packet))), "Packet internal alias")
    return packet


def load_fixed_source(root=ROOT):
    """Read and validate once; return (isolated source packet, evidence summary).

    All historical approval bindings remain in the pinned parent file. This
    source-interface check is not a new formal audit of those hundreds of files.
    """
    root = Path(root).resolve()
    require(not torch.cuda.is_initialized(), "Fresh CPU-only reader required")
    cpu_rng = torch.get_rng_state().clone()
    py_rng, np_rng = random.getstate(), copy.deepcopy(np.random.get_state())
    with zero_execution_guard(), torch.no_grad():
        for relative, expected in PINS.items():
            require(sha256(root / relative) == expected, "Changed pinned source file: " + relative)
        def read(relative):
            return json.loads((root / relative).read_text(encoding="utf-8-sig"))
        documents = {
            "approval": read("results/mel_lr_scale_import_20261004/approval.json"),
            "receipt": read("results/mel_lr_scale_20261004/checkpoint_4500.json"),
            "completion": read("results/mel_lr_scale_20261004/completion.json"),
            "development": read("results/mel_lr_scale_20261004/development_step_4500.json"),
        }
        review = read("results/mel_lr_scale_monitor_20261004/completion_review.json")
        launch_rel = "results/mel_lr_scale_launch_20261004/detached_launch_20261003_193445_5324751.json"
        exit_rel = "results/mel_lr_scale_launch_20261004/detached_exit_20261003_193445_5324751.json"
        launch, exit_doc = read(launch_rel), read(exit_rel)
        require(launch["probe_only"] is False and launch["error"] is None and
                launch["status"] == "worker_started_not_completed", "Actual nonprobe source launch")
        require(type(exit_doc["exit_code"]) is int and exit_doc["exit_code"] == 0 and
                exit_doc["launcher_process_id"] == launch["worker"]["LauncherProcessId"] and
                Path(exit_doc["launch_receipt"]).resolve() == (root / launch_rel).resolve(),
                "Source actual matched launch/exit")
        require(review["formal_round_completed"] is True and
                type(review["actual_training_exit_code"]) is int and review["actual_training_exit_code"] == 0 and
                review["actual_worker_or_shim"] == [] and review["hard_limit_preserved"] == 4500,
                "Original source completion/exit review")
        # Pinned historical PS review embeds normalized numeric JSON; it is NOT
        # substituted for the actual typed receipt/completion/PT documents.
        state = torch.load(root / SOURCE_REL, map_location="cpu", weights_only=True)
        packet = make_source_packet(state, documents)
        for relative, expected in PINS.items():
            require(sha256(root / relative) == expected, "Source changed during read: " + relative)
        require(torch.equal(torch.get_rng_state(), cpu_rng) and random.getstate() == py_rng and
                np.array_equal(np.random.get_state()[1], np_rng[1]) and
                np.random.get_state()[0] == np_rng[0] and np.random.get_state()[2:] == np_rng[2:] and
                not torch.cuda.is_initialized(), "Fresh reader changed RNG/CUDA")
        summary = {"source_sha256": SOURCE_SHA, "selected_arm": RAW_ARM,
                   "step": 4500, "adam_parameter_count": len(LAYOUT), "all_adam_steps": 4500,
                   "group_lr": packet["raw_arm"]["optimizer"]["param_groups"][0]["lr"],
                   "stale": state["schedule"]["stale"][RAW_ARM], "parent_stopped_at": 4500,
                   "model_sha256": state_digest(packet["raw_arm"]["model"]),
                   "cuda_initialized": False, "rng_unchanged": True, "source_only": True,
                   "new_training_authorized": False, "complete_resume_container": False,
                   "file_bindings_sha256": dict(PINS)}
    return packet, summary
