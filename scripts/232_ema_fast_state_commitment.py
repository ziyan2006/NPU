"""Allocation-efficient state commitments and scalar reporting, not a trainer.

Every tensor byte is hashed on every call; no version/mtime/identity digest
memoization. Full ordered typed205 seals and two transaction snapshots remain.
Blocking, owning CPU copies retain the original noalias/RNG/rollback contract.
"""
from __future__ import annotations
import ast
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts/205_ema_cpu_state_transaction.py"
SOURCE_SHA = "8a8f2da566b664abfb19e6c24fe5fb4bf78bd61c13408ebf385b5cf425cd90f5"
CHUNK = 262144


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def baseline_functions():
    """Only compile the exact six pure205 helpers; no205 imports/owner/PT."""
    require(sha(SOURCE) == SOURCE_SHA, "Changed sealed state helper source")
    names = ("portable", "typed_tree", "digest", "equal", "seal", "check_seal")
    nodes = [n for n in ast.parse(SOURCE.read_text(encoding="utf-8")).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    require(tuple(n.name for n in nodes) == names, "Exact pure helper order")
    namespace = dict(torch=torch, OrderedDict=OrderedDict, require=require, math=math,
                     hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


def tensor_commitment(value):
    require(type(value) is torch.Tensor and value.device.type == "cpu"
            and not value.requires_grad and value.grad_fn is None
            and value.layout == torch.strided, "Finite detached dense CPU state")
    # Same contiguous/reshape/byte-view as205, including dtype/endian/zeros.
    flat = value.contiguous().reshape(-1)
    raw = memoryview(flat.view(torch.uint8).numpy()).cast("B")
    try:
        array = flat.numpy()
    except (TypeError, RuntimeError):
        array = None  # e.g. bfloat16: preserve exact Torch finite semantics.
    for start in range(0, flat.numel(), CHUNK):
        end = min(start + CHUNK, flat.numel())
        finite = bool(np.isfinite(array[start:end]).all()) if array is not None else bool(torch.isfinite(flat[start:end]).all())
        require(finite, "Nonfinite typed state")
    return ["tensor", str(value.dtype), list(value.shape), hashlib.sha256(raw).hexdigest()]


def optimized_functions():
    """Keep every non-tensor recursive schema branch exactly as205's AST."""
    base = baseline_functions()
    nodes = [copy.deepcopy(n) for n in ast.parse(SOURCE.read_text(encoding="utf-8")).body
             if isinstance(n, ast.FunctionDef) and n.name in
             ("portable", "typed_tree", "digest", "equal", "seal", "check_seal")]
    portable = next(n for n in nodes if n.name == "portable")
    old_return = portable.body[0].body[-1]
    require(ast.unparse(old_return) == "return value.detach().cpu().clone()", "Exact portable tensor copy")
    portable.body[0].body[-1] = ast.copy_location(ast.parse(
        "return value.detach().to(device='cpu', copy=True, non_blocking=False)").body[0], old_return)
    tree = next(n for n in nodes if n.name == "typed_tree")
    tensor_if = tree.body[1] if isinstance(tree.body[0], ast.Expr) else tree.body[0]
    require(ast.unparse(tensor_if.test) == "type(value) is torch.Tensor", "Exact typed tensor branch")
    tensor_if.body = [ast.copy_location(ast.parse("return tensor_commitment(value)").body[0], tensor_if.body[0])]
    namespace = dict(base.digest.__globals__)
    namespace["tensor_commitment"] = tensor_commitment
    syntax = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(syntax, "new232[exact-metadata-single-copy-state]", "exec"), namespace)
    return types.SimpleNamespace(**{n.name: namespace[n.name] for n in nodes})


def read_scalars(values):
    """Batch pure log values by dtype/device; keep individual FP values/order.

    Not for loss branches or finite safety decisions. Source buffers may not
    be concurrently mutated during this blocking read; nothing is cached.
    """
    values = list(values)
    groups = OrderedDict()
    for index, value in enumerate(values):
        require(type(value) is torch.Tensor and value.numel() == 1 and value.layout == torch.strided
                and value.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64),
                "Real floating scalar log values only")
        groups.setdefault((value.device, value.dtype), []).append((index, value.detach().reshape(())))
    result = [None] * len(values)
    for rows in groups.values():
        packed = torch.stack([value for _, value in rows]).to("cpu", non_blocking=False)
        for (index, _), number in zip(rows, packed.tolist()):
            result[index] = float(number)
    return result


if __name__ == "__main__":
    raise SystemExit("State/performance helper only; no actual training/launch authority")
