"""Original194 math/6backwards, only batch pure scalar log readback per slot.

Not a new loss/optimizer/precision or training authority. No delayed loss
branch, changed division/reduction, fused Adam or enlarged microbatch.
"""
from __future__ import annotations
import ast
import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts/194_train_mel_lr_scale.py"
SOURCE_SHA = "56455bab403f5ffbab941980dde4cfc2d9fe88934d35f7dd5e841f34b5279f48"
spec = importlib.util.spec_from_file_location("ema233_state232", ROOT / "scripts/232_ema_fast_state_commitment.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def dump(value): return ast.dump(value, include_attributes=False)


def reporting_ast():
    f.require(f.sha(SOURCE) == SOURCE_SHA, "Original loss source changed")
    original = next(n for n in ast.parse(SOURCE.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "boundary_backward")
    f.require(not ({"_report_metrics", "_report_numbers"} & {n.id for n in ast.walk(original) if isinstance(n, ast.Name)}),
              "No new local variable collision")
    candidate = copy.deepcopy(original)
    slot_loop = next(n for n in candidate.body if isinstance(n, ast.For) and ast.unparse(n.target) == "(i, meta)")
    hits = [(i, n) for i, n in enumerate(slot_loop.body) if isinstance(n, ast.For) and ast.unparse(n.target) == "(key, value)"]
    f.require(len(hits) == 1, "One exact pure logging loop")
    index, loop = hits[0]
    f.require(isinstance(loop.iter, ast.Call) and isinstance(loop.iter.func, ast.Attribute)
              and loop.iter.func.attr == "items" and isinstance(loop.iter.func.value, ast.Dict)
              and ast.unparse(loop.body[0]) == "total[key] += float(value.detach()) * (1 / 6)", "Exact original scalar accumulation")
    assignments = ast.parse("_report_metrics = {}\n_report_numbers = dict(zip(_report_metrics, read_scalars(_report_metrics.values())))\n"
                            "for key, value in _report_numbers.items():\n    total[key] += value * (1 / 6)").body
    assignments[0].value = copy.deepcopy(loop.iter.func.value)
    slot_loop.body[index:index+1] = assignments
    restored = copy.deepcopy(candidate)
    restore_loop = next(n for n in restored.body if isinstance(n, ast.For) and ast.unparse(n.target) == "(i, meta)")
    f.require(dump(restore_loop.body[index]) == dump(assignments[0])
              and dump(restore_loop.body[index+1]) == dump(assignments[1])
              and dump(restore_loop.body[index+2]) == dump(assignments[2]), "Only declared reporting replacement")
    restore_loop.body[index:index+3] = [copy.deepcopy(loop)]
    f.require(dump(restored) == dump(original), "A numerical/backward/control-flow statement changed")
    return ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[]))


def make_reporting_function(original):
    """Compile in a private namespace; do not patch an original loss module."""
    f.require(original.__name__ == "boundary_backward" and Path(original.__code__.co_filename).resolve() == SOURCE,
              "Exact original compiled loss function required")
    namespace = dict(original.__globals__); namespace["read_scalars"] = f.read_scalars
    exec(compile(reporting_ast(), "new233[original194-batched-logging-only]", "exec"), namespace)
    return namespace["boundary_backward"]


if __name__ == "__main__":
    compile(reporting_ast(), "new233[compile-only]", "exec")
    print("One logging-site replacement; full math/backward AST reconstructed; no training executed")
