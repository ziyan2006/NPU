"""Isolated217 runtime-order recovery; original evidence remains unchanged.

common.py sets20 threads at import. Set CPU2/CUDA4 AFTER original loss imports,
then require exact source CUDA runtime before model construction/update.
One explicit input+target device round-trip verifies complete PCM bits; graph,
original194 backward, Adam, EMA, transaction and limits are unchanged.
Reuses SHA-bound217 algorithms, never runs its closed units or old attempts.
"""
from __future__ import annotations
import argparse
import ast
import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = "scripts/217_ema_actual_update_mechanism.py"
SOURCE_SHA = "468734d23c353c805a87fa10f3c6d134a14bf789adabb2659fb63237e4bc71fd"
PURPOSE = "NONRELEASE_EMA218_POST_IMPORT_RUNTIME_BOUND_ACTUAL_MECHANISM"
spec = importlib.util.spec_from_file_location("ema218_bound217_algorithms", ROOT / SOURCE)
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
require, sha = q.require, q.sha
require(sha(ROOT / SOURCE) == SOURCE_SHA, "Changed217 executed source")


def recovery_tree(tree):
    original = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"]
    require(len(original) == 1, "Exactly217 worker")
    node = copy.deepcopy(original[0])
    tries = [n for n in node.body if isinstance(n, ast.Try)]
    require(len(tries) == 1, "Exact worker try/failure boundary")
    body = tries[0].body
    matches = [i for i, n in enumerate(body) if isinstance(n, ast.Assign)
               and ast.dump(n, include_attributes=False) == ast.dump(ast.parse('loss = e.OriginalBoundaryLoss(device)').body[0], include_attributes=False)]
    require(len(matches) == 1, "Exactly one original loss construction")
    inserted = ast.parse('''
torch.set_num_threads(4 if device == "cuda" else 2)
require(e.runtime_identity(device)["threads"] == (4 if device == "cuda" else 2), "Post-import explicit runtime threads")
if device == "cuda":
    require(c.equal(e.runtime_identity("cuda"), parent["parent_metadata"]["runtime"]), "POST-IMPORT complete strict source CUDA runtime")
allowed()
device_pcm = {field: proof["input_packet"][field].to(device) for field in ("x", "v")}
require(all(c.equal(c.portable(device_pcm[field]), proof["input_packet"][field]) for field in device_pcm), "Complete cross-device input AND target PCM bits differ")
del device_pcm
allowed()
''').body
    index = matches[0] + 1
    body[index:index] = inserted
    # Removing just our inserted nodes must reproduce the entire original AST.
    restored = copy.deepcopy(node)
    restored_try = next(n for n in restored.body if isinstance(n, ast.Try))
    del restored_try.body[index:index+len(inserted)]
    require(ast.dump(restored, include_attributes=False) == ast.dump(original[0], include_attributes=False),
            "Recovery must not alter graph/Adam/transaction/reference/fault/arithmetic")
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), len(inserted)


def build():
    require(sha(ROOT / SOURCE) == SOURCE_SHA, "Changed217 source before exact AST selection")
    original = ast.parse((ROOT / SOURCE).read_text(encoding="utf-8"))
    worker_tree, count = recovery_tree(original)
    collectors = [copy.deepcopy(n) for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "collect"]
    require(len(collectors) == 1, "Exactly217 whole native supervisor")
    # Original exclusive error marker remains evidence; don't replace a marker
    # already written by external review. No other supervisor algorithm edits.
    new_marker_nodes = 0
    for node in ast.walk(collectors[0]):
        for field, children in ast.iter_fields(node):
            if type(children) is list:
                for i, child in enumerate(list(children)):
                    if (isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
                            and ast.unparse(child.value).startswith("obs.write_new(out / 'supervisor_refused.json',")):
                        children[i] = ast.If(test=ast.parse("not (out / 'supervisor_refused.json').exists()", mode="eval").body,
                                             body=[child], orelse=[])
                        new_marker_nodes += 1
    require(new_marker_nodes == 1, "One exclusive marker preservation repair")
    namespace = dict(q.__dict__)
    namespace.update(__file__=__file__, PURPOSE=PURPOSE, PINS=q.PINS | {SOURCE: SOURCE_SHA})
    exec(compile(worker_tree, str(ROOT / SOURCE) + "[218runtime-recovery]", "exec"), namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=collectors, type_ignores=[])), str(ROOT / SOURCE) + "[218supervisor]", "exec"), namespace)
    return namespace["worker"], namespace["collect"], count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("collect", "worker"))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    worker, collect, _ = build()
    (collect if args.operation == "collect" else worker)(args.out, args.device)
