"""Private original-engine reconstruction with exact fast state helpers.

No original module/class globals patched. Constructors, transaction, guards,
raw Adam, EMA, loss and rollback bodies are source-bound and unchanged. Only
their private storage/helper bindings differ; no training activation provided.
"""
from __future__ import annotations
import ast
import copy
import importlib.util
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema235_fast232", ROOT / "scripts/232_ema_fast_state_commitment.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
PINS = {
    "205_ema_cpu_state_transaction.py": f.SOURCE_SHA,
    "207_ema_live_cpu_state.py": "7a1ab6a2dbcf9b2af99030784b34bcf6c2a73732227e0abadbb0cf2f88f18d90",
    "212_ema_single_trajectory_engine.py": "8d10c304f3c3c2b4d2d6509dcec719e454b6afdabe651a329d1bc99b4a85d666",
    "219_ema_continuous_training_core.py": "cfd19e0b4059164621c190eda4f74819e0c8b73b358396b94e8f1bbb7f3619ca",
}


def clone_definitions(template, name, overrides):
    path = Path(template.__file__).resolve()
    f.require(path.parent == ROOT / "scripts" and path.name in PINS and f.sha(path) == PINS[path.name],
              "Exact original private core source: " + str(path))
    module = types.ModuleType(name); module.__dict__.update(vars(template)); module.__name__ = name
    module.__dict__.update(overrides)
    nodes = [copy.deepcopy(n) for n in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    # Every original function/class syntax unmodified; imports/data/constants
    # are inherited read-only by reference, functions/classes get new globals.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    module.__dict__.update(overrides)
    return module


def make_private_core(original):
    f.require(Path(original.__file__).resolve() == ROOT / "scripts/219_ema_continuous_training_core.py", "Original core required")
    fast = f.optimized_functions()
    overrides = {key: getattr(fast, key) for key in ("portable", "typed_tree", "digest", "equal", "seal", "check_seal")}
    storage = clone_definitions(original.c, "ema235_private_storage205", overrides)
    joint = clone_definitions(original.e.j, "ema235_private_joint207",
                              {"c": storage, "require": storage.require, "portable": storage.portable, "equal": storage.equal})
    engine = clone_definitions(original.e, "ema235_private_engine212",
                               {"j": joint, "c": storage, "require": storage.require, "portable": storage.portable, "equal": storage.equal})
    core = clone_definitions(original, "ema235_private_continuous219", {"e": engine, "c": storage})
    f.require(issubclass(core.ContinuousTrajectory, engine.SingleTrajectoryEngine)
              and issubclass(engine.SingleTrajectoryEngine, joint.LiveCpuStateOwner)
              and issubclass(joint.LiveCpuStateOwner, storage.CpuStateOwner), "Complete private class hierarchy")
    f.require(original.c.portable is not storage.portable and original.e.portable is not engine.portable,
              "No sealed original module patched")
    return core


if __name__ == "__main__":
    raise SystemExit("Private performance core factory only; no launch/training authority")
