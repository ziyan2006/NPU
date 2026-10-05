"""New private integration/guard tests; no old units, model or updates."""
import ast
import __future__
import importlib.util
import inspect
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parent.parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

f = load("ema235_new_unit_factory", "235_ema_private_fast_storage_core.py")
t = load("ema235_new_unit_original", "219_ema_continuous_training_core.py")


class PrivateIntegrationTests(unittest.TestCase):
    def test_entire_private_hierarchy_and_helper_bindings(self):
        new = f.make_private_core(t)
        self.assertIs(new.ContinuousTrajectory.__bases__[0], new.e.SingleTrajectoryEngine)
        self.assertIs(new.e.SingleTrajectoryEngine.__bases__[0], new.e.j.LiveCpuStateOwner)
        self.assertIs(new.e.j.LiveCpuStateOwner.__bases__[0], new.c.CpuStateOwner)
        self.assertIs(new.e.c, new.c); self.assertIs(new.e.j.c, new.c)
        self.assertIs(new.e.portable, new.c.portable)
        self.assertIs(new.e.j.portable, new.c.portable)
        self.assertIsNot(new.c.portable, t.c.portable)
        self.assertNotIn("read_scalars", new.__dict__)

    def test_all_guard_transaction_and_update_function_code_unchanged(self):
        new = f.make_private_core(t)
        pairs = ((t, new), (t.e, new.e), (t.e.j, new.e.j), (t.c, new.c))
        changed_helpers = {"portable", "typed_tree", "digest", "equal", "seal", "check_seal"}
        checked = 0
        for old, private in pairs:
            syntax = ast.parse(Path(old.__file__).read_text(encoding="utf-8"))
            # Original whole-module compilation recognizes imported module
            # names (Python3.13 LOAD_ATTR/PUSH_NULL call hints). Definition-only
            # compilation has no imports, so compare with an independent exact
            # original-AST compilation in the same definition-only setting.
            reference = types.ModuleType("ema235_independent_original_AST")
            reference.__dict__.update(vars(old))
            definitions = ast.Module(body=[n for n in syntax.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
            exec(compile(definitions, old.__file__, "exec", flags=__future__.annotations.compiler_flag,
                         dont_inherit=True), reference.__dict__)
            functions = [(n.name, getattr(reference, n.name), getattr(private, n.name)) for n in syntax.body
                         if isinstance(n, ast.FunctionDef) and n.name not in changed_helpers]
            for cls in (n for n in syntax.body if isinstance(n, ast.ClassDef)):
                for method in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                    functions.append((cls.name + "." + method.name,
                                      getattr(getattr(reference, cls.name), method.name),
                                      getattr(getattr(private, cls.name), method.name)))
            for name, a, b in functions:
                if isinstance(a, property): a, b = a.fget, b.fget
                a, b = inspect.unwrap(a), inspect.unwrap(b)
                self.assertEqual(a.__code__.co_code, b.__code__.co_code, name)
                self.assertEqual(a.__code__.co_names, b.__code__.co_names, name)
                self.assertEqual(a.__code__.co_varnames, b.__code__.co_varnames, name)
                self.assertEqual(a.__code__.co_argcount, b.__code__.co_argcount, name)
                checked += 1
        self.assertGreater(checked, 70)

    def test_original_all_global_objects_not_reassigned(self):
        originals = [dict(vars(module)) for module in (t, t.e, t.e.j, t.c)]
        f.make_private_core(t)
        for before, module in zip(originals, (t, t.e, t.e.j, t.c)):
            self.assertEqual(set(before), set(vars(module)))
            self.assertTrue(all(value is vars(module)[name] for name, value in before.items()))

    def test_wrong_entry_module_rejected(self):
        wrong = types.ModuleType("wrong")
        wrong.__file__ = str(ROOT / "scripts/232_ema_fast_state_commitment.py")
        self.assertRaises(ValueError, f.make_private_core, wrong)

    def test_changed_original_pin_rejected(self):
        with patch.dict(f.PINS, {"219_ema_continuous_training_core.py": "0"*64}):
            self.assertRaises(ValueError, f.make_private_core, t)


if __name__ == "__main__":
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA in static private tests")), \
         patch.object(torch, "load", side_effect=AssertionError("No PT in private definition tests")):
        unittest.main(verbosity=2)
