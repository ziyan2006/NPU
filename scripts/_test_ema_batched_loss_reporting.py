"""New CPU stub-path equivalence tests; no student/autograd/optimizer."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema233_new_units", ROOT / "scripts/233_ema_batched_loss_reporting.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def original_stub():
    trace = []; cursor = [0]
    def stft(x): trace.append(("stft", cursor[0])); return torch.zeros(1, 2, 1, 1)
    class Net:
        def modules(self): return []
        def __call__(self, x): trace.append(("stub_forward", cursor[0])); return torch.ones(1, 4, 1, 1)
    def reconstruction(*args):
        trace.append(("reconstruction", cursor[0]))
        return torch.tensor((cursor[0]+1)/64), {"wave_l1": torch.tensor(.125), "complex_l1": torch.tensor(.25)}, torch.zeros(1, 2, 89856)
    def auxiliary(*args):
        trace.append(("auxiliary", cursor[0])); return torch.tensor(.0625), {"active": False}
    def combine(*args):
        trace.append(("combine", cursor[0]))
        def backward(): trace.append(("stub_backward", cursor[0])); cursor[0] += 1
        return SimpleNamespace(backward=backward), {}
    ns = dict(torch=torch, validate_metadata=lambda meta: None,
              d=SimpleNamespace(finite_state=lambda pair: True),
              m=SimpleNamespace(core=SimpleNamespace(stft_batch=stft), fit=SimpleNamespace(reconstruction_loss=reconstruction,
                                 suite=SimpleNamespace(scoring_slice=lambda *args: slice(0, 8)))),
              k=SimpleNamespace(source_projection_component_loss=auxiliary), w=SimpleNamespace(combine_slot_loss=combine))
    node = next(n for n in ast.parse(f.SOURCE.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "boundary_backward")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(f.SOURCE), "exec"), ns)
    return ns["boundary_backward"], Net(), trace


class ReportingTests(unittest.TestCase):
    def test_full_AST_reverse_and_compilation(self):
        self.assertIsNotNone(compile(f.reporting_ast(), "new233_unit", "exec"))

    def test_six_stub_slots_whole_results_and_order_identical(self):
        x = torch.zeros(6, 2, 89856); meta = [{"slot": i} for i in range(6)]
        old, net, trace = original_stub()
        result = old(net, x, x, torch.ones(1, 1), None, "cpu", meta, 32)
        original_trace = list(trace)
        old2, net2, trace2 = original_stub(); fast = f.make_reporting_function(old2)
        actual = fast(net2, x, x, torch.ones(1, 1), None, "cpu", meta, 32)
        self.assertTrue(f.f.baseline_functions().equal(result, actual))
        self.assertEqual(original_trace, trace2)
        self.assertEqual(sum(name == "stub_backward" for name, _ in trace2), 6)

    def test_private_globals_no_original_patch(self):
        old, _, _ = original_stub(); fast = f.make_reporting_function(old)
        self.assertNotIn("read_scalars", old.__globals__)
        self.assertIs(fast.__globals__["read_scalars"], f.f.read_scalars)

    def test_wrong_source_callable_refused(self):
        self.assertRaises(ValueError, f.make_reporting_function, lambda: None)

    def test_changed_source_refused(self):
        with patch.object(f.f, "sha", return_value="0"*64): self.assertRaises(ValueError, f.reporting_ast)

    def test_invalid_input_refused_before_stub_forward(self):
        old, net, trace = original_stub(); fast = f.make_reporting_function(old)
        self.assertRaises(ValueError, fast, net, torch.zeros(1), torch.zeros(1), None, None, "cpu", [], 32)
        self.assertEqual(trace, [])


if __name__ == "__main__":
    torch.set_num_threads(4)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA in CPU stub tests")):
        unittest.main(verbosity=2)
