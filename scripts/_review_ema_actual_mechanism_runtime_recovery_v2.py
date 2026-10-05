"""Isolated native-exit field adapter for218's independent reader.

217 journals actual EXIT event sequence/PID and saves its native exit_code in
the sealed per-process table, not in each event row. Preserve failed v1 bytes
and log. Read exact table values; never invent exits or weaken any state gate.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts/_review_ema_actual_mechanism_runtime_recovery.py"
SOURCE_SHA = "ac84193ff086359c4a09cc285216c9f37fb98c13de214c6eb2af9a237aafc1c7"


def bound_reader():
    with SOURCE.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != SOURCE_SHA:
        raise ValueError("Changed failed independent reader; preserve original bytes")
    spec = importlib.util.spec_from_file_location("read218_exact_original", SOURCE)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "review")
    replacement = ast.parse('result["processes"][str(event["pid"])]["exit_code"]', mode="eval").body
    class Adapter(ast.NodeTransformer):
        count = 0
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "event" and isinstance(node.slice, ast.Constant) and node.slice.value == "exit_code":
                self.count += 1
                return copy.deepcopy(replacement)
            return self.generic_visit(node)
    adapter = Adapter(); new = adapter.visit(copy.deepcopy(function))
    if adapter.count != 2:
        raise ValueError("Exactly two absent event.exit_code reads; no other reader changes")
    namespace = dict(old.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[new], type_ignores=[])), str(SOURCE)+"[v2-per-process-exit-interface]", "exec"), namespace)
    return namespace["review"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", required=True, type=Path)
    bound_reader()(parser.parse_args().out)
