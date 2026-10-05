"""Non-launching integration recipe for two measured EMA hot-path changes.

Returns a SHA-bound worker AST, not a callable trainer or training authority.
The completed 4501..5000 run must never be replayed to time this candidate.
A future, separately bounded worker can use this recipe after its own actual
mechanism/activation checks. No source, optimizer or precision change here.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "scripts/221_ema_supervised_training.py"
PINS = {
    WORKER: "f1eb83d4abe087a182c761aa6e2e736a07bac0084d71822b2a70816ad6f3cc65",
    ROOT / "scripts/219_ema_continuous_training_core.py": "cfd19e0b4059164621c190eda4f74819e0c8b73b358396b94e8f1bbb7f3619ca",
    ROOT / "scripts/228_ema_fast_hotpath.py": "fb1f674052e0a2dd2601d99f9f89e950b17f98d8876712c722beab7d6c066cce",
    ROOT / "scripts/226_ema_decoder_hotpath.py": "142f81175dc9b2e22b8128eb5ef719afa412a290eac75108f48099cd3e0a11e7",
}
REPLACEMENTS = {
    "q.g.DirectDecoder(q.g.read_manifest())": "hot.make_decoder(q.g)",
    "a.AuthenticatedAudioStream._cache_identity": "hot.fast_cache_identity(a)",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_pins():
    for path, expected in PINS.items():
        require(sha(path) == expected, "Changed candidate dependency: " + str(path))


def dump(node):
    return ast.dump(node, include_attributes=False)


def worker_ast():
    """Only the decoder constructor and cache traversal binding may differ.

    `hot` is the private module228 binding in a future worker namespace, never
    a patch to a sealed original module. Returning syntax confers no authority.
    """
    check_pins()
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    workers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"]
    require(len(workers) == 1, "Exact single original worker")
    original = workers[0]
    candidate = copy.deepcopy(original)
    expected = {key: dump(ast.parse(key, mode="eval").body) for key in REPLACEMENTS}
    counts = {key: 0 for key in REPLACEMENTS}

    class Replace(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                key = ("q.g.DirectDecoder(q.g.read_manifest())" if target == "decoder" else
                       "a.AuthenticatedAudioStream._cache_identity" if target == "_cache_identity" else None)
                if key is not None:
                    require(dump(node.value) == expected[key], "Changed original hot-path assignment: " + target)
                    node.value = ast.copy_location(ast.parse(REPLACEMENTS[key], mode="eval").body, node.value)
                    counts[key] += 1
                    return node
            return self.generic_visit(node)

    candidate = Replace().visit(candidate)
    require(all(value == 1 for value in counts.values()), "Exactly two hot-path assignments required")
    # Independently reverse ONLY the declared substitutions and require the
    # entire worker syntax to equal the original, including every loss/update,
    # RNG, transaction, DEV, counter, supervision and activation statement.
    restored = copy.deepcopy(candidate)
    reverse_counts = {key: 0 for key in REPLACEMENTS}

    class Reverse(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                for key, value in REPLACEMENTS.items():
                    if target == ("decoder" if key.startswith("q.g.") else "_cache_identity"):
                        require(dump(node.value) == dump(ast.parse(value, mode="eval").body), "Undeclared substitution")
                        node.value = ast.copy_location(ast.parse(key, mode="eval").body, node.value)
                        reverse_counts[key] += 1
                        return node
            return self.generic_visit(node)

    restored = Reverse().visit(restored)
    require(dump(restored) == dump(original) and all(v == 1 for v in reverse_counts.values()),
            "Worker contains a change beyond the two hot paths")
    return ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[]))


def audit():
    syntax = worker_ast()
    compile(syntax, "ema230[non-launching-worker-recipe]", "exec")  # No exec.
    return {
        "purpose": "NONRELEASE_EMA230_TWO_HOTPATH_INTEGRATION_RECIPE",
        "source_bindings": {str(path): digest for path, digest in PINS.items()},
        "replacement_expressions": REPLACEMENTS.copy(),
        "candidate_worker_AST_sha256": hashlib.sha256(dump(syntax).encode()).hexdigest(),
        "exactly_two_changes": True, "original_worker_reconstructed_exactly": True,
        "compiled_not_executed": True, "training_authorized": False,
        "completed_fixed500_replay_authorized": False, "actual_CPU_CUDA_mechanism": "PENDING",
        "end_to_end_training_speedup": "UNMEASURED", "release_selection": "NONE",
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=True))
