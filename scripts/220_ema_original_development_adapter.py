"""Original119 policy / original194 LF32 evaluation of raw and EMA copies.

Pure source-SHA-bound AST reuse, no closed paired trainer/evaluate_pair CLI.
Original31-song/177-view corpus is kept; frozen44 scores are read from their
existing archive rather than recomputing a historical baseline. No optimizer,
teacher inference, acceptance, promotion, or audio-output operation exists.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SUITE = "e5990cdd2faf71e83469cbef7f94dea6b80d0b376ba49d5b81bca69fe442fb07"
PURPOSE = "NONRELEASE_EMA_RAW_SHADOW_OLD_DEVELOPMENT"
PINS = {
    "scripts/194_train_mel_lr_scale.py": "56455bab403f5ffbab941980dde4cfc2d9fe88934d35f7dd5e841f34b5279f48",
    "scripts/146_evaluate_paired_development.py": "4b2886c2fd01af2fccfe69d3e1da43cab1228c18d57047b6bac2c2ad05ce74f7",
    "scripts/119_model_selection_suite.py": "e57977037cf430a023b59587b1fde6b1f849ab01f43d1c269928f3c31f7d400e",
    "results/mel_lr_scale_20261004/frozen_scores.json": "710c69265dbcef531b1a1f4f37f6226a6f853c9bba84ac36ef8fb58a34afcacf",
    "results/mel_lr_scale_20261004/selection_suite.json": "54acd248571d7aa00137b28b1b8e615772437f278eeea52ccbc11c103946a9d2",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_files():
    for relative, expected in PINS.items():
        require(sha(ROOT / relative) == expected, "Original DEV archive/source changed: " + relative)


def original_ast(relative, names):
    """Exact whole selected definitions, decorators and bodies unmodified."""
    require(relative in PINS and sha(ROOT / relative) == PINS[relative], "SHA before original definition selection")
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
    selected = [copy.deepcopy(node) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    require([node.name for node in selected] == list(names), "Exact ordered original definitions")
    return ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))


def evaluation_functions(original_m):
    import torch
    suite = original_m.fit.suite
    require(sha(suite.__file__) == PINS["scripts/119_model_selection_suite.py"], "Original loaded policy source")
    tree = original_ast("scripts/194_train_mel_lr_scale.py", ("low_frequency_backing_metrics", "boundary_evaluation"))
    namespace = {"torch": torch, "math": math, "m": original_m, "old": SimpleNamespace(dev=SimpleNamespace(suite=suite))}
    exec(compile(tree, "original194[220unchanged-LF32-DEV]", "exec"), namespace)
    return namespace["boundary_evaluation"]


def corpus_types(original_m):
    tree = original_ast("scripts/146_evaluate_paired_development.py", ("LockedDevelopmentCorpus", "checked_suite"))
    namespace = {"m": original_m, "suite": original_m.fit.suite, "core": original_m.core,
                 "acq": original_m.pilot.acq, "OrderedDict": OrderedDict, "Path": Path,
                 "EXPECTED_SUITE": EXPECTED_SUITE, "torch": original_m.torch}
    exec(compile(tree, "original146[220unchanged-DEV-corpus]", "exec"), namespace)
    return namespace["LockedDevelopmentCorpus"], namespace["checked_suite"]


def development_source_bindings(lock):
    """All exact DEVELOPMENT source files; train/final sources excluded."""
    bindings, tracks = {}, []
    for row in lock["records"]:
        if row["role"] != "development":
            continue
        require(row["domain"] in ("musdb", "mir1k", "instrumental"), "Exact old DEV domain")
        tracks.append((row["domain"], row["track_id"]))
        for name in sorted({p for field in ("mix_files", "vocal_files", "stem_files") for p in row.get(field, [])}):
            bindings[name] = lock["files"][name]["sha256"]
    require(len(tracks) == len(set(tracks)) == 31 and bindings, "Whole unchanged31 old DEV songs")
    return bindings


class OriginalDevelopmentEvaluator:
    def __init__(self, original_m, rows, manifest, baseline, allowed):
        check_files()
        self.m, self.suite, self.c = original_m, original_m.fit.suite, None
        #212 is a storage/RNG library, not a historical test or trainer run.
        spec = importlib.util.spec_from_file_location("ema220_storage_rng", ROOT / "scripts/212_ema_single_trajectory_engine.py")
        self.e = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.e)
        self.c, self.torch = self.e.c, self.e.torch
        self.rows, self.manifest, self.baseline, self.allowed = rows, copy.deepcopy(manifest), copy.deepcopy(baseline), allowed
        self.function = evaluation_functions(original_m)
        self.wa = self.torch.from_numpy(original_m.core.t09.make_analysis_matrix())
        self.gs = self.torch.from_numpy(original_m.core.t09.make_synthesis_matrix())
        self.policy = copy.deepcopy(self.suite.POLICY)
        self.binding = {"source_sha256": dict(PINS), "suite_sha256": EXPECTED_SUITE,
                        "policy_sha256": original_m.pilot.acq.content_digest(self.policy),
                        "kill": 32, "frozen_kill": 44, "warmup": 96, "tracks": 31, "views": 177}
        self._binding_digest = self.c.digest(self.binding)
        self._baseline_digest = self.c.digest(self.baseline)
        self.exposure = {"forward_started": 0, "forward_completed": 0, "pair_started": 0, "pair_completed": 0}
        self._check()
        require(self.baseline["summary"] == self.suite.aggregate(self.baseline["rows"]), "Whole original frozen44 aggregation")

    @classmethod
    def prepare(cls, original_m, teacher_ids, source_lease, allowed):
        """Decode original DEV once under actual selected-source/native guards.

        Caller must scope213's decoder on THIS original_m.core and bulk. The
        whole-worker native-image supervisor remains mandatory. No baseline
        model, training optimizer, RNG migration or source PT is constructed.
        """
        check_files(); allowed()
        Corpus, _ = corpus_types(original_m)
        corpus = Corpus(original_m.bulk.OLD_LOCK, teacher_ids)
        bindings = development_source_bindings(corpus.lock)
        with source_lease(bindings) as held:
            rows = original_m.fit.suite.build_suite(corpus, 89856, 96)
            manifest = original_m.fit.suite.suite_manifest(rows, 89856, 96)
            historical = original_m.pilot.acq.read_sealed(ROOT / "results/mel_lr_scale_20261004/selection_suite.json")
            candidate = original_m.pilot.acq.seal(copy.deepcopy(manifest))
            # Both sides are complete own-sealed original plain-acq manifests.
            require(original_m.pilot.acq.content_digest(candidate) == candidate["content_sha256"]
                    and original_m.pilot.acq.content_digest(historical) == historical["content_sha256"], "Each manifest own seal")
            require(candidate == historical and manifest["sha256"] == EXPECTED_SUITE
                    and manifest["tracks"] == 31 and manifest["clips"] == 177, "Whole original fixed31/177 suite")
            held.check(); allowed()
        baseline = original_m.pilot.acq.read_sealed(ROOT / "results/mel_lr_scale_20261004/frozen_scores.json")
        return cls(original_m, rows, manifest, baseline, allowed)

    def _check(self):
        check_files(); self.allowed()
        require(self.c.digest(self.binding) == self._binding_digest and self.c.digest(self.baseline) == self._baseline_digest
                and self.suite.POLICY == self.policy, "Original DEV policy/frozen baseline binding changed")
        require(self.suite.suite_manifest(self.rows, 89856, 96) == self.manifest
                and self.manifest["sha256"] == EXPECTED_SUITE, "Unchanged entire DEV descriptors")
        require(all(self.suite.tensor_digest(row["x"], row["v"]) == row["samples_sha256"] for row in self.rows), "DEV input/target PCM changed")
        require(self.torch.get_num_threads() == 4 and self.torch.are_deterministic_algorithms_enabled()
                and not self.torch.is_deterministic_algorithms_warn_only_enabled()
                and not self.torch.backends.cudnn.benchmark and self.torch.backends.cudnn.deterministic
                and not self.torch.backends.cuda.matmul.allow_tf32 and not self.torch.backends.cudnn.allow_tf32,
                "Exact original CPU DEV4 strict FP32 settings after all imports")

    def _score(self, net):
        self._check()
        layout, values, device, modes = self.c.ema._model_layout(net)
        require(device == self.torch.device("cpu") and len(list(net.parameters())) == 22
                and len(list(net.modules())) == 27 and not list(net.buffers()), "Original actual CPU inference model")
        before = self.c.portable(values)
        gradients = OrderedDict((n, None if p.grad is None else self.c.portable(p.grad)) for n, p in net.named_parameters())
        runtime_device = "cuda" if self.torch.cuda.is_initialized() else "cpu"
        rng = self.e.capture_rng(runtime_device)
        require("forward" not in net.__dict__, "No evaluation forward override")
        original_forward = net.forward
        start = dict(self.exposure)
        def counted(instance, *args, **kwargs):
            self.allowed(); self.exposure["forward_started"] += 1
            result = original_forward(*args, **kwargs)
            self.exposure["forward_completed"] += 1; self.allowed()
            return result
        net.forward = MethodType(counted, net)
        try:
            result, low = self.function(net, self.rows, self.wa, self.gs, 32)
            require(self.exposure["forward_started"]-start["forward_started"] == 177
                    and self.exposure["forward_completed"]-start["forward_completed"] == 177, "Complete177 read-only views")
            self.c.typed_tree(result); self.c.typed_tree(low)
            require(self.c.equal(self.c.portable(net.state_dict()), before)
                    and all(self.c.equal(gradients[n], p.grad) for n, p in net.named_parameters()), "Evaluation changed model/gradients")
            self._check()
            descriptors = [{k: v for k, v in row.items() if k != "metrics"} for row in result["rows"]]
            require(descriptors == self.manifest["rows"] and result["summary"] == self.suite.aggregate(result["rows"])
                    and low["summary"] == self.suite.aggregate(low["rows"]), "Whole original DEV coverage/aggregation")
            return result, low
        finally:
            del net.forward
            with self.torch.no_grad():
                for name, value in values.items():
                    value.copy_(before[name])
                for name, parameter in net.named_parameters():
                    parameter.grad = None if gradients[name] is None else gradients[name].clone()
            for module, mode in zip(net.modules(), modes):
                module.training = mode
            self.e.restore_rng(rng, runtime_device)

    def evaluate(self, owner, factory):
        """Two CPU inference copies, ZERO second training optimizer/updates."""
        self._check()
        require(owner._actual and owner.live.validation_due and owner.live.step in (4750, 5000), "Actual committed due stage only")
        before = owner.state_dict()
        self.exposure["pair_started"] += 1
        rng = self.e.capture_rng("cuda")
        try:
            evaluations, low = {}, {}
            trajectory = before["trajectory"]
            for role, source in (("raw", trajectory["raw"]), ("ema", trajectory["shadow"])):
                model = factory()
                model.load_state_dict(self.c.portable(source["tensors"]), strict=True)
                modes = source["modes"] if role == "raw" else source["modes_at_copy"]
                for module, mode in zip(model.modules(), modes):
                    module.training = mode
                evaluations[role], low[role] = self._score(model)
                del model
            scores = {role: self.suite.assess(result["summary"], self.baseline["summary"]) for role, result in evaluations.items()}
            packet = self.c.seal({"purpose": PURPOSE, "step": owner.live.step, "scope": "actual_original31track177view_DEV",
                "evaluator_binding": self.c.portable(self.binding), "suite_sha256": EXPECTED_SUITE,
                "scores": scores, "evaluations": evaluations, "low_frequency_backing": low,
                "model_digest": {"raw": self.c.digest(trajectory["raw"]["tensors"]), "ema": self.c.digest(trajectory["shadow"]["tensors"])},
                "release_selection": "NONE", "independent_acceptance_scored": False,
                "scope_note": "Old true-reference DEV; pseudo is not final truth; no release approval"})
            self.exposure["pair_completed"] += 1
            return packet
        finally:
            self.e.restore_rng(rng, "cuda")
            require(self.c.equal(owner.state_dict(), before), "CPU DEV changed live raw/Adam/EMA/input/schedule/allRNG")


if __name__ == "__main__":
    check_files()
    print(json.dumps({"purpose": PURPOSE, "source_AST_unchanged": True, "actual_DEV_started": False,
                      "training_started": False, "release_selection": "NONE"}))
