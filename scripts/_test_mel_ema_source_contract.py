"""New zero-execution source-interface tests; NOT the closed203 unit suite."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("ema_source204", Path(__file__).with_name("204_mel_ema_source_contract.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def seal(doc):
    doc["content_sha256"] = s.content_digest(doc)
    return doc


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = s.zero_execution_guard()
        cls.guard.__enter__()
        cls.addClassCleanup(cls.guard.__exit__, None, None, None)
        cls.cpu_rng, cls.py_rng = torch.get_rng_state().clone(), random.getstate()
        cls.np_rng = copy.deepcopy(np.random.get_state())
        original_load = torch.load
        cls.deserialize_count = 0

        def capture(*args, **kwargs):
            cls.deserialize_count += 1
            cls.state = original_load(*args, **kwargs)
            return cls.state

        with patch.object(torch, "load", side_effect=capture):
            cls.packet, cls.summary = s.load_fixed_source()
        cls.documents = {
            key: json.loads((s.ROOT / path).read_text(encoding="utf-8-sig"))
            for key, path in {
                "approval": "results/mel_lr_scale_import_20261004/approval.json",
                "receipt": "results/mel_lr_scale_20261004/checkpoint_4500.json",
                "completion": "results/mel_lr_scale_20261004/completion.json",
                "development": "results/mel_lr_scale_20261004/development_step_4500.json",
            }.items()
        }
        cls.source_digest = {
            arm: s.state_digest(cls.state["arms"][arm]["model"]) for arm in s.ARMS
        }

    @classmethod
    def tearDownClass(cls):
        assert cls.deserialize_count == 1
        assert torch.equal(torch.get_rng_state(), cls.cpu_rng)
        assert random.getstate() == cls.py_rng
        nr = np.random.get_state()
        assert nr[0] == cls.np_rng[0] and np.array_equal(nr[1], cls.np_rng[1]) and nr[2:] == cls.np_rng[2:]
        assert not torch.cuda.is_initialized()
        for arm in s.ARMS:
            assert s.state_digest(cls.state["arms"][arm]["model"]) == cls.source_digest[arm]
        for path, expected in s.PINS.items():
            assert s.sha256(s.ROOT / path) == expected
        print("SOURCE_INTERFACE_EVIDENCE " + json.dumps({
            **cls.summary, "checkpoint_cpu_deserializations": cls.deserialize_count,
            "live_student_modules_loaded": 0, "model_forwards": 0,
            "autograd_engine_executions": 0, "adam_constructions": 0,
            "adam_steps": 0, "student_updates": 0, "real_audio_inputs": 0,
            "source_files_unchanged": True,
        }, sort_keys=True))

    def reject_state(self, mutation):
        state = copy.deepcopy(self.state)
        mutation(state)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            s.validate_source_state(state, self.documents)

    def test_actual_source_entry_once(self):
        self.assertEqual(self.deserialize_count, 1)
        self.assertEqual(self.summary["stale"], 18)
        self.assertEqual(self.summary["all_adam_steps"], 4500)
        self.assertEqual(self.summary["group_lr"], 6.281416799501188e-05)

    def test_packet_is_source_only_and_does_not_clear_parent_stop(self):
        self.assertIs(self.packet["new_training_authorized"], False)
        self.assertIs(self.packet["complete_resume_container"], False)
        self.assertEqual(self.packet["selected_arm"], s.ARMS[0])
        self.assertEqual(self.packet["parent_metadata"]["schedule"]["stopped_at"], 4500)
        self.assertEqual(self.packet["parent_metadata"]["limit"], 4500)

    def test_full_raw_packet_bits_types_modes_moments_preserved(self):
        self.assertTrue(s.typed_equal(self.packet["raw_arm"], self.state["arms"][s.RAW_ARM]))
        self.assertTrue(s.typed_equal(self.packet["parent_metadata"], {k: v for k, v in self.state.items() if k != "arms"}))

    def test_packet_no_source_or_internal_alias(self):
        pointers = s.storage_ids(self.packet)
        self.assertEqual(len(pointers), len(set(pointers)))
        self.assertFalse(set(pointers) & set(s.storage_ids(self.state)))

    def test_mutating_packet_copy_cannot_change_source(self):
        packet = copy.deepcopy(self.packet)
        packet["raw_arm"]["optimizer"]["state"][0]["exp_avg"].fill_(0)
        packet["parent_metadata"]["rng"]["torch_cuda"][0].fill_(0)
        packet["parent_metadata"]["schedule"]["stale"][s.RAW_ARM] = 0
        self.assertTrue(s.typed_equal(self.packet["raw_arm"], self.state["arms"][s.RAW_ARM]))
        self.assertEqual(self.state["schedule"]["stale"][s.RAW_ARM], 18)

    def test_typed_metadata_distinguishes_bool_int_float(self):
        self.assertFalse(s.typed_equal(True, 1))
        self.assertFalse(s.typed_equal(1, 1.0))
        self.assertFalse(s.typed_equal((.9, .999), [.9, .999]))
        self.assertTrue(s.typed_equal({"a": 1, "b": 2.0}, {"b": 2.0, "a": 1}))

    def test_tensor_signed_zero_compared_by_bits(self):
        self.assertFalse(s.typed_equal(torch.tensor([0.]), torch.tensor([-0.])))

    def test_full_sealed_parent_both_sides_success(self):
        s.compare_sealed_documents(self.documents["receipt"], self.documents["completion"]["final_checkpoint"])

    def test_original_single_side_seal_regression_is_rejected(self):
        embedded = copy.deepcopy(self.documents["receipt"])
        embedded.pop("content_sha256")
        with self.assertRaisesRegex(ValueError, "Missing own"):
            s.compare_sealed_documents(self.documents["receipt"], embedded)

    def test_sealed_totals_type_coercion_is_rejected(self):
        file_doc, embedded = seal({"totals": {"lr": 1.0}}), seal({"totals": {"lr": 1}})
        with self.assertRaisesRegex(ValueError, "Full parent"):
            s.compare_sealed_documents(file_doc, embedded)

    def test_changed_nested_field_without_new_seal_rejected(self):
        doc = copy.deepcopy(self.documents["receipt"])
        doc["binding"]["arm_lambdas"][s.RAW_ARM] = .3
        with self.assertRaisesRegex(ValueError, "Changed own"):
            s.check_seal(doc)

    def test_changed_pin_rejected_before_deserialize(self):
        original_hash = s.sha256
        def changed(path):
            return "0" * 64 if Path(path).name == Path(s.SOURCE_REL).name else original_hash(path)
        with patch.object(s, "sha256", side_effect=changed), patch.object(torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "Changed pinned"):
                s.load_fixed_source()
            load.assert_not_called()

    def test_guard_blocks_module_before_construction(self):
        with self.assertRaises(RuntimeError):
            torch.nn.Module()

    def test_guard_blocks_autograd_before_engine(self):
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(None, None)

    def test_guard_blocks_adam_before_construction(self):
        with self.assertRaises(RuntimeError):
            torch.optim.Adam([])

    def test_guard_blocks_cuda_before_initialization(self):
        with self.assertRaises(RuntimeError):
            torch.cuda._lazy_init()
        self.assertFalse(torch.cuda.is_initialized())


def field(path, value):
    def mutate(state):
        parent = state
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
    return mutate


arm = ("arms", s.RAW_ARM)
group = arm + ("optimizer", "param_groups", 0)
MUTATIONS = {
    "schema_bool": field(("schema",), True),
    "step_float": field(("step",), 4500.0),
    "limit_raised": field(("limit",), 5000),
    "deployment_enabled": field(("deployment_authorized",), True),
    "smoke_substitution": field(("smoke",), True),
    "wrong_teacher": field(("teacher",), "htdemucs"),
    "wrong_origin": field(("origin_sha256",), "0" * 64),
    "lr_scale_int_coercion": field(("arm_lr_scales", s.RAW_ARM), 1),
    "lambda_changed": field(("arm_lambdas", s.RAW_ARM), .1),
    "kill_float_coercion": field(("arm_kill_bands", s.RAW_ARM), 32.0),
    "inst_bool": field(("arm_instrumental_weights", s.RAW_ARM), True),
    "sampler_cursor_changed": field(("sampler", "cursor"), 4499),
    "sampler_seed_changed": field(("sampler", "seed"), 1),
    "sampler_binding_changed": field(("sampler", "approval_sha256"), "0" * 64),
    "cumulative_stale_reset": field(("schedule", "stale", s.RAW_ARM), 16),
    "cumulative_best_reset": field(("schedule", "best", s.RAW_ARM), 0.0),
    "cumulative_anchor_reset": field(("schedule", "patience_anchor", s.RAW_ARM), 0.0),
    "parent_stop_cleared": field(("schedule", "stopped_at"), None),
    "legacy_event_removed": field(("legacy_stop_events",), [4500]),
    "source_legacy_event_removed": field(("source_legacy_stop_events",), [4000]),
    "runtime_tf32_enabled": field(("runtime", "matmul_tf32"), True),
    "wrong_arm_exposure": field(arm + ("updates",), 4499),
    "mode_int_coercion": field(arm + ("modes",), [1] * 27),
    "mode_false": field(arm + ("modes",), [False] * 27),
    "parameter_order_changed": field(arm + ("parameter_names",), list(reversed(s.LAYOUT))),
    "optimizer_id_order_changed": field(group + ("params",), list(reversed(range(22)))),
    "adam_lr_changed": field(group + ("lr",), .0001),
    "adam_betas_list": field(group + ("betas",), [.9, .999]),
    "adam_beta_changed": field(group + ("betas",), (.8, .999)),
    "adam_eps_changed": field(group + ("eps",), 1e-7),
    "adam_weight_decay_float": field(group + ("weight_decay",), 0.0),
    "adam_foreach_changed": field(group + ("foreach",), True),
    "adam_fused_changed": field(group + ("fused",), True),
    "adam_maximize_changed": field(group + ("maximize",), True),
    "adam_amsgrad_changed": field(group + ("amsgrad",), True),
    "adam_capturable_changed": field(group + ("capturable",), True),
    "adam_differentiable_changed": field(group + ("differentiable",), True),
    "adam_decoupled_changed": field(group + ("decoupled_weight_decay",), True),
    "adam_step_wrong": field(arm + ("optimizer", "state", 0, "step"), torch.tensor(4499.)),
    "adam_step_float64": field(arm + ("optimizer", "state", 0, "step"), torch.tensor(4500., dtype=torch.float64)),
    "cuda_rng_erased": field(("rng", "torch_cuda"), []),
    "cuda_rng_wrong_shape": field(("rng", "torch_cuda"), [torch.zeros(15, dtype=torch.uint8)]),
    "numpy_rng_wrong_dtype": field(("rng", "numpy", 1), torch.zeros(624, dtype=torch.int32)),
    "python_rng_list": field(("rng", "python"), [3, (), None]),
    "missing_parent_field": lambda state: state.pop("source_stopped_at"),
    "extra_parent_field": lambda state: state.update(extra="unsupported"),
    "missing_adam_state": lambda state: state["arms"][s.RAW_ARM]["optimizer"]["state"].pop(21),
    "missing_adam_group_flag": lambda state: state["arms"][s.RAW_ARM]["optimizer"]["param_groups"][0].pop("fused"),
    "negative_adam_variance": lambda state: state["arms"][s.RAW_ARM]["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
    "nonfinite_adam_mean": lambda state: state["arms"][s.RAW_ARM]["optimizer"]["state"][0]["exp_avg"].fill_(float("nan")),
    "nonfinite_model": lambda state: state["arms"][s.RAW_ARM]["model"]["enc0.conv.bias"].fill_(float("inf")),
    "model_bits_changed": lambda state: state["arms"][s.RAW_ARM]["model"]["enc0.conv.bias"].add_(1),
    "gradient_required": lambda state: state["arms"][s.RAW_ARM]["model"]["enc0.conv.bias"].requires_grad_(True),
    "aliased_moments": lambda state: state["arms"][s.RAW_ARM]["optimizer"]["state"][0].update(exp_avg_sq=state["arms"][s.RAW_ARM]["optimizer"]["state"][0]["exp_avg_sq"].abs(), exp_avg=state["arms"][s.RAW_ARM]["model"]["enc0.conv.weight"]),
}
# Source mean alias above preserves valid dimensions; alias is rejected globally.
for name, mutation in MUTATIONS.items():
    def test(self, mutation=mutation):
        self.reject_state(mutation)
    setattr(SourceContractTests, "test_reject_" + name, test)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
