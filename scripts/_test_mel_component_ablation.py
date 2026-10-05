"""Loss-only fork authority, identical target/state, rollback and gradient tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("aux_tests", Path(__file__).with_name("178_train_mel_component_ablation.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
p, m, d, acq, ARMS = t.p, t.m, t.d, t.acq, t.ARMS
POLICY = json.loads(p.PROTOCOL.read_text(encoding="utf-8"))
SOURCE = json.loads(m.data.PROTOCOL.read_text(encoding="utf-8"))
torch.set_num_threads(2)


def factory():
    net = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.Tanh())
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.fill_(.03)
    return net


def metadata():
    return [{"domain": name, "role": "train" if i < 3 else "pseudo_label_train_candidate",
        "vocal_db": -12 if i == 0 else 0, "purpose": t.old.PURPOSE,
        "exploratory_eligible": True, "deployment_eligible": False} for i, name in enumerate(m.DOMAINS)]


def cheap_backward(net, x, v, *args):
    loss = sum(parameter.square().sum() for parameter in net.parameters()) + v.mean()
    loss.backward()
    return {"loss": float(loss.detach())}


def fake_stream(path):
    # Tiny model state machinery test; does not stand in for real-input smoke.
    inp = t.old.inp
    origin = inp.ApprovedPairStream.__new__(inp.ApprovedPairStream)
    origin.path, origin.bound, origin.cursor, origin.seed = path, acq.sha256(path), 0, 20261002
    origin.true = type("TruePool", (), {"bound": "locked"})()
    ids = [f"song_{i}" for i in range(24)]
    origin.doc = {"purpose": t.old.PURPOSE, "schema": 1, "protocol": json.loads(inp.PROTOCOL.read_text(encoding="utf-8")),
        "pair_ids": ids, "snapshots": {"htdemucs": {}, "kim_melband": {}}, "exploratory_training_authorized": True,
        "deployment_authorized": False, "original_htdemucs_exit_code": None,
        "accepted_completion_basis": "independent_full_waveform_verify_for_exploration_only",
        "approved_records": [{"song_id": song, "exploratory_eligible": True, "deployment_eligible": False} for song in ids]}
    def origin_batch():
        x = torch.zeros(6, 2, 89856)
        targets = {arm: x.clone() for arm in ARMS}
        origin.cursor += 1
        return {"x": x, "targets": targets, "metadata": metadata(), "domains": m.DOMAINS, "cursor": origin.cursor-1}
    origin.next_batch = origin_batch
    engine = t.old.ExplorationEngine(factory, SOURCE, {"fixture": "origin"}, "cpu", origin, cheap_backward)
    engine.update_next(origin)
    state = engine.state_dict(origin)
    state["step"] = state["sampler"]["cursor"] = state["schedule"]["step"] = state["schedule"]["last_validation"] = 3000
    for saved in state["arms"].values():
        saved["updates"] = 3000
        saved["optimizer"]["param_groups"][0]["lr"] = m.learning_rate(3000, m.validate_protocol(SOURCE))
        for value in saved["optimizer"]["state"].values():
            value["step"].fill_(3000)
    state["arms"][ARMS[1]]["model"]["0.weight"] += .01
    state["arms"][ARMS[1]]["modes"] = [False]*len(state["arms"][ARMS[1]]["modes"])
    state["schedule"]["stale"] = {ARMS[0]: 1, ARMS[1]: 2}
    state["schedule"]["stopped_at"] = 3000
    state["legacy_stop_events"] = [2750, 3000]
    value = p.ComponentStream.__new__(p.ComponentStream)
    value.path, value.bound, value.cursor, value.seed = path, acq.sha256(path), 3000, origin.seed
    value.true, value.source_state = origin.true, state
    value.doc = {"source_protocol": SOURCE, "protocol": POLICY, "purpose": p.PURPOSE}
    value.last_metadata = None
    def batch():
        cursor = value.cursor
        gen = torch.Generator().manual_seed(cursor)
        x = torch.randn(6, 2, 89856, generator=gen)*.025
        v = torch.randn(6, 2, 89856, generator=gen)*.008
        x = x+v
        v[2].zero_()
        value.cursor += 1
        value.last_metadata = metadata()
        return {"x": x, "targets": {arm: v.clone() for arm in ARMS}, "metadata": copy.deepcopy(value.last_metadata),
            "domains": m.DOMAINS, "cursor": cursor}
    value.next_batch = batch
    return value


class ComponentAblationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "approved.json"
        acq.write_new_json(self.path, {"fixture": "unit only"})
        runtime = d.deterministic_runtime("cpu")
        runtime.__enter__()
        self.addCleanup(runtime.__exit__, None, None, None)
        self.stream = fake_stream(self.path)

    def engine(self):
        with patch.object(p, "check_approval"):
            return t.ForkEngine(factory, self.stream, "cpu", smoke=True)

    def test_protocol_fixed_teacher_budget_and_no_release(self):
        p.check_protocol(POLICY)
        for key, value in (("teacher", "htdemucs"), ("absolute_limit", 4000), ("additional_common_steps", 3000),
                           ("deployment_authorized", True), ("origin_step", 0), ("approved_pseudo_songs", 275)):
            with self.assertRaises(ValueError):
                p.check_protocol(POLICY | {key: value})

    def test_changed_auxiliary_or_roles_rejected(self):
        for replacement in ({"arm_lambdas": p.LAMBDAS | {ARMS[1]: .4}},
                            {"arm_instrumental_weights": p.WEIGHTS | {ARMS[1]: True}},
                            {"auxiliary_domains": ["pseudo"]},
                            {"internal_arm_keys": {}}):
            with self.assertRaises(ValueError):
                p.check_protocol(POLICY | replacement)

    def test_metadata_recipe_role_geometry_rejected(self):
        t.validate_metadata(metadata())
        for slot, key, value in ((0, "vocal_db", True), (0, "vocal_db", -18), (2, "vocal_db", -12),
                                  (1, "role", "development"), (4, "domain", "musdb"), (3, "role", "train")):
            meta = metadata(); meta[slot][key] = value
            with self.assertRaises(ValueError):
                t.validate_metadata(meta)

    def test_initial_model_adam_modes_and_cpu_rng_equal_source(self):
        engine = self.engine()
        state = engine.state_dict(self.stream)
        for arm in ARMS:
            self.assertTrue(m.equal_state(state["arms"][arm], self.stream.source_state["arms"][ARMS[1]]))
        self.assertTrue(m.equal_state(state["rng"], self.stream.source_state["rng"]))
        self.assertEqual(state["sampler"]["cursor"], 3000)
        self.assertEqual(state["teacher"], "kim_melband")
        self.assertEqual(state["arm_roles"], p.ROLES)

    def test_cpu_long_training_or_unapproved_stream_before_optimizer(self):
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("Adam")):
            with self.assertRaises(ValueError):
                t.ForkEngine(factory, self.stream, "cpu", smoke=False)
            with self.assertRaises(ValueError):
                t.ForkEngine(factory, object(), "cuda")

    def test_control_identical_source_aux020_gradient(self):
        engine = self.engine()
        batch = self.stream.next_batch(); x, v = batch["x"], batch["targets"][ARMS[0]]
        a, b = factory(), factory()
        left = t.component_backward(a, x, v, engine.wa, engine.gs, "cpu", batch["metadata"], 1)
        right = p.t.protection_backward(b, x, v, engine.wa, engine.gs, "cpu", batch["metadata"], 4)
        for key, value in right.items():
            self.assertEqual(left[key], value)
        self.assertTrue(all(torch.equal(pa.grad, pb.grad) for pa, pb in zip(a.parameters(), b.parameters())))

    def test_candidate_gradient_matches_explicit_weighted_sum(self):
        engine = self.engine()
        batch = self.stream.next_batch(); x, v = batch["x"], batch["targets"][ARMS[0]]
        a, b = factory(), factory()
        result = t.component_backward(a, x, v, engine.wa, engine.gs, "cpu", batch["metadata"], 0)
        manual_loss = 0.
        for i, meta in enumerate(batch["metadata"]):
            xb, vb = x[i:i+1], v[i:i+1]
            spectrum = m.core.stft_batch(xb)
            output = b(torch.einsum("fk,bcft->bckt", engine.wa, spectrum.abs()))
            base, _, pv = m.fit.reconstruction_loss((output[:, :2]+1)/2, spectrum, xb, vb, engine.gs, 96, 44)
            region = m.fit.suite.scoring_slice(xb.shape[-1], 96)
            auxiliary, _ = t.k.source_projection_component_loss(pv[0, :, region], xb[0, :, region], vb[0, :, region], meta, 0)
            loss = (4 if i == 2 else 1)*base+.2*auxiliary
            (loss*(1/6)).backward()
            manual_loss += float(loss.detach())*(1/6)
        self.assertEqual(result["loss"], manual_loss)
        self.assertEqual(result["auxiliary_active_count"], 2)
        self.assertEqual(result["auxiliary_skip_count"], 4)
        self.assertGreater(result["instrumental_extra_contribution"], 0)
        self.assertTrue(all(torch.equal(pa.grad, pb.grad) for pa, pb in zip(a.parameters(), b.parameters())))

    def test_inactive_true_slots_have_zero_auxiliary(self):
        engine = self.engine()
        batch = self.stream.next_batch()
        batch["targets"][ARMS[0]][:3].zero_()
        result = t.component_backward(factory(), batch["x"], batch["targets"][ARMS[0]], engine.wa, engine.gs,
                            "cpu", batch["metadata"], 0)
        self.assertEqual(result["auxiliary_active_count"], 0)
        self.assertEqual(result["auxiliary_loss"], 0.)
        self.assertEqual(result["auxiliary_skip_count"], 6)

    def test_preparation_only_stream_cannot_construct_optimizer(self):
        self.stream.path = None
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("Adam")):
            with self.assertRaises(ValueError):
                t.ForkEngine(factory, self.stream, "cpu", smoke=True)

    def test_disk_resume_saved_state_no_alias_and_budget(self):
        engine = self.engine(); engine.backward = cheap_backward
        engine.update_next(self.stream)
        receipt = t.save_checkpoint(Path(self.folder.name), engine, self.stream)
        saved = m.load_checked_checkpoint(Path(self.folder.name) / receipt["checkpoint"], receipt["sha256"])
        untouched = d.portable(saved)
        expected_rows = [engine.update_next(self.stream) for _ in range(2)]
        expected = engine.state_dict(self.stream)
        self.stream.cursor = 3000
        other = self.engine(); other.backward = cheap_backward
        other.load_state_dict(saved, self.stream)
        actual_rows = [other.update_next(self.stream) for _ in range(2)]
        self.assertEqual([{k:v for k,v in row.items() if k != "seconds"} for row in expected_rows],
                         [{k:v for k,v in row.items() if k != "seconds"} for row in actual_rows])
        self.assertTrue(m.equal_state(expected, other.state_dict(self.stream)))
        self.assertTrue(m.equal_state(saved, untouched))
        with self.assertRaises(ValueError):
            other.update_next(self.stream)

    def test_second_arm_failure_rolls_back_entire_pair(self):
        engine = self.engine(); before = engine.state_dict(self.stream)
        def fail(net, *args):
            if net is engine.models[ARMS[1]]:
                torch.rand(2); m.random.random(); m.np.random.random()
                raise RuntimeError("second")
            return cheap_backward(net, *args)
        engine.backward = fail
        with self.assertRaises(RuntimeError):
            engine.update_next(self.stream)
        self.assertTrue(m.equal_state(before, engine.state_dict(self.stream)))

    def test_changed_pseudo_target_rejected_and_cursor_rolled_back(self):
        engine = self.engine(); before = engine.state_dict(self.stream)
        original = self.stream.next_batch
        def bad():
            batch = original(); batch["targets"][ARMS[1]][3] += .01
            return batch
        self.stream.next_batch = bad
        with self.assertRaises(ValueError):
            engine.update_next(self.stream)
        self.assertTrue(m.equal_state(before, engine.state_dict(self.stream)))

    def test_changed_teacher_roles_phase_origin_or_cursor_refused(self):
        engine = self.engine(); state = engine.state_dict(self.stream)
        for key, value in (("teacher", "htdemucs"), ("arm_roles", {}), ("smoke", False), ("origin_sha256", "bad"), ("limit", 3500)):
            with self.assertRaises(ValueError):
                engine.load_state_dict(state | {key: value}, self.stream)
        for key, value in (("teacher", "htdemucs"), ("cursor", 2999), ("cursor", 3501), ("cursor", True), ("approval_sha256", "bad")):
            with self.assertRaises(ValueError):
                self.stream.load_state_dict(state["sampler"] | {key: value})

    def test_old_origin_hash_checked_and_changed_stream_refuses(self):
        with self.assertRaises(ValueError):
            m.load_checked_checkpoint(self.path, "0"*64)
        engine = self.engine()
        other = Path(self.folder.name) / "changed.json"; acq.write_new_json(other, {"changed": True})
        self.stream.path = other
        with self.assertRaises(ValueError):
            engine.update_next(self.stream)

    def test_independent_listening_roles_and_common_gain(self):
        spec = importlib.util.spec_from_file_location("aux_review_unit", Path(__file__).with_name("180_review_mel_component_ablation.py"))
        review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
        models = {role: object() for role in ("frozen", "origin_instrumental4_3000", *p.ROLES.values())}
        x, v = torch.ones(2, 100), torch.ones(2, 100)*.2
        with patch.object(review.r, "vocal_wave", return_value=x*2):
            waves, gain = review.listening_waves(models, x, v)
        self.assertEqual(len(waves), 11)
        self.assertEqual(gain, .475)
        self.assertTrue(torch.equal(waves["reference_vocal"], v*gain))
        self.assertEqual(max(float(w.abs().max()) for w in waves.values()), float(torch.tensor(.95)))

    def test_review_rejects_wrong_source_arm_or_unequal_initial_scores(self):
        spec = importlib.util.spec_from_file_location("aux_review_baseline_unit", Path(__file__).with_name("180_review_mel_component_ablation.py"))
        review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
        origin = {"arms": {ARMS[0]: {"fixture": 10}, ARMS[1]: {"fixture": 20}}}
        state = {"arms": {arm: {"fixture": 20} for arm in ARMS}}
        packet = {"evaluations": {arm: {"score": 20} for arm in ARMS}}
        original = {"evaluations": {ARMS[0]: {"score": 10}, ARMS[1]: {"score": 20}}}
        review.check_fork_baseline(state, packet, origin, original)
        for bad_state, bad_packet in (({"arms": {arm: {"fixture": 10} for arm in ARMS}}, packet),
            (state, {"evaluations": {ARMS[0]: {"score": 10}, ARMS[1]: {"score": 11}}}),
            (state, {"evaluations": {arm: {"score": 10} for arm in ARMS}})):
            with self.assertRaises(ValueError):
                review.check_fork_baseline(bad_state, bad_packet, origin, original)

    def test_only_declared_weights_accept_and_different_gradients(self):
        engine = self.engine(); batch = self.stream.next_batch()
        gradients = []
        for weight in (1, 0):
            net = factory()
            row = t.component_backward(net, batch["x"], batch["targets"][ARMS[0]], engine.wa, engine.gs, "cpu", batch["metadata"], weight)
            self.assertGreater(row["auxiliary_contribution"], 0)
            self.assertGreater(row["instrumental_base_loss"], 0)
            gradients.append(torch.cat([parameter.grad.flatten() for parameter in net.parameters()]))
        self.assertGreater(float((gradients[1]-gradients[0]).norm()), 0)
        for weight in (False, True, 1., .2, 2, 10, float("nan")):
            with self.assertRaises(ValueError):
                t.component_backward(factory(), batch["x"], batch["targets"][ARMS[0]], engine.wa, engine.gs, "cpu", batch["metadata"], weight)

    def test_candidate_schedule_and_model_are_the_actual_fork_source(self):
        engine = self.engine(); state = engine.state_dict(self.stream)
        self.assertFalse(m.equal_state(state["arms"][ARMS[0]], self.stream.source_state["arms"][ARMS[0]]))
        self.assertEqual(state["schedule"]["stale"], {arm:2 for arm in ARMS})
        self.assertEqual(state["arm_lambdas"], p.LAMBDAS)
        self.assertEqual(state["source_stopped_at"], 3000)
        self.assertIsNone(state["schedule"]["stopped_at"])
        self.assertEqual(self.stream.source_state["schedule"]["stopped_at"], 3000)

    def test_new_fixed_budget_preserves_legacy_stop_and_resume_provenance(self):
        engine = self.engine()
        engine.limit = 3500
        engine.step = engine.schedule.step = self.stream.cursor = 3250
        def stopped(*args):
            engine.schedule.stopped_at = engine.step
            return {"scores": {}}
        with patch.object(t.old, "observe", side_effect=stopped):
            packet = engine.observe(None, None)
            self.assertIsNone(engine.schedule.stopped_at)
            self.assertEqual(packet["legacy_stop_events"], [3250])
            engine.step = engine.schedule.step = self.stream.cursor = 3500
            engine.observe(None, None)
            self.assertEqual(engine.schedule.stopped_at, 3500)
            self.assertEqual(engine.legacy_stop_events, [3250, 3500])

    def test_old_stop_cannot_be_silently_reintroduced(self):
        engine = self.engine(); state = engine.state_dict(self.stream)
        for changed in (state | {"source_stopped_at": None}, state | {"tranche_stopping": "resume_old"},
                        state | {"legacy_stop_events": [3000]}):
            with self.assertRaises(ValueError):
                engine.load_state_dict(changed, self.stream)
        bad = copy.deepcopy(state); bad["schedule"]["stopped_at"] = 3000
        with self.assertRaises(ValueError):
            engine.load_state_dict(bad, self.stream)

    def test_changed_strength_state_is_refused(self):
        engine = self.engine(); state = engine.state_dict(self.stream)
        with self.assertRaises(ValueError):
            engine.load_state_dict(state | {"arm_lambdas": p.LAMBDAS | {ARMS[1]: .1}}, self.stream)

    def test_new_weight_identity_normalizer_and_stop_history(self):
        engine = self.engine(); state = engine.state_dict(self.stream)
        self.assertEqual(state["source_legacy_stop_events"], [2750, 3000])
        for weights in ({}, p.WEIGHTS | {ARMS[0]: True}, p.WEIGHTS | {ARMS[1]: 2}):
            with self.assertRaises(ValueError):
                engine.load_state_dict(state | {"arm_instrumental_weights": weights}, self.stream)
        with self.assertRaises(ValueError):
            p.check_protocol(POLICY | {"base_normalizer": 9})
        with self.assertRaises(ValueError):
            engine.load_state_dict(state | {"source_legacy_stop_events": []}, self.stream)

    def test_new_metadata_state_has_no_alias(self):
        engine = self.engine(); state = engine.state_dict(self.stream)
        state["source_legacy_stop_events"].append(1)
        state["arm_instrumental_weights"][ARMS[0]] = 99
        other = engine.state_dict(self.stream)
        self.assertEqual(other["source_legacy_stop_events"], [2750, 3000])
        self.assertEqual(other["arm_instrumental_weights"], p.WEIGHTS)

    def test_nonzero_auxiliary_is_required_in_both_branches(self):
        row = {"losses": {arm: {"auxiliary_active_count": 2, "auxiliary_contribution": p.LAMBDAS[arm], "instrumental_weight": p.WEIGHTS[arm], "instrumental_base_loss": .01, "instrumental_weighted_contribution": p.WEIGHTS[arm]*.01, "instrumental_extra_contribution": (p.WEIGHTS[arm]-1)*.01,
            "accompaniment_weight": p.CA_WEIGHTS[arm], "remaining_vocal_component_contribution": .1,
            "accompaniment_component_contribution": .01 if p.CA_WEIGHTS[arm] else 0.}
                          for arm in ARMS}}
        proof = t.auxiliary_evidence([row]*3)
        self.assertEqual(proof[ARMS[0]]["active_count"], 6)
        self.assertEqual(proof[ARMS[1]]["coefficient"], .2)
        for arm in ARMS:
            for key, value in (("auxiliary_active_count", 0), ("auxiliary_contribution", 0.),
                               ("auxiliary_contribution", float("nan"))):
                bad = copy.deepcopy(row); bad["losses"][arm][key] = value
                with self.assertRaises(ValueError):
                    t.auxiliary_evidence([bad]*3)

    def test_cross_device_comparison_requires_same_pcm_and_metadata(self):
        row = {"step": 3001, "input_sha256": ["same"]*6, "metadata": metadata(),
               "arm_roles": p.ROLES, "arm_lambdas": p.LAMBDAS, "arm_instrumental_weights": p.WEIGHTS, "arm_accompaniment_weights": p.CA_WEIGHTS, "losses": {"different": 1}}
        cpu = {"draws": [copy.deepcopy(row)]}
        cuda = copy.deepcopy(cpu); cuda["draws"][0]["losses"] = {"different": 2}
        t.check_cross_device_inputs(cpu, cuda)
        for key, value in (("input_sha256", ["changed"]*6), ("metadata", []), ("step", 3002), ("arm_lambdas", {}), ("arm_instrumental_weights", {})):
            bad = copy.deepcopy(cuda); bad["draws"][0][key] = value
            with self.assertRaises(ValueError):
                t.check_cross_device_inputs(cpu, bad)

    def test_zero_missing_or_wrong_instrumental_branch_refused(self):
        row = {"losses": {arm: {"auxiliary_active_count": 2, "auxiliary_contribution": .2,
            "instrumental_weight": p.WEIGHTS[arm], "instrumental_base_loss": .01,
            "instrumental_weighted_contribution": p.WEIGHTS[arm]*.01,
            "instrumental_extra_contribution": (p.WEIGHTS[arm]-1)*.01,
            "accompaniment_weight": p.CA_WEIGHTS[arm], "remaining_vocal_component_contribution": .1,
            "accompaniment_component_contribution": .01 if p.CA_WEIGHTS[arm] else 0.} for arm in ARMS}}
        for arm in ARMS:
            for key, value in (("instrumental_base_loss", 0.), ("instrumental_weight", True),
                               ("instrumental_weighted_contribution", float("nan")),
                               ("instrumental_extra_contribution", -1.)):
                bad = copy.deepcopy(row); bad["losses"][arm][key] = value
                with self.assertRaises(ValueError):
                    t.auxiliary_evidence([bad]*3)

    def test_launcher_requires_detachment_exit_capture_and_all_mechanisms(self):
        launcher = Path(__file__).with_name("179_start_mel_component_ablation.ps1").read_text(encoding="utf-8")
        for required in ("Windows PowerShell 5.1", "WmiPrvSE.exe", "Wait-PairedProcessExit", "Hold-PairedProcessHandle",
                         "178_train_mel_component_ablation.py", "mel_component_ablation_cpu_20261003",
                         "mel_component_ablation_cuda_20261003", "detached_exit_", "-WindowStyle Hidden"):
            self.assertIn(required, launcher)

    def test_launcher_conflict_patterns_include_new_and_historical_tasks(self):
        launcher = Path(__file__).with_name("179_start_mel_component_ablation.ps1").read_text(encoding="utf-8")
        python_pattern = next(line for line in launcher.splitlines() if "134_generate_teacher_library|" in line)
        powershell_pattern = next(line for line in launcher.splitlines() if "151_start_paired_exploration|" in line)
        for name in ("172_train_mel_instrumental_protection", "175_diagnose_auxiliary_components",
                     "178_train_mel_component_ablation"):
            self.assertIn(name, python_pattern)
        for name in ("173_start_mel_instrumental_protection", "179_start_mel_component_ablation"):
            self.assertIn(name, powershell_pattern)
        self.assertNotIn("172_train_mel_component_ablation", launcher)
        self.assertNotIn("173_start_mel_component_ablation", launcher)


    def test_component_weight_identity_and_bool_refused(self):
        p.require_ca_weights(p.CA_WEIGHTS)
        engine = self.engine(); state = engine.state_dict(self.stream)
        self.assertEqual(state["arm_accompaniment_weights"], p.CA_WEIGHTS)
        for value in ({}, {arm:1 for arm in ARMS}, p.CA_WEIGHTS | {ARMS[0]: True}, p.CA_WEIGHTS | {ARMS[1]: .1}):
            with self.assertRaises(ValueError):
                engine.load_state_dict(state | {"arm_accompaniment_weights": value}, self.stream)
            with self.assertRaises(ValueError):
                p.check_protocol(POLICY | {"arm_accompaniment_weights": value})
        state["arm_accompaniment_weights"][ARMS[0]] = 99
        self.assertEqual(engine.state_dict(self.stream)["arm_accompaniment_weights"], p.CA_WEIGHTS)

    def test_required_removed_branch_cannot_fake_contribution(self):
        row = {"losses": {arm: {"auxiliary_active_count": 2, "auxiliary_contribution": .2,
            "instrumental_weight": 4, "instrumental_base_loss": .01, "instrumental_weighted_contribution": .04,
            "instrumental_extra_contribution": .03, "accompaniment_weight": p.CA_WEIGHTS[arm],
            "remaining_vocal_component_contribution": .1, "accompaniment_component_contribution": .01 if p.CA_WEIGHTS[arm] else 0.}
            for arm in ARMS}}
        t.auxiliary_evidence([row])
        for arm, key, value in ((ARMS[0], "accompaniment_component_contribution", 0.),
                               (ARMS[1], "accompaniment_component_contribution", .01),
                               (ARMS[1], "remaining_vocal_component_contribution", 0.),
                               (ARMS[0], "accompaniment_weight", True)):
            bad = copy.deepcopy(row); bad["losses"][arm][key] = value
            with self.assertRaises(ValueError):
                t.auxiliary_evidence([bad])


if __name__ == "__main__":
    unittest.main()
