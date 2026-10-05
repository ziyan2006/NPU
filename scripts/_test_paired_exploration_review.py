"""CPU-only review/export contracts; no real training, teachers or CUDA."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import soundfile as sf
import torch

spec = importlib.util.spec_from_file_location("exploration_review_test", Path(__file__).with_name("152_review_paired_exploration.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
torch.set_num_threads(2)


def fixture():
    descriptors = [{"domain": domain, "track": domain.split("/")[0], "start_sample": 123,
                    "samples_sha256": "fixed", "vocal_gain_db": 0} for domain in
                   (*r.dev.suite.VOCAL_DOMAINS, "instrumental/native")]
    manifest = {"rows": descriptors, "sha256": "synthetic", "clips": 177, "tracks": 31}
    def scores(gain):
        rows = [d | {"metrics": {"vocal_error_snr_db": gain, "accompaniment_error_snr_db": 12.,
                 "joint_fit_accompaniment_gain": .99, "residual_vocal_abs_gain": .9 - gain * .1}}
                for d in descriptors]
        return {"rows": rows, "summary": r.dev.suite.aggregate(rows)}
    baseline = scores(0.)
    bound = {"approval_sha256": "approved"}
    state = {"schema": 1, "purpose": r.PURPOSE, "step": 250, "limit": 1000, "binding": bound,
        "deployment_authorized": False, "sampler": {"cursor": 250, "approval_sha256": "approved"},
        "schedule": {"step": 250, "last_validation": 250},
        "arms": {arm: {"updates": 250, "model": {"weight": torch.tensor([float(i)])}}
                 for i, arm in enumerate(r.ARMS)}}
    evaluations = {arm: scores(float(i + 1)) for i, arm in enumerate(r.ARMS)}
    packet = {"step": 250, "suite_sha256": manifest["sha256"], "scope": r.SCOPE,
        "policy_sha256": r.acq.content_digest(r.dev.suite.POLICY),
        "baseline_scores_sha256": r.acq.content_digest(baseline), "evaluations": evaluations,
        "scores": {arm: r.dev.suite.assess(evaluations[arm]["summary"], baseline["summary"]) for arm in r.ARMS},
        "model_state_sha256": {arm: r.dev.state_digest(state["arms"][arm]["model"]) for arm in r.ARMS}}
    receipt = {"step": 250, "purpose": r.PURPOSE, "deployment_authorized": False, "binding": bound,
               "checkpoint": "NONRELEASE_pair_step_0250.pt"}
    return receipt, state, packet, bound, manifest, baseline


def net():
    model = r.core.t09.CausalSpectralUNet(enc=(4, 8, 12, 16), bottleneck_blocks=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.01)
    return model.train()


class ReviewTests(unittest.TestCase):
    def test_original_assessment_recomputed_not_training_loss_ranking(self):
        values = fixture()
        result = r.check_pair(*values)
        self.assertEqual(set(result), set(r.ARMS))
        for arm in r.ARMS:
            self.assertTrue(result[arm]["eligible_on_old_development"])
            self.assertLess(result[arm]["mean_projected_remaining_vocal_change_db"], 0.)
        self.assertNotIn("release_selection", result)

    def test_half_pair_wrong_step_model_hash_binding_and_promotion_rejected(self):
        for field in ("arm", "step", "cursor", "updates", "weights", "binding", "promotion", "filename", "policy", "baseline"):
            receipt, state, packet, bound, manifest, baseline = fixture()
            if field == "arm": state["arms"].pop(r.ARMS[1])
            if field == "step": packet["step"] = 500
            if field == "cursor": state["sampler"]["cursor"] = 249
            if field == "updates": state["arms"][r.ARMS[1]]["updates"] = 249
            if field == "weights": state["arms"][r.ARMS[1]]["model"]["weight"].add_(1)
            if field == "binding": receipt["binding"] = {"approval_sha256": "changed"}
            if field == "promotion": state["deployment_authorized"] = True
            if field == "filename": receipt["checkpoint"] = "../outside.pt"
            if field == "policy": packet["policy_sha256"] = "changed"
            if field == "baseline": packet["baseline_scores_sha256"] = "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.check_pair(receipt, state, packet, bound, manifest, baseline)

    def test_nonquarter_steps_and_bool_never_accepted(self):
        for step in (True, 0, 249, 251, 1250):
            values = fixture()
            values[0]["step"] = step
            with self.subTest(step=step), self.assertRaises(ValueError):
                r.check_pair(*values)

    def test_view_order_missing_metrics_nonfinite_forged_summary_rejected(self):
        for kind in ("missing", "reorder", "metric", "nonfinite", "summary", "assessment"):
            receipt, state, packet, bound, manifest, baseline = fixture()
            result = packet["evaluations"][r.ARMS[0]]
            if kind == "missing": result["rows"].pop()
            if kind == "reorder": result["rows"].reverse()
            if kind == "metric": result["rows"][0]["metrics"].pop("vocal_error_snr_db")
            if kind == "nonfinite": result["rows"][0]["metrics"]["vocal_error_snr_db"] = float("nan")
            if kind == "summary": result["summary"]["musdb/native"]["metrics"]["vocal_error_snr_db"]["mean"] += 100
            if kind == "assessment": packet["scores"][r.ARMS[0]]["rank_gain_db"] += 100
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                r.check_pair(receipt, state, packet, bound, manifest, baseline)

    def test_zero_gain_does_not_fake_improvement(self):
        receipt, state, packet, bound, manifest, baseline = fixture()
        for arm in r.ARMS:
            packet["evaluations"][arm] = copy.deepcopy(baseline)
            packet["scores"][arm] = r.dev.suite.assess(baseline["summary"], baseline["summary"])
        result = r.check_pair(receipt, state, packet, bound, manifest, baseline)
        self.assertTrue(all(not a["eligible_on_old_development"] for a in result.values()))
        self.assertTrue(all(a["mean_vocal_error_snr_gain_db"] == 0. for a in result.values()))

    def test_disk_commit_waits_for_receipt_then_checks_actual_checkpoint_hash(self):
        receipt, state, packet, bound, manifest, baseline = fixture()
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            historical = run / "historical.json"
            r.acq.write_new_json(historical, manifest)
            r.acq.write_new_json(run / "selection_suite.json", r.acq.seal(copy.deepcopy(manifest)))
            r.acq.write_new_json(run / "frozen_scores.json", r.acq.seal(copy.deepcopy(baseline)))
            r.acq.write_new_json(run / "development_step_0250.json", r.acq.seal(packet))
            with patch.object(r.inp, "verified_approval", return_value={}), \
                 patch.object(r, "expected_binding", return_value=bound), \
                 patch.object(r.dev, "OLD_MANIFEST", historical), \
                 patch.object(r.dev, "EXPECTED_SUITE", manifest["sha256"]), \
                 patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer")), \
                 patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA")):
                report, states, _ = r.inspect_run(run, historical)
                self.assertEqual(report["committed_evaluations"], [])
                self.assertEqual(report["pending_partial_steps"], [250])
                self.assertEqual(states, {})
                cp = run / receipt["checkpoint"]
                r.m.save_new(cp, state)
                receipt["sha256"] = r.acq.sha256(cp)
                r.acq.write_new_json(run / "checkpoint_0250.json", r.acq.seal(receipt))
                report, states, _ = r.inspect_run(run, historical)
                self.assertEqual(list(states), [250])
                self.assertEqual(report["release_selection"], "NONE")
                self.assertFalse(report["deployment"])
                with cp.open("ab") as stream: stream.write(b"changed")
                with self.assertRaises(ValueError): r.inspect_run(run, historical)

    def test_actual_frontend_cpu_readonly_preserves_all_modes_rng_weights(self):
        model = net()
        list(model.modules())[3].eval()  # deliberately heterogeneous modes
        modes = [v.training for v in model.modules()]
        before, rng = copy.deepcopy(model.state_dict()), r.m.capture_rng("cpu")
        time = torch.arange(80000, dtype=torch.float32)
        mix = (.1 * torch.sin(time * .021)).repeat(2, 1)
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA query")), \
             patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA init")):
            wave = r.vocal_wave(model, mix)
        self.assertEqual(wave.shape, mix.shape)
        self.assertTrue(torch.isfinite(wave).all())
        self.assertTrue(r.m.equal_state(before, model.state_dict()))
        self.assertTrue(r.m.equal_state(rng, r.m.capture_rng("cpu")))
        self.assertEqual(modes, [v.training for v in model.modules()])

    def test_failure_restores_readonly_state_and_bad_audio_rejected(self):
        model = net()
        before, rng = copy.deepcopy(model.state_dict()), r.m.capture_rng("cpu")
        mix = torch.ones(2, 80000) * .01
        def fail(*args):
            torch.rand(7)
            with torch.no_grad(): next(model.parameters()).add_(1)
            raise RuntimeError("injected")
        with patch.object(r.a, "context_masks", side_effect=fail), self.assertRaises(RuntimeError):
            r.vocal_wave(model, mix)
        self.assertTrue(r.m.equal_state(before, model.state_dict()))
        self.assertTrue(r.m.equal_state(rng, r.m.capture_rng("cpu")))
        for bad in (torch.ones(2, 1000), torch.ones(1, 80000), torch.ones(2, 80000).double(), mix * float("nan")):
            with self.assertRaises(ValueError): r.vocal_wave(model, bad)

    def test_one_playback_gain_for_all_nine_variants_preserves_pair_alignment(self):
        models = {name: object() for name in ("frozen", *r.ARMS)}
        mix, ref = torch.ones(2, 48000), torch.ones(2, 48000) * .25
        predictions = [mix * .1, mix * 2, mix * .6]
        with patch.object(r, "vocal_wave", side_effect=predictions) as predict:
            waves, gain = r.listening_waves(models, mix, ref)
        self.assertEqual(predict.call_count, 3)
        self.assertEqual(len(waves), 9)
        self.assertAlmostEqual(gain, .475)
        self.assertTrue(torch.equal(waves["mix"], mix * gain))
        for name in models:
            self.assertTrue(torch.allclose(waves[f"{name}_vocal"] + waves[f"{name}_backing"], waves["mix"], atol=1e-7))
        self.assertTrue(torch.allclose(waves["reference_vocal"] + waves["reference_backing"], waves["mix"]))
        self.assertLessEqual(max(float(v.abs().max()) for v in waves.values()), .950001)

    def test_float_audio_never_overwrites_and_roundtrips_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.wav"
            wave = torch.tensor([[.9, -.8, .6], [.9, -.8, .6]], dtype=torch.float32)
            r.a.save_wave(path, wave)
            actual, sr = sf.read(path, dtype="float32", always_2d=True)
            self.assertTrue(torch.equal(torch.from_numpy(actual.T.copy()), wave))
            self.assertEqual((sr, sf.info(path).subtype), (44100, "FLOAT"))
            with self.assertRaises(FileExistsError): r.a.save_wave(path, wave)

    def test_source_selection_remains_pre_training_and_fixed_prefix(self):
        self.assertEqual(r.LISTEN_SECONDS, 20)
        with patch.object(r.acq, "sha256", return_value="changed"), self.assertRaises(ValueError):
            r.checked_sources()

    def test_verifier_rejects_changed_binding_and_false_release_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            approval = out / "approval.json"
            r.acq.write_new_json(approval, {"fixture": True})
            review = {"purpose": r.PURPOSE, "release_selection": "NONE", "deployment": False,
                "cuda_used": False, "optimizer_constructed": False, "training_started": False,
                "independent_acceptance_scored": False, "approval": str(approval),
                "binding": {"approved": True}, "bindings_sha256": {str(approval): r.acq.sha256(approval)}}
            path = out / "review.json"
            r.acq.write_new_json(path, r.acq.seal(copy.deepcopy(review)))
            with patch.object(r.inp, "verified_approval", return_value={}), \
                 patch.object(r, "expected_binding", return_value=review["binding"]):
                r.verify_output(out)
                with approval.open("ab") as stream: stream.write(b"changed")
                with self.assertRaises(ValueError): r.verify_output(out)
            for flag in ("deployment", "cuda_used", "optimizer_constructed", "training_started", "independent_acceptance_scored"):
                changed = copy.deepcopy(review)
                changed[flag] = True
                with patch.object(r.acq, "read_sealed", return_value=changed), self.assertRaises(ValueError):
                    r.verify_output(out)
            with patch.object(r.acq, "read_sealed", return_value=review), \
                 patch.object(r.inp, "verified_approval", return_value={}), \
                 patch.object(r, "expected_binding", return_value={"approved": False}), self.assertRaises(ValueError):
                r.verify_output(out)

    def test_output_refuses_existing_directory_or_disk_reserve_violation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r.m.bulk, "guard_output"):
            with self.assertRaises(ValueError): r.fresh_output(Path(folder))
            with patch.object(r.shutil, "disk_usage", return_value=type("Disk", (), {"free": 100})()):
                with self.assertRaises(ValueError): r.fresh_output(Path(folder) / "new", audio=True)


if __name__ == "__main__":
    unittest.main()
