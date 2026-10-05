"""Loss-only fork authority, identical target/state, rollback and gradient tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("aux_tests", Path(__file__).with_name("186_train_mel_lf_boundary.py"))
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
    state["step"] = state["sampler"]["cursor"] = state["schedule"]["step"] = state["schedule"]["last_validation"] = 3500
    for saved in state["arms"].values():
        saved["updates"] = 3500
        saved["optimizer"]["param_groups"][0]["lr"] = m.learning_rate(3500, m.validate_protocol(SOURCE))
        for value in saved["optimizer"]["state"].values():
            value["step"].fill_(3500)
    state["arms"][ARMS[1]]["model"]["0.weight"] += .01
    state["arms"][ARMS[1]]["modes"] = [False]*len(state["arms"][ARMS[1]]["modes"])
    state["schedule"]["stale"] = {ARMS[0]: 14, ARMS[1]: 99}
    state["schedule"]["stopped_at"] = 3500
    state["legacy_stop_events"] = [3250, 3500]
    value = p.BoundaryStream.__new__(p.BoundaryStream)
    value.path, value.bound, value.cursor, value.seed = path, acq.sha256(path), 3500, origin.seed
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


class LFBoundaryTests(unittest.TestCase):
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

    def test_exact_protocol(self):
        p.check_protocol(POLICY)
        for key,value in (("absolute_limit",4500),("source_arm",ARMS[1]),("origin_step",3000),
                          ("deployment_authorized",True),("source_stale",0),("base_normalizer",9),
                          ("additional_common_steps",True),("teacher","htdemucs")):
            with self.assertRaises(ValueError):
                p.check_protocol(POLICY | {key:value})

    def test_kill_types_and_mapping(self):
        p.require_kills(p.KILLS)
        for value in ({}, {arm:44 for arm in ARMS}, p.KILLS | {ARMS[0]:44.}, p.KILLS | {ARMS[1]:True}):
            with self.assertRaises(ValueError): p.require_kills(value)

    def test_ca_and_instrumental_fixed(self):
        for key,value in (("arm_accompaniment_weights", {ARMS[0]:1,ARMS[1]:0}),
                          ("arm_instrumental_weights", {ARMS[0]:4,ARMS[1]:8}),
                          ("arm_lambdas", {ARMS[0]:.2,ARMS[1]:.4})):
            with self.assertRaises(ValueError): p.check_protocol(POLICY | {key:value})

    def test_actual_source_zero_not_candidate(self):
        engine = self.engine(); state = engine.state_dict(self.stream)
        for arm in ARMS:
            self.assertTrue(m.equal_state(state["arms"][arm], self.stream.source_state["arms"][ARMS[0]]))
            self.assertFalse(m.equal_state(state["arms"][arm], self.stream.source_state["arms"][ARMS[1]]))
        self.assertEqual(state["schedule"]["stale"], {arm:14 for arm in ARMS})
        self.assertEqual(state["sampler"]["cursor"],3500)

    def test_migration_rng_modes_and_stop(self):
        state = self.engine().state_dict(self.stream)
        self.assertTrue(m.equal_state(state["rng"],self.stream.source_state["rng"]))
        self.assertEqual(state["source_stopped_at"],3500)
        self.assertEqual(state["source_legacy_stop_events"],[3250,3500])
        self.assertIsNone(state["schedule"]["stopped_at"])
        self.assertEqual(self.stream.source_state["schedule"]["stopped_at"],3500)

    def test_cpu_long_run_preoptimizer_rejected(self):
        with patch.object(torch.optim,"Adam",side_effect=AssertionError("optimizer")):
            with self.assertRaises(ValueError): t.ForkEngine(factory,self.stream,"cpu",smoke=False)
            with self.assertRaises(ValueError): t.ForkEngine(factory,object(),"cuda")

    def test_preparation_stream_preoptimizer_rejected(self):
        self.stream.path=None
        with patch.object(torch.optim,"Adam",side_effect=AssertionError("optimizer")):
            with self.assertRaises(ValueError): t.ForkEngine(factory,self.stream,"cpu",smoke=True)

    def test_per_arm_backward_dispatch(self):
        engine=self.engine(); engine._stream=self.stream
        self.stream.last_metadata=metadata()
        with patch.object(t,"boundary_backward",return_value={"loss":1.}) as call:
            for arm in ARMS:
                row=engine._backward(engine.models[arm],None,None,None,None,"cpu",96,44,"reconstruction_l1",1)
                self.assertEqual(call.call_args.args[-1],p.KILLS[arm])
                self.assertEqual(row["kill_bands"],p.KILLS[arm])

    def test_original_call_geometry_guard(self):
        engine=self.engine()
        for warmup,kill,obj,micro in ((128,44,"reconstruction_l1",1),(96,32,"reconstruction_l1",1),
                                    (96,44,"different",1),(96,44,"reconstruction_l1",2)):
            with self.assertRaises(ValueError):
                engine._backward(engine.models[ARMS[0]],None,None,None,None,"cpu",warmup,kill,obj,micro)

    def test_control_gradient_bit_exact_old178(self):
        engine=self.engine(); batch=self.stream.next_batch()
        a,b=factory(),factory()
        left=t.boundary_backward(a,batch["x"],batch["targets"][ARMS[0]],engine.wa,engine.gs,"cpu",batch["metadata"],44)
        right=p.t.component_backward(b,batch["x"],batch["targets"][ARMS[0]],engine.wa,engine.gs,"cpu",batch["metadata"],1)
        self.assertEqual(left,right)
        self.assertTrue(all(torch.equal(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters())))

    def test_candidate_gradient_explicit_kill32(self):
        engine=self.engine(); batch=self.stream.next_batch(); a,b=factory(),factory()
        row=t.boundary_backward(a,batch["x"],batch["targets"][ARMS[0]],engine.wa,engine.gs,"cpu",batch["metadata"],32)
        scalar=0.
        for i,meta in enumerate(batch["metadata"]):
            x,v=batch["x"][i:i+1],batch["targets"][ARMS[0]][i:i+1]
            spectrum=m.core.stft_batch(x); output=b(torch.einsum("fk,bcft->bckt",engine.wa,spectrum.abs()))
            base,_,pv=m.fit.reconstruction_loss((output[:,:2]+1)/2,spectrum,x,v,engine.gs,96,32)
            region=m.fit.suite.scoring_slice(x.shape[-1],96)
            aux,_=t.k.source_projection_component_loss(pv[0,:,region],x[0,:,region],v[0,:,region],meta,1)
            loss=(4 if i==2 else 1)*base+.2*aux
            (loss*(1/6)).backward(); scalar+=float(loss.detach())*(1/6)
        self.assertEqual(row["loss"],scalar)
        self.assertTrue(all(torch.equal(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters())))
        self.assertEqual((row["auxiliary_active_count"],row["auxiliary_skip_count"]),(2,4))

    def test_invalid_boundary_refused(self):
        engine=self.engine();batch=self.stream.next_batch()
        for kill in (True,False,32.,44.,0,128,43):
            with self.assertRaises(ValueError):
                t.boundary_backward(factory(),batch["x"],batch["targets"][ARMS[0]],engine.wa,engine.gs,"cpu",batch["metadata"],kill)

    def test_actual_fft_zero_sets_and_matrix_unchanged(self):
        gs=torch.from_numpy(m.core.t09.make_synthesis_matrix())
        self.assertEqual(tuple(gs.shape),(513,128))
        for kill,expected in ((44,list(range(6))),(32,list(range(4)))):
            self.assertEqual(torch.nonzero(gs[:,kill:].eq(0).all(1)).flatten().tolist(),expected)

    def test_mask_no_alias_or_input_change(self):
        gs=torch.from_numpy(m.core.t09.make_synthesis_matrix())
        mask=torch.ones(1,2,128,16); original=mask.clone()
        a=m.fit.protected_full_mask(mask,gs,44); b=m.fit.protected_full_mask(mask,gs,32)
        self.assertTrue(torch.equal(mask,original)); self.assertGreater(float((b-a).abs().max()),0)
        self.assertTrue(bool(a[:,:,:6].eq(0).all())); self.assertTrue(bool(b[:,:,:4].eq(0).all()))

    def test_disk_bit_replay_no_alias(self):
        engine=self.engine();engine.backward=cheap_backward
        engine.update_next(self.stream)
        receipt=t.save_checkpoint(Path(self.folder.name),engine,self.stream)
        saved=m.load_checked_checkpoint(Path(self.folder.name)/receipt["checkpoint"],receipt["sha256"])
        untouched=d.portable(saved)
        expected_rows=[engine.update_next(self.stream) for _ in range(2)]
        expected=engine.state_dict(self.stream)
        self.stream.cursor=3500; other=self.engine();other.backward=cheap_backward
        other.load_state_dict(saved,self.stream)
        actual=[other.update_next(self.stream) for _ in range(2)]
        plain=lambda rows:[{k:v for k,v in row.items() if k!="seconds"} for row in rows]
        self.assertEqual(plain(expected_rows),plain(actual))
        self.assertTrue(m.equal_state(expected,other.state_dict(self.stream)))
        self.assertTrue(m.equal_state(saved,untouched))
        with self.assertRaises(ValueError): other.update_next(self.stream)

    def test_second_arm_fault_full_rollback(self):
        engine=self.engine();before=engine.state_dict(self.stream)
        def fail(net,*args):
            if net is engine.models[ARMS[1]]:
                torch.rand(2);m.random.random();m.np.random.random()
                raise RuntimeError("second")
            return cheap_backward(net,*args)
        engine.backward=fail
        with self.assertRaises(RuntimeError): engine.update_next(self.stream)
        self.assertTrue(m.equal_state(before,engine.state_dict(self.stream)))

    def test_changed_targets_full_rollback(self):
        engine=self.engine();before=engine.state_dict(self.stream);draw=self.stream.next_batch
        def bad():
            batch=draw();batch["targets"][ARMS[1]][3]+=.01
            return batch
        self.stream.next_batch=bad
        with self.assertRaises(ValueError): engine.update_next(self.stream)
        self.assertTrue(m.equal_state(before,engine.state_dict(self.stream)))

    def test_state_kills_teacher_stop_and_budget_guard(self):
        engine=self.engine();state=engine.state_dict(self.stream)
        for key,value in (("arm_kill_bands",{}),("source_stopped_at",None),("teacher","htdemucs"),
                          ("origin_sha256","bad"),("limit",4000),("legacy_stop_events",[3500])):
            with self.assertRaises(ValueError): engine.load_state_dict(state | {key:value},self.stream)
        bad=copy.deepcopy(state);bad["schedule"]["stopped_at"]=3500
        with self.assertRaises(ValueError): engine.load_state_dict(bad,self.stream)

    def test_state_sampler_type_boundaries(self):
        engine=self.engine();state=engine.state_dict(self.stream)
        for cursor in (True,3499,4001):
            with self.assertRaises(ValueError): self.stream.load_state_dict(state["sampler"] | {"cursor":cursor})

    def test_new_stopping_common_budget(self):
        engine=self.engine();engine.limit=4000
        def stopped(*args):
            engine.schedule.stopped_at=engine.step
            return {}
        with patch.object(t.old,"observe",side_effect=stopped):
            engine.step=engine.schedule.step=self.stream.cursor=3750
            engine.observe(None,None);self.assertIsNone(engine.schedule.stopped_at)
            engine.step=engine.schedule.step=self.stream.cursor=4000
            engine.observe(None,None);self.assertEqual(engine.schedule.stopped_at,4000)
        self.assertEqual(engine.legacy_stop_events,[3750,4000])

    def test_state_metadata_no_alias(self):
        engine=self.engine();state=engine.state_dict(self.stream)
        state["arm_kill_bands"][ARMS[1]]=0;state["source_legacy_stop_events"].append(1)
        other=engine.state_dict(self.stream)
        self.assertEqual(other["arm_kill_bands"],p.KILLS)
        self.assertEqual(other["source_legacy_stop_events"],[3250,3500])

    def test_cross_device_same_pcm_metadata_kills(self):
        row={"step":3501,"input_sha256":["same"]*6,"metadata":metadata(),"arm_roles":p.ROLES,
             "arm_lambdas":p.LAMBDAS,"arm_instrumental_weights":p.WEIGHTS,
             "arm_accompaniment_weights":p.CA_WEIGHTS,"arm_kill_bands":p.KILLS}
        cpu={"draws":[row]}; t.check_cross_device_inputs(cpu,copy.deepcopy(cpu))
        for key,value in (("input_sha256",[]),("metadata",[]),("arm_kill_bands",{}),("step",3502)):
            bad=copy.deepcopy(cpu);bad["draws"][0][key]=value
            with self.assertRaises(ValueError): t.check_cross_device_inputs(cpu,bad)

    def test_validator_dispatch_keeps_frozen44(self):
        legacy=type("Fixture",(),{"rows":[],"manifest":{"sha256":"fixed"},"policy":{},
            "policy_digest":"policy","evaluator":None,"wa":None,"gs":None,
            "baseline":{"summary":"frozen44"},"baseline_digest":acq.content_digest({"summary":"frozen44"})})()
        validator=t.BoundaryValidator(legacy)
        calls=[]
        def score(net,kill):
            calls.append(kill); return {"summary":kill}
        validator._score_at=score
        with patch.object(t.old.dev.suite,"assess",return_value={"eligible":False}), patch.object(t.old.dev,"state_digest",return_value="weights"):
            packet=validator.evaluate_pair({arm:factory() for arm in ARMS},3500)
        self.assertEqual(calls,[44,32]);self.assertEqual(packet["frozen_kill_bands"],44)
        self.assertEqual(packet["evaluations"][ARMS[1]]["summary"],32)

    def test_baseline_different_lf_scores_allowed_but_wrong_source_refused(self):
        review=t.p.load("review_tests","188_review_mel_lf_boundary.py")
        origin={"arms":{ARMS[0]:{"fixture":1},ARMS[1]:{"fixture":2}}}
        state={"arms":{arm:{"fixture":1} for arm in ARMS}}
        packet={"evaluations":{ARMS[0]:{"score":1},ARMS[1]:{"score":9}},"arm_kill_bands":p.KILLS}
        original={"evaluations":{ARMS[0]:{"score":1},ARMS[1]:{"score":2}}}
        review.check_fork_baseline(state,packet,origin,original)
        with self.assertRaises(ValueError):
            review.check_fork_baseline({"arms":{arm:{"fixture":2} for arm in ARMS}},packet,origin,original)

    def test_listening_both_origins_current_kills_common_gain(self):
        review=t.p.load("listening_tests","188_review_mel_lf_boundary.py")
        roles=("frozen","origin_full_3500_lf44","origin_full_3500_lf32",*p.ROLES.values())
        models={role:object() for role in roles};x=torch.ones(2,100);v=x*.2
        with patch.object(review,"vocal_wave",return_value=x*2) as call:
            waves,gain=review.listening_waves(models,x,v)
        self.assertEqual(len(waves),13);self.assertEqual(gain,.475)
        self.assertEqual([a.args[-1] for a in call.call_args_list],[44,44,32,44,32])
        self.assertTrue(torch.equal(waves["reference_vocal"],v*gain))

    def test_launcher_and_original_guards_unchanged(self):
        launcher=Path(__file__).with_name("187_start_mel_lf_boundary.ps1").read_text(encoding="utf-8")
        for text in ("Windows PowerShell 5.1","WmiPrvSE.exe","-WindowStyle Hidden","Wait-PairedProcessExit",
                     "186_train_mel_lf_boundary.py","184_diagnose_train_low_frequency_reference",
                     "178_train_mel_component_ablation","detached_exit_","mel_lf_boundary_cuda_20261003"):
            self.assertIn(text,launcher)
        historical=Path(p.t.__file__).read_text(encoding="utf-8")
        self.assertIn('(96, 44, "reconstruction_l1", 1)',historical)

    def test_auxiliary_actual_kill_evidence_required(self):
        row={"arm_kill_bands":p.KILLS,"losses":{arm:{"kill_bands":p.KILLS[arm],"auxiliary_active_count":2,
            "auxiliary_contribution":.2,"instrumental_weight":4,"instrumental_base_loss":.01,
            "instrumental_weighted_contribution":.04,"instrumental_extra_contribution":.03,
            "accompaniment_weight":1,"remaining_vocal_component_contribution":.1,
            "accompaniment_component_contribution":.01} for arm in ARMS}}
        t.auxiliary_evidence([row])
        for key,value in (("kill_bands",44),("auxiliary_contribution",0.),("accompaniment_component_contribution",0.)):
            bad=copy.deepcopy(row);bad["losses"][ARMS[1]][key]=value
            with self.assertRaises(ValueError): t.auxiliary_evidence([bad])

    def test_roles_and_pseudo_metadata(self):
        t.validate_metadata(metadata())
        bad=metadata();bad[3]["role"]="train"
        with self.assertRaises(ValueError): t.validate_metadata(bad)
        bad=metadata();bad[2]["vocal_db"]=-12
        with self.assertRaises(ValueError): t.validate_metadata(bad)

    def test_existing_evidence_never_overwritten(self):
        with self.assertRaises(ValueError): t.old.require_fresh(Path(self.folder.name))

    def test_new_baseline_filename_and_stop_history(self):
        review=Path(__file__).with_name("188_review_mel_lf_boundary.py").read_text(encoding="utf-8")
        trainer=Path(t.__file__).read_text(encoding="utf-8")
        self.assertIn('development_step_3500.json',trainer)
        self.assertNotIn('development_step_3000.json',trainer)
        self.assertNotIn('[2750, 3500]',review)

    def test_low_frequency_zero_error_finite_and_zero_reference_skip(self):
        gen=torch.Generator().manual_seed(10)
        v=torch.randn(1,2,89856,generator=gen)*.01; x=v+torch.randn(1,2,89856,generator=gen)*.03
        row=t.low_frequency_backing_metrics(v,x,v)
        self.assertEqual(row["lf_backing_error_power"],0.)
        self.assertTrue(torch.isfinite(torch.tensor(row["lf_backing_error_snr_db"])))
        empty=t.low_frequency_backing_metrics(v,v,v)
        self.assertIsNone(empty["lf_backing_error_snr_db"])

    def test_low_frequency_descriptor_does_not_modify_wave(self):
        gen=torch.Generator().manual_seed(11)
        x=torch.randn(1,2,89856,generator=gen)*.03;v=x*.2;pv=v*.8
        snapshots=[z.clone() for z in (x,v,pv)]
        rng=d.portable(m.capture_rng("cpu"))
        row=t.low_frequency_backing_metrics(pv,x,v)
        self.assertGreater(row["lf_backing_error_power"],0.)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(snapshots,(x,v,pv))))
        self.assertTrue(m.equal_state(rng,m.capture_rng("cpu")))
        self.assertFalse(torch.cuda.is_initialized())

    def test_new_evaluation_preserves_original119_scores(self):
        engine=self.engine();batch=self.stream.next_batch()
        rows=[{"x":batch["x"][0],"v":batch["targets"][ARMS[0]][0],"domain":"fixture", "track":"fixture"}]
        a,b=factory(),factory()
        with torch.no_grad():
            old=t.old.dev.suite.evaluate(a,rows,engine.wa,engine.gs,"cpu",96,44)
            new,lf=t.boundary_evaluation(b,rows,engine.wa,engine.gs,44)
        self.assertEqual(old,new);self.assertEqual(len(lf["rows"]),1)

    def test_independent_adam_layout_checks(self):
        engine=self.engine();state=engine.state_dict(self.stream)
        review=p.load("independent_adam_tests","188_review_mel_lf_boundary.py")
        review.check_layout_and_adam(state,self.stream.source_state)
        bad=copy.deepcopy(state);bad["arms"][ARMS[1]]["parameter_names"].reverse()
        with self.assertRaises(ValueError): review.check_layout_and_adam(bad,self.stream.source_state)
        bad=copy.deepcopy(state);bad["arms"][ARMS[0]]["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
        with self.assertRaises(ValueError): review.check_layout_and_adam(bad,self.stream.source_state)

    def test_actual_graph_layout_audit_preserves_rng(self):
        # Actual frozen graph construction, no forward/optimizer/CUDA or quality inference.
        source=p.m.frozen_factory(SOURCE)()
        saved={"model":source.state_dict(),"parameter_names":[n for n,_ in source.named_parameters()]}
        rng=d.portable(m.capture_rng("cpu"))
        result=p.hardware_identity({"arms":{p.SOURCE_ARM:saved}},SOURCE)
        self.assertTrue(result["source_and_both_arms_layout_equal"])
        self.assertEqual(result["added_model_mac"],0)
        self.assertFalse(result["new_board_implementation_verified"])
        self.assertTrue(m.equal_state(rng,m.capture_rng("cpu")))

if __name__ == "__main__":
    unittest.main()

