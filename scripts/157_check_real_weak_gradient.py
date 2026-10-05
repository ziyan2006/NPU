"""Supplementary REAL-graph weighted branch check, no optimizer or training.

The first three fork draws contain no -12 true slot. Exercise the separately
predeclared TRAIN audit draw on each backend without consuming update budget.
Verify weighted gradient against separately accumulated per-slot gradients.
Do not compare CPU and CUDA numerics as if they were bit-identical resumes.
"""
import argparse
import copy
import math
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("weak_real_grad", Path(__file__).with_name("154_train_mel_weak_weight.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def run(approval, audit_out, out, device):
    t.old.require_fresh(out)
    doc = t.p.verified_approval(approval)
    audit = t.acq.read_sealed(audit_out / "audit.json")
    if audit["binding"] != t.binding(approval) or not audit["model_unchanged"]:
        raise ValueError("Current predeclared TRAIN audit required")
    if device == "cuda" and not t.old.exploration_gpu_preflight(t.p.inp.verified_approval(Path(doc["origin_approval"])))['allowed']:
        return 2
    with t.d.deterministic_runtime(device):
        stream = t.p.MelForkStream(approval)
        batch = None
        for _ in range(audit["gradient_absolute_step"]-1000):
            batch = stream.next_batch()
        if batch["cursor"]+1 != audit["gradient_absolute_step"] or batch["metadata"] != [row["metadata"] for row in audit["gradients"]]:
            raise ValueError("Fixed audited TRAIN draw changed")
        weights = t.slot_weights(batch["metadata"], True)
        if 2. not in weights:
            raise ValueError("Weighted branch must actually be exercised")
        rng = t.d.portable(t.m.capture_rng(device))
        factory = t.m.frozen_factory(doc["source_protocol"])
        net = factory().to(device)
        net.load_state_dict(stream.source_state["arms"][t.ARMS[1]]["model"])
        net.train()
        digest = t.old.dev.state_digest(t.d.portable(net.state_dict()))
        wa = torch.from_numpy(t.m.core.t09.make_analysis_matrix()).to(device)
        gs = torch.from_numpy(t.m.core.t09.make_synthesis_matrix()).to(device)
        net.zero_grad(set_to_none=True)
        value = t.weighted_backward(net, batch["x"], batch["targets"][t.ARMS[0]], wa, gs, device, weights)
        actual = {name: param.grad.detach().clone() for name, param in net.named_parameters()}
        expected = {name: torch.zeros_like(grad) for name, grad in actual.items()}
        losses = 0.
        for i, weight in enumerate(weights):
            net.zero_grad(set_to_none=True)
            row = t.m.fit.backward_batch(net, batch["x"][i:i+1], batch["targets"][t.ARMS[0]][i:i+1], wa, gs,
                device, 96, 44, "reconstruction_l1", 1)
            losses += row["loss"]*weight/sum(weights)
            for name, param in net.named_parameters():
                expected[name].add_(param.grad, alpha=weight/sum(weights))
        # Scaling before versus after backward can differ by FP32 rounding.
        differences = {name: float((actual[name]-grad).abs().max()) for name, grad in expected.items()}
        if any(not torch.allclose(actual[name], grad, atol=2e-6, rtol=2e-4) for name, grad in expected.items()):
            raise ValueError("Weighted REAL-graph gradient not equal within predeclared FP32 tolerance")
        loss_difference = abs(value["loss"]-losses)
        if not math.isclose(value["loss"], losses, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Real weighted loss differs beyond scalar summation tolerance: {value['loss']} vs {losses}")
        if t.old.dev.state_digest(t.d.portable(net.state_dict())) != digest:
            raise ValueError("Diagnostic updated model")
        t.m.restore_rng(rng, device)
        evidence = {"schema": 1, "purpose": t.PURPOSE, "binding": t.binding(approval),
            "checker_sha256": t.acq.sha256(__file__), "audit_sha256": t.acq.sha256(audit_out / "audit.json"),
            "runtime": t.d.runtime_identity(device), "device": device, "absolute_draw": audit["gradient_absolute_step"],
            "slot_weights": weights, "actual_weight2_slots": [i for i, w in enumerate(weights) if w == 2.],
            "gradient_matches": True, "atol": 2e-6, "rtol": 2e-4, "max_parameter_abs_differences": differences,
            "loss_abs_difference": loss_difference, "scalar_summation_tolerance": 1e-12,
            "source_model_digest": digest, "model_unchanged": True, "optimizer_constructed": False,
            "updates": 0, "release_selection": "NONE", "deployment": False,
            "note": "TRAIN-only activity check; weighted FP32 accumulation tolerance, not cross-device exactness or quality"}
        out.mkdir(parents=True)
        t.acq.write_new_json(out / "real_weighted_gradient.json", t.acq.seal(evidence))
    print(f"REAL_WEAK_GRADIENT PASS device={device}; actual weight2 exercised; updates0", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, default=t.p.DEFAULT_APPROVAL)
    parser.add_argument("--audit-out", type=Path, default=t.AUDIT_OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    raise SystemExit(run(args.approval, args.audit_out, args.out or t.ROOT / f"results/mel_weak_real_gradient_{args.device}_20261003", args.device))
