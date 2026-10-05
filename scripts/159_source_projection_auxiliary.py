"""Prototype FP32 source-gain auxiliary. No CLI, optimizer or training launcher.

This kernel is NOT an approved trainer. A future experiment must bind locked
true TRAIN roles, identical Mel targets/state/input/budget, mechanism evidence
and fresh authority. Only true TRAIN vocal domains receive this auxiliary.
"""
import torch

RMS_FLOOR = 1e-4
MIN_GRAM_DETERMINANT = 1e-3
PROPOSED_LAMBDA = .02


def source_projection_auxiliary(predicted_vocal, mix, vocal, metadata):
    if (predicted_vocal.shape != mix.shape or vocal.shape != mix.shape or mix.ndim != 2 or mix.shape[0] != 2 or
        mix.numel() == 0 or any(value.dtype != torch.float32 or value.device != mix.device or not torch.isfinite(value).all()
                               for value in (predicted_vocal, mix, vocal))):
        raise ValueError("Aligned finite FP32 scored stereo required")
    domain, role = metadata.get("domain"), metadata.get("role")
    if (role not in ("train", "pseudo_label_train_candidate") or
        domain not in ("musdb", "mir1k", "instrumental", "pseudo") or
        (domain == "pseudo") != (role == "pseudo_label_train_candidate")):
        raise ValueError("Kernel cannot score development/acceptance or mismatched source roles")
    zero = predicted_vocal.sum()*0
    if domain not in ("musdb", "mir1k"):
        return zero, {"active": False, "reason": "Not true vocal TRAIN; original base loss only"}
    a, v = (mix-vocal).detach().flatten(), vocal.detach().flatten()
    if float(a.square().mean()) <= RMS_FLOOR**2 or float(v.square().mean()) <= RMS_FLOOR**2:
        return zero, {"active": False, "reason": "Inactive source; original base loss only"}
    na, nv = a.norm(), v.norm()
    u, w = a/na, v/nv
    rho = u@w
    determinant = 1-rho.square()
    if float(determinant) < MIN_GRAM_DETERMINANT:
        return zero, {"active": False, "reason": "Ill-conditioned correlated sources; original base loss only"}
    y = (mix-predicted_vocal).flatten()
    ya, yv = y@u, y@w
    ca = (ya-rho*yv)/(na*determinant)
    cv = (yv-rho*ya)/(nv*determinant)
    loss = cv.square()+(ca-1).square()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite source auxiliary; never silently update")
    return loss, {"active": True, "accompaniment_gain": float(ca.detach()),
                  "remaining_vocal_gain": float(cv.detach()), "gram_determinant": float(determinant)}
