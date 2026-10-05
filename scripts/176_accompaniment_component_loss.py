"""FP32 loss prototype only, not an approved trainer or launch entry point.

Single proposed variable: (ca-1)^2 component weight1 versus0. Original159
activity/role safeguards and cv^2 remain unchanged. The control returns159's
exact scalar/graph, preserving its FP32 operation order. Fresh authority,
real-TRAIN audit and state/runtime/rollback proofs are still required.
"""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("component_original_kernel", Path(__file__).with_name("159_source_projection_auxiliary.py"))
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
COEFFICIENT = .2
NORMALIZER = 6
PROPOSED_COMPONENT_WEIGHTS = (1, 0)


def source_projection_component_loss(predicted, mix, vocal, metadata, accompaniment_weight):
    if type(accompaniment_weight) is not int or accompaniment_weight not in PROPOSED_COMPONENT_WEIGHTS:
        raise ValueError("Only explicit accompaniment component weight1 or0, not bool")
    full, info = original.source_projection_auxiliary(predicted, mix, vocal, metadata)
    if accompaniment_weight == 1 or not info["active"]:
        return full, info
    # Exact159 FP32 arithmetic for the retained residual coefficient, not
    # full-minus-accompaniment subtraction or a changed activity criterion.
    a, v = (mix-vocal).detach().flatten(), vocal.detach().flatten()
    na, nv = a.norm(), v.norm()
    u, w = a/na, v/nv
    rho = u@w
    determinant = 1-rho.square()
    y = (mix-predicted).flatten()
    ya, yv = y@u, y@w
    cv = (yv-rho*ya)/(nv*determinant)
    loss = cv.square()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite retained residual auxiliary")
    return loss, info
