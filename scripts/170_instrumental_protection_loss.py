"""Prototype loss composition only. No CLI, optimizer or approved trainer.

The proposed single variable is pure-instrumental base-loss weight 1 versus4.
Both arms retain auxiliary lambda .2 and denominator6. Do not renormalize by
sum(weights): that would also reduce every other slot's gradient contribution.
Existing checkpoint, training and runtime proof gates are not supplied here.
"""
import torch

DOMAINS = ("musdb", "mir1k", "instrumental", "pseudo", "pseudo", "pseudo")
COEFFICIENT = .2
NORMALIZER = 6


def combine_slot_loss(base, auxiliary, metadata, index, instrumental_weight, auxiliary_info):
    if (type(index) is not int or not 0 <= index < 6 or
        type(instrumental_weight) is not int or instrumental_weight not in (1, 4)):
        raise ValueError("Fixed six-slot index and instrumental weight1/4 required")
    if (metadata.get("domain") != DOMAINS[index] or metadata.get("role") != (
            "train" if index < 3 else "pseudo_label_train_candidate")):
        raise ValueError("Exact TRAIN roles required; no development/acceptance")
    if (not isinstance(base, torch.Tensor) or not isinstance(auxiliary, torch.Tensor) or
        base.ndim != 0 or auxiliary.ndim != 0 or base.dtype != torch.float32 or
        auxiliary.dtype != torch.float32 or base.device != auxiliary.device or
        not torch.isfinite(base) or not torch.isfinite(auxiliary)):
        raise ValueError("Finite aligned scalar FP32 base/auxiliary required")
    active = auxiliary_info.get("active")
    if (type(active) is not bool or (active and index >= 2) or
        (not active and float(auxiliary.detach()) != 0.)):
        raise ValueError("Auxiliary must retain original true-vocal activity gate")
    weight = instrumental_weight if index == 2 else 1
    weighted_base = weight*base
    contribution = COEFFICIENT*auxiliary
    combined = weighted_base+contribution
    if not torch.isfinite(combined):
        raise ValueError("Nonfinite combined protection loss")
    # Match166's existing FP32 operation order for weight1 control.
    loss = combined*(1/NORMALIZER)
    return loss, {"base_weight": weight, "normalizer": NORMALIZER,
                  "coefficient": COEFFICIENT,
                  "base_loss": float(base.detach())/NORMALIZER,
                  "weighted_base_loss": float(weighted_base.detach())/NORMALIZER,
                  "auxiliary_loss": float(auxiliary.detach())/NORMALIZER,
                  "auxiliary_contribution": float(contribution.detach())/NORMALIZER}
