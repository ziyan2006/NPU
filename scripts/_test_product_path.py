"""Small regression tests for the product-aligned mask path."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("t09_test", HERE / "09_target_model.py")
t09 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t09
spec.loader.exec_module(t09)

spec = importlib.util.spec_from_file_location("t11_test", HERE / "11_smoke_train.py")
t11 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t11
spec.loader.exec_module(t11)


def main() -> None:
    torch.manual_seed(0)
    net = t09.CausalSpectralUNet().eval()
    x = torch.rand(2, 2, 128, 16)

    net.mask_mode = "independent"
    mv_old, ma_old = t11.forward_masks(net, x)
    assert mv_old.shape == ma_old.shape == x.shape

    net.mask_mode = "complement"
    mv, ma = t11.forward_masks(net, x)
    assert torch.equal(mv, mv_old)
    assert torch.allclose(mv + ma, torch.ones_like(mv), atol=1e-7)

    const = torch.full((2, 2, 128, 16), 0.37)
    assert torch.equal(t11.causal_smooth_mask(const, 1), const)
    assert torch.allclose(t11.causal_smooth_mask(const, 4), const, atol=1e-7)

    # A perturbation in the final frame must not affect an earlier smoothed mask.
    a = torch.zeros(1, 2, 8, 16)
    b = a.clone()
    b[..., -1] = 1.0
    sa = t11.causal_smooth_mask(a, 4)
    sb = t11.causal_smooth_mask(b, 4)
    assert torch.equal(sa[..., :-1], sb[..., :-1])
    assert float(sb[..., -1].mean()) == 0.25

    # New low-resolution blocks must be a no-regression warm start.
    base = t09.CausalSpectralUNet().eval()
    wider_context = t09.CausalSpectralUNet(bottleneck_blocks=2).eval()
    inc = wider_context.load_state_dict(base.state_dict(), strict=False)
    assert not inc.unexpected_keys
    assert inc.missing_keys and all(k.startswith("bott_blocks.")
                                    for k in inc.missing_keys)
    z = torch.rand(1, 2, 128, 16)
    assert torch.equal(base(z), wider_context(z))
    assert sum(p.numel() for p in wider_context.parameters()) > sum(
        p.numel() for p in base.parameters())

    # Dilated temporal blocks also warm-start as an exact identity and never
    # allow a future-frame perturbation to change an earlier output.
    temporal = t09.CausalSpectralUNet(
        bottleneck_blocks=2, temporal_dilations=(1, 2, 4)).eval()
    inc = temporal.load_state_dict(wider_context.state_dict(), strict=False)
    assert not inc.unexpected_keys
    assert inc.missing_keys and all(k.startswith("temporal_blocks.")
                                    for k in inc.missing_keys)
    assert torch.equal(wider_context(z), temporal(z))
    block = t09.CausalDepthwiseTemporalBlock(8, dilation=4).eval()
    torch.nn.init.normal_(block.pointwise.conv.weight, std=0.01)
    q0 = torch.zeros(1, 8, 4, 20)
    q1 = q0.clone()
    q1[..., -1] = 1.0
    assert torch.equal(block(q0)[..., :-1], block(q1)[..., :-1])

    pred = torch.rand(2, 2, 128, 16)
    target = torch.rand_like(pred)
    focal = t11.focal_mask_loss(pred, target, mid_weight=4.0)
    plain = (pred - target).abs().mean()
    assert torch.isfinite(focal) and focal >= plain

    # Phase-sensitive supervision is the closed-form real-mask projection.
    X = torch.tensor([1 + 2j, 2 - 1j, 1 + 0j, 1 + 0j])
    V = torch.stack((X[0], 0.5 * X[1], 1j * X[2], -X[3]))
    psm = t11.phase_sensitive_mask(X, V)
    assert torch.allclose(psm, torch.tensor([1.0, 0.5, 0.0, 0.0]))
    grid = torch.linspace(0.0, 1.0, 101)
    brute = ((grid[:, None] * X - V).abs().square()).argmin(dim=0)
    assert torch.allclose(psm, grid[brute], atol=0.011)

    # The replacement 128-band filterbank must have no dead input channels and
    # must interpolate a constant mask back to a constant bin mask.
    Wa = t09.make_analysis_matrix(layout="mel_unique")
    Gs = t09.make_synthesis_matrix(layout="mel_unique")
    assert Wa.shape == Gs.shape == (t09.N_BINS, t09.N_BANDS)
    assert np.all((Wa > 0).sum(axis=0) > 0)
    assert np.allclose(Wa.sum(axis=0), 1.0, atol=2e-6)
    assert np.allclose(Gs.sum(axis=1), 1.0, atol=1e-7)

    # Frequency height does not change any learned parameter shape.  A
    # 128-band checkpoint can therefore warm-start the wider 192-band path.
    Wa192 = t09.make_analysis_matrix(192, layout="legacy_log")
    Gs192 = t09.make_synthesis_matrix(192, layout="legacy_log")
    assert Wa192.shape == Gs192.shape == (t09.N_BINS, 192)
    wide = t09.CausalSpectralUNet(bottleneck_blocks=2).eval()
    with torch.no_grad():
        y192 = wide(torch.randn(1, 2, 192, 16))
    assert y192.shape == (1, 4, 192, 16)
    print("product-path regression tests: PASS")


if __name__ == "__main__":
    main()
