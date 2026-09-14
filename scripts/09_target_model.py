"""Build and verify the *target* model: a causal, band-compressed spectral U-Net.

Until now the target architecture only existed as an analytic cost model
(`06_fpga_budget.py`): a formula that multiplied channel counts by spatial sizes.
Nothing had been instantiated, let alone run.  This script closes that gap.

What it does
------------
1. Instantiates a real PyTorch network matching the spec's intent, with strict
   causality (no tap ever reads a future frame).
2. Counts parameters and MACs *of the real graph* and compares them against the
   analytic spec numbers, so we know whether the spec's budget is trustworthy.
3. Measures the true peak activation working set.
4. Runs the whole chain end to end on the user's own audio:
       STFT -> fixed filterbank (513 -> 128 bands) -> U-Net -> mask -> iSTFT
   Two passes are recorded:
     * identity mask  -> output must equal the input.  This is a pass/fail test
       of the plumbing (padding, strides, shapes, overlap-add alignment).
     * random mask    -> what an UNTRAINED network sounds like, i.e. the honest
       "we have not trained this yet" reference.

The model is NOT trained.  It has no weights.  Any audio it produces is either a
plumbing check (identity) or noise (random).  Real separation audio requires a
training run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ROOT, RESULTS, SR, WORK, read_wav, n_params

N_FFT = 1024
HOP = 256
N_BINS = N_FFT // 2 + 1
N_BANDS = 128
FMIN, FMAX = 30.0, 16000.0
DEFAULT_BAND_LAYOUT = "legacy_log"


# ---------------------------------------------------------------------------
# fixed (non-learned) filterbank: 513 bins <-> 128 bands, 0 MAC on FPGA
# ---------------------------------------------------------------------------
def band_edges(n_bands: int = N_BANDS) -> np.ndarray:
    return np.geomspace(FMIN, FMAX, n_bands + 1)


def _mel_points(n: int, fmin: float, fmax: float) -> np.ndarray:
    f2m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    m2f = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # noqa: E731
    return m2f(np.linspace(f2m(fmin), f2m(fmax), n))


def band_centers(n_bands: int = N_BANDS,
                 layout: str = DEFAULT_BAND_LAYOUT) -> np.ndarray:
    """Band centres for cutoff mapping and centre-based filterbanks."""
    if layout == DEFAULT_BAND_LAYOUT:
        edges = band_edges(n_bands)
        return np.sqrt(edges[:-1] * edges[1:])
    if layout != "mel_unique":
        raise ValueError(f"unknown band layout: {layout}")
    requested = _mel_points(n_bands, 0.0, FMAX)
    idx = np.rint(requested / (SR / 2) * (N_BINS - 1)).astype(int)
    idx = np.clip(idx, 0, N_BINS - 1)
    for k in range(1, len(idx)):
        idx[k] = max(idx[k], idx[k - 1] + 1)
    if idx[-1] > N_BINS - 1:
        idx[-1] = N_BINS - 1
        for k in range(len(idx) - 2, -1, -1):
            idx[k] = min(idx[k], idx[k + 1] - 1)
    if np.any(np.diff(idx) <= 0):
        raise ValueError("not enough FFT bins for unique band centres")
    return idx * (SR / 2) / (N_BINS - 1)


def _center_interpolation(centers: np.ndarray) -> np.ndarray:
    freqs = np.linspace(0.0, SR / 2, N_BINS)
    G = np.zeros((N_BINS, len(centers)), dtype=np.float32)
    for j, f in enumerate(freqs):
        if f <= centers[0]:
            G[j, 0] = 1.0
        elif f >= centers[-1]:
            G[j, -1] = 1.0
        else:
            k = int(np.searchsorted(centers, f) - 1)
            k = min(max(k, 0), len(centers) - 2)
            w = (f - centers[k]) / (centers[k + 1] - centers[k])
            G[j, k] = 1.0 - w
            G[j, k + 1] = w
    return G


def make_analysis_matrix(n_bands: int = N_BANDS,
                         layout: str = DEFAULT_BAND_LAYOUT) -> np.ndarray:
    """(513, n_bands) triangular weighting; each column sums to 1."""
    if layout == "mel_unique":
        W = _center_interpolation(band_centers(n_bands, layout))
        col = W.sum(axis=0, keepdims=True)
        if np.any(col == 0):
            raise AssertionError("mel_unique produced an empty band")
        return W / col
    if layout != DEFAULT_BAND_LAYOUT:
        raise ValueError(f"unknown band layout: {layout}")
    freqs = np.linspace(0.0, SR / 2, N_BINS)
    edges = band_edges(n_bands)
    W = np.zeros((N_BINS, n_bands), dtype=np.float32)
    for k in range(n_bands):
        lo, hi = float(edges[k]), float(edges[k + 1])
        c = math.sqrt(lo * hi)
        if c <= lo or hi <= c:
            continue
        left = (freqs >= lo) & (freqs < c)
        right = (freqs >= c) & (freqs <= hi)
        W[left, k] = (freqs[left] - lo) / (c - lo)
        W[right, k] = (hi - freqs[right]) / (hi - c)
    col = W.sum(axis=0, keepdims=True)
    col[col == 0] = 1.0
    return W / col


def make_synthesis_matrix(n_bands: int = N_BANDS,
                          layout: str = DEFAULT_BAND_LAYOUT) -> np.ndarray:
    """(513, n_bands) linear interpolation from band centres back to bins."""
    if layout == "mel_unique":
        return _center_interpolation(band_centers(n_bands, layout))
    if layout != DEFAULT_BAND_LAYOUT:
        raise ValueError(f"unknown band layout: {layout}")
    centers = band_centers(n_bands, layout)
    freqs = np.linspace(0.0, SR / 2, N_BINS)
    G = np.zeros((N_BINS, n_bands), dtype=np.float32)
    for j, f in enumerate(freqs):
        if f <= centers[0]:
            G[j, 0] = 1.0
        elif f >= centers[-1]:
            G[j, -1] = 1.0
        else:
            k = int(np.searchsorted(centers, f) - 1)
            k = min(max(k, 0), n_bands - 2)
            w = (f - centers[k]) / (centers[k + 1] - centers[k])
            G[j, k] = 1.0 - w
            G[j, k + 1] = w
    return G


# ---------------------------------------------------------------------------
# causal convolution
# ---------------------------------------------------------------------------
class CausalConv2d(nn.Module):
    """2-D conv that pads time on the LEFT only, so no future frame is read.

    Frequency is treated as a non-sequential axis (symmetric 'same' padding).
    """

    def __init__(self, cin: int, cout: int, kf: int, kt: int,
                 sf: int = 1, st: int = 1, dilation_t: int = 1,
                 groups: int = 1):
        super().__init__()
        self.pad_f = (kf - 1) // 2
        self.pad_t = (kt - 1) * dilation_t
        self.conv = nn.Conv2d(cin, cout, (kf, kt), stride=(sf, st),
                              dilation=(1, dilation_t), groups=groups,
                              bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_t or self.pad_f:
            x = F.pad(x, (self.pad_t, 0, self.pad_f, self.pad_f))
        return self.conv(x)


class CausalResidualBlock(nn.Module):
    """Low-resolution residual context block, initialised as exact identity.

    It lives at the 16-band / 1/8-rate bottleneck, where a 3x3 convolution adds
    useful temporal and cross-band context for little activation storage.  A
    zero-initialised residual branch lets an older checkpoint warm-start the
    larger model without changing its first forward pass.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv2d(channels, channels, 3, 3, 1, 1)
        nn.init.zeros_(self.conv.conv.weight)
        nn.init.zeros_(self.conv.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.leaky_relu(self.conv(x), 0.1)


class CausalDepthwiseTemporalBlock(nn.Module):
    """Cheap long-context residual block at the low-rate bottleneck.

    A dilated 1x3 depthwise convolution grows temporal context without mixing
    every input/output channel at every tap; the following 1x1 convolution does
    the channel mixing.  The pointwise projection starts at zero, making the
    complete block an exact identity when warm-starting an older checkpoint.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        if dilation < 1:
            raise ValueError("temporal dilation must be >= 1")
        self.dilation = int(dilation)
        self.depthwise = CausalConv2d(
            channels, channels, 1, 3, 1, 1,
            dilation_t=self.dilation, groups=channels)
        self.pointwise = CausalConv2d(channels, channels, 1, 1, 1, 1)
        nn.init.zeros_(self.pointwise.conv.weight)
        nn.init.zeros_(self.pointwise.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.leaky_relu(self.depthwise(x), 0.1)
        return x + self.pointwise(y)


# ---------------------------------------------------------------------------
# the target network
# ---------------------------------------------------------------------------
class CausalSpectralUNet(nn.Module):
    """Spec geometry: enc (32,64,96,128), 3 time downsamples, mirrored decoder.

    Resolution trace with T divisible by 8 (T frames at 172.27 fps):

        x0 (128, T)   -> enc0 (3,3) s1   -> (128, T)      32 ch   <- skip
        e1 ( 64, T/2) -> enc1 (3,3) s2   -> ( 64, T/2)    64 ch   <- skip
        e2 ( 32, T/4) -> enc2 (3,3) s2   -> ( 32, T/4)    96 ch   <- skip
        e3 ( 16, T/8) -> enc3 (3,3) s2   -> ( 16, T/8)   128 ch
        b  ( 16, T/8) -> bott (1,3) s1   -> ( 16, T/8)   128 ch
        up -> ( 32, T/4) + skip e2       -> dec3 (3,3)     96 ch
        up -> ( 64, T/2) + skip e1       -> dec2 (3,3)     64 ch
        up -> (128, T  ) + skip x0       -> dec1 (1,3)     32 ch
                                        -> out  (1,1)     4 ch
    """

    def __init__(self, n_in: int = 2, enc=(32, 64, 96, 128), n_out: int = 4,
                 bottleneck_blocks: int = 0, temporal_dilations=()):
        super().__init__()
        e0, e1, e2, e3 = enc
        self.enc0 = CausalConv2d(n_in, e0, 3, 3, 1, 1)
        self.enc1 = CausalConv2d(e0, e1, 3, 3, 2, 2)
        self.enc2 = CausalConv2d(e1, e2, 3, 3, 2, 2)
        self.enc3 = CausalConv2d(e2, e3, 3, 3, 2, 2)
        self.bott = CausalConv2d(e3, e3, 1, 3, 1, 1)
        self.bott_blocks = nn.ModuleList(
            CausalResidualBlock(e3) for _ in range(bottleneck_blocks))
        self.temporal_dilations = tuple(int(d) for d in temporal_dilations)
        self.temporal_blocks = nn.ModuleList(
            CausalDepthwiseTemporalBlock(e3, d)
            for d in self.temporal_dilations)
        self.dec3 = CausalConv2d(e3 + e2, e2, 3, 3, 1, 1)
        self.dec2 = CausalConv2d(e2 + e1, e1, 3, 3, 1, 1)
        self.dec1 = CausalConv2d(e1 + e0, e0, 1, 3, 1, 1)
        self.out = CausalConv2d(e0, n_out, 1, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = F.leaky_relu(self.enc0(x), 0.1)
        e1 = F.leaky_relu(self.enc1(x0), 0.1)
        e2 = F.leaky_relu(self.enc2(e1), 0.1)
        e3 = F.leaky_relu(self.enc3(e2), 0.1)
        b = F.leaky_relu(self.bott(e3), 0.1)
        for block in self.bott_blocks:
            b = block(b)
        for block in self.temporal_blocks:
            b = block(b)
        d = F.interpolate(b, scale_factor=(2, 2), mode="nearest")
        d = F.leaky_relu(self.dec3(torch.cat([d, e2], dim=1)), 0.1)
        d = F.interpolate(d, scale_factor=(2, 2), mode="nearest")
        d = F.leaky_relu(self.dec2(torch.cat([d, e1], dim=1)), 0.1)
        d = F.interpolate(d, scale_factor=(2, 2), mode="nearest")
        d = F.leaky_relu(self.dec1(torch.cat([d, x0], dim=1)), 0.1)
        return torch.tanh(self.out(d))

    @torch.no_grad()
    def shape_trace(self, frames: int):
        """Return (layer, input_shape, output_shape) for a T-frame input."""
        rows = []
        x = torch.zeros(1, 2, N_BANDS, frames)
        handles = []
        for name, mod in self.named_children():
            def hook(m, inp, out, _n=name):
                rows.append((_n, tuple(inp[0].shape), tuple(out.shape)))
            handles.append(mod.register_forward_hook(hook))
        self(x)
        for h in handles:
            h.remove()
        return rows


# ---------------------------------------------------------------------------
# verification helpers
# ---------------------------------------------------------------------------
def count_macs(model: nn.Module, frames: int) -> tuple[int, int, dict]:
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.randn(1, 2, N_BANDS, frames)
    counter = FlopCounterMode(display=False)
    model.eval()
    with counter, torch.no_grad():
        model(x)
    flops = int(counter.get_total_flops())
    per_layer = {str(k): int(v) for k, v in counter.get_flop_counts()["Global"].items()}
    return flops, flops // 2, per_layer


def param_breakdown(model: nn.Module) -> "OrderedDict[str, int]":
    return OrderedDict((n, sum(p.numel() for p in m.parameters()))
                       for n, m in model.named_children())


def activation_peak(model: nn.Module, frames: int, bytes_per: int = 1) -> dict:
    """Largest single activation tensor produced anywhere in the graph."""
    seen = []
    handles = [m.register_forward_hook(
        lambda mod, inp, out: seen.append(tuple(out.shape)))
        for m in model.modules() if not isinstance(m, nn.Sequential)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 2, N_BANDS, frames))
    for h in handles:
        h.remove()
    best = 0
    best_shape = None
    total = 0
    for s in seen:
        n = int(np.prod(s))
        total += n * bytes_per
        if n > best:
            best, best_shape = n, s
    return {"max_layer_bytes": best * bytes_per, "max_layer_shape": best_shape,
            "sum_all_layers_bytes": total}


# ---------------------------------------------------------------------------
# full chain on real audio
# ---------------------------------------------------------------------------
def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est.astype(np.float64).ravel()
    ref = ref.astype(np.float64).ravel()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    tgt = a * ref
    num = np.sum(tgt ** 2)
    den = np.sum((est - tgt) ** 2)
    return float(10 * np.log10((num + 1e-12) / (den + 1e-12)))


def lsd_db(a: np.ndarray, b: np.ndarray) -> float:
    """Log-spectral distance between two magnitude spectrograms."""
    return float(np.sqrt(np.mean((20 * np.log10(a + 1e-8) -
                                 20 * np.log10(b + 1e-8)) ** 2)))


def _stft(x: torch.Tensor) -> torch.Tensor:
    """centre=True: 512 samples (11.6 ms) of look-ahead inside the analyser.

    torch refuses a full overlap-add reconstruction with centre=False because
    the window envelope degenerates at the very first samples.  The 11.6 ms is
    already inside the spec's latency term (n_fft/sr = 23.2 ms), so the budget
    is unchanged.  Everything downstream of the STFT stays strictly causal.
    """
    win = torch.hann_window(N_FFT, device=x.device)
    return torch.stft(x, N_FFT, HOP, window=win, center=True,
                      return_complex=True)


def _istft(X: torch.Tensor, length: int) -> torch.Tensor:
    win = torch.hann_window(N_FFT, device=X.device)
    return torch.istft(X, N_FFT, HOP, window=win, center=True, length=length)


def mask_resolution_test(x: torch.Tensor, voc_ref: torch.Tensor, Wa, Gs):
    """How much does a 128-band mask representation cost?

    Build a real, stable mask (Wiener) that turns the mixture into a strong
    reference model's vocal estimate, then compare

        (a) mask kept at full 513-bin resolution
        (b) mask pushed through 513 -> 128 bands -> 513 bins

    SI-SDR(b, a) is the honest price of the band compression, measured on a
    real mask from a real model instead of on a synthetic one.
    """
    X = _stft(x)
    V = _stft(voc_ref)
    m = (V * X.conj()).real / (X.abs() ** 2 + 1e-6)

    n = x.shape[-1]
    a = _istft(X * m, n)

    Wa_t, Gs_t = torch.from_numpy(Wa), torch.from_numpy(Gs)
    mb = torch.einsum("fb,cft->cbt", Wa_t, m)
    m_up = torch.einsum("fb,cbt->cft", Gs_t, mb)
    b = _istft(X * m_up, n)

    L = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :L], b[..., :L]
    Lr = min(L, voc_ref.shape[-1])
    return {
        "a": a, "b": b,
        "b_vs_a_dB": si_sdr(b.numpy(), a.numpy()),
        "a_vs_ref_dB": si_sdr(a[..., :Lr].numpy(), voc_ref[..., :Lr].numpy()),
        "b_vs_ref_dB": si_sdr(b[..., :Lr].numpy(), voc_ref[..., :Lr].numpy()),
    }


def mask_band_sweep(x: torch.Tensor, voc_ref: torch.Tensor,
                    band_list=(32, 64, 96, 128, 192, 256, 384)) -> dict:
    """Price of band compression as a function of band count.

    Same real Wiener mask each time; only the band count changes.  This tells
    us whether 128 bands is actually the right operating point.
    """
    X = _stft(x)
    V = _stft(voc_ref)
    m = (V * X.conj()).real / (X.abs() ** 2 + 1e-6)
    n = x.shape[-1]
    a = _istft(X * m, n)
    out = {}
    for nb in band_list:
        Wa = make_analysis_matrix(nb)
        Gs = make_synthesis_matrix(nb)
        mb = torch.einsum("fb,cft->cbt", torch.from_numpy(Wa), m)
        m_up = torch.einsum("fb,cbt->cft", torch.from_numpy(Gs), mb)
        b = _istft(X * m_up, n)
        out[nb] = round(si_sdr(b.numpy(), a.numpy()), 2)
    return out


def run_chain(model, x, Wa, Gs, mask_mode: str):
    """STFT -> 128 bands -> [net] -> mask -> iSTFT."""
    n = x.shape[-1]
    X = _stft(x)
    mag = X.abs()

    Wa_t = torch.from_numpy(Wa)
    Gs_t = torch.from_numpy(Gs)

    if mask_mode == "identity":
        mask_bin = torch.ones_like(mag)
    else:
        bands = torch.einsum("fb,cft->cbt", Wa_t, mag)
        # the net downsamples time by 8, so the frame count has to be a
        # multiple of 8 -- a real constraint the spec never mentioned
        T = bands.shape[-1]
        pad = (-T) % 8
        if pad:
            bands = F.pad(bands, (0, pad))
        out = model(bands.unsqueeze(0))[0]
        if pad:
            out = out[..., :T]
        mask_band = torch.stack([out[0], out[1]], 0)
        mask_bin = torch.einsum("fb,cbt->cft", Gs_t,
                                mask_band.abs()).clamp(0, 1)

    return _istft(X * mask_bin, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(WORK / "excerpt_60_100.wav"))
    ap.add_argument("--frames", type=int, default=176,
                    help="frame count used for the complexity scan (~1 s)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    outdir = RESULTS / "target_check"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TARGET MODEL -- instantiate and verify")
    print("=" * 78)

    net = CausalSpectralUNet()
    net.eval()
    total_params = n_params(net)
    print(f"\nparameters: {total_params:,} ({total_params/1e6:.4f} M)")
    for n, v in param_breakdown(net).items():
        print(f"    {n:<8} {v:>9,}  ({v/total_params:5.1%})")

    spec = json.loads((RESULTS / "fpga_budget.json").read_text(encoding="utf-8"))
    rec = [d for d in spec["designs"] if d["name"] == "causal-recommended"][0]
    print(f"\nspec (analytic)  : {rec['params_M']:.4f} M params, "
          f"{rec['gmac_per_audio_s']:.3f} GMAC/audio-s, "
          f"RTF {rec['realistic_rtf_7020']:.3f}, "
          f"act {rec['activation_total_KB']:.0f} KB")

    frames = args.frames
    audio_s = frames * HOP / SR
    flops, macs, _ = count_macs(net, frames)
    gmac = macs / 1e9 / audio_s
    w_kb = total_params / 1024

    print(f"measured (real)  : {total_params/1e6:.4f} M params, "
          f"{gmac:.3f} GMAC/audio-s, RTF {gmac/22.0:.3f}")

    print(f"\nMAC probe: {frames} frames = {audio_s:.4f} s of audio, "
          f"{flops/1e9:.4f} GFLOP = {macs/1e9:.4f} GMAC")

    print("\nactivation as a function of processing chunk length")
    print(f"    {'chunk':>7}{'audio ms':>10}{'peak tensor':>26}{'peak KB':>10}")
    act_rows = {}
    for chunk in (8, 16, 24, 32, 48):
        a = activation_peak(net, chunk, bytes_per=1)
        act_rows[chunk] = a
        shape = "x".join(str(v) for v in a["max_layer_shape"][1:])
        print(f"    {chunk:>7}{chunk*HOP/SR*1000:>10.0f}{shape:>26}"
              f"{a['max_layer_bytes']/1024:>10.1f}")
    peak = act_rows[frames] if frames in act_rows else activation_peak(
        net, frames, bytes_per=1)
    print("    (peak = largest single tensor materialised by a naive layer-by-"
          "layer\n     schedule; a fused streaming datapath keeps only line "
          "buffers instead)")

    print("\nlayer shape trace")
    for n, i, o in net.shape_trace(frames):
        print(f"    {n:<8} in {str(i):<22} -> out {str(o)}")

    # ---- causality proof -------------------------------------------------
    print("\ncausality check: perturb the LAST frame, see what changes")
    a = torch.zeros(1, 2, N_BANDS, frames)
    b = a.clone()
    b[..., -1] = 1.0
    ya, yb = net(a), net(b)
    diff = (ya - yb).abs().amax(dim=(1, 2)).squeeze(0)
    nz = int((diff > 1e-6).sum())
    print(f"    frames whose output changed: {nz} of {frames} "
          f"(delta on last frame = {float(diff[-1]):.4f})")
    print(f"    -> {'PASS: no earlier frame reacts to future input'
                     if nz <= 1 else 'FAIL: output depends on future frames'}")

    # ---- front end -------------------------------------------------------
    print("\nfixed filterbank: 513 bins <-> 128 bands, 0 learned parameters")
    Wa, Gs = make_analysis_matrix(), make_synthesis_matrix()
    print(f"    analysis {Wa.shape}, synthesis {Gs.shape}, "
          f"covered band {FMIN:.0f} Hz - {FMAX:.0f} Hz")
    x = read_wav(args.inp)

    # ---- the measurement that actually matters ---------------------------
    print("\nmask-resolution test (real Wiener mask from HTDemucs, not synthetic)")
    ref_path = RESULTS / "stems" / "htdemucs" / "vocals.wav"
    res = None
    mask_cost = None
    if ref_path.exists():
        voc_ref = read_wav(ref_path)
        res = mask_resolution_test(x, voc_ref, Wa, Gs)
        voc_a, voc_b = res["a"], res["b"]
        mask_cost = res["b_vs_a_dB"]
        print(f"    513-bin mask -> {res['a_vs_ref_dB']:>6.2f} dB vs HTDemucs vocals"
              f"   (phase-free Wiener mask, so this is the achievable ceiling)")
        print(f"    128-band mask-> {res['b_vs_ref_dB']:>6.2f} dB vs HTDemucs vocals"
              f"   <- the real number")
        print(f"    band compression costs "
              f"{res['a_vs_ref_dB'] - res['b_vs_ref_dB']:>6.2f} dB of agreement")
        print("\nband-count sweep on the same real mask")
        sweep_rows = mask_band_sweep(x, voc_ref)
        for nb, v in sweep_rows.items():
            print(f"    {nb:>4} bands -> {v:>6.2f} dB vs full-resolution mask")
    else:
        sweep_rows = {}
        print(f"    skipped: {ref_path} not found (run 03_separate.py first)")

    # ---- end to end ------------------------------------------------------
    print("\nend-to-end plumbing test on real audio")
    n_samp = x.shape[-1]
    voc_id = run_chain(net, x, Wa, Gs, "identity")
    L = min(voc_id.shape[-1], n_samp)
    voc_id, xc = voc_id[..., :L], x[..., :L]
    print(f"    identity mask: SI-SDR vs input = "
          f"{si_sdr(voc_id.numpy(), xc.numpy()):.2f} dB (plumbing must be exact)")
    print(f"    samples compared: {L:,} of {n_samp:,}")

    voc_rnd = run_chain(net, x, Wa, Gs, "net")
    voc_rnd = voc_rnd[..., :L]
    print(f"    untrained random mask vs input = "
          f"{si_sdr(voc_rnd.numpy(), xc.numpy()):.2f} dB (meaningless, untrained)")

    import soundfile as sf
    outdir.mkdir(parents=True, exist_ok=True)
    sf.write(str(outdir / "00_original.wav"), xc.numpy().T, SR, subtype="FLOAT")
    sf.write(str(outdir / "01_identity_mask.wav"), voc_id.numpy().T, SR,
             subtype="FLOAT")
    sf.write(str(outdir / "02_untrained_random_mask.wav"), voc_rnd.numpy().T, SR,
             subtype="FLOAT")
    if voc_b is not None:
        sf.write(str(outdir / "03_band128_mask.wav"),
                 voc_b.numpy().T, SR, subtype="FLOAT")
        sf.write(str(outdir / "04_fullres_mask.wav"),
                 voc_a.numpy().T, SR, subtype="FLOAT")

    payload = {
        "trained": False,
        "weights": "none -- randomly initialised, no training run yet",
        "spec_analytic": rec,
        "measured": {
            "params": total_params,
            "params_M": round(total_params / 1e6, 4),
            "param_breakdown": dict(param_breakdown(net)),
            "weight_int8_KB": round(w_kb, 1),
            "macs_per_audio_s_G": round(gmac, 4),
            "realistic_rtf_7020": round(gmac / 22.0, 4),
            "probe_frames": frames,
            "probe_audio_s": round(audio_s, 4),
            "activation_peak_KB_int8": round(peak["max_layer_bytes"] / 1024, 1),
            "activation_peak_shape": list(peak["max_layer_shape"]),
            "activation_by_chunk_KB": {
                str(c): round(a["max_layer_bytes"] / 1024, 1)
                for c, a in act_rows.items()},
            "mismatch_vs_spec": {
                "params_M": round(total_params / 1e6 - rec["params_M"], 4),
                "gmac_per_audio_s": round(gmac - rec["gmac_per_audio_s"], 4),
            },
        },
        "causality": {
            "frames_affected_by_last_frame": nz,
            "frames_total": frames,
            "pass": bool(nz <= 1),
        },
        "mask_resolution": {
            "source": "HTDemucs vocal estimate as the mask target",
            "fullres_vs_reference_dB": (
                round(res["a_vs_ref_dB"], 2) if res else None),
            "band128_vs_reference_dB": (
                round(res["b_vs_ref_dB"], 2) if res else None),
            "band128_vs_fullres_dB": (round(mask_cost, 2)
                                      if mask_cost is not None else None),
            "compression_cost_dB": (
                round(res["a_vs_ref_dB"] - res["b_vs_ref_dB"], 2)
                if res else None),
            "band_sweep_dB": {str(k): v for k, v in sweep_rows.items()},
        },
        "end_to_end": {
            "identity_mask_si_sdr_dB": round(si_sdr(voc_id.numpy(), xc.numpy()), 2),
            "random_mask_si_sdr_dB": round(si_sdr(voc_rnd.numpy(), xc.numpy()), 2),
            "covered_samples": int(L),
            "total_samples": int(n_samp),
        },
        "audio_dir": str(outdir),
    }
    p = RESULTS / "target_model_check.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    print(f"wrote audio to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
