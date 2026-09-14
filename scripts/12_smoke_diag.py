"""12_smoke_diag.py -- trace every level in the distillation path.

The smoke run reported a big loss drop (val L1 1.96 -> 0.33) yet produced
essentially silent audio (SI-SDR -50 dB).  At least one of those is lying.

This walks the whole chain with a microscope, one stage at a time, printing
the signal level and the data distribution at each hop:

    mp3 -> wav -> teacher vocals -> STFT -> Wiener mask -> 128 bands
        -> student -> mask -> iSTFT -> wav

The first stage whose energy is wrong is the culprit.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import RESULTS, SR

_HERE = Path(__file__).resolve().parent
for _mod, _fn in (("t09", "09_target_model.py"), ("t11", "11_smoke_train.py")):
    _s = importlib.util.spec_from_file_location(_mod, _HERE / _fn)
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    globals()[_mod] = _m

t09, t11 = globals()["t09"], globals()["t11"]


def db(x: torch.Tensor) -> float:
    return float(20 * torch.log10(x.detach().float().pow(2).mean().sqrt() + 1e-12))


def describe(tag: str, m: torch.Tensor, extra: str = ""):
    m = m.detach().float().cpu()
    q = torch.quantile(m.flatten(), torch.tensor([0.5, 0.9, 0.99]))
    nan = bool(torch.isnan(m).any())
    inf = bool(torch.isinf(m).any())
    print(f"  {tag:<30} mean {m.mean():8.4f} std {m.std():8.4f} "
          f"min {m.min():8.4f} max {m.max():8.4f} "
          f"p50 {q[0]:7.4f} p99 {q[2]:7.4f}"
          + (f"  NaN={nan} Inf={inf}" if (nan or inf) else "")
          + (f"  {extra}" if extra else ""))


def probe_track(name: str, path: Path, device: str, model):
    """Full trace of one library track."""
    print("\n" + "-" * 78)
    print(f"TRACK: {name}")
    print("-" * 78)
    mix = t11.decode_excerpt(path, 60.0, 20.0)
    print(f"  mixture waveform  ({tuple(mix.shape)})  RMS {db(mix):7.2f} dBFS  "
          f"peak {float(mix.abs().max()):.4f}")

    voc = t11.teacher_vocals(model, mix, device)
    print(f"  teacher vocals    ({tuple(voc.shape)})  RMS {db(voc):7.2f} dBFS  "
          f"peak {float(voc.abs().max()):.4f}")
    ratio = db(voc) - db(mix)
    print(f"  vocal / mixture energy ratio: {ratio:+.1f} dB "
          f"({'plausible for a stem' if -20 < ratio < 3 else 'SUSPECT'})")
    noise = mix - voc
    print(f"  residual (mix-voc) RMS {db(noise):7.2f} dBFS")

    n = min(mix.shape[-1], voc.shape[-1])
    mix, voc = mix[:, :n], voc[:, :n]

    X = t09._stft(mix)
    V = t09._stft(voc)
    print(f"  STFT   |X| mean {X.abs().mean():.4f}   |V| mean {V.abs().mean():.4f}"
          f"   frames {X.shape[-1]}")

    num = (V * X.conj()).real
    den = X.abs() ** 2
    m_raw = num / (den + 1e-6)
    describe("mask raw (unclamped)", m_raw)
    m = m_raw.clamp(0.0, 1.0)
    describe("mask clamped [0,1]", m)
    print(f"  -> if this is ~0 the TARGET is broken, not the network")

    Wa = torch.from_numpy(t09.make_analysis_matrix())
    mix_band = torch.einsum("fb,cft->cbt", Wa, X.abs())
    m_band = torch.einsum("fb,cft->cbt", Wa, m)
    describe("mask in 128 bands", m_band)
    describe("mixture in 128 bands", mix_band)

    # what does band compression alone cost?  (oracle through the same path)
    Gs = torch.from_numpy(t09.make_synthesis_matrix())
    m_up = torch.einsum("fb,cbt->cft", Gs, m_band).clamp(0, 1)
    voc_or = t11._istft_dev(X * m_up, n)
    print(f"  oracle: band-compressed teacher mask -> iSTFT  "
          f"RMS {db(voc_or):7.2f} dBFS   "
          f"SI-SDR vs teacher {t09.si_sdr(voc_or.numpy(), voc.numpy()):6.2f} dB")

    # energy actually kept, per band, in dB relative to the mixture
    keep = (mix_band * m_band).pow(2).sum() / (mix_band.pow(2).sum() + 1e-12)
    print(f"  fraction of band-domain energy passed by the teacher mask: "
          f"{float(keep):.5f}  ({10*np.log10(float(keep)+1e-12):+.1f} dB)")
    return m_band


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print("=" * 78)
    print("DISTILLATION PATH TRACE")
    print("=" * 78)

    # ---------- part 1: is the teacher -> mask stage healthy? --------------
    print("\n[1] teacher -> Wiener mask, on real library tracks")
    from common import load_demucs
    model = load_demucs("htdemucs").to(device).eval()
    tracks = t11.find_tracks(seed=0, limit=4)
    for p in tracks[:3]:
        probe_track(p.name[:44], p, device, model)

    # ---------- part 2: what is in the cache? -----------------------------
    print("\n[2] cached targets (what training actually consumed)")
    items = t11.load_cache()
    print(f"  {len(items)} cached entries")
    print(f"  {'idx':>4}{'mix mean':>12}{'mask mean':>12}{'mask p99':>12}"
          f"{'mask max':>12}{'energy kept':>14}")
    for i, (mb, mm) in enumerate(items):
        keep = (mb * mm).pow(2).sum() / (mb.pow(2).sum() + 1e-12)
        q99 = torch.quantile(mm.flatten(), 0.99)
        flag = "  <-- near-empty target" if float(mm.mean()) < 0.01 else ""
        print(f"  {i:>4}{float(mb.mean()):>12.4f}{float(mm.mean()):>12.5f}"
              f"{float(q99):>12.4f}{float(mm.max()):>12.4f}"
              f"{10*np.log10(float(keep)+1e-12):>12.1f}dB{flag}")

    # ---------- part 3: the student -----------------------------------------
    print("\n[3] the trained student")
    ck = torch.load(RESULTS / "student_smoke.pt", map_location="cpu",
                    weights_only=False)
    net = t09.CausalSpectralUNet(2, (32, 64, 96, 128), 4).to(device)
    net.load_state_dict(ck["model"])
    net.eval()

    T = ck["crop"]
    mb, mm = items[-1]
    x = mb[:, :, 512:512 + T][None].to(device)
    y = mm[:, :, 512:512 + T][None].to(device)
    with torch.no_grad():
        o = net(x)
    print(f"  net output shape {tuple(o.shape)}  (batch, 4 mask channels, "
          f"128 bands, T)")
    describe("raw output (all 4 ch)", o)
    describe("raw output ch0:2", o[0, 0:2])
    describe("raw output ch2:4", o[0, 2:4])
    m_pred = ((o[0, 0:2].float() + 1.0) * 0.5).clamp(0, 1)
    describe("mapped mask ch0:2", m_pred)

    yt = 2.0 * y - 1.0
    print(f"\n  L1 of predictors on this crop (channel 0:2 only):")
    print(f"    {'trained net':<32}{float(F.l1_loss(2*m_pred-1, yt)):9.4f}")
    for lab, c in [("constant mask = 0 (silence)", -1.0),
                   ("constant mask = 0.5", 0.0),
                   ("constant mask = 1 (passthrough)", 1.0)]:
        print(f"    {lab:<32}{float(F.l1_loss(torch.full_like(yt, c), yt)):9.4f}")
    print(f"    E[target mask] on this crop = {float(y.mean()):.5f}")

    # ---------- part 4: what comes out the far end --------------------------
    print("\n[4] end-to-end output levels on the held-out track")
    val_wav = t09.WORK / "excerpt_60_100.wav"
    ref = RESULTS / "stems" / "htdemucs" / "vocals.wav"
    if val_wav.exists() and ref.exists():
        mix = t09.read_wav(val_wav)
        rv = t09.read_wav(ref)
        Wa = torch.from_numpy(t09.make_analysis_matrix())
        Gs = torch.from_numpy(t09.make_synthesis_matrix())
        print(f"  mixture           RMS {db(mix):7.2f} dBFS")
        print(f"  teacher vocals    RMS {db(rv):7.2f} dBFS")
        voc_s, acc_s = t11.separate(net, mix, Wa, Gs, device)
        print(f"  student vocals    RMS {db(voc_s):7.2f} dBFS")
        print(f"  student accomp    RMS {db(acc_s):7.2f} dBFS")

        # the model's mask on this track
        X = t11._stft_dev(mix.to(device))
        bands = torch.einsum("fb,cft->cbt", Wa.to(device), X.abs())
        Tb = bands.shape[-1]
        pad = (-Tb) % 8
        bb = F.pad(bands, (0, pad)) if pad else bands
        with torch.no_grad():
            oo = net(bb[None])[0]
        if pad:
            oo = oo[..., :Tb]
        mf = ((oo[0:2].float() + 1.0) * 0.5).clamp(0, 1)
        describe("student mask on the track", mf)

        # reference: the teacher's own mask, same path
        V = t11._stft_dev(rv.to(device))
        m_or = ((V * X.conj()).real / (X.abs() ** 2 + 1e-6)).clamp(0, 1)
        m_or_b = torch.einsum("fb,cft->cbt", Wa.to(device), m_or)
        describe("teacher mask, same track", m_or_b)

        n = min(mf.shape[-1], m_or_b.shape[-1])
        pred_energy = float((mf[:, :n] * mf[:, :n]).mean())
        ref_energy = float((m_or_b[:, :n] * m_or_b[:, :n]).mean())
        print(f"\n  mean squared mask value -- student {pred_energy:.5f} vs "
              f"teacher {ref_energy:.5f}")
    else:
        print("  skipped")
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
