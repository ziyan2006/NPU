"""21_render_demo.py -- render the button product on a track that actually sings.

Two problems with the demo the training script produces on its own:

  * the excerpt it renders is a fixed 40 s of one held-out track, and in
    house/EDM that usually lands on an instrumental, so pressing the button
    appears to do almost nothing;
  * it renders the mask exactly as the network emits it, which -- per
    19_filterbank_audit and 20_mask_postproc -- subtracts a *constant* quarter of
    the kick and bass, because the network's input below ~500 Hz is
    structurally zero.

So this script renders a chosen track twice, with and without the low-frequency
protection below, and reports the sub-120 Hz energy of each, which is the number
that explains the bass dip a listener hears on the button press.

    python 21_render_demo.py --ckpt v4_best.pt \
        --track "Jerro, Sophia Bel - Demons.mp3" --lf-hz 250
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common import RESULTS, SR, load_demucs                    # noqa: E402


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t09 = _load("t09", "09_target_model.py")
t11 = _load("t11", "11_smoke_train.py")


def band_energy_db(x: np.ndarray, lo: float, hi: float) -> float:
    """Energy in [lo, hi) Hz, via an FFT of the whole clip (dB, relative)."""
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    sel = (f >= lo) & (f < hi)
    return float(10.0 * np.log10(float((np.abs(X[:, sel]) ** 2).sum()) + 1e-20))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v4_best.pt")
    ap.add_argument("--track", default="Jerro, Sophia Bel - Demons.mp3")
    ap.add_argument("--seg", type=float, default=30.0)
    ap.add_argument("--lf-hz", type=float, default=250.0)
    ap.add_argument("--out-prefix", default="demo")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = None
    for cand in t11.LIB.rglob("*"):
        if cand.name == args.track:
            p = cand
            break
    if p is None:
        raise SystemExit(f"track not found in {t11.LIB}: {args.track}")

    total = t11.ffprobe_duration(p)
    start = 0.5 * max(0.0, total - args.seg)
    mix = t11.decode_excerpt(p, start, args.seg)
    n = mix.shape[-1]
    print(f"device {device}  {args.track}")
    print(f"  {args.seg:g} s from {start:.1f} s of a {total:.1f} s track")

    teacher = load_demucs("htdemucs").to(device).eval()
    ref_v = t11.teacher_vocals(teacher, mix, device)
    n = min(n, ref_v.shape[-1])
    mix, ref_v = mix[:, :n], ref_v[:, :n]
    floor = float(t09.si_sdr(mix.numpy(), ref_v.numpy()))
    print(f"  SI-SDR(mixture, teacher vocals) = {floor:+.2f} dB "
          f"({'vocals present' if floor > -8 else 'NO usable vocal here'})")
    del teacher
    torch.cuda.empty_cache()

    blob = torch.load(RESULTS / args.ckpt, map_location="cpu", weights_only=False)
    net = t09.CausalSpectralUNet(
        2, (32, 64, 96, 128), 4,
        bottleneck_blocks=int(blob.get("bottleneck_blocks", 0)),
        temporal_dilations=tuple(blob.get("temporal_dilations", ()))).to(device)
    net.load_state_dict(blob["model"])
    net.frontend = tuple(blob.get("frontend") or ("linear", 1.0))
    net.mask_mode = blob.get("mask_mode", "independent")
    net.band_layout = blob.get("band_layout", t09.DEFAULT_BAND_LAYOUT)
    net.n_bands = int(blob.get("n_bands", t09.N_BANDS))
    net.eval()
    layout = t11.band_layout_for(net)
    n_bands = t11.n_bands_for(net)
    Wa = torch.from_numpy(t09.make_analysis_matrix(n_bands, layout=layout))
    Gs = torch.from_numpy(t09.make_synthesis_matrix(n_bands, layout=layout))

    lfb = t11.lf_kill_band_for(args.lf_hz, layout, n_bands)
    print(f"  checkpoint {args.ckpt} (step {blob.get('step')})   "
          f"lf kill: {args.lf_hz:g} Hz -> bands 0..{lfb - 1} of "
          f"{n_bands} (layout {layout})")

    acc_ref = mix - ref_v
    report = {"track": args.track, "lf_hz": args.lf_hz, "lf_bands": lfb,
              "floor_dB": round(floor, 2)}
    for tag, lf in (("nolf", 0), ("lf", lfb)):
        voc, acc = t11.separate(net, mix, Wa, Gs, device, lf_kill_band=lf)
        voc, acc = voc[..., :n], acc[..., :n]
        d = RESULTS / f"{args.out_prefix}_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / "01_mixture.wav"), mix.T.numpy(), SR, subtype="PCM_16")
        sf.write(str(d / "02_student_vocals.wav"), voc.T.numpy(), SR,
                 subtype="PCM_16")
        sf.write(str(d / "03_student_accompaniment.wav"), acc.T.numpy(), SR,
                 subtype="PCM_16")
        sf.write(str(d / "04_teacher_vocals.wav"), ref_v.T.numpy(), SR,
                 subtype="PCM_16")
        sf.write(str(d / "05_teacher_accompaniment.wav"), acc_ref.T.numpy(), SR,
                 subtype="PCM_16")
        sz = float(t09.si_sdr(voc.numpy(), ref_v.numpy()))
        sa = float(t09.si_sdr(acc.numpy(), acc_ref.numpy()))
        # the number a listener hears as "the bass dropped"
        b_mix = band_energy_db(mix.numpy(), 25.0, 120.0)
        b_acc = band_energy_db(acc.numpy(), 25.0, 120.0)
        # how much teacher vocal energy survives in the accompaniment
        resid = band_energy_db(acc.numpy() - acc_ref.numpy(), 20.0, 16000.0) \
            - band_energy_db(ref_v.numpy(), 20.0, 16000.0)
        report[tag] = {"vocal_si_sdr": round(sz, 2), "accomp_si_sdr": round(sa, 2),
                       "sub120Hz_change_dB": round(b_acc - b_mix, 2),
                       "residual_vocal_vs_teacher_dB": round(resid, 2)}
        print(f"  [{tag:>4}] vocal {sz:+6.2f} dB   accomp {sa:+6.2f} dB   "
              f"25-120 Hz {b_acc - b_mix:+5.2f} dB   "
              f"residual vocal {resid:+6.2f} dB   -> {d}")

    out = RESULTS / f"{args.out_prefix}_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    print("next: python 14_button_demo.py --source demo_lf --out "
          "Z_button_demo_v4_lf.mp3 --outdir listen_v4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
