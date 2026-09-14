"""13_ab_compare.py -- A/B two student checkpoints on the same held-out audio.

    python 13_ab_compare.py                       # v1 vs v2 (the defaults)
    python 13_ab_compare.py --a results/v2_model.pt --b results/v3_model.pt \
                            --a-name v2 --b-name v3

The held-out set is pinned to one run's report (--names-from) so successive
comparisons stay on identical excerpts.

Why this script exists
----------------------
The v1 and v2 runs reported val log-L1 on *different* validation sets (3 tracks
vs 22), so those two numbers cannot be subtracted from each other.  And the
single-track SI-SDR the training script prints at the end is n=1 -- far too
little to call a result.  This script removes both problems at once:

  * one fixed set of held-out tracks, decoded from a window neither run trained
    on, so the comparison is apples to apples;
  * a waveform-domain metric (SI-SDR against the HTDemucs teacher) computed per
    track, so the spread is visible instead of hidden inside one number.

The teacher itself is the ceiling here: the student is never asked to beat
HTDemucs, only to get closer to it under the causal + 128-band constraints.

Outputs
-------
results/ab_v1_v2.json     per-track SI-SDR, medians, timings
results/ab/*.wav|mp3      teacher vs v1 vs v2 for a couple of tracks, to listen
"""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Some library filenames contain characters outside the active Windows console
# code page.  A metrics run must not die merely because a title cannot be
# printed; JSON output remains UTF-8 and preserves the original filename.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from common import RESULTS, SR, load_demucs          # noqa: E402


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t09 = _load("t09", "09_target_model.py")
t11 = _load("t11", "11_smoke_train.py")

LIB = t11.LIB
OUT = RESULTS / "ab"
MAX_TRACKS = 24

# defaults reproduce the original v1 -> v2 comparison; any two checkpoints of the
# same architecture can be compared by passing --a / --b
DEFAULT_A = RESULTS / "student_smoke.pt"     # 15.0 min of audio, 6000 steps
DEFAULT_B = RESULTS / "v2_model.pt"          # 110 min of audio, budget-limited


def held_out_tracks(limit: int, report: Path) -> list[str]:
    """The pinned holdout from one run's report.

    Pinned on purpose: the held-out excerpts must stay identical across
    comparisons, otherwise the SI-SDR columns are not comparable between one
    A/B and the next.  Newer runs record an explicit ``holdout_tracks`` list
    -- tracks carved out of the training pool in 11_smoke_train.find_tracks, so
    no checkpoint ever fitted them.  Older runs (v1..v3) have no such list and
    fall back to the last N of their own validation split, which those runs
    likewise never trained on.
    """
    rep = json.loads(report.read_text(encoding="utf-8"))
    if rep.get("holdout_tracks"):
        return list(rep["holdout_tracks"])[:limit] if limit else list(
            rep["holdout_tracks"])
    return [r["file"] for r in rep["stage_a_tracks"]][-limit:]


def find_file(name: str) -> Path | None:
    for p in LIB.rglob("*"):
        if p.name == name and p.suffix.lower() in t11.AUDIO_EXT:
            return p
    return None


def _db(x: torch.Tensor) -> float:
    return float(20 * torch.log10(x.float().pow(2).mean().sqrt() + 1e-12))


def ffprobe_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=nw=1:nk=1",
                        str(path)], capture_output=True, text=True, timeout=30)
    return float(r.stdout.strip())


def decode(path: Path, start_s: float, dur_s: float) -> torch.Tensor:
    """Decoded with a *fixed* 50% offset.  The cache entries used a random
    15-75% window, so a 50% window is at worst the same track a different time
    -- for these tracks the training run never used this window."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}",
           "-i", str(path), "-ar", str(SR), "-ac", "2", "-f", "wav", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or len(p.stdout) < 2000:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[:200])
    x, _ = sf.read(io.BytesIO(p.stdout), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(x.T))


def load_student(ckpt: Path, device: str):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = t09.CausalSpectralUNet(
        2, (32, 64, 96, 128), 4,
        bottleneck_blocks=int(blob.get("bottleneck_blocks", 0)),
        temporal_dilations=tuple(blob.get("temporal_dilations", ()))).to(device)
    net.load_state_dict(blob["model"])
    # the input front end is part of the model, not of the harness: a checkpoint
    # trained on a sqrt-compressed spectrum must be *evaluated* on one, or it
    # sees a completely different input distribution.  Old checkpoints have no
    # field and were trained on the raw linear spectrum.
    net.frontend = tuple(blob.get("frontend", ("linear", 1.0)))
    net.mask_mode = blob.get("mask_mode", "independent")
    net.band_layout = blob.get("band_layout", t09.DEFAULT_BAND_LAYOUT)
    net.n_bands = int(blob.get("n_bands", t09.N_BANDS))
    net.eval()
    return net, blob


def write_audio(tag: str, stems: dict[str, torch.Tensor], n: int):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, w in stems.items():
        wav = OUT / f"{tag}_{name}.wav"
        sf.write(str(wav), w[..., :n].numpy().T, SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav),
                        "-b:a", "256k", str(wav.with_suffix(".mp3"))],
                       capture_output=True)
        wav.unlink(missing_ok=True)


def frame_vocal_ratio_db(ref: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Per-STFT-frame vocal-to-mixture energy ratio, in dB.

    Returns one value per analysis frame (the same 172.27 fps grid the network
    works on).  ~0 dB means the frame is vocals with a bit of backing; below
    -20 dB means there is effectively nothing to remove.
    """
    V = t09._stft(ref)
    X = t09._stft(mix)
    ev = V.abs().pow(2).sum(dim=(0, 1))
    em = X.abs().pow(2).sum(dim=(0, 1)) + 1e-12
    return 10.0 * torch.log10(ev / em + 1e-12)


def sample_mask_from_frames(frame_db: torch.Tensor, n: int,
                            pred) -> torch.Tensor:
    """Expand a per-frame boolean test into a (1, n) sample-domain mask.

    With centre=True there is one more frame than n // HOP, so the expanded
    mask is a little longer than the signal and can simply be truncated.
    """
    m = pred(frame_db).repeat_interleave(t09.HOP)[:n]
    return m[None].float()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(DEFAULT_A), help="baseline checkpoint")
    ap.add_argument("--b", default=str(DEFAULT_B), help="candidate checkpoint")
    ap.add_argument("--a-name", default=None,
                    help="label for the baseline (default: file stem)")
    ap.add_argument("--b-name", default=None)
    ap.add_argument("--names-from", default=str(RESULTS / "v2_report.json"),
                    help="report whose stage_a_tracks defines the held-out set; "
                         "keep this pinned so runs stay comparable")
    ap.add_argument("--tracks", type=int, default=MAX_TRACKS)
    ap.add_argument("--seg", type=float, default=30.0,
                    help="excerpt length in seconds")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="where the excerpt sits inside the usable range; 0.5 is "
                         "the historical 50%% offset.  A single excerpt is noisy "
                         "on near-silent tracks (GPU float non-determinism gets "
                         "amplified when the reference is near the noise floor), "
                         "so the low-variance protocol is to run --frac 0.25 / "
                         "0.5 / 0.75 and average the per-track numbers")
    ap.add_argument("--write-audio", default="0,1",
                    help="comma-separated track indices to export for listening; "
                         "pick indices flagged as having a real vocal, otherwise "
                         "you are just listening to a breakdown")
    ap.add_argument("--a-lf-hz", type=float, default=0.0,
                    help="low-frequency protection cutoff for model A")
    ap.add_argument("--b-lf-hz", type=float, default=0.0,
                    help="low-frequency protection cutoff for model B")
    ap.add_argument("--a-smooth-frames", type=int, default=1,
                    help="causal trailing mask-average length for model A")
    ap.add_argument("--b-smooth-frames", type=int, default=1,
                    help="causal trailing mask-average length for model B")
    ap.add_argument("--a-mask-gain", type=float, default=1.0,
                    help="vocal-mask strength for model A")
    ap.add_argument("--b-mask-gain", type=float, default=1.0,
                    help="vocal-mask strength for model B")
    args = ap.parse_args()
    listen_idx = {int(x) for x in args.write_audio.split(",") if x.strip() != ""}

    def label(path_s: str, override: str | None) -> str:
        return override or Path(path_s).stem.replace("_model", "")

    la, lb = label(args.a, args.a_name), label(args.b, args.b_name)
    checkpoints = {la: Path(args.a), lb: Path(args.b)}
    render_cfg = {
        la: {"lf_hz": args.a_lf_hz, "smooth_frames": args.a_smooth_frames,
             "mask_gain": args.a_mask_gain},
        lb: {"lf_hz": args.b_lf_hz, "smooth_frames": args.b_smooth_frames,
             "mask_gain": args.b_mask_gain},
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg = args.seg
    print("=" * 78)
    print(f"A/B: {la} vs {lb} on the same held-out audio")
    print(f"excerpt {seg:g} s at frac {args.frac:g} of the usable range")
    print("=" * 78)

    names = held_out_tracks(args.tracks, Path(args.names_from))
    print(f"device {device}   held-out tracks: {len(names)} "
          f"(pinned to {Path(args.names_from).name})")

    teacher = load_demucs("htdemucs").to(device).eval()
    nets, meta = {}, {}
    filterbanks = {}
    for name, path in checkpoints.items():
        net, blob = load_student(path, device)
        nets[name] = net
        layout = t11.band_layout_for(net)
        n_bands = t11.n_bands_for(net)
        filterbanks[name] = (
            torch.from_numpy(t09.make_analysis_matrix(n_bands, layout=layout)),
            torch.from_numpy(t09.make_synthesis_matrix(n_bands, layout=layout)))
        meta[name] = {"checkpoint": path.name, "trained_steps": blob.get("step"),
                      "best_val_logL1": blob.get("best_val"),
                      "mask_mode": blob.get("mask_mode", "independent"),
                      "band_layout": layout,
                      "n_bands": n_bands,
                      **render_cfg[name]}
        print(f"  loaded {name}: {path.name}  params {blob['params']:,}  "
              f"step {blob.get('step')}")

    rows = []
    print(f"\n  {'track':<32}{'mix~T':>7}{'voc':>7}{la:>8}{lb:>8}{'gain':>8}"
          f"  |{'acc'+la:>8}{'acc'+lb:>8}{'gain':>8}")
    for i, nm in enumerate(names):
        p = find_file(nm)
        if p is None:
            print(f"  {i:>2}  {nm[:34]:<34} MISSING")
            continue
        try:
            total = ffprobe_duration(p)
            start = args.frac * max(0.0, total - seg)
            mix = decode(p, start, seg)
            ref = t11.teacher_vocals(teacher, mix, device)
            n = min(mix.shape[-1], ref.shape[-1])
            mix, ref = mix[:, :n], ref[:, :n]
        except Exception as exc:
            print(f"  {i:>2}  {nm[:34]:<34} SKIP ({type(exc).__name__})")
            continue

        # --- product metrics -------------------------------------------------
        # The button product outputs the ACCOMPANIMENT, so scoring only the
        # estimated vocal is scoring the wrong stem.  Two numbers matter:
        #   acc  = SI-SDR(student accompaniment, teacher accompaniment)
        #          -- how close the thing the user actually hears is to target;
        #   idle = SI-SDR(student accompaniment, mixture) measured only on
        #          frames where the teacher hears no vocals.  There the correct
        #          behaviour is to change nothing, so a low "idle" score means
        #          the model is subtracting energy it invented -- the offline
        #          face of the musical-noise / pumping artefact that dominates
        #          continuous playback.
        ratio = frame_vocal_ratio_db(ref, mix)
        quiet = sample_mask_from_frames(ratio, n, lambda r: r < -20.0)
        has_quiet = bool(quiet.sum() > t09.HOP)      # ignore ~empty selections

        sdr, acc_sdr, idle = {}, {}, {}
        vocs, accs = {}, {}
        for name, net in nets.items():
            cfg = render_cfg[name]
            Wa, Gs = filterbanks[name]
            lf_band = t11.lf_kill_band_for(
                cfg["lf_hz"], t11.band_layout_for(net),
                t11.n_bands_for(net))
            voc, acc = t11.separate(
                net, mix, Wa, Gs, device, lf_kill_band=lf_band,
                mask_smooth_frames=cfg["smooth_frames"],
                vocal_mask_gain=cfg["mask_gain"])
            vocs[name], accs[name] = voc, acc
            sdr[name] = float(t09.si_sdr(voc[..., :n].numpy(), ref[..., :n].numpy()))
            acc_ref = (mix - ref)[..., :n]
            acc_sdr[name] = float(t09.si_sdr(acc[..., :n].numpy(), acc_ref.numpy()))
            if has_quiet:
                idle[name] = float(t09.si_sdr(
                    (acc[..., :n] * quiet).numpy(),
                    (mix[..., :n] * quiet).numpy()))
        floor = float(t09.si_sdr(mix[..., :n].numpy(), ref[..., :n].numpy()))
        # Two excerpts in the first pass scored SI-SDR(mixture, teacher vocals)
        # near -30 dB, which is physically impossible when the vocals really are
        # part of the mix -- those windows landed on a breakdown, the teacher
        # output is noise there, and any SI-SDR against it is meaningless.  A
        # track only counts if the teacher's vocal is both audible in absolute
        # terms and a genuine component of the mixture.
        v_db = _db(ref)
        vocal_present = bool(v_db > -40.0 and floor > -8.0)
        rows.append({"track": nm, "mixture_vs_teacher_si_sdr_dB": round(floor, 2),
                     "teacher_vocal_dBFS": round(v_db, 2),
                     "mixture_dBFS": round(_db(mix), 2),
                     "vocal_frame_frac": round(float((ratio > -10).float().mean()), 3),
                     "quiet_frame_frac": round(float(quiet.mean()), 3),
                     "vocal_present": vocal_present,
                     **{f"{k}_si_sdr": round(v, 2) for k, v in sdr.items()},
                     **{f"{k}_acc_si_sdr": round(v, 2) for k, v in acc_sdr.items()},
                     **{f"{k}_acc_idle_si_sdr": round(v, 2) for k, v in idle.items()},
                     "gain_dB": round(sdr[lb] - sdr[la], 2),
                     "acc_gain_dB": round(acc_sdr[lb] - acc_sdr[la], 2)})
        print(f"  {nm[:32]:<32}{floor:>7.2f}{v_db:>7.1f}"
              f"{sdr[la]:>8.2f}{sdr[lb]:>8.2f}"
              f"{sdr[lb]-sdr[la]:>+8.2f}"
              f"  |{acc_sdr[la]:>8.2f}{acc_sdr[lb]:>8.2f}"
              f"{acc_sdr[lb]-acc_sdr[la]:>+8.2f}"
              f"{'' if vocal_present else '  (no vocal)'}", flush=True)

        if i in listen_idx:
            short = Path(nm).stem.split(" - ")[-1][:24].replace(" ", "_")
            write_audio(f"{i}_{short}",
                        {"01_mixture": mix, "02_teacher_vocals": ref,
                         f"03_{la}_vocals": vocs[la],
                         f"04_{lb}_vocals": vocs[lb],
                         f"05_{la}_accompaniment": accs[la],
                         f"06_{lb}_accompaniment": accs[lb],
                         "07_teacher_accompaniment": mix - ref}, n)

    def stats(sel, key: str = "{m}_si_sdr"):
        if not sel:
            return {}
        ka, kb = key.format(m=la), key.format(m=lb)
        sel = [r for r in sel if ka in r and kb in r]
        if not sel:
            return {}
        a = np.array([r[ka] for r in sel])
        b = np.array([r[kb] for r in sel])
        g = b - a
        return {
            "n_tracks": len(sel),
            f"{la}_median_dB": round(float(np.median(a)), 2),
            f"{lb}_median_dB": round(float(np.median(b)), 2),
            "median_gain_dB": round(float(np.median(g)), 2),
            "mean_gain_dB": round(float(g.mean()), 2),
            "tracks_improved": int((g > 0).sum()),
        }

    gated = [r for r in rows if r["vocal_present"]]
    summary = {
        "device": device,
        "segment_s": seg,
        "frac": args.frac,
        "compare": [la, lb],
        "held_out_from": Path(args.names_from).name,
        "per_track": rows,
        "models": meta,
        "scorable_header": "mix~T = SI-SDR(mixture, teacher vocals); voc = teacher "
                           "vocal level in dBFS; an excerpt is scorable only if the "
                           "teacher's vocal is audible AND is a real component of "
                           "the mixture (mix~T > -8 dB), otherwise SI-SDR against "
                           "it measures noise",
        "metric_notes": {
            "vocal": "SI-SDR(student vocals, teacher vocals)",
            "accompaniment": "SI-SDR(student accompaniment, teacher accompaniment) "
                             "-- the product's actual output stem",
            "idle": "SI-SDR(student accompaniment, mixture) restricted to frames "
                    "where the teacher hears no vocals.  Correct behaviour there "
                    "is to change nothing, so this is the offline face of the "
                    "musical-noise / pumping risk that only shows up in "
                    "continuous playback",
        },
        "caveat": "validation is against the HTDemucs teacher, not ground truth; "
                  "the 128-band mask architecture has its own 9.87 dB ceiling vs a "
                  "full-resolution oracle (reports/03).  The two runs' log-L1 "
                  "numbers are NOT comparable -- they used different val sets.",
        "all_excerpts": stats(rows),
        "scorable_excerpts": stats(gated),
        "all_excerpts_accompaniment": stats(rows, "{m}_acc_si_sdr"),
        "scorable_excerpts_accompaniment": stats(gated, "{m}_acc_si_sdr"),
        "all_excerpts_idle": stats(rows, "{m}_acc_idle_si_sdr"),
        "scorable_excerpts_idle": stats(gated, "{m}_acc_idle_si_sdr"),
        "per_track": rows,
    }
    out = RESULTS / f"ab_{la}_{lb}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print()
    for lbl, s in (("all excerpts", summary["all_excerpts"]),
                   ("scorable only", summary["scorable_excerpts"])):
        if s:
            ma = s[f"{la}_median_dB"]
            mb = s[f"{lb}_median_dB"]
            print(f"  {'vocal ' + lbl:<20} n={s['n_tracks']}  "
                  f"median SI-SDR vs teacher: {la} {ma:+.2f} -> {lb} {mb:+.2f} dB  "
                  f"(median gain {s['median_gain_dB']:+.2f}, "
                  f"mean {s['mean_gain_dB']:+.2f} dB, "
                  f"{lb} better on {s['tracks_improved']}/{s['n_tracks']})")
    for lbl, key in (("accompaniment (all)", "all_excerpts_accompaniment"),
                     ("accompaniment (scorable)", "scorable_excerpts_accompaniment"),
                     ("idle/no-vocal (all)", "all_excerpts_idle")):
        s = summary[key]
        if s:
            print(f"  {lbl:<20} n={s['n_tracks']}  median: "
                  f"{la} {s[f'{la}_median_dB']:+.2f} -> "
                  f"{lb} {s[f'{lb}_median_dB']:+.2f} dB  "
                  f"(median gain {s['median_gain_dB']:+.2f} dB)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
