"""11_smoke_train.py -- end-to-end smoke test of the distillation pipeline.

    decode -> teacher pseudo-labels -> student training -> inference -> audio

Deliberately small (N tracks x 30 s, a few hundred steps).  The point is NOT
quality.  It is to settle, in one run, three things that were still unverified:

  1. the data path (mp3 -> 44.1k stereo -> STFT -> 128 bands) works on the
     user's own DJ library, and how long it actually takes;
  2. the HTDemucs teacher forward pass runs on the GPU at a useful rate;
  3. **the causal U-Net actually learns the mask** -- loss goes down, and the
     weights it produces beat a random initialisation on held-out audio.
     That is the one question the whole design has been resting on.

Pseudo-labels are cached, so a later ablation run pays for the teacher only once.

Outputs
-------
results/smoke_cache/*.pt      cached pseudo-labels, reused by later runs
results/smoke/*.wav|mp3       the first audio this model has ever produced
results/smoke_report.json     timings, loss curve, SI-SDR
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from common import RESULTS, SR, load_demucs

# Some library filenames contain characters outside the active Windows console
# code page.  Logging must never abort an expensive teacher pass after the
# cache item has already been written; preserve the Unicode value in metadata
# and replace only an unrepresentable console glyph.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

# the target model lives in a module whose name starts with a digit
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("t09", _HERE / "09_target_model.py")
t09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t09)

LIB = Path(os.environ.get("STEM_AUDIO_LIBRARY", "D:/DJ_Music_Library"))
AUDIO_EXT = (".mp3", ".wav", ".flac", ".m4a", ".aiff", ".aif")
EXCLUDE_SUBSTR = "chills"          # the held-out validation track

CACHE = RESULTS / "smoke_cache"
OUT = RESULTS / "smoke"


# ---------------------------------------------------------------------------
# stage A -- data + teacher pseudo-labels
# ---------------------------------------------------------------------------
def find_tracks(seed: int, limit: int, holdout: int = 0):
    """The training pool, plus a pinned set of tracks that never enter it.

    The split matters more than it looks.  v2 and v3 both trained on the first
    220 tracks of this permutation and *validated* on the last 22, so the eight
    tracks the A/B harness scores were never fitted.  v4 widens the request to
    the whole library, which would have quietly pulled those eight songs into
    training and made the v3/v4 comparison meaningless.  A holdout that is
    carved out here -- and named in the report -- keeps every future A/B honest,
    at the cost of a couple of dozen tracks of training data.

    Duplicate *filenames* are dropped first (319 files resolve to 298 distinct
    songs -- the library keeps some tracks in two folders).  Without this the
    same song can land in both the training pool and the holdout: the two files
    would be labelled independently, the holdout name would still resolve, and
    an "unseen" track would in fact have been trained on.  Split integrity
    depends on song identity, and the filename is the only identity available.
    """
    if not LIB.exists():
        raise SystemExit(f"library not found: {LIB}")
    files = [p for p in LIB.rglob("*") if p.suffix.lower() in AUDIO_EXT]
    files = [p for p in files if EXCLUDE_SUBSTR not in p.name.lower()]
    files.sort()
    seen, uniq = set(), []
    for p in files:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    random.Random(seed).shuffle(uniq)
    sel = uniq[:limit]
    if not holdout:
        return sel, []
    return sel[:len(sel) - holdout], [p.name for p in sel[len(sel) - holdout:]]


def ffprobe_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 300.0


def decode_excerpt(path: Path, start_s: float, dur_s: float) -> torch.Tensor:
    """mp3 -> (2, N) float32 at 44.1 kHz, piped straight out of ffmpeg."""
    cmd = ["ffmpeg", "-v", "error",
           "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", str(path),
           "-ar", str(SR), "-ac", "2", "-f", "wav", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or len(p.stdout) < 2000:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[:200])
    x, _ = sf.read(io.BytesIO(p.stdout), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(x.T))


def teacher_vocals(model, mix: torch.Tensor, device: str) -> torch.Tensor:
    """Demucs-standard input normalisation around apply_model."""
    from demucs.apply import apply_model

    ref = mix.mean(0)
    norm = (mix - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, norm[None].to(device), device=device,
                          shifts=1, split=True, overlap=0.25, progress=False)[0]
    out = out.cpu() * ref.std() + ref.mean()
    return out[list(model.sources).index("vocals")]


def phase_sensitive_mask(X: torch.Tensor, V: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    """Best real [0,1] mask for projecting ``X`` onto complex target ``V``.

    Minimising ``|m X - V|^2`` for a real mask gives
    ``Re(V * conj(X)) / |X|^2``.  Clipping matches the product, whose sigmoid
    mask cannot amplify or flip phase.  Unlike a magnitude ratio, this target
    accounts for cancellation between vocals and accompaniment while changing
    nothing in the deployed inference graph.
    """
    return ((V * X.conj()).real / (X.abs().square() + eps)).clamp(0.0, 1.0)


def band_targets(mix: torch.Tensor, voc: torch.Tensor, Wa: torch.Tensor,
                 target_mask: str = "magnitude"):
    """(mixture, teacher vocals) -> (mix bands, target-mask bands), (2,128,T).

    ``magnitude`` is the historical soft mask ``|V| / (|V| + |A|)``.
    ``phase_sensitive`` is the clipped least-squares projection of the complex
    teacher vocal onto the mixture phase.  It is the target that the deployed
    real mixture mask can actually reproduce.
    """
    n = min(mix.shape[-1], voc.shape[-1])
    mix, voc = mix[..., :n], voc[..., :n]
    acc = mix - voc
    X = t09._stft(mix)
    V = t09._stft(voc)
    A = t09._stft(acc)
    if target_mask == "magnitude":
        m = V.abs() / (V.abs() + A.abs() + 1e-6)
    elif target_mask == "phase_sensitive":
        m = phase_sensitive_mask(X, V)
    else:
        raise ValueError(f"unknown target mask: {target_mask}")
    mix_band = torch.einsum("fb,cft->cbt", Wa, X.abs())
    mask_band = torch.einsum("fb,cft->cbt", Wa, m)
    return mix_band, mask_band


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------
EPS = 1e-3


def band_log_loss(x, m_v, m_a, t_v, t_a, w_v=None, accomp_weight: float = 1.0):
    """L1 between log-magnitudes of the reconstructed and target stems.

    A plain L1 on the *mask* has its optimum at the conditional median, and
    this mask's median is 0 in almost every time-frequency bin -- so "always
    output silence" is a strong attractor, and the first smoke run fell
    straight into it.  Taking logs before comparing gives every bin the same
    weight regardless of level, so a bin where the teacher hears vocals and the
    network outputs nothing costs the full dynamic range.  Silence stops being
    cheap.

    ``w_v`` optionally re-weights the *vocal* term per bin.  18_loss_attribution
    measured v4_best and found 86.8% of the vocal error sits in bins whose
    target mask is in (0, 0.5]: the network knows where to suppress, but
    under-shoots how much to keep.  Passing ``1 + lam * t_v`` makes the bins the
    teacher wants to retain cost more, which is the cheapest way to attack that
    without inventing an over-estimation attractor (bins with t_v = 0 keep
    weight 1, so silence is still not free).
    """
    lp = lambda z: (z + EPS).log()
    e_v = (lp(x * m_v) - lp(x * t_v)).abs()
    if w_v is not None:
        e_v = e_v * w_v
    return e_v.mean() + accomp_weight * F.l1_loss(lp(x * m_a), lp(x * t_a))


def band_rel_loss(x, m_v, m_a, t_v, t_a):
    """Energy-weighted relative L1 between predicted and target stem magnitudes.

    log-L1 is a *count* of bins, and 18_loss_attribution.py showed what that
    costs.  Measured on v4_best over the 27 validation excerpts:

      * bins with mixture magnitude < 0.01 are 39.6% of all bins and own 0.5%
        of the metric -- so the metric is not the silent-bin noise I first
        suspected, which is worth knowing;
      * but the loudest 10.3% of bins own only 23.4% of it, and the 28.7% in
        [1, 10) own 49.6%.  A count of bins spreads the loss roughly evenly
        across everything above x ~ 0.1, no matter how much energy is in it.

    SI-SDR is energy-weighted -- the loud bins decide it -- so a loss that
    weights every bin equally is optimising something else.  This term is the
    relative error of the reconstructed magnitude, which is the same shape as
    the ratio inside SI-SDR:

        sum |x*(m_hat - t)|  /  ( sum |x*t| + 0.02*sum|x| + eps )

    The 0.02*sum|x| floor keeps the denominator finite when a stem is silent in
    the crop (a non-vocal passage), where the target energy alone would divide
    by ~eps and make any leakage look catastrophic.

    Used as ``logl1 + w * band_rel_loss``: the log term still holds the
    silence-attractor line, the relative term spends the capacity where the
    energy is.  The validation metric stays log-L1 on purpose, so val curves
    remain comparable across a change of training loss.
    """
    tot = 0.0
    for m, t in ((m_v, t_v), (m_a, t_a)):
        p = (x * m).flatten(1)
        r = (x * t).flatten(1)
        num = (p - r).abs().sum(dim=1)
        den = r.abs().sum(dim=1) + 0.02 * x.flatten(1).abs().sum(dim=1) + EPS
        tot = tot + (num / den).mean()
    return 0.5 * tot


def focal_mask_loss(m_v, t_v, mid_weight: float = 4.0):
    """Directly calibrate partial-vocal masks without replacing log-L1.

    A historical mask-only L1 collapsed to zero, so this is deliberately an
    auxiliary term.  ``4*t*(1-t)`` peaks at the ambiguous 0.5 mask and is zero
    at the easy endpoints.  Loss attribution found that partial masks own
    86.8% of the current vocal error.
    """
    weight = 1.0 + mid_weight * (4.0 * t_v * (1.0 - t_v))
    return (weight * (m_v - t_v).abs()).mean()


def train_loss(args, x, m_v, m_a, ym):
    """(objective to backprop, log-L1 component) named by --loss.

    The second value is *always* the plain log-L1, because the logged training
    number and the validation metric must stay the same ruler across runs: once
    a weighting or a relative term is added, the raw objective is a different
    quantity and would not be comparable to any earlier run's column.
    """
    logl1 = band_log_loss(x, m_v, m_a, ym, 1.0 - ym)
    lam = getattr(args, "wmask_lambda", 0.0)
    if args.loss in ("logl1+wmask", "logl1+wmask+mask"):
        obj = band_log_loss(x, m_v, m_a, ym, 1.0 - ym,
                            w_v=1.0 + lam * ym,
                            accomp_weight=getattr(args, "accomp_weight", 1.0))
        if args.loss == "logl1+wmask+mask":
            obj = obj + args.mask_l1_weight * focal_mask_loss(
                m_v, ym, args.mask_mid_weight)
    elif args.loss == "logl1+rel":
        obj = logl1 + args.rel_weight * band_rel_loss(x, m_v, m_a, ym, 1.0 - ym)
    else:
        obj = logl1
    return obj, logl1


# ---------------------------------------------------------------------------
# input front end
# ---------------------------------------------------------------------------
# 17_quantize_check.py measured the problem this solves.  The network's input is
# a *linear* magnitude spectrum -- span ~250 with a median band at 0.3 of one
# INT8 step, so more than half the input bins round to zero on the PL, and the
# cost shows up as an 11 dB hit in the deployment arithmetic.  Compressing the
# input spreads the spectrum evenly over the 8-bit range and bought ~14 dB of
# input-domain SNR in that measurement.
#
# The transform has to be identical at training and inference time, hence one
# function used by both, and it is applied *after* the band filterbank and
# *before* the first conv.  The loss keeps using the raw magnitudes, so the
# thing being reconstructed never changes -- only what the network is fed.
#
# The spec travels *with the network* (``net.frontend``) rather than in a
# global, because the A/B harness holds two checkpoints at once and they may
# have been trained with different front ends.  Checkpoints written before this
# existed have no field and default to linear, which is what they were trained
# with.
DEFAULT_FRONTEND = ("linear", 1.0)
DEFAULT_MASK_MODE = "independent"


def frontend_spec(net) -> tuple:
    return getattr(net, "frontend", DEFAULT_FRONTEND)


def mask_mode_for(net) -> str:
    """How the accompaniment mask is obtained.

    Historical checkpoints predict vocal and accompaniment masks independently.
    The product, however, discards the accompaniment head and renders
    ``accompaniment = mixture - predicted_vocal``.  ``complement`` makes the
    training path match that product equation by using ``1 - vocal_mask`` for
    the accompaniment term as well.  Old checkpoints remain bit-compatible.
    """
    return getattr(net, "mask_mode", DEFAULT_MASK_MODE)


def band_layout_for(net) -> str:
    return getattr(net, "band_layout", t09.DEFAULT_BAND_LAYOUT)


def n_bands_for(net) -> int:
    """Frequency height carried by the checkpoint (128 for old weights)."""
    return int(getattr(net, "n_bands", t09.N_BANDS))


def apply_frontend(x: torch.Tensor, spec: tuple) -> torch.Tensor:
    """x -> network input.  Modes map [0, scale] onto [0, 1]."""
    mode, scale = spec
    if mode == "linear":
        return x
    y = (x / scale).clamp_min(0.0)
    if mode == "sqrt":
        y = y.sqrt()
    elif mode == "log":
        y = torch.log2(1.0 + y)          # log2 keeps the output in [0, 1]
    else:
        raise ValueError(f"unknown front end: {mode}")
    return torch.clamp(y, 0.0, 1.0)


def forward_masks(net, x):
    """Network -> (vocal mask, accompaniment mask), each (B, 2, 128, T) in [0,1]."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = net(apply_frontend(x, frontend_spec(net)))
    o = o.float()
    m_v = (o[:, 0:2] + 1.0) * 0.5
    if mask_mode_for(net) == "complement":
        return m_v, 1.0 - m_v
    return m_v, (o[:, 2:4] + 1.0) * 0.5


def causal_smooth_mask(mask: torch.Tensor, frames: int = 1) -> torch.Tensor:
    """Trailing moving average over mask frames, with no future look-ahead.

    ``frames=1`` is exactly the historical path.  Powers of two (2/4/8) are
    especially cheap in RTL because the divide is a right shift.  Replicating
    the first frame avoids a fade-in artefact during stream warm-up.
    """
    frames = int(frames)
    if frames <= 1:
        return mask
    shape = mask.shape
    flat = mask.reshape(-1, 1, shape[-1])
    flat = F.pad(flat, (frames - 1, 0), mode="replicate")
    kernel = torch.ones(1, 1, frames, device=mask.device,
                        dtype=mask.dtype) / frames
    return F.conv1d(flat, kernel).reshape(shape)



def _rand(a: float, b: float, g) -> float:
    return float(torch.rand(1, generator=g)) * (b - a) + a


def sample_batch(items, g, T, batch, aug=None):
    """Draw a batch of (mixture, soft-mask) crops, optionally augmented.

    Why augmentation, and why now
    -----------------------------
    v4 spent 2.5 h and 88 948 steps and finished *behind* the v3 weights it
    warm-started from (val log-L1 1.000 at init -> 1.032 at the end, best 0.9619
    at step 1500), while its *training* loss went 1.20 -> 0.71.  At the same time
    the held-out SI-SDR moved by -0.06 dB median.  So the extra data and the
    extra steps bought nothing, and the run ended with a 0.32 train/val gap.

    The training set is 272 excerpts -- one 60 s window per song.  Every crop the
    network ever sees comes from those 272 recordings, and a 0.53 M-parameter
    network has enough capacity to key on *which song* a crop came from (level,
    master-bus EQ, the specific arrangement) instead of on what a vocal sounds
    like.  More steps then buy memorisation, which is exactly what the curve
    shows.

    ``aug`` attacks that.  Two of the transforms are **exact** here and one is an
    approximation worth a flag:

    * ``gain_db`` (one gain for the whole crop) and ``eq_db`` (a smooth random
      per-band curve) scale the *mixture*.  Because the label is the ratio
      |V|/(|V|+|A|) and the same factor lands on both stems, the label does not
      move -- the pair stays a perfectly valid (input, label) pair, while the
      song's level and spectral fingerprint stop being a usable shortcut.
    * ``swap`` exchanges the two channels of input and label together (stereo
      augmentation); also exact.
    * ``stem_db`` (independent vocal/accompaniment gains) and ``remix`` (this
      excerpt's vocal over *another* excerpt's accompaniment) need the label
      recomputed, which is exact in the time domain but approximate in this
      magnitude-band representation: |V| + |A| overstates |V + A|.  Off by
      default for that reason.

    Inference is untouched -- augmentation only ever changes what the network is
    trained on, so the FPGA specification does not move.
    """
    aug = aug or {}
    eq_db = float(aug.get("eq_db") or 0.0)
    gain_db = float(aug.get("gain_db") or 0.0)
    stem_db = float(aug.get("stem_db") or 0.0)
    swap_p = float(aug.get("swap") or 0.0)
    remix_p = float(aug.get("remix") or 0.0)

    xs, ys = [], []
    for _ in range(batch):
        mix, mask = items[int(torch.randint(len(items), (1,), generator=g))]
        t = int(torch.randint(mix.shape[-1] - T + 1, (1,), generator=g))
        x, m = mix[:, :, t:t + T], mask[:, :, t:t + T]

        if remix_p and float(torch.rand(1, generator=g)) < remix_p:
            mix2, mask2 = items[int(torch.randint(len(items), (1,), generator=g))]
            t2 = int(torch.randint(mix2.shape[-1] - T + 1, (1,), generator=g))
            v = x * m
            a = mix2[:, :, t2:t2 + T] * (1.0 - mask2[:, :, t2:t2 + T])
        else:
            v, a = x * m, x * (1.0 - m)

        if stem_db:
            v = v * (10.0 ** (_rand(-stem_db, stem_db, g) / 20.0))
            a = a * (10.0 ** (_rand(-stem_db, stem_db, g) / 20.0))

        if stem_db or remix_p:
            x = v + a
            m = v / (x + EPS)

        if eq_db:
            f = int(x.shape[1])
            k = max(3, f // 16) | 1
            ctrl = torch.rand(f, generator=g).view(1, 1, f)
            ker = torch.ones(1, 1, k) / k
            ctrl = F.conv1d(ctrl, ker, padding=k // 2)[0, 0, :f]
            ctrl = (ctrl - ctrl.mean()) / (ctrl.std() + 1e-6)
            x = x * (10.0 ** (eq_db * ctrl.clamp(-2, 2) / 20.0)).view(1, f, 1)

        if gain_db:
            x = x * (10.0 ** (_rand(-gain_db, gain_db, g) / 20.0))

        if swap_p and float(torch.rand(1, generator=g)) < swap_p:
            x, m = x.flip(0), m.flip(0)

        xs.append(x)
        ys.append(m)
    return torch.stack(xs), torch.stack(ys)


def window_vocal_db(voc_w: torch.Tensor, mix_w: torch.Tensor) -> float:
    """How loud the teacher's vocals are inside one window, in dB.

    The vocal stem is by construction *part of* the mixture, so this ratio
    lives in roughly (-inf, 0] dB.  ~0 dB means vocals dominate the window;
    -25 dB means they are a whisper buried in the arrangement.
    """
    e_v = float(voc_w.float().pow(2).sum())
    e_m = float(mix_w.float().pow(2).sum()) + 1e-12
    return 10.0 * math.log10(max(e_v, 1e-20) / e_m)


def screening_meta(seg_sec, scan_sec, n_cand, min_vocal_db,
                   target_mask: str = "magnitude",
                   band_layout: str = t09.DEFAULT_BAND_LAYOUT,
                   n_bands: int = t09.N_BANDS) -> dict:
    return {"version": 2, "seg_s": seg_sec, "scan_s": scan_sec,
            "n_cand": n_cand, "min_vocal_db": min_vocal_db,
            **({"target_mask": target_mask}
               if target_mask != "magnitude" else {}),
            **({"band_layout": band_layout}
               if band_layout != t09.DEFAULT_BAND_LAYOUT else {}),
            **({"n_bands": int(n_bands)}
               if n_bands != t09.N_BANDS else {})}


def cache_srcs(files):
    """Filenames already labelled, read back from the cache files themselves.

    Only needed for caches built before ``_done.json`` existed.  The resume
    guard has to key on which *tracks* were labelled, and a cache that sifted
    tracks out holds fewer files than it processed paths, so the file count is
    not a usable stand-in.
    """
    names = set()
    for f in files:
        try:
            names.add(torch.load(f, map_location="cpu",
                                 weights_only=False)["src"])
        except Exception:
            pass
    return names


def build_cache(paths, device: str, seg_sec: float, rebuild: bool, seed: int = 0,
                scan_sec: float = 90.0, n_cand: int = 5,
                min_vocal_db: float = -60.0, crops_per_track: int = 1,
                target_mask: str = "magnitude",
                band_layout: str = t09.DEFAULT_BAND_LAYOUT,
                n_bands: int = t09.N_BANDS):
    """Teacher pseudo-labels, with the *window* chosen rather than gambled.

    The v2 run picked each track's 30 s excerpt uniformly at random from the
    middle of the song.  An A/B on eight held-out tracks then showed that five
    of the eight excerpts contained no usable vocals at all (teacher vocals
    -21..-43 dBFS against a -10 dBFS mixture) -- in house/EDM the vocals occupy
    a small fraction of the timeline, so a large share of the compute was
    teaching the model to output silence where there was nothing to remove, and
    feeding the ``mask -> 0`` attractor documented in 12_smoke_diag.py.

    So: decode a longer scan, run the teacher **once** over it, and keep the
    best 30 s sub-window according to vocal-to-mixture energy.  The teacher's
    cost is paid on the scan, not per candidate, so screening five windows
    costs one forward pass over 90 s instead of five over 30 s.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + 1)
    Wa = torch.from_numpy(t09.make_analysis_matrix(n_bands, layout=band_layout))
    model = load_demucs("htdemucs").to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())

    want = screening_meta(seg_sec, scan_sec, n_cand, min_vocal_db, target_mask,
                          band_layout, n_bands)
    # Recorded only when it differs from the historical default.  Adding the key
    # unconditionally would make every cache already on disk -- v3, v4, the
    # probes -- fail the meta check below, and force a needless re-screen.
    if crops_per_track != 1:
        want["crops_per_track"] = crops_per_track
    meta_path = CACHE / "_meta.json"
    got = None
    if meta_path.exists():
        try:
            got = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            got = None

    # With K > 1 the cache holds K files per song, so "number of files" is no
    # longer "number of songs done" and the resume counter has to key on the
    # track index instead.  K == 1 keeps the original file naming and counter
    # verbatim, so an old cache and an old --tag behave exactly as before.
    k_mode = crops_per_track > 1
    prog_path = CACHE / "_done.json"
    prog = []
    done_tracks = set()
    if prog_path.exists():
        try:
            prog = json.loads(prog_path.read_text(encoding="utf-8"))
            done_tracks = {int(r["t"]) for r in prog}
        except Exception:
            prog, done_tracks = [], set()

    done = sorted(CACHE.glob("*.pt"))
    base = 0
    if done and not rebuild:
        if got is None:
            print("  WARNING: this cache has no '_meta.json' -- it predates "
                  "window screening, so its excerpts are whatever the old run "
                  "picked.  Pass --rebuild to re-screen.")
        elif any(got.get(k) != v for k, v in want.items()):
            raise SystemExit(
                f"\n  cache {CACHE} was built with\n    {got}\n"
                f"  but this run asks for\n    {want}\n"
                f"  use a different --tag, or pass --rebuild.\n")
        if k_mode:
            if len(done_tracks) >= len(paths):
                print(f"  cache already holds {len(done)} crops from "
                      f"{len(done_tracks)} tracks -- reusing "
                      f"(pass --rebuild to regenerate)")
                return [], {"cached_entries": len(done),
                            "cached_tracks": len(done_tracks)}
            if done_tracks:
                print(f"  cache holds {len(done)} crops from "
                      f"{len(done_tracks)}/{len(paths)} tracks -- labelling the "
                      f"remaining {len(paths) - len(done_tracks)}")
        else:
            # Resume by *filename*, never by file count.  A previous run that
            # sifted tracks out writes fewer files than it processed paths, so
            # "skip the first len(files) tracks" re-labels -- and silently
            # duplicates -- the tail of the list.  That is exactly how
            # v4_cache ended up holding two tracks twice, so both the guard and
            # the remainder are keyed on the names already on disk.
            done_names = {r["file"] for r in prog} or cache_srcs(done)
            if len(done_names) >= len(paths):
                print(f"  cache already holds {len(done)} entries for "
                      f"{len(done_names)} tracks -- reusing "
                      f"(pass --rebuild to regenerate)")
                return [], {"cached_entries": len(done)}
            paths = [p for p in paths if p.name not in done_names]
            base = len(done)      # naming carries on where the files stop
            print(f"  cache holds {len(done)} entries for "
                  f"{len(done_names)} of {len(done_names) + len(paths)} tracks "
                  f"-- labelling the remaining {len(paths)}")

    scan_sec = max(float(scan_sec), float(seg_sec))
    W = int(round(seg_sec * SR))
    print(f"  teacher: HTDemucs, {n_par/1e6:.2f} M params, on {device}")
    _keep = ("the best" if not k_mode else
             f"the top {crops_per_track} non-overlapping")
    print(f"  window screening: decode {scan_sec:.0f}s -> teacher once -> keep "
          f"{_keep} of {n_cand} x {seg_sec:.0f}s windows by vocal energy"
          + (f"   [crops/track={crops_per_track}, teacher cost unchanged]"
             if k_mode else ""))
    print(f"  {'#':>4}  {'file':<38}{'dec':>7}{'teach':>7}{'spec':>7}"
          f"{'kHz':>6}{'voc dB':>8}{'off s':>7}{'cands':>18}")
    rows, t_dec, t_teach, t_spec = [], 0.0, 0.0, 0.0
    scan_audio = 0.0
    sifted = 0

    def _mark_done(ti: int, n_crop: int):
        """Checkpoint one song's outcome, so `--rebuild`-free resumption works."""
        prog.append({"t": ti, "n": n_crop, "file": paths[ti].name})
        prog_path.write_text(json.dumps(prog, ensure_ascii=False), encoding="utf-8")

    for i, p in enumerate(paths):
        if k_mode and i in done_tracks:
            continue
        t0 = time.perf_counter()
        try:
            total = ffprobe_duration(p)
            start = rng.uniform(0.15, 0.75) * max(0.0, total - scan_sec)
            scan = decode_excerpt(p, start, scan_sec)
        except Exception as exc:
            print(f"  {i:>4}  {p.name[:38]:<38} SKIP ({type(exc).__name__})")
            continue
        t1 = time.perf_counter()
        voc = teacher_vocals(model, scan, device)
        t2 = time.perf_counter()
        n = min(scan.shape[-1], voc.shape[-1])
        scan, voc = scan[:, :n], voc[:, :n]
        scan_audio += n / SR
        if n <= W:
            offs = [0]
        else:
            offs = sorted({int(round(x)) for x in
                           np.linspace(0, n - W, max(1, n_cand))})
        scores = [window_vocal_db(voc[:, o:o + W], scan[:, o:o + W])
                  for o in offs]
        # Greedy by score, rejecting any window that overlaps one already taken.
        # The first pick is the plain argmax, so rank 0 is exactly the window the
        # K=1 path has always cached.
        order = sorted(range(len(offs)), key=lambda j: -scores[j])
        picks = []
        for j in order:
            if len(picks) >= crops_per_track:
                break
            if all(abs(offs[j] - offs[q]) >= W for q in picks):
                picks.append(j)
        if not picks:
            picks = [order[0]]
        k = picks[0]
        best_off, best_db = offs[k], scores[k]
        if best_db < min_vocal_db:
            sifted += 1
            print(f"  {i:>4}  {p.name[:38]:<38}{t1-t0:>6.2f}s{t2-t1:>6.2f}s"
                  f"{'':>7}{'':>6}{best_db:>8.1f}{'':>7}  SIFTED OUT")
            if k_mode:
                _mark_done(i, 0)
            continue
        added = []
        for r, j in enumerate(picks):
            off = offs[j]
            mb, mm = band_targets(scan[:, off:off + W], voc[:, off:off + W],
                                  Wa, target_mask=target_mask)
            if k_mode:
                idx, fname = i * 100 + r, f"t{i:03d}_k{r}.pt"
            else:
                idx, fname = base + len(rows), f"{base + len(rows):03d}.pt"
            torch.save({"mix": mb, "mask": mm, "src": p.name, "dur": W / SR,
                        "target_mask": target_mask,
                        "band_layout": band_layout,
                        "n_bands": int(n_bands),
                        "vocal_db": round(best_db, 2), "k": r,
                        "offset_s": round(off / SR, 2),
                        "crop_db": round(scores[j], 2),
                        "scan_s": round(n / SR, 2),
                        "cand_db": [round(s, 2) for s in scores]},
                       CACHE / fname)
            added.append(off / SR)
            rows.append({"idx": idx, "file": p.name, "dur": round(W / SR, 2),
                         "vocal_db": round(best_db, 2), "k": r,
                         "offset_s": round(off / SR, 2)})
        if k_mode:
            _mark_done(i, len(picks))
        t3 = time.perf_counter()
        t_dec += t1 - t0
        t_teach += t2 - t1
        t_spec += t3 - t2
        print(f"  {i:>4}  {p.name[:38]:<38}{t1-t0:>6.2f}s{t2-t1:>6.2f}s"
              f"{t3-t2:>6.2f}s{(n/SR)/max(t2-t1,1e-6):>6.1f}{best_db:>8.1f}"
              f"{best_off/SR:>7.1f}   " +
              " ".join(f"{s:.0f}" for s in scores)
              + (f"  -> {len(picks)}c @ " + ",".join(f"{o:.0f}s" for o in added)
                 if k_mode else ""), flush=True)
    if not rows and not done:
        raise SystemExit("no usable tracks -- check the library and ffmpeg")
    meta_path.write_text(json.dumps(want, indent=2), encoding="utf-8")

    # the whole point of screening is visible in these two numbers: how much of
    # the labelled audio actually carries vocals
    every = []
    for f in sorted(CACHE.glob("*.pt")):
        try:
            every.append(torch.load(f, map_location="cpu",
                                    weights_only=False).get("vocal_db"))
        except Exception:
            pass
    every = [v for v in every if v is not None]
    stats = {}
    if every:
        s = sorted(every)
        stats = {"vocal_db_min": round(s[0], 1),
                 "vocal_db_p25": round(s[len(s) // 4], 1),
                 "vocal_db_median": round(s[len(s) // 2], 1),
                 "vocal_db_max": round(s[-1], 1),
                 "frac_above_-12dB": round(sum(v > -12 for v in every)
                                           / len(every), 3),
                 "frac_above_-18dB": round(sum(v > -18 for v in every)
                                           / len(every), 3)}
        print(f"  cached excerpts: {len(every)}, vocal-to-mixture energy "
              f"median {stats['vocal_db_median']:+.1f} dB, "
              f"{stats['frac_above_-12dB']*100:.0f}% above -12 dB")
    return rows, {"decode_s": t_dec, "teacher_s": t_teach,
                  "spectral_s": t_spec, "entries": len(rows),
                  "scan_audio_s": round(scan_audio, 1),
                  "sifted_out": sifted, **stats}


# ---------------------------------------------------------------------------
# stage B -- train the student
# ---------------------------------------------------------------------------
def load_cache(cache_dir: Path | None = None):
    """Cache entries as ``(song, rank, mix, mask)``, sorted by ``(song, rank)``.

    ``song`` is the index the track had in the screened list and ``rank`` says
    which of its K windows this file is; rank 0 is that song's best window.  A
    K == 1 cache uses plain ``NNN.pt`` names and reads back as ``(song, 0)``, so
    every caller sees the same shape whether the cache holds one window per song
    or K.
    """
    root = CACHE if cache_dir is None else Path(cache_dir)
    if not root.exists():
        raise FileNotFoundError(f"cache directory not found: {root}")
    out = []
    for f in sorted(root.glob("*.pt")):
        stem = f.stem
        if stem.startswith("t") and "_k" in stem:
            t_part, k_part = stem[1:].split("_k")
            song, rank = int(t_part), int(k_part)
        else:
            song, rank = int(stem), 0
        d = torch.load(f, map_location="cpu", weights_only=False)
        out.append((song, rank, d["mix"].float(), d["mask"].float()))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def save_ckpt(path: Path, net, args, n_par: int, step=None, best_val=None,
              state_dict=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    bv = None if best_val is None or best_val == float("inf") else best_val
    torch.save({"model": state_dict if state_dict is not None
                else net.state_dict(),
                "params": n_par, "crop": args.crop,
                "steps": args.steps, "lr": args.lr, "step": step,
                "frontend": tuple(frontend_spec(net)),
                "mask_mode": mask_mode_for(net),
                "band_layout": band_layout_for(net),
                "n_bands": n_bands_for(net),
                "bottleneck_blocks": len(getattr(net, "bott_blocks", ())),
                "temporal_dilations": tuple(
                    getattr(net, "temporal_dilations", ())),
                "best_val": bv}, path)


def scheduled_lr(args, step: int, elapsed: float) -> float:
    """Linear warmup, then cosine decay, with the cosine driven by the clock.

    Every run up to v3 used a constant learning rate, which is right for a
    hundred-step probe and wrong for a long one: a fixed lr keeps the weights
    bouncing around the minimum instead of settling into it.  The catch is that
    the number of steps a run will complete is not known when it starts -- it
    depends on the data size and on how loaded the machine is.  So when
    --time-budget-s is set, the cosine is a function of *elapsed wall time*
    rather than of step count, and the run anneals properly no matter how many
    steps fit.
    """
    if args.lr_schedule != "cosine":
        return args.lr
    if args.warmup_steps and step < args.warmup_steps:
        return args.lr * step / float(args.warmup_steps)
    if args.time_budget_s:
        p = min(max(elapsed / args.time_budget_s, 0.0), 1.0)
    else:
        span = max(args.steps - args.warmup_steps, 1)
        p = min(max((step - args.warmup_steps) / span, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))


def train(net, device: str, items, val_items, args, ckpt: Path):
    """Train until `--steps` or until the wall-clock budget runs out.

    The budget exists because a half-hour of machine time is a normal way to
    work on this project, and "how many steps fit" depends on the data size,
    which varies run to run.  Periodic checkpoints plus best-val tracking mean
    an interrupted run still leaves a usable model behind.
    """
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    T = args.crop
    aug = {"eq_db": args.aug_eq_db, "gain_db": args.aug_gain_db,
           "stem_db": args.aug_stem_db, "swap": args.aug_swap,
           "remix": args.aug_remix}
    if any(aug.values()):
        print("  augmentation: " + ", ".join(f"{k}={v}" for k, v in aug.items()
                                             if v))
    n_par = sum(p.numel() for p in net.parameters())
    curve = []
    best = {"val": float("inf"), "step": 0, "sd": None}
    print(f"  {'step':>7}{'lr':>10}{'train':>10}{'val':>10}{'sec':>8}{'aud-s/s':>10}")
    net.train()
    t0 = time.perf_counter()
    step = 0
    lr = args.lr
    for step in range(1, args.steps + 1):
        elapsed = time.perf_counter() - t0
        lr = scheduled_lr(args, step, elapsed)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, ym = sample_batch(items, g, T, args.batch, aug)
        x, ym = x.to(device), ym.to(device)

        opt.zero_grad(set_to_none=True)
        m_v, m_a = forward_masks(net, x)
        loss, logl1 = train_loss(args, x, m_v, m_a, ym)
        loss.backward()
        opt.step()

        if args.ckpt_every and step % args.ckpt_every == 0:
            save_ckpt(ckpt, net, args, n_par, step=step, best_val=best["val"])

        if step % args.log_every == 0 or step == 1:
            v = evaluate(net, val_items, device, T, args.batch,
                         args.val_batches) if val_items else float("nan")
            rate = args.batch * args.crop * step * t09.HOP / SR / elapsed
            curve.append({"step": step, "lr": round(lr, 7),
                          "train_logL1": round(float(logl1), 5),
                          "train_objective": round(float(loss), 5),
                          "val_logL1": round(v, 5) if v == v else None,
                          "wall_s": round(elapsed, 1)})
            print(f"  {step:>7}{lr:>10.2e}{float(logl1):>10.4f}{v:>10.4f}"
                  f"{elapsed:>8.1f}{rate:>10.1f}", flush=True)
            if v == v and v < best["val"]:
                best.update(val=v, step=step,
                            sd={k: t.detach().to("cpu", copy=True)
                                for k, t in net.state_dict().items()})
        if args.time_budget_s and elapsed > args.time_budget_s:
            print(f"  time budget {args.time_budget_s:.0f}s reached at step {step}")
            break
    wall = time.perf_counter() - t0
    if best["sd"] is not None:
        print(f"  best val {best['val']:.4f} at step {best['step']}")
    return curve, wall, best, step


@torch.no_grad()
def evaluate(net, items, device: str, T: int, batch: int, n_batches: int = 8,
             seed: int = 1234) -> float:
    """Val log-L1 over a *fixed* set of crops.

    The generator is deliberately re-seeded on every call.  The first version
    reused the training generator, so each evaluation drew different crops and
    the curve swung by +/-0.15 -- "best val" was recorded as 1.124 at step 6200,
    yet the same weights re-measured 1.296.  With a fixed set the numbers are
    comparable across steps, so best-val selection and early stopping mean
    something.  More batches also shrink the standard error of the estimate.
    """
    gen = torch.Generator().manual_seed(seed)
    net.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, ym = sample_batch(items, gen, T, batch)
        x, ym = x.to(device), ym.to(device)
        m_v, m_a = forward_masks(net, x)
        tot += float(band_log_loss(x, m_v, m_a, ym, 1.0 - ym))
    net.train()
    return tot / n_batches


@torch.no_grad()
def mask_health(net, items, device: str, T: int, batch: int, g) -> dict:
    """Detect the collapse mode directly: is the predicted mask a constant?

    The first smoke run looked healthy by loss and was in fact outputting
    silence, so the report has to watch the mask itself, not only the loss.
    """
    net.eval()
    x, ym = sample_batch(items, g, T, batch)
    m_v, _ = forward_masks(net, x.to(device))
    m_v = m_v.cpu()
    net.train()
    return {
        "target_mask_mean": round(float(ym.mean()), 4),
        "pred_mask_mean": round(float(m_v.mean()), 4),
        "pred_mask_std": round(float(m_v.std()), 4),
        "pred_pred_gt_0.5_frac": round(float((m_v > 0.5).float().mean()), 4),
        "target_gt_0.5_frac": round(float((ym > 0.5).float().mean()), 4),
    }


# ---------------------------------------------------------------------------
# stage C -- inference on held-out audio
# ---------------------------------------------------------------------------
def _stft_dev(x: torch.Tensor) -> torch.Tensor:
    """t09._stft builds the window on the CPU, which breaks on CUDA input."""
    win = torch.hann_window(t09.N_FFT, device=x.device)
    return torch.stft(x, t09.N_FFT, t09.HOP, window=win, center=True,
                      return_complex=True)


def _istft_dev(X: torch.Tensor, length: int) -> torch.Tensor:
    win = torch.hann_window(t09.N_FFT, device=X.device)
    return torch.istft(X, t09.N_FFT, t09.HOP, window=win, center=True,
                       length=length)


def _db(x: torch.Tensor) -> float:
    return float(20 * torch.log10(x.float().pow(2).mean().sqrt() + 1e-12))


def lf_kill_band_for(hz: float,
                     layout: str = t09.DEFAULT_BAND_LAYOUT,
                     n_bands: int = t09.N_BANDS) -> int:
    """Band index for a low-frequency-protection cutoff in Hz (0 = disabled).

    The band grid is geometric, so this is the last band whose top edge is still
    below ``hz``.  Kept next to separate() because the two must agree: the PL
    implements exactly this comparison.
    """
    if hz <= 0:
        return 0
    if layout == "mel_unique":
        return int(np.searchsorted(t09.band_centers(n_bands, layout),
                                   hz, side="left"))
    edges = t09.band_edges(n_bands)
    k = 0
    while k + 1 < n_bands and edges[k + 1] <= hz:
        k += 1
    return k + 1


def separate(net, mix: torch.Tensor, Wa, Gs, device: str,
             lf_kill_band: int = 0, mask_smooth_frames: int = 1,
             vocal_mask_gain: float = 1.0):
    """STFT -> 128 bands -> U-Net -> mask -> iSTFT.  Mask = (tanh + 1) / 2.

    ``lf_kill_band`` zeroes the vocal mask below a band index, which is the
    low-frequency protection the FPGA spec already called for.  20_mask_postproc
    measured it: on the pinned 24-track holdout (14 scorable) it is a paired
    gain of +0.74 dB on the vocal and +0.37 dB on the accompaniment, improving
    both on 12 of 14 tracks, and the optimum is a cutoff near 250 Hz.  Past
    ~350 Hz it starts deleting real vocal content and the vocal score collapses
    (3.52 -> 1.69 dB).

    It is not cosmetic.  19_filterbank_audit showed that the 49 lowest bands have
    no STFT bin inside them, so the network sees exactly zero in those input
    channels for every frame of every song; the *synthesis* matrix still routes
    sub-500 Hz bins through those bands, so the mask applied there is the
    network's response to a zero input -- a constant.  Measured, that constant
    averages 0.26 below 700 Hz, i.e. the button press would subtract a quarter of
    the kick and bass no matter what the song is doing.  Forcing the mask to zero
    there is both the correct behaviour and a free fix: on the PL it is a
    band-index comparator.

    Default 0 keeps every earlier measurement reproducible; callers that want the
    shipping behaviour pass the band for their cutoff (see hz_to_band in
    20_mask_postproc.py).
    """
    n = mix.shape[-1]
    X = _stft_dev(mix.to(device))
    bands = torch.einsum("fb,cft->cbt", Wa.to(device), X.abs())
    T = bands.shape[-1]
    pad = (-T) % 8
    if pad:
        bands = F.pad(bands, (0, pad))
    with torch.no_grad():
        out = net(apply_frontend(bands[None], frontend_spec(net)))[0]
    if pad:
        out = out[..., :T]
    m = ((out[0:2].float() + 1.0) * 0.5).clamp(0.0, 1.0)
    m = causal_smooth_mask(m, mask_smooth_frames)
    if vocal_mask_gain != 1.0:
        m = (m * vocal_mask_gain).clamp(0.0, 1.0)
    if lf_kill_band:
        m = m.clone()
        m[:, :lf_kill_band] = 0.0
    mask_bin = torch.einsum("fb,cbt->cft", Gs.to(device), m).clamp(0.0, 1.0)
    voc = _istft_dev(X * mask_bin, n).cpu()
    return voc, mix.cpu() - voc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=30)
    ap.add_argument("--holdout", type=int, default=0,
                    help="the last N of the selected tracks are removed from the "
                         "training pool and recorded in the report; the A/B "
                         "harness scores them, so they must never be trained on")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256,
                    help="training crop in frames; must be a multiple of 8")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--in-frontend", default="linear",
                    choices=["linear", "sqrt", "log"],
                    help="compression applied to the input spectrum before the "
                         "first conv.  17_quantize_check.py showed the linear "
                         "spectrum loses half its bins to INT8 rounding; sqrt "
                         "recovers ~14 dB of input-domain SNR for one square "
                         "root in the PL.  The same transform is applied at "
                         "inference and is stored in the checkpoint.")
    ap.add_argument("--in-scale", type=float, default=1.0,
                    help="reference band magnitude mapped to 1.0 by the front "
                         "end (linear: unused)")
    ap.add_argument("--mask-mode", choices=("independent", "complement"),
                    default="independent",
                    help="independent reproduces historical two-head training; "
                         "complement trains the accompaniment as 1-vocal_mask, "
                         "matching the actual product inference equation")
    ap.add_argument("--target-mask", choices=("magnitude", "phase_sensitive"),
                    default="magnitude",
                    help="teacher label stored in a newly built cache. "
                         "phase_sensitive is the best real [0,1] projection "
                         "onto the complex teacher and adds no inference cost")
    ap.add_argument("--band-layout", choices=("legacy_log", "mel_unique"),
                    default="legacy_log",
                    help="fixed 513-to-128 filterbank. mel_unique assigns every "
                         "band a distinct FFT-bin centre and removes the 49 "
                         "empty columns in the historical layout")
    ap.add_argument("--n-bands", type=int, default=t09.N_BANDS,
                    help="frequency bands presented to the U-Net. Must be a "
                         "multiple of 8; old checkpoints default to 128")
    ap.add_argument("--bottleneck-blocks", type=int, default=0,
                    help="zero-initialised causal 3x3 residual blocks at the "
                         "16-band/1/8-rate bottleneck")
    ap.add_argument("--temporal-dilations", default="",
                    help="comma-separated dilations for zero-initialised "
                         "depthwise 1x3 temporal blocks at the bottleneck; "
                         "for example 1,2,4 adds about 650 ms of history")
    ap.add_argument("--lr-schedule", choices=("const", "cosine"), default="const",
                    help="'cosine' = linear warmup then cosine decay to "
                         "--min-lr.  With --time-budget-s the decay follows the "
                         "clock, so a long run anneals however many steps fit.")
    ap.add_argument("--warmup-steps", type=int, default=500,
                    help="linear warmup length for --lr-schedule cosine")
    ap.add_argument("--min-lr", type=float, default=1e-5,
                    help="floor of the cosine decay")
    ap.add_argument("--seg", type=float, default=30.0, help="seconds per track")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--tag", default="smoke",
                    help="names every output: <tag>_cache/, <tag>/, "
                         "<tag>_model.pt, <tag>_report.json")
    ap.add_argument("--time-budget-s", type=float, default=0.0,
                    help="stop training once this many wall seconds elapse "
                         "(0 = run the full --steps)")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="write a checkpoint every N steps (0 = off)")
    ap.add_argument("--use-best", action="store_true",
                    help="run the listening test with the best-val checkpoint "
                         "instead of the final one")
    ap.add_argument("--val-batches", type=int, default=8,
                    help="validation batches per evaluation (each is --batch "
                         "crops); higher = less noisy val curve")
    ap.add_argument("--init-from", default=None,
                    help="warm-start the student from this checkpoint; combine "
                         "with --skip-train to score an existing model")
    ap.add_argument("--cache-dir", default=None,
                    help="reuse another run's pseudo-label cache instead of "
                         "<tag>_cache (the teacher pass is the expensive part)")
    ap.add_argument("--extra-cache-dir", action="append", default=[],
                    help="append a ground-truth cache to the training split only. "
                         "May be passed more than once.  The original cache still "
                         "defines validation, so its historical ruler is unchanged")
    ap.add_argument("--extra-cache-repeat", type=int, default=1,
                    help="repeat each extra-cache crop this many times in the "
                         "training sampler (useful for small, high-value true-stem "
                         "sets; default 1)")
    ap.add_argument("--base-cache-repeat", type=int, default=1,
                    help="repeat the original pseudo-label training pool before "
                         "adding true-stem caches.  This changes sampling weight "
                         "without copying tensor storage (default 1)")
    ap.add_argument("--scan-s", type=float, default=90.0,
                    help="seconds decoded per track for window screening.  The "
                         "teacher runs once over this, then the best --seg "
                         "sub-window is kept.  Costs --scan-s/--seg times the "
                         "teacher compute but removes the instrumental-only "
                         "excerpts that dominated v2.  Pass 0 for the old "
                         "single-window behaviour.")
    ap.add_argument("--n-cand", type=int, default=5,
                    help="candidate windows scored per track during screening "
                         "(1 = no screening, take the middle)")
    ap.add_argument("--crops-per-track", type=int, default=1,
                    help="keep this many non-overlapping windows per track "
                         "instead of only the best one.  The teacher already ran "
                         "over the whole scan, so K windows cost the same wall "
                         "clock as one and multiply the labelled audio by K -- "
                         "the one axis v4 never moved (274 songs -> 274 crops). "
                         "Validation keeps scoring rank 0 only, so the val ruler "
                         "is unchanged.  Default 1 = the historical behaviour, "
                         "byte for byte.")
    ap.add_argument("--min-vocal-db", type=float, default=-60.0,
                    help="drop a track whose best window is quieter than this "
                         "(-60 dB keeps everything)")
    ap.add_argument("--aug-eq-db", type=float, default=0.0,
                    help="training-time only: +-dB of smooth random per-band EQ "
                         "on the mixture.  Exact -- the soft-mask label is a "
                         "ratio and does not move.  Breaks the per-song "
                         "mastering fingerprint the model otherwise memorises.")
    ap.add_argument("--aug-gain-db", type=float, default=0.0,
                    help="training-time only: +-dB of random per-crop gain "
                         "(exact, same reason as --aug-eq-db)")
    ap.add_argument("--aug-swap", type=float, default=0.0,
                    help="training-time only: probability of swapping L/R "
                         "(exact)")
    ap.add_argument("--aug-stem-db", type=float, default=0.0,
                    help="training-time only: +-dB of independent vocal / "
                         "accompaniment gain.  Needs the label recomputed, so it "
                         "is approximate in the magnitude-band domain (|V|+|A| "
                         "overstates |V+A|) -- off by default")
    ap.add_argument("--aug-remix", type=float, default=0.0,
                    help="training-time only: probability of replacing this "
                         "crop's accompaniment with another song's.  Same domain "
                         "caveat as --aug-stem-db")
    ap.add_argument("--loss", default="logl1",
                    choices=("logl1", "logl1+rel", "logl1+wmask",
                             "logl1+wmask+mask"),
                    help="training objective.  logl1 = the loss every run up to "
                         "v4 used.  logl1+rel adds an energy-weighted relative "
                         "magnitude term, because log-L1 counts bins and the "
                         "loudest 10%% of them own only 23%% of the metric "
                         "while owning most of the energy (18_loss_attribution). "
                         "logl1+wmask keeps log-L1 but weights the vocal term by "
                         "1 + lambda*target, attacking the measured "
                         "under-estimation in (0, 0.5] target-mask bins. "
                         "The validation metric stays log-L1 either way, so "
                         "val curves remain comparable.")
    ap.add_argument("--wmask-lambda", type=float, default=2.0,
                    help="strength of the target-mask weighting used by "
                         "--loss logl1+wmask; 0 disables it")
    ap.add_argument("--accomp-weight", type=float, default=1.0,
                    help="accompaniment term weight in the training objective; "
                         "validation remains the historical equal-weight log-L1")
    ap.add_argument("--mask-l1-weight", type=float, default=0.25,
                    help="auxiliary direct-mask calibration weight for "
                         "--loss logl1+wmask+mask")
    ap.add_argument("--mask-mid-weight", type=float, default=4.0,
                    help="extra focus on partial masks in the calibration term")
    ap.add_argument("--rel-weight", type=float, default=1.0,
                    help="weight of the relative term in --loss logl1+rel")
    args = ap.parse_args()
    temporal_dilations = tuple(
        int(x) for x in args.temporal_dilations.split(",") if x.strip())
    if any(d < 1 for d in temporal_dilations):
        raise SystemExit("all --temporal-dilations must be >= 1")
    if args.n_bands < 8 or args.n_bands % 8:
        raise SystemExit("--n-bands must be a positive multiple of 8")
    if args.extra_cache_repeat < 1:
        raise SystemExit("--extra-cache-repeat must be >= 1")
    if args.base_cache_repeat < 1:
        raise SystemExit("--base-cache-repeat must be >= 1")

    if args.target_mask == "phase_sensitive" and (
            args.aug_stem_db or args.aug_remix):
        raise SystemExit("phase-sensitive labels cannot be recomputed from the "
                         "band cache after stem/remix augmentation; leave "
                         "--aug-stem-db and --aug-remix at zero")

    global CACHE, OUT
    CACHE = Path(args.cache_dir) if args.cache_dir else RESULTS / f"{args.tag}_cache"
    OUT = RESULTS / f"{args.tag}"
    CKPT = RESULTS / f"{args.tag}_model.pt"
    REPORT = RESULTS / f"{args.tag}_report.json"

    assert args.crop % 8 == 0, "crop must be a multiple of 8 (3 time downsamples)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 78)
    print("SMOKE TEST -- distillation pipeline end to end")
    print("=" * 78)
    print(f"device {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # ---- stage A ---------------------------------------------------------
    print("\n[1/3] data + teacher pseudo-labels")
    t_a0 = time.perf_counter()
    paths, holdout_names = find_tracks(args.seed, args.tracks, args.holdout)
    print(f"  library scan: {len(paths)} training tracks from {LIB}"
          + (f", {len(holdout_names)} held out for scoring" if holdout_names
             else ""))
    rows, timing = build_cache(paths, device, args.seg, args.rebuild, args.seed,
                               scan_sec=args.scan_s, n_cand=args.n_cand,
                               min_vocal_db=args.min_vocal_db,
                               crops_per_track=args.crops_per_track,
                               target_mask=args.target_mask,
                               band_layout=args.band_layout,
                               n_bands=args.n_bands)
    audio_s = sum(r["dur"] for r in rows)
    wall_a = time.perf_counter() - t_a0
    # With K > 1 the teacher pass is unchanged while the labelled audio triples,
    # so "labelled audio-s per wall-s" would look 3x better for free.  Report the
    # rate the teacher actually ran at (scan audio per wall second) -- the same
    # quantity 04_throughput measured, 4.9 audio-s/s with screening on.
    scan_total = timing.get("scan_audio_s") or 0.0
    if rows:
        print(f"  total {audio_s:.0f} labelled audio-s from "
              f"{scan_total or audio_s:.0f} scan audio-s in {wall_a:.1f} wall-s "
              f"-> {(scan_total or audio_s)/wall_a:.1f} audio-s/s (teacher)")
    else:
        audio_s = sum(d["dur"] for d in
                      (torch.load(f, weights_only=False) for f in
                       sorted(CACHE.glob("*.pt"))))

    # ---- stage B ---------------------------------------------------------
    print("\n[2/3] student training")
    entries = load_cache()
    # Validation stays exactly what it has always been: the best window (rank 0)
    # of the last 10% of *songs*, in the same order.  Scoring all K windows
    # instead would change the ruler, and this run's val log-L1 has to stay
    # comparable with v1-v4 and the probes.
    songs = sorted({e[0] for e in entries})
    n_val = max(1, round(len(songs) * 0.1))
    val_songs = set(songs[-n_val:])
    train_items = [(e[2], e[3]) for e in entries if e[0] not in val_songs]
    val_items = [(e[2], e[3]) for e in entries
                 if e[0] in val_songs and e[1] == 0]
    base_train_crops = len(train_items)
    if args.base_cache_repeat > 1:
        train_items *= args.base_cache_repeat
        print(f"  base pseudo labels: {base_train_crops} crops x "
              f"{args.base_cache_repeat} = {len(train_items)} sampler entries")
    extra_cache_stats = []
    for cache_s in args.extra_cache_dir:
        extra_path = Path(cache_s)
        extra_entries = load_cache(extra_path)
        extra_items = [(e[2], e[3]) for e in extra_entries]
        for x, y in extra_items:
            if x.shape[0] != 2 or y.shape != x.shape:
                raise ValueError(
                    f"incompatible extra cache item in {extra_path}: "
                    f"mix={tuple(x.shape)}, mask={tuple(y.shape)}")
            if x.shape[1] != args.n_bands:
                raise ValueError(
                    f"extra cache {extra_path} has {x.shape[1]} bands, "
                    f"but --n-bands is {args.n_bands}")
        train_items.extend(extra_items * args.extra_cache_repeat)
        extra_cache_stats.append({
            "path": str(extra_path.resolve()),
            "unique_crops": len(extra_items),
            "repeat": args.extra_cache_repeat,
            "effective_crops": len(extra_items) * args.extra_cache_repeat,
        })
        print(f"  extra ground truth: {extra_path} -> {len(extra_items)} crops "
              f"x {args.extra_cache_repeat} = "
              f"{len(extra_items) * args.extra_cache_repeat} sampler entries")
    print(f"  {len(train_items)} train crops / {len(val_items)} val crops "
          f"({len(songs) - len(val_songs)} train / {len(val_songs)} val songs), "
          f"crop {args.crop} frames = {args.crop*t09.HOP/SR*1000:.0f} ms, "
          f"batch {args.batch}, lr {args.lr}")

    net = t09.CausalSpectralUNet(
        2, (32, 64, 96, 128), 4,
        bottleneck_blocks=args.bottleneck_blocks,
        temporal_dilations=temporal_dilations).to(device)
    net.frontend = (args.in_frontend, args.in_scale)
    net.mask_mode = args.mask_mode
    net.band_layout = args.band_layout
    net.n_bands = args.n_bands
    n_par = sum(p.numel() for p in net.parameters())
    print(f"  model: CausalSpectralUNet, {n_par:,} params "
          f"({n_par/1e6:.4f} M), INT8 {n_par/1024:.0f} KB")
    if args.in_frontend != "linear":
        print(f"  front end: {args.in_frontend}, scale {args.in_scale:g} "
              f"(input mapped to [0,1] before the first conv)")
    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        incompatible = net.load_state_dict(blob["model"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        allowed_missing = [k for k in incompatible.missing_keys
                           if (k.startswith("bott_blocks.") or
                               k.startswith("temporal_blocks."))]
        disallowed_missing = [k for k in incompatible.missing_keys
                              if k not in allowed_missing]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                f"incompatible warm start: missing={disallowed_missing}, "
                f"unexpected={unexpected}")
        if allowed_missing:
            print(f"  warm start: {len(allowed_missing)} new bottleneck tensors "
                  "start as an identity residual path")
        print(f"  warm start: {args.init_from} "
              f"(trained {blob.get('step')} steps, best val {blob.get('best_val')})")

    init_val = evaluate(net, val_items, device, args.crop, args.batch,
                        args.val_batches) if val_items else float("nan")
    print(f"  val log-L1 at initialisation: {init_val:.5f}")

    curve, wall_b = [], 0.0
    best = {"val": float("inf"), "step": 0, "sd": None}
    n_done = 0
    if not args.skip_train:
        curve, wall_b, best, n_done = train(net, device, train_items,
                                            val_items, args, CKPT)
        # the run may have stopped on the wall-clock budget, so the steps that
        # actually happened -- not --steps -- are what the throughput means
        train_audio_s = args.batch * args.crop * n_done * t09.HOP / SR
        print(f"  {wall_b:.1f} wall-s for {train_audio_s:.0f} audio-s of "
              f"training -> {train_audio_s/wall_b:.1f} audio-s/s")

    final_val = evaluate(net, val_items, device, args.crop, args.batch,
                         args.val_batches) if val_items else float("nan")
    drop = init_val - final_val
    print(f"  val log-L1: {init_val:.5f} -> {final_val:.5f}  "
          f"({'LEARNED, -%.4f' % drop if drop > 0 else 'NO IMPROVEMENT'})")

    if args.use_best and best["sd"] is not None:
        net.load_state_dict(best["sd"])
        final_val = evaluate(net, val_items, device, args.crop, args.batch,
                             args.val_batches) if val_items else float("nan")
        drop = init_val - final_val
        print(f"  -> listening-test model restored from step {best['step']}: "
              f"val log-L1 {final_val:.5f} (-{drop:.5f} vs init)")

    health = (mask_health(net, val_items, device, args.crop, args.batch,
                          torch.Generator().manual_seed(1))
              if val_items else {})
    if health:
        print(f"  mask health (val): target mean {health['target_mask_mean']:.4f}, "
              f"predicted mean {health['pred_mask_mean']:.4f}, "
              f"std {health['pred_mask_std']:.4f}")
        collapsed = health["pred_mask_std"] < 0.02
        print(f"  -> {'COLLAPSED (mask is constant)' if collapsed else 'mask varies -- not collapsed'}")

    save_ckpt(CKPT, net, args, n_par, step=n_done, best_val=best["val"])
    BEST_CKPT = RESULTS / f"{args.tag}_best.pt"
    if best["sd"] is not None:
        # the final weights and the best-validation weights are different models
        # on a long run -- keep both so the A/B can score whichever is better
        save_ckpt(BEST_CKPT, net, args, n_par, step=best["step"],
                  best_val=best["val"], state_dict=best["sd"])
        print(f"  best-val checkpoint -> {BEST_CKPT.name} "
              f"(val {best['val']:.4f} @ step {best['step']})")

    # ---- stage C ---------------------------------------------------------
    print("\n[3/3] inference on the held-out track")
    OUT.mkdir(parents=True, exist_ok=True)
    val_wav = t09.WORK / "excerpt_60_100.wav"
    ref_voc = RESULTS / "stems" / "htdemucs" / "vocals.wav"
    metrics = {}
    if val_wav.exists():
        layout = band_layout_for(net)
        n_bands = n_bands_for(net)
        Wa = torch.from_numpy(t09.make_analysis_matrix(n_bands, layout=layout))
        Gs = torch.from_numpy(t09.make_synthesis_matrix(n_bands, layout=layout))
        mix = t09.read_wav(val_wav)
        print(f"  held-out: {val_wav.name}  {mix.shape[-1]/SR:.1f}s")
        t0 = time.perf_counter()
        voc, acc = separate(net, mix, Wa, Gs, device)
        wall_c = time.perf_counter() - t0
        print(f"  causal inference {wall_c:.2f} wall-s -> "
              f"RTF {wall_c/(mix.shape[-1]/SR):.3f} on {device}")
        s = min(voc.shape[-1], acc.shape[-1], mix.shape[-1])
        sf.write(str(OUT / "01_mixture.wav"), mix[..., :s].numpy().T, SR,
                 subtype="FLOAT")
        sf.write(str(OUT / "02_student_vocals.wav"), voc[..., :s].numpy().T, SR,
                 subtype="FLOAT")
        sf.write(str(OUT / "03_student_accompaniment.wav"),
                 acc[..., :s].numpy().T, SR, subtype="FLOAT")
        if ref_voc.exists():
            rv = t09.read_wav(ref_voc)
            sr_ = min(s, rv.shape[-1])
            levels = {
                "mixture": _db(mix[..., :sr_]),
                "teacher_vocals": _db(rv[..., :sr_]),
                "student_vocals": _db(voc[..., :sr_]),
                "student_accompaniment": _db(acc[..., :sr_]),
            }
            metrics["levels_dBFS"] = {k: round(v, 2) for k, v in levels.items()}
            metrics["student_vs_teacher_vocals_si_sdr_dB"] = round(
                t09.si_sdr(voc[..., :sr_].numpy(), rv[..., :sr_].numpy()), 2)
            metrics["mixture_vs_student_vocals_si_sdr_dB"] = round(
                t09.si_sdr(voc[..., :sr_].numpy(), mix[..., :sr_].numpy()), 2)
            print("  levels: " + "   ".join(f"{k} {v:.1f} dBFS"
                                            for k, v in levels.items()))
            silent = levels["student_vocals"] - levels["teacher_vocals"] < -20
            print(f"  SI-SDR(student, teacher vocals) = "
                  f"{metrics['student_vs_teacher_vocals_si_sdr_dB']} dB")
            if silent:
                print(f"  -> WARNING: student vocals are "
                      f"{levels['teacher_vocals']-levels['student_vocals']:.0f} dB "
                      f"below the teacher -- the model has collapsed to silence")
            else:
                print(f"  student vocals are "
                      f"{levels['student_vocals']-levels['teacher_vocals']:+.1f} dB "
                      f"relative to the teacher -- a real signal")
        metrics["inference_rtf_%s" % device] = round(wall_c / (mix.shape[-1] / SR), 4)
    else:
        print(f"  skipped: {val_wav} not found")

    report = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "tag": args.tag,
        "config": {k: getattr(args, k) for k in
                   ("tracks", "steps", "batch", "crop", "lr", "seg", "seed",
                    "scan_s", "n_cand", "crops_per_track", "min_vocal_db",
                    "lr_schedule", "min_lr", "warmup_steps", "in_frontend",
                    "in_scale", "mask_mode", "target_mask", "band_layout",
                    "n_bands",
                    "loss", "rel_weight",
                    "wmask_lambda", "accomp_weight", "mask_l1_weight",
                    "mask_mid_weight", "bottleneck_blocks",
                    "temporal_dilations")},
        "n_train_crops": len(train_items),
        "n_base_train_crops": base_train_crops,
        "base_cache_repeat": args.base_cache_repeat,
        "extra_caches": extra_cache_stats,
        "n_val_crops": len(val_items),
        "n_train_tracks": len(songs) - len(val_songs),
        "n_val_tracks": len(val_songs),
        "holdout_tracks": holdout_names,
        "warm_start_from": args.init_from,
        "steps_done": n_done,
        "time_budget_s": args.time_budget_s,
        "used_best_checkpoint": bool(args.use_best and best["sd"] is not None),
        "best_val_logL1": None if best["val"] == float("inf") else round(best["val"], 5),
        "best_val_step": best["step"],
        "stage_a_tracks": rows,
        "stage_a_timing": timing,
        "stage_a_audio_s": round(audio_s, 1),
        "stage_a_scan_audio_s": round(scan_total, 1),
        "stage_a_wall_s": round(wall_a, 1),
        # teacher throughput = scan seconds per wall second.  Deliberately NOT
        # labelled-audio per wall second: with --crops-per-track K that ratio
        # rises K-fold without the teacher getting any faster.
        "stage_a_audio_s_per_wall_s": (round((scan_total or audio_s) / wall_a, 2)
                                       if wall_a else None),
        "unique_audio_min": round(audio_s / 60, 1),
        "model_params": n_par,
        "objective": "log-magnitude L1 on the reconstructed 2 stems",
        "mask_target": ("soft mask |V| / (|V| + |A|)"
                        if args.target_mask == "magnitude" else
                        "clipped phase-sensitive Re(V*conj(X))/|X|^2"),
        "augmentation": {k: getattr(args, k) for k in
                         ("aug_eq_db", "aug_gain_db", "aug_swap",
                          "aug_stem_db", "aug_remix")},
        "val_logL1_initial": round(init_val, 5),
        "val_logL1_final": round(final_val, 5),
        "val_logL1_drop": round(drop, 5),
        "learned": bool(drop > 0),
        "mask_health": health,
        "loss_curve": curve,
        "train_wall_s": round(wall_b, 1),
        "metrics": metrics,
        "audio_dir": str(OUT),
    }
    p = REPORT
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {p}")
    print(f"wrote audio to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
