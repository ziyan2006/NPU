"""14_button_demo.py -- render what the button actually sounds like.

The product is `out = mix_delayed - g * vocal_est`, where the button only moves
`g` between 0 and 1.  With `g = 0` the output is bit-exactly the mixture; with
`g = 1` it is the model's accompaniment.  Because the model never stops running,
pressing the button costs no compute and no re-synchronisation -- the only
audible event is the 30 ms ramp, which exists purely to avoid a click.

This script concatenates a few seconds of each state so the whole thing can be
judged by ear in one file, ramps included.  That is the one question offline
metrics cannot answer: whether the switch feels instant and click-free, and
whether the accompaniment holds up when the vocal disappears underneath it.

Usage
-----
    python 14_button_demo.py                       # uses results/v2 + teacher stems
    python 14_button_demo.py --vocals A --source ab
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common import RESULTS, SR          # noqa: E402

OUT = RESULTS / "listen_v2"
RAMP_MS = 30.0                          # the spec: 30 ms, or it clicks


def read(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{path} is {sr} Hz, expected {SR}")
    return x.T


def fade(n: int, rising: bool) -> np.ndarray:
    """Equal-power-ish ramp.  A hard cut between two decorrelated signals clicks;
    30 ms is long enough to hide it and short enough to feel instant."""
    r = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return r if rising else 1.0 - r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="v2",
                    help="subdirectory of results/ holding 01_mixture / "
                         "02_student_vocals / 03_student_accompaniment")
    ap.add_argument("--out", default="Z_button_demo.mp3")
    ap.add_argument("--outdir", default="listen_v2",
                    help="subdirectory of results/ to write into")
    ap.add_argument("--seg", type=float, default=6.0,
                    help="seconds of each state")
    a = ap.parse_args()

    global OUT
    OUT = RESULTS / a.outdir

    d = RESULTS / a.source
    mix = read(d / "01_mixture.wav" if (d / "01_mixture.wav").exists()
               else d / "01_mixture.mp3")
    acc = read(d / "03_student_accompaniment.wav"
               if (d / "03_student_accompaniment.wav").exists()
               else d / "03_student_accompaniment.mp3")
    n = min(mix.shape[-1], acc.shape[-1])
    mix, acc = mix[:, :n], acc[:, :n]

    # the mixture as it goes in, and the mixture as `g = 0` returns it --
    # identical, which is the point: the un-pressed state is not a compromise
    g_off = mix
    g_on = acc

    s = int(a.seg * SR)
    r = int(RAMP_MS / 1000 * SR)
    if 4 * s + 2 * r > n:
        a.seg = (n - 2 * r) / 4 / SR
        s = int(a.seg * SR)
        print(f"  clip is short -- using {a.seg:.2f}s per state")

    parts = [
        g_off[:, :s],                              # vocals present
        g_off[:, s - r:s] * fade(r, False)
        + g_on[:, s - r:s] * fade(r, True),        # press: ramp down
        g_on[:, s:2 * s],                          # accompaniment only
        g_on[:, 2 * s - r:2 * s] * fade(r, False)
        + g_off[:, 2 * s - r:2 * s] * fade(r, True),  # release
        g_off[:, 2 * s:3 * s],
    ]
    y = np.concatenate(parts, axis=-1)

    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / (Path(a.out).stem + ".wav")
    sf.write(str(wav), y.T, SR, subtype="PCM_16")
    mp3 = OUT / a.out
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav),
                    "-b:a", "256k", str(mp3)], check=True)
    wav.unlink(missing_ok=True)

    print(f"  source      : results/{a.source}")
    print(f"  ramp        : {RAMP_MS:.0f} ms")
    print(f"  0.0-{a.seg:.1f}s  vocals present (g=0, output = mixture exactly)")
    print(f"  {a.seg:.1f}s        PRESS")
    print(f"  {a.seg:.1f}-{2*a.seg:.1f}s  accompaniment only (g=1)")
    print(f"  {2*a.seg:.1f}s        RELEASE")
    print(f"  {2*a.seg:.1f}-{3*a.seg:.1f}s  vocals back")
    print(f"  total {len(y.T)/SR:.1f}s -> {mp3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
