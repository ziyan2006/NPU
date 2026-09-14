"""16_build_listen_set.py -- assemble one listening directory for a run.

    python 16_build_listen_set.py --tag v4 --prev v3 \
        --ab-json ../results/ab_v3_v4.json

Last time this was done by hand with a dozen ``cp`` commands, and the result was
filenames like ``jerro_(Jerro_Ma_04_v2_vocals.mp3``.  It also silently exported
the first two tracks of the A/B, which happened to be the two excerpts with no
vocals in them -- the opposite of what a listening test needs.

So the rules live in code now:

  * only excerpts flagged ``vocal_present`` are exported, best vocal first;
  * names are ``<group><n>_<role>.mp3`` with the track name in the README, so
    the order is obvious and the filenames stay sortable;
  * a README is generated from the A/B json, so the numbers next to the audio
    are the measured ones and cannot drift from the report.

The script never re-runs separation: it consumes what 13_ab_compare wrote.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from common import RESULTS          # noqa: E402

AB = RESULTS / "ab"


def to_mp3(src: Path, dst: Path):
    if src.suffix == ".mp3":
        shutil.copyfile(src, dst)
        return
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                    "-b:a", "256k", str(dst)], check=True)


def pick_tracks(ab: dict, n: int) -> list[dict]:
    """Excerpts worth listening to: real vocals, loudest teacher vocal first."""
    rows = [r for r in ab["per_track"] if r.get("vocal_present")]
    rows.sort(key=lambda r: -r.get("teacher_vocal_dBFS", -99))
    return rows[:n]


def ab_prefix(ab_root: Path, index: int) -> list[Path]:
    return sorted(p for p in ab_root.glob(f"{index}_*"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run to showcase, e.g. v4")
    ap.add_argument("--prev", default=None, help="previous run to compare, e.g. v3")
    ap.add_argument("--ab-json", required=True)
    ap.add_argument("--n-tracks", type=int, default=3)
    args = ap.parse_args()

    tag, prev = args.tag, args.prev
    ab = json.loads(Path(args.ab_json).read_text(encoding="utf-8"))
    la, lb = ab["compare"]
    out = RESULTS / f"listen_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.mp3"):
        f.unlink()

    rows = pick_tracks(ab, args.n_tracks)
    if not rows:
        raise SystemExit("no excerpt in the A/B was flagged as containing vocals")

    # exported wavs are keyed by the index they had in that A/B run
    order = {r["track"]: i for i, r in enumerate(ab["per_track"])}
    letters = "ABCDEFGH"
    lines = []
    for k, row in enumerate(rows):
        i = order[row["track"]]
        letter = letters[k]
        prefix = ab_prefix(AB, i)
        if not prefix:
            print(f"  WARNING: no exported audio for {row['track']} -- re-run "
                  f"13_ab_compare with --write-audio {i}")
            continue
        for j, src in enumerate(prefix, start=1):
            # stems are "<index>_<track>_<nn>_<role>"; only the role survives
            role = "_".join(src.stem.split("_")[-2:])
            to_mp3(src, out / f"{letter}{j}_{role}.mp3")
        lines.append((letter, row))

    # the button demo: the only file that shows what the product sounds like
    for src in (tag, prev):
        if not src:
            continue
        subprocess.run([sys.executable, str(_HERE / "14_button_demo.py"),
                        "--source", src, "--out",
                        f"Z_button_demo_{src}.mp3", "--outdir",
                        out.name, "--seg", "6"], cwd=str(_HERE),
                       capture_output=True)

    # stage C: the dedicated held-out track (Le Youth - Chills, 60-100 s)
    for src_tag in (tag, prev):
        if not src_tag:
            continue
        d = RESULTS / src_tag
        for wav in sorted(d.glob("*.wav")):
            role = wav.stem.split("_", 1)[1] if "_" in wav.stem else wav.stem
            to_mp3(wav, out / f"D_{src_tag}_{role}.mp3")

    readme = [f"Listening set for {tag}"
              + (f" (compared against {prev})" if prev else ""),
              "=" * 60, ""]
    readme += ["START HERE", "  Z_button_demo_%s.mp3 -- the product's own sound:" % tag,
               "  original -> 30 ms ramp -> accompaniment only -> back.",
               "  This is rendered with out = mix - g*vocal_est, the exact",
               "  contract the board will implement.  Nothing else in this",
               "  directory tells you whether the button product works.", ""]
    readme += ["A/B BY TRACK (vocal-bearing excerpts, best vocal first)", ""]
    for letter, row in lines:
        readme += [f"  {letter}. {row['track']}",
                   f"     teacher vocal {row['teacher_vocal_dBFS']:+.1f} dBFS, "
                   f"SI-SDR(mix, teacher vocal) "
                   f"{row['mixture_vs_teacher_si_sdr_dB']:+.1f} dB",
                   f"     vocal SI-SDR vs teacher: {la} "
                   f"{row[f'{la}_si_sdr']:+.2f} dB -> {lb} "
                   f"{row[f'{lb}_si_sdr']:+.2f} dB "
                   f"({row['gain_dB']:+.2f})"]
        if f"{la}_acc_si_sdr" in row:
            readme.append(f"     accompaniment SI-SDR vs teacher: {la} "
                          f"{row[f'{la}_acc_si_sdr']:+.2f} dB -> {lb} "
                          f"{row[f'{lb}_acc_si_sdr']:+.2f} dB "
                          f"({row['acc_gain_dB']:+.2f})")
        readme.append("")
    readme += ["WHAT TO LISTEN FOR",
               "  1. how much vocal is left (residue) vs how much music was",
               "     damaged (over-subtraction);",
               "  2. bass and kick -- if they thin out, the button is unusable;",
               "  3. musical noise: watery / metallic residue.  This is the",
               "     number-one failure mode in continuous playback and offline",
               "     metrics cannot see it;",
               "  4. clicks or dropouts at the switch, in the Z file.", ""]
    readme += ["CAVEAT",
               "  The reference is HTDemucs, not ground truth.  The 128-band",
               "  mask architecture has its own 9.87 dB ceiling against a",
               "  full-resolution oracle (reports/03).", ""]
    (out / "README.txt").write_text("\n".join(readme), encoding="utf-8")

    print(f"  {len(list(out.glob('*.mp3')))} files -> {out}")
    for f in sorted(out.iterdir()):
        print(f"    {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
