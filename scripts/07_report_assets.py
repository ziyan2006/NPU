"""Generate report assets: spectrogram comparison, Pareto plot, gate matrix,
and a listening pack (mp3) so quality can be judged by ear, not just by numbers.

Usage
-----
python 07_report_assets.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from common import RESULTS, SR, STEMS

FIG = RESULTS / "figures"
LISTEN = RESULTS / "listen"
for d in (FIG, LISTEN):
    d.mkdir(parents=True, exist_ok=True)

ORDER = ["umx", "hdemucs_mmi", "htdemucs", "mdx", "mdx7ecf8ec1"]


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    return plt


def available() -> list[str]:
    return [t for t in ORDER if (STEMS / t / "vocals.wav").exists()]


# --------------------------------------------------------------------------
def fig_spectrograms(plt, tags: list[str]) -> Path:
    import librosa
    import librosa.display

    panels = [("mixture", STEMS / tags[0] / "mixture.wav")] + \
             [(f"{t} — vocals", STEMS / t / "vocals.wav") for t in tags]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.95 * n), sharex=True)
    for ax, (title, path) in zip(axes, panels):
        x, _ = sf.read(str(path), dtype="float32", always_2d=True)
        y = x.mean(1)
        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2,
            ref=np.max)
        img = librosa.display.specshow(S, sr=SR, hop_length=512, x_axis="time",
                                       y_axis="log", ax=ax, cmap="magma",
                                       vmin=-80, vmax=0)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(60, 12000)
    fig.colorbar(img, ax=axes, format="%+2.0f dB", pad=0.01)
    fig.suptitle("Vocal-stem spectrograms, 60-100 s of Le Youth - Chills",
                 fontsize=12)
    out = FIG / "01_spectrograms.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_pareto(plt) -> Path | None:
    cx = RESULTS / "complexity.json"
    mx = RESULTS / "metrics.json"
    bx = RESULTS / "fpga_budget.json"
    if not (cx.exists() and mx.exists()):
        return None
    comp = {r["tag"]: r for r in json.loads(cx.read_text(encoding="utf-8"))}
    met = json.loads(mx.read_text(encoding="utf-8"))["per_model"]
    budget = json.loads(bx.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    budget_line = budget["hardware"]["sustained_gmac_s"]

    pts = []
    for tag, c in comp.items():
        if tag not in met:
            continue
        pts.append((tag, c["mac_per_audio_s_G"],
                    met[tag].get("consensus_vocal_si_sdr_dB", np.nan),
                    c["params_M"]))

    for tag, g, sdr, p in pts:
        ax.scatter(g, sdr, s=90, zorder=3,
                   label=f"{tag}  ({p:.1f}M params)")
        ax.annotate(tag, (g, sdr), textcoords="offset points", xytext=(8, 6),
                    fontsize=9)

    # proposed designs: compute position is real, quality is projected
    for d in budget["designs"]:
        g = d["gmac_per_audio_s"]
        if g > budget_line:
            continue
        ax.axvline(g, color="tab:green", alpha=0.18, zorder=1)
    ax.axvspan(0.5, budget_line, color="tab:green", alpha=0.06, zorder=0,
               label="Zynq-7020 real-time compute window (RTF<=0.5)")

    ax.set_xscale("log")
    ax.set_xlabel("compute demand  (GMAC per second of audio, log scale)")
    ax.set_ylabel("consensus SI-SDR vs other models (dB)")
    ax.set_title("Compute vs agreement -- baselines only.\n"
                 "Green band = power budget of a 7020 at RTF 0.5.\n"
                 "Causal designs sit at 0.6-2.2 GMAC/s, i.e. the far left.",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    out = FIG / "02_pareto_compute_vs_quality.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_gate_matrix(plt) -> Path | None:
    bx = RESULTS / "fpga_budget.json"
    if not bx.exists():
        return None
    b = json.loads(bx.read_text(encoding="utf-8"))
    gate_names = ["causal_ok", "compute_ok", "memory_ok"]
    labels = ["causality\n<=100 ms", "compute\nRTF<=0.5", "on-chip\nBRAM"]

    rows, names = [], []
    for r in b["baselines"]:
        rows.append([1 if r["gates"].get(g) else 0 for g in gate_names])
        names.append(f"{r['name']}  (baseline)")
    for d in b["designs"]:
        rows.append([1 if d["gates"].get(g) else 0 for g in
                     ("causal_ok", "compute_ok", "memory_ok")])
        names.append(f"{d['name']}  (proposed)")

    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(names) + 1.9))
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "PASS" if M[i, j] else "FAIL", ha="center",
                    va="center", fontsize=8,
                    color="black" if M[i, j] else "white", fontweight="bold")
    ax.set_title("Feasibility gates on Zynq-7020\n"
                 "every strong baseline fails causality first", fontsize=11)
    out = FIG / "03_gate_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
def listening_pack(tags: list[str]) -> list[Path]:
    made = []
    for t in tags:
        for stem in ("vocals", "accompaniment"):
            src = STEMS / t / f"{stem}.wav"
            if not src.exists():
                continue
            dst = LISTEN / f"{t}_{stem}.mp3"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(src),
                 "-b:a", "192k", str(dst)],
                check=True)
            made.append(dst)
    mix_src = STEMS / tags[0] / "mixture.wav"
    if mix_src.exists():
        dst = LISTEN / "00_mixture_reference.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mix_src),
                        "-b:a", "192k", str(dst)], check=True)
        made.insert(0, dst)
    return made


def main() -> int:
    tags = available()
    if not tags:
        print("no stems yet; run 03_separate.py first")
        return 1
    plt = _setup_mpl()
    outs = [fig_spectrograms(plt, tags), fig_pareto(plt), fig_gate_matrix(plt)]
    for o in outs:
        if o:
            print(f"figure -> {o}")
    for p in listening_pack(tags):
        print(f"audio  -> {p}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
