"""Pool repeated A/B runs by track before taking the cohort median.

Each input may use different display labels; the two entries in ``compare``
identify the metric keys.  Vocal metrics only use excerpts that pass the
teacher-vocal gate.  This implements the project's low-variance protocol:
average repeated positions for each song first, then take paired medians.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summary(rows: list[dict], gain_key: str) -> dict:
    gains = np.asarray([r[gain_key] for r in rows], dtype=np.float64)
    return {
        "n_tracks": len(rows),
        "median_gain_dB": round(float(np.median(gains)), 3),
        "mean_gain_dB": round(float(np.mean(gains)), 3),
        "tracks_improved": int((gains > 0).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    vocal = defaultdict(list)
    accomp = defaultdict(list)
    idle = defaultdict(list)
    protocols = []
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        a, b = data["compare"]
        protocols.append({"file": path.name, "frac": data.get("frac"),
                          "a": a, "b": b})
        for row in data["per_track"]:
            name = row["track"]
            if row.get("vocal_present"):
                vocal[name].append(row[f"{b}_si_sdr"] - row[f"{a}_si_sdr"])
                accomp[name].append(
                    row[f"{b}_acc_si_sdr"] - row[f"{a}_acc_si_sdr"])
            ia, ib = f"{a}_acc_idle_si_sdr", f"{b}_acc_idle_si_sdr"
            if ia in row and ib in row:
                idle[name].append(row[ib] - row[ia])

    names = sorted(vocal)
    rows = []
    for name in names:
        rows.append({
            "track": name,
            "positions": len(vocal[name]),
            "vocal_gain_dB": round(float(np.mean(vocal[name])), 3),
            "accomp_gain_dB": round(float(np.mean(accomp[name])), 3),
            "idle_gain_dB": (round(float(np.mean(idle[name])), 3)
                             if idle[name] else None),
        })
    idle_rows = [{"track": name, "idle_gain_dB": float(np.mean(values))}
                 for name, values in idle.items() if values]
    payload = {
        "protocol": "mean positions per track, then paired cohort median",
        "inputs": protocols,
        "vocal": summary(rows, "vocal_gain_dB"),
        "accompaniment_on_vocal_tracks": summary(rows, "accomp_gain_dB"),
        "idle": summary(idle_rows, "idle_gain_dB"),
        "per_track": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("inputs", "per_track")},
                     indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
