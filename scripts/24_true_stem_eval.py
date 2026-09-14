"""A/B two student checkpoints against untouched, human-provided vocal stems."""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import soundfile as sf
import torch

from common import RESULTS, ROOT, SR


_HERE = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


t09 = _load_module("t09_true_eval", "09_target_model.py")
t11 = _load_module("t11_true_eval", "11_smoke_train.py")
t13 = _load_module("t13_true_eval", "13_ab_compare.py")
t23 = _load_module("t23_true_eval", "23_build_true_stem_cache.py")


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": round(statistics.fmean(values), 3),
            "median": round(statistics.median(values), 3)}


def _checkpoint_label(path: Path) -> str:
    return path.stem.replace("student_", "").replace("_model", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--a", required=True, help="baseline checkpoint")
    ap.add_argument("--b", required=True, help="candidate checkpoint")
    ap.add_argument("--a-name", default=None)
    ap.add_argument("--b-name", default=None)
    ap.add_argument("--segment-s", type=float, default=30.0)
    ap.add_argument("--max-segments", type=int, default=0,
                    help="0 evaluates every holdout segment")
    ap.add_argument("--lf-hz", type=float, default=250.0,
                    help="apply identical low-frequency protection to both models")
    ap.add_argument("--output", default=str(RESULTS / "ab_true_stems.json"))
    ap.add_argument("--audio-dir", default=str(ROOT / "试听文件" / "true_stem_ab"))
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    holdout = [r for r in manifest["tracks"] if r["split"] == "holdout"]
    if not holdout:
        raise SystemExit("manifest has no holdout tracks")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = {args.a_name or _checkpoint_label(Path(args.a)): Path(args.a),
             args.b_name or _checkpoint_label(Path(args.b)): Path(args.b)}
    if len(paths) != 2:
        raise SystemExit("A and B labels must be different")
    nets, banks, meta = {}, {}, {}
    for label, path in paths.items():
        net, blob = t13.load_student(path, device)
        layout = t11.band_layout_for(net)
        n_bands = t11.n_bands_for(net)
        Wa = torch.from_numpy(t09.make_analysis_matrix(n_bands, layout=layout))
        Gs = torch.from_numpy(t09.make_synthesis_matrix(n_bands, layout=layout))
        lf_band = t11.lf_kill_band_for(args.lf_hz, layout, n_bands)
        nets[label] = net
        banks[label] = (Wa, Gs, lf_band)
        meta[label] = {"path": str(path.resolve()), "params": blob.get("params"),
                       "layout": layout, "n_bands": n_bands,
                       "lf_kill_band": lf_band}

    print(f"device {device}; true-stem holdout: "
          f"{', '.join(r['track_id'] for r in holdout)}")
    rows = []
    segment_n = round(args.segment_s * SR)
    audio_dir = Path(args.audio_dir).resolve()
    wrote_audio = False
    for record in holdout:
        spec = t23.TrackSpec(record["track_id"], record["dataset"],
                             record["mix_files"], record["vocal_files"],
                             record.get("stem_files", []))
        mix, ref_vocal = t23.load_track(spec)
        rank = 0
        for start in range(0, mix.shape[-1], segment_n):
            end = min(start + segment_n, mix.shape[-1])
            if end - start < 5 * SR:
                continue
            x = mix[..., start:end]
            v = ref_vocal[..., start:end]
            a = x - v
            ratio_db = t23._vocal_ratio_db(v, x)
            row = {"track": spec.track_id, "segment": rank,
                   "start_s": round(start / SR, 3),
                   "dur_s": round((end - start) / SR, 3),
                   "vocal_to_mix_db": round(ratio_db, 3),
                   "mixture_vocal_si_sdr": round(float(t09.si_sdr(
                       x.numpy(), v.numpy())), 3)}
            rendered = {}
            for label, net in nets.items():
                Wa, Gs, lf_band = banks[label]
                pred_v, pred_a = t11.separate(net, x, Wa, Gs, device,
                                              lf_kill_band=lf_band)
                row[f"{label}_vocal_si_sdr"] = round(float(t09.si_sdr(
                    pred_v.numpy(), v.numpy())), 3)
                row[f"{label}_accompaniment_si_sdr"] = round(float(t09.si_sdr(
                    pred_a.numpy(), a.numpy())), 3)
                rendered[label] = (pred_v, pred_a)
            rows.append(row)
            print(f"  {spec.track_id} #{rank:02d} vocal {ratio_db:+.1f} dB  " +
                  "  ".join(f"{name} V={row[f'{name}_vocal_si_sdr']:+.2f} "
                             f"A={row[f'{name}_accompaniment_si_sdr']:+.2f}"
                             for name in nets))
            if not wrote_audio:
                audio_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(audio_dir / "00_mixture.wav"), x.numpy().T, SR,
                         subtype="PCM_16")
                sf.write(str(audio_dir / "01_reference_vocals.wav"), v.numpy().T,
                         SR, subtype="PCM_16")
                sf.write(str(audio_dir / "02_reference_accompaniment.wav"),
                         a.numpy().T, SR, subtype="PCM_16")
                for i, (label, (pv, pa)) in enumerate(rendered.items(), start=3):
                    sf.write(str(audio_dir / f"{i:02d}_{label}_vocals.wav"),
                             pv.numpy().T, SR, subtype="PCM_16")
                    sf.write(str(audio_dir / f"{i:02d}_{label}_accompaniment.wav"),
                             pa.numpy().T, SR, subtype="PCM_16")
                wrote_audio = True
            rank += 1
            if args.max_segments and len(rows) >= args.max_segments:
                break
        if args.max_segments and len(rows) >= args.max_segments:
            break

    labels = list(nets)
    vocal_gate = [r for r in rows if r["vocal_to_mix_db"] >= -30.0]
    summary = {}
    for label in labels:
        summary[label] = {
            "vocal_scorable": _summary(
                [r[f"{label}_vocal_si_sdr"] for r in vocal_gate]),
            "accompaniment_all": _summary(
                [r[f"{label}_accompaniment_si_sdr"] for r in rows]),
        }
    a_label, b_label = labels
    paired = {
        "vocal_b_minus_a": _summary([
            r[f"{b_label}_vocal_si_sdr"] - r[f"{a_label}_vocal_si_sdr"]
            for r in vocal_gate]),
        "accompaniment_b_minus_a": _summary([
            r[f"{b_label}_accompaniment_si_sdr"] -
            r[f"{a_label}_accompaniment_si_sdr"] for r in rows]),
    }
    report = {"manifest": str(manifest_path), "dataset": manifest["dataset"],
              "device": device, "lf_hz": args.lf_hz, "models": meta,
              "holdout_tracks": [r["track_id"] for r in holdout],
              "segments": len(rows), "scorable_vocal_segments": len(vocal_gate),
              "summary": summary, "paired": paired, "rows": rows,
              "audio_dir": str(audio_dir)}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(json.dumps({"summary": summary, "paired": paired}, indent=2,
                     ensure_ascii=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
