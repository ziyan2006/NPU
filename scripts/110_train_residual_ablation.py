"""Bounded, same-graph objective ablation; keeps deployed weights and SD intact.

Three arms use identical batches/steps: independent band loss; complement band
loss; complement plus differentiable product-waveform loss. Final holdouts and
the Cambridge counterexamples are never consumed here. Internal validation is
reserved from former training pools, not claimed to be unseen by the warm start.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import unicodedata
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


t13 = load_module("ab_residual_train", "13_ab_compare.py")
t23 = load_module("truth_residual_train", "23_build_true_stem_cache.py")
t09, t11 = t13.t09, t13.t11
SR, HOP, FFT = 44100, 256, 1024
CAMBRIDGE_HOLDOUTS = ("Skelpolu_HumanMistakes", "Triviul_Angelsaint")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def composition_key(track):
    """Conservative title/artist alias key, not an audio-fingerprint guarantee."""
    name = unicodedata.normalize("NFKC", Path(track).name).casefold()
    name = re.sub(r"\.(?:mp3|mp4|wav|flac|ogg|m4a)$", "", name)
    name = re.sub(r"\.stem$", "", name)
    return "".join(c for c in name if c.isalnum())


def exclude_final_compositions(records, final_ids):
    banned = {composition_key(name) for name in final_ids}
    eligible, excluded = [], []
    for record in records:
        (excluded if composition_key(record["track_id"]) in banned else eligible).append(record)
    return eligible, excluded


def split_training_records(records, seed, reserve):
    """Only former train IDs enter this fine-tune train/validation split."""
    train = sorted((r for r in records if r["split"] == "train"), key=lambda r: r["track_id"])
    if not 0 < reserve < len(train):
        raise ValueError("Invalid internal validation reserve")
    ids = [r["track_id"] for r in train]
    random.Random(seed).shuffle(ids)
    selected = set(ids[:reserve])
    a, b = [r for r in train if r["track_id"] not in selected], [r for r in train if r["track_id"] in selected]
    if ({r["track_id"] for r in a} | {r["track_id"] for r in b}) & {
            r["track_id"] for r in records if r["split"] == "holdout"}:
        raise ValueError("Final holdout contamination")
    return a, b


def decode_musdb(path):
    """Use decoded D+B+O and V; exact synthetic mixture avoids AAC sum mismatch."""
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=sample_rate,channels", "-of", "json", str(path)]))
    streams = probe["streams"]
    if len(streams) != 5 or any(s["channels"] != 2 or int(s["sample_rate"]) != SR for s in streams):
        raise ValueError("Expected five 44.1-kHz stereo MUSDB streams")
    graph = ("[0:a:1][0:a:2][0:a:3]amix=inputs=3:normalize=0:duration=shortest[a];"
             "[a][0:a:4]join=inputs=2:channel_layout=quad:"
             "map=0.0-FL|0.1-FR|1.0-BL|1.1-BR[out]")
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path),
                                   "-filter_complex", graph, "-map", "[out]", "-ar", str(SR),
                                   "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
    samples = np.frombuffer(raw, dtype="<f4").reshape(-1, 4).copy()
    accompaniment, vocal = torch.from_numpy(samples[:, :2].T), torch.from_numpy(samples[:, 2:].T)
    return accompaniment + vocal, vocal


class Corpus:
    def __init__(self, musdb_root, teacher_cache, seed):
        self.train, self.val, self.manifests = {}, {}, {}
        self.excluded = {}
        final_ids = list(CAMBRIDGE_HOLDOUTS)
        documents = {}
        for domain, folder in (("mir1k", "mir1k_gainmix_cache"),
                               ("instrumental", "mshoxx_stereo_instrumental_cache"),
                               ("onair", "onair_true_cache")):
            path = ROOT / "results" / folder / "manifest.json"
            documents[domain] = json.loads(path.read_text(encoding="utf-8"))
            final_ids.extend(r["track_id"] for r in documents[domain]["tracks"] if r["split"] == "holdout")
        original_test = sorted((musdb_root / "test").glob("*.stem.mp4"))
        if len(original_test) != 50:
            raise ValueError("Expected original 50-song MUSDB test inventory for split checks")
        final_ids.extend(p.stem for p in original_test)
        dj_holdout = json.loads((ROOT / "results/opt_bott2_mir1k_5m_report.json").read_text(
            encoding="utf-8"))["holdout_tracks"]
        final_ids.extend(dj_holdout)
        self.final_ids = sorted(set(final_ids))
        for domain, folder, reserve in (("mir1k", "mir1k_gainmix_cache", 9),
                                        ("instrumental", "mshoxx_stereo_instrumental_cache", 3)):
            path = ROOT / "results" / folder / "manifest.json"
            records = [r for r in documents[domain]["tracks"] if r["split"] == "train"]
            eligible, excluded = exclude_final_compositions(records, final_ids)
            self.excluded[domain] = [r["track_id"] for r in excluded]
            self.train[domain], self.val[domain] = split_training_records(eligible, seed, reserve)
            self.manifests[domain] = {"path": str(path), "sha256": digest(path)}
        path = ROOT / "results/onair_true_cache/manifest.json"
        records = [r for r in documents["onair"]["tracks"] if r["split"] == "train"]
        self.train["onair"], excluded = exclude_final_compositions(records, final_ids)
        self.excluded["onair"] = [r["track_id"] for r in excluded]
        self.manifests["onair"] = {"path": str(path), "sha256": digest(path)}
        musdb = sorted((musdb_root / "train").glob("*.stem.mp4"))
        if len(musdb) < 20:
            raise ValueError("MUSDB sample train pool incomplete")
        records = [{"track_id": p.stem, "dataset": "musdb", "split": "train",
                    "mix_files": [str(p)], "vocal_files": []} for p in musdb]
        records, excluded = exclude_final_compositions(records, final_ids)
        self.excluded["musdb"] = [r["track_id"] for r in excluded]
        self.train["musdb"], self.val["musdb"] = split_training_records(records, seed, 19)
        self.manifests["musdb"] = {"root": str(musdb_root), "official_test_used": False,
                                    "test_inventory_checked_without_audio": True,
                                    "tracks_available": len(musdb), "tracks_eligible": len(records),
                                    "source_sha256": {r["mix_files"][0]: digest(r["mix_files"][0]) for r in records}}
        self.lru = OrderedDict()
        self.onair_audio = {}
        self.musdb_audio = {}
        # Only ~7-second files are retained, rather than full long-song spectra.
        for i, r in enumerate(records):
            self.musdb_audio[r["track_id"]] = decode_musdb(Path(r["mix_files"][0]))
            if (i + 1) % 10 == 0:
                print(f"CORPUS MUSDB decoded={i+1}/{len(records)}", flush=True)
        files = sorted(teacher_cache.glob("*.pt"))
        if not files:
            raise ValueError("Teacher anchor cache absent")
        banned = {composition_key(name) for name in final_ids}
        self.teacher = []
        source_ids = set()
        for path in files:
            row = torch.load(path, map_location="cpu", weights_only=False)
            if composition_key(row["src"]) in banned:
                raise ValueError("Teacher cache includes a final holdout composition")
            if row.get("band_layout", "legacy_log") != "legacy_log" or row.get("target_mask", "magnitude") != "magnitude":
                raise ValueError("Incompatible teacher anchor representation")
            x, y = row["mix"].float(), row["mask"].float()
            if x.shape != y.shape or x.shape[:2] != (2, 128):
                raise ValueError("Invalid teacher anchor shapes")
            self.teacher.append((x, y))
            source_ids.add(row["src"])
        self.teacher_meta = {"path": str(teacher_cache), "crops": len(files),
                             "songs": len(source_ids), "source_ids": sorted(source_ids),
                             "cache_sha256": {str(p): digest(p) for p in files}}

    def audio(self, record):
        if record["dataset"] == "musdb":
            return self.musdb_audio[record["track_id"]]
        key = (record["dataset"], record["track_id"])
        if key in self.onair_audio:
            return self.onair_audio[key]
        if key not in self.lru:
            spec = t23.TrackSpec(record["track_id"], record["dataset"], record["mix_files"],
                                 record["vocal_files"], record.get("stem_files", []))
            self.lru[key] = t23.load_track(spec)
            if record["dataset"] == "onair":
                self.onair_audio[key] = self.lru.pop(key)
                return self.onair_audio[key]
            if len(self.lru) > 12:
                self.lru.popitem(last=False)
        self.lru.move_to_end(key)
        return self.lru[key]

    def crop(self, record, generator, length, augment):
        mix, vocal = self.audio(record)
        if mix.shape[-1] < length:
            raise ValueError("Source too short for context-aware crop")
        start = int(torch.randint(mix.shape[-1] - length + 1, (1,), generator=generator))
        x, v = mix[:, start:start+length], vocal[:, start:start+length]
        if augment:
            db = (-12, -6, 0, 6)[int(torch.randint(4, (1,), generator=generator))]
            a = x - v
            v = v * (10 ** (db / 20))
            x = a + v
            common_db = float(torch.rand(1, generator=generator) * 2 - 1)
            gain = min(1.0, 0.95 / max(float(x.abs().max()), 1e-12)) * 10 ** (common_db / 20)
            x, v = x * gain, v * gain
            if float(torch.rand(1, generator=generator)) < 0.5:
                x, v = x.flip(0), v.flip(0)
        return x.clone(), v.clone()

    def batch(self, generator, length):
        # Every arm sees the same domain ordering and random draws.
        xs, vs = [], []
        for domain in ("musdb", "musdb", "mir1k", "mir1k", "onair", "instrumental"):
            pool = self.train[domain]
            record = pool[int(torch.randint(len(pool), (1,), generator=generator))]
            x, v = self.crop(record, generator, length, True)
            xs.append(x); vs.append(v)
        return torch.stack(xs), torch.stack(vs)

    def validation(self, length):
        generator = torch.Generator().manual_seed(881)
        rows = []
        for domain in ("musdb", "mir1k", "instrumental"):
            for record in self.val[domain][:3]:
                x, v = self.crop(record, generator, length, False)
                rows.append((domain, record["track_id"], x, v))
        return rows


def stft_batch(x):
    shape = x.shape
    return t09._stft(x.reshape(-1, shape[-1])).reshape(*shape[:-1], 513, -1)


def prepare_truth(x, v, wa):
    spectrum, truth = stft_batch(x), stft_batch(v)
    bands = torch.einsum("fk,bcft->bckt", wa, spectrum.abs())
    target = truth.abs() / (truth.abs() + (spectrum - truth).abs() + 1e-6)
    target = torch.einsum("fk,bcft->bckt", wa, target)
    return spectrum, bands, target


def product_vocal(spectrum, band_mask, gs, length, kill):
    band_mask = band_mask.clone()
    band_mask[:, :, :kill] = 0
    full = torch.einsum("fk,bckt->bcft", gs, band_mask).clamp(0, 1)
    shape = spectrum.shape
    estimated = t09._istft((spectrum * full).reshape(-1, 513, shape[-1]), length)
    return estimated.reshape(shape[0], shape[1], length)


def waveform_loss(pv, x, v, warmup):
    # Discard STFT-edge effects and all causal warm-up. A=mixture-V, so the
    # two scale-dependent stem errors are identical; don't double-count them.
    start, end = (warmup + 2) * HOP, x.shape[-1] - 2 * HOP
    if end <= start:
        raise ValueError("No valid reconstruction scoring interval")
    error = (pv[..., start:end] - v[..., start:end]).square().mean(dim=(-1, -2))
    energy = x[..., start:end].square().mean(dim=(-1, -2))
    return (error / (energy + 1e-6)).mean()


def objective(arm, mv, ma, target, bands):
    options = SimpleNamespace(loss="logl1+wmask", wmask_lambda=2., accomp_weight=1.)
    if arm != "control":
        ma = 1 - mv
    return t11.train_loss(options, bands, mv, ma, target)[0]


@torch.no_grad()
def validate(net, data, wa, gs, device, warmup, kill):
    net.eval()
    errors = {}
    for domain, track, x, v in data:
        x, v = x[None].to(device), v[None].to(device)
        spectrum, bands, _ = prepare_truth(x, v, wa)
        output = net(bands)
        pv = product_vocal(spectrum, (output[:, :2] + 1) / 2, gs, x.shape[-1], kill)
        errors.setdefault(domain, []).append(float(waveform_loss(pv, x, v, warmup)))
    domain_means = {k: sum(v) / len(v) for k, v in errors.items()}
    net.train()
    return sum(domain_means.values()) / len(domain_means), domain_means


def save_checkpoint(path, net, initial, step, arm, metadata, validation):
    payload = {k: v for k, v in initial.items() if k != "model"}
    payload.update(model={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                   step=step, steps=metadata["steps"], lr=metadata["lr"], best_val=None,
                   mask_mode="independent" if arm == "control" else "complement",
                   finetune={**metadata, "arm": arm, "step": step, "internal_validation": validation,
                             "initial_checkpoint_step": initial.get("step")})
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/student_bott2_mir1k_candidate.pt")
    ap.add_argument("--musdb-root", type=Path, default=ROOT / "data/datasets/MUSDB18-7-STEMS")
    ap.add_argument("--teacher-cache", type=Path, default=ROOT / "results/opt_bott2_5m_cache")
    ap.add_argument("--out", type=Path, default=ROOT / "results/residual_finetune_clean_20261001")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--arms", default="control,complement,residual")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--wave-weight", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()
    arms = args.arms.split(",")
    if len(arms) != len(set(arms)) or not set(arms) <= {"control", "complement", "residual"}:
        ap.error("Invalid arms")
    if args.steps < 1 or args.crop < 32 or args.crop % 8 or args.warmup < 96 or args.warmup % 8 or args.log_every < 1:
        ap.error("Invalid step/crop/warm-up sizes")
    if args.out.exists():
        raise ValueError("Choose a new output directory; existing candidates are never overwritten")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, initial = t13.load_student(args.checkpoint, device)
    if (initial.get("bottleneck_blocks") != 2 or initial.get("temporal_dilations") or
            net.frontend != ("linear", 1.) or net.band_layout != "legacy_log" or net.n_bands != 128):
        raise ValueError("Requires the frozen deployed graph")
    original_hash = digest(args.checkpoint)
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    corpus = Corpus(args.musdb_root, args.teacher_cache, args.seed)
    length = (args.crop + args.warmup - 1) * HOP
    validation = corpus.validation(length)
    wa = torch.from_numpy(t09.make_analysis_matrix()).to(device)
    gs = torch.from_numpy(t09.make_synthesis_matrix()).to(device)
    kill = t11.lf_kill_band_for(250., "legacy_log", 128)
    metadata = {"checkpoint": str(args.checkpoint), "sha256": original_hash,
                "script_sha256": digest(__file__), "steps": args.steps, "seed": args.seed,
                "lr": args.lr, "wave_weight": args.wave_weight, "crop": args.crop,
                "warmup": args.warmup, "kill_bands": kill,
                "teacher_weight": .4, "truth_weight": .6,
                "batch_domains": ["musdb", "musdb", "mir1k", "mir1k", "onair", "instrumental"],
                "manifest_provenance": corpus.manifests, "teacher": corpus.teacher_meta,
                "excluded_before_split": corpus.excluded, "final_holdout_ids": corpus.final_ids,
                "composition_guard": "NFKC/casefold title+artist, ignoring punctuation and file suffix; not audio fingerprinting",
                "splits": {d: {"train": [r["track_id"] for r in corpus.train[d]],
                                "validation": [r["track_id"] for r in corpus.val.get(d, [])]}
                           for d in corpus.train},
                "validation_scope": "Reserved from former training pools; may have been seen by warm-start weights",
                "musdb_mix": "Decoded D+B+O+V synthetic sum, not independently AAC-coded mixture stream",
                "torch_version": torch.__version__, "device": device}
    report = {"metadata": metadata, "arms": {}}
    (args.out / "experiment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for arm in arms:
        net.load_state_dict(initial["model"])
        torch.manual_seed(args.seed)
        gen = torch.Generator().manual_seed(args.seed)
        teacher_gen = torch.Generator().manual_seed(args.seed + 1)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        net.train()
        base_score, domains = validate(net, validation, wa, gs, device, args.warmup, kill)
        best, best_step = base_score, 0
        save_checkpoint(args.out / f"{arm}_best.pt", net, initial, 0, arm, metadata, domains)
        curve = [{"step": 0, "validation": base_score, "domains": domains}]
        arm_start = time.perf_counter()
        for step in range(1, args.steps + 1):
            lr = args.lr * min(step / 20, 1.) * (0.2 + 0.8 * (1 + math.cos(math.pi * step / args.steps)) / 2)
            for group in opt.param_groups:
                group["lr"] = lr
            x, v = corpus.batch(gen, length)
            x, v = x.to(device), v.to(device)
            spectrum, bands, target = prepare_truth(x, v, wa)
            teacher_x, teacher_y = t11.sample_batch(corpus.teacher, teacher_gen,
                                                   args.crop + args.warmup, 4)
            teacher_x, teacher_y = teacher_x.to(device), teacher_y.to(device)
            opt.zero_grad(set_to_none=True)
            output, teacher_output = net(bands), net(teacher_x)
            mv, ma = (output[:, :2] + 1) / 2, (output[:, 2:] + 1) / 2
            tv, ta = (teacher_output[:, :2] + 1) / 2, (teacher_output[:, 2:] + 1) / 2
            selection = slice(args.warmup, None)
            band_loss = .6 * objective(arm, mv[..., selection], ma[..., selection],
                                       target[..., selection], bands[..., selection]) + .4 * objective(
                                           arm, tv[..., selection], ta[..., selection],
                                           teacher_y[..., selection], teacher_x[..., selection])
            wave = band_loss.new_zeros(())
            if arm == "residual":
                pv = product_vocal(spectrum, mv, gs, length, kill)
                wave = waveform_loss(pv, x, v, args.warmup)
            loss = band_loss + args.wave_weight * wave
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
            if step % args.log_every == 0 or step == args.steps:
                score, domains = validate(net, validation, wa, gs, device, args.warmup, kill)
                curve.append({"step": step, "loss": float(loss), "band": float(band_loss),
                              "wave": float(wave), "validation": score, "domains": domains,
                              "wall_s": time.perf_counter() - arm_start})
                if score < best:
                    best, best_step = score, step
                    save_checkpoint(args.out / f"{arm}_best.pt", net, initial, step, arm, metadata, domains)
                print(f"TRAIN arm={arm} step={step}/{args.steps} loss={float(loss):.5f} "
                      f"val={score:.6f} best_step={best_step}", flush=True)
        save_checkpoint(args.out / f"{arm}_final.pt", net, initial, step, arm, metadata, domains)
        report["arms"][arm] = {"best_step": best_step, "best_validation": best,
                              "baseline_validation": base_score, "curve": curve,
                              "wall_s": time.perf_counter() - arm_start,
                              "best_sha256": digest(args.out / f"{arm}_best.pt"),
                              "final_sha256": digest(args.out / f"{arm}_final.pt")}
        (args.out / "experiment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if digest(args.checkpoint) != original_hash:
        raise ValueError("Deployed checkpoint changed")
    report["total_wall_s"] = time.perf_counter() - started
    report["deployed_checkpoint_unchanged"] = True
    (args.out / "experiment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RESIDUAL_ABLATION PASS out={args.out}", flush=True)


if __name__ == "__main__":
    main()
