"""Authenticated single Mel input backend; NOT a trainer or CUDA gate.

Uses the closed143 true TRAIN loader and149 approved24 waveform loader through
the closed206 NEW local4500..5000 route. Old193/149 stops are not changed.
The historical approval authenticates input provenance, NOT new training
authority. No model, optimizer, teacher inference, STFT or output audio here.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import shutil

import torch

ROOT = Path(__file__).resolve().parent.parent
PURPOSE = "NONRELEASE_EMA_AUDIO_INPUT_NOT_TRAINER"
PINS = {
    "scripts/206_ema_live_input_schedule.py": "eaf1112263df6f74f97ea3e9f9a262db401dd24a819197ce0dbabdc6333d61ba",
    "scripts/149_prepare_exploratory_import.py": "d5738190df3f4debadfc920992104951c7ae68b429081e160fdc1a1399271a95",
    "scripts/134_generate_teacher_library.py": "fc136fc8250f3f788bdaf4c8d2501a13c2ecef213d38d5f1748b5da95cfa38f3",
    "scripts/131_run_teacher_pilot.py": "99a2f17a7787a7d89d259b5532d5cbee635adfdb92c000352f086b795fad48bb",
    "scripts/128_acquire_cambridge_candidates.py": "4d82316fe483c87e21fae654a87fb90f79e986c633f1698f143272dee237e037",
    "scripts/110_train_residual_ablation.py": "5476a5b6e38ed7b90044604e8f0e841ef46cc3ebcd6daa4249226eabce21bc07",
    "scripts/23_build_true_stem_cache.py": "29503781df9ff4a3e573cb572ca55456dab9ecceb5df112e916ada755812cd3c",
    "scripts/common.py": "2bb3cff47058eb09a2ee63bcdf77fe19fa7e954873846ee8e622e2bec7fc15a8",
    "results/paired_exploratory_import_20261003_r2/approval.json": "aa2eb50b82635cefb46fd4fc2d3cae611154eb064b26f2568968b756b139b7c6",
    "results/teacher_library_melband_20261002/plan.json": "aa351fbcb9ac489c6ea93fa436a8842e580390141de766f539dca4eabf6d41f5",
    "results/paired_distillation_prepare_20261002/kim_melband_snapshot.json": "55f6c74860f25aba488918eec9947f0ed5bbad7679a9b3c861638457b5234ae5",
    "results/training_protocol_20261001/dataset_lock.json": "36a9be8ade7822969a7fed661c8f226f49b89e8ece90e1e6192ae0ae1b4b2c38",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(bindings):
    for path, expected in bindings.items():
        if file_sha(path) != expected:
            raise ValueError("Authenticated input dependency changed: " + str(path))


verify_files({str(ROOT / p): h for p, h in PINS.items()})


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


live = load("ema208_live_route", "206_ema_live_input_schedule.py")
inp = load("ema208_original_approved_input", "149_prepare_exploratory_import.py")
storage, data, acq = live.storage, inp.data, inp.acq
equal, portable, require = storage.equal, storage.portable, storage.require
ORIGINAL_SAMPLER = {
    "approval_sha256": storage.source.PINS["results/mel_lr_scale_import_20261004/approval.json"],
    "true_lock_sha256": PINS["results/training_protocol_20261001/dataset_lock.json"],
    "seed": 20261002, "cursor": 4500, "teacher": "kim_melband",
}


def validate_initial_sampler(sampler):
    require(equal(sampler, ORIGINAL_SAMPLER), "Full typed original4500 sampler required")


def pcm_sha(value):
    require(type(value) is torch.Tensor and value.device.type == "cpu" and
            value.dtype == torch.float32 and not value.requires_grad and value.grad_fn is None and
            bool(torch.isfinite(value).all()), "Finite detached CPU FP32 PCM")
    return hashlib.sha256(value.contiguous().numpy().astype("<f4", copy=False).tobytes()).hexdigest()


def packet_identity(packet):
    require(type(packet) is dict and list(packet) == ["x", "v", "domains", "cursor", "metadata"],
            "Original single-target packet fields")
    require(equal(packet["domains"], live.DOMAINS) and type(packet["cursor"]) is int and
            4500 <= packet["cursor"] < 5000, "Bounded original domain route")
    live.validate_metadata(packet["metadata"])
    require(packet["x"].shape == packet["v"].shape == (6, 2, 89856), "Original input geometry")
    result = {"counter": packet["cursor"], "input_sha256": [pcm_sha(w) for w in packet["x"]],
              "target_sha256": [pcm_sha(w) for w in packet["v"]], "metadata": portable(packet["metadata"])}
    require(result["input_sha256"] == [r["input_pcm_sha256"] for r in result["metadata"]],
            "Actual input PCM metadata identity")
    return result


class AuthenticatedAudioStream:
    """One bounded route with pinned actual backend classes and cache guards.

    Sampling state is206 semantic cursor/metadata. Decoder LRUs are disposable,
    not new independent samples or training state. Any failed actual draw
    poisons THIS backend even after206 rolls back the semantic cursor/RNG;
    it cannot silently retry through possibly changed foreign cache state.
    This is not a full raw/Adam/shadow/CPU-CUDA transaction.
    """
    def __init__(self, sampler):
        validate_initial_sampler(sampler)  # Reject before audio constructors.
        rng = storage.capture_cpu_rng()
        bindings = {str(ROOT / p): h for p, h in PINS.items()}
        bindings.update({str(ROOT / p): h for p, h in live.PINS.items()})
        bindings.update({str(ROOT / p): h for p, h in storage.source.PINS.items()})
        approval = inp.verified_approval(inp.DEFAULT_APPROVAL)
        bindings.update(approval["bindings_sha256"])
        bindings.update({row["path"]: row["sha256"] for row in approval["snapshots"].values()})
        plan = acq.read_sealed(ROOT / "results/teacher_library_melband_20261002/plan.json")
        verify_files(bindings)
        actual_versions = inp.m.bulk.media_versions()
        require(equal(actual_versions, plan["media_versions"]), "Original ffmpeg/ffprobe decoder versions")
        self._executables = {name: str(Path(shutil.which(name) or "").resolve()) for name in actual_versions}
        require(all(Path(p).is_file() for p in self._executables.values()), "Actual decoder executables required")
        bindings.update({path: file_sha(path) for path in self._executables.values()})
        self.dataset = inp.ApprovedTeacherDataset(approval, "kim_melband")
        self.true = inp.m.LockedTruePool(inp.m.bulk.OLD_LOCK, self.dataset.config)
        self.route = live.BoundedSingleTargetStream(sampler, self.true, self.dataset, data.crop_recipe)
        for row in self.dataset.rows:
            bindings.update({info["path"]: info["sha256"] for info in (row["source"], *row["label_files"].values())})
        self._bindings = dict(bindings)
        self._signatures = {p: data.stat_signature(p) for p in bindings}
        self._backend = self._backend_identity()
        self._functions = self._function_identity()
        self._cache = self._cache_identity()
        self.poisoned = False
        self._contract = storage.seal({"schema": 1, "purpose": PURPOSE, "source_sampler": portable(sampler),
            "input_bindings_sha256": portable(bindings), "decoder_versions": actual_versions,
            "decoder_executables": portable(self._executables),
            "input_packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "soundfile", "scipy")},
            "true_train_counts": {d: len(rows) for d, rows in self.true.pools.items()},
            "pseudo_ids": [row["song_id"] for row in self.dataset.rows],
            "input_config": portable(self.dataset.config), "backend_identity_sha256": storage.digest(self._backend),
            "single_mel_target": True, "training_authorized": False, "cuda_resume_verified": False,
            "full_training_transaction_verified": False, "release_selection": "NONE"})
        self._contract_sha = storage.digest(self._contract)
        require(equal(rng, storage.capture_cpu_rng()), "Backend construction must preserve CPU/Python/NumPy RNG")

    def _backend_identity(self):
        return {"true_lock": portable(self.true.lock), "true_pools": portable(self.true.pools),
            "true_config": portable(self.true.config), "true_bound": self.true.bound,
            "true_path": str(self.true.path), "true_cache_limit": self.true.cache_songs,
            "teacher_doc": portable(self.dataset.doc), "teacher_rows": portable(self.dataset.rows),
            "teacher_by_id": portable(self.dataset.by_id), "teacher_config": portable(self.dataset.config),
            "teacher_bound": self.dataset.bound, "teacher_path": str(self.dataset.path),
            "teacher_seed": self.dataset.seed, "teacher_mix": self.dataset.mix_identity,
            "teacher_cache_limit": self.dataset.cache_songs, "teacher_cursor": self.dataset.cursor,
            "teacher_signatures": portable(self.dataset.signatures)}

    def _function_identity(self):
        return (type(self.true), type(self.dataset), self.true.crop.__func__, self.true.audio.__func__,
            self.dataset.crop.__func__, self.dataset._source.__func__, data.crop_recipe,
            data.read_window, data.common_gain, inp.m.core.decode_musdb, inp.m.core.t23.load_track,
            inp.m.bulk.decode, inp.m.core.t23._read, inp.m.core.t23.resample_poly,
            inp.m.core.t23.sf.read, data.sf.SoundFile, self.route.recipe,
            self.route.next_batch.__func__)

    def _cache_identity(self):
        require(type(self.true.cache) is OrderedDict and type(self.dataset.cache) is OrderedDict,
                "Original ordered LRU caches")
        return {"true": [(key, tuple(pcm_sha(w) for w in value)) for key, value in self.true.cache.items()],
                "teacher": [(key, pcm_sha(value)) for key, value in self.dataset.cache.items()],
                "true_signatures": portable(self.true.signatures)}

    def guard(self):
        require(not self.poisoned, "Failed audio backend is poisoned; no silent restart")
        require(storage.digest(self._contract) == self._contract_sha, "Input contract changed")
        require(equal(self._backend_identity(), self._backend) and self._function_identity() == self._functions,
                "Actual backend records/implementation/target mapping changed")
        require(equal(self._cache_identity(), self._cache), "Foreign decoded cache/signatures changed")
        require(equal(self._bindings, self._contract["input_bindings_sha256"]), "Input binding map changed")
        require(all(data.stat_signature(p) == signature for p, signature in self._signatures.items()),
                "Bound input/decoder file signature changed")
        require(all(str(Path(shutil.which(name) or "").resolve()) == path for name, path in self._executables.items()),
                "PATH-selected decoder changed")
        self.route._guard()

    @property
    def cursor(self):
        return self.route.cursor

    @property
    def last_metadata(self):
        return self.route.last_metadata

    def state_dict(self):
        self.guard()
        return self.route.state_dict()

    def load_state_dict(self, state, *, last_metadata=None):
        self.guard()
        self.route.load_state_dict(state, last_metadata=last_metadata)

    def contract(self):
        self.guard()
        return portable(self._contract)

    def verify_all_bindings(self):
        self.guard()
        # Input provenance, no old numerical verify/audit or corpus rescan.
        verify_files(self._bindings)
        for path in self.true.signatures:
            verify_files({path: self.true.lock["files"][path]["sha256"]})

    def next_batch(self):
        self.guard()
        try:
            result = self.route.next_batch()
            packet_identity(result)
            require(equal(self._backend_identity(), self._backend) and self._function_identity() == self._functions,
                    "Backend mutated during actual input draw")
            self._cache = self._cache_identity()
            require(len(self.true.cache) <= 3 and len(self.dataset.cache) <= 1, "Original bounded decode LRUs")
            self.guard()
            return result
        except BaseException:
            self.poisoned = True
            raise
