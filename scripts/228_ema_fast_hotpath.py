"""Drop-in CPU PCM commitment + decoder optimizations; no training authority.

All PCM bytes are still checked for finiteness and hashed on EVERY invocation.
No tensor-version/mtime-only hash cache, LRU enlargement, decode/crop change,
RNG, precision change, CUDA transfer or training operation is introduced.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "scripts/208_ema_authenticated_audio_input.py"
INPUT_SHA = "8d208e34ed1ce6b18997c64d75ad1aa03d67401b7752fa8d995da23d64982825"
DECODER_PATH = ROOT / "scripts/226_ema_decoder_hotpath.py"
DECODER_SHA = "142f81175dc9b2e22b8128eb5ef719afa412a290eac75108f48099cd3e0a11e7"
FINITE_CHUNK_ELEMENTS = 262144


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_decoder(module=None):
    require(sha(DECODER_PATH) == DECODER_SHA, "Changed accepted decoder optimization")
    spec = importlib.util.spec_from_file_location("ema228_decoder226", DECODER_PATH)
    decoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(decoder)
    return decoder.make_decoder(module)


def fast_pcm_sha(value):
    import numpy as np
    import torch
    require(type(value) is torch.Tensor and value.device.type == "cpu"
            and value.dtype == torch.float32 and value.layout == torch.strided
            and not value.requires_grad and value.grad_fn is None, "Detached dense CPU FP32 PCM only")
    # Preserve original C-order/little-endian FP32 bytes, including signed zero.
    # Zero-copy for contiguous CPU data; exactly one C-order copy for strides.
    array = np.ascontiguousarray(value.numpy(), dtype="<f4")
    flat = array.reshape(-1)
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for start in range(0, flat.size, FINITE_CHUNK_ELEMENTS):
        end = min(start+FINITE_CHUNK_ELEMENTS, flat.size)
        require(bool(np.isfinite(flat[start:end]).all()), "Nonfinite PCM")
        digest.update(view[start*4:end*4])
    return digest.hexdigest()


def fast_cache_identity(module):
    """Compile the exact original cache traversal, replacing ONLY pcm_sha.

    Returns an unbound method for the worker-local Backend class. No global
    input module, original loader, recipe, LRU or function identity is patched.
    """
    require(Path(module.__file__).resolve() == INPUT_PATH and sha(INPUT_PATH) == INPUT_SHA,
            "Exact original input module required")
    tree = ast.parse(INPUT_PATH.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AuthenticatedAudioStream")
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "_cache_identity")
    namespace = dict(vars(module)); namespace["pcm_sha"] = fast_pcm_sha
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(INPUT_PATH),"exec"),namespace)
    return namespace["_cache_identity"]


if __name__ == "__main__":
    raise SystemExit("Performance components only; no training/GPU launch")
