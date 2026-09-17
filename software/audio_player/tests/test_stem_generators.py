#!/usr/bin/env python3
"""Verify deterministic STEM filterbank and resident-task generators."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GENERATED = ROOT / "software" / "audio_player" / "generated"
PROGRAM = ROOT / "hardware" / "generated" / "bott2_mir1k_v1_program"
FILTERBANK_GENERATOR = ROOT / "scripts" / "98_generate_stem_constants.py"
TASK_GENERATOR = ROOT / "scripts" / "99_generate_stem_task_payload.py"
LEGACY_ANALYSIS_SHA256 = "f09eab6acb61f1b608a0f1cae99d876e3f3f6d9c02d879e915972ad0d4648e69"
LEGACY_SYNTHESIS_SHA256 = "52e424d91b7ecb1b765d46ee77b22e0ad7da380b6f1ef1a0b0a5dc18474ff32d"


def run_generator(path: Path, *arguments: str) -> None:
    subprocess.run([sys.executable, str(path), *arguments], cwd=ROOT, check=True)


def generated_snapshot() -> dict[str, bytes]:
    names = (
        "stem_filterbank.h",
        "stem_filterbank.c",
        "stem_task_metadata.h",
        "stem_task_payload.h",
        "stem_task_payload.c",
    )
    return {name: (GENERATED / name).read_bytes() for name in names}


def parse_float_matrix(source: str, symbol: str,
                       rows: int, columns: int) -> np.ndarray:
    pattern = (
        rf"const float {re.escape(symbol)}\[{rows}\]\[{columns}\] = \{{"
        rf"(?P<body>.*?)\n\}};"
    )
    match = re.search(pattern, source, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"missing {symbol}[{rows}][{columns}]")
    tokens = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?f",
        match.group("body"),
    )
    values = np.asarray([float(token[:-1]) for token in tokens], dtype="<f4")
    if values.size != rows * columns:
        raise AssertionError(
            f"{symbol} has {values.size} values, expected {rows * columns}"
        )
    return values.reshape(rows, columns)


def parse_macro(header: str, name: str) -> int:
    match = re.search(
        rf"^#define {re.escape(name)}\s+(0x[0-9a-fA-F]+|\d+)[uUlL]*$",
        header,
        flags=re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"missing integer macro {name}")
    return int(match.group(1), 0)


def parse_digest(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s*([0-9a-f]{{64}})", text)
    if match is None:
        raise AssertionError(f"missing SHA-256 comment for {label}")
    return match.group(1)


def test_filterbank() -> None:
    source = (GENERATED / "stem_filterbank.c").read_text(encoding="ascii")
    invalid_literals = re.findall(r"(?<![.\w])[-+]?\d+f(?![\w.])", source)
    if invalid_literals:
        raise AssertionError(
            f"filterbank contains invalid C float literals: {invalid_literals[:4]}"
        )
    analysis = parse_float_matrix(source, "stem_analysis", 513, 128)
    synthesis = parse_float_matrix(source, "stem_synthesis", 128, 513)

    if analysis.dtype != np.dtype("<f4") or synthesis.dtype != np.dtype("<f4"):
        raise AssertionError("filterbanks must serialize as little-endian float32")
    # The deployed checkpoint was trained with the historical legacy_log
    # matrix. Its 49 empty low-frequency columns are a known compatibility
    # constraint; changing them requires retraining and requantization.
    if int(np.count_nonzero(np.count_nonzero(analysis, axis=0) == 0)) != 49:
        raise AssertionError("legacy_log dead-band pattern changed")
    reconstructed_row_sums = synthesis.T.astype(np.float64).sum(axis=1)
    if not np.allclose(reconstructed_row_sums, 1.0, rtol=0.0, atol=1e-7):
        error = float(np.max(np.abs(reconstructed_row_sums - 1.0)))
        raise AssertionError(f"synthesis row sum error {error} exceeds 1e-7")

    analysis_bytes = analysis.tobytes(order="C")
    synthesis_bytes = synthesis.tobytes(order="C")
    analysis_sha = hashlib.sha256(analysis_bytes).hexdigest()
    synthesis_sha = hashlib.sha256(synthesis_bytes).hexdigest()
    if analysis_sha != LEGACY_ANALYSIS_SHA256:
        raise AssertionError("analysis matrix differs from the trained legacy layout")
    if synthesis_sha != LEGACY_SYNTHESIS_SHA256:
        raise AssertionError("synthesis matrix differs from the trained legacy layout")
    if parse_digest(source, "analysis-le-f32-sha256") != analysis_sha:
        raise AssertionError("analysis SHA-256 comment does not match matrix bytes")
    if parse_digest(source, "synthesis-le-f32-sha256") != synthesis_sha:
        raise AssertionError("synthesis SHA-256 comment does not match matrix bytes")


def test_task_metadata() -> None:
    program = json.loads((PROGRAM / "program.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (PROGRAM / "task_image.json").read_text(encoding="utf-8")
    )
    image = (PROGRAM / "task_image.bin").read_bytes()
    header = (GENERATED / "stem_task_metadata.h").read_text(encoding="ascii")

    activation_start = parse_macro(header, "STEM_TASK_ACTIVATION_OFFSET")
    activation_bytes = parse_macro(header, "STEM_TASK_ACTIVATION_BYTES")
    task_bytes = parse_macro(header, "STEM_TASK_IMAGE_BYTES")
    if task_bytes != len(image) or task_bytes % 64 != 0:
        raise AssertionError("task length or 64-byte alignment is wrong")
    if parse_macro(header, "STEM_TASK_COMMAND_COUNT") != manifest["command_count"]:
        raise AssertionError("task command count differs from the task manifest")
    if parse_digest(header, "task-image-sha256") != hashlib.sha256(image).hexdigest():
        raise AssertionError("task SHA-256 comment does not match task image")

    for prefix, entry_name, expected_channels in (
        ("INPUT", "input_tensor", 2),
        ("OUTPUT", "output_tensor", 4),
    ):
        tensor_index = int(program["entry"][entry_name])
        tensor = program["tensors"][tensor_index]
        absolute = activation_start + int(tensor["base_offset"])
        extent = int(tensor["allocation_bytes"])
        if parse_macro(header, f"STEM_TASK_{prefix}_OFFSET") != absolute:
            raise AssertionError(f"{prefix.lower()} offset was not derived correctly")
        if parse_macro(header, f"STEM_TASK_{prefix}_BYTES") != 32768:
            raise AssertionError(f"{prefix.lower()} tensor must occupy 32768 bytes")
        if extent != 32768:
            raise AssertionError(f"source {prefix.lower()} tensor size changed")
        if parse_macro(header, f"STEM_TASK_{prefix}_CHANNELS") != expected_channels:
            raise AssertionError(f"{prefix.lower()} channel count is wrong")
        if parse_macro(header, f"STEM_TASK_{prefix}_FREQUENCY_BANDS") != 128:
            raise AssertionError(f"{prefix.lower()} frequency extent is wrong")
        if parse_macro(header, f"STEM_TASK_{prefix}_TIME_FRAMES") != 16:
            raise AssertionError(f"{prefix.lower()} time extent is wrong")
        if tensor["layout"] != "NHWC8":
            raise AssertionError(f"source {prefix.lower()} tensor is not NHWC8")
        if not activation_start <= absolute <= absolute + extent <= (
            activation_start + activation_bytes
        ):
            raise AssertionError(f"{prefix.lower()} lies outside activation arena")
        if not 0 <= absolute <= absolute + extent <= task_bytes:
            raise AssertionError(f"{prefix.lower()} lies outside task image")

    payload_source = (GENERATED / "stem_task_payload.c").read_text(encoding="ascii")
    if parse_digest(payload_source, "task-image-sha256") != hashlib.sha256(
        image
    ).hexdigest():
        raise AssertionError("payload SHA-256 comment does not match task image")


def test_generators_do_not_embed_current_absolute_offsets() -> None:
    combined = FILTERBANK_GENERATOR.read_text(
        encoding="utf-8"
    ) + TASK_GENERATOR.read_text(encoding="utf-8")
    for forbidden in ("49280", "966784"):
        if forbidden in combined:
            raise AssertionError(f"generator embeds current absolute offset {forbidden}")


def main() -> None:
    run_generator(FILTERBANK_GENERATOR)
    run_generator(TASK_GENERATOR)
    first = generated_snapshot()
    run_generator(FILTERBANK_GENERATOR)
    run_generator(TASK_GENERATOR)
    second = generated_snapshot()
    if first != second:
        raise AssertionError("second generation was not byte-identical")

    run_generator(FILTERBANK_GENERATOR, "--check")
    run_generator(TASK_GENERATOR, "--check")
    test_filterbank()
    test_task_metadata()
    test_generators_do_not_embed_current_absolute_offsets()
    print("stem generators: PASS")


if __name__ == "__main__":
    main()
