#!/usr/bin/env python
"""Isolated Zimtohrli worker.

The third-party Python binding wraps native C++ and has been observed to abort
on degenerate input.  Keep it out of the long-running optimizer process: this
worker scores one already-prepared pair and exits.  It is intentionally small
and depends only on NumPy plus the ``zimtohrli`` package in the Pixi ``scoring``
environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 48_000
MIN_RMS = 1e-7


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if sample_width == 3:
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            u8[:, 0].astype(np.int32)
            | (u8[:, 1].astype(np.int32) << 8)
            | (u8[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width}")


def read_pcm_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    samples = _decode_pcm(raw, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), sample_rate


def score(reference: Path, candidate: Path) -> dict:
    ref, ref_rate = read_pcm_mono(reference)
    cand, cand_rate = read_pcm_mono(candidate)
    if ref_rate != SAMPLE_RATE or cand_rate != SAMPLE_RATE:
        raise ValueError(
            f"Zimtohrli requires 48 kHz input; got {ref_rate} and {cand_rate}")
    length = min(ref.size, cand.size)
    if length < 4096:
        raise ValueError(f"audio pair is too short for a stable score: {length} samples")
    ref = ref[:length]
    cand = cand[:length]
    ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    cand_rms = float(np.sqrt(np.mean(cand.astype(np.float64) ** 2)))
    if ref_rms < MIN_RMS or cand_rms < MIN_RMS:
        raise ValueError("Zimtohrli native binding is not called for silent input")

    import zimtohrli  # Native binding: import only inside this worker.

    metric = zimtohrli.Pyohrli()
    ref_spec = metric.analyze(ref)
    cand_spec = metric.analyze(cand)
    distance = float(metric.distance(ref_spec, cand_spec))
    mos = float(zimtohrli.mos_from_zimtohrli(distance))
    if not np.isfinite(distance) or not np.isfinite(mos):
        raise ValueError(f"non-finite Zimtohrli result: distance={distance}, mos={mos}")
    return {
        "distance": distance,
        "mos": mos,
        "sample_rate": SAMPLE_RATE,
        "samples": int(length),
        "binding": "zimtohrli.Pyohrli",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        result = score(args.reference, args.candidate)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
