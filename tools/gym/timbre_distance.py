#!/usr/bin/env python
"""Perceptual committee scorer for patch reversal and preset refinement.

The scorer deliberately keeps diagnosis and judgment separate:

* multiscale spectrum, envelope, centroid, and level terms explain where two
  signals differ;
* Zimtohrli estimates whether the difference is audible, but runs in an
  isolated Pixi worker because its native binding must not own the optimizer;
* the bootstrap composite is useful for search ordering, but is explicitly
  uncalibrated until its weights reproduce the project's human-verdict corpus.

Examples::

    pixi run python tools/gym/timbre_distance.py reference.wav candidate.wav
    pixi run python tools/gym/timbre_distance.py reference.wav candidate.wav \
        --zimtohrli wsl --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("zimtohrli_worker.py")
SAMPLE_RATE = 48_000
MIN_RMS = 1e-7


@dataclass(frozen=True)
class BootstrapWeights:
    """Uncalibrated search weights; calibration must replace these."""

    zimtohrli: float = 0.45
    spectral: float = 0.30
    envelope: float = 0.10
    centroid: float = 0.10
    level: float = 0.05


def _float_audio(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.floating):
        out = samples.astype(np.float64)
    elif np.issubdtype(samples.dtype, np.signedinteger):
        scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
        out = samples.astype(np.float64) / scale
    elif np.issubdtype(samples.dtype, np.unsignedinteger):
        info = np.iinfo(samples.dtype)
        midpoint = (info.max + 1) / 2.0
        out = (samples.astype(np.float64) - midpoint) / midpoint
    else:
        raise TypeError(f"unsupported audio dtype: {samples.dtype}")
    if out.ndim == 2:
        out = out.mean(axis=1)
    if out.ndim != 1:
        raise ValueError(f"expected mono or interleaved audio, got shape {out.shape}")
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    sample_rate, samples = wavfile.read(Path(path))
    return _float_audio(samples), int(sample_rate)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    divisor = gcd(source_rate, target_rate)
    return resample_poly(samples, target_rate // divisor, source_rate // divisor)


def _rms(samples: np.ndarray) -> float:
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _onset(samples: np.ndarray, sample_rate: int) -> int:
    if not samples.size:
        return 0
    frame = max(32, int(round(0.005 * sample_rate)))
    energy = np.convolve(samples * samples, np.ones(frame) / frame, mode="same")
    peak = float(energy.max(initial=0.0))
    if peak <= MIN_RMS**2:
        return 0
    hits = np.flatnonzero(energy >= peak * 0.01)
    return int(hits[0]) if hits.size else 0


def prepare_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_rate: int,
    candidate_rate: int,
    *,
    align_onsets: bool = True,
    normalize_loudness: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ref = _resample(_float_audio(np.asarray(reference)), reference_rate, SAMPLE_RATE)
    cand = _resample(_float_audio(np.asarray(candidate)), candidate_rate, SAMPLE_RATE)
    ref_onset = _onset(ref, SAMPLE_RATE)
    cand_onset = _onset(cand, SAMPLE_RATE)
    if align_onsets:
        pre_roll = int(0.010 * SAMPLE_RATE)
        ref = ref[max(0, ref_onset - pre_roll):]
        cand = cand[max(0, cand_onset - pre_roll):]
    length = min(ref.size, cand.size)
    if length < 256:
        raise ValueError(f"audio pair is too short after preparation: {length} samples")
    ref = np.ascontiguousarray(ref[:length], dtype=np.float64)
    cand = np.ascontiguousarray(cand[:length], dtype=np.float64)

    ref_rms = _rms(ref)
    cand_rms = _rms(cand)
    level_delta_db = 20.0 * math.log10((cand_rms + 1e-12) / (ref_rms + 1e-12))
    if normalize_loudness and ref_rms >= MIN_RMS and cand_rms >= MIN_RMS:
        target_rms = 0.1
        ref = ref * (target_rms / ref_rms)
        cand = cand * (target_rms / cand_rms)
        peak = max(float(np.max(np.abs(ref))), float(np.max(np.abs(cand))), 1.0)
        ref /= peak
        cand /= peak

    meta = {
        "sample_rate": SAMPLE_RATE,
        "samples": int(length),
        "duration_s": length / SAMPLE_RATE,
        "reference_onset_samples": ref_onset,
        "candidate_onset_samples": cand_onset,
        "onset_delta_ms": (cand_onset - ref_onset) * 1000.0 / SAMPLE_RATE,
        "reference_rms": ref_rms,
        "candidate_rms": cand_rms,
        "level_delta_db": level_delta_db,
        "aligned": align_onsets,
        "loudness_normalized": normalize_loudness,
    }
    return ref, cand, meta


def _frames(samples: np.ndarray, size: int, hop: int) -> np.ndarray:
    if samples.size < size:
        samples = np.pad(samples, (0, size - samples.size))
    count = 1 + (samples.size - size) // hop
    starts = np.arange(count) * hop
    return np.stack([samples[start:start + size] for start in starts])


def _spectrogram(samples: np.ndarray, size: int) -> np.ndarray:
    framed = _frames(samples, size, max(1, size // 2))
    magnitude = np.abs(np.fft.rfft(framed * np.hanning(size), axis=1))
    magnitude /= np.maximum(magnitude.sum(axis=1, keepdims=True), 1e-12)
    return np.log1p(1000.0 * magnitude) / math.log1p(1000.0)


def multiscale_spectral_loss(reference: np.ndarray, candidate: np.ndarray) -> float:
    losses = []
    for size in (256, 1024, 4096):
        ref_spec = _spectrogram(reference, size)
        cand_spec = _spectrogram(candidate, size)
        frames = min(ref_spec.shape[0], cand_spec.shape[0])
        losses.append(float(np.mean(np.abs(ref_spec[:frames] - cand_spec[:frames]))))
    return float(np.mean(losses))


def _rms_envelope(samples: np.ndarray, size: int = 960, hop: int = 480) -> np.ndarray:
    framed = _frames(samples, size, hop)
    env = np.sqrt(np.mean(framed * framed, axis=1))
    return env / max(float(env.max(initial=0.0)), 1e-12)


def envelope_loss(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_env = _rms_envelope(reference)
    cand_env = _rms_envelope(candidate)
    length = min(ref_env.size, cand_env.size)
    return float(np.mean(np.abs(ref_env[:length] - cand_env[:length])))


def _centroid_trajectory(samples: np.ndarray) -> np.ndarray:
    size = 1024
    magnitude = np.abs(np.fft.rfft(_frames(samples, size, size // 2) * np.hanning(size), axis=1))
    frequencies = np.fft.rfftfreq(size, 1.0 / SAMPLE_RATE)
    centroid = (magnitude * frequencies).sum(axis=1) / np.maximum(magnitude.sum(axis=1), 1e-12)
    return centroid / (SAMPLE_RATE / 2.0)


def centroid_loss(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_centroid = _centroid_trajectory(reference)
    cand_centroid = _centroid_trajectory(candidate)
    length = min(ref_centroid.size, cand_centroid.size)
    return float(np.mean(np.abs(ref_centroid[:length] - cand_centroid[:length])))


def _write_pcm16(path: Path, samples: np.ndarray) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"cannot translate path to WSL: {resolved}")
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def _wsl_home(distro: str, timeout: float) -> str:
    result = subprocess.run(
        ["wsl.exe", "-d", distro, "--", "sh", "-lc", 'printf %s "$HOME"'],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    home = result.stdout.strip()
    if not home.startswith("/"):
        raise RuntimeError(f"invalid WSL home returned: {home!r}")
    return home


def isolated_zimtohrli(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    backend: str = "auto",
    distro: str = "Ubuntu-24.04",
    timeout: float = 30.0,
) -> dict[str, Any]:
    if backend == "off":
        return {"status": "disabled"}
    ref_rms, cand_rms = _rms(reference), _rms(candidate)
    if ref_rms < MIN_RMS and cand_rms < MIN_RMS:
        return {"status": "synthetic-silence", "distance": 0.0, "mos": 5.0}
    if ref_rms < MIN_RMS or cand_rms < MIN_RMS:
        return {"status": "skipped-silence", "error": "one side is silent"}

    use_wsl = backend == "wsl" or (backend == "auto" and os.name == "nt")
    with tempfile.TemporaryDirectory(prefix="symcrash_metric_") as temp:
        temp_path = Path(temp)
        ref_path = temp_path / "reference.wav"
        cand_path = temp_path / "candidate.wav"
        _write_pcm16(ref_path, reference)
        _write_pcm16(cand_path, candidate)
        if use_wsl:
            try:
                home = _wsl_home(distro, timeout)
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                return {
                    "status": "unavailable",
                    "backend": f"wsl-pixi:{distro}",
                    "error": str(exc),
                }
            command = [
                "wsl.exe", "-d", distro, "--",
                f"{home}/.local/bin/pixi", "run",
                "--manifest-path", _windows_to_wsl(REPO_ROOT / "pyproject.toml"),
                "--environment", "scoring", "--frozen",
                "python", _windows_to_wsl(WORKER),
                _windows_to_wsl(ref_path), _windows_to_wsl(cand_path),
            ]
            backend_name = f"wsl-pixi:{distro}"
        else:
            command = [
                "pixi", "run", "--environment", "scoring", "--frozen",
                "python", str(WORKER), str(ref_path), str(cand_path),
            ]
            backend_name = "local-pixi"
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=REPO_ROOT,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "unavailable", "backend": backend_name, "error": str(exc)}
        if result.returncode != 0:
            error = (result.stderr or result.stdout or "native worker failed").strip()
            return {
                "status": "failed",
                "backend": backend_name,
                "returncode": result.returncode,
                "error": error[-1000:],
            }
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            return {
                "status": "failed",
                "backend": backend_name,
                "error": f"invalid worker output: {exc}: {result.stdout[-500:]}",
            }
        return {"status": "ok", "backend": backend_name, **payload}


def score_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    reference_rate: int = SAMPLE_RATE,
    candidate_rate: int = SAMPLE_RATE,
    align_onsets: bool = True,
    normalize_loudness: bool = True,
    zimtohrli_backend: str = "auto",
    distro: str = "Ubuntu-24.04",
    weights: BootstrapWeights | None = None,
) -> dict[str, Any]:
    ref, cand, preparation = prepare_pair(
        reference,
        candidate,
        reference_rate,
        candidate_rate,
        align_onsets=align_onsets,
        normalize_loudness=normalize_loudness,
    )
    weights = weights or BootstrapWeights()
    metrics = {
        "multiscale_log_spectral": multiscale_spectral_loss(ref, cand),
        "envelope": envelope_loss(ref, cand),
        "centroid_trajectory": centroid_loss(ref, cand),
        "level": min(abs(preparation["level_delta_db"]) / 24.0, 1.0),
    }
    zim = isolated_zimtohrli(ref, cand, backend=zimtohrli_backend, distro=distro)
    if zim.get("status") in {"ok", "synthetic-silence"}:
        # The raw Zimtohrli distance is retained.  Scaling by a provisional
        # 0.012 JND makes it numerically commensurate with the diagnostic
        # terms; calibration must replace this constant before release use.
        raw_zim = float(zim["distance"])
        metrics["zimtohrli_raw"] = raw_zim
        metrics["zimtohrli_jnd_scaled"] = min(raw_zim / 0.012, 1.0)

    components = {
        "spectral": metrics["multiscale_log_spectral"],
        "envelope": metrics["envelope"],
        "centroid": metrics["centroid_trajectory"],
        "level": metrics["level"],
    }
    component_weights = {
        "spectral": weights.spectral,
        "envelope": weights.envelope,
        "centroid": weights.centroid,
        "level": weights.level,
    }
    if "zimtohrli_jnd_scaled" in metrics:
        components["zimtohrli"] = metrics["zimtohrli_jnd_scaled"]
        component_weights["zimtohrli"] = weights.zimtohrli
    active_weight = sum(component_weights[name] for name in components)
    composite = sum(components[name] * component_weights[name] for name in components) / active_weight

    return {
        "schema": "symcrash.timbre-distance.v0",
        "calibration": {
            "profile": "bootstrap-v0",
            "calibrated": False,
            "warning": "search-ordering only; human-verdict calibration is pending",
            "weights": asdict(weights),
        },
        "preparation": preparation,
        "metrics": metrics,
        "zimtohrli": zim,
        "composite": float(composite),
    }


def score_audio_files(
    reference_path: str | Path,
    candidate_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    reference, reference_rate = read_audio(reference_path)
    candidate, candidate_rate = read_audio(candidate_path)
    result = score_arrays(
        reference,
        candidate,
        reference_rate=reference_rate,
        candidate_rate=candidate_rate,
        **kwargs,
    )
    result["reference"] = str(Path(reference_path).resolve())
    result["candidate"] = str(Path(candidate_path).resolve())
    return result


def _print_human(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    zim = result["zimtohrli"]
    print(f"COMPOSITE (UNCALIBRATED): {result['composite']:.6f}")
    print(f"  multiscale spectral : {metrics['multiscale_log_spectral']:.6f}")
    print(f"  envelope            : {metrics['envelope']:.6f}")
    print(f"  centroid trajectory : {metrics['centroid_trajectory']:.6f}")
    print(f"  level penalty       : {metrics['level']:.6f}")
    if "zimtohrli_raw" in metrics:
        print(f"  Zimtohrli raw       : {metrics['zimtohrli_raw']:.9f}")
        print(f"  Zimtohrli MOS       : {zim['mos']:.6f}")
    else:
        print(f"  Zimtohrli           : {zim.get('status')} ({zim.get('error', 'no score')})")
    print("STATUS: advisory until bootstrap-v0 is calibrated against human verdicts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--zimtohrli",
        choices=("auto", "wsl", "local", "off", "required"),
        default="auto",
        help="isolated perceptual backend (default: auto)",
    )
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--keep-level", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    backend = "wsl" if args.zimtohrli == "required" and os.name == "nt" else args.zimtohrli
    if backend == "required" and os.name != "nt":
        backend = "local"
    result = score_audio_files(
        args.reference,
        args.candidate,
        align_onsets=not args.no_align,
        normalize_loudness=not args.keep_level,
        zimtohrli_backend=backend,
        distro=args.wsl_distro,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_human(result)
    if args.zimtohrli == "required" and result["zimtohrli"].get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
