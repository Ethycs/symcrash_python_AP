"""Deterministic, side-effect-free exact render adapter (spec §6.3).

Calls the same IRFrameBuilder + vm_step_block path as tools/render/render_preset.py,
silences its console output, and returns exactly gate_s + tail_s seconds of audio.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from symcrash_python_ap.instrument_ir import InstrumentIR, load_instrument_from_dict

RENDERER_ID = "symcrash_inversion.render_service/v0"
SCHEMA_VERSIONS = {"instrument_ir": "v0", "probe_set": "symcrash.probe-set.v0", "render_request": "v0"}
Q15_ONE = 32767
_LEGACY = None


@dataclass(frozen=True)
class ControlPoint:
    time_s: float
    address: str
    value: float


@dataclass(frozen=True)
class RenderRequest:
    preset: Union[InstrumentIR, dict]
    note: int
    velocity: float                 # 0.0 .. 1.0 ; engine velocity is round(velocity * 32767)
    gate_s: float
    tail_s: float
    seed: int
    system_trajectory: Tuple[ControlPoint, ...] = ()


@dataclass
class RenderResult:
    audio: np.ndarray               # int16, shape (n,) mono or (2, n) stereo (authored unison)
    sample_rate_hz: int
    render_fingerprint: Dict[str, Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _legacy_renderer():
    """tools/render/render_preset.py loaded as a module (it is a script, not a package)."""
    global _LEGACY
    if _LEGACY is None:
        path = _repo_root() / "tools" / "render" / "render_preset.py"
        spec = importlib.util.spec_from_file_location("symcrash_inversion._legacy_render_preset", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
        _LEGACY = mod
    return _LEGACY


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("symcrash_python_ap")
    except Exception:  # noqa: BLE001
        return "unknown"


def velocity_q15(velocity: float) -> int:
    return max(0, min(Q15_ONE, int(round(float(velocity) * Q15_ONE))))


def render_fingerprint(seed: int, sample_rate_hz: int = 48000, block_size: int = 64) -> Dict[str, Any]:
    return {
        "git_revision": _git_revision(),
        "schema_versions": dict(SCHEMA_VERSIONS),
        "sample_rate_hz": int(sample_rate_hz),
        "block_size": int(block_size),
        "seed": int(seed),
        "renderer_id": RENDERER_ID,
        "compiler_version": _package_version(),
        "vm_version": _package_version(),
    }


def _as_ir(preset: Union[InstrumentIR, dict]) -> Tuple[InstrumentIR, int, int]:
    if isinstance(preset, dict):
        ir_dict = preset.get("instrument_ir_v0", preset)
        audio = ir_dict.get("audio") or {}
        sr = int(audio.get("sample_rate", 48000))
        bs = int(audio.get("block_size", 64))
        return load_instrument_from_dict(preset), sr, bs
    return preset, 48000, 64


def _fit_length(audio: np.ndarray, n: int) -> Tuple[np.ndarray, bool, bool]:
    cur = audio.shape[-1]
    if cur == n:
        return audio, False, False
    if cur > n:
        return audio[..., :n], False, True
    pad = [(0, 0)] * (audio.ndim - 1) + [(0, n - cur)]
    return np.pad(audio, pad, mode="constant"), True, False


def render(request: RenderRequest) -> RenderResult:
    """Exact render: the note is gated for gate_s, the preset's release follows, and the
    result is padded with silence or truncated so it holds exactly gate_s + tail_s seconds.
    No files are written and nothing is printed."""
    ir, sr, bs = _as_ir(request.preset)
    legacy = _legacy_renderer()
    vq15 = velocity_q15(request.velocity)
    unison = legacy.extract_unison_params(ir)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if unison:
            audio = legacy.render_preset_note(ir, note=int(request.note), duration_sec=float(request.gate_s),
                                              velocity_q15=vq15)
        else:
            audio = legacy._render_voice(ir, note=int(request.note), duration_sec=float(request.gate_s),
                                         velocity_q15=vq15, noise_seed_xor=int(request.seed), verbose=False)
    audio = np.asarray(audio)
    if audio.dtype != np.int16:
        audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)
    n_target = int(round((float(request.gate_s) + float(request.tail_s)) * sr))
    rendered = int(audio.shape[-1])
    audio, padded, truncated = _fit_length(audio, n_target)
    return RenderResult(
        audio=audio,
        sample_rate_hz=sr,
        render_fingerprint=render_fingerprint(request.seed, sr, bs),
        diagnostics={"velocity_q15": vq15, "rendered_samples": rendered, "target_samples": n_target,
                     "padded": padded, "truncated": truncated, "unison": bool(unison),
                     "system_trajectory_points": len(request.system_trajectory)},
    )
