"""Tests for the process-isolated perceptual committee scorer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "gym" / "timbre_distance.py"
SPEC = importlib.util.spec_from_file_location("timbre_distance", MODULE_PATH)
metric = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = metric
SPEC.loader.exec_module(metric)


def _tone(freq: float, seconds: float = 0.5, level: float = 0.2) -> np.ndarray:
    t = np.arange(int(metric.SAMPLE_RATE * seconds)) / metric.SAMPLE_RATE
    return level * np.sin(2.0 * np.pi * freq * t)


def test_identical_audio_scores_zero_without_native_backend():
    tone = _tone(440.0)
    result = metric.score_arrays(tone, tone, zimtohrli_backend="off")
    assert result["composite"] == 0.0
    assert result["metrics"]["multiscale_log_spectral"] == 0.0
    assert result["zimtohrli"]["status"] == "disabled"


def test_harmonic_change_is_detected_and_diagnosed():
    fundamental = _tone(220.0)
    brighter = fundamental + 0.12 * _tone(440.0, level=1.0)
    result = metric.score_arrays(fundamental, brighter, zimtohrli_backend="off")
    assert result["composite"] > 0.001
    assert result["metrics"]["multiscale_log_spectral"] > 0.001
    assert result["metrics"]["centroid_trajectory"] > 0.001


def test_onset_alignment_removes_leading_silence_penalty():
    tone = _tone(330.0)
    delayed = np.concatenate([np.zeros(2400), tone])
    reference = np.concatenate([tone, np.zeros(2400)])
    aligned = metric.score_arrays(reference, delayed, align_onsets=True, zimtohrli_backend="off")
    raw = metric.score_arrays(reference, delayed, align_onsets=False, zimtohrli_backend="off")
    assert aligned["composite"] < raw["composite"]
    assert aligned["preparation"]["onset_delta_ms"] > 40.0


def test_silence_never_calls_native_backend(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("native process must not be called")

    monkeypatch.setattr(metric.subprocess, "run", fail)
    silence = np.zeros(4800)
    result = metric.score_arrays(silence, silence, zimtohrli_backend="auto")
    assert result["zimtohrli"]["status"] == "synthetic-silence"
    assert result["composite"] == 0.0


def test_native_failure_falls_back_to_diagnostic_committee(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("backend unavailable")

    monkeypatch.setattr(metric.subprocess, "run", fail)
    result = metric.score_arrays(_tone(220.0), _tone(330.0), zimtohrli_backend="auto")
    assert result["zimtohrli"]["status"] == "unavailable"
    assert np.isfinite(result["composite"])
    assert result["composite"] > 0.0
