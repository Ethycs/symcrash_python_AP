"""FFT helpers for tools scripts."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows


def rfft_mag(samples: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return magnitude spectrum and frequency bins."""
    spectrum = np.abs(rfft(samples.astype(np.float64)))
    freqs = rfftfreq(len(samples), 1 / sample_rate)
    return spectrum, freqs


def rfft_mag_windowed(
    samples: np.ndarray,
    sample_rate: float,
    window_name: str = "blackman",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return windowed magnitude spectrum, frequency bins, and window gain."""
    window = windows.get_window(window_name, len(samples))
    windowed = samples.astype(np.float64) * window
    window_gain = np.sum(window) / len(window)
    spectrum = np.abs(rfft(windowed)) / window_gain
    freqs = rfftfreq(len(samples), 1 / sample_rate)
    return spectrum, freqs, window_gain
