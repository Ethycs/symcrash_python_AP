"""WAV helpers for tools scripts."""

from __future__ import annotations

from pathlib import Path
import wave
import numpy as np


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a WAV file into an int16 numpy array."""
    path = Path(path)
    with wave.open(str(path), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        rate = wav_file.getframerate()
        nframes = wav_file.getnframes()
        raw = wav_file.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(f"Unsupported WAV bit depth: {sampwidth * 8}-bit")

    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
    return rate, samples


def read_wav_mono(path: str | Path, channel: int = 0) -> tuple[int, np.ndarray]:
    """Read a WAV file and return a mono channel."""
    rate, samples = read_wav(path)
    if samples.ndim == 2:
        samples = samples[:, channel]
    return rate, samples
