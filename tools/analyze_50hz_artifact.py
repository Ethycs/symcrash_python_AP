"""Analyze violin audio for 50 Hz artifact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.io import wavfile
from numpy.fft import rfft, rfftfreq

# Load audio
audio_path = Path(__file__).parent.parent / "audio_generation_tests/audio_out/presets/violin_ir_single_note.wav"
sr, audio = wavfile.read(audio_path)
audio = audio.astype(np.float64)

print(f"Sample rate: {sr}")
print(f"Duration: {len(audio)/sr:.2f}s")
print(f"Samples: {len(audio)}")

# Skip attack, analyze sustain
start = int(0.5 * sr)  # Skip first 0.5s
end = int(2.5 * sr)    # Use 2s of sustain
segment = audio[start:end]

n = len(segment)
hann = np.hanning(n)
windowed = segment * hann
spectrum = np.abs(rfft(windowed)) * (2.0 / np.mean(hann))
freqs = rfftfreq(n, 1/sr)

# Find fundamental (should be around 440 Hz)
idx_440 = int(440 * n / sr)
amp_440 = spectrum[idx_440]

# Find 50 Hz
idx_50 = int(50 * n / sr)
amp_50 = spectrum[idx_50]

# Find 750 Hz (block rate)
idx_750 = int(750 * n / sr)
amp_750 = spectrum[idx_750]

print(f"\nSpectral analysis (sustain 0.5s-2.5s):")
print(f"  440 Hz (fundamental): {amp_440:.0f}")
print(f"  50 Hz: {amp_50:.0f} ({20*np.log10(amp_50/amp_440):.1f} dB rel)")
print(f"  750 Hz (block rate): {amp_750:.0f} ({20*np.log10(amp_750/amp_440):.1f} dB rel)")

# Check for spurs near 50 Hz
print(f"\nSpurs in 40-60 Hz range:")
for f in range(40, 61, 5):
    idx = int(f * n / sr)
    amp = spectrum[idx]
    db = 20*np.log10(amp/amp_440) if amp > 0 else -100
    print(f"  {f} Hz: {db:.1f} dB")

# Check harmonics
print(f"\n440 Hz harmonics:")
for h in range(1, 9):
    freq = 440 * h
    idx = int(freq * n / sr)
    if idx < len(spectrum):
        amp = spectrum[idx]
        db = 20*np.log10(amp/amp_440) if amp > 0 else -100
        print(f"  H{h} ({freq} Hz): {db:.1f} dB")
