"""Check pure sine for block-rate spurs."""
import wave
import numpy as np
from pathlib import Path

wav_path = Path(__file__).parent.parent / "audio_out" / "mode_b_test" / "mode_b_test_a440.wav"
with wave.open(str(wav_path), 'rb') as wf:
    framerate = wf.getframerate()
    raw_data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw_data, dtype=np.int16)[::2]  # Left channel

# FFT on pure sine
start = int(0.5 * framerate)
fft_size = 8192
chunk = samples[start:start + fft_size].astype(np.float64)
window = np.hanning(len(chunk))
spectrum = np.abs(np.fft.rfft(chunk * window))
freqs = np.fft.rfftfreq(len(chunk), 1/framerate)

max_val = np.max(spectrum)
spectrum_db = 20 * np.log10(spectrum / max_val + 1e-10)

print("Pure Sine (A440) Mode B - Top peaks:")
peak_indices = []
for i in range(1, len(spectrum_db) - 1):
    if spectrum_db[i] > spectrum_db[i-1] and spectrum_db[i] > spectrum_db[i+1] and spectrum_db[i] > -70:
        peak_indices.append(i)

sorted_peaks = sorted(peak_indices, key=lambda i: spectrum_db[i], reverse=True)[:10]
for i, idx in enumerate(sorted_peaks):
    print(f"  {i+1}. {freqs[idx]:.1f} Hz: {spectrum_db[idx]:.1f} dB")

# Check specifically for 440 +/- 750 Hz spurs
print()
print("Checking for block-rate spurs around 440 Hz:")
for offset in [-1500, -750, 750, 1500]:
    target = 440 + offset
    bin_idx = int(round(target * fft_size / framerate))
    if 0 < bin_idx < len(spectrum_db):
        print(f"  440{offset:+d} Hz = {target} Hz: {spectrum_db[bin_idx]:.1f} dB")
