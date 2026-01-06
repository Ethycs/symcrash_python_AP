"""FFT analysis of Mode B Fur Elise output."""

import sys
import numpy as np
from pathlib import Path
import wave

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load the audio file
wav_path = Path(__file__).parent.parent / "audio_out" / "fur_elise" / "fur_elise_mode_b.wav"

with wave.open(str(wav_path), 'rb') as wf:
    n_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    framerate = wf.getframerate()
    n_frames = wf.getnframes()

    raw_data = wf.readframes(n_frames)
    samples = np.frombuffer(raw_data, dtype=np.int16)

    # Get left channel only
    if n_channels == 2:
        samples = samples[::2]

print(f"Loaded: {wav_path}")
print(f"Sample rate: {framerate} Hz")
print(f"Duration: {len(samples) / framerate:.2f}s")
print(f"Samples: {len(samples)}")
print()

# Analyze multiple segments
segments = [
    ("Opening (0.5s)", int(0.5 * framerate), 4096),
    ("Middle (3.0s)", int(3.0 * framerate), 4096),
    ("Later (6.0s)", int(6.0 * framerate), 4096),
]

print("=" * 70)
print("FFT Analysis of Fur Elise Mode B")
print("=" * 70)

for name, start, fft_size in segments:
    if start + fft_size > len(samples):
        print(f"\n{name}: Skipped (not enough samples)")
        continue

    chunk = samples[start:start + fft_size].astype(np.float64)

    # Check if there's signal
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms < 100:
        print(f"\n{name}: Skipped (silence, RMS={rms:.0f})")
        continue

    # Window and FFT
    window = np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(chunk * window))
    freqs = np.fft.rfftfreq(len(chunk), 1/framerate)

    # Normalize to dB
    max_val = np.max(spectrum)
    if max_val > 0:
        spectrum_db = 20 * np.log10(spectrum / max_val + 1e-10)
    else:
        print(f"\n{name}: Skipped (no signal)")
        continue

    # Find peaks
    peak_indices = []
    for i in range(1, len(spectrum_db) - 1):
        if spectrum_db[i] > spectrum_db[i-1] and spectrum_db[i] > spectrum_db[i+1] and spectrum_db[i] > -60:
            peak_indices.append(i)

    print(f"\n{name}:")
    print("-" * 50)
    print(f"  RMS: {rms:.0f}")

    # Top peaks
    sorted_peaks = sorted(peak_indices, key=lambda i: spectrum_db[i], reverse=True)[:15]
    print(f"  Top 15 peaks:")
    for i, idx in enumerate(sorted_peaks):
        freq = freqs[idx]
        db = spectrum_db[idx]
        # Try to identify if it's a musical note
        if freq > 20:
            midi = 69 + 12 * np.log2(freq / 440)
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_idx = int(round(midi)) % 12
            octave = int(round(midi)) // 12 - 1
            note_name = f"{note_names[note_idx]}{octave}"
        else:
            note_name = "?"
        print(f"    {i+1:2d}. {freq:7.1f} Hz ({note_name:4s}): {db:6.1f} dB")

    # Check for non-harmonic artifacts (spurs at block rate)
    block_rate = framerate / 64  # 750 Hz
    print(f"\n  Checking for block-rate artifacts (750 Hz intervals):")

    # Find the fundamental (strongest peak)
    if sorted_peaks:
        fund_freq = freqs[sorted_peaks[0]]
        print(f"    Fundamental: {fund_freq:.1f} Hz")

        # Look for spurs at fund +/- N*750
        spur_found = False
        for n in range(1, 5):
            for sign in [-1, 1]:
                spur_freq = fund_freq + sign * n * block_rate
                if 20 < spur_freq < framerate / 2:
                    # Find closest bin
                    bin_idx = int(round(spur_freq * len(chunk) / framerate))
                    if 0 < bin_idx < len(spectrum_db):
                        spur_db = spectrum_db[bin_idx]
                        if spur_db > -60:
                            print(f"    SPUR at {spur_freq:.1f} Hz: {spur_db:.1f} dB")
                            spur_found = True

        if not spur_found:
            print(f"    No block-rate spurs detected above -60 dB")

# Overall spectral analysis
print("\n" + "=" * 70)
print("Full-length spectral analysis (8192-sample FFT from 1s)")
print("=" * 70)

start = int(1.0 * framerate)
fft_size = 8192
if start + fft_size <= len(samples):
    chunk = samples[start:start + fft_size].astype(np.float64)
    window = np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(chunk * window))
    freqs = np.fft.rfftfreq(len(chunk), 1/framerate)

    max_val = np.max(spectrum)
    spectrum_db = 20 * np.log10(spectrum / max_val + 1e-10)

    # Find all peaks above -80 dB
    peak_indices = []
    for i in range(1, len(spectrum_db) - 1):
        if spectrum_db[i] > spectrum_db[i-1] and spectrum_db[i] > spectrum_db[i+1] and spectrum_db[i] > -80:
            peak_indices.append(i)

    print(f"Peaks above -80 dB: {len(peak_indices)}")

    # Check noise floor
    noise_floor = np.median(spectrum_db)
    print(f"Median (noise floor): {noise_floor:.1f} dB")

    # Check for any suspicious non-musical content
    sorted_peaks = sorted(peak_indices, key=lambda i: spectrum_db[i], reverse=True)
    print(f"\nTop 20 frequency components:")
    for i, idx in enumerate(sorted_peaks[:20]):
        freq = freqs[idx]
        db = spectrum_db[idx]
        if freq > 20:
            midi = 69 + 12 * np.log2(freq / 440)
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_idx = int(round(midi)) % 12
            octave = int(round(midi)) // 12 - 1
            note_name = f"{note_names[note_idx]}{octave}"
            # Check if it's close to a real note
            cents_off = (midi - round(midi)) * 100
        else:
            note_name = "?"
            cents_off = 0
        print(f"  {i+1:2d}. {freq:7.1f} Hz ({note_name:4s}, {cents_off:+5.1f} cents): {db:6.1f} dB")
