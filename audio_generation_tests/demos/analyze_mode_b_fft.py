"""Detailed FFT analysis of Mode B sine wave output."""

import sys
import numpy as np
from pathlib import Path
import wave

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, mode_b_advance_phase,
    ModeBState, clear_basis_atlas
)
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15, PHASE_MASK
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_from_flat

SAMPLE_RATE = 48000
BLOCK_SIZE = 64

# Clear cache and regenerate
clear_basis_atlas()

print("=" * 70)
print("Mode B FFT Analysis - Looking for jitter sources")
print("=" * 70)

# Generate Mode B sine at A440
anchor_note = 69
basis = generate_basis_frames(anchor_note, H_max=16)

state = ModeBState()
state.reset(note=69, phase_inc=freq_to_phase_inc(440.0, SAMPLE_RATE))
state.weights.set_sine()
state.H_active = 1
state.H_bandlimit = 16

# Generate 1 second of audio
num_blocks = SAMPLE_RATE // BLOCK_SIZE
mode_b_samples = []

for _ in range(num_blocks):
    output_coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(state, basis, output_coeffs)
    audio = idwpt_polyphase_from_flat(output_coeffs, K=8, headroom_bits=1)
    mode_b_samples.extend(audio.tolist())
    mode_b_advance_phase(state)

mode_b_samples = np.array(mode_b_samples, dtype=np.float64)

# Generate reference sine
ref_phase = 0
phase_inc = freq_to_phase_inc(440.0, SAMPLE_RATE)
ref_samples = []
for _ in range(len(mode_b_samples)):
    ref_samples.append(sine_q15(ref_phase))
    ref_phase = (ref_phase + phase_inc) & PHASE_MASK
ref_samples = np.array(ref_samples, dtype=np.float64)

print(f"\nGenerated {len(mode_b_samples)} samples ({len(mode_b_samples)/SAMPLE_RATE:.2f}s)")

# FFT analysis
def analyze_spectrum(samples, name, fundamental_hz=440.0):
    print(f"\n{'='*60}")
    print(f"Spectrum Analysis: {name}")
    print(f"{'='*60}")

    # Use 8192 samples for good frequency resolution
    N = 8192
    chunk = samples[:N]

    # Window
    window = np.hanning(N)
    spectrum = np.abs(np.fft.rfft(chunk * window))
    freqs = np.fft.rfftfreq(N, 1/SAMPLE_RATE)

    # Convert to dB
    max_val = np.max(spectrum)
    spectrum_db = 20 * np.log10(spectrum / max_val + 1e-12)

    # Find fundamental
    fund_idx = np.argmin(np.abs(freqs - fundamental_hz))
    fund_freq = freqs[fund_idx]
    fund_db = spectrum_db[fund_idx]

    print(f"  Fundamental: {fund_freq:.1f} Hz @ {fund_db:.1f} dB")

    # Find all peaks above -80 dB
    peaks = []
    for i in range(2, len(spectrum_db) - 2):
        if (spectrum_db[i] > spectrum_db[i-1] and
            spectrum_db[i] > spectrum_db[i+1] and
            spectrum_db[i] > -80):
            peaks.append((freqs[i], spectrum_db[i]))

    # Sort by amplitude
    peaks.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top 15 spectral peaks:")
    harmonics = [fundamental_hz * (h+1) for h in range(50)]

    for i, (freq, db) in enumerate(peaks[:15]):
        # Check if harmonic
        is_harmonic = any(abs(freq - h) < 5 for h in harmonics)
        harmonic_num = None
        if is_harmonic:
            for h_idx, h in enumerate(harmonics):
                if abs(freq - h) < 5:
                    harmonic_num = h_idx + 1
                    break

        marker = f"H{harmonic_num}" if harmonic_num else "SPUR"
        print(f"    {i+1:2d}. {freq:8.1f} Hz: {db:6.1f} dB  [{marker}]")

    # Analyze spur content specifically
    print(f"\n  Non-harmonic content (spurs):")
    spurs = [(f, db) for f, db in peaks if not any(abs(f - h) < 5 for h in harmonics)]
    spurs.sort(key=lambda x: x[1], reverse=True)

    if spurs:
        for i, (freq, db) in enumerate(spurs[:10]):
            # Try to identify the spur pattern
            # Block-rate spurs would be at f0 +/- k * block_rate
            block_rate = SAMPLE_RATE / BLOCK_SIZE  # 750 Hz
            offset = (freq - fundamental_hz) % block_rate
            if offset > block_rate / 2:
                offset = offset - block_rate
            print(f"    {freq:8.1f} Hz: {db:6.1f} dB  (offset from harmonic: {offset:+.1f} Hz)")
    else:
        print("    None detected above -80 dB")

    return spectrum_db, freqs

# Analyze both
mode_b_db, freqs = analyze_spectrum(mode_b_samples, "Mode B Sine")
ref_db, _ = analyze_spectrum(ref_samples, "Reference Sine")

# Compare difference
print(f"\n{'='*60}")
print("Difference Analysis (Mode B - Reference)")
print(f"{'='*60}")

diff = mode_b_samples - ref_samples
diff_rms = np.sqrt(np.mean(diff**2))
print(f"  RMS difference: {diff_rms:.1f}")
print(f"  Max difference: {np.max(np.abs(diff)):.0f}")

# FFT of the difference (error signal)
N = 8192
diff_chunk = diff[:N]
window = np.hanning(N)
diff_spectrum = np.abs(np.fft.rfft(diff_chunk * window))
diff_freqs = np.fft.rfftfreq(N, 1/SAMPLE_RATE)

# Find peaks in error spectrum
diff_max = np.max(diff_spectrum)
if diff_max > 0:
    diff_db = 20 * np.log10(diff_spectrum / diff_max + 1e-12)

    print(f"\n  Error spectrum peaks:")
    peaks = []
    for i in range(2, len(diff_db) - 2):
        if (diff_db[i] > diff_db[i-1] and
            diff_db[i] > diff_db[i+1] and
            diff_db[i] > -40):
            peaks.append((diff_freqs[i], diff_db[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)
    for i, (freq, db) in enumerate(peaks[:10]):
        print(f"    {freq:8.1f} Hz: {db:6.1f} dB")

# Check for block-rate modulation
print(f"\n{'='*60}")
print("Block-rate analysis")
print(f"{'='*60}")
block_rate = SAMPLE_RATE / BLOCK_SIZE
print(f"  Block rate: {block_rate:.1f} Hz")
print(f"  Expected spur positions: 440 +/- N*{block_rate:.0f}")

# Look at envelope modulation
envelope = np.abs(mode_b_samples)
env_spectrum = np.abs(np.fft.rfft(envelope[:N] * window))
env_freqs = np.fft.rfftfreq(N, 1/SAMPLE_RATE)
env_max = np.max(env_spectrum[10:])  # Skip DC
env_db = 20 * np.log10(env_spectrum / env_max + 1e-12)

print(f"\n  Envelope modulation peaks (amplitude jitter):")
peaks = []
for i in range(10, len(env_db) - 2):  # Skip DC
    if (env_db[i] > env_db[i-1] and
        env_db[i] > env_db[i+1] and
        env_db[i] > -40):
        peaks.append((env_freqs[i], env_db[i]))

peaks.sort(key=lambda x: x[1], reverse=True)
for i, (freq, db) in enumerate(peaks[:5]):
    is_block_rate = abs(freq % block_rate) < 10 or abs(freq % block_rate - block_rate) < 10
    marker = " [BLOCK RATE!]" if is_block_rate else ""
    print(f"    {freq:8.1f} Hz: {db:6.1f} dB{marker}")
