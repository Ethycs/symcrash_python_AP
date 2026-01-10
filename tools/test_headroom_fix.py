"""Test: Does more headroom fix the Mode B coefficient error?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.vm.harmonic_bank import NUM_LEAVES, BLOCK_SIZE
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

SAMPLE_RATE = 48000

print("=" * 70)
print("HEADROOM FIX TEST")
print("=" * 70)

note = 69
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
h = 5

def generate_basis_with_headroom(headroom_bits):
    """Generate H5 basis frames with specified headroom."""
    sin_block = []
    cos_block = []
    phase = 0
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, cos_val = sin_cos_q15(h_phase)
        sin_block.append(sin_val)
        cos_block.append(cos_val)
        phase = (phase + phase_inc) & PHASE_MASK

    sin_coeffs = np.array(fwpt_jit(sin_block, headroom_bits=headroom_bits), dtype=np.int16)
    cos_coeffs = np.array(fwpt_jit(cos_block, headroom_bits=headroom_bits), dtype=np.int16)
    return sin_coeffs, cos_coeffs

def test_mode_b_accuracy(sin_coeffs, cos_coeffs, headroom_bits):
    """Test Mode B accuracy across 100 blocks."""
    errors = []
    phase = 0

    for block in range(100):
        # Generate target waveform
        target = []
        p = phase
        for n in range(BLOCK_SIZE):
            h_phase = (p * h) & PHASE_MASK
            sin_val, _ = sin_cos_q15(h_phase)
            target.append(sin_val)
            p = (p + phase_inc) & PHASE_MASK

        # Direct FWPT
        direct_coeffs = np.array(fwpt_jit(target, headroom_bits=headroom_bits), dtype=np.int16)

        # Mode B
        h_phase = (phase * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)
        w_sin = RSHIFT_RNE(Q15_ONE * cos_phi, 15)
        w_cos = RSHIFT_RNE(Q15_ONE * sin_phi, 15)

        mode_b_coeffs = []
        for leaf in range(NUM_LEAVES):
            sin_basis = int(sin_coeffs[leaf])
            cos_basis = int(cos_coeffs[leaf])
            val = SAT16(RSHIFT_RNE(w_sin * sin_basis, 15) + RSHIFT_RNE(w_cos * cos_basis, 15))
            mode_b_coeffs.append(val)

        # Error
        err = np.array(direct_coeffs, dtype=np.int32) - np.array(mode_b_coeffs, dtype=np.int32)
        errors.append(np.sqrt(np.mean(err**2)))

        phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

    return np.mean(errors), np.max(errors)

print("\n--- Mode B Coefficient Error vs Headroom ---")
print(f"{'Headroom':>10} {'Mean RMS':>12} {'Max RMS':>12}")
print("-" * 36)

for headroom_bits in [1, 2, 3, 4]:
    sin_coeffs, cos_coeffs = generate_basis_with_headroom(headroom_bits)
    mean_err, max_err = test_mode_b_accuracy(sin_coeffs, cos_coeffs, headroom_bits)
    print(f"{headroom_bits:>10} {mean_err:>12.1f} {max_err:>12.1f}")

print("\n" + "=" * 70)
print("50 Hz ARTIFACT TEST WITH DIFFERENT HEADROOM")
print("=" * 70)

def generate_audio_mode_b(sin_coeffs, cos_coeffs, headroom_bits, num_blocks=100):
    """Generate audio using Mode B and measure 50 Hz artifact."""
    all_audio = []
    phase = 0

    for block in range(num_blocks):
        # Mode B coefficients
        h_phase = (phase * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)
        w_sin = RSHIFT_RNE(Q15_ONE * cos_phi, 15)
        w_cos = RSHIFT_RNE(Q15_ONE * sin_phi, 15)

        coeffs = []
        for leaf in range(NUM_LEAVES):
            sin_basis = int(sin_coeffs[leaf])
            cos_basis = int(cos_coeffs[leaf])
            val = SAT16(RSHIFT_RNE(w_sin * sin_basis, 15) + RSHIFT_RNE(w_cos * cos_basis, 15))
            coeffs.append(val)

        # IDWPT to audio
        audio = idwpt_jit(coeffs, headroom_bits=headroom_bits)
        all_audio.extend(audio)

        phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

    return np.array(all_audio, dtype=np.float64)

print(f"\n{'Headroom':>10} {'50Hz dB':>12} {'2200Hz dB':>12} {'Ratio':>12}")
print("-" * 50)

for headroom_bits in [1, 2, 3, 4]:
    sin_coeffs, cos_coeffs = generate_basis_with_headroom(headroom_bits)
    audio = generate_audio_mode_b(sin_coeffs, cos_coeffs, headroom_bits, num_blocks=100)

    # Analyze spectrum
    n = len(audio)
    hann = np.hanning(n)
    spectrum = np.abs(rfft(audio * hann)) * (2.0 / np.sum(hann))

    # Find peaks
    idx_50 = int(50 * n / SAMPLE_RATE)
    idx_2200 = int(2200 * n / SAMPLE_RATE)

    # Use max in a small window around target
    amp_50 = np.max(spectrum[max(0, idx_50-2):idx_50+3])
    amp_2200 = np.max(spectrum[max(0, idx_2200-2):idx_2200+3])

    db_50 = 20 * np.log10(amp_50 + 1e-10)
    db_2200 = 20 * np.log10(amp_2200 + 1e-10)
    ratio = 20 * np.log10(amp_50 / (amp_2200 + 1e-10))

    print(f"{headroom_bits:>10} {db_50:>12.1f} {db_2200:>12.1f} {ratio:>12.1f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The coefficient error is caused by int16 quantization in the FWPT.
Using more headroom bits reduces the quantization error.

However, more headroom also reduces the dynamic range of the output.
The trade-off is:
- headroom_bits=1: Max dynamic range, but Mode B errors
- headroom_bits=2-3: Reduced errors, but 6-9 dB less headroom
- headroom_bits=4: Very low errors, but 12 dB less headroom

The proper fix may be to:
1. Use 32-bit coefficients internally
2. Or use Mode A (direct coefficient LUT) instead of Mode B for problematic harmonics
3. Or accept the artifact for harmonics that don't roundtrip cleanly
""")
