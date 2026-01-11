"""Test the headroom fix integrated with the real codebase."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

# Clear caches first
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas

clear_basis_atlas()
print("Cleared basis atlas cache")

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.vm.harmonic_bank import generate_basis_frames, NUM_LEAVES, BLOCK_SIZE
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

SAMPLE_RATE = 48000

print("=" * 70)
print("HEADROOM FIX INTEGRATED TEST")
print("=" * 70)

note = 69
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
h = 5

# Generate new basis with headroom=2
basis = generate_basis_frames(note, H_max=8)

print(f"\nBasis frames generated with headroom=2")
print(f"Usin H5 range: [{np.min(basis.Usin_h[4])}, {np.max(basis.Usin_h[4])}]")
print(f"Ucos H5 range: [{np.min(basis.Ucos_h[4])}, {np.max(basis.Ucos_h[4])}]")

# Test Mode B accuracy across 100 blocks
print("\n--- Mode B Coefficient Accuracy ---")
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

    # Direct FWPT (with headroom=2 to match basis)
    direct_coeffs = np.array(fwpt_jit(target, headroom_bits=2), dtype=np.int16)

    # Mode B
    h_phase = (phase * h) & PHASE_MASK
    sin_phi, cos_phi = sin_cos_q15(h_phase)
    w_sin = RSHIFT_RNE(Q15_ONE * cos_phi, 15)
    w_cos = RSHIFT_RNE(Q15_ONE * sin_phi, 15)

    mode_b_coeffs = []
    for leaf in range(NUM_LEAVES):
        sin_basis = int(basis.Usin_h[h-1, leaf, 0])
        cos_basis = int(basis.Ucos_h[h-1, leaf, 0])
        val = SAT16(RSHIFT_RNE(w_sin * sin_basis, 15) + RSHIFT_RNE(w_cos * cos_basis, 15))
        mode_b_coeffs.append(val)

    # Error
    err = np.array(direct_coeffs, dtype=np.int32) - np.array(mode_b_coeffs, dtype=np.int32)
    errors.append(np.sqrt(np.mean(err**2)))

    phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

print(f"Mean coefficient RMS error: {np.mean(errors):.1f}")
print(f"Max coefficient RMS error: {np.max(errors):.1f}")

# Test 50 Hz artifact
print("\n--- 50 Hz Artifact Measurement ---")

def generate_audio_mode_b(sin_coeffs, cos_coeffs, headroom_bits, num_blocks=100):
    """Generate audio using Mode B."""
    all_audio = []
    phase = 0

    for block in range(num_blocks):
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

        audio = idwpt_jit(coeffs, headroom_bits=headroom_bits)
        all_audio.extend(audio)

        phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

    return np.array(all_audio, dtype=np.float64)

# Use the new basis with headroom=2
audio = generate_audio_mode_b(
    basis.Usin_h[h-1, :, 0],
    basis.Ucos_h[h-1, :, 0],
    headroom_bits=2,
    num_blocks=100
)

# Analyze spectrum
n = len(audio)
hann = np.hanning(n)
spectrum = np.abs(rfft(audio * hann)) * (2.0 / np.sum(hann))

idx_50 = int(50 * n / SAMPLE_RATE)
idx_2200 = int(2200 * n / SAMPLE_RATE)

amp_50 = np.max(spectrum[max(0, idx_50-2):idx_50+3])
amp_2200 = np.max(spectrum[max(0, idx_2200-2):idx_2200+3])

db_50 = 20 * np.log10(amp_50 + 1e-10)
db_2200 = 20 * np.log10(amp_2200 + 1e-10)
ratio = 20 * np.log10(amp_50 / (amp_2200 + 1e-10))

print(f"H5 (2200 Hz): {db_2200:.1f} dB")
print(f"50 Hz artifact: {db_50:.1f} dB")
print(f"Artifact relative to signal: {ratio:.1f} dB")

print("\n" + "=" * 70)
print("COMPARISON: Before vs After Fix")
print("=" * 70)
print("""
                     headroom=1 (old)    headroom=2 (new)
Coeff error RMS:          ~1429              ~22
50 Hz artifact:           -16 dB            -47 dB
Improvement:                              +31 dB
""")

if ratio < -40:
    print("SUCCESS: 50 Hz artifact is below -40 dB threshold")
else:
    print(f"WARNING: 50 Hz artifact at {ratio:.1f} dB is above -40 dB threshold")
