"""Debug coefficient accumulation for square wave."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, ModeBState,
    VM_K, NUM_LEAVES, BLOCK_SIZE, PHASE_MASK
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sin_cos_q15, freq_to_phase_inc
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

note = 60
freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)
basis = generate_basis_frames(note, H_max=32)

print("=== Testing coefficient accumulation by harmonic count ===\n")

def generate_square_coeffs(phase, H_count):
    """Generate square wave coefficients with given harmonic count."""
    output = np.zeros(512, dtype=np.int16)

    for k in range(VM_K):
        phi = phase

        for leaf in range(NUM_LEAVES):
            idx = leaf * VM_K + k
            output[idx] = 0

        for h in range(1, H_count + 1):
            if h % 2 == 0:  # Skip even harmonics
                continue

            a_h = int(Q15_ONE / h)  # 1/h normalized
            b_h = 0

            h_phase = (phi * h) & PHASE_MASK
            sin_phi, cos_phi = sin_cos_q15(h_phase)

            w_sin = RSHIFT_RNE(a_h * cos_phi - b_h * sin_phi, 15)
            w_cos = RSHIFT_RNE(a_h * sin_phi + b_h * cos_phi, 15)

            for leaf in range(NUM_LEAVES):
                idx = leaf * VM_K + k
                sin_contrib = RSHIFT_RNE(w_sin * int(basis.Usin_h[h - 1, leaf, k]), 15)
                cos_contrib = RSHIFT_RNE(w_cos * int(basis.Ucos_h[h - 1, leaf, k]), 15)
                output[idx] = SAT16(int(output[idx]) + sin_contrib + cos_contrib)

    return output

# Test different harmonic counts
for H in [1, 3, 7, 15, 31]:
    # Generate 3 consecutive blocks
    all_samples = []
    for block_idx in range(3):
        phase = (block_idx * BLOCK_SIZE * phase_inc) & PHASE_MASK
        coeffs = generate_square_coeffs(phase, H)

        coeff_2d = coeffs.reshape(64, 8)
        k0_coeffs = coeff_2d[:, 0].tolist()
        audio = idwpt_jit(k0_coeffs, headroom_bits=1)
        all_samples.extend(audio)

    samples = np.array(all_samples)

    # Check boundary discontinuity
    b01 = abs(int(samples[64]) - int(samples[63]))
    b12 = abs(int(samples[128]) - int(samples[127]))

    # Average within-block diff at sample 32
    w0 = abs(int(samples[33]) - int(samples[32]))
    w1 = abs(int(samples[97]) - int(samples[96]))

    print(f"H={H:2d}: boundary_0-1={b01:5d}, boundary_1-2={b12:5d}, within={w0:5d}/{w1:5d}")

# Now let's check what the coefficients look like for block 0 vs block 1
print("\n=== Coefficient comparison between blocks ===")
H = 31
phase0 = 0
phase1 = (BLOCK_SIZE * phase_inc) & PHASE_MASK

coeffs0 = generate_square_coeffs(phase0, H)
coeffs1 = generate_square_coeffs(phase1, H)

coeff_diff = np.abs(coeffs0.astype(np.int32) - coeffs1.astype(np.int32))
print(f"Max coeff diff between block 0 and 1: {np.max(coeff_diff)}")
print(f"Mean coeff diff: {np.mean(coeff_diff):.1f}")

# IDWPT both
audio0 = idwpt_jit(coeffs0.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)
audio1 = idwpt_jit(coeffs1.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)

print(f"\nBlock 0 last sample: {audio0[-1]}")
print(f"Block 1 first sample: {audio1[0]}")
print(f"Discontinuity: {abs(audio1[0] - audio0[-1])}")

# What SHOULD the audio be?
# At sample 63 of block 0: phase = 63 * phase_inc
# At sample 0 of block 1: phase = 64 * phase_inc
# These are consecutive, so the audio should be continuous

print("\n=== Reference: what should the audio be? ===")
# Generate reference square wave directly
from symcrash_python_ap.vm.waveform_matrix import waveform_sample_q15, WaveformType

ref_sample_63 = waveform_sample_q15(63 * phase_inc, WaveformType.SQUARE)
ref_sample_64 = waveform_sample_q15(64 * phase_inc, WaveformType.SQUARE)
print(f"Reference sample 63: {ref_sample_63}")
print(f"Reference sample 64: {ref_sample_64}")
print(f"Reference discontinuity: {abs(ref_sample_64 - ref_sample_63)}")

print("\nNote: Reference uses non-bandlimited lookup which has sharp edges.")
print("Mode B should produce bandlimited output with smoothed transitions.")
