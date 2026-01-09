"""Test matrix method for Mode B."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, mode_b_generate_coeffs_matrix, mode_b_generate_coeffs,
    get_basis_for_note, PHASE_MASK
)
from symcrash_python_ap.vm.waveform_matrix import WaveformType
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

print("=== Testing Matrix Method for Mode B ===\n")

note = 60
freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)

# Create Mode B state
mb_state = ModeBState()
mb_state.reset(note=60, phase_inc=phase_inc)

print("Generating coefficient LUT for square wave...")
print("(This happens once per frequency, then cached)\n")

# Test matrix method
coeffs_matrix = np.zeros(512, dtype=np.int16)
mode_b_generate_coeffs_matrix(mb_state, WaveformType.SQUARE, coeffs_matrix)

# Test harmonic bank method for comparison
mb_state2 = ModeBState()
mb_state2.reset(note=60, phase_inc=phase_inc)
mb_state2.weights.set_square(H=32)
mb_state2.H_active = 32
basis = get_basis_for_note(60)
coeffs_harmonic = np.zeros(512, dtype=np.int16)
mode_b_generate_coeffs(mb_state2, basis, coeffs_harmonic)

# IDWPT both
audio_matrix = idwpt_jit(coeffs_matrix.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)
audio_harmonic = idwpt_jit(coeffs_harmonic.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)

print(f"Matrix method output (first 8 samples): {list(audio_matrix[:8])}")
print(f"Harmonic bank output (first 8 samples): {list(audio_harmonic[:8])}")

# Generate multiple blocks and check boundary discontinuity
print("\n=== Block boundary analysis ===")

def analyze_method(method_name, generate_fn, num_blocks=50):
    mb_state = ModeBState()
    mb_state.reset(note=60, phase_inc=phase_inc)
    if method_name == 'harmonic':
        mb_state.weights.set_square(H=32)
        mb_state.H_active = 32
        basis = get_basis_for_note(60)

    all_samples = []
    for block_idx in range(num_blocks):
        coeffs = np.zeros(512, dtype=np.int16)
        if method_name == 'matrix':
            mode_b_generate_coeffs_matrix(mb_state, WaveformType.SQUARE, coeffs)
        else:
            mode_b_generate_coeffs(mb_state, basis, coeffs)

        audio = idwpt_jit(coeffs.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)
        all_samples.extend(audio)
        mb_state.advance_block()

    samples = np.array(all_samples, dtype=np.int16)

    # Calculate boundary vs within discontinuity
    boundary_diffs = []
    within_diffs = []
    for i in range(num_blocks - 1):
        boundary_idx = (i + 1) * 64 - 1
        boundary_diffs.append(abs(int(samples[boundary_idx + 1]) - int(samples[boundary_idx])))

    for i in range(1000, len(samples) - 1):
        if i % 64 != 63:
            within_diffs.append(abs(int(samples[i + 1]) - int(samples[i])))

    avg_boundary = np.mean(boundary_diffs)
    avg_within = np.mean(within_diffs[:1000])
    ratio = avg_boundary / avg_within if avg_within > 0 else 0

    # Check flatness in top region
    top_mask = samples[1000:2000] > 10000
    top_std = np.std(samples[1000:2000][top_mask]) if np.sum(top_mask) > 0 else 0

    print(f"\n{method_name}:")
    print(f"  Avg boundary diff: {avg_boundary:.1f}")
    print(f"  Avg within diff:   {avg_within:.1f}")
    print(f"  Boundary/Within:   {ratio:.2f}x")
    print(f"  Top region std:    {top_std:.1f} (lower = flatter)")

    return samples

samples_matrix = analyze_method('matrix', mode_b_generate_coeffs_matrix)
samples_harmonic = analyze_method('harmonic', mode_b_generate_coeffs)

print("\n=== Comparison complete ===")
