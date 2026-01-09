"""Test flat region quality of square wave."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, mode_b_generate_coeffs_matrix, mode_b_generate_coeffs,
    get_basis_for_note, clear_waveform_coeff_lut_cache
)
from symcrash_python_ap.vm.waveform_matrix import WaveformType
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

print("=== Flat Region Quality Test ===\n")

# Clear cache to test fresh
clear_waveform_coeff_lut_cache()

note = 60
freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)

def analyze_flat_regions(method_name, generate_fn, num_blocks=100):
    """Analyze the quality of flat regions in square wave output."""
    mb_state = ModeBState()
    mb_state.reset(note=60, phase_inc=phase_inc)

    if 'harmonic' in method_name:
        mb_state.weights.set_square(H=32)
        mb_state.H_active = 32
        basis = get_basis_for_note(60)

    all_samples = []
    for _ in range(num_blocks):
        coeffs = np.zeros(512, dtype=np.int16)
        if 'matrix' in method_name:
            mode_b_generate_coeffs_matrix(mb_state, WaveformType.SQUARE, coeffs)
        else:
            mode_b_generate_coeffs(mb_state, basis, coeffs)

        audio = idwpt_jit(coeffs.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)
        all_samples.extend(audio)
        mb_state.advance_block()

    samples = np.array(all_samples, dtype=np.int16)

    # Skip first 1000 samples for settling
    samples = samples[1000:]

    # Find flat regions (top and bottom)
    # Square wave at C4 has ~183 samples per cycle
    # Top region: samples > 90% of max
    # Bottom region: samples < 90% of min

    max_val = np.max(samples)
    min_val = np.min(samples)

    top_mask = samples > (0.8 * max_val)
    bottom_mask = samples < (0.8 * min_val)

    top_samples = samples[top_mask]
    bottom_samples = samples[bottom_mask]

    print(f"\n{method_name}:")
    print(f"  Max: {max_val}, Min: {min_val}")
    print(f"  Top region: {len(top_samples)} samples")
    if len(top_samples) > 0:
        print(f"    Mean: {np.mean(top_samples):.1f}")
        print(f"    Std:  {np.std(top_samples):.1f} (lower = flatter)")
        print(f"    Range: {np.min(top_samples)} to {np.max(top_samples)}")
    print(f"  Bottom region: {len(bottom_samples)} samples")
    if len(bottom_samples) > 0:
        print(f"    Mean: {np.mean(bottom_samples):.1f}")
        print(f"    Std:  {np.std(bottom_samples):.1f}")
        print(f"    Range: {np.min(bottom_samples)} to {np.max(bottom_samples)}")

    return samples

# Test both methods
samples_matrix = analyze_flat_regions('matrix', mode_b_generate_coeffs_matrix)
samples_harmonic = analyze_flat_regions('harmonic', mode_b_generate_coeffs)

# Also test ideal bandlimited square (summing sines directly)
print("\n=== Reference: Direct harmonic synthesis (no wavelet) ===")
from symcrash_python_ap.wavelet.sine_core import sin_cos_q15

ref_samples = []
phase = 0
for i in range(6000):
    val = 0
    for h in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
        h_phase = (phase * h) & 0xFFFFFFFF
        s, _ = sin_cos_q15(h_phase)
        val += s // h
    ref_samples.append(np.clip(val, -32768, 32767))
    phase = (phase + phase_inc) & 0xFFFFFFFF

ref_samples = np.array(ref_samples[1000:], dtype=np.int16)

max_val = np.max(ref_samples)
min_val = np.min(ref_samples)
top_mask = ref_samples > (0.8 * max_val)
bottom_mask = ref_samples < (0.8 * min_val)

top = ref_samples[top_mask]
bottom = ref_samples[bottom_mask]

print(f"Direct synthesis:")
print(f"  Max: {max_val}, Min: {min_val}")
print(f"  Top region: {len(top)} samples")
if len(top) > 0:
    print(f"    Std: {np.std(top):.1f}")
print(f"  Bottom region: {len(bottom)} samples")
if len(bottom) > 0:
    print(f"    Std: {np.std(bottom):.1f}")
