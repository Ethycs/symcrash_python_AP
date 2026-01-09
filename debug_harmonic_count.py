"""Test block boundary discontinuity vs harmonic count."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    mode_b_generate_coeffs, get_basis_for_note, ModeBState
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

def test_harmonic_count(H_count, wave_type='square'):
    """Test boundary discontinuity for given harmonic count."""
    note = 60
    phase_inc = freq_to_phase_inc(261.63, 48000)

    mb_state = ModeBState()
    mb_state.reset(note=60, phase_inc=phase_inc)

    if wave_type == 'square':
        mb_state.weights.set_square(H=H_count)
    elif wave_type == 'triangle':
        mb_state.weights.set_triangle(H=H_count)
    elif wave_type == 'sine':
        mb_state.weights.set_sine()
        H_count = 1

    mb_state.H_active = H_count
    basis = get_basis_for_note(60)

    all_samples = []
    for block_idx in range(50):
        coeffs = np.zeros(512, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, coeffs)

        coeff_2d = coeffs.reshape(64, 8)
        k0_coeffs = coeff_2d[:, 0].tolist()
        audio = idwpt_jit(k0_coeffs, headroom_bits=1)
        all_samples.extend(audio)

        mb_state.advance_block()

    samples = np.array(all_samples, dtype=np.int16)

    # Calculate average boundary discontinuity
    boundary_diffs = []
    within_diffs = []
    for i in range(49):
        boundary_idx = (i + 1) * 64 - 1
        boundary_diffs.append(abs(int(samples[boundary_idx + 1]) - int(samples[boundary_idx])))

    for i in range(1000, len(samples) - 1):
        if i % 64 != 63:  # Not at boundary
            within_diffs.append(abs(int(samples[i + 1]) - int(samples[i])))

    avg_boundary = np.mean(boundary_diffs)
    avg_within = np.mean(within_diffs[:1000])  # Take first 1000
    ratio = avg_boundary / avg_within if avg_within > 0 else 0

    return avg_boundary, avg_within, ratio

print("=== Square wave: Boundary discontinuity vs harmonic count ===")
print(f"{'H':>4}  {'Boundary':>10}  {'Within':>10}  {'Ratio':>8}")
print("-" * 40)
for H in [1, 2, 4, 8, 16, 24, 32]:
    b, w, r = test_harmonic_count(H, 'square')
    print(f"{H:4d}  {b:10.1f}  {w:10.1f}  {r:8.2f}x")

print("\n=== Triangle wave: Boundary discontinuity vs harmonic count ===")
print(f"{'H':>4}  {'Boundary':>10}  {'Within':>10}  {'Ratio':>8}")
print("-" * 40)
for H in [1, 2, 4, 8, 16, 24, 32]:
    b, w, r = test_harmonic_count(H, 'triangle')
    print(f"{H:4d}  {b:10.1f}  {w:10.1f}  {r:8.2f}x")

# Also test saw
print("\n=== Sawtooth wave: Boundary discontinuity vs harmonic count ===")
print(f"{'H':>4}  {'Boundary':>10}  {'Within':>10}  {'Ratio':>8}")
print("-" * 40)
from symcrash_python_ap.vm.harmonic_bank import HarmonicWeights

def test_saw(H_count):
    note = 60
    phase_inc = freq_to_phase_inc(261.63, 48000)

    mb_state = ModeBState()
    mb_state.reset(note=60, phase_inc=phase_inc)
    mb_state.weights.set_saw(H=H_count)
    mb_state.H_active = H_count
    basis = get_basis_for_note(60)

    all_samples = []
    for block_idx in range(50):
        coeffs = np.zeros(512, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, coeffs)

        coeff_2d = coeffs.reshape(64, 8)
        k0_coeffs = coeff_2d[:, 0].tolist()
        audio = idwpt_jit(k0_coeffs, headroom_bits=1)
        all_samples.extend(audio)

        mb_state.advance_block()

    samples = np.array(all_samples, dtype=np.int16)

    boundary_diffs = []
    within_diffs = []
    for i in range(49):
        boundary_idx = (i + 1) * 64 - 1
        boundary_diffs.append(abs(int(samples[boundary_idx + 1]) - int(samples[boundary_idx])))

    for i in range(1000, len(samples) - 1):
        if i % 64 != 63:
            within_diffs.append(abs(int(samples[i + 1]) - int(samples[i])))

    avg_boundary = np.mean(boundary_diffs)
    avg_within = np.mean(within_diffs[:1000])
    ratio = avg_boundary / avg_within if avg_within > 0 else 0

    return avg_boundary, avg_within, ratio

for H in [1, 2, 4, 8, 16, 24, 32]:
    b, w, r = test_saw(H)
    print(f"{H:4d}  {b:10.1f}  {w:10.1f}  {r:8.2f}x")
