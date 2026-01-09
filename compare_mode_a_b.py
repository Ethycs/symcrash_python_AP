"""Compare Mode A (LUT with matrix method) vs Mode B (harmonic bank) for square wave."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.coeff_lut_streaming import StreamingCoeffLUT, BLOCK_SIZE, NUM_LEAVES
from symcrash_python_ap.vm.waveform_matrix import waveform_sample_q15, WaveformType
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15, freq_to_phase_inc
from symcrash_python_ap.vm.harmonic_bank import (
    get_basis_for_note, ModeBState, mode_b_generate_coeffs, PHASE_MASK
)

print("=== Comparing Mode A (matrix) vs Mode B (harmonic bank) ===\n")

note = 60
freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)

# ==================================================
# Mode A: Generate LUT using matrix method (c-vector lookup)
# ==================================================
print("Generating Mode A LUT using matrix method...")

def square_matrix_sample(phase: int) -> int:
    """Square wave via matrix method (c-vector lookup)."""
    return waveform_sample_q15(phase, WaveformType.SQUARE)

mode_a_lut = StreamingCoeffLUT.generate_for_frequency(
    wave_id=99,
    name='square_matrix',
    target_freq_hz=freq_hz,
    generate_sample_fn=square_matrix_sample,
    P=1024,  # High resolution
    normalize=False,  # Don't normalize
)

# Generate blocks using Mode A
mode_a_samples = []
phase = 0
for block_idx in range(50):
    output = np.zeros((NUM_LEAVES, 8), dtype=np.int16)
    mode_a_lut.sample(phase, output)

    # IDWPT for k=0
    audio = idwpt_jit(output[:, 0].tolist(), headroom_bits=1)
    mode_a_samples.extend(audio)

    phase = (phase + BLOCK_SIZE * phase_inc) & 0xFFFFFFFF

mode_a_samples = np.array(mode_a_samples, dtype=np.int16)

# ==================================================
# Mode A: Generate LUT using harmonic synthesis (old method)
# ==================================================
print("Generating Mode A LUT using harmonic synthesis...")

def square_harmonic_sample(phase: int) -> int:
    """Square wave via harmonic synthesis."""
    val = 0
    for h in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
        h_phase = (phase * h) & 0xFFFFFFFF
        val += sine_q15(h_phase) // h
    return int(np.clip(val, -32768, 32767))

mode_a_harmonic_lut = StreamingCoeffLUT.generate_for_frequency(
    wave_id=98,
    name='square_harmonic',
    target_freq_hz=freq_hz,
    generate_sample_fn=square_harmonic_sample,
    P=1024,
    normalize=False,
)

mode_a_harmonic_samples = []
phase = 0
for block_idx in range(50):
    output = np.zeros((NUM_LEAVES, 8), dtype=np.int16)
    mode_a_harmonic_lut.sample(phase, output)

    audio = idwpt_jit(output[:, 0].tolist(), headroom_bits=1)
    mode_a_harmonic_samples.extend(audio)

    phase = (phase + BLOCK_SIZE * phase_inc) & 0xFFFFFFFF

mode_a_harmonic_samples = np.array(mode_a_harmonic_samples, dtype=np.int16)

# ==================================================
# Mode B: Harmonic bank
# ==================================================
print("Generating Mode B using harmonic bank...")

mb_state = ModeBState()
mb_state.reset(note=60, phase_inc=phase_inc)
mb_state.weights.set_square(H=32)
mb_state.H_active = 32

basis = get_basis_for_note(60)

mode_b_samples = []
for block_idx in range(50):
    coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(mb_state, basis, coeffs)

    coeff_2d = coeffs.reshape(64, 8)
    k0_coeffs = coeff_2d[:, 0].tolist()
    audio = idwpt_jit(k0_coeffs, headroom_bits=1)
    mode_b_samples.extend(audio)

    mb_state.advance_block()

mode_b_samples = np.array(mode_b_samples, dtype=np.int16)

# ==================================================
# Compare
# ==================================================
print("\n=== Block boundary analysis ===\n")

def analyze_boundary(samples, name):
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

    print(f"{name}:")
    print(f"  Avg boundary diff: {avg_boundary:.1f}")
    print(f"  Avg within diff:   {avg_within:.1f}")
    print(f"  Ratio:             {ratio:.2f}x")

    # Check flatness in top region
    top_mask = samples[1000:2000] > 10000
    if np.sum(top_mask) > 0:
        top_std = np.std(samples[1000:2000][top_mask])
        print(f"  Top region std:    {top_std:.1f}")
    print()

analyze_boundary(mode_a_samples, "Mode A (matrix method)")
analyze_boundary(mode_a_harmonic_samples, "Mode A (harmonic synthesis)")
analyze_boundary(mode_b_samples, "Mode B (harmonic bank)")

print("=== Sample values at flat region ===")
print("(Around sample 1500 - should be in flat top of square wave)\n")

for i in [1500, 1501, 1502, 1503, 1504]:
    print(f"Sample {i}: A_matrix={mode_a_samples[i]:6d}, A_harmonic={mode_a_harmonic_samples[i]:6d}, B={mode_b_samples[i]:6d}")
