"""Debug the fundamental basis frame issue causing H5 artifact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, BLOCK_SIZE, SAMPLE_RATE, VM_K, K_STRIDE, NUM_LEAVES
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK

# =============================================================================
# ROOT CAUSE ANALYSIS
# =============================================================================

print("=" * 70)
print("FUNDAMENTAL ANALYSIS: Why does H5 at 440 Hz create 50 Hz artifact?")
print("=" * 70)

# Parameters
freq_hz = 440.0
note = 69
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

print(f"\nSetup:")
print(f"  Base frequency: {freq_hz} Hz")
print(f"  Block size: {BLOCK_SIZE} samples")
print(f"  Sample rate: {SAMPLE_RATE} Hz")
print(f"  Phase increment: {phase_inc} (Q32)")

# Calculate block phase advance
block_phase_advance = (BLOCK_SIZE * phase_inc) & PHASE_MASK
cycles_per_block = BLOCK_SIZE * freq_hz / SAMPLE_RATE

print(f"\nBlock phase behavior:")
print(f"  Phase advance per block: {block_phase_advance} Q32 units")
print(f"  Cycles per block (H1): {cycles_per_block:.4f}")

# For H5
h5_cycles_per_block = 5 * cycles_per_block
h5_phase_advance = (5 * block_phase_advance) & PHASE_MASK

print(f"  Cycles per block (H5): {h5_cycles_per_block:.4f}")
print(f"  Fractional part (H5): {h5_cycles_per_block % 1:.4f}")

# =============================================================================
# THE ACTUAL BUG HUNT
# =============================================================================

print("\n" + "=" * 70)
print("EXAMINING COEFFICIENT GENERATION FOR H5")
print("=" * 70)

# Generate basis frames
basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

# Let's trace what happens for H5 (index 4) over multiple blocks
print("\nH5 basis frame examination:")
print(f"  Usin_h[4] shape: {basis.Usin_h[4].shape}")
print(f"  First few Usin values for k=0: {basis.Usin_h[4, :5, 0].tolist()}")

# Now let's manually trace the coefficient generation
print("\n--- Manual coefficient generation trace ---")

# Simulate 20 blocks of coefficient generation for H5 only
num_trace_blocks = 20
phase = 0
a_h = 32767  # Full amplitude for H5

print("\nBlock-by-block coefficient analysis for H5:")
coeff_traces = []

for block_idx in range(num_trace_blocks):
    # Compute sin(5*phi), cos(5*phi)
    h_phase = (phase * 5) & PHASE_MASK
    sin_phi, cos_phi = sin_cos_q15(h_phase)

    # The weight for Usin is: a_h * cos(h*phi) (simplified, no b_h)
    w_sin = (a_h * cos_phi) >> 15
    w_cos = (a_h * sin_phi) >> 15

    # Sample one coefficient (leaf 0, k=0)
    sin_basis = basis.Usin_h[4, 0, 0]
    cos_basis = basis.Ucos_h[4, 0, 0]

    coeff_contrib = ((w_sin * sin_basis) >> 15) + ((w_cos * cos_basis) >> 15)

    coeff_traces.append(coeff_contrib)

    if block_idx < 5:
        print(f"  Block {block_idx}: phase(Q32)={phase:10d}, sin_phi={sin_phi:6d}, "
              f"cos_phi={cos_phi:6d}, coeff_0={coeff_contrib:6d}")

    # Advance phase
    phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

# Analyze periodicity of coefficient values
coeff_arr = np.array(coeff_traces, dtype=np.float64)
print(f"\nCoefficient trace stats:")
print(f"  Mean: {np.mean(coeff_arr):.1f}")
print(f"  Std: {np.std(coeff_arr):.1f}")
print(f"  Min: {np.min(coeff_arr):.0f}")
print(f"  Max: {np.max(coeff_arr):.0f}")

# FFT of coefficient trace to find periodicity
if len(coeff_arr) > 10:
    coeff_spectrum = np.abs(rfft(coeff_arr - np.mean(coeff_arr)))
    coeff_freqs = np.arange(len(coeff_spectrum)) * (SAMPLE_RATE / BLOCK_SIZE) / len(coeff_arr)

    print(f"\nCoefficient modulation frequencies:")
    for i, (f, a) in enumerate(zip(coeff_freqs[:10], coeff_spectrum[:10])):
        if a > np.max(coeff_spectrum) * 0.1:
            print(f"  {f:.1f} Hz: amplitude = {a:.0f}")

# =============================================================================
# THE REAL ISSUE: Phase discretization in basis frames
# =============================================================================

print("\n" + "=" * 70)
print("THE CORE ISSUE: Basis Frame Phase Discretization")
print("=" * 70)

print("""
The basis frames are computed at discrete phase positions.
When we advance by BLOCK_SIZE samples, the phase advances by:

  delta_phase = BLOCK_SIZE * phase_inc

For H5 at 440 Hz:
  - H5 phase advance per block = 5 * 2.933... cycles = 14.667 cycles
  - This is NOT an integer, so H5's phase relative to block start
    slowly drifts through the cycle.

The basis frames Usin_h, Ucos_h are computed assuming the signal
starts at a fixed phase offset (k * K_STRIDE samples). But the
actual signal's H5 component has a different phase relationship
that changes block to block.

This creates a 50 Hz beat because:
  - H5 fractional cycles per block = 0.9333
  - Beat = block_rate * (1 - 0.9333) = 750 * 0.0667 = 50 Hz
""")

# =============================================================================
# VISUALIZE THE ISSUE
# =============================================================================

print("\n" + "=" * 70)
print("DEMONSTRATION: H5 Phase vs Block Boundary")
print("=" * 70)

# Show how H5's phase at block boundaries evolves
print("\nH5 phase at start of each block (in cycles):")
phase = 0
for block_idx in range(20):
    h5_phase_cycles = ((phase * 5) & PHASE_MASK) / (2**32)
    print(f"  Block {block_idx:2d}: H5 phase = {h5_phase_cycles:.4f} cycles "
          f"(fractional: {h5_phase_cycles % 1:.4f})")
    phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

# Calculate the expected beat pattern
print("\n--- Expected Beat Pattern ---")
block_rate = SAMPLE_RATE / BLOCK_SIZE
h5_phase_slip_per_block = (5 * cycles_per_block) % 1
beat_freq = block_rate * h5_phase_slip_per_block if h5_phase_slip_per_block < 0.5 else \
            block_rate * (1 - h5_phase_slip_per_block)
print(f"H5 phase slip per block: {h5_phase_slip_per_block:.4f} cycles")
print(f"Expected beat frequency: {beat_freq:.1f} Hz")
print(f"Beat period in blocks: {block_rate / beat_freq:.1f}")
