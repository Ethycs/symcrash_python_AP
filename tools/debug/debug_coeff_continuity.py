"""Debug coefficient continuity across blocks for H5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, PHASE_MASK
from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, ModeBState, mode_b_generate_coeffs, VM_K, K_STRIDE,
    NUM_LEAVES, BLOCK_SIZE
)
from symcrash_python_ap.fixed import Q15_ONE

SAMPLE_RATE = 48000

print("=" * 70)
print("COEFFICIENT CONTINUITY DEBUG")
print(f"VM_K={VM_K}, K_STRIDE={K_STRIDE}")
print("=" * 70)

note = 69  # A4 = 440 Hz
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

print(f"\nNote: {note} ({freq_hz} Hz)")
print(f"phase_inc: {phase_inc}")
print(f"phase_inc per block: {BLOCK_SIZE * phase_inc}")

# Test with H5 only
print("\n--- Testing H5 (2200 Hz) alone ---")
mb_state = ModeBState()
mb_state.reset(note=note, phase_inc=phase_inc)
mb_state.weights.a_h_q15.fill(0)
mb_state.weights.a_h_q15[4] = Q15_ONE  # H5 only
mb_state.H_active = 5

basis = generate_basis_frames(note, H_max=8)

# Generate coefficients for 3 consecutive blocks
blocks_coeffs = []
for block_idx in range(3):
    coeffs = np.zeros(NUM_LEAVES * VM_K, dtype=np.int16)
    mode_b_generate_coeffs(mb_state, basis, coeffs)
    blocks_coeffs.append(coeffs.copy())

    print(f"\nBlock {block_idx}:")
    print(f"  Phase before: {mb_state.phase}")

    mb_state.advance_block()

    print(f"  Phase after: {mb_state.phase}")

# Analyze continuity: compare end of block N with start of block N+1
print("\n--- Coefficient Continuity Analysis ---")
for i in range(len(blocks_coeffs) - 1):
    c0 = blocks_coeffs[i]
    c1 = blocks_coeffs[i + 1]

    # Compare k=3 of block i with k=0 of block i+1 (should be continuous for interleaved)
    # Or compare all coefficients
    diff = c1.astype(np.int32) - c0.astype(np.int32)

    print(f"\nBlock {i} -> Block {i+1}:")
    print(f"  Max coeff diff: {np.max(np.abs(diff))}")
    print(f"  RMS coeff diff: {np.sqrt(np.mean(diff**2)):.1f}")

    # Check specific k-slots
    for k in range(VM_K):
        k_slice_0 = c0[k::VM_K]  # Coefficients for slot k
        k_slice_1 = c1[k::VM_K]
        k_diff = k_slice_1.astype(np.int32) - k_slice_0.astype(np.int32)
        print(f"  k={k}: max_diff={np.max(np.abs(k_diff))}, rms={np.sqrt(np.mean(k_diff**2)):.1f}")

# Now check what happens at IDWPT level
print("\n--- IDWPT Output Continuity ---")
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block

audio_blocks = []
for coeffs in blocks_coeffs:
    coeff_2d = coeffs.reshape(NUM_LEAVES, VM_K)
    audio = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
    audio_blocks.append(audio)

for i in range(len(audio_blocks) - 1):
    a0 = audio_blocks[i]
    a1 = audio_blocks[i + 1]

    # The last sample of block i should be close to first sample of block i+1
    boundary_diff = int(a1[0]) - int(a0[-1])

    # Also check second derivative (acceleration) at boundary
    if i > 0:
        a_prev = audio_blocks[i - 1]
        d2_before = int(a0[-1]) - 2*int(a0[-2]) + int(a0[-3])
        d2_after = int(a1[2]) - 2*int(a1[1]) + int(a1[0])
        d2_boundary = int(a1[0]) - 2*int(a0[-1]) + int(a0[-2])
    else:
        d2_before = 0
        d2_after = 0
        d2_boundary = 0

    print(f"\nBlock {i} -> Block {i+1}:")
    print(f"  Last sample block {i}: {a0[-1]}")
    print(f"  First sample block {i+1}: {a1[0]}")
    print(f"  Boundary discontinuity: {boundary_diff}")
    print(f"  d2 before boundary: {d2_before}")
    print(f"  d2 at boundary: {d2_boundary}")
    print(f"  d2 after boundary: {d2_after}")

# Expected signal value at boundaries
print("\n--- Expected vs Actual at Boundaries ---")
h = 5
for block_idx in range(3):
    block_start_phase = block_idx * BLOCK_SIZE * phase_inc

    # Expected value at start of block (H5 sine)
    h_phase = (block_start_phase * h) & PHASE_MASK
    from symcrash_python_ap.wavelet.sine_core import sin_cos_q15
    expected_sin, _ = sin_cos_q15(h_phase)

    actual = audio_blocks[block_idx][0] if block_idx < len(audio_blocks) else 0

    print(f"Block {block_idx} start: expected ~{expected_sin}, actual {actual}")
