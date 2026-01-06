"""Debug polyphase interleaving in Mode B."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, mode_b_advance_phase,
    ModeBState, clear_basis_atlas, VM_K, NUM_LEAVES
)
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15, PHASE_MASK
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

SAMPLE_RATE = 48000
BLOCK_SIZE = 64

clear_basis_atlas()

# Generate basis at A440
anchor_note = 69
basis = generate_basis_frames(anchor_note, H_max=16)

state = ModeBState()
state.reset(note=69, phase_inc=freq_to_phase_inc(440.0, SAMPLE_RATE))
state.weights.set_sine()
state.H_active = 1

print("=" * 70)
print("Polyphase Interleave Debug")
print("=" * 70)

# Generate one block of coefficients
output_coeffs = np.zeros(512, dtype=np.int16)
mode_b_generate_coeffs(state, basis, output_coeffs)

# Reshape to [64, 8] for polyphase IDWPT
coeff_2d = output_coeffs.reshape(NUM_LEAVES, VM_K)

print(f"\nCoefficient array shape: {coeff_2d.shape}")
print(f"Phase at generation: {state.phase}")
print(f"Phase increment: {state.phase_inc}")

# Run IDWPT for each k-slot separately and look at the results
print(f"\nIDWPT outputs per k-slot (first 8 samples each):")
print("-" * 60)

k_outputs = []
for k in range(VM_K):
    coeffs_k = coeff_2d[:, k].astype(np.int16).tolist()
    audio_k = np.array(idwpt_jit(coeffs_k, headroom_bits=1), dtype=np.int16)
    k_outputs.append(audio_k)
    print(f"  k={k}: {audio_k[:8].tolist()}")

# The polyphase interleave takes y_k[0::K] and puts it at out[k::K]
print(f"\nPolyphase interleave pattern:")
print(f"  out[k::K] = y_k[0::K]")
print(f"  So out[0,8,16,...] = y_0[0,8,16,...], out[1,9,17,...] = y_1[0,8,16,...], etc.")

# Manually do the interleave
manual_out = np.zeros(64, dtype=np.int16)
for k in range(VM_K):
    manual_out[k::VM_K] = k_outputs[k][0::VM_K]

print(f"\nManual interleave first 16 samples:")
for i in range(16):
    k = i % VM_K
    src_idx = (i // VM_K) * VM_K  # Source index in y_k
    print(f"  out[{i:2d}] = y_{k}[{src_idx:2d}] = {manual_out[i]:6d}")

# Compare with expected sine values
phase_inc = freq_to_phase_inc(440.0, SAMPLE_RATE)
print(f"\nExpected sine values:")
phase = 0
for i in range(16):
    expected = sine_q15(phase)
    actual = manual_out[i]
    err = actual - expected
    print(f"  out[{i:2d}] = {actual:6d}, expected = {expected:6d}, error = {err:6d}")
    phase = (phase + phase_inc) & PHASE_MASK

# Now analyze what's happening
print(f"\n" + "=" * 70)
print("Analysis: What does y_k[0::K] actually represent?")
print("=" * 70)

# Each y_k is the IDWPT of coeff[:, k]
# The coeff[:, k] was generated from a time-shifted basis frame
# Specifically, for k-slot k, the basis was:
#   sin_block = [sin(n*phase_inc) for n in range(64)]
#   sin_shifted = sin_block[k:] + sin_block[:k]  # Roll by k
#   Usin[k] = FWPT(sin_shifted)
#
# At runtime, we modulate with phase phi:
#   coeff[k] = a_1 * sin(phi) * Usin[k] + ...
#
# The IDWPT of coeff[k] reconstructs: a_1 * sin(phi) * sin_shifted
# = a_1 * sin(phi) * sin((n-k) * phase_inc)
#
# Wait, that's the issue! The roll shifts the indices, so position n in
# the IDWPT output corresponds to time (n-k) in the original signal!

print("""
The rolled basis creates a TIME SHIFT in the IDWPT output.

For k-slot k with basis Usin_k = FWPT(sin_block[k:] + sin_block[:k]):
  - IDWPT(Usin_k) reconstructs the rolled signal
  - Position n in output corresponds to sample (n-k) mod 64 of original

But the polyphase interleave assumes:
  - out[k::K] = y_k[0::K]
  - This puts y_k[0] at out[k], y_k[K] at out[k+K], etc.

The issue is that y_k[0] is NOT sample 0 of the original signal!
Due to the roll, y_k[0] is actually sample (64-k) mod 64 = 64-k of original.

So when we interleave:
  - out[0] gets y_0[0] = sample 0 of original (correct)
  - out[1] gets y_1[0] = sample 63 of original (WRONG! Should be sample 1)
  - out[2] gets y_2[0] = sample 62 of original (WRONG! Should be sample 2)
  etc.

The fix: Don't roll the basis frames. Instead, generate them directly
for the correct k-slot phase relationship.
""")

print("\nVerifying the roll issue:")
print("-" * 60)

# Generate unrolled sine block
sin_unrolled = []
phase = 0
for n in range(BLOCK_SIZE):
    sin_unrolled.append(sine_q15(phase))
    phase = (phase + phase_inc) & PHASE_MASK

for k in range(3):
    # Roll by k (current approach)
    sin_rolled = sin_unrolled[k:] + sin_unrolled[:k]

    print(f"\nk={k}: Roll by {k}")
    print(f"  Original: sin_unrolled[0:4] = {sin_unrolled[0:4]}")
    print(f"  Rolled:   sin_rolled[0:4]   = {sin_rolled[0:4]}")
    print(f"  After roll, position 0 has what was at position {k}")
