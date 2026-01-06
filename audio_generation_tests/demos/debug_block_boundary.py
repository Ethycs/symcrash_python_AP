"""Debug block boundary discontinuity."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, mode_b_advance_phase,
    ModeBState, clear_basis_atlas, VM_K, NUM_LEAVES, BLOCK_SIZE, PHASE_MASK
)
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_from_flat

SAMPLE_RATE = 48000

clear_basis_atlas()

anchor_note = 69
basis = generate_basis_frames(anchor_note, H_max=16)

state = ModeBState()
phase_inc = freq_to_phase_inc(440.0, SAMPLE_RATE)
state.reset(note=69, phase_inc=phase_inc)
state.weights.set_sine()
state.H_active = 1

print("=" * 70)
print("Block Boundary Analysis")
print("=" * 70)

# Generate 2 blocks
blocks = []
phases_at_start = []

for block_num in range(2):
    phases_at_start.append(state.phase)
    output_coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(state, basis, output_coeffs)
    audio = idwpt_polyphase_from_flat(output_coeffs, K=8, headroom_bits=1)
    blocks.append(audio)
    mode_b_advance_phase(state)

# Concatenate
full_audio = np.concatenate(blocks)

print(f"\nPhase increment: {phase_inc}")
print(f"Block 0 start phase: {phases_at_start[0]}")
print(f"Block 1 start phase: {phases_at_start[1]}")
print(f"Expected block 1 phase: {(phases_at_start[0] + BLOCK_SIZE * phase_inc) & PHASE_MASK}")

# Show samples around boundary
print(f"\nSamples around block boundary (samples 60-68):")
print("-" * 60)

# Reference sine
ref_phase = 0
ref = []
for i in range(128):
    ref.append(sine_q15(ref_phase))
    ref_phase = (ref_phase + phase_inc) & PHASE_MASK

for i in range(60, 69):
    mode_b_val = full_audio[i]
    ref_val = ref[i]
    err = mode_b_val - ref_val
    boundary = " <-- BOUNDARY" if i == 63 or i == 64 else ""
    print(f"  Sample {i:2d}: Mode B = {mode_b_val:6d}, Ref = {ref_val:6d}, Error = {err:6d}{boundary}")

# Analyze the last samples of block 0 and first of block 1
print(f"\n\nBlock 0 last 8 samples (samples 56-63):")
for i in range(56, 64):
    k = i % VM_K
    print(f"  Sample {i:2d} (k={k}): Mode B = {full_audio[i]:6d}, Ref = {ref[i]:6d}, Err = {full_audio[i]-ref[i]:6d}")

print(f"\nBlock 1 first 8 samples (samples 64-71):")
for i in range(64, 72):
    k = i % VM_K
    print(f"  Sample {i:2d} (k={k}): Mode B = {full_audio[i]:6d}, Ref = {ref[i]:6d}, Err = {full_audio[i]-ref[i]:6d}")

# Check what sample 63 and 64 SHOULD be
print(f"\n\nPhase analysis:")
print("-" * 60)

# Sample 63 is the last sample of block 0
# For k=7, it's at position 7 + 7*8 = 63 in output
# So it comes from y_7[56] in the k=7 IDWPT output

# Actually wait - let me verify the interleave pattern
# out[k::K] = y_k[0::K]
# So:
#   out[0] = y_0[0], out[8] = y_0[8], out[16] = y_0[16], ..., out[56] = y_0[56]
#   out[1] = y_1[0], out[9] = y_1[8], out[17] = y_1[16], ..., out[57] = y_1[56]
#   ...
#   out[7] = y_7[0], out[15] = y_7[8], out[23] = y_7[16], ..., out[63] = y_7[56]

# So sample 63 = y_7[56]
# And sample 64 = y_0[0] of NEXT block

print(f"Sample 63 comes from: y_7[56] of block 0")
print(f"Sample 64 comes from: y_0[0] of block 1")

# What phase does y_7[56] correspond to?
# Block 0 uses phi = 0 for coefficient generation
# The basis for k=7 was generated with roll(sin_block, 7)
# So y_7 is IDWPT of coeffs that reconstruct the rolled signal
# y_7[56] reconstructs sample (56 + 7) mod 64 = 63 of the unrolled signal

# For the unrolled signal at phase phi=0:
#   sample n = sin(n * phase_inc)
#   sample 63 = sin(63 * phase_inc)

phase_sample_63 = (63 * phase_inc) & PHASE_MASK
expected_63 = sine_q15(phase_sample_63)
print(f"\nBlock 0, sample 63:")
print(f"  Phase: {phase_sample_63}")
print(f"  Expected sin value: {expected_63}")
print(f"  Actual Mode B value: {full_audio[63]}")

# For block 1, sample 0 (which is global sample 64):
# Block 1 starts at phi = BLOCK_SIZE * phase_inc
block1_start_phase = (BLOCK_SIZE * phase_inc) & PHASE_MASK
# Sample 0 of block 1 (k=0) = sin(block1_start_phase)
expected_64 = sine_q15(block1_start_phase)
print(f"\nBlock 1, sample 0 (global sample 64):")
print(f"  Phase: {block1_start_phase}")
print(f"  Expected sin value: {expected_64}")
print(f"  Actual Mode B value: {full_audio[64]}")

# But wait - the coefficient generation uses phi (block start phase)
# and doesn't account for the k-slot time offset!
# That's the bug!

print(f"\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print("""
The issue is in mode_b_generate_coeffs:

Current code uses: phi = state.phase (same for all k-slots)

But the basis frames Usin_k were generated as FWPT of a rolled signal.
Rolling shifts the TIME index, not the PHASE.

So Usin_k at phase phi reconstructs: sin((n + k) * phase_inc + phi)
                             NOT:    sin(n * phase_inc + phi + k * phase_inc)

This means k-slot k naturally produces samples SHIFTED by k positions.
The polyphase interleave then puts y_k[0] at position k, which is correct!

But the phase advance at block boundary is wrong. We advance by
BLOCK_SIZE * phase_inc, which is 64 samples worth of phase.

However, each k-slot's output y_k covers samples k, k+8, k+16, ..., k+56
(8 samples per k-slot, interleaved).

At the block boundary:
- Last sample of block 0 is out[63] = y_7[56], representing sample 63
- First sample of block 1 is out[64] = y_0[0] of next block

For continuity, y_0[0] of block 1 should equal sin(64 * phase_inc).

Currently, block 1's phase is set to (64 * phase_inc).
Coefficient generation uses phi = 64 * phase_inc.
y_0 is IDWPT of coeffs using Usin_0 (rolled by 0 = no roll).
So y_0[0] = sin(0 * phase_inc + 64 * phase_inc) = sin(64 * phase_inc). Correct!

Hmm, the math seems right. Let me verify the actual phase advance...
""")

print(f"\nVerifying phase advance:")
print(f"  phase_inc = {phase_inc}")
print(f"  BLOCK_SIZE = {BLOCK_SIZE}")
print(f"  BLOCK_SIZE * phase_inc = {BLOCK_SIZE * phase_inc}")
print(f"  Expected phase after 1 block: {(BLOCK_SIZE * phase_inc) & PHASE_MASK}")
print(f"  Actual phases_at_start[1]: {phases_at_start[1]}")
print(f"  Match: {phases_at_start[1] == (BLOCK_SIZE * phase_inc) & PHASE_MASK}")
