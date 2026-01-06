"""Debug the phase model for Mode A vs Mode B."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, mode_b_advance_phase,
    ModeBState, PHASE_MASK, VM_K, NUM_LEAVES, clear_basis_atlas, BLOCK_SIZE
)
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15, sin_cos_q15
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_from_flat
from symcrash_python_ap.wavelet.coeff_lut_streaming import get_lut_for_note
from symcrash_python_ap.vm.state import CCSState

clear_basis_atlas()

SAMPLE_RATE = 48000
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

print("Phase Model Debug")
print("=" * 60)
print(f"Frequency: {freq_hz} Hz")
print(f"Phase inc: {phase_inc}")
print(f"Block size: {BLOCK_SIZE}")
print(f"K (polyphase lanes): {VM_K}")
print()

# The key question: what is the correct phase for each k-slot?
#
# In polyphase IDWPT with K=8:
# - k=0 produces samples at positions 0, 8, 16, 24, 32, 40, 48, 56
# - k=1 produces samples at positions 1, 9, 17, 25, 33, 41, 49, 57
# - etc.
#
# For a sine wave at sample n, we want sin(phase + n * phase_inc)
#
# So for k-slot k, the FIRST sample produced is at position k,
# which should have value sin(phase + k * phase_inc)

print("Mode A: What phase does it use for each k-slot?")
print("-" * 60)

lut = get_lut_for_note(0, 69)  # Sine at A440
state_a = CCSState()
state_a.reset(note=69, phase_inc=phase_inc, wave_id=0)

# Get coefficients from Mode A
output_a = np.zeros((64, 8), dtype=np.int16)
lut.get_subband_samples(state_a.phase, output_a)

print(f"Mode A starting phase: {state_a.phase}")
print()

# Now let's see what Mode A produces
flat_a = output_a.flatten()
audio_a = idwpt_polyphase_from_flat(flat_a, K=8, headroom_bits=1)

print("Mode A first 8 audio samples (what k-slots produce):")
for i in range(8):
    # Reference: what should sample i be?
    ref_phase = (state_a.phase + i * phase_inc) & PHASE_MASK
    ref_val = sine_q15(ref_phase)
    print(f"  Sample {i} (from k={i}): Mode A={audio_a[i]:6d}, Ref sin={ref_val:6d}, err={audio_a[i]-ref_val:4d}")

print()
print("Mode B: What phase should we use for each k-slot?")
print("-" * 60)

# The basis frames encode sin(h * n * phase_inc) for n=0..63
# But k-slot k is ROLLED by k, so it encodes samples at positions k, k+1, ..., 63, 0, 1, ..., k-1
# After IDWPT, k-slot k produces samples at positions k, k+8, k+16, ... (due to polyphase interleave)

# Wait - let me think about this more carefully.
# The FWPT input is 64 time-domain samples.
# For k-slot k, we ROLL the input by k, so:
#   - Input to FWPT: [sample_k, sample_{k+1}, ..., sample_63, sample_0, ..., sample_{k-1}]
#   - This is NOT the same as generating samples at positions k, k+8, k+16...

# Actually the polyphase structure is:
# - We have K=8 sets of coefficients
# - Each set has 64 values (one per WPT leaf)
# - After IDWPT, we get 64 audio samples
# - The interleave pattern means k-slot k contributes to samples k, k+8, k+16, ...

# So the question is: what is the relationship between the ROLLED time-domain input
# and the interleaved output?

print("Let's verify by generating basis frame directly and checking reconstruction:")
print()

basis = generate_basis_frames(69, H_max=16)
print(f"Basis anchor: {basis.anchor_freq_hz} Hz")

# Test: if we use Usin_h for h=1 with weight 1.0, we should get sin(n * phase_inc)
# after IDWPT

# For k=0: basis encodes sin(n * phase_inc) for n=0..63, no roll
# For k=1: basis encodes sin(n * phase_inc) for n=0..63, then rolled by 1

# Generate coeffs using just harmonic 1, unity weight
test_coeffs = np.zeros(512, dtype=np.int16)
for k in range(VM_K):
    for leaf in range(NUM_LEAVES):
        idx = leaf * VM_K + k
        # Just use Usin for h=1 (a pure sine)
        test_coeffs[idx] = basis.Usin_h[0, leaf, k]

test_audio = idwpt_polyphase_from_flat(test_coeffs, K=8, headroom_bits=1)

print("Basis Usin_h[h=1] reconstruction (no phase modulation):")
print("This should reconstruct sin(n * phase_inc) for n=0..63")
print()

for i in range(8):
    ref_phase = (i * phase_inc) & PHASE_MASK
    ref_val = sine_q15(ref_phase)
    print(f"  Sample {i}: Recon={test_audio[i]:6d}, Ref={ref_val:6d}, err={test_audio[i]-ref_val:4d}")

print()
print("Now let's see what happens when we modulate by a phase offset...")
print("-" * 60)

# If we want sin(phase + n * phase_inc), we use the trig identity:
#   sin(A + B) = sin(A)cos(B) + cos(A)sin(B)
# where A = phase (runtime), B = n * phase_inc (basis)
#
# So: sin(phase + n*phase_inc) = sin(phase)*cos(n*phase_inc) + cos(phase)*sin(n*phase_inc)
#                               = cos(phase)*Usin + sin(phase)*Ucos
#
# Wait, that's what the code does! Let me verify...

# For h=1, pure sine (a_1=1, b_1=0):
# w_sin = a_h*cos(phi) - b_h*sin(phi) = 1*cos(phi) - 0 = cos(phi)
# w_cos = a_h*sin(phi) + b_h*cos(phi) = 1*sin(phi) + 0 = sin(phi)
#
# coeff = w_sin * Usin + w_cos * Ucos
#       = cos(phi) * sin(n*phase_inc) + sin(phi) * cos(n*phase_inc)
#       = sin(n*phase_inc + phi)  [using sin(A+B) identity]
#
# Hmm, that's sin(n*phase_inc + phi), not sin(phi + n*phase_inc).
# They're the same thing! So the math is correct.

# Let me verify by manually computing:
test_phase = 0x10000000  # Some arbitrary phase

sin_phi, cos_phi = sin_cos_q15(test_phase)
print(f"Test phase: {test_phase}, sin={sin_phi}, cos={cos_phi}")

# Compute coefficients manually
manual_coeffs = np.zeros(512, dtype=np.int32)
for k in range(VM_K):
    for leaf in range(NUM_LEAVES):
        idx = leaf * VM_K + k
        # w_sin = cos(phi) for pure sine (a_h=1, b_h=0)
        # w_cos = sin(phi)
        from symcrash_python_ap.fixed import RSHIFT_RNE, SAT16
        sin_contrib = RSHIFT_RNE(cos_phi * int(basis.Usin_h[0, leaf, k]), 15)
        cos_contrib = RSHIFT_RNE(sin_phi * int(basis.Ucos_h[0, leaf, k]), 15)
        manual_coeffs[idx] = SAT16(sin_contrib + cos_contrib)

manual_audio = idwpt_polyphase_from_flat(manual_coeffs.astype(np.int16), K=8, headroom_bits=1)

print("\nManual phase-modulated reconstruction:")
for i in range(8):
    # Expected: sin(test_phase + i*phase_inc)
    ref_phase = (test_phase + i * phase_inc) & PHASE_MASK
    ref_val = sine_q15(ref_phase)
    print(f"  Sample {i}: Recon={manual_audio[i]:6d}, Ref={ref_val:6d}, err={manual_audio[i]-ref_val:4d}")

print()
print("=" * 60)
print("KEY INSIGHT: The phase model seems correct.")
print("The issue might be elsewhere...")
print("=" * 60)

# Let me check if the issue is in how Mode B uses the same phase for all k-slots
# vs Mode A which might use different phases

print()
print("Comparing Mode A vs Mode B coefficient access patterns:")
print("-" * 60)

# Mode A: For each phase P, it reads coefficients from LUT at phase P
# The LUT was pre-computed with coefficients that reconstruct the waveform
# when the polyphase structure is applied.

# Let's see what phases Mode A uses for the 8 k-slots
print("\nMode A gets_subband_samples at phase:", state_a.phase)
print("Mode A uses LINEAR INTERPOLATION between phase bins in the LUT")
print()

# The key difference might be:
# Mode A: Pre-computed LUT where each entry is the correct coefficient for that phase
# Mode B: We compute coefficients using basis modulation

# For a single block with phase P:
# - Sample 0 should be sin(P)
# - Sample 1 should be sin(P + phase_inc)
# - Sample k should be sin(P + k*phase_inc)
# ...
# - Sample 63 should be sin(P + 63*phase_inc)

# After the block, phase advances by 64*phase_inc

# In Mode B, we use phase P for ALL k-slots
# But the basis for k-slot k was generated by ROLLING a signal

# Let me check what the roll actually does to the reconstructed signal...

print("Testing: Does rolling the basis introduce time shift in reconstruction?")
print()

# Generate unrolled basis for k=0
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit

# Pure sine at anchor frequency
sin_block_unrolled = []
phase = 0
for n in range(BLOCK_SIZE):
    sin_block_unrolled.append(sine_q15(phase))
    phase = (phase + phase_inc) & PHASE_MASK

# FWPT of unrolled
coeffs_unrolled = fwpt_jit(sin_block_unrolled)

# FWPT of rolled by 1
sin_block_rolled_1 = sin_block_unrolled[1:] + sin_block_unrolled[:1]
coeffs_rolled_1 = fwpt_jit(sin_block_rolled_1)

print("Coeffs for k=0 (unrolled) first 8 leaves:", coeffs_unrolled[:8])
print("Coeffs for k=1 (rolled by 1) first 8 leaves:", coeffs_rolled_1[:8])
print()

# Reconstruct
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

recon_unrolled = idwpt_jit(coeffs_unrolled)
recon_rolled_1 = idwpt_jit(coeffs_rolled_1)

print("Reconstruction (first 4 samples):")
print(f"  Unrolled: {recon_unrolled[:4]}")
print(f"  Rolled 1: {recon_rolled_1[:4]}")
print(f"  Original (rolled 1): {sin_block_rolled_1[:4]}")
print()

# The rolled reconstruction should match the rolled input!
# So after IDWPT, rolled coeffs give rolled output.

# But in polyphase IDWPT, each k-slot's coeffs go through separate processing
# and then get INTERLEAVED. So the roll in the time domain doesn't directly
# translate to a time shift in the interleaved output.

print("=" * 60)
print("The issue might be the polyphase interleave pattern!")
print("=" * 60)
print()
print("In polyphase IDWPT:")
print("- k=0 coeffs produce samples at positions 0, 8, 16, ...")
print("- k=1 coeffs produce samples at positions 1, 9, 17, ...")
print()
print("The roll by k in basis generation was meant to handle the interleave,")
print("but the math might not work out correctly...")
print()

# Let me verify: what does Mode A's LUT structure look like?
print("Mode A LUT structure:")
print(f"  Shape: {lut.data.shape}")  # [64 leaves, K=8, P=1024 phases]
print(f"  For a given phase P, all K=8 slots read from the same phase bin")
print()

# Actually, let me look at what get_subband_samples does more carefully
print("Looking at Mode A's get_subband_samples implementation...")
