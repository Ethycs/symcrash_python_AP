"""Debug Mode B coefficient generation."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, mode_b_advance_phase,
    ModeBState, HarmonicWeights, PHASE_MASK, VM_K, NUM_LEAVES, clear_basis_atlas
)

# Clear any cached basis frames (to pick up algorithm changes)
clear_basis_atlas()
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_from_flat
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

SAMPLE_RATE = 48000
BLOCK_SIZE = 64

# Generate A440 basis frames
anchor_note = 69  # A4 = 440 Hz
basis = generate_basis_frames(anchor_note, H_max=16)
print(f"Anchor: {basis.anchor_note}, freq: {basis.anchor_freq_hz:.2f} Hz")

# Create state for A440
state = ModeBState()
state.reset(note=69, phase_inc=freq_to_phase_inc(440.0, SAMPLE_RATE))

# Set to sine wave
state.weights.set_sine()
state.H_active = 1
state.H_bandlimit = 16

print(f"Phase inc: {state.phase_inc}")
print(f"Phase: {state.phase}")
print(f"Weights: dc={state.weights.dc_q15}, a_h[0]={state.weights.a_h_q15[0]}, b_h[0]={state.weights.b_h_q15[0]}")

# Generate 4 blocks and look at the seams
all_audio = []
coeff_seams = []

for block_num in range(4):
    output_coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(state, basis, output_coeffs)

    # Store last coeff of k=7 and first coeff of k=0 (these should be continuous)
    coeff_k0_leaf0 = output_coeffs[0]  # leaf=0, k=0
    coeff_k7_leaf0 = output_coeffs[7]  # leaf=0, k=7

    # Run IDWPT
    coeff_2d = output_coeffs.reshape(64, 8)
    audio = idwpt_polyphase_from_flat(output_coeffs, K=8, headroom_bits=1)
    all_audio.append(audio)

    # Store phase and first/last samples for analysis
    coeff_seams.append({
        'block': block_num,
        'phase': state.phase,
        'coeff_k0_l0': coeff_k0_leaf0,
        'coeff_k7_l0': coeff_k7_leaf0,
        'audio_first': audio[0],
        'audio_last': audio[63],
    })

    mode_b_advance_phase(state)

# Analyze block boundary continuity
print("\nBlock boundary analysis:")
print("-" * 60)
for i, seam in enumerate(coeff_seams):
    print(f"Block {seam['block']}: phase={seam['phase']:10d}, "
          f"audio[0]={seam['audio_first']:6d}, audio[63]={seam['audio_last']:6d}")

# Check discontinuity at each block boundary
print("\nDiscontinuities at block boundaries:")
for i in range(1, len(coeff_seams)):
    prev_last = coeff_seams[i-1]['audio_last']
    curr_first = coeff_seams[i]['audio_first']
    disc = abs(curr_first - prev_last)
    print(f"  Block {i-1} to {i}: {prev_last} -> {curr_first} (jump={disc})")

# Concatenate audio and look at sample-by-sample
full_audio = np.concatenate(all_audio)
print(f"\nFull audio shape: {full_audio.shape}")

# Check diffs around block boundaries
for boundary in [63, 64, 127, 128, 191, 192]:
    if boundary < len(full_audio) - 1:
        diff = full_audio[boundary + 1] - full_audio[boundary]
        print(f"  Sample {boundary} to {boundary+1}: {full_audio[boundary]} -> {full_audio[boundary+1]} (diff={diff})")

# Compare with direct sine generation
print("\n\nReference: Direct sine at 440 Hz")
print("-" * 60)
from symcrash_python_ap.wavelet.sine_core import sine_q15
ref_phase = 0
phase_inc = freq_to_phase_inc(440.0, SAMPLE_RATE)
ref_samples = []
for i in range(256):
    ref_samples.append(sine_q15(ref_phase))
    ref_phase = (ref_phase + phase_inc) & PHASE_MASK

ref_samples = np.array(ref_samples, dtype=np.int16)

# Compare block boundaries in reference
for boundary in [63, 64, 127, 128, 191, 192]:
    if boundary < len(ref_samples) - 1:
        diff = ref_samples[boundary + 1] - ref_samples[boundary]
        print(f"  Sample {boundary} to {boundary+1}: {ref_samples[boundary]} -> {ref_samples[boundary+1]} (diff={diff})")

# Compare Mode B output to reference
print("\n\nMode B vs Reference comparison (first 64 samples):")
print("-" * 60)
for i in range(min(10, len(full_audio))):
    err = full_audio[i] - ref_samples[i]
    print(f"  [{i:2d}] Mode B: {full_audio[i]:6d}, Ref: {ref_samples[i]:6d}, Error: {err:6d}")

print(f"\n  Max error: {np.max(np.abs(full_audio[:256] - ref_samples[:256]))}")
print(f"  RMS error: {np.sqrt(np.mean((full_audio[:256].astype(np.float64) - ref_samples[:256].astype(np.float64))**2)):.1f}")
