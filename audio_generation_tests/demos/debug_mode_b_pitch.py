"""Debug Mode B pitch mismatch - C4 vs A3 anchor."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, mode_b_advance_phase,
    ModeBState, HarmonicWeights, PHASE_MASK, VM_K, NUM_LEAVES,
    clear_basis_atlas, get_basis_for_note
)

# Clear any cached basis frames
clear_basis_atlas()

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_from_flat

SAMPLE_RATE = 48000
BLOCK_SIZE = 64

# Test 1: Note 60 (C4) - what the demo uses
target_note = 60
target_freq = 440.0 * (2.0 ** ((target_note - 69) / 12.0))
print(f"Target: Note {target_note} (C4), freq: {target_freq:.2f} Hz")

# Get basis frames - this uses anchor!
basis = get_basis_for_note(target_note)
print(f"Basis anchor: Note {basis.anchor_note}, freq: {basis.anchor_freq_hz:.2f} Hz")
print(f"MISMATCH: Target {target_freq:.2f} Hz != Anchor {basis.anchor_freq_hz:.2f} Hz")
print()

# Create state for C4
state = ModeBState()
state.reset(note=target_note, phase_inc=freq_to_phase_inc(target_freq, SAMPLE_RATE))

# Set to sine wave
state.weights.set_sine()
state.H_active = 1
state.H_bandlimit = 16

print(f"Phase inc (for C4): {state.phase_inc}")
anchor_phase_inc = freq_to_phase_inc(basis.anchor_freq_hz, SAMPLE_RATE)
print(f"Phase inc (anchor A3): {anchor_phase_inc}")
print(f"Ratio: {state.phase_inc / anchor_phase_inc:.4f}")
print()

# Generate 10 blocks
all_audio = []
for block_num in range(10):
    output_coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(state, basis, output_coeffs)
    audio = idwpt_polyphase_from_flat(output_coeffs, K=8, headroom_bits=1)
    all_audio.append(audio)
    mode_b_advance_phase(state)

full_audio = np.concatenate(all_audio)

# FFT analysis
from scipy import fft
fft_result = np.abs(fft.rfft(full_audio.astype(np.float64)))
freqs = fft.rfftfreq(len(full_audio), 1.0/SAMPLE_RATE)

# Find peak
peak_idx = np.argmax(fft_result[1:]) + 1  # Skip DC
peak_freq = freqs[peak_idx]

print(f"FFT Analysis:")
print(f"  Expected fundamental: {target_freq:.2f} Hz")
print(f"  Detected peak: {peak_freq:.2f} Hz")
print(f"  Error: {peak_freq - target_freq:.2f} Hz ({1200 * np.log2(peak_freq/target_freq):.1f} cents)")
print()

# Compare with reference
print("Reference: Direct sine at C4")
ref_phase = 0
phase_inc = freq_to_phase_inc(target_freq, SAMPLE_RATE)
ref_samples = []
for i in range(len(full_audio)):
    ref_samples.append(sine_q15(ref_phase))
    ref_phase = (ref_phase + phase_inc) & PHASE_MASK

ref_samples = np.array(ref_samples, dtype=np.int16)

# FFT of reference
ref_fft = np.abs(fft.rfft(ref_samples.astype(np.float64)))
ref_peak_idx = np.argmax(ref_fft[1:]) + 1
ref_peak_freq = freqs[ref_peak_idx]

print(f"  Reference peak: {ref_peak_freq:.2f} Hz")
print()

# Check where the problem is
print("\nDiagnosis:")
print("=" * 60)
print(f"The basis frames were generated for {basis.anchor_freq_hz:.2f} Hz (A3)")
print(f"But we're running the oscillator at {target_freq:.2f} Hz (C4)")
print()
print("The basis frames encode sin/cos at the ANCHOR frequency's phase progression.")
print("When we use a different phase_inc at runtime, the frequency relationship breaks.")
print()

# Test 2: What if we generate basis frames for the EXACT target pitch?
print("\n" + "=" * 60)
print("FIX TEST: Generate basis frames for exact target pitch")
print("=" * 60)

# Generate basis for exact C4
exact_basis = generate_basis_frames(target_note, H_max=16)
print(f"Exact basis anchor: Note {exact_basis.anchor_note}, freq: {exact_basis.anchor_freq_hz:.2f} Hz")

# Reset state
state2 = ModeBState()
state2.reset(note=target_note, phase_inc=freq_to_phase_inc(target_freq, SAMPLE_RATE))
state2.weights.set_sine()
state2.H_active = 1
state2.H_bandlimit = 16

# Generate 10 blocks
all_audio2 = []
for block_num in range(10):
    output_coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(state2, exact_basis, output_coeffs)
    audio = idwpt_polyphase_from_flat(output_coeffs, K=8, headroom_bits=1)
    all_audio2.append(audio)
    mode_b_advance_phase(state2)

full_audio2 = np.concatenate(all_audio2)

# FFT analysis
fft_result2 = np.abs(fft.rfft(full_audio2.astype(np.float64)))
peak_idx2 = np.argmax(fft_result2[1:]) + 1
peak_freq2 = freqs[peak_idx2]

print(f"\nFFT Analysis (exact basis):")
print(f"  Expected fundamental: {target_freq:.2f} Hz")
print(f"  Detected peak: {peak_freq2:.2f} Hz")
print(f"  Error: {peak_freq2 - target_freq:.2f} Hz ({1200 * np.log2(peak_freq2/target_freq):.1f} cents)")

# Compare waveforms
max_err_mismatched = np.max(np.abs(full_audio.astype(float) - ref_samples.astype(float)))
max_err_exact = np.max(np.abs(full_audio2.astype(float) - ref_samples.astype(float)))
print(f"\nMax error vs reference (mismatched anchor): {max_err_mismatched:.0f}")
print(f"Max error vs reference (exact basis): {max_err_exact:.0f}")
