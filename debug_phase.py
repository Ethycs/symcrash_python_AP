"""Debug phase alignment between basis frames and runtime."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    mode_b_generate_coeffs, get_basis_for_note, ModeBState, HarmonicWeights,
    VM_K, NUM_LEAVES, BLOCK_SIZE, PHASE_MASK
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sin_cos_q15, freq_to_phase_inc

# Generate 3 consecutive blocks with square wave and examine phase continuity

note = 60
phase_inc = freq_to_phase_inc(261.63, 48000)
print(f"Phase inc per sample: {phase_inc}")
print(f"Phase inc per block: {BLOCK_SIZE * phase_inc}")
print(f"Full cycle in Q32: {2**32}")
print(f"Samples per cycle: {2**32 / phase_inc:.1f}")
print(f"Blocks per cycle: {(2**32 / phase_inc) / BLOCK_SIZE:.2f}")

# Create Mode B state
mb_state = ModeBState()
mb_state.reset(note=60, phase_inc=phase_inc)
mb_state.weights.set_square(H=32)
mb_state.H_active = 32

basis = get_basis_for_note(60)

# Generate 10 blocks
all_samples = []
phases = []

for block_idx in range(10):
    phases.append(mb_state.phase)

    coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(mb_state, basis, coeffs)

    # IDWPT for k=0 only (like lane_ccs_render)
    coeff_2d = coeffs.reshape(64, 8)
    k0_coeffs = coeff_2d[:, 0].tolist()
    audio = idwpt_jit(k0_coeffs, headroom_bits=1)
    all_samples.extend(audio)

    mb_state.advance_block()

samples = np.array(all_samples, dtype=np.int16)

# Check for discontinuities at block boundaries
print("\n=== Block boundary discontinuities ===")
for i in range(9):
    boundary_idx = (i + 1) * 64 - 1
    diff = abs(int(samples[boundary_idx + 1]) - int(samples[boundary_idx]))
    print(f"Block {i}->{i+1}: last={samples[boundary_idx]:6d}, first={samples[boundary_idx+1]:6d}, diff={diff}")

# Now compare to what we'd expect from continuous sine
print("\n=== Expected from continuous sine(phi) ===")
expected_samples = []
phase = 0
for n in range(len(samples)):
    sin_val, _ = sin_cos_q15(phase)
    expected_samples.append(sin_val)
    phase = (phase + phase_inc) & PHASE_MASK

expected = np.array(expected_samples, dtype=np.int16)
print(f"Max deviation from expected: {np.max(np.abs(samples - expected))}")

# Check block boundary of pure sine (single harmonic)
print("\n=== Now test with SINE (single harmonic) ===")
mb_state2 = ModeBState()
mb_state2.reset(note=60, phase_inc=phase_inc)
mb_state2.weights.set_sine()
mb_state2.H_active = 1

all_samples2 = []
for block_idx in range(10):
    coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(mb_state2, basis, coeffs)

    coeff_2d = coeffs.reshape(64, 8)
    k0_coeffs = coeff_2d[:, 0].tolist()
    audio = idwpt_jit(k0_coeffs, headroom_bits=1)
    all_samples2.extend(audio)

    mb_state2.advance_block()

sine_samples = np.array(all_samples2, dtype=np.int16)

print("Sine block boundary discontinuities:")
for i in range(9):
    boundary_idx = (i + 1) * 64 - 1
    diff = abs(int(sine_samples[boundary_idx + 1]) - int(sine_samples[boundary_idx]))
    print(f"Block {i}->{i+1}: last={sine_samples[boundary_idx]:6d}, first={sine_samples[boundary_idx+1]:6d}, diff={diff}")

# Check how k-slots are related for the square wave
print("\n=== Analyzing k-slot phase relationship ===")
mb_state3 = ModeBState()
mb_state3.reset(note=60, phase_inc=phase_inc)
mb_state3.weights.set_square(H=32)
mb_state3.H_active = 32

coeffs3 = np.zeros(512, dtype=np.int16)
mode_b_generate_coeffs(mb_state3, basis, coeffs3)

# Extract k=0 to k=7 coefficient sets
print("First leaf (DC) coefficients across k-slots:")
for k in range(8):
    print(f"  k={k}: coeff={coeffs3[k]}")

# IDWPT each k-slot independently and check continuity
print("\nIDWPT output for each k-slot:")
for k in range(8):
    k_coeffs = coeffs3.reshape(64, 8)[:, k].tolist()
    audio_k = idwpt_jit(k_coeffs, headroom_bits=1)
    print(f"  k={k}: first sample = {audio_k[0]}, last sample = {audio_k[63]}")

# The issue: lane_ccs_render only uses k=0
# But the basis frames encode different phase offsets in each k-slot
# For continuous audio, we should be interleaving the k-slots somehow
