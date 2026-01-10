"""Debug k-slot phase alignment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.vm.harmonic_bank import VM_K, K_STRIDE, NUM_LEAVES, BLOCK_SIZE
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

SAMPLE_RATE = 48000

print("=" * 70)
print("K-SLOT PHASE ALIGNMENT DEBUG")
print(f"VM_K={VM_K}, K_STRIDE={K_STRIDE}")
print("=" * 70)

note = 69
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
h = 5  # H5 = 2200 Hz

print(f"\nH5 at 440 Hz base:")
print(f"  H5 freq: {freq_hz * h} Hz")
print(f"  phase_inc: {phase_inc}")
print(f"  phase_inc per sample (H5): {phase_inc * h}")

# Generate what the time-domain signal SHOULD look like for 2 blocks
print("\n--- Expected Time Domain Signal (H5 sine) ---")
expected_samples = []
phase = 0
for block in range(2):
    block_samples = []
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, _ = sin_cos_q15(h_phase)
        block_samples.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK
    expected_samples.extend(block_samples)

print(f"Expected boundary: block0[-1]={expected_samples[63]}, block1[0]={expected_samples[64]}")
print(f"Expected boundary diff: {expected_samples[64] - expected_samples[63]}")

# Now generate via Mode B approach WITHOUT k-slot phase offset (current bug)
print("\n--- Mode B WITHOUT k-slot phase offset (CURRENT) ---")
def generate_mode_b_current(base_phase, h, phase_inc):
    """Current implementation: same phase for all k-slots."""
    # Generate basis for each k-slot
    basis_sin = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
    basis_cos = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)

    for k in range(VM_K):
        k_phase_offset = (k * K_STRIDE * phase_inc) & PHASE_MASK

        sin_block = []
        cos_block = []
        phase = k_phase_offset
        for n in range(BLOCK_SIZE):
            h_phase = (phase * h) & PHASE_MASK
            s, c = sin_cos_q15(h_phase)
            sin_block.append(s)
            cos_block.append(c)
            phase = (phase + phase_inc) & PHASE_MASK

        basis_sin[:, k] = fwpt_jit(sin_block, headroom_bits=1)
        basis_cos[:, k] = fwpt_jit(cos_block, headroom_bits=1)

    # Modulate with runtime phase (SAME for all k-slots - this is the bug)
    coeffs = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
    h_phase = (base_phase * h) & PHASE_MASK
    sin_phi, cos_phi = sin_cos_q15(h_phase)

    a_h = Q15_ONE
    w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
    w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

    for k in range(VM_K):
        for leaf in range(NUM_LEAVES):
            sin_contrib = RSHIFT_RNE(w_sin * int(basis_sin[leaf, k]), 15)
            cos_contrib = RSHIFT_RNE(w_cos * int(basis_cos[leaf, k]), 15)
            coeffs[leaf, k] = SAT16(sin_contrib + cos_contrib)

    return coeffs

# Generate 2 blocks
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block

phase = 0
audio_current = []
for block in range(2):
    coeffs = generate_mode_b_current(phase, h, phase_inc)
    audio = idwpt_polyphase_block(coeffs, headroom_bits=1)
    audio_current.extend(audio.tolist())
    phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

print(f"Current boundary: block0[-1]={audio_current[63]}, block1[0]={audio_current[64]}")
print(f"Current boundary diff: {audio_current[64] - audio_current[63]}")

# Now generate via Mode B WITH k-slot phase offset (proposed fix)
print("\n--- Mode B WITH k-slot phase offset (PROPOSED FIX) ---")
def generate_mode_b_fixed(base_phase, h, phase_inc):
    """Fixed: apply k-slot phase offset at runtime."""
    # Generate basis for each k-slot
    basis_sin = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
    basis_cos = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)

    for k in range(VM_K):
        k_phase_offset = (k * K_STRIDE * phase_inc) & PHASE_MASK

        sin_block = []
        cos_block = []
        phase = k_phase_offset
        for n in range(BLOCK_SIZE):
            h_phase = (phase * h) & PHASE_MASK
            s, c = sin_cos_q15(h_phase)
            sin_block.append(s)
            cos_block.append(c)
            phase = (phase + phase_inc) & PHASE_MASK

        basis_sin[:, k] = fwpt_jit(sin_block, headroom_bits=1)
        basis_cos[:, k] = fwpt_jit(cos_block, headroom_bits=1)

    # Modulate with runtime phase (DIFFERENT for each k-slot - the fix)
    coeffs = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)

    for k in range(VM_K):
        # Apply k-slot phase offset at runtime too
        k_phase_offset = (k * K_STRIDE * phase_inc) & PHASE_MASK
        phi_k = (base_phase + k_phase_offset) & PHASE_MASK

        h_phase = (phi_k * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)

        a_h = Q15_ONE
        w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
        w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

        for leaf in range(NUM_LEAVES):
            sin_contrib = RSHIFT_RNE(w_sin * int(basis_sin[leaf, k]), 15)
            cos_contrib = RSHIFT_RNE(w_cos * int(basis_cos[leaf, k]), 15)
            coeffs[leaf, k] = SAT16(sin_contrib + cos_contrib)

    return coeffs

phase = 0
audio_fixed = []
for block in range(2):
    coeffs = generate_mode_b_fixed(phase, h, phase_inc)
    audio = idwpt_polyphase_block(coeffs, headroom_bits=1)
    audio_fixed.extend(audio.tolist())
    phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

print(f"Fixed boundary: block0[-1]={audio_fixed[63]}, block1[0]={audio_fixed[64]}")
print(f"Fixed boundary diff: {audio_fixed[64] - audio_fixed[63]}")

# Compare errors
print("\n--- Error Analysis ---")
expected = np.array(expected_samples[:128], dtype=np.float64)
current = np.array(audio_current[:128], dtype=np.float64)
fixed = np.array(audio_fixed[:128], dtype=np.float64)

err_current = expected - current
err_fixed = expected - fixed

print(f"Current implementation:")
print(f"  RMS error: {np.sqrt(np.mean(err_current**2)):.1f}")
print(f"  Max error: {np.max(np.abs(err_current)):.0f}")

print(f"\nFixed implementation:")
print(f"  RMS error: {np.sqrt(np.mean(err_fixed**2)):.1f}")
print(f"  Max error: {np.max(np.abs(err_fixed)):.0f}")
