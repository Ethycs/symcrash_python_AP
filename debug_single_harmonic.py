"""Debug single harmonic reconstruction through Mode B."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, VM_K, NUM_LEAVES, BLOCK_SIZE, PHASE_MASK
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sin_cos_q15, freq_to_phase_inc
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

note = 60
freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)
basis = generate_basis_frames(note, H_max=32)

print("=== Single harmonic (h=1) reconstruction ===\n")

def generate_single_harmonic_block(phase, h, basis):
    """Generate block with single harmonic using Mode B method."""
    output = np.zeros(512, dtype=np.int16)

    for k in range(VM_K):
        phi = phase  # Same phase for all k (as in mode_b_generate_coeffs)

        h_phase = (phi * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)

        # a_h = 1 (sine coefficient), b_h = 0
        a_h = Q15_ONE
        w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
        w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

        for leaf in range(NUM_LEAVES):
            idx = leaf * VM_K + k
            sin_contrib = RSHIFT_RNE(w_sin * int(basis.Usin_h[h - 1, leaf, k]), 15)
            cos_contrib = RSHIFT_RNE(w_cos * int(basis.Ucos_h[h - 1, leaf, k]), 15)
            output[idx] = SAT16(sin_contrib + cos_contrib)

    return output

def generate_reference_block(start_phase, h, phase_inc):
    """Generate reference sine block directly (no wavelet)."""
    block = []
    phase = start_phase
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, _ = sin_cos_q15(h_phase)
        block.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK
    return block

# Test h=1
print("h=1 (fundamental):")
for block_idx in range(3):
    phase = (block_idx * BLOCK_SIZE * phase_inc) & PHASE_MASK

    coeffs = generate_single_harmonic_block(phase, 1, basis)
    audio = idwpt_jit(coeffs.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)

    ref = generate_reference_block(phase, 1, phase_inc)

    # Compare first and last samples
    print(f"  Block {block_idx}:")
    print(f"    First: Mode B={audio[0]:6d}, Ref={ref[0]:6d}, diff={abs(audio[0]-ref[0])}")
    print(f"    Last:  Mode B={audio[-1]:6d}, Ref={ref[-1]:6d}, diff={abs(audio[-1]-ref[-1])}")

# Test h=3
print("\nh=3 (third harmonic):")
for block_idx in range(3):
    phase = (block_idx * BLOCK_SIZE * phase_inc) & PHASE_MASK

    coeffs = generate_single_harmonic_block(phase, 3, basis)
    audio = idwpt_jit(coeffs.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)

    ref = generate_reference_block(phase, 3, phase_inc)

    print(f"  Block {block_idx}:")
    print(f"    First: Mode B={audio[0]:6d}, Ref={ref[0]:6d}, diff={abs(audio[0]-ref[0])}")
    print(f"    Last:  Mode B={audio[-1]:6d}, Ref={ref[-1]:6d}, diff={abs(audio[-1]-ref[-1])}")

# Test h=15
print("\nh=15 (high harmonic):")
for block_idx in range(3):
    phase = (block_idx * BLOCK_SIZE * phase_inc) & PHASE_MASK

    coeffs = generate_single_harmonic_block(phase, 15, basis)
    audio = idwpt_jit(coeffs.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)

    ref = generate_reference_block(phase, 15, phase_inc)

    print(f"  Block {block_idx}:")
    print(f"    First: Mode B={audio[0]:6d}, Ref={ref[0]:6d}, diff={abs(audio[0]-ref[0])}")
    print(f"    Last:  Mode B={audio[-1]:6d}, Ref={ref[-1]:6d}, diff={abs(audio[-1]-ref[-1])}")

# Test h=31
print("\nh=31 (very high harmonic):")
for block_idx in range(3):
    phase = (block_idx * BLOCK_SIZE * phase_inc) & PHASE_MASK

    coeffs = generate_single_harmonic_block(phase, 31, basis)
    audio = idwpt_jit(coeffs.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)

    ref = generate_reference_block(phase, 31, phase_inc)

    max_diff = max(abs(a - r) for a, r in zip(audio, ref))
    rms_diff = np.sqrt(np.mean([(a - r)**2 for a, r in zip(audio, ref)]))

    print(f"  Block {block_idx}:")
    print(f"    Max diff: {max_diff}, RMS diff: {rms_diff:.1f}")
    print(f"    First: Mode B={audio[0]:6d}, Ref={ref[0]:6d}")
    print(f"    Last:  Mode B={audio[-1]:6d}, Ref={ref[-1]:6d}")
