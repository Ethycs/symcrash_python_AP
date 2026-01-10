"""Test if bit-exact FWPT makes a difference for basis frames."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.vm.harmonic_bank import VM_K, K_STRIDE, NUM_LEAVES, BLOCK_SIZE
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

SAMPLE_RATE = 48000

print("=" * 70)
print("BIT-EXACT VS FAST FWPT FOR BASIS FRAMES")
print("=" * 70)

note = 69
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
h = 5

# Generate H5 sin/cos basis using fast FWPT
print("\n--- Generating H5 basis (k=0) ---")
k_phase_offset = 0  # k=0

sin_block = []
cos_block = []
phase = k_phase_offset
for n in range(BLOCK_SIZE):
    h_phase = (phase * h) & PHASE_MASK
    sin_val, cos_val = sin_cos_q15(h_phase)
    sin_block.append(sin_val)
    cos_block.append(cos_val)
    phase = (phase + phase_inc) & PHASE_MASK

# FWPT with fast path
sin_coeffs_fast = np.array(fwpt_jit(sin_block, headroom_bits=1), dtype=np.int16)
cos_coeffs_fast = np.array(fwpt_jit(cos_block, headroom_bits=1), dtype=np.int16)

print(f"Sin basis (fast): range [{np.min(sin_coeffs_fast)}, {np.max(sin_coeffs_fast)}]")
print(f"Cos basis (fast): range [{np.min(cos_coeffs_fast)}, {np.max(cos_coeffs_fast)}]")

# Check roundtrip
sin_recon = idwpt_jit(sin_coeffs_fast.tolist(), headroom_bits=1)
sin_recon = np.array(sin_recon, dtype=np.int16)
sin_input = np.array(sin_block, dtype=np.int16)
sin_err = sin_input.astype(np.int32) - sin_recon.astype(np.int32)

print(f"\nH5 Sin roundtrip error:")
print(f"  RMS: {np.sqrt(np.mean(sin_err**2)):.1f}")
print(f"  Max: {np.max(np.abs(sin_err))}")

# Now simulate Mode B synthesis for multiple blocks
print("\n--- Mode B synthesis for 100 blocks ---")

def generate_mode_b_audio(sin_basis, cos_basis, num_blocks):
    """Generate audio using Mode B with given basis."""
    all_audio = []
    phase = 0

    for block in range(num_blocks):
        # Compute modulation weights for this block
        h_phase = (phase * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)

        # Weights: w_sin = cos(h*phi), w_cos = sin(h*phi)
        a_h = Q15_ONE
        w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
        w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

        # Generate coefficients for all k-slots
        coeffs = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)
        for k in range(VM_K):
            for leaf in range(NUM_LEAVES):
                sin_contrib = RSHIFT_RNE(w_sin * int(sin_basis[leaf]), 15)
                cos_contrib = RSHIFT_RNE(w_cos * int(cos_basis[leaf]), 15)
                coeffs[leaf, k] = SAT16(sin_contrib + cos_contrib)

        # IDWPT polyphase
        audio = idwpt_polyphase_block(coeffs, headroom_bits=1)
        all_audio.extend(audio.tolist())

        phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

    return np.array(all_audio, dtype=np.float64)

audio = generate_mode_b_audio(sin_coeffs_fast, cos_coeffs_fast, 100)

# Analyze 50 Hz artifact
n = len(audio)
hann = np.hanning(n)
spectrum = np.abs(rfft(audio * hann)) * (2.0 / np.mean(hann))

idx_h5 = int(2200 * n / SAMPLE_RATE)
idx_50 = int(50 * n / SAMPLE_RATE)
amp_h5 = spectrum[idx_h5]
amp_50 = spectrum[idx_50]

rel_50 = 20 * np.log10(amp_50 / amp_h5) if amp_h5 > 0 else -100
print(f"50 Hz artifact: {rel_50:.1f} dB")

# Now test: what if we use K separate basis frames (one per k-slot)?
print("\n--- Test with per-k basis frames ---")

k_sin_bases = []
k_cos_bases = []

for k in range(VM_K):
    k_phase_offset = (k * K_STRIDE * phase_inc) & PHASE_MASK

    sin_block = []
    cos_block = []
    phase = k_phase_offset
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, cos_val = sin_cos_q15(h_phase)
        sin_block.append(sin_val)
        cos_block.append(cos_val)
        phase = (phase + phase_inc) & PHASE_MASK

    k_sin_bases.append(np.array(fwpt_jit(sin_block, headroom_bits=1), dtype=np.int16))
    k_cos_bases.append(np.array(fwpt_jit(cos_block, headroom_bits=1), dtype=np.int16))

def generate_mode_b_audio_per_k(sin_bases, cos_bases, num_blocks):
    """Generate audio using Mode B with per-k basis frames."""
    all_audio = []
    phase = 0

    for block in range(num_blocks):
        coeffs = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)

        for k in range(VM_K):
            # Use k-specific basis
            sin_basis = sin_bases[k]
            cos_basis = cos_bases[k]

            # Same phase for all k (current implementation)
            h_phase = (phase * h) & PHASE_MASK
            sin_phi, cos_phi = sin_cos_q15(h_phase)

            a_h = Q15_ONE
            w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
            w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

            for leaf in range(NUM_LEAVES):
                sin_contrib = RSHIFT_RNE(w_sin * int(sin_basis[leaf]), 15)
                cos_contrib = RSHIFT_RNE(w_cos * int(cos_basis[leaf]), 15)
                coeffs[leaf, k] = SAT16(sin_contrib + cos_contrib)

        audio = idwpt_polyphase_block(coeffs, headroom_bits=1)
        all_audio.extend(audio.tolist())

        phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

    return np.array(all_audio, dtype=np.float64)

audio_per_k = generate_mode_b_audio_per_k(k_sin_bases, k_cos_bases, 100)

n = len(audio_per_k)
hann = np.hanning(n)
spectrum = np.abs(rfft(audio_per_k * hann)) * (2.0 / np.mean(hann))

idx_h5 = int(2200 * n / SAMPLE_RATE)
idx_50 = int(50 * n / SAMPLE_RATE)
amp_h5 = spectrum[idx_h5]
amp_50 = spectrum[idx_50]

rel_50_per_k = 20 * np.log10(amp_50 / amp_h5) if amp_h5 > 0 else -100
print(f"50 Hz artifact with per-k basis: {rel_50_per_k:.1f} dB")

print("\n--- Comparison ---")
print(f"Same basis for all k: {rel_50:.1f} dB")
print(f"Per-k basis frames:   {rel_50_per_k:.1f} dB")
