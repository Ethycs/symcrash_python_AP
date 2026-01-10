"""Debug polyphase slot interaction for H5 artifact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, BLOCK_SIZE, SAMPLE_RATE, VM_K, K_STRIDE, NUM_LEAVES
)
from symcrash_python_ap.vm.tables import COEFF_LEN
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.fixed import RSHIFT_RNE, SAT16

# Parameters
freq_hz = 440.0
note = 69
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
H = 5  # Focus on H5

print("=" * 70)
print("POLYPHASE SLOT ANALYSIS FOR H5")
print("=" * 70)

# Generate basis frames
basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

print(f"\nVM_K = {VM_K}, K_STRIDE = {K_STRIDE}")
print(f"Basis frame phase offsets per k-slot:")
for k in range(VM_K):
    offset_samples = k * K_STRIDE
    offset_phase = (offset_samples * phase_inc) & PHASE_MASK
    offset_cycles = offset_phase / (2**32)
    h5_offset_cycles = (5 * offset_cycles) % 1
    print(f"  k={k}: {offset_samples} samples = {offset_cycles:.6f} H1 cycles, "
          f"H5 offset = {h5_offset_cycles:.6f} cycles")

# =============================================================================
# Generate coefficients manually for analysis
# =============================================================================

def generate_h5_coeffs_for_block(base_phase, a_h=32767):
    """Generate H5 coefficients for all K slots at a given phase."""
    h = 5
    coeffs = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)

    for k in range(VM_K):
        # Phase for this k-slot
        phi = base_phase

        # Compute sin(h*phi), cos(h*phi)
        h_phase = (phi * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)

        # Weights for basis
        w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
        w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

        # Generate coefficients for all leaves
        for leaf in range(NUM_LEAVES):
            sin_basis = int(basis.Usin_h[h-1, leaf, k])
            cos_basis = int(basis.Ucos_h[h-1, leaf, k])
            sin_contrib = RSHIFT_RNE(w_sin * sin_basis, 15)
            cos_contrib = RSHIFT_RNE(w_cos * cos_basis, 15)
            coeffs[leaf, k] = SAT16(sin_contrib + cos_contrib)

    return coeffs


def idwpt_polyphase_block_debug(coeff_2d, headroom_bits=1):
    """IDWPT polyphase with debug output."""
    K = coeff_2d.shape[1]
    out = np.zeros(64, dtype=np.int16)

    slot_outputs = []
    for k in range(K):
        coeffs_k = coeff_2d[:, k].astype(np.int16).tolist()
        audio_k = np.array(idwpt_jit(coeffs_k, headroom_bits=headroom_bits), dtype=np.int16)
        slot_outputs.append(audio_k.copy())
        out[k::K] = audio_k[0::K]

    return out, slot_outputs


# =============================================================================
# Trace coefficient and output for multiple blocks
# =============================================================================

print("\n" + "=" * 70)
print("BLOCK-BY-BLOCK TRACE")
print("=" * 70)

num_blocks = 20
phase = 0
all_samples = []
slot_data = {k: [] for k in range(VM_K)}

for block_idx in range(num_blocks):
    coeffs = generate_h5_coeffs_for_block(phase)
    out, slot_outputs = idwpt_polyphase_block_debug(coeffs)
    all_samples.extend(out.tolist())

    # Track per-slot outputs
    for k in range(VM_K):
        slot_data[k].extend(slot_outputs[k].tolist())

    if block_idx < 5:
        h5_phase_cycles = ((phase * 5) & PHASE_MASK) / (2**32)
        print(f"\nBlock {block_idx}: H5 phase = {h5_phase_cycles:.4f} cycles")
        print(f"  Coeff[0,:] = {coeffs[0,:].tolist()}")
        print(f"  Out first 8: {out[:8].tolist()}")

    # Advance phase
    phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

# =============================================================================
# Analyze the combined output vs individual slots
# =============================================================================

print("\n" + "=" * 70)
print("SLOT COMBINATION ANALYSIS")
print("=" * 70)

all_samples = np.array(all_samples, dtype=np.float64)

# Analyze each slot's contribution
for k in range(VM_K):
    slot_arr = np.array(slot_data[k], dtype=np.float64)

    # Calculate 50 Hz component
    n = len(slot_arr)
    if n > 100:
        hann = np.hanning(n)
        windowed = slot_arr * hann
        spectrum = np.abs(rfft(windowed))
        freqs = rfftfreq(n, 1/SAMPLE_RATE)

        idx_50 = int(50 * n / SAMPLE_RATE)
        idx_h5 = int(2200 * n / SAMPLE_RATE)

        amp_50 = spectrum[idx_50]
        amp_h5 = spectrum[idx_h5] if idx_h5 < len(spectrum) else 1

        rel_db = 20 * np.log10(amp_50 / amp_h5) if amp_h5 > 0 else -100

        print(f"Slot {k}: 50 Hz = {rel_db:.1f} dB relative to H5")

# Combined output
print(f"\nCombined output:")
n = len(all_samples)
hann = np.hanning(n)
windowed = all_samples * hann
spectrum = np.abs(rfft(windowed))
freqs = rfftfreq(n, 1/SAMPLE_RATE)

idx_50 = int(50 * n / SAMPLE_RATE)
idx_h5 = int(2200 * n / SAMPLE_RATE)

amp_50 = spectrum[idx_50]
amp_h5 = spectrum[idx_h5] if idx_h5 < len(spectrum) else 1

rel_db = 20 * np.log10(amp_50 / amp_h5) if amp_h5 > 0 else -100
print(f"  50 Hz = {rel_db:.1f} dB relative to H5")

# =============================================================================
# Check if the issue is in the interleaving itself
# =============================================================================

print("\n" + "=" * 70)
print("INTERLEAVING ANALYSIS")
print("=" * 70)

print("""
The interleave takes:
  out[0, 4, 8, 12, ...] from slot 0's IDWPT output [0, 4, 8, 12, ...]
  out[1, 5, 9, 13, ...] from slot 1's IDWPT output [0, 4, 8, 12, ...]
  etc.

If the slots have different phase relationships, the interleaved
output may have discontinuities at the slot boundaries (every K samples).
""")

# Analyze sample-to-sample differences
if len(all_samples) > 100:
    diff = np.diff(all_samples)
    print(f"Sample-to-sample differences:")
    print(f"  Mean: {np.mean(diff):.1f}")
    print(f"  Std: {np.std(diff):.1f}")
    print(f"  Max: {np.max(np.abs(diff)):.1f}")

    # Check if there's a pattern every K samples
    print(f"\nDifference pattern (first 16 samples):")
    for i in range(min(16, len(diff))):
        slot = i % VM_K
        print(f"  sample {i} (slot {slot}): diff = {diff[i]:.0f}")

# Check for discontinuities at slot boundaries
print(f"\nSlot boundary discontinuities (every {VM_K} samples):")
slot_boundary_diffs = diff[VM_K-1::VM_K][:10]  # At positions K-1, 2K-1, etc.
print(f"  First 10 slot-boundary diffs: {slot_boundary_diffs.astype(int).tolist()}")
print(f"  Mean slot-boundary diff: {np.mean(np.abs(slot_boundary_diffs)):.1f}")
