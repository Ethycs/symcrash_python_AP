"""Debug LUT discontinuity for H5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15
from symcrash_python_ap.vm.coeff_lut_streaming import StreamingCoeffLUT, VM_K, BLOCK_SIZE, NUM_LEAVES

SAMPLE_RATE = 48000

print("=" * 70)
print("LUT DISCONTINUITY ANALYSIS")
print("=" * 70)

freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

def analyze_lut(h, label):
    """Analyze LUT for harmonic h."""
    def gen_h(phase):
        h_phase = (phase * h) & 0xFFFFFFFF
        return sine_q15(h_phase)

    lut = StreamingCoeffLUT.generate_for_frequency(
        wave_id=0,
        name=label,
        target_freq_hz=freq_hz,
        generate_sample_fn=gen_h,
        K=VM_K,
        P=64,
        normalize=False,
    )

    print(f"\n--- {label} ---")
    print(f"LUT shape: {lut.data.shape} (leaves, k-slots, phase-bins)")

    # Check coefficient continuity across phase bins
    data = lut.data  # [64 leaves, K slots, P phase bins]

    # For each leaf and k-slot, check max jump between adjacent phase bins
    max_jumps = []
    for leaf in range(NUM_LEAVES):
        for k in range(VM_K):
            coeffs_vs_phase = data[leaf, k, :]  # P values
            # Wrap-around: compare bin P-1 to bin 0
            diffs = np.diff(coeffs_vs_phase.astype(np.int32))
            wrap_diff = int(coeffs_vs_phase[0]) - int(coeffs_vs_phase[-1])
            all_diffs = np.append(diffs, wrap_diff)
            max_jump = np.max(np.abs(all_diffs))
            max_jumps.append(max_jump)

    max_jumps = np.array(max_jumps)
    print(f"Max coefficient jump between phase bins:")
    print(f"  Overall max: {np.max(max_jumps)}")
    print(f"  Mean max: {np.mean(max_jumps):.1f}")
    print(f"  Std: {np.std(max_jumps):.1f}")

    # Check specific leaf 0, k=0 across all phase bins
    print(f"\nLeaf 0, k=0 coefficients vs phase:")
    coeffs = data[0, 0, :]
    print(f"  Min: {np.min(coeffs)}, Max: {np.max(coeffs)}")
    print(f"  First 8 bins: {coeffs[:8].tolist()}")
    print(f"  Last 8 bins:  {coeffs[-8:].tolist()}")
    print(f"  Wrap discontinuity: {int(coeffs[0]) - int(coeffs[-1])}")

    # Find the phase bin with largest discontinuity
    diffs = np.abs(np.diff(coeffs.astype(np.int32)))
    max_diff_idx = np.argmax(diffs)
    print(f"  Largest jump at bin {max_diff_idx}->{max_diff_idx+1}: {diffs[max_diff_idx]}")

    return lut

# Test H1 (good) and H5 (bad)
lut_h1 = analyze_lut(1, "H1 (440 Hz)")
lut_h5 = analyze_lut(5, "H5 (2200 Hz)")

# Compare coefficient evolution rate
print("\n" + "=" * 70)
print("COEFFICIENT EVOLUTION RATE")
print("=" * 70)

print("\nPhase increment per block (Q32):", BLOCK_SIZE * phase_inc)
print("As fraction of 2^32:", (BLOCK_SIZE * phase_inc) / (2**32))
print("Phase bins per block:", (BLOCK_SIZE * phase_inc) / (2**32) * 64)

# For H5, the effective frequency is 5x higher
print("\nFor H5:")
print("  Effective phase inc (Q32):", BLOCK_SIZE * phase_inc * 5)
print("  As fraction of 2^32:", (BLOCK_SIZE * phase_inc * 5) / (2**32))
print("  Effective phase bins per block:", ((BLOCK_SIZE * phase_inc * 5) % (2**32)) / (2**32) * 64)
