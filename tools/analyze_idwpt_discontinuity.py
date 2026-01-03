#!/usr/bin/env python3
"""
Analyze IDWPT block discontinuities - find why wavelet output doesn't match carrier phase.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit

BLOCK_SIZE = 64
SAMPLE_RATE = 48000

def generate_carrier_block(phase, f0_hz, n_samples):
    """Generate a block of sine carrier samples."""
    phase_inc = int((f0_hz / SAMPLE_RATE) * (2**32))
    samples = []
    for i in range(n_samples):
        # 16-bit phase for lookup (top 16 bits of 32-bit phase)
        phase_16bit = (phase >> 16) & 0xFFFF
        # Simple sine approximation (use actual sine table for real test)
        sin_val = int(32767 * np.sin(2 * np.pi * phase_16bit / 65536))
        samples.append(sin_val)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return samples, phase

def main():
    f0_hz = 261.63  # Middle C

    print("Analyzing IDWPT block discontinuities")
    print("=" * 60)

    # Generate 3 consecutive blocks of carrier
    phase = 0
    blocks = []
    for i in range(3):
        carrier, phase = generate_carrier_block(phase, f0_hz, BLOCK_SIZE)
        blocks.append(carrier)

    # Apply IDWPT (roundtrip with identity mask) to each block
    wet_blocks = []
    for i, carrier in enumerate(blocks):
        # Identity mask = None means passthrough
        wet = effect_mode_jit(carrier, None, headroom_bits=1)
        wet_blocks.append(wet)

    # Check block boundaries
    print("\nBlock boundary analysis:")
    for i in range(len(blocks) - 1):
        carrier_end = blocks[i][-1]
        carrier_start = blocks[i+1][0]
        carrier_diff = carrier_start - carrier_end

        wet_end = wet_blocks[i][-1]
        wet_start = wet_blocks[i+1][0]
        wet_diff = wet_start - wet_end

        print(f"\nBlock {i} -> Block {i+1}:")
        print(f"  Carrier: {carrier_end} -> {carrier_start} (diff={carrier_diff})")
        print(f"  Wet:     {wet_end} -> {wet_start} (diff={wet_diff})")
        print(f"  Carrier is continuous: {abs(carrier_diff) < 1000}")
        print(f"  Wet is continuous:     {abs(wet_diff) < 1000}")

    # Show last 8 and first 8 samples around boundary
    print("\nSamples around block 0->1 boundary:")
    print("  Carrier end:", blocks[0][-8:])
    print("  Carrier start:", blocks[1][:8])
    print("  Wet end:", wet_blocks[0][-8:])
    print("  Wet start:", wet_blocks[1][:8])

    # Check if wet matches carrier within each block
    print("\nWithin-block comparison (wet vs carrier):")
    for i in range(len(blocks)):
        carrier = np.array(blocks[i])
        wet = np.array(wet_blocks[i])
        diff = wet - carrier
        max_diff = np.max(np.abs(diff))
        rms_diff = np.sqrt(np.mean(diff.astype(np.float64)**2))
        print(f"  Block {i}: max_diff={max_diff}, rms_diff={rms_diff:.1f}")


if __name__ == "__main__":
    main()
