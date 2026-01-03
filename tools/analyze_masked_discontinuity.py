#!/usr/bin/env python3
"""
Analyze IDWPT discontinuities WITH masking applied.
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
        phase_16bit = (phase >> 16) & 0xFFFF
        sin_val = int(32767 * np.sin(2 * np.pi * phase_16bit / 65536))
        samples.append(sin_val)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return samples, phase

def main():
    f0_hz = 261.63  # Middle C

    print("Analyzing IDWPT block discontinuities WITH masking")
    print("=" * 60)

    # Test different mask scenarios
    mask_scenarios = [
        ("Identity (None)", None),
        ("All ones [32767]*64", [32767] * 64),
        ("Half amplitude [16384]*64", [16384] * 64),
        ("Varying mask", [32767 if i % 2 == 0 else 16384 for i in range(64)]),
    ]

    for mask_name, mask in mask_scenarios:
        print(f"\n{'='*60}")
        print(f"Mask: {mask_name}")
        print("=" * 60)

        # Generate 3 consecutive blocks of carrier
        phase = 0
        blocks = []
        for i in range(3):
            carrier, phase = generate_carrier_block(phase, f0_hz, BLOCK_SIZE)
            blocks.append(carrier)

        # Apply IDWPT with mask
        wet_blocks = []
        for i, carrier in enumerate(blocks):
            wet = effect_mode_jit(carrier, mask, headroom_bits=1)
            wet_blocks.append(wet)

        # Check block boundaries
        print("\nBlock boundary discontinuities:")
        for i in range(len(blocks) - 1):
            carrier_end = blocks[i][-1]
            carrier_start = blocks[i+1][0]
            carrier_diff = carrier_start - carrier_end

            wet_end = wet_blocks[i][-1]
            wet_start = wet_blocks[i+1][0]
            wet_diff = wet_start - wet_end

            print(f"  Block {i}->{i+1}: carrier_diff={carrier_diff:+5d}, wet_diff={wet_diff:+5d}")

        # Within-block error
        print("\nWithin-block wet vs carrier:")
        for i in range(len(blocks)):
            carrier = np.array(blocks[i])
            wet = np.array(wet_blocks[i])
            diff = wet - carrier
            max_diff = np.max(np.abs(diff))
            rms_diff = np.sqrt(np.mean(diff.astype(np.float64)**2))
            print(f"  Block {i}: max_diff={max_diff:5d}, rms_diff={rms_diff:7.1f}")


if __name__ == "__main__":
    main()
