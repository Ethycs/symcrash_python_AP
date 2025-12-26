"""
Check for Discontinuities in Audio Output
==========================================

Tests the anti-click fix by checking for large sample-to-sample jumps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame


def check_discontinuities():
    print("=" * 60)
    print("CHECKING FOR DISCONTINUITIES")
    print("=" * 60)

    # Generate a simple test tone
    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    warmup_samples = []
    samples = []

    # First 50 blocks for warmup/transition
    for i in range(50):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=261.63,
            gain_q15=16384,
            beta_target_q15=16384,  # 50% wet
        )
        L, R = pipeline.render_block(frame)
        warmup_samples.extend(L)

    # Next 100 blocks for steady-state analysis
    for i in range(100):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=261.63,
            gain_q15=16384,
            beta_target_q15=16384,  # 50% wet
        )
        L, R = pipeline.render_block(frame)
        samples.extend(L)

    print(f"\n[Skipped first 50 blocks for warmup/transition]")

    # Check for discontinuities (large sample-to-sample jumps)
    arr = np.array(samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))

    # Find largest jumps
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10  # Jumps > 10x mean are suspicious

    spikes = np.where(diffs > threshold)[0]
    print(f"\nTotal samples: {len(arr)}")
    print(f"Max sample-to-sample diff: {max_diff:.0f}")
    print(f"Mean sample-to-sample diff: {mean_diff:.2f}")
    print(f"Threshold (10x mean): {threshold:.0f}")
    print(f"Suspicious jumps (>threshold): {len(spikes)}")

    if len(spikes) > 0:
        print(f"\nFirst 10 spike locations: {spikes[:10].tolist()}")
        for loc in spikes[:5]:
            print(f"  Sample {loc}: {arr[loc]:.0f} -> {arr[loc+1]:.0f} (diff={diffs[loc]:.0f})")
    else:
        print("\nNo suspicious discontinuities detected!")

    # Check block boundaries specifically (where bugs usually occur)
    print("\n" + "-" * 40)
    print("Block boundary analysis:")
    block_size = 64
    boundary_issues = 0
    for block in range(1, 20):
        idx = block * block_size - 1
        if idx < len(arr) - 1:
            diff = abs(arr[idx+1] - arr[idx])
            if diff > threshold:
                print(f"  Block {block} boundary (sample {idx}): diff={diff:.0f}")
                boundary_issues += 1

    if boundary_issues == 0:
        print("  No block boundary discontinuities!")

    # Check around the wet transition point (around block 5-6)
    print("\n" + "-" * 40)
    print("Wet transition analysis (blocks 4-8):")
    for block in range(4, 9):
        start = block * block_size
        end = (block + 1) * block_size
        if end <= len(arr):
            block_samples = arr[start:end]
            block_diffs = np.abs(np.diff(block_samples))
            max_block_diff = np.max(block_diffs)
            print(f"  Block {block}: max diff = {max_block_diff:.0f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    check_discontinuities()
