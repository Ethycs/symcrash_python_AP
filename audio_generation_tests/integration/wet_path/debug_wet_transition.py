"""
Debug wet transition - trace what happens during the beta ramp
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame


def test_wet_transition():
    """Detailed trace of wet transition."""
    print("=" * 60)
    print("WET TRANSITION DETAILED TRACE")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    samples = []
    prev_sample = 0

    for block in range(30):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=261.63,
            gain_q15=32767,
            beta_target_q15=16384,  # 50% wet
        )

        # Get state before rendering
        ac_state = pipeline.ac_states[0]
        beta_before = ac_state.beta
        wet_ready = ac_state.wet_ready
        fifo_warmup = ac_state.fifo_warmup
        tail_before = ac_state.tail.copy() if ac_state.tail is not None else None

        # Render
        L, R = pipeline.render_block(frame)
        samples.extend(L)

        # Get state after
        beta_after = ac_state.beta
        tail_after = ac_state.tail.copy() if ac_state.tail is not None else None

        # Check block boundary
        first_sample = L[0]
        boundary_diff = abs(first_sample - prev_sample) if block > 0 else 0
        prev_sample = L[-1]

        # Print trace for first 15 blocks
        if block < 15:
            use_wet = beta_before > 0.0 and wet_ready
            print(f"Block {block:2d}: beta={beta_before:.3f}->{beta_after:.3f}, wet_ready={wet_ready}, warmup={fifo_warmup}, use_wet={use_wet}")
            print(f"         boundary_diff={boundary_diff:6d}, first={first_sample:6d}, last={prev_sample:6d}")
            if tail_after is not None:
                # Check if tail values are in valid range
                tail_min = np.min(tail_after)
                tail_max = np.max(tail_after)
                if tail_max > 1.0 or tail_min < -1.0:
                    print(f"         [WARN] tail out of range: min={tail_min:.3f}, max={tail_max:.3f}")

    # Analyze
    arr = np.array(samples, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    threshold = mean_diff * 10

    spikes = np.where(diffs > threshold)[0]
    print(f"\nAnalysis:")
    print(f"  Max diff: {max_diff:.0f}, Mean diff: {mean_diff:.2f}")
    print(f"  Threshold (10x mean): {threshold:.0f}")
    print(f"  Spikes: {len(spikes)}")

    for block in range(1, 15):
        idx = block * 64 - 1
        if idx < len(diffs):
            if diffs[idx] > threshold:
                print(f"  Block {block} boundary spike: {diffs[idx]:.0f}")


if __name__ == "__main__":
    test_wet_transition()
