"""Debug individual components of the WET path."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.carrier_lift import Q15_ONE, Q15_HALF, BLOCK_SIZE

FS = 48000
BLOCKS_PER_SEC = FS // BLOCK_SIZE


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def analyze(samples, name):
    arr = np.array(samples, dtype=np.float64)
    deltas = np.diff(arr)
    d2 = np.diff(deltas)

    print(f"\n{name}:")
    print(f"  Samples: {len(samples)}")
    print(f"  Peak: {int(np.max(np.abs(arr)))}")
    print(f"  Max delta: {int(np.max(np.abs(deltas)))}")
    print(f"  Max d2: {int(np.max(np.abs(d2)))}")
    print(f"  Spikes: >500={np.sum(np.abs(d2) > 500)}, >1k={np.sum(np.abs(d2) > 1000)}, >3k={np.sum(np.abs(d2) > 3000)}")


def test_carrier_only():
    """Test carrier path only."""
    print("\n" + "=" * 60)
    print("TEST: Carrier only (x_carrier)")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(1.0 * BLOCKS_PER_SEC)):
        # Get carrier block directly
        x_carrier = pipeline.carrier.render_block(0, BLOCK_SIZE)
        samples.extend(x_carrier)

    pipeline.voice_off(0)
    analyze(samples, "Carrier only")


def test_wavelet_only():
    """Test wavelet path only (IDWPT output)."""
    print("\n" + "=" * 60)
    print("TEST: Wavelet only (x_wpt before DC blocker)")
    print("=" * 60)

    from symcrash_python_ap.unified_renderer import RenderFrame
    from symcrash_python_ap.leaf_generator import LeafGenerator

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    leaf_gen = LeafGenerator()
    samples = []

    for _ in range(int(1.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=Q15_ONE,
        )

        # Generate leaf content
        f0_hz = midi_to_freq(60)
        # Use pure sine harmonic amplitudes
        harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]
        leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

        # Apply tuning
        tuning_gains = pipeline.tuning.get_current_gains()
        leaf_content.apply_per_leaf_gains(tuning_gains)

        # Build coefficients (64 leaves -> 256 for compatibility)
        # Each leaf expands to 4 consecutive coefficients (NODE_SIZE=4)
        from symcrash_python_ap.wavelet_trees import NODE_SIZE as WPT_NODE_SIZE
        coeffs_256 = []
        for i in range(256):
            leaf_idx = i // WPT_NODE_SIZE  # leaf 0 -> coeffs[0:4], leaf 1 -> coeffs[4:8], etc.
            coeffs_256.append(leaf_content.coeffs_q15[leaf_idx])

        # Build wavelet frame
        wavelet_frame = RenderFrame(
            voice_id=0,
            note=60,
            cents=0,
            gain_q15=Q15_ONE,
            coeff_q15=coeffs_256,
        )

        pipeline.wavelet_backend.apply_params(wavelet_frame)
        x_wpt = pipeline.wavelet_backend.render_block(wavelet_frame, BLOCK_SIZE)

        samples.extend(x_wpt)

        # Advance blend state
        if 0 in pipeline.blend_states:
            pipeline.blend_states[0].advance_block()

        pipeline.tuning.advance_block()

    pipeline.voice_off(0)
    analyze(samples, "Wavelet only (before DC blocker)")


def test_wavelet_with_dc_blocker():
    """Test wavelet with DC blocker."""
    print("\n" + "=" * 60)
    print("TEST: Wavelet with DC blocker")
    print("=" * 60)

    from symcrash_python_ap.unified_renderer import RenderFrame
    from symcrash_python_ap.leaf_generator import LeafGenerator
    from symcrash_python_ap.wavelet_trees import NODE_SIZE as WPT_NODE_SIZE

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    leaf_gen = LeafGenerator()
    samples = []

    for _ in range(int(1.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=Q15_ONE,
        )

        # Generate leaf content
        f0_hz = midi_to_freq(60)
        # Use pure sine harmonic amplitudes
        harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]
        leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

        # Apply tuning
        tuning_gains = pipeline.tuning.get_current_gains()
        leaf_content.apply_per_leaf_gains(tuning_gains)

        # Build coefficients (each leaf expands to 4 consecutive coefficients)
        coeffs_256 = []
        for i in range(256):
            leaf_idx = i // WPT_NODE_SIZE
            coeffs_256.append(leaf_content.coeffs_q15[leaf_idx])

        wavelet_frame = RenderFrame(
            voice_id=0,
            note=60,
            cents=0,
            gain_q15=Q15_ONE,
            coeff_q15=coeffs_256,
        )

        pipeline.wavelet_backend.apply_params(wavelet_frame)
        x_wpt = pipeline.wavelet_backend.render_block(wavelet_frame, BLOCK_SIZE)

        # Apply DC blocker
        x_wpt = pipeline._apply_dc_blocker(0, x_wpt)

        samples.extend(x_wpt)

        if 0 in pipeline.blend_states:
            pipeline.blend_states[0].advance_block()

        pipeline.tuning.advance_block()

    pipeline.voice_off(0)
    analyze(samples, "Wavelet with DC blocker")


def test_blend_output():
    """Test the blended output from render_voice."""
    print("\n" + "=" * 60)
    print("TEST: Blended output (render_voice)")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(1.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=Q15_ONE,
        )
        block = pipeline.render_voice(frame)
        samples.extend(block)

    pipeline.voice_off(0)
    analyze(samples, "Blended output (render_voice)")


if __name__ == "__main__":
    print("=" * 60)
    print("WET COMPONENT DEBUG")
    print("=" * 60)

    test_carrier_only()
    test_wavelet_only()
    test_wavelet_with_dc_blocker()
    test_blend_output()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
