"""Trace exact source of remaining spikes in render_voice."""
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
    print(f"  Spikes: >500={np.sum(np.abs(d2) > 500)}, >1k={np.sum(np.abs(d2) > 1000)}")

    # Find where spikes occur
    spike_locs = np.where(np.abs(d2) > 500)[0]
    if len(spike_locs) > 0 and len(spike_locs) <= 20:
        print(f"  Spike locations (first 20): {spike_locs[:20]}")
        # Show spike context
        for loc in spike_locs[:5]:
            block = loc // BLOCK_SIZE
            pos = loc % BLOCK_SIZE
            print(f"    @{loc} (block {block}, pos {pos}): d2={int(d2[loc])}")


def test_render_voice_breakdown():
    """Break down render_voice into components."""
    print("\n" + "=" * 60)
    print("TEST: render_voice component breakdown")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    # Set pure sine on carrier
    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    carrier_samples = []
    wavelet_samples = []
    blended_samples = []

    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=Q15_ONE,  # 100% WET
        )

        # Get carrier
        x_carrier = pipeline.carrier.render_block(0, BLOCK_SIZE)
        carrier_samples.extend(x_carrier)

        # Full render_voice
        block = pipeline.render_voice(frame)
        blended_samples.extend(block)

    pipeline.voice_off(0)

    analyze(carrier_samples, "Carrier only (direct)")
    analyze(blended_samples, "render_voice output (100% WET)")


def test_wavelet_backend_direct():
    """Test wavelet backend directly with harmonic coeffs."""
    print("\n" + "=" * 60)
    print("TEST: WaveletBackend direct")
    print("=" * 60)

    from symcrash_python_ap.wavelet_engine import WaveletEngine
    from symcrash_python_ap.frames import VoiceFrame as WE_VoiceFrame
    from symcrash_python_ap.harmonic_injector import create_harmonic_coeffs, WaveType

    engine = WaveletEngine(block_size=BLOCK_SIZE)
    engine.voice_allocate(0)

    f0_hz = 261.63
    coeffs_256 = create_harmonic_coeffs(
        frequency=f0_hz,
        amplitude=Q15_HALF,
        wave_type=WaveType.SINE,
        partials_n=1
    )

    samples = []
    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        vf = WE_VoiceFrame(
            voice_id=0,
            tree_a_id=1,
            tree_b_id=1,
            tree_blend_q15=0,
            gain_q15=Q15_ONE,
            coeff_q15=coeffs_256,
        )
        block = engine.render_voice(vf, apply_regularity=False)
        samples.extend(block)

    analyze(samples, "WaveletEngine direct (no regularity)")


if __name__ == "__main__":
    print("=" * 60)
    print("RENDER_VOICE SPIKE TRACE")
    print("=" * 60)

    test_render_voice_breakdown()
    test_wavelet_backend_direct()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
