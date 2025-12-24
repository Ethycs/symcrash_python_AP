"""Debug WET path phase continuity after coefficient fix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.wavelet_packet_trees import PacketTreeSet, LEAF_COUNT, NODE_SIZE, BLOCK_SIZE
from symcrash_python_ap.leaf_generator import LeafGenerator
from symcrash_python_ap.carrier_lift import Q15_ONE, Q15_HALF

FS = 48000
BLOCKS_PER_SEC = FS // BLOCK_SIZE


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


def test_direct_synthesize_with_phase():
    """Test direct synthesis with proper coefficient expansion and phase."""
    print("\n" + "=" * 60)
    print("TEST: Direct synthesis with proper leaf->coeff expansion")
    print("=" * 60)

    trees = PacketTreeSet()
    leaf_gen = LeafGenerator()

    # Generate leaf content for C4 (pure sine)
    f0_hz = 261.63
    harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]
    leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

    # Build proper 256 coefficients (each leaf expands to NODE_SIZE coefficients)
    coeffs_256 = []
    for i in range(256):
        leaf_idx = i // NODE_SIZE  # leaf 0 -> coeffs[0:4], etc.
        coeffs_256.append(leaf_content.coeffs_q15[leaf_idx])

    # Initialize phase state
    import math
    phase_state = []
    for leaf_idx in range(LEAF_COUNT):
        for coeff_idx in range(NODE_SIZE):
            phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
            phase_state.append(phase)

    # Render multiple blocks
    samples = []
    synth = trees.synthesizers[1]  # Harmonic tree
    nodes = trees.unpack_coeffs(1, coeffs_256)

    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        block = synth.synthesize(nodes, phase_state=phase_state)
        samples.extend(block)

    analyze(samples, "Direct synthesis (proper expansion, with phase)")


def test_via_wavelet_engine():
    """Test via WaveletEngine (like render_voice does)."""
    print("\n" + "=" * 60)
    print("TEST: Via WaveletEngine.render_voice (no regularity)")
    print("=" * 60)

    from symcrash_python_ap.wavelet_engine import WaveletEngine
    from symcrash_python_ap.frames import VoiceFrame

    engine = WaveletEngine(block_size=BLOCK_SIZE)
    leaf_gen = LeafGenerator()

    f0_hz = 261.63
    harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    engine.voice_allocate(0)

    samples = []
    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

        # Build proper 256 coefficients
        coeffs_256 = []
        for i in range(256):
            leaf_idx = i // NODE_SIZE
            coeffs_256.append(leaf_content.coeffs_q15[leaf_idx])

        vf = VoiceFrame(
            voice_id=0,
            tree_a_id=1,
            tree_b_id=1,
            tree_blend_q15=0,
            gain_q15=Q15_ONE,
            coeff_q15=coeffs_256,
        )

        block = engine.render_voice(vf, apply_regularity=False)
        samples.extend(block)

    analyze(samples, "WaveletEngine (no regularity)")


def test_via_wavelet_engine_with_regularity():
    """Test via WaveletEngine with regularity operator."""
    print("\n" + "=" * 60)
    print("TEST: Via WaveletEngine.render_voice (with regularity)")
    print("=" * 60)

    from symcrash_python_ap.wavelet_engine import WaveletEngine
    from symcrash_python_ap.frames import VoiceFrame

    engine = WaveletEngine(block_size=BLOCK_SIZE)
    leaf_gen = LeafGenerator()

    f0_hz = 261.63
    harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    engine.voice_allocate(0)

    samples = []
    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        leaf_content = leaf_gen.generate_from_fundamental(f0_hz, harmonic_amps)

        # Build proper 256 coefficients
        coeffs_256 = []
        for i in range(256):
            leaf_idx = i // NODE_SIZE
            coeffs_256.append(leaf_content.coeffs_q15[leaf_idx])

        vf = VoiceFrame(
            voice_id=0,
            tree_a_id=1,
            tree_b_id=1,
            tree_blend_q15=0,
            gain_q15=Q15_ONE,
            coeff_q15=coeffs_256,
        )

        block = engine.render_voice(vf, apply_regularity=True)
        samples.extend(block)

    analyze(samples, "WaveletEngine (with regularity)")


def test_render_voice_raw():
    """Test render_voice directly (like CarrierLiftPipeline but raw)."""
    print("\n" + "=" * 60)
    print("TEST: CarrierLiftPipeline.render_voice (raw, no safety)")
    print("=" * 60)

    from symcrash_python_ap.unified_renderer import CarrierLiftPipeline
    from symcrash_python_ap.frames import VoiceFrame

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    # Set pure sine harmonics on carrier
    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(0.5 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=261.63,
            gain_q15=Q15_HALF,
            beta_target_q15=Q15_ONE,  # 100% WET
        )
        block = pipeline.render_voice(frame)
        samples.extend(block)

    pipeline.voice_off(0)
    analyze(samples, "CarrierLiftPipeline.render_voice (raw)")


if __name__ == "__main__":
    print("=" * 60)
    print("WET PATH PHASE CONTINUITY DEBUG")
    print("=" * 60)

    test_direct_synthesize_with_phase()
    test_via_wavelet_engine()
    test_via_wavelet_engine_with_regularity()
    test_render_voice_raw()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
