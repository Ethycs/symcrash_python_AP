"""Debug the WET (wavelet) pipeline to find spike sources."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame, ParamFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.carrier_lift import Q15_ONE, Q15_HALF

FS = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SEC = FS // BLOCK_SIZE

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


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

    # Check block boundaries
    block_boundary_d2 = []
    for i in range(len(d2)):
        if (i + 2) % BLOCK_SIZE < 4:  # Near block boundary
            block_boundary_d2.append(abs(d2[i]))
    if block_boundary_d2:
        print(f"  Max d2 at block boundaries: {int(max(block_boundary_d2))}")


def test_raw_wet_only():
    """Test raw wavelet output without any safety."""
    print("\n" + "=" * 60)
    print("TEST 1: Raw WET (render_voice, beta=100%)")
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
            beta_target_q15=Q15_ONE,  # 100% wet
        )
        block = pipeline.render_voice(frame)
        samples.extend(block)

    pipeline.voice_off(0)
    analyze(samples, "Raw WET (render_voice)")


def test_wet_no_safety():
    """Test wet with no safety processing at all."""
    print("\n" + "=" * 60)
    print("TEST 2: WET with stereo conversion only (no safety)")
    print("=" * 60)

    import math
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
        # Simple stereo conversion without safety
        L = [int(s * 0.707) for s in block]
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "WET stereo only (no safety)")


def test_wet_with_declick_no_ola():
    """Test wet with declick but no OLA (seam=False)."""
    print("\n" + "=" * 60)
    print("TEST 3: WET with declick(seam=False, c1=True)")
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
        block = pipeline.safety.apply_declick(block, 0, seam=False, apply_c1_patch=True)
        samples.extend(block)

    pipeline.voice_off(0)
    analyze(samples, "WET with declick(seam=False)")


def test_wet_with_step_block_all_disabled():
    """Test wet with step_block but all safety disabled."""
    print("\n" + "=" * 60)
    print("TEST 4: WET with step_block (all disabled)")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    params = ParamFrame()
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
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.step_block(
            L, R, params,
            apply_master_ola=False,
            apply_slew_limit=False,
            apply_dc_removal=False
        )
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "WET with step_block (all disabled)")


def test_wet_with_dc_removal():
    """Test wet with DC removal enabled."""
    print("\n" + "=" * 60)
    print("TEST 5: WET with step_block (DC removal only)")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    params = ParamFrame()
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
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.step_block(
            L, R, params,
            apply_master_ola=False,
            apply_slew_limit=False,
            apply_dc_removal=True  # Only DC removal
        )
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "WET with DC removal only")


def test_wet_with_master_ola():
    """Test wet with master OLA enabled."""
    print("\n" + "=" * 60)
    print("TEST 6: WET with step_block (master OLA only)")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    params = ParamFrame()
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
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.step_block(
            L, R, params,
            apply_master_ola=True,  # Only master OLA
            apply_slew_limit=False,
            apply_dc_removal=False
        )
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "WET with master OLA only")


def test_full_wet_render_block():
    """Test full render_block with wet."""
    print("\n" + "=" * 60)
    print("TEST 7: Full render_block (WET)")
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
        L, R = pipeline.render_block(frame)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "Full render_block (WET)")


if __name__ == "__main__":
    print("=" * 60)
    print("WET PIPELINE DEBUG")
    print("=" * 60)

    test_raw_wet_only()
    test_wet_no_safety()
    test_wet_with_declick_no_ola()
    test_wet_with_step_block_all_disabled()
    test_wet_with_dc_removal()
    test_wet_with_master_ola()
    test_full_wet_render_block()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
