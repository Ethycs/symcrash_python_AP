"""
Debug: Which step_block component causes remaining jaggies?
===========================================================

Test each step_block component in isolation to find the culprit.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame, ParamFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, Q15_HALF
from symcrash_python_ap.safety.safety_engine import SafetyEngine

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def save_audio(samples: List[int], name: str) -> str:
    metadata = audio_io.write_audio(name, samples, samples, subdirectory="stepblock_debug", timestamp=False)
    return metadata.filepath


def analyze(samples: List[int], name: str):
    arr = np.array(samples, dtype=np.float64)
    deltas = np.diff(arr)
    d2 = np.diff(deltas)

    print(f"\n{name}:")
    print(f"  Samples: {len(samples)}")
    print(f"  Peak: {int(np.max(np.abs(arr)))}")
    print(f"  Max delta: {int(np.max(np.abs(deltas)))}")
    print(f"  Max d2: {int(np.max(np.abs(d2)))}")
    print(f"  Spikes (d2>500): {np.sum(np.abs(d2) > 500)}")


def test_raw_render_voice():
    """Raw render_voice - no safety at all."""
    print("\n" + "=" * 60)
    print("TEST 1: Raw render_voice (no safety)")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        samples.extend(block)

    pipeline.voice_off(0)
    analyze(samples, "Raw render_voice")
    save_audio(samples, "1_raw_render_voice")


def test_with_declick_only():
    """Apply only apply_declick (seam=False)."""
    print("\n" + "=" * 60)
    print("TEST 2: render_voice + apply_declick(seam=False)")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        block = pipeline.safety.apply_declick(block, 0, seam=False)
        samples.extend(block)

    pipeline.voice_off(0)
    analyze(samples, "With apply_declick(seam=False)")
    save_audio(samples, "2_declick_only")


def test_with_q_only():
    """Apply only apply_q."""
    print("\n" + "=" * 60)
    print("TEST 3: render_voice + apply_q only")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    params = ParamFrame()
    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.apply_q(L, R, params.K_mode)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "With apply_q only")
    save_audio(samples, "3_q_only")


def test_with_limiter_only():
    """Apply only apply_limiter."""
    print("\n" + "=" * 60)
    print("TEST 4: render_voice + apply_limiter only")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.apply_limiter(L, R)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "With apply_limiter only")
    save_audio(samples, "4_limiter_only")


def test_with_slew_limit_only():
    """Apply only apply_slew_limit."""
    print("\n" + "=" * 60)
    print("TEST 5: render_voice + apply_slew_limit only")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    prev_L, prev_R = 0, 0
    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)
        L, R, prev_L, prev_R = pipeline.safety.apply_slew_limit(L, R, prev_L, prev_R)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "With apply_slew_limit only")
    save_audio(samples, "5_slew_limit_only")


def test_with_dc_removal_only():
    """Apply only remove_dc."""
    print("\n" + "=" * 60)
    print("TEST 6: render_voice + remove_dc only")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.remove_dc(L, R)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "With remove_dc only")
    save_audio(samples, "6_dc_removal_only")


def test_full_step_block():
    """Full step_block with apply_master_ola=False."""
    print("\n" + "=" * 60)
    print("TEST 7: Full step_block (apply_master_ola=False)")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    params = ParamFrame()
    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.step_block(L, R, params, apply_master_ola=False)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "Full step_block (master_ola=False)")
    save_audio(samples, "7_full_step_block")


def test_q_then_slew():
    """Apply apply_q then apply_slew_limit."""
    print("\n" + "=" * 60)
    print("TEST 8: apply_q + apply_slew_limit")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    params = ParamFrame()
    prev_L, prev_R = 0, 0
    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.apply_q(L, R, params.K_mode)
        L, R, prev_L, prev_R = pipeline.safety.apply_slew_limit(L, R, prev_L, prev_R)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "apply_q + apply_slew_limit")
    save_audio(samples, "8_q_then_slew")


def test_q_then_limiter():
    """Apply apply_q then apply_limiter."""
    print("\n" + "=" * 60)
    print("TEST 9: apply_q + apply_limiter")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    params = ParamFrame()
    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.apply_q(L, R, params.K_mode)
        L, R = pipeline.safety.apply_limiter(L, R)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "apply_q + apply_limiter")
    save_audio(samples, "9_q_then_limiter")


def test_slew_then_limiter():
    """Apply apply_slew_limit then apply_limiter."""
    print("\n" + "=" * 60)
    print("TEST 10: apply_slew_limit + apply_limiter")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    prev_L, prev_R = 0, 0
    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)
        L, R, prev_L, prev_R = pipeline.safety.apply_slew_limit(L, R, prev_L, prev_R)
        L, R = pipeline.safety.apply_limiter(L, R)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "apply_slew_limit + apply_limiter")
    save_audio(samples, "10_slew_then_limiter")


def test_step_without_slew():
    """Full step_block but skip slew limit."""
    print("\n" + "=" * 60)
    print("TEST 11: step_block without slew_limit")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    params = ParamFrame()
    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        block = pipeline.render_voice(frame)
        L = list(block)
        R = list(block)
        L, R = pipeline.safety.step_block(L, R, params, apply_master_ola=False, apply_slew_limit=False)
        samples.extend(L)

    pipeline.voice_off(0)
    analyze(samples, "step_block (no slew)")
    save_audio(samples, "11_step_without_slew")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP_BLOCK COMPONENT DEBUG")
    print("=" * 60)

    test_raw_render_voice()
    test_with_declick_only()
    test_with_q_only()
    test_with_limiter_only()
    test_with_slew_limit_only()
    test_with_dc_removal_only()
    test_full_step_block()
    test_q_then_slew()
    test_q_then_limiter()
    test_slew_then_limiter()
    test_step_without_slew()

    print("\n" + "=" * 60)
    print("Check audio_out/stepblock_debug/ for files")
    print("=" * 60)
