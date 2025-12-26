"""
Debug: SafetyEngine causing jaggies?
====================================

Compare CarrierLiftPipeline output WITH and WITHOUT SafetyEngine.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, Q15_HALF

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def save_audio(samples: List[int], name: str) -> str:
    metadata = audio_io.write_audio(name, samples, samples, subdirectory="safety_debug", timestamp=False)
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

    # Check block boundaries specifically
    block_boundary_d2 = []
    for i in range(BLOCK_SIZE, len(d2), BLOCK_SIZE):
        # Check samples around block boundary
        for offset in range(-3, 4):
            if 0 <= i + offset < len(d2):
                block_boundary_d2.append(abs(d2[i + offset]))

    if block_boundary_d2:
        print(f"  Max d2 at block boundaries: {int(max(block_boundary_d2))}")


def test_with_safety():
    """Standard pipeline with SafetyEngine."""
    print("\n" + "=" * 60)
    print("TEST: WITH SafetyEngine")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    # Set pure sine
    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,  # DRY only
        )
        L, R = pipeline.render_block(frame)  # Uses SafetyEngine
        samples.extend(L)

    pipeline.voice_off(0)

    analyze(samples, "With SafetyEngine")
    filepath = save_audio(samples, "1_with_safety")
    print(f"  Saved: {filepath}")


def test_without_safety():
    """Pipeline bypassing SafetyEngine."""
    print("\n" + "=" * 60)
    print("TEST: WITHOUT SafetyEngine (bypass)")
    print("=" * 60)

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=60)

    # Set pure sine
    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    samples = []
    for _ in range(int(3.0 * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=60,
            f0_hz=midi_to_freq(60),
            gain_q15=Q15_HALF,
            beta_target_q15=0,  # DRY only
        )

        # Call render_voice directly, bypassing SafetyEngine
        block = pipeline.render_voice(frame)
        samples.extend(block)

    pipeline.voice_off(0)

    analyze(samples, "Without SafetyEngine")
    filepath = save_audio(samples, "2_without_safety")
    print(f"  Saved: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("SAFETY ENGINE JAGGIE DEBUG")
    print("=" * 60)

    test_with_safety()
    test_without_safety()

    print("\n" + "=" * 60)
    print("Compare the two files in audio_out/safety_debug/")
    print("If '2_without_safety' is clean, SafetyEngine OLA is the cause.")
    print("=" * 60)
