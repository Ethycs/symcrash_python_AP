"""
Test Brass Stab Rendering with Operators
=========================================

Quick test to render brass_stab and verify operators are applied.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # for test_config

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.presets import load_preset
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude

Q15_ONE = 32767
Q15_HALF = 16384

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def generate_partial_amps(wave_type: WaveType, num_partials: int = 16):
    return [get_partial_amplitude(wave_type, k + 1) for k in range(num_partials)]


def test_render():
    print("=" * 60)
    print("BRASS STAB RENDER TEST")
    print("=" * 60)

    # Load preset
    ir = load_preset("brass_stab")
    print(f"\nPreset: {ir.meta.name}")

    # Get operators
    stable = ir.layers.L3_cmmm.emit_templates.get("Stable")
    ops_enable = stable.ops_enable if stable else []
    families = stable.families if stable else {}

    # Get beta
    policy = ir.compile.completion_policy
    beta_float = getattr(policy.default_tree, 'beta', 0.0)
    beta_q15 = int(beta_float * Q15_ONE)

    print(f"Ops: {ops_enable}")
    print(f"Families: {families}")
    print(f"Beta: {beta_q15} ({int(beta_float * 100)}% wet)")

    # Create pipeline
    pipeline = CarrierLiftPipeline()

    # Render A4 for 1 second
    note = 69
    duration_sec = 1.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    partial_amps = generate_partial_amps(WaveType.SAW)

    samples_L = []
    samples_R = []

    pipeline.voice_on(0, note=note)

    for block_idx in range(num_blocks):
        # Simple ADSR (fast attack)
        if block_idx < 2:
            env = block_idx / 2.0
        elif block_idx < 20:
            env = 1.0 - (block_idx - 2) / 18.0 * 0.5
        else:
            env = 0.5

        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=int(Q15_ONE * 0.85 * env),
            beta_target_q15=beta_q15,
            partial_amps_q15=partial_amps,
            ops_enable=ops_enable,
            families=families,
        )

        L, R = pipeline.render_block(frame)
        samples_L.extend(L)
        samples_R.extend(R)

    pipeline.voice_off(0)

    # Analyze
    arr = np.array(samples_L, dtype=np.float64)
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr ** 2))

    print(f"\nRendered {num_blocks} blocks ({duration_sec}s)")
    print(f"Peak: {peak} ({100*peak/32767:.1f}%)")
    print(f"RMS: {rms:.0f}")

    # Save
    filepath = audio_io.write_audio(
        "brass_stab_ops_test", samples_L, samples_R,
        subdirectory="test",
        timestamp=False
    ).filepath
    print(f"Saved: {filepath}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_render()
