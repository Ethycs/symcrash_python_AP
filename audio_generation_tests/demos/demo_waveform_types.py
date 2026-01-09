"""
CCS Waveform Types Demo
=======================

Demonstrates all waveform types available via the cyclic shift matrix approach:

1. Sine - Pure sine (quarter-wave LUT)
2. Square - 50% duty square wave
3. Pulse - 25% duty pulse wave
4. Sawtooth - Rising ramp with jump
5. Triangle - Symmetric triangle
6. PWM Sweep - Animated pulse width

Each waveform is generated bandlimited via Mode B harmonic bank.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState, OSC_MODE_B
from symcrash_python_ap.vm.waveform_matrix import WaveformType
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter

# Constants
FS = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SEC = FS // BLOCK_SIZE

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def samples_to_stereo(samples):
    """Convert mono int16 samples to stereo lists."""
    return samples.tolist(), samples.tolist()


def save_demo(name: str, samples: np.ndarray) -> str:
    """Save demo to WAV file."""
    L, R = samples_to_stereo(samples)
    metadata = audio_io.write_audio(name, L, R, subdirectory="waveforms", timestamp=False)
    return metadata.filepath


def render_blocks(state: VMState, frame: VoiceFrame, num_blocks: int) -> np.ndarray:
    """Render multiple blocks and concatenate."""
    samples = []
    for _ in range(num_blocks):
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())
    return np.array(samples, dtype=np.int16)


# =============================================================================
# DEMO 1: Pure Sine
# =============================================================================

def demo_sine():
    """Pure sine wave (default waveform_type=0)."""
    print("\n" + "=" * 60)
    print("DEMO 1: Pure Sine Wave")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B
        H_request=16,
        waveform_type=WaveformType.SINE,
    )

    duration_sec = 2.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("01_sine", samples)
    print(f"  Waveform: Sine")
    print(f"  Note: C4 (MIDI 60)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 2: Square Wave
# =============================================================================

def demo_square():
    """Square wave (50% duty cycle)."""
    print("\n" + "=" * 60)
    print("DEMO 2: Square Wave")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B
        H_request=32,  # More harmonics for square
        waveform_type=WaveformType.SQUARE,
    )

    duration_sec = 2.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("02_square", samples)
    print(f"  Waveform: Square (50% duty)")
    print(f"  Note: C4 (MIDI 60)")
    print(f"  Harmonics: 32 (odd only)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 3: Pulse Wave (25% duty)
# =============================================================================

def demo_pulse():
    """Pulse wave with 25% duty cycle."""
    print("\n" + "=" * 60)
    print("DEMO 3: Pulse Wave (25% Duty)")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B
        H_request=32,
        waveform_type=WaveformType.PULSE,
        pwm_width_q15=8192,  # 25% duty
    )

    duration_sec = 2.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("03_pulse_25", samples)
    print(f"  Waveform: Pulse (25% duty)")
    print(f"  Note: C4 (MIDI 60)")
    print(f"  Harmonics: 32")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 4: Sawtooth Wave
# =============================================================================

def demo_sawtooth():
    """Sawtooth wave (rising ramp)."""
    print("\n" + "=" * 60)
    print("DEMO 4: Sawtooth Wave")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B
        H_request=32,  # More harmonics for saw
        waveform_type=WaveformType.SAW,
    )

    duration_sec = 2.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("04_sawtooth", samples)
    print(f"  Waveform: Sawtooth (rising ramp)")
    print(f"  Note: C4 (MIDI 60)")
    print(f"  Harmonics: 32 (1/n rolloff)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 5: Triangle Wave
# =============================================================================

def demo_triangle():
    """Triangle wave (symmetric, smooth)."""
    print("\n" + "=" * 60)
    print("DEMO 5: Triangle Wave")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B
        H_request=32,
        waveform_type=WaveformType.TRIANGLE,
    )

    duration_sec = 2.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("05_triangle", samples)
    print(f"  Waveform: Triangle (symmetric)")
    print(f"  Note: C4 (MIDI 60)")
    print(f"  Harmonics: 32 (odd, 1/n^2 rolloff)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 6: PWM Sweep
# =============================================================================

def demo_pwm_sweep():
    """PWM sweep from 10% to 90% duty cycle."""
    print("\n" + "=" * 60)
    print("DEMO 6: PWM Sweep (10% -> 90%)")
    print("=" * 60)

    all_samples = []
    duration_sec = 4.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    for i in range(num_blocks):
        # Sweep duty cycle from 10% to 90%
        progress = i / num_blocks
        duty = int(3277 + progress * (29491 - 3277))  # 10% to 90% in Q15

        frame = VoiceFrame(
            note=60,
            coeff_osc_mode=1,
            H_request=32,
            waveform_type=WaveformType.PULSE,
            pwm_width_q15=duty,
        )

        state, output = vm_step_block(state, frame)
        all_samples.extend(output.tolist())

    samples = np.array(all_samples, dtype=np.int16)
    filepath = save_demo("06_pwm_sweep", samples)
    print(f"  Waveform: PWM Sweep")
    print(f"  Note: C4 (MIDI 60)")
    print(f"  Duty: 10% -> 90% over {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 7: Waveform Comparison (A-B-C-D sequence)
# =============================================================================

def demo_waveform_comparison():
    """Play all waveforms in sequence for comparison."""
    print("\n" + "=" * 60)
    print("DEMO 7: Waveform Comparison")
    print("=" * 60)

    waveforms = [
        (WaveformType.SINE, "Sine"),
        (WaveformType.TRIANGLE, "Triangle"),
        (WaveformType.SAW, "Sawtooth"),
        (WaveformType.SQUARE, "Square"),
        (WaveformType.PULSE, "Pulse 25%"),
    ]

    all_samples = []
    duration_per_wave = 0.75  # 750ms each

    for wave_type, name in waveforms:
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=60)

        frame = VoiceFrame(
            note=60,
            coeff_osc_mode=1,
            H_request=32,
            waveform_type=wave_type,
            pwm_width_q15=8192 if wave_type == WaveformType.PULSE else 16384,
        )

        num_blocks = int(duration_per_wave * BLOCKS_PER_SEC)
        samples = render_blocks(state, frame, num_blocks)
        all_samples.extend(samples.tolist())
        print(f"    {name}: {len(samples)} samples")

    samples = np.array(all_samples, dtype=np.int16)
    filepath = save_demo("07_waveform_comparison", samples)
    print(f"  Sequence: Sine -> Triangle -> Saw -> Square -> Pulse")
    print(f"  Duration: {len(waveforms) * duration_per_wave}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 8: High Pitch Aliasing Test
# =============================================================================

def demo_aliasing_test():
    """Test bandlimiting at high frequencies."""
    print("\n" + "=" * 60)
    print("DEMO 8: Aliasing Test (High Pitch)")
    print("=" * 60)

    # Play square wave at progressively higher pitches
    # At high pitches, fewer harmonics should be active to avoid aliasing

    notes = [60, 72, 84, 96]  # C4, C5, C6, C7
    all_samples = []

    for note in notes:
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=note)

        frame = VoiceFrame(
            note=note,
            coeff_osc_mode=1,
            H_request=64,  # Request many, but bandlimiter will reduce
            waveform_type=WaveformType.SAW,
        )

        num_blocks = int(0.5 * BLOCKS_PER_SEC)
        samples = render_blocks(state, frame, num_blocks)
        all_samples.extend(samples.tolist())

        # Check how many harmonics were actually used
        H_used = state.mode_b_state.H_active
        print(f"    Note {note} (C{note//12 - 1}): H_active={H_used}")

    samples = np.array(all_samples, dtype=np.int16)
    filepath = save_demo("08_aliasing_test", samples)
    print(f"  Waveform: Sawtooth at C4-C5-C6-C7")
    print(f"  Duration: 2.0s (0.5s per octave)")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CCS WAVEFORM TYPES DEMO")
    print("=" * 60)
    print(f"\nSample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Output: audio_out/waveforms/")

    # Run all demos
    demo_sine()
    demo_square()
    demo_pulse()
    demo_sawtooth()
    demo_triangle()
    demo_pwm_sweep()
    demo_waveform_comparison()
    demo_aliasing_test()

    print("\n" + "=" * 60)
    print("ALL WAVEFORM DEMOS COMPLETE")
    print("=" * 60)
    print("\nFiles saved to: audio_out/waveforms/")
    print("  - 01_sine.wav")
    print("  - 02_square.wav")
    print("  - 03_pulse_25.wav")
    print("  - 04_sawtooth.wav")
    print("  - 05_triangle.wav")
    print("  - 06_pwm_sweep.wav")
    print("  - 07_waveform_comparison.wav")
    print("  - 08_aliasing_test.wav")


if __name__ == "__main__":
    main()
