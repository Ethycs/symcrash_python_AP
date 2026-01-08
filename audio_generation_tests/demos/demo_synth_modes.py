"""
CCS Multi-Architecture Synth Modes Demo
=======================================

Demonstrates all synth modes implemented in the CCS pipeline:

1. Mode A - LUT Wavetable (coefficient-native)
2. Mode B - Harmonic Bank (coefficient-native)
3. Mode B + FM - DX-style feedback FM synthesis
4. Mode B + Drone - Slow-evolving pad with low base_mix
5. Sample Mode - Coefficient-native sample playback
6. Sample Mode + Pitch Shift - Block-interpolated pitch shifting

Each demo generates a short audio clip showcasing the mode.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState, OSC_MODE_A, OSC_MODE_B, OSC_MODE_SAMPLE
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
    metadata = audio_io.write_audio(name, L, R, subdirectory="synth_modes", timestamp=False)
    return metadata.filepath


def render_blocks(state: VMState, frame: VoiceFrame, num_blocks: int) -> np.ndarray:
    """Render multiple blocks and concatenate."""
    samples = []
    for _ in range(num_blocks):
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())
    return np.array(samples, dtype=np.int16)


# =============================================================================
# DEMO 1: Mode A - LUT Wavetable
# =============================================================================

def demo_mode_a_wavetable():
    """Mode A: LUT-based wavetable synthesis."""
    print("\n" + "=" * 60)
    print("DEMO 1: Mode A - LUT Wavetable")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    frame = VoiceFrame(
        note=60,  # Middle C
        coeff_osc_mode=0,  # MODE_A
        H_request=16,
    )

    duration_sec = 3.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("01_mode_a_wavetable", samples)
    print(f"  Mode: A (LUT Wavetable)")
    print(f"  Note: C4 (MIDI 60)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 2: Mode B - Harmonic Bank
# =============================================================================

def demo_mode_b_harmonic():
    """Mode B: Harmonic bank with dynamic generation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Mode B - Harmonic Bank")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B
        H_request=16,  # 16 harmonics
    )

    duration_sec = 3.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("02_mode_b_harmonic", samples)
    print(f"  Mode: B (Harmonic Bank)")
    print(f"  Harmonics: 16")
    print(f"  Note: C4 (MIDI 60)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 3: Mode B + FM Feedback
# =============================================================================

def demo_mode_b_fm():
    """Mode B with DX-style FM feedback."""
    print("\n" + "=" * 60)
    print("DEMO 3: Mode B + FM Feedback")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=1,  # MODE_B
        H_request=8,
        fm_depth_q15=16384,  # 50% FM depth
        fm_source=1,  # FM_SOURCE_FEEDBACK
    )

    duration_sec = 3.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("03_mode_b_fm_feedback", samples)
    print(f"  Mode: B + FM Feedback")
    print(f"  FM Depth: 50%")
    print(f"  Note: C4 (MIDI 60)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 4: Mode B + Drone Mode
# =============================================================================

def demo_mode_b_drone():
    """Mode B with drone mode (low base_mix for slow evolution)."""
    print("\n" + "=" * 60)
    print("DEMO 4: Mode B + Drone Mode")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=48)  # C3 for deeper drone

    frame = VoiceFrame(
        note=48,
        coeff_osc_mode=1,  # MODE_B
        H_request=8,
        drone_mode=True,  # Enables 5% base_mix for slow evolution
    )

    duration_sec = 5.0  # Longer for drone
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("04_mode_b_drone", samples)
    print(f"  Mode: B + Drone")
    print(f"  Base Mix: ~5% (slow evolution)")
    print(f"  Note: C3 (MIDI 48)")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 5: Sample Mode - Coefficient Native
# =============================================================================

def demo_sample_mode():
    """Sample playback in coefficient domain."""
    print("\n" + "=" * 60)
    print("DEMO 5: Sample Mode - Coefficient Native")
    print("=" * 60)

    # Create a sample buffer (sine wave at 440Hz)
    sample_rate = 48000
    duration_samples = 4800  # 100ms of sample
    freq = 440.0
    t = np.arange(duration_samples) / sample_rate
    sample_buffer = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)

    # Load sample with precomputed coefficients (coefficient-native mode)
    state.sample_state.set_buffer(
        sample_buffer,
        loop_enabled=True,
        precompute_coeffs=True  # Key: stays in coefficient domain
    )

    frame = VoiceFrame(
        note=60,
        coeff_osc_mode=2,  # OSC_MODE_SAMPLE
    )

    duration_sec = 3.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)
    samples = render_blocks(state, frame, num_blocks)

    filepath = save_demo("05_sample_coeff_native", samples)
    print(f"  Mode: Sample (Coefficient Native)")
    print(f"  Sample: 440Hz sine, 100ms looped")
    print(f"  Duration: {duration_sec}s")
    print(f"  Saved: {filepath}")

    return samples


# =============================================================================
# DEMO 6: Sample Mode + Pitch Shift (Coefficient Interpolation)
# =============================================================================

def demo_sample_pitch_shift():
    """Sample playback with coefficient-domain pitch shifting."""
    print("\n" + "=" * 60)
    print("DEMO 6: Sample Mode + Pitch Shift")
    print("=" * 60)

    # Create a richer sample buffer (saw-ish wave at 220Hz)
    sample_rate = 48000
    duration_samples = 9600  # 200ms of sample
    freq = 220.0
    t = np.arange(duration_samples) / sample_rate
    # Create a simple saw-like wave
    phase = (t * freq) % 1.0
    sample_buffer = ((phase * 2 - 1) * 16000).astype(np.int16)

    all_samples = []

    # Play at different pitches: original, octave up, octave down
    pitches = [
        (1.0, "Original (1.0x)"),
        (2.0, "Octave Up (2.0x)"),
        (0.5, "Octave Down (0.5x)"),
        (1.5, "Fifth Up (1.5x)"),
    ]

    for ratio, name in pitches:
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=60)

        # Load sample with precomputed coefficients
        state.sample_state.set_buffer(
            sample_buffer.copy(),
            loop_enabled=True,
            precompute_coeffs=True
        )
        state.sample_state.set_pitch_ratio(ratio)

        # Verify still in coefficient mode
        assert state.sample_state.use_coeff_buffer, f"Should stay in coeff mode at {ratio}x"

        frame = VoiceFrame(
            note=60,
            coeff_osc_mode=2,  # OSC_MODE_SAMPLE
        )

        # 1 second per pitch
        num_blocks = int(1.0 * BLOCKS_PER_SEC)
        samples = render_blocks(state, frame, num_blocks)
        all_samples.extend(samples.tolist())
        print(f"    {name}: {len(samples)} samples")

    all_samples = np.array(all_samples, dtype=np.int16)
    filepath = save_demo("06_sample_pitch_shift", all_samples)
    print(f"  Mode: Sample + Pitch Shift (Coefficient Interpolation)")
    print(f"  Pitches: 1.0x, 2.0x, 0.5x, 1.5x (1s each)")
    print(f"  Duration: 4.0s")
    print(f"  Saved: {filepath}")

    return all_samples


# =============================================================================
# DEMO 7: Mode B Melodic Sequence
# =============================================================================

def demo_mode_b_melody():
    """Mode B playing a simple melody with smooth pitch transitions (legato style)."""
    print("\n" + "=" * 60)
    print("DEMO 7: Mode B - Melodic Sequence (Legato)")
    print("=" * 60)

    # Simple melody: C E G C' G E C (arpeggio up and down)
    # Using legato_mode for smooth transitions (no re-trigger clicks)
    melody = [
        (60, 0.3),  # C4
        (64, 0.3),  # E4
        (67, 0.3),  # G4
        (72, 0.5),  # C5 (longer)
        (67, 0.3),  # G4
        (64, 0.3),  # E4
        (60, 0.5),  # C4 (longer)
    ]

    # Use single continuous state for smooth pitch changes
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=melody[0][0])  # Start at first note

    all_samples = []

    for note, duration in melody:
        # Use legato_mode=True for smooth pitch transitions without phase reset
        frame = VoiceFrame(
            note=note,
            coeff_osc_mode=1,  # MODE_B
            H_request=12,
            legato_mode=True,  # Key: preserves phase across pitch changes
        )

        num_blocks = int(duration * BLOCKS_PER_SEC)
        for _ in range(num_blocks):
            state, output = vm_step_block(state, frame)
            all_samples.extend(output.tolist())

    all_samples = np.array(all_samples, dtype=np.int16)
    filepath = save_demo("07_mode_b_melody", all_samples)
    print(f"  Mode: B (Harmonic Bank)")
    print(f"  Style: Legato (continuous state, smooth pitch changes)")
    print(f"  Melody: C4-E4-G4-C5-G4-E4-C4")
    print(f"  Duration: ~2.5s")
    print(f"  Saved: {filepath}")

    return all_samples


# =============================================================================
# DEMO 8: FM Bell (High FM Depth)
# =============================================================================

def demo_fm_bell():
    """FM synthesis with high depth for bell-like tones."""
    print("\n" + "=" * 60)
    print("DEMO 8: FM Bell (High Depth)")
    print("=" * 60)

    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=72)  # C5 for bell

    # Start with high FM depth, let it ring
    all_samples = []

    # Attack with high FM
    frame_attack = VoiceFrame(
        note=72,
        coeff_osc_mode=1,  # MODE_B
        H_request=6,  # Fewer harmonics for cleaner FM
        fm_depth_q15=24576,  # 75% FM depth (bright attack)
        fm_source=1,  # FM_SOURCE_FEEDBACK
    )

    # Sustain with moderate FM
    frame_sustain = VoiceFrame(
        note=72,
        coeff_osc_mode=1,
        H_request=6,
        fm_depth_q15=8192,  # 25% FM depth (mellow sustain)
        fm_source=1,
    )

    # Render attack (0.1s)
    num_attack = int(0.1 * BLOCKS_PER_SEC)
    for _ in range(num_attack):
        state, output = vm_step_block(state, frame_attack)
        all_samples.extend(output.tolist())

    # Render sustain (2s)
    num_sustain = int(2.0 * BLOCKS_PER_SEC)
    for _ in range(num_sustain):
        state, output = vm_step_block(state, frame_sustain)
        all_samples.extend(output.tolist())

    all_samples = np.array(all_samples, dtype=np.int16)
    filepath = save_demo("08_fm_bell", all_samples)
    print(f"  Mode: B + FM Feedback")
    print(f"  FM Depth: 75% attack -> 25% sustain")
    print(f"  Note: C5 (MIDI 72)")
    print(f"  Duration: 2.1s")
    print(f"  Saved: {filepath}")

    return all_samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CCS MULTI-ARCHITECTURE SYNTH MODES DEMO")
    print("=" * 60)
    print(f"\nSample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Output: audio_out/synth_modes/")

    # Run all demos
    demo_mode_a_wavetable()
    demo_mode_b_harmonic()
    demo_mode_b_fm()
    demo_mode_b_drone()
    demo_sample_mode()
    demo_sample_pitch_shift()
    demo_mode_b_melody()
    demo_fm_bell()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
    print("\nFiles saved to: audio_out/synth_modes/")
    print("  - 01_mode_a_wavetable.wav")
    print("  - 02_mode_b_harmonic.wav")
    print("  - 03_mode_b_fm_feedback.wav")
    print("  - 04_mode_b_drone.wav")
    print("  - 05_sample_coeff_native.wav")
    print("  - 06_sample_pitch_shift.wav")
    print("  - 07_mode_b_melody.wav")
    print("  - 08_fm_bell.wav")


if __name__ == "__main__":
    main()
