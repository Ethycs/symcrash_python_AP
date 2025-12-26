"""Generate comparison audio files to demonstrate WET pipeline fixes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "symcrash_python_ap"))

import numpy as np
from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.wavelet.carrier_lift import Q15_ONE, Q15_HALF, BLOCK_SIZE

FS = 48000
BLOCKS_PER_SEC = FS // BLOCK_SIZE

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def analyze(samples, name):
    """Analyze audio quality metrics."""
    arr = np.array(samples, dtype=np.float64)
    deltas = np.diff(arr)
    d2 = np.diff(deltas)

    print(f"  {name}:")
    print(f"    Peak: {int(np.max(np.abs(arr)))}")
    print(f"    Max d2: {int(np.max(np.abs(d2)))}")
    print(f"    Spikes: >500={np.sum(np.abs(d2) > 500)}, >1k={np.sum(np.abs(d2) > 1000)}")


def generate_dry_vs_wet(note: int, note_name: str, duration_sec: float = 3.0):
    """Generate DRY (carrier only) vs WET (wavelet blend) comparison."""
    print(f"\n{note_name} (MIDI {note}):")

    # DRY (beta=0, carrier only)
    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=note)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    L_dry, R_dry = [], []
    for _ in range(int(duration_sec * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=0,  # DRY - carrier only
        )
        L, R = pipeline.render_block(frame)
        L_dry.extend(L)
        R_dry.extend(R)

    pipeline.voice_off(0)
    analyze(L_dry, "DRY (carrier only)")

    # WET (beta=75%, wavelet blend)
    pipeline2 = CarrierLiftPipeline()
    pipeline2.voice_on(0, note=note)

    if 0 in pipeline2.carrier.states:
        pipeline2.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    L_wet, R_wet = [], []
    for _ in range(int(duration_sec * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=24576,  # 75% WET
        )
        L, R = pipeline2.render_block(frame)
        L_wet.extend(L)
        R_wet.extend(R)

    pipeline2.voice_off(0)
    analyze(L_wet, "WET (75% wavelet)")

    # Write files
    audio_io.write_audio(f"{note_name}_DRY", L_dry, R_dry, subdirectory="comparison")
    audio_io.write_audio(f"{note_name}_WET_75pct", L_wet, R_wet, subdirectory="comparison")


def generate_blend_sweep(note: int, note_name: str, duration_sec: float = 5.0):
    """Generate a sweep from 0% to 100% wet blend."""
    print(f"\n{note_name} Blend Sweep (0% -> 100%):")

    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=note)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    L_out, R_out = [], []
    total_blocks = int(duration_sec * BLOCKS_PER_SEC)

    for block_idx in range(total_blocks):
        # Sweep beta from 0 to Q15_ONE over duration
        progress = block_idx / total_blocks
        beta = int(progress * Q15_ONE)

        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=beta,
        )
        L, R = pipeline.render_block(frame)
        L_out.extend(L)
        R_out.extend(R)

    pipeline.voice_off(0)
    analyze(L_out, "Blend sweep")

    audio_io.write_audio(f"{note_name}_blend_sweep", L_out, R_out, subdirectory="comparison")


def generate_saw_wave(note: int, note_name: str, duration_sec: float = 3.0):
    """Generate saw wave comparison (DRY vs WET)."""
    print(f"\n{note_name} Saw Wave:")

    # Saw wave harmonics: 1/n falloff
    saw_harmonics = [Q15_ONE // n for n in range(1, 9)]

    # DRY
    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=note)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = saw_harmonics

    L_dry, R_dry = [], []
    for _ in range(int(duration_sec * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
            partial_amps_q15=saw_harmonics,
        )
        L, R = pipeline.render_block(frame)
        L_dry.extend(L)
        R_dry.extend(R)

    pipeline.voice_off(0)
    analyze(L_dry, "Saw DRY")

    # WET
    pipeline2 = CarrierLiftPipeline()
    pipeline2.voice_on(0, note=note)

    if 0 in pipeline2.carrier.states:
        pipeline2.carrier.states[0].harmonic_amps_q15 = saw_harmonics

    L_wet, R_wet = [], []
    for _ in range(int(duration_sec * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=24576,
            partial_amps_q15=saw_harmonics,
        )
        L, R = pipeline2.render_block(frame)
        L_wet.extend(L)
        R_wet.extend(R)

    pipeline2.voice_off(0)
    analyze(L_wet, "Saw WET")

    audio_io.write_audio(f"{note_name}_saw_DRY", L_dry, R_dry, subdirectory="comparison")
    audio_io.write_audio(f"{note_name}_saw_WET", L_wet, R_wet, subdirectory="comparison")


def generate_chromatic_run(start_note: int, end_note: int, note_duration_sec: float = 0.3):
    """Generate chromatic run with WET blend (legato style - no voice reset between notes)."""
    print(f"\nChromatic Run ({start_note} -> {end_note}):")

    pipeline = CarrierLiftPipeline()

    L_out, R_out = [], []

    # Start voice once at the beginning
    pipeline.voice_on(0, note=start_note)

    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    for note in range(start_note, end_note + 1):
        # Change pitch without resetting phase (legato)
        pipeline.carrier.set_pitch(0, note, slew_time_ms=5.0)  # 5ms glide

        for _ in range(int(note_duration_sec * BLOCKS_PER_SEC)):
            frame = VoiceFrame(
                voice_id=0,
                note=note,
                f0_hz=midi_to_freq(note),
                gain_q15=Q15_HALF,
                beta_target_q15=24576,  # 75% WET
            )
            L, R = pipeline.render_block(frame)
            L_out.extend(L)
            R_out.extend(R)

    pipeline.voice_off(0)

    analyze(L_out, "Chromatic run")
    audio_io.write_audio("chromatic_C3_C5_WET", L_out, R_out, subdirectory="comparison")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING COMPARISON AUDIO FILES")
    print("=" * 60)

    # Pure sine comparisons at different pitches
    generate_dry_vs_wet(60, "C4")   # Middle C
    generate_dry_vs_wet(48, "C3")   # Low C
    generate_dry_vs_wet(72, "C5")   # High C
    generate_dry_vs_wet(33, "A1")   # Very low

    # Blend sweep
    generate_blend_sweep(60, "C4")

    # Saw wave (rich harmonics)
    generate_saw_wave(60, "C4")
    generate_saw_wave(48, "C3")

    # Chromatic run
    generate_chromatic_run(48, 72)

    print("\n" + "=" * 60)
    print("Files written to audio_out/comparison/")
    print("=" * 60)
