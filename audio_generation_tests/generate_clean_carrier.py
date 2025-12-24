"""Generate clean carrier lift signals (DRY only, beta=0)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from symcrash_python_ap.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.carrier_lift import Q15_ONE, Q15_HALF

FS = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SEC = FS // BLOCK_SIZE

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def generate_steady_note(name: str, note: int, duration_sec: float = 3.0):
    """Generate a steady note with clean carrier."""
    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=note)

    # Pure sine (fundamental only)
    if 0 in pipeline.carrier.states:
        pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

    L_out, R_out = [], []
    for _ in range(int(duration_sec * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=0,  # DRY only - clean carrier
        )
        L, R = pipeline.render_block(frame)
        L_out.extend(L)
        R_out.extend(R)

    pipeline.voice_off(0)

    metadata = audio_io.write_audio(name, L_out, R_out, subdirectory="clean_carrier")
    print(f"  {name}: {metadata.filepath}")


def generate_chromatic_scale(name: str, start_note: int, end_note: int, note_duration_sec: float = 0.25):
    """Generate chromatic scale with clean carrier."""
    pipeline = CarrierLiftPipeline()

    # Pure sine
    L_out, R_out = [], []

    for note in range(start_note, end_note + 1):
        pipeline.voice_on(0, note=note)
        if 0 in pipeline.carrier.states:
            pipeline.carrier.states[0].harmonic_amps_q15 = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]

        for _ in range(int(note_duration_sec * BLOCKS_PER_SEC)):
            frame = VoiceFrame(
                voice_id=0,
                note=note,
                f0_hz=midi_to_freq(note),
                gain_q15=Q15_HALF,
                beta_target_q15=0,
            )
            L, R = pipeline.render_block(frame)
            L_out.extend(L)
            R_out.extend(R)

        pipeline.voice_off(0)

    metadata = audio_io.write_audio(name, L_out, R_out, subdirectory="clean_carrier")
    print(f"  {name}: {metadata.filepath}")


def generate_saw_wave(name: str, note: int, duration_sec: float = 3.0):
    """Generate saw wave using carrier harmonics."""
    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=note)

    # Saw wave: harmonics decay as 1/n
    if 0 in pipeline.carrier.states:
        harmonics = []
        for n in range(1, 9):
            amp = Q15_ONE // n
            harmonics.append(amp)
        pipeline.carrier.states[0].harmonic_amps_q15 = harmonics

    L_out, R_out = [], []
    for _ in range(int(duration_sec * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        L, R = pipeline.render_block(frame)
        L_out.extend(L)
        R_out.extend(R)

    pipeline.voice_off(0)

    metadata = audio_io.write_audio(name, L_out, R_out, subdirectory="clean_carrier")
    print(f"  {name}: {metadata.filepath}")


def generate_square_wave(name: str, note: int, duration_sec: float = 3.0):
    """Generate square wave using carrier harmonics (odd harmonics only)."""
    pipeline = CarrierLiftPipeline()
    pipeline.voice_on(0, note=note)

    # Square wave: odd harmonics decay as 1/n
    if 0 in pipeline.carrier.states:
        harmonics = [0] * 8
        for i, n in enumerate([1, 3, 5, 7]):
            if i < 8:
                harmonics[n - 1] = Q15_ONE // n
        pipeline.carrier.states[0].harmonic_amps_q15 = harmonics

    L_out, R_out = [], []
    for _ in range(int(duration_sec * BLOCKS_PER_SEC)):
        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=Q15_HALF,
            beta_target_q15=0,
        )
        L, R = pipeline.render_block(frame)
        L_out.extend(L)
        R_out.extend(R)

    pipeline.voice_off(0)

    metadata = audio_io.write_audio(name, L_out, R_out, subdirectory="clean_carrier")
    print(f"  {name}: {metadata.filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING CLEAN CARRIER SIGNALS")
    print("=" * 60)

    print("\nSteady Notes (Pure Sine):")
    generate_steady_note("sine_C4_3s", 60, 3.0)
    generate_steady_note("sine_A1_3s", 33, 3.0)
    generate_steady_note("sine_C6_3s", 84, 3.0)
    generate_steady_note("sine_A4_440Hz_3s", 69, 3.0)

    print("\nChromatic Scales:")
    generate_chromatic_scale("chromatic_C3_C5", 48, 72, 0.25)
    generate_chromatic_scale("chromatic_C4_C6", 60, 84, 0.2)

    print("\nClassic Waveforms (C4):")
    generate_saw_wave("saw_C4_3s", 60, 3.0)
    generate_square_wave("square_C4_3s", 60, 3.0)

    print("\n" + "=" * 60)
    print("Done! Files in audio_out/clean_carrier/")
    print("=" * 60)
