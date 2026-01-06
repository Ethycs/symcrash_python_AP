"""
Für Elise Demo - Single Preset Mode B Test
===========================================

Generates the opening theme of Beethoven's Für Elise using the
WSE-STD-210 Mode B harmonic bank rotator with a single preset.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC
from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude

Q15_ONE = 32767
Q15_HALF = 16384
OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def generate_partial_amps(wave_type: WaveType, num_partials: int = 16):
    return [get_partial_amplitude(wave_type, k + 1) for k in range(num_partials)]


# Für Elise opening theme (public domain - composed 1810)
FUR_ELISE_MELODY = [
    (76, 1), (75, 1), (76, 1), (75, 1), (76, 1), (71, 1), (74, 1), (72, 1), (69, 2),
    (0, 1), (60, 1), (64, 1), (69, 1), (71, 2),
    (0, 1), (64, 1), (68, 1), (71, 1), (72, 2),
    (0, 1), (64, 1), (76, 1), (75, 1), (76, 1), (75, 1), (76, 1), (71, 1), (74, 1), (72, 1), (69, 2),
    (0, 1), (60, 1), (64, 1), (69, 1), (71, 2),
    (0, 1), (64, 1), (72, 1), (71, 1), (69, 3),
]


def render_melody_mode_b(bpm: float = 120):
    """Render Für Elise with Mode B harmonic bank rotator."""
    partial_amps = generate_partial_amps(WaveType.SAW)
    beta = Q15_HALF  # 50% wet

    eighth_note_sec = 60.0 / bpm / 2
    eighth_note_blocks = int(eighth_note_sec * BLOCKS_PER_SEC)
    attack_blocks = 6
    decay_blocks = 12
    sustain_level = 0.7
    release_blocks = 14

    pipeline = CarrierLiftPipeline()
    samples_L = []
    samples_R = []

    voice_active = False
    prev_note = None

    for i, (midi_note, duration) in enumerate(FUR_ELISE_MELODY):
        note_blocks = int(duration * eighth_note_blocks)
        next_is_rest = (i + 1 >= len(FUR_ELISE_MELODY) or FUR_ELISE_MELODY[i + 1][0] == 0)

        if midi_note == 0:
            if voice_active:
                last_env = sustain_level * 0.85
                for rel_idx in range(release_blocks):
                    env = last_env * (1.0 - (rel_idx + 1) / release_blocks)
                    frame = VoiceFrame(
                        voice_id=0, note=prev_note or 60,
                        f0_hz=midi_to_freq(prev_note or 60),
                        gain_q15=int(Q15_ONE * 0.85 * env),
                        beta_target_q15=beta,
                        partial_amps_q15=partial_amps,
                        coeff_osc_mode=1, H_request=16,
                    )
                    L, R = pipeline.render_block(frame)
                    samples_L.extend(L)
                    samples_R.extend(R)
                pipeline.voice_off(0)
                voice_active = False

            silence_blocks = max(0, note_blocks - release_blocks)
            for _ in range(silence_blocks):
                samples_L.extend([0] * BLOCK_SIZE)
                samples_R.extend([0] * BLOCK_SIZE)
        else:
            if not voice_active:
                pipeline.voice_on(0, note=midi_note)
                voice_active = True
                is_new_phrase = True
            else:
                is_new_phrase = False

            for block_idx in range(note_blocks):
                if is_new_phrase and block_idx < attack_blocks:
                    env = 0.5 + 0.5 * (block_idx + 1) / attack_blocks
                elif is_new_phrase and block_idx < attack_blocks + decay_blocks:
                    env = 1.0 - (1.0 - sustain_level) * (block_idx - attack_blocks) / decay_blocks
                else:
                    env = sustain_level * (1.0 - 0.15 * block_idx / max(1, note_blocks))

                frame = VoiceFrame(
                    voice_id=0, note=midi_note,
                    f0_hz=midi_to_freq(midi_note),
                    gain_q15=int(Q15_ONE * 0.85 * env),
                    beta_target_q15=beta,
                    partial_amps_q15=partial_amps,
                    coeff_osc_mode=1, H_request=16,
                )
                L, R = pipeline.render_block(frame)
                samples_L.extend(L)
                samples_R.extend(R)

            prev_note = midi_note

    if voice_active:
        last_env = sustain_level * 0.85
        for rel_idx in range(release_blocks):
            env = last_env * (1.0 - (rel_idx + 1) / release_blocks)
            frame = VoiceFrame(
                voice_id=0, note=prev_note or 60,
                f0_hz=midi_to_freq(prev_note or 60),
                gain_q15=int(Q15_ONE * 0.85 * env),
                beta_target_q15=beta,
                partial_amps_q15=partial_amps,
                coeff_osc_mode=1, H_request=16,
            )
            L, R = pipeline.render_block(frame)
            samples_L.extend(L)
            samples_R.extend(R)
        pipeline.voice_off(0)

    return samples_L, samples_R


def main():
    print("=" * 60)
    print("FÜR ELISE - Mode B (WSE-STD-210) Demo")
    print("=" * 60)
    print(f"Sample rate: {FS} Hz")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Using: Mode B Harmonic Bank/Rotator")
    print()

    print("Rendering...")
    L, R = render_melody_mode_b(bpm=120)

    arr = np.array(L, dtype=np.float64)
    duration = len(L) / FS
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr ** 2))

    print(f"  Duration: {duration:.2f}s")
    print(f"  Peak: {peak}, RMS: {rms:.0f}")

    metadata = audio_io.write_audio(
        "fur_elise_mode_b", L, R,
        subdirectory="fur_elise", timestamp=False
    )
    print(f"  Saved: {metadata.filepath}")
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
