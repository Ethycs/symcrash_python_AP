"""
Render FM Brass Presets to WAV Files
=====================================

Generates audio samples for each brass preset configuration.
"""

import numpy as np
import wave
import struct
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.vm.fm6_state import FM6State, Q15_ONE
from symcrash_python_ap.vm.fm6_engine import (
    fm6_generate_block_full,
    configure_brass,
    configure_synth_brass,
    configure_jump_brass,
)

SAMPLE_RATE = 48000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'brass_samples')


def midi_to_phase_inc(midi_note: int) -> int:
    """Convert MIDI note to Q32 phase increment."""
    freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    phase_inc = int((freq / SAMPLE_RATE) * (1 << 32))
    return phase_inc


def apply_adsr(audio: np.ndarray, a_ms: float, d_ms: float, s: float, r_ms: float) -> np.ndarray:
    """Apply ADSR envelope to audio."""
    total_samples = len(audio)

    a_samples = int((a_ms / 1000) * SAMPLE_RATE)
    d_samples = int((d_ms / 1000) * SAMPLE_RATE)
    r_samples = int((r_ms / 1000) * SAMPLE_RATE)

    # Sustain duration = total - attack - decay - release
    s_samples = max(0, total_samples - a_samples - d_samples - r_samples)

    envelope = np.zeros(total_samples, dtype=np.float64)
    idx = 0

    # Attack
    if a_samples > 0:
        envelope[idx:idx+a_samples] = np.linspace(0, 1, a_samples)
        idx += a_samples

    # Decay
    if d_samples > 0 and idx < total_samples:
        decay_end = min(idx + d_samples, total_samples)
        envelope[idx:decay_end] = np.linspace(1, s, decay_end - idx)
        idx = decay_end

    # Sustain
    if s_samples > 0 and idx < total_samples:
        sustain_end = min(idx + s_samples, total_samples)
        envelope[idx:sustain_end] = s
        idx = sustain_end

    # Release
    if r_samples > 0 and idx < total_samples:
        release_end = min(idx + r_samples, total_samples)
        envelope[idx:release_end] = np.linspace(s, 0, release_end - idx)
        idx = release_end

    # Apply envelope
    return (audio.astype(np.float64) * envelope).astype(np.int16)


def render_preset(name: str, configure_fn, midi_note: int, duration_ms: float,
                  a_ms: float, d_ms: float, s: float, r_ms: float) -> np.ndarray:
    """Render a preset to audio."""
    state = FM6State()
    configure_fn(state)

    # Set all operator envelopes to full (we apply ADSR separately)
    for op in state.operators:
        op.env_value_q15 = Q15_ONE

    phase_inc = midi_to_phase_inc(midi_note)

    # Calculate blocks needed
    total_samples = int((duration_ms / 1000) * SAMPLE_RATE)
    num_blocks = (total_samples + 63) // 64

    # Generate audio
    audio_blocks = []
    for _ in range(num_blocks):
        block = fm6_generate_block_full(state, phase_inc)
        audio_blocks.append(block)

    audio = np.concatenate(audio_blocks)[:total_samples]

    # Apply ADSR envelope
    audio = apply_adsr(audio, a_ms, d_ms, s, r_ms)

    return audio


def write_wav(filename: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Write audio to WAV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)

        # Convert to bytes
        data = struct.pack(f'<{len(audio)}h', *audio)
        wav.writeframes(data)

    print(f"  Written: {filename}")


def main():
    print("=" * 60)
    print("FM Brass Preset Audio Renderer")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define presets to render
    presets = [
        # (name, configure_fn, note, duration_ms, a_ms, d_ms, s, r_ms)
        ("brass_stab_v0", configure_brass, 60, 1500, 8, 85, 0.55, 95),      # C4
        ("brass_stab_v0_chord", configure_brass, 60, 1500, 8, 85, 0.55, 95),  # Will be chord
        ("synth_brass_v1", configure_synth_brass, 60, 1500, 3, 120, 0.7, 80),
        ("jump_brass_v0", configure_jump_brass, 60, 1500, 2, 140, 0.5, 100),
    ]

    # Render single notes
    print("\n[1/4] Rendering single notes...")
    for name, configure_fn, note, dur, a, d, s, r in presets:
        if "chord" in name:
            continue
        print(f"  Rendering {name} @ MIDI {note}...")
        audio = render_preset(name, configure_fn, note, dur, a, d, s, r)
        write_wav(os.path.join(OUTPUT_DIR, f"{name}_C4.wav"), audio)

    # Render at different octaves
    print("\n[2/4] Rendering octave range...")
    for note in [48, 60, 72]:  # C3, C4, C5
        note_name = {48: "C3", 60: "C4", 72: "C5"}[note]
        print(f"  Rendering brass_stab @ {note_name}...")
        audio = render_preset("brass_stab", configure_brass, note, 1500, 8, 85, 0.55, 95)
        write_wav(os.path.join(OUTPUT_DIR, f"brass_stab_{note_name}.wav"), audio)

    # Render chord (C major: C4, E4, G4)
    print("\n[3/4] Rendering brass chord (C major)...")
    chord_notes = [60, 64, 67]  # C4, E4, G4
    chord_audio = np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float64)

    for note in chord_notes:
        audio = render_preset("chord", configure_brass, note, 1500, 8, 85, 0.55, 95)
        chord_audio[:len(audio)] += audio.astype(np.float64)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(chord_audio))
    if max_val > 32767:
        chord_audio = chord_audio * (32767 / max_val)

    write_wav(os.path.join(OUTPUT_DIR, "brass_stab_Cmaj_chord.wav"),
              chord_audio.astype(np.int16))

    # Render stab pattern (short notes)
    print("\n[4/4] Rendering stab pattern...")
    stab_duration = 200  # 200ms stabs
    gap_duration = 100   # 100ms gap
    pattern_notes = [60, 60, 64, 67, 72]  # C4, C4, E4, G4, C5

    total_duration = len(pattern_notes) * (stab_duration + gap_duration)
    pattern_audio = np.zeros(int((total_duration / 1000) * SAMPLE_RATE), dtype=np.int16)

    offset = 0
    for note in pattern_notes:
        audio = render_preset("stab", configure_jump_brass, note, stab_duration, 2, 50, 0.3, 50)
        end = offset + len(audio)
        if end <= len(pattern_audio):
            pattern_audio[offset:end] = audio
        offset += int(((stab_duration + gap_duration) / 1000) * SAMPLE_RATE)

    write_wav(os.path.join(OUTPUT_DIR, "jump_brass_stab_pattern.wav"), pattern_audio)

    print("\n" + "=" * 60)
    print(f"All files written to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
