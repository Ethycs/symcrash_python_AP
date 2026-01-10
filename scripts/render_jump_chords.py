"""
Render Van Halen "Jump" Brass Chord Progression
================================================

The iconic synth brass riff from Van Halen's "Jump" (1984).
Key: C major, tempo ~130 BPM

Chord progression (intro riff):
- C major (C-E-G)
- F/C (C-F-A)
- G/B (B-D-G)
- C major (C-E-G)

Uses the jump_brass preset with heavy detune for that OB-Xa character.
"""

import numpy as np
import wave
import struct
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.vm.fm6_state import FM6State, Q15_ONE
from symcrash_python_ap.vm.fm6_engine import fm6_generate_block_full, configure_jump_brass

SAMPLE_RATE = 48000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'brass_samples')
BPM = 130


def midi_to_phase_inc(midi_note: int) -> int:
    """Convert MIDI note to Q32 phase increment."""
    freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    return int((freq / SAMPLE_RATE) * (1 << 32))


def ms_to_samples(ms: float) -> int:
    return int((ms / 1000) * SAMPLE_RATE)


def beats_to_samples(beats: float) -> int:
    """Convert beats to samples at current BPM."""
    seconds = (beats / BPM) * 60
    return int(seconds * SAMPLE_RATE)


def apply_adsr(audio: np.ndarray, a_ms: float, d_ms: float, s: float, r_ms: float) -> np.ndarray:
    """Apply ADSR envelope."""
    total = len(audio)
    a = min(ms_to_samples(a_ms), total)
    d = min(ms_to_samples(d_ms), total - a)
    r = min(ms_to_samples(r_ms), total)

    env = np.zeros(total, dtype=np.float64)
    idx = 0

    # Attack
    if a > 0:
        env[idx:idx+a] = np.linspace(0, 1, a)
        idx += a

    # Decay
    if d > 0 and idx < total:
        end = min(idx + d, total)
        env[idx:end] = np.linspace(1, s, end - idx)
        idx = end

    # Sustain (fill until release point)
    sustain_end = total - r
    if idx < sustain_end:
        env[idx:sustain_end] = s
        idx = sustain_end

    # Release
    if idx < total:
        env[idx:total] = np.linspace(s, 0, total - idx)

    return (audio.astype(np.float64) * env).astype(np.int16)


def render_note(midi_note: int, duration_samples: int, a_ms=2, d_ms=100, s=0.6, r_ms=80,
                mod_attack_ms=30) -> np.ndarray:
    """Render a single note with jump brass.

    mod_attack_ms: Time for modulators to fade in (softens harsh attack)
    """
    state = FM6State()
    configure_jump_brass(state)

    # Carriers at full, modulators will ramp up
    for op in state.operators:
        op.env_value_q15 = Q15_ONE

    phase_inc = midi_to_phase_inc(midi_note)
    num_blocks = (duration_samples + 63) // 64

    # Calculate modulator envelope ramp
    mod_attack_blocks = max(1, int((mod_attack_ms / 1000) * SAMPLE_RATE / 64))

    # Store original modulator levels
    mod_levels = [state.operators[4].level_q15, state.operators[5].level_q15]

    blocks = []
    for block_idx in range(num_blocks):
        # Ramp modulator levels during attack for softer onset
        if block_idx < mod_attack_blocks:
            ramp = block_idx / mod_attack_blocks
            # Smooth ramp (sine curve for natural feel)
            ramp = 0.5 - 0.5 * np.cos(ramp * np.pi)
            state.operators[4].level_q15 = int(mod_levels[0] * ramp)
            state.operators[5].level_q15 = int(mod_levels[1] * ramp)
        else:
            state.operators[4].level_q15 = mod_levels[0]
            state.operators[5].level_q15 = mod_levels[1]

        blocks.append(fm6_generate_block_full(state, phase_inc))

    audio = np.concatenate(blocks)[:duration_samples]
    return apply_adsr(audio, a_ms, d_ms, s, r_ms)


def render_chord(notes: list, duration_samples: int, a_ms=2, d_ms=100, s=0.6, r_ms=80,
                 mod_attack_ms=30) -> np.ndarray:
    """Render a chord (multiple notes)."""
    chord = np.zeros(duration_samples, dtype=np.float64)

    for note in notes:
        audio = render_note(note, duration_samples, a_ms, d_ms, s, r_ms, mod_attack_ms)
        chord += audio.astype(np.float64)

    # Normalize
    max_val = np.max(np.abs(chord))
    if max_val > 32767:
        chord = chord * (30000 / max_val)  # Leave headroom

    return chord.astype(np.int16)


def write_wav(filename: str, audio: np.ndarray):
    """Write WAV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(struct.pack(f'<{len(audio)}h', *audio))
    print(f"  Written: {os.path.basename(filename)}")


def main():
    print("=" * 60)
    print("Van Halen 'Jump' Brass Chord Renderer")
    print(f"BPM: {BPM}")
    print("=" * 60)

    # Jump chord voicings (MIDI notes)
    # The actual Jump riff uses these inversions:
    C_maj = [60, 64, 67]      # C4, E4, G4
    F_C = [60, 65, 69]        # C4, F4, A4 (F/C)
    G_B = [59, 62, 67]        # B3, D4, G4 (G/B)
    Am = [57, 60, 64]         # A3, C4, E4

    # Higher voicing for brightness
    C_maj_high = [60, 64, 67, 72]   # Add C5
    F_C_high = [60, 65, 69, 72]     # Add C5

    # === Individual chords ===
    print("\n[1/4] Rendering individual chords...")

    chord_duration = beats_to_samples(2)  # 2 beats per chord

    chords = [
        ("C_major", C_maj),
        ("F_over_C", F_C),
        ("G_over_B", G_B),
        ("A_minor", Am),
        ("C_major_high", C_maj_high),
    ]

    for name, notes in chords:
        audio = render_chord(notes, chord_duration, a_ms=2, d_ms=140, s=0.5, r_ms=100)
        write_wav(os.path.join(OUTPUT_DIR, f"jump_{name}.wav"), audio)

    # === Main Jump riff (4 bars) ===
    print("\n[2/4] Rendering Jump intro riff...")

    # Timing: | C (2 beats) | F/C (2 beats) | G/B (1 beat) | C (1 beat) | rest |
    # This is simplified - real riff has more syncopation

    riff_sequence = [
        (C_maj_high, 1.5),    # C major - dotted quarter
        (None, 0.5),          # rest - eighth
        (F_C_high, 1.5),      # F/C - dotted quarter
        (None, 0.5),          # rest
        (G_B, 1.0),           # G/B - quarter
        (C_maj, 1.0),         # C - quarter
        (None, 2.0),          # rest - half
    ]

    total_beats = sum(dur for _, dur in riff_sequence)
    total_samples = beats_to_samples(total_beats)
    riff_audio = np.zeros(total_samples, dtype=np.int16)

    pos = 0
    for notes, beats in riff_sequence:
        duration = beats_to_samples(beats)
        if notes is not None:
            # Staccato for stabs
            chord = render_chord(notes, duration, a_ms=2, d_ms=80, s=0.4, r_ms=60)
            end = min(pos + len(chord), total_samples)
            riff_audio[pos:end] = chord[:end-pos]
        pos += duration

    write_wav(os.path.join(OUTPUT_DIR, "jump_intro_riff.wav"), riff_audio)

    # === Full 4-bar phrase (repeat riff) ===
    print("\n[3/4] Rendering 4-bar phrase...")

    phrase = np.tile(riff_audio, 2)  # Repeat twice
    write_wav(os.path.join(OUTPUT_DIR, "jump_4bar_phrase.wav"), phrase)

    # === Sustained chord swell ===
    print("\n[4/4] Rendering sustained chord swell...")

    swell_duration = beats_to_samples(8)  # 2 bars sustained
    swell = render_chord(C_maj_high, swell_duration, a_ms=50, d_ms=500, s=0.7, r_ms=300)
    write_wav(os.path.join(OUTPUT_DIR, "jump_C_swell.wav"), swell)

    print("\n" + "=" * 60)
    print(f"All Jump brass files written to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
