"""
Classic Synth Audio Generation Tests (Baby Fourier)
====================================================

Tests the Baby Fourier / Harmonic Target Injector for classic synthesizer sounds.

Generates:
1. Classic waveforms (sine, saw, pulse/square, triangle)
2. Classic synth patches (bass, lead, pad, pluck)
3. Filter sweeps (brightness parameter)
4. Detuned unison / supersaw
5. Arpeggios and sequences
6. Chord stabs

Uses direct additive synthesis via synthesize_harmonic_audio() for precise pitch.

Output: audio_generation_tests/audio_out/classic_synth_tests/
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE
from audio_io import write_audio

from symcrash_python_ap.wavelet.harmonic_injector import (
    WaveType,
    synthesize_harmonic_audio,
    create_harmonic_coeffs,
    note_to_freq,
    Q15_ONE,
    HarmonicTargetGenerator,
)
from symcrash_python_ap.fixed import sat16, mul_q15_sat


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_SUBDIR = "classic_synth_tests"
SAMPLE_RATE = 48000
DURATION_SEC = 2.0


# =============================================================================
# AUDIO HELPERS
# =============================================================================

def save_audio(samples_L: List[int], samples_R: List[int], name: str) -> str:
    """Save audio using audio_io module."""
    metadata = write_audio(name, samples_L, samples_R, timestamp=True,
                           subdirectory=OUTPUT_SUBDIR, verbose=False)
    return metadata.filepath


def compute_rms(samples: List[int]) -> float:
    """Compute RMS of sample list."""
    if not samples:
        return 0.0
    return np.sqrt(sum(s * s for s in samples) / len(samples))


def apply_envelope(samples: List[int], attack_sec: float, decay_sec: float,
                   sustain_level: float, release_sec: float,
                   gate_off_sec: Optional[float] = None) -> List[int]:
    """Apply ADSR envelope to samples."""
    n = len(samples)
    if gate_off_sec is None:
        gate_off_sec = n / SAMPLE_RATE - release_sec

    attack_samples = int(attack_sec * SAMPLE_RATE)
    decay_samples = int(decay_sec * SAMPLE_RATE)
    gate_off_sample = int(gate_off_sec * SAMPLE_RATE)
    release_samples = int(release_sec * SAMPLE_RATE)

    result = []
    for i, s in enumerate(samples):
        if i < attack_samples:
            # Attack phase
            env = i / attack_samples if attack_samples > 0 else 1.0
        elif i < attack_samples + decay_samples:
            # Decay phase
            decay_pos = (i - attack_samples) / decay_samples if decay_samples > 0 else 1.0
            env = 1.0 - (1.0 - sustain_level) * decay_pos
        elif i < gate_off_sample:
            # Sustain phase
            env = sustain_level
        elif i < gate_off_sample + release_samples:
            # Release phase
            release_pos = (i - gate_off_sample) / release_samples if release_samples > 0 else 1.0
            env = sustain_level * (1.0 - release_pos)
        else:
            env = 0.0

        result.append(int(s * env))

    return result


def mix_to_stereo(mono: List[int], pan: float = 0.0) -> Tuple[List[int], List[int]]:
    """Convert mono to stereo with panning (-1 to +1)."""
    l_gain = 0.5 * (1 - pan)
    r_gain = 0.5 * (1 + pan)
    L = [int(s * l_gain) for s in mono]
    R = [int(s * r_gain) for s in mono]
    return L, R


# =============================================================================
# SYNTHESIS HELPERS
# =============================================================================

def render_note(note: int, duration_sec: float, wave_type: WaveType,
                amplitude: int = 0x5000, partials: int = 12,
                brightness: int = Q15_ONE) -> List[int]:
    """Render a single note using Baby Fourier synthesis."""
    freq = note_to_freq(note)
    n_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)

    samples = []
    phase_state = None

    for _ in range(n_blocks):
        block, phase_state = synthesize_harmonic_audio(
            frequency=freq,
            amplitude=amplitude,
            wave_type=wave_type,
            partials_n=partials,
            brightness_q15=brightness,
            block_size=BLOCK_SIZE,
            sample_rate=SAMPLE_RATE,
            phase_state=phase_state
        )
        samples.extend(block)

    return samples


def render_chord(notes: List[int], duration_sec: float, wave_type: WaveType,
                 amplitude: int = 0x4000, partials: int = 8,
                 brightness: int = Q15_ONE) -> List[int]:
    """Render multiple notes as a chord."""
    n_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)
    n_samples = n_blocks * BLOCK_SIZE

    # Mix all notes
    mixed = [0] * n_samples
    amp_per_voice = amplitude // len(notes)

    for note in notes:
        note_samples = render_note(note, duration_sec, wave_type, amp_per_voice, partials, brightness)
        for i, s in enumerate(note_samples):
            if i < n_samples:
                mixed[i] = sat16(mixed[i] + s)

    return mixed


def render_unison(note: int, n_voices: int, detune_cents: float,
                  duration_sec: float, wave_type: WaveType,
                  amplitude: int = 0x5000, partials: int = 12) -> List[int]:
    """Render detuned unison voices (supersaw-style)."""
    n_blocks = int(duration_sec * SAMPLE_RATE / BLOCK_SIZE)
    n_samples = n_blocks * BLOCK_SIZE

    mixed = [0] * n_samples
    amp_per_voice = amplitude // n_voices

    for v in range(n_voices):
        # Spread detune across voices
        if n_voices > 1:
            detune = detune_cents * (v / (n_voices - 1) * 2 - 1)
        else:
            detune = 0

        freq = note_to_freq(note, int(detune))
        phase_state = None

        offset = 0
        for _ in range(n_blocks):
            block, phase_state = synthesize_harmonic_audio(
                frequency=freq,
                amplitude=amp_per_voice,
                wave_type=wave_type,
                partials_n=partials,
                block_size=BLOCK_SIZE,
                phase_state=phase_state
            )
            for i, s in enumerate(block):
                if offset + i < n_samples:
                    mixed[offset + i] = sat16(mixed[offset + i] + s)
            offset += BLOCK_SIZE

    return mixed


# =============================================================================
# CLASSIC WAVEFORM TESTS
# =============================================================================

class TestClassicWaveforms:
    """Test classic synthesizer waveforms."""

    @pytest.mark.parametrize("wave_type", [WaveType.SINE, WaveType.SAW, WaveType.PULSE, WaveType.TRIANGLE])
    def test_waveform(self, wave_type):
        """Test each classic waveform."""
        samples = render_note(60, 2.0, wave_type, amplitude=0x6000, partials=16)
        samples = apply_envelope(samples, 0.01, 0.1, 0.8, 0.3)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, f"waveform_{wave_type.name.lower()}")

        rms = compute_rms(samples)
        print(f"\n  {wave_type.name}: RMS={rms:.0f}")
        print(f"    Saved: {filepath}")

    def test_all_waveforms_sequence(self):
        """Play all waveforms in sequence."""
        all_samples = []

        for wave_type in [WaveType.SINE, WaveType.SAW, WaveType.PULSE, WaveType.TRIANGLE]:
            samples = render_note(60, 1.0, wave_type, amplitude=0x5000, partials=12)
            samples = apply_envelope(samples, 0.01, 0.1, 0.8, 0.2)
            all_samples.extend(samples)
            # Small gap between waveforms
            all_samples.extend([0] * int(SAMPLE_RATE * 0.25))

        L, R = mix_to_stereo(all_samples)
        filepath = save_audio(L, R, "waveforms_sequence")
        print(f"\n  Waveforms sequence: {len(all_samples)} samples")
        print(f"    Saved: {filepath}")


# =============================================================================
# BRIGHTNESS / FILTER SWEEP TESTS
# =============================================================================

class TestBrightness:
    """Test brightness parameter (simulates filter cutoff)."""

    @pytest.mark.parametrize("brightness_pct", [25, 50, 75, 100])
    def test_brightness_levels(self, brightness_pct):
        """Test different brightness levels."""
        brightness = int(Q15_ONE * brightness_pct / 100)
        samples = render_note(48, 2.0, WaveType.SAW, amplitude=0x5000,
                              partials=16, brightness=brightness)
        samples = apply_envelope(samples, 0.01, 0.1, 0.8, 0.3)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, f"brightness_{brightness_pct}pct")

        print(f"\n  Brightness {brightness_pct}%: saved to {filepath}")

    def test_filter_sweep(self):
        """Sweep brightness from dark to bright (filter sweep effect)."""
        n_blocks = int(3.0 * SAMPLE_RATE / BLOCK_SIZE)
        freq = note_to_freq(48)  # C3

        samples = []
        phase_state = None

        for block_idx in range(n_blocks):
            # Sweep brightness from 10% to 100%
            t = block_idx / n_blocks
            brightness = int(Q15_ONE * (0.1 + 0.9 * t))

            block, phase_state = synthesize_harmonic_audio(
                frequency=freq,
                amplitude=0x5000,
                wave_type=WaveType.SAW,
                partials_n=16,
                brightness_q15=brightness,
                block_size=BLOCK_SIZE,
                phase_state=phase_state
            )
            samples.extend(block)

        samples = apply_envelope(samples, 0.01, 0.0, 1.0, 0.5)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "filter_sweep")

        print(f"\n  Filter sweep: {len(samples)} samples")
        print(f"    Saved: {filepath}")


# =============================================================================
# CLASSIC SYNTH PATCHES
# =============================================================================

class TestClassicPatches:
    """Classic synthesizer patches."""

    def test_bass_saw(self):
        """Classic saw bass (Moog-style)."""
        samples = render_note(36, 2.0, WaveType.SAW, amplitude=0x6000, partials=8)
        # Punchy envelope
        samples = apply_envelope(samples, 0.005, 0.2, 0.6, 0.1)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "patch_bass_saw")

        print(f"\n  Saw Bass: saved to {filepath}")

    def test_bass_square(self):
        """Classic square bass (sub bass)."""
        samples = render_note(36, 2.0, WaveType.PULSE, amplitude=0x6000, partials=6)
        samples = apply_envelope(samples, 0.002, 0.1, 0.7, 0.15)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "patch_bass_square")

        print(f"\n  Square Bass: saved to {filepath}")

    def test_lead_saw(self):
        """Classic saw lead."""
        samples = render_note(72, 2.0, WaveType.SAW, amplitude=0x5000, partials=12)
        samples = apply_envelope(samples, 0.01, 0.1, 0.8, 0.2)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "patch_lead_saw")

        print(f"\n  Saw Lead: saved to {filepath}")

    def test_lead_square(self):
        """Classic square lead (hollow sound)."""
        samples = render_note(72, 2.0, WaveType.PULSE, amplitude=0x5000, partials=10)
        samples = apply_envelope(samples, 0.01, 0.15, 0.7, 0.25)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "patch_lead_square")

        print(f"\n  Square Lead: saved to {filepath}")

    def test_pad_triangle(self):
        """Soft triangle pad."""
        samples = render_note(60, 4.0, WaveType.TRIANGLE, amplitude=0x5000, partials=8)
        samples = apply_envelope(samples, 0.5, 0.5, 0.7, 1.0)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "patch_pad_triangle")

        print(f"\n  Triangle Pad: saved to {filepath}")

    def test_pluck_saw(self):
        """Plucked saw (like a plucked string)."""
        samples = render_note(60, 1.5, WaveType.SAW, amplitude=0x6000, partials=12)
        samples = apply_envelope(samples, 0.002, 0.3, 0.0, 0.0, gate_off_sec=0.002)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "patch_pluck_saw")

        print(f"\n  Saw Pluck: saved to {filepath}")

    def test_organ_sine(self):
        """Organ-like sound using pure sine."""
        samples = render_note(60, 3.0, WaveType.SINE, amplitude=0x6000, partials=1)
        samples = apply_envelope(samples, 0.01, 0.0, 1.0, 0.1)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "patch_organ_sine")

        print(f"\n  Sine Organ: saved to {filepath}")


# =============================================================================
# SUPERSAW / UNISON TESTS
# =============================================================================

class TestUnison:
    """Test detuned unison / supersaw sounds."""

    def test_supersaw_2_voices(self):
        """2-voice detuned saw."""
        samples = render_unison(60, n_voices=2, detune_cents=10,
                                duration_sec=2.0, wave_type=WaveType.SAW)
        samples = apply_envelope(samples, 0.01, 0.1, 0.8, 0.3)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "unison_2voice")

        print(f"\n  2-voice unison: saved to {filepath}")

    def test_supersaw_5_voices(self):
        """5-voice supersaw (classic trance lead)."""
        samples = render_unison(60, n_voices=5, detune_cents=15,
                                duration_sec=2.0, wave_type=WaveType.SAW,
                                partials=10)
        samples = apply_envelope(samples, 0.01, 0.1, 0.8, 0.3)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "supersaw_5voice")

        print(f"\n  5-voice supersaw: saved to {filepath}")

    def test_supersaw_7_voices(self):
        """7-voice ultra-wide supersaw."""
        samples = render_unison(60, n_voices=7, detune_cents=20,
                                duration_sec=2.0, wave_type=WaveType.SAW,
                                partials=12)
        samples = apply_envelope(samples, 0.05, 0.2, 0.8, 0.5)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "supersaw_7voice")

        print(f"\n  7-voice supersaw: saved to {filepath}")

    def test_unison_square(self):
        """Detuned square unison (PWM-like thickness)."""
        samples = render_unison(60, n_voices=3, detune_cents=8,
                                duration_sec=2.0, wave_type=WaveType.PULSE,
                                partials=8)
        samples = apply_envelope(samples, 0.01, 0.1, 0.7, 0.3)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "unison_square")

        print(f"\n  Square unison: saved to {filepath}")


# =============================================================================
# CHORD TESTS
# =============================================================================

class TestChords:
    """Test chord voicings."""

    def test_major_chord(self):
        """C major chord."""
        notes = [60, 64, 67]  # C-E-G
        samples = render_chord(notes, 2.0, WaveType.SAW, partials=8)
        samples = apply_envelope(samples, 0.01, 0.1, 0.8, 0.3)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "chord_c_major")

        print(f"\n  C Major chord: saved to {filepath}")

    def test_minor_chord(self):
        """A minor chord."""
        notes = [57, 60, 64]  # A-C-E
        samples = render_chord(notes, 2.0, WaveType.SAW, partials=8)
        samples = apply_envelope(samples, 0.01, 0.1, 0.8, 0.3)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "chord_a_minor")

        print(f"\n  A Minor chord: saved to {filepath}")

    def test_seventh_chord(self):
        """C major 7th chord."""
        notes = [60, 64, 67, 71]  # C-E-G-B
        samples = render_chord(notes, 2.0, WaveType.TRIANGLE, partials=6)
        samples = apply_envelope(samples, 0.1, 0.2, 0.7, 0.5)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "chord_cmaj7")

        print(f"\n  CMaj7 chord: saved to {filepath}")

    def test_power_chord(self):
        """Power chord (root + fifth + octave)."""
        notes = [36, 43, 48]  # C2-G2-C3
        samples = render_chord(notes, 2.0, WaveType.SAW, amplitude=0x5000, partials=10)
        samples = apply_envelope(samples, 0.005, 0.1, 0.9, 0.2)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "chord_power")

        print(f"\n  Power chord: saved to {filepath}")

    def test_stab_chord(self):
        """Short stab chord (dance music style)."""
        notes = [60, 64, 67, 72]  # C major with octave
        samples = render_chord(notes, 0.5, WaveType.SAW, amplitude=0x6000, partials=10)
        samples = apply_envelope(samples, 0.001, 0.15, 0.0, 0.0, gate_off_sec=0.001)

        L, R = mix_to_stereo(samples)
        filepath = save_audio(L, R, "chord_stab")

        print(f"\n  Stab chord: saved to {filepath}")


# =============================================================================
# ARPEGGIO / SEQUENCE TESTS
# =============================================================================

class TestArpeggios:
    """Test arpeggios and sequences."""

    def test_arpeggio_up(self):
        """Upward arpeggio."""
        notes = [60, 64, 67, 72, 67, 64]  # C-E-G-C'-G-E
        note_duration = 0.15
        all_samples = []

        for note in notes:
            samples = render_note(note, note_duration, WaveType.SAW,
                                  amplitude=0x5000, partials=10)
            samples = apply_envelope(samples, 0.002, 0.05, 0.5, 0.05)
            all_samples.extend(samples)

        # Repeat once
        all_samples = all_samples * 2

        L, R = mix_to_stereo(all_samples)
        filepath = save_audio(L, R, "arpeggio_up")

        print(f"\n  Upward arpeggio: saved to {filepath}")

    def test_arpeggio_down(self):
        """Downward arpeggio."""
        notes = [72, 67, 64, 60, 64, 67]  # C'-G-E-C-E-G
        note_duration = 0.12
        all_samples = []

        for note in notes:
            samples = render_note(note, note_duration, WaveType.PULSE,
                                  amplitude=0x5000, partials=8)
            samples = apply_envelope(samples, 0.001, 0.03, 0.4, 0.03)
            all_samples.extend(samples)

        # Repeat
        all_samples = all_samples * 3

        L, R = mix_to_stereo(all_samples)
        filepath = save_audio(L, R, "arpeggio_down")

        print(f"\n  Downward arpeggio: saved to {filepath}")

    def test_bass_sequence(self):
        """Classic synth bass sequence."""
        # Classic dance bassline pattern
        notes = [36, 36, 48, 36, 38, 36, 43, 41]  # C-C-C'-C-D-C-G-F
        durations = [0.2, 0.15, 0.1, 0.2, 0.15, 0.2, 0.1, 0.15]
        all_samples = []

        for note, dur in zip(notes, durations):
            samples = render_note(note, dur, WaveType.SAW,
                                  amplitude=0x6000, partials=6)
            samples = apply_envelope(samples, 0.002, 0.08, 0.6, 0.05)
            all_samples.extend(samples)

        # Repeat twice
        all_samples = all_samples * 2

        L, R = mix_to_stereo(all_samples)
        filepath = save_audio(L, R, "bass_sequence")

        print(f"\n  Bass sequence: saved to {filepath}")

    def test_lead_melody(self):
        """Simple lead melody."""
        # Happy melody
        notes = [72, 74, 76, 77, 79, 77, 76, 74, 72]
        durations = [0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.5]
        all_samples = []

        for note, dur in zip(notes, durations):
            samples = render_note(note, dur, WaveType.SAW,
                                  amplitude=0x5000, partials=12)
            samples = apply_envelope(samples, 0.01, 0.1, 0.7, 0.1)
            all_samples.extend(samples)

        L, R = mix_to_stereo(all_samples)
        filepath = save_audio(L, R, "lead_melody")

        print(f"\n  Lead melody: saved to {filepath}")


# =============================================================================
# STEREO TESTS
# =============================================================================

class TestStereo:
    """Test stereo effects."""

    def test_stereo_spread(self):
        """Stereo spread with different notes panned."""
        notes = [60, 64, 67, 72]
        pans = [-0.8, -0.3, 0.3, 0.8]
        n_samples = int(2.0 * SAMPLE_RATE)

        all_L = [0] * n_samples
        all_R = [0] * n_samples

        for note, pan in zip(notes, pans):
            samples = render_note(note, 2.0, WaveType.SAW,
                                  amplitude=0x4000, partials=8)
            samples = apply_envelope(samples, 0.1, 0.2, 0.8, 0.3)
            L, R = mix_to_stereo(samples, pan)

            for i in range(min(len(L), n_samples)):
                all_L[i] = sat16(all_L[i] + L[i])
                all_R[i] = sat16(all_R[i] + R[i])

        filepath = save_audio(all_L, all_R, "stereo_spread")

        print(f"\n  Stereo spread: saved to {filepath}")

    def test_ping_pong(self):
        """Ping-pong between left and right."""
        note_duration = 0.2
        notes = [60, 64, 67, 72, 67, 64] * 2

        all_L = []
        all_R = []

        for i, note in enumerate(notes):
            samples = render_note(note, note_duration, WaveType.SAW,
                                  amplitude=0x5000, partials=10)
            samples = apply_envelope(samples, 0.002, 0.05, 0.4, 0.05)

            # Alternate pan
            pan = -0.7 if i % 2 == 0 else 0.7
            L, R = mix_to_stereo(samples, pan)

            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "stereo_ping_pong")

        print(f"\n  Ping-pong stereo: saved to {filepath}")


# =============================================================================
# DEMO / SHOWCASE
# =============================================================================

class TestDemo:
    """Demonstration showcases."""

    def test_classic_synth_showcase(self):
        """Comprehensive showcase of classic synth sounds."""
        all_L = []
        all_R = []
        silence = [0] * int(SAMPLE_RATE * 0.3)

        # 1. Waveform parade
        print("\n  Building showcase...")
        for wave_type in [WaveType.SINE, WaveType.TRIANGLE, WaveType.PULSE, WaveType.SAW]:
            samples = render_note(60, 0.8, wave_type, amplitude=0x5000, partials=12)
            samples = apply_envelope(samples, 0.01, 0.1, 0.7, 0.15)
            L, R = mix_to_stereo(samples)
            all_L.extend(L + silence)
            all_R.extend(R + silence)

        # 2. Filter sweep
        n_blocks = int(2.0 * SAMPLE_RATE / BLOCK_SIZE)
        freq = note_to_freq(48)
        sweep_samples = []
        phase_state = None
        for block_idx in range(n_blocks):
            t = block_idx / n_blocks
            brightness = int(Q15_ONE * (0.1 + 0.9 * t))
            block, phase_state = synthesize_harmonic_audio(
                frequency=freq, amplitude=0x5000, wave_type=WaveType.SAW,
                partials_n=16, brightness_q15=brightness,
                block_size=BLOCK_SIZE, phase_state=phase_state
            )
            sweep_samples.extend(block)
        sweep_samples = apply_envelope(sweep_samples, 0.01, 0.0, 1.0, 0.3)
        L, R = mix_to_stereo(sweep_samples)
        all_L.extend(L + silence)
        all_R.extend(R + silence)

        # 3. Supersaw chord
        chord_notes = [48, 55, 60, 64, 67]  # C major with bass
        chord_samples = render_chord(chord_notes, 2.0, WaveType.SAW, partials=8)
        chord_samples = apply_envelope(chord_samples, 0.05, 0.2, 0.8, 0.5)
        L, R = mix_to_stereo(chord_samples)
        all_L.extend(L + silence)
        all_R.extend(R + silence)

        # 4. Arpeggio
        arp_notes = [60, 64, 67, 72, 76, 72, 67, 64] * 2
        arp_samples = []
        for note in arp_notes:
            s = render_note(note, 0.1, WaveType.SAW, amplitude=0x4000, partials=10)
            s = apply_envelope(s, 0.001, 0.03, 0.5, 0.03)
            arp_samples.extend(s)
        L, R = mix_to_stereo(arp_samples)
        all_L.extend(L)
        all_R.extend(R)

        filepath = save_audio(all_L, all_R, "classic_synth_showcase")

        print(f"  Classic Synth Showcase: {len(all_L)} samples ({len(all_L)/SAMPLE_RATE:.1f}s)")
        print(f"    Saved: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CLASSIC SYNTH AUDIO GENERATION")
    print("=" * 60)

    # Run showcase
    demo = TestDemo()
    demo.test_classic_synth_showcase()

    print("\nDone!")
