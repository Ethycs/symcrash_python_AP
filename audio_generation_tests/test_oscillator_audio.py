"""
WPO v1 Oscillator Audio Generation Tests
=========================================

Comprehensive audio tests for the WPO v1 oscillator module.

Tests each feature:
1. Tree Archetypes (HARMONIC, BRIGHT, HOLLOW, NOISY, TRANSIENT)
2. Waveforms (saw, square, triangle, sine)
3. Two-stream blend (scaffold vs texture)
4. Analog drift (phase/amplitude)
5. BLEP bandlimiting
6. Multi-voice rendering
7. MIDI note range

Output: audio_generation_tests/audio_out/oscillator_tests/
"""

import pytest
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE
from audio_io import write_audio

from symcrash_python_ap.oscillator import (
    # Constants
    BLOCK_SIZE as OSC_BLOCK_SIZE,
    SAMPLE_RATE,
    Q15_ONE,
    Q15_HALF,
    Q15_SCALE,
    # Tree archetypes
    TreeArchetype,
    TREE_ARCHETYPES,
    get_archetype_config,
    # Components
    PhaseAccumulator,
    HarmonicScaffold,
    PacketTexture,
    BLEPState,
    AnalogDrift,
    blend_two_stream,
    # Full oscillator
    WPOState,
    OscillatorManager,
    midi_to_frequency,
)
from symcrash_python_ap.prng import VoicePRNG


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_SUBDIR = "oscillator_tests"
DURATION_SEC = 2.0
BLOCKS_PER_SEC = SAMPLE_RATE // BLOCK_SIZE


# =============================================================================
# AUDIO OUTPUT HELPER
# =============================================================================

def save_audio(samples_L: List[int], samples_R: List[int], name: str) -> str:
    """Save audio using audio_io module. Returns filepath."""
    metadata = write_audio(name, samples_L, samples_R, timestamp=True,
                           subdirectory=OUTPUT_SUBDIR, verbose=False)
    return metadata.filepath


def compute_rms(samples: List[int]) -> float:
    """Compute RMS of sample list."""
    if not samples:
        return 0.0
    return np.sqrt(sum(s * s for s in samples) / len(samples))


def compute_peak(samples: List[int]) -> int:
    """Compute peak absolute value."""
    if not samples:
        return 0
    return max(abs(s) for s in samples)


def samples_to_stereo(mono: List[int], pan: float = 0.0) -> tuple:
    """Convert mono to stereo with panning (-1 to +1)."""
    l_gain = 0.5 * (1 - pan)
    r_gain = 0.5 * (1 + pan)
    L = [int(s * l_gain) for s in mono]
    R = [int(s * r_gain) for s in mono]
    return L, R


# =============================================================================
# RENDER HELPERS
# =============================================================================

def render_oscillator(
    note: int = 60,
    waveform: str = "saw",
    archetype: TreeArchetype = TreeArchetype.HARMONIC,
    eta_q15: int = Q15_HALF,
    duration_sec: float = DURATION_SEC,
    use_prng: bool = False,
    voice_id: int = 0,
) -> List[int]:
    """
    Render audio from a single oscillator.

    Args:
        note: MIDI note number
        waveform: "saw", "square", "triangle", or "sine"
        archetype: Tree archetype
        eta_q15: Two-stream blend (0=scaffold, Q15_ONE=texture)
        duration_sec: Duration in seconds
        use_prng: Whether to use PRNG for drift/noise
        voice_id: Voice ID for PRNG seeding

    Returns:
        List of Q15 samples
    """
    n_blocks = int(duration_sec * BLOCKS_PER_SEC)

    osc = WPOState(voice_id=voice_id)
    osc.configure(
        note=note,
        cents=0,
        archetype=archetype,
        eta_q15=eta_q15,
        waveform=waveform,
    )

    prng = VoicePRNG.create(voice_id=voice_id, global_seed=12345) if use_prng else None

    samples = []
    for _ in range(n_blocks):
        block = osc.render_block(BLOCK_SIZE, prng)
        samples.extend(block)

    return samples


def render_multivoice(
    notes: List[int],
    waveform: str = "saw",
    archetype: TreeArchetype = TreeArchetype.HARMONIC,
    duration_sec: float = DURATION_SEC,
    pans: Optional[List[float]] = None,
) -> tuple:
    """
    Render multiple voices simultaneously.

    Args:
        notes: List of MIDI notes
        waveform: Waveform for all voices
        archetype: Archetype for all voices
        duration_sec: Duration
        pans: Pan positions for each voice (-1 to +1)

    Returns:
        (samples_L, samples_R) tuple
    """
    n_blocks = int(duration_sec * BLOCKS_PER_SEC)
    n_voices = len(notes)

    if pans is None:
        # Spread voices across stereo field
        pans = [((i / (n_voices - 1)) * 2 - 1) if n_voices > 1 else 0.0
                for i in range(n_voices)]

    mgr = OscillatorManager(max_voices=n_voices)

    for i, note in enumerate(notes):
        mgr.allocate(voice_id=i)
        mgr.configure_voice(i, note=note, waveform=waveform, archetype=archetype)

    prngs = {i: VoicePRNG.create(voice_id=i, global_seed=i * 1000) for i in range(n_voices)}

    all_L = []
    all_R = []

    for _ in range(n_blocks):
        outputs = mgr.render_all(block_size=BLOCK_SIZE, prngs=prngs)

        L_block = [0] * BLOCK_SIZE
        R_block = [0] * BLOCK_SIZE

        for vid, samples in outputs.items():
            pan = pans[vid]
            l_gain = 0.5 * (1 - pan)
            r_gain = 0.5 * (1 + pan)

            for i, s in enumerate(samples):
                L_block[i] += int(s * l_gain)
                R_block[i] += int(s * r_gain)

        # Soft clip
        L_block = [max(-Q15_SCALE, min(Q15_ONE, s)) for s in L_block]
        R_block = [max(-Q15_SCALE, min(Q15_ONE, s)) for s in R_block]

        all_L.extend(L_block)
        all_R.extend(R_block)

    return all_L, all_R


# =============================================================================
# TREE ARCHETYPE TESTS
# =============================================================================

class TestTreeArchetypes:
    """Test each tree archetype produces distinct audio character."""

    @pytest.mark.parametrize("archetype", list(TreeArchetype))
    def test_archetype_renders(self, archetype):
        """Each archetype should render without errors."""
        samples = render_oscillator(
            note=60,
            waveform="saw",
            archetype=archetype,
            duration_sec=1.0,
        )

        config = get_archetype_config(archetype)
        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, f"archetype_{archetype.name.lower()}")

        rms = compute_rms(samples)
        peak = compute_peak(samples)

        assert len(samples) > 0, "No samples generated"
        assert peak < Q15_SCALE, f"Clipping detected: peak={peak}"

        print(f"\n  {archetype.name}: RMS={rms:.0f}, peak={peak}, config={config.name}")
        print(f"    harmonic_lock={config.harmonic_lock}, noise={config.noise_inject_q15}")
        print(f"    Saved: {filepath}")

    def test_all_archetypes_comparison(self):
        """Render all archetypes side by side for comparison."""
        all_L = []
        all_R = []

        for archetype in TreeArchetype:
            samples = render_oscillator(
                note=60,
                waveform="saw",
                archetype=archetype,
                duration_sec=1.0,
            )
            L, R = samples_to_stereo(samples)
            all_L.extend(L)
            all_R.extend(R)
            # Add brief silence between
            all_L.extend([0] * (SAMPLE_RATE // 4))
            all_R.extend([0] * (SAMPLE_RATE // 4))

        filepath = save_audio(all_L, all_R, "all_archetypes_comparison")
        print(f"\n  All archetypes comparison saved: {filepath}")


# =============================================================================
# WAVEFORM TESTS
# =============================================================================

class TestWaveforms:
    """Test each waveform type."""

    @pytest.mark.parametrize("waveform", ["saw", "square", "triangle", "sine"])
    def test_waveform_renders(self, waveform):
        """Each waveform should render correctly."""
        samples = render_oscillator(
            note=60,
            waveform=waveform,
            archetype=TreeArchetype.HARMONIC,
            eta_q15=0,  # Pure scaffold to hear waveform clearly
            duration_sec=1.0,
        )

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, f"waveform_{waveform}")

        rms = compute_rms(samples)
        peak = compute_peak(samples)

        assert len(samples) > 0
        assert peak <= Q15_SCALE  # Allow hitting but not exceeding limit

        print(f"\n  {waveform}: RMS={rms:.0f}, peak={peak}")
        print(f"    Saved: {filepath}")

    def test_all_waveforms_comparison(self):
        """Render all waveforms in sequence."""
        all_L = []
        all_R = []

        for waveform in ["saw", "square", "triangle", "sine"]:
            samples = render_oscillator(
                note=60,
                waveform=waveform,
                eta_q15=0,
                duration_sec=1.0,
            )
            L, R = samples_to_stereo(samples)
            all_L.extend(L)
            all_R.extend(R)
            all_L.extend([0] * (SAMPLE_RATE // 4))
            all_R.extend([0] * (SAMPLE_RATE // 4))

        filepath = save_audio(all_L, all_R, "all_waveforms_comparison")
        print(f"\n  All waveforms comparison saved: {filepath}")


# =============================================================================
# TWO-STREAM BLEND TESTS
# =============================================================================

class TestTwoStreamBlend:
    """Test scaffold/texture blend parameter."""

    @pytest.mark.parametrize("eta_pct", [0, 25, 50, 75, 100])
    def test_blend_levels(self, eta_pct):
        """Test different blend levels."""
        eta_q15 = int(Q15_ONE * eta_pct / 100)

        samples = render_oscillator(
            note=60,
            waveform="saw",
            archetype=TreeArchetype.HARMONIC,
            eta_q15=eta_q15,
            duration_sec=1.0,
        )

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, f"blend_eta_{eta_pct}pct")

        rms = compute_rms(samples)
        print(f"\n  Blend {eta_pct}%: RMS={rms:.0f}")
        print(f"    Saved: {filepath}")

    def test_blend_sweep(self):
        """Sweep blend from scaffold to texture over time."""
        n_blocks = int(3.0 * BLOCKS_PER_SEC)

        osc = WPOState(voice_id=0)
        osc.configure(note=60, waveform="saw", archetype=TreeArchetype.BRIGHT)

        samples = []
        for block_idx in range(n_blocks):
            # Sweep eta from 0 to Q15_ONE
            t = block_idx / n_blocks
            osc.eta_q15 = int(t * Q15_ONE)

            block = osc.render_block(BLOCK_SIZE)
            samples.extend(block)

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, "blend_sweep")

        print(f"\n  Blend sweep (0->100%): {len(samples)} samples")
        print(f"    Saved: {filepath}")


# =============================================================================
# ANALOG DRIFT TESTS
# =============================================================================

class TestAnalogDrift:
    """Test analog drift/instability features."""

    def test_drift_with_prng(self):
        """Compare rendering with and without PRNG drift."""
        samples_no_drift = render_oscillator(
            note=60, waveform="saw", use_prng=False, duration_sec=2.0
        )
        samples_with_drift = render_oscillator(
            note=60, waveform="saw", use_prng=True, duration_sec=2.0
        )

        # They should differ due to drift
        L1, R1 = samples_to_stereo(samples_no_drift)
        L2, R2 = samples_to_stereo(samples_with_drift)

        filepath1 = save_audio(L1, R1, "drift_disabled")
        filepath2 = save_audio(L2, R2, "drift_enabled")

        rms1 = compute_rms(samples_no_drift)
        rms2 = compute_rms(samples_with_drift)

        print(f"\n  No drift: RMS={rms1:.0f}")
        print(f"  With drift: RMS={rms2:.0f}")
        print(f"    Saved: {filepath1}, {filepath2}")

    def test_noisy_archetype_with_drift(self):
        """NOISY archetype should benefit from PRNG drift."""
        samples = render_oscillator(
            note=60,
            waveform="saw",
            archetype=TreeArchetype.NOISY,
            use_prng=True,
            duration_sec=2.0,
        )

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, "noisy_with_drift")

        rms = compute_rms(samples)
        print(f"\n  NOISY + drift: RMS={rms:.0f}")
        print(f"    Saved: {filepath}")


# =============================================================================
# BLEP BANDLIMITING TESTS
# =============================================================================

class TestBLEP:
    """Test BLEP bandlimited edge correction."""

    def test_high_frequency_saw(self):
        """High frequency saw should still sound clean (BLEP prevents aliasing)."""
        # C7 = MIDI 96 = ~2093 Hz (high, near Nyquist/24 for 48kHz)
        samples = render_oscillator(
            note=96,
            waveform="saw",
            archetype=TreeArchetype.HARMONIC,
            eta_q15=0,
            duration_sec=1.0,
        )

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, "blep_high_freq_saw")

        rms = compute_rms(samples)
        peak = compute_peak(samples)

        print(f"\n  High freq saw (C7): RMS={rms:.0f}, peak={peak}")
        print(f"    Saved: {filepath}")

    def test_low_frequency_saw(self):
        """Low frequency saw for reference."""
        # A1 = MIDI 33 = ~55 Hz
        samples = render_oscillator(
            note=33,
            waveform="saw",
            eta_q15=0,
            duration_sec=2.0,
        )

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, "blep_low_freq_saw")

        print(f"\n  Low freq saw (A1): {len(samples)} samples")
        print(f"    Saved: {filepath}")


# =============================================================================
# MIDI NOTE RANGE TESTS
# =============================================================================

class TestMIDIRange:
    """Test oscillator across MIDI note range."""

    @pytest.mark.parametrize("note", [36, 48, 60, 72, 84, 96])
    def test_note_range(self, note):
        """Test various MIDI notes across the range."""
        freq = midi_to_frequency(note)

        samples = render_oscillator(
            note=note,
            waveform="saw",
            duration_sec=1.0,
        )

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, f"note_midi_{note}")

        rms = compute_rms(samples)
        print(f"\n  MIDI {note} ({freq:.1f} Hz): RMS={rms:.0f}")
        print(f"    Saved: {filepath}")

    def test_chromatic_scale(self):
        """Render a chromatic scale from C3 to C5."""
        all_L = []
        all_R = []

        for note in range(48, 73):  # C3 to C5
            samples = render_oscillator(
                note=note,
                waveform="saw",
                duration_sec=0.25,
            )
            L, R = samples_to_stereo(samples)
            all_L.extend(L)
            all_R.extend(R)

        filepath = save_audio(all_L, all_R, "chromatic_scale_c3_c5")
        print(f"\n  Chromatic scale C3-C5: {len(all_L)} samples")
        print(f"    Saved: {filepath}")


# =============================================================================
# MULTI-VOICE TESTS
# =============================================================================

class TestMultiVoice:
    """Test multi-voice rendering."""

    def test_unison_2_voices(self):
        """Two voices in unison with slight detuning."""
        mgr = OscillatorManager(max_voices=2)

        mgr.allocate(0)
        mgr.allocate(1)

        # Same note, slight detune via cents
        osc0 = mgr.get(0)
        osc1 = mgr.get(1)
        osc0.configure(note=60, waveform="saw")
        osc1.configure(note=60, waveform="saw")
        osc0.drift.detune_cents = -5
        osc1.drift.detune_cents = +5

        n_blocks = int(2.0 * BLOCKS_PER_SEC)
        prngs = {0: VoicePRNG.create(0, 111), 1: VoicePRNG.create(1, 222)}

        samples = []
        for _ in range(n_blocks):
            outputs = mgr.render_all(BLOCK_SIZE, prngs)
            block = [0] * BLOCK_SIZE
            for vid, out in outputs.items():
                for i, s in enumerate(out):
                    block[i] += s // 2
            samples.extend(block)

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, "unison_2_voices")

        print(f"\n  Unison 2 voices: {len(samples)} samples")
        print(f"    Saved: {filepath}")

    def test_chord_major(self):
        """Major chord (C-E-G)."""
        notes = [60, 64, 67]  # C4, E4, G4
        L, R = render_multivoice(notes, waveform="saw", duration_sec=2.0)

        filepath = save_audio(L, R, "chord_c_major")

        rms_L = compute_rms(L)
        print(f"\n  C Major chord: RMS={rms_L:.0f}")
        print(f"    Saved: {filepath}")

    def test_chord_minor(self):
        """Minor chord (A-C-E)."""
        notes = [57, 60, 64]  # A3, C4, E4
        L, R = render_multivoice(notes, waveform="triangle", duration_sec=2.0)

        filepath = save_audio(L, R, "chord_a_minor")

        rms_L = compute_rms(L)
        print(f"\n  A Minor chord: RMS={rms_L:.0f}")
        print(f"    Saved: {filepath}")

    def test_octave_stack(self):
        """Stack of octaves (A2, A3, A4, A5)."""
        notes = [45, 57, 69, 81]
        pans = [-0.6, -0.2, 0.2, 0.6]
        L, R = render_multivoice(notes, waveform="saw", duration_sec=2.0, pans=pans)

        filepath = save_audio(L, R, "octave_stack")

        print(f"\n  Octave stack A2-A5: {len(L)} samples")
        print(f"    Saved: {filepath}")


# =============================================================================
# FEATURE COMBINATIONS
# =============================================================================

class TestFeatureCombinations:
    """Test combinations of features."""

    @pytest.mark.parametrize("archetype", [TreeArchetype.HARMONIC, TreeArchetype.HOLLOW])
    @pytest.mark.parametrize("waveform", ["saw", "square"])
    def test_archetype_waveform_combo(self, archetype, waveform):
        """Test archetype + waveform combinations."""
        samples = render_oscillator(
            note=60,
            waveform=waveform,
            archetype=archetype,
            duration_sec=1.0,
        )

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, f"combo_{archetype.name.lower()}_{waveform}")

        rms = compute_rms(samples)
        print(f"\n  {archetype.name} + {waveform}: RMS={rms:.0f}")
        print(f"    Saved: {filepath}")

    def test_transient_attack(self):
        """TRANSIENT archetype for percussive attack."""
        n_blocks = int(1.0 * BLOCKS_PER_SEC)

        osc = WPOState(voice_id=0)
        osc.configure(note=60, waveform="saw", archetype=TreeArchetype.TRANSIENT)

        prng = VoicePRNG.create(0, 12345)
        samples = []

        for block_idx in range(n_blocks):
            # Decay envelope
            t = block_idx / n_blocks
            envelope = max(0, int(Q15_ONE * (1.0 - t * 2)))
            block = osc.render_block(BLOCK_SIZE, prng)
            # Apply envelope
            block = [int(s * envelope / Q15_ONE) for s in block]
            samples.extend(block)

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, "transient_attack")

        print(f"\n  Transient attack: {len(samples)} samples")
        print(f"    Saved: {filepath}")

    def test_bright_high_notes(self):
        """BRIGHT archetype on high notes."""
        samples = render_oscillator(
            note=84,  # C6
            waveform="saw",
            archetype=TreeArchetype.BRIGHT,
            use_prng=True,
            duration_sec=1.5,
        )

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, "bright_high_notes")

        print(f"\n  BRIGHT + high note (C6): {len(samples)} samples")
        print(f"    Saved: {filepath}")


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for oscillator stability."""

    def test_long_render(self):
        """Extended rendering for stability check."""
        samples = render_oscillator(
            note=60,
            waveform="saw",
            duration_sec=10.0,
        )

        # Check stability - RMS shouldn't drift
        quarter = len(samples) // 4
        rms_q1 = compute_rms(samples[:quarter])
        rms_q4 = compute_rms(samples[3*quarter:])

        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, "stress_long_render")

        assert rms_q4 > rms_q1 * 0.9, "Signal decayed unexpectedly"

        print(f"\n  Long render (10s): RMS_start={rms_q1:.0f}, RMS_end={rms_q4:.0f}")
        print(f"    Saved: {filepath}")

    def test_many_voices(self):
        """Test with many simultaneous voices."""
        notes = [48, 52, 55, 60, 64, 67, 72, 76]  # C major scale + octave
        pans = [-0.9, -0.6, -0.3, 0.0, 0.0, 0.3, 0.6, 0.9]

        L, R = render_multivoice(notes, waveform="saw", duration_sec=3.0, pans=pans)

        filepath = save_audio(L, R, "stress_8_voices")

        clip_count = sum(1 for s in L if abs(s) >= Q15_ONE - 100)
        clip_rate = clip_count / len(L)

        print(f"\n  8 voices: {len(L)} samples, clip_rate={clip_rate*100:.2f}%")
        print(f"    Saved: {filepath}")


# =============================================================================
# COMPREHENSIVE DEMO
# =============================================================================

class TestDemo:
    """Generate demonstration audio files."""

    def test_feature_showcase(self):
        """Create a showcase of all major features."""
        all_L = []
        all_R = []
        silence = [0] * (SAMPLE_RATE // 2)

        # 1. Waveforms
        for waveform in ["sine", "triangle", "square", "saw"]:
            samples = render_oscillator(note=60, waveform=waveform, eta_q15=0, duration_sec=1.0)
            L, R = samples_to_stereo(samples)
            all_L.extend(L)
            all_R.extend(R)
            all_L.extend(silence)
            all_R.extend(silence)

        # 2. Archetypes
        for archetype in TreeArchetype:
            samples = render_oscillator(note=60, waveform="saw", archetype=archetype, duration_sec=1.0)
            L, R = samples_to_stereo(samples)
            all_L.extend(L)
            all_R.extend(R)
            all_L.extend(silence)
            all_R.extend(silence)

        # 3. Chord
        L, R = render_multivoice([60, 64, 67, 72], waveform="saw", duration_sec=2.0)
        all_L.extend(L)
        all_R.extend(R)

        filepath = save_audio(all_L, all_R, "wpo_feature_showcase")

        print(f"\n  WPO Feature Showcase: {len(all_L)} samples ({len(all_L)/SAMPLE_RATE:.1f}s)")
        print(f"    Saved: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WPO v1 OSCILLATOR AUDIO GENERATION")
    print("=" * 60)

    # Run all tests manually
    print("\n--- Tree Archetypes ---")
    for archetype in TreeArchetype:
        samples = render_oscillator(note=60, waveform="saw", archetype=archetype, duration_sec=1.0)
        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, f"archetype_{archetype.name.lower()}")
        print(f"  {archetype.name}: saved to {filepath}")

    print("\n--- Waveforms ---")
    for waveform in ["saw", "square", "triangle", "sine"]:
        samples = render_oscillator(note=60, waveform=waveform, eta_q15=0, duration_sec=1.0)
        L, R = samples_to_stereo(samples)
        filepath = save_audio(L, R, f"waveform_{waveform}")
        print(f"  {waveform}: saved to {filepath}")

    print("\n--- Chords ---")
    L, R = render_multivoice([60, 64, 67], waveform="saw", duration_sec=2.0)
    filepath = save_audio(L, R, "chord_c_major")
    print(f"  C Major: saved to {filepath}")

    print("\nDone!")
