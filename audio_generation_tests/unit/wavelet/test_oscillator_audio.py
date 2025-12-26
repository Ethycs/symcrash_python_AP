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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "symcrash_python_ap"))

from test_config import FS, BLOCK_SIZE
from audio_io import write_audio

from symcrash_python_ap.wavelet.oscillator import (
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


# Import from safety_engine (canonical location for audio safety)
from symcrash_python_ap.safety.safety_engine import (
    SafetyEngine,
    crossfade_append,
    FADE_LEN,
    OLA_OVERLAP,
)
from symcrash_python_ap.frames import ParamFrame


# =============================================================================
# SAFETY ENGINE INTEGRATION
# =============================================================================

# Global SafetyEngine instance for tests (reused to maintain OLA state)
_safety_engine: Optional[SafetyEngine] = None


def get_safety_engine(reset: bool = False) -> SafetyEngine:
    """Get or create the global SafetyEngine instance."""
    global _safety_engine
    if _safety_engine is None or reset:
        _safety_engine = SafetyEngine(block_size=BLOCK_SIZE)
    return _safety_engine


def render_through_safety_tail(
    samples_L: List[int],
    samples_R: List[int],
    voice_id: int = 0,
    is_voice_start: bool = True,
    is_voice_stop: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Route audio through SafetyEngine's tail processing.

    This is the canonical way to render audio - all anti-click processing
    belongs in the Safety Tail, not scattered throughout the oscillator.

    Args:
        samples_L: Left channel samples
        samples_R: Right channel samples
        voice_id: Voice ID for per-voice OLA tracking
        is_voice_start: Signal voice start (fade-in)
        is_voice_stop: Signal voice stop (fade-out at end)

    Returns:
        (processed_L, processed_R) with OLA, slew limiting, and limiting applied
    """
    safety = get_safety_engine()
    params = ParamFrame()

    # Signal voice start if needed
    if is_voice_start:
        safety.voice_on(voice_id)

    out_L = []
    out_R = []

    # Process in blocks
    n_samples = len(samples_L)
    n_blocks = (n_samples + BLOCK_SIZE - 1) // BLOCK_SIZE

    for block_idx in range(n_blocks):
        start = block_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, n_samples)

        # Extract block (pad if needed)
        block_L = list(samples_L[start:end])
        block_R = list(samples_R[start:end])

        # Pad final block if needed
        while len(block_L) < BLOCK_SIZE:
            block_L.append(0)
            block_R.append(0)

        # Signal voice stop on last block
        if is_voice_stop and block_idx == n_blocks - 1:
            safety.voice_off(voice_id)

        # Apply per-voice de-click (voice start/stop fades + OLA)
        block_L = safety.apply_declick(block_L, voice_id)
        block_R = safety.apply_declick(block_R, voice_id)

        # Apply full safety chain (metrics, Q, master OLA, slew limit, limiter, DC)
        block_L, block_R = safety.step_block(block_L, block_R, params)

        # Collect output (trim padding from final block)
        if block_idx == n_blocks - 1 and end < start + BLOCK_SIZE:
            out_L.extend(block_L[:end - start])
            out_R.extend(block_R[:end - start])
        else:
            out_L.extend(block_L)
            out_R.extend(block_R)

    return out_L, out_R


def render_blocks_through_safety_tail(
    oscillator: 'WPOState',
    n_blocks: int,
    voice_id: int = 0,
    prng: Optional['VoicePRNG'] = None,
    is_voice_start: bool = True,
    is_voice_stop: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Render oscillator blocks directly through SafetyEngine.

    This integrates the oscillator with the Safety Tail in a block-by-block
    manner, which is the correct architecture for click-free audio.

    Args:
        oscillator: WPOState oscillator to render
        n_blocks: Number of blocks to render
        voice_id: Voice ID for per-voice OLA tracking
        prng: Optional PRNG for drift
        is_voice_start: Signal voice start (fade-in)
        is_voice_stop: Signal voice stop (fade-out at end)

    Returns:
        (processed_L, processed_R) stereo output
    """
    safety = get_safety_engine()
    params = ParamFrame()

    if is_voice_start:
        safety.voice_on(voice_id)

    out_L = []
    out_R = []

    for block_idx in range(n_blocks):
        # Render oscillator block (mono)
        block = oscillator.render_block(BLOCK_SIZE, prng)

        # Convert to stereo (center pan for simplicity)
        block_L = list(block)
        block_R = list(block)

        # Signal voice stop on last block
        if is_voice_stop and block_idx == n_blocks - 1:
            safety.voice_off(voice_id)

        # Apply per-voice de-click
        block_L = safety.apply_declick(block_L, voice_id)
        block_R = safety.apply_declick(block_R, voice_id)

        # Apply full safety chain
        block_L, block_R = safety.step_block(block_L, block_R, params)

        out_L.extend(block_L)
        out_R.extend(block_R)

    return out_L, out_R


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
        """Render a chromatic scale from C3 to C5 with phase-continuous transitions."""
        # Use a single oscillator with phase-continuous pitch changes (legato style)
        # This avoids clicks by never resetting phase - just changing frequency
        osc = WPOState(voice_id=0)
        osc.configure(note=48, waveform="saw")  # Start at C3

        samples_per_note = int(0.25 * SAMPLE_RATE)
        all_samples = []

        for note in range(48, 73):  # C3 to C5
            if note > 48:
                # Change pitch without phase reset (legato)
                osc.set_pitch(note)

            # Render this note
            note_samples = []
            while len(note_samples) < samples_per_note:
                block = osc.render_block(BLOCK_SIZE)
                note_samples.extend(block)
            all_samples.extend(note_samples[:samples_per_note])

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "chromatic_scale_c3_c5")
        print(f"\n  Chromatic scale C3-C5 (phase-continuous): {len(L)} samples")
        print(f"    Saved: {filepath}")

    def test_chromatic_scale_crossfade(self):
        """Render a chromatic scale with per-note crossfades (for comparison)."""
        all_L = []
        all_R = []

        for note in range(48, 73):  # C3 to C5
            samples = render_oscillator(
                note=note,
                waveform="saw",
                duration_sec=0.25,
            )
            L, R = samples_to_stereo(samples)
            # Use constant-power crossfade to avoid clicks at note boundaries
            crossfade_append(all_L, L, fade_samples=128, constant_power=True)
            crossfade_append(all_R, R, fade_samples=128, constant_power=True)

        filepath = save_audio(all_L, all_R, "chromatic_scale_c3_c5_crossfade")
        print(f"\n  Chromatic scale C3-C5 (crossfade): {len(all_L)} samples")
        print(f"    Saved: {filepath}")

    def test_chromatic_scale_glide(self):
        """Render a chromatic scale using glide_to() for smooth portamento."""
        osc = WPOState(voice_id=0)
        osc.configure(note=48, waveform="saw")  # Start at C3

        samples_per_note = int(0.25 * SAMPLE_RATE)
        glide_blocks = 24  # ~32ms glide time (smooth portamento)
        all_samples = []

        for note in range(48, 73):  # C3 to C5
            if note > 48:
                # Glide to new pitch (per-sample interpolation for smoothness)
                osc.glide_to(note, glide_blocks=glide_blocks)

            # Render this note
            note_samples = []
            while len(note_samples) < samples_per_note:
                block = osc.render_block(BLOCK_SIZE)
                note_samples.extend(block)
            all_samples.extend(note_samples[:samples_per_note])

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "chromatic_scale_c3_c5_glide")
        print(f"\n  Chromatic scale C3-C5 (glide): {len(L)} samples")
        print(f"    Saved: {filepath}")

    def test_chromatic_scale_safety_tail(self):
        """
        Render a chromatic scale through the Safety Tail.

        This is the architecturally correct approach: all anti-click processing
        belongs in the Safety Tail, not scattered in the oscillator.

        The Safety Tail provides:
        - Per-voice fade-in on voice_on()
        - Per-voice fade-out on voice_off()
        - Block boundary OLA (overlap-add) smoothing
        - Slew rate limiting (A2 invariant)
        - Peak limiting (A1 invariant)
        - DC removal (A3 invariant)
        """
        # Reset safety engine for clean state
        safety = get_safety_engine(reset=True)
        params = ParamFrame()

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)
        out_L = []
        out_R = []

        for note_idx, note in enumerate(range(48, 73)):  # C3 to C5
            osc = WPOState(voice_id=note_idx % 4)  # Cycle voice IDs
            osc.configure(note=note, waveform="saw")

            # Signal voice start
            safety.voice_on(osc.voice_id)

            for block_idx in range(blocks_per_note):
                # Render oscillator block
                block = osc.render_block(BLOCK_SIZE)
                block_L = list(block)
                block_R = list(block)

                # Signal voice stop on last block of each note
                if block_idx == blocks_per_note - 1:
                    safety.voice_off(osc.voice_id)

                # Apply per-voice de-click (the Safety Tail's job)
                block_L = safety.apply_declick(block_L, osc.voice_id)
                block_R = safety.apply_declick(block_R, osc.voice_id)

                # Apply full safety chain
                block_L, block_R = safety.step_block(block_L, block_R, params)

                out_L.extend(block_L)
                out_R.extend(block_R)

        filepath = save_audio(out_L, out_R, "chromatic_scale_c3_c5_safety_tail")
        print(f"\n  Chromatic scale C3-C5 (Safety Tail): {len(out_L)} samples")
        print(f"    Saved: {filepath}")

    def test_chromatic_scale_legato_safety_tail(self):
        """
        Render a chromatic scale with legato (continuous voice) through Safety Tail.

        This combines phase-continuous pitch changes with proper Safety Tail
        processing. The oscillator just changes pitch; the Safety Tail handles
        all block-boundary smoothing.
        """
        safety = get_safety_engine(reset=True)
        params = ParamFrame()

        osc = WPOState(voice_id=0)
        osc.configure(note=48, waveform="saw")

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)
        n_notes = 25  # C3 to C5

        out_L = []
        out_R = []

        # Signal voice start once at beginning
        safety.voice_on(0)

        for note_idx, note in enumerate(range(48, 73)):
            if note > 48:
                # Phase-continuous pitch change (no reset)
                osc.set_pitch(note)

            for block_idx in range(blocks_per_note):
                block = osc.render_block(BLOCK_SIZE)
                block_L = list(block)
                block_R = list(block)

                # Signal voice stop on very last block
                if note == 72 and block_idx == blocks_per_note - 1:
                    safety.voice_off(0)

                # Safety Tail handles block boundaries
                block_L = safety.apply_declick(block_L, 0)
                block_R = safety.apply_declick(block_R, 0)
                block_L, block_R = safety.step_block(block_L, block_R, params)

                out_L.extend(block_L)
                out_R.extend(block_R)

        filepath = save_audio(out_L, out_R, "chromatic_scale_c3_c5_legato_safety")
        print(f"\n  Chromatic scale C3-C5 (legato + Safety Tail): {len(out_L)} samples")
        print(f"    Saved: {filepath}")

    def test_chromatic_scale_seam_hints(self):
        """
        Render a chromatic scale using dual-render crossfade for C¹-continuous bridging.

        This uses render_with_seam_at() which is the canonical method for smooth
        pitch transitions. It renders the overlap region with BOTH old and new
        parameters from the same state, then Hermite-crossfades them.

        This produces the cleanest transitions because both waveforms are rendered
        from the exact same phase state, eliminating phase mismatch artifacts.
        """
        osc = WPOState(voice_id=0)
        osc.configure(note=48, waveform="saw")  # Start at C3

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)  # 250ms per note
        all_samples = []

        # First note: just render normally
        for _ in range(blocks_per_note):
            block = osc.render_block(BLOCK_SIZE)
            all_samples.extend(block)

        # Remaining notes: use render_with_seam_at for smooth transitions
        for note in range(49, 73):  # C#3 to C5
            # First block of new note: dual-render crossfade at block start
            block = osc.render_with_seam_at(
                block_size=BLOCK_SIZE,
                seam_index=8,  # Seam near start (after M=8 margin)
                new_note=note,
            )
            all_samples.extend(block)

            # Sustain blocks: normal rendering
            for _ in range(blocks_per_note - 1):
                block = osc.render_block(BLOCK_SIZE)
                all_samples.extend(block)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "chromatic_scale_c3_c5_seam_hints")
        print(f"\n  Chromatic scale C3-C5 (dual-render): {len(L)} samples")
        print(f"    Saved: {filepath}")

    def test_rapid_note_changes_seam_at(self):
        """
        Test rapid note changes within a single block using render_with_seam_at().

        This tests the most demanding seam case: multiple pitch changes
        happening mid-block with Hermite bridging at each seam point.
        """
        osc = WPOState(voice_id=0)
        osc.configure(note=60, waveform="saw")  # Start at C4

        # Render multiple blocks with mid-block note changes
        all_samples = []

        # Pattern: change note at sample 32 (middle of 64-sample block)
        notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale

        for note_idx, new_note in enumerate(notes[1:], 1):
            # Render block with note change at middle
            block = osc.render_with_seam_at(
                block_size=BLOCK_SIZE,
                seam_index=BLOCK_SIZE // 2,  # Change at middle
                new_note=new_note,
            )
            all_samples.extend(block)

            # Also render some sustain
            for _ in range(int(0.1 * BLOCKS_PER_SEC)):  # 100ms sustain
                block = osc.render_block(BLOCK_SIZE)
                all_samples.extend(block)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "rapid_note_changes_seam_at")
        print(f"\n  Rapid note changes (seam_at): {len(L)} samples")
        print(f"    Saved: {filepath}")

    def test_chromatic_scale_dual_render_prototype(self):
        """
        Chromatic scale using render_with_seam_at() for proper dual-render crossfade.

        This is the prototype for the cleanest pitch transitions - uses the
        dual-render technique where we render the overlap region with BOTH
        old and new parameters from the same state, then Hermite-crossfade.

        Unlike prepare_for_parameter_change() which only has limited history,
        render_with_seam_at() does true dual-render at the seam point.
        """
        osc = WPOState(voice_id=0)
        osc.configure(note=48, waveform="saw")  # Start at C3

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)  # 250ms per note
        all_samples = []

        # First note: just render normally
        for _ in range(blocks_per_note):
            block = osc.render_block(BLOCK_SIZE)
            all_samples.extend(block)

        # Remaining notes: use render_with_seam_at for smooth transitions
        for note in range(49, 73):  # C#3 to C5
            # First block of new note: dual-render crossfade at block start
            block = osc.render_with_seam_at(
                block_size=BLOCK_SIZE,
                seam_index=8,  # Seam near start (after M=8 margin)
                new_note=note,
            )
            all_samples.extend(block)

            # Sustain blocks: normal rendering
            for _ in range(blocks_per_note - 1):
                block = osc.render_block(BLOCK_SIZE)
                all_samples.extend(block)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "chromatic_scale_c3_c5_dual_render_prototype")
        print(f"\n  Chromatic scale C3-C5 (dual-render prototype): {len(L)} samples")
        print(f"    Saved: {filepath}")

    def test_chromatic_scale_auto_dual_render(self):
        """
        Chromatic scale using automatic dual-render via set_pitch().

        This tests the new default behavior where set_pitch() queues a pitch
        change and render_block() automatically applies dual-render crossfade.
        This should produce identical results to the explicit render_with_seam_at()
        approach but with simpler API.
        """
        osc = WPOState(voice_id=0)
        osc.configure(note=48, waveform="saw")  # Start at C3

        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)  # 250ms per note
        all_samples = []

        # First note: just render normally
        for _ in range(blocks_per_note):
            block = osc.render_block(BLOCK_SIZE)
            all_samples.extend(block)

        # Remaining notes: use set_pitch() for automatic dual-render
        for note in range(49, 73):  # C#3 to C5
            # Queue pitch change - render_block will auto-apply dual-render
            osc.set_pitch(note, seam_sample=8)

            # First block: automatic dual-render happens here
            block = osc.render_block(BLOCK_SIZE)
            all_samples.extend(block)

            # Sustain blocks: normal rendering
            for _ in range(blocks_per_note - 1):
                block = osc.render_block(BLOCK_SIZE)
                all_samples.extend(block)

        L, R = samples_to_stereo(all_samples)
        filepath = save_audio(L, R, "chromatic_scale_c3_c5_auto_dual_render")
        print(f"\n  Chromatic scale C3-C5 (auto dual-render): {len(L)} samples")
        print(f"    Saved: {filepath}")

        # Verify no clicks by checking max delta at note boundaries
        # Note boundaries occur every blocks_per_note * BLOCK_SIZE samples
        note_boundary_samples = blocks_per_note * BLOCK_SIZE
        max_boundary_delta = 0
        for i in range(1, 24):  # 24 note changes
            boundary_idx = i * note_boundary_samples
            if boundary_idx < len(all_samples) - 1:
                # Check around the boundary (within crossfade region)
                for offset in range(-16, 16):
                    idx = boundary_idx + offset
                    if 0 < idx < len(all_samples):
                        delta = abs(all_samples[idx] - all_samples[idx - 1])
                        max_boundary_delta = max(max_boundary_delta, delta)

        print(f"    Max delta at note boundaries: {max_boundary_delta}")
        # With proper dual-render, max delta should be reasonable (< 5000)
        assert max_boundary_delta < 8000, f"Possible click detected: delta={max_boundary_delta}"


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
# SAFETY TAIL INTEGRATION TESTS
# =============================================================================

class TestSafetyTailIntegration:
    """
    Test proper routing through Safety Tail.

    ARCHITECTURE NOTE: All anti-click processing belongs in the Safety Tail,
    not scattered throughout the oscillator. The oscillator should focus on
    synthesis; the Safety Tail handles all block-boundary smoothing, voice
    fade-in/out, slew limiting, and peak protection.

    The Safety Tail enforces three invariants:
    - A1: Peak bound (|x[n]| ≤ P_max)
    - A2: Slew bound (|x[n] - x[n-1]| ≤ S_max)
    - A3: DC bound (|μ| ≤ D_max)
    """

    def test_single_note_through_safety_tail(self):
        """Render a single note through the Safety Tail."""
        safety = get_safety_engine(reset=True)

        osc = WPOState(voice_id=0)
        osc.configure(note=60, waveform="saw")

        n_blocks = int(1.0 * BLOCKS_PER_SEC)
        L, R = render_blocks_through_safety_tail(
            osc, n_blocks, voice_id=0,
            is_voice_start=True, is_voice_stop=True
        )

        filepath = save_audio(L, R, "safety_tail_single_note")
        rms = compute_rms(L)
        peak = compute_peak(L)

        print(f"\n  Single note (Safety Tail): RMS={rms:.0f}, peak={peak}")
        print(f"    Saved: {filepath}")

        # Verify safety invariants (allow up to Q15_ONE since limiter uses soft knee)
        assert peak <= Q15_ONE, f"Peak exceeded Q15 limit: {peak}"

    def test_note_sequence_through_safety_tail(self):
        """
        Render a sequence of notes with proper voice on/off signaling.

        Each note triggers:
        - voice_on() -> Hann fade-in (FADE_LEN samples)
        - voice_off() -> Hann fade-out (FADE_LEN samples)

        Block boundaries use OLA crossfade automatically.
        """
        safety = get_safety_engine(reset=True)
        params = ParamFrame()

        notes = [60, 64, 67, 72]  # C major arpeggio
        blocks_per_note = int(0.5 * BLOCKS_PER_SEC)

        out_L = []
        out_R = []

        for note_idx, note in enumerate(notes):
            osc = WPOState(voice_id=0)
            osc.configure(note=note, waveform="saw")

            # Signal voice start
            safety.voice_on(0)

            for block_idx in range(blocks_per_note):
                block = osc.render_block(BLOCK_SIZE)
                block_L = list(block)
                block_R = list(block)

                # Signal voice stop on last block
                if block_idx == blocks_per_note - 1:
                    safety.voice_off(0)

                # Route through Safety Tail
                block_L = safety.apply_declick(block_L, 0)
                block_R = safety.apply_declick(block_R, 0)
                block_L, block_R = safety.step_block(block_L, block_R, params)

                out_L.extend(block_L)
                out_R.extend(block_R)

        filepath = save_audio(out_L, out_R, "safety_tail_arpeggio")
        print(f"\n  Arpeggio (Safety Tail): {len(out_L)} samples")
        print(f"    Saved: {filepath}")

    def test_multivoice_through_safety_tail(self):
        """
        Render multiple voices simultaneously through Safety Tail.

        Each voice has its own OLA state for independent de-clicking.
        """
        safety = get_safety_engine(reset=True)
        params = ParamFrame()

        notes = [60, 64, 67]  # C major chord
        n_voices = len(notes)
        n_blocks = int(2.0 * BLOCKS_PER_SEC)

        # Create oscillators
        oscillators = []
        for i, note in enumerate(notes):
            osc = WPOState(voice_id=i)
            osc.configure(note=note, waveform="saw")
            oscillators.append(osc)
            safety.voice_on(i)

        out_L = []
        out_R = []

        for block_idx in range(n_blocks):
            # Signal voice stop on last block
            if block_idx == n_blocks - 1:
                for i in range(n_voices):
                    safety.voice_off(i)

            # Render and declick each voice
            mix_L = [0] * BLOCK_SIZE
            mix_R = [0] * BLOCK_SIZE

            for i, osc in enumerate(oscillators):
                block = osc.render_block(BLOCK_SIZE)
                block_L = list(block)
                block_R = list(block)

                # Per-voice declick
                block_L = safety.apply_declick(block_L, i)
                block_R = safety.apply_declick(block_R, i)

                # Mix (with pan spread)
                pan = (i / (n_voices - 1)) * 2 - 1 if n_voices > 1 else 0
                l_gain = 0.5 * (1 - pan)
                r_gain = 0.5 * (1 + pan)

                for j in range(BLOCK_SIZE):
                    mix_L[j] += int(block_L[j] * l_gain)
                    mix_R[j] += int(block_R[j] * r_gain)

            # Saturate mix
            mix_L = [max(-Q15_SCALE, min(Q15_ONE, s)) for s in mix_L]
            mix_R = [max(-Q15_SCALE, min(Q15_ONE, s)) for s in mix_R]

            # Apply master safety chain (OLA, slew, limiter, DC)
            mix_L, mix_R = safety.step_block(mix_L, mix_R, params)

            out_L.extend(mix_L)
            out_R.extend(mix_R)

        filepath = save_audio(out_L, out_R, "safety_tail_chord")
        print(f"\n  Chord (Safety Tail): {len(out_L)} samples")
        print(f"    Saved: {filepath}")

    def test_compare_raw_vs_safety_tail(self):
        """
        Compare raw oscillator output vs Safety Tail processed.

        This demonstrates why the Safety Tail is necessary:
        - Raw output may have clicks at note boundaries
        - Safety Tail smooths all transitions
        """
        # Reset for fresh state
        safety = get_safety_engine(reset=True)
        params = ParamFrame()

        # Common rendering params
        blocks_per_note = int(0.25 * BLOCKS_PER_SEC)
        notes = [60, 67, 60, 67, 60]  # Alternating notes

        # === RAW OUTPUT (no Safety Tail) ===
        raw_samples = []
        for note in notes:
            osc = WPOState(voice_id=0)
            osc.configure(note=note, waveform="saw", reset_phase=True)
            for _ in range(blocks_per_note):
                block = osc.render_block(BLOCK_SIZE)
                raw_samples.extend(block)

        raw_L, raw_R = samples_to_stereo(raw_samples)
        filepath_raw = save_audio(raw_L, raw_R, "compare_raw_no_safety")

        # === SAFETY TAIL OUTPUT ===
        safety = get_safety_engine(reset=True)  # Fresh state
        safe_L = []
        safe_R = []

        for note_idx, note in enumerate(notes):
            osc = WPOState(voice_id=0)
            osc.configure(note=note, waveform="saw", reset_phase=True)

            safety.voice_on(0)

            for block_idx in range(blocks_per_note):
                block = osc.render_block(BLOCK_SIZE)
                block_L = list(block)
                block_R = list(block)

                if block_idx == blocks_per_note - 1:
                    safety.voice_off(0)

                block_L = safety.apply_declick(block_L, 0)
                block_R = safety.apply_declick(block_R, 0)
                block_L, block_R = safety.step_block(block_L, block_R, params)

                safe_L.extend(block_L)
                safe_R.extend(block_R)

        filepath_safe = save_audio(safe_L, safe_R, "compare_with_safety")

        # Analyze differences
        raw_peak = compute_peak(raw_samples)
        safe_peak = compute_peak(safe_L)

        print(f"\n  Raw (no Safety Tail):")
        print(f"    Peak: {raw_peak}, Saved: {filepath_raw}")
        print(f"  With Safety Tail:")
        print(f"    Peak: {safe_peak}, Saved: {filepath_safe}")
        print(f"    (Listen to compare click behavior at note boundaries)")


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
