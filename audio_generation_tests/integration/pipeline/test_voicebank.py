"""
Voicebank Instrument Audio Generation Tests
============================================

Renders audio for each voicebank instrument (1001-1004) and saves to WAV files.
Tests the full Voice Engine -> Symbol Engine -> Wavelet Engine -> Safety Engine pipeline.

Uses Baby Fourier / Harmonic Target Injector for proper harmonic synthesis.

Instruments:
- 1001: Glitch Violin
- 1002: Shepard Random
- 1003: Turbulent Choir
- 1004: Static Organ

Output: audio_generation_tests/voicebank_output/
"""

import pytest
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # for test_config

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC
from symcrash_python_ap.audio_io import write_audio
from symcrash_python_ap.wavelet.harmonic_injector import (
    HarmonicTargetGenerator, WaveType, create_harmonic_coeffs as baby_fourier_coeffs,
    synthesize_harmonic_audio, note_to_freq, freq_to_leaf
)


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "voicebank_output"

Q15_ONE = 32767
Q15_HALF = 16384

INSTRUMENTS = {
    1001: {
        'name': 'glitch_violin',
        'description': 'Glitchy string sound with stochastic modulation',
    },
    1002: {
        'name': 'shepard_random',
        'description': 'Endless rising/falling shepard tones with randomness',
    },
    1003: {
        'name': 'turbulent_choir',
        'description': 'Choir-like pads with turbulent modulation',
    },
    1004: {
        'name': 'static_organ',
        'description': 'Stable organ tone for sustained sounds',
    },
}


# =============================================================================
# AUDIO OUTPUT HELPER
# =============================================================================

def save_audio(samples_L: List[int], samples_R: List[int], name: str) -> str:
    """Save audio using audio_io module. Returns filepath."""
    # Use default audio_out directory with timestamps and subdirectory for voicebank tests
    metadata = write_audio(name, samples_L, samples_R, timestamp=True,
                           subdirectory="voicebank_tests", verbose=False)
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


# =============================================================================
# GESTURE ENVELOPES
# =============================================================================

class GestureEnvelope:
    """Defines how gesture parameters evolve over time."""
    name: str = "base"

    def pressure(self, t: float) -> int:
        """Return pressure_q15 at time t (0-1)."""
        return Q15_HALF

    def chaos(self, t: float) -> int:
        """Return chaos_q15 at time t (0-1)."""
        return Q15_HALF // 4

    def speed(self, t: float) -> int:
        """Return speed_q15 at time t (0-1)."""
        return Q15_HALF // 2

    def tilt(self, t: float) -> int:
        """Return tilt_q15 at time t (0-1)."""
        return 0


class SustainEnvelope(GestureEnvelope):
    """Steady sustain at moderate level."""
    name = "sustain"

    def pressure(self, t: float) -> int:
        return Q15_HALF


class SwellEnvelope(GestureEnvelope):
    """Swell up then down."""
    name = "swell"

    def pressure(self, t: float) -> int:
        if t < 0.5:
            return int(Q15_HALF * 0.3 + Q15_HALF * 0.7 * (t * 2))
        else:
            return int(Q15_HALF * 0.3 + Q15_HALF * 0.7 * (1 - (t - 0.5) * 2))

    def chaos(self, t: float) -> int:
        return int(Q15_HALF * 0.4 * (0.5 + 0.5 * np.sin(t * 2 * np.pi * 2)))


class DecayEnvelope(GestureEnvelope):
    """Fast attack, slow decay."""
    name = "decay"

    def pressure(self, t: float) -> int:
        if t < 0.1:
            return int(Q15_ONE * t * 10)
        else:
            return int(Q15_ONE * (1 - (t - 0.1) / 0.9) ** 2)


class ExpressiveEnvelope(GestureEnvelope):
    """Complex modulation with all gesture parameters."""
    name = "expressive"

    def pressure(self, t: float) -> int:
        base = Q15_HALF * (0.4 + 0.6 * (0.5 + 0.5 * np.sin(t * 2 * np.pi * 0.5)))
        tremolo = Q15_HALF * 0.1 * np.sin(t * 2 * np.pi * 5)
        return int(base + tremolo)

    def chaos(self, t: float) -> int:
        return int(Q15_HALF * 0.5 * (0.5 + 0.5 * np.sin(t * 2 * np.pi * 0.7)))

    def speed(self, t: float) -> int:
        return int(Q15_HALF * (0.3 + 0.7 * t))

    def tilt(self, t: float) -> int:
        return int(Q15_HALF * 0.3 * np.sin(t * 2 * np.pi * 0.3))


ENVELOPES = {
    'sustain': SustainEnvelope(),
    'swell': SwellEnvelope(),
    'decay': DecayEnvelope(),
    'expressive': ExpressiveEnvelope(),
}


# =============================================================================
# RENDERER
# =============================================================================

@dataclass
class RenderResult:
    """Result of audio rendering."""
    samples_L: List[int]
    samples_R: List[int]
    duration_sec: float
    n_blocks: int
    rms_L: float
    rms_R: float
    peak_L: int
    peak_R: int
    regimes_seen: List[str]
    program_id: int
    envelope_name: str


def voice_frame_abi_to_vf(abi_frame):
    """Convert VoiceFrameABI from voice_engine to VoiceFrame for other engines."""
    from frames import VoiceFrame

    return VoiceFrame(
        voice_id=abi_frame.voice_id,
        tree_id=abi_frame.tree_a_id,
        tree_a_id=abi_frame.tree_a_id,
        tree_b_id=abi_frame.tree_b_id,
        tree_blend_q15=abi_frame.tree_blend_q15,
        gain_q15=abi_frame.gain_q15,
        pan_q15=abi_frame.pan_q15 if hasattr(abi_frame, 'pan_q15') else 0,
        is_active=abi_frame.is_active,
        coeff_q15=[0] * 256,
    )


def create_harmonic_coeffs(frequency: float = 220.0, amplitude: int = 0x5000,
                           wave_type: WaveType = WaveType.SAW,
                           partials_n: int = 12,
                           brightness_q15: int = Q15_ONE) -> List[int]:
    """
    Create harmonic coefficient pattern for a pitched note using Baby Fourier.

    Uses the Harmonic Target Injector to generate frequency-encoded coefficients
    that properly map harmonic partials to wavelet band positions.

    Args:
        frequency: Fundamental frequency in Hz
        amplitude: Base amplitude (Q15)
        wave_type: Wave type (SINE, SAW, PULSE, TRIANGLE)
        partials_n: Number of harmonic partials to generate
        brightness_q15: Brightness control for high-frequency rolloff

    Returns:
        256-coefficient vector with harmonics placed at correct frequency positions
    """
    return baby_fourier_coeffs(frequency, amplitude, wave_type, partials_n, brightness_q15)


def render_instrument(program_id: int,
                      duration_sec: float = 2.0,
                      envelope: GestureEnvelope = None) -> RenderResult:
    """
    Render an instrument through the full pipeline.

    Args:
        program_id: Instrument program ID (1001-1004)
        duration_sec: Duration in seconds
        envelope: Gesture envelope to use

    Returns:
        RenderResult with audio and metadata
    """
    from voice_engine import VoiceEngine, GestureBasis
    from symbol_engine import SymbolExecutor
    from wavelet_engine import WaveletEngine
    from safety_engine import SafetyEngine
    from frames import ParamFrame

    if envelope is None:
        envelope = SustainEnvelope()

    n_blocks = int(duration_sec * FS / BLOCK_SIZE)

    # Initialize engines
    voice_engine = VoiceEngine(max_voices=4)
    symbol_engine = SymbolExecutor(n_voices=4)
    wavelet_engine = WaveletEngine(block_size=BLOCK_SIZE)
    safety_engine = SafetyEngine(block_size=BLOCK_SIZE)

    # Allocate voice
    vid = voice_engine.allocate_voice(program_id=program_id)

    # Initial gesture
    gesture = GestureBasis(
        gate=True,
        pressure_q15=Q15_HALF,
        speed_q15=Q15_HALF // 2,
        chaos_q15=Q15_HALF // 4,
    )
    voice_engine.note_on(vid, gesture=gesture)
    safety_engine.voice_on(vid)

    # Harmonic configuration for this instrument
    # Different instruments get different base frequencies and wave types for variety
    instrument_configs = {
        1001: (220.0, WaveType.SAW, 12, Q15_ONE),       # Glitch Violin - A3, saw wave (rich harmonics)
        1002: (110.0, WaveType.TRIANGLE, 8, Q15_HALF),  # Shepard Random - A2, triangle (mellow drone)
        1003: (330.0, WaveType.PULSE, 10, Q15_ONE),     # Turbulent Choir - E4, pulse (hollow choir)
        1004: (196.0, WaveType.SAW, 16, Q15_ONE),       # Static Organ - G3, saw (organ harmonics)
    }
    base_freq, wave_type, partials, brightness = instrument_configs.get(
        program_id, (220.0, WaveType.SAW, 12, Q15_ONE)
    )

    # Phase state for continuous harmonic synthesis across blocks
    harmonic_phase_state = None

    params = ParamFrame(K_mode=1)

    all_L = []
    all_R = []
    regimes_seen = set()

    for block_idx in range(n_blocks):
        t = block_idx / n_blocks

        # Apply envelope
        gesture.pressure_q15 = envelope.pressure(t)
        gesture.chaos_q15 = envelope.chaos(t)
        gesture.speed_q15 = envelope.speed(t)
        gesture.tilt_q15 = envelope.tilt(t)

        voice_engine.update_gesture(vid, gesture)

        abi_frame = voice_engine.step_voice(vid)

        if abi_frame:
            if hasattr(abi_frame.regime, 'name'):
                regimes_seen.add(abi_frame.regime.name)

            # Direct harmonic synthesis using Baby Fourier (additive synthesis)
            # This generates audio at exact harmonic frequencies
            harmonic_amplitude = int(0x5000 * (gesture.pressure_q15 / Q15_ONE))
            harmonic_audio, harmonic_phase_state = synthesize_harmonic_audio(
                frequency=base_freq,
                amplitude=harmonic_amplitude,
                wave_type=wave_type,
                partials_n=partials,
                brightness_q15=brightness,
                block_size=BLOCK_SIZE,
                phase_state=harmonic_phase_state
            )

            # Apply voice gain
            gain_scale = abi_frame.gain_q15 / 32768.0
            audio_scaled = [int(s * gain_scale) for s in harmonic_audio]

            L_safe = safety_engine.apply_declick(audio_scaled, vid)
            R_safe = safety_engine.apply_declick(audio_scaled, vid)

            all_L.extend(L_safe)
            all_R.extend(R_safe)
        else:
            all_L.extend([0] * BLOCK_SIZE)
            all_R.extend([0] * BLOCK_SIZE)

    return RenderResult(
        samples_L=all_L,
        samples_R=all_R,
        duration_sec=duration_sec,
        n_blocks=n_blocks,
        rms_L=compute_rms(all_L),
        rms_R=compute_rms(all_R),
        peak_L=compute_peak(all_L),
        peak_R=compute_peak(all_R),
        regimes_seen=list(regimes_seen),
        program_id=program_id,
        envelope_name=envelope.name,
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def output_dir():
    """Create output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


# =============================================================================
# INDIVIDUAL INSTRUMENT TESTS
# =============================================================================

class TestGlitchViolin:
    """Tests for Program 1001: Glitch Violin."""

    PROGRAM_ID = 1001
    NAME = "glitch_violin"

    def test_sustain(self, output_dir):
        """Glitch Violin with sustained pressure."""
        result = render_instrument(self.PROGRAM_ID, duration_sec=3.0,
                                   envelope=ENVELOPES['sustain'])

        filepath = save_audio(result.samples_L, result.samples_R, f"{self.NAME}_sustain")

        assert result.rms_L > 50, "Output too quiet"
        assert result.peak_L < 35000, "Clipping detected"

        print(f"\n  {self.NAME} sustain: RMS={result.rms_L:.0f}, peak={result.peak_L}")
        print(f"    Regimes: {result.regimes_seen}")
        print(f"    Saved: {filepath}")

    def test_expressive(self, output_dir):
        """Glitch Violin with expressive modulation."""
        result = render_instrument(self.PROGRAM_ID, duration_sec=4.0,
                                   envelope=ENVELOPES['expressive'])

        filepath = save_audio(result.samples_L, result.samples_R, f"{self.NAME}_expressive")

        assert result.rms_L > 30, "Output too quiet"

        print(f"\n  {self.NAME} expressive: RMS={result.rms_L:.0f}")
        print(f"    Saved: {filepath}")


class TestShepardRandom:
    """Tests for Program 1002: Shepard Random."""

    PROGRAM_ID = 1002
    NAME = "shepard_random"

    def test_sustain(self, output_dir):
        """Shepard Random with sustained pressure."""
        result = render_instrument(self.PROGRAM_ID, duration_sec=5.0,
                                   envelope=ENVELOPES['sustain'])

        filepath = save_audio(result.samples_L, result.samples_R, f"{self.NAME}_sustain")

        assert result.rms_L > 50, "Output too quiet"

        print(f"\n  {self.NAME} sustain: RMS={result.rms_L:.0f}")
        print(f"    Saved: {filepath}")

    def test_swell(self, output_dir):
        """Shepard Random with swell envelope."""
        result = render_instrument(self.PROGRAM_ID, duration_sec=4.0,
                                   envelope=ENVELOPES['swell'])

        filepath = save_audio(result.samples_L, result.samples_R, f"{self.NAME}_swell")

        print(f"\n  {self.NAME} swell: RMS={result.rms_L:.0f}")
        print(f"    Saved: {filepath}")


class TestTurbulentChoir:
    """Tests for Program 1003: Turbulent Choir."""

    PROGRAM_ID = 1003
    NAME = "turbulent_choir"

    def test_sustain(self, output_dir):
        """Turbulent Choir with sustained pressure."""
        result = render_instrument(self.PROGRAM_ID, duration_sec=4.0,
                                   envelope=ENVELOPES['sustain'])

        filepath = save_audio(result.samples_L, result.samples_R, f"{self.NAME}_sustain")

        assert result.rms_L > 50, "Output too quiet"

        print(f"\n  {self.NAME} sustain: RMS={result.rms_L:.0f}")
        print(f"    Saved: {filepath}")

    def test_decay(self, output_dir):
        """Turbulent Choir with decay envelope."""
        result = render_instrument(self.PROGRAM_ID, duration_sec=3.0,
                                   envelope=ENVELOPES['decay'])

        filepath = save_audio(result.samples_L, result.samples_R, f"{self.NAME}_decay")

        print(f"\n  {self.NAME} decay: RMS={result.rms_L:.0f}")
        print(f"    Saved: {filepath}")


class TestStaticOrgan:
    """Tests for Program 1004: Static Organ."""

    PROGRAM_ID = 1004
    NAME = "static_organ"

    def test_sustain(self, output_dir):
        """Static Organ with sustained pressure."""
        result = render_instrument(self.PROGRAM_ID, duration_sec=3.0,
                                   envelope=ENVELOPES['sustain'])

        filepath = save_audio(result.samples_L, result.samples_R, f"{self.NAME}_sustain")

        assert result.rms_L > 50, "Output too quiet"

        print(f"\n  {self.NAME} sustain: RMS={result.rms_L:.0f}")
        print(f"    Regimes: {result.regimes_seen}")
        print(f"    Saved: {filepath}")

    def test_long_sustain(self, output_dir):
        """Static Organ extended sustain for stability test."""
        result = render_instrument(self.PROGRAM_ID, duration_sec=10.0,
                                   envelope=ENVELOPES['sustain'])

        filepath = save_audio(result.samples_L, result.samples_R, f"{self.NAME}_long_sustain")

        # Check for stability - RMS shouldn't drop to zero
        quarter = len(result.samples_L) // 4
        rms_q1 = compute_rms(result.samples_L[:quarter])
        rms_q4 = compute_rms(result.samples_L[3*quarter:])

        assert rms_q4 > rms_q1 * 0.5, "Signal decayed unexpectedly"

        print(f"\n  {self.NAME} long sustain: RMS_start={rms_q1:.0f}, RMS_end={rms_q4:.0f}")
        print(f"    Saved: {filepath}")


# =============================================================================
# PARAMETRIC TESTS
# =============================================================================

class TestAllInstrumentsAllEnvelopes:
    """Generate all combinations of instruments and envelopes."""

    @pytest.mark.parametrize("program_id", [1001, 1002, 1003, 1004])
    @pytest.mark.parametrize("envelope_name", ['sustain', 'swell', 'decay', 'expressive'])
    def test_instrument_envelope_combo(self, output_dir, program_id, envelope_name):
        """Test each instrument with each envelope type."""
        instrument = INSTRUMENTS[program_id]
        envelope = ENVELOPES[envelope_name]

        result = render_instrument(program_id, duration_sec=2.0, envelope=envelope)

        filepath = save_audio(result.samples_L, result.samples_R,
                              f"{instrument['name']}_{envelope_name}")

        # Basic sanity checks
        assert len(result.samples_L) > 0, "No samples generated"
        assert result.peak_L < 35000, f"Clipping: peak={result.peak_L}"

        print(f"\n  {instrument['name']} + {envelope_name}: RMS={result.rms_L:.0f}")


# =============================================================================
# MULTI-VOICE TESTS
# =============================================================================

class TestMultiVoice:
    """Tests with multiple simultaneous voices."""

    def test_all_instruments_chord(self, output_dir):
        """Play all 4 instruments simultaneously."""
        from voice_engine import VoiceEngine, GestureBasis
        from symbol_engine import SymbolExecutor
        from wavelet_engine import WaveletEngine
        from safety_engine import SafetyEngine
        from frames import ParamFrame

        duration_sec = 4.0
        n_blocks = int(duration_sec * FS / BLOCK_SIZE)

        voice_engine = VoiceEngine(max_voices=8)
        symbol_engine = SymbolExecutor(n_voices=8)
        wavelet_engine = WaveletEngine(block_size=BLOCK_SIZE)
        safety_engine = SafetyEngine(block_size=BLOCK_SIZE)
        params = ParamFrame(K_mode=1)

        # Allocate one voice per instrument
        voices = []
        programs = [1001, 1002, 1003, 1004]
        pans = [-0.6, -0.2, 0.2, 0.6]
        # Different frequencies for each instrument for a chord (A major)
        freqs = [220.0, 277.0, 330.0, 440.0]  # A3, C#4, E4, A4
        # Different wave types per voice for variety
        wave_types = [WaveType.SAW, WaveType.TRIANGLE, WaveType.PULSE, WaveType.SAW]

        # Track phase state per voice for continuous synthesis
        voice_phase_states = {}
        for prog, pan, freq, wave_type in zip(programs, pans, freqs, wave_types):
            vid = voice_engine.allocate_voice(program_id=prog)
            gesture = GestureBasis(
                gate=True,
                pressure_q15=Q15_HALF,
                pan_q15=int(pan * Q15_ONE),
            )
            voice_engine.note_on(vid, gesture=gesture)
            safety_engine.voice_on(vid)
            # Store config and phase state
            voice_phase_states[vid] = {
                'freq': freq,
                'wave_type': wave_type,
                'phase': None
            }
            voices.append((vid, gesture, pan))

        all_L = []
        all_R = []

        for block_idx in range(n_blocks):
            t = block_idx / n_blocks

            L_block = [0] * BLOCK_SIZE
            R_block = [0] * BLOCK_SIZE

            for vid, gesture, pan in voices:
                gesture.chaos_q15 = int(Q15_HALF * 0.3 * (0.5 + 0.5 * np.sin(t * 2 * np.pi * 0.5 + vid)))
                voice_engine.update_gesture(vid, gesture)

                abi_frame = voice_engine.step_voice(vid)

                if abi_frame:
                    # Direct harmonic synthesis
                    cfg = voice_phase_states[vid]
                    harmonic_audio, cfg['phase'] = synthesize_harmonic_audio(
                        frequency=cfg['freq'],
                        amplitude=0x4000,
                        wave_type=cfg['wave_type'],
                        partials_n=10,
                        block_size=BLOCK_SIZE,
                        phase_state=cfg['phase']
                    )

                    gain_scale = abi_frame.gain_q15 / 32768.0
                    audio_scaled = [int(s * gain_scale) for s in harmonic_audio]

                    l_gain = 0.5 * (1 - pan)
                    r_gain = 0.5 * (1 + pan)

                    for i, s in enumerate(audio_scaled):
                        L_block[i] += int(s * l_gain)
                        R_block[i] += int(s * r_gain)

            # Soft clip
            L_block = [max(-32767, min(32767, s)) for s in L_block]
            R_block = [max(-32767, min(32767, s)) for s in R_block]

            L_safe, R_safe = safety_engine.step_block(L_block, R_block, params)

            all_L.extend(L_safe)
            all_R.extend(R_safe)

        filepath = save_audio(all_L, all_R, "all_instruments_chord")

        rms_L = compute_rms(all_L)
        rms_R = compute_rms(all_R)

        assert rms_L > 50, "Chord too quiet"
        assert rms_R > 50, "Chord too quiet"

        print(f"\n  All instruments chord: L_RMS={rms_L:.0f}, R_RMS={rms_R:.0f}")
        print(f"    Saved: {filepath}")


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for instrument stability."""

    def test_rapid_gesture_changes(self, output_dir):
        """Test rapid gesture parameter changes."""
        from voice_engine import VoiceEngine, GestureBasis
        from symbol_engine import SymbolExecutor
        from wavelet_engine import WaveletEngine
        from safety_engine import SafetyEngine
        from frames import ParamFrame

        duration_sec = 3.0
        n_blocks = int(duration_sec * FS / BLOCK_SIZE)

        voice_engine = VoiceEngine(max_voices=4)
        symbol_engine = SymbolExecutor(n_voices=4)
        wavelet_engine = WaveletEngine(block_size=BLOCK_SIZE)
        safety_engine = SafetyEngine(block_size=BLOCK_SIZE)
        params = ParamFrame(K_mode=2)

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=Q15_HALF)
        voice_engine.note_on(vid, gesture=gesture)
        safety_engine.voice_on(vid)

        # Phase state for continuous synthesis
        harmonic_phase = None

        all_L = []
        all_R = []

        for block_idx in range(n_blocks):
            t = block_idx / n_blocks

            # Rapid oscillations
            gesture.pressure_q15 = int(Q15_HALF * (0.5 + 0.5 * np.sin(t * 2 * np.pi * 20)))
            gesture.chaos_q15 = int(Q15_HALF * (0.5 + 0.5 * np.sin(t * 2 * np.pi * 15)))
            gesture.speed_q15 = int(Q15_HALF * (0.5 + 0.5 * np.sin(t * 2 * np.pi * 10)))

            voice_engine.update_gesture(vid, gesture)
            abi_frame = voice_engine.step_voice(vid)

            if abi_frame:
                # Direct harmonic synthesis with amplitude modulation
                harmonic_amplitude = int(0x4000 * (gesture.pressure_q15 / Q15_ONE))
                harmonic_audio, harmonic_phase = synthesize_harmonic_audio(
                    frequency=220.0,
                    amplitude=harmonic_amplitude,
                    wave_type=WaveType.SAW,
                    partials_n=12,
                    block_size=BLOCK_SIZE,
                    phase_state=harmonic_phase
                )

                gain_scale = abi_frame.gain_q15 / 32768.0
                audio_scaled = [int(s * gain_scale) for s in harmonic_audio]

                L_safe = safety_engine.apply_declick(audio_scaled, vid)
                R_safe = safety_engine.apply_declick(audio_scaled, vid)

                all_L.extend(L_safe)
                all_R.extend(R_safe)
            else:
                all_L.extend([0] * BLOCK_SIZE)
                all_R.extend([0] * BLOCK_SIZE)

        filepath = save_audio(all_L, all_R, "rapid_gesture_changes")

        # Should not have excessive clipping
        clip_count = sum(1 for s in all_L if abs(s) >= 32700)
        clip_rate = clip_count / len(all_L)

        assert clip_rate < 0.05, f"Too much clipping: {clip_rate*100:.1f}%"

        print(f"\n  Rapid gesture changes: clip_rate={clip_rate*100:.2f}%")
        print(f"    Saved: {filepath}")

    def test_extreme_parameters(self, output_dir):
        """Test with extreme parameter values."""
        from voice_engine import VoiceEngine, GestureBasis
        from symbol_engine import SymbolExecutor
        from wavelet_engine import WaveletEngine
        from safety_engine import SafetyEngine
        from frames import ParamFrame

        duration_sec = 2.0
        n_blocks = int(duration_sec * FS / BLOCK_SIZE)

        voice_engine = VoiceEngine(max_voices=4)
        symbol_engine = SymbolExecutor(n_voices=4)
        wavelet_engine = WaveletEngine(block_size=BLOCK_SIZE)
        safety_engine = SafetyEngine(block_size=BLOCK_SIZE)
        params = ParamFrame(K_mode=3)

        vid = voice_engine.allocate_voice(program_id=1002)
        gesture = GestureBasis(
            gate=True,
            pressure_q15=Q15_ONE,
            chaos_q15=Q15_ONE,
            speed_q15=Q15_ONE,
            tilt_q15=Q15_ONE,
        )
        voice_engine.note_on(vid, gesture=gesture)
        safety_engine.voice_on(vid)

        # Phase state for continuous synthesis
        harmonic_phase = None

        all_L = []
        all_R = []

        for block_idx in range(n_blocks):
            abi_frame = voice_engine.step_voice(vid)

            if abi_frame:
                # Direct harmonic synthesis
                harmonic_audio, harmonic_phase = synthesize_harmonic_audio(
                    frequency=110.0,
                    amplitude=0x4000,
                    wave_type=WaveType.TRIANGLE,
                    partials_n=8,
                    block_size=BLOCK_SIZE,
                    phase_state=harmonic_phase
                )

                gain_scale = abi_frame.gain_q15 / 32768.0
                audio_scaled = [int(s * gain_scale) for s in harmonic_audio]

                L_safe, R_safe = safety_engine.step_block(audio_scaled, audio_scaled, params)

                all_L.extend(L_safe)
                all_R.extend(R_safe)
            else:
                all_L.extend([0] * BLOCK_SIZE)
                all_R.extend([0] * BLOCK_SIZE)

        filepath = save_audio(all_L, all_R, "extreme_parameters")

        # Should not overflow
        peak = compute_peak(all_L)
        assert peak < 35000, f"Overflow detected: peak={peak}"

        print(f"\n  Extreme parameters: peak={peak}")
        print(f"    Saved: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run as standalone script
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("VOICEBANK AUDIO GENERATION")
    print("=" * 60)

    for program_id, info in INSTRUMENTS.items():
        print(f"\nRendering {info['name']} ({program_id})...")

        for env_name, envelope in ENVELOPES.items():
            result = render_instrument(program_id, duration_sec=2.0, envelope=envelope)

            filepath = save_audio(result.samples_L, result.samples_R,
                                  f"{info['name']}_{env_name}")

            print(f"  {env_name}: RMS={result.rms_L:.0f}, peak={result.peak_L}")

    print(f"\nOutput written to: {OUTPUT_DIR}")
