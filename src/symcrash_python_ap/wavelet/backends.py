"""
Synthesis Backend Adapters
==========================

Backend adapters that implement the SynthesisBackend protocol for different
synthesis engines. These are used by the UnifiedOutputStage for rendering.

Backends are "automatic transmissions" - they do pure block synthesis.
All discontinuity handling lives in the AntiClickOperator, not here.
"""

from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass

from ..primitives import Q15_ONE, Q15_HALF, Q15_SCALE, BLOCK_SIZE


# =============================================================================
# SYNTHESIS BACKEND PROTOCOL
# =============================================================================

class SynthesisBackend(Protocol):
    """
    Protocol defining what all synthesis backends must implement.

    Backends are responsible for pure block synthesis. They should NOT
    implement anti-click logic - that's the AntiClickOperator's job.
    """

    def render_block(self, frame: Any, block_size: int) -> List[int]:
        """
        Render one block of audio samples.

        Args:
            frame: Frame containing all parameters for this block
            block_size: Number of samples to render

        Returns:
            List of Q15 samples
        """
        ...

    def snapshot(self) -> Any:
        """
        Capture current state for dual-render.

        Returns an opaque state object that can be restored later.
        Must be deterministic and cheap.
        """
        ...

    def restore(self, state: Any) -> None:
        """
        Restore previously captured state.

        Args:
            state: State object from snapshot()
        """
        ...

    def apply_params(self, frame: Any) -> None:
        """
        Apply parameters from frame to internal state.

        This is called BEFORE render_block when parameters change.
        """
        ...


# =============================================================================
# HARMONIC SCAFFOLD BACKEND
# =============================================================================

class HarmonicScaffoldBackend:
    """
    Adapter wrapping oscillator.py's HarmonicScaffold.

    This is the additive synthesis backend using Baby Fourier.
    """

    def __init__(self):
        from .oscillator import (
            HarmonicScaffold, PhaseAccumulator,
            midi_to_phase_inc
        )

        self.HarmonicScaffold = HarmonicScaffold
        self.PhaseAccumulator = PhaseAccumulator
        self.midi_to_phase_inc = midi_to_phase_inc

        self.scaffold = HarmonicScaffold()
        self.phase = PhaseAccumulator()
        self.current_waveform = "saw"
        self.scaffold.set_saw_harmonics()

    def render_block(self, frame: Any, block_size: int) -> List[int]:
        """Render one block using additive harmonic synthesis."""
        samples = []
        gain_q15 = getattr(frame, 'gain_q15', Q15_HALF)
        for _ in range(block_size):
            phase, wrapped, frac = self.phase.advance()
            sample = self.scaffold.render_sample(phase)
            # Apply gain
            sample = (sample * gain_q15) >> 15
            samples.append(sample)
        return samples

    def snapshot(self) -> Dict:
        """Capture current state."""
        return {
            'phase': self.phase.phase,
            'phase_inc': self.phase.phase_inc,
            'cycle_count': self.phase.cycle_count,
            'harmonic_amps': list(self.scaffold.harmonic_amps_q15),
            'norm_gain': self.scaffold.norm_gain_q15,
            'waveform': self.current_waveform,
        }

    def restore(self, state: Dict) -> None:
        """Restore previously captured state."""
        self.phase.phase = state['phase']
        self.phase.phase_inc = state['phase_inc']
        self.phase.cycle_count = state['cycle_count']
        self.scaffold.harmonic_amps_q15 = list(state['harmonic_amps'])
        self.scaffold.norm_gain_q15 = state['norm_gain']
        self.current_waveform = state['waveform']

    def apply_params(self, frame: Any) -> None:
        """Apply parameters from frame."""
        note = getattr(frame, 'note', 60)
        cents = getattr(frame, 'cents', 0)
        waveform = getattr(frame, 'waveform', 'saw')

        # Set pitch
        self.phase.phase_inc = self.midi_to_phase_inc(note, cents)

        # Set waveform if changed
        if waveform != self.current_waveform:
            self.current_waveform = waveform
            if waveform == "saw":
                self.scaffold.set_saw_harmonics()
            elif waveform == "square":
                self.scaffold.set_square_harmonics()
            elif waveform == "triangle":
                self.scaffold.set_triangle_harmonics()
            else:  # sine
                self.scaffold.harmonic_amps_q15 = [Q15_ONE] + [0] * 7
                self.scaffold._compute_normalization()


# =============================================================================
# WAVELET BACKEND
# =============================================================================

class WaveletBackend:
    """
    Adapter wrapping WaveletEngine for IDWPT synthesis.

    This is the wavelet packet synthesis backend.
    """

    def __init__(self, block_size: int = BLOCK_SIZE):
        from .wavelet_engine import WaveletEngine
        from ..frames import VoiceFrame

        self.WaveletEngine = WaveletEngine
        self.VoiceFrame = VoiceFrame

        self.engine = WaveletEngine(block_size=block_size)
        self.block_size = block_size
        self._allocated_voices: set = set()
        self._last_vf: Optional[Any] = None

    def render_block(self, frame: Any, block_size: int) -> List[int]:
        """Render one block using IDWPT wavelet synthesis."""
        voice_id = getattr(frame, 'voice_id', 0)

        # Ensure voice is allocated in engine
        if voice_id not in self._allocated_voices:
            self.engine.voice_allocate(voice_id)
            self._allocated_voices.add(voice_id)

        # Build VoiceFrame if we got a different frame type
        if hasattr(frame, 'tree_a_id'):
            vf = self.VoiceFrame(
                voice_id=voice_id,
                tree_a_id=getattr(frame, 'tree_a_id', 1),
                tree_b_id=getattr(frame, 'tree_b_id', 1),
                tree_blend_q15=getattr(frame, 'tree_blend_q15', 0),
                gain_q15=getattr(frame, 'gain_q15', Q15_HALF),
                pan_q15=getattr(frame, 'pan_q15', 0),
                coeff_q15=getattr(frame, 'coeff_q15', [0] * 256),
            )
        else:
            vf = frame

        self._last_vf = vf

        # Render through engine with regularity operator
        return self.engine.render_voice(vf, apply_regularity=True)

    def snapshot(self) -> Dict:
        """Capture current state."""
        state = {
            'allocated_voices': set(self._allocated_voices),
            'prev_coeffs': {},
            'boundary_state': {},
            'voice_tree_state': {},
        }

        # Capture per-voice state
        for vid in self._allocated_voices:
            if vid in self.engine.prev_coeffs:
                state['prev_coeffs'][vid] = list(self.engine.prev_coeffs[vid])
            if vid in self.engine.boundary_state:
                state['boundary_state'][vid] = dict(self.engine.boundary_state[vid])
            if vid in self.engine.voice_tree_state:
                state['voice_tree_state'][vid] = dict(self.engine.voice_tree_state[vid])

        return state

    def restore(self, state: Dict) -> None:
        """Restore previously captured state."""
        self._allocated_voices = set(state.get('allocated_voices', set()))

        for vid, coeffs in state.get('prev_coeffs', {}).items():
            self.engine.prev_coeffs[vid] = list(coeffs)
        for vid, bs in state.get('boundary_state', {}).items():
            self.engine.boundary_state[vid] = dict(bs)
        for vid, ts in state.get('voice_tree_state', {}).items():
            self.engine.voice_tree_state[vid] = dict(ts)

    def apply_params(self, frame: Any) -> None:
        """Apply parameters from frame."""
        # Parameters are applied via VoiceFrame in render_block
        pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SynthesisBackend',
    'HarmonicScaffoldBackend',
    'WaveletBackend',
]
