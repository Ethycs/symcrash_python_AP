"""
UnifiedRenderer - Shared Render Hygiene for All Synthesis Backends
===================================================================

WSE-STD-0xx: Render Hygiene & Transitions

This module provides a unified rendering wrapper that applies consistent
anti-click hygiene (dual-render crossfade, microfade) regardless of which
synthesis backend is used:

- HarmonicScaffold (additive synthesis from oscillator.py)
- WaveletEngine/BabyFourier (IDWPT synthesis)
- Any future backend implementing SynthesisBackend protocol

Key Principles:
1. Backends are "automatic transmissions" - they do pure block synthesis
2. All discontinuity handling lives in the wrapper, not scattered in backends
3. One place to solve clicks = consistent behavior across all synthesis methods

Discontinuity Triggers (v0):
- Pitch change (note or cents step exceeds threshold)
- Tree/basis change (tree_id changes, tree_blend step exceeds threshold)
- Operator word jump (enable-mask toggles, permutation events)
- Voice lifecycle (allocate, gate rise/fall)

Glide vs Crossfade:
- Glide: continuous pitch path, no dual-render needed if backend is truly continuous
- Crossfade: used for discontinuous state changes (tree swap, big coeff jump)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol, Tuple, TypeVar
from copy import deepcopy
import math

# Q15 constants
Q15_ONE = 32767
Q15_HALF = 16384
Q15_SCALE = 32768

# Default block size
BLOCK_SIZE = 64

# Discontinuity thresholds
PITCH_STEP_THRESH_CENTS = 10  # > 10 cents triggers dual-render
TREE_BLEND_STEP_THRESH = 3277  # ~10% of Q15 range

# Crossfade parameters
CROSSFADE_SAMPLES = 32  # Dual-render crossfade width (samples) - larger for smoother transitions
MICROFADE_SAMPLES = 16  # Voice on/off fade (samples)


# =============================================================================
# SYNTHESIS BACKEND PROTOCOL
# =============================================================================

class SynthesisBackend(Protocol):
    """
    Protocol defining what all synthesis backends must implement.

    Backends are responsible for pure block synthesis. They should NOT
    implement anti-click logic - that's the UnifiedRenderer's job.
    """

    def render_block(self, frame: 'RenderFrame', block_size: int) -> List[int]:
        """
        Render one block of audio samples.

        Args:
            frame: RenderFrame containing all parameters for this block
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

    def apply_params(self, frame: 'RenderFrame') -> None:
        """
        Apply parameters from frame to internal state.

        This is called BEFORE render_block when parameters change.
        """
        ...


# =============================================================================
# RENDER FRAME - UNIFIED PARAMETER CONTAINER
# =============================================================================

@dataclass
class RenderFrame:
    """
    Unified parameter container for all synthesis backends.

    Contains all parameters needed to render one block, regardless
    of which backend is being used.
    """
    # Identity
    voice_id: int = 0

    # Pitch (common to all backends)
    note: int = 60  # MIDI note
    cents: int = 0  # Fine tuning

    # Amplitude
    gain_q15: int = Q15_HALF
    pan_q15: int = 0  # -32768 = full left, +32767 = full right

    # Tree/basis (for wavelet backend)
    tree_a_id: int = 1
    tree_b_id: int = 1
    tree_blend_q15: int = 0

    # Coefficients (for wavelet backend)
    coeff_q15: List[int] = field(default_factory=list)

    # Waveform (for additive backend)
    waveform: str = "saw"  # "saw", "square", "triangle", "sine"

    # Blend between scaffold/texture (for oscillator)
    eta_q15: int = Q15_HALF

    # Glide state
    glide_active: bool = False
    glide_target_note: int = 60
    glide_target_cents: int = 0
    glide_samples_remaining: int = 0


@dataclass
class DiscontinuityFlags:
    """Flags indicating what changed between frames."""
    pitch_changed: bool = False
    tree_changed: bool = False
    coeff_changed: bool = False
    operator_changed: bool = False
    voice_allocated: bool = False

    @property
    def any_discontinuity(self) -> bool:
        """True if any discontinuity flag is set."""
        return (self.pitch_changed or self.tree_changed or
                self.coeff_changed or self.operator_changed or
                self.voice_allocated)


# =============================================================================
# CROSSFADE UTILITIES
# =============================================================================

def generate_crossfade_window(length: int, kind: str = "hann") -> List[float]:
    """
    Generate a crossfade window.

    Args:
        length: Window length in samples
        kind: "hann" (constant-power) or "linear"

    Returns:
        List of fade-in weights [0.0 -> 1.0]
    """
    if kind == "linear":
        return [i / (length - 1) if length > 1 else 1.0 for i in range(length)]
    else:  # hann (constant-power)
        return [math.sin(math.pi * i / (2 * (length - 1))) ** 2
                if length > 1 else 1.0 for i in range(length)]


def crossfade_samples(
    old_samples: List[int],
    new_samples: List[int],
    fade_samples: int = CROSSFADE_SAMPLES,
) -> List[int]:
    """
    Crossfade between two sample buffers using Hermite-weighted blend.

    Uses constant-power fade for smooth transitions.

    Args:
        old_samples: Samples rendered with old parameters
        new_samples: Samples rendered with new parameters
        fade_samples: Number of samples for crossfade

    Returns:
        Crossfaded samples (same length as inputs)
    """
    if len(old_samples) != len(new_samples):
        raise ValueError("Sample buffers must be same length")

    result = list(old_samples)
    fade_samples = min(fade_samples, len(old_samples))

    # Generate fade window
    window = generate_crossfade_window(fade_samples, "hann")

    for i in range(fade_samples):
        fade_in = window[i]
        fade_out = 1.0 - fade_in

        blended = int(old_samples[i] * fade_out + new_samples[i] * fade_in)

        # Saturate
        if blended > Q15_ONE:
            blended = Q15_ONE
        elif blended < -Q15_SCALE:
            blended = -Q15_SCALE

        result[i] = blended

    # After crossfade region, use new samples
    for i in range(fade_samples, len(result)):
        result[i] = new_samples[i]

    return result


def apply_microfade(
    samples: List[int],
    fade_in: bool = False,
    fade_out: bool = False,
    fade_samples: int = MICROFADE_SAMPLES,
) -> List[int]:
    """
    Apply microfade for voice start/stop.

    Args:
        samples: Audio samples
        fade_in: Apply fade-in at start
        fade_out: Apply fade-out at end
        fade_samples: Fade length

    Returns:
        Samples with microfade applied
    """
    result = list(samples)
    fade_samples = min(fade_samples, len(samples) // 2)

    if fade_in:
        window = generate_crossfade_window(fade_samples, "hann")
        for i in range(fade_samples):
            result[i] = int(result[i] * window[i])

    if fade_out:
        window = generate_crossfade_window(fade_samples, "hann")
        for i in range(fade_samples):
            idx = len(result) - fade_samples + i
            result[idx] = int(result[idx] * (1.0 - window[i]))

    return result


# =============================================================================
# DISCONTINUITY DETECTION
# =============================================================================

def detect_discontinuities(
    old_frame: Optional[RenderFrame],
    new_frame: RenderFrame,
) -> DiscontinuityFlags:
    """
    Detect what changed between frames that might cause discontinuity.

    Args:
        old_frame: Previous frame (None if first frame)
        new_frame: Current frame

    Returns:
        DiscontinuityFlags indicating what changed
    """
    flags = DiscontinuityFlags()

    if old_frame is None:
        flags.voice_allocated = True
        return flags

    # Pitch change detection
    old_cents_total = old_frame.note * 100 + old_frame.cents
    new_cents_total = new_frame.note * 100 + new_frame.cents
    if abs(new_cents_total - old_cents_total) > PITCH_STEP_THRESH_CENTS:
        # Only flag if not in glide mode
        if not new_frame.glide_active:
            flags.pitch_changed = True

    # Tree change detection
    if (old_frame.tree_a_id != new_frame.tree_a_id or
        old_frame.tree_b_id != new_frame.tree_b_id):
        flags.tree_changed = True

    # Tree blend step detection
    if abs(new_frame.tree_blend_q15 - old_frame.tree_blend_q15) > TREE_BLEND_STEP_THRESH:
        flags.tree_changed = True

    # Coefficient change detection (simplified - check if list changed)
    if old_frame.coeff_q15 != new_frame.coeff_q15:
        # Only flag major changes, not smooth transitions
        if len(old_frame.coeff_q15) > 0 and len(new_frame.coeff_q15) > 0:
            # Compute rough energy difference
            old_energy = sum(abs(c) for c in old_frame.coeff_q15[:32])
            new_energy = sum(abs(c) for c in new_frame.coeff_q15[:32])
            if abs(new_energy - old_energy) > old_energy * 0.3:  # >30% change
                flags.coeff_changed = True

    # Waveform change
    if old_frame.waveform != new_frame.waveform:
        flags.operator_changed = True

    return flags


# =============================================================================
# UNIFIED RENDERER
# =============================================================================

@dataclass
class VoiceState:
    """Per-voice state tracking for UnifiedRenderer."""
    voice_id: int
    backend_state: Any = None  # Snapshot of backend state
    last_frame: Optional[RenderFrame] = None
    is_active: bool = False
    pending_fade_in: bool = False
    pending_fade_out: bool = False


class UnifiedRenderer:
    """
    Unified rendering wrapper with shared anti-click hygiene.

    Wraps any SynthesisBackend and applies consistent dual-render
    crossfade and microfade handling regardless of backend.

    Usage:
        backend = HarmonicScaffoldBackend()  # or WaveletBackend()
        renderer = UnifiedRenderer(backend)

        renderer.voice_allocate(0)
        frame = RenderFrame(voice_id=0, note=60)
        samples = renderer.render_voice(frame)

        # Change pitch - dual-render happens automatically
        frame.note = 62
        samples = renderer.render_voice(frame)
    """

    def __init__(
        self,
        backend: SynthesisBackend,
        block_size: int = BLOCK_SIZE,
        crossfade_samples: int = CROSSFADE_SAMPLES,
        microfade_samples: int = MICROFADE_SAMPLES,
    ):
        self.backend = backend
        self.block_size = block_size
        self.crossfade_samples = crossfade_samples
        self.microfade_samples = microfade_samples

        # Per-voice state
        self.voices: Dict[int, VoiceState] = {}

    def voice_allocate(self, voice_id: int) -> None:
        """
        Allocate a voice slot.

        Clears any existing state and prepares for rendering.
        First render will apply microfade-in.
        """
        self.voices[voice_id] = VoiceState(
            voice_id=voice_id,
            is_active=True,
            pending_fade_in=True,
        )

    def voice_deallocate(self, voice_id: int) -> None:
        """Deallocate a voice slot."""
        if voice_id in self.voices:
            del self.voices[voice_id]

    def voice_off(self, voice_id: int) -> None:
        """
        Signal voice stop - next render will apply fade-out.
        """
        if voice_id in self.voices:
            self.voices[voice_id].pending_fade_out = True

    def render_voice(self, frame: RenderFrame) -> List[int]:
        """
        Render one block for a voice with automatic anti-click handling.

        Detects discontinuities and applies dual-render crossfade as needed.
        Also handles voice start/stop microfades.

        Args:
            frame: RenderFrame with all parameters

        Returns:
            List of Q15 samples
        """
        voice_id = frame.voice_id

        # Ensure voice is allocated
        if voice_id not in self.voices:
            self.voice_allocate(voice_id)

        voice = self.voices[voice_id]

        # Detect discontinuities
        flags = detect_discontinuities(voice.last_frame, frame)

        # Decide rendering strategy
        if flags.any_discontinuity and voice.last_frame is not None:
            # Dual-render crossfade
            samples = self._render_with_dual_crossfade(voice, frame, flags)
        else:
            # Normal render
            self.backend.apply_params(frame)
            samples = self.backend.render_block(frame, self.block_size)

        # Apply microfades
        if voice.pending_fade_in:
            samples = apply_microfade(samples, fade_in=True,
                                      fade_samples=self.microfade_samples)
            voice.pending_fade_in = False

        if voice.pending_fade_out:
            samples = apply_microfade(samples, fade_out=True,
                                      fade_samples=self.microfade_samples)
            voice.pending_fade_out = False
            voice.is_active = False

        # Update voice state
        voice.last_frame = deepcopy(frame)
        voice.backend_state = self.backend.snapshot()

        return samples

    def _render_with_dual_crossfade(
        self,
        voice: VoiceState,
        new_frame: RenderFrame,
        flags: DiscontinuityFlags,
    ) -> List[int]:
        """
        Render with dual-render crossfade for discontinuous transitions.

        Key insight: Both renders must start from the SAME state (phase, etc.)
        to ensure they diverge smoothly and can be crossfaded.

        1. Snapshot state at START of block
        2. Apply OLD params, render block with old params
        3. Restore to START of block state
        4. Apply NEW params, render block with new params
        5. Crossfade (both start from same phase, diverge with different freq)
        """
        old_frame = voice.last_frame

        # Snapshot state at START of this block (before any rendering)
        start_state = self.backend.snapshot()

        # === Render with OLD parameters ===
        if old_frame is not None:
            self.backend.apply_params(old_frame)
        old_samples = self.backend.render_block(old_frame or new_frame, self.block_size)

        # === Restore to START of block state ===
        self.backend.restore(start_state)

        # === Render with NEW parameters (from same starting state) ===
        self.backend.apply_params(new_frame)
        new_samples = self.backend.render_block(new_frame, self.block_size)

        # Crossfade - both renders started from same phase, just different frequency
        return crossfade_samples(old_samples, new_samples, self.crossfade_samples)

    def render_stereo(
        self,
        frame: RenderFrame,
    ) -> Tuple[List[int], List[int]]:
        """
        Render voice and apply pan to stereo.

        Returns (L, R) sample lists.
        """
        mono = self.render_voice(frame)

        # Simple pan law (linear for now)
        # pan_q15: -32768 = full left, 0 = center, +32767 = full right
        pan = frame.pan_q15

        # Convert to 0..1 range where 0=left, 0.5=center, 1=right
        pan_norm = (pan + Q15_SCALE) / (2 * Q15_SCALE)

        # Constant-power pan
        left_gain = math.cos(pan_norm * math.pi / 2)
        right_gain = math.sin(pan_norm * math.pi / 2)

        L = [int(s * left_gain) for s in mono]
        R = [int(s * right_gain) for s in mono]

        return L, R


# =============================================================================
# HARMONIC SCAFFOLD BACKEND ADAPTER
# =============================================================================

class HarmonicScaffoldBackend:
    """
    Adapter wrapping oscillator.py's HarmonicScaffold for UnifiedRenderer.

    This is the additive synthesis backend.
    """

    def __init__(self):
        # Import here to avoid circular dependency
        from ..wavelet.oscillator import (
            HarmonicScaffold, PhaseAccumulator,
            midi_to_phase_inc, Q15_ONE as OSC_Q15_ONE
        )

        self.HarmonicScaffold = HarmonicScaffold
        self.PhaseAccumulator = PhaseAccumulator
        self.midi_to_phase_inc = midi_to_phase_inc

        self.scaffold = HarmonicScaffold()
        self.phase = PhaseAccumulator()
        self.current_waveform = "saw"
        self.scaffold.set_saw_harmonics()

    def render_block(self, frame: RenderFrame, block_size: int) -> List[int]:
        """Render one block using additive harmonic synthesis."""
        samples = []
        for _ in range(block_size):
            phase, wrapped, frac = self.phase.advance()
            sample = self.scaffold.render_sample(phase)
            # Apply gain
            sample = (sample * frame.gain_q15) >> 15
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

    def apply_params(self, frame: RenderFrame) -> None:
        """Apply parameters from frame."""
        # Set pitch
        self.phase.phase_inc = self.midi_to_phase_inc(frame.note, frame.cents)

        # Set waveform if changed
        if frame.waveform != self.current_waveform:
            self.current_waveform = frame.waveform
            if frame.waveform == "saw":
                self.scaffold.set_saw_harmonics()
            elif frame.waveform == "square":
                self.scaffold.set_square_harmonics()
            elif frame.waveform == "triangle":
                self.scaffold.set_triangle_harmonics()
            else:  # sine
                self.scaffold.harmonic_amps_q15 = [Q15_ONE] + [0] * 7
                self.scaffold._compute_normalization()


# =============================================================================
# WAVELET/BABY FOURIER BACKEND ADAPTER
# =============================================================================

class WaveletBackend:
    """
    Adapter wrapping WaveletEngine for UnifiedRenderer.

    This is the IDWPT/Baby Fourier synthesis backend.
    """

    def __init__(self, block_size: int = BLOCK_SIZE):
        from ..wavelet.wavelet_engine import WaveletEngine
        from ..frames import VoiceFrame

        self.WaveletEngine = WaveletEngine
        self.VoiceFrame = VoiceFrame

        self.engine = WaveletEngine(block_size=block_size)
        self.block_size = block_size
        self._allocated_voices: set = set()
        self._last_vf: Optional[Any] = None

    def render_block(self, frame: RenderFrame, block_size: int) -> List[int]:
        """Render one block using IDWPT wavelet synthesis."""
        voice_id = frame.voice_id

        # Ensure voice is allocated in engine
        if voice_id not in self._allocated_voices:
            self.engine.voice_allocate(voice_id)
            self._allocated_voices.add(voice_id)

        # Build VoiceFrame
        vf = self.VoiceFrame(
            voice_id=voice_id,
            tree_a_id=frame.tree_a_id,
            tree_b_id=frame.tree_b_id,
            tree_blend_q15=frame.tree_blend_q15,
            gain_q15=frame.gain_q15,
            pan_q15=frame.pan_q15,
            coeff_q15=frame.coeff_q15 if frame.coeff_q15 else [0] * 256,
        )

        self._last_vf = vf

        # Render through engine with regularity operator
        return self.engine.render_voice(vf, apply_regularity=True)

    def snapshot(self) -> Dict:
        """Capture current state."""
        # WaveletEngine state is in various dicts keyed by voice_id
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
        # Restore allocated voices
        self._allocated_voices = set(state.get('allocated_voices', set()))

        # Restore per-voice state
        for vid, coeffs in state.get('prev_coeffs', {}).items():
            self.engine.prev_coeffs[vid] = list(coeffs)
        for vid, bs in state.get('boundary_state', {}).items():
            self.engine.boundary_state[vid] = dict(bs)
        for vid, ts in state.get('voice_tree_state', {}).items():
            self.engine.voice_tree_state[vid] = dict(ts)

    def apply_params(self, frame: RenderFrame) -> None:
        """Apply parameters from frame."""
        # Parameters are applied via VoiceFrame in render_block
        # Nothing to do here for wavelet backend
        pass


# =============================================================================
# STANDARD PIPELINE FACTORY
# =============================================================================

class StandardWaveletPipeline:
    """
    Standard pipeline combining UnifiedRenderer + WaveletBackend + SafetyEngine.

    This is the recommended way to use the wavelet synthesis engine.
    Provides automatic dual-render crossfade for pitch transitions
    plus safety limiting.

    Usage:
        pipeline = StandardWaveletPipeline()
        pipeline.voice_on(0)

        for note in notes:
            L, R = pipeline.render_note(voice_id=0, note=note, coeffs=coeffs)
            # L, R are stereo samples ready for output

        pipeline.voice_off(0)
    """

    def __init__(self, block_size: int = BLOCK_SIZE):
        from .safety_engine import SafetyEngine
        from ..frames import ParamFrame

        self.backend = WaveletBackend(block_size=block_size)
        self.renderer = UnifiedRenderer(self.backend, block_size=block_size)
        self.safety = SafetyEngine(block_size=block_size)
        self.params = ParamFrame()
        self.block_size = block_size

    def voice_on(self, voice_id: int) -> None:
        """Allocate and activate a voice."""
        self.renderer.voice_allocate(voice_id)
        self.safety.voice_on(voice_id)

    def voice_off(self, voice_id: int) -> None:
        """Signal voice release (applies fade-out)."""
        self.renderer.voice_off(voice_id)

    def voice_deallocate(self, voice_id: int) -> None:
        """Fully deallocate a voice slot."""
        self.renderer.voice_deallocate(voice_id)
        self.safety.voice_off(voice_id)

    def render_block(
        self,
        voice_id: int,
        note: int,
        coeffs: List[int],
        gain_q15: int = Q15_HALF,
        tree_a_id: int = 1,
        tree_b_id: int = 1,
        tree_blend_q15: int = 0,
    ) -> Tuple[List[int], List[int]]:
        """
        Render one block through the full pipeline.

        Args:
            voice_id: Voice slot
            note: MIDI note
            coeffs: Wavelet coefficients (256 values)
            gain_q15: Volume (Q15)
            tree_a_id: Tree A selection
            tree_b_id: Tree B selection
            tree_blend_q15: Blend between trees

        Returns:
            (L, R) stereo sample lists
        """
        frame = RenderFrame(
            voice_id=voice_id,
            note=note,
            gain_q15=gain_q15,
            coeff_q15=coeffs,
            tree_a_id=tree_a_id,
            tree_b_id=tree_b_id,
            tree_blend_q15=tree_blend_q15,
        )

        # Render through UnifiedRenderer (applies dual-render crossfade)
        samples = self.renderer.render_voice(frame)

        # Apply SafetyEngine declick
        samples = self.safety.apply_declick(samples, voice_id)

        # Convert to stereo
        L = list(samples)
        R = list(samples)

        # Apply SafetyEngine limiting
        L, R = self.safety.step_block(L, R, self.params)

        return L, R


class StandardHarmonicPipeline:
    """
    Standard pipeline combining UnifiedRenderer + HarmonicScaffoldBackend + SafetyEngine.

    This is the recommended way to use the additive synthesis engine.

    Usage:
        pipeline = StandardHarmonicPipeline()
        pipeline.voice_on(0)

        for note in notes:
            L, R = pipeline.render_note(voice_id=0, note=note, waveform="saw")
    """

    def __init__(self, block_size: int = BLOCK_SIZE):
        from .safety_engine import SafetyEngine
        from ..frames import ParamFrame

        self.backend = HarmonicScaffoldBackend()
        self.renderer = UnifiedRenderer(self.backend, block_size=block_size)
        self.safety = SafetyEngine(block_size=block_size)
        self.params = ParamFrame()
        self.block_size = block_size

    def voice_on(self, voice_id: int) -> None:
        """Allocate and activate a voice."""
        self.renderer.voice_allocate(voice_id)
        self.safety.voice_on(voice_id)

    def voice_off(self, voice_id: int) -> None:
        """Signal voice release (applies fade-out)."""
        self.renderer.voice_off(voice_id)

    def voice_deallocate(self, voice_id: int) -> None:
        """Fully deallocate a voice slot."""
        self.renderer.voice_deallocate(voice_id)
        self.safety.voice_off(voice_id)

    def render_block(
        self,
        voice_id: int,
        note: int,
        waveform: str = "saw",
        gain_q15: int = Q15_HALF,
        cents: int = 0,
    ) -> Tuple[List[int], List[int]]:
        """
        Render one block through the full pipeline.

        Args:
            voice_id: Voice slot
            note: MIDI note
            waveform: "saw", "square", "triangle", or "sine"
            gain_q15: Volume (Q15)
            cents: Fine tuning

        Returns:
            (L, R) stereo sample lists
        """
        frame = RenderFrame(
            voice_id=voice_id,
            note=note,
            cents=cents,
            gain_q15=gain_q15,
            waveform=waveform,
        )

        # Render through UnifiedRenderer (applies dual-render crossfade)
        samples = self.renderer.render_voice(frame)

        # Apply SafetyEngine declick
        samples = self.safety.apply_declick(samples, voice_id)

        # Convert to stereo
        L = list(samples)
        R = list(samples)

        # Apply SafetyEngine limiting
        L, R = self.safety.step_block(L, R, self.params)

        return L, R


# =============================================================================
# CARRIER + LIFT PIPELINE (v2 Architecture)
# =============================================================================

class CarrierLiftPipeline:
    """
    Carrier + IDWPT Lift Pipeline.

    This implements the v2 architecture where:
    - Carrier (Baby Fourier) owns pitch + phase continuity
    - IDWPT becomes a "lift" that shapes timbre, not an oscillator
    - Beta (β) is an audio-domain dry/wet blend with latency handling

    The carrier provides immediate pitch response while the wavelet
    "blooms in" over ~5 blocks of latency for rich timbral shaping.

    Key invariants:
    1. Pitch identity lives in carrier - wavelet never determines pitch
    2. Master leaf space fixed - D=6, 64 leaves, forever
    3. Trees = slow global tunings (1-5s morph), not per-voice swaps
    4. All discrete changes -> continuous morph (slew/crossfade everything)

    Anti-click handling is consolidated in AntiClickOperator which:
    - Aligns latency (5-block FIFO for wet/carrier sync)
    - Mixes carrier/wet with equal-power crossfade
    - Applies optional OLA seam smoothing

    Usage:
        pipeline = CarrierLiftPipeline()
        pipeline.voice_on(0, note=60)

        for block in range(num_blocks):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6, ...)
            L, R = pipeline.render_block(frame)

        pipeline.voice_off(0)
    """

    def __init__(self, block_size: int = BLOCK_SIZE):
        from .safety_engine import SafetyEngine
        from ..frames import ParamFrame, VoiceFrame
        from ..wavelet.carrier import CarrierOscillator
        from ..wavelet.carrier_lift import (
            CarrierLiftVoiceState, NUM_LEAVES
        )
        from ..wavelet.leaf_generator import LeafGenerator
        from ..tuning import TuningMorpher, TUNING_HARMONIC
        from .anti_click import AntiClickOperator, AntiClickConfig, AntiClickState

        # Store imports for use
        self.VoiceFrame = VoiceFrame
        self.NUM_LEAVES = NUM_LEAVES
        self.AntiClickState = AntiClickState

        # Core components
        self.carrier = CarrierOscillator()
        self.leaf_gen = LeafGenerator()
        self.tuning = TuningMorpher()
        self.safety = SafetyEngine(block_size=block_size)
        self.params = ParamFrame()
        self.block_size = block_size

        # Wavelet backend for IDWPT lift
        self.wavelet_backend = WaveletBackend(block_size=block_size)

        # AntiClick operator (single source of truth for click prevention)
        self.anticlick = AntiClickOperator(AntiClickConfig(
            block_size=block_size,
            sample_rate=48000,
            wet_latency_blocks=5,
            bloom_time_s=0.05,
            ola_K=16,
            equal_power=True
        ))

        # Per-voice state
        self.voice_states: Dict[int, CarrierLiftVoiceState] = {}
        self.ac_states: Dict[int, AntiClickState] = {}  # AntiClick state per voice

        # Initialize with harmonic tuning
        self.tuning.set_tuning_immediate(TUNING_HARMONIC)

    def voice_on(self, voice_id: int, note: int = 60, cents: int = 0,
                 velocity_q15: int = Q15_ONE) -> None:
        """
        Allocate and start a voice.

        Args:
            voice_id: Voice identifier
            note: MIDI note number
            cents: Fine tuning in cents
            velocity_q15: Note velocity (Q15)
        """
        from ..wavelet.carrier_lift import CarrierLiftVoiceState

        # Create voice state
        state = CarrierLiftVoiceState(voice_id=voice_id)
        state.note_on(note, cents, velocity_q15)
        self.voice_states[voice_id] = state

        # Initialize AntiClick state (replaces BlendState + DC blocker)
        self.ac_states[voice_id] = self.anticlick.init_state()
        self.anticlick.note_on(self.ac_states[voice_id])

        # Configure carrier
        f0_hz = 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))
        self.carrier.configure_voice(voice_id, note, cents)
        self.carrier.gate_on(voice_id)

        # Allocate wavelet voice AND reset wet state to prevent drones
        self.wavelet_backend.engine.voice_allocate(voice_id)
        self.wavelet_backend.engine.reset_wet_state(voice_id)
        self.wavelet_backend._allocated_voices.add(voice_id)

        # Safety
        self.safety.voice_on(voice_id)

    def voice_off(self, voice_id: int) -> None:
        """Signal voice release."""
        if voice_id in self.voice_states:
            self.voice_states[voice_id].note_off()
            self.carrier.gate_off(voice_id)

    def voice_deallocate(self, voice_id: int) -> None:
        """Fully deallocate a voice slot."""
        self.voice_states.pop(voice_id, None)
        self.ac_states.pop(voice_id, None)
        self.carrier.remove_state(voice_id)
        self.safety.voice_off(voice_id)

    def set_tuning(self, tuning_name: str, morph_time_sec: float = 2.0) -> None:
        """
        Change global instrument tuning.

        All voices share the same tuning, and changes morph slowly.

        Args:
            tuning_name: Tuning preset name (e.g., "harmonic", "bright")
            morph_time_sec: Transition time (1-5 seconds typical)
        """
        self.tuning.set_target_by_name(tuning_name, morph_time_sec)

    def render_voice(self, frame: 'VoiceFrame') -> List[int]:
        """
        Render one block for a voice using Carrier + Lift architecture.

        Args:
            frame: VoiceFrame containing all voice parameters

        Returns:
            List of Q15 audio samples
        """
        import numpy as np
        voice_id = frame.voice_id

        # Ensure voice is allocated
        if voice_id not in self.voice_states:
            self.voice_on(voice_id, frame.note, frame.cents)

        state = self.voice_states[voice_id]
        ac_state = self.ac_states[voice_id]

        # Update pitch if changed
        if frame.f0_hz != state.carrier.target_phase_inc:
            self.carrier.set_pitch(voice_id, frame.note, frame.cents)

        # Update carrier harmonics from frame (waveform control)
        if frame.partial_amps_q15:
            self.carrier.set_harmonics(voice_id, frame.partial_amps_q15)

        # Set wet target from frame (AntiClick handles ramping)
        beta_target = frame.beta_target_q15 / 32767.0
        self.anticlick.set_wet_target(ac_state, beta_target)

        # === Step 1: Render carrier (phase-continuous, immediate) ===
        x_carrier = self.carrier.render_block(voice_id, self.block_size)

        # === Step 2: Generate wet path (FWPT -> mask -> IDWPT) ===
        # Only render wet if beta target > 0
        x_wet = None
        if frame.beta_target_q15 > 0:
            # Get frequency for coefficient generation
            f0_hz = frame.f0_hz if frame.f0_hz > 0 else 440.0 * (2.0 ** ((frame.note - 69 + frame.cents / 100.0) / 12.0))

            from ..wavelet.harmonic_injector import create_harmonic_coeffs, WaveType

            # Determine waveform from partial amplitudes
            partial_amps = frame.partial_amps_q15 if frame.partial_amps_q15 else state.partial_amps_q15

            if partial_amps and partial_amps[0] > 0 and all(a == 0 for a in partial_amps[1:]):
                wave_type = WaveType.SINE
            else:
                wave_type = WaveType.SAW

            coeffs_256 = create_harmonic_coeffs(
                frequency=f0_hz,
                amplitude=Q15_HALF,
                wave_type=wave_type,
                partials_n=8,
                brightness_q15=Q15_ONE
            )

            # Apply global tuning gains
            tuning_gains = self.tuning.get_current_gains()
            from ..wavelet_trees import NODE_SIZE as WPT_NODE_SIZE
            for i in range(256):
                leaf_idx = i // WPT_NODE_SIZE
                if leaf_idx < len(tuning_gains):
                    coeffs_256[i] = (coeffs_256[i] * tuning_gains[leaf_idx]) >> 15

            # Effect mode: FWPT(carrier) -> mask -> IDWPT
            from ..wavelet.wavelet_packet_trees import PacketTreeSet
            if not hasattr(self, '_packet_tree_set'):
                self._packet_tree_set = PacketTreeSet()

            tree_id = frame.tree_a_id if frame.tree_blend_q15 < Q15_HALF else frame.tree_b_id

            x_wet = self._packet_tree_set.effect_mode(
                tree_id=tree_id,
                carrier_audio=x_carrier,
                mask_q15=None  # Identity mask for now
            )

        # === Step 3: AntiClick operator does ALL mixing + click prevention ===
        # Convert to numpy arrays for AntiClick
        carrier_np = np.array(x_carrier, dtype=np.float32) / 32768.0
        wet_np = None
        if x_wet is not None:
            wet_np = np.array(x_wet, dtype=np.float32) / 32768.0

        # AntiClick handles:
        # - Latency alignment (5-block FIFO)
        # - Beta ramping (equal-power crossfade)
        # - OLA seam smoothing
        output_np = self.anticlick.step(ac_state, carrier_np, wet_np)

        # Convert back to Q15
        output = (output_np * 32768.0).astype(np.int32)
        output = np.clip(output, -32768, 32767).tolist()

        # === Step 4: Apply voice gain ===
        output = [(s * frame.gain_q15) >> 15 for s in output]

        # Advance tuning morph
        self.tuning.advance_block()

        # Mark first block done
        if state.is_starting:
            state.after_first_block()

        return output

    def render_block(self, frame: 'VoiceFrame') -> Tuple[List[int], List[int]]:
        """
        Render one block through the full pipeline with safety.

        Args:
            frame: VoiceFrame containing all parameters

        Returns:
            (L, R) stereo sample lists
        """
        # Render voice (AntiClick handles all click prevention)
        samples = self.render_voice(frame)

        # Convert to stereo (using pan from frame)
        pan_q15 = frame.pan_q15
        pan_norm = (pan_q15 + Q15_SCALE) / (2 * Q15_SCALE)
        left_gain = math.cos(pan_norm * math.pi / 2)
        right_gain = math.sin(pan_norm * math.pi / 2)

        L = [int(s * left_gain) for s in samples]
        R = [int(s * right_gain) for s in samples]

        # Apply SafetyEngine limiting (minimal now that AntiClick handles clicks)
        # AntiClick already handles:
        #   - Latency alignment (5-block FIFO)
        #   - Beta ramping (equal-power crossfade)
        #   - OLA seam smoothing
        # SafetyEngine now only handles master limiting
        is_carrier_only = frame.beta_target_q15 == 0
        L, R = self.safety.step_block(
            L, R, self.params,
            apply_master_ola=False,  # AntiClick handles OLA
            apply_dc_removal=not is_carrier_only
        )

        return L, R

    def render_block_simple(
        self,
        voice_id: int,
        note: int,
        gain_q15: int = Q15_HALF,
        beta_target_q15: int = 24576,
    ) -> Tuple[List[int], List[int]]:
        """
        Simplified render for common use case.

        Args:
            voice_id: Voice identifier
            note: MIDI note
            gain_q15: Volume (Q15)
            beta_target_q15: Carrier/lift blend target

        Returns:
            (L, R) stereo sample lists
        """
        frame = self.VoiceFrame(
            voice_id=voice_id,
            note=note,
            gain_q15=gain_q15,
            beta_target_q15=beta_target_q15,
            f0_hz=440.0 * (2.0 ** ((note - 69) / 12.0)),
        )
        return self.render_block(frame)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Protocol
    'SynthesisBackend',
    # Data classes
    'RenderFrame',
    'DiscontinuityFlags',
    'VoiceState',
    # Utilities
    'generate_crossfade_window',
    'crossfade_samples',
    'apply_microfade',
    'detect_discontinuities',
    # Main class
    'UnifiedRenderer',
    # Backend adapters
    'HarmonicScaffoldBackend',
    'WaveletBackend',
    # Standard pipelines (recommended)
    'StandardWaveletPipeline',
    'StandardHarmonicPipeline',
    # Carrier + Lift Pipeline (v2 architecture)
    'CarrierLiftPipeline',
    # Constants
    'CROSSFADE_SAMPLES',
    'MICROFADE_SAMPLES',
    'PITCH_STEP_THRESH_CENTS',
    'TREE_BLEND_STEP_THRESH',
]
