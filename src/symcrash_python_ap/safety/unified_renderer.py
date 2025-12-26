"""
UnifiedOutputStage - The Closure Layer for Click-Free Audio
============================================================

WSE-STD-0xx: Render Hygiene & Transitions

This module is the SINGLE AUTHORITY for ensuring no discontinuities reach
the audio output. It implements the v2 Carrier + Lift architecture:

    Carrier (Baby Fourier) -> AntiClick -> Safety -> Output
           \-> IDWPT Lift -/

Key Responsibilities:
1. **note-off / steal never pops** - AntiClickOperator handles all transitions
2. **wet appears exactly after L blocks and fades in smoothly** - FIFO + beta ramp
3. **param jumps are mediated** - no zipper noise from abrupt changes

Architecture:
- Carrier (Baby Fourier) owns pitch + phase continuity (immediate response)
- IDWPT becomes a "lift" that shapes timbre (5-block latency)
- AntiClickOperator is the single source of truth for click prevention:
  * Latency alignment (5-block FIFO for wet/carrier sync)
  * Carrier/wet mixing (equal-power crossfade with beta ramp)
  * OLA seam smoothing

The old dual-render crossfade approach (UnifiedRenderer) is deprecated.
In v2 architecture, the carrier is always continuous, and wet blooms in
gradually - no need for dual-render.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import math

# Q15 constants
Q15_ONE = 32767
Q15_HALF = 16384
Q15_SCALE = 32768

# Default block size
BLOCK_SIZE = 64


# =============================================================================
# IMPORTS FROM CANONICAL LOCATIONS
# =============================================================================

# Backends live in wavelet/backends.py
from ..wavelet.backends import (
    SynthesisBackend,
    HarmonicScaffoldBackend,
    WaveletBackend,
)


# =============================================================================
# DEPRECATED: RenderFrame (use VoiceFrame from frames.py instead)
# =============================================================================

@dataclass
class RenderFrame:
    """
    DEPRECATED: Use VoiceFrame from frames.py instead.

    Kept for backwards compatibility with old code.
    """
    voice_id: int = 0
    note: int = 60
    cents: int = 0
    gain_q15: int = Q15_HALF
    pan_q15: int = 0
    tree_a_id: int = 1
    tree_b_id: int = 1
    tree_blend_q15: int = 0
    coeff_q15: List[int] = field(default_factory=list)
    waveform: str = "saw"
    eta_q15: int = Q15_HALF
    glide_active: bool = False
    glide_target_note: int = 60
    glide_target_cents: int = 0
    glide_samples_remaining: int = 0


# =============================================================================
# UNIFIED OUTPUT STAGE (v2 Architecture)
# =============================================================================

class UnifiedOutputStage:
    """
    Unified Output Stage - The Closure Layer for Click-Free Audio.

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

    AntiClickOperator is the single source of truth for click prevention:
    - Aligns latency (5-block FIFO for wet/carrier sync)
    - Mixes carrier/wet with equal-power crossfade
    - Applies optional OLA seam smoothing

    Usage:
        stage = UnifiedOutputStage()
        stage.voice_on(0, note=60)

        for block in range(num_blocks):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6, ...)
            L, R = stage.render_block(frame)

        stage.voice_off(0)
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
        self.ac_states: Dict[int, AntiClickState] = {}

        # Initialize with harmonic tuning
        self.tuning.set_tuning_immediate(TUNING_HARMONIC)

    def voice_on(self, voice_id: int, note: int = 60, cents: int = 0,
                 velocity_q15: int = Q15_ONE, external_adsr: bool = True) -> None:
        """
        Allocate and start a voice.

        Args:
            voice_id: Voice identifier
            note: MIDI note number
            cents: Fine tuning in cents
            velocity_q15: Note velocity (Q15)
            external_adsr: If True, caller handles ADSR via gain_q15 in VoiceFrame.
        """
        from ..wavelet.carrier_lift import CarrierLiftVoiceState

        # Create voice state
        state = CarrierLiftVoiceState(voice_id=voice_id)
        state.note_on(note, cents, velocity_q15)
        self.voice_states[voice_id] = state

        # Initialize AntiClick state
        self.ac_states[voice_id] = self.anticlick.init_state()
        self.anticlick.note_on(self.ac_states[voice_id])

        # Configure carrier
        self.carrier.configure_voice(voice_id, note, cents)

        # If external ADSR, set carrier envelope to instant
        if external_adsr:
            self.carrier.set_envelope_instant(voice_id)

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
        x_wet = None
        if frame.beta_target_q15 > 0:
            f0_hz = frame.f0_hz if frame.f0_hz > 0 else 440.0 * (2.0 ** ((frame.note - 69 + frame.cents / 100.0) / 12.0))

            from ..wavelet.harmonic_injector import create_harmonic_coeffs, WaveType

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

            # === Apply operators from preset ===
            if frame.ops_enable:
                from ..operator_library import (
                    build_operator_library, get_operator_id, apply_operator
                )

                if not hasattr(self, '_op_library'):
                    self._op_library = build_operator_library()

                coeffs_np = np.array(coeffs_256, dtype=np.int16)

                for op_name in frame.ops_enable:
                    op_id = get_operator_id(op_name)
                    if op_id in self._op_library:
                        op = self._op_library[op_id]
                        strength = frame.families.get(op_name, 1.0)

                        processed = apply_operator(coeffs_np, op)

                        if strength < 1.0:
                            coeffs_np = (
                                (coeffs_np.astype(np.int32) * int((1.0 - strength) * 32768) +
                                 processed.astype(np.int32) * int(strength * 32768)) >> 15
                            ).astype(np.int16)
                        else:
                            coeffs_np = processed

                coeffs_256 = coeffs_np.tolist()

            # Effect mode: FWPT(carrier) -> mask -> IDWPT
            from ..wavelet.wavelet_packet_trees import PacketTreeSet
            if not hasattr(self, '_packet_tree_set'):
                self._packet_tree_set = PacketTreeSet()

            tree_id = frame.tree_a_id if frame.tree_blend_q15 < Q15_HALF else frame.tree_b_id

            x_wet = self._packet_tree_set.effect_mode(
                tree_id=tree_id,
                carrier_audio=x_carrier,
                mask_q15=None
            )

        # === Step 3: AntiClick operator does ALL mixing + click prevention ===
        # Normalize to [-1, 1] - anticlick handles clipping internally
        carrier_np = np.array(x_carrier, dtype=np.float32) / 32768.0
        wet_np = None
        if x_wet is not None:
            wet_np = np.array(x_wet, dtype=np.float32) / 32768.0

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
        samples = self.render_voice(frame)

        # Convert to stereo
        pan_q15 = frame.pan_q15
        pan_norm = (pan_q15 + Q15_SCALE) / (2 * Q15_SCALE)
        left_gain = math.cos(pan_norm * math.pi / 2)
        right_gain = math.sin(pan_norm * math.pi / 2)

        L = [int(s * left_gain) for s in samples]
        R = [int(s * right_gain) for s in samples]

        # Apply SafetyEngine limiting (AntiClick already handled clicks)
        is_carrier_only = frame.beta_target_q15 == 0
        L, R = self.safety.step_block(
            L, R, self.params,
            apply_master_ola=False,
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
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================

# CarrierLiftPipeline is now UnifiedOutputStage
CarrierLiftPipeline = UnifiedOutputStage

# UnifiedRenderer is deprecated - use UnifiedOutputStage instead
# Kept as stub for import compatibility
class UnifiedRenderer:
    """
    DEPRECATED: Use UnifiedOutputStage instead.

    The old dual-render crossfade approach is obsolete in v2 architecture.
    CarrierLiftPipeline (now UnifiedOutputStage) uses AntiClickOperator
    for all click prevention.
    """
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "UnifiedRenderer is deprecated. Use UnifiedOutputStage instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Create a UnifiedOutputStage as fallback
        self._stage = UnifiedOutputStage()

    def __getattr__(self, name):
        return getattr(self._stage, name)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class (v2 architecture)
    'UnifiedOutputStage',
    # Backwards compatibility aliases
    'CarrierLiftPipeline',
    'UnifiedRenderer',  # Deprecated
    # Backend protocol and adapters (re-exported from wavelet.backends)
    'SynthesisBackend',
    'HarmonicScaffoldBackend',
    'WaveletBackend',
    # Deprecated data class
    'RenderFrame',
]
