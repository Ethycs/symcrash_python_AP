"""
UnifiedOutputStage - The Closure Layer for Click-Free Audio
============================================================

WSE-STD-0xx: Render Hygiene & Transitions

This module is the SINGLE AUTHORITY for ensuring no discontinuities reach
the audio output. It implements the v2 Carrier + Lift architecture:

    Carrier (Baby Fourier) -> AntiClick -> Safety -> Output
           |-> IDWPT Lift -|

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
import numpy as np

from ..primitives import Q15_ONE, Q15_HALF, Q15_SCALE, BLOCK_SIZE


# =============================================================================
# IMPORTS FROM CANONICAL LOCATIONS
# =============================================================================

# Backends live in wavelet/backends.py
from ..wavelet.backends import (
    SynthesisBackend,
    HarmonicScaffoldBackend,
    WaveletBackend,
)
from ..primitives import Q15_3_4

# Use sine_core for bit-exact trig (no math.sin/cos in hot paths)
from ..wavelet.sine_core import sine_q15, cosine_q15, PHASE_MASK

# Fixed-point math for Q15 multiplication without float truncation
from ..fixed import mul_q15_sat

# VM Executor (WSE-STD-VM-HYBRID v0)
from ..vm import vm_step_block, VMState, DecodedPlan, Tables


# =============================================================================
# UNIFIED OUTPUT STAGE (v2 Architecture)
# =============================================================================
# (The deprecated RenderFrame dataclass and UnifiedRenderer stub were
# removed 2026-07-02; frames.VoiceFrame and UnifiedOutputStage are the
# single owners.)

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

        # VM executor state (WSE-STD-VM-HYBRID v0)
        self.vm_states: Dict[int, VMState] = {}
        self.vm_tables = Tables()

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

        # Initialize VM state (WSE-STD-VM-HYBRID v0)
        vm_state = VMState.create(voice_id)
        vm_state.reset_for_note_on(note=note, cents=cents)
        self.vm_states[voice_id] = vm_state

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
        # Clean up VM state
        self.vm_states.pop(voice_id, None)

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
        voice_id = frame.voice_id

        # Ensure voice is allocated
        if voice_id not in self.voice_states:
            self.voice_on(voice_id, frame.note, frame.cents)

        # =====================================================================
        # VM EXECUTOR PATH (WSE-STD-VM-HYBRID v0)
        # =====================================================================
        # Get or create VM state
        if voice_id not in self.vm_states:
            vm_state = VMState.create(voice_id)
            vm_state.reset_for_note_on(note=frame.note, cents=frame.cents)
            self.vm_states[voice_id] = vm_state

        # Execute one block through VM
        vm_state = self.vm_states[voice_id]
        vm_state, output_block = vm_step_block(vm_state, frame, self.vm_tables)
        self.vm_states[voice_id] = vm_state

        # VM output is int16 numpy array, convert to list
        output = output_block.tolist()

        # Advance tuning morph (still needed for global state)
        self.tuning.advance_block()

        # Mark first block done
        state = self.voice_states[voice_id]
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

        # Convert to stereo using sine_core for bit-exact pan law
        # pan_q15 range: -32768 (full left) to +32767 (full right)
        # Map to phase [0, pi/2] for equal-power panning
        # All integer math to avoid floating-point jitter
        pan_q15 = frame.pan_q15
        # Map pan_q15 from [-32768, 32767] to [0, 65535] (Q16 unsigned)
        pan_u16 = pan_q15 + Q15_SCALE
        # Scale to phase [0, 0x40000000]: phase = pan_u16 * 0x40000000 / 65536
        # = (pan_u16 << 14) since 0x40000000 / 65536 = 0x4000 = 16384 = 2^14
        pan_phase = (pan_u16 << 14) & PHASE_MASK
        # left = cos(phase), right = sin(phase) for equal-power pan
        left_q15 = cosine_q15(pan_phase)
        right_q15 = sine_q15(pan_phase)

        # Apply pan gains in Q15 (no float truncation)
        L = [mul_q15_sat(s, left_q15) for s in samples]
        R = [mul_q15_sat(s, right_q15) for s in samples]

        # Apply SafetyEngine processing
        # NOTE: OLA is now applied in lane_idwpt_render to the wet path ONLY.
        # The carrier is phase-continuous and applying OLA to it corrupts the signal.
        # Master OLA is disabled here since wet path OLA is handled at the correct layer.
        #
        # DC removal is DISABLED for VM/CCS path:
        # - CCS path produces phase-continuous sine waves with zero inherent DC
        # - Block-based "DC offset" is just incomplete cycles, not true DC
        # - Removing it causes block boundary discontinuities
        L, R = self.safety.step_block(
            L, R, self.params,
            apply_master_ola=False,  # OLA handled in VM lane_idwpt_render
            apply_dc_removal=False   # CCS path is phase-continuous with zero DC
        )

        return L, R

    def render_block_simple(
        self,
        voice_id: int,
        note: int,
        gain_q15: int = Q15_HALF,
        beta_target_q15: int = Q15_3_4,
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


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class (v2 architecture)
    'UnifiedOutputStage',
    # Backwards compatibility alias
    'CarrierLiftPipeline',
    # Backend protocol and adapters (re-exported from wavelet.backends)
    'SynthesisBackend',
    'HarmonicScaffoldBackend',
    'WaveletBackend',
]
