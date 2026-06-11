"""
Emitter Bridge - Factory for Unified Voice Management
=====================================================

This module provides factory functions to create voice emitters that integrate
with VoiceEngine, unifying all synthesis paths through the CMMM architecture.

Per WSE-ARCHITECTURE-v1: Voice Engine owns musical time (LFO phases).

Three modes of operation:
1. **Standalone**: VoiceFrameEmitter with internal phase storage (no VoiceEngine)
2. **State-linked**: VoiceFrameEmitter sharing arc_phase_q16 with VoiceState
3. **Engine-backed**: Full VoiceEngine integration with CMMM regime/intent pipeline

Usage:
    from symcrash_python_ap.voice.emitter_bridge import (
        create_standalone_emitter,      # Mode 1: Simple, standalone
        create_emitter_with_state,      # Mode 2: Shared phase storage
        create_emitter_with_engine,     # Mode 3: Full CMMM integration
        VoiceEngineAdapter,             # Direct VoiceEngine wrapper
    )

Mode 3 Example (Full CMMM):
    from symcrash_python_ap.voice import VoiceEngine
    from symcrash_python_ap.library.preset_compiler import PresetCompiler

    # Create VoiceEngine as central voice manager
    engine = VoiceEngine(max_voices=8)

    # Compile preset
    compiler = PresetCompiler()
    preset = compiler.compile_file("brass_stab_v0.yaml")

    # Create engine-backed emitter (uses VoiceEngine for voice management)
    adapter = create_emitter_with_engine(preset, engine, voice_id=0)

    # Now all operations go through VoiceEngine
    adapter.note_on(note=60, velocity_q15=32767)
    frame = adapter.emit_frame()  # Produces VoiceFrame compatible with VM
"""

from typing import TYPE_CHECKING, Optional, List
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..library.preset_compiler import CompiledPreset
    from .voice_engine import VoiceState, VoiceEngine, VoiceFrameABI, GestureBasis


def create_emitter_with_state(
    preset: 'CompiledPreset',
    voice_state: 'VoiceState',
    voice_id: int = 0,
) -> 'VoiceFrameEmitter':
    """
    Create a VoiceFrameEmitter that shares phase storage with VoiceState.

    This unifies LFO phase storage between VoiceFrameEmitter and VoiceEngine,
    per WSE-ARCHITECTURE-v1 which specifies Voice Engine owns musical time.

    The emitter's arc_phase property will delegate to voice_state.arc_phase_q16,
    ensuring both systems use the same phase accumulators.

    Args:
        preset: CompiledPreset with LFO definitions
        voice_state: VoiceState instance to share phase storage with
        voice_id: Voice ID for this emitter (default 0)

    Returns:
        VoiceFrameEmitter with unified phase storage

    Example:
        >>> state = VoiceState(voice_id=0)
        >>> emitter = create_emitter_with_state(preset, state)
        >>> emitter.note_on(60, 32767)
        >>> # LFO phases now stored in state.arc_phase_q16
        >>> frame = emitter.emit_frame()
    """
    from ..voiceframe_emitter import VoiceFrameEmitter

    return VoiceFrameEmitter(
        preset=preset,
        voice_state=voice_state,
        voice_id=voice_id,
    )


def create_standalone_emitter(
    preset: 'CompiledPreset',
    voice_id: int = 0,
) -> 'VoiceFrameEmitter':
    """
    Create a standalone VoiceFrameEmitter with internal phase storage.

    This creates an emitter that uses its own _arc_phase_q16 storage,
    without connection to VoiceState. Use this for standalone operation
    (e.g., pico2_live_synth) where VoiceEngine is not used.

    Args:
        preset: CompiledPreset with LFO definitions
        voice_id: Voice ID for this emitter (default 0)

    Returns:
        VoiceFrameEmitter with internal phase storage

    Example:
        >>> emitter = create_standalone_emitter(preset)
        >>> emitter.note_on(60, 32767)
        >>> frame = emitter.emit_frame()
    """
    from ..voiceframe_emitter import VoiceFrameEmitter

    return VoiceFrameEmitter(
        preset=preset,
        voice_state=None,  # Standalone mode
        voice_id=voice_id,
    )


# =============================================================================
# MODE 3: VOICE ENGINE ADAPTER
# =============================================================================

@dataclass
class VoiceEngineAdapter:
    """
    Adapter that wraps VoiceEngine to provide VoiceFrameEmitter-compatible interface.

    This enables all synthesis paths to go through VoiceEngine (CMMM) while
    producing VoiceFrame output compatible with the VM.

    Key responsibilities:
    - Converts note_on/note_off to VoiceEngine API
    - Converts CompiledPreset to InstrumentProgram-like behavior
    - Converts VoiceFrameABI to VoiceFrame for VM consumption
    - Manages LFO state through VoiceState.arc_phase_q16

    Usage:
        engine = VoiceEngine(max_voices=8)
        adapter = VoiceEngineAdapter(preset, engine, voice_id=0)
        adapter.note_on(note=60, velocity_q15=32767)
        frame = adapter.emit_frame()  # VoiceFrame for VM
    """
    preset: 'CompiledPreset'
    engine: 'VoiceEngine' = None
    voice_id: int = 0

    # Note state (mirrors VoiceFrameEmitter interface)
    note: int = 60
    cents: int = 0
    velocity_q15: int = 32767

    # Block counter
    block_index: int = 0
    _is_note_on: bool = False

    def __post_init__(self):
        """Initialize voice in engine."""
        if self.engine is not None:
            # Allocate voice in engine
            self.engine.allocate_voice(self.voice_id)

            # Transfer LFO definitions from preset to VoiceState
            state = self.engine.voices.get(self.voice_id)
            if state is not None:
                lfos = getattr(self.preset, 'lfos', None)
                if lfos:
                    state.lfos = lfos

    def note_on(self, note: int, velocity_q15: int = 32767, cents: int = 0) -> None:
        """
        Trigger note-on through VoiceEngine.

        Converts VoiceFrameEmitter-style note_on to VoiceEngine API.
        """
        self.note = note
        self.cents = cents
        self.velocity_q15 = velocity_q15
        self.block_index = 0
        self._is_note_on = True

        if self.engine is not None:
            # Import here to avoid circular imports
            from .voice_engine import GestureBasis

            # Create gesture from note parameters
            gesture = GestureBasis.from_note_on(
                velocity_q15=velocity_q15,
                pitch_q15=(note << 8),  # Note to pitch_q15
                pan_q15=0
            )

            # Update note info in state
            state = self.engine.voices.get(self.voice_id)
            if state is not None:
                state.note_u8 = note
                state.cents_i16 = cents

            self.engine.note_on(self.voice_id, gesture, velocity_q15)

    def note_off(self) -> None:
        """Trigger note-off through VoiceEngine."""
        self._is_note_on = False

        if self.engine is not None:
            self.engine.note_off(self.voice_id)

    def emit_frame(self) -> 'VoiceFrame':
        """
        Emit VoiceFrame by stepping VoiceEngine.

        Converts VoiceFrameABI from VoiceEngine to VoiceFrame for VM.
        """
        from ..frames import VoiceFrame
        from ..voiceframe_emitter import midi_to_hz

        if self.engine is not None:
            # Step the voice through VoiceEngine CMMM
            abi = self.engine.step_voice(self.voice_id)

            if abi is not None:
                # Convert VoiceFrameABI to VoiceFrame
                frame = self._abi_to_voiceframe(abi)
            else:
                # Voice not active, return silent frame
                frame = self._create_silent_frame()
        else:
            # No engine, create basic frame
            frame = self._create_basic_frame()

        self.block_index += 1
        return frame

    def _abi_to_voiceframe(self, abi: 'VoiceFrameABI') -> 'VoiceFrame':
        """Convert VoiceFrameABI to VoiceFrame."""
        from ..frames import VoiceFrame
        from ..voiceframe_emitter import midi_to_hz

        # Get f0_hz from note
        state = self.engine.voices.get(self.voice_id)
        total_cents = self.cents
        if state is not None:
            total_cents += state.detune_cents

        f0_hz = midi_to_hz(self.note, total_cents)

        # Get preset fields
        osc_mode = getattr(self.preset, 'osc_mode', 0)
        fm6_config = getattr(self.preset, 'fm6', None)
        pwm_width_q15 = getattr(self.preset, 'pwm_width_q15', 16384)
        cmmm = getattr(self.preset, 'cmmm', None)

        # Build VoiceFrame from ABI + preset
        frame = VoiceFrame(
            voice_id=abi.voice_id,
            f0_hz=f0_hz,
            note=self.note,
            cents=total_cents,
            gain_q15=abi.gain_q15,
            beta_target_q15=self.preset.material.beta_target_q15,
            partial_amps_q15=list(self.preset.harmonic_amps_q15),
            ops_enable=self.preset.backbone_ops(),
            families=list(abi.w_family_q15),  # From VoiceEngine family weights
            tree_a_id=abi.tree_a_id,
            tree_b_id=abi.tree_b_id,
            tree_blend_q15=abi.tree_blend_q15,
            is_starting=abi.is_starting,
            is_active=abi.is_active,
            tilt_env_q15=abi.tilt_env_q15,
            density_env_q15=abi.density_env_q15,
            # PWM
            pwm_width_q15=pwm_width_q15,
            # FM6 fields
            coeff_osc_mode=osc_mode,
            fm_algorithm_id=fm6_config.algorithm_id if fm6_config else None,
            fm_op_ratios=list(fm6_config.op_ratios) if fm6_config else None,
            fm_op_levels=list(fm6_config.op_levels) if fm6_config else None,
            fm_feedback_depth=fm6_config.feedback_depth if fm6_config else None,
            fm_post_filter_enabled=(
                bool(getattr(fm6_config, 'post_filter_hz', None)) or None
                if fm6_config else None
            ),
            fm_post_filter_cutoff_hz=(
                getattr(fm6_config, 'post_filter_hz', None)
                if fm6_config else None
            ),
            # CMMM
            cmmm_node_mask=cmmm.node_mask_u32 if cmmm else None,
            cmmm_mask_ramp_blocks=cmmm.mask_ramp_blocks if cmmm else 5,
            cmmm_tail_mode=cmmm.tail_mode if cmmm else None,
            cmmm_traction_q15=cmmm.traction_q15 if cmmm else 32767,
        )

        return frame

    def _create_silent_frame(self) -> 'VoiceFrame':
        """Create a silent frame when voice is inactive."""
        from ..frames import VoiceFrame

        return VoiceFrame(
            voice_id=self.voice_id,
            f0_hz=440.0,
            note=self.note,
            gain_q15=0,
            is_active=False,
        )

    def _create_basic_frame(self) -> 'VoiceFrame':
        """Create basic frame when no engine (fallback)."""
        from ..frames import VoiceFrame
        from ..voiceframe_emitter import midi_to_hz

        f0_hz = midi_to_hz(self.note, self.cents)

        return VoiceFrame(
            voice_id=self.voice_id,
            f0_hz=f0_hz,
            note=self.note,
            cents=self.cents,
            gain_q15=self.velocity_q15 if self._is_note_on else 0,
            is_active=self._is_note_on,
        )

    @property
    def is_active(self) -> bool:
        """True if voice is active."""
        if self.engine is not None:
            state = self.engine.voices.get(self.voice_id)
            if state is not None:
                from .voice_engine import VoicePhase
                return state.phase != VoicePhase.FREE
        return self._is_note_on

    @property
    def is_releasing(self) -> bool:
        """True if voice is in release phase."""
        if self.engine is not None:
            state = self.engine.voices.get(self.voice_id)
            if state is not None:
                from .voice_engine import VoicePhase
                return state.phase == VoicePhase.RELEASE
        return not self._is_note_on

    def reset(self) -> None:
        """Reset adapter state."""
        self.block_index = 0
        self._is_note_on = False
        # Note: VoiceEngine state persists; use engine methods to fully reset


def create_emitter_with_engine(
    preset: 'CompiledPreset',
    engine: 'VoiceEngine',
    voice_id: int = 0,
) -> VoiceEngineAdapter:
    """
    Create a VoiceEngineAdapter that routes all operations through VoiceEngine.

    This is the recommended way to create voice emitters for unified synthesis
    where all paths go through the CMMM architecture.

    Args:
        preset: CompiledPreset with instrument definition
        engine: VoiceEngine instance (central voice manager)
        voice_id: Voice ID to use in engine

    Returns:
        VoiceEngineAdapter with VoiceFrameEmitter-compatible interface

    Example:
        >>> engine = VoiceEngine(max_voices=8)
        >>> adapter = create_emitter_with_engine(preset, engine, voice_id=0)
        >>> adapter.note_on(60, 32767)
        >>> frame = adapter.emit_frame()
    """
    return VoiceEngineAdapter(
        preset=preset,
        engine=engine,
        voice_id=voice_id,
    )


__all__ = [
    'create_emitter_with_state',
    'create_standalone_emitter',
    'create_emitter_with_engine',
    'VoiceEngineAdapter',
]
