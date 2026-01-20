"""
VM State (WSE-STD-VM-HYBRID v0)
===============================

Per-voice persistent state for the hybrid VM.
Snapshotable for deterministic replay.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional
import numpy as np

from .tables import COEFF_LEN, BLOCK_SAMPLES
from ..fixed import midi_to_phase_inc_fixed
from .coeff_lut_streaming import CCSState, midi_to_hz, SAMPLE_RATE
from ..primitives import Q15_ONE

# Defer import to avoid circular dependency - ModeBState imported when needed
# from .harmonic_bank import ModeBState, OSC_MODE_A, OSC_MODE_B

# Oscillator mode constants (WSE-STD-210)
OSC_MODE_A = 0       # LUT-based (CCSState)
OSC_MODE_B = 1       # Rotator/Harmonic Bank (ModeBState)
OSC_MODE_SAMPLE = 2  # Sample playback via FWPT gateway (Phase 3)
OSC_MODE_FM6 = 3     # 6-operator FM synthesis (WSE-STD-FM-TOPOLOGY-v0)


# =============================================================================
# EXECUTOR LATCH (CMMM Runtime State)
# =============================================================================
# The executor latch holds runtime state that is updated by CMMM event payloads.
# This enables dynamic control of node masks, tail modes, and other CMMM features.

# Tail mode enum (imported from tables at runtime to avoid circular import)
# TailMode: FULL=0, MID=1, MIN=2, EMERGENCY=3

# Event payload types for CMMM control
PAYLOAD_TYPE_NODEMASK_SET = 0x30   # Sets node mask with optional ramp
PAYLOAD_TYPE_TAILMODE_SET = 0x31   # Sets tail mode with traction


@dataclass
class ExecutorLatch:
    """
    CMMM executor latch state.

    Holds latched values that control operator execution, updated by
    event payloads during the CMMM step.

    Attributes:
        node_mask_u32: Current node mask (32-bit, one bit per node)
        mask_target_u32: Target mask for ramping
        mask_ramp_remaining: Blocks remaining in mask ramp
        tail_mode: Current tail mode (0=FULL, 1=MID, 2=MIN, 3=EMERGENCY)
        traction_q15: Projection hardness (0-32767)
    """
    # Node mask control (payload_type 0x30)
    node_mask_u32: int = 0xFFFFFFFF      # Current active node mask
    mask_target_u32: int = 0xFFFFFFFF    # Target mask for ramping
    mask_ramp_remaining: int = 0          # Blocks remaining in crossfade

    # Tail mode control (payload_type 0x31)
    tail_mode: int = 0                    # TailMode enum value (0=FULL)
    traction_q15: int = Q15_ONE            # Projection hardness

    # Ramp tracking for crossfade
    _mask_ramp_total: int = 0             # Total blocks for current ramp
    _mask_start_u32: int = 0xFFFFFFFF     # Mask at ramp start

    def apply_nodemask_event(self, new_mask: int, ramp_blocks: int = 5) -> None:
        """
        Apply a node mask change event (payload_type 0x30).

        Args:
            new_mask: New target node mask
            ramp_blocks: Number of blocks to crossfade (0 = immediate)
        """
        if ramp_blocks <= 0:
            # Immediate change
            self.node_mask_u32 = new_mask
            self.mask_target_u32 = new_mask
            self.mask_ramp_remaining = 0
            self._mask_ramp_total = 0
        else:
            # Start ramping - save current mask as start
            self._mask_start_u32 = self.node_mask_u32
            self.mask_target_u32 = new_mask
            self.mask_ramp_remaining = ramp_blocks
            self._mask_ramp_total = ramp_blocks

    def apply_tailmode_event(self, tail_mode: int, traction_q15: int = Q15_ONE) -> None:
        """
        Apply a tail mode change event (payload_type 0x31).

        Args:
            tail_mode: New tail mode (0=FULL, 1=MID, 2=MIN, 3=EMERGENCY)
            traction_q15: Projection hardness (0-32767)
        """
        self.tail_mode = tail_mode
        self.traction_q15 = traction_q15

    def step_mask_ramp(self) -> None:
        """
        Advance mask ramp by one block.

        Call this at the start of each block to update the current mask
        based on the ramp progress.
        """
        if self.mask_ramp_remaining > 0:
            self.mask_ramp_remaining -= 1
            if self.mask_ramp_remaining <= 0:
                # Ramp complete: snap to target
                self.node_mask_u32 = self.mask_target_u32
            # Note: Intermediate mask blending happens in the operator dispatcher

    def get_effective_k(self) -> int:
        """Get active node count (K) from current mask."""
        return (self.node_mask_u32 & 0xFFFFFFFF).bit_count()

    def get_crossfade_weights(self) -> tuple:
        """
        Get crossfade weights for smooth mask transitions.

        During a mask ramp, returns (start_weight_q15, target_weight_q15, start_mask, target_mask)
        where weights sum to ~32767.

        If not ramping, returns (0, 32767, 0, current_mask).

        Returns:
            (start_weight_q15, target_weight_q15, start_mask, target_mask)
        """
        if self._mask_ramp_total <= 0 or self.mask_ramp_remaining <= 0:
            # Not ramping - use target mask at full weight
            return (0, 32767, 0, self.node_mask_u32)

        # Compute blend factor: how far through the ramp
        # ramp_remaining=total → progress=0 (all start)
        # ramp_remaining=0 → progress=1 (all target)
        progress_blocks = self._mask_ramp_total - self.mask_ramp_remaining
        alpha_q15 = (progress_blocks * 32767) // self._mask_ramp_total

        start_weight = Q15_ONE - alpha_q15
        target_weight = alpha_q15

        return (start_weight, target_weight, self._mask_start_u32, self.mask_target_u32)

    def is_ramping(self) -> bool:
        """Check if currently in a mask ramp transition."""
        return self.mask_ramp_remaining > 0


# =============================================================================
# ENVELOPE STATE
# =============================================================================

class EnvelopePhase(IntEnum):
    """ADSR envelope phases."""
    ATTACK = 0
    DECAY = 1
    SUSTAIN = 2
    RELEASE = 3
    FREE = 4  # Voice is free


@dataclass
class EnvelopeState:
    """ADSR envelope state."""
    phase: EnvelopePhase = EnvelopePhase.FREE
    phase_sample: int = 0  # Sample counter within current phase
    level_q15: int = 0     # Current level (0-32767)

    # Envelope parameters (set at note-on)
    attack_samples: int = 0
    decay_samples: int = 0
    sustain_level_q15: int = 24576  # ~0.75
    release_samples: int = 0


# =============================================================================
# VM STATE
# =============================================================================

@dataclass
class VMState:
    """
    Per-voice persistent state for the hybrid VM.

    All fields are deterministic and snapshotable.
    """
    voice_id: int = 0

    # -------------------------------------------------------------------------
    # Carrier oscillator state
    # -------------------------------------------------------------------------
    phase: int = 0                    # 32-bit phase accumulator (0..2^32-1)
    phase_inc: int = 0                # Phase increment per sample
    target_phase_inc: int = 0         # Target for pitch slew
    pitch_slew_rate: int = 0          # Per-sample delta for portamento

    # Pitch tracking (to avoid floating-point noise triggering unnecessary updates)
    last_note: int = -1               # Last set MIDI note (-1 = not set)
    last_cents: int = 0               # Last set cents value

    # Harmonics (up to 16)
    harmonic_amps_q15: List[int] = field(default_factory=lambda: [0] * 16)

    # Envelope
    envelope: EnvelopeState = field(default_factory=EnvelopeState)

    # DC blocking filter state (1-pole highpass)
    dc_block_x1: int = 0  # Previous input
    dc_block_y1: int = 0  # Previous output

    # -------------------------------------------------------------------------
    # Coefficient state
    # -------------------------------------------------------------------------
    coeff_state: np.ndarray = field(
        default_factory=lambda: np.zeros(COEFF_LEN, dtype=np.int16)
    )

    # -------------------------------------------------------------------------
    # PRNG state
    # -------------------------------------------------------------------------
    prng_state: int = 0x12345678  # xorshift32 seed

    # -------------------------------------------------------------------------
    # Tree state (optional, for IDWPT)
    # -------------------------------------------------------------------------
    tree_a_id: int = 0
    tree_b_id: int = 0

    # -------------------------------------------------------------------------
    # Latency tracking (microcode contract)
    # -------------------------------------------------------------------------
    blocks_since_start: int = 0  # Blocks since voice allocation
    wet_ready: bool = False       # Set after latency period

    # -------------------------------------------------------------------------
    # Previous block values (for ramping)
    # -------------------------------------------------------------------------
    eta_prev_q15: int = 0         # Previous block's eta (dry/wet)
    gain_prev_q15: int = 0        # Previous block's gain

    # -------------------------------------------------------------------------
    # Wet path phase-coherent state (no windowing)
    # -------------------------------------------------------------------------
    # Mask smoothing state: interpolate mask values across blocks
    mask_state_q15: np.ndarray = field(
        default_factory=lambda: np.zeros(64, dtype=np.int16)
    )
    mask_state_initialized: bool = False

    # C⁰ seam stitching: track previous block's end sample for continuity
    wet_prev_end: int = 0
    wet_seam_initialized: bool = False

    # Legacy OLA state (kept for backward compatibility, may be removed)
    wet_ola_tail: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.int16)
    )
    wet_ola_initialized: bool = False

    # -------------------------------------------------------------------------
    # CCS State (WSE-STD-121 Streaming Coefficients)
    # -------------------------------------------------------------------------
    # CCS generates coefficients directly from phase-indexed tables,
    # bypassing FWPT in voice path. This eliminates phase-variant combing.
    ccs_state: CCSState = field(default_factory=CCSState)

    # Flag to enable CCS path (coefficient synthesis vs carrier→FWPT)
    use_coeff_osc: bool = True  # Default to new CCS path

    # -------------------------------------------------------------------------
    # Mode B State (WSE-STD-210 Harmonic Bank)
    # -------------------------------------------------------------------------
    # Mode B generates coefficients dynamically from harmonic basis frames.
    # Supports PWM, FM/PM, and other modulation types.
    # Note: mode_b_state is created lazily to avoid circular import
    _mode_b_state: Optional['ModeBState'] = None  # type: ignore

    # Oscillator mode selection (WSE-STD-210 Section 10.1)
    osc_mode: int = OSC_MODE_A  # Default to Mode A (LUT-based)

    # -------------------------------------------------------------------------
    # FM Feedback Buffer (Phase 3 - WSE-STD-210 Section 8.3)
    # -------------------------------------------------------------------------
    # z⁻¹ block-delayed modulator signal for FM synthesis.
    # Extracted from coeff_state after backbone operators, used next block.
    prev_fm_mod_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.int16)
    )

    # -------------------------------------------------------------------------
    # Sample Playback State (Phase 3 - Audio-Native FWPT Gateway)
    # -------------------------------------------------------------------------
    # Lazy-initialized like mode_b_state to avoid import overhead
    _sample_state: Optional['SampleOscState'] = None  # type: ignore

    # -------------------------------------------------------------------------
    # FM6 State (WSE-STD-FM-TOPOLOGY-v0 - 6-Operator FM Synthesis)
    # -------------------------------------------------------------------------
    # Lazy-initialized to avoid import overhead when not using FM6 mode
    _fm6_state: Optional['FM6State'] = None  # type: ignore

    # FM6 Morph State (WSE-STD-FM-TOPOLOGY-v0 Section 8 - Topology Morphing)
    # -------------------------------------------------------------------------
    # Used when L1Lift.fm6_morph is configured for blending between archetypes
    _fm6_morph_state: Optional['FM6MorphState'] = None  # type: ignore

    # -------------------------------------------------------------------------
    # CMMM Executor Latch (WSE-STD-070 Tower Runtime State)
    # -------------------------------------------------------------------------
    # Holds latched values for node mask and tail mode, updated by CMMM events
    executor_latch: ExecutorLatch = field(default_factory=ExecutorLatch)

    @property
    def fm6_state(self) -> 'FM6State':
        """Get or create FM6 state (lazy init)."""
        if self._fm6_state is None:
            from .fm6_state import FM6State
            self._fm6_state = FM6State()
        return self._fm6_state

    @fm6_state.setter
    def fm6_state(self, value: 'FM6State') -> None:
        """Set FM6 state."""
        self._fm6_state = value

    @property
    def fm6_morph_state(self) -> 'FM6MorphState':
        """Get or create FM6 morph state (lazy init)."""
        if self._fm6_morph_state is None:
            from .fm6_morph_engine import FM6MorphState
            self._fm6_morph_state = FM6MorphState()
        return self._fm6_morph_state

    @fm6_morph_state.setter
    def fm6_morph_state(self, value: 'FM6MorphState') -> None:
        """Set FM6 morph state."""
        self._fm6_morph_state = value

    @property
    def sample_state(self) -> 'SampleOscState':
        """Get or create Sample state (lazy init)."""
        if self._sample_state is None:
            from .sample_source import SampleOscState
            self._sample_state = SampleOscState()
        return self._sample_state

    @sample_state.setter
    def sample_state(self, value: 'SampleOscState') -> None:
        """Set Sample state."""
        self._sample_state = value

    @property
    def mode_b_state(self) -> 'ModeBState':
        """Get or create Mode B state (lazy init to avoid circular import)."""
        if self._mode_b_state is None:
            from .harmonic_bank import ModeBState
            self._mode_b_state = ModeBState()
        return self._mode_b_state

    @mode_b_state.setter
    def mode_b_state(self, value: 'ModeBState') -> None:
        """Set Mode B state."""
        self._mode_b_state = value

    @classmethod
    def create(cls, voice_id: int = 0, prng_seed: Optional[int] = None) -> 'VMState':
        """
        Create a new VMState for a voice.

        Args:
            voice_id: Voice slot ID
            prng_seed: Optional PRNG seed (default: voice_id based)

        Returns:
            Initialized VMState
        """
        state = cls(voice_id=voice_id)
        if prng_seed is not None:
            state.prng_state = prng_seed & 0xFFFFFFFF
        else:
            # Deterministic seed based on voice_id
            state.prng_state = (0x12345678 ^ (voice_id * 0x9E3779B9)) & 0xFFFFFFFF
        return state

    def reset_for_note_on(
        self,
        note: int = 60,
        cents: int = 0,
        attack_samples: int = 480,
        decay_samples: int = 960,
        sustain_level_q15: int = 24576,
        release_samples: int = 4800,
        *,
        f0_hz: Optional[float] = None,
        sample_rate: int = 48000,  # Unused, kept for backward compat
    ) -> None:
        """
        Reset state for a new note.

        Args:
            note: MIDI note number (0-127) - preferred
            cents: Fine tuning in cents
            attack_samples: Attack time in samples
            decay_samples: Decay time in samples
            sustain_level_q15: Sustain level (0-32767)
            release_samples: Release time in samples
            f0_hz: DEPRECATED - Fundamental frequency (use note/cents instead)
            sample_rate: Unused, kept for backward compatibility
        """
        _ = sample_rate  # Suppress unused warning
        # Calculate phase increment
        if f0_hz is not None:
            # DEPRECATED: Convert f0_hz to note/cents for backward compatibility
            # This uses floating-point, but only for legacy callers
            import math
            note = int(round(69 + 12 * math.log2(f0_hz / 440.0)))
            cents = 0  # Can't easily recover cents from f0_hz
            phase_inc = midi_to_phase_inc_fixed(note, cents)
        else:
            # Preferred: Use fixed-point LUT (WSE-STD-001 compliant)
            phase_inc = midi_to_phase_inc_fixed(note, cents)

        self.phase_inc = phase_inc
        self.target_phase_inc = phase_inc
        self.pitch_slew_rate = 0

        # Track initial note/cents for comparison
        self.last_note = note
        self.last_cents = cents

        # Reset phase (or keep for legato)
        # self.phase = 0

        # Reset envelope
        self.envelope = EnvelopeState(
            phase=EnvelopePhase.ATTACK,
            phase_sample=0,
            level_q15=0,
            attack_samples=attack_samples,
            decay_samples=decay_samples,
            sustain_level_q15=sustain_level_q15,
            release_samples=release_samples,
        )

        # Reset coefficients
        self.coeff_state.fill(0)

        # Reset latency tracking
        self.blocks_since_start = 0
        self.wet_ready = False

        # Reset previous values
        self.eta_prev_q15 = 0
        self.gain_prev_q15 = 0

        # Reset wet path phase-coherent state
        self.mask_state_q15.fill(0)
        self.mask_state_initialized = False
        self.wet_prev_end = 0
        self.wet_seam_initialized = False

        # Reset legacy wet OLA state
        self.wet_ola_tail.fill(0)
        self.wet_ola_initialized = False

        # Reset CCS state (WSE-STD-121)
        self.ccs_state.reset(note=note, phase_inc=phase_inc, wave_id=0)

        # Reset Mode B state (WSE-STD-210) if initialized
        if self._mode_b_state is not None:
            self._mode_b_state.reset(note=note, phase_inc=phase_inc)

        # Reset FM6 state (WSE-STD-FM-TOPOLOGY-v0) if initialized
        if self._fm6_state is not None:
            self._fm6_state.reset()

        # Reset FM6 morph state if initialized
        if self._fm6_morph_state is not None:
            self._fm6_morph_state.reset()

        # Reset FM feedback buffer (Phase 3)
        self.prev_fm_mod_buffer.fill(0)

        # Reset CMMM Executor Latch (WSE-STD-070)
        self.executor_latch = ExecutorLatch()

    def trigger_release(self) -> None:
        """Trigger note release (enter release phase)."""
        if self.envelope.phase in (EnvelopePhase.ATTACK, EnvelopePhase.DECAY, EnvelopePhase.SUSTAIN):
            self.envelope.phase = EnvelopePhase.RELEASE
            self.envelope.phase_sample = 0

    def is_active(self) -> bool:
        """Check if voice is currently active."""
        return self.envelope.phase != EnvelopePhase.FREE

    def snapshot(self) -> dict:
        """
        Create a snapshot of the state for deterministic replay.

        Returns:
            Dictionary with all state values
        """
        return {
            'voice_id': self.voice_id,
            'phase': self.phase,
            'phase_inc': self.phase_inc,
            'target_phase_inc': self.target_phase_inc,
            'pitch_slew_rate': self.pitch_slew_rate,
            'harmonic_amps_q15': list(self.harmonic_amps_q15),
            'envelope_phase': int(self.envelope.phase),
            'envelope_sample': self.envelope.phase_sample,
            'envelope_level': self.envelope.level_q15,
            'dc_block_x1': self.dc_block_x1,
            'dc_block_y1': self.dc_block_y1,
            'coeff_state': self.coeff_state.tolist(),
            'prng_state': self.prng_state,
            'tree_a_id': self.tree_a_id,
            'tree_b_id': self.tree_b_id,
            'blocks_since_start': self.blocks_since_start,
            'wet_ready': self.wet_ready,
            'eta_prev_q15': self.eta_prev_q15,
            'gain_prev_q15': self.gain_prev_q15,
            # Wet path phase-coherent state
            'mask_state_q15': self.mask_state_q15.tolist(),
            'mask_state_initialized': self.mask_state_initialized,
            'wet_prev_end': self.wet_prev_end,
            'wet_seam_initialized': self.wet_seam_initialized,
            # Legacy wet path OLA state
            'wet_ola_tail': self.wet_ola_tail.tolist(),
            'wet_ola_initialized': self.wet_ola_initialized,
            # CCS state (WSE-STD-121)
            'ccs_phase': self.ccs_state.phase,
            'ccs_phase_inc': self.ccs_state.phase_inc,
            'ccs_target_phase_inc': self.ccs_state.target_phase_inc,
            'ccs_phase_slew_rate': self.ccs_state.phase_slew_rate,
            'ccs_note': self.ccs_state.note,
            'ccs_wave_id': self.ccs_state.wave_id,
            'use_coeff_osc': self.use_coeff_osc,
            # FM feedback buffer (Phase 3)
            'prev_fm_mod_buffer': self.prev_fm_mod_buffer.tolist(),
            # CMMM Executor Latch (WSE-STD-070)
            'executor_latch_node_mask': self.executor_latch.node_mask_u32,
            'executor_latch_mask_target': self.executor_latch.mask_target_u32,
            'executor_latch_mask_ramp': self.executor_latch.mask_ramp_remaining,
            'executor_latch_tail_mode': self.executor_latch.tail_mode,
            'executor_latch_traction': self.executor_latch.traction_q15,
            'executor_latch_mask_ramp_total': self.executor_latch._mask_ramp_total,
            'executor_latch_mask_start': self.executor_latch._mask_start_u32,
        }

    @classmethod
    def restore(cls, snapshot: dict) -> 'VMState':
        """
        Restore state from a snapshot.

        Args:
            snapshot: Dictionary from snapshot()

        Returns:
            Restored VMState
        """
        state = cls(voice_id=snapshot['voice_id'])
        state.phase = snapshot['phase']
        state.phase_inc = snapshot['phase_inc']
        state.target_phase_inc = snapshot['target_phase_inc']
        state.pitch_slew_rate = snapshot['pitch_slew_rate']
        state.harmonic_amps_q15 = snapshot['harmonic_amps_q15']
        state.envelope.phase = EnvelopePhase(snapshot['envelope_phase'])
        state.envelope.phase_sample = snapshot['envelope_sample']
        state.envelope.level_q15 = snapshot['envelope_level']
        state.dc_block_x1 = snapshot['dc_block_x1']
        state.dc_block_y1 = snapshot['dc_block_y1']
        state.coeff_state = np.array(snapshot['coeff_state'], dtype=np.int16)
        state.prng_state = snapshot['prng_state']
        state.tree_a_id = snapshot['tree_a_id']
        state.tree_b_id = snapshot['tree_b_id']
        state.blocks_since_start = snapshot['blocks_since_start']
        state.wet_ready = snapshot['wet_ready']
        state.eta_prev_q15 = snapshot['eta_prev_q15']
        state.gain_prev_q15 = snapshot['gain_prev_q15']
        # Restore wet path phase-coherent state
        if 'mask_state_q15' in snapshot:
            state.mask_state_q15 = np.array(snapshot['mask_state_q15'], dtype=np.int16)
            state.mask_state_initialized = snapshot.get('mask_state_initialized', False)
            state.wet_prev_end = snapshot.get('wet_prev_end', 0)
            state.wet_seam_initialized = snapshot.get('wet_seam_initialized', False)
        # Restore legacy wet OLA state
        if 'wet_ola_tail' in snapshot:
            state.wet_ola_tail = np.array(snapshot['wet_ola_tail'], dtype=np.int16)
            state.wet_ola_initialized = snapshot.get('wet_ola_initialized', False)
        # Restore CCS state (WSE-STD-121)
        if 'ccs_phase' in snapshot:
            state.ccs_state.phase = snapshot['ccs_phase']
            state.ccs_state.phase_inc = snapshot['ccs_phase_inc']
            state.ccs_state.target_phase_inc = snapshot['ccs_target_phase_inc']
            state.ccs_state.phase_slew_rate = snapshot['ccs_phase_slew_rate']
            state.ccs_state.note = snapshot['ccs_note']
            state.ccs_state.wave_id = snapshot['ccs_wave_id']
        state.use_coeff_osc = snapshot.get('use_coeff_osc', True)
        # Restore FM feedback buffer (Phase 3)
        if 'prev_fm_mod_buffer' in snapshot:
            state.prev_fm_mod_buffer = np.array(snapshot['prev_fm_mod_buffer'], dtype=np.int16)
        # Restore CMMM Executor Latch (WSE-STD-070)
        if 'executor_latch_node_mask' in snapshot:
            state.executor_latch.node_mask_u32 = snapshot['executor_latch_node_mask']
            state.executor_latch.mask_target_u32 = snapshot['executor_latch_mask_target']
            state.executor_latch.mask_ramp_remaining = snapshot['executor_latch_mask_ramp']
            state.executor_latch.tail_mode = snapshot['executor_latch_tail_mode']
            state.executor_latch.traction_q15 = snapshot['executor_latch_traction']
            state.executor_latch._mask_ramp_total = snapshot.get('executor_latch_mask_ramp_total', 0)
            state.executor_latch._mask_start_u32 = snapshot.get('executor_latch_mask_start', 0xFFFFFFFF)
        return state


# =============================================================================
# PRNG FUNCTIONS
# =============================================================================

def xorshift32(state: int) -> tuple[int, int]:
    """
    xorshift32 PRNG step.

    Args:
        state: Current PRNG state (32-bit)

    Returns:
        (new_state, random_value)
    """
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17)
    x ^= (x << 5) & 0xFFFFFFFF
    return x, x


def prng_step(state: VMState) -> int:
    """
    Advance PRNG and return random value.

    Modifies state.prng_state in place.

    Args:
        state: VM state

    Returns:
        Random 32-bit value
    """
    state.prng_state, value = xorshift32(state.prng_state)
    return value


__all__ = [
    'EnvelopePhase',
    'EnvelopeState',
    'VMState',
    'xorshift32',
    'prng_step',
    # WSE-STD-210 oscillator modes
    'OSC_MODE_A',
    'OSC_MODE_B',
    'OSC_MODE_SAMPLE',
    'OSC_MODE_FM6',
    # CMMM Executor Latch (WSE-STD-070)
    'ExecutorLatch',
    'PAYLOAD_TYPE_NODEMASK_SET',
    'PAYLOAD_TYPE_TAILMODE_SET',
]
