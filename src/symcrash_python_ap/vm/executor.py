"""
VM Executor (WSE-STD-VM-HYBRID v0)
==================================

Main execution entry point for the hybrid VM.
Executes one block for one voice with fixed lane schedule.
"""

from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np

from .state import VMState, OSC_MODE_A, OSC_MODE_B
from .decode import DecodedPlan, CarrierRole, decode_frame
from .tables import Tables, DEFAULT_TABLES, BLOCK_SAMPLES
from .lanes import (
    lane_carrier_render,
    lane_inject,
    dispatch_coeff_op,
    lane_tail,
    lane_idwpt_render,
    lane_ccs_render,
    lane_mix_gain,
)
from ..fixed import RSHIFT_RNE

if TYPE_CHECKING:
    from ..frames import VoiceFrame


# =============================================================================
# LATENCY CONSTANTS
# =============================================================================

IDWPT_LATENCY_BLOCKS = 5  # Blocks until wet signal is ready


# =============================================================================
# MAIN EXECUTOR
# =============================================================================

def vm_step_block(
    state: VMState,
    frame: 'VoiceFrame',
    tables: Tables = DEFAULT_TABLES,
) -> Tuple[VMState, np.ndarray]:
    """
    Execute one block for one voice.

    This is the main entry point for the hybrid VM. It executes the
    fixed lane schedule deterministically.

    Args:
        state: Per-voice VM state (modified in place)
        frame: VoiceFrame microcode for this block
        tables: Immutable tables for lookups

    Returns:
        (state, output_block) where:
            state: Updated VM state
            output_block: int16 array of BLOCK_SAMPLES
    """
    # -------------------------------------------------------------------------
    # Step 0: Decode configuration
    # -------------------------------------------------------------------------
    cfg = decode_frame(
        frame,
        tables,
        prev_eta_q15=state.eta_prev_q15,
        prev_gain_q15=state.gain_prev_q15,
    )

    # Track blocks since start (for latency model)
    state.blocks_since_start += 1

    # Check if wet is ready (microcode contract)
    if state.blocks_since_start >= IDWPT_LATENCY_BLOCKS:
        state.wet_ready = True
    else:
        # Force eta to 0 (pure carrier) during latency period
        cfg.eta_curr_q15 = 0
        cfg.idwpt_enable = False

    # Update tree state
    state.tree_a_id = cfg.tree_a
    state.tree_b_id = cfg.tree_b

    # =========================================================================
    # CCS PATH (Coefficient Synthesis) - WSE-STD-ARCH-CCS COMPLIANT
    # =========================================================================
    # When use_coeff_osc is True, we use the TRUE CCS path per spec section 2.1:
    #   Phase → CoeffTable → Coefficients → Operators → IDWPT → Audio
    #
    # This eliminates FWPT from the voice path and ensures operators modify
    # the carrier coefficients before IDWPT reconstruction.

    if cfg.use_coeff_osc and state.use_coeff_osc:
        # -----------------------------------------------------------------
        # CCS PATH: Coefficient-domain architecture (WSE-STD-121 COMPLIANT)
        # -----------------------------------------------------------------
        # Uses StreamingCoeffLUT with per-note tables that have frequency
        # baked into the FWPT coefficients. No FWPT in runtime voice path.
        #
        # WSE-STD-210: Mode A (LUT) vs Mode B (Rotator/Harmonic Bank)
        # - Mode A: Phase → LUT lookup → Coefficients (existing)
        # - Mode B: Phase → sin/cos weights × basis frames → Coefficients

        from .tables import COEFF_LEN

        # Get oscillator params
        params = cfg.coeff_osc_params
        osc_mode = params.osc_mode

        # Update state if pitch changed
        pitch_changed = state.last_note != params.note or state.last_cents != params.cents
        if pitch_changed:
            state.ccs_state.note = params.note
            state.ccs_state.wave_id = params.table_id
            state.last_note = params.note
            state.last_cents = params.cents

        # Update Mode B state if using rotator (always check for first-time init)
        if osc_mode == OSC_MODE_B:
            mb_state = state.mode_b_state
            if pitch_changed or mb_state.note != params.note:
                mb_state.reset(note=params.note, phase_inc=params.phase_inc)
                mb_state.pwm_width_q15 = params.pwm_width_q15
                mb_state.pwm_enabled = params.pwm_enabled
                mb_state.H_active = min(params.H_request, mb_state.H_bandlimit)

            # Update FM depth from VoiceFrame (modulator signal comes from fm_mod_buffer)
            # Phase 3 FM/PM will provide the modulator via operator graph
            mb_state.fm_depth_q15 = params.fm_depth_q15

        if state.ccs_state.phase_inc != params.phase_inc:
            state.ccs_state.set_pitch(params.phase_inc, slew_blocks=8)
            if osc_mode == OSC_MODE_B:
                state.mode_b_state.phase_inc = params.phase_inc

        # Generate polyphase carrier: 64 leaves × K k-slots = 512 values (K=8)
        carrier_512 = np.zeros(COEFF_LEN, dtype=np.int16)

        if osc_mode == OSC_MODE_B:
            # -----------------------------------------------------------------
            # MODE B: Rotator/Harmonic Bank (WSE-STD-210 Section 10)
            # -----------------------------------------------------------------
            from .harmonic_bank import mode_b_generate_coeffs, get_basis_for_note

            # Get or generate basis frames for this pitch anchor
            basis = get_basis_for_note(params.note)

            # Update PWM if enabled
            if params.pwm_enabled:
                state.mode_b_state.set_pwm_weights(params.pwm_width_q15)

            # Generate coefficients using harmonic bank
            mode_b_generate_coeffs(state.mode_b_state, basis, carrier_512)

            # Advance Mode B phase for next block
            state.mode_b_state.advance_block()
        else:
            # -----------------------------------------------------------------
            # MODE A: LUT-based (existing path)
            # -----------------------------------------------------------------
            from .coeff_lut_streaming import ccs_generate_coeffs_polyphase
            ccs_generate_coeffs_polyphase(state.ccs_state, carrier_512)

        # PERSISTENT EVOLUTION per WSE-STD-120 section 4.1:
        # Coefficients persist and evolve, carrier provides stabilization

        base_mix = params.base_mix_q15

        # POLYPHASE INJECTION: Each k-slot gets its own carrier value
        # This is the key to eliminating block-boundary discontinuities
        for idx in range(COEFF_LEN):
            carrier_val = int(carrier_512[idx])
            old_val = int(state.coeff_state[idx])

            # Blend: new = old * (1 - base_mix) + carrier * base_mix
            # base_mix = 32767 → pure carrier (full oscillation)
            # base_mix = 16384 → 50/50 blend
            # base_mix = 0 → pure persistence
            inv_mix = 32767 - base_mix
            blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)

            state.coeff_state[idx] = np.clip(blended, -32768, 32767)

        # Step 1: LANE_INJECT (events modify carrier-seeded coeff_state)
        lane_inject(state, cfg)

        # Steps 2-5: Backbone lanes (operators modify carrier)
        for instr in cfg.backbone_instrs:
            dispatch_coeff_op(state, instr, cfg.mask_u32)

        # Step 6: LANE_TAIL (safety constraints)
        lane_tail(state, cfg)

        # Step 7: LANE_CCS_RENDER (now just extracts and calls IDWPT)
        output = lane_ccs_render(state, cfg)

        # Apply gain with proper rounding (WSE-STD-001 compliant)
        gain_curr = cfg.gain_curr_q15
        gain_prev = cfg.gain_prev_q15
        if gain_curr != 32767 or gain_prev != 32767:
            # Interpolate gain across block using round-to-nearest
            for i in range(BLOCK_SAMPLES):
                # Round-to-nearest for alpha: (i * 32767 + 32) / 64
                alpha = (i * 32767 + (BLOCK_SAMPLES >> 1)) // BLOCK_SAMPLES
                gain = gain_prev + RSHIFT_RNE((gain_curr - gain_prev) * alpha, 15)
                output[i] = RSHIFT_RNE(int(output[i]) * gain, 15)

        # Advance CCS phase for next block (Mode A only - Mode B advances in generate)
        if osc_mode == OSC_MODE_A:
            state.ccs_state.advance_block()

    else:
        # -----------------------------------------------------------------
        # LEGACY PATH: Old architecture (has combing when wet enabled)
        # -----------------------------------------------------------------

        # Step 1: LANE_CARRIER_RENDER
        if cfg.carrier_role != CarrierRole.OFF:
            carrier_buf = lane_carrier_render(state, cfg)
        else:
            carrier_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        # Step 2: LANE_INJECT (events)
        lane_inject(state, cfg)

        # Steps 3-6: Backbone lanes (in order)
        for instr in cfg.backbone_instrs:
            dispatch_coeff_op(state, instr, cfg.mask_u32)

        # Step 7: LANE_TAIL
        lane_tail(state, cfg)

        # Step 8: LANE_IDWPT_RENDER
        if cfg.idwpt_enable:
            texture_buf = lane_idwpt_render(state, cfg, carrier_buf)
        else:
            texture_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        # Step 9: LANE_MIX_GAIN
        output = lane_mix_gain(cfg, carrier_buf, texture_buf)

    # -------------------------------------------------------------------------
    # Update state for next block
    # -------------------------------------------------------------------------
    state.eta_prev_q15 = cfg.eta_curr_q15
    state.gain_prev_q15 = cfg.gain_curr_q15

    return state, output


def vm_step_block_raw(
    state: VMState,
    cfg: DecodedPlan,
) -> Tuple[VMState, np.ndarray]:
    """
    Execute one block with a pre-decoded plan.

    Lower-level API for when you want to bypass decode_frame().

    Args:
        state: Per-voice VM state (modified in place)
        cfg: Pre-decoded execution plan

    Returns:
        (state, output_block)
    """
    # Track blocks
    state.blocks_since_start += 1
    if state.blocks_since_start >= IDWPT_LATENCY_BLOCKS:
        state.wet_ready = True

    if cfg.use_coeff_osc and state.use_coeff_osc:
        # CCS PATH: WSE-STD-121 compliant with TRUE POLYPHASE INJECTION
        from .coeff_lut_streaming import ccs_generate_coeffs_polyphase
        from .tables import COEFF_LEN

        # Generate polyphase carrier: 64 leaves × K k-slots = 512 values (K=8)
        carrier_512 = np.zeros(COEFF_LEN, dtype=np.int16)
        ccs_generate_coeffs_polyphase(state.ccs_state, carrier_512)

        # Inject with base_mix blending (polyphase: each k-slot has own value)
        params = cfg.coeff_osc_params
        base_mix = params.base_mix_q15

        for idx in range(COEFF_LEN):
            carrier_val = int(carrier_512[idx])
            old_val = int(state.coeff_state[idx])
            inv_mix = 32767 - base_mix
            blended = RSHIFT_RNE(old_val * inv_mix + carrier_val * base_mix, 15)
            state.coeff_state[idx] = np.clip(blended, -32768, 32767)

        lane_inject(state, cfg)

        for instr in cfg.backbone_instrs:
            dispatch_coeff_op(state, instr, cfg.mask_u32)

        lane_tail(state, cfg)

        output = lane_ccs_render(state, cfg)

        # Apply gain with proper rounding (WSE-STD-001 compliant)
        gain_curr = cfg.gain_curr_q15
        gain_prev = cfg.gain_prev_q15
        if gain_curr != 32767 or gain_prev != 32767:
            for i in range(BLOCK_SAMPLES):
                alpha = (i * 32767 + (BLOCK_SAMPLES >> 1)) // BLOCK_SAMPLES
                gain = gain_prev + RSHIFT_RNE((gain_curr - gain_prev) * alpha, 15)
                output[i] = RSHIFT_RNE(int(output[i]) * gain, 15)

        # Advance CCS phase for next block
        state.ccs_state.advance_block()
    else:
        # LEGACY PATH
        if cfg.carrier_role != CarrierRole.OFF:
            carrier_buf = lane_carrier_render(state, cfg)
        else:
            carrier_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        lane_inject(state, cfg)

        for instr in cfg.backbone_instrs:
            dispatch_coeff_op(state, instr, cfg.mask_u32)

        lane_tail(state, cfg)

        if cfg.idwpt_enable and state.wet_ready:
            texture_buf = lane_idwpt_render(state, cfg, carrier_buf)
        else:
            texture_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

        output = lane_mix_gain(cfg, carrier_buf, texture_buf)

    # Update state
    state.eta_prev_q15 = cfg.eta_curr_q15
    state.gain_prev_q15 = cfg.gain_curr_q15

    return state, output


# =============================================================================
# MULTI-VOICE EXECUTOR
# =============================================================================

class VMExecutor:
    """
    Multi-voice VM executor.

    Manages state for multiple voices and provides batch execution.
    """

    def __init__(self, max_voices: int = 128, tables: Optional[Tables] = None):
        """
        Initialize VM executor.

        Args:
            max_voices: Maximum number of simultaneous voices
            tables: Tables instance (default: DEFAULT_TABLES)
        """
        self.max_voices = max_voices
        self.tables = tables or DEFAULT_TABLES
        self.states: dict[int, VMState] = {}

    def get_or_create_state(self, voice_id: int) -> VMState:
        """Get existing state or create new one for a voice."""
        if voice_id not in self.states:
            self.states[voice_id] = VMState.create(voice_id)
        return self.states[voice_id]

    def step_voice(
        self,
        voice_id: int,
        frame: 'VoiceFrame',
    ) -> np.ndarray:
        """
        Execute one block for a voice.

        Args:
            voice_id: Voice slot ID
            frame: VoiceFrame microcode

        Returns:
            output_block: int16 array of BLOCK_SAMPLES
        """
        state = self.get_or_create_state(voice_id)
        state, output = vm_step_block(state, frame, self.tables)
        self.states[voice_id] = state
        return output

    def release_voice(self, voice_id: int) -> None:
        """Release a voice slot."""
        if voice_id in self.states:
            del self.states[voice_id]

    def reset_voice(self, voice_id: int) -> None:
        """Reset a voice to initial state."""
        if voice_id in self.states:
            self.states[voice_id] = VMState.create(voice_id)

    def active_voices(self) -> list[int]:
        """Get list of active voice IDs."""
        return [vid for vid, state in self.states.items() if state.is_active()]

    def snapshot_all(self) -> dict[int, dict]:
        """Snapshot all voice states for deterministic replay."""
        return {vid: state.snapshot() for vid, state in self.states.items()}

    def restore_all(self, snapshots: dict[int, dict]) -> None:
        """Restore all voice states from snapshots."""
        self.states = {vid: VMState.restore(snap) for vid, snap in snapshots.items()}


__all__ = [
    'vm_step_block',
    'vm_step_block_raw',
    'VMExecutor',
    'IDWPT_LATENCY_BLOCKS',
]
