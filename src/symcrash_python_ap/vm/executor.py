"""
VM Executor (WSE-STD-VM-HYBRID v0)
==================================

Main execution entry point for the hybrid VM.
Executes one block for one voice with fixed lane schedule.
"""

from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np

from .state import VMState
from .decode import DecodedPlan, CarrierRole, decode_frame
from .tables import Tables, DEFAULT_TABLES, BLOCK_SAMPLES
from .lanes import (
    lane_carrier_render,
    lane_inject,
    dispatch_coeff_op,
    lane_tail,
    lane_idwpt_render,
    lane_mix_gain,
)

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

    # -------------------------------------------------------------------------
    # Step 1: LANE_CARRIER_RENDER
    # -------------------------------------------------------------------------
    if cfg.carrier_role != CarrierRole.OFF:
        carrier_buf = lane_carrier_render(state, cfg)
    else:
        carrier_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

    # -------------------------------------------------------------------------
    # Step 2: LANE_INJECT (events)
    # -------------------------------------------------------------------------
    lane_inject(state, cfg)

    # -------------------------------------------------------------------------
    # Steps 3-6: Backbone lanes (in order)
    # -------------------------------------------------------------------------
    for instr in cfg.backbone_instrs:
        dispatch_coeff_op(state, instr, cfg.mask_u32)

    # -------------------------------------------------------------------------
    # Step 7: LANE_TAIL
    # -------------------------------------------------------------------------
    lane_tail(state, cfg)

    # -------------------------------------------------------------------------
    # Step 8: LANE_IDWPT_RENDER
    # -------------------------------------------------------------------------
    if cfg.idwpt_enable:
        texture_buf = lane_idwpt_render(state, cfg, carrier_buf)
    else:
        texture_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

    # -------------------------------------------------------------------------
    # Step 9: LANE_MIX_GAIN
    # -------------------------------------------------------------------------
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

    # Lane 1: Carrier
    if cfg.carrier_role != CarrierRole.OFF:
        carrier_buf = lane_carrier_render(state, cfg)
    else:
        carrier_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

    # Lane 2: Inject
    lane_inject(state, cfg)

    # Lanes 3-6: Backbone
    for instr in cfg.backbone_instrs:
        dispatch_coeff_op(state, instr, cfg.mask_u32)

    # Lane 7: Tail
    lane_tail(state, cfg)

    # Lane 8: IDWPT
    if cfg.idwpt_enable and state.wet_ready:
        texture_buf = lane_idwpt_render(state, cfg, carrier_buf)
    else:
        texture_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

    # Lane 9: Mix + Gain
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
