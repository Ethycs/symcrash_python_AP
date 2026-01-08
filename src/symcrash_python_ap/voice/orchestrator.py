"""
Block Orchestrator
==================

Connects BlockScheduler to the rendering pipeline.

The orchestrator is the execution engine that:
1. Takes scheduling decisions from BlockScheduler
2. Applies emission group flags (TOP/MID/BASE/TAIL) to control operations
3. Manages voice lifecycle (allocate/on/free) when scheduler reassigns voices
4. Produces audio blocks using only serviced voices

Architecture Note:
- Scheduler runs on Pico (decides what to do)
- Orchestrator is this module (Python reference for FPGA execution)
- CTRL-DELTA (MaskedUpdate) is the wire format between them
- VoiceFramePacket is internal/debug format, not sent per-block

Per WSE-STD-070:
- voice_allocate: New owner of voice slot, arms microfade
- voice_on: Retrigger/gate rise, NEVER resets state
- voice_free: Mark voice as available
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import IntEnum

from ..frames import VoiceFrame, VoiceFramePacketV0b, VoiceFlags, ParamFrame
from .scheduler import (
    BlockScheduler, ScheduleResult, VoiceScheduleInfo, EmissionGroup,
    HealthTelemetry, BLOCK_SIZE
)
from ..wavelet import WaveletEngine
from ..symbol import SymbolEngine, EmissionFlags
from ..budgets import (
    apply_degradation, DegradationResult, RegimeBudget, TailMode,
    get_regime_budget, RegimeId, popcount32,
)
from ..fixed import Q15_ONE, sat16, mul_q15_sat


# =============================================================================
# VOICE ALLOCATION STATE
# =============================================================================

class VoiceState(IntEnum):
    """Voice slot state for lifecycle tracking."""
    FREE = 0       # Available for allocation
    ALLOCATED = 1  # Allocated but not yet playing
    ACTIVE = 2     # Playing (gate high or activity > threshold)
    RELEASING = 3  # Gate fell, activity decaying


@dataclass
class VoiceSlot:
    """State for a voice slot."""
    state: VoiceState = VoiceState.FREE
    owner_id: int = 0        # Logical note/sound owner
    note_u8: int = 60        # Current MIDI note
    last_activity_q15: int = 0  # For eligibility checks
    blocks_since_gate: int = 0  # Blocks since last gate event


# =============================================================================
# BLOCK ORCHESTRATOR
# =============================================================================

@dataclass
class BlockOrchestrator:
    """
    Orchestrates block rendering using scheduler decisions.

    This is the execution layer that:
    1. Receives ScheduleResult from BlockScheduler
    2. Manages voice allocation lifecycle
    3. Applies emission group flags to control Symbol Engine operations
    4. Renders only serviced voices through Wavelet Engine
    5. Returns mixed audio and updated telemetry
    """

    # Configuration
    n_voices: int = 128
    block_size: int = BLOCK_SIZE

    # Engines
    wavelet: WaveletEngine = field(default_factory=lambda: WaveletEngine(BLOCK_SIZE))
    symbol: Optional[SymbolEngine] = None

    # Scheduler
    scheduler: BlockScheduler = field(default_factory=BlockScheduler)

    # Voice slot state
    voice_slots: List[VoiceSlot] = field(default_factory=list)

    # Current block state
    block_count: int = 0

    # Voice frames (populated by Symbol Engine or external input)
    voice_frames: List[Optional[VoiceFramePacketV0b]] = field(default_factory=list)

    # Budget for degradation (can be set from regime)
    current_budget: Optional[RegimeBudget] = None

    # Default backbone operators (for voices without explicit op list)
    default_backbone_ops: List[int] = field(default_factory=lambda: [1, 3, 4, 5])  # Tilt, IntraMix, DiffuseUp/Down

    # Degradation stats (accumulated per block for telemetry)
    degradation_steps_total: int = 0
    emergency_voices: int = 0

    def __post_init__(self):
        """Initialize voice slots and frame storage."""
        if not self.voice_slots:
            self.voice_slots = [VoiceSlot() for _ in range(self.n_voices)]
        if not self.voice_frames:
            self.voice_frames = [None] * self.n_voices

    # -------------------------------------------------------------------------
    # Voice Lifecycle
    # -------------------------------------------------------------------------

    def voice_allocate(self, voice_id: int, owner_id: int, note_u8: int = 60) -> bool:
        """
        Allocate voice slot to new logical owner.

        Per WSE-STD-070 Section 4:
        - Arms microfade to prevent clicks
        - Clears tree state for fresh start
        - Does NOT clear prev_sample (used for microfade)

        Args:
            voice_id: Voice slot index
            owner_id: Logical owner (e.g., note number, sound ID)
            note_u8: MIDI note number

        Returns:
            True if allocation succeeded
        """
        if voice_id < 0 or voice_id >= self.n_voices:
            return False

        slot = self.voice_slots[voice_id]

        # Get last sample for microfade continuity
        last_sample = self.wavelet.get_last_sample(voice_id)

        # Allocate in WaveletEngine (arms microfade)
        self.wavelet.voice_allocate(voice_id, last_sample)

        # Update slot state
        slot.state = VoiceState.ALLOCATED
        slot.owner_id = owner_id
        slot.note_u8 = note_u8
        slot.blocks_since_gate = 0

        return True

    def voice_on(self, voice_id: int) -> bool:
        """
        Signal voice retrigger / gate rise.

        Per WSE-STD-070: voice_on is a RETRIGGER, not a new allocation.
        - MUST NOT reset prev_sample or arm microfade
        - Only transitions state to ACTIVE

        Args:
            voice_id: Voice slot index

        Returns:
            True if voice was in valid state for voice_on
        """
        if voice_id < 0 or voice_id >= self.n_voices:
            return False

        slot = self.voice_slots[voice_id]

        # Can only voice_on if allocated or already active
        if slot.state == VoiceState.FREE:
            return False

        # Signal to WaveletEngine (does NOT reset)
        self.wavelet.voice_on(voice_id)

        # Transition to active
        slot.state = VoiceState.ACTIVE
        slot.blocks_since_gate = 0

        return True

    def voice_free(self, voice_id: int) -> bool:
        """
        Mark voice as free / available for reallocation.

        Per WSE-STD-070:
        - Keeps prev_sample for next allocate's microfade
        - Slot becomes available for scheduler

        Args:
            voice_id: Voice slot index

        Returns:
            True if voice was successfully freed
        """
        if voice_id < 0 or voice_id >= self.n_voices:
            return False

        slot = self.voice_slots[voice_id]

        # Signal to WaveletEngine (preserves prev_sample)
        self.wavelet.voice_free(voice_id)

        # Update slot state
        slot.state = VoiceState.FREE
        slot.owner_id = 0

        return True

    def get_free_voice(self) -> Optional[int]:
        """
        Find a free voice slot.

        Returns voice_id if found, None if all slots are occupied.
        """
        for vid, slot in enumerate(self.voice_slots):
            if slot.state == VoiceState.FREE:
                return vid
        return None

    def set_regime_budget(self, budget: RegimeBudget) -> None:
        """
        Set the current regime budget for degradation.

        Args:
            budget: RegimeBudget from regime or capsule
        """
        self.current_budget = budget

    def _degrade_for_voice(
        self,
        frame: VoiceFramePacketV0b,
        backbone_ops: Optional[List[int]] = None,
        event_ops: Optional[List[int]] = None,
    ) -> DegradationResult:
        """
        Apply degradation ladder to fit voice within budget.

        Args:
            frame: VoiceFramePacket with node_mask, etc.
            backbone_ops: Override backbone operators (default: default_backbone_ops)
            event_ops: Event operators pending for this voice

        Returns:
            DegradationResult with fitted ops, K, tail_mode
        """
        # Use defaults if not provided
        if backbone_ops is None:
            backbone_ops = self.default_backbone_ops
        if event_ops is None:
            event_ops = []

        # Get budget (fall back to STABLE_TONE if not set)
        budget = self.current_budget
        if budget is None:
            budget = get_regime_budget(RegimeId.STABLE_TONE)

        # Get node mask from frame (full mask if not available)
        node_mask = getattr(frame, 'node_mask', 0xFFFFFFFF)
        if node_mask == 0:
            node_mask = 0xFFFFFFFF  # Default to all nodes

        return apply_degradation(
            backbone_ops=backbone_ops,
            event_ops=event_ops,
            node_mask=node_mask,
            tail_mode=TailMode.FULL,
            budget=budget,
        )

    # -------------------------------------------------------------------------
    # Frame Management
    # -------------------------------------------------------------------------

    def update_frame(self, voice_id: int, frame: VoiceFramePacketV0b) -> None:
        """
        Update voice frame from external source (e.g., CTRL-DELTA).

        This is how parameter updates arrive from the wire protocol.
        """
        if 0 <= voice_id < self.n_voices:
            self.voice_frames[voice_id] = frame

            # Update slot activity for scheduling
            slot = self.voice_slots[voice_id]
            slot.last_activity_q15 = frame.activity_q15

            # Handle gate events
            if frame.flags & VoiceFlags.GATE_RISE:
                if slot.state == VoiceState.FREE:
                    # Auto-allocate on gate rise
                    self.voice_allocate(voice_id, voice_id, frame.note_u8)
                self.voice_on(voice_id)

            elif frame.flags & VoiceFlags.GATE_FALL:
                slot.state = VoiceState.RELEASING

    # -------------------------------------------------------------------------
    # Block Rendering
    # -------------------------------------------------------------------------

    def step_block(
        self,
        params: Optional[ParamFrame] = None,
        tail_stats: Optional[dict] = None,
    ) -> Tuple[List[int], List[int], ScheduleResult]:
        """
        Process one block through scheduled rendering.

        Args:
            params: Parameter frame for Symbol Engine
            tail_stats: Previous block's tail statistics (for D update)

        Returns:
            (L_audio, R_audio, schedule_result)
        """
        # 0. Reset per-block degradation stats
        self.degradation_steps_total = 0
        self.emergency_voices = 0

        # 1. Get scheduling decisions
        schedule = self.scheduler.schedule_block(self.voice_frames, tail_stats)

        # 2. Render only serviced voices
        voice_blocks: List[Tuple[VoiceFrame, List[int]]] = []

        for sched_info in schedule.serviced:
            vid = sched_info.voice_id
            frame = self.voice_frames[vid]

            if frame is None:
                continue

            # Apply degradation ladder to fit voice within budget
            degrade_result = self._degrade_for_voice(frame)

            # Track degradation stats for telemetry
            self.degradation_steps_total += degrade_result.degradation_steps
            if degrade_result.is_emergency:
                self.emergency_voices += 1

            # Convert to internal VoiceFrame for rendering
            vf = frame.to_voice_frame()

            # Apply emission group filtering to Symbol Engine
            # TOP/MID flags from scheduler gate which operators run
            if self.symbol is not None and params is not None:
                emission_flags = EmissionFlags(
                    mid_due=sched_info.mid_due,
                    top_due=sched_info.top_due,
                )
                # Symbol engine step_voice would use:
                # - emission_flags to gate MID/TOP operator groups
                # - degrade_result.K for active node count
                # - degrade_result.get_full_op_chain() for fitted operator list
                # - degrade_result.tail_mode for tail configuration
                #
                # NOTE: Full wiring requires SymbolEngine.step_voice() to accept
                # these parameters. For now, store on vf for potential use.
                vf.emission_flags = emission_flags
                vf.degrade_K = degrade_result.K
                vf.degrade_tail_mode = degrade_result.tail_mode

            # Render through Wavelet Engine
            audio = self.wavelet.render_voice(vf)
            voice_blocks.append((vf, audio))

        # 3. Mix to stereo
        if voice_blocks:
            L, R = self.wavelet.mix_block(voice_blocks)
        else:
            L = [0] * self.block_size
            R = [0] * self.block_size

        # 4. Update block counter and voice states
        self.block_count += 1
        self._update_voice_states(schedule.serviced_ids)

        # 5. End block (update telemetry for next block)
        self.scheduler.end_block(tail_stats)

        return L, R, schedule

    def _update_voice_states(self, serviced_ids: List[int]) -> None:
        """Update voice slot states after block processing."""
        for vid, slot in enumerate(self.voice_slots):
            if slot.state != VoiceState.FREE:
                slot.blocks_since_gate += 1

            # Auto-free voices that have decayed to silence
            if slot.state == VoiceState.RELEASING:
                frame = self.voice_frames[vid]
                if frame is None or frame.activity_q15 == 0:
                    self.voice_free(vid)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_active_voice_count(self) -> int:
        """Count currently active voices."""
        return sum(1 for slot in self.voice_slots
                   if slot.state in (VoiceState.ACTIVE, VoiceState.RELEASING))

    def get_voice_summary(self) -> Dict[str, int]:
        """Get voice state summary."""
        counts = {state.name: 0 for state in VoiceState}
        for slot in self.voice_slots:
            counts[slot.state.name] += 1
        return counts

    def get_degradation_stats(self) -> Dict[str, int]:
        """
        Get degradation statistics from last block.

        Returns:
            Dict with 'degradation_steps_total' and 'emergency_voices'
        """
        return {
            'degradation_steps_total': self.degradation_steps_total,
            'emergency_voices': self.emergency_voices,
        }


# =============================================================================
# ORCHESTRATED PIPELINE
# =============================================================================

@dataclass
class OrchestratedPipeline:
    """
    Complete pipeline with scheduling and orchestration.

    This replaces the simple Pipeline class for production use.
    """
    orchestrator: BlockOrchestrator = field(default_factory=BlockOrchestrator)
    params: ParamFrame = field(default_factory=ParamFrame)

    # Output accumulator
    L_out: List[int] = field(default_factory=list)
    R_out: List[int] = field(default_factory=list)

    # Tail statistics (accumulated per block)
    tail_stats: Dict = field(default_factory=dict)

    def step(self) -> Tuple[List[int], List[int], ScheduleResult]:
        """Process one block."""
        L, R, schedule = self.orchestrator.step_block(self.params, self.tail_stats)

        self.L_out.extend(L)
        self.R_out.extend(R)

        return L, R, schedule

    def run(self, n_blocks: int) -> Tuple[List[int], List[int]]:
        """Run for n_blocks."""
        for _ in range(n_blocks):
            self.step()
        return self.L_out, self.R_out


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'VoiceState',
    'VoiceSlot',
    'BlockOrchestrator',
    'OrchestratedPipeline',
]
