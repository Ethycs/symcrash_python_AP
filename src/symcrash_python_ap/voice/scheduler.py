"""
Block-Level Scheduler
=====================

Implements Scheduler Spec v01 for OrangeCrab (ECP5).

Goal: Each audio block, run the highest effective Tower level (L5->L0) the FPGA
can afford without clicks, without divergence, and with tail always enforced.

Key Components:
- VoiceSelector: Ring cursor voice selection with eligibility
- TileScheduler: Tile-based TOP/MID throttling with counters
- DegradeController: Global degrade state D with hysteresis
- HealthTelemetry: CPU/mem/stab bars and gear
- BlockScheduler: Main entry point combining all components
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from enum import IntEnum

from ..frames import VoiceFramePacketV0b, VoiceFlags


# =============================================================================
# CONSTANTS (from Scheduler Spec v01)
# =============================================================================

BLOCK_SIZE = 64  # Samples per block
V_MAX = 128      # Maximum voices
TILE_SIZE = 16   # Voices per tile (power of 2)
NUM_TILES = V_MAX // TILE_SIZE  # 8 tiles

# Q15 fixed-point constants
Q15_ONE = 32767
Q15_SCALE = 32768

# Eligibility threshold: activity > 0.05 in Q15
THETA_ACTIVE_Q15 = int(0.05 * Q15_SCALE)  # ~1638

# Stability threshold for TOP suppression
THETA_STAB_RED_Q15 = int(0.3 * Q15_SCALE)  # ~9830

# Hysteresis: blocks before D can decrease
H_DOWN = 4

# Degrade thresholds in Q15
# D=0 if p < 0.70, D=1 if 0.70 <= p < 0.85, etc.
DEGRADE_THRESHOLDS_Q15 = [
    int(0.70 * Q15_SCALE),  # 22938 - boundary between D=0 and D=1
    int(0.85 * Q15_SCALE),  # 27853 - boundary between D=1 and D=2
    int(1.00 * Q15_SCALE),  # 32768 - boundary between D=2 and D=3
    int(1.15 * Q15_SCALE),  # 37683 - boundary between D=3 and D=4
]

# Degrade -> period table (K_top, K_mid)
PERIOD_TABLE = {
    0: (1, 1),   # D=0: no throttling
    1: (2, 1),   # D=1: TOP every 2 blocks
    2: (4, 2),   # D=2: TOP every 4, MID every 2
    3: (8, 4),   # D=3: TOP every 8, MID every 4
    4: (16, 8),  # D=4: TOP every 16 (or suppressed), MID every 8
}

# Tail cost per voice (in cost units)
# FineClamp(2) + NormalizeScale(11) + SheafProject(23) = ~3N each = 9N
# Add margin for GluePair and EmergencySimplify
C_TAIL_VOICE = 12  # Cost units per voice for mandatory tail


# =============================================================================
# EMISSION GROUPS
# =============================================================================

class EmissionGroup(IntEnum):
    """Tower level emission groups."""
    TAIL = 0   # Phase C: Always runs for serviced voices
    BASE = 1   # L0-L1: Minimal injections for continuity
    MID = 2    # L2-L3: BANDED + Diffuse, LOWRANK, richer events
    TOP = 3    # L4-L5: Chaos overlays, catastrophe corridor, dense patterning


# =============================================================================
# HEALTH TELEMETRY
# =============================================================================

@dataclass
class HealthTelemetry:
    """
    Health telemetry output per block.

    These are observational outputs and MUST NOT affect decisions
    in the same block except via D computed at block start.
    """
    cpu_bar: int = 0      # Q15: used_compute / B_compute
    mem_bar: int = 0      # Q15: used_mem / B_mem
    stab_bar: int = Q15_ONE  # Q15: derived from tail metrics (starts healthy)
    gear: int = 5         # 5 - D (higher = more headroom)

    # Tail stability counters (aggregated from serviced voices)
    clamp_hits: int = 0
    norm_gain_extreme: int = 0
    emergency_fired: int = 0

    # Service stats
    voices_serviced: int = 0
    voices_eligible: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'cpu_bar': self.cpu_bar,
            'mem_bar': self.mem_bar,
            'stab_bar': self.stab_bar,
            'gear': self.gear,
            'clamp_hits': self.clamp_hits,
            'norm_gain_extreme': self.norm_gain_extreme,
            'emergency_fired': self.emergency_fired,
            'voices_serviced': self.voices_serviced,
            'voices_eligible': self.voices_eligible,
        }


# =============================================================================
# VOICE SELECTOR (Ring Scheduler)
# =============================================================================

@dataclass
class VoiceSelector:
    """
    Ring cursor voice selection per Scheduler Spec v01 Section 6.

    Maintains cursor_fg over [0..V_MAX-1] and selects eligible foreground
    voices up to the budget-determined V_serviced limit.
    """
    cursor_fg: int = 0  # Ring cursor position
    v_max: int = V_MAX  # Total voice count

    def is_eligible(self, frame: VoiceFramePacketV0b) -> bool:
        """
        Check if a voice is eligible foreground.

        Eligible if:
        - reject_code == 0 AND
        - (flags.gate == 1 OR activity_q15 > theta_active)
        """
        if frame.reject_code != 0:
            return False

        gate_active = (frame.flags & VoiceFlags.GATE) != 0
        activity_above_threshold = frame.activity_q15 > THETA_ACTIVE_Q15

        return gate_active or activity_above_threshold

    def compute_v_serviced(self, b_compute: int, b_mem: int) -> int:
        """
        Compute maximum V_serviced based on tail reservation.

        V_serviced * C_TAIL_VOICE <= min(b_compute, b_mem)
        """
        budget = min(b_compute, b_mem)
        if C_TAIL_VOICE <= 0:
            return self.v_max
        return min(budget // C_TAIL_VOICE, self.v_max)

    def select_voices(
        self,
        frames: List[Optional[VoiceFramePacketV0b]],
        v_serviced_max: int,
    ) -> Tuple[List[int], int]:
        """
        Select eligible foreground voices using ring scheduler.

        Args:
            frames: List of VoiceFramePacketV0b (or None for absent voices)
            v_serviced_max: Maximum voices to service this block

        Returns:
            Tuple of (serviced_indices, eligible_count)
        """
        serviced: List[int] = []
        eligible_count = 0

        # Scan from cursor, wrapping around
        scanned = 0
        pos = self.cursor_fg

        while scanned < self.v_max and len(serviced) < v_serviced_max:
            frame = frames[pos] if pos < len(frames) else None

            if frame is not None and self.is_eligible(frame):
                eligible_count += 1
                serviced.append(pos)

            scanned += 1
            pos = (pos + 1) % self.v_max

        # Count remaining eligible voices we didn't get to
        while scanned < self.v_max:
            frame = frames[pos] if pos < len(frames) else None
            if frame is not None and self.is_eligible(frame):
                eligible_count += 1
            scanned += 1
            pos = (pos + 1) % self.v_max

        # Update cursor to next position after last scanned
        self.cursor_fg = pos

        return serviced, eligible_count


# =============================================================================
# TILE SCHEDULER (TOP/MID Throttling)
# =============================================================================

def tile_id(v: int) -> int:
    """Get tile ID for voice v."""
    return v >> 4  # v // TILE_SIZE (TILE_SIZE = 16 = 2^4)


def tile_off(v: int) -> int:
    """Get offset within tile for voice v."""
    return v & (TILE_SIZE - 1)  # v % TILE_SIZE


@dataclass
class TileScheduler:
    """
    Tile-based TOP/MID throttling per Scheduler Spec v01 Section 5.

    Maintains per-tile counters for MID and TOP emission groups.
    Counters only advance when at least one voice in the tile is serviced.
    """
    num_tiles: int = NUM_TILES
    ctr_mid: List[int] = field(default_factory=lambda: [0] * NUM_TILES)
    ctr_top: List[int] = field(default_factory=lambda: [0] * NUM_TILES)

    def get_periods(self, D: int) -> Tuple[int, int]:
        """Get (K_top, K_mid) periods for degrade state D."""
        return PERIOD_TABLE.get(D, (16, 8))

    def phase_mid(self, v: int, K_mid: int) -> int:
        """Compute MID phase for voice v."""
        return tile_off(v) & (K_mid - 1)

    def phase_top(self, v: int, K_top: int) -> int:
        """Compute TOP phase for voice v."""
        return tile_off(v) & (K_top - 1)

    def is_mid_due(self, v: int, D: int) -> bool:
        """
        Check if MID emission is due for voice v.

        MID_due(v) = ((ctr_mid[t] + phase_mid(v)) & (K_mid-1)) == 0
        """
        K_top, K_mid = self.get_periods(D)
        t = tile_id(v)
        phase = self.phase_mid(v, K_mid)
        return ((self.ctr_mid[t] + phase) & (K_mid - 1)) == 0

    def is_top_due(self, v: int, D: int, top_suppressed: bool = False) -> bool:
        """
        Check if TOP emission is due for voice v.

        TOP_due(v) = ((ctr_top[t] + phase_top(v)) & (K_top-1)) == 0

        Returns False if TOP is suppressed (D=4 and stab_bar < threshold).
        """
        if top_suppressed:
            return False

        K_top, K_mid = self.get_periods(D)
        t = tile_id(v)
        phase = self.phase_top(v, K_top)
        return ((self.ctr_top[t] + phase) & (K_top - 1)) == 0

    def compute_due_flags(
        self,
        serviced: List[int],
        D: int,
        top_suppressed: bool = False,
    ) -> List[Tuple[int, bool, bool]]:
        """
        Compute due flags for all serviced voices.

        Returns:
            List of (voice_id, mid_due, top_due) tuples
        """
        result = []
        for v in serviced:
            mid_due = self.is_mid_due(v, D)
            top_due = self.is_top_due(v, D, top_suppressed)
            result.append((v, mid_due, top_due))
        return result

    def advance_counters(self, serviced: List[int], D: int) -> None:
        """
        Advance tile counters for tiles with serviced voices.

        A tile counter advances only if at least one serviced voice
        in that tile exists this block.
        """
        K_top, K_mid = self.get_periods(D)

        # Find tiles with serviced voices
        serviced_tiles: Set[int] = set()
        for v in serviced:
            serviced_tiles.add(tile_id(v))

        # Advance counters for serviced tiles only
        for t in serviced_tiles:
            self.ctr_mid[t] = (self.ctr_mid[t] + 1) % K_mid
            self.ctr_top[t] = (self.ctr_top[t] + 1) % K_top


# =============================================================================
# DEGRADE CONTROLLER (Hysteresis)
# =============================================================================

@dataclass
class DegradeController:
    """
    Global degrade state D with hysteresis per Scheduler Spec v01 Section 4.

    D ranges from 0 (full budget) to 4 (maximum degradation).
    D increases immediately when pressure exceeds threshold.
    D decreases only after H_DOWN consecutive blocks below threshold.
    """
    D: int = 0  # Current degrade state [0..4]
    down_counter: int = 0  # Blocks spent below threshold

    def compute_target_D(self, p_q15: int) -> int:
        """
        Compute target D based on pressure p.

        D=0 if p < 0.70
        D=1 if 0.70 <= p < 0.85
        D=2 if 0.85 <= p < 1.00
        D=3 if 1.00 <= p < 1.15
        D=4 if p >= 1.15
        """
        if p_q15 < DEGRADE_THRESHOLDS_Q15[0]:
            return 0
        elif p_q15 < DEGRADE_THRESHOLDS_Q15[1]:
            return 1
        elif p_q15 < DEGRADE_THRESHOLDS_Q15[2]:
            return 2
        elif p_q15 < DEGRADE_THRESHOLDS_Q15[3]:
            return 3
        else:
            return 4

    def update(self, cpu_bar: int, mem_bar: int) -> int:
        """
        Update degrade state based on previous block telemetry.

        Args:
            cpu_bar: CPU usage bar (Q15)
            mem_bar: Memory usage bar (Q15)

        Returns:
            New degrade state D
        """
        # Combined pressure
        p = max(cpu_bar, mem_bar)
        target_D = self.compute_target_D(p)

        if target_D > self.D:
            # Increase immediately (but at most +1 per block)
            self.D = min(self.D + 1, 4)
            self.down_counter = 0
        elif target_D < self.D:
            # Decrease only after H_DOWN blocks below threshold
            self.down_counter += 1
            if self.down_counter >= H_DOWN:
                self.D = max(self.D - 1, 0)
                self.down_counter = 0
        else:
            # At target, reset down counter
            self.down_counter = 0

        return self.D

    def get_gear(self) -> int:
        """Get gear (5 - D). Higher gear = more headroom."""
        return 5 - self.D


# =============================================================================
# SCHEDULE RESULT
# =============================================================================

@dataclass
class VoiceScheduleInfo:
    """Schedule information for a single voice."""
    voice_id: int
    mid_due: bool
    top_due: bool

    def to_dict(self) -> dict:
        return {
            'voice_id': self.voice_id,
            'mid_due': self.mid_due,
            'top_due': self.top_due,
        }


@dataclass
class ScheduleResult:
    """Result of block scheduling."""
    serviced: List[VoiceScheduleInfo]
    degrade_state: int
    top_suppressed: bool
    telemetry: HealthTelemetry

    @property
    def serviced_ids(self) -> List[int]:
        """Get list of serviced voice IDs."""
        return [s.voice_id for s in self.serviced]

    def to_dict(self) -> dict:
        return {
            'serviced': [s.to_dict() for s in self.serviced],
            'degrade_state': self.degrade_state,
            'top_suppressed': self.top_suppressed,
            'telemetry': self.telemetry.to_dict(),
        }


# =============================================================================
# BLOCK SCHEDULER (Main Entry Point)
# =============================================================================

@dataclass
class BlockScheduler:
    """
    Main block scheduler combining all components.

    Implements Scheduler Spec v01:
    1. Update D from previous block telemetry (with hysteresis)
    2. Select voices using ring scheduler
    3. Compute due flags for TOP/MID
    4. Advance tile counters
    5. Return schedule result with telemetry
    """
    voice_selector: VoiceSelector = field(default_factory=VoiceSelector)
    tile_scheduler: TileScheduler = field(default_factory=TileScheduler)
    degrade_controller: DegradeController = field(default_factory=DegradeController)

    # Budget configuration
    b_compute: int = 1000  # Compute budget per block
    b_mem: int = 1000      # Memory budget per block

    # Previous block telemetry (for D update)
    prev_cpu_bar: int = 0
    prev_mem_bar: int = 0
    prev_stab_bar: int = Q15_ONE

    # Accumulated costs this block
    used_compute: int = 0
    used_mem: int = 0

    def reset_block_costs(self) -> None:
        """Reset accumulated costs for new block."""
        self.used_compute = 0
        self.used_mem = 0

    def add_costs(self, compute: int, mem: int) -> None:
        """Add costs from executed operations."""
        self.used_compute += compute
        self.used_mem += mem

    def compute_health_bars(self) -> Tuple[int, int]:
        """
        Compute CPU and memory bars from used costs.

        Returns (cpu_bar, mem_bar) in Q15.
        """
        if self.b_compute > 0:
            cpu_bar = min((self.used_compute * Q15_SCALE) // self.b_compute, 0xFFFF)
        else:
            cpu_bar = Q15_ONE

        if self.b_mem > 0:
            mem_bar = min((self.used_mem * Q15_SCALE) // self.b_mem, 0xFFFF)
        else:
            mem_bar = Q15_ONE

        return cpu_bar, mem_bar

    def is_top_suppressed(self, D: int, stab_bar: int) -> bool:
        """
        Check if TOP should be suppressed.

        TOP suppressed if D=4 AND stab_bar < theta_stab_red
        """
        return D == 4 and stab_bar < THETA_STAB_RED_Q15

    def compute_stab_bar(
        self,
        clamp_hits: int,
        norm_gain_extreme: int,
        emergency_fired: int,
        voices_serviced: int,
    ) -> int:
        """
        Compute stability bar from tail metrics.

        stab_bar decreases with clamp_hits and emergency events.
        """
        if voices_serviced == 0:
            return Q15_ONE  # No voices = stable

        # Start at full stability
        stab = Q15_ONE

        # Decrease by 10% for each emergency
        stab -= emergency_fired * (Q15_SCALE // 10)

        # Decrease by 5% for each extreme gain
        stab -= norm_gain_extreme * (Q15_SCALE // 20)

        # Decrease by 1% for each clamp hit (normalized to voice count)
        clamp_penalty = (clamp_hits * Q15_SCALE) // (voices_serviced * 100)
        stab -= clamp_penalty

        return max(0, stab)

    def schedule_block(
        self,
        frames: List[Optional[VoiceFramePacketV0b]],
        tail_stats: Optional[dict] = None,
    ) -> ScheduleResult:
        """
        Schedule a block of voices.

        Args:
            frames: List of VoiceFramePacketV0b (or None for absent voices)
            tail_stats: Optional dict with clamp_hits, norm_gain_extreme, emergency_fired

        Returns:
            ScheduleResult with serviced voices and telemetry
        """
        # Reset costs for new block
        self.reset_block_costs()

        # 1. Update D from previous block telemetry
        D = self.degrade_controller.update(self.prev_cpu_bar, self.prev_mem_bar)

        # 2. Check TOP suppression
        top_suppressed = self.is_top_suppressed(D, self.prev_stab_bar)

        # 3. Compute V_serviced based on budget
        v_serviced_max = self.voice_selector.compute_v_serviced(
            self.b_compute, self.b_mem
        )

        # 4. Select voices using ring scheduler
        serviced_ids, eligible_count = self.voice_selector.select_voices(
            frames, v_serviced_max
        )

        # 5. Compute due flags for serviced voices
        due_flags = self.tile_scheduler.compute_due_flags(
            serviced_ids, D, top_suppressed
        )

        # 6. Build schedule info
        serviced_info = [
            VoiceScheduleInfo(v, mid_due, top_due)
            for v, mid_due, top_due in due_flags
        ]

        # 7. Advance tile counters (only for tiles with serviced voices)
        self.tile_scheduler.advance_counters(serviced_ids, D)

        # 8. Process tail stats
        clamp_hits = 0
        norm_gain_extreme = 0
        emergency_fired = 0
        if tail_stats:
            clamp_hits = tail_stats.get('clamp_hits', 0)
            norm_gain_extreme = tail_stats.get('norm_gain_extreme', 0)
            emergency_fired = tail_stats.get('emergency_fired', 0)

        # 9. Build telemetry
        telemetry = HealthTelemetry(
            cpu_bar=self.prev_cpu_bar,
            mem_bar=self.prev_mem_bar,
            stab_bar=self.prev_stab_bar,
            gear=self.degrade_controller.get_gear(),
            clamp_hits=clamp_hits,
            norm_gain_extreme=norm_gain_extreme,
            emergency_fired=emergency_fired,
            voices_serviced=len(serviced_ids),
            voices_eligible=eligible_count,
        )

        return ScheduleResult(
            serviced=serviced_info,
            degrade_state=D,
            top_suppressed=top_suppressed,
            telemetry=telemetry,
        )

    def end_block(self, tail_stats: Optional[dict] = None) -> None:
        """
        Called at end of block to update telemetry for next block.

        Args:
            tail_stats: Optional dict with clamp_hits, norm_gain_extreme, emergency_fired
        """
        # Compute health bars from this block's costs
        self.prev_cpu_bar, self.prev_mem_bar = self.compute_health_bars()

        # Compute stability bar
        clamp_hits = 0
        norm_gain_extreme = 0
        emergency_fired = 0
        voices = 1  # Prevent div by zero

        if tail_stats:
            clamp_hits = tail_stats.get('clamp_hits', 0)
            norm_gain_extreme = tail_stats.get('norm_gain_extreme', 0)
            emergency_fired = tail_stats.get('emergency_fired', 0)
            voices = max(tail_stats.get('voices_serviced', 1), 1)

        self.prev_stab_bar = self.compute_stab_bar(
            clamp_hits, norm_gain_extreme, emergency_fired, voices
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'BLOCK_SIZE', 'V_MAX', 'TILE_SIZE', 'NUM_TILES',
    'Q15_ONE', 'Q15_SCALE', 'THETA_ACTIVE_Q15', 'THETA_STAB_RED_Q15',
    'H_DOWN', 'DEGRADE_THRESHOLDS_Q15', 'PERIOD_TABLE', 'C_TAIL_VOICE',
    # Enums
    'EmissionGroup',
    # Helpers
    'tile_id', 'tile_off',
    # Classes
    'HealthTelemetry',
    'VoiceSelector',
    'TileScheduler',
    'DegradeController',
    'VoiceScheduleInfo',
    'ScheduleResult',
    'BlockScheduler',
]
