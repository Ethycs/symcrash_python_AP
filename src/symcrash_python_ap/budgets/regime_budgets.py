"""
Per-Regime Budget Limits (WSE-STD-032)
======================================

Defines resource budgets for each regime and voice class.
Budgets are FPGA-native: derived from hardware constraints.

Reference configuration:
- 100 MHz clock
- 64-sample block @ 48kHz
- Block time = 1.33 ms = 133,000 clock cycles
- Symbol Engine gets ~40% = 53,000 cycles per block (shared among voices)

Voice classes:
- MAIN: Up to 12 foreground voices with generous budgets
- RESIDUAL: Up to 128 background/cloud voices with tight budgets
"""

from dataclasses import dataclass
from typing import Dict
from enum import IntEnum


class RegimeId(IntEnum):
    """Regime identifiers matching WSE-STD-030."""
    IDLE = 0
    TRANSIENT = 1
    STABLE_TONE = 2
    GLITCH_CHATTER = 3
    RECOVERY = 4
    # Special regimes
    RESIDUAL = 100      # Cloud/background voices
    CARRIER_ONLY = 200  # Emergency fallback


class VoiceClass(IntEnum):
    """Voice classification for budget allocation."""
    MAIN = 0       # Foreground voices (slots 0-11)
    RESIDUAL = 1   # Background/cloud voices (slots 12+)


# =============================================================================
# REGIME BUDGET DATACLASS
# =============================================================================

@dataclass(frozen=True)
class RegimeBudget:
    """
    Resource budget for a single regime.

    All values in integer tick units matching operator costs.
    """
    regime_id: int
    regime_name: str

    # Primary budget constraint (ticks per block per voice)
    max_ticks_per_block: int

    # Node constraint
    max_active_nodes_K: int

    # Feature flags
    allow_nonlinear: bool      # Allow sqrt, tanh, etc.

    # Tail reservation (must fit within max_ticks_per_block)
    tail_reserved_ticks: int

    @property
    def available_for_backbone(self) -> int:
        """Ticks available for backbone + events after tail reservation."""
        return self.max_ticks_per_block - self.tail_reserved_ticks


# =============================================================================
# BUDGET TABLE
# =============================================================================

REGIME_BUDGETS: Dict[int, RegimeBudget] = {

    # -------------------------------------------------------------------------
    # MAIN Voice Regimes (generous budgets)
    # -------------------------------------------------------------------------

    RegimeId.IDLE: RegimeBudget(
        regime_id=RegimeId.IDLE,
        regime_name="IDLE",
        max_ticks_per_block=1300,      # Must fit EMERGENCY tail (1152 ticks)
        max_active_nodes_K=8,          # Only low-frequency nodes
        allow_nonlinear=False,
        tail_reserved_ticks=1152,      # EMERGENCY tail: FineClamp + EmergencySimplify
    ),

    RegimeId.TRANSIENT: RegimeBudget(
        regime_id=RegimeId.TRANSIENT,
        regime_name="TRANSIENT",
        max_ticks_per_block=2000,
        max_active_nodes_K=16,
        allow_nonlinear=False,         # No expensive ops during attack
        tail_reserved_ticks=400,
    ),

    RegimeId.STABLE_TONE: RegimeBudget(
        regime_id=RegimeId.STABLE_TONE,
        regime_name="STABLE_TONE",
        max_ticks_per_block=8000,      # Full processing for main voices
        max_active_nodes_K=32,         # All nodes
        allow_nonlinear=True,          # Allow sqrt, etc.
        tail_reserved_ticks=600,
    ),

    RegimeId.GLITCH_CHATTER: RegimeBudget(
        regime_id=RegimeId.GLITCH_CHATTER,
        regime_name="GLITCH_CHATTER",
        max_ticks_per_block=3000,
        max_active_nodes_K=24,
        allow_nonlinear=False,
        tail_reserved_ticks=500,
    ),

    RegimeId.RECOVERY: RegimeBudget(
        regime_id=RegimeId.RECOVERY,
        regime_name="RECOVERY",
        max_ticks_per_block=1500,      # Must fit EMERGENCY tail (1152 ticks)
        max_active_nodes_K=12,
        allow_nonlinear=False,
        tail_reserved_ticks=1152,      # EMERGENCY tail: FineClamp + EmergencySimplify
    ),

    # -------------------------------------------------------------------------
    # Special Regimes
    # -------------------------------------------------------------------------

    RegimeId.RESIDUAL: RegimeBudget(
        regime_id=RegimeId.RESIDUAL,
        regime_name="RESIDUAL",
        max_ticks_per_block=1300,      # Must fit EMERGENCY tail (1152 ticks)
        max_active_nodes_K=8,          # Limited nodes
        allow_nonlinear=False,
        tail_reserved_ticks=1152,      # EMERGENCY tail: FineClamp + EmergencySimplify
    ),

    RegimeId.CARRIER_ONLY: RegimeBudget(
        regime_id=RegimeId.CARRIER_ONLY,
        regime_name="CARRIER_ONLY",
        max_ticks_per_block=1200,      # Must fit EMERGENCY tail [2,19] = 1152 ticks
        max_active_nodes_K=1,          # Single carrier frequency
        allow_nonlinear=False,
        tail_reserved_ticks=1152,      # FineClamp + EmergencySimplify
    ),
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================

def get_regime_budget(regime_id: int, voice_class: VoiceClass = VoiceClass.MAIN) -> RegimeBudget:
    """
    Get budget for regime and voice class.

    Args:
        regime_id: Regime ID (0-4 or special)
        voice_class: MAIN or RESIDUAL

    Returns:
        RegimeBudget for this regime + voice class
    """
    # RESIDUAL class always uses RESIDUAL budget
    if voice_class == VoiceClass.RESIDUAL:
        return REGIME_BUDGETS[RegimeId.RESIDUAL]

    # MAIN class uses regime-specific budget
    if regime_id in REGIME_BUDGETS:
        return REGIME_BUDGETS[regime_id]

    # Default to STABLE_TONE for unknown regimes
    return REGIME_BUDGETS[RegimeId.STABLE_TONE]


def get_voice_class(voice_id: int, main_voice_limit: int = 12) -> VoiceClass:
    """
    Determine voice class from voice ID.

    Args:
        voice_id: Voice slot number
        main_voice_limit: Threshold for MAIN vs RESIDUAL

    Returns:
        VoiceClass.MAIN or VoiceClass.RESIDUAL
    """
    if voice_id < main_voice_limit:
        return VoiceClass.MAIN
    return VoiceClass.RESIDUAL


def budget_allows_operator(budget: RegimeBudget, op_cost: int, uses_nonlinear: bool) -> bool:
    """
    Quick check if an operator can fit in budget.

    Args:
        budget: Current regime budget
        op_cost: Computed tick cost of operator
        uses_nonlinear: Whether operator uses nonlinear ops

    Returns:
        True if operator is allowed
    """
    if uses_nonlinear and not budget.allow_nonlinear:
        return False
    return op_cost <= budget.available_for_backbone


__all__ = [
    'RegimeId', 'VoiceClass', 'RegimeBudget',
    'REGIME_BUDGETS',
    'get_regime_budget', 'get_voice_class', 'budget_allows_operator',
]
