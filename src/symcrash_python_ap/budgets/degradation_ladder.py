"""
Deterministic Degradation Ladder (WSE-STD-CMMM)
===============================================

Applies deterministic degradation to operator chains to fit budget.
Guarantees bounded work and instrument-safe output.

Degradation order (highest priority dropped first):
    1. Drop INJECT ops (events) by drop_priority
    2. Drop PERMUTE ops (global reordering)
    3. Drop LOWRANK ops (global dense)
    4. Reduce K (32→24→16→12→8→4→1)
    5. Tail downgrade (FULL→MID→MIN→EMERGENCY)
    6. Emergency mode (carrier-only)

This module provides the core apply_degradation() function that
returns a deterministic plan fitting the budget.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from .operator_costs import (
    get_operator_cost, OpCategory, L_TOTAL_NODES, OPERATOR_COSTS
)
from .tail_modes import (
    TailMode, get_tail_ops, tail_uses_nonlinear,
    get_allowed_tail_mode, downgrade_tail_mode
)
from .pattern_table import (
    popcount32, shrink_mask_to_k, K_LADDER, NODE_PRIORITY,
    get_pattern_for_k, build_mask_for_k
)
from .regime_budgets import RegimeBudget
from .budget_predictor import predict_chain_cost, check_budget_fit


# =============================================================================
# DEGRADATION RESULT
# =============================================================================

@dataclass
class DegradationResult:
    """Result of applying degradation ladder."""
    backbone_ops: List[int]
    event_ops: List[int]
    K: int
    node_mask: int
    tail_mode: TailMode
    pattern_id: int  # Canonical pattern_id for this K rung (VoiceFrame-ready)

    # Metadata
    degradation_steps: int
    final_ticks: int
    is_emergency: bool

    def get_full_op_chain(self) -> List[int]:
        """Get complete operator chain including tail."""
        return self.backbone_ops + self.event_ops + get_tail_ops(self.tail_mode)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_droppable_ops(op_ids: List[int], category: Optional[OpCategory] = None) -> List[int]:
    """
    Get operators that can be dropped, sorted by drop priority (highest first).

    Args:
        op_ids: List of operator IDs
        category: Optional category filter

    Returns:
        Sorted list of droppable operator IDs
    """
    droppable = []
    for op_id in op_ids:
        cost = get_operator_cost(op_id)
        # Never drop tail ops
        if cost.is_tail:
            continue
        # Category filter
        if category is not None and cost.category != category:
            continue
        droppable.append(op_id)

    # Sort by drop priority (highest = drop first)
    droppable.sort(key=lambda op: get_operator_cost(op).drop_priority, reverse=True)
    return droppable


def _is_global_op(op_id: int) -> bool:
    """Check if operator is global (L-scaling)."""
    cost = get_operator_cost(op_id)
    return cost.ticks_per_total_node > 0


def _get_next_k(current_k: int) -> int:
    """Get next lower K in degradation ladder."""
    for k in K_LADDER:
        if k < current_k:
            return k
    return 1


def _tail_fits_at_emergency(budget: RegimeBudget) -> bool:
    """Check if emergency tail fits in budget with K=1."""
    predicted = predict_chain_cost([], [], 1, TailMode.EMERGENCY)
    fits, _ = check_budget_fit(predicted, budget)
    return fits


# =============================================================================
# MAIN DEGRADATION FUNCTION
# =============================================================================

def apply_degradation(
    backbone_ops: List[int],
    event_ops: List[int],
    node_mask: int,
    tail_mode: TailMode,
    budget: RegimeBudget,
    max_iterations: int = 128,
) -> DegradationResult:
    """
    Deterministically apply degradation ladder to fit budget.

    This function is the core of the CMMM budget system. It guarantees:
    - Deterministic output for identical inputs
    - Bounded work (fits within budget)
    - Instrument-safe output (always has valid tail)

    Args:
        backbone_ops: Initial backbone operator IDs
        event_ops: Initial event operator IDs
        node_mask: Initial 32-bit node mask
        tail_mode: Initial tail mode
        budget: Regime budget constraints
        max_iterations: Safety limit on degradation steps

    Returns:
        DegradationResult with final plan that fits budget
    """
    # Work with copies to avoid mutation
    backbone = list(backbone_ops)
    events = list(event_ops)
    K = popcount32(node_mask)
    mask = node_mask
    tail = tail_mode
    steps = 0

    # -------------------------------------------------------------------------
    # Step 0: Enforce hard K cap from budget
    # -------------------------------------------------------------------------
    if K > budget.max_active_nodes_K:
        K = budget.max_active_nodes_K
        mask = shrink_mask_to_k(mask, K)

    # -------------------------------------------------------------------------
    # Step 0b: Handle nonlinear constraint on tail
    # -------------------------------------------------------------------------
    if not budget.allow_nonlinear and tail in (TailMode.FULL, TailMode.MID):
        tail = TailMode.MIN

    # -------------------------------------------------------------------------
    # Step 0c: Ensure tail is feasible at K=1 (last resort)
    # -------------------------------------------------------------------------
    if not _tail_fits_at_emergency(budget):
        # Budget is impossibly tight - return minimal emergency with actual predicted cost
        emergency_predicted = predict_chain_cost([], [], 1, TailMode.EMERGENCY)
        return DegradationResult(
            backbone_ops=[],
            event_ops=[],
            K=1,
            node_mask=shrink_mask_to_k(mask, 1),
            tail_mode=TailMode.EMERGENCY,
            pattern_id=get_pattern_for_k(1).pattern_id,
            degradation_steps=1,
            final_ticks=emergency_predicted.total_ticks,  # v0: Real predicted ticks, not 0
            is_emergency=True,
        )

    # -------------------------------------------------------------------------
    # Main degradation loop
    # -------------------------------------------------------------------------
    for iteration in range(max_iterations):
        # Predict current cost
        predicted = predict_chain_cost(backbone, events, K, tail)
        fits, violations = check_budget_fit(predicted, budget)

        # Check nonlinear constraint
        if predicted.uses_nonlinear and not budget.allow_nonlinear:
            old_backbone, old_events, old_tail = backbone, events, tail

            # First: force tail down to a linear tail (MIN or EMERGENCY)
            if tail in (TailMode.FULL, TailMode.MID):
                tail = TailMode.MIN

            # Then: drop any remaining nonlinear non-tail ops from backbone/events
            backbone = [op for op in backbone if not get_operator_cost(op).nonlinear]
            events = [op for op in events if not get_operator_cost(op).nonlinear]

            steps += 1

            # If nothing changed and we're still nonlinear, force emergency tail
            if (backbone, events, tail) == (old_backbone, old_events, old_tail):
                tail = TailMode.EMERGENCY
            continue

        # Success - fits budget
        if fits:
            return DegradationResult(
                backbone_ops=backbone,
                event_ops=events,
                K=K,
                node_mask=mask,
                tail_mode=tail,
                pattern_id=get_pattern_for_k(K).pattern_id,
                degradation_steps=steps,
                final_ticks=predicted.total_ticks,
                is_emergency=(tail == TailMode.EMERGENCY and K == 1 and not backbone and not events),
            )

        steps += 1

        # ---------------------------------------------------------------------
        # Ladder Step 1: Drop event ops (INJECT category, highest priority first)
        # ---------------------------------------------------------------------
        if events:
            droppable = _get_droppable_ops(events)
            if droppable:
                drop_op = droppable[0]
                events = [op for op in events if op != drop_op]
                continue

        # ---------------------------------------------------------------------
        # Ladder Step 2: Drop PERMUTE ops (global reordering)
        # ---------------------------------------------------------------------
        permutes = _get_droppable_ops(backbone, OpCategory.PERMUTE)
        if permutes:
            drop_op = permutes[0]
            backbone = [op for op in backbone if op != drop_op]
            continue

        # ---------------------------------------------------------------------
        # Ladder Step 3: Drop LOWRANK ops (global dense)
        # ---------------------------------------------------------------------
        lowranks = _get_droppable_ops(backbone, OpCategory.LOWRANK)
        if lowranks:
            drop_op = lowranks[0]
            backbone = [op for op in backbone if op != drop_op]
            continue

        # ---------------------------------------------------------------------
        # Ladder Step 4: Reduce K (32→24→16→12→8→4→1)
        # ---------------------------------------------------------------------
        next_k = _get_next_k(K)
        if next_k < K:
            K = next_k
            mask = shrink_mask_to_k(mask, K)
            continue

        # ---------------------------------------------------------------------
        # Ladder Step 5: Drop remaining backbone ops by priority
        # ---------------------------------------------------------------------
        droppable_backbone = _get_droppable_ops(backbone)
        if droppable_backbone:
            drop_op = droppable_backbone[0]
            backbone = [op for op in backbone if op != drop_op]
            continue

        # ---------------------------------------------------------------------
        # Ladder Step 6: Tail downgrade (FULL→MID→MIN→EMERGENCY)
        # ---------------------------------------------------------------------
        if tail == TailMode.FULL:
            tail = TailMode.MID if budget.allow_nonlinear else TailMode.MIN
            continue
        elif tail == TailMode.MID:
            tail = TailMode.MIN
            continue
        elif tail == TailMode.MIN:
            tail = TailMode.EMERGENCY
            continue

        # ---------------------------------------------------------------------
        # Ladder Step 7: Emergency mode (nothing left to drop)
        # ---------------------------------------------------------------------
        backbone = []
        events = []
        K = 1
        mask = shrink_mask_to_k(mask, K)
        tail = TailMode.EMERGENCY
        break

    # Fallback (should not reach here normally)
    final_predicted = predict_chain_cost(backbone, events, K, tail)
    return DegradationResult(
        backbone_ops=backbone,
        event_ops=events,
        K=K,
        node_mask=mask,
        tail_mode=tail,
        pattern_id=get_pattern_for_k(K).pattern_id,
        degradation_steps=steps,
        final_ticks=final_predicted.total_ticks,
        is_emergency=True,
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def degrade_to_fit(
    ops: List[int],
    K: int,
    budget: RegimeBudget,
    tail_mode: TailMode = TailMode.FULL,
) -> DegradationResult:
    """
    Simplified interface for degradation.

    Treats all ops as backbone (no event distinction).

    Args:
        ops: All operator IDs
        K: Initial active nodes
        budget: Budget constraints
        tail_mode: Initial tail mode

    Returns:
        DegradationResult fitting budget
    """
    # Build full mask from K
    from .pattern_table import build_mask_for_k
    mask = build_mask_for_k(K)

    return apply_degradation(
        backbone_ops=ops,
        event_ops=[],
        node_mask=mask,
        tail_mode=tail_mode,
        budget=budget,
    )


def emergency_fallback() -> DegradationResult:
    """
    Get minimal emergency configuration.

    Returns the absolute minimum viable configuration.
    """
    return DegradationResult(
        backbone_ops=[],
        event_ops=[],
        K=1,
        node_mask=build_mask_for_k(1),
        tail_mode=TailMode.EMERGENCY,
        pattern_id=get_pattern_for_k(1).pattern_id,
        degradation_steps=0,
        final_ticks=0,
        is_emergency=True,
    )


def simulate_degradation(
    backbone_ops: List[int],
    event_ops: List[int],
    K: int,
    tail_mode: TailMode,
    budget: RegimeBudget,
) -> List[Tuple[str, int, int, TailMode]]:
    """
    Simulate degradation steps for debugging/visualization.

    Returns list of (action, K, remaining_ops, tail_mode) at each step.
    """
    from .pattern_table import build_mask_for_k

    steps = []
    mask = build_mask_for_k(K)
    backbone = list(backbone_ops)
    events = list(event_ops)
    tail = tail_mode
    current_K = K

    for _ in range(128):
        predicted = predict_chain_cost(backbone, events, current_K, tail)
        fits, _ = check_budget_fit(predicted, budget)

        if fits:
            steps.append(("FIT", current_K, len(backbone) + len(events), tail))
            break

        # Record current state before degradation
        if events:
            steps.append(("DROP_EVENT", current_K, len(backbone) + len(events), tail))
            events.pop()
        elif any(_is_global_op(op) for op in backbone):
            steps.append(("DROP_GLOBAL", current_K, len(backbone) + len(events), tail))
            backbone = [op for op in backbone if not _is_global_op(op)][:len(backbone)-1]
        elif current_K > 1:
            current_K = _get_next_k(current_K)
            steps.append(("REDUCE_K", current_K, len(backbone) + len(events), tail))
        elif tail != TailMode.EMERGENCY:
            tail = downgrade_tail_mode(tail)
            steps.append(("DOWNGRADE_TAIL", current_K, len(backbone) + len(events), tail))
        else:
            steps.append(("EMERGENCY", 1, 0, TailMode.EMERGENCY))
            break

    return steps


__all__ = [
    'DegradationResult',
    'apply_degradation', 'degrade_to_fit', 'emergency_fallback',
    'simulate_degradation',
]
