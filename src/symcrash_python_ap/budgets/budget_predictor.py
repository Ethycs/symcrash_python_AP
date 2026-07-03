"""
Budget Prediction & Verification (WSE-STD-DEGRADE-v0 cost-model interface)
===============================================

Predicts resource costs for operator chains BEFORE execution.
Enables deterministic degradation by computing costs upfront.

Usage:
    predicted = predict_chain_cost(backbone_ops, event_ops, K, tail_mode)
    fits, violations = check_budget_fit(predicted, budget)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from .operator_costs import (
    get_operator_cost, OperatorCost, L_TOTAL_NODES, OPERATOR_COSTS
)
from .tail_modes import TailMode, get_tail_ops, tail_uses_nonlinear
from .regime_budgets import RegimeBudget


# =============================================================================
# PREDICTED COST DATACLASS
# =============================================================================

@dataclass
class PredictedCost:
    """Predicted resource cost for an operator chain."""

    # Total costs
    total_ticks: int

    # Breakdown by category
    backbone_ticks: int
    event_ticks: int
    tail_ticks: int

    # Active nodes
    active_nodes_K: int
    total_nodes_L: int

    # Flags
    uses_nonlinear: bool

    @property
    def non_tail_ticks(self) -> int:
        """Ticks used by backbone + events (not tail)."""
        return self.backbone_ticks + self.event_ticks


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_operator_ticks(op_id: int, K: int, L: int = L_TOTAL_NODES) -> Tuple[int, bool]:
    """
    Predict ticks for a single operator.

    Args:
        op_id: Operator ID
        K: Active nodes count
        L: Total nodes count (default 32)

    Returns:
        (ticks, uses_nonlinear)
    """
    cost = get_operator_cost(op_id)
    ticks = cost.compute_ticks(K, L)
    return ticks, cost.nonlinear


def predict_ops_ticks(op_ids: List[int], K: int, L: int = L_TOTAL_NODES) -> Tuple[int, bool]:
    """
    Predict total ticks for a list of operators.

    Args:
        op_ids: List of operator IDs
        K: Active nodes count
        L: Total nodes count

    Returns:
        (total_ticks, uses_nonlinear)
    """
    total = 0
    uses_nl = False
    for op_id in op_ids:
        ticks, nl = predict_operator_ticks(op_id, K, L)
        total += ticks
        uses_nl = uses_nl or nl
    return total, uses_nl


def predict_chain_cost(
    backbone_ops: List[int],
    event_ops: List[int],
    K: int,
    tail_mode: TailMode,
    L: int = L_TOTAL_NODES,
) -> PredictedCost:
    """
    Predict cost for full operator chain including tail.

    Args:
        backbone_ops: List of backbone operator IDs
        event_ops: List of event operator IDs
        K: Active nodes count
        tail_mode: Tail mode (FULL/MID/MIN/EMERGENCY)
        L: Total nodes count

    Returns:
        PredictedCost with full breakdown
    """
    # Backbone cost
    backbone_ticks, backbone_nl = predict_ops_ticks(backbone_ops, K, L)

    # Event cost
    event_ticks, event_nl = predict_ops_ticks(event_ops, K, L)

    # Tail cost
    tail_ops = get_tail_ops(tail_mode)
    tail_ticks, tail_nl = predict_ops_ticks(tail_ops, K, L)

    # Combined flags
    uses_nonlinear = backbone_nl or event_nl or tail_nl

    return PredictedCost(
        total_ticks=backbone_ticks + event_ticks + tail_ticks,
        backbone_ticks=backbone_ticks,
        event_ticks=event_ticks,
        tail_ticks=tail_ticks,
        active_nodes_K=K,
        total_nodes_L=L,
        uses_nonlinear=uses_nonlinear,
    )


# =============================================================================
# BUDGET CHECKING
# =============================================================================

@dataclass
class BudgetViolation:
    """Details of a budget violation."""
    category: str
    actual: int
    limit: int
    message: str


def check_budget_fit(
    predicted: PredictedCost,
    budget: RegimeBudget,
) -> Tuple[bool, List[BudgetViolation]]:
    """
    Check if predicted cost fits within budget.

    Args:
        predicted: Predicted cost from predict_chain_cost
        budget: Regime budget constraints

    Returns:
        (fits, violations) where violations lists what exceeded budget
    """
    violations = []

    # Check total ticks
    if predicted.total_ticks > budget.max_ticks_per_block:
        violations.append(BudgetViolation(
            category="ticks",
            actual=predicted.total_ticks,
            limit=budget.max_ticks_per_block,
            message=f"Total ticks {predicted.total_ticks} > {budget.max_ticks_per_block}",
        ))

    # Check active nodes
    if predicted.active_nodes_K > budget.max_active_nodes_K:
        violations.append(BudgetViolation(
            category="nodes",
            actual=predicted.active_nodes_K,
            limit=budget.max_active_nodes_K,
            message=f"Active nodes {predicted.active_nodes_K} > {budget.max_active_nodes_K}",
        ))

    # Check nonlinear ops
    if predicted.uses_nonlinear and not budget.allow_nonlinear:
        violations.append(BudgetViolation(
            category="nonlinear",
            actual=1,
            limit=0,
            message="Nonlinear ops not allowed in this regime",
        ))

    return (len(violations) == 0, violations)


def estimate_max_backbone_ticks(budget: RegimeBudget, K: int, tail_mode: TailMode) -> int:
    """
    Estimate maximum ticks available for backbone+events.

    Subtracts tail reservation from budget.

    Args:
        budget: Regime budget
        K: Active nodes count
        tail_mode: Tail mode to use

    Returns:
        Available ticks for non-tail operators
    """
    # Predict tail cost
    tail_ops = get_tail_ops(tail_mode)
    tail_ticks, _ = predict_ops_ticks(tail_ops, K)

    # Use the larger of predicted tail or reserved amount
    tail_cost = max(tail_ticks, budget.tail_reserved_ticks)

    return budget.max_ticks_per_block - tail_cost


def can_add_operator(
    current_ticks: int,
    op_id: int,
    K: int,
    budget: RegimeBudget,
    tail_mode: TailMode,
) -> bool:
    """
    Quick check if adding an operator would exceed budget.

    Args:
        current_ticks: Current backbone+event ticks
        op_id: Operator to potentially add
        K: Active nodes count
        budget: Regime budget
        tail_mode: Current tail mode

    Returns:
        True if operator can be added within budget
    """
    op_ticks, uses_nl = predict_operator_ticks(op_id, K)

    # Check nonlinear constraint
    if uses_nl and not budget.allow_nonlinear:
        return False

    # Check tick budget
    max_available = estimate_max_backbone_ticks(budget, K, tail_mode)
    return (current_ticks + op_ticks) <= max_available


# =============================================================================
# COST ESTIMATION UTILITIES
# =============================================================================

def estimate_k_for_budget(
    backbone_ops: List[int],
    event_ops: List[int],
    tail_mode: TailMode,
    budget: RegimeBudget,
) -> int:
    """
    Estimate the maximum K that fits in budget.

    Binary search for largest K where chain fits.

    Args:
        backbone_ops: Backbone operators
        event_ops: Event operators
        tail_mode: Tail mode
        budget: Regime budget

    Returns:
        Maximum K that fits (1-32)
    """
    # Start with budget's max K
    max_k = budget.max_active_nodes_K

    # Binary search
    low, high = 1, max_k
    result = 1

    while low <= high:
        mid = (low + high) // 2
        predicted = predict_chain_cost(backbone_ops, event_ops, mid, tail_mode)
        fits, _ = check_budget_fit(predicted, budget)

        if fits:
            result = mid
            low = mid + 1
        else:
            high = mid - 1

    return result


def rank_operators_by_efficiency(op_ids: List[int], K: int) -> List[Tuple[int, float]]:
    """
    Rank operators by cost efficiency (lower is better).

    Useful for deciding which operators to drop first.

    Args:
        op_ids: Operator IDs to rank
        K: Active nodes count

    Returns:
        List of (op_id, cost_per_priority) sorted by efficiency
    """
    ranked = []
    for op_id in op_ids:
        cost = get_operator_cost(op_id)
        ticks = cost.compute_ticks(K)
        # Lower priority = more important = higher efficiency
        # Avoid division by zero
        priority = max(cost.drop_priority, 1)
        efficiency = ticks / priority
        ranked.append((op_id, efficiency))

    # Sort by efficiency (lower is better to keep)
    ranked.sort(key=lambda x: x[1])
    return ranked


__all__ = [
    'PredictedCost', 'BudgetViolation',
    'predict_operator_ticks', 'predict_ops_ticks', 'predict_chain_cost',
    'check_budget_fit', 'estimate_max_backbone_ticks', 'can_add_operator',
    'estimate_k_for_budget', 'rank_operators_by_efficiency',
]
