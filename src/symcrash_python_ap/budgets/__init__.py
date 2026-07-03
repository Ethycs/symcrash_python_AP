"""
CMMM Budget System (WSE-STD-DEGRADE-v0)
=======================================

Normative sources: WSE-STD-DEGRADE-v0.md (frozen degradation/budget
standard) and CMMM-SPEC.md (operator taxonomy). The WSE-STD-031/032/033
numbers formerly cited here were planned but never published; their
content was consolidated into those documents.

Deterministic budget prediction and degradation for the Symbol Engine.
Ensures bounded work and instrument-safe output.

Core concepts:
    - K: Active nodes (1-32), can be reduced by degradation
    - L: Total nodes (always 32), used for global ops
    - Ticks: Abstract cost units (≈1 Q15 mul-add per tick)
    - Tail: Mandatory safety operators at end of chain

Usage:
    from symcrash_python_ap.budgets import (
        apply_degradation, predict_chain_cost, check_budget_fit,
        get_regime_budget, TailMode, RegimeId,
    )

    # Get budget for current regime
    budget = get_regime_budget(RegimeId.STABLE_TONE)

    # Predict cost
    predicted = predict_chain_cost(backbone_ops, event_ops, K=32, tail_mode=TailMode.FULL)

    # Check if it fits
    fits, violations = check_budget_fit(predicted, budget)

    # Apply degradation if needed
    if not fits:
        result = apply_degradation(
            backbone_ops, event_ops,
            node_mask=0xFFFFFFFF,
            tail_mode=TailMode.FULL,
            budget=budget,
        )
        # result.backbone_ops, result.K, result.tail_mode now fit budget

Modules:
    operator_costs: Per-operator cost table with K/L scaling
    regime_budgets: Per-regime budget limits
    tail_modes: FULL/MID/MIN/EMERGENCY tail definitions
    pattern_table: Node mask patterns for K ladder
    budget_predictor: Cost prediction before execution
    degradation_ladder: Deterministic degradation algorithm
"""

# Operator costs
from .operator_costs import (
    COEFF_LEN, NODE_LEN, L_TOTAL_NODES,
    OpCategory, OperatorCost,
    COST_TABLE_VERSION, OPERATOR_COSTS,
    get_operator_cost, get_tail_operators, get_nonlinear_operators,
    get_operators_by_category, is_global_op,
)

# Regime budgets
from .regime_budgets import (
    RegimeId, VoiceClass, RegimeBudget,
    REGIME_BUDGETS,
    get_regime_budget, get_voice_class, budget_allows_operator,
)

# Tail modes
from .tail_modes import (
    TailMode,
    OP_FINE_CLAMP, OP_NORMALIZE_SCALE, OP_SHEAF_PROJECT, OP_EMERGENCY_SIMPLIFY,
    TAIL_OPS, TAIL_USES_NONLINEAR,
    get_tail_ops, tail_uses_nonlinear,
    get_allowed_tail_mode, downgrade_tail_mode,
    get_tail_mode_for_voice_class, TailConfig,
)

# Pattern table
from .pattern_table import (
    BAND_SLICES, NODES_PER_BAND, K_LADDER,
    NODE_PRIORITY, build_node_priority,
    popcount32, build_mask_for_k, shrink_mask_to_k,
    mask_to_node_list, node_list_to_mask,
    PatternEntry, PATTERN_TABLE,
    get_pattern_by_id, get_pattern_for_k, get_next_lower_k,
    node_mask_to_coeff_mask, get_active_coeff_indices,
)

# Budget prediction
from .budget_predictor import (
    PredictedCost, BudgetViolation,
    predict_operator_ticks, predict_ops_ticks, predict_chain_cost,
    check_budget_fit, estimate_max_backbone_ticks, can_add_operator,
    estimate_k_for_budget, rank_operators_by_efficiency,
)

# Degradation ladder
from .degradation_ladder import (
    DegradationResult,
    apply_degradation, degrade_to_fit, emergency_fallback,
    simulate_degradation,
)


__all__ = [
    # Constants
    'COEFF_LEN', 'NODE_LEN', 'L_TOTAL_NODES',
    'BAND_SLICES', 'NODES_PER_BAND', 'K_LADDER',
    'NODE_PRIORITY', 'COST_TABLE_VERSION',

    # Enums
    'OpCategory', 'RegimeId', 'VoiceClass', 'TailMode',

    # Dataclasses
    'OperatorCost', 'RegimeBudget', 'TailConfig',
    'PatternEntry', 'PredictedCost', 'BudgetViolation', 'DegradationResult',

    # Tables
    'OPERATOR_COSTS', 'REGIME_BUDGETS', 'TAIL_OPS', 'PATTERN_TABLE',

    # Operator cost functions
    'get_operator_cost', 'get_tail_operators', 'get_nonlinear_operators',
    'get_operators_by_category', 'is_global_op',

    # Regime budget functions
    'get_regime_budget', 'get_voice_class', 'budget_allows_operator',

    # Tail mode functions
    'get_tail_ops', 'tail_uses_nonlinear',
    'get_allowed_tail_mode', 'downgrade_tail_mode',
    'get_tail_mode_for_voice_class',
    'OP_FINE_CLAMP', 'OP_NORMALIZE_SCALE', 'OP_SHEAF_PROJECT', 'OP_EMERGENCY_SIMPLIFY',

    # Pattern table functions
    'build_node_priority', 'popcount32', 'build_mask_for_k', 'shrink_mask_to_k',
    'mask_to_node_list', 'node_list_to_mask',
    'get_pattern_by_id', 'get_pattern_for_k', 'get_next_lower_k',
    'node_mask_to_coeff_mask', 'get_active_coeff_indices',

    # Prediction functions
    'predict_operator_ticks', 'predict_ops_ticks', 'predict_chain_cost',
    'check_budget_fit', 'estimate_max_backbone_ticks', 'can_add_operator',
    'estimate_k_for_budget', 'rank_operators_by_efficiency',

    # Degradation functions
    'apply_degradation', 'degrade_to_fit', 'emergency_fallback',
    'simulate_degradation',
]
