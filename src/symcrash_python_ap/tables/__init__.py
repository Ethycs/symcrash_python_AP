"""
Canonical Tables Module
=======================

Frozen, versioned tables that define runtime behavior:
- PatternTable: node mask patterns for K scheduling
- OperatorCosts: operator cost model for budgeting
- TailModes: tail operator sets and degradation ladder
"""

from .tail_modes import TailMode, TAIL_OPS, get_tail_ops
from .operator_costs import OperatorCost, OPERATOR_COSTS, predict_ticks
from .pattern_table import PatternEntry, PATTERN_TABLE, get_node_mask, K_STAIRCASE

__all__ = [
    # Tail modes
    'TailMode', 'TAIL_OPS', 'get_tail_ops',
    # Operator costs
    'OperatorCost', 'OPERATOR_COSTS', 'predict_ticks',
    # Pattern table
    'PatternEntry', 'PATTERN_TABLE', 'get_node_mask', 'K_STAIRCASE',
]
