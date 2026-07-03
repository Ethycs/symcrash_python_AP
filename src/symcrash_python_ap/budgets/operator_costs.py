"""
Operator Cost Table (WSE-STD-DEGRADE-v0)
========================================

Re-exports from tables/operator_costs.py (canonical source).

Historical note: this module's docstring used to cite "WSE-STD-031",
a planned standard number that was never published; its content was
consolidated into WSE-STD-DEGRADE-v0.md (cost-model interface) and
CMMM-SPEC.md (operator taxonomy). See the canonical module for the
tick-unit definition (per coefficient) and the errata note.
"""

# Re-export canonical definitions from tables/operator_costs.py
from ..tables.operator_costs import (
    COEFF_LEN, NODE_LEN, L_TOTAL_NODES,
    OpCategory, OperatorCost,
    COST_TABLE_VERSION, OPERATOR_COSTS,
    get_operator_cost, get_tail_operators, get_nonlinear_operators,
    get_operators_by_category, is_global_op,
)

__all__ = [
    'COEFF_LEN', 'NODE_LEN', 'L_TOTAL_NODES',
    'OpCategory', 'OperatorCost',
    'COST_TABLE_VERSION', 'OPERATOR_COSTS',
    'get_operator_cost', 'get_tail_operators', 'get_nonlinear_operators',
    'get_operators_by_category', 'is_global_op',
]
