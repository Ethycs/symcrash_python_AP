"""
DX7-Style 6-Operator FM Algorithm Definitions
==============================================

Per WSE-STD-FM-TOPOLOGY-v0:
- All 32 DX7 algorithms as FMAlgorithm instances
- Modulation matrix [6,6] where M[i,j]=True means Op_i modulates Op_j
- Carrier mask (bitmask): which operators sum to output
- Feedback operator: which op self-modulates (-1 if none)
- Eval order: topological sort (modulators before carriers)

Key insight: DX7 algorithms are DAGs with at most one feedback edge.
Standard graph theory is sufficient - no knot invariants needed.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class FMAlgorithm:
    """
    FM algorithm topology definition (DX7-compatible).

    Attributes:
        alg_id: Algorithm number (1-32)
        name: Human-readable name
        modulation_matrix: [6,6] bool array - M[i,j]=True if Op_i modulates Op_j
        carrier_mask: Bitmask where bit i set means Op_i is a carrier
        feedback_op: Which operator has self-feedback (0-5), or -1 if none
        eval_order: List of operator indices in evaluation order (modulators first)
    """
    alg_id: int
    name: str
    modulation_matrix: np.ndarray  # [6,6] bool
    carrier_mask: int              # 6-bit mask
    feedback_op: int               # -1 if no feedback
    eval_order: List[int] = field(default_factory=list)

    def __post_init__(self):
        # Ensure modulation_matrix is numpy array
        if not isinstance(self.modulation_matrix, np.ndarray):
            self.modulation_matrix = np.array(self.modulation_matrix, dtype=bool)

        # Compute modulator lists for each operator
        self._modulator_lists = []
        for op in range(6):
            modulators = []
            for src in range(6):
                if self.modulation_matrix[src, op]:
                    modulators.append(src)
            self._modulator_lists.append(modulators)

    def get_modulators(self, op_idx: int) -> List[int]:
        """Get list of operators that modulate the given operator."""
        return self._modulator_lists[op_idx]

    def get_carriers(self) -> List[int]:
        """Get list of carrier operator indices."""
        return [i for i in range(6) if self.carrier_mask & (1 << i)]

    def is_carrier(self, op_idx: int) -> bool:
        """Check if operator is a carrier (contributes to output)."""
        return bool(self.carrier_mask & (1 << op_idx))

    def has_feedback(self, op_idx: int) -> bool:
        """Check if operator has self-feedback."""
        return self.feedback_op == op_idx


def _make_matrix(*edges) -> np.ndarray:
    """
    Create modulation matrix from edge list.

    Each edge is (src, dst) meaning Op_src modulates Op_dst.
    Operators are 0-indexed (Op1=0, Op2=1, etc).
    """
    mat = np.zeros((6, 6), dtype=bool)
    for src, dst in edges:
        mat[src, dst] = True
    return mat


# =============================================================================
# DX7 Algorithm Definitions (32 algorithms)
# =============================================================================
#
# Naming convention:
# - Operators are numbered 1-6 in diagrams (DX7 convention)
# - Internally we use 0-5 (Python convention)
# - Feedback is shown with ↺ symbol
# - Carriers are marked with * or → OUT
#
# The eval_order ensures modulators compute before their targets.

ALGORITHMS = {
    # =========================================================================
    # Algorithm 1: 6-op chain
    # [6] → [5] → [4] → [3] → [2] → [1*]
    #                               ↺fb
    # =========================================================================
    1: FMAlgorithm(
        alg_id=1,
        name="6-Op Chain",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (3, 2),  # Op4 → Op3
            (2, 1),  # Op3 → Op2
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b000001,  # Op1 is carrier
        feedback_op=0,          # Op1 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 2: 5-op chain + 1 carrier
    # [6] → [5] → [4] → [3] → [2] → [1*]
    #                         ↺fb
    # =========================================================================
    2: FMAlgorithm(
        alg_id=2,
        name="5-Op Chain + FB on Op2",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (3, 2),  # Op4 → Op3
            (2, 1),  # Op3 → Op2
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b000001,  # Op1 is carrier
        feedback_op=1,          # Op2 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 3: 4-op chain + 2-op chain
    # [6] → [5] → [4] → [3*]
    #             [2] → [1*]
    #             ↺fb
    # =========================================================================
    3: FMAlgorithm(
        alg_id=3,
        name="4-Op + 2-Op Chains",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (3, 2),  # Op4 → Op3
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b000101,  # Op1, Op3 are carriers
        feedback_op=1,          # Op2 has feedback
        eval_order=[5, 4, 3, 1, 2, 0],
    ),

    # =========================================================================
    # Algorithm 4: 3-op chain + 3-op chain
    # [6] → [5] → [4*]
    # [3] → [2] → [1*]
    #             ↺fb
    # =========================================================================
    4: FMAlgorithm(
        alg_id=4,
        name="3-Op + 3-Op Chains",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (2, 1),  # Op3 → Op2
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b001001,  # Op1, Op4 are carriers
        feedback_op=0,          # Op1 has feedback
        eval_order=[5, 4, 2, 1, 3, 0],
    ),

    # =========================================================================
    # Algorithm 5: Dual stack into carrier (E.Piano)
    # [6] → [5] → [4] ─┐
    #                   ├─→ [1*]
    # [3] → [2] ───────┘
    #       ↺fb
    # =========================================================================
    5: FMAlgorithm(
        alg_id=5,
        name="E.Piano (Dual Stack)",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (3, 0),  # Op4 → Op1
            (2, 1),  # Op3 → Op2
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b000001,  # Op1 is carrier
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 2, 1, 3, 0],
    ),

    # =========================================================================
    # Algorithm 6: Parallel modulators into carrier
    # [6]   [5] → [4] ─┐
    #  │              ├─→ [1*]
    #  └── [3] → [2] ─┘
    #       ↺fb
    # =========================================================================
    6: FMAlgorithm(
        alg_id=6,
        name="Parallel Stacks",
        modulation_matrix=_make_matrix(
            (5, 2),  # Op6 → Op3
            (4, 3),  # Op5 → Op4
            (3, 0),  # Op4 → Op1
            (2, 1),  # Op3 → Op2
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b000001,  # Op1 is carrier
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 2, 1, 3, 0],
    ),

    # =========================================================================
    # Algorithm 7: 3-op stack + 3 carriers
    # [6] → [5] → [4] → [1*]
    #             [3*]
    #             [2*]
    #             ↺fb
    # =========================================================================
    7: FMAlgorithm(
        alg_id=7,
        name="3-Op Stack + 2 Carriers",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (3, 0),  # Op4 → Op1
        ),
        carrier_mask=0b000111,  # Op1, Op2, Op3 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 8: 2-op pair modulating 2 carriers + 2 carriers
    # [6] → [5] ─┬─→ [4*]
    #            └─→ [3*]
    #            [2*]
    #            [1*]
    #            ↺fb on Op6
    # =========================================================================
    8: FMAlgorithm(
        alg_id=8,
        name="Split Modulator",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (4, 2),  # Op5 → Op3
        ),
        carrier_mask=0b001111,  # Op1-4 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 9: 3+2+1 modulator arrangement
    # [6] → [5] → [4*]
    # [3] → [2] → [1*]
    #       ↺fb on Op3
    # =========================================================================
    9: FMAlgorithm(
        alg_id=9,
        name="Dual 3-Op Chains",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (2, 1),  # Op3 → Op2
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b001001,  # Op1, Op4 are carriers
        feedback_op=2,          # Op3 has feedback
        eval_order=[5, 4, 2, 1, 3, 0],
    ),

    # =========================================================================
    # Algorithm 10: 2+2+2 modulator pairs
    # [6] → [5*]
    # [4] → [3*]
    # [2] → [1*]
    #       ↺fb on Op4
    # =========================================================================
    10: FMAlgorithm(
        alg_id=10,
        name="Three 2-Op Pairs",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (3, 2),  # Op4 → Op3
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b010101,  # Op1, Op3, Op5 are carriers
        feedback_op=3,          # Op4 has feedback
        eval_order=[5, 3, 1, 4, 2, 0],
    ),

    # =========================================================================
    # Algorithm 11: Like 10 but different feedback
    # [6] → [5*]
    # [4] → [3*]
    # [2] → [1*]
    #       ↺fb on Op6
    # =========================================================================
    11: FMAlgorithm(
        alg_id=11,
        name="Three 2-Op Pairs (FB6)",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (3, 2),  # Op4 → Op3
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b010101,  # Op1, Op3, Op5 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 3, 1, 4, 2, 0],
    ),

    # =========================================================================
    # Algorithm 12: 4 modulators into 2 carriers
    # [6] → [5] → [4] → [3*]
    #             [2] → [1*]
    #             ↺fb on Op6
    # =========================================================================
    12: FMAlgorithm(
        alg_id=12,
        name="4+2 Split",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (3, 2),  # Op4 → Op3
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b000101,  # Op1, Op3 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 1, 2, 0],
    ),

    # =========================================================================
    # Algorithm 13: 3+3 arrangement
    # [6] → [5] → [4*]
    #       [3] → [2] → [1*]
    #       ↺fb on Op6
    # =========================================================================
    13: FMAlgorithm(
        alg_id=13,
        name="3+3 Arrangement",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (2, 1),  # Op3 → Op2
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b001001,  # Op1, Op4 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 2, 1, 3, 0],
    ),

    # =========================================================================
    # Algorithm 14: 4-op chain + 2 carriers
    # [6] → [5] → [4] → [3*]
    #                   [2*]
    #                   [1*]
    #                   ↺fb on Op6
    # =========================================================================
    14: FMAlgorithm(
        alg_id=14,
        name="4-Op + 2 Carriers",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (3, 2),  # Op4 → Op3
        ),
        carrier_mask=0b000111,  # Op1, Op2, Op3 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 15: 2-op modulator + 2 carriers + 2 carriers
    # [6] → [5*]
    #       [4*]
    # [2] → [3*]
    #       [1*]
    #       ↺fb on Op6
    # =========================================================================
    15: FMAlgorithm(
        alg_id=15,
        name="2-Op + 4 Carriers",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (1, 2),  # Op2 → Op3
        ),
        carrier_mask=0b011101,  # Op1, Op3, Op4, Op5 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 1, 4, 3, 2, 0],
    ),

    # =========================================================================
    # Algorithm 16: Brass (parallel into carrier)
    # [6] ─┐
    # [5] ─┼─→ [1*]
    # [4] ─┤
    # [3] ─┤
    # [2] ─┘
    # ↺fb on Op6
    # =========================================================================
    16: FMAlgorithm(
        alg_id=16,
        name="Brass (5 into 1)",
        modulation_matrix=_make_matrix(
            (5, 0),  # Op6 → Op1
            (4, 0),  # Op5 → Op1
            (3, 0),  # Op4 → Op1
            (2, 0),  # Op3 → Op1
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b000001,  # Op1 is carrier
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 17: Similar to 16, different routing
    # [6] ─┐
    # [5] ─┼─→ [1*]
    # [4] ─┘
    # [3] ─┤
    # [2] ─┘
    # ↺fb on Op6
    # =========================================================================
    17: FMAlgorithm(
        alg_id=17,
        name="5 Modulators Split",
        modulation_matrix=_make_matrix(
            (5, 0),  # Op6 → Op1
            (4, 0),  # Op5 → Op1
            (3, 0),  # Op4 → Op1
            (2, 0),  # Op3 → Op1
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b000001,  # Op1 is carrier
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 18: 3 stacks feeding carrier
    # [6] → [5] ─┐
    # [4] ───────┼─→ [1*]
    # [3] → [2] ─┘
    # ↺fb on Op6
    # =========================================================================
    18: FMAlgorithm(
        alg_id=18,
        name="3 Stacks into 1",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 0),  # Op5 → Op1
            (3, 0),  # Op4 → Op1
            (2, 1),  # Op3 → Op2
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b000001,  # Op1 is carrier
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 3, 2, 4, 1, 0],
    ),

    # =========================================================================
    # Algorithm 19: 3-op + 3 carriers
    # [6] → [5] ─┬─→ [4*]
    #            ├─→ [3*]
    #            └─→ [2*]
    #            [1*]
    # ↺fb on Op6
    # =========================================================================
    19: FMAlgorithm(
        alg_id=19,
        name="Split to 3 Carriers",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (4, 2),  # Op5 → Op3
            (4, 1),  # Op5 → Op2
        ),
        carrier_mask=0b001111,  # Op1, Op2, Op3, Op4 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 20: 2+2 modulators into 2 carriers
    # [6] → [5] → [4*]
    #       [3] ─┬─→ [2*]
    #            └─→ [1*]
    # ↺fb on Op4
    # =========================================================================
    20: FMAlgorithm(
        alg_id=20,
        name="2+2 into 2 Carriers",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (2, 1),  # Op3 → Op2
            (2, 0),  # Op3 → Op1
        ),
        carrier_mask=0b001011,  # Op1, Op2, Op4 are carriers
        feedback_op=3,          # Op4 has feedback
        eval_order=[5, 4, 2, 3, 1, 0],
    ),

    # =========================================================================
    # Algorithm 21: 2 modulators into 4 carriers
    # [6] → [5] ─┬─→ [4*]
    #            └─→ [3*]
    #       [2] ─┬─→ [1*]
    #            (no mod to 2nd carrier)
    # ↺fb on Op6
    # =========================================================================
    21: FMAlgorithm(
        alg_id=21,
        name="2 Mod Pairs + 2 Carriers",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (4, 2),  # Op5 → Op3
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b001111,  # Op1, Op2, Op3, Op4 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 1, 3, 2, 0],
    ),

    # =========================================================================
    # Algorithm 22: Brass/Organ
    # [6] → [5] ─┬─→ [4*]
    #            ├─→ [3*]
    #            └─→ [2*]
    #            [1*]
    # ↺fb on Op6
    # =========================================================================
    22: FMAlgorithm(
        alg_id=22,
        name="Brass (Split 3)",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (4, 2),  # Op5 → Op3
            (4, 1),  # Op5 → Op2
        ),
        carrier_mask=0b001111,  # Op1, Op2, Op3, Op4 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 23: 5 carriers + 1 modulator
    # [6] → [5*]
    #       [4*]
    #       [3*]
    #       [2*]
    #       [1*]
    # ↺fb on Op6
    # =========================================================================
    23: FMAlgorithm(
        alg_id=23,
        name="1 Mod + 5 Carriers",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
        ),
        carrier_mask=0b011111,  # Op1-5 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 24: All carriers (slight modulation)
    # [6] → [5*]
    #       [4*]
    #       [3*]
    #       [2*]
    #       [1*]
    # ↺fb on Op6
    # =========================================================================
    24: FMAlgorithm(
        alg_id=24,
        name="1 Mod + 5 Carriers (v2)",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
        ),
        carrier_mask=0b011111,  # Op1-5 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 25: Similar with different feedback
    # [6] → [5*]
    #       [4*]
    #       [3*]
    #       [2*]
    #       [1*]
    # ↺fb on Op6
    # =========================================================================
    25: FMAlgorithm(
        alg_id=25,
        name="1 Mod + 5 Carriers (v3)",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
        ),
        carrier_mask=0b011111,  # Op1-5 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 26: 2 modulators + 4 carriers
    # [6] → [5*]
    #       [4*]
    # [3] → [2*]
    #       [1*]
    # ↺fb on Op6
    # =========================================================================
    26: FMAlgorithm(
        alg_id=26,
        name="2 Mods + 4 Carriers",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (2, 1),  # Op3 → Op2
        ),
        carrier_mask=0b010011,  # Op1, Op2, Op5 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 2, 4, 3, 1, 0],
    ),

    # =========================================================================
    # Algorithm 27: 2 modulators + 4 carriers (different config)
    # [6] → [5*]
    #       [4*]
    # [3] → [2*]
    #       [1*]
    # ↺fb on Op3
    # =========================================================================
    27: FMAlgorithm(
        alg_id=27,
        name="2 Mods + 4 Carriers (v2)",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (2, 1),  # Op3 → Op2
        ),
        carrier_mask=0b010011,  # Op1, Op2, Op5 are carriers
        feedback_op=2,          # Op3 has feedback
        eval_order=[5, 2, 4, 3, 1, 0],
    ),

    # =========================================================================
    # Algorithm 28: 3-op chain + 3 carriers
    # [6] → [5] → [4*]
    #             [3*]
    #             [2*]
    #             [1*]
    # ↺fb on Op6
    # =========================================================================
    28: FMAlgorithm(
        alg_id=28,
        name="3-Op Chain + 3 Carriers",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
        ),
        carrier_mask=0b001111,  # Op1, Op2, Op3, Op4 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 29: 2+2+2 carriers with modulation
    # [6] → [5*]
    # [4] → [3*]
    #       [2*]
    #       [1*]
    # ↺fb on Op6
    # =========================================================================
    29: FMAlgorithm(
        alg_id=29,
        name="2+2+2 Pairs",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (3, 2),  # Op4 → Op3
        ),
        carrier_mask=0b010111,  # Op1, Op2, Op3, Op5 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 3, 4, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 30: 2 chains of 3
    # [6] → [5] → [4*]
    # [3] → [2] → [1*]
    # ↺fb on Op6
    # =========================================================================
    30: FMAlgorithm(
        alg_id=30,
        name="2x 3-Op Chains",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
            (4, 3),  # Op5 → Op4
            (2, 1),  # Op3 → Op2
            (1, 0),  # Op2 → Op1
        ),
        carrier_mask=0b001001,  # Op1, Op4 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 2, 1, 3, 0],
    ),

    # =========================================================================
    # Algorithm 31: 5 carriers + 1 modulator
    # [6] → [5*]
    #       [4*]
    #       [3*]
    #       [2*]
    #       [1*]
    # ↺fb on Op6
    # =========================================================================
    31: FMAlgorithm(
        alg_id=31,
        name="1 Mod + 5 Carriers (v4)",
        modulation_matrix=_make_matrix(
            (5, 4),  # Op6 → Op5
        ),
        carrier_mask=0b011111,  # Op1-5 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],
    ),

    # =========================================================================
    # Algorithm 32: All carriers (Additive)
    # [6*]
    # [5*]
    # [4*]
    # [3*]
    # [2*]
    # [1*]
    # ↺fb on Op6
    # =========================================================================
    32: FMAlgorithm(
        alg_id=32,
        name="Additive (6 Carriers)",
        modulation_matrix=_make_matrix(),  # No modulation
        carrier_mask=0b111111,  # All 6 are carriers
        feedback_op=5,          # Op6 has feedback
        eval_order=[5, 4, 3, 2, 1, 0],  # Any order works
    ),
}


def get_algorithm(alg_id: int) -> FMAlgorithm:
    """
    Get algorithm by ID (1-32).

    Args:
        alg_id: Algorithm number (1-32)

    Returns:
        FMAlgorithm instance

    Raises:
        KeyError: If algorithm ID not found
    """
    if alg_id not in ALGORITHMS:
        raise KeyError(f"Algorithm {alg_id} not found. Valid range: 1-32")
    return ALGORITHMS[alg_id]


def list_algorithms() -> List[Tuple[int, str]]:
    """
    List all algorithms with their names.

    Returns:
        List of (alg_id, name) tuples
    """
    return [(alg.alg_id, alg.name) for alg in ALGORITHMS.values()]


__all__ = [
    'FMAlgorithm',
    'ALGORITHMS',
    'get_algorithm',
    'list_algorithms',
]
