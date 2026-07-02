"""
DX7-Style 6-Operator FM Algorithm Definitions
==============================================

Per WSE-STD-FM-TOPOLOGY-v0:
- Algorithms 1-32: the CANONICAL DX7 algorithm chart, generated from a compact
  spec audited bit-for-bit against Dexed/msfa's reference table
  (music-synthesizer-for-android fm_core.cc, the de-facto DX7 ground truth).
  Fixed 2026-06-25: the previous hand-written table diverged from the real
  DX7 chart on 29 of 32 algorithms (tests/test_fm_algorithms_dx7_conformance.py
  now pins the table so it cannot drift again).
- Algorithms 33-38: LEGACY WSE topologies — the pre-fix structures that
  existing ear-tuned presets were voiced against. Kept verbatim so those
  presets render bit-exact; new presets and DX7 imports use 1-32.
- Modulation matrix [6,6] where M[i,j]=True means Op_i modulates Op_j
- Carrier mask (bitmask): which operators sum to output
- Feedback operator: which op self-modulates (-1 if none).
  KNOWN APPROXIMATION: real DX7 algorithms 4 and 6 use *loop* feedback
  (op4->op6 and op5->op6 respectively); this engine models self-feedback
  only, so both are approximated as self-feedback on op6 (the loop reader).
  TODO(loop-fb): support distinct feedback reader/writer ops.

Key insight: DX7 algorithms are DAGs with at most one feedback edge.
Standard graph theory is sufficient - no knot invariants needed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class FMAlgorithm:
    """
    FM algorithm topology definition (DX7-compatible).

    Attributes:
        alg_id: Algorithm number (1-32 canonical DX7; 33-38 legacy WSE)
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
# Canonical DX7 algorithm spec (1-32)
# =============================================================================
#
# Format: alg_id: (name, carriers, edges, feedback_op) — ALL 1-BASED
#   carriers: tuple of carrier operator numbers (sum to audio out)
#   edges:    tuple of (modulator, target) pairs
#   feedback: operator with self-feedback
#
# Source of truth: decoded from Dexed/msfa fm_core.cc FmCore::algorithms[32]
# (flags: OUT_BUS 0=audio out, IN/OUT mod buses, FB_IN/FB_OUT; rows op6..op1).
# The conformance test embeds the same reference independently.
#
# Algorithms 4 and 6: real DX7 feedback is a loop (4->6, 5->6); approximated
# here as self-feedback on op6 — see module docstring.

_DX7_SPEC: Dict[int, Tuple[str, Tuple[int, ...], Tuple[Tuple[int, int], ...], int]] = {
    1:  ("Stack+Pair A",            (1, 3),          ((2, 1), (4, 3), (5, 4), (6, 5)), 6),
    2:  ("Stack+Pair B (fb2)",      (1, 3),          ((2, 1), (4, 3), (5, 4), (6, 5)), 2),
    3:  ("Dual Stacks",             (1, 4),          ((2, 1), (3, 2), (5, 4), (6, 5)), 6),
    4:  ("Dual Stacks (loop fb)",   (1, 4),          ((2, 1), (3, 2), (5, 4), (6, 5)), 6),
    5:  ("Three Pairs (E.Piano)",   (1, 3, 5),       ((2, 1), (4, 3), (6, 5)), 6),
    6:  ("Three Pairs (loop fb)",   (1, 3, 5),       ((2, 1), (4, 3), (6, 5)), 6),
    7:  ("Pair + Branched Stack",   (1, 3),          ((2, 1), (4, 3), (5, 3), (6, 5)), 6),
    8:  ("Pair + Branch (fb4)",     (1, 3),          ((2, 1), (4, 3), (5, 3), (6, 5)), 4),
    9:  ("Pair + Branch (fb2)",     (1, 3),          ((2, 1), (4, 3), (5, 3), (6, 5)), 2),
    10: ("Stack + Dual Mod (fb3)",  (1, 4),          ((2, 1), (3, 2), (5, 4), (6, 4)), 3),
    11: ("Stack + Dual Mod (fb6)",  (1, 4),          ((2, 1), (3, 2), (5, 4), (6, 4)), 6),
    12: ("Pair + Triple Mod (fb2)", (1, 3),          ((2, 1), (4, 3), (5, 3), (6, 3)), 2),
    13: ("Pair + Triple Mod (fb6)", (1, 3),          ((2, 1), (4, 3), (5, 3), (6, 3)), 6),
    14: ("Pair + Forked Stack",     (1, 3),          ((2, 1), (4, 3), (5, 4), (6, 4)), 6),
    15: ("Forked Stack (fb2)",      (1, 3),          ((2, 1), (4, 3), (5, 4), (6, 4)), 2),
    16: ("Three-Branch Mono",       (1,),            ((2, 1), (3, 1), (4, 3), (5, 1), (6, 5)), 6),
    17: ("Three-Branch Mono (fb2)", (1,),            ((2, 1), (3, 1), (4, 3), (5, 1), (6, 5)), 2),
    18: ("Three-Branch Deep",       (1,),            ((2, 1), (3, 1), (4, 1), (5, 4), (6, 5)), 3),
    19: ("Stack + Split Mod",       (1, 4, 5),       ((2, 1), (3, 2), (6, 4), (6, 5)), 6),
    20: ("Shared Mod + Pair",       (1, 2, 4),       ((3, 1), (3, 2), (5, 4), (6, 4)), 3),
    21: ("Dual Shared Mods",        (1, 2, 4, 5),    ((3, 1), (3, 2), (6, 4), (6, 5)), 3),
    22: ("Pair + Triple Split",     (1, 3, 4, 5),    ((2, 1), (6, 3), (6, 4), (6, 5)), 6),
    23: ("Free 1 + Pair + Split",   (1, 2, 4, 5),    ((3, 2), (6, 4), (6, 5)), 6),
    24: ("Triple Split + Free",     (1, 2, 3, 4, 5), ((6, 3), (6, 4), (6, 5)), 6),
    25: ("Dual Split + Free",       (1, 2, 3, 4, 5), ((6, 4), (6, 5)), 6),
    26: ("Two Pairs + Dual Mod",    (1, 2, 4),       ((3, 2), (5, 4), (6, 4)), 6),
    27: ("Two Pairs + Dual (fb3)",  (1, 2, 4),       ((3, 2), (5, 4), (6, 4)), 3),
    28: ("Stack + Free 6",          (1, 3, 6),       ((2, 1), (4, 3), (5, 4)), 5),
    29: ("Two Free + Two Pairs",    (1, 2, 3, 5),    ((4, 3), (6, 5)), 6),
    30: ("Free + Stack + Free 6",   (1, 2, 3, 6),    ((4, 3), (5, 4)), 5),
    31: ("Five Free + Pair",        (1, 2, 3, 4, 5), ((6, 5),), 6),
    32: ("All Carriers",            (1, 2, 3, 4, 5, 6), (), 6),
}

# =============================================================================
# Legacy WSE topologies (33-38)
# =============================================================================
#
# These are the structures the pre-2026-06-25 hand-written table shipped under
# DX7 numbers. They do not exist in the real DX7 chart, but 17 ear-tuned
# presets were voiced against them; renumbering the presets onto these ids
# preserves their sound bit-exact. Mapping (old wrong id -> legacy id):
#   old 1  -> 33   old 4 -> 34   old 5 -> 35   old 16 -> 36
#   old 18 -> 37   old 19 and old 22 (identical) -> 38

_LEGACY_SPEC: Dict[int, Tuple[str, Tuple[int, ...], Tuple[Tuple[int, int], ...], int]] = {
    33: ("Legacy 6-Op Chain (old 1)",       (1,),         ((2, 1), (3, 2), (4, 3), (5, 4), (6, 5)), 1),
    34: ("Legacy Dual Stack (old 4)",       (1, 4),       ((2, 1), (3, 2), (5, 4), (6, 5)), 1),
    35: ("Legacy E.Piano Stack (old 5)",    (1,),         ((2, 1), (3, 2), (4, 1), (5, 4), (6, 5)), 6),
    36: ("Legacy Flat Bright (old 16)",     (1,),         ((2, 1), (3, 1), (4, 1), (5, 1), (6, 1)), 6),
    37: ("Legacy Brass Branch (old 18)",    (1,),         ((2, 1), (3, 2), (4, 1), (5, 1), (6, 5)), 6),
    38: ("Legacy Fan-Out (old 19/22)",      (1, 2, 3, 4), ((5, 2), (5, 3), (5, 4), (6, 5)), 6),
}


def _topological_eval_order(edges: Tuple[Tuple[int, int], ...]) -> List[int]:
    """Order all 6 ops (0-indexed) so every modulator precedes its targets."""
    deps = {op: set() for op in range(6)}          # op -> its modulators
    for src, dst in edges:
        deps[dst - 1].add(src - 1)
    order: List[int] = []
    remaining = set(range(6))
    while remaining:
        ready = sorted(op for op in remaining if deps[op] <= set(order))
        if not ready:  # cycle — cannot happen for valid DAG specs
            raise ValueError(f"cyclic FM spec: {edges}")
        # Deterministic: highest op number first (modulators live high, matches
        # the historical 5..0 ordering for pure chains).
        nxt = ready[-1]
        order.append(nxt)
        remaining.discard(nxt)
    return order


def _build(alg_id: int, spec) -> FMAlgorithm:
    name, carriers, edges, fb = spec
    return FMAlgorithm(
        alg_id=alg_id,
        name=name,
        modulation_matrix=_make_matrix(*(((s - 1), (d - 1)) for s, d in edges)),
        carrier_mask=sum(1 << (c - 1) for c in carriers),
        feedback_op=fb - 1,
        eval_order=_topological_eval_order(edges),
    )


ALGORITHMS: Dict[int, FMAlgorithm] = {
    aid: _build(aid, spec)
    for aid, spec in {**_DX7_SPEC, **_LEGACY_SPEC}.items()
}


def get_algorithm(alg_id: int) -> FMAlgorithm:
    """
    Get algorithm by ID.

    Args:
        alg_id: Algorithm number — 1-32 canonical DX7, 33-38 legacy WSE

    Returns:
        FMAlgorithm instance

    Raises:
        KeyError: If algorithm ID not found
    """
    if alg_id not in ALGORITHMS:
        raise KeyError(
            f"Algorithm {alg_id} not found. Valid: 1-32 (DX7), 33-38 (legacy WSE)")
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
