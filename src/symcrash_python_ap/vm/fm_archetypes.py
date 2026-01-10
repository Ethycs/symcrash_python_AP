"""
FM Topology Archetypes - 8 Canonical FM Structures
===================================================

Per WSE-STD-FM-TOPOLOGY-v0 Section 8:
- 8 canonical archetypes that cover all 32 DX7 algorithms
- Edge lists with Q15 weights for continuous morphing
- DX7 → archetype mapping table

All 32 DX7 algorithms map to one of these 8 structural families,
enabling continuous topology morphing via A/B blend.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np


# Archetype IDs
ARCH_CHAIN = 0      # Serial 6-op chain
ARCH_STACK = 1      # 5 modulators into 1 carrier
ARCH_DUAL = 2       # Two parallel 3-op chains
ARCH_SPLIT = 3      # Modulator tree to multiple carriers
ARCH_YPAIR = 4      # Y-branch structure
ARCH_CLOUD = 5      # Parallel modulators (thick)
ARCH_ADDITIVE = 6   # All carriers, no modulation
ARCH_FEEDBACK = 7   # Chain with enhanced feedback

Q15_ONE = 32767
Q15_3_4 = 24576     # 0.75
Q15_1_2 = 16384     # 0.5


@dataclass
class FMArchetype:
    """
    Canonical FM topology archetype.

    Attributes:
        arch_id: Archetype ID (0-7)
        name: Human-readable name
        edges: List of (src, dst, weight_q15) modulation edges
        carrier_mask: Bitmask of carrier operators
        feedback_op: Which operator has feedback (-1 if none)
        feedback_enhanced: If True, apply 2x feedback depth
    """
    arch_id: int
    name: str
    edges: List[Tuple[int, int, int]]  # (src, dst, weight_q15)
    carrier_mask: int
    feedback_op: int
    feedback_enhanced: bool = False

    def get_edge_weight(self, src: int, dst: int) -> int:
        """Get modulation weight for edge (src, dst), or 0 if not present."""
        for s, d, w in self.edges:
            if s == src and d == dst:
                return w
        return 0

    def has_edge(self, src: int, dst: int) -> bool:
        """Check if edge exists."""
        return self.get_edge_weight(src, dst) > 0

    def get_carriers(self) -> List[int]:
        """Get list of carrier operator indices."""
        return [i for i in range(6) if self.carrier_mask & (1 << i)]


# =============================================================================
# 8 Canonical Archetypes
# =============================================================================

ARCHETYPES: Dict[int, FMArchetype] = {
    # -------------------------------------------------------------------------
    # 0: CHAIN - Serial 6-op chain
    # [6] → [5] → [4] → [3] → [2] → [1*]
    #                                 ↺fb
    # -------------------------------------------------------------------------
    ARCH_CHAIN: FMArchetype(
        arch_id=ARCH_CHAIN,
        name="Chain",
        edges=[
            (5, 4, Q15_ONE),  # Op6 → Op5
            (4, 3, Q15_ONE),  # Op5 → Op4
            (3, 2, Q15_ONE),  # Op4 → Op3
            (2, 1, Q15_ONE),  # Op3 → Op2
            (1, 0, Q15_ONE),  # Op2 → Op1
        ],
        carrier_mask=0b000001,  # Op1
        feedback_op=0,
    ),

    # -------------------------------------------------------------------------
    # 1: STACK - 5 modulators into 1 carrier
    # [6] ─┐
    # [5] ─┤
    # [4] ─┼──→ [1*]
    # [3] ─┤     ↺fb
    # [2] ─┘
    # -------------------------------------------------------------------------
    ARCH_STACK: FMArchetype(
        arch_id=ARCH_STACK,
        name="Stack",
        edges=[
            (5, 0, Q15_ONE),  # Op6 → Op1
            (4, 0, Q15_ONE),  # Op5 → Op1
            (3, 0, Q15_ONE),  # Op4 → Op1
            (2, 0, Q15_ONE),  # Op3 → Op1
            (1, 0, Q15_ONE),  # Op2 → Op1
        ],
        carrier_mask=0b000001,  # Op1
        feedback_op=5,
    ),

    # -------------------------------------------------------------------------
    # 2: DUAL - Two parallel 3-op chains
    # [6] → [5] → [4*]
    # [3] → [2] → [1*]
    # -------------------------------------------------------------------------
    ARCH_DUAL: FMArchetype(
        arch_id=ARCH_DUAL,
        name="Dual",
        edges=[
            (5, 4, Q15_ONE),  # Op6 → Op5
            (4, 3, Q15_ONE),  # Op5 → Op4
            (2, 1, Q15_ONE),  # Op3 → Op2
            (1, 0, Q15_ONE),  # Op2 → Op1
        ],
        carrier_mask=0b001001,  # Op1, Op4
        feedback_op=5,
    ),

    # -------------------------------------------------------------------------
    # 3: SPLIT - Modulator tree to multiple carriers
    #               ┌──→ [4*]
    # [6] → [5] ────┼──→ [3*]
    #               ├──→ [2*]
    #               └──→ [1*]
    # -------------------------------------------------------------------------
    ARCH_SPLIT: FMArchetype(
        arch_id=ARCH_SPLIT,
        name="Split",
        edges=[
            (5, 4, Q15_ONE),  # Op6 → Op5
            (4, 3, Q15_ONE),  # Op5 → Op4
            (4, 2, Q15_ONE),  # Op5 → Op3
            (4, 1, Q15_ONE),  # Op5 → Op2
        ],
        carrier_mask=0b001111,  # Op1-4
        feedback_op=5,
    ),

    # -------------------------------------------------------------------------
    # 4: YPAIR - Y-branch structure
    #         ┌──→ [5] ──┐
    # [6] ────┤          ├──→ [3*]
    #         └──→ [4] ──┘
    #               [2] → [1*]
    # -------------------------------------------------------------------------
    ARCH_YPAIR: FMArchetype(
        arch_id=ARCH_YPAIR,
        name="Y-Pair",
        edges=[
            (5, 4, Q15_ONE),  # Op6 → Op5
            (5, 3, Q15_ONE),  # Op6 → Op4
            (4, 2, Q15_ONE),  # Op5 → Op3
            (3, 2, Q15_ONE),  # Op4 → Op3
            (1, 0, Q15_ONE),  # Op2 → Op1
        ],
        carrier_mask=0b000101,  # Op1, Op3
        feedback_op=5,
    ),

    # -------------------------------------------------------------------------
    # 5: CLOUD - Parallel modulators (thick, weighted)
    # [6] ═══╗
    # [5] ═══╬═══→ [1*]   (weighted parallel)
    # [4] ═══╣
    # [3] ═══╣
    # [2] ═══╝
    # -------------------------------------------------------------------------
    ARCH_CLOUD: FMArchetype(
        arch_id=ARCH_CLOUD,
        name="Cloud",
        edges=[
            (5, 0, Q15_ONE),    # Op6 → Op1 (full)
            (4, 0, Q15_3_4),    # Op5 → Op1 (0.75)
            (3, 0, Q15_1_2),    # Op4 → Op1 (0.5)
            (2, 0, Q15_1_2),    # Op3 → Op1 (0.5)
            (1, 0, Q15_3_4),    # Op2 → Op1 (0.75)
        ],
        carrier_mask=0b000001,  # Op1
        feedback_op=5,
    ),

    # -------------------------------------------------------------------------
    # 6: ADDITIVE - All carriers, no modulation
    # [6*]  [5*]  [4*]  [3*]  [2*]  [1*]
    #   │     │     │     │     │     │
    #   └─────┴─────┴─────┴─────┴─────┘
    #                   ↓
    #                 OUTPUT
    # -------------------------------------------------------------------------
    ARCH_ADDITIVE: FMArchetype(
        arch_id=ARCH_ADDITIVE,
        name="Additive",
        edges=[],  # No modulation
        carrier_mask=0b111111,  # All carriers
        feedback_op=5,
    ),

    # -------------------------------------------------------------------------
    # 7: FEEDBACK - Chain with enhanced feedback
    # [6] → [5] → [4] → [3] → [2] → [1*]
    #                                 ⟲2x
    # -------------------------------------------------------------------------
    ARCH_FEEDBACK: FMArchetype(
        arch_id=ARCH_FEEDBACK,
        name="Feedback",
        edges=[
            (5, 4, Q15_ONE),
            (4, 3, Q15_ONE),
            (3, 2, Q15_ONE),
            (2, 1, Q15_ONE),
            (1, 0, Q15_ONE),
        ],
        carrier_mask=0b000001,  # Op1
        feedback_op=0,
        feedback_enhanced=True,  # 2x feedback depth
    ),
}


# =============================================================================
# DX7 → Archetype Mapping
# =============================================================================

DX7_TO_ARCHETYPE: Dict[int, int] = {
    # CHAIN (0) - Serial chains
    1: ARCH_CHAIN,
    2: ARCH_CHAIN,

    # DUAL (2) - Parallel chains
    3: ARCH_DUAL,
    4: ARCH_DUAL,
    9: ARCH_DUAL,
    10: ARCH_DUAL,
    11: ARCH_DUAL,
    12: ARCH_DUAL,
    13: ARCH_DUAL,
    30: ARCH_DUAL,

    # YPAIR (4) - Y-branch structures
    5: ARCH_YPAIR,
    6: ARCH_YPAIR,
    20: ARCH_YPAIR,

    # SPLIT (3) - Multi-carrier splits
    7: ARCH_SPLIT,
    8: ARCH_SPLIT,
    14: ARCH_SPLIT,
    15: ARCH_SPLIT,
    19: ARCH_SPLIT,
    21: ARCH_SPLIT,
    22: ARCH_SPLIT,
    28: ARCH_SPLIT,
    29: ARCH_SPLIT,

    # STACK (1) - All-to-one
    16: ARCH_STACK,
    17: ARCH_STACK,

    # CLOUD (5) - Thick parallel
    18: ARCH_CLOUD,

    # ADDITIVE (6) - Mostly/all carriers
    23: ARCH_ADDITIVE,
    24: ARCH_ADDITIVE,
    25: ARCH_ADDITIVE,
    26: ARCH_ADDITIVE,
    27: ARCH_ADDITIVE,
    31: ARCH_ADDITIVE,
    32: ARCH_ADDITIVE,
}


def get_archetype(arch_id: int) -> FMArchetype:
    """
    Get archetype by ID (0-7).

    Args:
        arch_id: Archetype ID (0-7)

    Returns:
        FMArchetype instance

    Raises:
        KeyError: If archetype ID not found
    """
    if arch_id not in ARCHETYPES:
        raise KeyError(f"Archetype {arch_id} not found. Valid range: 0-7")
    return ARCHETYPES[arch_id]


def dx7_to_archetype(alg_id: int) -> int:
    """
    Map DX7 algorithm to archetype ID.

    Args:
        alg_id: DX7 algorithm number (1-32)

    Returns:
        Archetype ID (0-7)

    Raises:
        KeyError: If algorithm ID not found
    """
    if alg_id not in DX7_TO_ARCHETYPE:
        raise KeyError(f"DX7 algorithm {alg_id} not found. Valid range: 1-32")
    return DX7_TO_ARCHETYPE[alg_id]


# =============================================================================
# Blend Functions
# =============================================================================

def blend_edge_weight(weight_a: int, weight_b: int, blend_q15: int) -> int:
    """
    Interpolate edge weight between A and B topologies.

    Args:
        weight_a: Edge weight in topology A (Q15)
        weight_b: Edge weight in topology B (Q15)
        blend_q15: Blend amount (0=A, 32767=B)

    Returns:
        Blended edge weight (Q15)
    """
    inv_blend = Q15_ONE - blend_q15
    result = (weight_a * inv_blend + weight_b * blend_q15) >> 15
    return result


def blend_carrier_level(is_carrier_a: bool, is_carrier_b: bool,
                        level_q15: int, blend_q15: int) -> int:
    """
    Blend carrier output level based on topology transition.

    Args:
        is_carrier_a: Is this op a carrier in topology A?
        is_carrier_b: Is this op a carrier in topology B?
        level_q15: Base operator level (Q15)
        blend_q15: Blend amount (0=A, 32767=B)

    Returns:
        Blended carrier output level (Q15)
    """
    if is_carrier_a and is_carrier_b:
        return level_q15  # Carrier in both - full level
    elif is_carrier_a and not is_carrier_b:
        # Fading out as carrier
        return (level_q15 * (Q15_ONE - blend_q15)) >> 15
    elif not is_carrier_a and is_carrier_b:
        # Fading in as carrier
        return (level_q15 * blend_q15) >> 15
    else:
        return 0  # Carrier in neither


def compute_blended_matrix(topo_a_id: int, topo_b_id: int,
                           blend_q15: int) -> np.ndarray:
    """
    Compute blended modulation matrix from two archetypes.

    Args:
        topo_a_id: Source archetype ID (0-7)
        topo_b_id: Target archetype ID (0-7)
        blend_q15: Blend amount (0=A, 32767=B)

    Returns:
        [6,6] modulation weight matrix (Q15)
    """
    arch_a = get_archetype(topo_a_id)
    arch_b = get_archetype(topo_b_id)

    mat = np.zeros((6, 6), dtype=np.int16)

    # Collect all edges from both topologies
    all_edges = set()
    for src, dst, _ in arch_a.edges:
        all_edges.add((src, dst))
    for src, dst, _ in arch_b.edges:
        all_edges.add((src, dst))

    # Blend each edge
    for src, dst in all_edges:
        w_a = arch_a.get_edge_weight(src, dst)
        w_b = arch_b.get_edge_weight(src, dst)
        mat[src, dst] = blend_edge_weight(w_a, w_b, blend_q15)

    return mat


def compute_blended_carriers(topo_a_id: int, topo_b_id: int,
                             blend_q15: int) -> Tuple[int, np.ndarray]:
    """
    Compute blended carrier mask and fade levels.

    Args:
        topo_a_id: Source archetype ID (0-7)
        topo_b_id: Target archetype ID (0-7)
        blend_q15: Blend amount (0=A, 32767=B)

    Returns:
        (union_mask, fade_levels) where:
        - union_mask: Bitmask of all carriers in either topology
        - fade_levels: [6] Q15 level multipliers for carrier fade
    """
    arch_a = get_archetype(topo_a_id)
    arch_b = get_archetype(topo_b_id)

    mask_a = arch_a.carrier_mask
    mask_b = arch_b.carrier_mask

    # Union of carriers
    union_mask = mask_a | mask_b

    # Compute per-carrier fade level
    fade_levels = np.zeros(6, dtype=np.int16)
    for op in range(6):
        is_a = bool(mask_a & (1 << op))
        is_b = bool(mask_b & (1 << op))
        fade_levels[op] = blend_carrier_level(is_a, is_b, Q15_ONE, blend_q15)

    return union_mask, fade_levels


def blend_feedback_op(fb_op_a: int, fb_op_b: int, blend_q15: int,
                      op_outputs: np.ndarray) -> np.ndarray:
    """
    Blend feedback from two different operators.

    Args:
        fb_op_a: Feedback operator in topology A
        fb_op_b: Feedback operator in topology B
        blend_q15: Blend amount (0=A, 32767=B)
        op_outputs: [6, K] operator outputs

    Returns:
        [K] blended feedback signal
    """
    if fb_op_a == fb_op_b:
        return op_outputs[fb_op_a].copy()

    # Handle negative (no feedback) cases
    if fb_op_a < 0 and fb_op_b < 0:
        return np.zeros_like(op_outputs[0])
    elif fb_op_a < 0:
        # Fade in B feedback
        return (op_outputs[fb_op_b] * blend_q15) >> 15
    elif fb_op_b < 0:
        # Fade out A feedback
        return (op_outputs[fb_op_a] * (Q15_ONE - blend_q15)) >> 15

    # Crossfade between feedback sources
    inv_blend = Q15_ONE - blend_q15
    fb_a = (op_outputs[fb_op_a].astype(np.int32) * inv_blend) >> 15
    fb_b = (op_outputs[fb_op_b].astype(np.int32) * blend_q15) >> 15
    return (fb_a + fb_b).astype(np.int16)


__all__ = [
    # Archetype IDs
    'ARCH_CHAIN',
    'ARCH_STACK',
    'ARCH_DUAL',
    'ARCH_SPLIT',
    'ARCH_YPAIR',
    'ARCH_CLOUD',
    'ARCH_ADDITIVE',
    'ARCH_FEEDBACK',
    # Classes
    'FMArchetype',
    # Data
    'ARCHETYPES',
    'DX7_TO_ARCHETYPE',
    # Functions
    'get_archetype',
    'dx7_to_archetype',
    'blend_edge_weight',
    'blend_carrier_level',
    'compute_blended_matrix',
    'compute_blended_carriers',
    'blend_feedback_op',
]
