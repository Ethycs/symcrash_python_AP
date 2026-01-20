"""
Central Registry (WSE-STD-REGISTRY v0)
======================================

Unified access point for all runtime registries in the Tower system.

Provides:
1. Single import for all registries
2. Global initialization and lifecycle management
3. Cross-registry lookups (e.g., find all resources for a preset)
4. Registry introspection and validation

Usage:
    from symcrash_python_ap.library.central_registry import get_central_registry

    registry = get_central_registry()

    # Access individual registries
    regime = registry.regimes.get(1)
    tree = registry.trees.get(TreeID.HARMONIC)
    control = registry.controls.get_entry(ControlID.TIMBRE)

    # Access primitives/constants
    block_size = registry.constants.BLOCK_SIZE
    q15_one = registry.constants.Q15_ONE

    # Introspection
    registry.list_all_regimes()
    registry.list_all_trees()
    registry.list_all_controls()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import IntEnum

# Import individual registries
from .regime_registry import (
    RegimeRegistry, RegimeGraphSpec, RegimeNodeSpec, EdgeSpec,
    get_regime_registry, create_minimal_regime_graph,
    create_standard_regime_graph, create_expressive_regime_graph,
)
from .tree_registry import (
    TreeRegistry, TreeFamily, TreeID, TreeCoeffProfile,
    get_tree_registry, create_balanced_tree, create_harmonic_tree,
    create_texture_tree, create_impact_tree, create_drone_tree,
)
from ..control_registry import (
    ControlRegistry, ControlEntry, ControlID, ValueType, SlewPolicy,
    CONTROL_REGISTRY_V0,
)

# Import primitives as a constants namespace
from .. import primitives


# =============================================================================
# OPERATOR REGISTRY
# =============================================================================

class OperatorID(IntEnum):
    """Canonical operator IDs for CMMM lanes."""
    # Core operators (0-7)
    NORMALIZE = 0       # Energy normalization
    COLOR = 1           # Timbral coloring
    TRANSPORT = 2       # Coefficient transport
    LOCAL = 3           # Local detail injection
    GLOBAL = 4          # Global structure
    CATASTROPHE = 5     # Controlled instability
    TAIL_FINE = 6       # Fine tail adjustment
    TAIL_COARSE = 7     # Coarse tail adjustment


@dataclass
class OperatorSpec:
    """Specification for a CMMM operator."""
    id: OperatorID
    name: str
    description: str
    cost_cycles: int        # Approximate FPGA cycles
    is_tail: bool = False   # True if this is a tail operator
    lane_index: int = 0     # Which lane this operator runs in


# Default operator specifications (from CMMM-2 spec)
OPERATOR_SPECS: List[OperatorSpec] = [
    OperatorSpec(OperatorID.NORMALIZE, "Normalize", "Energy normalization", 50, False, 0),
    OperatorSpec(OperatorID.COLOR, "Color", "Timbral coloring via spectral shaping", 80, False, 1),
    OperatorSpec(OperatorID.TRANSPORT, "Transport", "Coefficient transport/blur", 120, False, 2),
    OperatorSpec(OperatorID.LOCAL, "Local", "Local detail injection", 60, False, 3),
    OperatorSpec(OperatorID.GLOBAL, "Global", "Global structure shaping", 40, False, 4),
    OperatorSpec(OperatorID.CATASTROPHE, "Catastrophe", "Controlled instability", 100, False, 5),
    OperatorSpec(OperatorID.TAIL_FINE, "TailFine", "Fine-grained tail coefficient clamping", 30, True, 6),
    OperatorSpec(OperatorID.TAIL_COARSE, "TailCoarse", "Coarse tail coefficient zeroing", 20, True, 7),
]


class OperatorRegistry:
    """Registry for CMMM operator specifications."""

    def __init__(self, specs: List[OperatorSpec] = None):
        self._by_id: Dict[int, OperatorSpec] = {}
        specs = specs or OPERATOR_SPECS
        for spec in specs:
            self._by_id[spec.id] = spec

    def get(self, op_id: int) -> Optional[OperatorSpec]:
        """Get operator spec by ID."""
        return self._by_id.get(op_id)

    def get_by_name(self, name: str) -> Optional[OperatorSpec]:
        """Get operator spec by name."""
        for spec in self._by_id.values():
            if spec.name.lower() == name.lower():
                return spec
        return None

    def list_all(self) -> List[OperatorSpec]:
        """List all operator specs."""
        return list(self._by_id.values())

    def get_tail_operators(self) -> List[OperatorSpec]:
        """Get all tail operators."""
        return [s for s in self._by_id.values() if s.is_tail]

    def get_backbone_operators(self) -> List[OperatorSpec]:
        """Get all non-tail operators."""
        return [s for s in self._by_id.values() if not s.is_tail]

    def total_cost(self, op_ids: List[int]) -> int:
        """Calculate total cost for a set of operators."""
        return sum(self._by_id[oid].cost_cycles for oid in op_ids if oid in self._by_id)


# =============================================================================
# PATTERN REGISTRY (for node masks)
# =============================================================================

@dataclass
class PatternSpec:
    """Specification for a leaf node mask pattern."""
    id: int
    name: str
    mask_u32: int           # 32-bit mask for leaves 0-31 (lower half)
    mask_u32_hi: int = 0    # 32-bit mask for leaves 32-63 (upper half)
    description: str = ""

    @property
    def mask_u64(self) -> int:
        """Get full 64-bit mask."""
        return (self.mask_u32_hi << 32) | self.mask_u32

    @property
    def active_count(self) -> int:
        """Count of active leaves (K value)."""
        return bin(self.mask_u64).count('1')


# Default patterns from K ladder
DEFAULT_PATTERNS: List[PatternSpec] = [
    PatternSpec(0, "K64_FULL", 0xFFFFFFFF, 0xFFFFFFFF, "All 64 leaves active"),
    PatternSpec(1, "K32_EVEN", 0x55555555, 0x55555555, "32 leaves (even indices)"),
    PatternSpec(2, "K16_HARMONIC", 0x0000FFFF, 0x00000000, "16 leaves (harmonics)"),
    PatternSpec(3, "K8_CORE", 0x000000FF, 0x00000000, "8 core leaves"),
    PatternSpec(4, "K4_MINIMAL", 0x0000000F, 0x00000000, "4 minimal leaves"),
    PatternSpec(5, "K2_STEREO", 0x00000003, 0x00000000, "2 stereo leaves"),
    PatternSpec(6, "K1_MONO", 0x00000001, 0x00000000, "1 mono leaf"),
]


class PatternRegistry:
    """Registry for node mask patterns."""

    def __init__(self, patterns: List[PatternSpec] = None):
        self._by_id: Dict[int, PatternSpec] = {}
        self._by_name: Dict[str, int] = {}
        patterns = patterns or DEFAULT_PATTERNS
        for p in patterns:
            self._by_id[p.id] = p
            self._by_name[p.name.lower()] = p.id

    def get(self, pattern_id: int) -> Optional[PatternSpec]:
        """Get pattern by ID."""
        return self._by_id.get(pattern_id)

    def get_by_name(self, name: str) -> Optional[PatternSpec]:
        """Get pattern by name."""
        pid = self._by_name.get(name.lower())
        if pid is not None:
            return self._by_id.get(pid)
        return None

    def register(self, spec: PatternSpec) -> int:
        """Register a new pattern."""
        if spec.id in self._by_id:
            raise ValueError(f"Pattern ID {spec.id} already registered")
        self._by_id[spec.id] = spec
        self._by_name[spec.name.lower()] = spec.id
        return spec.id

    def list_all(self) -> List[PatternSpec]:
        """List all patterns."""
        return sorted(self._by_id.values(), key=lambda p: p.id)


# =============================================================================
# CENTRAL REGISTRY
# =============================================================================

class CentralRegistry:
    """
    Unified access point for all runtime registries.

    Provides single import for accessing:
    - Regime graphs (L2 state machines)
    - Tree families (wavelet coefficient structures)
    - Controls (user-facing parameters)
    - Operators (CMMM processing lanes)
    - Patterns (node mask configurations)
    - Constants (Q15, block size, etc.)
    """

    def __init__(self):
        """Initialize all registries."""
        self._regimes: Optional[RegimeRegistry] = None
        self._trees: Optional[TreeRegistry] = None
        self._controls: Optional[ControlRegistry] = None
        self._operators: Optional[OperatorRegistry] = None
        self._patterns: Optional[PatternRegistry] = None

    @property
    def regimes(self) -> RegimeRegistry:
        """Access regime registry (lazy init)."""
        if self._regimes is None:
            self._regimes = get_regime_registry()
        return self._regimes

    @property
    def trees(self) -> TreeRegistry:
        """Access tree registry (lazy init)."""
        if self._trees is None:
            self._trees = get_tree_registry()
        return self._trees

    @property
    def controls(self) -> ControlRegistry:
        """Access control registry (lazy init)."""
        if self._controls is None:
            self._controls = ControlRegistry()
        return self._controls

    @property
    def operators(self) -> OperatorRegistry:
        """Access operator registry (lazy init)."""
        if self._operators is None:
            self._operators = OperatorRegistry()
        return self._operators

    @property
    def patterns(self) -> PatternRegistry:
        """Access pattern registry (lazy init)."""
        if self._patterns is None:
            self._patterns = PatternRegistry()
        return self._patterns

    @property
    def constants(self):
        """Access primitives module as constants namespace."""
        return primitives

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def list_all_regimes(self) -> List[Tuple[int, str, str]]:
        """List all regimes as (id, hash, start_regime) tuples."""
        return [
            (rid, graph_hash, spec.start_regime)
            for rid, graph_hash, spec in self.regimes.list_all()
        ]

    def list_all_trees(self) -> List[Tuple[int, str, str]]:
        """List all trees as (id, name, description) tuples."""
        return [
            (t.id, t.name, t.description)
            for t in self.trees.list_all()
        ]

    def list_all_controls(self) -> List[Tuple[int, str, str]]:
        """List all controls as (id, name, binding) tuples."""
        return [
            (e.control_id, e.name, e.binding_target)
            for e in self.controls.all_entries()
        ]

    def list_all_operators(self) -> List[Tuple[int, str, int]]:
        """List all operators as (id, name, cost) tuples."""
        return [
            (o.id, o.name, o.cost_cycles)
            for o in self.operators.list_all()
        ]

    def list_all_patterns(self) -> List[Tuple[int, str, int]]:
        """List all patterns as (id, name, active_count) tuples."""
        return [
            (p.id, p.name, p.active_count)
            for p in self.patterns.list_all()
        ]

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_preset_resources(
        self,
        regime_graph_id: Optional[int] = None,
        tree_a_id: Optional[int] = None,
        tree_b_id: Optional[int] = None,
        pattern_id: Optional[int] = None,
    ) -> List[str]:
        """
        Validate that a preset's resource references exist.

        Returns list of error messages (empty if valid).
        """
        errors = []

        if regime_graph_id is not None and self.regimes.get(regime_graph_id) is None:
            errors.append(f"Regime graph ID {regime_graph_id} not found")

        if tree_a_id is not None and self.trees.get(tree_a_id) is None:
            errors.append(f"Tree A ID {tree_a_id} not found")

        if tree_b_id is not None and self.trees.get(tree_b_id) is None:
            errors.append(f"Tree B ID {tree_b_id} not found")

        if pattern_id is not None and self.patterns.get(pattern_id) is None:
            errors.append(f"Pattern ID {pattern_id} not found")

        return errors

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summary(self) -> str:
        """Generate human-readable registry summary."""
        lines = [
            "=== Central Registry Summary ===",
            "",
            f"Regimes:   {len(self.regimes.list_all())} registered",
            f"Trees:     {len(self.trees.list_all())} registered",
            f"Controls:  {len(self.controls.all_entries())} registered",
            f"Operators: {len(self.operators.list_all())} registered",
            f"Patterns:  {len(self.patterns.list_all())} registered",
            "",
            "Constants:",
            f"  Q15_ONE = {primitives.Q15_ONE}",
            f"  Q15_HALF = {primitives.Q15_HALF}",
            f"  BLOCK_SIZE = {primitives.BLOCK_SIZE}",
            f"  NUM_LEAVES = {primitives.NUM_LEAVES}",
            f"  SAMPLE_RATE_HZ = {primitives.SAMPLE_RATE_HZ}",
        ]
        return "\n".join(lines)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_global_central_registry: Optional[CentralRegistry] = None


def get_central_registry() -> CentralRegistry:
    """Get the global central registry instance."""
    global _global_central_registry
    if _global_central_registry is None:
        _global_central_registry = CentralRegistry()
    return _global_central_registry


def reset_central_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _global_central_registry
    _global_central_registry = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Central registry
    "CentralRegistry",
    "get_central_registry",
    "reset_central_registry",
    # Operator registry
    "OperatorID",
    "OperatorSpec",
    "OperatorRegistry",
    "OPERATOR_SPECS",
    # Pattern registry
    "PatternSpec",
    "PatternRegistry",
    "DEFAULT_PATTERNS",
    # Re-exports from sub-registries
    "RegimeRegistry",
    "RegimeGraphSpec",
    "TreeRegistry",
    "TreeFamily",
    "TreeID",
    "ControlRegistry",
    "ControlID",
]
