"""
Regime Registry (WSE-STD-070 Tower Runtime)
============================================

Registry for regime graph specifications. Replaces hash-only IDs with
actual storage and resolution of regime graphs.

Per the Tower plan:
- Regime graphs define L2 state machine topology
- Each regime has an emit template reference (T_IDLE, T_ATTACK, etc.)
- Edges define transitions between regimes

Registry provides:
1. Storage of RegimeGraphSpec by ID
2. Hash-based ID generation for determinism
3. Lookup by ID or hash
4. Validation of regime graph structure
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# EDGE TRANSITION TYPES
# =============================================================================

class EdgeTransitionType(Enum):
    """Transition types between regimes (from InstrumentIR Tower Spec)."""
    HARD = "hard"           # Immediate switch, no crossfade
    SOFT = "soft"           # Smooth crossfade over duration_blocks
    CHAOTIC = "chaotic"     # Stochastic switching with jitter
    CORRIDOR = "corridor"   # Entry into degradation corridor
    RECOVERY = "recovery"   # Return from instability to stable


# =============================================================================
# REGIME GRAPH DATA STRUCTURES
# =============================================================================

@dataclass
class Hysteresis:
    """Hysteresis thresholds for transitions."""
    enter: float = 0.0
    exit: float = 0.0


@dataclass
class EdgeSpec:
    """
    Defines a transition between two regimes.

    Attributes:
        src_regime: Source regime ID (e.g., "R0_IDLE")
        dst_regime: Destination regime ID (e.g., "R1_ATTACK")
        transition_type: Type of transition (hard, soft, chaotic, etc.)
        guard: Optional guard condition expression
        hysteresis: Optional hysteresis thresholds
        min_dwell_blocks: Minimum blocks before transition allowed
        duration_blocks: Crossfade duration for soft transitions
        overlay: Optional overlay template during transition
    """
    src_regime: str
    dst_regime: str
    transition_type: str = "soft"
    guard: str = ""
    hysteresis: Optional[Hysteresis] = None
    min_dwell_blocks: int = 0
    duration_blocks: int = 4
    overlay: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for hashing."""
        d = {
            "src": self.src_regime,
            "dst": self.dst_regime,
            "type": self.transition_type,
            "guard": self.guard,
            "min_dwell": self.min_dwell_blocks,
            "duration": self.duration_blocks,
        }
        if self.hysteresis:
            d["hysteresis"] = {"enter": self.hysteresis.enter, "exit": self.hysteresis.exit}
        if self.overlay:
            d["overlay"] = self.overlay
        return d


@dataclass
class RegimeNodeSpec:
    """Specification for a single regime node."""
    id: str                        # Regime ID (e.g., "R0_IDLE")
    kind: str = "Idle"             # Regime kind (Idle, Transient, StableTone, etc.)
    emit_template: str = ""        # Reference to emit template (e.g., "T_IDLE")
    dwell_ms: Optional[float] = None  # Optional dwell time for transient regimes


@dataclass
class RegimeGraphSpec:
    """
    Full state machine topology for regime system.

    The regime graph defines how a voice transitions between
    different behavioral states during performance.
    """
    start_regime: str = "R0_IDLE"
    nodes: List[RegimeNodeSpec] = field(default_factory=list)
    edges: List[EdgeSpec] = field(default_factory=list)

    def compute_hash(self) -> str:
        """Compute deterministic hash of the graph specification."""
        import json
        # Build canonical representation
        data = {
            "start": self.start_regime,
            "nodes": [
                {"id": n.id, "kind": n.kind, "emit": n.emit_template}
                for n in sorted(self.nodes, key=lambda x: x.id)
            ],
            "edges": [
                e.to_dict()
                for e in sorted(self.edges, key=lambda x: (x.src_regime, x.dst_regime))
            ],
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def validate(self) -> List[str]:
        """Validate the regime graph structure."""
        errors = []

        # Check start regime exists
        node_ids = {n.id for n in self.nodes}
        if self.start_regime not in node_ids:
            errors.append(f"Start regime '{self.start_regime}' not found in nodes")

        # Check all edge endpoints exist
        for edge in self.edges:
            if edge.src_regime != "any" and edge.src_regime not in node_ids:
                errors.append(f"Edge source '{edge.src_regime}' not found")
            if edge.dst_regime not in node_ids:
                errors.append(f"Edge destination '{edge.dst_regime}' not found")

        # Check for unreachable nodes
        reachable = {self.start_regime}
        changed = True
        while changed:
            changed = False
            for edge in self.edges:
                if edge.src_regime in reachable or edge.src_regime == "any":
                    if edge.dst_regime not in reachable:
                        reachable.add(edge.dst_regime)
                        changed = True

        unreachable = node_ids - reachable
        if unreachable:
            errors.append(f"Unreachable nodes: {unreachable}")

        return errors


# =============================================================================
# REGIME REGISTRY
# =============================================================================

class RegimeRegistry:
    """
    Registry for regime graph specifications.

    Provides storage, lookup, and validation of regime graphs.
    Uses hash-based IDs for determinism.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._by_id: Dict[int, RegimeGraphSpec] = {}
        self._by_hash: Dict[str, int] = {}
        self._next_id: int = 1

    def register(self, spec: RegimeGraphSpec, regime_graph_id: int = None) -> Tuple[int, str]:
        """
        Register a regime graph specification.

        Args:
            spec: The regime graph specification
            regime_graph_id: Optional explicit ID (auto-generated if None)

        Returns:
            (regime_graph_id, hash) tuple
        """
        # Compute hash
        graph_hash = spec.compute_hash()

        # Check if already registered by hash
        if graph_hash in self._by_hash:
            existing_id = self._by_hash[graph_hash]
            return existing_id, graph_hash

        # Assign ID
        if regime_graph_id is not None:
            if regime_graph_id in self._by_id:
                # ID collision - check if same content
                if self._by_id[regime_graph_id].compute_hash() != graph_hash:
                    raise ValueError(f"ID {regime_graph_id} already registered with different content")
                return regime_graph_id, graph_hash
            rid = regime_graph_id
        else:
            rid = self._next_id
            self._next_id += 1

        # Store
        self._by_id[rid] = spec
        self._by_hash[graph_hash] = rid

        return rid, graph_hash

    def get(self, regime_graph_id: int) -> Optional[RegimeGraphSpec]:
        """Look up regime graph by ID."""
        return self._by_id.get(regime_graph_id)

    def get_by_hash(self, graph_hash: str) -> Optional[RegimeGraphSpec]:
        """Look up regime graph by hash."""
        rid = self._by_hash.get(graph_hash)
        if rid is not None:
            return self._by_id.get(rid)
        return None

    def get_id_by_hash(self, graph_hash: str) -> Optional[int]:
        """Get ID from hash."""
        return self._by_hash.get(graph_hash)

    def list_all(self) -> List[Tuple[int, str, RegimeGraphSpec]]:
        """List all registered regime graphs as (id, hash, spec) tuples."""
        return [
            (rid, spec.compute_hash(), spec)
            for rid, spec in self._by_id.items()
        ]

    def clear(self) -> None:
        """Clear all registrations."""
        self._by_id.clear()
        self._by_hash.clear()
        self._next_id = 1


# =============================================================================
# DEFAULT REGIME GRAPHS
# =============================================================================

def create_minimal_regime_graph() -> RegimeGraphSpec:
    """Create minimal 2-state regime graph (IDLE ↔ ACTIVE)."""
    return RegimeGraphSpec(
        start_regime="R0_IDLE",
        nodes=[
            RegimeNodeSpec(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
            RegimeNodeSpec(id="R1_ACTIVE", kind="StableTone", emit_template="T_STABLE"),
        ],
        edges=[
            EdgeSpec(src_regime="R0_IDLE", dst_regime="R1_ACTIVE",
                     transition_type="hard", guard="gate_rise"),
            EdgeSpec(src_regime="R1_ACTIVE", dst_regime="R0_IDLE",
                     transition_type="soft", guard="gate_fall", duration_blocks=8),
        ],
    )


def create_standard_regime_graph() -> RegimeGraphSpec:
    """Create standard 4-state regime graph (IDLE → ATTACK → STABLE → RELEASE)."""
    return RegimeGraphSpec(
        start_regime="R0_IDLE",
        nodes=[
            RegimeNodeSpec(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
            RegimeNodeSpec(id="R1_ATTACK", kind="Transient", emit_template="T_ATTACK", dwell_ms=20),
            RegimeNodeSpec(id="R2_STABLE", kind="StableTone", emit_template="T_STABLE"),
            RegimeNodeSpec(id="R3_RELEASE", kind="Release", emit_template="T_RELEASE"),
        ],
        edges=[
            EdgeSpec(src_regime="R0_IDLE", dst_regime="R1_ATTACK",
                     transition_type="hard", guard="gate_rise"),
            EdgeSpec(src_regime="R1_ATTACK", dst_regime="R2_STABLE",
                     transition_type="soft", min_dwell_blocks=15, duration_blocks=8),
            EdgeSpec(src_regime="any", dst_regime="R3_RELEASE",
                     transition_type="soft", guard="gate_fall", duration_blocks=4),
            EdgeSpec(src_regime="R3_RELEASE", dst_regime="R0_IDLE",
                     transition_type="soft", min_dwell_blocks=30, duration_blocks=4),
        ],
    )


def create_expressive_regime_graph() -> RegimeGraphSpec:
    """Create expressive 5-state regime graph with glitch state."""
    return RegimeGraphSpec(
        start_regime="R0_IDLE",
        nodes=[
            RegimeNodeSpec(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
            RegimeNodeSpec(id="R1_ATTACK", kind="Transient", emit_template="T_ATTACK", dwell_ms=20),
            RegimeNodeSpec(id="R2_STABLE", kind="StableTone", emit_template="T_STABLE"),
            RegimeNodeSpec(id="R3_GLITCH", kind="GlitchChatter", emit_template="T_GLITCH"),
            RegimeNodeSpec(id="R4_RELEASE", kind="Release", emit_template="T_RELEASE"),
        ],
        edges=[
            EdgeSpec(src_regime="R0_IDLE", dst_regime="R1_ATTACK",
                     transition_type="hard", guard="gate_rise"),
            EdgeSpec(src_regime="R1_ATTACK", dst_regime="R2_STABLE",
                     transition_type="soft", min_dwell_blocks=12, duration_blocks=8),
            EdgeSpec(src_regime="R2_STABLE", dst_regime="R3_GLITCH",
                     transition_type="chaotic", guard="intent.glitch > 0.7",
                     hysteresis=Hysteresis(enter=0.7, exit=0.5)),
            EdgeSpec(src_regime="R3_GLITCH", dst_regime="R2_STABLE",
                     transition_type="recovery", guard="intent.calm > 0.6"),
            EdgeSpec(src_regime="any", dst_regime="R4_RELEASE",
                     transition_type="soft", guard="gate_fall", duration_blocks=4),
            EdgeSpec(src_regime="R4_RELEASE", dst_regime="R0_IDLE",
                     transition_type="soft", min_dwell_blocks=30, duration_blocks=4),
        ],
    )


# Global registry instance
_global_regime_registry: Optional[RegimeRegistry] = None


def get_regime_registry() -> RegimeRegistry:
    """Get the global regime registry instance."""
    global _global_regime_registry
    if _global_regime_registry is None:
        _global_regime_registry = RegimeRegistry()
        # Pre-register standard graphs
        _global_regime_registry.register(create_minimal_regime_graph(), regime_graph_id=1)
        _global_regime_registry.register(create_standard_regime_graph(), regime_graph_id=2)
        _global_regime_registry.register(create_expressive_regime_graph(), regime_graph_id=3)
    return _global_regime_registry


__all__ = [
    "EdgeTransitionType",
    "Hysteresis",
    "EdgeSpec",
    "RegimeNodeSpec",
    "RegimeGraphSpec",
    "RegimeRegistry",
    "create_minimal_regime_graph",
    "create_standard_regime_graph",
    "create_expressive_regime_graph",
    "get_regime_registry",
]
