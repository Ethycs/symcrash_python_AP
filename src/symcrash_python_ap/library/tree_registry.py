"""
Tree Registry (WSE-STD-070 Tower Runtime)
=========================================

Registry for tree family specifications. Replaces hash-only IDs with
actual storage and resolution of tree families.

Per the Tower plan:
- Tree families define wavelet coefficient tree structures
- Each tree affects timbre character (balanced, harmonic, texture, etc.)
- Trees are blended via tree_a, tree_b, and blend controls

Registry provides:
1. Storage of TreeFamily by ID
2. Hash-based ID generation for determinism
3. Lookup by ID or hash
4. Validation of tree family structure

Canonical Tree IDs (from existing codebase):
- 0: Balanced (default)
- 1: Harmonic
- 2: Texture
- 3: Impact
- 4: Drone
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum


# =============================================================================
# TREE FAMILY CONSTANTS
# =============================================================================

class TreeID(IntEnum):
    """Canonical tree IDs."""
    BALANCED = 0
    HARMONIC = 1
    TEXTURE = 2
    IMPACT = 3
    DRONE = 4


# =============================================================================
# TREE FAMILY DATA STRUCTURES
# =============================================================================

@dataclass
class TreeCoeffProfile:
    """
    Defines how coefficients are distributed across tree leaves.

    Attributes:
        leaf_weights: Weight for each of 64 leaves (normalized)
        band_emphasis: Which frequency bands are emphasized
        temporal_spread: How coefficients spread over time
    """
    leaf_weights: List[float] = field(default_factory=lambda: [1.0] * 64)
    band_emphasis: str = "balanced"  # "low", "mid", "high", "balanced"
    temporal_spread: float = 1.0     # 0.0 = tight, 1.0 = spread

    def compute_hash(self) -> str:
        """Compute deterministic hash."""
        import json
        data = {
            "weights": [round(w, 4) for w in self.leaf_weights],
            "band": self.band_emphasis,
            "spread": round(self.temporal_spread, 4),
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:8]


@dataclass
class TreeFamily:
    """
    Defines a tree family for wavelet coefficient structure.

    A tree family determines the timbre character by specifying
    how coefficients are organized and blended across leaves.
    """
    id: int                        # Canonical tree ID
    name: str                      # Human-readable name
    description: str = ""          # Description of timbre character
    coeff_profile: TreeCoeffProfile = field(default_factory=TreeCoeffProfile)
    blend_mode: str = "linear"     # How to blend with other trees

    def compute_hash(self) -> str:
        """Compute deterministic hash of tree family."""
        import json
        data = {
            "id": self.id,
            "name": self.name,
            "profile": self.coeff_profile.compute_hash(),
            "blend": self.blend_mode,
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# =============================================================================
# TREE REGISTRY
# =============================================================================

class TreeRegistry:
    """
    Registry for tree family specifications.

    Provides storage, lookup, and validation of tree families.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._by_id: Dict[int, TreeFamily] = {}
        self._by_hash: Dict[str, int] = {}

    def register(self, family: TreeFamily) -> Tuple[int, str]:
        """
        Register a tree family.

        Args:
            family: The tree family specification

        Returns:
            (tree_family_id, hash) tuple
        """
        tree_hash = family.compute_hash()

        # Check if already registered
        if family.id in self._by_id:
            existing = self._by_id[family.id]
            if existing.compute_hash() != tree_hash:
                raise ValueError(f"Tree ID {family.id} already registered with different content")
            return family.id, tree_hash

        # Store
        self._by_id[family.id] = family
        self._by_hash[tree_hash] = family.id

        return family.id, tree_hash

    def get(self, tree_family_id: int) -> Optional[TreeFamily]:
        """Look up tree family by ID."""
        return self._by_id.get(tree_family_id)

    def get_by_hash(self, tree_hash: str) -> Optional[TreeFamily]:
        """Look up tree family by hash."""
        tid = self._by_hash.get(tree_hash)
        if tid is not None:
            return self._by_id.get(tid)
        return None

    def list_all(self) -> List[TreeFamily]:
        """List all registered tree families."""
        return list(self._by_id.values())

    def clear(self) -> None:
        """Clear all registrations."""
        self._by_id.clear()
        self._by_hash.clear()


# =============================================================================
# DEFAULT TREE FAMILIES
# =============================================================================

def create_balanced_tree() -> TreeFamily:
    """Create balanced tree (default, even distribution)."""
    return TreeFamily(
        id=TreeID.BALANCED,
        name="Balanced",
        description="Even coefficient distribution across all leaves",
        coeff_profile=TreeCoeffProfile(
            leaf_weights=[1.0] * 64,
            band_emphasis="balanced",
            temporal_spread=1.0,
        ),
    )


def create_harmonic_tree() -> TreeFamily:
    """Create harmonic tree (emphasizes harmonic partials)."""
    # Harmonic emphasis: stronger weights on leaves corresponding to harmonics
    weights = [0.5] * 64
    # Fundamental and low harmonics get more weight
    for i in range(8):
        weights[i] = 1.0
    for i in range(8, 16):
        weights[i] = 0.8
    for i in range(16, 32):
        weights[i] = 0.6

    return TreeFamily(
        id=TreeID.HARMONIC,
        name="Harmonic",
        description="Emphasizes harmonic partials for tonal sounds",
        coeff_profile=TreeCoeffProfile(
            leaf_weights=weights,
            band_emphasis="low",
            temporal_spread=0.5,
        ),
    )


def create_texture_tree() -> TreeFamily:
    """Create texture tree (emphasizes high-frequency content)."""
    # Texture emphasis: stronger weights on high-frequency leaves
    weights = [0.3] * 64
    for i in range(32, 64):
        weights[i] = 1.0
    for i in range(16, 32):
        weights[i] = 0.7

    return TreeFamily(
        id=TreeID.TEXTURE,
        name="Texture",
        description="Emphasizes high-frequency texture and noise",
        coeff_profile=TreeCoeffProfile(
            leaf_weights=weights,
            band_emphasis="high",
            temporal_spread=0.8,
        ),
    )


def create_impact_tree() -> TreeFamily:
    """Create impact tree (tight transient response)."""
    # Impact: tight temporal spread, mid emphasis
    weights = [0.5] * 64
    for i in range(16, 48):
        weights[i] = 1.0

    return TreeFamily(
        id=TreeID.IMPACT,
        name="Impact",
        description="Tight transient response for percussive sounds",
        coeff_profile=TreeCoeffProfile(
            leaf_weights=weights,
            band_emphasis="mid",
            temporal_spread=0.2,
        ),
    )


def create_drone_tree() -> TreeFamily:
    """Create drone tree (sustained, evolving texture)."""
    # Drone: spread temporal response, low emphasis
    weights = [0.8] * 64
    for i in range(16):
        weights[i] = 1.0

    return TreeFamily(
        id=TreeID.DRONE,
        name="Drone",
        description="Sustained, slowly evolving texture",
        coeff_profile=TreeCoeffProfile(
            leaf_weights=weights,
            band_emphasis="low",
            temporal_spread=1.5,
        ),
    )


# Global registry instance
_global_tree_registry: Optional[TreeRegistry] = None


def get_tree_registry() -> TreeRegistry:
    """Get the global tree registry instance."""
    global _global_tree_registry
    if _global_tree_registry is None:
        _global_tree_registry = TreeRegistry()
        # Pre-register canonical trees
        _global_tree_registry.register(create_balanced_tree())
        _global_tree_registry.register(create_harmonic_tree())
        _global_tree_registry.register(create_texture_tree())
        _global_tree_registry.register(create_impact_tree())
        _global_tree_registry.register(create_drone_tree())
    return _global_tree_registry


__all__ = [
    "TreeID",
    "TreeCoeffProfile",
    "TreeFamily",
    "TreeRegistry",
    "create_balanced_tree",
    "create_harmonic_tree",
    "create_texture_tree",
    "create_impact_tree",
    "create_drone_tree",
    "get_tree_registry",
]
