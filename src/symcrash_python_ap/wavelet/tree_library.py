"""
Tree Library Contract (WSE-STD-010 v0)
======================================

Conforms to WSE-STD-010 — Tree Library Contract (v0)

Defines the canonical wavelet packet tree "library" used by Voice/Symbol/Wavelet engines.
Freezes: IDs, invariants, leaf ordering, band semantics, and morph legality.

v0 Freeze Summary:
- B=64, L=32, K=8, coeff packing per leaf
- Canonical leaf order: low→high (leaf 0 = lowest frequency)
- Tree IDs + descriptor CRC enforcement
- Morph whitelist + M1 (coefficient-space blend) semantic mode
- Canonical targeting + bijective mapping functions
- Deterministic fallback behavior
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import zlib

from ..fixed import MUL_Q15, Q15_ONE, ONE_Q15

# Import tree definitions
from ..wavelet_trees import (
    BLOCK_SIZE, MAX_DEPTH, LEAF_COUNT, COEFF_LEN, NODE_SIZE, NODE_COUNT,
    PacketNode, TreeROM, enumerate_leaves,
    build_tree_0_balanced,
    build_tree_1_harmonic,
    build_tree_2_texture,
    build_tree_3_transient,
    build_tree_4_drone,
    SAMPLE_RATE, NYQUIST,
)


# =============================================================================
# WSE-STD-010 CONSTANTS (Section 0)
# =============================================================================

# Block and coefficient layout (MUST per Section 0.1)
B = BLOCK_SIZE          # 64 audio samples per block
L = LEAF_COUNT          # 32 leaves (bands) per block
K = NODE_SIZE           # 8 coefficients per leaf

# Family IDs (Section 5.1)
FAMILY_CDF53_WPT = 1    # CDF 5/3 wavelet packet tree (lifting-based)

# Leaf order IDs (Section 2.2)
LEAF_ORDER_CANON_LOW_TO_HIGH = 0  # Canonical: leaf 0 = lowest frequency

# Tree morph mode (Section 6.2) - v0 uses M1
MORPH_MODE_COEFFICIENT_BLEND = 1  # M1: Coefficient-space blend (RECOMMENDED)
MORPH_MODE_AUDIO_CROSSFADE = 2    # M2: Audio crossfade

# v0 default morph mode
DEFAULT_MORPH_MODE = MORPH_MODE_COEFFICIENT_BLEND

# Reserved tree IDs
TREE_BYPASS_DEFAULT = 0  # Default stable tree (Section 2.1)


# =============================================================================
# TREE DESCRIPTOR (Section 2.2)
# =============================================================================

@dataclass
class TreeDescriptor:
    """
    Tree descriptor per WSE-STD-010 Section 2.2.

    Each tree_id maps to a TreeDescriptor with frozen structure.
    """
    tree_id: int                    # u16: unique key
    family_id: int = FAMILY_CDF53_WPT  # u8: CDF53_WPT=1
    leaf_count: int = L             # u8: MUST be 32 in v0
    coeffs_per_leaf: int = K        # u8: MUST be 8 in v0
    leaf_order_id: int = LEAF_ORDER_CANON_LOW_TO_HIGH  # u8: MUST be CANON_LOW_TO_HIGH in v0

    # Reconstruction normalization (Section 4)
    recon_gain_q15: int = Q15_ONE   # s16: unity calibration for this tree
    coeff_energy_gain_q15: int = Q15_ONE  # s16: optional, maps coeff energy to audio RMS

    # Frequency mapping (Section 1.2) - optional if derivable
    f_lo_q10_6: List[int] = field(default_factory=list)  # u16[32]: low edge Hz in Q10.6
    f_hi_q10_6: List[int] = field(default_factory=list)  # u16[32]: high edge Hz in Q10.6

    # Schedule payload (implementation-defined but frozen)
    schedule_words: List[int] = field(default_factory=list)

    # Descriptor CRC for toolchain agreement (Section 3.3)
    descriptor_crc32: int = 0

    # Name (for debugging/display)
    name: str = ""

    # Internal: reference to TreeROM for synthesis
    _rom: Optional[TreeROM] = field(default=None, repr=False)

    def compute_crc32(self) -> int:
        """Compute CRC32 of the descriptor (excluding CRC field itself)."""
        data = (
            f"{self.tree_id}:{self.family_id}:{self.leaf_count}:"
            f"{self.coeffs_per_leaf}:{self.leaf_order_id}:"
            f"{self.recon_gain_q15}:{self.coeff_energy_gain_q15}:"
            f"{len(self.f_lo_q10_6)}:{len(self.schedule_words)}"
        ).encode('utf-8')
        return zlib.crc32(data) & 0xFFFFFFFF


# =============================================================================
# BAND BOUNDARY TABLE (Section 1.2)
# =============================================================================

def compute_band_boundaries(fs: int = SAMPLE_RATE) -> Tuple[List[float], List[float]]:
    """
    Compute band boundaries for uniform distribution (Section 1.2).

    For v0, we use uniform band widths: each of 32 leaves covers
    equal portions of [0, Fs/2].

    Args:
        fs: Sample rate in Hz

    Returns:
        Tuple of (f_lo[32], f_hi[32]) in Hz
    """
    nyquist = fs / 2
    band_width = nyquist / L

    f_lo = [i * band_width for i in range(L)]
    f_hi = [(i + 1) * band_width for i in range(L)]

    # Enforce exact boundaries per Section 1.2
    f_lo[0] = 0.0
    f_hi[L - 1] = nyquist

    return f_lo, f_hi


def hz_to_q10_6(hz: float) -> int:
    """Convert Hz to Q10.6 fixed-point (max ~1024 Hz per integer part)."""
    return int(hz * 64) & 0xFFFF


def q10_6_to_hz(q: int) -> float:
    """Convert Q10.6 fixed-point to Hz."""
    return q / 64.0


# =============================================================================
# TREE LIBRARY (Section 8)
# =============================================================================

class TreeLibrary:
    """
    Tree library implementing WSE-STD-010 API (Section 8).

    Provides:
    1. get_tree_descriptor(tree_id) -> TreeDescriptor
    2. validate_tree(tree_id, Fs) -> bool
    3. get_band_limits(tree_id, Fs, leaf) -> (f_lo, f_hi)
    4. allowed_morph(tree_a, tree_b) -> bool
    5. map_canon_to_phys(tree_id, leaf) -> leaf_phys
    6. map_phys_to_canon(tree_id, leaf_phys) -> leaf
    """

    def __init__(self, fs: int = SAMPLE_RATE):
        """Initialize tree library for given sample rate."""
        self.fs = fs
        self.descriptors: Dict[int, TreeDescriptor] = {}
        self.morph_whitelist: Dict[Tuple[int, int], bool] = {}

        # Build band boundaries
        self.f_lo, self.f_hi = compute_band_boundaries(fs)

        # Build descriptors for all trees
        self._build_library()

        # Build morph whitelist
        self._build_morph_whitelist()

    def _build_library(self):
        """Build tree descriptors for all supported trees."""
        # Tree builders
        builders = [
            (0, build_tree_0_balanced, "Balanced"),
            (1, build_tree_1_harmonic, "Harmonic"),
            (2, build_tree_2_texture, "Texture"),
            (3, build_tree_3_transient, "Transient"),
            (4, build_tree_4_drone, "Drone"),
        ]

        for tree_id, builder, name in builders:
            rom = builder()

            # Convert band boundaries to Q10.6
            f_lo_q = [hz_to_q10_6(f) for f in self.f_lo]
            f_hi_q = [hz_to_q10_6(f) for f in self.f_hi]

            # Build schedule from leaf list
            schedule = []
            for depth, k in rom.leaf_list:
                schedule.append((depth << 8) | k)

            desc = TreeDescriptor(
                tree_id=tree_id,
                family_id=FAMILY_CDF53_WPT,
                leaf_count=L,
                coeffs_per_leaf=K,
                leaf_order_id=LEAF_ORDER_CANON_LOW_TO_HIGH,
                recon_gain_q15=Q15_ONE,
                f_lo_q10_6=f_lo_q,
                f_hi_q10_6=f_hi_q,
                schedule_words=schedule,
                name=name,
                _rom=rom,
            )

            # Compute CRC
            desc.descriptor_crc32 = desc.compute_crc32()

            self.descriptors[tree_id] = desc

    def _build_morph_whitelist(self):
        """
        Build morph whitelist (Section 6.1).

        In v0, all trees with same leaf_count can morph to each other.
        """
        tree_ids = list(self.descriptors.keys())

        for a in tree_ids:
            for b in tree_ids:
                # Same leaf count = morphable
                desc_a = self.descriptors[a]
                desc_b = self.descriptors[b]

                allowed = (desc_a.leaf_count == desc_b.leaf_count and
                          desc_a.coeffs_per_leaf == desc_b.coeffs_per_leaf)

                self.morph_whitelist[(a, b)] = allowed

    # -------------------------------------------------------------------------
    # API Functions (Section 8)
    # -------------------------------------------------------------------------

    def get_tree_descriptor(self, tree_id: int) -> TreeDescriptor:
        """
        Get tree descriptor by ID (Section 8.1).

        Falls back to TREE_BYPASS_DEFAULT if invalid (Section 9.1).
        """
        if tree_id in self.descriptors:
            return self.descriptors[tree_id]

        # Fallback to default (Section 9.1)
        return self.descriptors[TREE_BYPASS_DEFAULT]

    def validate_tree(self, tree_id: int, fs: int = None) -> bool:
        """
        Validate tree for given sample rate (Section 8.2).

        Checks:
        - tree_id exists
        - Descriptor CRC matches
        - Structural invariants (Section 3.1)
        """
        if fs is None:
            fs = self.fs

        if tree_id not in self.descriptors:
            return False

        desc = self.descriptors[tree_id]

        # Check structural invariants (Section 3.1)
        if desc.leaf_count != L:
            return False
        if desc.coeffs_per_leaf != K:
            return False

        # Check CRC (Section 3.3)
        expected_crc = desc.compute_crc32()
        if desc.descriptor_crc32 != expected_crc:
            return False

        return True

    def get_band_limits(self, tree_id: int, fs: int, leaf: int) -> Tuple[float, float]:
        """
        Get frequency band limits for a leaf (Section 8.3).

        Args:
            tree_id: Tree identifier
            fs: Sample rate
            leaf: Canonical leaf index (0..31)

        Returns:
            (f_lo, f_hi) in Hz
        """
        if leaf < 0 or leaf >= L:
            return (0.0, 0.0)

        # Compute for given sample rate
        nyquist = fs / 2
        band_width = nyquist / L

        f_lo = leaf * band_width
        f_hi = (leaf + 1) * band_width

        return (f_lo, f_hi)

    def allowed_morph(self, tree_a: int, tree_b: int) -> bool:
        """
        Check if morph between trees is allowed (Section 6.1).
        """
        return self.morph_whitelist.get((tree_a, tree_b), False)

    def map_canon_to_phys(self, tree_id: int, canon_leaf: int) -> int:
        """
        Map canonical leaf index to physical (Section 7.2).

        In v0 with canonical ordering, this is identity.
        """
        if canon_leaf < 0 or canon_leaf >= L:
            return 0

        # v0: canonical == physical (Section 1.1)
        return canon_leaf

    def map_phys_to_canon(self, tree_id: int, phys_leaf: int) -> int:
        """
        Map physical leaf index to canonical (Section 7.2).

        In v0 with canonical ordering, this is identity.
        """
        if phys_leaf < 0 or phys_leaf >= L:
            return 0

        # v0: physical == canonical (Section 1.1)
        return phys_leaf

    # -------------------------------------------------------------------------
    # Extended API
    # -------------------------------------------------------------------------

    def get_all_tree_ids(self) -> List[int]:
        """Get list of all valid tree IDs."""
        return list(self.descriptors.keys())

    def get_tree_rom(self, tree_id: int) -> Optional[TreeROM]:
        """Get TreeROM for synthesis (internal use)."""
        desc = self.get_tree_descriptor(tree_id)
        return desc._rom


# =============================================================================
# MORPH BLENDING (Section 6.2 - Mode M1)
# =============================================================================

def blend_coefficients(c_a: List[int], c_b: List[int], alpha_q15: int) -> List[int]:
    """
    Blend coefficients for tree morph (Section 6.2, Mode M1).

    c_blend[i] = (1 - α) * c_a[i] + α * c_b[i]

    Uses WSE-STD-001 MIX_Q15 semantics.

    Args:
        c_a: Coefficient vector for tree A (256 values)
        c_b: Coefficient vector for tree B (256 values)
        alpha_q15: Blend factor in Q1.15 (0 = pure A, ONE_Q15 = pure B)

    Returns:
        Blended coefficient vector (256 values)
    """
    from ..fixed import MIX_Q15

    n = min(len(c_a), len(c_b), COEFF_LEN)
    c_blend = [0] * COEFF_LEN

    for i in range(n):
        c_blend[i] = MIX_Q15(c_a[i], c_b[i], alpha_q15)

    return c_blend


# =============================================================================
# GLOBAL LIBRARY INSTANCE
# =============================================================================

# Singleton library instance
_library: Optional[TreeLibrary] = None


def get_tree_library(fs: int = SAMPLE_RATE) -> TreeLibrary:
    """Get the global tree library instance."""
    global _library
    if _library is None or _library.fs != fs:
        _library = TreeLibrary(fs)
    return _library


# Convenience API wrappers
def get_tree_descriptor(tree_id: int) -> TreeDescriptor:
    """Get tree descriptor by ID."""
    return get_tree_library().get_tree_descriptor(tree_id)


def validate_tree(tree_id: int, fs: int = SAMPLE_RATE) -> bool:
    """Validate tree for given sample rate."""
    return get_tree_library(fs).validate_tree(tree_id, fs)


def get_band_limits(tree_id: int, fs: int, leaf: int) -> Tuple[float, float]:
    """Get frequency band limits for a leaf."""
    return get_tree_library(fs).get_band_limits(tree_id, fs, leaf)


def allowed_morph(tree_a: int, tree_b: int) -> bool:
    """Check if morph between trees is allowed."""
    return get_tree_library().allowed_morph(tree_a, tree_b)


def map_canon_to_phys(tree_id: int, leaf: int) -> int:
    """Map canonical leaf index to physical."""
    return get_tree_library().map_canon_to_phys(tree_id, leaf)


def map_phys_to_canon(tree_id: int, leaf_phys: int) -> int:
    """Map physical leaf index to canonical."""
    return get_tree_library().map_phys_to_canon(tree_id, leaf_phys)


# Tests moved to pipeline_tests/test_tree_library_std010.py
