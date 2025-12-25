"""
WSE-STD-010 TREE LIBRARY CONFORMANCE TESTS
==========================================

Tests for tree_library.py conformance with WSE-STD-010.
Extracted from tree_library.py test_std010_conformance().

NOTE: This test requires the symcrash_python_ap package to be importable.
Run with: pixi run pytest pipeline_tests/test_tree_library_std010.py -v
"""

import pytest
import sys
from pathlib import Path

# Ensure package is importable
pkg_path = Path(__file__).parent.parent / "src"
if str(pkg_path) not in sys.path:
    sys.path.insert(0, str(pkg_path))

# Now import from the package
from symcrash_python_ap.tree_library import (
    get_tree_library, TREE_BYPASS_DEFAULT, blend_coefficients, ONE_Q15,
    B, L, K, COEFF_LEN, SAMPLE_RATE, NYQUIST
)
from symcrash_python_ap.fixed import Q15_HALF


class TestBlockLayout:
    """Section 0: Block/Coefficient Layout."""

    def test_block_size_is_64(self):
        """B must be 64."""
        assert B == 64

    def test_leaf_count_is_64(self):
        """L must be 64 (refactored from 32)."""
        assert L == 64

    def test_coeffs_per_leaf_is_4(self):
        """K must be 4 (refactored from 8)."""
        assert K == 4

    def test_coeff_len_is_256(self):
        """COEFF_LEN must be 256."""
        assert COEFF_LEN == 256


class TestCanonicalLeafOrder:
    """Section 1.1: Canonical Leaf Order."""

    def test_leaf_0_is_lowest_frequency(self):
        """Leaf 0 must start at 0 Hz."""
        lib = get_tree_library()
        f_lo, f_hi = lib.get_band_limits(0, SAMPLE_RATE, 0)
        assert f_lo == 0.0, f"Leaf 0 f_lo must be 0, got {f_lo}"

    def test_last_leaf_is_highest_frequency(self):
        """Last leaf (L-1) must end at Nyquist."""
        lib = get_tree_library()
        f_lo, f_hi = lib.get_band_limits(0, SAMPLE_RATE, L - 1)
        assert f_hi == NYQUIST, f"Leaf {L-1} f_hi must be {NYQUIST}, got {f_hi}"

    def test_frequency_order_low_to_high(self):
        """Leaf 0 must be lower frequency than last leaf."""
        lib = get_tree_library()
        f_lo_0, _ = lib.get_band_limits(0, SAMPLE_RATE, 0)
        f_lo_last, _ = lib.get_band_limits(0, SAMPLE_RATE, L - 1)
        assert f_lo_0 < f_lo_last


class TestBandBoundaryContinuity:
    """Section 1.2: Band Boundary Continuity."""

    def test_bands_are_contiguous(self):
        """Bands must be contiguous (no gaps)."""
        lib = get_tree_library()

        for i in range(L - 1):
            _, f_hi_i = lib.get_band_limits(0, SAMPLE_RATE, i)
            f_lo_next, _ = lib.get_band_limits(0, SAMPLE_RATE, i + 1)
            assert f_hi_i == f_lo_next, f"Gap between leaf {i} and {i+1}"


class TestTreeBypassDefault:
    """Section 2.1: TREE_BYPASS_DEFAULT."""

    def test_tree_bypass_default_is_zero(self):
        """TREE_BYPASS_DEFAULT must be 0."""
        assert TREE_BYPASS_DEFAULT == 0

    def test_tree_0_exists(self):
        """Tree 0 must exist."""
        lib = get_tree_library()
        desc = lib.get_tree_descriptor(0)
        assert desc.tree_id == 0


class TestTreeValidation:
    """Section 3: Tree Validation."""

    def test_all_trees_valid(self):
        """All defined trees should be valid."""
        lib = get_tree_library()
        for tree_id in lib.get_all_tree_ids():
            assert lib.validate_tree(tree_id), f"Tree {tree_id} should be valid"

    def test_all_trees_have_64_leaves(self):
        """All trees must have 64 leaves (refactored from 32)."""
        lib = get_tree_library()
        for tree_id in lib.get_all_tree_ids():
            desc = lib.get_tree_descriptor(tree_id)
            assert desc.leaf_count == 64, f"Tree {tree_id} leaf_count must be 64"

    def test_all_trees_have_4_coeffs_per_leaf(self):
        """All trees must have 4 coefficients per leaf (refactored from 8)."""
        lib = get_tree_library()
        for tree_id in lib.get_all_tree_ids():
            desc = lib.get_tree_descriptor(tree_id)
            assert desc.coeffs_per_leaf == 4, f"Tree {tree_id} coeffs_per_leaf must be 4"

    def test_invalid_tree_fails_validation(self):
        """Invalid tree_id should fail validation."""
        lib = get_tree_library()
        assert not lib.validate_tree(9999)


class TestFallbackBehavior:
    """Section 9.1: Fallback Behavior."""

    def test_invalid_tree_returns_default(self):
        """Invalid tree_id should return default descriptor."""
        lib = get_tree_library()
        desc = lib.get_tree_descriptor(9999)
        assert desc.tree_id == TREE_BYPASS_DEFAULT


class TestMorphWhitelist:
    """Section 6.1: Morph Whitelist."""

    def test_all_trees_morphable(self):
        """All v0 trees should be morphable with each other."""
        lib = get_tree_library()
        tree_ids = lib.get_all_tree_ids()

        for a in tree_ids:
            for b in tree_ids:
                assert lib.allowed_morph(a, b), f"Morph {a}->{b} should be allowed"


class TestLeafMappingBijection:
    """Section 7.2: Leaf Mapping Bijection."""

    def test_leaf_mappings_bijective(self):
        """Leaf mappings must be bijective (round-trip preserving)."""
        lib = get_tree_library()

        for tree_id in lib.get_all_tree_ids():
            for canon in range(L):
                phys = lib.map_canon_to_phys(tree_id, canon)
                back = lib.map_phys_to_canon(tree_id, phys)
                assert back == canon, \
                    f"Mapping not bijective for tree {tree_id}: {canon}->{phys}->{back}"


class TestCoefficientBlending:
    """Section 6.2 M1: Coefficient Blending."""

    def test_blend_alpha_zero_gives_a(self):
        """alpha=0 should give c_a (approximately)."""
        c_a = [10000] * COEFF_LEN
        c_b = [20000] * COEFF_LEN

        c_blend = blend_coefficients(c_a, c_b, 0)
        # Due to Q15 math, allow some tolerance
        assert abs(c_blend[0] - 10000) < 100, f"alpha=0 gave {c_blend[0]}, expected ~10000"

    def test_blend_alpha_one_gives_b(self):
        """alpha=ONE should give c_b (approximately)."""
        c_a = [10000] * COEFF_LEN
        c_b = [20000] * COEFF_LEN

        c_blend = blend_coefficients(c_a, c_b, ONE_Q15)
        # Due to Q15 math, allow some tolerance
        assert abs(c_blend[0] - 20000) < 100, f"alpha=ONE gave {c_blend[0]}, expected ~20000"

    def test_blend_alpha_half_gives_average(self):
        """alpha=HALF should give average (approximately)."""
        c_a = [10000] * COEFF_LEN
        c_b = [20000] * COEFF_LEN

        c_blend = blend_coefficients(c_a, c_b, Q15_HALF)
        # Due to Q15 math, allow some tolerance
        assert abs(c_blend[0] - 15000) < 500, f"alpha=HALF gave {c_blend[0]}, expected ~15000"


class TestDescriptorCRC:
    """Section 3.3: Descriptor CRC."""

    def test_all_descriptor_crcs_match(self):
        """All descriptor CRCs should match computed values."""
        lib = get_tree_library()

        for tree_id in lib.get_all_tree_ids():
            desc = lib.get_tree_descriptor(tree_id)
            computed = desc.compute_crc32()
            assert computed == desc.descriptor_crc32, \
                f"CRC mismatch for tree {tree_id}: computed {computed}, stored {desc.descriptor_crc32}"
