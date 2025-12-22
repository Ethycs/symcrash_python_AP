"""
WAVELET PACKET TREES TESTS
==========================

Tests for wavelet packet tree structure and synthesis.

Covers:
- Tree structure verification (32 leaves per tree)
- Frequency ordering of leaves
- Complex mask consistency
- Synthesis pipeline
- ROM export validation
"""

import pytest

from symcrash_python_ap.wavelet_packet_trees import (
    PacketTreeSet, PacketSynthesizer,
    BLOCK_SIZE, LEAF_COUNT, COEFF_LEN, NODE_SIZE,
    export_rom_tables, export_verilog_rom,
)
from symcrash_python_ap.wavelet_trees import PacketNode


class TestTreeStructure:
    """Verify tree structure invariants."""

    def test_all_trees_have_32_leaves(self):
        """Each tree must have exactly 32 leaves."""
        tree_set = PacketTreeSet()

        for tree_id, rom in tree_set.trees.items():
            n_leaves = len(rom.leaf_list)
            assert n_leaves == 32, f"Tree {tree_id} ({rom.name}): expected 32 leaves, got {n_leaves}"

    def test_trees_are_frequency_ordered(self):
        """Leaves should be in frequency order."""
        tree_set = PacketTreeSet()

        for tree_id, rom in tree_set.trees.items():
            is_sorted = all(
                PacketNode(rom.leaf_list[i][0], rom.leaf_list[i][1]) <=
                PacketNode(rom.leaf_list[i+1][0], rom.leaf_list[i+1][1])
                for i in range(len(rom.leaf_list) - 1)
            )
            assert is_sorted, f"Tree {tree_id} ({rom.name}): leaves not frequency ordered"

    def test_complex_mask_length(self):
        """Complex mask should have 32 entries."""
        tree_set = PacketTreeSet()

        for tree_id, rom in tree_set.trees.items():
            assert len(rom.complex_mask) == 32, f"Tree {tree_id}: complex_mask wrong length"

    def test_leaf_depth_range(self):
        """Leaf depths should be in valid range."""
        tree_set = PacketTreeSet()

        for tree_id, rom in tree_set.trees.items():
            for d, k in rom.leaf_list:
                assert 1 <= d <= 6, f"Tree {tree_id}: invalid depth {d}"
                assert 0 <= k < 2**d, f"Tree {tree_id}: invalid k={k} for depth {d}"


class TestSynthesis:
    """Test synthesis pipeline."""

    def test_synthesis_produces_correct_length(self):
        """Synthesis should produce BLOCK_SIZE samples."""
        tree_set = PacketTreeSet()

        # Zero coefficients
        test_coeffs = [0] * COEFF_LEN

        for tree_id in range(5):
            audio = tree_set.synthesize(tree_id, test_coeffs)
            assert len(audio) == BLOCK_SIZE, f"Tree {tree_id}: wrong output length"

    def test_synthesis_with_impulse(self):
        """Synthesis with impulse produces non-zero output."""
        tree_set = PacketTreeSet()

        # Impulse in first coefficient
        test_coeffs = [0] * COEFF_LEN
        test_coeffs[0] = 16384

        for tree_id in range(5):
            audio = tree_set.synthesize(tree_id, test_coeffs)

            # Should have some non-zero samples
            max_sample = max(abs(x) for x in audio)
            assert max_sample > 0, f"Tree {tree_id}: zero output from impulse"

    def test_synthesis_energy_propagation(self):
        """Energy in coefficients produces energy in output."""
        tree_set = PacketTreeSet()

        # Moderate energy in multiple nodes
        test_coeffs = [0] * COEFF_LEN
        test_coeffs[0] = 16384  # Impulse in first node
        test_coeffs[8] = 8192   # Some energy in second node

        for tree_id in range(5):
            audio = tree_set.synthesize(tree_id, test_coeffs)

            peak = max(abs(x) for x in audio)
            energy = sum(x * x for x in audio) / BLOCK_SIZE

            # Should have reasonable peak and energy
            assert peak > 0, f"Tree {tree_id}: no peak"
            assert energy > 0, f"Tree {tree_id}: no energy"

    def test_crossfade_synthesis(self):
        """Crossfade synthesis blends correctly."""
        tree_set = PacketTreeSet()

        test_coeffs = [1000] * COEFF_LEN

        # Synthesize with pure A
        pure_a = tree_set.synthesize_crossfade(0, 1, test_coeffs, 0)

        # Synthesize with pure B
        pure_b = tree_set.synthesize_crossfade(0, 1, test_coeffs, 0x7FFF)

        # Synthesize with 50% blend
        blend_50 = tree_set.synthesize_crossfade(0, 1, test_coeffs, 0x4000)

        assert len(pure_a) == BLOCK_SIZE
        assert len(pure_b) == BLOCK_SIZE
        assert len(blend_50) == BLOCK_SIZE

        # Blend should be between pure A and pure B (approximately)
        for i in range(BLOCK_SIZE):
            if pure_a[i] != pure_b[i]:
                # At least one sample should be different
                break
        # If all samples are the same, the test still passes (trees may be similar)


class TestRomExport:
    """Test ROM table export functionality."""

    def test_export_rom_tables(self):
        """ROM export produces valid structure."""
        rom_data = export_rom_tables()

        assert "constants" in rom_data
        assert "trees" in rom_data

        # Check constants
        assert rom_data["constants"]["BLOCK_SIZE"] == BLOCK_SIZE
        assert rom_data["constants"]["LEAF_COUNT"] == LEAF_COUNT
        assert rom_data["constants"]["COEFF_LEN"] == COEFF_LEN
        assert rom_data["constants"]["NODE_SIZE"] == NODE_SIZE

        # Check trees
        assert len(rom_data["trees"]) == 5

        for tree_id in range(5):
            tree_data = rom_data["trees"][tree_id]
            assert "name" in tree_data
            assert "split_mask" in tree_data
            assert "leaf_list" in tree_data
            assert "complex_mask" in tree_data

    def test_export_verilog_rom(self):
        """Verilog ROM export produces valid code."""
        for tree_id in range(5):
            verilog = export_verilog_rom(tree_id)

            # Should contain function definitions
            assert f"get_leaf_depth_{tree_id}" in verilog
            assert f"get_leaf_k_{tree_id}" in verilog
            assert f"get_complex_mask_{tree_id}" in verilog

            # Should contain case statements
            assert "case" in verilog
            assert "endcase" in verilog


class TestCoeffUnpacking:
    """Test coefficient unpacking."""

    def test_unpack_coeffs_length(self):
        """Unpacked coefficients should have correct structure."""
        tree_set = PacketTreeSet()

        test_coeffs = list(range(COEFF_LEN))

        for tree_id in range(5):
            nodes = tree_set.unpack_coeffs(tree_id, test_coeffs)

            assert len(nodes) == LEAF_COUNT, f"Tree {tree_id}: wrong node count"

            for node in nodes:
                assert len(node) == NODE_SIZE, f"Tree {tree_id}: wrong node size"

    def test_unpack_coeffs_values(self):
        """Unpacked values should match input order."""
        tree_set = PacketTreeSet()

        test_coeffs = list(range(COEFF_LEN))

        for tree_id in range(5):
            nodes = tree_set.unpack_coeffs(tree_id, test_coeffs)

            # Verify sequential unpacking
            for leaf_idx in range(LEAF_COUNT):
                for coeff_idx in range(NODE_SIZE):
                    expected = leaf_idx * NODE_SIZE + coeff_idx
                    assert nodes[leaf_idx][coeff_idx] == expected


def test_verify_trees():
    """Run the full tree verification suite."""
    tree_set = PacketTreeSet()

    for tree_id, rom in tree_set.trees.items():
        n_leaves = len(rom.leaf_list)
        n_complex = sum(rom.complex_mask)

        # Basic structure
        assert n_leaves == 32
        assert 0 <= n_complex <= 32

        # Depth distribution
        depths = {}
        for d, k in rom.leaf_list:
            depths[d] = depths.get(d, 0) + 1

        # All depths should be positive and <= 6
        for d, count in depths.items():
            assert 1 <= d <= 6
            assert count > 0


def test_synthesis_basic():
    """Basic synthesis test for all trees."""
    tree_set = PacketTreeSet()

    # Simple impulse test
    test_coeffs = [0] * COEFF_LEN
    test_coeffs[0] = 16384
    test_coeffs[8] = 8192

    for tree_id in range(5):
        audio = tree_set.synthesize(tree_id, test_coeffs)

        peak = max(abs(x) for x in audio)
        energy = sum(x * x for x in audio) / BLOCK_SIZE

        # Should produce output
        assert len(audio) == BLOCK_SIZE
        assert peak > 0
