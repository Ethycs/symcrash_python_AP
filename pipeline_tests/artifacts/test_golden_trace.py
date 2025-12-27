"""
Test for Golden Trace v0 ABI Regression

Runs the golden trace through the VM and verifies bit-exact output.

This test catches:
- Accidental changes to VM execution semantics
- Platform-dependent floating point behavior (should not happen with integer math)
- Changes that are "numerically valid" but musically different

Usage:
    python -m pytest pipeline_tests/artifacts/test_golden_trace.py -v
"""

import pytest

from pipeline_tests.artifacts.golden_trace_v0 import (
    VM_VERSION,
    TRACE_NAME,
    TRACE_DURATION_BLOCKS,
    FROZEN_BLOCK_HASHES,
    FROZEN_TRACE_HASH,
    compute_trace_hashes,
)


class TestGoldenTrace:
    """Tests for golden trace ABI regression."""

    def test_golden_trace_matches_frozen(self):
        """
        Verify VM output matches frozen golden trace.

        This is the primary ABI regression test. If this fails:
        1. Check if the change was intentional
        2. If intentional, regenerate hashes: python golden_trace_v0.py --regenerate
        3. If unintentional, fix the regression
        """
        block_hashes, trace_hash = compute_trace_hashes()

        # Verify overall trace hash first (fast fail)
        assert trace_hash == FROZEN_TRACE_HASH, (
            f"Trace hash mismatch for {TRACE_NAME}!\n"
            f"  Expected: {FROZEN_TRACE_HASH}\n"
            f"  Got:      {trace_hash}\n"
            f"  This indicates the VM output has changed since {VM_VERSION} was frozen."
        )

        # Then verify each block (for debugging which block diverged)
        assert len(block_hashes) == len(FROZEN_BLOCK_HASHES), (
            f"Block count mismatch: got {len(block_hashes)}, expected {len(FROZEN_BLOCK_HASHES)}"
        )

        for i, (actual, expected) in enumerate(zip(block_hashes, FROZEN_BLOCK_HASHES)):
            assert actual == expected, (
                f"Block {i} hash mismatch!\n"
                f"  Expected: {expected}\n"
                f"  Got:      {actual}\n"
                f"  This is the first block that diverged."
            )

    def test_golden_trace_has_expected_duration(self):
        """Verify trace has expected number of blocks."""
        block_hashes, _ = compute_trace_hashes()
        assert len(block_hashes) == TRACE_DURATION_BLOCKS, (
            f"Duration mismatch: got {len(block_hashes)}, expected {TRACE_DURATION_BLOCKS}"
        )

    def test_block_hashes_are_unique(self):
        """Verify blocks produce unique hashes (not stuck in loop)."""
        block_hashes, _ = compute_trace_hashes()

        # Most blocks should be unique (some may repeat during sustain)
        unique_hashes = set(block_hashes)

        # At least 50% should be unique (attack/release have changing params)
        assert len(unique_hashes) >= len(block_hashes) // 2, (
            f"Too few unique hashes: {len(unique_hashes)} out of {len(block_hashes)}. "
            f"VM may be stuck in a degenerate state."
        )

    def test_determinism_across_runs(self):
        """Verify same input produces same output on repeated runs."""
        hashes1_blocks, hashes1_trace = compute_trace_hashes()
        hashes2_blocks, hashes2_trace = compute_trace_hashes()

        assert hashes1_trace == hashes2_trace, "Trace hash differs between runs!"

        for i, (h1, h2) in enumerate(zip(hashes1_blocks, hashes2_blocks)):
            assert h1 == h2, f"Block {i} differs between runs!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
