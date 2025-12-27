"""
Test for Golden Trace Stress v0 ABI Regression

Runs the stress golden trace through the VM and verifies bit-exact output.

This test catches:
- Accidental changes to VM execution semantics
- PRNG advancement bugs (noise should only advance on event lane 2)
- Permute/tail/K-ladder regressions
- Platform-dependent behavior (should not happen with integer math)

Coverage assertions verify the stress trace actually exercises:
- NoiseBand via event lane 2
- Permute ops (RotateNodes)
- Tail mode ladder (FULL and MIN)
- K ladder via pattern_id (0 and 2)

Usage:
    python -m pytest pipeline_tests/artifacts/test_golden_trace_stress.py -v
"""

import pytest

from pipeline_tests.artifacts.golden_trace_stress_v0 import (
    GOLDEN_TRACE_STRESS_VERSION,
    TRACE_NAME,
    TRACE_DURATION_BLOCKS,
    FROZEN_BLOCK_HASHES,
    FROZEN_TRACE_HASH,
    compute_trace_hashes,
    generate_golden_stress_sequence,
)


class TestGoldenTraceStress:
    """Tests for golden trace stress ABI regression."""

    def test_golden_trace_matches_frozen(self):
        """
        Verify VM output matches frozen golden trace.

        This is the primary ABI regression test. If this fails:
        1. Check if the change was intentional
        2. If intentional, regenerate hashes: python golden_trace_stress_v0.py --regenerate
        3. If unintentional, fix the regression
        """
        block_hashes, trace_hash = compute_trace_hashes()

        # Verify overall trace hash first (fast fail)
        assert trace_hash == FROZEN_TRACE_HASH, (
            f"Trace hash mismatch for {TRACE_NAME}!\n"
            f"  Expected: {FROZEN_TRACE_HASH}\n"
            f"  Got:      {trace_hash}\n"
            f"  This indicates the VM output has changed since {GOLDEN_TRACE_STRESS_VERSION} was frozen."
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

        # Most blocks should be unique (stress trace has lots of variation)
        unique_hashes = set(block_hashes)

        # At least 80% should be unique for stress trace
        assert len(unique_hashes) >= len(block_hashes) * 0.8, (
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


class TestStressTraceCoverage:
    """Tests that verify the stress trace exercises specific VM paths."""

    def test_stress_trace_exercises_noise_lane(self):
        """
        Assert noise lane 2 (NoiseBand) fires in the stress trace.

        NoiseBand advances PRNG, so this verifies PRNG gating is exercised.
        Region B (blocks 16-23) has noise on even blocks.
        """
        frames = generate_golden_stress_sequence()

        # Check that at least one frame has event lane 2 set
        noise_lane_mask = (1 << 2)  # Lane 2 = NoiseBand
        noise_frames = [
            i for i, f in enumerate(frames)
            if (f.event_mask & noise_lane_mask) != 0
        ]

        assert len(noise_frames) >= 4, (
            f"Expected at least 4 frames with noise lane 2, got {len(noise_frames)}. "
            f"Frames with noise: {noise_frames}"
        )

        # Verify they're in region B (16-23)
        for frame_idx in noise_frames:
            assert 16 <= frame_idx <= 23, (
                f"Noise frame {frame_idx} is outside expected region B (16-23)"
            )

    def test_stress_trace_exercises_permutes(self):
        """
        Assert permute ops (RotateNodes, BandShear) are enabled in the trace.

        Region C (blocks 24-31) has permutes enabled.
        """
        frames = generate_golden_stress_sequence()

        permute_ops = {"rotatenodes", "bandshear", "swapbands", "quasirotate"}
        permute_frames = [
            i for i, f in enumerate(frames)
            if any(op in permute_ops for op in f.ops_enable)
        ]

        assert len(permute_frames) >= 8, (
            f"Expected at least 8 frames with permute ops, got {len(permute_frames)}. "
            f"Frames with permutes: {permute_frames}"
        )

        # Verify they're in region C (24-31)
        for frame_idx in permute_frames:
            assert 24 <= frame_idx <= 31, (
                f"Permute frame {frame_idx} is outside expected region C (24-31)"
            )

    def test_stress_trace_exercises_tail_ladder(self):
        """
        Assert both FULL (0) and MIN (2) tail modes are used.

        Region A/B: FULL, Region C/D: MIN, Region E: FULL restored.
        """
        frames = generate_golden_stress_sequence()

        # Collect unique tail modes used
        tail_modes_used = set(f.tail_mode for f in frames)

        assert 0 in tail_modes_used, "FULL tail mode (0) not exercised"
        assert 2 in tail_modes_used, "MIN tail mode (2) not exercised"

        # Count frames per tail mode
        full_count = sum(1 for f in frames if f.tail_mode == 0)
        min_count = sum(1 for f in frames if f.tail_mode == 2)

        assert full_count >= 16, f"Expected at least 16 FULL tail frames, got {full_count}"
        assert min_count >= 8, f"Expected at least 8 MIN tail frames, got {min_count}"

    def test_stress_trace_exercises_k_ladder(self):
        """
        Assert pattern_id 0 (K=32) and 2 (K=16) are both used.

        Region A: pattern_id=0, Regions B/C/D: pattern_id=2, Region E: pattern_id=0.
        """
        frames = generate_golden_stress_sequence()

        # Collect unique pattern IDs used
        pattern_ids_used = set(f.pattern_id for f in frames)

        assert 0 in pattern_ids_used, "pattern_id 0 (K=32) not exercised"
        assert 2 in pattern_ids_used, "pattern_id 2 (K=16) not exercised"

        # Count frames per pattern
        p0_count = sum(1 for f in frames if f.pattern_id == 0)
        p2_count = sum(1 for f in frames if f.pattern_id == 2)

        assert p0_count >= 16, f"Expected at least 16 pattern_id=0 frames, got {p0_count}"
        assert p2_count >= 16, f"Expected at least 16 pattern_id=2 frames, got {p2_count}"

    def test_stress_trace_exercises_event_lanes(self):
        """
        Assert multiple event lanes are exercised.

        - Lane 0 (Impulse): Region D
        - Lane 2 (NoiseBand): Region B
        - Lane 4 (GateMask): Region C
        - Lane 7 (CuspInjection): Region D
        """
        frames = generate_golden_stress_sequence()

        lanes_exercised = set()
        for f in frames:
            for lane in range(8):
                if f.event_mask & (1 << lane):
                    lanes_exercised.add(lane)

        expected_lanes = {0, 2, 4, 7}
        missing = expected_lanes - lanes_exercised
        assert not missing, (
            f"Expected event lanes {expected_lanes}, missing: {missing}"
        )

    def test_stress_trace_has_gain_ramps(self):
        """Assert gain_q15 varies across the trace (not constant)."""
        frames = generate_golden_stress_sequence()

        gains = [f.gain_q15 for f in frames]
        unique_gains = set(gains)

        assert len(unique_gains) >= 3, (
            f"Expected at least 3 different gain values, got {len(unique_gains)}: {unique_gains}"
        )

    def test_stress_trace_has_beta_ramps(self):
        """Assert beta_target_q15 varies across the trace (not constant)."""
        frames = generate_golden_stress_sequence()

        betas = [f.beta_target_q15 for f in frames]
        unique_betas = set(betas)

        assert len(unique_betas) >= 3, (
            f"Expected at least 3 different beta values, got {len(unique_betas)}"
        )

        # Verify beta ramp in region A (should go from 0 to ~0.6)
        region_a_betas = [frames[i].beta_target_q15 for i in range(16)]
        assert region_a_betas[0] < region_a_betas[15], (
            f"Region A beta should rise: got {region_a_betas[0]} -> {region_a_betas[15]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
