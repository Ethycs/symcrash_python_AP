"""
Test for Golden Trace FM/PM v0 ABI Regression

Runs the FM/PM golden trace through the VM and verifies bit-exact output.

This test catches:
- Accidental changes to Mode FM6 execution semantics
- Changes to FM algorithm routing
- Operator level/ratio changes
- Feedback implementation changes
- Changes that are "numerically valid" but musically different

Usage:
    python -m pytest tests/artifacts/test_golden_trace_fm.py -v
"""

import pytest

from tests.artifacts.golden_trace_fm_v0 import (
    VM_VERSION,
    TRACE_NAME,
    TRACE_DURATION_BLOCKS,
    FROZEN_BLOCK_HASHES,
    FROZEN_TRACE_HASH,
    compute_trace_hashes,
)


class TestGoldenTraceFM:
    """Tests for FM/PM golden trace ABI regression."""

    @pytest.mark.skipif(
        not FROZEN_TRACE_HASH,
        reason="Frozen hashes not yet generated - run --regenerate first"
    )
    def test_golden_trace_matches_frozen(self):
        """
        Verify VM output matches frozen golden trace.

        This is the primary ABI regression test. If this fails:
        1. Check if the change was intentional
        2. If intentional, regenerate hashes: python golden_trace_fm_v0.py --regenerate
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

        # Most blocks should be unique (FM changes every block due to mod envelope)
        unique_hashes = set(block_hashes)

        # At least 50% should be unique
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

    def test_produces_audio_output(self):
        """Verify the trace produces non-zero audio output."""
        import numpy as np
        from symcrash_python_ap.vm import vm_step_block, VMState
        from symcrash_python_ap.vm.tables import DEFAULT_TABLES
        from tests.artifacts.golden_trace_fm_v0 import generate_golden_sequence

        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=72, cents=0)
        frames = generate_golden_sequence()

        total_nonzero = 0
        for frame in frames:
            state, output = vm_step_block(state, frame, DEFAULT_TABLES)
            total_nonzero += np.count_nonzero(output)

        # Should have significant non-zero samples
        assert total_nonzero > 100, f"Too few non-zero samples: {total_nonzero}"

    def test_fm_modulation_creates_harmonics(self):
        """Verify FM modulation creates expected harmonic content."""
        import numpy as np
        from symcrash_python_ap.vm import vm_step_block, VMState
        from symcrash_python_ap.vm.tables import DEFAULT_TABLES
        from tests.artifacts.golden_trace_fm_v0 import generate_golden_sequence

        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=72, cents=0)
        frames = generate_golden_sequence()

        # Collect first few blocks (attack phase with high modulation)
        outputs = []
        for frame in frames[:5]:
            state, output = vm_step_block(state, frame, DEFAULT_TABLES)
            outputs.append(output)

        # Concatenate and analyze spectrum
        audio = np.concatenate(outputs)

        # FFT analysis
        fft = np.abs(np.fft.rfft(audio))

        # Should have energy at multiple frequencies (FM creates sidebands)
        # Find peaks above noise floor
        noise_floor = np.median(fft)
        peaks = np.where(fft > noise_floor * 10)[0]

        # FM should create multiple frequency components
        assert len(peaks) > 1, "FM should create multiple frequency components (sidebands)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
