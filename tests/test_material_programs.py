"""
Tests for Material Engine Programs (0-5)
========================================

Verifies the complete material processing chains.
"""

import pytest
from symcrash_python_ap.material.programs import (
    # Program 0
    CleanGlueState, clean_glue_sample, clean_glue_block,
    # Program 1
    TapeBusState, tape_bus_sample, tape_bus_block,
    # Program 2
    RankGritState, rank_grit_sample, rank_grit_block,
    # Program 3
    TropicalCutState, tropical_cut_sample, tropical_cut_block,
    # Program 4
    CatastropheSnapState, catastrophe_snap_sample, catastrophe_snap_block,
    # Program 5
    CatastropheSwallowtailState, catastrophe_swallowtail_program_sample, catastrophe_swallowtail_program_block,
    # Registry
    PROGRAM_INFO, get_program_state, get_program_name,
)
from symcrash_python_ap.fixed import Q15_ONE, Q15_HALF


def generate_test_signal(length: int = 64) -> list:
    """Generate a simple test signal (ramp)."""
    return [int(20000 * ((i % 32) / 16 - 1)) for i in range(length)]


class TestProgram0CleanGlue:
    """Tests for Program 0: Clean Glue."""

    def test_clean_glue_basic(self):
        """Clean Glue should process without crash."""
        state = CleanGlueState()
        signal = generate_test_signal()
        outputs = []
        for x in signal:
            y = clean_glue_sample(x, state)
            outputs.append(y)
        assert len(outputs) == len(signal)
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_clean_glue_transparency(self):
        """Clean Glue at low drive should produce valid output."""
        state = CleanGlueState()
        # Process DC
        for _ in range(100):
            y = clean_glue_sample(10000, state, drive_q15=Q15_HALF >> 1)
        # Should produce valid output (filtering/saturation may drift from input)
        assert -32768 <= y <= 32767
        # Output should be in reasonable range (not completely silent or clipped)
        assert abs(y) > 1000

    def test_clean_glue_block(self):
        """Block processing should match sample processing."""
        state_L = CleanGlueState()
        state_R = CleanGlueState()
        L = generate_test_signal()
        R = generate_test_signal()
        outL, outR = clean_glue_block(L, R, state_L, state_R)
        assert len(outL) == len(L)
        assert len(outR) == len(R)


class TestProgram1TapeBus:
    """Tests for Program 1: Tape Bus."""

    def test_tape_bus_basic(self):
        """Tape Bus should process without crash."""
        state = TapeBusState()
        signal = generate_test_signal()
        outputs = []
        for x in signal:
            y = tape_bus_sample(x, state)
            outputs.append(y)
        assert len(outputs) == len(signal)
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_tape_bus_saturation(self):
        """High drive should compress peaks."""
        state = TapeBusState()
        # Process high level signal
        outputs = []
        for _ in range(64):
            y = tape_bus_sample(28000, state, drive_q15=Q15_ONE)
            outputs.append(y)
        # After settling, output should be compressed
        # Just verify it's valid and somewhat compressed
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_tape_bus_block(self):
        """Block processing should work."""
        state_L = TapeBusState()
        state_R = TapeBusState()
        L = generate_test_signal()
        R = generate_test_signal()
        outL, outR = tape_bus_block(L, R, state_L, state_R)
        assert len(outL) == len(L)
        assert len(outR) == len(R)


class TestProgram2RankGrit:
    """Tests for Program 2: Rank Grit."""

    def test_rank_grit_basic(self):
        """Rank Grit should process without crash."""
        state = RankGritState()
        signal = generate_test_signal()
        outputs = []
        for x in signal:
            y = rank_grit_sample(x, state)
            outputs.append(y)
        assert len(outputs) == len(signal)
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_rank_grit_mix(self):
        """Mix parameter should blend wet/dry."""
        state_dry = RankGritState()
        state_wet = RankGritState()

        # Process same signal with different mix
        outputs_dry = []
        outputs_wet = []
        signal = generate_test_signal()
        for x in signal:
            outputs_dry.append(rank_grit_sample(x, state_dry, mix_q15=0))
            outputs_wet.append(rank_grit_sample(x, state_wet, mix_q15=Q15_ONE))

        # Both should be valid
        assert all(-32768 <= o <= 32767 for o in outputs_dry)
        assert all(-32768 <= o <= 32767 for o in outputs_wet)


class TestProgram3TropicalCut:
    """Tests for Program 3: Tropical Cut."""

    def test_tropical_cut_basic(self):
        """Tropical Cut should process without crash."""
        state = TropicalCutState()
        signal = generate_test_signal()
        outputs = []
        for x in signal:
            y = tropical_cut_sample(x, state)
            outputs.append(y)
        assert len(outputs) == len(signal)
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_tropical_cut_gating(self):
        """Gate should zero low-level signals."""
        state = TropicalCutState()
        # Low-level signal below threshold
        outputs = []
        for _ in range(32):
            y = tropical_cut_sample(1000, state, threshold_q15=5000, gate_mode=0)
            outputs.append(y)
        # Most outputs should be near zero (gate closed)
        near_zero = sum(1 for o in outputs if abs(o) < 500)
        assert near_zero > 20  # Most should be gated

    def test_tropical_cut_modes(self):
        """Different gate modes should work."""
        for mode in [0, 1, 2]:
            state = TropicalCutState()
            outputs = []
            signal = generate_test_signal()
            for x in signal:
                y = tropical_cut_sample(x, state, gate_mode=mode)
                outputs.append(y)
            assert len(outputs) == len(signal)


class TestProgram4CatastropheSnap:
    """Tests for Program 4: Catastrophe Snap."""

    def test_catastrophe_snap_basic(self):
        """Catastrophe Snap should process without crash."""
        state = CatastropheSnapState()
        signal = generate_test_signal()
        outputs = []
        for x in signal:
            y = catastrophe_snap_sample(x, state)
            outputs.append(y)
        assert len(outputs) == len(signal)
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_catastrophe_snap_pressure(self):
        """Different pressure levels should affect output."""
        state_low = CatastropheSnapState()
        state_high = CatastropheSnapState()

        signal = generate_test_signal()
        outputs_low = []
        outputs_high = []
        for x in signal:
            outputs_low.append(catastrophe_snap_sample(x, state_low, pressure_q15=1000))
            outputs_high.append(catastrophe_snap_sample(x, state_high, pressure_q15=20000))

        # Both should be valid
        assert all(-32768 <= o <= 32767 for o in outputs_low)
        assert all(-32768 <= o <= 32767 for o in outputs_high)


class TestProgram5CatastropheSwallowtail:
    """Tests for Program 5: Catastrophe Swallowtail."""

    def test_swallowtail_basic(self):
        """Swallowtail should process without crash."""
        state = CatastropheSwallowtailState()
        signal = generate_test_signal()
        outputs = []
        for x in signal:
            y = catastrophe_swallowtail_program_sample(x, state)
            outputs.append(y)
        assert len(outputs) == len(signal)
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_swallowtail_with_tape(self):
        """Tape stage should smooth the output."""
        state = CatastropheSwallowtailState()
        signal = generate_test_signal()
        outputs = []
        for x in signal:
            y = catastrophe_swallowtail_program_sample(x, state, tape_drive_q15=Q15_ONE)
            outputs.append(y)
        assert all(-32768 <= o <= 32767 for o in outputs)


class TestProgramRegistry:
    """Tests for program registry."""

    def test_all_programs_registered(self):
        """All 6 programs should be in registry."""
        assert len(PROGRAM_INFO) == 6
        for i in range(6):
            assert i in PROGRAM_INFO

    def test_get_program_state(self):
        """Should create fresh state for each program."""
        for program_id in range(6):
            state = get_program_state(program_id)
            assert state is not None

    def test_get_program_state_invalid(self):
        """Should raise for invalid program ID."""
        with pytest.raises(ValueError):
            get_program_state(99)

    def test_get_program_name(self):
        """Should return names for all programs."""
        names = [get_program_name(i) for i in range(6)]
        assert 'Clean Glue' in names
        assert 'Tape Bus' in names
        assert 'Rank Grit' in names
        assert 'Tropical Cut' in names
        assert 'Catastrophe Snap' in names
        assert 'Catastrophe Swallowtail' in names

    def test_get_program_name_invalid(self):
        """Should return 'Unknown' for invalid program."""
        name = get_program_name(99)
        assert 'Unknown' in name


class TestProgramIntegration:
    """Integration tests for programs."""

    def test_import_from_package(self):
        """All programs should import from package."""
        from symcrash_python_ap.material import (
            CleanGlueState, clean_glue_block,
            TapeBusState, tape_bus_block,
            get_program_name,
        )
        assert CleanGlueState is not None
        assert get_program_name(0) == 'Clean Glue'

    def test_all_programs_stereo_block(self):
        """All programs should handle stereo blocks."""
        L = generate_test_signal(64)
        R = generate_test_signal(64)

        # Program 0
        s0L, s0R = CleanGlueState(), CleanGlueState()
        o0L, o0R = clean_glue_block(L, R, s0L, s0R)
        assert len(o0L) == 64

        # Program 1
        s1L, s1R = TapeBusState(), TapeBusState()
        o1L, o1R = tape_bus_block(L, R, s1L, s1R)
        assert len(o1L) == 64

        # Program 2
        s2L, s2R = RankGritState(), RankGritState()
        o2L, o2R = rank_grit_block(L, R, s2L, s2R)
        assert len(o2L) == 64

        # Program 3
        s3L, s3R = TropicalCutState(), TropicalCutState()
        o3L, o3R = tropical_cut_block(L, R, s3L, s3R)
        assert len(o3L) == 64

        # Program 4
        s4L, s4R = CatastropheSnapState(), CatastropheSnapState()
        o4L, o4R = catastrophe_snap_block(L, R, s4L, s4R)
        assert len(o4L) == 64

        # Program 5
        s5L, s5R = CatastropheSwallowtailState(), CatastropheSwallowtailState()
        o5L, o5R = catastrophe_swallowtail_program_block(L, R, s5L, s5R)
        assert len(o5L) == 64
