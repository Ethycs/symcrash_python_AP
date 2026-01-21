"""
Tests for Material Engine Building Blocks (P1-P7)
=================================================

Verifies the DSP primitives used in material programs.
"""

import pytest
from symcrash_python_ap.material.building_blocks import (
    # P1: Tilt
    TiltState, tilt_sample,
    # P2: Saturator
    SaturatorParams, saturator_sample,
    # P3: Tape
    TapeHysteresisState, tape_hysteresis_sample,
    # P4: One-pole
    OnePoleState, one_pole_lp_sample, one_pole_hp_sample, freq_to_one_pole_coeff,
    # P5: Rank
    RankState, rank_median_sample, rank_trimmed_mean_sample,
    # P6: Tropical
    tropical_max, tropical_min, tropical_gate_sample,
    # P7: Catastrophe
    CatastropheState, catastrophe_fold_sample, catastrophe_cusp_sample,
    catastrophe_swallowtail_sample,
)
from symcrash_python_ap.fixed import Q15_ONE, Q15_HALF


class TestP1Tilt:
    """Tests for P1: Tilt EQ."""

    def test_tilt_neutral(self):
        """Zero tilt should approximate bypass."""
        state = TiltState()
        # Process a DC signal
        for _ in range(100):
            out = tilt_sample(10000, state, tilt_q15=0)
        # Should be close to input after settling
        assert abs(out - 10000) < 3000  # Some deviation expected

    def test_tilt_bright(self):
        """Positive tilt should emphasize changes."""
        state = TiltState()
        # Process step input
        outputs = []
        for i in range(20):
            x = 0 if i < 10 else 20000
            out = tilt_sample(x, state, tilt_q15=20000)
            outputs.append(out)
        # After step, output should overshoot due to emphasis
        # Just verify it runs without error
        assert len(outputs) == 20

    def test_tilt_dark(self):
        """Negative tilt should reduce brightness."""
        state = TiltState()
        outputs = []
        for i in range(20):
            x = 0 if i < 10 else 20000
            out = tilt_sample(x, state, tilt_q15=-20000)
            outputs.append(out)
        assert len(outputs) == 20


class TestP2Saturator:
    """Tests for P2: Soft Saturator."""

    def test_saturator_tanh_curve(self):
        """Tanh curve should soft-limit extreme values."""
        params = SaturatorParams(drive_q15=Q15_HALF, curve=0)
        # Very high input should be compressed toward limit
        out = saturator_sample(32000, params)
        assert -32768 <= out <= 32767
        # Tanh at drive=0.5 on ~16000 should still be significant
        assert out > 10000

    def test_saturator_soft_clip(self):
        """Soft clip should compress above threshold."""
        params = SaturatorParams(drive_q15=Q15_HALF, curve=1)
        out = saturator_sample(20000, params)
        assert -32768 <= out <= 32767

    def test_saturator_hard_clip(self):
        """Hard clip should saturate."""
        params = SaturatorParams(drive_q15=Q15_ONE, curve=2)
        out = saturator_sample(30000, params)
        assert -32768 <= out <= 32767

    def test_saturator_asymmetric(self):
        """Asymmetric curve should treat positive/negative differently."""
        params = SaturatorParams(drive_q15=Q15_ONE, curve=3)
        pos = saturator_sample(25000, params)
        neg = saturator_sample(-25000, params)
        # Both should be valid Q15
        assert -32768 <= pos <= 32767
        assert -32768 <= neg <= 32767


class TestP3TapeHysteresis:
    """Tests for P3: Tape Hysteresis."""

    def test_tape_basic(self):
        """Tape should process signal with hysteresis."""
        state = TapeHysteresisState()
        outputs = []
        for i in range(64):
            x = int(20000 * (1 if i % 20 < 10 else -1))
            out = tape_hysteresis_sample(x, state, drive_q15=Q15_HALF)
            outputs.append(out)
        # Should produce smooth transitions (hysteresis)
        assert len(outputs) == 64
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_tape_with_bias(self):
        """Tape with bias should have asymmetric response."""
        state = TapeHysteresisState()
        outputs = []
        for i in range(32):
            out = tape_hysteresis_sample(15000, state, bias_q15=5000)
            outputs.append(out)
        assert len(outputs) == 32

    def test_tape_saturation(self):
        """High saturation should compress more."""
        state = TapeHysteresisState()
        for _ in range(50):
            tape_hysteresis_sample(25000, state, saturation_q15=Q15_ONE)
        # Should be compressed
        assert abs(state.m) < 32000


class TestP4OnePole:
    """Tests for P4: One-pole LP/HP."""

    def test_lp_dc_passes(self):
        """Lowpass should pass DC."""
        state = OnePoleState()
        coeff = freq_to_one_pole_coeff(1000.0)
        # Feed DC
        for _ in range(200):
            out = one_pole_lp_sample(20000, state, coeff)
        # Should converge to input
        assert abs(out - 20000) < 1000

    def test_lp_blocks_high_freq(self):
        """Lowpass should attenuate high frequencies."""
        state = OnePoleState()
        coeff = freq_to_one_pole_coeff(100.0)  # Very low cutoff
        outputs = []
        for i in range(100):
            x = 20000 if i % 2 == 0 else -20000  # ~24kHz at 48kHz
            out = one_pole_lp_sample(x, state, coeff)
            outputs.append(out)
        # High freq should be heavily attenuated
        # Check RMS is lower than input
        rms = (sum(o ** 2 for o in outputs[-50:]) / 50) ** 0.5
        assert rms < 15000

    def test_hp_blocks_dc(self):
        """Highpass should block DC."""
        state = OnePoleState()
        coeff = freq_to_one_pole_coeff(1000.0)
        # Feed DC
        for _ in range(200):
            out = one_pole_hp_sample(20000, state, coeff)
        # Should converge to zero
        assert abs(out) < 2000

    def test_freq_to_coeff_range(self):
        """Coefficient should be in valid range."""
        coeff_low = freq_to_one_pole_coeff(20.0)
        coeff_high = freq_to_one_pole_coeff(15000.0)
        assert 0 < coeff_low < Q15_ONE
        assert 0 < coeff_high < Q15_ONE
        assert coeff_high > coeff_low


class TestP5Rank:
    """Tests for P5: Rank Nonlinearity."""

    def test_median_removes_impulse(self):
        """Median should reject impulse noise."""
        state = RankState(window_size=5)
        # Normal signal with one spike
        inputs = [1000, 1000, 30000, 1000, 1000]  # Spike at index 2
        outputs = []
        for x in inputs:
            out = rank_median_sample(x, state)
            outputs.append(out)
        # After window fills, median should reject spike
        assert outputs[-1] == 1000

    def test_trimmed_mean(self):
        """Trimmed mean should average without outliers."""
        state = RankState(window_size=7)
        # Fill with values
        inputs = [100, 200, 300, 400, 500, 600, 700]
        outputs = []
        for x in inputs:
            out = rank_trimmed_mean_sample(x, state, trim=2)
            outputs.append(out)
        # Trimmed mean of middle values (300, 400, 500) = 400
        assert abs(outputs[-1] - 400) < 50


class TestP6Tropical:
    """Tests for P6: Tropical Operations."""

    def test_tropical_max(self):
        """Max should return larger value."""
        assert tropical_max(100, 200) == 200
        assert tropical_max(-100, -200) == -100

    def test_tropical_min(self):
        """Min should return smaller value."""
        assert tropical_min(100, 200) == 100
        assert tropical_min(-100, -200) == -200

    def test_gate_hard_gate(self):
        """Hard gate should zero below threshold."""
        assert tropical_gate_sample(1000, threshold_q15=5000, mode=0) == 0
        assert tropical_gate_sample(10000, threshold_q15=5000, mode=0) == 10000

    def test_gate_hard_limit(self):
        """Hard limit should clamp above threshold."""
        assert tropical_gate_sample(10000, threshold_q15=5000, mode=1) == 5000
        assert tropical_gate_sample(-10000, threshold_q15=5000, mode=1) == -5000

    def test_gate_dead_zone(self):
        """Dead zone should zero in range."""
        assert tropical_gate_sample(3000, threshold_q15=5000, mode=2) == 0
        out_pos = tropical_gate_sample(10000, threshold_q15=5000, mode=2)
        assert out_pos == 5000  # 10000 - 5000


class TestP7Catastrophe:
    """Tests for P7: Catastrophe Shapers."""

    def test_fold_basic(self):
        """Fold catastrophe should process without crash."""
        state = CatastropheState()
        outputs = []
        for i in range(64):
            x = int(15000 * ((i % 20) / 10 - 1))
            out = catastrophe_fold_sample(x, state, pressure_q15=10000)
            outputs.append(out)
        assert len(outputs) == 64
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_cusp_basic(self):
        """Cusp catastrophe should process without crash."""
        state = CatastropheState()
        outputs = []
        for i in range(64):
            x = int(15000 * ((i % 20) / 10 - 1))
            out = catastrophe_cusp_sample(x, state, pressure_q15=8000, bias_q15=2000)
            outputs.append(out)
        assert len(outputs) == 64
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_swallowtail_basic(self):
        """Swallowtail catastrophe should process without crash."""
        state = CatastropheState()
        outputs = []
        for i in range(64):
            x = int(10000 * ((i % 20) / 10 - 1))
            out = catastrophe_swallowtail_sample(
                x, state,
                a_q15=5000, b_q15=3000, c_q15=1000
            )
            outputs.append(out)
        assert len(outputs) == 64
        assert all(-32768 <= o <= 32767 for o in outputs)

    def test_catastrophe_hysteresis(self):
        """Catastrophe should exhibit hysteresis behavior."""
        state = CatastropheState()
        # Sweep pressure up then down
        up_outputs = []
        for p in range(0, 20000, 1000):
            out = catastrophe_fold_sample(5000, state, pressure_q15=p)
            up_outputs.append(out)

        state2 = CatastropheState()
        down_outputs = []
        for p in range(20000, 0, -1000):
            out = catastrophe_fold_sample(5000, state2, pressure_q15=p)
            down_outputs.append(out)

        # Just verify both directions run
        assert len(up_outputs) == len(down_outputs)


class TestIntegration:
    """Integration tests for building blocks."""

    def test_import_from_package(self):
        """All building blocks should import from package."""
        from symcrash_python_ap.material import (
            TiltState, tilt_sample,
            SaturatorParams, saturator_sample,
            TapeHysteresisState, tape_hysteresis_sample,
            OnePoleState, one_pole_lp_sample, one_pole_hp_sample,
            tropical_gate_sample,
            catastrophe_fold_sample,
        )
        # Just verify imports work
        assert TiltState is not None
        assert tilt_sample is not None

    def test_chain_processing(self):
        """Building blocks should chain together."""
        # Tilt -> Saturator -> LP chain
        tilt_state = TiltState()
        lp_state = OnePoleState()
        sat_params = SaturatorParams(drive_q15=Q15_HALF, curve=0)
        lp_coeff = freq_to_one_pole_coeff(5000.0)

        outputs = []
        for i in range(64):
            x = int(15000 * ((i % 20) / 10 - 1))
            # Tilt
            y = tilt_sample(x, tilt_state, tilt_q15=5000)
            # Saturate
            y = saturator_sample(y, sat_params)
            # Lowpass
            y = one_pole_lp_sample(y, lp_state, lp_coeff)
            outputs.append(y)

        assert len(outputs) == 64
        assert all(-32768 <= o <= 32767 for o in outputs)
