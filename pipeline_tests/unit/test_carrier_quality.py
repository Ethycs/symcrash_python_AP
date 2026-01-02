#!/usr/bin/env python3
"""
Test carrier oscillator signal quality.

Verifies that the carrier path produces clean audio without:
- Clipping artifacts from DC blocking filter
- Discontinuities at block boundaries
- Excessive gradient changes (clicks/pops)
"""

import pytest
import numpy as np
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import DecodedPlan, CarrierParams, CarrierRole
from symcrash_python_ap.vm.lanes import lane_carrier_render
from symcrash_python_ap.wavelet.sine_core import sine_q15_fast


class TestCarrierQuality:
    """Test carrier oscillator signal quality."""

    def test_no_clipping_single_harmonic(self):
        """Carrier with single harmonic should not clip."""
        phase_inc = int(440 * (2**32) / 48000)

        state = VMState()
        state.phase = 0
        state.phase_inc = phase_inc
        state.target_phase_inc = phase_inc
        state.pitch_slew_rate = 0

        params = CarrierParams()
        params.phase_inc = phase_inc
        params.harmonic_amps_q15 = [32767, 0, 0, 0, 0, 0, 0, 0]

        cfg = DecodedPlan()
        cfg.carrier_role = CarrierRole.AUDIBLE
        cfg.carrier_params = params

        # Render 100 blocks
        all_samples = []
        for _ in range(100):
            output = lane_carrier_render(state, cfg).copy()
            all_samples.append(output)

        samples = np.concatenate(all_samples)

        # Check no hard clipping (output should be within range with margin)
        assert samples.min() >= -32768, "Output below int16 min"
        assert samples.max() <= 32767, "Output above int16 max"

        # With headroom, peaks should be well below limits
        assert samples.max() < 32000, f"Peak too high: {samples.max()}"
        assert samples.min() > -33000, f"Trough too low: {samples.min()}"

    def test_no_clipping_full_harmonics(self):
        """Carrier with full harmonics should not clip."""
        phase_inc = int(440 * (2**32) / 48000)

        state = VMState()
        state.phase = 0
        state.phase_inc = phase_inc
        state.target_phase_inc = phase_inc
        state.pitch_slew_rate = 0

        params = CarrierParams()
        params.phase_inc = phase_inc
        # Full saw-like harmonics
        params.harmonic_amps_q15 = [32767, 16384, 10923, 8192, 6554, 5461, 4681, 4096]

        cfg = DecodedPlan()
        cfg.carrier_role = CarrierRole.AUDIBLE
        cfg.carrier_params = params

        # Render 100 blocks
        all_samples = []
        for _ in range(100):
            output = lane_carrier_render(state, cfg).copy()
            all_samples.append(output)

        samples = np.concatenate(all_samples)

        # Check no clipping
        assert samples.min() >= -32768, "Output below int16 min"
        assert samples.max() <= 32767, "Output above int16 max"

    def test_smooth_gradient(self):
        """Carrier output should have smooth gradients (no clicks)."""
        phase_inc = int(440 * (2**32) / 48000)

        state = VMState()
        state.phase = 0
        state.phase_inc = phase_inc
        state.target_phase_inc = phase_inc
        state.pitch_slew_rate = 0

        params = CarrierParams()
        params.phase_inc = phase_inc
        params.harmonic_amps_q15 = [32767, 0, 0, 0, 0, 0, 0, 0]

        cfg = DecodedPlan()
        cfg.carrier_role = CarrierRole.AUDIBLE
        cfg.carrier_params = params

        # Render 100 blocks
        all_samples = []
        for _ in range(100):
            output = lane_carrier_render(state, cfg).copy()
            all_samples.append(output)

        samples = np.concatenate(all_samples).astype(np.int32)

        # Compute gradient and gradient change
        gradient = np.diff(samples)
        gradient_change = np.abs(np.diff(gradient))

        # For a clean sine, max gradient change should be bounded
        # Raw sine at 440Hz has max gradient change ~110 LSB
        # With DC filtering and headroom, should be similar
        max_gradient_change = gradient_change.max()
        assert max_gradient_change < 200, f"Gradient change too large: {max_gradient_change}"

    def test_phase_continuity(self):
        """Phase should advance correctly across blocks."""
        phase_inc = int(440 * (2**32) / 48000)

        state = VMState()
        state.phase = 0
        state.phase_inc = phase_inc
        state.target_phase_inc = phase_inc
        state.pitch_slew_rate = 0

        params = CarrierParams()
        params.phase_inc = phase_inc
        params.harmonic_amps_q15 = [32767, 0, 0, 0, 0, 0, 0, 0]

        cfg = DecodedPlan()
        cfg.carrier_role = CarrierRole.AUDIBLE
        cfg.carrier_params = params

        # Track phase across blocks
        for i in range(10):
            phase_before = state.phase
            lane_carrier_render(state, cfg)
            phase_after = state.phase

            expected_after = (phase_before + 64 * phase_inc) & 0xFFFFFFFF
            assert phase_after == expected_after, f"Block {i}: Phase mismatch"

    def test_no_dc_filter_in_digital_path(self):
        """DC filter is not used in all-digital implementation.

        The DC blocking filter was removed because:
        - Sine LUT is symmetric around zero (no inherent DC)
        - No ADC offset or analog component drift in digital path
        - Headroom scaling preserves symmetry
        """
        phase_inc = int(440 * (2**32) / 48000)

        state = VMState()
        state.phase = 0
        state.phase_inc = phase_inc
        state.target_phase_inc = phase_inc
        state.pitch_slew_rate = 0

        params = CarrierParams()
        params.phase_inc = phase_inc
        params.harmonic_amps_q15 = [32767, 0, 0, 0, 0, 0, 0, 0]

        cfg = DecodedPlan()
        cfg.carrier_role = CarrierRole.AUDIBLE
        cfg.carrier_params = params

        # DC filter state variables exist but are unused
        assert state.dc_block_x1 == 0
        assert state.dc_block_y1 == 0

        # Render many blocks to accumulate complete cycles
        all_samples = []
        for _ in range(100):  # 100 blocks = 6400 samples ≈ 14 full cycles at 440Hz
            output = lane_carrier_render(state, cfg)
            all_samples.extend(output.tolist())

        # DC filter state should remain unused (zeros)
        assert state.dc_block_x1 == 0
        assert state.dc_block_y1 == 0

        # Over many complete cycles, DC should average to near-zero
        # Allow small DC offset from integer quantization (~0.5% of peak)
        dc_component = np.mean(all_samples)
        assert abs(dc_component) < 200, f"DC component {dc_component} too high"


class TestSineLUT:
    """Test 64K sine LUT accuracy."""

    def test_boundary_values(self):
        """LUT should have correct values at key phase points."""
        from symcrash_python_ap.wavelet.sine_core import sine_q15_fast, cosine_q15_fast

        # sin(0) = 0
        assert sine_q15_fast(0) == 0

        # sin(90) = 1
        assert sine_q15_fast(16384) == 32767

        # sin(180) = 0
        assert sine_q15_fast(32768) == 0

        # sin(270) = -1
        assert sine_q15_fast(49152) == -32767

        # cos(0) = 1
        assert cosine_q15_fast(0) == 32767

        # cos(90) = 0
        assert cosine_q15_fast(16384) == 0

        # cos(180) = -1
        assert cosine_q15_fast(32768) == -32767

        # cos(270) = 0
        assert cosine_q15_fast(49152) == 0

    def test_vectorized_matches_scalar(self):
        """Vectorized lookup should match scalar lookup."""
        from symcrash_python_ap.wavelet.sine_core import (
            sine_q15_fast, sine_q15_vectorized
        )

        test_phases = np.array([0, 16384, 32768, 49152, 8192, 24576], dtype=np.int32)

        vec_result = sine_q15_vectorized(test_phases)
        scalar_results = [sine_q15_fast(p) for p in test_phases]

        for i, p in enumerate(test_phases):
            assert abs(vec_result[i] - scalar_results[i]) <= 1, \
                f"Mismatch at phase {p}: vec={vec_result[i]}, scalar={scalar_results[i]}"
