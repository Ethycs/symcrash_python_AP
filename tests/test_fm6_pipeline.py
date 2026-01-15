#!/usr/bin/env python3
"""
Test FM6 Pipeline Integration
============================

Tests the FM6 synthesis pipeline with FWPT adapter for effects processing.

FM6 Pipeline architecture:
- Pipeline A (no effects): audio -> gain -> output
- Pipeline B (with effects): audio -> FWPT -> coeff_ops -> IDWPT -> output
"""

import pytest
import numpy as np

from symcrash_python_ap.vm.fm6_state import FM6State, Q15_ONE
from symcrash_python_ap.vm.fm6_engine import (
    fm6_generate_block_full,
    fwpt_analyze_block,
    configure_jump_brass,
    configure_brass,
    configure_bell,
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState, OSC_MODE_FM6
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.tables import BLOCK_SAMPLES


class TestFM6DirectAudio:
    """Test FM6 direct audio generation."""

    def test_jump_brass_generates_audio(self):
        """Jump brass preset generates valid audio."""
        state = FM6State()
        configure_jump_brass(state)
        for op in state.operators:
            op.env_value_q15 = Q15_ONE

        phase_inc = int((440 / 48000) * (2**32))
        audio = fm6_generate_block_full(state, phase_inc)

        assert audio.shape == (64,)
        assert audio.min() < 0  # Has negative samples
        assert audio.max() > 0  # Has positive samples
        assert np.sqrt(np.mean(audio.astype(np.float64)**2)) > 1000  # Has energy

    def test_dx7_brass_generates_audio(self):
        """DX7 brass preset generates valid audio."""
        state = FM6State()
        configure_brass(state)
        for op in state.operators:
            op.env_value_q15 = Q15_ONE

        phase_inc = int((440 / 48000) * (2**32))
        audio = fm6_generate_block_full(state, phase_inc)

        assert audio.shape == (64,)
        assert audio.min() < 0
        assert audio.max() > 0

    def test_bell_generates_audio(self):
        """Bell preset generates valid audio."""
        state = FM6State()
        configure_bell(state)
        for op in state.operators:
            op.env_value_q15 = Q15_ONE

        phase_inc = int((440 / 48000) * (2**32))
        audio = fm6_generate_block_full(state, phase_inc)

        assert audio.shape == (64,)
        assert audio.min() < 0
        assert audio.max() > 0


class TestFWPTAdapter:
    """Test FWPT adapter for coefficient-domain effects."""

    def test_fwpt_produces_64_coefficients(self):
        """FWPT analysis produces 64 coefficients."""
        state = FM6State()
        configure_jump_brass(state)
        for op in state.operators:
            op.env_value_q15 = Q15_ONE

        phase_inc = int((440 / 48000) * (2**32))
        audio = fm6_generate_block_full(state, phase_inc)
        coeffs = fwpt_analyze_block(audio)

        assert coeffs.shape == (64,)
        assert coeffs.dtype == np.int16

    def test_fwpt_idwpt_round_trip_preserves_audio(self):
        """FWPT->IDWPT round-trip preserves audio fidelity."""
        state = FM6State()
        configure_jump_brass(state)
        for op in state.operators:
            op.env_value_q15 = Q15_ONE

        phase_inc = int((440 / 48000) * (2**32))
        audio = fm6_generate_block_full(state, phase_inc)

        # Forward transform
        coeffs = fwpt_analyze_block(audio)

        # Inverse transform
        reconstructed = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)

        # Compare
        diff = audio.astype(np.float64) - reconstructed.astype(np.float64)
        max_error = np.max(np.abs(diff))
        corr = np.corrcoef(audio.astype(np.float64), reconstructed.astype(np.float64))[0, 1]

        assert max_error <= 2, f"Max error {max_error} > 2 LSB"
        assert corr > 0.999, f"Correlation {corr} < 0.999"

    def test_sine_wave_round_trip(self):
        """Pure sine wave round-trip is near-perfect."""
        freq = 440
        sample_rate = 48000
        t = np.arange(64) / sample_rate
        test_signal = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

        coeffs = fwpt_analyze_block(test_signal)
        reconstructed = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)

        corr = np.corrcoef(test_signal.astype(np.float64), reconstructed.astype(np.float64))[0, 1]
        assert corr > 0.9999, f"Sine wave correlation {corr} < 0.9999"


class TestExecutorIntegration:
    """Test FM6 integration with VM executor."""

    def test_executor_fm6_direct_path(self):
        """Executor FM6 produces valid audio (no effects)."""
        frame = VoiceFrame(
            voice_id=1,
            note=69,
            cents=0,
            f0_hz=440.0,
            gain_q15=32767,
            beta_target_q15=0,
            partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
            is_active=True,
            coeff_osc_mode=OSC_MODE_FM6,
            fm_algorithm_id=19,
        )

        vm_state = VMState.create(1)
        vm_state.use_coeff_osc = True

        vm_state, output = vm_step_block(vm_state, frame)

        assert output.shape == (BLOCK_SAMPLES,)
        assert output.min() < 0 or output.max() > 0  # Has audio

    def test_executor_fm6_multiple_blocks(self):
        """Executor FM6 maintains state across blocks."""
        frame = VoiceFrame(
            voice_id=1,
            note=69,
            cents=0,
            f0_hz=440.0,
            gain_q15=32767,
            beta_target_q15=0,
            partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
            is_active=True,
            coeff_osc_mode=OSC_MODE_FM6,
            fm_algorithm_id=22,  # DX7 Brass algorithm
        )

        vm_state = VMState.create(1)
        vm_state.use_coeff_osc = True

        audio_blocks = []
        for _ in range(10):
            vm_state, output = vm_step_block(vm_state, frame)
            audio_blocks.append(output)

        full_audio = np.concatenate(audio_blocks)

        # Check for zero crossings (not just DC)
        zero_crossings = np.sum(np.diff(np.sign(full_audio.astype(np.float64))) != 0)
        assert zero_crossings > 10, f"Only {zero_crossings} zero crossings - might be DC"

        # Check RMS is reasonable
        rms = np.sqrt(np.mean(full_audio.astype(np.float64)**2))
        assert rms > 1000, f"RMS {rms} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
