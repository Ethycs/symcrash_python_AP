"""
FM6 Gateway Headroom Tests (Packet E2)
======================================

The FM6->FWPT gateway auto-normalizes carrier levels when their sum exceeds
0.5 (Q15 16384): the FWPT analysis stage saturates above that threshold and
used to render ~23% H3 distortion for a pure sine carrier at level 1.0.
The IDWPT render stage's built-in 2x makeup gain (analysis headroom_bits=1,
render headroom_bits=2) preserves loudness, so normalizing is free.

Covers:
- Sweep: pure sine carrier, levels 0.3 -> 1.0, H3/H1 < 1% everywhere
  (including carrier sum 1.0, which previously hit ~23%)
- Bit-exactness: compliant level sets (sum <= 0.5) render identically to
  the no-gateway pipeline (gateway is a strict no-op)
- Gateway math: threshold edge, proportional scaling, modulators exempt
- JIT/reference equivalence through the gateway path
"""

import numpy as np
import pytest

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.vm import fm6_engine
from symcrash_python_ap.vm.fm6_engine import (
    FM6_CARRIER_SUM_MAX_Q15,
    fm6_gateway_level_overrides,
)
from symcrash_python_ap.vm.fm6_state import create_fm6_state
from symcrash_python_ap.vm.fm_algorithms import get_algorithm

SAMPLE_RATE = 48000
BLOCK = 64
NOTE = 69         # A4 = 440 Hz
F0 = 440.0
NUM_BLOCKS = 64
SKIP_BLOCKS = 16  # IDWPT warmup / attack
FFT_LEN = 2048


@pytest.fixture(autouse=True)
def _restore_jit_flag():
    """Always leave the global JIT flag as we found it."""
    before = hot.jit_enabled()
    yield
    hot.set_jit_enabled(before)


def _render_sine_carrier(level_q15: int, num_blocks: int = NUM_BLOCKS) -> np.ndarray:
    """Render a pure sine carrier (algorithm 1, modulators silent)."""
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=NOTE, cents=0)
    out = []
    for _ in range(num_blocks):
        frame = VoiceFrame(
            voice_id=0,
            note=NOTE,
            f0_hz=F0,
            coeff_osc_mode=3,  # OSC_MODE_FM6
            gain_q15=32767,
            fm_algorithm_id=1,  # single carrier (Op1)
            fm_op_ratios=[65536] * 6,
            fm_op_levels=[level_q15, 0, 0, 0, 0, 0],
            fm_feedback_depth=0,
            is_active=True,
        )
        state, block = vm_step_block(state, frame)
        out.append(np.asarray(block, dtype=np.int64).copy())
    return np.concatenate(out)


def _harmonic_ratio(audio: np.ndarray, harmonic: int) -> float:
    """Return |H_n| / |H_1| from a windowed FFT of the steady-state segment."""
    seg = audio[SKIP_BLOCKS * BLOCK: SKIP_BLOCKS * BLOCK + FFT_LEN].astype(np.float64)
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win))
    bin_hz = SAMPLE_RATE / len(seg)

    def peak(freq):
        center = int(round(freq / bin_hz))
        lo, hi = max(center - 3, 0), min(center + 4, len(spec))
        return spec[lo:hi].max()

    return peak(F0 * harmonic) / peak(F0)


class TestGatewayMath:
    """Unit behavior of fm6_gateway_level_overrides."""

    def test_noop_at_and_below_threshold(self):
        state = create_fm6_state(algorithm_id=1)
        state.operators[0].level_q15 = FM6_CARRIER_SUM_MAX_Q15  # exactly 0.5
        state.operators[1].level_q15 = 32767  # modulator: exempt
        for i in range(2, 6):
            state.operators[i].level_q15 = 0
        assert fm6_gateway_level_overrides(state) is None

    def test_engages_above_threshold(self):
        state = create_fm6_state(algorithm_id=1)
        state.operators[0].level_q15 = FM6_CARRIER_SUM_MAX_Q15 + 1
        overrides = fm6_gateway_level_overrides(state)
        assert overrides is not None
        assert overrides[0] <= FM6_CARRIER_SUM_MAX_Q15

    def test_proportional_scaling_and_modulator_exemption(self):
        # Algorithm 22: carriers Op1-4 (idx 0-3); Op5/Op6 modulators
        state = create_fm6_state(algorithm_id=22)
        levels = [32767, 16384, 16384, 8192, 30000, 25000]
        for i, lvl in enumerate(levels):
            state.operators[i].level_q15 = lvl

        overrides = fm6_gateway_level_overrides(state)
        assert overrides is not None

        alg = get_algorithm(22)
        carriers = alg.get_carriers()
        carrier_sum = sum(levels[c] for c in carriers)

        # Carriers: scaled proportionally, sum normalized to <= 0.5
        scaled_sum = sum(overrides[c] for c in carriers)
        assert scaled_sum <= FM6_CARRIER_SUM_MAX_Q15
        assert scaled_sum >= FM6_CARRIER_SUM_MAX_Q15 - len(carriers)  # floor rounding
        for c in carriers:
            assert overrides[c] == (levels[c] * FM6_CARRIER_SUM_MAX_Q15) // carrier_sum

        # Modulators: untouched
        for m in (4, 5):
            assert overrides[m] == levels[m]


class TestSweepClean:
    """Pure sine carrier sweep: harmonics stay below 1% at all levels."""

    @pytest.mark.parametrize("level", [0.3, 0.5, 0.6, 0.8, 1.0])
    def test_h3_below_one_percent(self, level):
        q15 = int(round(level * 32767))
        audio = _render_sine_carrier(q15)
        h3 = _harmonic_ratio(audio, 3)
        assert h3 < 0.01, f"H3/H1 = {h3 * 100:.2f}% at carrier level {level}"

    def test_carrier_sum_one_clean_and_loud(self):
        """Carrier sum 1.0: H2/H3/H5 < 1% and full-scale output (no loudness loss)."""
        audio = _render_sine_carrier(32767)
        for harmonic in (2, 3, 5):
            ratio = _harmonic_ratio(audio, harmonic)
            assert ratio < 0.01, (
                f"H{harmonic}/H1 = {ratio * 100:.2f}% at carrier sum 1.0"
            )
        peak = np.abs(audio[SKIP_BLOCKS * BLOCK:]).max()
        assert peak > 30000, f"makeup gain lost loudness: peak {peak}"


class TestCompliantBitExact:
    """Gateway is a strict no-op for carrier sums <= 0.5."""

    def test_render_matches_no_gateway_expectation(self, monkeypatch):
        level = 13107  # 0.4: compliant
        with_gateway = _render_sine_carrier(level, num_blocks=32)

        # Force the gateway off and re-render: must be bit-identical
        monkeypatch.setattr(
            fm6_engine, "fm6_gateway_level_overrides", lambda state, alg=None: None
        )
        without_gateway = _render_sine_carrier(level, num_blocks=32)

        assert np.array_equal(with_gateway, without_gateway), (
            "gateway changed output for a compliant (<= 0.5) carrier sum"
        )


class TestJitEquivalence:
    """JIT and reference paths agree through the gateway (sum > 0.5)."""

    def test_gateway_path_jit_matches_reference(self):
        if not hot.NUMBA_AVAILABLE:
            pytest.skip("numba not available")
        hot.set_jit_enabled(False)
        ref = _render_sine_carrier(32767, num_blocks=24)
        hot.set_jit_enabled(True)
        jit = _render_sine_carrier(32767, num_blocks=24)
        assert np.array_equal(ref, jit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
