"""
FIXED-POINT ARITHMETIC TESTS
=============================

WSE-STD-001 Conformance tests for the fixed-point arithmetic layer.

Tests:
- RSHIFT_RNE (round-to-nearest, ties-to-even)
- MUL_Q15 edge cases
- ABS16 saturating behavior
- SAT16 boundaries
- ADD_Q15 saturation
- MIX_Q15 crossfade
- Vector operations
"""

import pytest
from symcrash_python_ap.fixed import (
    RSHIFT_RNE, MUL_Q15, ABS16, SAT16, ADD_Q15, MIX_Q15,
    vec_mul_q15, vec_add_sat,
    Q15_ONE, ONE_Q15, Q15_HALF, MIN_S16, MAX_S16,
)


class TestRshiftRne:
    """RSHIFT_RNE - round-to-nearest, ties-to-even right shift."""

    def test_positive_tie_odd_quotient_rounds_up(self):
        """6 >> 2: q=1 (odd), r=2=half -> rounds to 2."""
        assert RSHIFT_RNE(6, 2) == 2

    def test_positive_no_remainder(self):
        """4 >> 2: q=1, r=0 -> 1."""
        assert RSHIFT_RNE(4, 2) == 1

    def test_positive_tie_even_quotient_stays(self):
        """10 >> 2: q=2 (even), r=2=half -> stays 2."""
        assert RSHIFT_RNE(10, 2) == 2

    def test_positive_tie_odd_quotient_rounds_up_2(self):
        """14 >> 2: q=3 (odd), r=2=half -> rounds to 4."""
        assert RSHIFT_RNE(14, 2) == 4

    def test_positive_round_up(self):
        """7 >> 2: q=1, r=3 > half -> rounds up to 2."""
        assert RSHIFT_RNE(7, 2) == 2

    def test_positive_truncate(self):
        """5 >> 2: q=1, r=1 < half -> truncates to 1."""
        assert RSHIFT_RNE(5, 2) == 1

    def test_negative_tie_odd_quotient(self):
        """-6 >> 2: abs=6, q=1 (odd) -> rounds to -2."""
        assert RSHIFT_RNE(-6, 2) == -2

    def test_negative_tie_even_quotient(self):
        """-10 >> 2: abs=10, q=2 (even) -> -2."""
        assert RSHIFT_RNE(-10, 2) == -2

    def test_large_shift_q15(self):
        """0x8000 >> 15 with RNE."""
        assert RSHIFT_RNE(0x8000, 15) == 1


class TestMulQ15:
    """MUL_Q15 Q1.15 multiplication edge cases."""

    def test_negative_min_squared_saturates(self):
        """-32768 * -32768 should saturate to MAX_S16."""
        result = MUL_Q15(-32768, -32768)
        assert result == MAX_S16

    def test_positive_max_squared(self):
        """32767 * 32767 = 32766 (just under max)."""
        result = MUL_Q15(32767, 32767)
        assert result == 32766

    def test_mixed_signs(self):
        """-32768 * 32767 = -32767."""
        result = MUL_Q15(-32768, 32767)
        assert result == -32767

    def test_multiply_by_half(self):
        """20000 * 16384 (half) = 10000."""
        result = MUL_Q15(20000, 16384)
        assert result == 10000


class TestAbs16:
    """ABS16 saturating absolute value."""

    def test_minint_saturates(self):
        """ABS16(-32768) = 32767 (saturating)."""
        assert ABS16(-32768) == 32767

    def test_negative(self):
        """ABS16(-100) = 100."""
        assert ABS16(-100) == 100

    def test_positive(self):
        """ABS16(100) = 100."""
        assert ABS16(100) == 100

    def test_zero(self):
        """ABS16(0) = 0."""
        assert ABS16(0) == 0


class TestSat16:
    """SAT16 boundary saturation tests."""

    def test_positive_overflow(self):
        """Values above 32767 saturate to 32767."""
        assert SAT16(40000) == 32767

    def test_negative_overflow(self):
        """Values below -32768 saturate to -32768."""
        assert SAT16(-40000) == -32768

    def test_max_boundary(self):
        """32767 stays 32767."""
        assert SAT16(32767) == 32767

    def test_min_boundary(self):
        """-32768 stays -32768."""
        assert SAT16(-32768) == -32768

    def test_zero(self):
        """0 stays 0."""
        assert SAT16(0) == 0


class TestAddQ15:
    """ADD_Q15 saturating addition."""

    def test_positive_overflow(self):
        """30000 + 30000 saturates to 32767."""
        result = ADD_Q15(30000, 30000)
        assert result == 32767

    def test_negative_overflow(self):
        """-30000 + -30000 saturates to -32768."""
        result = ADD_Q15(-30000, -30000)
        assert result == -32768

    def test_normal_add(self):
        """1000 + 2000 = 3000."""
        result = ADD_Q15(1000, 2000)
        assert result == 3000


class TestMixQ15:
    """MIX_Q15 weighted crossfade."""

    def test_t_zero_returns_x(self):
        """t=0 should return ~x (scaled by ONE)."""
        result = MIX_Q15(10000, 20000, 0)
        # MUL_Q15(10000, 32767) = 10000 * 32767 >> 15 = 9999
        assert 9998 <= result <= 10001

    def test_t_one_returns_y(self):
        """t=ONE should return ~y."""
        result = MIX_Q15(10000, 20000, ONE_Q15)
        assert 19998 <= result <= 20001

    def test_t_half_returns_average(self):
        """t=HALF should return approximately average."""
        result = MIX_Q15(10000, 20000, Q15_HALF)
        # Should be around 15000
        assert 14000 <= result <= 16000


class TestVectorOps:
    """Vector operations."""

    def test_vec_mul_q15(self):
        """Element-wise multiply by 0.5."""
        a = [10000, 20000, 30000]
        b = [16384, 16384, 16384]  # 0.5 in Q15
        result = vec_mul_q15(a, b)
        # Should be approximately [5000, 10000, 15000]
        assert 4900 <= result[0] <= 5100
        assert 9900 <= result[1] <= 10100
        assert 14900 <= result[2] <= 15100

    def test_vec_add_sat_overflow(self):
        """Vector add with saturation."""
        result = vec_add_sat([30000, 30000], [10000, 10000])
        assert result == [32767, 32767]


def test_std001_conformance():
    """Run complete WSE-STD-001 conformance suite."""
    # All tests above cover the conformance requirements.
    # This test ensures the full suite passes.

    # RSHIFT_RNE tests
    assert RSHIFT_RNE(6, 2) == 2
    assert RSHIFT_RNE(10, 2) == 2
    assert RSHIFT_RNE(14, 2) == 4
    assert RSHIFT_RNE(-6, 2) == -2
    assert RSHIFT_RNE(-10, 2) == -2

    # MUL_Q15 tests
    assert MUL_Q15(-32768, -32768) == MAX_S16
    assert MUL_Q15(32767, 32767) == 32766
    assert MUL_Q15(-32768, 32767) == -32767
    assert MUL_Q15(20000, 16384) == 10000

    # ABS16 tests
    assert ABS16(-32768) == 32767
    assert ABS16(-100) == 100
    assert ABS16(0) == 0

    # SAT16 tests
    assert SAT16(40000) == 32767
    assert SAT16(-40000) == -32768

    # ADD_Q15 tests
    assert ADD_Q15(30000, 30000) == 32767
    assert ADD_Q15(-30000, -30000) == -32768
