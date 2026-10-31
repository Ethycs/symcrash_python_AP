"""E3 interpolated VM kernel parity, independent RNE oracle, and scratch bounds."""
from fractions import Fraction

import numpy as np
import pytest
from symcrash_python_ap.vm import hotpath_jit as hot, harmonic_bank as bank

original = hot._mode_b_interpolated_coefficients_leaf_order
coefficients = hot.mode_b_interpolated_coefficients


def inputs(harmonics, slots, seed=75):
    rng = np.random.default_rng(seed)
    anchors = rng.integers(-32768, 32768, 12 * 64 * slots, dtype=np.int16)
    anchors[:8] = [-32768, 32767, 0, 1, -1, 16384, -16384, 3]
    offsets = rng.integers(0, 12, (harmonics, 4, 2), dtype=np.int32) * (64 * slots)
    weights = rng.integers(-(1 << 28), 1 << 28, (harmonics, 4), dtype=np.int32)
    weights[:, 3] = (1 << 30) - weights[:, :3].astype(np.int64).sum(axis=1)
    if harmonics:
        weights[0] = [-(1 << 30) + 1, (1 << 31) - 1, 0, 0]
    a = rng.integers(-32768, 32768, harmonics, dtype=np.int64)
    b = rng.integers(-32768, 32768, harmonics, dtype=np.int64)
    phases = rng.integers(0, 1 << 32, slots, dtype=np.int64)
    phases[:min(slots, 4)] = [0, 0xffffffff, 0x80000000, 0x40000000][:slots]
    dc = rng.integers(-32768, 32768, (64, slots), dtype=np.int16)
    arrays = [hot._get_sine_lut_np(), a, b, -32768, phases, dc, anchors, offsets, weights]
    for array in arrays:
        if isinstance(array, np.ndarray):
            array.flags.writeable = False
    return arrays


@pytest.mark.parametrize('harmonics', [0, 1, 8, 16, 31, 32, 64])
@pytest.mark.parametrize('slots', [2, 4, 8])
def test_kernel_preserves_every_coefficient_and_output_tail(harmonics, slots):
    args = inputs(harmonics, slots)
    before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
    reference = np.full(64 * slots + 7, -12345, dtype=np.int16)
    actual = reference.copy()
    original(*args, reference)
    coefficients(*args, actual)
    np.testing.assert_array_equal(actual, reference)
    assert actual[-7:].tolist() == [-12345] * 7
    for old, new in zip(before, args):
        np.testing.assert_array_equal(old, new)


def test_reordered_h32_path_matches_independent_fraction_arithmetic():
    args = inputs(32, 4, 752)
    lut, a, b, dc, phases, dc_frame, anchors, offsets, weights = args
    clip = lambda x: max(-32768, min(32767, x))
    rne = lambda x, shift: round(Fraction(int(x), 1 << shift))
    output = np.empty(256, dtype=np.int16)
    coefficients(*args, output)
    expected = []
    for leaf in range(64):
        for k in range(4):
            acc = rne(dc * int(dc_frame[leaf, k]), 15)
            for h in range(32):
                sine, cosine = bank.sin_cos_q15((int(phases[k]) * (h + 1)) & 0xffffffff)
                ws = rne(int(a[h]) * cosine - int(b[h]) * sine, 15)
                wc = rne(int(a[h]) * sine + int(b[h]) * cosine, 15)
                basis = [clip(rne(sum(int(weights[h, j]) * int(anchors[int(offsets[h, j, component]) + leaf * 4 + k])
                                     for j in range(4)), 30)) for component in (0, 1)]
                acc += rne(ws * basis[1], 15)
                acc += rne(wc * basis[0], 15)
            expected.append(clip(acc))
    np.testing.assert_array_equal(output, expected)


def test_scratch_and_arithmetic_bounds():
    for h in (32, 64):
        for k in (2, 4, 8):
            assert 64 * k * 8 <= 2 * h * k * 8
    assert 64 * 4 * 8 == 2048
    # Four signed Q30/int32 weights multiply signed i16 anchor samples.
    assert 4 * (1 << 31) * (1 << 15) < 1 << 63
    # Signed i16 a/b rotate against i16 sine/cosine; the RNE result fits 17 bits.
    assert 2 * (1 << 15) ** 2 < 1 << 32
    assert (1 << 16) * (1 << 15) < 1 << 32
    assert (1 << 15) + 2 * 64 * (1 << 16) < 1 << 63


@pytest.mark.parametrize('shift', [15, 30])
def test_mode_b_rounding_matches_fraction_at_signed_ties_and_product_bounds(shift):
    values = {0, -(1 << 48), (1 << 48) - 1}
    for quotient in (-65536, -32768, -3, -2, -1, 0, 1, 2, 3, 32767, 65535):
        for remainder in (0, 1, (1 << (shift - 1)) - 1,
                          1 << (shift - 1), (1 << (shift - 1)) + 1, (1 << shift) - 1):
            values.add((quotient << shift) + remainder)
    rng = np.random.default_rng(9821 + shift)
    values.update(int(x) for x in rng.integers(-(1 << 48), 1 << 48, 500))
    for value in values:
        expected = round(Fraction(value, 1 << shift))
        assert hot._mode_b_rne(value, shift) == expected
        assert hot._rshift_rne(value, shift) == expected
