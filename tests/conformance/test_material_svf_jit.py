"""Exact Material SVF acceleration, state continuity and admitted-width checks."""
from dataclasses import replace
from copy import deepcopy
import itertools

import numpy as np
import pytest

from symcrash_python_ap.frames import ParamFrame
from symcrash_python_ap.material.material_engine import MaterialEngine, SVFState
from symcrash_python_ap.material.svf import coefficients, process_sample
from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.material import svf_jit


@pytest.fixture(autouse=True)
def enabled():
    old = hot.jit_enabled()
    if not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    hot.set_jit_enabled(True)
    svf_jit.prepare()
    try:
        yield
    finally:
        hot.set_jit_enabled(old)


def test_every_admitted_intermediate_fits_signed_64_bits():
    # Conservative interval proof over the adapter's entire admitted domain,
    # including saturated int32 histories that normal audio rarely reaches.
    state = 1 << 31
    audio = 1 << 27
    a1, a2, a3, k = 1 << 30, (1 << 29) + 8, (1 << 30) + 16, 1 << 31
    v3 = audio + state
    n1, n2 = a1 * state + a2 * v3, a2 * state + a3 * v3
    v1, delta = (n1 + (1 << 29)) >> 30, (n2 + (1 << 29)) >> 30
    v2 = state + delta
    hp = audio + ((k * v1 + (1 << 29)) >> 30) + v2
    assert max(n1, n2) < 1 << 62
    assert k * v1 < (1 << 63) - 1
    assert max(2 * v1 + state, 2 * v2 + state, hp + v2) < 1 << 35


@pytest.mark.parametrize('mode', range(4))
def test_stereo_audio_and_state_match_reference_through_extremes(mode):
    rng = np.random.default_rng(6910 + mode)
    for left_start, right_start in itertools.product((-(1 << 31), 0, (1 << 31) - 1), repeat=2):
        actual = [SVFState(left_start, right_start), SVFState(right_start, left_start)]
        expected = deepcopy(actual)
        for block in range(24):
            cutoff = (0, 32767, 1, 32766, 9000, 18000)[block % 6]
            resonance = (0, 32767, 8000, 24000)[block % 4]
            coeff = coefficients(cutoff, resonance)
            left = rng.integers(-32768, 32768, 64, dtype=np.int16)
            right = np.zeros(64, dtype=np.int16) if block % 2 else -np.ones(64, dtype=np.int16) * 32767
            before_l, before_r = left.copy(), right.copy()
            result = svf_jit.try_process(left, right, 64, *actual, coeff, mode)
            reference = tuple([process_sample(x, state, coeff, mode) for x in samples]
                              for state, samples in zip(expected, (left, right)))
            assert result == reference
            assert actual == expected
            np.testing.assert_array_equal(left, before_l)
            np.testing.assert_array_equal(right, before_r)


@pytest.mark.parametrize('target', ['audio', 'state', 'coefficient', 'fractional'])
def test_unproven_widths_fall_back_before_mutating_state(target):
    left, right = [1] * 64, [2] * 64
    a, b = SVFState(3, 4), SVFState(5, 6)
    coeff = coefficients(16000, 32000)
    if target == 'audio':
        left[0] = 1 << 100
    elif target == 'state':
        a.ic1eq = 1 << 40
    elif target == 'coefficient':
        coeff = replace(coeff, a1=1 << 31)
    else:
        left[0] = 0.5
    before = deepcopy((a, b))
    assert svf_jit.try_process(left, right, 64, a, b, coeff, 0) is None
    assert (a, b) == before


def test_readonly_and_strided_audio_use_the_prepared_signature():
    signatures = list(svf_jit.channel.signatures)
    left = np.arange(128, dtype=np.int64)[::2]
    right = left[::-1]
    left.setflags(write=False)
    right.setflags(write=False)
    assert svf_jit.try_process(left, right, 64, SVFState(), SVFState(), coefficients(7000, 12000), 3)
    assert svf_jit.channel.signatures == signatures


def test_python_and_jit_can_alternate_on_the_same_live_state():
    a, b = SVFState(), SVFState()
    expected_a, expected_b = SVFState(), SVFState()
    for block in range(64):
        coeff = coefficients((block * 607) % 32768, (block * 7907) % 32768)
        left = [((block * 64 + i) * 7919 % 65536) - 32768 for i in range(64)]
        right = [x // -3 for x in left]
        hot.set_jit_enabled(block % 2 == 0)
        result = svf_jit.try_process(left, right, 64, a, b, coeff, block % 4)
        if result is None:
            result = ([process_sample(x, a, coeff, block % 4) for x in left],
                      [process_sample(x, b, coeff, block % 4) for x in right])
        expected = ([process_sample(x, expected_a, coeff, block % 4) for x in left],
                    [process_sample(x, expected_b, coeff, block % 4) for x in right])
        assert result == expected
        assert (a, b) == (expected_a, expected_b)
