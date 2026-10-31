"""Independent exact rounding and wide-sum checks for the live stereo mixer."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.wavelet.sine_core import cosine_q15, sine_q15
from symcrash_python_ap.safety import mix_jit


@pytest.fixture(autouse=True)
def enabled():
    old = hot.jit_enabled()
    if not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    hot.set_jit_enabled(True)
    mix_jit.prepare()
    try:
        yield
    finally:
        hot.set_jit_enabled(old)


def multiply(a, b):
    return max(-32768, min(32767, round(Fraction(int(a) * int(b), 32768))))


def reference(rendered, n):
    result = [[0] * n for _ in range(6)]
    for frame, samples in rendered:
        pan = min(32767, max(-32768, frame.pan_q15))
        phase = (pan + 32768) << 14
        gains = (cosine_q15(phase), sine_q15(phase))
        sends = (min(32767, max(0, int(frame.sendA_q15))),
                 min(32767, max(0, int(frame.sendB_q15))))
        for i, sample in enumerate(samples):
            for ch, gain in enumerate(gains):
                value = multiply(sample, gain)
                result[ch][i] += value
                for bus, send in enumerate(sends):
                    result[2 + bus * 2 + ch][i] += multiply(value, send)
    left, right, al, ar, bl, br = result
    return ([min(32767, max(-32768, value)) for value in left],
            [min(32767, max(-32768, value)) for value in right],
            {'A': (al, ar), 'B': (bl, br)})


@pytest.mark.parametrize('voices', [0, 1, 2, 8, 32, 256])
def test_pan_sends_and_polyphonic_clipping_match_fraction_oracle(voices):
    rng = np.random.default_rng(6900 + voices)
    rendered = []
    for voice in range(voices):
        frame = SimpleNamespace(pan_q15=int(rng.integers(-40000, 40001)),
                                sendA_q15=(0, 32767, 16384, 1)[voice % 4],
                                sendB_q15=(-10, 65536, 12345)[voice % 3])
        source = rng.integers(-32768, 32768, 64, dtype=np.int64)[::-1]
        source.setflags(write=False)
        rendered.append((frame, source))
    before = [source.copy() for _, source in rendered]
    signatures = list(mix_jit.mix_one.signatures)
    assert mix_jit.try_mix(rendered, 64) == reference(rendered, 64)
    assert mix_jit.mix_one.signatures == signatures
    for (_, source), old in zip(rendered, before):
        np.testing.assert_array_equal(source, old)


def test_opposite_polarities_cancel_before_main_bus_saturation():
    frame = SimpleNamespace(pan_q15=-32768, sendA_q15=32767, sendB_q15=0)
    rendered = [(frame, [32767] * 64)] * 3 + [(frame, [-32768] * 64)] * 3
    actual = mix_jit.try_mix(rendered, 64)
    assert actual == reference(rendered, 64)
    assert max(abs(x) for x in actual[0]) < 10
    # A per-voice clipped accumulation would lose the cancellation above.
    over = mix_jit.try_mix(rendered[:3], 64)
    assert over[0] == [32767] * 64
    assert min(over[2]['A'][0]) > 32767  # Sends retain wide sums for their owner.


@pytest.mark.parametrize('sample', [-32768, -16384, -1, 0, 1, 16384, 32767])
def test_pan_endpoints_and_half_lsb_send_products(sample):
    for pan in (-32768, -32767, -1, 0, 1, 32766, 32767):
        frame = SimpleNamespace(pan_q15=pan, sendA_q15=16384, sendB_q15=1)
        rendered = [(frame, [sample] * 64)]
        assert mix_jit.try_mix(rendered, 64) == reference(rendered, 64)


@pytest.mark.parametrize('samples', [[1 << 80] * 64, [0.5] * 64,
                                   np.full(64, (1 << 64) - 1, dtype=np.uint64), [1] * 63])
def test_outside_compiled_domain_falls_back_without_modifying_input(samples):
    frame = SimpleNamespace(pan_q15=0, sendA_q15=0, sendB_q15=0)
    before = list(samples)
    assert mix_jit.try_mix([(frame, samples)], 64) is None
    assert list(samples) == before


def test_reference_switch_disables_compiled_mixing():
    hot.set_jit_enabled(False)
    assert mix_jit.try_mix([], 64) is None


@pytest.mark.parametrize('n', [1, 7, 64])
@pytest.mark.parametrize('gain', [-32768, -1, 0, 1, 16384, 32766, 32767])
@pytest.mark.parametrize('mono', [False, True])
def test_final_return_mono_gain_order_matches_independent_oracle(n, gain, mono):
    rng = np.random.default_rng(7600 + n)
    left = rng.integers(-32768, 32768, n).tolist()
    right = rng.integers(-32768, 32768, n).tolist()
    left[0], right[0] = -32768, 32767
    returns = (rng.integers(-(1 << 31), 1 << 31, n).tolist(),
               rng.integers(-(1 << 31), 1 << 31, n).tolist())
    before = [x.copy() for x in (left, right, *returns)]
    clip = lambda x: max(-32768, min(32767, x))
    for aux in (None, returns):
        expected_l, expected_r = [], []
        for i in range(n):
            l = clip(left[i] + (aux[0][i] if aux else 0))
            r = clip(right[i] + (aux[1][i] if aux else 0))
            if mono:
                l = r = round(Fraction(l + r, 2))
            expected_l.append(l if gain == 32767 else multiply(l, gain))
            expected_r.append(r if gain == 32767 else multiply(r, gain))
        signatures = list(mix_jit.finish_output.signatures)
        assert mix_jit.try_finish_output(left, right, aux, mono, gain) == (expected_l, expected_r)
        assert mix_jit.finish_output.signatures == signatures
    assert before == [left, right, *returns]


def test_final_half_lsb_ties_and_unity_bypass():
    left, right = [-3, -1, 1, 3, 32767], [0] * 5
    assert mix_jit.try_finish_output(left, right, None, True, 32767) == (
        [-2, 0, 0, 2, 16384], [-2, 0, 0, 2, 16384])
    assert mix_jit.try_finish_output([32767], [32767], None, False, 32767) == ([32767], [32767])


@pytest.mark.parametrize('value', [np.int16(1), np.uint16(1), 0.5, 1 << 80, True])
def test_final_custom_or_out_of_range_arithmetic_keeps_reference(value):
    assert mix_jit.try_finish_output([value], [0], None, False, 32767) is None
    assert mix_jit.try_finish_output([0], [0], ([value], [0]), False, 32767) is None
    assert mix_jit.try_finish_output([0], [0], None, False, value) is None


def test_final_reference_switch_and_malformed_inputs_keep_fallback():
    assert mix_jit.try_finish_output([0], [], None, False, 1) is None
    assert mix_jit.try_finish_output([0], [0], ([1 << 31], [0]), False, 1) is None
    hot.set_jit_enabled(False)
    assert mix_jit.try_finish_output([0], [0], None, False, 1) is None
