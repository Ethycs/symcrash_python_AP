"""Independent arithmetic/rendering checks for candidate-only Mode A analysis."""
from fractions import Fraction
import json

import numpy as np
import pytest

from tools.audit_mode_a_anchors import bracket, groups, interpolate, render_carrier, spectral_metrics
from symcrash_python_ap.vm.coeff_lut_streaming import StreamingCoeffLUT
from symcrash_python_ap.vm import mode_a_resource as resource
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block


@pytest.mark.parametrize('scheme', ['midi', 'octave'])
def test_every_supported_cent_is_bracketed_without_pitch_wrap(scheme):
    for cent in range(12701):
        low, high = bracket(cent, scheme)
        assert 0 <= low <= cent <= high <= 12700
        assert low % 100 == high % 100 == 0
        assert (low == high) == (cent == low)
    assert bracket(12700, scheme) == (12700, 12700)
    assert bracket(0, scheme) == (0, 0)


@pytest.mark.parametrize('num,den', [(0, 2), (1, 2), (2, 2), (17, 101),
                                  (1212754, 92134891)])
def test_interpolation_matches_independent_fraction_rounding(num, den):
    a = np.array([-32768, 32767, -2, -1, 0, 1, 2], dtype=np.int16)
    b = np.array([32767, -32768, -1, 0, 1, 2, 3], dtype=np.int16)
    before = a.copy(), b.copy()
    expected = [round(Fraction(int(x) * (den - num) + int(y) * num, den))
                for x, y in zip(a, b)]
    np.testing.assert_array_equal(interpolate(a, b, num, den), expected)
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])


@pytest.mark.parametrize('kind,count', [('midi_anchors', 643),
                                      ('midi_octave_offsets', 643),
                                      ('octave_anchors', 63)])
def test_storage_groups_cover_each_anchor_once_and_have_encodable_stride(kind, count):
    selected = list(groups(kind))
    keys = [key for group in selected for key in group]
    assert len(keys) == len(set(keys)) == count
    assert keys[-3:] == list(range(resource.NOTE_TABLES, resource.TABLE_COUNT))
    for group in selected:
        assert 1 <= len(group) <= 16
        strides = set(np.diff(group))
        # Last octave group contains MIDI120->127: explicit final endpoint.
        assert strides <= {1, 100, 700, 1200}
    if kind.startswith('midi'):
        expected = {wave * resource.PITCH_COUNT + note * 100
                    for wave in range(5) for note in range(128)}
        assert set(keys[:-3]) == expected


@pytest.mark.parametrize('phase', [0, (1 << 25), 0xfffffff0])
def test_candidate_audio_harness_matches_production_lookup_and_scalar_decoder(phase):
    data = np.random.default_rng(29).integers(-800, 801, (64, 4, 64), dtype=np.int16)
    table = StreamingCoeffLUT(0, 'audit', data)
    increment = 39370534
    coeff = np.empty((64, 4), dtype=np.int16)
    expected = []
    p = phase
    for _ in range(3):
        table.get_subband_samples(p, coeff)
        expected.append(idwpt_polyphase_block(coeff, headroom_bits=2))
        p = (p + 64 * increment) & 0xffffffff
    np.testing.assert_array_equal(render_carrier(data, increment, 3, phase),
                                  np.concatenate(expected))


def test_audio_measurements_are_serializable_and_resolve_a_known_tone():
    samples = np.rint(10000 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.int16)
    metrics = spectral_metrics(samples, 440, sine=True)
    assert metrics['pitch_below_0_1_percent']
    assert metrics['residual_below_minus_60_dbc']
    assert json.loads(json.dumps(metrics)) == metrics
