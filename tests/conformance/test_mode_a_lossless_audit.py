"""Lossless diagnostic encodings retain every int16 bit and reject bad extents."""
import zlib

import numpy as np
import pytest

from tools.audit_mode_a_lossless import encode_block, decode_block, VALUES_PER_TABLE


@pytest.mark.parametrize('predicted', [False, True])
@pytest.mark.parametrize('pattern', ['zero', 'extrema', 'all_int16', 'random'])
def test_lossless_roundtrip_including_mixed_difference_extrema(predicted, pattern):
    if pattern == 'zero':
        values = np.zeros((2, 64, 4, 64), dtype=np.int16)
    elif pattern == 'extrema':
        table, leaf, lane, phase = np.indices((2, 64, 4, 64))
        values = np.where((table + leaf + lane + phase) % 2, -32768, 32767).astype(np.int16)
    elif pattern == 'all_int16':
        values = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16).reshape(4, 64, 4, 64)
    else:
        values = np.random.default_rng(4601).integers(-32768, 32768, (3, 64, 4, 64), dtype=np.int16)
    before = values.copy()
    payload = encode_block(values, predicted=predicted)
    np.testing.assert_array_equal(decode_block(payload, len(values), predicted=predicted), values)
    np.testing.assert_array_equal(values, before)


def test_encoded_predictor_matches_independent_scalar_formula():
    values = np.random.default_rng(4602).integers(-32768, 32768, (2, 64, 4, 64), dtype=np.int16)
    payload = zlib.decompress(encode_block(values))
    actual = np.frombuffer(payload, dtype=np.uint8).reshape(4, -1).T.copy().view('<u4').reshape(values.shape)
    for t, leaf, lane, phase in np.ndindex(values.shape):
        value = int(values[t, leaf, lane, phase])
        left = int(values[t, leaf, lane, phase-1]) if phase else 0
        above = int(values[t-1, leaf, lane, phase]) if t else 0
        diagonal = int(values[t-1, leaf, lane, phase-1]) if t and phase else 0
        residual = value - left - above + diagonal
        expected = 2*residual if residual >= 0 else -2*residual-1
        assert actual[t, leaf, lane, phase] == expected


@pytest.mark.parametrize('damage', ['truncated', 'trailing', 'too_long', 'wide_residual', 'out_of_range'])
def test_decoder_rejects_corruption_instead_of_wrapping_or_clipping(damage):
    values = np.zeros((1, 64, 4, 64), dtype=np.int16)
    payload = encode_block(values)
    if damage == 'truncated':
        payload = payload[:-1]
    elif damage == 'trailing':
        payload += b'x'
    elif damage == 'too_long':
        payload = zlib.compress(bytes(4*VALUES_PER_TABLE + 1))
    else:
        codes = np.zeros(VALUES_PER_TABLE, dtype='<u4')
        if damage == 'wide_residual':
            codes[0] = 262141
        else:
            codes[0], codes[1] = 65534, 2  # 32767, then +1: must not wrap to -32768
        payload = zlib.compress(codes.view(np.uint8).reshape(-1, 4).T.copy().tobytes())
    with pytest.raises((ValueError, zlib.error)):
        decode_block(payload, 1)


@pytest.mark.parametrize('count', [0, 33, True, 1.5])
def test_decoder_requires_bounded_integer_block_count(count):
    with pytest.raises(ValueError, match='1..32'):
        decode_block(b'', count)


def test_encoder_does_not_silently_narrow_values():
    values = np.zeros((1, 64, 4, 64), dtype=np.int32)
    values.flat[0] = 32768
    with pytest.raises(ValueError, match='int16'):
        encode_block(values)
