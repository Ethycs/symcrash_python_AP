"""Exact optional Reverb acceleration with bounded copies of touched ring cells.

The Python delay lists and indices remain authoritative. No delay is shortened
or copied in full: gather at most N cells per ring, run the existing recurrence,
then scatter those cells after both channels finish. Six rings per channel use
12*N int16 scratch (1536 bytes at N64), plus 4*N int16 input/output (512 bytes)
and twelve int64 spans (96 bytes). Python packing/conversion overhead is extra.
Existing stereo delay capacity remains 30,400 logical Q15 bytes.

Each state write retains SAT16. The four-comb sum fits 18 signed bits; comb
feedback and dry/wet products fit signed int64, including inverse mix up to
65535. Arithmetic right shifts, individual Q15 RNE products and allpass order
are unchanged. Custom scalars, aliases and methods retain reference behavior.
"""
import numpy as np

from ..primitives import Q15_HALF, Q15_ONE, Q15_NEG_ONE
from ..vm.hotpath_jit import njit, jit_enabled, _clamp16
from .material_sample_jit import _audio, _scalar, _multiply

_prepared = False
_NAMES = ('comb1', 'comb2', 'comb3', 'comb4', 'ap1', 'ap2')


@njit(cache=True)
def reverb_channel(source, rings, spans, decay, mix):
    output = np.empty(len(source), dtype=np.int16)
    feedback = (decay * 28000) >> 15
    for i in range(len(source)):
        sample = np.int64(source[i])
        scaled = sample >> 2
        total = np.int64(0)
        for ring in range(4):
            at = i % spans[ring]
            delayed = np.int64(rings[ring, at])
            total += delayed
            rings[ring, at] = _clamp16(scaled + _multiply(delayed, feedback))
        value = total >> 2
        for ring in range(4, 6):
            at = i % spans[ring]
            delayed = np.int64(rings[ring, at])
            result = _clamp16(delayed - _multiply(Q15_HALF, _clamp16(value - delayed)))
            rings[ring, at] = value
            value = result
        wet = _clamp16(value << 1)
        output[i] = _clamp16(_multiply(sample, Q15_ONE - mix) + _multiply(wet, mix))
    return output


def prepare():
    global _prepared
    if not _prepared and jit_enabled():
        reverb_channel(np.zeros(1, dtype=np.int16),
                       np.zeros((6, 1), dtype=np.int16),
                       np.ones(6, dtype=np.int64), 0, 0)
        _prepared = True


def _gather(state, count):
    delays = (*state.COMB_DELAYS, *state.AP_DELAYS)
    if (len(delays) != 6 or type(state.comb_idx) is not list
            or type(state.ap_idx) is not list
            or len(state.comb_idx) != 4 or len(state.ap_idx) != 2):
        raise ValueError('Nonstandard reverb layout')
    indices = (*state.comb_idx, *state.ap_idx)
    buffers = [getattr(state, name) for name in _NAMES]
    rings = np.zeros((6, count), dtype=np.int16)
    spans = np.empty(6, dtype=np.int64)
    for ring, (buffer, delay, start) in enumerate(zip(buffers, delays, indices)):
        if (type(buffer) is not list or type(delay) is not int or delay <= 0
                or len(buffer) < delay or type(start) is not int or not 0 <= start < delay):
            raise ValueError('Nonstandard ring storage/index')
        span = min(count, delay)
        first = min(span, delay - start)
        values = buffer[start:start + first] + buffer[:span - first]
        # Builtin scans avoid a Python predicate call for every ring sample.
        # Reject non-builtin integers before comparisons or conversion.
        if (set(map(type, values)) != {int}
                or min(values) < Q15_NEG_ONE or max(values) > Q15_ONE):
            raise ValueError('Non-Q15 or custom ring scalars')
        rings[ring, :span] = values
        spans[ring] = span
    return rings, spans, buffers, delays, indices


def _scatter(state, packed, count):
    rings, spans, buffers, delays, indices = packed
    for ring, (buffer, delay, start) in enumerate(zip(buffers, delays, indices)):
        span = int(spans[ring])
        first = min(span, delay - start)
        values = rings[ring, :span].tolist()
        buffer[start:start + first] = values[:first]
        buffer[:span - first] = values[first:]
    state.comb_idx[:] = [(i + count) % d for i, d in zip(indices[:4], delays[:4])]
    state.ap_idx[:] = [(i + count) % d for i, d in zip(indices[4:], delays[4:])]


def try_reverb(engine, left, right):
    """Commit only private computed windows; reject aliases before mutation."""
    if not jit_enabled() or not _prepared:
        return None
    try:
        count = engine.B
        if type(count) is not int or count <= 0 or engine.reverb_L is engine.reverb_R:
            return None
        decay = _scalar(engine.params.drive_q15)
        mix = _scalar(engine.params.pressure_q15)
        states = (engine.reverb_L, engine.reverb_R)
        owned = [getattr(s, name) for s in states for name in (*_NAMES, 'comb_idx', 'ap_idx')]
        if len({id(v) for v in owned}) != len(owned) or any(v is left or v is right for v in owned):
            return None
        sources = (_audio(left, count), _audio(right, count))
        packed = [_gather(state, count) for state in states]
    except (OverflowError, TypeError, ValueError, AttributeError):
        return None
    output = [reverb_channel(source, data[0], data[1], decay, mix)
              for source, data in zip(sources, packed)]
    for state, data in zip(states, packed):
        _scatter(state, data, count)
    return output[0].tolist(), output[1].tolist()
