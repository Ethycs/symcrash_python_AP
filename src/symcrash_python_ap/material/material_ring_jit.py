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

from ..primitives import Q15_HALF, Q15_ONE, Q15_NEG_ONE, Q15_3_4
from ..vm.hotpath_jit import njit, jit_enabled, _clamp16, _rshift_rne, _sine_q15_k, _get_sine_lut_np
from .material_sample_jit import _audio, _scalar, _multiply
from .svf import div_rne

_prepared = False
_chorus_lut = None
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
    global _prepared, _chorus_lut
    if not _prepared and jit_enabled():
        reverb_channel(np.zeros(1, dtype=np.int16),
                       np.zeros((6, 1), dtype=np.int16),
                       np.ones(6, dtype=np.int64), 0, 0)
        source = np.zeros(1, dtype=np.int16)
        delay_channel(source, source, 1, 0, 0)
        ks_channel(source, source, 1, 0, 0, 0, 0)
        # Read-only view of the existing VM quarter-wave ROM; no second table.
        _chorus_lut = _get_sine_lut_np().view()
        _chorus_lut.flags.writeable = False
        delays, _ = chorus_delays(_chorus_lut, 1, 0, 0, 0, 0, 960)
        chorus_channel(source, source, delays, 0)
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


@njit(cache=True)
def delay_channel(source, initial, delay, feedback, mix):
    writes = np.empty(len(source), dtype=np.int16)
    output = np.empty(len(source), dtype=np.int16)
    for i in range(len(source)):
        sample = np.int64(source[i])
        delayed = np.int64(initial[i] if i < delay else writes[i - delay])
        writes[i] = _clamp16(sample + _multiply(delayed, feedback))
        value = _clamp16(_multiply(sample, Q15_ONE - mix) + _multiply(delayed, mix))
        if abs(value) > Q15_3_4:
            clipped = min(Q15_3_4 + ((abs(value) - Q15_3_4) >> 1), Q15_ONE)
            value = clipped if value >= 0 else -clipped
        output[i] = value
    return output, writes


@njit(cache=True)
def ks_channel(source, initial, delay, exciter, damping, feedback, previous):
    writes = np.empty(len(source), dtype=np.int16)
    output = np.empty(len(source), dtype=np.int16)
    for i in range(len(source)):
        sample = np.int64(source[i])
        delayed = np.int64(initial[i] if i < delay else writes[i - delay])
        average = (previous + delayed) >> 1
        filtered = delayed + _multiply(average - delayed, damping)
        writes[i] = _clamp16(_multiply(sample, exciter) + _multiply(filtered, feedback))
        previous = delayed
        output[i] = delayed
    return output, writes, previous


@njit(cache=True)
def chorus_delays(lut, count, phase, increment, offset, depth, capacity):
    delays = np.empty(count, dtype=np.int64)
    for i in range(count):
        phase = (phase + increment) & 0xffffffff
        sine = _sine_q15_k(lut, (phase + offset) & 0xffffffff)
        delay = 720 + _rshift_rne(sine * depth, 15)
        delays[i] = max(1, min(delay, capacity - 1))
    return delays, phase


@njit(cache=True)
def chorus_channel(source, initial, delays, mix):
    output = np.empty(len(source), dtype=np.int16)
    for i in range(len(source)):
        delay = delays[i]
        sample = np.int64(source[i])
        delayed = np.int64(initial[i] if i < delay else source[i - delay])
        output[i] = _clamp16(_multiply(sample, Q15_ONE - mix) + _multiply(delayed, mix))
    return output


def _ring_header(state):
    buffer, capacity, start = state.buffer, state.max_samples, state.write_idx
    if (type(buffer) is not list or type(capacity) is not int or capacity < 1
            or len(buffer) < capacity or type(start) is not int or not 0 <= start < capacity):
        raise ValueError('Nonstandard delay ring')
    return buffer, capacity, start


def _initial_reads(header, delay, count):
    buffer, capacity, start = header
    span = min(delay, count)
    begin = (start - delay) % capacity
    first = min(span, capacity - begin)
    values = buffer[begin:begin + first] + buffer[:span - first]
    return _ring_values(values)


def _ring_values(values):
    if (set(map(type, values)) != {int}
            or min(values) < Q15_NEG_ONE or max(values) > Q15_ONE):
        raise ValueError('Non-Q15/custom ring cells')
    return np.array(values, dtype=np.int16)


def _commit_writes(state, header, writes):
    buffer, capacity, start = header
    count = len(writes)
    span = min(count, capacity)
    begin = (start + count - span) % capacity
    first = min(span, capacity - begin)
    values = writes[-span:].tolist()
    buffer[begin:begin + first] = values[:first]
    buffer[:span - first] = values[first:]
    state.write_idx = (start + count) % capacity


def try_delay_program(engine, left, right, program):
    """Exact Delay/Chorus/KS with bounded read/write windows, not ring copies.

    At N64, Delay/KS numeric input, initial-read, write and output arrays use
    at most 1024 bytes; Chorus uses 768 bytes plus 1024 bytes of delay indices.
    The sine LUT view shares existing storage. Python packing overhead is extra.
    Persistent capacities/indices, Q15 cells and Q32 phase are unchanged.
    KS's unclamped filtered intermediate is preserved before feedback RNE/SAT.
    All admitted products fit int64. No intermediate is narrowed to int16
    before the reference write/saturation boundary.
    """
    if not jit_enabled() or not _prepared:
        return None
    try:
        count = engine.B
        if type(count) is not int or count <= 0:
            return None
        states = {6: (engine.delay_L, engine.delay_R),
                  7: (engine.chorus_L, engine.chorus_R),
                  10: (engine.ks_L, engine.ks_R)}[program]
        if states[0] is states[1] or states[0].buffer is states[1].buffer:
            return None
        if any(s.buffer is left or s.buffer is right for s in states):
            return None
        drive, tone, pressure, bias = map(_scalar, (engine.params.drive_q15,
            engine.params.tone_q15, engine.params.pressure_q15, engine.params.bias_q15))
        if program == 7 and (type(left) not in (list, tuple) or type(right) not in (list, tuple)
                or any(type(x) is not int for x in (*left[:count], *right[:count]))):
            # Chorus writes x directly; preserve bool/custom scalar identities.
            return None
        sources = (_audio(left, count), _audio(right, count))
        headers = [_ring_header(s) for s in states]
        if program == 7:
            increment = div_rne((Q15_ONE + 49 * drive) << 32, 10 * Q15_ONE * 48000)
            depth = (tone * 480) >> 15
            offset = div_rne(bias << 31, Q15_ONE)
            for state in states:
                if type(state.lfo_phase) is not int or not 0 <= state.lfo_phase <= 0xffffffff:
                    return None
            planned = [chorus_delays(_chorus_lut, count, state.lfo_phase,
                increment, off, depth, head[1]) for state, head, off in zip(states, headers, (0, offset))]
            initial = [_ring_values([head[0][(head[2] + i - int(d)) % head[1]]
                for i, d in enumerate(plan[0])]) for head, plan in zip(headers, planned)]
        else:
            if program == 6:
                delays = [max(1, min((drive * 48000) >> 15, h[1] - 1)) for h in headers]
            else:
                delays = [s.delay_samples for s in states]
                if any(type(d) is not int or not 1 <= d <= h[1] for d, h in zip(delays, headers)):
                    return None
                previous = [_scalar(s.prev_sample) for s in states]
            initial = [_initial_reads(h, d, count) for h, d in zip(headers, delays)]
    except (OverflowError, TypeError, ValueError, AttributeError, KeyError):
        return None
    if program == 6:
        result = [delay_channel(x, old, delay, min(tone, 30000), pressure)
                  for x, old, delay in zip(sources, initial, delays)]
    elif program == 10:
        feedback = 29491 + ((pressure * 3276) >> 15)
        result = [ks_channel(x, old, delay, drive, tone, feedback, prev)
                  for x, old, delay, prev in zip(sources, initial, delays, previous)]
    else:
        result = [(chorus_channel(x, old, plan[0], pressure), x, plan[1])
                  for x, old, plan in zip(sources, initial, planned)]
    for state, head, values in zip(states, headers, result):
        _commit_writes(state, head, values[1])
        if program == 7:
            state.lfo_phase = int(values[2])
        elif program == 10:
            state.prev_sample = int(values[2])
    return result[0][0].tolist(), result[1][0].tolist()
