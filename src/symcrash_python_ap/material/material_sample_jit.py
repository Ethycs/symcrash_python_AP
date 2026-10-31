"""Exact optional Tape/Rank acceleration; Python processing remains the reference.

Stored signal/state ranges remain signed Q15, with the existing 256-entry
tanh ROM, 17-sample rings and 9/17 windows. Arithmetic products widen to int64
as required by MUL_Q15. Admitted scalars/audio/history fit int16. Even signed
drive/tone extremes are safe: |target-m|<=65535, |lambda|<=32767,
0<=damp<=40959, and 0<=32767-tone<=65535. Every product is <2**31 in
magnitude; the int16 square is bounded by 2**30. Trimmed sums contain
at most 13 int16 samples (magnitude <=425984). State writes and mixes retain
the reference's individual RNE/SAT16 order and arithmetic delta >> 1.

Python state objects/lists remain authoritative. Guard failures return None
before changing them. Kernels work on temporary int16 input/output/history
arrays; intermediates widen explicitly. There is one read-only 512-byte host
mirror of the existing ROM, no new curve or approximation. No persistent
ring, transform, sample/coefficient type or hardware budget changes.
"""
import operator

import numpy as np

from symcrash_python_ap.vm.hotpath_jit import njit, jit_enabled, _rshift_rne, _clamp16
from symcrash_python_ap.primitives import Q15_ONE, Q15_NEG_ONE, Q15_HALF
from symcrash_python_ap.fixed_rom import TANH_Q15
from symcrash_python_ap import primitives as arithmetic

_TANH_ROM = np.array(TANH_Q15, dtype=np.int16)
_TANH_ROM.flags.writeable = False
_prepared = False


@njit(cache=True)
def _multiply(a, b):
    return _clamp16(_rshift_rne(a * b, 15))


@njit(cache=True)
def tape_channel(source, drive, bias, tone, magnetization, pole):
    result = np.empty(len(source), dtype=np.int16)
    lam = min(Q15_ONE, drive + Q15_HALF)
    rho = drive >> 2
    inverse_tone = Q15_ONE - tone
    for index in range(len(source)):
        sample = np.int64(source[index])
        driven = _clamp16(_multiply(sample, drive) + bias)
        target = np.int64(_TANH_ROM[(driven - Q15_NEG_ONE) >> 8])
        square = _multiply(magnetization, magnetization)
        damp = max(0, Q15_ONE - _multiply(rho, square))
        delta = _multiply(target - magnetization, lam)
        delta = _multiply(delta, damp)
        magnetization = _clamp16(magnetization + (delta >> 1))
        pole = _clamp16(_multiply(inverse_tone, pole) + _multiply(tone, magnetization))
        result[index] = pole
    return result, magnetization, pole


@njit(cache=True)
def rank_channel(source, drive, history, position):
    result = np.empty(len(source), dtype=np.int16)
    ordered = np.empty(17, dtype=np.int16)
    window = 17 if drive < Q15_HALF else 9
    divisor = window - 4
    for index in range(len(source)):
        sample = np.int64(source[index])
        history[position] = sample
        position = (position + 1) % 17
        # Sort the same most-recent window without allocating a list per sample.
        for offset in range(window):
            value = history[(position - window + offset) % 17]
            slot = offset
            while slot > 0 and ordered[slot - 1] > value:
                ordered[slot] = ordered[slot - 1]
                slot -= 1
            ordered[slot] = value
        total = np.int64(0)
        for offset in range(2, window - 2):
            total += np.int64(ordered[offset])
        quotient = total // divisor
        remainder = total - quotient * divisor
        twice = 2 * remainder
        filtered = _clamp16(quotient + int(twice > divisor or
                                          (twice == divisor and quotient & 1)))
        result[index] = _clamp16(_multiply(filtered, drive) +
                                 _multiply(sample, Q15_ONE - drive))
    return result, position


def prepare():
    """Warm both accepted signatures before the render callback."""
    global _prepared
    if not _prepared and jit_enabled():
        audio = np.zeros(1, dtype=np.int16)
        tape_channel(audio, 0, 0, 0, 0, 0)
        rank_channel(audio, 0, np.zeros(17, dtype=np.int16), 0)
        _prepared = True


def _scalar(value):
    # Ordinary declared state is Python int. Preserve unusual scalar objects
    # and their reference-path behavior instead of changing their representation.
    if type(value) is not int or not Q15_NEG_ONE <= value <= Q15_ONE:
        raise ValueError('Outside the proven signed-Q15 domain')
    return value


def _audio(source, count):
    # NumPy scalars can multiply at their stored width in the reference.
    # Do not silently widen them or change overflow/exception behavior.
    if type(source) not in (list, tuple):
        raise ValueError('Non-Python containers retain reference arithmetic and access order')
    values = source[:count]
    if any(type(x) not in (int, bool) for x in values):
        raise ValueError('Custom scalar conversion retains the reference path')
    array = np.asarray(values)
    if array.shape != (count,) or array.dtype.kind not in 'biu':
        raise ValueError('Complete integer audio block required')
    if np.any(array < Q15_NEG_ONE) or np.any(array > Q15_ONE):
        raise ValueError('Wide audio retains the reference path')
    return np.array(array, dtype=np.int16, order='C', copy=True)


def try_tape(engine, left, right):
    """Process ordinary separate stereo state, or return None without mutation."""
    if not jit_enabled() or not _prepared or arithmetic._TANH_LUT is not TANH_Q15:
        return None
    try:
        count = operator.index(engine.B)
        if count <= 0 or engine.tape_L is engine.tape_R:
            return None
        drive, bias, tone, ml, mr, zl, zr = map(_scalar, (
            engine.params.drive_q15, engine.params.bias_q15, engine.params.tone_q15,
            engine.tape_L.m, engine.tape_R.m, engine.lp_zL, engine.lp_zR))
        left, right = _audio(left, count), _audio(right, count)
    except (OverflowError, TypeError, ValueError):
        return None
    out_l, ml, zl = tape_channel(left, drive, bias, tone, ml, zl)
    out_r, mr, zr = tape_channel(right, drive, bias, tone, mr, zr)
    engine.tape_L.m, engine.tape_R.m = int(ml), int(mr)
    engine.lp_zL, engine.lp_zR = int(zl), int(zr)
    return out_l.tolist(), out_r.tolist()


def try_rank(engine, left, right):
    """Keep the existing 17-value lists and their identities on successful commit."""
    if not jit_enabled() or not _prepared:
        return None
    try:
        count = operator.index(engine.B)
        lstate, rstate = engine.rank_L, engine.rank_R
        if (count <= 0 or lstate is rstate or lstate.buffer is rstate.buffer
                or type(lstate.buffer) is not list or type(rstate.buffer) is not list
                or len(lstate.buffer) != 17 or len(rstate.buffer) != 17):
            return None
        drive = _scalar(engine.params.drive_q15)
        li, ri = _scalar(lstate.idx), _scalar(rstate.idx)
        if not 0 <= li < 17 or not 0 <= ri < 17:
            return None
        for value in lstate.buffer + rstate.buffer:
            _scalar(value)
        left, right = _audio(left, count), _audio(right, count)
        lh = np.array(lstate.buffer, dtype=np.int16)
        rh = np.array(rstate.buffer, dtype=np.int16)
    except (OverflowError, TypeError, ValueError):
        return None
    out_l, li = rank_channel(left, drive, lh, li)
    out_r, ri = rank_channel(right, drive, rh, ri)
    lstate.buffer[:], rstate.buffer[:] = lh.tolist(), rh.tolist()
    lstate.idx, rstate.idx = int(li), int(ri)
    return out_l.tolist(), out_r.tolist()
