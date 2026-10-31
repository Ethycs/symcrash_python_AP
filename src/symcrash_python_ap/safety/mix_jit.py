"""Exact host acceleration of per-voice pan, wide summation and auxiliary sends.

Each input, pan/send weight and reduced product fits signed int16. Products
therefore fit signed int64. Accumulators admit at most floor(INT64_MAX/32768)
voices, so even full negative rails cannot wrap. No early mix saturation is
allowed: opposite-polarity voices must cancel before the final main-bus clamp.
Auxiliary sums stay wide until their existing output owner clamps them.

Six int64 block accumulators are transient host scratch (3072 numeric bytes
at N64); input conversion adds at most two block arrays (1024 bytes). This
does not change persistent PCM formats or allocate hardware memory.
"""
import operator

import numpy as np

from symcrash_python_ap.vm.hotpath_jit import njit, jit_enabled, _rshift_rne, _clamp16
from symcrash_python_ap.primitives import Q15_ONE, Q15_NEG_ONE, Q15_SCALE
from symcrash_python_ap.wavelet.sine_core import cosine_q15, sine_q15

_prepared = False


@njit(cache=True)
def mix_one(source, gain_l, gain_r, send_a, send_b, mixed):
    for index in range(len(source)):
        sample = int(source[index])
        left = _clamp16(_rshift_rne(sample * gain_l, 15))
        right = _clamp16(_rshift_rne(sample * gain_r, 15))
        mixed[0, index] += left
        mixed[1, index] += right
        if send_a:
            mixed[2, index] += _clamp16(_rshift_rne(left * send_a, 15))
            mixed[3, index] += _clamp16(_rshift_rne(right * send_a, 15))
        if send_b:
            mixed[4, index] += _clamp16(_rshift_rne(left * send_b, 15))
            mixed[5, index] += _clamp16(_rshift_rne(right * send_b, 15))


def prepare():
    global _prepared
    if not _prepared and jit_enabled():
        mix_one(np.zeros(1, dtype=np.int64), 0, 0, 0, 0, np.zeros((6, 1), dtype=np.int64))
        finish_output(np.zeros((4, 1), dtype=np.int64), False, Q15_ONE)
        _prepared = True


def try_mix(rendered, block_size):
    """Return (left, right, sends), or None for the existing Python fallback."""
    if not jit_enabled() or len(rendered) > ((1 << 63) - 1) // Q15_SCALE:
        return None
    mixed = np.zeros((6, block_size), dtype=np.int64)
    for frame, samples in rendered:
        try:
            pan = max(Q15_NEG_ONE, min(Q15_ONE, operator.index(frame.pan_q15)))
            send_a = max(0, min(Q15_ONE, int(frame.sendA_q15)))
            send_b = max(0, min(Q15_ONE, int(frame.sendB_q15)))
            source = np.asarray(samples)
            if (source.shape != (block_size,) or source.dtype.kind not in 'biu'
                    or np.any(source < Q15_NEG_ONE) or np.any(source > Q15_ONE)):
                return None
            source = np.array(source, dtype=np.int64, order='C', copy=True)
        except (OverflowError, TypeError, ValueError):
            return None
        phase = (pan + Q15_SCALE) << 14
        mix_one(source, cosine_q15(phase), sine_q15(phase), send_a, send_b, mixed)
    np.clip(mixed[:2], Q15_NEG_ONE, Q15_ONE, out=mixed[:2])
    left, right, a_left, a_right, b_left, b_right = mixed.tolist()
    return left, right, {'A': (a_left, a_right), 'B': (b_left, b_right)}


@njit(cache=True)
def finish_output(audio, mono, gain):
    """Return clamp -> optional mono RNE -> optional Q15 gain, in that order."""
    for i in range(audio.shape[1]):
        left = _clamp16(audio[0, i] + audio[2, i])
        right = _clamp16(audio[1, i] + audio[3, i])
        if mono:
            left = _rshift_rne(left + right, 1)
            right = left
        if gain != Q15_ONE:
            left = _clamp16(_rshift_rne(left * gain, 15))
            right = _clamp16(_rshift_rne(right * gain, 15))
        audio[0, i], audio[1, i] = left, right


def try_finish_output(left, right, returns, mono, gain):
    """Exact optional output arithmetic; no Material/Safety state is bypassed.

    Plain Python int sequences only: custom/NumPy scalar arithmetic keeps its
    reference path. Main audio and gain fit signed Q15; return sums fit int32.
    Sums and products fit int64 without overflow. One private 4*N int64 array
    (2048 bytes at N64) replaces Python loop intermediates. Persistent formats,
    buffers, effect lengths and device/hardware budgets are unchanged.
    """
    if not jit_enabled() or type(gain) is not int or not Q15_NEG_ONE <= gain <= Q15_ONE:
        return None
    rows = (left, right) if returns is None else (left, right, *returns)
    if len(rows) not in (2, 4):
        return None
    n = len(left)
    for index, row in enumerate(rows):
        if type(row) not in (list, tuple) or len(row) != n:
            return None
        lower, upper = (Q15_NEG_ONE, Q15_ONE) if index < 2 else (-(1 << 31), (1 << 31) - 1)
        if not all(type(x) is int and lower <= x <= upper for x in row):
            return None
    audio = np.zeros((4, n), dtype=np.int64)
    for index, row in enumerate(rows):
        audio[index] = row
    finish_output(audio, bool(mono), gain)
    return audio[0].tolist(), audio[1].tolist()
