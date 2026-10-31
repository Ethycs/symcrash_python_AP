"""Optional exact acceleration of the Material SVF's existing recurrence.

Q30 coefficients, signed 32-bit integrator writes, RNE and SAT16 are unchanged.
Temporary input/output arrays and products are signed int64. The adapter admits
only int16 audio and int32 state. Coefficient guards bound a1 <= 2**30,
a2 <= 2**29+8, a3 <= 2**30+16 and k <= 2**31. Then |v3| <= 17*2**27,
the v1 numerator is < 2**62, and |k*v1| < 2**63. Both v2 products/sum are
also < 2**62. Remaining additions and state doubling are far below int64.
Unusual inputs outside this proven domain retain the Python reference path.

State remains on the caller's SVFState objects. No delay buffers, ROMs,
transform geometry or persistent numeric formats are added or changed.
"""
import operator

import numpy as np

from symcrash_python_ap.vm.hotpath_jit import njit, jit_enabled, _rshift_rne, _clamp16
from symcrash_python_ap.primitives import Q15_ONE, Q15_NEG_ONE
from symcrash_python_ap.material.svf import COEFF_BITS, STATE_GUARD_BITS

_prepared = False


@njit(cache=True)
def channel(source, a1, a2, a3, k, ic1, ic2, mode):
    output = np.empty(len(source), dtype=np.int64)
    for index in range(len(source)):
        x = int(source[index]) << STATE_GUARD_BITS
        v3 = x - ic2
        v1 = _rshift_rne(a1 * ic1 + a2 * v3, COEFF_BITS)
        v2 = ic2 + _rshift_rne(a2 * ic1 + a3 * v3, COEFF_BITS)
        hp = x - _rshift_rne(k * v1, COEFF_BITS) - v2
        ic1 = max(-(1 << 31), min((1 << 31) - 1, 2 * v1 - ic1))
        ic2 = max(-(1 << 31), min((1 << 31) - 1, 2 * v2 - ic2))
        value = v2 if mode == 0 else hp if mode == 1 else v1 if mode == 2 else v2 + hp
        output[index] = _clamp16(_rshift_rne(value, STATE_GUARD_BITS))
    return output, ic1, ic2


def prepare():
    """Compile/load the one kernel signature during Material construction."""
    global _prepared
    if not _prepared and jit_enabled():
        channel(np.zeros(1, dtype=np.int64), 1 << 30, 0, 0, 1 << 30, 0, 0, 0)
        _prepared = True


def try_process(left, right, count, state_l, state_r, coeff, mode):
    """Return stereo lists, or None for inputs needing unbounded Python math."""
    if not jit_enabled():
        return None
    try:
        mode = operator.index(mode)
        if not 0 <= mode <= 3:
            return None
        state = tuple(operator.index(value) for value in
                      (state_l.ic1eq, state_l.ic2eq, state_r.ic1eq, state_r.ic2eq))
        values = tuple(operator.index(value) for value in (coeff.a1, coeff.a2, coeff.a3, coeff.k))
        if any(not -(1 << 31) <= value < (1 << 31) for value in state):
            return None
        limits = (1 << 30, (1 << 29) + 8, (1 << 30) + 16, 1 << 31)
        if any(not 0 <= value <= limit for value, limit in zip(values, limits)):
            return None
        # Copies also normalize strided/read-only input to the warmed signature.
        left = np.asarray(left[:count])
        right = np.asarray(right[:count])
        if (left.shape != (count,) or right.shape != (count,)
                or left.dtype.kind not in 'biu' or right.dtype.kind not in 'biu'):
            return None
        if (np.any(left < Q15_NEG_ONE) or np.any(left > Q15_ONE)
                or np.any(right < Q15_NEG_ONE) or np.any(right > Q15_ONE)):
            return None
        left = np.array(left, dtype=np.int64, order='C', copy=True)
        right = np.array(right, dtype=np.int64, order='C', copy=True)
    except (OverflowError, TypeError, ValueError):
        return None
    out_l, l1, l2 = channel(left, *values, state[0], state[1], mode)
    out_r, r1, r2 = channel(right, *values, state[2], state[3], mode)
    state_l.ic1eq, state_l.ic2eq = int(l1), int(l2)
    state_r.ic1eq, state_r.ic2eq = int(r1), int(r2)
    return out_l.tolist(), out_r.tolist()
