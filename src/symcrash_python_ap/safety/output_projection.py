"""Integer projection used inside the final audio Safety window.

The same ordinary Python functions compile into the Numba tail. A disabled
constraint is represented by -1. The first sample can be anchored to the last
emitted sample, so correcting DC cannot undo the OLA boundary guarantee.
"""
import numpy as np

try:
    from numba.extending import register_jitable
except ImportError:  # pragma: no cover - reference-only installation
    def register_jitable(function):
        return function


@register_jitable
def output_feasible(previous, count, peak, slew, dc, anchor):
    """Whether any integer block can satisfy the enabled constraints."""
    if count <= 0 or peak < 0:
        return False
    low = previous
    high = previous
    sum_low = 0
    sum_high = 0
    for i in range(count):
        if anchor and i == 0:
            if abs(previous) > peak:
                return False
        elif slew >= 0:
            low = max(-peak, low - slew)
            high = min(peak, high + slew)
            if low > peak or high < -peak:
                return False
        else:
            low, high = -peak, peak
        # With DC disabled, any legal first sample can be repeated for the
        # rest of the block: constant continuation keeps peak and slew legal.
        # The checks above still reject an impossible first sample/anchor.
        if dc < 0:
            return True
        sum_low += low
        sum_high += high
    return dc < 0 or (sum_low <= count * dc and sum_high >= -count * dc)


@register_jitable
def _project_offset(source, out, previous, peak, slew, anchor, sign, offset):
    total = 0
    p = sign * previous
    for i in range(len(source)):
        low, high = -peak, peak
        if slew >= 0:
            low, high = max(low, p - slew), min(high, p + slew)
        if anchor and i == 0:
            value = p
        else:
            value = max(low, min(high, sign * int(source[i]) - offset))
        out[i] = value
        total += value
        p = value
    return total


@register_jitable
def project_output(source, previous, peak, slew, dc, anchor):
    """Preserve legal samples; otherwise find a bounded integer DC correction.

    For a fixed offset the forward clip enforces peak/slew. Its sum decreases
    monotonically with offset, so binary search finds the smallest correction.
    Adjacent offsets differ by at most one per sample. If an integer offset
    skips the DC band (possible with dc=0), fill the gap one legal LSB at a time.
    This is a feasibility projection, not a least-squares optimizer.
    """
    n = len(source)
    if not output_feasible(previous, n, peak, slew, dc, anchor):
        raise ValueError('Safety output constraints are infeasible for this block/history')
    out = np.empty(n, dtype=np.int64)
    total = _project_offset(source, out, previous, peak, slew, anchor, 1, 0)
    bound = n * dc
    if dc < 0 or abs(total) <= bound:
        return out
    sign = 1 if total > 0 else -1
    lo, hi = 0, 1
    for i in range(n):
        hi = max(hi, sign * int(source[i]) + peak)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        value = _project_offset(source, out, previous, peak, slew, anchor, sign, mid)
        if value > bound:
            lo = mid
        else:
            hi = mid
    total = _project_offset(source, out, previous, peak, slew, anchor, sign, hi)
    if total < -bound:
        upper = np.empty(n, dtype=np.int64)
        _project_offset(source, upper, previous, peak, slew, anchor, sign, hi - 1)
        while total < -bound:
            changed = False
            for i in range(n):
                if (anchor and i == 0) or out[i] >= upper[i]:
                    continue
                value = out[i] + 1
                p = sign * previous if i == 0 else out[i - 1]
                if slew >= 0 and (abs(value - p) > slew or
                                  (i + 1 < n and abs(value - out[i + 1]) > slew)):
                    continue
                out[i] = value
                total += 1
                changed = True
                if total >= -bound:
                    break
            if not changed:  # Unreachable for a feasible interval of path constraints.
                raise ValueError('No integer Safety output satisfies the DC bound')
    for i in range(n):
        out[i] *= sign
    return out


@register_jitable
def dc_excess_offset(source, dc):
    """Constant offset removing excess mean, rounded away from zero."""
    n = len(source)
    if n == 0:
        return 0
    total = 0
    for i in range(n):
        total += int(source[i])
    excess = abs(total) - n * dc
    if excess <= 0:
        return 0
    offset = (excess + n - 1) // n
    return offset if total > 0 else -offset
