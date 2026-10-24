"""
Safety Tail Hot-Path JIT Kernel
===============================

Numba-compiled port of the per-block Safety Tail chain executed by
``SafetyEngine.step_block``:

    Metrics -> RG -> Q -> Master OLA -> Slew Limit -> Limiter -> DC removal

DESIGN CONTRACT (same as vm/hotpath_jit.py):
    The kernel is a BIT-EXACT port of the pure-Python methods in
    safety_engine.py, which remain the conformance reference. Every
    intermediate fits in int64 (largest product is 32767 * 32767), so Python
    int and numba int64 agree exactly. Floor division and arithmetic right
    shift on negative values have identical semantics in CPython and numba.
    No floats are used.

SWITCHING:
    The kernel honours the VM switch: SYMCRASH_VM_REFERENCE=1 or
    hotpath_jit.set_jit_enabled(False) routes step_block through the
    pure-Python methods. State lives on the SafetyEngine instance in both
    paths, so the two can be alternated block by block.
"""
from __future__ import annotations

import numpy as np

from ..vm.hotpath_jit import njit, NUMBA_AVAILABLE, jit_enabled  # noqa: F401

Q15_ONE = 32767
Q15_HALF = 16384
MAX_S16 = 32767
MIN_S16 = -32768

# State vector layout shared with SafetyEngine._step_block_jit
ST_E_HAT = 0
ST_HF_HAT = 1
ST_PEAK_HAT = 2
ST_DC_HAT = 3
ST_Q_GAIN = 4
ST_Q_HF_CUT = 5
ST_PREV_L = 6
ST_PREV_R = 7
ST_LIM_L = 8
ST_LIM_R = 9
ST_OLA_PRIMED = 10
ST_ALPHA = 11
ST_LEN = 12

FLAG_MASTER_OLA = 1
FLAG_SLEW_LIMIT = 2
FLAG_DC_REMOVAL = 4


@njit(cache=True)
def _sat16(x):
    if x > MAX_S16:
        return MAX_S16
    if x < MIN_S16:
        return MIN_S16
    return x


@njit(cache=True)
def _rshift_rne(x, k):
    """Port of primitives.RSHIFT_RNE for 0 < k <= 63."""
    half = 1 << (k - 1)
    if x >= 0:
        q = x >> k
        r = x & ((1 << k) - 1)
        if r > half:
            return q + 1
        elif r < half:
            return q
        else:
            if q & 1:
                return q + 1
            return q
    else:
        abs_x = -x
        q = abs_x >> k
        r = abs_x & ((1 << k) - 1)
        if r > half:
            return -(q + 1)
        elif r < half:
            return -q
        else:
            if q & 1:
                return -(q + 1)
            return -q


@njit(cache=True)
def _mul_q15(a, b):
    """primitives.MUL_Q15 / mul_q15_sat: SAT16(RSHIFT_RNE(a*b, 15))."""
    return _sat16(_rshift_rne(a * b, 15))


@njit(cache=True)
def _slew(cur, target, max_delta):
    delta = target - cur
    if delta > max_delta:
        return cur + max_delta
    elif delta < -max_delta:
        return cur - max_delta
    return target


@njit(cache=True)
def _slew_asym(cur, target, attack, release):
    delta = target - cur
    if delta > 0:
        return cur + min(delta, attack)
    return cur + max(delta, -release)


@njit(cache=True)
def _sqrt_q15(x, sqrt_lut):
    if x <= 0:
        return 0
    if x >= Q15_ONE:
        return Q15_ONE
    idx = (x * (sqrt_lut.shape[0] - 1)) >> 15
    if idx > sqrt_lut.shape[0] - 1:
        idx = sqrt_lut.shape[0] - 1
    return sqrt_lut[idx]


@njit(cache=True)
def _hann_fade_in(n, fade_len, hann_lut):
    if n <= 0:
        return 0
    if n >= fade_len:
        return Q15_ONE
    return hann_lut[n]


@njit(cache=True)
def safety_tail_step(L, R, n, st, flags, k_mode,
                     e_targets, hf_budgets, sqrt_lut, hann_lut,
                     tail_L, tail_R,
                     limit_threshold, limit_ceiling, attack, release,
                     slew_max, ola_overlap,
                     outL, outR):
    """Run one block through the whole chain. Mutates st, tail_L/tail_R, outL/outR."""
    # ---- 1) measure_metrics -------------------------------------------------
    peak = 0
    energy = 0
    for i in range(n):
        a = abs(L[i])
        b = abs(R[i])
        if a > peak:
            peak = a
        if b > peak:
            peak = b
        energy += (L[i] * L[i] + R[i] * R[i]) >> 16
    e = energy // (2 * n)
    if e > Q15_ONE:
        e = Q15_ONE
    rms = _sqrt_q15(e, sqrt_lut)

    hf = 0
    for i in range(1, n):
        dL = L[i] - L[i - 1]
        dR = R[i] - R[i - 1]
        hf += (dL * dL + dR * dR) >> 18
    hf_energy = hf // n
    if hf_energy > Q15_ONE:
        hf_energy = Q15_ONE

    dc = 0
    for i in range(n):
        dc += L[i] + R[i]
    dc_offset = _sat16(dc // (2 * n))

    # ---- 2) update_rg -------------------------------------------------------
    alpha = st[ST_ALPHA]
    alpha_inv = Q15_ONE - alpha
    st[ST_E_HAT] = _rshift_rne(alpha * rms + alpha_inv * st[ST_E_HAT], 15)
    st[ST_HF_HAT] = _rshift_rne(alpha * hf_energy + alpha_inv * st[ST_HF_HAT], 15)
    st[ST_PEAK_HAT] = _rshift_rne(alpha * peak + alpha_inv * st[ST_PEAK_HAT], 15)
    st[ST_DC_HAT] = _rshift_rne(alpha * dc_offset + alpha_inv * st[ST_DC_HAT], 15)

    # ---- 3) apply_q (compute_q_targets + slewed gain + HF blend) -----------
    idx = k_mode
    if idx > 3:
        idx = 3
    if idx < 0:
        idx = idx + 4          # Python negative indexing of the 4-entry tables
    e_target = e_targets[idx]
    hf_budget = hf_budgets[idx]

    if st[ST_E_HAT] > 0:
        gain_target = (e_target << 15) // st[ST_E_HAT]
        if gain_target > Q15_ONE:
            gain_target = Q15_ONE
    else:
        gain_target = Q15_ONE

    if st[ST_HF_HAT] > hf_budget:
        hf_cut_target = (hf_budget << 15) // st[ST_HF_HAT]
        if hf_cut_target > Q15_ONE:
            hf_cut_target = Q15_ONE
        if hf_cut_target < Q15_HALF:
            hf_cut_target = Q15_HALF
    else:
        hf_cut_target = Q15_ONE

    st[ST_Q_GAIN] = _slew(st[ST_Q_GAIN], gain_target, 256)
    st[ST_Q_HF_CUT] = _slew(st[ST_Q_HF_CUT], hf_cut_target, 128)
    q_gain = st[ST_Q_GAIN]
    q_hf_cut = st[ST_Q_HF_CUT]
    blend = Q15_ONE - q_hf_cut

    # The HF blend is seeded from the slew limiter's last outputs of the previous block.
    prev_yL = st[ST_PREV_L]
    prev_yR = st[ST_PREV_R]
    for i in range(n):
        yL = _mul_q15(L[i], q_gain)
        yR = _mul_q15(R[i], q_gain)
        if blend > 0:
            yL = _sat16(_mul_q15(yL, q_hf_cut) + _mul_q15(prev_yL, blend))
            yR = _sat16(_mul_q15(yR, q_hf_cut) + _mul_q15(prev_yR, blend))
        prev_yL = yL
        prev_yR = yR
        outL[i] = yL
        outR[i] = yR

    # ---- 4) apply_master_ola --------------------------------------------------
    if flags & FLAG_MASTER_OLA:
        if st[ST_OLA_PRIMED] == 0:
            for i in range(ola_overlap):
                tail_L[i] = outL[n - ola_overlap + i]
                tail_R[i] = outR[n - ola_overlap + i]
            st[ST_OLA_PRIMED] = 1
        else:
            # The saved tail is taken from this block's OLA *input*, before crossfading.
            newtail_L = np.empty(ola_overlap, np.int64)
            newtail_R = np.empty(ola_overlap, np.int64)
            for i in range(ola_overlap):
                newtail_L[i] = outL[n - ola_overlap + i]
                newtail_R[i] = outR[n - ola_overlap + i]
            for i in range(ola_overlap):
                fade_out_gain = _hann_fade_in(ola_overlap - i, ola_overlap, hann_lut)
                fade_in_gain = _hann_fade_in(i, ola_overlap, hann_lut)
                prev_L = _mul_q15(tail_L[i], fade_out_gain)
                curr_L = _mul_q15(outL[i], fade_in_gain)
                outL[i] = _sat16(prev_L + curr_L)
                prev_R = _mul_q15(tail_R[i], fade_out_gain)
                curr_R = _mul_q15(outR[i], fade_in_gain)
                outR[i] = _sat16(prev_R + curr_R)
            for i in range(ola_overlap):
                tail_L[i] = newtail_L[i]
                tail_R[i] = newtail_R[i]

    # ---- 5) apply_slew_limit --------------------------------------------------
    if flags & FLAG_SLEW_LIMIT:
        pL = st[ST_PREV_L]
        pR = st[ST_PREV_R]
        for i in range(n):
            dl = outL[i] - pL
            if dl > slew_max:
                sL = pL + slew_max
            elif dl < -slew_max:
                sL = pL - slew_max
            else:
                sL = outL[i]
            outL[i] = sL
            pL = sL
            dr = outR[i] - pR
            if dr > slew_max:
                sR = pR + slew_max
            elif dr < -slew_max:
                sR = pR - slew_max
            else:
                sR = outR[i]
            outR[i] = sR
            pR = sR
        st[ST_PREV_L] = pL
        st[ST_PREV_R] = pR

    # ---- 6) apply_limiter -----------------------------------------------------
    gL = st[ST_LIM_L]
    gR = st[ST_LIM_R]
    for i in range(n):
        a = abs(outL[i])
        b = abs(outR[i])
        pk = a if a > b else b
        if pk > limit_threshold:
            denom = pk if pk > 1 else 1
            target_gain = (limit_ceiling << 15) // denom
            if target_gain > Q15_ONE:
                target_gain = Q15_ONE
        else:
            target_gain = Q15_ONE
        gL = _slew_asym(gL, target_gain, attack, release)
        gR = _slew_asym(gR, target_gain, attack, release)
        outL[i] = _sat16(_mul_q15(outL[i], gL))
        outR[i] = _sat16(_mul_q15(outR[i], gR))
    st[ST_LIM_L] = gL
    st[ST_LIM_R] = gR

    # ---- 7) remove_dc ---------------------------------------------------------
    if flags & FLAG_DC_REMOVAL:
        dch = st[ST_DC_HAT]
        if abs(dch) >= 100:
            for i in range(n):
                outL[i] = _sat16(outL[i] - dch)
                outR[i] = _sat16(outR[i] - dch)


# =============================================================================
# PER-VOICE DE-CLICK: C1 DISCONTINUITY PATCHER (bit-exact ports)
# =============================================================================
# Ports of compute_local_rms, detect_discontinuities_energy_aware,
# apply_degenerate_hold, hermite_bridge and SafetyEngine.apply_c1_patch.
# hermite_bridge and apply_degenerate_hold use IEEE-754 double arithmetic in the
# reference; the same expressions are evaluated here in the same order, with no
# fastmath, so CPython and numba agree bit for bit. int() truncates toward zero
# in both.

@njit(cache=True)
def _local_rms(x, n, center, window, sqrt_lut):
    """compute_local_rms(samples, center, window)."""
    if n == 0:
        return 0
    start = center - window
    if start < 0:
        start = 0
    end = center + window
    if end > n:
        end = n
    m = end - start
    if m == 0:
        return 0
    energy = 0
    for i in range(start, end):
        s = x[i]
        energy += (s * s) >> 16
    avg = energy // m
    if avg > Q15_ONE:
        avg = Q15_ONE
    return _sqrt_q15(avg, sqrt_lut)


@njit(cache=True)
def _detect_energy_aware(x, n, prev_sample, slew_thr, curv_thr, energy_floor,
                         degen_slew, sqrt_lut, idx_out):
    """detect_discontinuities_energy_aware -> (count, is_degenerate); indices in idx_out."""
    if n < 3:
        return 0, False
    block_rms = _local_rms(x, n, n // 2, n // 2, sqrt_lut)
    is_deg = block_rms < energy_floor
    cnt = 0
    prev = prev_sample
    prev_delta = 0
    for i in range(n):
        s = x[i]
        delta = s - prev
        abs_delta = abs(delta)
        if i > 0:
            curvature = abs(delta - prev_delta)
        else:
            curvature = 0
        if is_deg:
            local_rms = _local_rms(x, n, i, 8, sqrt_lut)
            if local_rms < energy_floor:
                if abs_delta > degen_slew:
                    idx_out[cnt] = i
                    cnt += 1
            else:
                if abs_delta > slew_thr or curvature > curv_thr:
                    idx_out[cnt] = i
                    cnt += 1
        else:
            if abs_delta > slew_thr or curvature > curv_thr:
                idx_out[cnt] = i
                cnt += 1
        prev_delta = delta
        prev = s
    return cnt, is_deg


@njit(cache=True)
def _degenerate_hold(x, n, out, prev_output, energy_floor, sqrt_lut):
    """apply_degenerate_hold(samples, prev_output, energy_floor) into out."""
    block_rms = _local_rms(x, n, n // 2, n // 2, sqrt_lut)
    if block_rms >= energy_floor:
        for i in range(n):
            out[i] = x[i]
        return
    fade_target = prev_output
    for i in range(n):
        s = x[i]
        local_rms = _local_rms(x, n, i, 8, sqrt_lut)
        if local_rms < energy_floor // 2:
            fade_amount = 0.8
        elif local_rms < energy_floor:
            fade_amount = 0.5
        else:
            fade_amount = 0.2
        blended = int(s * (1 - fade_amount) + fade_target * fade_amount)
        out[i] = _sat16(blended)
        fade_target = int(fade_target * 0.95)


@njit(cache=True)
def _hermite_bridge(out, n, k, bl):
    """hermite_bridge(samples, k, bridge_len) applied in place.

    The reference reads p0/p1/m0/m1 from its input before writing a copy, and each
    successive bridge reads the previous bridge's output, so reading the context
    first and then writing in place is equivalent.
    """
    if k < bl or k >= n - bl:
        return
    left = k - bl
    right = k + bl
    p0 = out[left]
    p1 = out[right]
    if left >= 2:
        m0 = ((out[left] - out[left - 1]) + (out[left - 1] - out[left - 2])) // 2
    elif left >= 1:
        m0 = out[left] - out[left - 1]
    else:
        m0 = 0
    if right < n - 2:
        m1 = ((out[right + 1] - out[right]) + (out[right + 2] - out[right + 1])) // 2
    elif right < n - 1:
        m1 = out[right + 1] - out[right]
    else:
        m1 = 0
    interval = 2 * bl
    m0s = m0 * interval // 4
    m1s = m1 * interval // 4
    for i in range(2 * bl + 1):
        idx = left + i
        if idx < 0 or idx >= n:
            continue
        t = i / (2 * bl)
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        interp = int(h00 * p0 + h10 * m0s + h01 * p1 + h11 * m1s)
        out[idx] = _sat16(interp)


@njit(cache=True)
def c1_patch_block(x, n, out, idx_buf, was_degenerate, prev_out,
                   slew_thr, curv_thr, energy_floor, hysteresis, degen_slew,
                   patch_radius, sqrt_lut):
    """SafetyEngine.apply_c1_patch for one voice block -> (is_degenerate, last_output)."""
    cnt, is_deg_now = _detect_energy_aware(x, n, 0, slew_thr, curv_thr, energy_floor,
                                           degen_slew, sqrt_lut, idx_buf)
    if was_degenerate:
        block_rms = _local_rms(x, n, n // 2, n // 2, sqrt_lut)
        is_deg = not (block_rms > hysteresis)
    else:
        is_deg = is_deg_now
    if is_deg:
        _degenerate_hold(x, n, out, prev_out, energy_floor, sqrt_lut)
    else:
        for i in range(n):
            out[i] = x[i]
    for j in range(cnt):
        k = idx_buf[j]
        if k < patch_radius or k >= n - patch_radius:
            continue
        _hermite_bridge(out, n, k, patch_radius)
    last = out[n - 1] if n > 0 else 0
    return is_deg, last


@njit(cache=True)
def mix_sat16(acc, src, n):
    """step_block_with_voices mixing: acc[i] = SAT16(acc[i] + src[i])."""
    for i in range(n):
        acc[i] = _sat16(acc[i] + src[i])
