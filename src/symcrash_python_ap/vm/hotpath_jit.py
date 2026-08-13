"""
VM Hot-Path JIT Kernels (Packet E3 - realtime budget)
=====================================================

Numba-compiled kernels for the per-block VM hot path.

DESIGN CONTRACT (NON-NEGOTIABLE):
    Every kernel in this module is a BIT-EXACT port of the pure-Python
    reference implementation in vm/executor.py and vm/lanes.py. The pure
    Python code remains the conformance reference; these kernels are a
    switchable acceleration layer.

    - RSHIFT_RNE (round-to-nearest, ties-to-even) is ported verbatim from
      fixed.py. All intermediates fit comfortably in int64 (max ~2^31), so
      Python arbitrary-precision int and numba int64 agree exactly.
    - Where the reference relies on numpy int16 wraparound semantics
      (e.g. _inject_gate_mask's `coeff_state[i] * factor` which multiplies
      a np.int16 scalar by a Python int and wraps), the kernel emulates the
      wraparound explicitly with _wrap16.
    - Floor division (//) and arithmetic right shift (>>) on negative ints
      have identical semantics in CPython and numba (floor / sign-extend).
    - Float usage (golden-ratio gates, tail RMS sqrt) follows IEEE-754
      double semantics identically in CPython and numba (sqrt is correctly
      rounded; fastmath is intentionally NOT used anywhere).

SWITCHING:
    Set environment variable SYMCRASH_VM_REFERENCE=1 to force the pure
    Python reference path, or call set_jit_enabled(False) at runtime.
    The JIT path is used by default when numba is available.

    Kernels use cache=True so compilation cost is paid once per machine,
    not per process.
"""

import os
import math

import numpy as np
from ..primitives import Q15_HALF, Q15_NEG_ONE, Q15_ONE, Q15_SCALE

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a project dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator


# =============================================================================
# ENABLE SWITCH
# =============================================================================

def _env_forces_reference() -> bool:
    val = os.environ.get('SYMCRASH_VM_REFERENCE', '').strip().lower()
    return val in ('1', 'true', 'yes', 'on')


_JIT_ENABLED = NUMBA_AVAILABLE and not _env_forces_reference()


def jit_enabled() -> bool:
    """True when the JIT hot path should be used (else pure-Python reference)."""
    return _JIT_ENABLED


def set_jit_enabled(enabled: bool) -> None:
    """Enable/disable the JIT hot path at runtime (tests, A/B verification)."""
    global _JIT_ENABLED
    _JIT_ENABLED = bool(enabled) and NUMBA_AVAILABLE


# =============================================================================
# CONSTANTS (must mirror vm/tables.py and primitives.py)
# =============================================================================

Q15_ONE = 32767
Q15_HALF = 16384
NODE_LEN = 8
L_TOTAL_NODES = 32
BLOCK_SAMPLES = 64
NUM_LEAVES = 64

# CCS tail limiter target (lanes.py _CCS_MAX_RMS_Q15)
_CCS_MAX_RMS_Q15 = 23170

# Golden ratio conjugate used by GateMask / QuasiRotate (lanes.py)
_PHI = 0.618033988749895


# =============================================================================
# CORE PRIMITIVES (bit-exact ports)
# =============================================================================

@njit(cache=True)
def _rshift_rne(x, k):
    """Port of fixed.RSHIFT_RNE for k in (0, 63]. x must fit in int64."""
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
def _clamp16(x):
    """clamp_q15 / np.clip(x, -32768, 32767)."""
    if x > Q15_ONE:
        return Q15_ONE
    if x < Q15_NEG_ONE:
        return Q15_NEG_ONE
    return x


@njit(cache=True)
def _wrap16(x):
    """Emulate numpy int16 wraparound for a value already in int64."""
    return ((x + Q15_SCALE) & 0xFFFF) - Q15_SCALE


# =============================================================================
# EXECUTOR KERNELS
# =============================================================================

@njit(cache=True)
def inject_blend(coeff_state, carrier, base_mix):
    """
    Polyphase carrier injection (executor.py vm_step_block):

        inv_mix = Q15_ONE - base_mix
        blended = RSHIFT_RNE(old * inv_mix + carrier * base_mix, 15)
        coeff_state[idx] = clip(blended, -32768, 32767)
    """
    inv_mix = Q15_ONE - base_mix
    n = coeff_state.shape[0]
    for idx in range(n):
        old_val = np.int64(coeff_state[idx])
        carrier_val = np.int64(carrier[idx])
        blended = _rshift_rne(old_val * inv_mix + carrier_val * base_mix, 15)
        coeff_state[idx] = _clamp16(blended)


@njit(cache=True)
def evolution_scale(coeff_state, prev_coeffs, evolution_rate):
    """
    CMMM evolution-rate scaling (executor.py vm_step_block):

        delta = new - old
        scaled = RSHIFT_RNE(delta * evolution_rate, 15)
        coeff_state[idx] = clip(old + scaled, -32768, 32767)
    """
    n = coeff_state.shape[0]
    for idx in range(n):
        new_val = np.int64(coeff_state[idx])
        old_val = np.int64(prev_coeffs[idx])
        delta = new_val - old_val
        scaled_delta = _rshift_rne(delta * evolution_rate, 15)
        coeff_state[idx] = _clamp16(old_val + scaled_delta)


@njit(cache=True)
def gain_interp(output, gain_prev, gain_curr):
    """
    Per-sample interpolated gain (executor.py vm_step_block):

        alpha = (i * Q15_ONE + 32) // 64
        gain  = gain_prev + RSHIFT_RNE((gain_curr - gain_prev) * alpha, 15)
        out[i] = RSHIFT_RNE(out[i] * gain, 15)
    """
    delta = gain_curr - gain_prev
    for i in range(BLOCK_SAMPLES):
        alpha = (i * Q15_ONE + (BLOCK_SAMPLES >> 1)) // BLOCK_SAMPLES
        gain = gain_prev + _rshift_rne(delta * alpha, 15)
        output[i] = _rshift_rne(np.int64(output[i]) * gain, 15)


# =============================================================================
# BACKBONE OPERATOR KERNELS (lanes.py ports)
# =============================================================================

@njit(cache=True)
def op_tilt(cs, mask, strength):
    """lane_diag op 1 (Tilt)."""
    n = cs.shape[0]
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue
        tilt_factor = Q15_HALF + (node * Q15_ONE) // L_TOTAL_NODES
        tilt_factor = Q15_ONE + ((tilt_factor - Q15_ONE) * strength) // Q15_ONE
        coeff_idx = node * NODE_LEN
        end = coeff_idx + NODE_LEN
        if end > n:
            end = n
        for i in range(coeff_idx, end):
            val = np.int64(cs[i])
            cs[i] = _clamp16((val * tilt_factor) >> 15)


@njit(cache=True)
def op_pressure(cs, mask, strength):
    """lane_diag ops 15/20 (PressureRamp / FoldPressure)."""
    n = cs.shape[0]
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue
        coeff_idx = node * NODE_LEN
        end = coeff_idx + NODE_LEN
        if end > n:
            end = n
        for i in range(coeff_idx, end):
            x = np.int64(cs[i])
            ax = -x if x < 0 else x
            if ax > Q15_HALF:
                sign = 1 if x > 0 else -1
                x = sign * (Q15_HALF + ((ax - Q15_HALF) * strength) // Q15_ONE)
                cs[i] = _clamp16(x)


@njit(cache=True)
def op_swallowtail(cs, mask, strength):
    """lane_diag op 22 (SwallowtailMask)."""
    n = cs.shape[0]
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue
        phase = (node * 3) % 5
        gate = 1 if phase < 3 else 0
        if not gate:
            coeff_idx = node * NODE_LEN
            end = coeff_idx + NODE_LEN
            if end > n:
                end = n
            for i in range(coeff_idx, end):
                val = np.int64(cs[i])
                cs[i] = _wrap16((val * (Q15_ONE - strength)) >> 15)


@njit(cache=True)
def op_gate_mask(cs, mask, strength):
    """
    _inject_gate_mask (lanes.py).

    NOTE: the reference computes `cs[i] * (Q15_ONE - strength)` on a
    np.int16 scalar, which wraps in int16 BEFORE the >> 15 shift. The
    wraparound is emulated here for bit-exactness.
    """
    n = cs.shape[0]
    factor = Q15_ONE - strength
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue
        gate_open = int((node + 1) * _PHI) - int(node * _PHI)
        if gate_open:
            continue
        coeff_idx = node * NODE_LEN
        end = coeff_idx + NODE_LEN
        if end > n:
            end = n
        for i in range(coeff_idx, end):
            prod16 = _wrap16(np.int64(cs[i]) * factor)
            cs[i] = prod16 >> 15


@njit(cache=True)
def op_intramix(cs, mask, strength):
    """lane_banded op 3 (IntraMix).

    Mirrors the reference exactly, including its expression
    `base + ((mixed - base) * strength) >> 15` which (by Python operator
    precedence) shifts the SUM, not just the scaled delta.
    """
    n = cs.shape[0]
    tmp = cs.copy()
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue
        coeff_idx = node * NODE_LEN
        end = coeff_idx + NODE_LEN - 1
        if end > n - 1:
            end = n - 1
        for i in range(coeff_idx + 1, end):
            mixed = (np.int64(tmp[i - 1]) + 2 * np.int64(tmp[i]) + np.int64(tmp[i + 1])) >> 2
            base = np.int64(tmp[i])
            cs[i] = _wrap16((base + (mixed - base) * strength) >> 15)


@njit(cache=True)
def _transfer_energy_k(cs, src_idx, dst_idx, strength):
    """_transfer_energy (lanes.py)."""
    src_val = np.int64(cs[src_idx])
    dst_val = np.int64(cs[dst_idx])
    transfer = (src_val * strength) >> 16
    cs[dst_idx] = _clamp16(dst_val + transfer)
    cs[src_idx] = _clamp16(src_val - transfer)


@njit(cache=True)
def op_diffuse_up(cs, mask, strength):
    """lane_banded op 4 (DiffuseUp)."""
    for node in range(1, L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue
        if not (mask & (1 << (node - 1))):
            continue
        _transfer_energy_k(cs, node * NODE_LEN, (node - 1) * NODE_LEN, strength)


@njit(cache=True)
def op_diffuse_down(cs, mask, strength):
    """lane_banded op 5 (DiffuseDown)."""
    for node in range(L_TOTAL_NODES - 1):
        if not (mask & (1 << node)):
            continue
        if not (mask & (1 << (node + 1))):
            continue
        _transfer_energy_k(cs, node * NODE_LEN, (node + 1) * NODE_LEN, strength)


@njit(cache=True)
def op_band_shear(cs, mask):
    """lane_permute op 6 (BandShear). Caller enforces strength gate."""
    n = cs.shape[0]
    tmp = cs.copy()
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue
        coeff_idx = node * NODE_LEN
        shift = node % NODE_LEN
        for i in range(NODE_LEN):
            src = coeff_idx + i
            dst = coeff_idx + ((i + shift) % NODE_LEN)
            if src < n and dst < n:
                cs[dst] = tmp[src]


@njit(cache=True)
def op_swap_bands(cs, mask):
    """lane_permute op 9 (SwapBands). Caller enforces strength gate."""
    n = cs.shape[0]
    tmp = cs.copy()
    for node in range(0, L_TOTAL_NODES - 1, 2):
        if not (mask & (1 << node)):
            continue
        if not (mask & (1 << (node + 1))):
            continue
        src_idx = node * NODE_LEN
        dst_idx = (node + 1) * NODE_LEN
        for i in range(NODE_LEN):
            if src_idx + i < n and dst_idx + i < n:
                cs[src_idx + i] = tmp[dst_idx + i]
                cs[dst_idx + i] = tmp[src_idx + i]


@njit(cache=True)
def op_rotate_nodes(cs, mask):
    """lane_permute op 10 (RotateNodes). Caller enforces strength gate."""
    n = cs.shape[0]
    tmp = cs.copy()
    active = np.empty(L_TOTAL_NODES, dtype=np.int64)
    count = 0
    for node in range(L_TOTAL_NODES):
        if mask & (1 << node):
            active[count] = node
            count += 1
    if count > 1:
        for i in range(count):
            node = active[i]
            next_node = active[(i + 1) % count]
            src_idx = node * NODE_LEN
            dst_idx = next_node * NODE_LEN
            for j in range(NODE_LEN):
                if src_idx + j < n and dst_idx + j < n:
                    cs[dst_idx + j] = tmp[src_idx + j]


@njit(cache=True)
def op_quasi_rotate(cs, mask, strength):
    """lane_permute op 17 (QuasiRotate). Caller enforces strength gate."""
    n = cs.shape[0]
    tmp = cs.copy()
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue
        target = int((node + _PHI * L_TOTAL_NODES) % L_TOTAL_NODES)
        if not (mask & (1 << target)):
            continue
        src_idx = node * NODE_LEN
        dst_idx = target * NODE_LEN
        for i in range(NODE_LEN):
            if src_idx + i < n and dst_idx + i < n:
                dst_val = np.int64(tmp[dst_idx + i])
                src_val = np.int64(tmp[src_idx + i])
                cs[dst_idx + i] = _clamp16(
                    (dst_val * (Q15_ONE - strength) + src_val * strength) >> 15
                )


@njit(cache=True)
def op_lowrank_morph(cs, mask, strength):
    """lane_lowrank op 7 (LowRankMorph)."""
    n = cs.shape[0]
    active_count = 0
    mean = np.int64(0)
    for node in range(L_TOTAL_NODES):
        if mask & (1 << node):
            coeff_idx = node * NODE_LEN
            if coeff_idx < n:
                mean += np.int64(cs[coeff_idx])
                active_count += 1
    if active_count > 0:
        # Python floor division semantics (mean may be negative)
        mean = mean // active_count
        for node in range(L_TOTAL_NODES):
            if not (mask & (1 << node)):
                continue
            coeff_idx = node * NODE_LEN
            end = coeff_idx + NODE_LEN
            if end > n:
                end = n
            for i in range(coeff_idx, end):
                val = np.int64(cs[i])
                diff = mean - val
                cs[i] = _clamp16(val + ((diff * strength) >> 15))


@njit(cache=True)
def op_glue_pair(cs, mask, strength):
    """lane_proj op 12 (GluePair).

    Mirrors the reference's `val + ((avg - val) * strength) >> 15`
    precedence (the sum is shifted) exactly.
    """
    n = cs.shape[0]
    for node in range(L_TOTAL_NODES - 1):
        if not (mask & (1 << node)):
            continue
        if not (mask & (1 << (node + 1))):
            continue
        idx_a = node * NODE_LEN
        idx_b = (node + 1) * NODE_LEN
        if idx_a < n and idx_b < n:
            val_a = np.int64(cs[idx_a])
            val_b = np.int64(cs[idx_b])
            avg = (val_a + val_b) >> 1
            cs[idx_a] = _clamp16((val_a + (avg - val_a) * strength) >> 15)
            cs[idx_b] = _clamp16((val_b + (avg - val_b) * strength) >> 15)


def dispatch_backbone_op(coeff_state, op_id, strength_q15, mask) -> bool:
    """
    JIT dispatch for a backbone operator. Returns True when handled.

    Mirrors lanes.dispatch_coeff_op routing exactly (including the
    lane_permute strength < Q15_HALF early-out and the documented no-ops).
    Returns False for anything not covered so the caller can fall back to
    the pure-Python reference.
    """
    s = int(strength_q15)
    m = int(mask)
    if op_id == 1:
        op_tilt(coeff_state, m, s)
        return True
    if op_id == 3:
        op_intramix(coeff_state, m, s)
        return True
    if op_id == 4:
        op_diffuse_up(coeff_state, m, s)
        return True
    if op_id == 5:
        op_diffuse_down(coeff_state, m, s)
        return True
    if op_id in (6, 9, 10, 17):
        if s < Q15_HALF:
            return True  # lane_permute: weak permutation is no-op
        if op_id == 6:
            op_band_shear(coeff_state, m)
        elif op_id == 9:
            op_swap_bands(coeff_state, m)
        elif op_id == 10:
            op_rotate_nodes(coeff_state, m)
        else:
            op_quasi_rotate(coeff_state, m, s)
        return True
    if op_id == 7:
        op_lowrank_morph(coeff_state, m, s)
        return True
    if op_id == 12:
        op_glue_pair(coeff_state, m, s)
        return True
    if op_id in (15, 20):
        op_pressure(coeff_state, m, s)
        return True
    if op_id == 16:
        op_gate_mask(coeff_state, m, s)
        return True
    if op_id == 22:
        op_swallowtail(coeff_state, m, s)
        return True
    if op_id in (0, 2, 11, 18, 19, 23):
        # Identity / tail-only ops dispatched through backbone lanes are
        # no-ops in the reference (lane_diag/lane_proj have no branch).
        return True
    return False


# =============================================================================
# TAIL KERNEL (lanes.py _tail_ccs_soft_limit)
# =============================================================================

@njit(cache=True)
def tail_ccs_soft_limit(cs):
    """
    CCS soft energy limiter. Bit-exact port:
        energy = sum(c*c)            (int64)
        rms = int(sqrt(energy / N))  (IEEE double sqrt, truncate)
        if rms <= 23170: return
        scaled = (c * 23170 + rms // 2) // rms   (floor division)
        clip to int16
    """
    n = cs.shape[0]
    energy = np.int64(0)
    for i in range(n):
        v = np.int64(cs[i])
        energy += v * v
    rms = int(math.sqrt(energy / n))
    if rms <= _CCS_MAX_RMS_Q15:
        return
    gain_den = rms if rms > 1 else 1
    half = gain_den // 2
    for i in range(n):
        scaled = (np.int64(cs[i]) * _CCS_MAX_RMS_Q15 + half) // gain_den
        cs[i] = _clamp16(scaled)


# =============================================================================
# IDWPT POLYPHASE KERNEL (lanes.py lane_ccs_render path)
# =============================================================================
# Bit-exact port of fast_lifting._idwpt_jit_core driven for all K slots,
# replacing the per-slot list round-trips in idwpt_polyphase_block.

@njit(cache=True)
def _idwpt_core_k(coeffs, headroom_bits, output_buf, inv_levels, temp_s, temp_d):
    """Single 64-coefficient inverse transform (port of _idwpt_jit_core)."""
    block_size = 64
    depth = 6

    for i in range(block_size):
        inv_levels[depth, i] = coeffs[i]

    for level in range(depth - 1, -1, -1):
        num_nodes = 1 << level
        node_size = block_size >> level

        for node in range(num_nodes):
            out_start = node * node_size
            half = node_size >> 1
            if half == 0:
                inv_levels[level, out_start] = inv_levels[level + 1, node]
                continue

            approx_start = node * 2 * half
            detail_start = approx_start + half

            for i in range(half):
                s_val = inv_levels[level + 1, approx_start + i]
                if i > 0:
                    d_left = inv_levels[level + 1, detail_start + i - 1]
                else:
                    d_left = inv_levels[level + 1, detail_start]
                d_right = inv_levels[level + 1, detail_start + i]
                update = (d_left + d_right + 2) >> 2
                temp_s[i] = s_val - update

            for i in range(half):
                even_left = temp_s[i]
                if i + 1 < half:
                    even_right = temp_s[i + 1]
                else:
                    even_right = temp_s[i]
                avg = (even_left + even_right + 1) >> 1
                temp_d[i] = inv_levels[level + 1, detail_start + i] + avg

            for i in range(half):
                v = temp_s[i]
                if v < Q15_NEG_ONE:
                    v = -32768
                elif v > Q15_ONE:
                    v = 32767
                inv_levels[level, out_start + 2 * i] = v

                v = temp_d[i]
                if v < Q15_NEG_ONE:
                    v = -32768
                elif v > Q15_ONE:
                    v = 32767
                inv_levels[level, out_start + 2 * i + 1] = v

    for i in range(block_size):
        v = inv_levels[0, i] << headroom_bits
        if v < Q15_NEG_ONE:
            v = -32768
        elif v > Q15_ONE:
            v = 32767
        output_buf[i] = v


@njit(cache=True)
def idwpt_polyphase(coeff_state, out, headroom_bits):
    """
    K-slot interleaved polyphase IDWPT (lane_ccs_render):

        coeffs_k[leaf] = coeff_state[leaf * K + k]
        audio_k = idwpt(coeffs_k, headroom_bits)
        out[k::K] = audio_k[0::K]
    """
    K = coeff_state.shape[0] // NUM_LEAVES
    coeffs = np.empty(64, dtype=np.int64)
    audio = np.empty(64, dtype=np.int64)
    inv_levels = np.empty((7, 64), dtype=np.int64)
    temp_s = np.empty(64, dtype=np.int64)
    temp_d = np.empty(64, dtype=np.int64)

    for k in range(K):
        for leaf in range(NUM_LEAVES):
            coeffs[leaf] = coeff_state[leaf * K + k]
        _idwpt_core_k(coeffs, headroom_bits, audio, inv_levels, temp_s, temp_d)
        m = 0
        for i in range(k, BLOCK_SAMPLES, K):
            out[i] = audio[m * K]
            m += 1


# =============================================================================
# MODE B SIN/COS TABLE KERNEL (harmonic_bank.mode_b_generate_coeffs_vectorized)
# =============================================================================
# The reference computes sin_cos_q15 per (harmonic, k-slot) via the
# quarter-wave LUT in wavelet/sine_core.py. This is the same lookup ported
# to numba, fed with the LUT contents generated by sine_core itself so the
# tables can never diverge.

_SINE_LUT_NP = None


def _get_sine_lut_np():
    global _SINE_LUT_NP
    if _SINE_LUT_NP is None:
        from ..wavelet.sine_core import _get_lut
        _SINE_LUT_NP = np.array(_get_lut(), dtype=np.int64)
    return _SINE_LUT_NP


@njit(cache=True)
def _sine_q15_k(lut, phase):
    """Bit-exact port of sine_core.sine_q15 (quarter-wave LUT, RNE interp)."""
    TABLE_SIZE = 1024
    FRAC_BITS = 12
    INDEX_MASK = TABLE_SIZE - 1
    FRAC_MASK = (1 << FRAC_BITS) - 1

    quadrant = (phase >> 30) & 0x3
    pos_in_quad = phase & 0x3FFFFFFF
    scaled = pos_in_quad >> (30 - 10 - FRAC_BITS)
    index = (scaled >> FRAC_BITS) & INDEX_MASK
    frac = scaled & FRAC_MASK

    if quadrant & 1:
        if scaled == 0:
            y = lut[TABLE_SIZE]
        else:
            max_scaled = TABLE_SIZE << FRAC_BITS
            mirror_scaled = max_scaled - scaled
            mirror_index = (mirror_scaled >> FRAC_BITS) & INDEX_MASK
            mirror_frac = mirror_scaled & FRAC_MASK
            y0 = lut[mirror_index]
            i1 = mirror_index + 1
            if i1 > TABLE_SIZE:
                i1 = TABLE_SIZE
            y1 = lut[i1]
            y = y0 + _rshift_rne((y1 - y0) * mirror_frac, FRAC_BITS)
    else:
        y0 = lut[index]
        i1 = index + 1
        if i1 > TABLE_SIZE:
            i1 = TABLE_SIZE
        y1 = lut[i1]
        y = y0 + _rshift_rne((y1 - y0) * frac, FRAC_BITS)

    if quadrant >= 2:
        return -y
    return y


@njit(cache=True)
def sincos_hk(lut, h_phases, sin_out, cos_out):
    """
    Fill sin/cos tables for [H, K] phase array (Q32 phases).

    Bit-exact with sine_core.sin_cos_q15:
        sin = sine_q15(phase)
        cos = sine_q15((phase + 0x40000000) & 0xFFFFFFFF)
    """
    H = h_phases.shape[0]
    K = h_phases.shape[1]
    for h in range(H):
        for k in range(K):
            p = h_phases[h, k]
            sin_out[h, k] = _sine_q15_k(lut, p)
            cos_out[h, k] = _sine_q15_k(lut, (p + 0x40000000) & 0xFFFFFFFF)


# =============================================================================
# FWPT KERNEL (fast_lifting._fwpt_jit_core, array-native)
# =============================================================================

@njit(cache=True)
def fwpt_64(input_buf, headroom_bits, output_coeffs):
    """
    Forward wavelet packet transform for one 64-sample block.

    Bit-exact port of fast_lifting._fwpt_jit_core, taking/returning arrays
    directly (no list round-trips).
    """
    block_size = 64
    depth = 6

    fwd_levels = np.empty((7, 64), dtype=np.int64)
    temp_s = np.empty(64, dtype=np.int64)
    temp_d = np.empty(64, dtype=np.int64)

    for i in range(block_size):
        fwd_levels[0, i] = np.int64(input_buf[i]) >> headroom_bits

    for level in range(depth):
        num_nodes = 1 << level
        node_size = block_size >> level

        for node in range(num_nodes):
            start = node * node_size
            half = node_size >> 1
            if half == 0:
                fwd_levels[level + 1, node] = fwd_levels[level, start]
                continue

            for i in range(half):
                idx = start + 2 * i
                even_left = fwd_levels[level, idx]
                if idx + 2 < start + node_size:
                    even_right = fwd_levels[level, idx + 2]
                else:
                    even_right = even_left
                odd = fwd_levels[level, idx + 1]
                avg = (even_left + even_right + 1) >> 1
                temp_d[i] = odd - avg

            for i in range(half):
                if i > 0:
                    d_left = temp_d[i - 1]
                else:
                    d_left = temp_d[0]
                d_right = temp_d[i]
                update = (d_left + d_right + 2) >> 2
                temp_s[i] = fwd_levels[level, start + 2 * i] + update

            approx_start = node * 2 * half
            detail_start = approx_start + half
            for i in range(half):
                v = temp_s[i]
                if v < Q15_NEG_ONE:
                    v = -32768
                elif v > Q15_ONE:
                    v = 32767
                fwd_levels[level + 1, approx_start + i] = v

                v = temp_d[i]
                if v < Q15_NEG_ONE:
                    v = -32768
                elif v > Q15_ONE:
                    v = 32767
                fwd_levels[level + 1, detail_start + i] = v

    for i in range(block_size):
        output_coeffs[i] = fwd_levels[depth, i]


# =============================================================================
# FM6 BLOCK KERNEL (fm6_engine.fm6_generate_block_full)
# =============================================================================

@njit(cache=True)
def _wrap32(x):
    """Emulate numpy int32 wraparound for a value already in int64."""
    return ((x + 0x80000000) & 0xFFFFFFFF) - 0x80000000


@njit(cache=True)
def _fm6_rshift_rne(val, shift):
    """Port of fm6_engine._rshift_rne (biased round, ties cleared to even)."""
    half = 1 << (shift - 1)
    result = (val + half) >> shift
    if (val & ((1 << shift) - 1)) == half:
        result = result & ~np.int64(1)
    return result


@njit(cache=True)
def fm6_block_full(
    lut,
    eval_order,
    mod_matrix,
    feedback_op,
    feedback_q15,
    fb_prev,
    phases,
    phase_incs,
    levels,
    envs,
    carrier_mask,
    num_carriers,
    master_level,
    mod_index_q16,
    fb_row,
    out_audio,
):
    """
    Bit-exact port of fm6_engine.fm6_generate_block_full inner loops.

    Args:
        lut: quarter-wave sine LUT (int64[1025], from sine_core)
        eval_order: int64[6] operator evaluation order
        mod_matrix: uint8[6,6], mod_matrix[src, op] != 0 if src modulates op
        feedback_op: feedback operator index (-1 if none)
        feedback_q15: feedback depth of the feedback operator
        fb_prev: previous block's feedback sample (feedback_buffer[VM_K-1])
        phases: int64[6] in/out Q32 phase accumulators
        phase_incs: int64[6] per-operator Q32 phase increments
        levels, envs: int64[6] Q15 level / envelope values
        carrier_mask: 6-bit carrier mask
        num_carriers: number of carriers (for normalization)
        master_level: Q15 master level
        mod_index_q16: Q16 phase-modulation depth scale (65536 == legacy << 16)
        fb_row: int64[64] out - feedback operator's sample outputs
        out_audio: int16[64] out - final audio block (pre post-filter)
    """
    num_samples = 64
    Q32_MASK = 0xFFFFFFFF

    op_out = np.zeros((6, num_samples), dtype=np.int64)

    for oi in range(6):
        op_idx = eval_order[oi]
        op_phase_inc = phase_incs[op_idx]
        level = levels[op_idx]
        env = envs[op_idx]
        phase = phases[op_idx]
        is_fb = (feedback_op == op_idx) and (feedback_q15 > 0)

        for s in range(num_samples):
            mod_sum = np.int64(0)
            for src in range(6):
                if mod_matrix[src, op_idx]:
                    mod_sum += op_out[src, s]

            if is_fb:
                if s == 0:
                    fb_val = fb_prev
                else:
                    fb_val = op_out[op_idx, s - 1]
                mod_sum += (fb_val * feedback_q15) >> 15

            phase = (phase + op_phase_inc) & Q32_MASK

            mod_phase = (mod_sum * mod_index_q16) & Q32_MASK
            effective_phase = (phase + mod_phase) & Q32_MASK

            sin_val = _sine_q15_k(lut, effective_phase)

            output = _fm6_rshift_rne(sin_val * level, 15)
            output = _fm6_rshift_rne(output * env, 15)
            if output > Q15_ONE:
                output = 32767
            elif output < Q15_NEG_ONE:
                output = -32768

            op_out[op_idx, s] = output

        phases[op_idx] = phase

    if feedback_op >= 0:
        for s in range(num_samples):
            fb_row[s] = op_out[feedback_op, s]

    for s in range(num_samples):
        acc = np.int64(0)
        for c in range(6):
            if carrier_mask & (1 << c):
                acc += op_out[c, s]
        # Reference computes (sum * master) in an int32 numpy array, which
        # can wrap; emulate the wraparound at each int32-array step.
        prod = _wrap32(acc * master_level)
        val = _wrap32(prod + Q15_HALF) >> 15
        if num_carriers > 1:
            # numpy int32 floor division (matches Python floor semantics)
            val = val // num_carriers
        if val > Q15_ONE:
            val = 32767
        elif val < Q15_NEG_ONE:
            val = -32768
        out_audio[s] = val


__all__ = [
    'NUMBA_AVAILABLE',
    'jit_enabled',
    'set_jit_enabled',
    # executor kernels
    'inject_blend',
    'evolution_scale',
    'gain_interp',
    # operator dispatch
    'dispatch_backbone_op',
    # tail
    'tail_ccs_soft_limit',
    # render
    'idwpt_polyphase',
    # mode B helpers
    'sincos_hk',
    '_get_sine_lut_np',
]
