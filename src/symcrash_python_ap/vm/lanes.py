"""
VM Lanes (WSE-STD-VM-HYBRID v0)
===============================

Fixed execution lane implementations for the hybrid VM.
Each lane is a deterministic function on VM state.
"""

import numpy as np
from typing import List, Optional

from .state import VMState, EnvelopePhase, prng_step
from .decode import DecodedPlan, BackboneEntry, CarrierRole
from .tables import Lane, get_operator_lane, BLOCK_SAMPLES, COEFF_LEN, NODE_LEN, L_TOTAL_NODES
from ..fixed import RSHIFT_RNE

# Use canonical sine_core for bit-exact trig lookups
from ..wavelet.sine_core import (
    sine_q15 as _sine_q15,
    cosine_q15 as _cosine_q15,
    sine_q15_vectorized as _sine_vectorized,
)

# Q15 constants
Q15_ONE = 32767
Q15_HALF = 16384


# =============================================================================
# BIT-EXACT INTEGER MATH (WSE-STD-VM-HYBRID determinism requirement)
# =============================================================================

def isqrt(x: int) -> int:
    """
    Integer square root using Newton-Raphson.

    Returns floor(sqrt(x)) for non-negative x.
    Deterministic across all platforms.
    """
    if x < 0:
        raise ValueError("isqrt requires non-negative input")
    if x == 0:
        return 0

    # Initial guess: x >> (bit_length // 2)
    guess = 1 << ((x.bit_length() + 1) >> 1)

    # Newton-Raphson iteration
    while True:
        next_guess = (guess + x // guess) >> 1
        if next_guess >= guess:
            return guess
        guess = next_guess


def isqrt_ratio_q15(num: int, denom: int) -> int:
    """
    Compute sqrt(num / denom) * Q15_ONE using integer math only.

    Returns Q15 fixed-point result.
    For tail operations that need sqrt(target / energy) scaling.
    """
    if denom <= 0 or num <= 0:
        return 0

    # Compute (num << 30) / denom, then sqrt
    # This gives us sqrt(num/denom) * (1 << 15) = sqrt(num/denom) in Q15
    scaled = (num << 30) // denom
    return isqrt(scaled)


def cos_q15(eta_q15: int) -> int:
    """
    Cosine lookup for equal-power crossfade using sine_core.py.

    Input: eta in Q15, representing 0 to pi/2 (0 to 32767)
    Output: cos(eta * pi/2) in Q15

    Uses the canonical 1024-entry quarter-wave LUT with interpolation.
    """
    # Map eta (0 to 32767) to phase (0 to 0x40000000 for 0° to 90°)
    # 0x40000000 = 90° in sine_core's 32-bit phase format
    phase = (eta_q15 * 0x40000000) >> 15
    return _cosine_q15(phase)


def sin_q15(eta_q15: int) -> int:
    """
    Sine lookup for equal-power crossfade using sine_core.py.

    Input: eta in Q15, representing 0 to pi/2 (0 to 32767)
    Output: sin(eta * pi/2) in Q15

    Uses the canonical 1024-entry quarter-wave LUT with interpolation.
    """
    # Map eta (0 to 32767) to phase (0 to 0x40000000 for 0° to 90°)
    phase = (eta_q15 * 0x40000000) >> 15
    return _sine_q15(phase)


# =============================================================================
# FAST VECTORIZED SINE/COSINE FOR MIX LANE
# =============================================================================
# Precompute full equal-power crossfade tables for 0-32767 eta range.
# This eliminates per-sample function call overhead.

# Quarter-wave sine table (1025 entries for interpolation)
_QW_SIZE = 1024
_QW_TABLE: Optional[np.ndarray] = None


def _init_qw_table() -> np.ndarray:
    """Initialize quarter-wave sine table (lazy)."""
    global _QW_TABLE
    if _QW_TABLE is None:
        # Generate quarter-wave sine [0, pi/2]
        angles = np.linspace(0, np.pi / 2, _QW_SIZE + 1)
        _QW_TABLE = (np.sin(angles) * Q15_ONE).astype(np.int32)
    return _QW_TABLE


def _sincos_batch_q15(phases: np.ndarray) -> tuple:
    """
    Batch sine/cosine lookup for equal-power crossfade.

    Args:
        phases: uint32 array of phases (0 to 0x40000000 for 0° to 90°)

    Returns:
        (cos_vals, sin_vals) - both int32 arrays in Q15

    Note: Phases are restricted to [0, 0x40000000] (first quadrant only)
    since eta is always in [0, 32767] for crossfade.
    """
    table = _init_qw_table()

    # Scale phases to table indices with 12-bit fraction
    # phases are in [0, 0x40000000] (30-bit), scale to [0, 1024.0] with 12-bit frac
    # scaled = phase * (1024 << 12) / 0x40000000 = phase >> 8
    scaled = phases >> 8  # 22-bit value: 10-bit index + 12-bit frac

    indices = (scaled >> 12).astype(np.int32)
    fracs = (scaled & 0xFFF).astype(np.int32)

    # Clamp indices to valid range
    indices = np.clip(indices, 0, _QW_SIZE - 1)

    # Linear interpolation for sine: y = y0 + (y1 - y0) * frac / 4096
    y0 = table[indices]
    y1 = table[np.minimum(indices + 1, _QW_SIZE)]
    sin_vals = y0 + ((y1 - y0) * fracs >> 12)

    # Cosine = sin(90° - θ) = table[1024 - index] with reversed interpolation
    # At index=0: cos = table[1024] = peak
    # At index=1024: cos = table[0] = 0
    cos_indices = _QW_SIZE - indices
    cos_indices = np.clip(cos_indices, 1, _QW_SIZE)  # 1 to 1024

    # For descending lookup, we interpolate from higher to lower index
    # As frac increases, cosine should DECREASE (move toward lower table index)
    cy0 = table[cos_indices]
    cy1 = table[cos_indices - 1]
    cos_vals = cy0 - ((cy0 - cy1) * fracs >> 12)

    return cos_vals, sin_vals


# =============================================================================
# WET PATH OLA (Block Boundary Smoothing for IDWPT)
# =============================================================================
# OLA is applied to the wet (IDWPT) path ONLY, not the carrier.
# The carrier is phase-continuous across blocks and needs no OLA.
# Applying OLA to a continuous signal corrupts it by crossfading non-matching samples.

WET_OLA_SIZE = 8  # Overlap samples (must match state.py)

# Precomputed Hann window for OLA (fade-out and fade-in)
_WET_OLA_FADE_OUT = np.array([
    (np.cos(np.pi * i / WET_OLA_SIZE) + 1) * 16384  # Q15 Hann fade-out
    for i in range(WET_OLA_SIZE)
], dtype=np.int32)

_WET_OLA_FADE_IN = np.array([
    (1 - np.cos(np.pi * i / WET_OLA_SIZE)) * 16384  # Q15 Hann fade-in
    for i in range(WET_OLA_SIZE)
], dtype=np.int32)


def apply_wet_ola(state: VMState, wet_buf: np.ndarray) -> np.ndarray:
    """
    Apply OLA to wet path output to smooth IDWPT block boundaries.

    Uses Hann window crossfade at block start.

    Args:
        state: VMState containing wet_ola_tail from previous block
        wet_buf: Current block's wet (IDWPT) output (int16 array)

    Returns:
        Smoothed wet output (modifies wet_buf in place and returns it)
    """
    if not state.wet_ola_initialized:
        # First block - save tail and return unchanged
        state.wet_ola_tail[:] = wet_buf[-WET_OLA_SIZE:]
        state.wet_ola_initialized = True
        return wet_buf

    # Crossfade at block start: prev_tail fades out, new_head fades in
    for i in range(WET_OLA_SIZE):
        prev_sample = int(state.wet_ola_tail[i])
        curr_sample = int(wet_buf[i])
        # Hann crossfade with proper rounding: prev * fade_out + curr * fade_in
        blended = RSHIFT_RNE(
            prev_sample * _WET_OLA_FADE_OUT[i] + curr_sample * _WET_OLA_FADE_IN[i], 15
        )
        # Clamp to int16
        wet_buf[i] = max(-32768, min(32767, blended))

    # Save tail for next block
    state.wet_ola_tail[:] = wet_buf[-WET_OLA_SIZE:]

    return wet_buf


# =============================================================================
# LANE 1: CARRIER RENDER (Optimized with pre-allocated buffers)
# =============================================================================

# Pre-allocated buffers for carrier render (avoid per-block allocations)
_CARRIER_SAMPLES = np.zeros(BLOCK_SAMPLES, dtype=np.int32)
_CARRIER_PHASES = np.zeros(BLOCK_SAMPLES, dtype=np.int64)
_CARRIER_OUTPUT = np.zeros(BLOCK_SAMPLES, dtype=np.int16)


def lane_carrier_render(state: VMState, cfg: DecodedPlan) -> np.ndarray:
    """
    Render carrier oscillator for one block (vectorized).

    Phase-continuous additive synthesis with up to 16 harmonics.
    Uses NumPy vectorization with pre-allocated output buffer.

    Args:
        state: VM state (modified in place)
        cfg: Decoded plan

    Returns:
        carrier_buf: int16 array of BLOCK_SAMPLES
    """
    global _CARRIER_SAMPLES, _CARRIER_PHASES, _CARRIER_OUTPUT

    if cfg.carrier_role == CarrierRole.OFF:
        _CARRIER_OUTPUT.fill(0)
        return _CARRIER_OUTPUT

    # Get carrier parameters
    target_phase_inc = cfg.carrier_params.phase_inc
    harmonic_amps = cfg.carrier_params.harmonic_amps_q15
    note = cfg.carrier_params.note
    cents = cfg.carrier_params.cents

    # Update pitch slew target - use note/cents comparison for determinism
    # (Avoids floating-point noise causing constant triggering)
    if state.last_note != note or state.last_cents != cents:
        state.last_note = note
        state.last_cents = cents
        state.target_phase_inc = target_phase_inc
        diff = target_phase_inc - state.phase_inc
        state.pitch_slew_rate = diff // (BLOCK_SAMPLES * 8)

    # Fast path: no slew (common case) - avoid loop entirely
    slew_rate = state.pitch_slew_rate
    current_inc = state.phase_inc

    if slew_rate == 0:
        # If slew rounded to 0 but we have a target mismatch, snap to target immediately
        # This prevents phase discontinuity from using wrong frequency
        if current_inc != target_phase_inc:
            current_inc = target_phase_inc
            state.phase_inc = target_phase_inc
        # Vectorized phase generation
        sample_indices = np.arange(BLOCK_SAMPLES, dtype=np.int64)
        base_phases = (state.phase + sample_indices * current_inc) & 0xFFFFFFFF
        state.phase = (state.phase + BLOCK_SAMPLES * current_inc) & 0xFFFFFFFF
    else:
        # Slew case - must iterate
        phase = state.phase
        for i in range(BLOCK_SAMPLES):
            _CARRIER_PHASES[i] = phase
            if slew_rate > 0 and current_inc < target_phase_inc:
                current_inc = min(current_inc + slew_rate, target_phase_inc)
            elif slew_rate < 0 and current_inc > target_phase_inc:
                current_inc = max(current_inc + slew_rate, target_phase_inc)
            else:
                slew_rate = 0
            phase = (phase + current_inc) & 0xFFFFFFFF
        state.phase = phase
        state.phase_inc = current_inc
        state.pitch_slew_rate = slew_rate
        base_phases = _CARRIER_PHASES

    # Clear samples buffer
    _CARRIER_SAMPLES.fill(0)

    # Compute normalization factor: sum of all harmonic amplitudes
    # This prevents clipping when harmonics add in phase
    total_amp = sum(harmonic_amps)
    if total_amp == 0:
        total_amp = Q15_ONE

    # Vectorized harmonic synthesis
    for h in range(len(harmonic_amps)):
        amp = harmonic_amps[h]
        if amp == 0:
            continue

        # Phase for this harmonic: base_phase * (h + 1)
        h_phases = (base_phases * (h + 1)) & 0xFFFFFFFF
        # Extract 16-bit phase for lookup
        phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
        # Compute sine values using canonical 64K LUT from sine_core
        sin_vals = _sine_vectorized(phase_norm)
        # Accumulate (normalized by Q15)
        _CARRIER_SAMPLES += (sin_vals * amp) >> 15

    # Normalize output by total amplitude and add headroom for DC filter
    # The DC blocking filter has gain > 1 at high frequencies (~5.8% overshoot)
    # Scale down to prevent clipping: headroom = 30983/32767 ≈ 0.9456
    DC_HEADROOM = 30983  # Q15 scale factor to prevent DC filter clipping
    # Always use Q15 scaling for consistency - normalize by max(total_amp, Q15_ONE)
    divisor = max(total_amp, Q15_ONE)
    _CARRIER_SAMPLES = (_CARRIER_SAMPLES * DC_HEADROOM) // divisor

    # No DC filter needed in all-digital implementation:
    # - Sine LUT is symmetric around zero (no inherent DC)
    # - No ADC offset or analog component drift
    # - Headroom scaling preserves symmetry

    # Clamp to int16 range (should rarely trigger with headroom applied)
    np.clip(_CARRIER_SAMPLES, -32768, 32767, out=_CARRIER_SAMPLES)
    _CARRIER_OUTPUT[:] = _CARRIER_SAMPLES.astype(np.int16)

    return _CARRIER_OUTPUT


# =============================================================================
# LANE 2: INJECT
# =============================================================================

def lane_inject(state: VMState, cfg: DecodedPlan) -> None:
    """
    Apply event injections to coefficient state.

    Modifies state.coeff_state in place.
    PRNG advances ONLY if noise ops execute.

    Args:
        state: VM state (modified in place)
        cfg: Decoded plan
    """
    for event in cfg.event_instrs:
        op_id = event.op_id
        strength = event.strength_q15

        if op_id == 13:  # Impulse
            _inject_impulse(state, cfg.mask_u32, strength, event.payload64)
        elif op_id == 14:  # NoiseBand
            _inject_noise(state, cfg.mask_u32, strength, event.lane)
        elif op_id == 8:  # GlobalPulse
            _inject_global_pulse(state, cfg.mask_u32, strength)
        elif op_id == 21:  # CuspInjection
            _inject_cusp(state, cfg.mask_u32, strength)
        elif op_id == 16:  # GateMask
            _inject_gate_mask(state, cfg.mask_u32, strength)


def _inject_impulse(state: VMState, mask: int, strength: int, payload: int) -> None:
    """Inject sparse impulse into coefficients."""
    # Target specific nodes based on payload
    target_node = payload & 0x1F  # Use low 5 bits for node selection

    if not (mask & (1 << target_node)):
        return  # Node not active

    # Inject into first coefficient of target node
    coeff_idx = target_node * NODE_LEN
    if coeff_idx < COEFF_LEN:
        state.coeff_state[coeff_idx] = min(32767, max(-32768,
            state.coeff_state[coeff_idx] + ((strength * (payload >> 8 & 0xFF)) >> 8)
        ))


def _inject_noise(state: VMState, mask: int, strength: int, lane: int) -> None:
    """Inject band-limited noise into coefficients."""
    # Determine target band from lane
    band_start = (lane * COEFF_LEN) // 8
    band_end = ((lane + 1) * COEFF_LEN) // 8

    for i in range(band_start, min(band_end, COEFF_LEN)):
        # Check if node is active
        node = i // NODE_LEN
        if not (mask & (1 << node)):
            continue

        # Get random value (advances PRNG)
        rand_val = prng_step(state)

        # Scale to Q15 range and apply strength
        noise = ((rand_val & 0xFFFF) - 32768)
        noise = (noise * strength) >> 15

        state.coeff_state[i] = min(32767, max(-32768,
            state.coeff_state[i] + noise
        ))


def _inject_global_pulse(state: VMState, mask: int, strength: int) -> None:
    """Inject global pulse with coarse-heavy bias."""
    # Weight toward low nodes (coarse)
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue

        # Coarse-heavy weight: node 0 gets full, node 31 gets 1/8
        weight = (L_TOTAL_NODES - node) * strength // L_TOTAL_NODES

        coeff_idx = node * NODE_LEN
        if coeff_idx < COEFF_LEN:
            state.coeff_state[coeff_idx] = min(32767, max(-32768,
                state.coeff_state[coeff_idx] + weight
            ))


def _inject_cusp(state: VMState, mask: int, strength: int) -> None:
    """Inject asymmetric cusp perturbation."""
    # Asymmetric injection: positive on even nodes, negative on odd
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue

        sign = 1 if (node % 2) == 0 else -1
        coeff_idx = node * NODE_LEN
        if coeff_idx < COEFF_LEN:
            state.coeff_state[coeff_idx] = min(32767, max(-32768,
                state.coeff_state[coeff_idx] + sign * (strength >> 2)
            ))


def _inject_gate_mask(state: VMState, mask: int, strength: int) -> None:
    """Apply deterministic gate pattern."""
    # Sturmian gate: based on golden ratio
    phi = 0.618033988749895
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue

        # Sturmian sequence: floor((n+1)*phi) - floor(n*phi)
        gate_open = int((node + 1) * phi) - int(node * phi)

        if gate_open:
            # Gate is open - preserve coefficient
            pass
        else:
            # Gate is closed - attenuate
            coeff_idx = node * NODE_LEN
            for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
                state.coeff_state[i] = (state.coeff_state[i] * (Q15_ONE - strength)) >> 15


# =============================================================================
# LANE 3-6: COEFFICIENT BACKBONE OPERATORS
# =============================================================================

def dispatch_coeff_op(state: VMState, instr: BackboneEntry, mask: int) -> None:
    """
    Dispatch a backbone operator to its lane.

    Args:
        state: VM state (modified in place)
        instr: Backbone instruction
        mask: Active node mask
    """
    op_lane = get_operator_lane(instr.op_id)
    lane = op_lane.lane

    if lane == Lane.DIAG:
        lane_diag(state, instr, mask)
    elif lane == Lane.BANDED:
        lane_banded(state, instr, mask)
    elif lane == Lane.PERMUTE:
        lane_permute(state, instr, mask)
    elif lane == Lane.LOWRANK:
        lane_lowrank(state, instr, mask)
    elif lane == Lane.PROJ:
        lane_proj(state, instr, mask)
    elif lane == Lane.INJECT:
        # Inject in backbone is treated as immediate injection
        _inject_impulse(state, mask, instr.strength_q15, 0)


def lane_diag(state: VMState, instr: BackboneEntry, mask: int) -> None:
    """Apply diagonal (elementwise) operator."""
    op_id = instr.op_id
    strength = instr.strength_q15

    if op_id == 0:  # Identity
        pass  # No-op

    elif op_id == 1:  # Tilt
        # Frequency-dependent gain: boost high, attenuate low
        for node in range(L_TOTAL_NODES):
            if not (mask & (1 << node)):
                continue

            # Tilt factor: node 0 = 0.5, node 31 = 1.5
            tilt_factor = Q15_HALF + (node * Q15_ONE) // L_TOTAL_NODES
            tilt_factor = Q15_ONE + ((tilt_factor - Q15_ONE) * strength) // Q15_ONE

            coeff_idx = node * NODE_LEN
            for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
                # Use Python int to avoid numpy overflow
                val = int(state.coeff_state[i])
                state.coeff_state[i] = min(32767, max(-32768,
                    (val * tilt_factor) >> 15
                ))

    elif op_id in (15, 20):  # PressureRamp, FoldPressure
        # Catastrophe regime pressure
        for node in range(L_TOTAL_NODES):
            if not (mask & (1 << node)):
                continue

            coeff_idx = node * NODE_LEN
            for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
                # Soft saturation (use Python int to avoid overflow)
                x = int(state.coeff_state[i])
                if abs(x) > Q15_HALF:
                    sign = 1 if x > 0 else -1
                    x = sign * (Q15_HALF + ((abs(x) - Q15_HALF) * strength) // Q15_ONE)
                    state.coeff_state[i] = min(32767, max(-32768, x))

    elif op_id == 16:  # GateMask
        _inject_gate_mask(state, mask, strength)

    elif op_id == 22:  # SwallowtailMask
        # 3-parameter catastrophe gate
        for node in range(L_TOTAL_NODES):
            if not (mask & (1 << node)):
                continue

            # Swallowtail pattern based on node position
            phase = (node * 3) % 5
            gate = 1 if phase < 3 else 0

            if not gate:
                coeff_idx = node * NODE_LEN
                for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
                    # Use Python int to avoid overflow
                    val = int(state.coeff_state[i])
                    state.coeff_state[i] = (val * (Q15_ONE - strength)) >> 15


def lane_banded(state: VMState, instr: BackboneEntry, mask: int) -> None:
    """Apply banded (local stencil) operator."""
    op_id = instr.op_id
    strength = instr.strength_q15

    if op_id == 3:  # IntraMix
        # Tridiagonal smoothing within nodes
        tmp = state.coeff_state.copy()
        for node in range(L_TOTAL_NODES):
            if not (mask & (1 << node)):
                continue

            coeff_idx = node * NODE_LEN
            for i in range(coeff_idx + 1, min(coeff_idx + NODE_LEN - 1, COEFF_LEN - 1)):
                # [0.25, 0.5, 0.25] kernel (use Python int to avoid overflow)
                mixed = (int(tmp[i-1]) + 2 * int(tmp[i]) + int(tmp[i+1])) >> 2
                base = int(tmp[i])
                state.coeff_state[i] = base + ((mixed - base) * strength) >> 15

    elif op_id == 4:  # DiffuseUp
        # Energy transport: fine → coarse
        for node in range(1, L_TOTAL_NODES):
            if not (mask & (1 << node)):
                continue
            if not (mask & (1 << (node - 1))):
                continue

            # Transfer energy up (use Python int to avoid overflow)
            src_idx = node * NODE_LEN
            dst_idx = (node - 1) * NODE_LEN

            src_val = int(state.coeff_state[src_idx])
            dst_val = int(state.coeff_state[dst_idx])
            transfer = (src_val * strength) >> 16
            state.coeff_state[dst_idx] = min(32767, max(-32768, dst_val + transfer))
            state.coeff_state[src_idx] = min(32767, max(-32768, src_val - transfer))

    elif op_id == 5:  # DiffuseDown
        # Energy transport: coarse → fine
        for node in range(L_TOTAL_NODES - 1):
            if not (mask & (1 << node)):
                continue
            if not (mask & (1 << (node + 1))):
                continue

            # Transfer energy down (use Python int to avoid overflow)
            src_idx = node * NODE_LEN
            dst_idx = (node + 1) * NODE_LEN

            src_val = int(state.coeff_state[src_idx])
            dst_val = int(state.coeff_state[dst_idx])
            transfer = (src_val * strength) >> 16
            state.coeff_state[dst_idx] = min(32767, max(-32768, dst_val + transfer))
            state.coeff_state[src_idx] = min(32767, max(-32768, src_val - transfer))


def lane_permute(state: VMState, instr: BackboneEntry, mask: int) -> None:
    """Apply permutation operator."""
    op_id = instr.op_id
    strength = instr.strength_q15

    if strength < Q15_HALF:
        return  # Weak permutation is no-op

    if op_id == 6:  # BandShear
        # Structured permutation within bands
        tmp = state.coeff_state.copy()
        for node in range(L_TOTAL_NODES):
            if not (mask & (1 << node)):
                continue

            coeff_idx = node * NODE_LEN
            # Rotate coefficients within node
            shift = (node % NODE_LEN)
            for i in range(NODE_LEN):
                src = coeff_idx + i
                dst = coeff_idx + ((i + shift) % NODE_LEN)
                if src < COEFF_LEN and dst < COEFF_LEN:
                    state.coeff_state[dst] = tmp[src]

    elif op_id == 9:  # SwapBands
        # Swap adjacent node pairs
        tmp = state.coeff_state.copy()
        for node in range(0, L_TOTAL_NODES - 1, 2):
            if not (mask & (1 << node)):
                continue
            if not (mask & (1 << (node + 1))):
                continue

            src_idx = node * NODE_LEN
            dst_idx = (node + 1) * NODE_LEN
            for i in range(NODE_LEN):
                if src_idx + i < COEFF_LEN and dst_idx + i < COEFF_LEN:
                    state.coeff_state[src_idx + i] = tmp[dst_idx + i]
                    state.coeff_state[dst_idx + i] = tmp[src_idx + i]

    elif op_id == 10:  # RotateNodes
        # Circular rotation of all nodes
        tmp = state.coeff_state.copy()
        active_nodes = [n for n in range(L_TOTAL_NODES) if mask & (1 << n)]
        if len(active_nodes) > 1:
            for i, node in enumerate(active_nodes):
                next_node = active_nodes[(i + 1) % len(active_nodes)]
                src_idx = node * NODE_LEN
                dst_idx = next_node * NODE_LEN
                for j in range(NODE_LEN):
                    if src_idx + j < COEFF_LEN and dst_idx + j < COEFF_LEN:
                        state.coeff_state[dst_idx + j] = tmp[src_idx + j]

    elif op_id == 17:  # QuasiRotate
        # Irrational rotation (golden ratio based)
        phi = 0.618033988749895
        tmp = state.coeff_state.copy()
        for node in range(L_TOTAL_NODES):
            if not (mask & (1 << node)):
                continue

            # Target node based on golden ratio
            target = int((node + phi * L_TOTAL_NODES) % L_TOTAL_NODES)
            if not (mask & (1 << target)):
                continue

            src_idx = node * NODE_LEN
            dst_idx = target * NODE_LEN
            for i in range(NODE_LEN):
                if src_idx + i < COEFF_LEN and dst_idx + i < COEFF_LEN:
                    # Blend instead of hard swap (use Python int to avoid overflow)
                    dst_val = int(tmp[dst_idx + i])
                    src_val = int(tmp[src_idx + i])
                    state.coeff_state[dst_idx + i] = min(32767, max(-32768,
                        (dst_val * (Q15_ONE - strength) + src_val * strength) >> 15
                    ))


def lane_lowrank(state: VMState, instr: BackboneEntry, mask: int) -> None:
    """Apply low-rank operator."""
    op_id = instr.op_id
    strength = instr.strength_q15

    if op_id == 7:  # LowRankMorph
        # UV^T rank-2 motif
        # Compute mean across active nodes (use Python int to avoid overflow)
        active_count = 0
        mean = 0
        for node in range(L_TOTAL_NODES):
            if mask & (1 << node):
                coeff_idx = node * NODE_LEN
                if coeff_idx < COEFF_LEN:
                    mean += int(state.coeff_state[coeff_idx])
                    active_count += 1

        if active_count > 0:
            mean //= active_count

            # Apply rank-1 update: x += strength * (mean - x)
            for node in range(L_TOTAL_NODES):
                if not (mask & (1 << node)):
                    continue

                coeff_idx = node * NODE_LEN
                for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
                    val = int(state.coeff_state[i])
                    diff = mean - val
                    state.coeff_state[i] = min(32767, max(-32768,
                        val + ((diff * strength) >> 15)
                    ))


def lane_proj(state: VMState, instr: BackboneEntry, mask: int) -> None:
    """Apply projection operator (non-tail PROJ ops)."""
    op_id = instr.op_id
    strength = instr.strength_q15

    if op_id == 12:  # GluePair
        # Sheaf compatibility between adjacent bands
        for node in range(L_TOTAL_NODES - 1):
            if not (mask & (1 << node)):
                continue
            if not (mask & (1 << (node + 1))):
                continue

            # Make adjacent nodes more similar (use Python int to avoid overflow)
            idx_a = node * NODE_LEN
            idx_b = (node + 1) * NODE_LEN

            if idx_a < COEFF_LEN and idx_b < COEFF_LEN:
                val_a = int(state.coeff_state[idx_a])
                val_b = int(state.coeff_state[idx_b])
                avg = (val_a + val_b) >> 1
                state.coeff_state[idx_a] = min(32767, max(-32768,
                    val_a + ((avg - val_a) * strength) >> 15
                ))
                state.coeff_state[idx_b] = min(32767, max(-32768,
                    val_b + ((avg - val_b) * strength) >> 15
                ))


# =============================================================================
# LANE 7: TAIL
# =============================================================================

def lane_tail(state: VMState, cfg: DecodedPlan) -> None:
    """
    Apply tail operators.

    Tail enforces "legal/pleasant" bounds on coefficients.

    For CCS mode, uses a simple soft energy limiter instead of NODE-based
    operators (which assume a different layout).

    Args:
        state: VM state (modified in place)
        cfg: Decoded plan
    """
    # CCS mode: use soft energy limiter instead of NODE-based operators
    if cfg.use_coeff_osc:
        _tail_ccs_soft_limit(state)
        return

    # Legacy mode: use NODE-based operators
    for op_id in cfg.tail_ops:
        if op_id == 2:  # FineClamp
            _tail_fine_clamp(state, cfg.mask_u32)
        elif op_id == 11:  # NormalizeScale
            _tail_normalize_scale(state, cfg.mask_u32)
        elif op_id == 23:  # SheafProject
            _tail_sheaf_project(state, cfg.mask_u32)
        elif op_id == 19:  # EmergencySimplify
            _tail_emergency_simplify(state, cfg.mask_u32)


# Maximum RMS energy target for CCS coefficients (Q15)
# Allows ~0.7 peak amplitude before limiting kicks in
_CCS_MAX_RMS_Q15 = 23170  # sqrt(0.5) * 32767


def _tail_ccs_soft_limit(state: VMState) -> None:
    """
    CCS-specific soft energy limiter.

    Prevents coefficient runaway by applying gentle gain reduction when
    total energy exceeds target. Works with polyphase layout (64 leaves × 8 k-slots).

    This replaces the NODE-based tail operators which assume a different layout.
    """
    # Compute total RMS energy across all coefficients
    coeffs = state.coeff_state[:COEFF_LEN].astype(np.int64)
    energy = np.sum(coeffs * coeffs)
    rms = int(np.sqrt(energy / COEFF_LEN))

    if rms <= _CCS_MAX_RMS_Q15:
        return  # Energy within bounds

    # Apply soft gain reduction: gain = target / current
    # Implemented as: new_coeff = old_coeff * target / rms
    # Using integer math: (coeff * target + rms//2) // rms
    gain_num = _CCS_MAX_RMS_Q15
    gain_den = max(1, rms)

    # Vectorized scaling
    scaled = (coeffs * gain_num + gain_den // 2) // gain_den
    state.coeff_state[:COEFF_LEN] = np.clip(scaled, -32768, 32767).astype(np.int16)


def _tail_fine_clamp(state: VMState, mask: int) -> None:
    """Per-band energy cap using vectorized operations."""
    # Max energy per node: allow each coefficient to be at most Q15_ONE
    max_energy_per_node = NODE_LEN * Q15_ONE * Q15_ONE

    # Reshape coefficients into nodes for vectorized operations
    # Only process first L_TOTAL_NODES * NODE_LEN coefficients
    n_coeffs = min(L_TOTAL_NODES * NODE_LEN, COEFF_LEN)
    coeff_view = state.coeff_state[:n_coeffs].reshape(L_TOTAL_NODES, NODE_LEN).astype(np.int64)

    # Compute energy per node (sum of squares)
    energies = (coeff_view * coeff_view).sum(axis=1)

    # Find nodes that are over budget and in mask
    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue
        if energies[node] > max_energy_per_node:
            scale_q15 = isqrt_ratio_q15(max_energy_per_node, int(energies[node]))
            coeff_idx = node * NODE_LEN
            end_idx = min(coeff_idx + NODE_LEN, COEFF_LEN)
            for i in range(coeff_idx, end_idx):
                val = int(state.coeff_state[i])
                state.coeff_state[i] = (val * scale_q15) >> 15


def _tail_normalize_scale(state: VMState, mask: int) -> None:
    """Per-band energy normalization using vectorized operations."""
    target_energy = Q15_HALF * NODE_LEN

    # Reshape for vectorized energy computation
    n_coeffs = min(L_TOTAL_NODES * NODE_LEN, COEFF_LEN)
    coeff_view = state.coeff_state[:n_coeffs].reshape(L_TOTAL_NODES, NODE_LEN).astype(np.int64)

    # Compute energy per node (sum of val^2 >> 15)
    energies = ((coeff_view * coeff_view) >> 15).sum(axis=1)

    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue

        energy = int(energies[node])
        if energy > 0:
            scale = isqrt_ratio_q15(target_energy, energy)
            scale = max(Q15_HALF // 2, min(Q15_ONE * 2, scale))

            coeff_idx = node * NODE_LEN
            end_idx = min(coeff_idx + NODE_LEN, COEFF_LEN)
            for i in range(coeff_idx, end_idx):
                val = int(state.coeff_state[i])
                state.coeff_state[i] = min(32767, max(-32768, (val * scale) >> 15))


def _tail_sheaf_project(state: VMState, mask: int) -> None:
    """Local-to-global consistency projection."""
    # Ensure smooth transitions between adjacent nodes
    for node in range(L_TOTAL_NODES - 1):
        if not (mask & (1 << node)):
            continue
        if not (mask & (1 << (node + 1))):
            continue

        # Boundary coefficients should be similar
        idx_a = node * NODE_LEN + NODE_LEN - 1  # Last of node
        idx_b = (node + 1) * NODE_LEN           # First of next node

        if idx_a < COEFF_LEN and idx_b < COEFF_LEN:
            # Use Python int to avoid numpy overflow
            avg = (int(state.coeff_state[idx_a]) + int(state.coeff_state[idx_b])) >> 1
            state.coeff_state[idx_a] = avg
            state.coeff_state[idx_b] = avg


def _tail_emergency_simplify(state: VMState, mask: int) -> None:
    """Hard projection to safe envelope."""
    # Zero out fine nodes, keep only coarse
    for node in range(L_TOTAL_NODES):
        if node >= 8:  # Keep only first 8 nodes (coarse)
            coeff_idx = node * NODE_LEN
            for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
                state.coeff_state[i] = 0

    # Also clamp remaining to safe range
    for i in range(min(8 * NODE_LEN, COEFF_LEN)):
        state.coeff_state[i] = max(-Q15_HALF, min(Q15_HALF, state.coeff_state[i]))


# =============================================================================
# LANE 8: IDWPT RENDER
# =============================================================================

# Import JIT-accelerated lifting for real-time IDWPT (fastest)
from ..wavelet.fast_lifting import (
    effect_mode_jit,
    idwpt_balanced,
    NUMBA_AVAILABLE,
    # Separated transforms for CCS architecture
    fwpt_jit,
    idwpt_jit,
)

# Import polyphase IDWPT for CCS K=4 interleaving
from ..wavelet.idwpt_streaming import idwpt_polyphase_block

# Import CCS functions for coefficient-domain synthesis (WSE-STD-121)
from .coeff_lut_streaming import (
    CCSState,
    ccs_generate_coeffs,
    get_lut_for_note,
    VM_K,
)


# Pre-allocated buffers for IDWPT lane (avoid per-block allocations)
_IDWPT_OUTPUT = np.zeros(BLOCK_SAMPLES, dtype=np.int16)
_IDWPT_LEAF_ENERGIES = np.zeros(64, dtype=np.int32)
_IDWPT_NODE_MASK_TEMPLATE = np.zeros(32, dtype=np.int32)


def lane_idwpt_render(
    state: VMState,
    cfg: DecodedPlan,
    carrier_buf: np.ndarray,
) -> np.ndarray:
    """
    Render IDWPT texture from coefficient state using fast_lifting.

    Uses phase/time coherent approach (no windowing):
    1. Mask smoothing: interpolate mask values across blocks
    2. C⁰ seam stitching: linear ramp correction for block continuity

    Args:
        state: VM state (contains coeff_state as mask source)
        cfg: Decoded plan (contains tree info, mask_u32)
        carrier_buf: Carrier buffer (int16 numpy array)

    Returns:
        texture_buf: int16 array of BLOCK_SAMPLES
    """
    global _IDWPT_OUTPUT, _IDWPT_LEAF_ENERGIES, _IDWPT_NODE_MASK_TEMPLATE

    if not cfg.idwpt_enable:
        _IDWPT_OUTPUT.fill(0)
        return _IDWPT_OUTPUT

    # Convert carrier to list (tolist() is fast for small arrays)
    carrier_list = carrier_buf.tolist()

    # Build target mask from coeff_state (existing logic)
    coeff_view = state.coeff_state[:COEFF_LEN].reshape(64, COEFF_LEN // 64)
    np.abs(coeff_view).sum(axis=1, out=_IDWPT_LEAF_ENERGIES)

    # Check if coeff_state has any energy
    total_energy = _IDWPT_LEAF_ENERGIES.sum()
    if total_energy == 0:
        # No coefficient energy - use None for true passthrough
        mask_target = None
    else:
        # Apply mask_u32: expand 32 nodes to 64 leaves
        mask_u32 = cfg.mask_u32
        for i in range(32):
            _IDWPT_NODE_MASK_TEMPLATE[i] = (mask_u32 >> i) & 1
        leaf_mask = np.repeat(_IDWPT_NODE_MASK_TEMPLATE, 2)

        # Compute target mask (before smoothing)
        mask_target = np.minimum(_IDWPT_LEAF_ENERGIES * leaf_mask, Q15_ONE)

    # =========================================================================
    # MASK SMOOTHING (Control Continuity)
    # =========================================================================
    # Interpolate mask values across blocks to prevent sudden operator changes.
    # This removes the "block phase lands differently → mask produces different
    # waveform" shock by making the operator time-smooth.
    MASK_ALPHA = 16384  # Q15 0.5 - blend 50% per block (~10ms for full change)

    if mask_target is not None:
        if state.mask_state_initialized:
            # Interpolate: smoothed = prev * (1-α) + target * α
            mask_smoothed = (
                (state.mask_state_q15.astype(np.int32) * (Q15_ONE - MASK_ALPHA) +
                 mask_target.astype(np.int32) * MASK_ALPHA) >> 15
            ).astype(np.int16)
        else:
            # First block with mask - initialize state
            mask_smoothed = mask_target.astype(np.int16)
            state.mask_state_initialized = True

        # Update state for next block
        state.mask_state_q15[:] = mask_smoothed
        mask_q15 = mask_smoothed.tolist()
    else:
        # Identity mask - decay mask state toward zero
        if state.mask_state_initialized:
            # Decay by half each block
            state.mask_state_q15[:] = (state.mask_state_q15.astype(np.int32) * MASK_ALPHA) >> 15
            # If energy drops below threshold, stop using smoothed mask
            if np.sum(np.abs(state.mask_state_q15)) < 64:
                mask_q15 = None
            else:
                mask_q15 = state.mask_state_q15.tolist()
        else:
            mask_q15 = None

    # Apply JIT effect mode: carrier -> FWPT -> mask -> IDWPT -> wet
    wet_list = effect_mode_jit(carrier_list, mask_q15)
    wet_arr = np.array(wet_list, dtype=np.int32)

    # =========================================================================
    # C⁰ SEAM STITCHING (Boundary Constraint)
    # =========================================================================
    # After IDWPT, correct wet output to match previous block's end value.
    # This is NOT windowing - it's a minimal linear ramp correction.
    # delta = prev_end - curr_start
    # wet[i] += delta * (1 - i/63) → forces wet[0] = prev_end, smooth ramp
    if state.wet_seam_initialized:
        delta = state.wet_prev_end - int(wet_arr[0])
        # Linear ramp correction: delta at i=0, 0 at i=63
        for i in range(BLOCK_SAMPLES):
            wet_arr[i] += (delta * (BLOCK_SAMPLES - 1 - i)) // (BLOCK_SAMPLES - 1)

    # Store end sample for next block
    state.wet_prev_end = int(wet_arr[BLOCK_SAMPLES - 1])
    if not state.wet_seam_initialized:
        state.wet_seam_initialized = True

    # Clamp and copy to output buffer
    _IDWPT_OUTPUT[:] = np.clip(wet_arr, -32768, 32767).astype(np.int16)

    return _IDWPT_OUTPUT


# =============================================================================
# LANE 8b: CCS RENDER (Coefficient-Synthesis path)
# =============================================================================
# This replaces the carrier→FWPT→mask→IDWPT path with:
# CoeffOsc → blend(coeff_state) → IDWPT → audio
#
# Eliminates phase-variant combing by generating coefficients directly
# from oscillator phases + tables, bypassing FWPT entirely.

# =============================================================================
# COEFFICIENT MAPPING HELPERS (256 ↔ 64)
# =============================================================================

def inject_64_to_512(coeffs_64: np.ndarray, coeff_state: np.ndarray) -> None:
    """
    Inject 64 carrier coefficients into 512-element state.

    Replicates each leaf across K=8 positions for polyphase compatibility.
    This allows the 64-leaf carrier to be modified by operators that
    work on the coefficient structure.

    Args:
        coeffs_64: 64 leaf coefficients (int16)
        coeff_state: 512-element coefficient state (int16, modified in place)
    """
    for i in range(64):
        val = coeffs_64[i]
        for j in range(VM_K):  # K=8
            coeff_state[i * VM_K + j] = val


def extract_64_leaves(coeff_state: np.ndarray) -> List[int]:
    """
    Extract 64 leaf coefficients from 512-element state.

    Uses energy-preserving summation: each group of K=8 coefficients
    is averaged to produce one leaf value.

    Args:
        coeff_state: 512-element coefficient state (int16)

    Returns:
        List of 64 leaf coefficients (int16)
    """
    # Reshape to (64, K) and sum along axis 1
    coeff_view = coeff_state[:COEFF_LEN].reshape(64, VM_K)
    # Sum with int32 to avoid overflow, then average (>> 3 for K=8)
    leaves_64 = (coeff_view.astype(np.int32).sum(axis=1) >> 3).astype(np.int16)
    return leaves_64.tolist()


# Pre-allocated buffer for CCS output
_CCS_OUTPUT = np.zeros(BLOCK_SAMPLES, dtype=np.int16)
_CCS_COEFFS_64 = np.zeros(64, dtype=np.int16)

# K=4 polyphase IDWPT (sample every 2nd slot from K=8 state)
_CCS_POLYPHASE_K = 4


def lane_ccs_render(
    state: VMState,
    cfg: DecodedPlan,
) -> np.ndarray:
    """
    Render audio from operator-modified coefficient state (CCS final step).

    ARCHITECTURE (CORRECT FLOW per executor.py):
        1. Carrier injected into coeff_state (executor.py, before operators)
        2. Events modify coeff_state (lane_inject)
        3. Operators modify coeff_state (backbone)
        4. Tail constrains coeff_state (lane_tail)
        5. THIS FUNCTION: Extract + IDWPT

    K=4 POLYPHASE IDWPT (WSE-STD-120 section 3.1):
        The VM's coeff_state is [64 leaves × 8 time samples] = 512 values.
        For K=4 polyphase reconstruction:
        - Sample k-slots 0, 2, 4, 6 (every 2nd slot from K=8)
        - Run 4 IDWPTs and interleave results
        - Produces 64 samples with proper phase continuity

    This provides smoother transitions and better phase coherence than
    single-slot (k=0 only) reconstruction.

    Args:
        state: VM state (coeff_state contains final operator-modified coefficients)
        cfg: Decoded plan

    Returns:
        audio_buf: int16 array of BLOCK_SAMPLES
    """
    global _CCS_OUTPUT

    # Reshape to [64 leaves, 8 k-slots] for K=8 interleaved polyphase
    coeff_2d = state.coeff_state[:COEFF_LEN].reshape(64, VM_K).astype(np.int16)

    # Run K=8 polyphase IDWPT with interleaving
    # Each sample n uses the reconstruction from slot (n % 8)
    _CCS_OUTPUT[:] = idwpt_polyphase_block(coeff_2d, headroom_bits=2)

    return _CCS_OUTPUT


# =============================================================================
# LANE 9: MIX + GAIN (Vectorized)
# =============================================================================

# Precomputed alpha values for interpolation (Q15)
_ALPHA_Q15 = np.array([(i * Q15_ONE) // BLOCK_SAMPLES for i in range(BLOCK_SAMPLES)], dtype=np.int32)


def lane_mix_gain(
    cfg: DecodedPlan,
    carrier_buf: np.ndarray,
    texture_buf: np.ndarray,
) -> np.ndarray:
    """
    Apply eta blend and gain ramp to produce final output (vectorized).

    Uses orthogonal residual projection for eta blend:
        output = carrier + eta * (texture - carrier)

    This approach reduces block discontinuity artifacts by blending the
    *difference* rather than the full signals. The residual is typically
    smaller than the wet signal, so discontinuities are quieter.

    All math is integer-only for bit-exact determinism.

    Args:
        cfg: Decoded plan
        carrier_buf: Carrier oscillator output
        texture_buf: IDWPT texture output

    Returns:
        output: int16 array of BLOCK_SAMPLES
    """
    # Vectorized eta interpolation
    eta_delta = cfg.eta_curr_q15 - cfg.eta_prev_q15
    eta_vals = cfg.eta_prev_q15 + ((_ALPHA_Q15 * eta_delta) >> 15)

    # Orthogonal residual projection: output = carrier + eta * (texture - carrier)
    # This is algebraically equivalent to linear blend (1-eta)*carrier + eta*texture
    # but computed as carrier + eta*residual for better discontinuity handling.
    carrier_i32 = carrier_buf.astype(np.int32)
    texture_i32 = texture_buf.astype(np.int32)

    # Compute residual (what the wavelet processing added/changed)
    residual = texture_i32 - carrier_i32

    # Blend: carrier stays, add scaled residual
    # At eta=0: output = carrier (pure carrier)
    # At eta=32767: output = carrier + residual = texture (pure wet)
    mixed = carrier_i32 + ((residual * eta_vals) >> 15)

    # Apply headroom to prevent clipping.
    # Residual peaks are typically smaller than wet peaks, so we can use
    # a higher factor (~0.866) compared to equal-power crossfade (~0.707).
    RESIDUAL_HEADROOM = 28378  # Q15 for ~0.866
    mixed = (mixed * RESIDUAL_HEADROOM) >> 15

    # Vectorized gain interpolation
    gain_delta = cfg.gain_curr_q15 - cfg.gain_prev_q15
    gain_vals = cfg.gain_prev_q15 + ((_ALPHA_Q15 * gain_delta) >> 15)

    # Apply gain
    samples = (mixed * gain_vals) >> 15

    # Clamp and return
    return np.clip(samples, -32768, 32767).astype(np.int16)


__all__ = [
    'lane_carrier_render',
    'lane_inject',
    'dispatch_coeff_op',
    'lane_diag',
    'lane_banded',
    'lane_permute',
    'lane_lowrank',
    'lane_proj',
    'lane_tail',
    'lane_idwpt_render',
    'lane_ccs_render',
    'lane_mix_gain',
    # Separated transforms (re-exported from fast_lifting)
    'fwpt_jit',
    'idwpt_jit',
]
