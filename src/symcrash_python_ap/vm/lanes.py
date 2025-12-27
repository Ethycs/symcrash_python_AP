"""
VM Lanes (WSE-STD-VM-HYBRID v0)
===============================

Fixed execution lane implementations for the hybrid VM.
Each lane is a deterministic function on VM state.
"""

import numpy as np
from typing import Optional

from .state import VMState, EnvelopePhase, prng_step
from .decode import DecodedPlan, BackboneEntry, CarrierRole
from .tables import Lane, get_operator_lane, BLOCK_SAMPLES, COEFF_LEN, NODE_LEN, L_TOTAL_NODES

# Use canonical sine_core for bit-exact trig lookups
from ..wavelet.sine_core import sine_q15 as _sine_q15, cosine_q15 as _cosine_q15

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
# LANE 1: CARRIER RENDER
# =============================================================================

def lane_carrier_render(state: VMState, cfg: DecodedPlan) -> np.ndarray:
    """
    Render carrier oscillator for one block.

    Phase-continuous additive synthesis with up to 16 harmonics.

    Args:
        state: VM state (modified in place)
        cfg: Decoded plan

    Returns:
        carrier_buf: int16 array of BLOCK_SAMPLES
    """
    carrier_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

    if cfg.carrier_role == CarrierRole.OFF:
        return carrier_buf

    # Get carrier parameters
    target_phase_inc = cfg.carrier_params.phase_inc
    harmonic_amps = cfg.carrier_params.harmonic_amps_q15

    # Update pitch slew target
    if state.target_phase_inc != target_phase_inc:
        state.target_phase_inc = target_phase_inc
        # Calculate slew rate (smooth over ~10ms)
        diff = target_phase_inc - state.phase_inc
        state.pitch_slew_rate = diff // (BLOCK_SAMPLES * 8)

    # Render samples
    for i in range(BLOCK_SAMPLES):
        # Apply pitch slew
        if state.pitch_slew_rate != 0:
            if state.pitch_slew_rate > 0 and state.phase_inc < state.target_phase_inc:
                state.phase_inc = min(state.phase_inc + state.pitch_slew_rate, state.target_phase_inc)
            elif state.pitch_slew_rate < 0 and state.phase_inc > state.target_phase_inc:
                state.phase_inc = max(state.phase_inc + state.pitch_slew_rate, state.target_phase_inc)
            else:
                state.pitch_slew_rate = 0

        # Additive synthesis with harmonics
        sample = 0
        for h in range(16):
            amp = harmonic_amps[h]
            if amp == 0:
                continue

            # Phase for this harmonic
            h_phase = (state.phase * (h + 1)) & 0xFFFFFFFF

            # Convert phase to sine value (Q15)
            # phase is 32-bit, use top bits for lookup
            phase_norm = (h_phase >> 16) & 0xFFFF
            # Simple sine approximation using parabola
            # sin(x) ≈ 4x(1-x) for x in [0,1]
            if phase_norm < 16384:
                # First quarter: rising
                x = phase_norm
                sin_val = (4 * x * (16384 - x)) >> 14
            elif phase_norm < 32768:
                # Second quarter: falling
                x = 32768 - phase_norm
                sin_val = (4 * x * (16384 - x)) >> 14
            elif phase_norm < 49152:
                # Third quarter: negative rising
                x = phase_norm - 32768
                sin_val = -((4 * x * (16384 - x)) >> 14)
            else:
                # Fourth quarter: negative falling
                x = 65536 - phase_norm
                sin_val = -((4 * x * (16384 - x)) >> 14)

            # Accumulate with harmonic amplitude
            sample += (sin_val * amp) >> 15

        # Advance phase
        state.phase = (state.phase + state.phase_inc) & 0xFFFFFFFF

        # DC blocking (1-pole highpass)
        # y[n] = x[n] - x[n-1] + 0.995 * y[n-1]
        dc_coeff = 32604  # ~0.995 in Q15
        x = sample
        y = x - state.dc_block_x1 + ((state.dc_block_y1 * dc_coeff) >> 15)
        state.dc_block_x1 = x
        state.dc_block_y1 = y
        sample = y

        # Clamp to int16
        sample = max(-32768, min(32767, sample))
        carrier_buf[i] = sample

    return carrier_buf


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
                # [0.25, 0.5, 0.25] kernel
                mixed = (tmp[i-1] + 2 * tmp[i] + tmp[i+1]) >> 2
                state.coeff_state[i] = tmp[i] + ((mixed - tmp[i]) * strength) >> 15

    elif op_id == 4:  # DiffuseUp
        # Energy transport: fine → coarse
        for node in range(1, L_TOTAL_NODES):
            if not (mask & (1 << node)):
                continue
            if not (mask & (1 << (node - 1))):
                continue

            # Transfer energy up
            src_idx = node * NODE_LEN
            dst_idx = (node - 1) * NODE_LEN

            transfer = (state.coeff_state[src_idx] * strength) >> 16
            state.coeff_state[dst_idx] = min(32767, max(-32768,
                state.coeff_state[dst_idx] + transfer
            ))
            state.coeff_state[src_idx] = min(32767, max(-32768,
                state.coeff_state[src_idx] - transfer
            ))

    elif op_id == 5:  # DiffuseDown
        # Energy transport: coarse → fine
        for node in range(L_TOTAL_NODES - 1):
            if not (mask & (1 << node)):
                continue
            if not (mask & (1 << (node + 1))):
                continue

            # Transfer energy down
            src_idx = node * NODE_LEN
            dst_idx = (node + 1) * NODE_LEN

            transfer = (state.coeff_state[src_idx] * strength) >> 16
            state.coeff_state[dst_idx] = min(32767, max(-32768,
                state.coeff_state[dst_idx] + transfer
            ))
            state.coeff_state[src_idx] = min(32767, max(-32768,
                state.coeff_state[src_idx] - transfer
            ))


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
                    # Blend instead of hard swap
                    state.coeff_state[dst_idx + i] = min(32767, max(-32768,
                        (tmp[dst_idx + i] * (Q15_ONE - strength) + tmp[src_idx + i] * strength) >> 15
                    ))


def lane_lowrank(state: VMState, instr: BackboneEntry, mask: int) -> None:
    """Apply low-rank operator."""
    op_id = instr.op_id
    strength = instr.strength_q15

    if op_id == 7:  # LowRankMorph
        # UV^T rank-2 motif
        # Compute mean across active nodes
        active_count = 0
        mean = 0
        for node in range(L_TOTAL_NODES):
            if mask & (1 << node):
                coeff_idx = node * NODE_LEN
                if coeff_idx < COEFF_LEN:
                    mean += state.coeff_state[coeff_idx]
                    active_count += 1

        if active_count > 0:
            mean //= active_count

            # Apply rank-1 update: x += strength * (mean - x)
            for node in range(L_TOTAL_NODES):
                if not (mask & (1 << node)):
                    continue

                coeff_idx = node * NODE_LEN
                for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
                    diff = mean - state.coeff_state[i]
                    state.coeff_state[i] = min(32767, max(-32768,
                        state.coeff_state[i] + ((diff * strength) >> 15)
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

            # Make adjacent nodes more similar
            idx_a = node * NODE_LEN
            idx_b = (node + 1) * NODE_LEN

            if idx_a < COEFF_LEN and idx_b < COEFF_LEN:
                avg = (state.coeff_state[idx_a] + state.coeff_state[idx_b]) >> 1
                state.coeff_state[idx_a] = min(32767, max(-32768,
                    state.coeff_state[idx_a] + ((avg - state.coeff_state[idx_a]) * strength) >> 15
                ))
                state.coeff_state[idx_b] = min(32767, max(-32768,
                    state.coeff_state[idx_b] + ((avg - state.coeff_state[idx_b]) * strength) >> 15
                ))


# =============================================================================
# LANE 7: TAIL
# =============================================================================

def lane_tail(state: VMState, cfg: DecodedPlan) -> None:
    """
    Apply tail operators.

    Tail enforces "legal/pleasant" bounds on coefficients.

    Args:
        state: VM state (modified in place)
        cfg: Decoded plan
    """
    for op_id in cfg.tail_ops:
        if op_id == 2:  # FineClamp
            _tail_fine_clamp(state, cfg.mask_u32)
        elif op_id == 11:  # NormalizeScale
            _tail_normalize_scale(state, cfg.mask_u32)
        elif op_id == 23:  # SheafProject
            _tail_sheaf_project(state, cfg.mask_u32)
        elif op_id == 19:  # EmergencySimplify
            _tail_emergency_simplify(state, cfg.mask_u32)


def _tail_fine_clamp(state: VMState, mask: int) -> None:
    """Per-band energy cap using integer-only math."""
    # Max energy per node: allow each coefficient to be at most Q15_ONE
    # Total squared energy = NODE_LEN * Q15_ONE^2
    max_energy_per_node = NODE_LEN * Q15_ONE * Q15_ONE

    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue

        # Compute node energy (sum of squared coefficients) - raw, not Q15
        coeff_idx = node * NODE_LEN
        energy = 0
        for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
            val = int(state.coeff_state[i])  # Convert to Python int to avoid overflow
            energy += val * val

        # If over budget, scale down using integer sqrt
        if energy > max_energy_per_node:
            # scale = sqrt(max_energy / energy) in Q15
            scale_q15 = isqrt_ratio_q15(max_energy_per_node, energy)
            for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
                val = int(state.coeff_state[i])
                state.coeff_state[i] = (val * scale_q15) >> 15


def _tail_normalize_scale(state: VMState, mask: int) -> None:
    """Per-band energy normalization using integer-only math."""
    target_energy = Q15_HALF * NODE_LEN

    for node in range(L_TOTAL_NODES):
        if not (mask & (1 << node)):
            continue

        # Compute node energy (use Python int to avoid numpy overflow)
        coeff_idx = node * NODE_LEN
        energy = 0
        for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
            val = int(state.coeff_state[i])  # Convert to Python int
            energy += (val * val) >> 15

        if energy > 0:
            # Scale to target energy using integer sqrt
            # scale = sqrt(target_energy / energy) in Q15
            scale = isqrt_ratio_q15(target_energy, energy)
            scale = max(Q15_HALF // 2, min(Q15_ONE * 2, scale))  # Limit scale range

            for i in range(coeff_idx, min(coeff_idx + NODE_LEN, COEFF_LEN)):
                # Use Python int to avoid numpy overflow
                val = int(state.coeff_state[i])
                state.coeff_state[i] = min(32767, max(-32768,
                    (val * scale) >> 15
                ))


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

def lane_idwpt_render(
    state: VMState,
    cfg: DecodedPlan,
    carrier_buf: np.ndarray,
) -> np.ndarray:
    """
    Render IDWPT texture from coefficient state.

    This is a placeholder - actual IDWPT requires the wavelet engine.
    For now, generates simple AM/FM modulated tone based on coefficients.

    Args:
        state: VM state
        cfg: Decoded plan
        carrier_buf: Carrier buffer (for excitation)

    Returns:
        texture_buf: int16 array of BLOCK_SAMPLES
    """
    texture_buf = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

    if not cfg.idwpt_enable:
        return texture_buf

    # Simple placeholder: modulate carrier by coefficient energy
    # Real implementation would use proper IDWPT synthesis

    # Compute overall coefficient energy (use Python int to avoid overflow)
    total_energy = 0
    for node in range(L_TOTAL_NODES):
        if cfg.mask_u32 & (1 << node):
            coeff_idx = node * NODE_LEN
            if coeff_idx < COEFF_LEN:
                total_energy += abs(int(state.coeff_state[coeff_idx]))

    # Normalize energy to modulation depth
    mod_depth = min(Q15_ONE, total_energy // 8)

    # Apply AM to carrier (use Python int to avoid overflow)
    for i in range(BLOCK_SAMPLES):
        # Simple AM modulation
        modulated = (int(carrier_buf[i]) * mod_depth) >> 15
        texture_buf[i] = modulated

    return texture_buf


# =============================================================================
# LANE 9: MIX + GAIN
# =============================================================================

def lane_mix_gain(
    cfg: DecodedPlan,
    carrier_buf: np.ndarray,
    texture_buf: np.ndarray,
) -> np.ndarray:
    """
    Apply eta blend and gain ramp to produce final output.

    Uses equal-power crossfade for eta and linear ramp for gain.
    All math is integer-only for bit-exact determinism.

    Args:
        cfg: Decoded plan
        carrier_buf: Carrier oscillator output
        texture_buf: IDWPT texture output

    Returns:
        output: int16 array of BLOCK_SAMPLES
    """
    output = np.zeros(BLOCK_SAMPLES, dtype=np.int16)

    for i in range(BLOCK_SAMPLES):
        # Interpolation factor for this sample (Q15 fixed-point)
        # alpha = i / BLOCK_SAMPLES, but we compute in Q15
        alpha_q15 = (i * Q15_ONE) // BLOCK_SAMPLES

        # Eta interpolation (dry/wet blend) - all integer
        eta_delta = cfg.eta_curr_q15 - cfg.eta_prev_q15
        eta = cfg.eta_prev_q15 + ((alpha_q15 * eta_delta) >> 15)

        # Equal-power crossfade weights using lookup tables
        # dry = cos(eta * pi/2), wet = sin(eta * pi/2)
        # eta is 0 to Q15_ONE, representing 0 to pi/2 angle
        dry_weight = cos_q15(eta)
        wet_weight = sin_q15(eta)

        # Mix carrier and texture (use Python ints to avoid overflow)
        carrier_val = int(carrier_buf[i])
        texture_val = int(texture_buf[i])
        mixed = (carrier_val * dry_weight + texture_val * wet_weight) >> 15

        # Gain interpolation - all integer
        gain_delta = cfg.gain_curr_q15 - cfg.gain_prev_q15
        gain = cfg.gain_prev_q15 + ((alpha_q15 * gain_delta) >> 15)

        # Apply gain
        sample = (mixed * gain) >> 15

        # Clamp to int16
        output[i] = max(-32768, min(32767, sample))

    return output


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
    'lane_mix_gain',
]
