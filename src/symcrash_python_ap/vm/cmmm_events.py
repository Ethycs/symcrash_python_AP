"""Execute semantic CMMM events without reinterpreting frozen wire lane IDs.

The six authored bands and 32 node mask address the operator library's
256-coefficient domain. Coefficients outside that domain are untouched.
All events are instantaneous; VoiceEngine owns repetition and arc duration.
"""
from ..fixed import RSHIFT_RNE
from ..frames import CMMMEventPayload
from ..operator_library import BAND_SLICES
from ..primitives import Q15_ONE, Q15_SCALE, clamp_q15
from .decode import BackboneEntry
from .state import VMState, prng_step
from .tables import NODE_LEN, L_TOTAL_NODES


def _band_nodes(band: int) -> int:
    if not 0 <= band < len(BAND_SLICES):
        raise ValueError(f'CMMM band out of range: {band}')
    start, end = BAND_SLICES[band]
    return ((1 << (end // NODE_LEN)) - 1) ^ ((1 << (start // NODE_LEN)) - 1)


def _selected_nodes(bands: int) -> int:
    return sum(_band_nodes(b) for b in range(len(BAND_SLICES)) if bands & (1 << b))


def apply_cmmm_events(state: VMState, events: CMMMEventPayload, node_mask: int) -> None:
    """Apply E0..E7 in order; only an executing noise lane advances the PRNG."""
    from .lanes import dispatch_coeff_op, _inject_cusp

    mask = node_mask & 0xFFFFFFFF
    active = events.active_mask
    if active & 1:  # Impulse at the first coefficient of the selected band.
        target_mask = _band_nodes(events.impulse_band)
        idx = BAND_SLICES[events.impulse_band][0]
        if mask & target_mask & (1 << (idx // NODE_LEN)):
            amount = max(0, min(Q15_ONE, events.impulse_strength_q15))
            state.coeff_state[idx] = clamp_q15(int(state.coeff_state[idx]) + amount)

    if active & 2:  # Noise over the selected authored band.
        target_mask = mask & _band_nodes(events.noise_band)
        amount = max(0, min(Q15_ONE, events.noise_amount_q15))
        if amount:
            start, end = BAND_SLICES[events.noise_band]
            for idx in range(start, min(end, len(state.coeff_state))):
                if target_mask & (1 << (idx // NODE_LEN)):
                    noise = (prng_step(state) & 0xFFFF) - Q15_SCALE
                    value = int(state.coeff_state[idx]) + RSHIFT_RNE(noise * amount, 15)
                    state.coeff_state[idx] = clamp_q15(value)

    if active & 4:  # PressureRamp, not wire lane E3's additive GlobalPulse.
        dispatch_coeff_op(state, BackboneEntry(15, max(0, min(Q15_ONE, events.pulse_pressure_q15))), mask)

    if active & 8:  # Gate bitmap: a set band bit leaves that band open.
        closed = mask & ~_selected_nodes(events.gate_mask)
        for node in range(L_TOTAL_NODES):
            if closed & (1 << node):
                start = node * NODE_LEN
                state.coeff_state[start:start + NODE_LEN] = 0

    if active & 16:
        _inject_cusp(state, mask, max(0, min(Q15_ONE, events.cusp_amount_q15)))

    if active & 32:  # Apply SwallowtailMask only to the selected bands.
        dispatch_coeff_op(state, BackboneEntry(22, Q15_ONE), mask & _selected_nodes(events.swallowtail_bands))

    if active & 64:
        dispatch_coeff_op(state, BackboneEntry(20, max(0, min(Q15_ONE, events.fold_pressure_q15))), mask)

    if active & 128:  # Project incompatible adjacent node boundaries.
        threshold = max(0, min(Q15_ONE, events.sheaf_threshold_q15))
        for node in range(L_TOTAL_NODES - 1):
            if mask & (1 << node) and mask & (1 << (node + 1)):
                left, right = (node + 1) * NODE_LEN - 1, (node + 1) * NODE_LEN
                a, b = int(state.coeff_state[left]), int(state.coeff_state[right])
                if abs(a - b) > threshold:
                    state.coeff_state[left] = state.coeff_state[right] = RSHIFT_RNE(a + b, 1)
