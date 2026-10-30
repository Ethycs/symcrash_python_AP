"""Resolve the complete Mode B spectrum without borrowing oscillator phase.

The bounded cache contains immutable derived weights, never per-voice time.
Reconstruction from a frame therefore needs no new checkpoint fields.
"""
from functools import lru_cache
import math

from ..primitives import Q15_ONE, Q15_NEG_ONE, Q15_HALF


H_MAX = 32


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')


def validate_spectrum(frame):
    """Check native spectral inputs before any VM musical state changes."""
    request = getattr(frame, 'H_request', 16)
    _integer(request, 0, 255, 'H_request')
    _integer(getattr(frame, 'waveform_type', 0), 0, 5, 'Mode B waveform')
    _integer(getattr(frame, 'oeis_sequence_id', -1), -1, 7, 'OEIS sequence')
    rolloff = getattr(frame, 'oeis_rolloff_db', -3.0)
    if type(rolloff) not in (int, float) or not math.isfinite(rolloff) or not -120 <= rolloff <= 0:
        raise ValueError('OEIS rolloff must be finite in -120..0 dB/octave')
    _integer(getattr(frame, 'partial_phase_spread_q15', 0), 0, Q15_ONE, 'Partial phase spread')
    width = getattr(frame, 'pwm_width_q15', Q15_HALF)
    # Legacy optional width explicitly defaults to half duty.
    _integer(Q15_HALF if width is None else width, 0, Q15_ONE, 'PWM width')
    if type(getattr(frame, 'pwm_enabled', False)) is not bool:
        raise ValueError('PWM enabled must be boolean')
    partials = getattr(frame, 'partial_amps_q15', None)
    if partials is not None:
        if type(partials) not in (list, tuple) or len(partials) > 255:
            raise ValueError('Custom partials must contain at most 255 Q15 amplitudes')
        for value in partials:
            _integer(value, Q15_NEG_ONE, Q15_ONE, 'Custom partial amplitude')


def harmonic_limit(phase_inc):
    """0.9 Nyquist margin, computed from the actual Q32 pitch increment."""
    return min(H_MAX, (9 * (1 << 31)) // (10 * phase_inc)) if phase_inc else H_MAX


@lru_cache(maxsize=512)
def _weights(kind, count, values, spread):
    from .harmonic_bank import HarmonicWeights
    weights = HarmonicWeights()
    if count:
        if kind == 'oeis':
            weights.set_oeis(values[0], count, values[1])
        elif kind == 'custom':
            weights.set_custom(values[:count], spread)
        elif kind == 'pwm':
            weights.set_pwm(values[0], count)
        else:
            weights.set_from_waveform_type(values[0], count, values[1])
    return (int(weights.dc_q15), tuple(int(v) for v in weights.a_h_q15),
            tuple(int(v) for v in weights.b_h_q15))


def apply_spectrum(state, params):
    """Resolve OEIS > custom > enabled PWM > named waveform every block."""
    state.H_bandlimit = harmonic_limit(params.phase_inc)
    count = min(params.H_request, state.H_bandlimit,
                len(state.weights.a_h_q15), len(state.weights.b_h_q15))
    state.pwm_width_q15 = params.pwm_width_q15
    state.pwm_enabled = False
    state.last_custom_harmonics = None
    if params.oeis_sequence_id >= 0:
        kind, values, spread = 'oeis', (params.oeis_sequence_id, params.oeis_rolloff_db), 0
    elif params.custom_harmonics:
        count = min(count, len(params.custom_harmonics))
        kind, values, spread = 'custom', tuple(params.custom_harmonics[:H_MAX]), params.phase_spread_q15
        state.last_custom_harmonics = (list(params.custom_harmonics), spread)
    elif params.pwm_enabled:
        kind, values, spread = 'pwm', (params.pwm_width_q15,), 0
        state.pwm_enabled = True
    else:
        kind, values, spread = 'wave', (params.waveform_type, params.pwm_width_q15), 0
        if params.waveform_type == 0:
            count = min(count, 1)
    dc, a, b = _weights(kind, count, values, spread)
    state.weights.dc_q15 = dc
    state.weights.a_h_q15.fill(0)
    state.weights.b_h_q15.fill(0)
    capacity = min(H_MAX, len(state.weights.a_h_q15), len(state.weights.b_h_q15))
    state.weights.a_h_q15[:capacity] = a[:capacity]
    state.weights.b_h_q15[:capacity] = b[:capacity]
    state.H_active = state.weights.H_active = count
