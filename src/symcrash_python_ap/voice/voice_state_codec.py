"""Integer VM payloads: complete v1 by default, explicit frozen legacy v0.

The 644-byte v0 payload remains readable but omits modern state and resources.
The complete v1 schema is implemented separately in vm_binary_codec.
"""
import operator
import struct

import numpy as np

from ..vm.state import VMState, EnvelopeState, EnvelopePhase
from ..vm.coeff_lut_streaming import CCSState
from ..vm.tables import COEFF_LEN
from ..primitives import Q15_ONE, Q15_NEG_ONE


VMSTATE_VERSION = 2
LEGACY_VMSTATE_VERSION = 0
ENVELOPE_SIZE = 24
VMSTATE_HEADER_SIZE = 4
V0_COEFF_COUNT = 256
COEFF_ARRAY_SIZE = V0_COEFF_COUNT * 2
VMSTATE_SIZE = 644


def _bytes(data):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError('Binary snapshot requires a bytes-like value')
    return bytes(data)


def _integer(value, minimum, maximum, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer, not a boolean')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer') from exc
    if not minimum <= value <= maximum:
        raise ValueError(f'{name} is outside {minimum}..{maximum}')
    return value


def _pack(fmt, *values):
    try:
        if any(isinstance(v, (bool, np.bool_)) for v in values):
            raise ValueError('Boolean supplied for an integer payload field')
        return struct.pack(fmt, *(operator.index(v) for v in values))
    except (TypeError, struct.error) as exc:
        raise ValueError(f'Invalid integer snapshot field ({fmt})') from exc


def _flag(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return _integer(value, 0, 1, 'Snapshot flag')


def envelope_to_bytes(env: EnvelopeState) -> bytes:
    """Pack the legacy envelope subset: phase/count/level, A/D and sustain.

    Layout: BBHIhHIIhH. Reserved fields are zero. Release duration/level are
    absent from v0 and cannot be recovered from these bytes.
    """
    phase = _integer(env.phase, 0, 4, 'Envelope phase')
    return _pack('<BBHIhHIIhH', phase, 0, 0, env.phase_sample,
                 env.level_q15, 0, env.attack_samples, env.decay_samples,
                 env.sustain_level_q15, 0)


def envelope_from_bytes(data: bytes) -> EnvelopeState:
    data = _bytes(data)
    if len(data) != ENVELOPE_SIZE:
        raise ValueError('Legacy envelope payload must contain exactly 24 bytes')
    phase, pad1, pad2, count, level, pad3, attack, decay, sustain, pad4 = struct.unpack('<BBHIhHIIhH', data)
    if pad1 or pad2 or pad3 or pad4:
        raise ValueError('Nonzero reserved envelope field')
    return EnvelopeState(phase=EnvelopePhase(phase), phase_sample=count,
        level_q15=level, attack_samples=attack, decay_samples=decay,
        sustain_level_q15=sustain, release_samples=0)


def _vmstate_to_bytes_v0(state: VMState) -> bytes:
    """Encode the frozen v0 subset without truncation or narrowing overflow.

    Offsets: header 0, carrier 4, harmonics 24, envelope 56, DC 80,
    coefficients 84, PRNG 596, trees 600, latency 604, previous values 610,
    flags 614, seam 618, CCS 620. All integers are explicitly little-endian.
    """
    if COEFF_LEN != V0_COEFF_COUNT:
        raise ValueError('Legacy VM core requires the 256-coefficient runtime layout')
    voice_id = _integer(state.voice_id, 0, 0xFFFE, 'Voice ID')
    _integer(state.last_note, -1, 127, 'Last MIDI note')
    if len(state.harmonic_amps_q15) != 16:
        raise ValueError('Legacy harmonic vector must contain exactly 16 values')
    coeffs = np.asarray(state.coeff_state)
    if (coeffs.shape != (V0_COEFF_COUNT,) or coeffs.dtype.kind not in 'iu'
            or np.any(coeffs < Q15_NEG_ONE) or np.any(coeffs > Q15_ONE)):
        raise ValueError('Legacy coefficient vector must contain 256 signed-int16 values')
    parts = [
        _pack('<HH', LEGACY_VMSTATE_VERSION, voice_id),
        _pack('<IIIIbbh', state.phase, state.phase_inc, state.target_phase_inc,
              state.pitch_slew_rate, state.last_note, 0, state.last_cents),
        _pack('<16h', *state.harmonic_amps_q15),
        envelope_to_bytes(state.envelope),
        _pack('<hh', state.dc_block_x1, state.dc_block_y1),
        coeffs.astype('<i2', copy=False).tobytes(),
        _pack('<I', state.prng_state),
        _pack('<BBH', state.tree_a_id, state.tree_b_id, 0),
        _pack('<IBB', state.blocks_since_start, _flag(state.wet_ready), 0),
        _pack('<hh', state.eta_prev_q15, state.gain_prev_q15),
        _pack('<BBBB', _flag(state.mask_state_initialized), _flag(state.wet_seam_initialized),
              _flag(state.wet_ola_initialized), _flag(state.use_coeff_osc)),
        _pack('<h', state.wet_prev_end),
        # CCS phases are modulo-2^32 accumulators, including their upper half.
        _pack('<6I', state.ccs_state.wave_id, state.ccs_state.note, state.ccs_state.phase,
              state.ccs_state.phase_inc, state.ccs_state.target_phase_inc,
              state.ccs_state.phase_slew_rate),
    ]
    return b''.join(parts)


def _vmstate_from_bytes_v0(data: bytes) -> VMState:
    """Read exactly one v0 core payload, retaining legacy defaults for omissions."""
    data = _bytes(data)
    if COEFF_LEN != V0_COEFF_COUNT:
        raise ValueError('Legacy VM core requires the 256-coefficient runtime layout')
    if len(data) != VMSTATE_SIZE:
        raise ValueError('Legacy VM core payload must contain exactly 644 bytes')
    version, voice_id = struct.unpack_from('<HH', data, 0)
    if version != LEGACY_VMSTATE_VERSION:
        raise ValueError(f'Unsupported VMState version: {version}')
    _integer(voice_id, 0, 0xFFFE, 'Voice ID')
    phase, inc, target, slew, note, pad1, cents = struct.unpack_from('<IIIIbbh', data, 4)
    _integer(note, -1, 127, 'Last MIDI note')
    tree_a, tree_b, pad2 = struct.unpack_from('<BBH', data, 600)
    count, ready, pad3 = struct.unpack_from('<IBB', data, 604)
    flags = struct.unpack_from('<4B', data, 614)
    if pad1 or pad2 or pad3:
        raise ValueError('Nonzero reserved VM core field')
    if ready not in (0, 1) or any(flag not in (0, 1) for flag in flags):
        raise ValueError('Invalid boolean VM core field')
    dc_x, dc_y = struct.unpack_from('<hh', data, 80)
    eta, gain = struct.unpack_from('<hh', data, 610)
    wave, ccs_note, ccs_phase, ccs_inc, ccs_target, ccs_slew = struct.unpack_from('<6I', data, 620)
    return VMState(voice_id=voice_id, phase=phase, phase_inc=inc,
        target_phase_inc=target, pitch_slew_rate=slew, last_note=note, last_cents=cents,
        harmonic_amps_q15=list(struct.unpack_from('<16h', data, 24)),
        envelope=envelope_from_bytes(data[56:80]), dc_block_x1=dc_x, dc_block_y1=dc_y,
        coeff_state=np.frombuffer(data[84:596], dtype='<i2').astype(np.int16, copy=True),
        prng_state=struct.unpack_from('<I', data, 596)[0], tree_a_id=tree_a, tree_b_id=tree_b,
        blocks_since_start=count, wet_ready=bool(ready), eta_prev_q15=eta, gain_prev_q15=gain,
        mask_state_initialized=bool(flags[0]), wet_seam_initialized=bool(flags[1]),
        wet_ola_initialized=bool(flags[2]), use_coeff_osc=bool(flags[3]),
        wet_prev_end=struct.unpack_from('<h', data, 618)[0],
        ccs_state=CCSState(wave_id=wave, note=ccs_note, phase=ccs_phase,
            phase_inc=ccs_inc, target_phase_inc=ccs_target, phase_slew_rate=ccs_slew))


def vmstate_to_bytes(state: VMState, *, version: int = VMSTATE_VERSION) -> bytes:
    """Encode complete v2 state, or an explicitly requested older format."""
    version = _integer(version, 0, VMSTATE_VERSION, 'VMState version')
    if version == LEGACY_VMSTATE_VERSION:
        return _vmstate_to_bytes_v0(state)
    from .vm_binary_codec import encode
    return encode(state, version=version)


def vmstate_from_bytes(data: bytes) -> VMState:
    """Dispatch by payload version; require exact consumption of either format."""
    data = _bytes(data)
    if len(data) < VMSTATE_HEADER_SIZE:
        raise ValueError('Truncated VM core header')
    version = struct.unpack_from('<H', data)[0]
    if version == LEGACY_VMSTATE_VERSION:
        return _vmstate_from_bytes_v0(data)
    if version in (1, VMSTATE_VERSION):
        from .vm_binary_codec import decode
        return decode(data)
    raise ValueError(f'Unsupported VMState version: {version}')


def get_vmstate_size() -> int:
    """Size of the frozen v0 core subset, independent of current VM layout."""
    return VMSTATE_SIZE


__all__ = ['vmstate_to_bytes', 'vmstate_from_bytes', 'get_vmstate_size',
           'envelope_to_bytes', 'envelope_from_bytes', 'VMSTATE_VERSION']
