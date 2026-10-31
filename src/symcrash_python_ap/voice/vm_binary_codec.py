"""Integer-only VM CORE v2 payload, with frozen v1 compatibility.

The schema below freezes record IDs, field order, types and array dimensions.
No field names, Python type names, JSON, floats or executable objects are stored.
Every declared runtime field must be registered; additions require a new schema.
See WSE-IMPL-VM-BINARY-v1 for the byte grammar and replay boundary.
"""
from dataclasses import fields
from functools import lru_cache
import math
import operator
import struct

import numpy as np

from ..vm.tables import VM_K, COEFF_LEN
from ..primitives import Q15_NEG_ONE, Q15_ONE, Q15_SCALE, Q16_ONE


VERSION = 2
HEADER = struct.Struct('<4H')  # version, voice ID, VM_K, coefficient count
I64 = ('int', -(1 << 63), (1 << 63) - 1)
BOOL = ('bool',)
TEXT = ('text',)
PHASE = ('phase',)


def integer(low, high):
    return ('int', low, high)


def optional(spec):
    return ('optional', spec)


def array(dtype, *shape):
    return ('array', dtype, shape)


def sequence(spec, count=None, *, maximum=65535, kind=list):
    return ('sequence', spec, count, maximum, kind)


def record(name):
    return ('record', name)


@lru_cache(maxsize=2)
def _schema(version=VERSION):
    from ..vm.state import VMState, EnvelopeState, ExecutorLatch, CMMMemory
    from ..vm.coeff_lut_streaming import CCSState
    from ..vm.wire_events import WireExecutionState
    from ..wire_frame import WireEvent
    from ..vm.noise_exciter import NoiseExciterState
    from ..vm.harmonic_bank import ModeBState, HarmonicWeights
    from ..vm.fm6_state import FM6State, FMOperator
    from ..vm.fm6_morph_engine import FM6MorphState
    from ..vm.sample_source import SampleOscState
    from ..vm.wavetable_engine import WavetableState, WavetableAtlas
    from ..vm.granular_engine import GranularState, GrainState
    from ..vm.mode_switch import ModeSwitchState

    result = {}

    def add(code, cls, names, **overrides):
        names = names.split()
        omitted = {'mode_switch'} if version == 1 and cls is VMState else set()
        if set(names) != {f.name for f in fields(cls)} - omitted or not set(overrides) <= set(names):
            raise ValueError(f'Runtime fields have changed: {cls.__name__}; binary schema update required')
        result[cls.__name__] = (code, cls, tuple((n, overrides.get(n, I64)) for n in names))

    add(1, VMState, '''voice_id phase phase_inc target_phase_inc pitch_slew_rate last_note
        last_cents harmonic_amps_q15 envelope dc_block_x1 dc_block_y1 coeff_state
        tail_previous_coeffs wire_state prng_state tree_a_id tree_b_id blocks_since_start
        wet_ready eta_prev_q15 gain_prev_q15 mask_state_q15 mask_state_initialized
        wet_prev_end wet_seam_initialized wet_ola_tail wet_ola_initialized ccs_state
        use_coeff_osc _mode_b_state osc_mode prev_fm_mod_buffer _sample_state _fm6_state
        _fm6_morph_state _wavetable_state _granular_state _noise_state executor_latch cmmm_memory'''
        + (' mode_switch' if version == 2 else ''),
        **({'mode_switch': record('ModeSwitchState')} if version == 2 else {}),
        voice_id=integer(0, 65534), last_note=integer(-1, 127),
        harmonic_amps_q15=sequence(integer(Q15_NEG_ONE, Q15_ONE), 16),
        envelope=record('EnvelopeState'), coeff_state=array('i2', COEFF_LEN),
        tail_previous_coeffs=array('i2', COEFF_LEN), wire_state=record('WireExecutionState'),
        wet_ready=BOOL, mask_state_q15=array('i2', 64), mask_state_initialized=BOOL,
        wet_seam_initialized=BOOL, wet_ola_tail=array('i2', 8), wet_ola_initialized=BOOL,
        ccs_state=record('CCSState'), use_coeff_osc=BOOL,
        _mode_b_state=optional(record('ModeBState')), osc_mode=integer(0, 5),
        prev_fm_mod_buffer=array('i2', 8), _sample_state=optional(record('SampleOscState')),
        _fm6_state=optional(record('FM6State')), _fm6_morph_state=optional(record('FM6MorphState')),
        _wavetable_state=optional(record('WavetableState')), _granular_state=optional(record('GranularState')),
        _noise_state=optional(record('NoiseExciterState')), executor_latch=record('ExecutorLatch'),
        cmmm_memory=record('CMMMemory'))
    add(2, EnvelopeState, '''phase phase_sample level_q15 attack_samples decay_samples
        sustain_level_q15 release_samples''', phase=PHASE)
    add(3, WireExecutionState, '''program_id frame_seq forced_on forced_off tree_held
        tree_blend_q15 carrier_role carrier_model pulse_width_q15 events''', tree_held=BOOL,
        events=sequence(record('WireEvent'), maximum=8, kind=tuple))
    add(4, WireEvent, '''lane variant strength_q15 target_kind target_id pattern_id param0 param1 param2''')
    add(5, CCSState, 'phase phase_inc target_phase_inc phase_slew_rate note wave_id')
    add(6, ExecutorLatch, '''node_mask_u32 tail_mode traction_q15 symbol_mode mode_dwell_blocks
        phrase_quantize_pending timbral_mode timbral_target timbral_ramp_blocks''',
        phrase_quantize_pending=BOOL)
    add(7, CMMMemory, '''tension_q15 fatigue_q15 harmonic_memory_q15 blocks_since_transition
        TENSION_DECAY FATIGUE_DECAY HARMONIC_DECAY''')
    add(8, ModeBState, '''phase phase_inc target_phase_inc phase_slew_rate note cents H_active
        H_bandlimit weights last_custom_harmonics pwm_width_q15 pwm_enabled fm_depth_q15
        fm_mod_buffer sync_enabled sync_master_phase sync_master_inc unison_voices
        unison_spread_cents phase_offset_q32''', weights=record('HarmonicWeights'),
        last_custom_harmonics=optional(('pair', sequence(I64), I64)), pwm_enabled=BOOL,
        fm_mod_buffer=optional(array('i2', None)), sync_enabled=BOOL)
    add(9, HarmonicWeights, 'dc_q15 a_h_q15 b_h_q15 H_active',
        a_h_q15=array('i2', None), b_h_q15=array('i2', None))
    add(10, FM6State, '''operators algorithm_id op_outputs feedback_buffer master_level_q15
        master_pitch_bend mod_index_q16 post_filter_enabled post_filter_cutoff_q15 _post_filter_state''',
        operators=sequence(record('FMOperator'), 6), op_outputs=array('i4', 6, VM_K),
        feedback_buffer=array('i4', VM_K), post_filter_enabled=BOOL)
    add(11, FMOperator, '''op_id ratio_q16 detune_cents level_q15 velocity_sens feedback_q15
        phase env_rates env_levels env_phase env_value_q15''',
        env_rates=sequence(I64, 4), env_levels=sequence(I64, 4), env_phase=integer(0, 3))
    add(12, FM6MorphState, '''operators op_outputs feedback_buffer master_level_q15 mod_index_q16
        topo_a_id topo_b_id blend_q15 dx7_compat_id _cached_blend _cached_matrix
        _cached_carrier_mask _cached_carrier_fades''', operators=sequence(record('FMOperator'), 6),
        op_outputs=array('i4', 6, VM_K), feedback_buffer=array('i2', VM_K),
        _cached_matrix=optional(array('i2', 6, 6)), _cached_carrier_fades=optional(array('i2', 6)))
    add(13, SampleOscState, '''buffer coeff_buffer use_coeff_buffer position_q16 block_position_q16
        rate_q16 loop_start loop_end loop_enabled is_playing is_finished root_note''',
        buffer=optional(array('i2', None)), coeff_buffer=optional(array('i2', None, 64)),
        use_coeff_buffer=BOOL, loop_enabled=BOOL, is_playing=BOOL, is_finished=BOOL,
        root_note=optional(integer(0, 127)))
    add(14, WavetableState, '''atlas phase_q32 phase_inc_q32 position_q16 frame_q16 frame_rate_q16
        position_mod_q15 frame_mod_q15 is_playing''', atlas=optional(record('WavetableAtlas')), is_playing=BOOL)
    add(15, WavetableAtlas, 'data num_positions num_frames frame_size name base_note',
        data=array('i2', None, None, None), name=TEXT)
    add(16, GranularState, '''grains coeff_buffer num_source_blocks scan_position scan_position_q16
        scan_rate_q16 density_q15 grain_length grain_rate_q16 spray_q15 mix_gain_q15 prng_state _output_512''',
        grains=sequence(record('GrainState'), 8), coeff_buffer=optional(array('i2', None, 64)),
        _output_512=array('i2', COEFF_LEN))
    add(17, GrainState, '''start_block current_block_q16 length_blocks rate_q16 pan_q15 attack_blocks
        release_blocks amplitude_q15 blocks_played is_active''', is_active=BOOL)
    add(18, NoiseExciterState, 'pink_b0 pink_b1 pink_b2 brown_acc')
    if version == 2:
        add(19, ModeSwitchState, '''initialized mix_b_q16 start_mix_b_q16 remaining_blocks
            b_harmonic_gains b_gain_bound b_fm_source''', initialized=BOOL,
            mix_b_q16=integer(0, Q16_ONE), start_mix_b_q16=integer(0, Q16_ONE),
            remaining_blocks=integer(0, 3), b_fm_source=integer(0, 2),
            b_gain_bound=integer(Q15_SCALE, 4 * Q15_SCALE),
            b_harmonic_gains=optional(sequence(integer(0, 4 * Q15_SCALE), 32, kind=tuple)))
    return result


def _int(value, low, high, path):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{path}: expected integer, got boolean')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{path}: expected integer') from exc
    if not low <= value <= high:
        raise ValueError(f'{path}: integer out of range')
    return value


def _shape(shape, expected, path):
    if len(shape) != len(expected) or any(n < 0 or n > 0xFFFFFFFF or
            fixed is not None and n != fixed for n, fixed in zip(shape, expected)):
        raise ValueError(f'{path}: invalid array shape {shape}, expected {expected}')


def _count(n, spec, path):
    if n > spec[3] or spec[2] is not None and n != spec[2]:
        raise ValueError(f'{path}: invalid sequence length')


def _consistent(value, path):
    """Check dimensions used together by the renderer, without rebuilding caches."""
    name = type(value).__name__
    if name == 'HarmonicWeights':
        if value.a_h_q15.shape != value.b_h_q15.shape or not 0 <= value.H_active <= len(value.a_h_q15):
            raise ValueError(f'{path}: inconsistent harmonic weights')
    elif name == 'ModeBState':
        if value.fm_mod_buffer is not None and len(value.fm_mod_buffer) < VM_K:
            raise ValueError(f'{path}: insufficient FM modulation samples')
    elif name == 'WavetableAtlas':
        if value.data.shape != (value.num_positions, value.num_frames, value.frame_size) or min(value.data.shape) < 1:
            raise ValueError(f'{path}: inconsistent atlas dimensions')
    elif name == 'GranularState':
        count = 0 if value.coeff_buffer is None else len(value.coeff_buffer)
        if value.num_source_blocks != count:
            raise ValueError(f'{path}: inconsistent granular source length')
    elif name == 'ModeSwitchState':
        value.validate()
    elif name == 'VMState':
        value.mode_switch.validate(value.osc_mode)


def _write(out, spec, value, path, version=VERSION):
    kind = spec[0]
    if kind == 'int':
        out.extend(struct.pack('<q', _int(value, spec[1], spec[2], path)))
    elif kind == 'bool':
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{path}: expected boolean')
        out.append(int(value))
    elif kind == 'phase':
        from ..vm.state import EnvelopePhase
        out.append(EnvelopePhase(_int(value, 0, 4, path)).value)
    elif kind == 'optional':
        out.append(int(value is not None))
        if value is not None:
            _write(out, spec[1], value, path, version)
    elif kind == 'text':
        if type(value) is not str:
            raise ValueError(f'{path}: expected UTF-8 metadata')
        try:
            data = value.encode('utf-8')
        except UnicodeError as exc:
            raise ValueError(f'{path}: invalid UTF-8 metadata') from exc
        out.extend(struct.pack('<I', _int(len(data), 0, 0xFFFFFFFF, path)))
        out.extend(data)
    elif kind == 'array':
        dtype = np.dtype(spec[1])
        if not isinstance(value, np.ndarray) or value.dtype.newbyteorder('=') != dtype:
            raise ValueError(f'{path}: expected {dtype} array')
        _shape(value.shape, spec[2], path)
        out.append(value.ndim)
        out.extend(struct.pack('<' + 'I' * value.ndim, *value.shape))
        out.extend(value.astype(dtype.newbyteorder('<'), copy=False).tobytes(order='C'))
    elif kind == 'sequence':
        if type(value) is not spec[4]:
            raise ValueError(f'{path}: expected {spec[4].__name__}')
        _count(len(value), spec, path)
        out.extend(struct.pack('<I', len(value)))
        for i, item in enumerate(value):
            _write(out, spec[1], item, f'{path}[{i}]', version)
    elif kind == 'pair':
        if type(value) is not tuple or len(value) != 2:
            raise ValueError(f'{path}: expected harmonic cache pair')
        for i in range(2):
            _write(out, spec[i + 1], value[i], f'{path}[{i}]', version)
    elif kind == 'record':
        code, cls, members = _schema(version)[spec[1]]
        omitted = {'mode_switch'} if version == 1 and spec[1] == 'VMState' else set()
        if type(value) is not cls or set(vars(value)) - omitted != {name for name, _ in members}:
            raise ValueError(f'{path}: unregistered runtime fields or incorrect record type')
        out.extend(struct.pack('<HH', code, len(members)))
        for name, member in members:
            _write(out, member, getattr(value, name), path + '.' + name, version)
        _consistent(value, path)
    else:
        raise AssertionError(f'Unknown internal binary schema kind: {kind}')


class _Reader:
    def __init__(self, data):
        self.data = memoryview(data)
        self.offset = 0
        self.version = VERSION

    def take(self, size):
        if size > len(self.data) - self.offset:
            raise ValueError('Truncated VM binary payload')
        start = self.offset
        self.offset += size
        return self.data[start:self.offset]

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def read(self, spec, path):
        kind = spec[0]
        if kind == 'int':
            return _int(self.unpack('<q')[0], spec[1], spec[2], path)
        if kind in ('bool', 'optional'):
            flag = self.unpack('<B')[0]
            if flag > 1:
                raise ValueError(f'{path}: invalid boolean/presence flag')
            return bool(flag) if kind == 'bool' else self.read(spec[1], path) if flag else None
        if kind == 'phase':
            from ..vm.state import EnvelopePhase
            return EnvelopePhase(self.unpack('<B')[0])
        if kind == 'text':
            try:
                return bytes(self.take(self.unpack('<I')[0])).decode('utf-8')
            except UnicodeError as exc:
                raise ValueError(f'{path}: invalid UTF-8 metadata') from exc
        if kind == 'array':
            rank = self.unpack('<B')[0]
            if rank != len(spec[2]):
                raise ValueError(f'{path}: invalid array rank')
            shape = self.unpack('<' + 'I' * rank)
            _shape(shape, spec[2], path)
            dtype = np.dtype(spec[1])
            # Check the entire byte extent BEFORE constructing or copying any array.
            raw = self.take(math.prod(shape) * dtype.itemsize)
            return np.frombuffer(raw, dtype=dtype.newbyteorder('<')).reshape(shape).astype(dtype, copy=True)
        if kind == 'sequence':
            count = self.unpack('<I')[0]
            _count(count, spec, path)
            if count > len(self.data) - self.offset:
                raise ValueError(f'{path}: truncated sequence')
            return spec[4](self.read(spec[1], f'{path}[{i}]') for i in range(count))
        if kind == 'pair':
            return tuple(self.read(spec[i + 1], f'{path}[{i}]') for i in range(2))
        if kind == 'record':
            code, cls, members = _schema(self.version)[spec[1]]
            if self.unpack('<HH') != (code, len(members)):
                raise ValueError(f'{path}: incorrect record ID or field count')
            # Only registered classes can be instantiated. All fields are supplied;
            # constructors must not normalize saved resources or recompute caches.
            values = {name: self.read(member, path + '.' + name) for name, member in members}
            value = object.__new__(cls)
            for name, item in values.items():
                object.__setattr__(value, name, item)
            if self.version == 1 and spec[1] == 'VMState':
                from ..vm.mode_switch import ModeSwitchState
                value.mode_switch = ModeSwitchState()
            _consistent(value, path)
            return value
        raise AssertionError(f'Unknown internal binary schema kind: {kind}')


def encode(state, *, version=VERSION):
    if version not in (1, 2):
        raise ValueError('Unsupported VM binary version')
    voice = _int(state.voice_id, 0, 65534, 'Voice ID')
    if version == 1 and state.mode_switch.initialized:
        raise ValueError('CORE v1 cannot preserve initialized mode-switch state; use v2')
    out = bytearray(HEADER.pack(version, voice, VM_K, COEFF_LEN))
    _write(out, record('VMState'), state, 'VMState', version)
    return bytes(out)


def decode(data):
    reader = _Reader(data)
    version, voice, k, coeffs = reader.unpack(HEADER.format)
    if version not in (1, 2) or k != VM_K or coeffs != COEFF_LEN:
        raise ValueError('Unsupported VM binary version or runtime layout')
    reader.version = version
    state = reader.read(record('VMState'), 'VMState')
    if reader.offset != len(data):
        raise ValueError('Trailing VM binary payload data')
    if state.voice_id != voice:
        raise ValueError('VM payload header and record voice IDs disagree')
    # Binary records bypass dataclass constructors to preserve saved state.
    # Prepare shared immutable assets after validation, before adopting a voice.
    from ..vm.resources import prepare_oscillator_resources
    prepare_oscillator_resources()
    return state
