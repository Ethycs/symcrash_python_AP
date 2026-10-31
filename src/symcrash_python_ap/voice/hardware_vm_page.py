"""Lossless VM boundary image for the existing 2 KiB opaque voice cache.

The first KiB holds a fixed control layout; the second holds every int16
coefficient slot. Complete optional engine records live in explicitly bounded
external storage, including their resources. No implicit paging, truncation,
quantization or resource sharing is performed. This preparation/inspection
codec runs off the audio path; engine RTL and global allocation are separate.
"""
from dataclasses import dataclass
from functools import lru_cache
import struct

import numpy as np

from ..budgets.hardware_memory_map import MemorySpan, check_memory_map
from ..budgets.hardware_resources import EXTERNAL_RAM_BYTES
from ..primitives import INT16_MIN, INT16_MAX
from ..vm.state import VMState, EnvelopePhase
from ..vm.tables import VM_K
from . import vm_binary_codec as binary


VERSION = 1
PAGE_BYTES = 2048
COEFF_OFFSET = 1024
COEFF_WORD_OFFSET = COEFF_OFFSET // 16
HEADER = struct.Struct('<4sHHIHHII8s')
EXTERNAL_HEADER = struct.Struct('<4sHHII')
REFERENCE = struct.Struct('<II')
FORMAT_LIMITS = {'B': (0, 255), 'h': (INT16_MIN, INT16_MAX), 'I': (0, 0xffffffff),
                 'i': (-(1 << 31), (1 << 31)-1), 'q': (-(1 << 63), (1 << 63)-1)}


def _integer(value, low, high, name):
    return binary._int(value, low, high, name)


def _bytes(value, name):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f'{name}: expected bytes-like data')
    return bytes(value)


def _align(size):
    return (size + 15) & ~15


def _aligned(offset, alignment):
    return (offset + alignment - 1) // alignment * alignment


def _alignment(spec):
    if isinstance(spec, str):
        return 1 if spec in ('bool', 'phase') else struct.calcsize('<' + spec)
    return 8 if spec[0] == 'external' else 16


def _identity(voice_id, context_tag):
    return (_integer(voice_id, 0, 65534, 'Voice ID'),
            _integer(context_tag, 0, 0xffffffff, 'Context tag'))


def _regions(page_address, external_start, external_bytes):
    for name, value in (('Page address', page_address), ('External address', external_start),
                        ('External byte count', external_bytes)):
        _integer(value, 0, EXTERNAL_RAM_BYTES, name)
    if page_address % PAGE_BYTES or external_start % 16 or external_bytes % 16:
        raise ValueError('Page/extension addresses and capacity must be aligned')
    spans = [MemorySpan('voice page', page_address, PAGE_BYTES)]
    if external_bytes:
        spans.append(MemorySpan('voice extension', external_start, external_bytes))
    if check_memory_map(spans, EXTERNAL_RAM_BYTES):
        raise ValueError('Voice page/extension overlaps or exceeds external RAM')


# A scalar has only a storage width, not a new arithmetic rule. Q32 phases and
# PRNG use their normative u32 domain; Q15 vectors retain int16. Unspecified
# counters/selectors retain the complete CORE v2 i64 domain. No silent narrowing.
@lru_cache(maxsize=1)
def _schema():
    base = binary._schema(2)
    specs = {
        'VMState': dict(
            phase='I', phase_inc='I', target_phase_inc='I', pitch_slew_rate='I',
            last_note='h', last_cents='q', harmonic_amps_q15=('list', 'h', 16),
            envelope=('record', 'EnvelopeState'), dc_block_x1='i', dc_block_y1='i',
            wire_state=('record', 'WireExecutionState'), prng_state='I', tree_a_id='q', tree_b_id='q',
            blocks_since_start='q', wet_ready='bool', eta_prev_q15='h', gain_prev_q15='h',
            mask_state_q15=('array', 64), mask_state_initialized='bool',
            wet_prev_end='h', wet_seam_initialized='bool', wet_ola_tail=('array', 8),
            wet_ola_initialized='bool', ccs_state=('record', 'CCSState'), use_coeff_osc='bool',
            _mode_b_state=('external', 1), osc_mode='B', prev_fm_mod_buffer=('array', 8),
            _sample_state=('external', 2), _fm6_state=('external', 3),
            _fm6_morph_state=('external', 4), _wavetable_state=('external', 5),
            _granular_state=('external', 6), _noise_state=('external', 7),
            executor_latch=('record', 'ExecutorLatch'), cmmm_memory=('record', 'CMMMemory'),
            mode_switch=('record', 'ModeSwitchState')),
        'EnvelopeState': dict(phase='phase', phase_sample='q', level_q15='h',
            attack_samples='q', decay_samples='q', sustain_level_q15='h', release_samples='q'),
        'WireExecutionState': dict(program_id='q', frame_seq='q', forced_on='I', forced_off='I',
            tree_held='bool', tree_blend_q15='h', carrier_role='q', carrier_model='q',
            pulse_width_q15='q', events=('external', 8)),
        'CCSState': dict(phase='I', phase_inc='I', target_phase_inc='I', phase_slew_rate='I', note='q', wave_id='q'),
        'ExecutorLatch': dict(node_mask_u32='I', tail_mode='B', traction_q15='h', symbol_mode='B',
            mode_dwell_blocks='q', phrase_quantize_pending='bool', timbral_mode='B',
            timbral_target='B', timbral_ramp_blocks='q'),
        'CMMMemory': dict(tension_q15='h', fatigue_q15='h', harmonic_memory_q15='h',
            blocks_since_transition='q', TENSION_DECAY='q', FATIGUE_DECAY='q', HARMONIC_DECAY='q'),
        'ModeSwitchState': dict(initialized='bool', mix_b_q16='I', start_mix_b_q16='I',
            remaining_blocks='B', b_harmonic_gains=('gains',), b_gain_bound='I', b_fm_source='B'),
    }
    result = {}
    for name, overrides in specs.items():
        _, cls, original = base[name]
        skipped = {'voice_id', 'coeff_state', 'tail_previous_coeffs'} if name == 'VMState' else set()
        if set(overrides) != {field for field, _ in original} - skipped:
            raise ValueError(f'Hardware page schema must cover every {name} field')
        result[name] = (cls, tuple((field, overrides[field], spec) for field, spec in original if field not in skipped))
    return result


def _size(spec):
    if isinstance(spec, str):
        return 1 if spec in ('bool', 'phase') else struct.calcsize('<' + spec)
    kind = spec[0]
    if kind == 'array':
        return spec[1] * 2
    if kind == 'list':
        return spec[2] * struct.calcsize('<' + spec[1])
    if kind == 'external':
        return REFERENCE.size
    if kind == 'gains':
        return 16 + 32 * 4
    offset = 0
    for _, member, _ in _schema()[spec[1]][1]:
        offset = _aligned(offset, _alignment(member)) + _size(member)
    return offset


def page_layout():
    """Return all fixed byte offsets, including subrecords, for HDL consumers."""
    rows = []
    def visit(name, path, offset):
        for field, spec, _ in _schema()[name][1]:
            member_path = f'{path}.{field}'
            offset = _aligned(offset, _alignment(spec))
            size = _size(spec)
            rows.append(dict(field=member_path, offset=offset, bytes=size, storage=spec))
            if isinstance(spec, tuple) and spec[0] == 'record':
                visit(spec[1], member_path, offset)
            offset += size
        return offset
    end = visit('VMState', 'VMState', HEADER.size)
    if end > COEFF_OFFSET:
        raise ValueError('Hardware control layout exceeds its fixed 1 KiB region')
    return dict(version=VERSION, page_bytes=PAGE_BYTES, header_bytes=HEADER.size,
                metadata_end=end, coefficient_offset=COEFF_OFFSET, fields=rows)


@dataclass(frozen=True)
class ExternalRecord:
    address: int
    field_id: int
    data: bytes  # Includes owner header and zero padding to 16-byte DMA words.


@dataclass(frozen=True)
class PackedVMPage:
    page_address: int
    page: bytes
    external: tuple[ExternalRecord, ...]

    @property
    def external_bytes(self):
        return sum(len(item.data) for item in self.external)

    @property
    def total_bytes(self):
        return PAGE_BYTES + self.external_bytes


class _Writer:
    def __init__(self, voice_id, context_tag, external_start, capacity):
        self.voice_id, self.context_tag = voice_id, context_tag
        self.external_start, self.capacity = external_start, capacity
        self.used = 0
        self.external = []

    def write(self, spec, original, value, path):
        if isinstance(spec, str):
            if spec == 'bool':
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError(f'{path}: expected boolean')
                return bytes([int(value)])
            if spec == 'phase':
                return bytes([EnvelopePhase(_integer(value, 0, 4, path)).value])
            low, high = FORMAT_LIMITS[spec]
            return struct.pack('<' + spec, _integer(value, low, high, path))
        kind = spec[0]
        if kind == 'record':
            name = spec[1]
            cls, fields = _schema()[name]
            expected = {field for field, _ in binary._schema(2)[name][2]}
            if type(value) is not cls or set(vars(value)) != expected:
                raise ValueError(f'{path}: incorrect record type or unregistered runtime fields')
            data = bytearray()
            for field, s, o in fields:
                data.extend(bytes(_aligned(len(data), _alignment(s))-len(data)))
                data.extend(self.write(s, o, getattr(value, field), path + '.' + field))
            binary._consistent(value, path)
            return bytes(data)
        if kind == 'array':
            if (not isinstance(value, np.ndarray) or value.shape != (spec[1],)
                    or value.dtype.newbyteorder('=') != np.dtype('int16')):
                raise ValueError(f'{path}: expected {spec[1]} int16 values')
            return value.astype('<i2', copy=False).tobytes(order='C')
        if kind == 'list':
            if type(value) is not list or len(value) != spec[2]:
                raise ValueError(f'{path}: incorrect list type or length')
            return b''.join(self.write(spec[1], None, item, path) for item in value)
        if kind == 'gains':
            if value is None:
                return bytes(144)
            if type(value) is not tuple or len(value) != 32:
                raise ValueError(f'{path}: expected 32 harmonic gains')
            return b'\x01' + bytes(15) + struct.pack('<32I', *(_integer(v, 0, 131072, path) for v in value))
        if kind == 'external':
            field_id = spec[1]
            empty = value is None if field_id != 8 else type(value) is tuple and not value
            if empty:
                return bytes(REFERENCE.size)
            payload = bytearray()
            binary._write(payload, original, value, path, version=2)
            data = EXTERNAL_HEADER.pack(b'SVE1', VERSION, self.voice_id, self.context_tag, field_id) + payload
            logical_size = len(data)
            padded_size = _align(logical_size)
            if padded_size > self.capacity - self.used:
                raise ValueError(f'{path}: external state exceeds caller-owned capacity')
            address = self.external_start + self.used
            self.used += padded_size
            self.external.append(ExternalRecord(address, field_id, bytes(data) + bytes(padded_size-logical_size)))
            return REFERENCE.pack(address, logical_size)
        raise AssertionError(f'Unknown hardware page field: {spec}')


def pack_vm_page(state, *, context_tag, page_address, external_start, external_capacity):
    """Capture complete VM state at a stable boundary, without mutating it.

    The caller must own disjoint external spans and stop/synchronize rendering.
    Capacity failure publishes no image and allocates no device memory. This
    bridge uses the reference VM's K; changing K is not an export conversion.
    """
    if type(state) is not VMState:
        raise ValueError('Expected a VMState')
    voice_id, context_tag = _identity(state.voice_id, context_tag)
    _regions(page_address, external_start, external_capacity)
    if VM_K not in (2, 4, 8):
        raise ValueError('Unsupported hardware K')
    count = 64 * VM_K
    writer = _Writer(voice_id, context_tag, external_start, external_capacity)
    coefficients = writer.write(('array', count), None, state.coeff_state, 'VMState.coeff_state')
    history = writer.write(('array', count), None, state.tail_previous_coeffs, 'VMState.tail_previous_coeffs')
    if coefficients != history:
        raise ValueError('Hardware page requires equal committed coefficients and tail history')
    _integer(state.last_note, -1, 127, 'Last MIDI note')
    _integer(state.osc_mode, 0, 5, 'Oscillator mode')
    metadata = writer.write(('record', 'VMState'), None, state, 'VMState')
    end = page_layout()['metadata_end']
    assert len(metadata) == end - HEADER.size
    header = HEADER.pack(b'SVP1', VERSION, voice_id, context_tag, VM_K, len(metadata),
                         external_start if writer.used else 0, writer.used, bytes(8))
    page = header + metadata + bytes(COEFF_OFFSET-end) + coefficients + bytes(1024-len(coefficients))
    assert len(page) == PAGE_BYTES
    return PackedVMPage(page_address, page, tuple(writer.external))


class _Reader:
    def __init__(self, page, read_external, voice_id, context_tag, start, extent):
        self.page, self.offset = page, HEADER.size
        self.read_external = read_external
        self.voice_id, self.context_tag = voice_id, context_tag
        self.start, self.extent, self.used = start, extent, 0

    def take(self, count):
        start = self.offset
        self.offset += count
        return self.page[start:self.offset]

    def read(self, spec, original, path):
        if isinstance(spec, str):
            if spec in ('bool', 'phase'):
                value = self.take(1)[0]
                if spec == 'phase':
                    return EnvelopePhase(value)
                if value > 1:
                    raise ValueError(f'{path}: invalid boolean')
                return bool(value)
            return struct.unpack('<' + spec, self.take(_size(spec)))[0]
        kind = spec[0]
        if kind == 'record':
            cls, fields = _schema()[spec[1]]
            value = cls.__new__(cls)
            for field, member, orig in fields:
                if any(self.take(_aligned(self.offset, _alignment(member))-self.offset)):
                    raise ValueError(f'{path}.{field}: nonzero alignment padding')
                setattr(value, field, self.read(member, orig, path + '.' + field))
            # Root's three specially stored fields are populated by the caller.
            if spec[1] != 'VMState':
                binary._consistent(value, path)
            return value
        if kind == 'array':
            return np.frombuffer(self.take(_size(spec)), dtype='<i2').astype(np.int16, copy=True)
        if kind == 'list':
            return [self.read(spec[1], None, path) for _ in range(spec[2])]
        if kind == 'gains':
            flag = self.take(1)[0]
            if any(self.take(15)):
                raise ValueError(f'{path}: nonzero gain alignment padding')
            raw = self.take(128)
            if flag == 0:
                if any(raw):
                    raise ValueError(f'{path}: nonzero absent gain padding')
                return None
            if flag != 1:
                raise ValueError(f'{path}: invalid gain presence')
            return struct.unpack('<32I', raw)
        if kind == 'external':
            address, size = REFERENCE.unpack(self.take(REFERENCE.size))
            field_id = spec[1]
            if address == size == 0:
                return () if field_id == 8 else None
            padded = _align(size)
            if (size <= EXTERNAL_HEADER.size or address != self.start + self.used
                    or padded > self.extent-self.used):
                raise ValueError(f'{path}: invalid or overlapping external reference')
            raw = _bytes(self.read_external(address, padded), path)
            if len(raw) != padded or any(raw[size:]):
                raise ValueError(f'{path}: truncated external record or nonzero padding')
            expected = (b'SVE1', VERSION, self.voice_id, self.context_tag, field_id)
            if EXTERNAL_HEADER.unpack_from(raw) != expected:
                raise ValueError(f'{path}: external record owner/version/type mismatch')
            reader = binary._Reader(raw[EXTERNAL_HEADER.size:size])
            reader.version = 2
            value = reader.read(original, path)
            if reader.offset != len(reader.data):
                raise ValueError(f'{path}: trailing external bytes')
            if value is None or field_id == 8 and not value:
                raise ValueError(f'{path}: noncanonical empty external record')
            self.used += padded
            return value
        raise AssertionError(f'Unknown hardware page field: {spec}')


def unpack_vm_page(page, read_external, *, voice_id, context_tag, page_address):
    """Restore a detached VM from a complete page and bounded external reads.

    Owner, K, layout, reserved bytes and reference extents are checked before
    adopting state. The caller applies the result only after success; this
    function never mutates a live VM or requests an out-of-extent external read.
    """
    voice_id, context_tag = _identity(voice_id, context_tag)
    page = _bytes(page, 'Voice page')
    if len(page) != PAGE_BYTES:
        raise ValueError('Voice page must contain exactly 2048 bytes')
    magic, version, owner, tag, k, meta_bytes, start, extent, reserved = HEADER.unpack_from(page)
    if (magic, version, owner, tag, k) != (b'SVP1', VERSION, voice_id, context_tag, VM_K):
        raise ValueError('Voice page owner/version/K mismatch')
    _regions(page_address, start, extent)
    end = page_layout()['metadata_end']
    if (meta_bytes != end-HEADER.size or any(reserved) or any(page[end:COEFF_OFFSET])
            or any(page[COEFF_OFFSET+64*k*2:]) or not extent and start):
        raise ValueError('Invalid page layout, extension extent or reserved padding')
    reader = _Reader(page, read_external, voice_id, context_tag, start, extent)
    state = reader.read(('record', 'VMState'), None, 'VMState')
    assert reader.offset == end
    if reader.used != extent:
        raise ValueError('Unreferenced external state extent')
    state.voice_id = voice_id
    state.coeff_state = np.frombuffer(page[COEFF_OFFSET:COEFF_OFFSET+64*k*2], dtype='<i2').astype(np.int16, copy=True)
    state.tail_previous_coeffs = state.coeff_state.copy()
    _integer(state.last_note, -1, 127, 'Last MIDI note')
    _integer(state.osc_mode, 0, 5, 'Oscillator mode')
    binary._consistent(state, 'VMState')
    from ..vm.resources import prepare_oscillator_resources
    prepare_oscillator_resources()
    return state


def check_page_images(images):
    """Reject cross-voice aliasing and physical overflow in a prepared image set."""
    spans = []
    for index, image in enumerate(images):
        if type(image) is not PackedVMPage or len(image.page) != PAGE_BYTES:
            raise ValueError('Expected a complete PackedVMPage')
        spans.append(MemorySpan(f'page {index}', image.page_address, PAGE_BYTES))
        spans.extend(MemorySpan(f'page {index} extension {j}', record.address, len(record.data))
                     for j, record in enumerate(image.external))
    failures = check_memory_map(spans, EXTERNAL_RAM_BYTES)
    if failures:
        raise ValueError(f'Prepared voice images overlap or exceed external RAM: {failures}')
