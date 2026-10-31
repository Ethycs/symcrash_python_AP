"""WSES framing and legacy fixed-layout fields, checked against independent bytes."""
import binascii
from copy import deepcopy
from functools import partial
import struct
from typing import get_type_hints

import numpy as np
import pytest

from symcrash_python_ap.voice.snapshot_bundle import (
    SnapshotBundle, SnapshotChunk, SnapshotHeader, ChunkType)
from symcrash_python_ap.voice.voice_state_codec import (
    vmstate_to_bytes, vmstate_from_bytes, envelope_to_bytes, envelope_from_bytes,
    get_vmstate_size)
from symcrash_python_ap.vm import VMState
from symcrash_python_ap.vm.state import EnvelopePhase


# This module freezes the legacy subset explicitly; new writers default to v1.
vmstate_to_bytes = partial(vmstate_to_bytes, version=0)


def record(kind, version, voice, payload, reserved=0):
    body = struct.pack('<HHHHI', kind, version, voice, reserved, len(payload)) + payload
    return body + struct.pack('<I', binascii.crc32(body))


def container(*records, version=0, reserved=0, count=None):
    prefix = struct.pack('<4sHHHH', b'WSES', version, 0xfffe,
                         len(records) if count is None else count, reserved)
    body = b''.join(records)
    return prefix + struct.pack('<I', binascii.crc32(prefix+body)) + body


def rehash_global(data):
    return data[:12] + struct.pack('<I', binascii.crc32(data[:12]+data[16:])) + data[16:]


def raw_core(voice=7):
    """Literal layout, independent of the production serializer's field list."""
    data = bytearray(644)
    struct.pack_into('<HH', data, 0, 0, voice)
    struct.pack_into('<4I', data, 4, 0x89abcdef, 0xfedcba98, 0x87654321, 0xffffffff)
    struct.pack_into('<bbh', data, 20, 127, 0, -1234)
    struct.pack_into('<16h', data, 24, *range(-8,8))
    struct.pack_into('<BBHIhHIIhH', data, 56, 3, 0, 0, 123, 17000, 0, 456, 789, 21000, 0)
    struct.pack_into('<hh', data, 80, -12345, 23456)
    struct.pack_into('<256h', data, 84, *range(-128,128))
    struct.pack_into('<I', data, 596, 0xf1234567)
    struct.pack_into('<BBH', data, 600, 2, 4, 0)
    struct.pack_into('<IBB', data, 604, 0x12345678, 1, 0)
    struct.pack_into('<hh', data, 610, 12345, -23456)
    struct.pack_into('<4B', data, 614, 1, 1, 1, 0)
    struct.pack_into('<h', data, 618, -22222)
    struct.pack_into('<6I', data, 620, 3, 96, 0xffffffff, 0x80000000, 0xabcdef01, 0xdeadbeef)
    return bytes(data)


def test_literal_core_layout_unsigned_phases_and_little_endian_vectors():
    raw = raw_core()
    state = vmstate_from_bytes(raw)
    assert state.voice_id == 7
    assert (state.phase, state.phase_inc, state.target_phase_inc, state.pitch_slew_rate) == (
        0x89abcdef, 0xfedcba98, 0x87654321, 0xffffffff)
    assert (state.last_note,state.last_cents) == (127,-1234)
    assert state.harmonic_amps_q15 == list(range(-8,8))
    assert state.envelope.phase == EnvelopePhase.RELEASE
    assert state.envelope.phase_sample == 123
    assert state.envelope.attack_samples == 456 and state.envelope.decay_samples == 789
    assert state.envelope.sustain_level_q15 == 21000
    np.testing.assert_array_equal(state.coeff_state, np.arange(-128,128,dtype=np.int16))
    assert state.coeff_state.dtype == np.int16
    assert state.prng_state == 0xf1234567
    assert state.blocks_since_start == 0x12345678
    assert state.ccs_state.phase == 0xffffffff
    assert state.ccs_state.phase_inc == 0x80000000
    assert state.ccs_state.target_phase_inc == 0xabcdef01
    assert state.ccs_state.phase_slew_rate == 0xdeadbeef
    assert vmstate_to_bytes(state) == raw
    # Endian of the source ndarray must not change the file format.
    state.coeff_state = state.coeff_state.astype('>i2')
    assert vmstate_to_bytes(state) == raw
    assert len(raw) == get_vmstate_size() == 644


def test_literal_container_and_chunk_crcs_match_documented_byte_scope():
    raw = container(record(1,0,7,raw_core()), record(5,0,0xffff,b'globals'))
    bundle = SnapshotBundle.from_bytes(raw)
    assert bundle.header.chunk_count == 2
    assert bundle.list_voice_ids() == [7]
    assert bundle.get_global_state() == b'globals'
    assert bundle.get_voice_state(7).phase == 0x89abcdef
    assert bundle.to_bytes() == raw
    header = raw[:16]
    assert header[:12].hex() == '575345530000feff02000000'
    assert SnapshotHeader.from_bytes(header).to_bytes() == header


@pytest.mark.parametrize('position', [0,1,2])
def test_unknown_chunk_types_are_checked_then_skipped(position):
    parts = [record(1,0,7,raw_core()), record(5,0,0xffff,b'globals')]
    parts.insert(position, record(0xbeef,0xffff,3,b'future payload'))
    bundle = SnapshotBundle.from_bytes(container(*parts))
    assert bundle.header.chunk_count == 3  # Original framing, including skipped records.
    assert [c.chunk_type for c in bundle.chunks] == [ChunkType.CORE,ChunkType.GLOBAL]
    assert bundle.list_voice_ids() == [7]
    assert bundle.get_global_state() == b'globals'
    # Resaving emits only the known retained records, with a corrected count/CRC.
    saved = SnapshotBundle.from_bytes(bundle.to_bytes())
    assert saved.header.chunk_count == 2


def test_unknown_only_bundle_is_valid_and_does_not_invent_voices():
    loaded = SnapshotBundle.from_bytes(container(record(77,1,11,b'a'),record(77,2,11,b'b')))
    assert loaded.chunks == [] and loaded.list_voice_ids() == []
    assert loaded.get_global_state() is None and loaded.get_voice_state(11) is None


@pytest.mark.parametrize('kind', [1,2,3,4,5])
def test_known_chunk_versions_are_not_silently_interpreted(kind):
    raw = container(record(kind,3,0xffff if kind==5 else 7,b'new version'))
    with pytest.raises(ValueError, match='Unsupported version'):
        SnapshotBundle.from_bytes(raw)


@pytest.mark.parametrize('version', [1,9,65535])
def test_unsupported_header_version_is_rejected_even_with_valid_crc(version):
    with pytest.raises(ValueError, match='Unsupported snapshot version'):
        SnapshotBundle.from_bytes(container(version=version))


@pytest.mark.parametrize('kind', [1,2,3,4,5])
def test_known_chunks_must_address_the_correct_voice_or_global_scope(kind):
    with pytest.raises(ValueError, match='scope'):
        SnapshotBundle.from_bytes(container(record(kind,0,7 if kind==5 else 0xffff,b'x')))


def test_both_reserved_fields_are_checked_independently_of_valid_crcs():
    with pytest.raises(ValueError, match='reserved snapshot header'):
        SnapshotBundle.from_bytes(container(reserved=1))
    with pytest.raises(ValueError, match='reserved snapshot chunk'):
        SnapshotBundle.from_bytes(container(record(77,0,7,b'x',reserved=1)))


@pytest.mark.parametrize('kind,voice,payload', [(1,7,raw_core()),(5,0xffff,b'globals'),(3,7,b'ccs')])
def test_duplicate_chunk_identities_are_rejected(kind,voice,payload):
    rec = record(kind,0,voice,payload)
    with pytest.raises(ValueError, match='Duplicate'):
        SnapshotBundle.from_bytes(container(rec,rec))


def test_core_inner_version_and_voice_id_must_match_the_supported_chunk():
    with pytest.raises(ValueError, match='voice ID'):
        SnapshotBundle.from_bytes(container(record(1,0,8,raw_core(7))))
    altered = b'\x03\x00' + raw_core()[2:]
    with pytest.raises(ValueError, match='Unsupported VMState version'):
        SnapshotBundle.from_bytes(container(record(1,0,7,altered)))


def test_mutable_chunk_and_bundle_recompute_both_crcs_after_edits():
    chunk = SnapshotChunk(77,0,1,b'old')
    first = chunk.to_bytes()
    chunk.payload, chunk.voice_id, chunk.chunk_version = b'edited', 8, 9
    second = chunk.to_bytes()
    assert first != second
    restored, end = SnapshotChunk.from_bytes(second)
    assert restored.payload == b'edited' and restored.voice_id == 8 and restored.chunk_version == 9
    assert end == len(second)
    bundle = SnapshotBundle()
    bundle.add_global_state(b'old')
    previous_crc = SnapshotBundle.from_bytes(bundle.to_bytes()).header.global_crc32
    bundle.chunks[0].payload = b'edited'
    loaded = SnapshotBundle.from_bytes(bundle.to_bytes())
    assert loaded.get_global_state() == b'edited'
    assert loaded.header.global_crc32 != previous_crc


def test_chunk_crc_is_verified_even_when_global_crc_is_repaired():
    for kind in (5,0xbeef):
        raw = container(record(kind,0,0xffff,b'payload'))
        corrupt = raw[:-1] + bytes([raw[-1] ^ 1])
        with pytest.raises(ValueError, match='CRC mismatch in chunk'):
            SnapshotBundle.from_bytes(rehash_global(corrupt))
        # A zero checksum never suppresses validation.
        with pytest.raises(ValueError, match='CRC mismatch in chunk'):
            SnapshotBundle.from_bytes(rehash_global(raw[:-4]+b'\0'*4))


def test_header_and_body_corruption_are_covered_by_the_global_crc():
    raw = container(record(5,0,0xffff,b'payload'))
    for pos in (8,12,28):
        changed = bytearray(raw)
        changed[pos] ^= 1
        with pytest.raises(ValueError, match='Global CRC'):
            SnapshotBundle.from_bytes(changed)


def test_every_truncation_is_rejected_and_chunk_lengths_cannot_escape_the_buffer():
    raw = container(record(1,0,7,raw_core()), record(5,0,0xffff,b'globals'))
    for length in range(len(raw)):
        with pytest.raises(ValueError):
            SnapshotBundle.from_bytes(raw[:length])
    huge = bytearray(container(record(5,0,0xffff,b'x')))
    struct.pack_into('<I', huge, 24, 0xffffffff)
    with pytest.raises(ValueError, match='Truncated'):
        SnapshotBundle.from_bytes(rehash_global(huge))


@pytest.mark.parametrize('count', [0,2,65535])
def test_chunk_count_must_match_exactly(count):
    with pytest.raises(ValueError):
        SnapshotBundle.from_bytes(container(record(5,0,0xffff,b'x'),count=count))


def test_trailing_data_cannot_hide_behind_a_valid_global_crc():
    raw = container(record(5,0,0xffff,b'x'))
    with pytest.raises(ValueError, match='Trailing'):
        SnapshotBundle.from_bytes(rehash_global(raw+b'extra'))


def test_chunk_offsets_bytes_like_inputs_and_resolved_annotations():
    raw = record(5,0,0xffff,b'x')
    chunk, end = SnapshotChunk.from_bytes(memoryview(b'prefix'+raw),6)
    assert chunk.payload == b'x' and end == len(raw)+6
    assert get_type_hints(SnapshotChunk.from_bytes)['return']
    for offset in (-1, len(raw)+1, True, .5):
        with pytest.raises(ValueError):
            SnapshotChunk.from_bytes(raw,offset)
    for bad in (None, 'bytes', [0]*16):
        with pytest.raises(ValueError):
            SnapshotBundle.from_bytes(bad)


def test_failed_append_and_decode_do_not_replace_existing_data():
    bundle = SnapshotBundle()
    bundle.add_voice_state(7,VMState.create(7))
    before = bundle.to_bytes()
    for voice in (7,8):
        with pytest.raises(ValueError):
            bundle.add_voice_state(voice,VMState.create(7))
        assert bundle.to_bytes() == before
    with pytest.raises(ValueError):
        SnapshotBundle.from_bytes(before[:-1])
    assert bundle.to_bytes() == before
    bundle.add_global_state(b'x')
    with pytest.raises(ValueError, match='Duplicate'):
        bundle.add_global_state(b'y')
    assert bundle.get_global_state() == b'x'


def test_direct_chunk_mutation_cannot_bypass_scope_identity_or_version_checks():
    bundle = SnapshotBundle()
    bundle.add_voice_state(7,VMState.create(7))
    original = deepcopy(bundle.chunks[0])
    for field,value in [('voice_id',8),('chunk_version',3),('voice_id',0xffff)]:
        bundle.chunks = [deepcopy(original)]
        setattr(bundle.chunks[0],field,value)
        for action in (bundle.to_bytes, lambda:bundle.get_voice_state(7), bundle.list_voice_ids):
            with pytest.raises(ValueError):
                action()


@pytest.mark.parametrize('length', [0,1,23,25,643,645])
def test_legacy_payload_decoders_require_exact_lengths(length):
    with pytest.raises(ValueError):
        vmstate_from_bytes(bytes(length))
    with pytest.raises(ValueError):
        envelope_from_bytes(bytes(length))


@pytest.mark.parametrize('offset', [21,57,58,66,78,602,609])
def test_reserved_legacy_payload_fields_are_rejected(offset):
    raw = bytearray(raw_core())
    raw[offset] = 1
    with pytest.raises(ValueError, match='reserved'):
        vmstate_from_bytes(raw)


@pytest.mark.parametrize('offset', [608,614,615,616,617])
def test_legacy_boolean_fields_reject_noncanonical_values(offset):
    raw = bytearray(raw_core())
    raw[offset] = 2
    with pytest.raises(ValueError, match='boolean'):
        vmstate_from_bytes(raw)


@pytest.mark.parametrize('field,value', [('phase',-1),('phase',1<<32),('prng_state',1<<32),
    ('last_note',-2),('voice_id',65535),('voice_id',True),('phase_inc',1.5),
    ('last_cents',1<<15),('wet_ready',2)])
def test_legacy_writer_does_not_wrap_or_coerce_invalid_fields(field,value):
    state = VMState.create(7)
    setattr(state,field,value)
    with pytest.raises(ValueError):
        vmstate_to_bytes(state)


@pytest.mark.parametrize('bad', [np.zeros(255,dtype=np.int16),np.zeros(512,dtype=np.int16),
    np.zeros((16,16),dtype=np.int16),np.zeros(256,dtype=np.float32),
    np.full(256,32768,dtype=np.int32),np.full(256,-32769,dtype=np.int32)])
def test_legacy_coefficient_dimensions_and_range_are_checked_before_narrowing(bad):
    state = VMState.create(7)
    state.coeff_state = bad
    with pytest.raises(ValueError, match='coefficient'):
        vmstate_to_bytes(state)


@pytest.mark.parametrize('harmonics', [[0]*15,[0]*17,[32768]*16,[1.5]*16])
def test_legacy_harmonic_vector_never_truncates_or_wraps(harmonics):
    state = VMState.create(7)
    state.harmonic_amps_q15 = harmonics
    with pytest.raises(ValueError):
        vmstate_to_bytes(state)


def test_legacy_restores_own_their_coefficient_buffers():
    data = bytearray(raw_core())
    a,b = vmstate_from_bytes(data),vmstate_from_bytes(data)
    a.coeff_state[0] = 999
    data[84:86] = b'\0\0'
    assert b.coeff_state[0] == -128
    assert not np.shares_memory(a.coeff_state,b.coeff_state)
    assert envelope_to_bytes(b.envelope) == raw_core()[56:80]


def test_frozen_core_layout_is_not_reinterpreted_under_another_vm_layout(monkeypatch):
    import symcrash_python_ap.voice.voice_state_codec as codec
    monkeypatch.setattr(codec, 'COEFF_LEN', 512)
    for action in (lambda:codec.vmstate_from_bytes(raw_core()),
                   lambda:codec.vmstate_to_bytes(VMState.create(7), version=0)):
        with pytest.raises(ValueError, match='runtime layout'):
            action()
