"""Exact 2 KiB page, external ownership, and future VM behavior."""
from dataclasses import replace
import json
import struct
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.voice.hardware_vm_page import (
    pack_vm_page, unpack_vm_page, page_layout, check_page_images, HEADER,
    PAGE_BYTES, COEFF_OFFSET, COEFF_WORD_OFFSET,
)
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.scheduling import ExecutionPolicy
from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes
from tests.conformance.test_vm_snapshot_replay import prepared, assert_state_equal, execution_path


OPTIONS = dict(context_tag=0xfedcba98, page_address=0x2000,
               external_start=0x04000000, external_capacity=512 * 1024)
KINDS = ('a', 'b', 'pwm', 'fm6', 'morph', 'sample', 'sample_coeff', 'wavetable', 'granular')


def pack(state, **options):
    return pack_vm_page(state, **(OPTIONS | options))


def reader_for(image):
    records = {record.address: record.data for record in image.external}
    def read(address, length):
        assert address in records, 'Reader must never receive an unchecked reference'
        assert len(records[address]) == length
        return records[address]
    return read


def restore(image, voice_id, **options):
    return unpack_vm_page(image.page, reader_for(image), voice_id=voice_id,
                          context_tag=OPTIONS['context_tag'], page_address=image.page_address, **options)


def offset(field):
    return next(row['offset'] for row in page_layout()['fields'] if row['field'] == 'VMState.' + field)


@pytest.mark.parametrize('kind', KINDS)
def test_complete_state_and_future_audio_roundtrip(kind, execution_path):
    state, frame = prepared(kind, voice_id=0xe237)
    before = vmstate_to_bytes(state)
    image = pack(state)
    assert len(image.page) == 2048
    assert vmstate_to_bytes(state) == before
    restored = restore(image, state.voice_id)
    assert_state_equal(state, restored)
    assert pack(restored) == image
    assert not np.shares_memory(restored.coeff_state, restored.tail_previous_coeffs)
    nonzero = False
    for block in range(14):
        if block == 3:
            frame.note, frame.f0_hz = 74, 587.33
        if block == 8:
            state.trigger_release()
            restored.trigger_release()
        policy = ExecutionPolicy(serviced=block % 5 != 0)
        expected = vm_step_block(state, frame, policy=policy)[1].copy()
        actual = vm_step_block(restored, frame, policy=policy)[1].copy()
        np.testing.assert_array_equal(actual, expected)
        nonzero |= bool(np.any(expected))
    assert nonzero
    assert_state_equal(state, restored)


def test_literal_header_offsets_and_full_width_payload():
    state = VMState.create(0xfffe)
    state.phase, state.phase_inc = 0xffffffff, 0x80000000
    state.target_phase_inc, state.pitch_slew_rate = 0x87654321, 0xffffffff
    state.last_cents = -(1 << 50)
    state.dc_block_x1, state.dc_block_y1 = -(1 << 31), (1 << 31)-1
    state.harmonic_amps_q15 = [-32768, 32767] * 8
    state.coeff_state[:] = (np.arange(256, dtype=np.int32) * 257 - 32768).astype(np.int16)
    state.tail_previous_coeffs[:] = state.coeff_state
    image = pack(state)
    assert COEFF_WORD_OFFSET == 64 and COEFF_OFFSET == 1024
    assert page_layout()['metadata_end'] == 837
    assert image.page[:32] == struct.pack('<4sHHIHHII8s', b'SVP1', 1, 65534,
        0xfedcba98, 4, 805, 0, 0, bytes(8))
    assert struct.unpack_from('<4Ih6xq16h', image.page, 32) == (
        0xffffffff, 0x80000000, 0x87654321, 0xffffffff, -1, -(1 << 50), *([-32768, 32767] * 8))
    assert struct.unpack_from('<ii', image.page, 152) == (-(1 << 31), (1 << 31)-1)
    assert image.page[1024:1536] == struct.pack('<256h', *map(int, state.coeff_state))
    assert image.page[837:1024] == bytes(187)
    assert image.page[1536:] == bytes(512)
    assert_state_equal(state, restore(image, state.voice_id))


def test_no_fixed_scalar_crosses_a_ram_word_and_vectors_are_word_aligned():
    for row in page_layout()['fields']:
        start, size, spec = row['offset'], row['bytes'], row['storage']
        if isinstance(spec, str):
            assert start // 16 == (start+size-1) // 16, row['field']
            assert start % size == 0, row['field']
        elif spec[0] in ('array', 'list', 'gains', 'record'):
            assert start % 16 == 0, row['field']
        else:
            assert start % 8 == 0 and size == 8, row['field']


@pytest.mark.parametrize('byte', [50, 51, 55, 97, 98, 103, 114, 119, 188, 191, 688+15])
def test_alignment_padding_is_canonical_and_never_discarded(byte):
    image = pack(VMState.create(1))
    page = bytearray(image.page)
    assert page[byte] == 0
    page[byte] = 1
    with pytest.raises(ValueError, match='padding'):
        restore(replace(image, page=bytes(page)), 1)


def test_all_loaded_engines_resources_and_noise_survive_together():
    state, _ = prepared('a')
    for kind, field in [('b', '_mode_b_state'), ('fm6', '_fm6_state'), ('morph', '_fm6_morph_state'),
                        ('sample_coeff', '_sample_state'), ('wavetable', '_wavetable_state'),
                        ('granular', '_granular_state')]:
        setattr(state, field, getattr(prepared(kind)[0], field))
    from symcrash_python_ap.vm.noise_exciter import NoiseExciterState
    state._noise_state = NoiseExciterState(pink_b0=123, pink_b1=-234, pink_b2=345, brown_acc=-456)
    state.wavetable_state.atlas.name = '音色 — checkpoint'
    state.mode_switch.b_harmonic_gains = tuple(range(32))
    state.mode_switch.b_gain_bound = 131072
    state.cmmm_memory.TENSION_DECAY = (1 << 48) + 3
    image = pack(state)
    assert len(image.external) == 7
    assert_state_equal(state, restore(image, state.voice_id))
    assert image.total_bytes == 2048 + sum(len(record.data) for record in image.external)
    assert all(record.address % 16 == 0 and len(record.data) % 16 == 0 for record in image.external)


def test_wire_event_owner_is_complete_and_independent():
    from symcrash_python_ap.wire_frame import WireEvent
    state = VMState.create(41)
    state.wire_state.events = tuple(WireEvent(lane=i, variant=1, strength_q15=-32768,
        target_kind=3, target_id=31, pattern_id=65535, param0=-123, param1=456, param2=789) for i in range(8))
    state.wire_state.frame_seq = 0xffffffff
    image = pack(state)
    assert len(image.external) == 1 and image.external[0].field_id == 8
    assert_state_equal(state, restore(image, 41))


@pytest.mark.parametrize('path,value', [
    ('phase', -1), ('phase_inc', 1 << 32), ('pitch_slew_rate', True),
    ('prng_state', 1.0), ('dc_block_x1', 1 << 31), ('eta_prev_q15', 32768),
    ('last_note', 128), ('osc_mode', 6), ('voice_id', 65535),
])
def test_unrepresentable_or_invalid_values_fail_without_coercion(path, value):
    state = VMState.create(1)
    setattr(state, path, value)
    with pytest.raises(ValueError):
        pack(state)


def test_no_history_or_unknown_field_is_silently_dropped():
    state = VMState.create(1)
    state.tail_previous_coeffs[200] = 1
    with pytest.raises(ValueError, match='committed'):
        pack(state)
    state.tail_previous_coeffs.fill(0)
    state.unknown_future_state = 99
    with pytest.raises(ValueError, match='unregistered'):
        pack(state)
    del state.unknown_future_state
    state.ccs_state.unknown_future_state = 0
    with pytest.raises(ValueError, match='unregistered'):
        pack(state)


def test_arrays_retain_endian_strides_and_independent_ownership():
    state = VMState.create(1)
    values = np.arange(512, dtype=np.int16)[::-2]
    state.coeff_state = values.astype('>i2')
    state.tail_previous_coeffs = values.copy()
    state.mask_state_q15 = np.arange(128, dtype=np.int16)[::2]
    image = pack(state)
    restored = restore(image, 1)
    np.testing.assert_array_equal(restored.coeff_state, values)
    np.testing.assert_array_equal(restored.mask_state_q15, state.mask_state_q15)
    restored.coeff_state.fill(-7)
    assert state.coeff_state[0] != -7 and restored.tail_previous_coeffs[0] != -7
    for malformed in (np.zeros(256, dtype=np.float64), np.zeros(256, dtype=np.int32),
                      np.zeros(255, dtype=np.int16), np.zeros((64, 4), dtype=np.int16)):
        state.coeff_state = malformed
        with pytest.raises(ValueError, match='int16'):
            pack(state)


def test_external_capacity_is_exact_and_failure_does_not_mutate_state():
    state, _ = prepared('morph')
    before = vmstate_to_bytes(state)
    image = pack(state)
    assert image.external_bytes > 0
    assert pack(state, external_capacity=image.external_bytes) == image
    with pytest.raises(ValueError, match='capacity'):
        pack(state, external_capacity=image.external_bytes-16)
    assert vmstate_to_bytes(state) == before
    end = 128 * 1024 * 1024
    exact = pack(state, external_start=end-image.external_bytes, external_capacity=image.external_bytes)
    assert_state_equal(state, restore(exact, state.voice_id))
    with pytest.raises(ValueError, match='exceeds'):
        pack(state, external_start=end-image.external_bytes+16, external_capacity=image.external_bytes)


@pytest.mark.parametrize('options', [dict(page_address=1), dict(external_start=17),
    dict(external_capacity=1), dict(external_start=0x2000, external_capacity=16),
    dict(page_address=128*1024*1024), dict(external_capacity=True)])
def test_invalid_owned_regions_are_rejected(options):
    with pytest.raises(ValueError):
        pack(VMState.create(1), **options)


def test_every_page_truncation_and_appended_byte_fails_before_external_read():
    image = pack(VMState.create(1))
    def forbidden(*args):
        pytest.fail('Invalid page must not request external state')
    for length in range(2048):
        with pytest.raises(ValueError, match='2048'):
            unpack_vm_page(image.page[:length], forbidden, voice_id=1, context_tag=OPTIONS['context_tag'], page_address=0x2000)
    with pytest.raises(ValueError, match='2048'):
        unpack_vm_page(image.page+b'\x00', forbidden, voice_id=1, context_tag=OPTIONS['context_tag'], page_address=0x2000)


@pytest.mark.parametrize('byte', [0, 4, 6, 8, 12, 14, 24, 691, 1023, 1536, 2047])
def test_bad_identity_version_k_layout_or_padding_is_rejected(byte):
    image = pack(VMState.create(1))
    page = bytearray(image.page)
    page[byte] ^= 1
    with pytest.raises(ValueError):
        restore(replace(image, page=bytes(page)), 1)


@pytest.mark.parametrize('kind', ['address', 'size', 'extent', 'owner', 'context', 'type', 'padding', 'truncate'])
def test_external_reference_cannot_escape_owner_or_extent(kind):
    state, _ = prepared('b')
    image = pack(state)
    page = bytearray(image.page)
    records = list(image.external)
    where = offset('_mode_b_state')
    if kind == 'address':
        struct.pack_into('<I', page, where, OPTIONS['external_start']+16)
    elif kind == 'size':
        struct.pack_into('<I', page, where+4, 0xffffffff)
    elif kind == 'extent':
        struct.pack_into('<I', page, 20, 0)
    else:
        raw = bytearray(records[0].data)
        if kind in ('owner', 'context', 'type'):
            raw[{'owner': 6, 'context': 8, 'type': 12}[kind]] ^= 1
        elif kind == 'padding':
            logical_size = struct.unpack_from('<I', page, where+4)[0]
            assert logical_size < len(raw)
            raw[-1] = 1
        else:
            raw = raw[:-1]
        records[0] = replace(records[0], data=bytes(raw))
    corrupt = replace(image, page=bytes(page), external=tuple(records))
    if kind in ('address', 'size', 'extent'):
        def read(*args):
            pytest.fail('Malformed reference must fail before external access')
    else:
        # A transport can return a short record; unlike the fixture accessor,
        # let the production decoder detect that length error itself.
        read = lambda address, size: corrupt.external[0].data
    with pytest.raises(ValueError):
        unpack_vm_page(corrupt.page, read, voice_id=state.voice_id,
            context_tag=OPTIONS['context_tag'], page_address=0x2000)


def test_two_optional_engines_cannot_alias_the_same_external_record():
    state, _ = prepared('b')
    state._fm6_state = prepared('fm6')[0]._fm6_state
    image = pack(state)
    page = bytearray(image.page)
    a, b = offset('_mode_b_state'), offset('_fm6_state')
    page[b:b+8] = page[a:a+8]
    with pytest.raises(ValueError, match='reference'):
        restore(replace(image, page=bytes(page)), state.voice_id)


def test_image_set_cannot_alias_pages_or_external_allocations():
    state, _ = prepared('b')
    first = pack(state)
    other = pack(state, page_address=0x2800, external_start=OPTIONS['external_start'] + first.external_bytes)
    assert check_page_images([first, other]) is None
    with pytest.raises(ValueError, match='overlap'):
        check_page_images([first, first])
    with pytest.raises(ValueError, match='overlap'):
        check_page_images([first, pack(state, page_address=0x2800)])


def test_fresh_process_reconstructs_resources_and_future_audio(tmp_path):
    from symcrash_python_ap.vm import hotpath_jit as hot
    state, frame = prepared('wavetable')
    image = pack(state)
    (tmp_path / 'page.bin').write_bytes(image.page)
    records = {}
    for record in image.external:
        name = f'{record.field_id}.bin'
        (tmp_path / name).write_bytes(record.data)
        records[record.address] = name
    (tmp_path / 'records.json').write_text(json.dumps(records), encoding='utf-8')
    script = tmp_path / 'restore.py'
    script.write_text('''from pathlib import Path
import json, numpy as np
from symcrash_python_ap.voice.hardware_vm_page import unpack_vm_page
from symcrash_python_ap.vm import vm_step_block, hotpath_jit as hot
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.scheduling import ExecutionPolicy
root = Path(__file__).parent
records = json.loads((root / 'records.json').read_text())
hot.set_jit_enabled(False)
state = unpack_vm_page((root / 'page.bin').read_bytes(),
    lambda address, size: (root / records[str(address)]).read_bytes(),
    voice_id=7, context_tag=0xfedcba98, page_address=0x2000)
frame = VoiceFrame(voice_id=7, note=69, f0_hz=440.0,
    coeff_osc_mode=4, gain_q15=20000, beta_target_q15=16384)
audio = [vm_step_block(state, frame, policy=ExecutionPolicy(serviced=i%5!=0))[1].copy() for i in range(12)]
np.save(root / 'audio.npy', np.asarray(audio), allow_pickle=False)
''', encoding='utf-8')
    prior = hot.jit_enabled()
    hot.set_jit_enabled(False)
    try:
        expected = [vm_step_block(state, frame, policy=ExecutionPolicy(serviced=i%5!=0))[1].copy() for i in range(12)]
        subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True, timeout=120)
    finally:
        hot.set_jit_enabled(prior)
    actual = np.load(tmp_path / 'audio.npy', allow_pickle=False)
    assert np.any(actual)
    np.testing.assert_array_equal(actual, expected)
