"""Integer CORE v1: independent byte fixtures, future audio and atomic VM restore."""
from dataclasses import asdict, replace
import hashlib
import json
import struct
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.frames import VoiceFramePacketV0b
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.executor import VMExecutor
from symcrash_python_ap.vm.scheduling import ExecutionPolicy
from symcrash_python_ap.voice.snapshot_bundle import SnapshotBundle
from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes, vmstate_from_bytes
from tests.conformance.test_vm_snapshot_replay import prepared, assert_state_equal, execution_path
from tests.conformance.test_wire_frame_transfer import FIXTURE
from tests.conformance.test_binary_snapshot import container, record


KINDS = ('a', 'b', 'pwm', 'fm6', 'morph', 'sample', 'sample_coeff', 'wavetable', 'granular')


@pytest.mark.parametrize('kind', KINDS)
def test_binary_reproduces_all_runtime_fields_and_future_audio(kind, execution_path):
    state, frame = prepared(kind)
    payload = vmstate_to_bytes(state)
    restored = vmstate_from_bytes(payload)
    assert_state_equal(state, restored)
    assert vmstate_to_bytes(restored) == payload
    outputs = []
    for block in range(18):
        if block == 4:
            frame.note, frame.f0_hz, frame.gain_q15 = 74, 587.33, 15000
        if block == 10:
            state.trigger_release()
            restored.trigger_release()
        policy = ExecutionPolicy(serviced=block % 5 != 0)
        expected = vm_step_block(state, frame, policy=policy)[1].copy()
        actual = vm_step_block(restored, frame, policy=policy)[1].copy()
        np.testing.assert_array_equal(actual, expected, err_msg=f'{kind}, block {block}')
        outputs.append(expected)
    assert np.any(outputs)
    assert_state_equal(state, restored)


@pytest.mark.parametrize('color', [0, 1, 2])
def test_binary_noise_filter_and_rng_replay(color, execution_path):
    state, frame = prepared('a')
    frame.noise_color, frame.noise_level_q15, frame.noise_seed_u32 = color, 8192, 0x87654321
    for _ in range(7):
        vm_step_block(state, frame)
    assert state._noise_state is not None
    restored = vmstate_from_bytes(vmstate_to_bytes(state))
    assert_state_equal(state, restored)
    for _ in range(8):
        expected = vm_step_block(state, frame)[1].copy()
        np.testing.assert_array_equal(vm_step_block(restored, frame)[1], expected)
    assert_state_equal(state, restored)


def test_binary_wire_sequence_and_events_survive_and_are_not_replayed_twice():
    frame = VoiceFramePacketV0b.from_bytes(FIXTURE).to_voice_frame()
    state = VMState.create(frame.voice_id)
    vm_step_block(state, frame)
    assert state.wire_state.events
    restored = vmstate_from_bytes(vmstate_to_bytes(state))
    assert_state_equal(state, restored)
    for seq in (frame.wire_data.frame_seq, frame.wire_data.frame_seq - 1,
                frame.wire_data.frame_seq + 1):
        frame.wire_data = replace(frame.wire_data, frame_seq=seq)
        expected = vm_step_block(state, frame)[1].copy()
        np.testing.assert_array_equal(vm_step_block(restored, frame)[1], expected)
        assert_state_equal(state, restored)


def test_unselected_loaded_resources_and_pending_latches_are_preserved():
    state = VMState.create(3)
    for kind, name in [('b', '_mode_b_state'), ('fm6', '_fm6_state'),
                       ('morph', '_fm6_morph_state'), ('sample_coeff', '_sample_state'),
                       ('wavetable', '_wavetable_state'), ('granular', '_granular_state')]:
        setattr(state, name, getattr(prepared(kind)[0], name))
    state.wavetable_state.atlas.name = '波形 — café 🎵'
    state.executor_latch.phrase_quantize_pending = True
    state.cmmm_memory.TENSION_DECAY = 93
    state.mode_b_state.phase_offset_q32 = 0xf1234567
    state.mode_b_state.fm_mod_buffer = np.arange(8, dtype=np.int16)
    restored = vmstate_from_bytes(vmstate_to_bytes(state))
    assert_state_equal(state, restored)
    restored.sample_state.buffer[:] = 0
    restored.wavetable_state.atlas.data[:] = 0
    restored.granular_state.coeff_buffer[:] = 0
    assert np.any(state.sample_state.buffer)
    assert np.any(state.wavetable_state.atlas.data)
    assert np.any(state.granular_state.coeff_buffer)


def test_lazy_defaults_and_explicit_empty_resource_arrays_survive():
    state = VMState.create(2)
    assert_state_equal(state, vmstate_from_bytes(vmstate_to_bytes(state)))
    state.sample_state.buffer = np.empty(0, dtype=np.int16)
    state.sample_state.coeff_buffer = np.empty((0, 64), dtype=np.int16)
    state.granular_state.coeff_buffer = np.empty((0, 64), dtype=np.int16)
    _ = state.wavetable_state, state.fm6_state, state.fm6_morph_state, state.mode_b_state
    assert_state_equal(state, vmstate_from_bytes(vmstate_to_bytes(state)))


def literal_core_v1():
    """Hand assembled grammar: no production schema, defaults, codec or ndarray bytes."""
    def ints(*values):
        return struct.pack('<' + 'q' * len(values), *values)
    def vector(values):
        return b'\x01' + struct.pack('<I', len(values)) + struct.pack('<' + 'h' * len(values), *values)
    return b''.join([
        struct.pack('<4H', 1, 7, 4, 256), struct.pack('<HH', 1, 40),
        ints(7, 0xffffffff, 0x87654321, 2345, 6789, 69, -23),
        struct.pack('<I', 16), ints(*range(-8, 8)),
        struct.pack('<HHB', 2, 7, 3), ints(123, 17000, 121, 235, 19000, 711),
        ints(-12345, 23456), vector(list(range(-128, 128))), vector([17] * 256),
        struct.pack('<HH', 3, 10), ints(0x1234, 0xffffffff, 5, 8), b'\x01',
        ints(12345, 2, 1, 8192), struct.pack('<IHH', 1, 4, 9),
        ints(4, 1, 200, 3, 3, 0, 7, 0, 0),
        ints(0xfedcba98, 1, 2, 456), b'\x01', ints(1111, 2222),
        vector([3333] * 64), b'\x01', ints(-4444), b'\x01', vector([5555] * 8), b'\x01',
        struct.pack('<HH', 5, 6), ints(0xdeadbeef, 67890, 12345, 678, 69, 2),
        b'\x01\x00', ints(0), vector([6666] * 8), b'\x00' * 6,
        struct.pack('<HH', 6, 9), ints(0xffffffff, 2, 1234, 3, 56), b'\x01', ints(4, 5, 6),
        struct.pack('<HH', 7, 7), ints(100, 200, 300, 400, 500, 600, 700),
    ])


def test_literal_v1_fields_endianness_versions_and_detached_buffers():
    raw = literal_core_v1()
    state = vmstate_from_bytes(raw)
    assert state.phase == 0xffffffff and state.phase_inc == 0x87654321
    assert state.envelope.release_samples == 711
    assert state.last_cents == -23
    assert state.ccs_state.phase == 0xdeadbeef
    assert state.wire_state.events[0].param0 == 7
    assert state.wire_state.frame_seq == 0xffffffff
    assert state.executor_latch.phrase_quantize_pending
    assert state.cmmm_memory.HARMONIC_DECAY == 700
    np.testing.assert_array_equal(state.coeff_state, np.arange(-128, 128))
    np.testing.assert_array_equal(state.tail_previous_coeffs, np.full(256, 17))
    assert vmstate_to_bytes(state) == raw
    # Neither byte order nor storage strides alter the canonical representation.
    backing = np.empty(512, dtype='>i2')
    backing[::2] = state.coeff_state
    state.coeff_state = backing[::2]
    assert vmstate_to_bytes(state) == raw
    input_buffer = bytearray(raw)
    restored = vmstate_from_bytes(memoryview(input_buffer))
    input_buffer[:] = bytes(len(input_buffer))
    assert vmstate_to_bytes(restored) == raw
    assert vmstate_from_bytes(raw).coeff_state.flags.writeable


@pytest.mark.parametrize('offset,fmt,value', [
    (0, 'H', 2), (2, 'H', 8), (4, 'H', 8), (6, 'H', 512),
    (8, 'H', 99), (10, 'H', 39), (12, 'q', 65535),
    (68, 'I', 17), (200, 'H', 3), (204, 'B', 9),
    (269, 'B', 2), (270, 'I', 0xffffffff),
    (1339, 'B', 2), (1372, 'I', 9), (1376, 'H', 19),
    (1719, 'B', 2), (1750, 'B', 2), (1799, 'B', 2),
])
def test_literal_v1_rejects_corrupt_identity_layout_types_counts_and_extents(offset, fmt, value):
    raw = bytearray(literal_core_v1())
    struct.pack_into('<' + fmt, raw, offset, value)
    with pytest.raises(ValueError):
        vmstate_from_bytes(raw)


def test_every_core_truncation_and_trailing_byte_is_rejected():
    raw = literal_core_v1()
    for size in range(len(raw)):
        with pytest.raises(ValueError):
            vmstate_from_bytes(raw[:size])
    with pytest.raises(ValueError, match='Trailing'):
        vmstate_from_bytes(raw + b'\x00')


@pytest.mark.parametrize('change', ['extent', 'rank', 'metadata', 'utf8'])
def test_loaded_resource_shape_extent_and_utf8_are_validated(change):
    state, _ = prepared('wavetable')
    data = bytearray(vmstate_to_bytes(state))
    marker = struct.pack('<HHBIII', 15, 6, 3, 2, 3, 256)
    at = data.index(marker)
    if change == 'extent':
        struct.pack_into('<III', data, at + 5, 0xffffffff, 0xffffffff, 0xffffffff)
    elif change == 'rank':
        data[at + 4] = 8
    elif change == 'metadata':
        struct.pack_into('<q', data, at + 17 + 1536 * 2, 9)
    else:
        data[data.index(b'checkpoint-only atlas')] = 0xff
    with pytest.raises(ValueError):
        vmstate_from_bytes(data)


def test_explicit_legacy_default_fixture_stays_byte_identical():
    raw = vmstate_to_bytes(VMState.create(7), version=0)
    assert len(raw) == 644
    assert hashlib.sha256(raw).hexdigest() == 'd9c289f6a31ed753da5e3acd0ad156dd740856d5a233f02f59f552bdf80a4c87'
    assert vmstate_to_bytes(VMState.create(7))[:2] == b'\x01\x00'


@pytest.mark.parametrize('field,value', [
    ('phase', 1.5), ('phase', True), ('phase', 1 << 63), ('last_cents', -(1 << 63) - 1),
    ('voice_id', 65535), ('voice_id', -1), ('last_note', 128), ('osc_mode', 6),
    ('wet_ready', 1), ('envelope', None), ('_sample_state', VMState()),
    ('coeff_state', np.zeros(256, dtype=np.float32)),
    ('coeff_state', np.zeros(256, dtype=np.int32)), ('coeff_state', np.zeros((16, 16), dtype=np.int16)),
    ('tail_previous_coeffs', np.zeros(255, dtype=np.int16)),
    ('mask_state_q15', np.zeros(63, dtype=np.int16)),
    ('harmonic_amps_q15', [0] * 15), ('harmonic_amps_q15', [32768] * 16),
])
def test_writer_rejects_noninteger_values_wrong_types_and_dimensions(field, value):
    state = VMState.create(7)
    setattr(state, field, value)
    with pytest.raises(ValueError):
        vmstate_to_bytes(state)


@pytest.mark.parametrize('kind,field,value', [
    ('b', 'fm_mod_buffer', np.zeros(3, dtype=np.int16)),
    ('fm6', 'operators', []), ('fm6', 'op_outputs', np.zeros((6, 8), dtype=np.int32)),
    ('morph', '_cached_matrix', np.zeros((5, 6), dtype=np.int16)),
    ('sample', 'root_note', 128), ('granular', 'num_source_blocks', 999),
])
def test_oscillator_record_shapes_and_linked_dimensions_are_checked(kind, field, value):
    state, _ = prepared(kind)
    name = {'b': 'mode_b_state', 'fm6': 'fm6_state', 'morph': 'fm6_morph_state',
            'sample': 'sample_state', 'granular': 'granular_state'}[kind]
    setattr(getattr(state, name), field, value)
    with pytest.raises(ValueError):
        vmstate_to_bytes(state)


@pytest.mark.parametrize('path', ['', 'mode_b_state', 'mode_b_state.weights', 'envelope',
                                  'wire_state', 'ccs_state', 'executor_latch', 'cmmm_memory'])
def test_runtime_field_additions_never_disappear_silently(path):
    state, _ = prepared('b')
    target = state
    for name in path.split('.') if path else []:
        target = getattr(target, name)
    target.unregistered_state = 7
    with pytest.raises(ValueError, match='unregistered'):
        vmstate_to_bytes(state)


def test_full_and_subset_executor_restores_are_atomic_and_have_explicit_scope():
    source, target = VMExecutor(), VMExecutor()
    frames = {}
    for vid, kind in [(9, 'fm6'), (2, 'b'), (12, 'granular')]:
        source.states[vid], frames[vid] = prepared(kind, vid)
    old = target.get_or_create_state(99)
    payload = source.snapshot_binary()
    bundle = SnapshotBundle.from_bytes(payload)
    assert [c.voice_id for c in bundle.chunks] == [2, 9, 12]
    assert all(c.chunk_version == 1 for c in bundle.chunks)
    target.restore_binary(payload, [12, 2])
    assert set(target.states) == {2, 12, 99} and target.states[99] is old
    target.restore_binary(payload)
    assert set(target.states) == {2, 9, 12}
    for _ in range(4):
        for vid, frame in frames.items():
            expected = source.step_voice(vid, frame).copy()
            np.testing.assert_array_equal(target.step_voice(vid, frame), expected)
    for vid in frames:
        assert_state_equal(source.states[vid], target.states[vid])
    assert SnapshotBundle.from_bytes(source.snapshot_binary([12])).list_voice_ids() == [12]
    target.restore_binary(SnapshotBundle().to_bytes(), [])
    assert set(target.states) == {2, 9, 12}
    target.restore_binary(SnapshotBundle().to_bytes())
    assert target.states == {}


@pytest.mark.parametrize('selection', [[7, 7], [99], [True], [-1], [65535], ['7']])
def test_invalid_capture_and_restore_selections_preserve_target(selection):
    source, target = VMExecutor(), VMExecutor()
    source.get_or_create_state(7)
    old = target.get_or_create_state(8)
    before = target.states
    with pytest.raises(ValueError):
        source.snapshot_binary(selection)
    with pytest.raises(ValueError):
        target.restore_binary(source.snapshot_binary(), selection)
    assert target.states is before and target.states == {8: old}


def test_bad_later_core_cannot_install_an_earlier_valid_voice_even_with_correct_crcs():
    target = VMExecutor()
    old = target.get_or_create_state(99)
    before = target.states
    raw = container(record(1, 1, 2, vmstate_to_bytes(VMState.create(2))),
                    record(1, 1, 7, literal_core_v1()[:-1]))
    with pytest.raises(ValueError):
        target.restore_binary(raw)
    assert target.states is before and target.states[99] is old


@pytest.mark.parametrize('outer,inner', [(0, 1), (1, 0)])
def test_outer_and_inner_core_versions_must_agree(outer, inner):
    raw = container(record(1, outer, 7, vmstate_to_bytes(VMState.create(7), version=inner)))
    with pytest.raises(ValueError, match='version does not match'):
        SnapshotBundle.from_bytes(raw)


def test_executor_does_not_accept_legacy_or_other_owners_payloads_as_complete():
    target = VMExecutor()
    old = target.get_or_create_state(99)
    legacy = SnapshotBundle()
    legacy.add_voice_state(7, VMState.create(7), version=0)
    global_bundle = SnapshotBundle()
    global_bundle.add_global_state(b'caller owned')
    for bundle in (legacy, global_bundle):
        with pytest.raises(ValueError, match='complete CORE v1'):
            target.restore_binary(bundle.to_bytes())
        assert target.states == {99: old}


def test_binary_checkpoint_replays_all_modes_in_a_fresh_process(tmp_path):
    source = VMExecutor()
    frames, expected = {}, {}
    for vid, kind in enumerate(KINDS):
        source.states[vid], frame = prepared(kind, vid)
        frames[vid] = asdict(frame)
    checkpoint, controls = tmp_path / 'state.wses', tmp_path / 'frames.json'
    checkpoint.write_bytes(source.snapshot_binary())
    controls.write_text(json.dumps(frames), encoding='utf-8')
    from symcrash_python_ap.frames import VoiceFrame
    for vid in source.states:
        digest = hashlib.sha256()
        for _ in range(12):
            digest.update(source.step_voice(vid, VoiceFrame(**frames[vid])).astype('<i2').tobytes())
        expected[str(vid)] = digest.hexdigest()
    script = '''
import hashlib, json, sys
from pathlib import Path
from symcrash_python_ap.vm.executor import VMExecutor
from symcrash_python_ap.frames import VoiceFrame
engine = VMExecutor()
engine.restore_binary(Path(sys.argv[1]).read_bytes())
frames = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
result = {}
for vid, data in frames.items():
    frame, digest = VoiceFrame(**data), hashlib.sha256()
    for _ in range(12):
        digest.update(engine.step_voice(int(vid), frame).astype('<i2').tobytes())
    result[vid] = digest.hexdigest()
print(json.dumps(result))
'''
    result = subprocess.run([sys.executable, '-c', script, str(checkpoint), str(controls)],
                            check=True, capture_output=True, text=True, timeout=60)
    assert json.loads(result.stdout) == expected, result.stderr
