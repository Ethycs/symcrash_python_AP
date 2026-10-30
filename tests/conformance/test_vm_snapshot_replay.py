"""Checkpoints must reproduce future audio, including loaded oscillator resources."""
import copy
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.executor import VMExecutor
from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.vm.scheduling import ExecutionPolicy
from symcrash_python_ap.vm.snapshot_codec import OSCILLATOR_TYPES, decode
from symcrash_python_ap.vm.wavetable_engine import WavetableAtlas


def json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def assert_state_equal(expected, actual, path='state'):
    """Walk actual runtime fields, independently of the serializer's field list."""
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
        assert not np.shares_memory(expected, actual), path
    elif is_dataclass(expected):
        assert type(actual) is type(expected), path
        for item in fields(expected):
            assert_state_equal(getattr(expected, item.name), getattr(actual, item.name),
                               path + '.' + item.name)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), path
        assert len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(expected, actual)):
            assert_state_equal(left, right, f'{path}[{i}]')
    else:
        assert actual == expected, path


def prepared(kind, voice_id=7):
    mode = {'a': 0, 'b': 1, 'pwm': 1, 'fm6': 3, 'morph': 3,
            'sample': 2, 'sample_coeff': 2, 'wavetable': 4, 'granular': 5}[kind]
    state = VMState.create(voice_id, prng_seed=123456)
    state.reset_for_note_on(note=69, attack_samples=121, decay_samples=235,
                            sustain_level_q15=19000, release_samples=711)
    frame = VoiceFrame(voice_id=voice_id, note=69, f0_hz=440.0,
                       coeff_osc_mode=mode, gain_q15=20000, beta_target_q15=16384)
    if kind == 'b':
        frame.partial_amps_q15 = [30000, 11000, 7000, 4000]
        frame.partial_phase_spread_q15 = 14321
    if kind == 'pwm':
        frame.waveform_type = 2
        frame.pwm_enabled = True
        frame.pwm_width_q15 = 11000
    if kind in ('fm6', 'morph'):
        frame.fm_algorithm_id = 5
        frame.fm_feedback_depth = 4
        frame.fm_op_ratios = [65536, 131072, 65536, 98304, 65536, 131072]
        frame.fm_op_levels = [30000, 16000, 24000, 8000, 12000, 16000]
    if kind == 'fm6':
        state.fm6_state.set_post_filter(True, 4321.0)
    if kind == 'morph':
        frame.fm_topo_a_id, frame.fm_topo_b_id, frame.fm_topo_blend_q15 = 0, 3, 12000
    # A deterministic nonzero, nonperiodic-at-block-boundaries resource.
    audio = ((np.arange(4096, dtype=np.int32) * 509) % 30001 - 15000).astype(np.int16)
    if kind in ('sample', 'sample_coeff'):
        state.sample_state.set_buffer(audio, loop_start=64, loop_end=4032,
                                      loop_enabled=True, root_note=69,
                                      precompute_coeffs=kind == 'sample_coeff')
    if kind == 'wavetable':
        state.wavetable_state.atlas = WavetableAtlas(
            data=audio[:1536].reshape(2, 3, 256), name='checkpoint-only atlas')
        state.wavetable_state.position_q16 = 12345
        state.wavetable_state.frame_q16 = 45678
        state.wavetable_state.frame_rate_q16 = 811
    if kind == 'granular':
        state.granular_state.set_source(audio)
        state.granular_state.density_q15 = 24000
        state.granular_state.spray_q15 = 17000
        state.granular_state.scan_rate_q16 = 131071
        state.granular_state.grain_rate_q16 = 80000
        state.granular_state.prng_state = 98273
    for _ in range(11):
        vm_step_block(state, frame)
    return state, frame


@pytest.fixture(params=[False, True], ids=['reference', 'jit'])
def execution_path(request):
    if request.param and not hot.NUMBA_AVAILABLE:
        pytest.skip('numba unavailable')
    previous = hot.jit_enabled()
    hot.set_jit_enabled(request.param)
    yield
    hot.set_jit_enabled(previous)


@pytest.mark.parametrize('kind', ['a', 'b', 'pwm', 'fm6', 'morph', 'sample',
                                 'sample_coeff', 'wavetable', 'granular'])
def test_json_checkpoint_resumes_every_oscillator(kind, execution_path, monkeypatch):
    state, frame = prepared(kind)
    checkpoint = json_copy(state.snapshot())
    restored = VMState.restore(checkpoint)
    assert_state_equal(state, restored)
    if kind == 'wavetable':
        def unexpected_default():
            pytest.fail('Restoring a loaded atlas must not replace it with a default')
        monkeypatch.setattr('symcrash_python_ap.vm.wavetable_engine.create_default_bank',
                            unexpected_default)
    if kind == 'granular':
        assert any(grain.is_active for grain in state.granular_state.grains)
    output = []
    for block in range(18):
        if block == 4:
            frame.note = 74
            frame.f0_hz = 587.33
            frame.gain_q15 = 15000
        if block == 10:
            state.trigger_release()
            restored.trigger_release()
        policy = ExecutionPolicy(serviced=block % 5 != 0)
        expected = vm_step_block(state, frame, policy=policy)[1].copy()
        actual = vm_step_block(restored, frame, policy=policy)[1].copy()
        np.testing.assert_array_equal(actual, expected, err_msg=f'{kind} block {block}')
        output.append(expected)
    assert np.any(output), 'A silent fixture cannot establish audible replay'
    assert_state_equal(state, restored)
    # A returned checkpoint is a value, not a view into changing render state.
    assert checkpoint == json_copy(VMState.restore(checkpoint).snapshot())


def test_snapshot_preserves_configuration_and_pending_control_state():
    state, _ = prepared('b')
    state.last_cents = -37
    state.osc_mode = 1
    state.executor_latch.phrase_quantize_pending = True
    state.cmmm_memory.TENSION_DECAY = 31234
    state.cmmm_memory.FATIGUE_DECAY = 32123
    state.cmmm_memory.HARMONIC_DECAY = 32345
    state.harmonic_amps_q15[2] = 9876
    snapshot = json_copy(state.snapshot())
    restored = VMState.restore(snapshot)
    assert_state_equal(state, restored)
    restored.harmonic_amps_q15[2] = 1
    assert state.harmonic_amps_q15[2] == snapshot['harmonic_amps_q15'][2] == 9876


def test_snapshot_does_not_initialize_unused_oscillators():
    state = VMState.create(1)
    restored = VMState.restore(json_copy(state.snapshot()))
    assert_state_equal(state, restored)
    for name in OSCILLATOR_TYPES:
        assert getattr(state, name) is getattr(restored, name) is None


@pytest.mark.parametrize('color', [0, 1, 2])
def test_noise_filter_history_survives_json_checkpoint(color, execution_path):
    state, frame = prepared('a')
    frame.noise_color, frame.noise_level_q15 = color, 8192
    frame.noise_seed_u32 = 0x87654321
    for _ in range(7):
        vm_step_block(state, frame)
    assert state._noise_state is not None
    restored = VMState.restore(json_copy(state.snapshot()))
    assert_state_equal(state, restored)
    for _ in range(8):
        expected = vm_step_block(state, frame)[1].copy()
        np.testing.assert_array_equal(vm_step_block(restored, frame)[1], expected)
    assert_state_equal(state, restored)


def test_all_loaded_resources_survive_even_when_not_the_selected_mode():
    state = VMState.create(3)
    for kind, name in [('b', '_mode_b_state'), ('fm6', '_fm6_state'),
                       ('morph', '_fm6_morph_state'), ('sample_coeff', '_sample_state'),
                       ('wavetable', '_wavetable_state'), ('granular', '_granular_state')]:
        source, _ = prepared(kind)
        setattr(state, name, getattr(source, name))
    restored = VMState.restore(json_copy(state.snapshot()))
    assert_state_equal(state, restored)
    restored.sample_state.buffer[:] = 0
    restored.wavetable_state.atlas.data[:] = 0
    restored.granular_state.coeff_buffer[:] = 0
    assert np.any(state.sample_state.buffer)
    assert np.any(state.wavetable_state.atlas.data)
    assert np.any(state.granular_state.coeff_buffer)


def test_unversioned_snapshots_remain_loadable_with_legacy_defaults():
    state, _ = prepared('a')
    snapshot = state.snapshot()
    for name in ('snapshot_version', 'last_note', 'last_cents', 'osc_mode',
                 'oscillators', 'envelope_config', 'executor_latch_phrase_pending',
                 'cmmm_memory_rates'):
        del snapshot[name]
    restored = VMState.restore(json_copy(snapshot))
    assert restored.phase == state.phase
    np.testing.assert_array_equal(restored.coeff_state, state.coeff_state)
    assert restored.envelope.attack_samples == VMState().envelope.attack_samples


@pytest.mark.parametrize('version', [-1, 2, True, '1', None])
def test_unknown_snapshot_versions_are_rejected(version):
    snapshot = VMState.create().snapshot()
    snapshot['snapshot_version'] = version
    with pytest.raises(ValueError, match='version'):
        VMState.restore(snapshot)


@pytest.mark.parametrize('mutate', [
    lambda x: x.update(kind='unregistered.module.Class'),
    lambda x: x['fields'].pop('phase'),
    lambda x: x['fields'].update(unknown_runtime_field=1),
])
def test_unknown_or_incomplete_oscillator_payloads_are_rejected(mutate):
    state, _ = prepared('b')
    snapshot = state.snapshot()
    mutate(snapshot['oscillators']['_mode_b_state'])
    with pytest.raises(ValueError):
        VMState.restore(snapshot)


def test_wrong_oscillator_class_is_rejected():
    state, _ = prepared('fm6')
    snapshot = state.snapshot()
    snapshot['oscillators']['_mode_b_state'] = snapshot['oscillators']['_fm6_state']
    with pytest.raises(ValueError, match='Incorrect oscillator'):
        VMState.restore(snapshot)


@pytest.mark.parametrize('change', [
    {'dtype': 'object'}, {'dtype': 'complex64'}, {'shape': [-1]},
    {'shape': [True]}, {'shape': [2]}, {'data': [32768]},
    {'data': [-32769]}, {'data': [True]}, {'data': [1.5]},
    {'data': [float('nan')], 'dtype': 'float64'},
])
def test_array_payload_rejects_unsafe_or_lossy_casts(change):
    payload = {'kind': 'array', 'dtype': 'int16', 'shape': [1], 'data': [1]}
    payload.update(change)
    with pytest.raises(ValueError):
        decode(payload)


def test_runtime_fields_cannot_be_silently_dropped():
    state, _ = prepared('b')
    state.mode_b_state.undeclared_phase = 999
    with pytest.raises(ValueError, match='Unregistered runtime fields'):
        state.snapshot()


def test_executor_bundle_restores_integer_voice_keys_after_json():
    original = VMExecutor()
    frames = {}
    for voice_id, kind in [(2, 'b'), (9, 'fm6'), (12, 'granular')]:
        original.states[voice_id], frames[voice_id] = prepared(kind, voice_id)
    restored = VMExecutor()
    restored.restore_all(json_copy(original.snapshot_all()))
    assert set(restored.states) == {2, 9, 12}
    for _ in range(4):
        for voice_id, frame in frames.items():
            expected = original.step_voice(voice_id, frame).copy()
            np.testing.assert_array_equal(restored.step_voice(voice_id, frame), expected)
    for voice_id in frames:
        assert_state_equal(original.states[voice_id], restored.states[voice_id])


@pytest.mark.parametrize('key', [3, '03', '4', True, 3.0, None])
def test_bad_bundle_id_is_rejected_without_replacing_existing_voices(key):
    executor = VMExecutor()
    state = executor.get_or_create_state(8)
    snapshot = VMState.create(4).snapshot()
    # '4' matches, so corrupt its oscillator instead; preceding valid state must
    # not be partially installed when the second snapshot fails.
    if key == '4':
        snapshot['oscillators'] = {}
    with pytest.raises(ValueError):
        executor.restore_all({2: VMState.create(2).snapshot(), key: snapshot})
    assert executor.states == {8: state}


def test_duplicate_integer_and_string_bundle_id_is_rejected():
    snapshot = VMState.create(3).snapshot()
    with pytest.raises(ValueError, match='duplicate'):
        VMExecutor().restore_all({3: snapshot, '3': copy.deepcopy(snapshot)})


@pytest.mark.parametrize('kind', ['fm6', 'morph', 'granular'])
def test_checkpoint_replays_in_a_fresh_process(kind, tmp_path):
    state, frame = prepared(kind)
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'state': state.snapshot(), 'frame': asdict(frame)},
                                     allow_nan=False), encoding='utf-8')
    expected = hashlib.sha256()
    for _ in range(12):
        expected.update(vm_step_block(state, frame)[1].astype('<i2').tobytes())
    script = '''
import hashlib, json, sys
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.frames import VoiceFrame
with open(sys.argv[1], encoding='utf-8') as stream:
    checkpoint = json.load(stream)
state = VMState.restore(checkpoint['state'])
frame = VoiceFrame(**checkpoint['frame'])
digest = hashlib.sha256()
for _ in range(12):
    digest.update(vm_step_block(state, frame)[1].astype('<i2').tobytes())
print(digest.hexdigest())
'''
    result = subprocess.run([sys.executable, '-c', script, str(checkpoint)],
                            check=True, capture_output=True, text=True, timeout=60)
    assert result.stdout.strip() == expected.hexdigest(), result.stderr


def test_fm_morph_gateway_advances_exactly_64_sample_phases():
    from symcrash_python_ap.vm.fm6_morph_engine import (
        FM6MorphState, fm6_generate_block_morphed, fm6_generate_block_morphed_full,
    )
    from symcrash_python_ap.vm.tables import VM_K
    state = FM6MorphState()
    state.set_morph(1, 4, 19000)
    state.operators[5].feedback_q15 = 8192
    reference = copy.deepcopy(state)
    phase_increment = 39370534
    expected = np.concatenate([fm6_generate_block_morphed(reference, phase_increment)
                               for _ in range(64 // VM_K)])
    actual = fm6_generate_block_morphed_full(state, phase_increment)
    assert actual.shape == (64,)
    np.testing.assert_array_equal(actual, expected)
    for op in state.operators:
        assert op.phase == (64 * phase_increment) & 0xFFFFFFFF
    assert_state_equal(reference, state)


def test_fm_morph_vm_reference_and_jit_match():
    previous = hot.jit_enabled()
    try:
        hot.set_jit_enabled(False)
        reference, frame = prepared('morph')
        hot.set_jit_enabled(True)
        accelerated, _ = prepared('morph')
        assert_state_equal(reference, accelerated)
        for block in range(12):
            frame.fm_topo_blend_q15 = (block * 3000) % 32768
            hot.set_jit_enabled(False)
            expected = vm_step_block(reference, frame)[1].copy()
            hot.set_jit_enabled(True)
            actual = vm_step_block(accelerated, frame)[1].copy()
            np.testing.assert_array_equal(actual, expected)
        assert_state_equal(reference, accelerated)
    finally:
        hot.set_jit_enabled(previous)


@pytest.mark.parametrize('input_size,output_size', [(0, 64), (8, 64), (63, 64),
                                                  (65, 64), (64, 8), (64, 65)])
def test_array_transform_rejects_wrong_block_sizes(input_size, output_size):
    source = np.zeros(input_size, dtype=np.int16)
    target = np.full(output_size, 12345, dtype=np.int64)
    with pytest.raises(ValueError, match='exactly 64'):
        hot.fwpt_64(source, 1, target)
    np.testing.assert_array_equal(target, np.full(output_size, 12345))


@pytest.mark.parametrize('size', [0, 4, 63, 65])
def test_reference_transform_rejects_short_and_oversized_input(size, monkeypatch):
    from symcrash_python_ap.wavelet import fast_lifting
    # Enforce the same interface with and without numba installed.
    for numba in (False, fast_lifting.NUMBA_AVAILABLE):
        monkeypatch.setattr(fast_lifting, 'NUMBA_AVAILABLE', numba)
        with pytest.raises(ValueError, match='exactly 64'):
            fast_lifting.fwpt_jit([1234] * size)
    processor = fast_lifting.SeparatedWaveletProcessor()
    before = processor._input_buf.copy()
    with pytest.raises(ValueError, match='complete input block'):
        processor.fwpt([1234] * size)
    np.testing.assert_array_equal(processor._input_buf, before)


def test_retrigger_reloads_custom_weights_after_reset(execution_path):
    state, frame = prepared('b')
    restored = VMState.restore(json_copy(state.snapshot()))
    for voice in (state, restored):
        voice.reset_for_note_on(note=69)
        assert voice.mode_b_state.last_custom_harmonics is None
    fresh = VMState.create(state.voice_id)
    fresh.reset_for_note_on(note=69)
    expected = vm_step_block(fresh, frame)[1].copy()
    for voice in (state, restored):
        actual = vm_step_block(voice, frame)[1].copy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(voice.mode_b_state.weights.a_h_q15,
                                      fresh.mode_b_state.weights.a_h_q15)
        np.testing.assert_array_equal(voice.mode_b_state.weights.b_h_q15,
                                      fresh.mode_b_state.weights.b_h_q15)
