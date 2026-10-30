"""Mode A preparation preserves the original tables and removes playback FWPT."""
from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.vm import VMState, vm_step_block, coeff_lut_streaming as lut
from symcrash_python_ap.vm import mode_a_resource as resource, mode_a_resource_builder as builder
from symcrash_python_ap.voice.voice_state_codec import vmstate_to_bytes, vmstate_from_bytes
from tests.conformance.test_basis_resource import forbidden, prohibit_preparation as prohibit_basis
from tests.conformance.test_carrier_spectrum import frame
from tests.conformance.test_vm_snapshot_replay import execution_path


@pytest.fixture(scope='module')
def atlas():
    return resource.prepare_mode_a_resources()


def original_table(wave, note):
    return lut.StreamingCoeffLUT.generate_for_frequency(wave,
        f'{resource.WAVE_NAMES[wave]}_{note}', lut.midi_to_hz(note),
        lut.get_waveform_generator(wave)[0])


def prohibit_preparation(monkeypatch):
    prohibit_basis(monkeypatch)
    monkeypatch.setattr(builder, 'build_mode_a_resource', forbidden)
    monkeypatch.setattr(builder, '_generate_table', forbidden)
    monkeypatch.setattr(resource.PreparedModeAAtlas, '__init__', forbidden)
    for name in ('midi_to_hz', 'get_waveform_generator', 'sine_q15'):
        monkeypatch.setattr(lut, name, forbidden)
    for name in ('generate_for_frequency', 'generate_from_waveform',
                 'generate_sine', 'generate_saw', 'generate_square'):
        monkeypatch.setattr(lut.StreamingCoeffLUT, name, staticmethod(forbidden))


@pytest.mark.parametrize('wave', range(5))
@pytest.mark.parametrize('note', [0, 69, 127])
def test_all_coefficients_match_original_scalar_recipe(atlas, wave, note):
    expected = original_table(wave, note)
    actual = lut.get_lut_for_note(wave, note)
    assert (actual.wave_id, actual.name, actual.target_freq_hz) == (
        expected.wave_id, expected.name, expected.target_freq_hz)
    np.testing.assert_array_equal(actual.data, expected.data)
    assert not actual.data.flags.writeable
    with pytest.raises(ValueError):
        actual.data.flags.writeable = True
    with pytest.raises(ValueError):
        actual.data[0, 0, 0] = 1


@pytest.mark.parametrize('wave', range(3))
def test_legacy_tables_keep_their_distinct_c4_recipe(atlas, wave):
    expected = getattr(lut.StreamingCoeffLUT, 'generate_' + resource.WAVE_NAMES[wave])()
    actual = lut.get_streaming_lut(wave)
    assert (actual.name, actual.target_freq_hz) == (expected.name, 261.63)
    np.testing.assert_array_equal(actual.data, expected.data)
    assert not np.array_equal(actual.data, lut.get_lut_for_note(wave, 60).data)


def test_complete_domain_and_bounded_views_without_generation(atlas, monkeypatch):
    prohibit_preparation(monkeypatch)
    lut.clear_mode_a_views()
    assert len(atlas.keys) == 643
    assert atlas.path.stat().st_size == 21072476
    for wave in range(5):
        for note in range(128):
            table = lut.get_lut_for_note(wave, note)
            assert (table.wave_id, table.name, table.data.shape) == (
                wave, f'{resource.WAVE_NAMES[wave]}_{note}', (64, 4, 64))
            assert not table.data.flags.writeable
    assert len(lut._NOTE_LUT_CACHE) == 128
    retained = lut.get_lut_for_note(4, 0)
    lut.get_lut_for_note(0, 0)
    assert lut.get_lut_for_note(4, 0) is retained
    assert (4, 1) not in lut._NOTE_LUT_CACHE  # least recently used view was evicted
    for wave in range(3):
        assert lut.get_streaming_lut(wave).data.shape == (64, 4, 64)
    lut.clear_mode_a_views()
    assert resource.prepared_mode_a_atlas() is atlas


@pytest.mark.parametrize('wave,note,canonical', [
    (0, -1000, 0), (1, 1000, 127), (np.uint32(2), np.int16(69), 69),
    (np.uint32(0xffffffff), np.uint64(127), 127)])
def test_addresses_widen_and_clamp_like_shared_pitch_law(atlas, wave, note, canonical):
    table = lut.get_lut_for_note(wave, note)
    expected = lut.get_lut_for_note(wave if wave < 5 else 0, canonical)
    assert table.wave_id == int(wave)
    np.testing.assert_array_equal(table.data, expected.data)
    assert lut.get_streaming_lut(0xffffffff) is lut.get_streaming_lut(0)


@pytest.mark.parametrize('wave', [True, np.bool_(False), 0.0, -1, 2**32, None])
def test_invalid_lookup_ids_are_rejected(atlas, wave):
    with pytest.raises(ValueError):
        lut.get_lut_for_note(wave, 60)
    with pytest.raises(ValueError):
        lut.get_streaming_lut(wave)


@pytest.fixture
def small_resource(tmp_path):
    keys = (0, 255, 640, 642)
    path = builder.build_mode_a_resource(tmp_path / 'small.bin', table_keys=keys)
    return path, keys


def test_deterministic_builder_and_explicit_partial_images(small_resource, tmp_path):
    path, keys = small_resource
    other = builder.build_mode_a_resource(tmp_path / 'other.bin', table_keys=keys)
    assert path.read_bytes() == other.read_bytes()
    with pytest.raises(ValueError, match='coverage'):
        resource.PreparedModeAAtlas(path)
    partial = resource.PreparedModeAAtlas(path, expected_keys=keys)
    with pytest.raises(ValueError, match='absent'):
        partial.table(1)


@pytest.mark.parametrize('damage', ['magic', 'version', 'count', 'recipe',
    'digest', 'directory', 'coeff', 'truncate', 'append'])
def test_rejects_corrupt_resources(small_resource, damage):
    path, keys = small_resource
    data = bytearray(path.read_bytes())
    offsets = dict(magic=0, version=8, count=12, recipe=16, digest=48,
                   directory=resource.HEADER.size, coeff=len(data)-1)
    if damage == 'truncate':
        data = data[:-1]
    elif damage == 'append':
        data += b'x'
    else:
        data[offsets[damage]] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError):
        resource.PreparedModeAAtlas(path, expected_keys=keys)


@pytest.mark.parametrize('keys', [(), (1, 1), (2, 1), (-1,), (643,), (True,), (1.5,)])
def test_invalid_directories_leave_no_files(tmp_path, keys):
    with pytest.raises(ValueError):
        builder.build_mode_a_resource(tmp_path / 'invalid.bin', table_keys=keys)
    assert not list(tmp_path.iterdir())


def test_interrupted_build_and_failed_adoption_keep_previous_resource(atlas, tmp_path):
    target = tmp_path / 'tables.bin'
    target.write_bytes(b'previous resource')
    with pytest.raises(AssertionError):
        builder.build_mode_a_resource(target, table_keys=(0,), progress=forbidden)
    assert target.read_bytes() == b'previous resource'
    assert list(tmp_path.iterdir()) == [target]
    with pytest.raises(ValueError):
        resource.prepare_mode_a_resources(target, build_if_missing=False)
    with pytest.raises(FileNotFoundError):
        resource.prepare_mode_a_resources(tmp_path / 'absent.bin', build_if_missing=False)
    assert resource.prepared_mode_a_atlas() is atlas


def test_adoption_clears_views_and_keeps_retained_arrays_valid(atlas, monkeypatch):
    old_view = lut.get_lut_for_note(3, 69)
    expected = old_view.data.copy()
    monkeypatch.setattr(resource, '_active', atlas)
    adopted = resource.prepare_mode_a_resources(atlas.path, build_if_missing=False)
    assert adopted is not atlas
    assert not lut._NOTE_LUT_CACHE
    new_view = lut.get_lut_for_note(3, 69)
    assert new_view is not old_view
    np.testing.assert_array_equal(new_view.data, expected)
    np.testing.assert_array_equal(old_view.data, expected)


@pytest.mark.parametrize('mode', [0, None])
def test_unavailable_resource_fails_before_musical_state_changes(atlas, monkeypatch, mode):
    state = VMState.create(0)
    lut.get_lut_for_note(0, 69)  # cached wrappers must not bypass the availability check
    lut.get_streaming_lut(0)
    before = state.snapshot()
    monkeypatch.setattr(resource, '_active', None)
    with pytest.raises(RuntimeError, match='Prepare'):
        vm_step_block(state, frame(coeff_osc_mode=mode))
    with pytest.raises(RuntimeError, match='Prepare'):
        lut.get_streaming_lut(0)
    assert state.snapshot() == before


@pytest.mark.parametrize('wave', [True, np.bool_(False), 0.0, -1, 2**32])
def test_invalid_native_ids_fail_before_musical_state_changes(atlas, wave):
    state = VMState.create(0)
    before = state.snapshot()
    with pytest.raises(ValueError):
        vm_step_block(state, frame(coeff_osc_mode=0, coeff_table_id=wave))
    assert state.snapshot() == before


def test_optional_table_id_retains_sine_default(atlas):
    a, b = VMState.create(0), VMState.create(0)
    np.testing.assert_array_equal(
        vm_step_block(a, frame(coeff_osc_mode=0, coeff_table_id=None))[1],
        vm_step_block(b, frame(coeff_osc_mode=0, coeff_table_id=0))[1])
    assert a.snapshot() == b.snapshot()


@pytest.mark.parametrize('note,canonical', [(-5, 0), (1000, 127)])
def test_clamped_tables_preserve_raw_note_labels_and_restoration(atlas, note, canonical):
    actual, expected = VMState.create(0), VMState.create(0)
    command = frame(coeff_osc_mode=0, coeff_table_id=1, note=note)
    for _ in range(3):
        np.testing.assert_array_equal(vm_step_block(actual, command)[1],
            vm_step_block(expected, replace(command, note=canonical))[1])
    assert actual.last_note == actual.ccs_state.note == note
    restored = VMState.restore(json.loads(json.dumps(actual.snapshot())))
    np.testing.assert_array_equal(vm_step_block(restored, command)[1],
                                  vm_step_block(actual, command)[1])
    assert restored.snapshot() == actual.snapshot()


def test_playback_cache_misses_never_open_files_or_use_transcendentals(atlas, monkeypatch):
    import builtins
    import math
    from pathlib import Path
    state = VMState.create(0)
    prohibit_preparation(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', forbidden)
        patch.setattr(builtins, 'open', forbidden)
        for module, names in [(math, ('sin', 'cos', 'exp', 'log', 'log2', 'pow')),
                              (np, ('sin', 'cos', 'exp', 'log', 'log2', 'power'))]:
            for name in names:
                patch.setattr(module, name, forbidden)
        for command in commands():
            lut.clear_mode_a_views()
            vm_step_block(state, command)
            lut.get_streaming_lut(command.coeff_table_id)


def test_unavailable_mode_a_does_not_affect_holds_rejections_or_mode_b(atlas, monkeypatch):
    from symcrash_python_ap.vm.scheduling import ExecutionPolicy
    from tests.conformance.test_scheduled_vm import packet
    state = VMState.create(0)
    vm_step_block(state, frame(coeff_osc_mode=0))
    before = state.snapshot()
    rejected = packet().to_voice_frame()
    rejected.wire_data = replace(rejected.wire_data, reject_code=1)
    monkeypatch.setattr(resource, '_active', None)
    vm_step_block(state, frame(coeff_osc_mode=0), policy=ExecutionPolicy(serviced=False))
    vm_step_block(state, rejected)
    assert state.snapshot() == before
    vm_step_block(state, frame(coeff_osc_mode=1))
    vm_step_block(state, packet(carrier_model_u8=1).to_voice_frame())


def commands():
    return [frame(coeff_osc_mode=0, coeff_table_id=wave, note=note,
                  cents=cents, is_starting=i == 0, legato_mode=True)
            for i, (wave, note, cents) in enumerate([
                (0, 0, 0), (1, 69, -99), (2, 127, 0), (3, 60, 37),
                (4, 69, 99), (0, 127, -37), (1, 0, 100), (2, 60, 0),
                (3, 69, 0), (4, 127, -199)])]


def test_cache_misses_and_json_core_restore_preserve_audio(atlas, monkeypatch, execution_path):
    frames = commands()
    originals = {(f.coeff_table_id, f.note): original_table(f.coeff_table_id, f.note) for f in frames}
    reference = VMState.create(0)
    with monkeypatch.context() as patch:
        patch.setattr(lut, 'get_lut_for_note', lambda wave, note: originals[wave, note])
        expected = [vm_step_block(reference, f)[1].copy() for f in frames]
    state = VMState.create(0)
    prohibit_preparation(monkeypatch)
    for i, f in enumerate(frames):
        lut.clear_mode_a_views()
        np.testing.assert_array_equal(vm_step_block(state, f)[1], expected[i])
        if i == 4:
            restored = [VMState.restore(json.loads(json.dumps(state.snapshot()))),
                        vmstate_from_bytes(vmstate_to_bytes(state))]
            for copy in restored:
                for later, output in zip(frames[i+1:], expected[i+1:]):
                    lut.clear_mode_a_views()
                    np.testing.assert_array_equal(vm_step_block(copy, later)[1], output)
                assert copy.snapshot() == reference.snapshot()
    assert state.snapshot() == reference.snapshot()


@pytest.mark.parametrize('binary', [False, True], ids=['json', 'core'])
def test_fresh_process_restore_needs_no_generation(atlas, tmp_path, binary):
    state = VMState.create(0)
    frames = commands()
    for f in frames[:5]:
        vm_step_block(state, f)
    path = tmp_path / ('checkpoint.bin' if binary else 'checkpoint.json')
    if binary:
        path.write_bytes(vmstate_to_bytes(state))
    else:
        path.write_text(json.dumps(state.snapshot()))
    expected = [vm_step_block(state, f)[1].tolist() for f in frames[5:]]
    code = '''
import json, sys
from pathlib import Path
from pytest import MonkeyPatch
from symcrash_python_ap.vm import VMState, vm_step_block, coeff_lut_streaming as lut
from symcrash_python_ap.vm import mode_a_resource_builder as builder
from symcrash_python_ap.voice.voice_state_codec import vmstate_from_bytes
from tests.conformance.test_mode_a_resource import commands, forbidden, prohibit_preparation
builder.build_mode_a_resource = builder._generate_table = forbidden
path = Path(sys.argv[1])
state = vmstate_from_bytes(path.read_bytes()) if path.suffix == '.bin' else VMState.restore(json.loads(path.read_text()))
with MonkeyPatch.context() as patch:
    prohibit_preparation(patch)
    output = []
    for frame in commands()[5:]:
        lut.clear_mode_a_views()
        output.append(vm_step_block(state, frame)[1].tolist())
print(json.dumps(output))
'''
    child = subprocess.run([sys.executable, '-c', code, str(path)], check=True, capture_output=True, text=True)
    assert json.loads(child.stdout) == expected


@pytest.mark.parametrize('kind', ['master', 'workbench', 'executor', 'orchestrator', 'unified'])
def test_hosts_prepare_before_late_voice_allocation(atlas, monkeypatch, tmp_path, kind):
    from tests.conformance.test_native_preset_transport import source, routes
    monkeypatch.setattr(resource, '_active', None)
    monkeypatch.setattr(resource, 'PreparedModeAAtlas', lambda path: atlas)
    session = None
    if kind == 'master':
        from tests.conformance.test_host_preset_ownership import host as make_host
        host = make_host(tmp_path, 'host', routes(source('saw', osc_mode=0))[0])
    elif kind == 'workbench':
        from symcrash_workbench.session import WorkbenchSession
        from symcrash_workbench.streaming import StreamingVoice
        from symcrash_python_ap.instrument_ir import save_instrument
        path = tmp_path / 'preset.yaml'
        save_instrument(source('saw', osc_mode=0), path)
        session = WorkbenchSession(path, autoplay=False, log_dir=tmp_path)
        host = StreamingVoice(session)
    elif kind == 'executor':
        from symcrash_python_ap.vm.executor import VMExecutor
        host = VMExecutor()
    elif kind == 'orchestrator':
        from symcrash_python_ap.voice.orchestrator import BlockOrchestrator
        host = BlockOrchestrator(n_voices=4)
    else:
        from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
        host = UnifiedOutputStage()
    assert resource.prepared_mode_a_atlas() is atlas
    monkeypatch.setattr(resource, 'PreparedModeAAtlas', type(atlas))
    prohibit_preparation(monkeypatch)
    lookups = []
    lookup = lut.get_lut_for_note
    def observing(wave, note):
        lookups.append((wave, note))
        return lookup(wave, note)
    monkeypatch.setattr(lut, 'get_lut_for_note', observing)
    try:
        if kind in ('master', 'workbench'):
            host.note_on(69)
            for _ in range(12):
                lut.clear_mode_a_views()
                host.render_block()
        else:
            for i, f in enumerate(commands()):
                lut.clear_mode_a_views()
                if kind == 'executor':
                    host.step_voice(0, f)
                elif kind == 'orchestrator':
                    from symcrash_python_ap.frames import VoiceFramePacketV0b, VoiceFlags
                    host.update_frame(0, VoiceFramePacketV0b(voice_id=0, frame_seq=i+1,
                        flags=VoiceFlags.GATE, note_u8=f.note, cents_i16=f.cents))
                    host.step_block()
                else:
                    host.render_block(f)
        assert lookups, 'The host must actually exercise Mode A'
    finally:
        if session is not None:
            host.stop()
            session.close()


def test_frozen_recipe_matches_offline_generator():
    child = subprocess.run([sys.executable, 'tools/generate_mode_a_recipe.py', '--check'],
                           check=True, capture_output=True, text=True)
    assert 'matches' in child.stdout
