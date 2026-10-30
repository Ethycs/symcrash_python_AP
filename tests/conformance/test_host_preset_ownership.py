"""SynthMaster must render the parameters owned by each individual voice."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from symcrash_python_ap import synth_master as master
from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
from tests.conformance.test_live_unison import authored
from tests.conformance.test_native_preset_transport import source, routes
from tests.conformance.test_scheduled_vm import schedule
from tests.conformance.test_vm_snapshot_replay import execution_path


def host(tmp_path, name, preset=None):
    result = SynthMaster(SynthConfig(n_voices=8,
        output_dir=str(tmp_path / name / 'audio'),
        library_dir=str(tmp_path / name / 'library')))
    result.disable('cockpit')
    if preset is not None:
        result.load_compiled_preset(preset)
    return result


@pytest.mark.parametrize('route', [0, 1], ids=['native', 'capsule'])
@pytest.mark.parametrize('mode,wave', [(0, 'sine'), (0, 'saw'), (1, 'sine'),
    (1, 'saw'), (1, 'square'), (1, 'triangle')])
def test_emitted_frames_and_audio_reach_live_vm_without_defaults(
        tmp_path, monkeypatch, route, mode, wave, execution_path):
    ir = source(wave, osc_mode=mode)
    ir.compile.material.beta_start_q15 = 0
    ir.compile.material.beta_target_q15 = 12000
    ir.compile.material.beta_ramp_blocks = 8
    preset = routes(ir)[route]
    actual, reference = [host(tmp_path, name, preset) for name in ('actual', 'reference')]
    for owner in (actual, reference):
        owner.note_on(69, 100)
    original = master.vm_step_block
    captured = []
    def record(state, frame, **kwargs):
        captured.append(asdict(frame))
        return original(state, frame, **kwargs)
    audible = 0
    for block in range(36):
        if block == 20:
            actual.note_off(69)
            reference.note_off(69)
        emitted = reference.voice_engine.step_block()
        frames = [item.frame for item in emitted.render_batch]
        expected = [asdict(frame) for frame in frames]
        assert all(frame.coeff_osc_mode == mode and frame.partial_amps_q15 == [] for frame in frames)
        if block == 0:
            assert frames[0].beta_target_q15 == 0
        captured.clear()
        with monkeypatch.context() as context:
            context.setattr(master, 'vm_step_block', record)
            actual_audio = actual.render_block()[:2]
        assert captured == expected
        reference_audio = reference.render_frames(frames, params=emitted.param_frame)[:2]
        np.testing.assert_array_equal(actual_audio, reference_audio)
        audible += np.count_nonzero(actual_audio)
    assert audible
    assert actual._voice_slots[0].vm_state.snapshot() == reference._voice_slots[0].vm_state.snapshot()


@pytest.mark.parametrize('mode', [0, 1, 3])
def test_each_unison_copy_executes_its_authored_mode(tmp_path, monkeypatch, mode, execution_path):
    preset = routes(authored(mode=mode))[0]
    synth = host(tmp_path, 'unison', preset)
    synth.note_on(69, 100)
    original = master.vm_step_block
    seen = []
    def record(state, frame, **kwargs):
        seen.append((frame.voice_id, frame.coeff_osc_mode))
        return original(state, frame, **kwargs)
    monkeypatch.setattr(master, 'vm_step_block', record)
    for _ in range(3):
        synth.render_block()
    assert seen == [(voice, mode) for _ in range(3) for voice in range(3)]


@pytest.mark.parametrize('release', [False, True], ids=['held-note', 'releasing-note'])
@pytest.mark.parametrize('native', [False, True], ids=['legacy-compiled', 'native-compiled'])
def test_voice_ownership_survives_preset_change(tmp_path, release, native):
    first = routes(source('sine', osc_mode=0))[0]
    first.material.beta_start_q15 = first.material.beta_target_q15 = 0
    if not native:
        first = replace(first, native_carrier=None, harmonic_amps_q15=[])
    second = routes(source('saw'))[0]
    second.material.beta_start_q15 = second.material.beta_target_q15 = 16000
    synth = host(tmp_path, 'change', first)
    first_id = synth.note_on(69, 100)
    synth.render_block()
    if release:
        synth.note_off(69)
    synth.load_compiled_preset(second)
    second_id = synth.note_on(72, 100)
    for _ in range(3):
        synth.render_block()
        old = synth._last_render_frames[first_id]
        new = synth._last_render_frames[second_id]
        assert old.coeff_osc_mode == 0 and old.partial_amps_q15 == []
        assert old.beta_target_q15 == 0
        assert new.coeff_osc_mode == 1 and new.beta_target_q15 == 16000
        assert new.partial_amps_q15 == []
    assert synth.voice_engine.voices[first_id].compiled_preset is first


def test_unowned_voice_keeps_fallbacks_with_another_preset_selected(tmp_path):
    synth = host(tmp_path, 'fallback')
    voice_id = synth.note_on(60, 100)
    assert synth.voice_engine.voices[voice_id].compiled_preset is None
    synth.load_compiled_preset(routes(source('sine', osc_mode=0))[0])
    synth._harmonic_amps = [21000, 7000]
    synth._beta_target_q15 = 9000
    synth.set_osc_mode('mode_b')
    synth.render_block()
    rendered = synth._last_render_frames[voice_id]
    assert rendered.coeff_osc_mode == 1
    assert rendered.partial_amps_q15 == [21000, 7000]
    emitted = synth.voice_engine._render_frames[voice_id]
    assert emitted is not rendered and emitted.partial_amps_q15 == []
    assert emitted.coeff_osc_mode == 0


def test_held_fallback_frame_ignores_later_global_defaults(tmp_path):
    synth = host(tmp_path, 'held')
    synth.set_osc_mode('mode_a')
    synth._beta_target_q15 = 0
    voice_id = synth.note_on(60, 100)
    synth.render_block()
    before = deepcopy(synth._last_render_frames[voice_id])
    state = synth._voice_slots[voice_id].vm_state.snapshot()
    synth.set_osc_mode('mode_b')
    synth._beta_target_q15 = 23000
    synth._harmonic_amps[:] = [12345]
    for _ in range(3):
        synth.render_block(schedule=schedule())
        assert synth._last_render_frames[voice_id] == before
        assert synth._voice_slots[voice_id].vm_state.snapshot() == state
