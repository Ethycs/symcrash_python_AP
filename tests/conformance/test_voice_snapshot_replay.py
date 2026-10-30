"""Per-voice time checkpoints: compare future behavior, not just codec output."""
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import IntEnum
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.control_frame import ControlFrame, VoiceControlPatch, ControlItem
from symcrash_python_ap.control_registry import ControlID, ControlRegistry
from symcrash_python_ap.envelope_scheduler import EnvelopePhase
from symcrash_python_ap.instrument_ir import ADSR, EnvelopeBinding, ControlConfig
from symcrash_python_ap.library.preset_compiler import PresetCompiler
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceState, VoiceStateWithProgram
from symcrash_python_ap.vm import VMState, Tables, vm_step_block
from symcrash_python_ap.vm import hotpath_jit as hot
from symcrash_python_ap.vm.harmonic_bank import LFOState, LFORouting, LFOTarget, LFOWaveform, LFOSyncMode


ROOT = Path(__file__).resolve().parents[2]


def structural(value):
    """Independent walk of actual runtime fields, including cached/dynamic data."""
    if is_dataclass(value):
        assert not set(vars(value)) - {f.name for f in fields(value)}
        return {f.name: structural(getattr(value, f.name)) for f in fields(value) if f.name != 'program'}
    if isinstance(value, dict):
        return {k: structural(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [structural(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, IntEnum):
        return int(value)
    return value


def preset():
    result = PresetCompiler().compile_file(str(ROOT/'presets/instruments/dx_marimba_v0.yaml'))
    result.amp_env.attack_ms = 12
    result.amp_env.decay_ms = 40
    result.amp_env.sustain = .63
    result.amp_env.release_ms = 30
    result.amp_key_scale_rate = .7
    result.amp_velocity_sens = .55
    result.material.beta_ramp_blocks = 25
    result.control_config = ControlConfig(brightness=.72, density=.31)
    result.envelope_bindings = [EnvelopeBinding('mod', ADSR(15,50,.25,18),
        ['FM_OP2','FM_OP4'], velocity_sens=.4, key_scale_rate=.8)]
    result.lfos = []
    for slot, shape in enumerate((LFOWaveform.TRIANGLE, LFOWaveform.SINE)):
        lfo = LFOState(id=f'lfo{slot}')
        lfo.configure_lfo(rate_hz=2.3+slot*4.1, depth_q15=11000, waveform=shape,
            sync_mode=LFOSyncMode.GATE_RESET, delay_ms=3, ramp_ms=23, arc_slot=slot,
            routings=[LFORouting(LFOTarget.PITCH, 36), LFORouting(LFOTarget.FILTER_CUTOFF, -4000)])
        result.lfos.append(lfo)
    return result


def engine(p=None):
    result = VoiceEngine(max_voices=2)
    result.allocate_voice_from_preset(p or preset(), voice_id=0)
    result.note_on_simple(0, note=79, velocity_q15=23000, cents=-21)
    result.voices[0].channel = 7
    result.voices[0].controls.detune_override = 33
    return result


def controls(**values):
    registry = ControlRegistry()
    return ControlFrame(voice_patches=[VoiceControlPatch(0, [ControlItem(
        int(ControlID[name]), registry.get_entry(ControlID[name]).value_type, value)
        for name, value in values.items()])])


@pytest.fixture(params=[False, True], ids=['python','jit'])
def execution_path(request):
    previous = hot.jit_enabled()
    if request.param and not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    hot.set_jit_enabled(request.param)
    try:
        yield
    finally:
        hot.set_jit_enabled(previous)


@pytest.mark.parametrize('checkpoint', ['start','attack','decay','sustain','pending_release','release','free'])
def test_simple_replay_preserves_actual_runtime_frames_and_vm_audio(checkpoint, execution_path):
    original = engine()
    vm, tables = VMState.create(0), Tables()
    count = {'start':0, 'attack':3, 'decay':12, 'sustain':50,
             'pending_release':4, 'release':4, 'free':4}[checkpoint]
    original.consume_control_frame(controls(PITCH_BEND=120, MORPH=17000))
    for _ in range(count):
        frame = original.step_voice_simple(0)
        if frame is not None:
            vm, _ = vm_step_block(vm, frame, tables)
    if checkpoint in ('pending_release','release','free'):
        original.note_off_simple(0)
        for _ in range(3 if checkpoint == 'release' else 60 if checkpoint == 'free' else 0):
            frame = original.step_voice_simple(0)
            if frame is not None:
                vm, _ = vm_step_block(vm, frame, tables)
    snapshot = json.loads(json.dumps(original.voices[0].snapshot(), allow_nan=False))
    restored = VoiceEngine(max_voices=2)
    restored.voices[0] = VoiceStateWithProgram.restore(snapshot)
    replay_vm = VMState.restore(json.loads(json.dumps(vm.snapshot())))
    assert structural(restored.voices[0]) == structural(original.voices[0])
    assert restored.voices[0].compiled_preset is not original.voices[0].compiled_preset
    heard = False
    for tick in range(55):
        for e in (original, restored):
            if tick == 2:
                e.consume_control_frame(controls(ATK_BLOCKS=7, DCY_BLOCKS=9, SUS_Q15=12345,
                                                REL_BLOCKS=13, PITCH_BEND=-90))
            if tick == 5:
                e.note_off_simple(0)
            if tick == 19:
                e.note_on_simple(0, note=43, velocity_q15=11000, cents=41)
            if tick == 35:
                e.note_off_simple(0)
        a, b = original.step_voice_simple(0), restored.step_voice_simple(0)
        assert a == b, (checkpoint,tick)
        assert structural(restored.voices[0]) == structural(original.voices[0])
        if a is not None:
            vm, samples = vm_step_block(vm, a, tables)
            expected = samples.copy()
            replay_vm, actual = vm_step_block(replay_vm, b, tables)
            np.testing.assert_array_equal(actual, expected)
            heard |= bool(np.any(actual))
    assert heard


def test_saved_live_envelope_cache_is_not_recomputed_from_authored_ms():
    source = engine()
    source.consume_control_frame(controls(ATK_BLOCKS=101, DCY_BLOCKS=37, REL_BLOCKS=91, SUS_Q15=4321))
    for _ in range(6):
        source.step_voice_simple(0)
    state = VoiceStateWithProgram.restore(json.loads(json.dumps(source.voices[0].snapshot())))
    assert state.amp_envelope._attack_blocks == 101
    assert state.amp_envelope._decay_blocks == 37
    assert state.amp_envelope._release_blocks == 91
    assert state.amp_envelope._sustain_override_q15 == 4321
    assert state.amp_envelope.phase == EnvelopePhase.ATTACK
    assert state.amp_envelope.phase_block == 6
    assert structural(state.amp_envelope) == structural(source.voices[0].amp_envelope)


def test_lfo_state_is_per_voice_and_does_not_mutate_the_preset():
    p = preset()
    authored = structural(p)
    poly, single = engine(p), engine(deepcopy(p))
    poly.allocate_voice_from_preset(p, voice_id=1)
    poly.note_on_simple(1, note=48)
    assert poly.voices[0].lfos[0] is not poly.voices[1].lfos[0]
    for tick in range(25):
        assert poly.step_voice_simple(0) == single.step_voice_simple(0)
        poly.step_voice_simple(1)
        if tick == 8:
            poly.note_on_simple(1, note=84)
        assert structural(poly.voices[0].lfos) == structural(single.voices[0].lfos)
        assert structural(p) == authored


@pytest.mark.parametrize('path', sorted((ROOT/'presets/instruments').glob('*.yaml')), ids=lambda p:p.stem)
def test_every_shipped_preset_voice_survives_json_checkpoint(path):
    p = PresetCompiler().compile_file(str(path))
    original = engine(p)
    for _ in range(5):
        original.step_voice_simple(0)
    state = original.voices[0]
    restored = VoiceStateWithProgram.restore(json.loads(json.dumps(state.snapshot(), allow_nan=False)))
    assert structural(restored) == structural(state)
    replay = VoiceEngine()
    replay.voices[0] = restored
    for _ in range(4):
        assert original.step_voice_simple(0) == replay.step_voice_simple(0)


def test_snapshot_and_restore_own_detached_resources_and_runtime_buffers():
    source = engine()
    source.step_voice_simple(0)
    saved = source.voices[0].snapshot()
    before = deepcopy(saved)
    a = VoiceStateWithProgram.restore(saved)
    b = VoiceStateWithProgram.restore(saved)
    a.compiled_preset.fm6.op_levels[0] = 0
    a.compiled_preset.control_config.temperament.stickiness_q15 = 99
    a.amp_envelope.spec.attack_ms = 999
    a.lfos[0].routings[0].amount_q15 = 0
    a.arc_phase_q16[0] = 0
    a.controls.values[123] = 45
    a.envelope_matrix['mod'][0].current_q15 = 555
    assert saved == before
    assert structural(b) == structural(source.voices[0])


@pytest.mark.parametrize('change', [
    lambda s:s.update(snapshot_version=99),
    lambda s:s.update(snapshot_version=True),
    lambda s:s.pop('lfos'),
    lambda s:s.update(invented=0),
    lambda s:s.update(voice_id=-1),
    lambda s:s.update(voice_id=True),
    lambda s:s.update(engine_mode=99),
    lambda s:s.update(compiled_preset=None),
    lambda s:s.update(arc_phase_q16=[0]),
    lambda s:s.update(amp_envelope={'kind':'Path', 'fields':{}}),
    lambda s:s['amp_envelope']['fields'].update(phase_block='6'),
    lambda s:s['amp_envelope']['fields']['spec']['fields'].update(attack_ms=float('nan')),
    lambda s:s['controls']['fields']['values'].update(items=[[1,2],[1,3]]),
    lambda s:s['lfos'][0]['fields'].update(arc_slot=99),
    lambda s:s['lfos'][0]['fields'].update(elapsed_samples=-1),
    lambda s:s['amp_envelope']['fields'].update(_attack_blocks=0),
    lambda s:s['amp_envelope']['fields'].update(release_level_q15=-1),
    lambda s:s['envelope_matrix']['items'][0][1].update(items=[1]),
])
def test_malformed_checkpoints_fail_without_mutating_source(change):
    source = engine()
    before = structural(source.voices[0])
    snapshot = source.voices[0].snapshot()
    change(snapshot)
    with pytest.raises(ValueError):
        VoiceStateWithProgram.restore(snapshot)
    assert structural(source.voices[0]) == before


def test_unregistered_dynamic_fields_fail_instead_of_silently_disappearing():
    source = engine()
    source.voices[0].unknown_history = [1,2,3]
    with pytest.raises(ValueError, match='Unregistered runtime fields'):
        source.voices[0].snapshot()


@pytest.mark.parametrize('snapshot', [None, [], 'voice'])
def test_invalid_root_is_rejected(snapshot):
    with pytest.raises(ValueError, match='dictionary'):
        VoiceState.restore(snapshot)


def test_versioned_cmmm_checkpoint_requires_the_custom_program_resource():
    from symcrash_python_ap.voice.voice_engine import InstrumentProgram
    source = VoiceEngine()
    source.programs[23456] = InstrumentProgram(program_id=23456, sustain_level_q15=7890)
    source.allocate_voice(program_id=23456)
    source.note_on(0)
    source.step_voice(0)
    snapshot = source.voices[0].snapshot()
    with pytest.raises(ValueError, match='Program 23456'):
        VoiceStateWithProgram.restore(snapshot)
    restored = VoiceStateWithProgram.restore(snapshot, programs=source.programs)
    assert restored.program is source.programs[23456]
    assert structural(restored) == structural(source.voices[0])


def test_all_cmmm_fields_are_preserved_including_current_events_and_lfos():
    from symcrash_python_ap.voice.voice_engine import OverlaySpec
    source = VoiceEngine()
    source.allocate_voice(program_id=1001)
    source.note_on(0)
    state = source.voices[0]
    state.lfos = preset().lfos
    state.detune_cents = -98
    state.eta_q15, state.harmonics_q15, state.master_gain_q15 = 1234, 4567, 9900
    state.overlay_spec = OverlaySpec(duration_blocks=13, family_slew_mod_q15=[1111]*8,
                                    event_suppress_mask=3, tail_cap_mod_q15=3456)
    state.overlay_active, state.overlay_blocks_remaining = True, 11
    for _ in range(4):
        source.step_voice(0)
    restored = VoiceStateWithProgram.restore(json.loads(json.dumps(state.snapshot())))
    assert structural(restored) == structural(state)
    replay = VoiceEngine()
    replay.voices[0] = restored
    for _ in range(8):
        assert replay.step_voice(0) == source.step_voice(0)
        assert structural(restored) == structural(state)


def test_legacy_unversioned_cmmm_snapshot_still_loads():
    source = VoiceEngine()
    source.allocate_voice(program_id=1001)
    source.note_on(0)
    source.step_voice(0)
    saved = source.voices[0]._snapshot_legacy()
    assert 'snapshot_version' not in saved
    restored = VoiceStateWithProgram.restore(json.loads(json.dumps(saved)))
    assert restored.program is source.voices[0].program
    replay = VoiceEngine()
    replay.voices[0] = restored
    for _ in range(5):
        assert replay.step_voice(0) == source.step_voice(0)


def test_simple_checkpoint_replays_in_a_fresh_process_without_preset_files(tmp_path):
    source = engine()
    for _ in range(12):
        source.step_voice_simple(0)
    path = tmp_path/'voice.json'
    path.write_text(json.dumps(source.voices[0].snapshot()), encoding='utf-8')
    expected = [structural(source.step_voice_simple(0)) for _ in range(6)]
    script = '''
import json, sys
from dataclasses import asdict
from pathlib import Path
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
engine = VoiceEngine()
engine.voices[0] = VoiceStateWithProgram.restore(json.loads(Path(sys.argv[1]).read_text()))
print(json.dumps([asdict(engine.step_voice_simple(0)) for _ in range(6)]))
'''
    run = subprocess.run([sys.executable, '-c', script, str(path)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(run.stdout) == json.loads(json.dumps(expected))
