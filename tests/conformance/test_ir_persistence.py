"""Saving an authored instrument must preserve the sound and editor data."""
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap import instrument_ir as ir
from symcrash_python_ap.editor import EditorSession
from symcrash_python_ap.editor.commands import SetParam
from symcrash_python_ap.library import ir_codec
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir
from symcrash_python_ap.vm import VMState, vm_step_block
from tests.conformance.test_ir_event_compilation import author, engine
from tests.conformance.test_vm_snapshot_replay import execution_path


PRESETS = sorted((Path(__file__).resolve().parents[2] / 'presets/instruments').glob('*.yaml'))


@pytest.mark.parametrize('path', PRESETS, ids=lambda path: path.stem)
@pytest.mark.parametrize('extension', ['.yaml', '.json'])
def test_every_shipped_preset_file_preserves_all_model_and_compiled_fields(path, extension, tmp_path):
    source = ir.load_instrument(path)
    original = deepcopy(source)
    target = tmp_path / ('instrument' + extension)
    ir.save_instrument(source, target)
    saved = target.read_bytes()
    restored = ir.load_instrument(target)
    assert source == original == restored
    assert compiled_preset_from_ir(source) == compiled_preset_from_ir(restored)
    ir.save_instrument(restored, target)
    assert target.read_bytes() == saved
    assert b'!!python' not in saved


def rich_instrument():
    source = ir.load_instrument_from_dict(author())
    source.meta.name = 'Mélodie 雨'
    source.layers.L0_graph.macros = [ir.Macro('M_COLOR', 'Color', (-1.0, 1.0), 0.125)]
    source.layers.L1_lift = ir.L1Lift(
        gesture_basis={'Pressure': ir.GestureSource((0, 1), 'aftertouch')},
        macro_semantics={'M_COLOR': ir.MacroSemantic([ir.MacroAffect('intent.tone', 'linear')])},
        control=ir.ControlConfig(brightness=0.72, temperament=ir.TemperamentVector(reactivity_q15=12345)),
        intent_mapping=ir.IntentMappingConfig(mode='custom', custom_map={'tone': 'Pressure'}),
        family_mapping=ir.FamilyMappingConfig(mode='custom', custom_map={'tone': 'energy'}),
        fm6_morph=ir.FM6MorphConfig(topo_a=2, topo_b=7, blend_q15=123, blend_source='macro(M_COLOR)'),
        vibrato=ir.VibratoParams(rate_hz=2.5, depth_amp=0.3),
        body_warble=ir.BodyWarbleParams(depth=0.4),
        attack_texture=ir.AttackTextureParams(bright_amount=0.5, harmonics=[3, 7]))
    source.layers.L2_regimes.transitions = [ir.RegimeTransition('R1_ATTACK', 'R2_STABLE',
        'dwell_done', ir.Hysteresis(0.7, 0.3))]
    source.layers.L2_regimes.regime_graph_spec = ir.RegimeGraphSpec(start_regime='R1_ATTACK',
        edges=[ir.EdgeSpec('R1_ATTACK', 'R2_STABLE', guard='dwell_done',
            hysteresis=ir.Hysteresis(0.6, 0.2), overlay='accent')])
    source.layers.L2_regimes.envelope_overrides = {'R2_STABLE': [
        ir.RegimeEnvelopeOverride('amp', ir.ADSR(1, 2, 0.3, 4), 12000)]}
    source.layers.L3_cmmm.state.variables = {
        'counter': ir.StateVariable('u32', 9),
        'table': ir.StateVariable('json', {'steps': [0, 1, None], 'optional': None})}
    source.layers.L3_cmmm.inputs = ir.CmmmInputs(['Pressure'], {'velocity': 'q15'}, 'u64')
    source.layers.L3_cmmm.outputs = ir.CmmmOutputs(['gain_q15', 'event_mask'])
    source.layers.L3_cmmm.transition_kernel = [ir.TransitionRule(name='risk', from_regime='R2_STABLE',
        to='R4_GLITCH', type='weighted', when='risk > 0.5',
        choices=[ir.TransitionChoice('R4_GLITCH', 0.3), ir.TransitionChoice('R2_STABLE', 0.7)],
        prng_draw_site='risk_42')]
    source.layers.L3_cmmm.overlays = [ir.Overlay(id='accent', enable_when='risk > 0.5',
        add=ir.OverlayAdd(families={'energy': 0.3}, events=[ir.EventPolicy(lane=5,
            condition='gate_rise', payload_template=ir.EventPayloadTemplate(variant=1, p0=42))]),
        suppression_rules=[ir.SuppressionRule('risk > 0.8', ['settling'])])]
    source.compile.map_library = {'family': 'custom-v2'}
    source.compile.completion_policy = ir.CompletionPolicy('derive', 'derive', 'derive', ['FineClamp'],
        ir.TreeTarget('harmonic', 'percussive', 0.4))
    source.compile.degrade_profile.top_group = ir.DegradeGroup(['DiffuseUp'], 'first')
    source.compile.frame_emitter_policy.reject_on = ['unknown_operator']
    source.compile.material = ir.MaterialConfig(material_id=8, beta_target_q15=22345)
    source.compile.safety = ir.SafetyConfig(slew_limit_q15=1234)
    return source


@pytest.mark.parametrize('extension', ['.yaml', '.JSON'])
def test_full_tower_graph_rules_overlays_compile_settings_and_unicode_survive(extension, tmp_path):
    source = rich_instrument()
    original = deepcopy(source)
    target = tmp_path / ('rich' + extension)
    ir.save_instrument(source, target)
    assert ir.load_instrument(target) == original == source
    data = ir.instrument_to_dict(source)['instrument_ir_v0']
    assert data['layers']['L2_regimes']['transitions'][0]['from'] == 'R1_ATTACK'
    assert data['layers']['L3_cmmm']['transition_kernel'][0]['from'] == 'R2_STABLE'
    assert '_control_compat' not in data
    data['layers']['L0_graph']['macros'][0]['default'] = 0.75
    assert source == original  # Returned authoring data is detached.


@pytest.mark.parametrize('extension', ['.yaml', '.json'])
def test_every_node_parameter_including_deprecated_modulation_is_preserved(extension, tmp_path):
    params = ir.NodeParams()
    optional = dict(waveform='sine', osc_mode=0, waveform_type=0, role='audible', archetype='harmonic',
        harmonics=ir.HarmonicProfile([]), cutoff_hz=4321.25, vibrato=ir.VibratoParams(depth_amp=0.2),
        body_warble=ir.BodyWarbleParams(depth=0.3), attack_texture=ir.AttackTextureParams(bright_amount=0.4),
        color='pink', noise_level=0.125, mode='hp', adsr=ir.ADSR(2, 3, 0.5, 8),
        algorithm=17, feedback=3, post_filter_hz=5432.25, op_ratios=[0, 1, 2, 3, 4, 5],
        op_levels=[0, 1, 2, 32765, 32766, 32767], mod_index=1.5)
    for item in fields(params):
        old = getattr(params, item.name)
        if old is None:
            value = optional[item.name]
        elif type(old) is bool:
            value = not old
        elif type(old) is int:
            value = old + 1
        else:
            value = old + 0.125
        setattr(params, item.name, value)
    source = ir.InstrumentIR(layers=ir.Layers(L0_graph=ir.L0Graph(nodes=[ir.L0Node('all', 'Osc', params)])))
    target = tmp_path / ('node' + extension)
    ir.save_instrument(source, target)
    restored = ir.load_instrument(target)
    assert restored == source
    raw = ir.instrument_to_dict(source)['instrument_ir_v0']['layers']['L0_graph']['nodes'][0]['params']
    assert raw['op_ratios_q16'] == [0, 1, 2, 3, 4, 5]
    assert raw['harmonics'] == {'amplitudes': []}


@pytest.mark.parametrize('name,cls', [('wavetable_spec', ir.WavetableSpec),
    ('granular_spec', ir.GranularSpec), ('warp_spec', ir.WarpSpec)])
@pytest.mark.parametrize('extension', ['.yaml', '.json'])
def test_populated_advanced_l1_specifications_preserve_every_field(name, cls, extension, tmp_path):
    spec = cls()
    for item in fields(spec):
        value = getattr(spec, item.name)
        if value is None:
            if item.name == 'warp_table':
                value = [i * 128 - 16384 for i in range(256)]
            elif item.name in ('atlas_index', 'prng_seed'):
                value = 42
            elif item.name.endswith('_hz'):
                value = 5432.5
            else:
                value = 'authored-resource'
        elif type(value) is bool:
            value = not value
        elif type(value) in (int, float):
            value += 1 if type(value) is int else 0.125
        else:
            value = 'custom-' + value
        setattr(spec, item.name, value)
    source = ir.InstrumentIR()
    setattr(source.layers.L1_lift, name, spec)
    path = tmp_path / ('advanced' + extension)
    ir.save_instrument(source, path)
    restored = ir.load_instrument(path)
    assert restored == source and getattr(restored.layers.L1_lift, name) == spec


@pytest.mark.parametrize('extension', ['.yaml', '.json'])
def test_lfo_exact_units_disabled_state_and_every_routing_survive(extension, tmp_path):
    lfo = ir.LFOState(id='precise', waveform=ir.LFOWaveform.SAW_UP,
        sync_mode=ir.LFOSyncMode.LEGATO_RESET, rate_hz=3.14159, enabled=False,
        phase=0xF1234567, phase_inc=280001, depth_q15=32123, arc_slot=6,
        delay_samples=101, ramp_samples=103, elapsed_samples=107, current_depth_scale=12345,
        depth_pitch_cents=17, depth_amp_q15=19, depth_timbre_q15=23,
        routings=[ir.LFORouting(target, (-1 if i % 2 else 1) * (12345 + i))
                  for i, target in enumerate(ir.LFOTarget)])
    source = ir.InstrumentIR(modulators=[lfo])
    target = tmp_path / ('lfo' + extension)
    ir.save_instrument(source, target)
    assert ir.load_instrument(target) == source
    raw = ir.instrument_to_dict(source)['instrument_ir_v0']['modulators']['lfos'][0]
    assert raw['delay_samples'] == 101 and 'delay_ms' not in raw
    assert raw['targets'][1]['amount_q15'] == -12346


def render(preset):
    producer, state, output, frames = engine(preset), VMState.create(0), [], []
    for block in range(24):
        if block == 16:
            producer.note_off_simple(0)
        frame = producer.step_voice_simple(0)
        if frame is None:
            break
        frames.append(frame)
        output.append(vm_step_block(state, frame)[1].copy())
    return frames, np.concatenate(output)


@pytest.mark.parametrize('name', ['dx_marimba_v0', 'pwm_pad_v0', 'saw_lead_v0', 'authored_events'])
def test_reloaded_instruments_reproduce_actual_frames_and_audio(name, execution_path, tmp_path):
    source = (ir.load_instrument_from_dict(author()) if name == 'authored_events' else
              ir.load_instrument(next(p for p in PRESETS if p.stem == name)))
    path = tmp_path / 'render.yaml'
    ir.save_instrument(source, path)
    a, audio_a = render(compiled_preset_from_ir(source))
    b, audio_b = render(compiled_preset_from_ir(ir.load_instrument(path)))
    assert a == b and np.any(audio_a)
    np.testing.assert_array_equal(audio_a, audio_b)


def test_saved_authored_event_instrument_renders_identically_in_a_fresh_process(tmp_path):
    source = ir.load_instrument_from_dict(author())
    expected = hashlib.sha256(render(compiled_preset_from_ir(source))[1].tobytes()).hexdigest()
    target = tmp_path / 'portable.json'
    ir.save_instrument(source, target)
    code = '''import hashlib, sys
from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir
from tests.conformance.test_ir_persistence import render
print(hashlib.sha256(render(compiled_preset_from_ir(load_instrument(sys.argv[1])))[1].tobytes()).hexdigest())
'''
    result = subprocess.run([sys.executable, '-c', code, str(target)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected


def test_editor_edit_save_reload_and_undo_preserve_the_sound(tmp_path):
    session = EditorSession(ir.load_instrument_from_dict(author()))
    before = deepcopy(session.ir)
    assert session.execute(SetParam('layers.L0_graph.nodes[0].params.detune_cents', 37.5)).ok
    target = tmp_path / 'edited.yaml'
    assert session.is_dirty and session.save(target) and not session.is_dirty
    reopened = EditorSession.from_preset(target)
    assert reopened.ir == session.ir and reopened.save_path == target
    assert reopened.ir.layers.L0_graph.nodes[0].params.detune_cents == 37.5
    assert session.undo().ok
    session.save()
    assert ir.load_instrument(target) == before


@pytest.mark.parametrize('failure', ['dump', 'flush', 'replace', 'validation'])
def test_failed_save_preserves_existing_file_editor_dirty_state_and_cleans_temporary(failure, tmp_path, monkeypatch):
    session = EditorSession(ir.load_instrument_from_dict(author()))
    session.execute(SetParam('meta.name', 'changed'))
    target = tmp_path / 'existing.yaml'
    target.write_bytes(b'preserve original bytes\n')

    def fail(*args, **kwargs):
        raise OSError('injected save failure')

    if failure == 'dump': monkeypatch.setattr(ir.yaml, 'safe_dump', fail)
    if failure == 'flush': monkeypatch.setattr(ir_codec.os, 'fsync', fail)
    if failure == 'replace': monkeypatch.setattr(ir_codec.os, 'replace', fail)
    if failure == 'validation': session.ir.layers.L0_graph.nodes[0].params.detune_cents = float('nan')
    with pytest.raises((OSError, ValueError)):
        session.save(target)
    assert target.read_bytes() == b'preserve original bytes\n'
    assert session.is_dirty and session.save_path is None
    assert list(tmp_path.iterdir()) == [target]


def test_export_rejects_a_parser_that_would_drop_a_declared_parameter(tmp_path, monkeypatch):
    source = ir.load_instrument_from_dict(author())
    source.layers.L0_graph.nodes[0].params.detune_cents = 17
    original = ir._parse_node_params

    def dropping_parser(data):
        value = original(data)
        value.detune_cents = 0
        return value

    monkeypatch.setattr(ir, '_parse_node_params', dropping_parser)
    target = tmp_path / 'untouched.json'
    with pytest.raises(ValueError, match='detune_cents'):
        ir.save_instrument(source, target)
    assert not target.exists() and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), object(), {1: 'ambiguous key'}])
def test_nonportable_authoring_values_fail_before_destination_creation(bad, tmp_path):
    source = ir.InstrumentIR()
    source.layers.L3_cmmm.state.variables['x'] = ir.StateVariable('json', bad)
    with pytest.raises(ValueError):
        ir.save_instrument(source, tmp_path / 'invalid.json')
    assert list(tmp_path.iterdir()) == []


def test_unknown_runtime_attributes_are_not_silently_discarded():
    source = ir.InstrumentIR()
    source.layers.L0_graph.future_extension = 42
    with pytest.raises(ValueError, match='undeclared'):
        ir.instrument_to_dict(source)


def test_legacy_envelope_control_fallbacks_migrate_without_mutating_the_source():
    source = ir.InstrumentIR(_envelopes_compat=[ir.EnvelopeBinding('legacy', targets=['GAIN'])],
        _control_compat=ir.ControlConfig(brightness=0.73))
    original = deepcopy(source)
    loaded = ir.load_instrument_from_dict(ir.instrument_to_dict(source))
    assert loaded.envelopes == source.envelopes and loaded.control == source.control
    assert loaded._envelopes_compat == [] and loaded._control_compat is None
    assert source == original
    source.control = ir.ControlConfig(brightness=0.25)
    with pytest.raises(ValueError, match='conflicting legacy'):
        ir.instrument_to_dict(source)
