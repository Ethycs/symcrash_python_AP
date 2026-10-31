"""Native authoring must survive compilation and reach identical audio owners."""
from copy import deepcopy
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap import instrument_ir as ir
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.capsule_codec import capsule_to_bytes, capsule_from_bytes
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir, compiled_preset_from_capsule
from symcrash_python_ap.library.native_carrier import NativeCarrierConfig
from symcrash_python_ap.library.ir_extract import lfo_to_dict, lfo_from_dict
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.harmonic_bank import LFOState, LFORouting, LFOTarget, LFOWaveform, LFOSyncMode
from tests.conformance.test_vm_snapshot_replay import execution_path


ROOT = Path(__file__).resolve().parents[2]
PRESETS = sorted((ROOT / 'presets/instruments').glob('*.yaml'))


def capsule_for(source):
    result = CapsuleCompiler().compile_from_ir(source)
    assert result.success, result.errors
    return capsule_from_bytes(capsule_to_bytes(result.capsule))


def routes(source):
    return compiled_preset_from_ir(source), compiled_preset_from_capsule(capsule_for(source))


def owner(preset, note=69):
    engine = VoiceEngine(max_voices=2)
    engine.allocate_voice_from_preset(preset, voice_id=0)
    engine.note_on_simple(0, note=note, velocity_q15=24000)
    return engine


def source(waveform='saw', **params):
    result = ir.InstrumentIR(layers=ir.Layers(L0_graph=ir.L0Graph(nodes=[
        ir.L0Node('osc', 'Osc', ir.NodeParams(waveform=waveform, **params)),
        ir.L0Node('amp', 'Amp', ir.NodeParams(adsr=ir.ADSR(5.5, 17.25, .71321, 150.0))),
    ])))
    result.meta.instrument_id = 'native-transport-fixture'
    result.meta.name = 'Native transport fixture'
    return result


@pytest.mark.parametrize('path', PRESETS, ids=lambda p: p.stem)
def test_every_factory_preset_has_identical_native_frames_and_audio_after_capsule(path, execution_path):
    authored = ir.load_instrument(path)
    before = deepcopy(authored)
    direct, capsule = routes(authored)
    # Metadata may acquire a capsule UUID/version; all playback fields agree.
    a, b = asdict(direct), asdict(capsule)
    for key in ('preset_id', 'preset_version', 'description'):
        a.pop(key)
        b.pop(key)
    assert a == b
    engines = [owner(p) for p in (direct, capsule)]
    states = [VMState.create(0), VMState.create(0)]
    nonzero = 0
    for block in range(30):
        for engine in engines:
            engine.voices[0].gesture.chaos_q15 = 32767
            if block == 20:
                engine.note_off_simple(0)
        frames = [engine.step_voice_simple(0) for engine in engines]
        assert frames[0] == frames[1]
        if frames[0] is None:
            continue
        samples = [vm_step_block(state, frame)[1].copy() for state, frame in zip(states, frames)]
        np.testing.assert_array_equal(*samples)
        nonzero += np.count_nonzero(samples[0])
    assert nonzero and authored == before


@pytest.mark.parametrize('waveform,expected', [('sine', 0), ('square', 1), ('pulse', 2),
    ('saw', 3), ('saw_down', 4), ('tri', 5), ('triangle', 5)])
def test_named_waveform_reaches_mode_b_without_an_eight_partial_substitute(waveform, expected):
    for preset in routes(source(waveform)):
        frame = owner(preset).step_voice_simple(0)
        assert frame.coeff_osc_mode == 1 and frame.waveform_type == expected
        assert frame.partial_amps_q15 == []
        assert decode_frame(frame).coeff_osc_params.custom_harmonics is None


@pytest.mark.parametrize('waveform,table', [('sine', 0), ('saw', 1), ('square', 2), ('triangle', 3)])
def test_explicit_mode_a_selects_its_lut_even_after_same_note_reset(waveform, table):
    for preset in routes(source(waveform, osc_mode=0)):
        frame = owner(preset).step_voice_simple(0)
        assert frame.coeff_osc_mode == 0 and frame.coeff_table_id == table
        vm = VMState.create(0)
        vm.reset_for_note_on(note=69)
        vm.last_note = 69
        vm_step_block(vm, frame)
        assert vm.ccs_state.wave_id == table


def test_named_waveforms_are_audibly_distinct(execution_path):
    results = []
    for wave in ('sine', 'square', 'saw', 'triangle'):
        engine, vm = owner(compiled_preset_from_ir(source(wave))), VMState.create(0)
        samples = np.concatenate([vm_step_block(vm, engine.step_voice_simple(0))[1].copy() for _ in range(8)])
        results.append(samples.tobytes())
    assert len(set(results)) == 4


def test_custom_partials_phase_oeis_detune_and_noise_reach_actual_vm(execution_path):
    authored = source('partials', harmonics=ir.HarmonicProfile([.7, .2, .4, .1]),
        phase_spread=.713, detune_cents=17.4, oeis_sequence_id=1, oeis_rolloff_db=-5.25,
        noise_level=.2, color='pink', noise_attack_ms=2, noise_decay_ms=19)
    authored.meta.seed_u32 = 0xdeadbeef
    preset = compiled_preset_from_ir(authored)
    engine = owner(preset)
    actual, clean = VMState.create(0), VMState.create(0)
    changed = 0
    for _ in range(12):
        frame = engine.step_voice_simple(0)
        assert frame.partial_amps_q15 == [22937, 6553, 13107, 3277]
        assert frame.partial_phase_spread_q15 == round(.713*32767)
        assert (frame.oeis_sequence_id, frame.oeis_rolloff_db, frame.cents) == (1, -5.25, 17)
        assert (frame.noise_color, frame.noise_level_q15, frame.noise_seed_u32) == (1, 6553, 0xdeadbeef)
        baseline = replace(frame, noise_level_q15=0)
        a = vm_step_block(actual, frame)[1].copy()
        b = vm_step_block(clean, baseline)[1].copy()
        changed += np.count_nonzero(a != b)
    assert changed > 0


@pytest.mark.parametrize('width', [0, 1, 8192, 16384, 32767])
def test_pwm_zero_and_all_widths_survive_authoring_and_decode(width):
    for preset in routes(source('pulse', pwm_enable=True, pwm_width_q15=width)):
        frame = owner(preset).step_voice_simple(0)
        assert frame.pwm_enabled and frame.pwm_width_q15 == width
        assert decode_frame(frame).coeff_osc_params.pwm_width_q15 == width


@pytest.mark.parametrize('ms', [0, 4/3, 5.0, 5.5, 17.25, 150.0])
def test_exact_envelope_survives_quantized_control_defaults_and_key_scaling(ms):
    authored = source()
    amp = authored.layers.L0_graph.nodes[1].params
    amp.adsr = ir.ADSR(ms, ms + 1.125, .713217, ms + 3.75)
    amp.key_scale_rate = .713
    a, b = routes(authored)
    assert a.amp_env == b.amp_env
    assert a.amp_env.attack_ms == ms and b.amp_env.sustain == .713217
    ea, eb = owner(a, 82), owner(b, 82)
    assert ea.voices[0].amp_envelope == eb.voices[0].amp_envelope
    assert [ea.step_voice_simple(0) for _ in range(10)] == [eb.step_voice_simple(0) for _ in range(10)]


def test_no_amp_stays_absent_instead_of_becoming_a_capsule_control_envelope():
    authored = source()
    authored.layers.L0_graph.nodes.pop()
    assert all(p.amp_env is None for p in routes(authored))


def test_control_defaults_and_compiled_mutable_data_are_detached():
    authored = ir.load_instrument(ROOT/'presets/instruments/warm_pad_v0.yaml')
    a, b = routes(authored)
    assert a.control_config == b.control_config and a.control_config is not None
    assert a.control_config is not authored.control
    a.control_config.risk = .9
    assert b.control_config.risk == authored.control.risk != .9
    if a.envelope_bindings:
        a.envelope_bindings[0].adsr.a_ms = 999
        assert authored.envelopes[0].adsr.a_ms != 999


def test_lfo_capsule_keeps_disabled_state_raw_phase_and_all_declared_fields():
    lfo = LFOState(id='raw', waveform=LFOWaveform.TRIANGLE, sync_mode=LFOSyncMode.FREE,
        routings=[LFORouting(LFOTarget.PITCH, -12345)], enabled=False, phase=123456789,
        phase_inc=76543, elapsed_samples=999, current_depth_scale=12345, delay_samples=101,
        ramp_samples=203, depth_pitch_cents=17, depth_amp_q15=4567, depth_timbre_q15=7654,
        depth_q15=11111, rate_hz=3.17, arc_slot=2)
    restored = lfo_from_dict(json.loads(json.dumps(lfo_to_dict(lfo))))
    assert restored == lfo and restored.routings is not lfo.routings
    authored = source()
    authored.modulators = [lfo]
    a, b = routes(authored)
    assert a.lfos == b.lfos == [lfo]
    a.lfos[0].phase += 1
    assert authored.modulators[0].phase == b.lfos[0].phase == 123456789
    frame = owner(b).step_voice_simple(0)
    assert frame.cents == 0  # A disabled LFO did not silently become enabled.


def test_old_lfo_payload_still_loads_without_raw_state():
    data = lfo_to_dict(LFOState())
    del data['raw_state'], data['raw_state_version']
    restored = lfo_from_dict(data)
    assert restored.id == '' and restored.waveform == LFOWaveform.SINE


def test_old_capsules_keep_their_legacy_fallback_instead_of_inventing_exact_data():
    capsule = capsule_for(source())
    del capsule.sound_extras['native_carrier'], capsule.sound_extras['amp_envelope']
    preset = compiled_preset_from_capsule(capsule)
    assert preset.native_carrier is None
    assert preset.amp_env.attack_ms == 4 * (4/3)  # Four stored control blocks.


@pytest.mark.parametrize('field,value', [('version', 4), ('version', True), ('waveform_type', 6),
    ('oeis_sequence_id', 8), ('oeis_rolloff_db', float('nan')), ('pwm_enabled', 1),
    ('noise_seed_u32', -1), ('noise_attack_blocks', 0), ('noise_level_q15', 32768),
    ('partial_phase_spread_q15', -1), ('fm_topo_a_id', 9), ('filter_mode', 4)])
def test_malformed_native_capsule_fails(field, value):
    capsule = capsule_for(source())
    capsule.sound_extras['native_carrier'][field] = value
    with pytest.raises(ValueError):
        compiled_preset_from_capsule(capsule)


@pytest.mark.parametrize('field,value', [('attack_ms', -1), ('decay_ms', float('inf')),
    ('sustain', 1.01), ('release_ms', True)])
def test_malformed_exact_envelope_fails(field, value):
    capsule = capsule_for(source())
    capsule.sound_extras['amp_envelope'][field] = value
    with pytest.raises(ValueError):
        compiled_preset_from_capsule(capsule)


def test_missing_native_or_lfo_fields_are_not_silently_defaulted():
    data = NativeCarrierConfig().to_dict()
    del data['waveform_type']
    with pytest.raises(ValueError, match='Incomplete'):
        NativeCarrierConfig.from_dict(data)
    data = lfo_to_dict(LFOState())
    del data['raw_state']['enabled']
    with pytest.raises(ValueError, match='Incomplete'):
        lfo_from_dict(data)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_old_voice_json_preserves_legacy_carrier_absence(version):
    engine = owner(compiled_preset_from_ir(source()))
    saved = engine.voices[0].snapshot()
    saved['snapshot_version'] = version
    del saved['compiled_preset']['fields']['native_carrier']
    before = deepcopy(saved)
    restored = VoiceStateWithProgram.restore(saved)
    assert restored.compiled_preset.native_carrier is None and saved == before
    current = restored.snapshot()
    assert current['snapshot_version'] == 7
    del current['compiled_preset']['fields']['native_carrier']
    with pytest.raises(ValueError, match='Incomplete'):
        VoiceStateWithProgram.restore(current)


def test_native_snapshot_replays_phase_noise_pwm_and_release_audio(execution_path):
    authored = source('pulse', pwm_enable=True, pwm_width_q15=9012,
        noise_level=.15, color='brown', noise_decay_ms=200)
    first = owner(compiled_preset_from_ir(authored))
    vm = VMState.create(0)
    for _ in range(9):
        vm_step_block(vm, first.step_voice_simple(0))
    second = VoiceEngine()
    second.voices[0] = VoiceStateWithProgram.restore(json.loads(json.dumps(first.voices[0].snapshot())))
    replay_vm = VMState.restore(json.loads(json.dumps(vm.snapshot())))
    for block in range(18):
        if block == 11:
            first.note_off_simple(0)
            second.note_off_simple(0)
        a, b = first.step_voice_simple(0), second.step_voice_simple(0)
        assert a == b
        np.testing.assert_array_equal(vm_step_block(vm, a)[1], vm_step_block(replay_vm, b)[1])


def test_fresh_process_loads_capsule_and_matches_direct_render(tmp_path):
    authored = source('partials', harmonics=ir.HarmonicProfile([.7, .2, .3]), phase_spread=.3,
                      noise_level=.1, color='pink')
    path = tmp_path/'voice.cap'
    path.write_bytes(capsule_to_bytes(capsule_for(authored)))
    producer, vm = owner(compiled_preset_from_ir(authored)), VMState.create(0)
    expected = [vm_step_block(vm, producer.step_voice_simple(0))[1].tolist() for _ in range(12)]
    code = '''import json, sys
from pathlib import Path
from tests.conformance.test_native_preset_transport import owner
from symcrash_python_ap.library.capsule_codec import capsule_from_bytes
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_capsule
from symcrash_python_ap.vm import VMState, vm_step_block
e=owner(compiled_preset_from_capsule(capsule_from_bytes(Path(sys.argv[1]).read_bytes())))
vm=VMState.create(0)
print(json.dumps([vm_step_block(vm,e.step_voice_simple(0))[1].tolist() for _ in range(12)]))
'''
    result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def test_explicit_zero_mode_and_waveform_survive_ir_save_load(tmp_path):
    authored = source(None, osc_mode=0, waveform_type=0)
    path = tmp_path/'explicit.yaml'
    ir.save_instrument(authored, path)
    restored = ir.load_instrument(path)
    assert restored == authored
    frame = owner(compiled_preset_from_ir(restored)).step_voice_simple(0)
    assert (frame.coeff_osc_mode, frame.waveform_type, frame.coeff_table_id) == (0, 0, 0)


def test_static_l1_fm_topology_uses_declared_pair_even_when_both_endpoints_are_zero():
    authored = ir.load_instrument(ROOT/'presets/instruments/dx_marimba_v0.yaml')
    authored.layers.L1_lift.fm6_morph = ir.FM6MorphConfig(topo_a=0, topo_b=0, blend_q15=0)
    for preset in routes(authored):
        frame = owner(preset).step_voice_simple(0)
        assert (frame.fm_topo_a_id, frame.fm_topo_b_id, frame.fm_topo_blend_q15) == (0, 0, 0)
        vm = VMState.create(0)
        vm_step_block(vm, frame)
        assert vm._fm6_morph_state is not None


def test_oeis_has_priority_over_custom_partials_on_every_block(execution_path):
    preset = compiled_preset_from_ir(source('partials', oeis_sequence_id=0,
        harmonics=ir.HarmonicProfile([1.0, .2, .1]), phase_spread=.75))
    engine = owner(preset)
    a, b = VMState.create(0), VMState.create(0)
    for _ in range(9):
        frame = engine.step_voice_simple(0)
        without_partials = replace(frame, partial_amps_q15=[], partial_phase_spread_q15=0)
        np.testing.assert_array_equal(vm_step_block(a, frame)[1], vm_step_block(b, without_partials)[1])


@pytest.mark.parametrize('enabled', [False, True])
def test_only_enabled_pwm_lfos_enable_pwm_implicitly(enabled):
    authored = source('pulse')
    authored.modulators = [LFOState(enabled=enabled, routings=[LFORouting(LFOTarget.PWM_WIDTH, 10000)])]
    assert compiled_preset_from_ir(authored).native_carrier.pwm_enabled is enabled


def test_offline_builder_uses_shared_native_identity_without_losing_noise_decorrelation():
    from symcrash_python_ap.voice.frame_builder import IRFrameBuilder, NoteClock
    authored = source(None, osc_mode=0, waveform_type=3, detune_cents=13,
                      noise_level=.1, color='pink')
    authored.meta.seed_u32 = 12345
    frame = IRFrameBuilder(authored, 69, noise_seed_xor=42).build(NoteClock(0))
    assert (frame.coeff_osc_mode, frame.waveform_type, frame.coeff_table_id, frame.cents) == (0, 3, 1, 13)
    assert frame.noise_seed_u32 == 12345 ^ 42


@pytest.mark.parametrize('params', [dict(osc_mode=2), dict(osc_mode=3),
    dict(osc_mode=0, pwm_enable=True), dict(osc_mode=0, oeis_sequence_id=1)])
def test_unavailable_resource_and_mode_combinations_fail_explicitly(params):
    with pytest.raises(ValueError):
        compiled_preset_from_ir(source(**params))


@pytest.mark.parametrize('name', ['warm_pad_v0', 'pwm_pad_v0', 'oeis_bell_v0',
                               'acoustic_marimba_v0', 'glitch_lead_v0'])
def test_actual_synthmaster_yaml_and_library_capsules_match_final_stereo(name, tmp_path, execution_path):
    from tests.conformance.test_output_configuration import host
    path = ROOT/'presets/instruments'/f'{name}.yaml'
    direct, loaded = host(tmp_path, 'direct'), host(tmp_path, 'capsule')
    assert direct.load_preset_from_yaml(str(path))
    capsule = capsule_for(ir.load_instrument(path))
    result = loaded.library.import_capsule(capsule_to_bytes(capsule))
    assert result.success, result.errors
    assert loaded.load_preset(capsule.identity.capsule_id)
    direct.note_on(69, 100)
    loaded.note_on(69, 100)
    nonzero = 0
    for block in range(32):
        if block == 24:
            direct.note_off(69)
            loaded.note_off(69)
        a, b = direct.render_block()[:2], loaded.render_block()[:2]
        assert a == b
        nonzero += sum(x != 0 for channel in a for x in channel)
    assert nonzero


def test_named_waveform_has_priority_over_legacy_serialized_default_numeric_zero():
    # Previous full IR writers emitted waveform_type=0 even beside waveform=saw.
    preset = compiled_preset_from_ir(source('saw', osc_mode=0, waveform_type=0))
    frame = owner(preset).step_voice_simple(0)
    assert (frame.waveform_type, frame.coeff_table_id) == (3, 1)
