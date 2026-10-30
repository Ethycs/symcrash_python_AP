"""Authoring -> capsule/compiled preset -> actual stereo output ownership."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from symcrash_python_ap.control_frame import ControlFrame
from symcrash_python_ap.frames import ParamFrame, VoiceFramePacketV1, VoiceFlags
from symcrash_python_ap.instrument_ir import SafetyConfig, load_instrument_from_dict
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.capsule_codec import (
    capsule_to_bytes, capsule_from_bytes, save_capsule_bundle, load_capsule_bundle)
from symcrash_python_ap.library.preset_compiler import (
    PresetCompiler, PresetCompileError, compiled_preset_from_ir, compiled_preset_from_capsule)
from symcrash_python_ap.safety.safety_engine import SafetyEngine
from symcrash_python_ap.synth_master import SynthMaster, SynthConfig
from symcrash_python_ap.vm import hotpath_jit as hot


def authored(**settings):
    path = Path(__file__).resolve().parents[2] / 'presets/instruments/dx_marimba_v0.yaml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    compile_data = data['instrument_ir_v0'].setdefault('compile', {})
    compile_data['safety'] = settings
    compile_data.setdefault('material', {})['material_id'] = 2
    return data


def packet(seq, note=96):
    return VoiceFramePacketV1(voice_id=0, frame_seq=seq, flags=VoiceFlags.GATE,
                              gain_q15=32767, note_u8=note, carrier_role_u8=1).to_voice_frame()


def host(tmp_path, name='host'):
    result = SynthMaster(SynthConfig(n_voices=1, output_dir=str(tmp_path/name/'audio'),
                                     library_dir=str(tmp_path/name/'library')))
    result.disable('cockpit')
    return result


@pytest.fixture(params=[False, True], ids=['python', 'jit'])
def execution_path(request):
    original = hot.jit_enabled()
    if request.param and not hot.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    hot.set_jit_enabled(request.param)
    try:
        yield
    finally:
        hot.set_jit_enabled(original)


@pytest.mark.parametrize('codec', ['direct', 'binary', 'bundle'])
def test_all_declared_settings_survive_authoring_and_capsules_without_aliasing(tmp_path, codec):
    values = dict(enabled=False, dc_removal=True, master_ola=True,
                  limiter_threshold_q15=12001, slew_limit_q15=321,
                  energy_cap_soft_q32=123456, energy_cap_hard_q32=234567)
    ir = load_instrument_from_dict(authored(**values))
    assert asdict(ir.compile.safety) == values
    if codec == 'direct':
        preset = compiled_preset_from_ir(ir)
    else:
        compiled = CapsuleCompiler().compile_from_ir(ir)
        assert compiled.success, compiled.errors
        capsule = compiled.capsule
        assert capsule.sound_extras['safety'] == values
        if codec == 'binary':
            capsule = capsule_from_bytes(capsule_to_bytes(capsule))
        else:
            save_capsule_bundle(capsule, str(tmp_path/'bundle'))
            capsule = load_capsule_bundle(str(tmp_path/'bundle'))
        preset = compiled_preset_from_capsule(capsule)
        capsule.sound_extras['safety']['slew_limit_q15'] = 999
    assert asdict(preset.safety) == values
    assert preset.material.material_id == 2
    ir.compile.safety.slew_limit_q15 = 777
    assert preset.safety.slew_limit_q15 == 321


def test_legacy_capsule_has_explicit_ccs_defaults():
    from symcrash_python_ap.library.capsule import CapsuleV0, CapsuleIdentity, CapsuleKind
    capsule = CapsuleV0(identity=CapsuleIdentity('legacy', CapsuleKind.SYNTH, 'Legacy'))
    preset = compiled_preset_from_capsule(capsule)
    assert preset.safety == SafetyConfig()
    assert preset.material.material_id == 0


def test_flat_preset_and_documented_aliases_use_the_same_parser():
    preset = PresetCompiler().compile({'preset': {'preset_id': 'flat'},
        'safety': {'ola_enabled': True, 'max_slew_per_sample': 731, 'dc_removal': True}})
    assert preset.safety == SafetyConfig(master_ola=True, slew_limit_q15=731, dc_removal=True)
    with pytest.raises(PresetCompileError, match='Unsupported Safety'):
        PresetCompiler().compile({'preset': {'preset_id': 'bad'}, 'safety': {'invented': 1}})


@pytest.mark.parametrize('settings', [
    {'enabled': 1}, {'dc_removal': 'false'}, {'master_ola': None},
    {'limiter_threshold_q15': 0}, {'limiter_threshold_q15': 32768},
    {'limiter_threshold_q15': 123.0}, {'slew_limit_q15': -1},
    {'slew_limit_q15': True}, {'slew_limit_q15': 32768},
    {'energy_cap_hard_q32': -1}, {'energy_cap_soft_q32': 1 << 32},
    {'energy_cap_soft_q32': 20, 'energy_cap_hard_q32': 19},
    {'master_ola': True, 'ola_enabled': False}, {'dc_cutoff_hz': 10},
    {'ola_window': 'blackman'}, {'limiter_attack_ms': 0.1},
    {'limiter_threshold_db': -0.3}, {'slew_limiting': False},
])
def test_invalid_or_unimplemented_safety_settings_are_not_silently_lost(settings):
    with pytest.raises(ValueError):
        load_instrument_from_dict(authored(**settings))


@pytest.mark.parametrize('mode', ['yaml', 'compiled', 'library', 'cockpit'])
def test_every_load_path_applies_material_and_safety_before_audio(tmp_path, mode, execution_path):
    data = authored(dc_removal=True, master_ola=True,
                    limiter_threshold_q15=9001, slew_limit_q15=301)
    ir = load_instrument_from_dict(data)
    synth = host(tmp_path)
    if mode == 'yaml':
        path = tmp_path/'sound.yaml'
        path.write_text(yaml.safe_dump(data), encoding='utf-8')
        assert synth.load_preset_from_yaml(str(path))
    elif mode == 'compiled':
        synth.load_compiled_preset(compiled_preset_from_ir(ir))
    elif mode == 'library':
        result = CapsuleCompiler().compile_from_ir(ir)
        assert result.success, result.errors
        imported = synth.library.import_capsule(capsule_to_bytes(result.capsule))
        assert imported.success, imported.errors
        assert synth.load_preset(result.capsule.identity.capsule_id)
    else:
        synth.register_control_preset(77, compiled_preset_from_ir(ir))
        control = ControlFrame()
        control.add_preset_load(77)
        synth._apply_control_frame(control)
    # A second host supplies the actual pre-Safety stream for an independently
    # configured output engine. This checks routing, not just stored attributes.
    raw = host(tmp_path, 'raw')
    raw.set_material(2)
    raw.disable('safety')
    oracle = SafetyEngine()
    oracle.LIMIT_THRESHOLD = oracle.LIMIT_CEILING = 9001
    oracle.max_slew = 301
    before = [0,0]
    for seq in range(1,20):
        params = ParamFrame(K_mode=3, drive_q15=25000, bias_q15=14000)
        frames = [packet(seq)]
        pre_l, pre_r, _ = raw.render_frames(deepcopy(frames), params=params)
        expected = oracle.step_block(pre_l, pre_r, params)
        actual = synth.render_frames(frames, params=params)[:2]
        assert actual == expected
        for ch, values in enumerate(actual):
            assert max(abs(x) for x in values) <= 9001
            assert max(abs(b-a) for a,b in zip([before[ch]]+values, values)) <= 301
            assert abs(sum(values)) <= 64*2048
            if seq > 1:
                assert values[0] == before[ch]
            before[ch] = values[-1]
    assert synth._material_id == 2
    assert synth.safety.max_slew == 301
    assert synth._safety_master_ola and synth._safety_dc_removal


def test_default_ccs_path_preserves_low_frequency_block_means(tmp_path, execution_path):
    synth, raw = host(tmp_path), host(tmp_path, 'raw')
    synth.disable('material')
    raw.disable('material')
    raw.disable('safety')
    exceeded_dc = False
    for seq in range(1,20):
        frames = [packet(seq, note=24)]
        expected = raw.render_frames(deepcopy(frames))[:2]
        actual = synth.render_frames(frames)[:2]
        # Three Q15 near-unity gain multiplications can account for small LSBs;
        # block DC subtraction or master OLA would produce much larger errors.
        assert max(abs(a-b) for x,y in zip(actual,expected) for a,b in zip(x,y)) <= 4
        exceeded_dc |= abs(sum(actual[0])) > 64*2048
    assert exceeded_dc
    assert not synth._safety_master_ola and not synth._safety_dc_removal


def test_cockpit_switches_global_defaults_without_rewriting_existing_voices(tmp_path):
    synth = host(tmp_path)
    old = PresetCompiler().compile({'preset': {'preset_id': 'old'}})
    new = PresetCompiler().compile({'preset': {'preset_id': 'new'},
                                    'safety': {'limiter_threshold_q15': 8000, 'slew_limit_q15': 500},
                                    'material': {'material_id': 1}})
    synth.load_compiled_preset(old)
    vid = synth.note_on(60)
    synth.render_block()
    synth.register_control_preset(5, new)
    control = ControlFrame()
    control.add_preset_load(5)
    synth._render_block_voice_engine(control)
    assert synth._material_id == 1
    assert synth.safety.LIMIT_CEILING == 8000
    assert synth.safety.max_slew == 500
    assert synth.voice_engine.voices[vid].compiled_preset is old


def test_loading_defaults_resets_previous_preset_configuration(tmp_path):
    synth = host(tmp_path)
    first = PresetCompiler().compile({'preset': {'preset_id': 'one'},
        'material': {'material_id': 4}, 'safety': {'dc_removal': True, 'master_ola': True,
        'limiter_threshold_q15': 6000, 'slew_limit_q15': 100}})
    synth.load_compiled_preset(first)
    synth.load_compiled_preset(PresetCompiler().compile({'preset': {'preset_id': 'two'}}))
    assert synth._material_id == 0
    assert synth._safety_config == SafetyConfig()
    assert synth.safety.LIMIT_CEILING == 32000 and synth.safety.max_slew == 2000


def test_bypass_freezes_dsp_but_reenable_uses_actual_emitted_history(tmp_path, execution_path):
    synth = host(tmp_path)
    synth.disable('material')
    synth.configure_safety(SafetyConfig(slew_limit_q15=17))
    synth.render_frames([packet(1,24)])
    before = deepcopy(synth.safety.rg)
    synth.disable('safety')
    bypass = synth.render_frames([packet(2,24)])[:2]
    assert synth.safety.rg == before
    synth.enable('safety')
    actual = synth.render_frames([packet(3,24)])[:2]
    for values, previous in zip(actual,bypass):
        assert max(abs(b-a) for a,b in zip([previous[-1]]+values,values)) <= 17
    assert synth.safety._last_block_prev_L == bypass[0][-1]


def test_invalid_or_infeasible_configuration_does_not_replace_active_settings(tmp_path):
    synth = host(tmp_path)
    before = deepcopy(synth._safety_config)
    invalid = SafetyConfig()
    invalid.slew_limit_q15 = -1
    with pytest.raises(ValueError):
        synth.configure_safety(invalid)
    assert synth._safety_config == before
    synth._emitted_tail_L = synth._emitted_tail_R = [30000]*8
    with pytest.raises(ValueError, match='infeasible'):
        synth.configure_safety(SafetyConfig(limiter_threshold_q15=1000, slew_limit_q15=1))
    assert synth._safety_config == before
    assert synth.safety.LIMIT_CEILING == 32000
    synth.disable('safety')
    synth.configure_safety(SafetyConfig(limiter_threshold_q15=1000, slew_limit_q15=1))
    with pytest.raises(ValueError, match='infeasible'):
        synth.enable('safety')
    assert not synth._enable_safety and synth._safety_override is False


def test_declared_limits_cannot_relax_the_normative_bounds(tmp_path):
    synth = host(tmp_path)
    synth.configure_safety(SafetyConfig(limiter_threshold_q15=32767, slew_limit_q15=32767))
    assert synth.safety.LIMIT_CEILING == 32512
    assert synth.safety.max_slew == 8192


def test_debug_override_persists_and_authored_disable_applies_without_override(tmp_path):
    synth = host(tmp_path)
    preset = PresetCompiler().compile({'preset': {'preset_id': 'disabled'}, 'safety': {'enabled': False}})
    synth.load_compiled_preset(preset)
    assert not synth._enable_safety
    synth.enable('safety')
    synth.load_compiled_preset(preset)
    assert synth._enable_safety
    synth.disable('safety')
    synth.load_compiled_preset(PresetCompiler().compile({'preset': {'preset_id': 'enabled'}}))
    assert not synth._enable_safety


@pytest.mark.parametrize('slew', [0, 1, 63, 500, 8192])
def test_configured_slew_has_bit_exact_python_jit_output_and_state(slew):
    old = hot.jit_enabled()
    engines = [SafetyEngine(),SafetyEngine()]
    for engine in engines:
        engine.max_slew = slew
    try:
        for tick in range(12):
            outputs = []
            for index, engine in enumerate(engines):
                hot.set_jit_enabled(bool(index))
                outputs.append(engine.step_block([32767 if tick % 2 else -32768]*64,
                    [-32768,32767]*32, ParamFrame(K_mode=3)))
                assert all(result.a2_slew_ok for result in engine.check_audio_safety(*outputs[-1]))
            assert outputs[0] == outputs[1]
            for name in ('rg','lim_L','lim_R','prev_sample_L','prev_sample_R','master_ola_L','master_ola_R'):
                assert getattr(engines[0],name) == getattr(engines[1],name)
    finally:
        hot.set_jit_enabled(old)
