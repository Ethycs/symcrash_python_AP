"""Authored event expressions must reach real command consumers without loss."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from symcrash_python_ap.frames import VoiceFrame, VoiceFramePacketV0b, VoiceFramePacketV1
from symcrash_python_ap.instrument_ir import load_instrument_from_dict
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.capsule_codec import capsule_to_bytes, capsule_from_bytes
from symcrash_python_ap.library.preset_compiler import (
    compiled_preset_from_ir, compiled_preset_from_capsule, PresetCompiler, PresetCompileError)
from symcrash_python_ap.voice.ir_events import (
    compile_ir_events, evaluate, CompiledEventProgram, AuthoredEventState)
from symcrash_python_ap.voice.frame_builder import IRFrameBuilder, NoteClock
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram, InstrumentProgram
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.scheduling import ExecutionPolicy, ExecutionReport
from symcrash_python_ap.wire_frame import decode_wire_events
from tests.conformance.test_vm_snapshot_replay import execution_path, assert_state_equal
from tests.conformance.test_scheduled_vm import schedule


def event(lane=0, when='always', strength='0.5', **payload):
    return dict(lane=lane, when=when, strength=strength, payload=payload)


def author(events=None):
    data = yaml.safe_load((Path(__file__).resolve().parents[2] /
        'presets/instruments/dx_marimba_v0.yaml').read_text(encoding='utf-8'))
    root = data['instrument_ir_v0']
    from symcrash_python_ap.instrument_ir import _translate_sound_design_format
    if 'graph' in root:
        root = _translate_sound_design_format(root)
        data['instrument_ir_v0'] = root
    root['meta']['instrument_id'] = 'authored-event-fixture'
    layers = root.setdefault('layers', {})
    layers['L3_cmmm'] = {'emit_templates': {
        'T_ATTACK': {'events': events or [event(0, 'gate_rise', '20000', op='Impulse', target_kind='band_id', target_id=3)]},
        'T_STABLE': {'events': [event(0, 'block_index % 3 == 0', '12000', op='NoiseBand', target_kind='band_id', target_id=5)]},
        'T_RECOVERY': {'events': [event(5, 'gate_fall', '16000', variant='drive', p0=90, p1=3)]},
        'T_IDLE': {'events': []},
    }}
    layers['L2_regimes'] = {'regime_nodes': [
        {'id': 'R1_ATTACK', 'kind': 'Transient', 'emit_template': 'T_ATTACK'},
        {'id': 'R2_STABLE', 'kind': 'StableTone', 'emit_template': 'T_STABLE'},
        {'id': 'R6_RECOVERY', 'kind': 'Recovery', 'emit_template': 'T_RECOVERY'},
    ]}
    return data


def engine(preset):
    result = VoiceEngine(max_voices=4)
    result.allocate_voice_from_preset(preset, voice_id=0)
    result.note_on_simple(0, note=69, velocity_q15=28000)
    return result


@pytest.mark.parametrize('source,context,expected', [
    ('always', {}, 1), ('never', {}, 0), ('0.5', {}, 16384),
    ('1.0', {}, 32767), ('-0.5', {}, -16384), ('32767', {}, 32767),
    ('gesture.gate and intent.snap_q15 > 16384', {'gesture.gate': True, 'intent.snap_q15': 16384}, 0),
    ('gesture.gate and intent.snap_q15 >= 16384', {'gesture.gate': True, 'intent.snap_q15': 16384}, 1),
    ('macro(M_GAIN) - (macro.M_GAIN >> 1)', {'macro.M_GAIN': 20001}, 10001),
    ('0 < risk <= 0.5', {'risk': 16384}, 1),
    ('mul_q15(Q15_HALF, 3)', {}, 2),  # nearest-even tie
    ('mul_q15_sat(-16384, 3)', {}, -2),
    ('sat16(100000)', {}, 32767), ('clamp(-5, 0, 10)', {}, 0),
    ('max(1, min(9, abs(-11)))', {}, 9), ('block_index % 4 == 2', {'block_index': 6}, 1),
    ('(5 << 2) | 3', {}, 23), ('(7 & 3) + 8', {}, 11),
    ('not gate_fall', {'gate_fall': 0}, 1), ('never and macro(M_UNDEFINED)', {}, 0),
])
def test_expression_integer_arithmetic_edges_and_units(source, context, expected):
    assert evaluate(source, context) == expected


@pytest.mark.parametrize('source', ['__import__("os")', 'open("x")', 'gesture.__class__',
    'macro.__dict__', 'intent.missing', 'unknown', '1 / 0', '2 ** 100', 'x[0]',
    '[1, 2]', '(lambda: 1)()', 'abs(x=1)', '1 << block_index', '1 << 32',
    'block_index % 0', '3.14159', 'float("nan")', '', '1 if gate else 0'])
def test_unimplemented_or_executable_expression_syntax_is_rejected(source):
    with pytest.raises(ValueError):
        evaluate(source, {})


def test_expression_size_missing_inputs_and_runtime_float_rejection():
    for source, context in [('gate', {}), ('risk', {'risk': 0.5}),
                             ('1+' * 100 + '1', {}), ('x' * 2049, {})]:
        with pytest.raises(ValueError):
            evaluate(source, context)


@pytest.mark.parametrize('condition,variable', [('Gate rises', 'gate_rise'), ('Gate falls', 'gate_fall')])
def test_documented_gate_aliases_compile_inside_rule_guards(condition, variable):
    program = compile_ir_events(load_instrument_from_dict(author([event(0, condition)])))
    assert program.emit('T_ATTACK', {variable: 0}).event_mask == 0
    assert program.emit('T_ATTACK', {variable: 1}).event_mask == 1


def test_when_payload_and_variant_aliases_survive_loader():
    source = author([dict(lane=1, when='gate_rise', variant='enable', payload={'target_id': 3}, strength=5)])
    ir = load_instrument_from_dict(source)
    rule = ir.layers.L3_cmmm.emit_templates['T_ATTACK'].events[0]
    assert rule.condition == 'gate_rise'
    assert rule.payload_template.variant == 'enable'
    assert rule.payload_template.target_id == 3
    assert rule.strength == '5'
    wire = compile_ir_events(ir).emit('T_ATTACK', {'gate_rise': 1})
    # Independent MSB-lane layout: lane=1, variant=1, target op=3.
    assert wire.event_payload_u64[1] == 0x1100000300000000
    assert decode_wire_events(wire)[0].target_id == 3


@pytest.mark.parametrize('extra', [{'condition': 'never'}, {'payload_template': {'p0': 3}}, {'variant': 2}])
def test_conflicting_aliases_are_not_silently_preferred(extra):
    row = event(0, 'gate_rise', variant=1)
    row.update(extra)
    with pytest.raises(ValueError, match='Conflicting'):
        load_instrument_from_dict(author([row]))


@pytest.mark.parametrize('row', [event(7), event(True), event('EVT_CUSP'),
    event('EVT_CARRIER_CFG'), event(0, op='SwapBands'), event(0, p0=256),
    event(0, target_kind='voice_id'), event(1, target_id=24), event(1, p0=1),
    event(2, p0=99), event(5, p2=1), event(0, carrier_model='saw'),
    event(0, strength='macro(M_UNDECLARED)'), event(0, when='__import__("os")'),
    event(6), event(0, p1=1), event(4, p2=1), event(0, p0=2),
    event(2, p1=1), event(3, p1=1), event(0, target_kind='pattern_id', pattern_id=255),
])
def test_invalid_payloads_and_unresolved_program_inputs_fail_compile(row):
    source = author([row])
    with pytest.raises(PresetCompileError):
        PresetCompiler().compile(source)
    result = CapsuleCompiler().compile_from_ir(load_instrument_from_dict(source))
    assert not result.success and result.errors


def test_strongest_wins_ties_keep_order_overlays_merge_and_mask_before_evaluation():
    source = author([event(0, strength=12000, target_kind='band_id', target_id=2),
                     event(0, strength=20000, target_kind='band_id', target_id=3),
                     event(0, strength=20000, target_kind='band_id', target_id=4),
                     event(1, when='gesture.gate', strength=0, variant='enable', target_id=3)])
    source['instrument_ir_v0']['layers']['L3_cmmm']['overlays'] = [{
        'id': 'loud', 'enable_when': 'risk > 0.5', 'add': {'events': [
            event(0, strength=21000, target_kind='band_id', target_id=5)]}}]
    program = compile_ir_events(load_instrument_from_dict(source))
    quiet = program.emit('T_ATTACK', {'risk': 0, 'gesture.gate': 1})
    assert quiet.event_mask == 3 and quiet.event_strength_q15[:2] == (20000, 0)
    assert decode_wire_events(quiet)[0].target_id == 3
    loud = program.emit('T_ATTACK', {'risk': 20000, 'gesture.gate': 1})
    assert decode_wire_events(loud)[0].target_id == 5
    # Masked lane 1 must not even read its missing gesture input.
    assert program.emit('T_ATTACK', {'risk': 0}, allowed_mask=1).event_mask == 1


def test_macro_defaults_overrides_and_capsule_program_roundtrip():
    source = author([event(0, when='macro(M_CUSTOM) > 0.25', strength='macro.M_CUSTOM')])
    source['instrument_ir_v0']['layers']['L0_graph']['macros'] = [
        {'id': 'M_CUSTOM', 'name': 'Custom', 'default': 0.75}]
    program = compile_ir_events(load_instrument_from_dict(source))
    assert program.emit('T_ATTACK', {}).event_strength_q15[0] == 24575
    assert program.emit('T_ATTACK', {'macro.M_CUSTOM': 0}).event_mask == 0
    restored = CompiledEventProgram.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored == program and restored is not program


def test_lifecycle_edges_exit_entry_and_dwell_use_private_integer_clocks():
    source = author([event(0, 'gate_rise', strength=10000), event(1, 'regime_exit', strength=1, variant='enable', target_id=3)])
    root = source['instrument_ir_v0']['layers']
    root['L3_cmmm']['emit_templates']['T_STABLE']['events'] = [
        event(2, 'regime_enter', strength=1, variant='snap_b'),
        event(5, 'dwell_done', strength=20000, p0=3)]
    root['L2_regimes']['regime_nodes'][1]['dwell_ms'] = 4  # Three 64-sample blocks.
    program = compile_ir_events(load_instrument_from_dict(source))
    state = AuthoredEventState()
    assert state.emit(program, 'T_ATTACK', {}, gate=True, regime=1).event_mask == 1
    assert state.emit(program, 'T_ATTACK', {}, gate=True, regime=1).event_mask == 0
    assert state.emit(program, 'T_STABLE', {}, gate=True, regime=2).event_mask == 6
    for _ in range(2):
        assert state.emit(program, 'T_STABLE', {}, gate=True, regime=2).event_mask == 0
    assert state.emit(program, 'T_STABLE', {}, gate=True, regime=2).event_mask == 32
    assert state.emit(program, 'T_RECOVERY', {}, gate=False, regime=6).event_mask == 32
    assert state.emit(program, 'T_RECOVERY', {}, gate=False, regime=6).event_mask == 0
    state.rise_pending = True
    state.fall_pending = True
    program.templates['T_ATTACK'].append(program.templates['T_RECOVERY'][0])
    assert state.emit(program, 'T_ATTACK', {}, gate=True, regime=1).event_mask == 33


@pytest.mark.parametrize('route', ['direct', 'capsule'])
def test_authoring_through_live_voice_and_vm_changes_audio(route, execution_path):
    ir = load_instrument_from_dict(author())
    if route == 'direct':
        preset = compiled_preset_from_ir(ir)
    else:
        result = CapsuleCompiler().compile_from_ir(ir)
        assert result.success, result.errors
        preset = compiled_preset_from_capsule(capsule_from_bytes(capsule_to_bytes(result.capsule)))
    producer = engine(preset)
    actual, baseline = VMState.create(0), VMState.create(0)
    changed = 0
    for block in range(15):
        frame = producer.step_voice_simple(0)
        assert frame.event_commands is not None
        if block == 0:
            assert frame.event_commands.event_mask == 1
        clean = replace(frame, event_commands=None)
        expected = vm_step_block(baseline, clean)[1].copy()
        output = vm_step_block(actual, frame)[1].copy()
        changed += np.count_nonzero(output != expected)
    assert changed > 0
    producer.note_off_simple(0)
    released = producer.step_voice_simple(0)
    report = ExecutionReport()
    vm_step_block(actual, released, report=report)
    assert [e.lane for e in report.consumer_events] == [5]


def test_ir_frame_builder_emits_commands_without_replacing_fm_or_gain_configuration():
    ir = load_instrument_from_dict(author())
    dirty = IRFrameBuilder(ir, 69)
    clean_ir = deepcopy(ir)
    for template in clean_ir.layers.L3_cmmm.emit_templates.values():
        template.events.clear()
    clean = IRFrameBuilder(clean_ir, 69)
    for block in range(8):
        a, b = dirty.build(NoteClock(block)), clean.build(NoteClock(block))
        assert a.event_commands is not None and b.event_commands is None
        assert replace(a, event_commands=None) == b
    assert ir.layers.L3_cmmm.emit_templates['T_ATTACK'].events[0].condition == 'gate_rise'


def test_native_event_commands_use_existing_toggle_and_tree_consumers():
    program = compile_ir_events(load_instrument_from_dict(author([
        event(1, variant='disable', target_id=3), event(2, variant='snap_b', p0=4)])))
    frame = VoiceFrame(note=69, coeff_osc_mode=1, ops_enable=['DIFFUSE_DOWN'],
                       event_commands=program.emit('T_ATTACK', {}))
    state = VMState.create(0)
    vm_step_block(state, frame)
    assert state.wire_state.forced_off == 1 << 3
    assert state.tree_b_id == 4 and state.wire_state.tree_blend_q15 == 32767
    saved = deepcopy(state)
    vm_step_block(state, frame, policy=ExecutionPolicy(serviced=False))
    assert_state_equal(saved, state)


def test_native_commands_packet_export_preserves_payload_and_rejects_implicit_migration():
    program = compile_ir_events(load_instrument_from_dict(author()))
    frame = VoiceFrame(event_commands=program.emit('T_ATTACK', {'gate_rise': 1}, sequence=37))
    packet = VoiceFramePacketV0b.from_voice_frame(frame)
    assert packet.frame_seq == 37
    assert packet.event_payload_u64[0] == frame.event_commands.event_payload_u64[0]
    assert packet.event_strength_q15[0] == 20000
    with pytest.raises(ValueError, match='Cross-version'):
        VoiceFramePacketV1.from_voice_frame(frame)
    frame.wire_data = frame.event_commands
    with pytest.raises(ValueError, match='both'):
        VoiceFramePacketV0b.from_voice_frame(frame)
    with pytest.raises(ValueError, match='both'):
        vm_step_block(VMState.create(), frame)


def test_cmmm_program_uses_authored_wire_events_and_has_no_duplicate_builtin_bursts():
    plan = compile_ir_events(load_instrument_from_dict(author([event(0, 'gate_rise', strength=24000)])))
    producer = VoiceEngine()
    producer.programs[42] = InstrumentProgram(program_id=42, authored_events=plan)
    producer.allocate_voice(0, program_id=42)
    producer.note_on(0)
    output = producer.step_block()
    frame = output.render_batch[0].frame
    assert frame.cmmm_events.active_mask == 0
    assert frame.event_commands.event_mask == 1
    producer.note_on(0)  # Retrigger while gate remains up is another edge.
    assert producer.step_block().render_batch[0].frame.event_commands.event_mask == 1


def test_event_snapshot_and_service_skips_preserve_future_frames_and_audio():
    source = engine(compiled_preset_from_ir(load_instrument_from_dict(author())))
    vm = VMState.create(0)
    for _ in range(7):
        vm_step_block(vm, source.step_voice_simple(0))
    before = source.voices[0].snapshot()
    source.step_block(schedule=schedule())
    assert source.voices[0].snapshot() == before
    restored = VoiceEngine()
    restored.voices[0] = VoiceStateWithProgram.restore(json.loads(json.dumps(before)))
    from symcrash_python_ap.voice.voice_state_codec import vmstate_from_bytes, vmstate_to_bytes
    restored_vm = vmstate_from_bytes(vmstate_to_bytes(vm))
    for block in range(14):
        if block == 4:
            source.note_off_simple(0)
            restored.note_off_simple(0)
        a, b = source.step_voice_simple(0), restored.step_voice_simple(0)
        assert a == b
        if a is not None:
            expected = vm_step_block(vm, a)[1].copy()
            np.testing.assert_array_equal(vm_step_block(restored_vm, b)[1], expected)
    assert source.voices[0].snapshot() == restored.voices[0].snapshot()


def test_previous_voice_snapshot_version_migrates_without_inventing_events():
    preset = compiled_preset_from_ir(load_instrument_from_dict(author()))
    preset.event_program = None
    source = engine(preset)
    source.step_voice_simple(0)
    old = source.voices[0].snapshot()
    old['snapshot_version'] = 1
    del old['authored_event_state']
    del old['compiled_preset']['fields']['event_program']
    frozen = deepcopy(old)
    restored = VoiceStateWithProgram.restore(old)
    assert old == frozen
    assert restored.authored_event_state == AuthoredEventState()
    assert restored.compiled_preset.event_program is None
    current = restored.snapshot()
    assert current['snapshot_version'] == 7
    del current['authored_event_state']
    with pytest.raises(ValueError, match='Incomplete'):
        VoiceStateWithProgram.restore(current)


def test_authored_program_and_state_replay_in_fresh_process(tmp_path):
    producer = engine(compiled_preset_from_ir(load_instrument_from_dict(author())))
    for _ in range(5):
        producer.step_voice_simple(0)
    saved = tmp_path / 'voice.json'
    saved.write_text(json.dumps(producer.voices[0].snapshot()), encoding='utf-8')
    expected = [asdict(producer.step_voice_simple(0).event_commands) for _ in range(6)]
    script = '''
import json, sys
from pathlib import Path
from dataclasses import asdict
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
engine = VoiceEngine()
engine.voices[0] = VoiceStateWithProgram.restore(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
print(json.dumps([asdict(engine.step_voice_simple(0).event_commands) for _ in range(6)]))
'''
    result = subprocess.run([sys.executable, '-c', script, str(saved)], check=True,
                            capture_output=True, text=True, timeout=60)
    assert json.loads(result.stdout) == json.loads(json.dumps(expected)), result.stderr


def test_final_release_block_delivers_material_event_to_actual_output_owner():
    from symcrash_python_ap.envelope_scheduler import ADSRSpec
    from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
    from symcrash_python_ap.frames import ParamFrame
    preset = compiled_preset_from_ir(load_instrument_from_dict(author()))
    preset.amp_env = ADSRSpec(attack_ms=1, decay_ms=1, sustain=1.0, release_ms=1)
    producer = engine(preset)
    output = UnifiedOutputStage()
    for _ in range(3):
        frame = producer.step_voice_simple(0)
        output.render_frames([frame], ParamFrame())
    producer.note_off_simple(0)
    final = producer.step_voice_simple(0)
    assert final is not None and final.event_commands.event_mask == 32
    assert not final.is_active and final.gain_q15 == 0
    output.render_frames([final], ParamFrame())
    assert output.post_mix_controls.bursts
    assert output.last_post_mix_params.drive_q15 > ParamFrame().drive_q15
    assert producer.step_voice_simple(0) is None


def test_cmmm_frame_conversion_never_advances_event_time():
    producer = VoiceEngine()
    producer.programs[42] = InstrumentProgram(program_id=42,
        authored_events=compile_ir_events(load_instrument_from_dict(author())))
    producer.allocate_voice(0, program_id=42)
    producer.note_on(0)
    abi = producer.step_voice(0)
    assert abi.event_commands.event_mask == 1
    saved = producer.voices[0].snapshot()
    a = producer._abi_to_voice_frame(abi, 0)
    b = producer._abi_to_voice_frame(abi, 0)
    assert a == b and a.event_commands == abi.event_commands
    assert producer.voices[0].snapshot() == saved
    with pytest.raises(ValueError, match='Legacy VoiceFrameABI'):
        abi.to_bytes()
    assert VoiceFramePacketV0b.from_voice_frame(a).event_mask == 1


@pytest.mark.parametrize('retrigger', [False, True])
def test_exit_rules_observe_departing_dwell_and_age_before_clock_reset(retrigger):
    program = compile_ir_events(load_instrument_from_dict(author([
        event(0, 'regime_exit and dwell_done and age >= 4', '20000')])))
    program.regime_dwell_blocks = {1: 2, 2: 90}
    state = AuthoredEventState()
    for _ in range(3):
        assert state.emit(program, 'T_ATTACK', {}, gate=True, regime=1).event_mask == 0
    state.rise_pending = retrigger
    output = state.emit(program, 'T_ATTACK' if retrigger else 'T_STABLE',
        {'block_index': 4}, gate=True, regime=1 if retrigger else 2)
    assert output.event_mask == 1 and output.event_strength_q15[0] == 20000
    assert state.note_blocks == (1 if retrigger else 4)


@pytest.mark.parametrize('payload', [dict(pattern_id=1), dict(target_id=1),
                                    dict(target_kind='pattern_id', pattern_id=1, target_id=1)])
def test_unused_payload_selectors_are_rejected_instead_of_silently_ignored(payload):
    with pytest.raises(ValueError):
        compile_ir_events(load_instrument_from_dict(author([event(0, **payload)])))


def test_offline_clock_and_explicit_regime_template_drive_event_edges():
    ir = load_instrument_from_dict(author())
    builder = IRFrameBuilder(ir, 69)
    builder.build(NoteClock(0, sustain_until_block=3))
    released = builder.build(NoteClock(3, sustain_until_block=3), template_name='T_RECOVERY')
    assert released.event_commands.event_mask == 32
    held = builder.build(NoteClock(4, sustain_until_block=3), template_name='T_RECOVERY')
    assert held.event_commands.event_mask == 0


def test_offline_early_release_selects_the_release_template_without_an_override():
    builder = IRFrameBuilder(load_instrument_from_dict(author()), 69)
    builder.attack_blocks = 20
    builder.build(NoteClock(0, sustain_until_block=1))
    released = builder.build(NoteClock(1, sustain_until_block=1))
    assert builder.phase_at(NoteClock(1, sustain_until_block=1)) == 'RELEASE'
    assert released.event_commands.event_mask == 32


def test_offline_render_api_delivers_attack_release_and_normalized_risk():
    from tools.render.render_preset import render_preset_note
    ir = load_instrument_from_dict(author([
        event(0, 'gate_rise and risk == 0.5', '20000')]))
    frames = []

    def capture(frame):
        frames.append(frame)
        return frame

    audio = render_preset_note(ir, note=69, duration_sec=0.004,
        risk_profile=[(0, 0.5)], frame_transform=capture)
    assert len(audio) == 64 * len(frames) and np.any(audio)
    assert frames[0].event_commands.event_mask == 1
    assert sum(bool(frame.event_commands.event_mask & 32) for frame in frames) == 1
    assert frames[3].event_commands.event_mask == 32


def test_macro_overrides_are_per_voice_snapshotable_and_do_not_change_program_defaults():
    source = author([event(0, when='gate', strength='macro(M_CUSTOM)')])
    source['instrument_ir_v0']['layers']['L0_graph']['macros'] = [
        {'id': 'M_CUSTOM', 'name': 'Custom', 'default': 0.5}]
    preset = compiled_preset_from_ir(load_instrument_from_dict(source))
    producer = engine(preset)
    producer.allocate_voice_from_preset(preset, voice_id=1)
    producer.note_on_simple(1, note=69)
    producer.set_event_macro(0, 'M_CUSTOM', 23456)
    assert producer.step_voice_simple(0).event_commands.event_strength_q15[0] == 23456
    assert producer.step_voice_simple(1).event_commands.event_strength_q15[0] == 16384
    assert preset.event_program.macros['M_CUSTOM'] == 16384
    restored = VoiceStateWithProgram.restore(json.loads(json.dumps(producer.voices[0].snapshot())))
    assert restored.authored_event_state.macros == {'M_CUSTOM': 23456}
    with pytest.raises(ValueError, match='Undeclared'):
        producer.set_event_macro(0, 'M_OTHER', 1)
    with pytest.raises(ValueError):
        producer.set_event_macro(0, 'M_CUSTOM', 1.5)


@pytest.mark.parametrize('field,value', [('sequence', -1), ('sequence', 1 << 32),
    ('regime', 7), ('template_blocks', -1), ('note_blocks', -1), ('previous_gate', 1)])
def test_malformed_event_checkpoint_state_is_rejected(field, value):
    producer = engine(compiled_preset_from_ir(load_instrument_from_dict(author())))
    snapshot = producer.voices[0].snapshot()
    snapshot['authored_event_state']['fields'][field] = value
    with pytest.raises(ValueError):
        VoiceStateWithProgram.restore(snapshot)


@pytest.mark.parametrize('change', [
    lambda p: p.update(version=True), lambda p: p.update(version=2),
    lambda p: p.pop('macros'), lambda p: p['regime_templates'].update({'01': 'T_ATTACK'}),
    lambda p: p['regime_templates'].update({'1': 'missing'}),
    lambda p: p['templates']['T_ATTACK'][0].update(payload=-1),
    lambda p: p['templates']['T_ATTACK'][0].update(condition='__import__("os")'),
    lambda p: p['templates']['T_ATTACK'][0].update(strength='macro(M_UNKNOWN)'),
])
def test_malformed_compiled_program_never_bypasses_validation(change):
    program = compile_ir_events(load_instrument_from_dict(author()))
    data = json.loads(json.dumps(program.to_dict()))
    change(data)
    with pytest.raises(ValueError):
        CompiledEventProgram.from_dict(data)


def test_sound_design_behavior_events_survive_translation_and_compilation():
    data = yaml.safe_load((Path(__file__).resolve().parents[2] /
        'presets/instruments/dx_marimba_v0.yaml').read_text(encoding='utf-8'))
    data['instrument_ir_v0']['behavior']['attack']['events'] = [event(0, 'gate_rise', strength=20000)]
    preset = PresetCompiler().compile(data)
    assert engine(preset).step_voice_simple(0).event_commands.event_mask == 1


@pytest.mark.parametrize('where', ['event', 'payload'])
def test_unrecognized_authored_fields_are_rejected_before_they_can_disappear(where):
    row = event()
    (row if where == 'event' else row['payload'])['invented'] = 1
    with pytest.raises(ValueError, match='Unsupported event'):
        load_instrument_from_dict(author([row]))
