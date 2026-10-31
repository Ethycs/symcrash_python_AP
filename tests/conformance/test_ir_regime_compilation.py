"""Execute authored graphs through real preset, capsule, offline and VM paths."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap import instrument_ir as ir
from symcrash_python_ap.frames import VoiceFramePacketV0b, VoiceFramePacketV1
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.capsule_codec import capsule_to_bytes, capsule_from_bytes
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir, compiled_preset_from_capsule
from symcrash_python_ap.voice.ir_regimes import compile_ir_regimes, CompiledRegimeProgram, AuthoredRegimeState
from symcrash_python_ap.voice.ir_events import event_context, compile_ir_events
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
from symcrash_python_ap.voice.frame_builder import IRFrameBuilder, NoteClock
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.decode import decode_frame
from tests.conformance.test_vm_snapshot_replay import execution_path
from tests.conformance.test_scheduled_vm import schedule


ROOT = Path(__file__).resolve().parents[2]


def author(duration=3):
    source = ir.load_instrument(ROOT / 'presets/instruments/dx_marimba_v0.yaml')
    source.layers.L2_regimes = ir.L2Regimes(regime_nodes=[
        ir.RegimeNode('waiting', 'Idle', 'quiet'),
        ir.RegimeNode('first', 'Transient', 'one', dwell_ms=4),
        ir.RegimeNode('second', 'StableTone', 'two'),
        ir.RegimeNode('last', 'Recovery', 'three'),
    ], regime_graph_spec=ir.RegimeGraphSpec(start_regime='waiting', edges=[
        ir.EdgeSpec('any', 'last', 'hard', 'Gate falls'),
        ir.EdgeSpec('waiting', 'first', 'hard', 'Gate rises'),
        ir.EdgeSpec('first', 'second', 'soft', 'dwell_done', duration_blocks=duration),
        ir.EdgeSpec('second', 'first', 'hard', 'risk > 0.7', hysteresis=ir.Hysteresis(0.7, 0.3)),
    ]))
    source.layers.L3_cmmm = ir.L3Cmmm(emit_templates={
        'quiet': ir.EmitTemplate(),
        'one': ir.EmitTemplate(ops_enable=['Tilt'], families={'COLOR': 1.0}),
        'two': ir.EmitTemplate(ops_enable=['IntraMix'], families={'LOCAL': 1.0}),
        'three': ir.EmitTemplate(ops_enable=['GluePair'], families={'GLUE': 0.5},
            trees=ir.TreeTarget('harmonic', 'texture', 0.5),
            cmmm=ir.CmmmControl(node_mask=0xff, mask_ramp_blocks=2, tail_mode='MID', traction_q15=12345)),
    })
    return source


def engine(source=None, route='direct'):
    source = source or author()
    if route == 'direct':
        preset = compiled_preset_from_ir(source)
    else:
        result = CapsuleCompiler().compile_from_ir(source)
        assert result.success, result.errors
        preset = compiled_preset_from_capsule(capsule_from_bytes(capsule_to_bytes(result.capsule)))
    producer = VoiceEngine(max_voices=4)
    producer.allocate_voice_from_preset(preset, voice_id=0)
    producer.note_on_simple(0, note=69, velocity_q15=28000)
    return producer


def step(program, state, *, gate=True, risk=0, **inputs):
    context = event_context(risk=risk)
    context.update(inputs)
    return state.step(program, context, gate=gate)[0]


def test_names_are_distinct_and_typed_edges_override_legacy_transitions():
    source = author()
    source.layers.L2_regimes.transitions = [ir.RegimeTransition('waiting', 'last', 'always')]
    program = compile_ir_regimes(source)
    assert len({n.event_id for n in program.nodes.values()}) == 4
    state = AuthoredRegimeState()
    assert step(program, state) == {'Tilt': 32767}
    assert state.node == 'first'
    assert CompiledRegimeProgram.from_dict(json.loads(json.dumps(program.to_dict()))) == program


@pytest.mark.parametrize('duration', [0, 1, 2, 3, 4, 17])
def test_soft_edges_have_exact_integer_endpoints_and_authored_dwell(duration):
    program = compile_ir_regimes(author(duration))
    state = AuthoredRegimeState()
    for i in range(3):
        assert step(program, state) == {'Tilt': 32767}
        assert state.node == 'first' and state.blocks == i + 1
    for i in range(max(1, duration)):
        actual = step(program, state)
        expected = ({'Tilt': 32767 * (duration-i-1) // duration,
                     'IntraMix': 32767 * (i+1) // duration} if duration else {'IntraMix': 32767})
        assert actual == expected
        assert state.node == 'second'
    assert step(program, state) == {'IntraMix': 32767}


def test_interrupted_soft_transition_uses_last_emitted_weights():
    source = author(6)
    source.layers.L2_regimes.regime_graph_spec.edges[0] = ir.EdgeSpec(
        'any', 'last', 'soft', 'Gate falls', duration_blocks=2)
    program = compile_ir_regimes(source)
    state = AuthoredRegimeState()
    for _ in range(4):
        previous = step(program, state)
    actual = step(program, state, gate=False)
    assert actual == {'Tilt': previous['Tilt']//2, 'IntraMix': previous['IntraMix']//2, 'GluePair': 8192}
    assert state.node == 'last'
    assert step(program, state, gate=False) == {'Tilt': 0, 'IntraMix': 0, 'GluePair': 16384}


def test_hysteresis_rearms_across_nodes_only_beyond_exit_threshold():
    program = compile_ir_regimes(author(0))
    state = AuthoredRegimeState()
    for _ in range(4):
        step(program, state)
    step(program, state, risk=round(.7*32767))
    assert state.node == 'second'
    step(program, state, risk=round(.7*32767)+1)
    assert state.node == 'first' and state.latched_edges == [3]
    for _ in range(8):
        step(program, state, risk=32767)
    assert state.node == 'second'
    step(program, state, risk=round(.3*32767))
    assert state.latched_edges == [3]
    step(program, state, risk=round(.3*32767)-1)
    assert state.latched_edges == []
    step(program, state, risk=32767)
    assert state.node == 'first'


@pytest.mark.parametrize('guard,inputs,eligible', [
    ('pressure > 0.5', {'gesture.pressure_q15': 16385}, True),
    ('pressure > 0.5', {'gesture.pressure_q15': 16384}, False),
    ('chaos > 0.5', {'gesture.chaos_q15': 16385}, True),
    ('energy < 0.5', {'energy_state': 16383}, True),
    ('energy < 0.5', {'energy_state': 16384}, False),
    ('age > 2', {}, False), ('age > 1', {}, True),
    ('gate and not gate_fall', {}, True),
])
def test_documented_guard_units_and_strict_boundaries(guard, inputs, eligible):
    source = author()
    source.layers.L2_regimes.regime_graph_spec.edges[2] = ir.EdgeSpec('first', 'second', 'hard', guard)
    program = compile_ir_regimes(source)
    state = AuthoredRegimeState()
    step(program, state)
    step(program, state, **inputs)  # age = floor(4/3) = 1 ms
    step(program, state, **inputs)  # age = floor(8/3) = 2 ms
    assert (state.node == 'second') is eligible


def test_wildcard_order_and_min_dwell_gate_edges_are_not_queued_implicitly():
    source = author()
    source.layers.L2_regimes.regime_graph_spec.edges[0].min_dwell_blocks = 2
    program = compile_ir_regimes(source)
    state = AuthoredRegimeState()
    step(program, state)
    step(program, state, gate=False)
    assert state.node == 'first'
    step(program, state, gate=False)
    assert state.node == 'first'  # gate fall was an edge, not a latched level
    state.fall_pending = True
    step(program, state, gate=False)
    assert state.node == 'last'


def test_template_fields_and_exact_weights_reach_decoder_without_float_loss():
    source = author(0)
    source.layers.L3_cmmm.emit_templates['one'].families = {'Tilt': '49'}
    producer = engine(source)
    frame = producer.step_voice_simple(0)
    assert frame.operator_strengths_q15 == {'Tilt': 49}
    plan = decode_frame(frame)
    assert plan.backbone_instrs[0].strength_q15 == 49
    producer.note_off_simple(0)
    frame = producer.step_voice_simple(0)
    assert (frame.tree_a_id, frame.tree_b_id, frame.tree_blend_q15) == (0, 3, 16384)
    assert (frame.cmmm_node_mask, frame.cmmm_mask_ramp_blocks, frame.cmmm_tail_mode,
            frame.cmmm_traction_q15) == (0xff, 2, 1, 12345)
    assert frame.fm_op_ratios == producer.voices[0].compiled_preset.fm6.op_ratios


def test_live_and_offline_use_the_same_authored_state_and_emission_sequence():
    source = author()
    producer = engine(source)
    builder = IRFrameBuilder(source, 69, 28000)
    original = deepcopy(source)
    for block in range(16):
        if block == 10:
            producer.note_off_simple(0)
        live = producer.step_voice_simple(0)
        offline = builder.build(NoteClock(block, sustain_until_block=10))
        assert producer.voices[0].authored_regime_state.node == builder._regime_state.node
        for name in ('ops_enable', 'operator_strengths_q15', 'tree_a_id', 'tree_b_id',
                     'tree_blend_q15', 'cmmm_node_mask', 'cmmm_tail_mode', 'cmmm_traction_q15'):
            assert getattr(live, name) == getattr(offline, name)
    assert source == original


@pytest.mark.parametrize('route', ['direct', 'capsule'])
def test_actual_graph_playback_changes_audio_and_preserves_native_carrier(route, execution_path):
    producer = engine(route=route)
    actual_vm, old_vm = VMState.create(0), VMState.create(0)
    changed = 0
    for _ in range(18):
        frame = producer.step_voice_simple(0)
        assert frame.coeff_osc_mode == 3 and frame.fm_op_ratios
        former = replace(frame, ops_enable=[], families={}, operator_strengths_q15=None)
        a = vm_step_block(actual_vm, frame)[1].copy()
        b = vm_step_block(old_vm, former)[1].copy()
        changed += np.count_nonzero(a != b)
    assert changed > 0


def test_arbitrary_named_regimes_select_entry_exit_events_and_keep_distinct_ids():
    source = author(0)
    for template in source.layers.L3_cmmm.emit_templates.values():
        template.events = [ir.EventPolicy('EVT_MATERIAL_HIT', 'regime_enter', strength='12000')]
    program = compile_ir_events(source)
    producer = engine(source)
    frames = [producer.step_voice_simple(0) for _ in range(6)]
    assert [f.event_commands.event_mask for f in frames] == [32, 0, 0, 32, 0, 0]
    state = producer.voices[0]
    assert state.authored_event_state.regime == state.compiled_preset.regime_program.nodes['second'].event_id
    assert state.authored_event_state.regime in program.regime_templates
    VoiceStateWithProgram.restore(json.loads(json.dumps(state.snapshot())))


def test_snapshot_replays_mid_fade_hysteresis_and_future_audio(execution_path):
    producer = engine(author(8))
    vm = VMState.create(0)
    for _ in range(5):
        vm_step_block(vm, producer.step_voice_simple(0))
    snapshot = json.loads(json.dumps(producer.voices[0].snapshot()))
    assert snapshot['snapshot_version'] == 7
    restored = VoiceEngine()
    restored.voices[0] = VoiceStateWithProgram.restore(snapshot)
    restored_vm = VMState.restore(json.loads(json.dumps(vm.snapshot())))
    for block in range(22):
        for owner in (producer, restored):
            owner.voices[0].gesture.chaos_q15 = 32767 if block < 16 else 0
            if block == 18:
                owner.note_off_simple(0)
        a, b = producer.step_voice_simple(0), restored.step_voice_simple(0)
        assert a == b
        np.testing.assert_array_equal(vm_step_block(vm, a)[1], vm_step_block(restored_vm, b)[1])
        assert producer.voices[0].snapshot() == restored.voices[0].snapshot()


@pytest.mark.parametrize('version', [1, 2, 3])
def test_old_snapshot_migration_does_not_invent_a_graph_or_change_input(version):
    source = author()
    source.layers.L2_regimes.regime_graph_spec = None
    producer = engine(source)
    snapshot = producer.voices[0].snapshot()
    snapshot['snapshot_version'] = version
    del snapshot['authored_regime_state']
    del snapshot['compiled_preset']['fields']['regime_program']
    original = deepcopy(snapshot)
    restored = VoiceStateWithProgram.restore(snapshot)
    assert restored.authored_regime_state == AuthoredRegimeState()
    assert restored.compiled_preset.regime_program is None and snapshot == original


@pytest.mark.parametrize('field,value', [('node', 'missing'), ('blocks', -1),
    ('fade_blocks', 999), ('active_edge', 100), ('latched_edges', [0]), ('previous_gate', 1)])
def test_invalid_snapshot_graph_state_is_rejected(field, value):
    producer = engine()
    producer.step_voice_simple(0)
    snapshot = producer.voices[0].snapshot()
    snapshot['authored_regime_state']['fields'][field] = value
    with pytest.raises(ValueError):
        VoiceStateWithProgram.restore(snapshot)


def test_voice_graphs_are_private_and_scheduler_holds_freeze_every_clock():
    producer = engine()
    preset = producer.voices[0].compiled_preset
    original = deepcopy(preset)
    producer.allocate_voice_from_preset(preset, voice_id=1)
    producer.note_on_simple(1)
    producer.step_voice_simple(0)
    assert producer.voices[1].authored_regime_state.node == ''
    before = producer.voices[0].snapshot()
    producer.step_block(schedule=schedule())
    assert producer.voices[0].snapshot() == before
    assert preset == original


def test_retrigger_resets_graph_and_preserves_same_block_gate_fall():
    producer = engine()
    for _ in range(8):
        producer.step_voice_simple(0)
    producer.note_off_simple(0)
    producer.note_on_simple(0)
    producer.step_voice_simple(0)
    assert producer.voices[0].authored_regime_state.node == 'last'  # first wildcard wins both edges


def test_graph_identity_includes_order_dwell_hysteresis_and_typed_edges():
    compiler = CapsuleCompiler()
    source = author()
    baseline = compiler._extract_regime_graph_id(source)
    for mutate in (lambda x: x.regime_nodes[1].__setattr__('dwell_ms', 6),
                   lambda x: x.regime_graph_spec.edges.reverse(),
                   lambda x: x.regime_graph_spec.edges[2].__setattr__('duration_blocks', 9),
                   lambda x: x.regime_graph_spec.edges[3].hysteresis.__setattr__('exit', .2)):
        changed = deepcopy(source)
        mutate(changed.layers.L2_regimes)
        assert compiler._extract_regime_graph_id(changed) != baseline


def test_fresh_process_restores_the_graph_without_a_registry(tmp_path):
    producer = engine()
    for _ in range(4):
        producer.step_voice_simple(0)
    path = tmp_path / 'voice.json'
    path.write_text(json.dumps(producer.voices[0].snapshot()), encoding='utf-8')
    expected = [asdict(producer.step_voice_simple(0)) for _ in range(9)]
    code = '''import json, sys
from dataclasses import asdict
from symcrash_python_ap.voice.voice_engine import VoiceEngine, VoiceStateWithProgram
e = VoiceEngine()
e.voices[0] = VoiceStateWithProgram.restore(json.load(open(sys.argv[1])))
print(json.dumps([asdict(e.step_voice_simple(0)) for _ in range(9)]))
'''
    result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == json.loads(json.dumps(expected))


@pytest.mark.parametrize('packet', [VoiceFramePacketV0b, VoiceFramePacketV1])
def test_frozen_packets_cannot_silently_drop_per_operator_crossfade_weights(packet):
    frame = engine().step_voice_simple(0)
    with pytest.raises(ValueError, match='per-operator'):
        packet.from_voice_frame(frame)


def test_stock_glitch_graph_obeys_gate_dwell_risk_hysteresis_and_recovery():
    source = ir.load_instrument(ROOT / 'presets/instruments/glitch_lead_v0.yaml')
    producer = engine(source)
    states = []
    for block in range(160):
        producer.voices[0].gesture.chaos_q15 = 32767
        if block == 150:
            producer.note_off_simple(0)
        producer.step_voice_simple(0)
        states.append(producer.voices[0].authored_regime_state.node)
    assert states[:11] == ['R1_ATTACK'] * 11
    assert states[11] == 'R2_STABLE'
    assert states[12:72] == ['R3_GLITCH'] * 60
    assert states[72:117] == ['R4_RECOVERY'] * 45
    assert states[117:150] == ['R2_STABLE'] * 33
    assert states[150:] == ['R0_IDLE'] * 10


@pytest.mark.parametrize('field,value', [
    ('start', 'missing'), ('version', 9), ('version', True),
])
def test_bad_compiled_program_header_fails(field, value):
    data = compile_ir_regimes(author()).to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        CompiledRegimeProgram.from_dict(data)


@pytest.mark.parametrize('field,value', [('source', 'missing'), ('target', 'missing'),
    ('guard', 'open("x")'), ('guard', 'macro(M_MISSING)'), ('min_dwell', -1),
    ('min_dwell', True), ('duration', -1), ('overlay', 'missing'),
    ('hysteresis_input', 'risk + 1'), ('enter', 40000), ('exit', 30000)])
def test_bad_compiled_edge_fails(field, value):
    data = compile_ir_regimes(author()).to_dict()
    data['edges'][3][field] = value
    with pytest.raises(ValueError):
        CompiledRegimeProgram.from_dict(data)


@pytest.mark.parametrize('field,value', [('ops', ['MissingOp']), ('ops', ['Tilt', 'TILT']),
    ('families', {'missing': '12000'}), ('families', {'COLOR': 'macro(M_MISSING)'}),
    ('tree_a', 9), ('blend', '2 ** 100'), ('mask', -1), ('tail', 9), ('traction', 32768)])
def test_bad_compiled_template_fails(field, value):
    data = compile_ir_regimes(author()).to_dict()
    data['templates']['one'][field] = value
    with pytest.raises(ValueError):
        CompiledRegimeProgram.from_dict(data)


def test_compiled_graph_requires_every_declared_field():
    data = compile_ir_regimes(author()).to_dict()
    del data['edges'][0]['min_dwell']
    with pytest.raises(ValueError, match='Incomplete'):
        CompiledRegimeProgram.from_dict(data)


@pytest.mark.parametrize('duration', [0, 3])
def test_transition_overlay_applies_for_the_transition_then_expires(duration):
    source = author(duration)
    source.layers.L3_cmmm.emit_templates['two'].families = {'LOCAL': .25}
    source.layers.L3_cmmm.overlays = [ir.Overlay('burst', 'never', add=ir.OverlayAdd(families={'LOCAL': .5}))]
    source.layers.L2_regimes.regime_graph_spec.edges[2].overlay = 'burst'
    program = compile_ir_regimes(source)
    state = AuthoredRegimeState()
    for _ in range(3):
        step(program, state)
    for i in range(max(1, duration)):
        result = step(program, state)
        assert result['IntraMix'] == ((24576 * (i+1) // duration) if duration else 24576)
    assert step(program, state) == {'IntraMix': 8192}


def test_live_registry_family_control_scales_exact_graph_weights():
    from symcrash_python_ap.control_registry import ControlID
    producer = engine()
    producer.voices[0].controls.values[ControlID.TIMBRE] = 12345
    frame = producer.step_voice_simple(0)
    assert decode_frame(frame).backbone_instrs[0].strength_q15 == 12345


def test_graph_macros_are_private_and_work_without_an_event_program():
    source = author(0)
    from symcrash_python_ap.instrument_ir import Macro
    source.layers.L0_graph.macros = [Macro(id='M_GAIN', name='Gain', range=(-1, 1), default=-0.25)]
    source.layers.L3_cmmm.emit_templates['one'].families = {'COLOR': 'abs(macro(M_GAIN))'}
    producer = engine(source)
    assert producer.step_voice_simple(0).operator_strengths_q15 == {'Tilt': 8192}
    producer.set_event_macro(0, 'M_GAIN', 12345)
    assert producer.step_voice_simple(0).operator_strengths_q15 == {'Tilt': 12345}
    assert producer.voices[0].compiled_preset.regime_program.macros == {'M_GAIN': -8192}


def test_self_transition_resets_event_dwell_and_emits_entry_again():
    source = author(0)
    source.layers.L2_regimes.regime_graph_spec.edges[2].dst_regime = 'first'
    source.layers.L3_cmmm.emit_templates['one'].events = [
        ir.EventPolicy('EVT_MATERIAL_HIT', 'regime_enter', strength='12000')]
    producer = engine(source)
    builder = IRFrameBuilder(source, 69, 28000)
    for block in range(8):
        live = producer.step_voice_simple(0)
        offline = builder.build(NoteClock(block))
        assert live.event_commands == offline.event_commands
        assert live.event_commands.event_mask == (32 if block % 3 == 0 else 0)


def test_overlay_sum_is_independent_of_serialized_map_order():
    source = author()
    source.layers.L3_cmmm.emit_templates['one'].families = {'COLOR': .75}
    source.layers.L3_cmmm.overlays = [
        ir.Overlay('z_positive', 'always', add=ir.OverlayAdd(families={'COLOR': .75})),
        ir.Overlay('a_negative', 'always', add=ir.OverlayAdd(families={'COLOR': '-0.5'})),
    ]
    program = compile_ir_regimes(source)
    reordered = CompiledRegimeProgram.from_dict(json.loads(json.dumps(program.to_dict(), sort_keys=True)))
    assert step(program, AuthoredRegimeState()) == step(reordered, AuthoredRegimeState()) == {'Tilt': 32766}


def test_ambiguous_per_operator_aliases_fail_before_serialization():
    source = author()
    source.layers.L3_cmmm.emit_templates['one'].families = {'Tilt': .2, 'TILT': .8}
    with pytest.raises(ValueError, match='Duplicate aliases'):
        compile_ir_regimes(source)


@pytest.mark.parametrize('base,overlay', [('COLOR', 'tone'), ('Tilt', 'TILT')])
def test_overlays_combine_equivalent_family_and_operator_aliases(base, overlay):
    source = author()
    source.layers.L3_cmmm.emit_templates['one'].families = {base: .75}
    source.layers.L3_cmmm.overlays = [ir.Overlay('lower', 'always',
        add=ir.OverlayAdd(families={overlay: '-0.5'}))]
    assert step(compile_ir_regimes(source), AuthoredRegimeState()) == {'Tilt': 8191}
