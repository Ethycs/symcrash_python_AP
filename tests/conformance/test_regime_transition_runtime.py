"""Graph guards and probabilities must govern real per-voice transitions."""
from copy import deepcopy
from dataclasses import asdict
import json
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap.instrument_ir import TemperamentVector
from symcrash_python_ap.primitives import Q15_ONE, Regime
from symcrash_python_ap.voice.voice_engine import (
    ConditionSpec, EdgeSpec, EventPolicySpec, GestureBasis, InstrumentProgram,
    IntentVector, RegimeGraph, REGIME_SPECS, VoiceEngine, VoiceStateWithProgram,
    compute_transition_probability, should_transition_stochastic)
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.state import CMMMemory
from tests.conformance.test_scheduled_vm import schedule
from tests.conformance.test_vm_snapshot_replay import execution_path


def neutral():
    return TemperamentVector(hysteresis_q15=0, tension_influence_q15=0,
                             fatigue_sensitivity_q15=0, chaos_amount_q15=0)


def xorshift(seed):
    value = seed & 0xffffffff
    value ^= (value << 13) & 0xffffffff
    value ^= value >> 17
    value ^= (value << 5) & 0xffffffff
    return value & 0xffffffff


def engine(edges=None, probability=12345, temperament=None, voices=1):
    if edges is None:
        edges = [EdgeSpec(Regime.R1_ATTACK, Regime.R4_GLITCH, base_probability_q15=probability),
                 EdgeSpec(Regime.R4_GLITCH, Regime.R1_ATTACK, base_probability_q15=probability)]
    program = InstrumentProgram(program_id=60002, event_policy=EventPolicySpec(),
        regime_graph=RegimeGraph(edges=edges, regime_specs=deepcopy(REGIME_SPECS)),
        attack_blocks=1000, release_blocks=100, temperament=temperament)
    result = VoiceEngine(max_voices=voices)
    result.programs[60002] = program
    for voice_id in range(voices):
        result.allocate_voice(voice_id, program_id=60002)
        result.note_on(voice_id)
    return result


@pytest.mark.parametrize('condition,gesture,rise,fall,expected', [
    (ConditionSpec(), None, None, None, True),
    (ConditionSpec(gate_required=True), None, None, None, False),
    (ConditionSpec(gate_required=False), None, None, None, False),
    (ConditionSpec(gate_required=True), GestureBasis(gate=False), None, None, False),
    (ConditionSpec(gate_required=True), GestureBasis(gate=True), None, None, True),
    (ConditionSpec(gate_required=False), GestureBasis(gate=False), None, None, True),
    (ConditionSpec(gate_rise_required=True), GestureBasis(gate=True), False, False, False),
    (ConditionSpec(gate_rise_required=True), GestureBasis(gate=True), True, False, True),
    (ConditionSpec(gate_fall_required=True), GestureBasis(gate=False), False, False, False),
    (ConditionSpec(gate_fall_required=True), GestureBasis(gate=False), False, True, True),
    (ConditionSpec(gate_rise_required=True, gate_fall_required=True), GestureBasis(gate=True), True, True, True),
    (ConditionSpec(intent_thresholds={0: 100}), None, None, None, True),
    (ConditionSpec(intent_thresholds={0: 101}), None, None, None, False),
    (ConditionSpec(intent_ranges={0: (100, 100)}), None, None, None, True),
    (ConditionSpec(intent_ranges={0: (101, 300)}), None, None, None, False),
    (ConditionSpec(intent_ranges={0: (0, 99)}), None, None, None, False),
])
def test_graph_delivers_every_guard_form_without_inventing_missing_input(condition, gesture, rise, fall, expected):
    edge = EdgeSpec(Regime.R1_ATTACK, Regime.R2_STABLE, condition=condition)
    graph = RegimeGraph(edges=[edge])
    target, seed = graph.check_transition(Regime.R1_ATTACK, IntentVector([100] + [0] * 7),
        10, prng_state=123, gesture=gesture, gate_rise=rise, gate_fall=fall)
    assert (target == Regime.R2_STABLE) is expected
    assert seed == 123


def test_legacy_and_structured_thresholds_and_dwell_are_conjunctive():
    edge = EdgeSpec(Regime.R1_ATTACK, Regime.R2_STABLE,
        condition=ConditionSpec(intent_ranges={0: (100, 200)}),
        intent_thresholds={1: 300}, min_dwell_blocks=3)
    assert not edge.check_condition(IntentVector([100, 299] + [0] * 6), 3)
    assert not edge.check_condition(IntentVector([100, 300] + [0] * 6), 2)
    assert edge.check_condition(IntentVector([100, 300] + [0] * 6), 3)


@pytest.mark.parametrize('probability', [0, 1, 12345, 16384, 32766, 32767])
@pytest.mark.parametrize('with_memory,with_temperament', [(False, False), (True, False), (False, True), (True, True)])
def test_optional_modulation_inputs_never_disable_probability_and_draw_order(probability, with_memory, with_temperament):
    graph = RegimeGraph(edges=[EdgeSpec(Regime.R1_ATTACK, Regime.R2_STABLE,
                                      base_probability_q15=probability)])
    seed = 0x12345678
    for _ in range(40):
        expected_seed = seed if probability == Q15_ONE else xorshift(seed)
        expected = probability == Q15_ONE or (expected_seed & 0x7fff) < probability
        target, next_seed = graph.check_transition(Regime.R1_ATTACK, IntentVector(), 10,
            memory=CMMMemory() if with_memory else None,
            temperament=neutral() if with_temperament else None, prng_state=seed)
        assert (target == Regime.R2_STABLE) is expected
        assert next_seed == expected_seed
        seed = next_seed


def test_rejected_guard_spends_no_rng_and_priority_order_remains_stable():
    edges = [EdgeSpec(Regime.R1_ATTACK, Regime.R4_GLITCH, priority=10,
        condition=ConditionSpec(gate_fall_required=True), base_probability_q15=10000),
        EdgeSpec(Regime.R1_ATTACK, Regime.R2_STABLE, priority=5),
        EdgeSpec(Regime.R1_ATTACK, Regime.R3_AIR, priority=5)]
    target, seed = RegimeGraph(edges=edges).check_transition(Regime.R1_ATTACK,
        IntentVector(), 0, prng_state=42, gesture=GestureBasis(gate=True), gate_fall=False)
    assert target == Regime.R2_STABLE and seed == 42


def test_probability_unity_has_no_identity_attenuation_or_maximum_draw_hole(monkeypatch):
    edge = EdgeSpec(Regime.R1_ATTACK, Regime.R2_STABLE)
    probability, _ = compute_transition_probability(edge, IntentVector(), CMMMemory(), neutral(), 1)
    assert probability == 32767
    monkeypatch.setattr('symcrash_python_ap.vm.state.xorshift32', lambda seed: (seed + 1, 32767))
    assert should_transition_stochastic(edge, IntentVector(), CMMMemory(), neutral(), 1) == (True, 2)
    edge.base_probability_q15 = 0
    assert should_transition_stochastic(edge, IntentVector(), CMMMemory(), neutral(), 1) == (False, 2)


def test_explicit_intent_probability_influence_runs_even_at_full_base_probability():
    edge = EdgeSpec(Regime.R1_ATTACK, Regime.R2_STABLE,
        stochastic_intent_influences={0: -32767})
    target, seed = RegimeGraph(edges=[edge]).check_transition(Regime.R1_ATTACK,
        IntentVector([32767] + [0] * 7), 20, prng_state=123)
    assert target is None and seed == xorshift(123)


def test_voice_delivers_pending_edges_before_event_history_is_consumed():
    producer = engine(edges=[
        EdgeSpec(Regime.R1_ATTACK, Regime.R2_STABLE, condition=ConditionSpec(gate_rise_required=True)),
        EdgeSpec(Regime.R2_STABLE, Regime.R6_RECOVERY, condition=ConditionSpec(gate_fall_required=True))])
    assert producer.step_voice(0).regime_id == int(Regime.R2_STABLE)
    for _ in range(3):
        assert producer.step_voice(0).regime_id == int(Regime.R2_STABLE)
    producer.note_off(0)
    assert producer.step_voice(0).regime_id == int(Regime.R6_RECOVERY)


def test_both_edges_of_same_block_retrigger_reach_the_graph():
    producer = engine(edges=[EdgeSpec(Regime.R1_ATTACK, Regime.R4_GLITCH,
        condition=ConditionSpec(gate_rise_required=True, gate_fall_required=True))])
    assert producer.step_voice(0).regime_id == int(Regime.R1_ATTACK)
    producer.note_off(0)
    producer.note_on(0)
    assert producer.step_voice(0).regime_id == int(Regime.R4_GLITCH)


def test_explicit_graph_prevents_legacy_attack_fallback_from_bypassing_a_blocked_edge():
    producer = engine(probability=0)
    producer.programs[60002].attack_blocks = 1
    producer.programs[60002].sustain_regime = Regime.R4_GLITCH
    for _ in range(20):
        assert producer.step_voice(0).regime_id == int(Regime.R1_ATTACK)


def test_live_probability_trace_has_exact_draws_and_private_memory():
    producer = engine(voices=2)
    before = deepcopy(producer.programs[60002].regime_graph)
    seed, regime = 0x12345678, Regime.R1_ATTACK
    for _ in range(40):
        seed = xorshift(seed)
        if (seed & 0x7fff) < 12345:
            regime = Regime.R4_GLITCH if regime == Regime.R1_ATTACK else Regime.R1_ATTACK
        for voice_id in range(2):
            assert producer.step_voice(voice_id).regime_id == int(regime)
            assert producer.voices[voice_id].rng_u32 == seed
    assert producer.programs[60002].regime_graph == before
    assert producer.voices[0].transition_memory is not producer.voices[1].transition_memory
    producer.voices[0].transition_memory.tension_q15 = 30000
    assert producer.voices[1].transition_memory.tension_q15 != 30000
    producer.note_on(0)
    assert producer.voices[0].transition_memory == CMMMemory()


def test_transition_memory_updates_once_per_service_and_stays_frozen_when_held():
    producer = engine(probability=0)
    producer.programs[60002].gesture_to_intent = lambda gesture: IntentVector([32767] * 8)
    tension = fatigue = 0
    for i in range(12):
        producer.step_voice(0)
        tension = max(0, min(32767, tension + (32767 >> 3)) - 328)
        fatigue = max(0, min(32767, fatigue + 32 + ((32767 - 16384) >> 6)) - 164)
        memory = producer.voices[0].transition_memory
        assert (memory.tension_q15, memory.fatigue_q15, memory.blocks_since_transition) == (tension, fatigue, i + 1)
    before = producer.voices[0].snapshot()
    producer.step_block(schedule=schedule())
    assert producer.voices[0].snapshot() == before


def test_live_temperament_uses_owned_memory_and_resets_transition_age():
    producer = engine(probability=1, temperament=TemperamentVector(hysteresis_q15=0))
    memory = producer.voices[0].transition_memory
    memory.tension_q15 = 32767
    memory.fatigue_q15 = 1000
    memory.blocks_since_transition = 9
    edge = producer.programs[60002].regime_graph.edges[0]
    intent = producer.programs[60002].gesture_to_intent(producer.voices[0].gesture)
    expected, seed = should_transition_stochastic(edge, intent, deepcopy(memory),
        producer.programs[60002].temperament, producer.voices[0].rng_u32)
    producer.step_voice(0)
    assert (producer.voices[0].regime == Regime.R4_GLITCH) is expected
    assert producer.voices[0].rng_u32 == seed
    assert memory.blocks_since_transition == (0 if expected else 10)


def test_version_three_checkpoint_replays_transitions_memory_and_actual_audio(execution_path):
    original = engine(temperament=TemperamentVector(chaos_amount_q15=10000))
    vm = VMState.create(0)
    for _ in range(11):
        frame = original.step_block().render_batch[0].frame
        vm_step_block(vm, frame)
    snapshot = json.loads(json.dumps(original.voices[0].snapshot()))
    assert snapshot['snapshot_version'] == 7
    restored = VoiceEngine()
    restored.voices[0] = VoiceStateWithProgram.restore(snapshot, programs=original.programs)
    restored_vm = VMState.restore(json.loads(json.dumps(vm.snapshot())))
    nonzero = 0
    for block in range(24):
        if block == 12:
            original.note_off(0)
            restored.note_off(0)
        a = original.step_block().render_batch[0].frame
        b = restored.step_block().render_batch[0].frame
        assert a == b
        actual = vm_step_block(vm, a)[1].copy()
        replay = vm_step_block(restored_vm, b)[1].copy()
        np.testing.assert_array_equal(actual, replay)
        nonzero += np.count_nonzero(actual)
        assert original.voices[0].snapshot() == restored.voices[0].snapshot()
    assert nonzero


@pytest.mark.parametrize('old_version', [1, 2])
def test_old_checkpoint_migration_initializes_memory_without_modifying_source(old_version):
    producer = engine()
    snapshot = producer.voices[0].snapshot()
    snapshot['snapshot_version'] = old_version
    del snapshot['transition_memory']
    if old_version == 1:
        del snapshot['authored_event_state']
    before = deepcopy(snapshot)
    restored = VoiceStateWithProgram.restore(snapshot, programs=producer.programs)
    assert restored.transition_memory == CMMMemory() and snapshot == before
    current = restored.snapshot()
    del current['transition_memory']
    with pytest.raises(ValueError, match='Incomplete'):
        VoiceStateWithProgram.restore(current, programs=producer.programs)


@pytest.mark.parametrize('field,value', [('tension_q15', -1), ('fatigue_q15', 32768),
    ('harmonic_memory_q15', -1), ('blocks_since_transition', -1), ('TENSION_DECAY', -1)])
def test_invalid_transition_checkpoint_memory_fails(field, value):
    producer = engine()
    snapshot = producer.voices[0].snapshot()
    snapshot['transition_memory']['fields'][field] = value
    with pytest.raises(ValueError, match='transition memory'):
        VoiceStateWithProgram.restore(snapshot, programs=producer.programs)


def test_checkpoint_continues_the_same_stochastic_trace_in_a_fresh_process(tmp_path):
    producer = engine(temperament=neutral())
    for _ in range(11):
        producer.step_voice(0)
    path = tmp_path / 'voice.json'
    path.write_text(json.dumps(producer.voices[0].snapshot()), encoding='utf-8')
    expected = []
    for _ in range(20):
        frame = producer.step_voice(0)
        expected.append((frame.regime_id, producer.voices[0].rng_u32,
                         asdict(producer.voices[0].transition_memory)))
    code = '''import json, sys
from dataclasses import asdict
from tests.conformance.test_regime_transition_runtime import engine, neutral
from symcrash_python_ap.voice.voice_engine import VoiceStateWithProgram
producer = engine(temperament=neutral())
producer.voices[0] = VoiceStateWithProgram.restore(json.load(open(sys.argv[1])), programs=producer.programs)
result = []
for _ in range(20):
    frame = producer.step_voice(0)
    result.append((frame.regime_id, producer.voices[0].rng_u32, asdict(producer.voices[0].transition_memory)))
print(json.dumps(result))
'''
    child = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True, check=True)
    assert json.loads(child.stdout) == json.loads(json.dumps(expected))
