"""Authored CMMM rules, block clocks, independent RNG and snapshot replay."""
from copy import deepcopy
from fractions import Fraction
import json

import pytest

from symcrash_python_ap.voice import (
    ArcSpec, EdgeSpec, EventPatternSpec, EventPolicySpec, GestureBasis,
    InstrumentProgram, IntentVector, LaneRuleSpec, LaneRuleType as Rule,
    Regime, RegimeGraph, RegimeSpec, VoiceEngine, VoiceStateWithProgram,
)
from symcrash_python_ap.voice.voice_engine import INSTRUMENT_PROGRAMS


def policy(**lanes):
    result = EventPolicySpec()
    for lane, rule in lanes.items():
        result.lane_rules[int(lane.removeprefix('e'))] = rule
    return result


def engine(authored, *, duration=12, patterns=None, probabilities=None, voices=1):
    arc = ArcSpec(arc_id=91, duration_blocks=duration, event_patterns=patterns or {})
    graph = RegimeGraph(
        regime_specs={Regime.R1_ATTACK: RegimeSpec(
            regime=Regime.R1_ATTACK, event_enable_mask=255,
            event_probability=probabilities or [32767] * 8)},
        arc_specs={91: arc},
        edges=[EdgeSpec(from_regime=Regime.R0_IDLE, to_regime=Regime.R1_ATTACK, arc_id=91)],
    )
    program = InstrumentProgram(program_id=60001, event_policy=authored,
                                regime_graph=graph, attack_blocks=1000)
    e = VoiceEngine(max_voices=voices)
    e.programs[60001] = program
    for vid in range(voices):
        e.allocate_voice(vid, program_id=60001)
    return e


def xorshift(seed):
    seed ^= (seed << 13) & 0xFFFFFFFF
    seed ^= seed >> 17
    seed ^= (seed << 5) & 0xFFFFFFFF
    return seed


def qmul(a, b):
    return max(-32768, min(32767, round(Fraction(a * b, 32768))))


def test_explicit_empty_policy_disables_defaults_and_arc_events():
    default, silent = VoiceEngine(), VoiceEngine()
    for e in (default, silent):
        e.allocate_voice()
        e.note_on(0)
    silent.voices[0].program = deepcopy(silent.voices[0].program)
    silent.voices[0].program.event_policy = EventPolicySpec()
    assert default.step_voice(0).event_mask
    assert [silent.step_voice(0).event_mask for _ in range(6)] == [0] * 6


def test_gate_edges_fire_once_and_retrigger_records_both_edges_in_one_block():
    e = engine(policy(e0=LaneRuleSpec(Rule.GATE_EDGE),
                      e1=LaneRuleSpec(Rule.GATE_EDGE, on_rise=False, on_fall=True)))
    e.note_on(0)
    assert [e.step_voice(0).event_mask for _ in range(3)] == [1, 0, 0]
    e.note_off(0)
    e.note_on(0)
    assert e.step_voice(0).event_mask == 3
    assert e.step_voice(0).event_mask == 0
    e.note_off(0)
    assert e.step_voice(0).event_mask == 2
    e.note_off(0)  # Duplicate release is not another falling edge.
    assert e.step_voice(0).event_mask == 0


def test_held_gate_does_not_create_edges_when_policy_is_enabled_later():
    e = engine(None)
    e.note_on(0)
    e.step_voice(0)
    e.voices[0].program.event_policy = policy(e0=LaneRuleSpec(Rule.GATE_EDGE))
    assert e.step_voice(0).event_mask == 0
    e.note_on(0)
    assert e.step_voice(0).event_mask == 1


def test_per_voice_gate_history_and_input_gestures_are_independent():
    e = engine(policy(e0=LaneRuleSpec(Rule.GATE_EDGE, on_fall=True)), voices=2)
    gesture = GestureBasis(gate=True, pressure_q15=10000)
    e.note_on(0, gesture)
    e.note_on(1, gesture)
    assert [e.step_voice(v).event_mask for v in (0, 1)] == [1, 1]
    e.note_off(0)
    assert gesture.gate and e.voices[1].gesture.gate
    assert [e.step_voice(v).event_mask for v in (0, 1)] == [1, 0]
    update = GestureBasis(pressure_q15=20000)
    e.update_gesture(1, update)
    assert not update.gate and e.voices[1].gesture.gate


@pytest.mark.parametrize('value,fire', [(9999, False), (10000, True), (20000, True)])
def test_threshold_boundary_and_amplitude_are_exact(value, fire):
    rule = LaneRuleSpec(Rule.THRESHOLD, intent_lane=2, threshold_q15=10000, strength_mod_q15=16384)
    intent = IntentVector(lanes=[0, 0, value, 0, 0, 0, 0, 0])
    assert rule.check(intent, GestureBasis(), 42) == (fire, qmul(value, 16384) if fire else 0, 42)


@pytest.mark.parametrize('loop', [False, True])
def test_arc_pattern_uses_zero_based_blocks_rests_and_exact_duration(loop):
    e = engine(policy(e0=LaneRuleSpec(Rule.PATTERN, pattern_id=7)), duration=11,
               patterns={7: EventPatternSpec([10000, 0, 20000], step_blocks=2, loop=loop)})
    e.note_on(0)
    frames = [e.step_voice(0) for _ in range(14)]
    expected_blocks = {0: qmul(10000, 32767), 4: qmul(20000, 32767)}
    if loop:
        expected_blocks.update({6: qmul(10000, 32767), 10: qmul(20000, 32767)})
    assert {i: f.event_strengths[0] for i, f in enumerate(frames) if f.event_mask} == expected_blocks
    assert [f.arc_active for f in frames] == [True] * 11 + [False] * 3
    assert frames[0].arc_progress_q15 == 0
    assert frames[10].arc_progress_q15 == (10 * 32767) // 11


def test_one_block_arc_fires_its_first_and_only_pattern_step():
    e = engine(policy(e0=LaneRuleSpec(Rule.PATTERN)), duration=1,
               patterns={0: EventPatternSpec([10000])})
    e.note_on(0)
    a, b = e.step_voice(0), e.step_voice(0)
    assert a.arc_active and a.event_mask == 1 and a.arc_progress_q15 == 0
    assert not b.arc_active and b.event_mask == 0


def test_retrigger_restarts_pattern_without_resetting_other_voice():
    e = engine(policy(e0=LaneRuleSpec(Rule.PATTERN)), voices=2,
               patterns={0: EventPatternSpec([10000, 0, 0, 0])})
    for v in (0, 1):
        e.note_on(v)
        assert e.step_voice(v).event_mask == 1
        assert e.step_voice(v).event_mask == 0
    e.note_on(0)
    assert e.step_voice(0).event_mask == 1
    assert e.step_voice(1).event_mask == 0


@pytest.mark.parametrize('rate,scale', [(0, 32767), (32767, 0), (32767, 32767), (16384, 32767), (16384, 16384)])
def test_stochastic_probability_and_draw_order_match_independent_stream(rate, scale):
    p = policy(e1=LaneRuleSpec(Rule.STOCHASTIC, rate_q15=rate, strength_mod_q15=20000))
    e = engine(p, probabilities=[scale] * 8)
    e.note_on(0)
    expected_rng = e.voices[0].rng_u32
    probability = (rate * scale + 16383) // 32767
    for _ in range(60):
        fire = probability > 0
        if 0 < probability < 32767:
            expected_rng = xorshift(expected_rng)
            fire = (expected_rng & 0x7FFF) < probability
        expected_strength = 0
        if fire:
            expected_rng = xorshift(expected_rng)
            expected_strength = qmul(expected_rng & 0x7FFF, 20000)
        frame = e.step_voice(0)
        assert frame.event_mask == (2 if fire else 0)
        assert frame.event_strengths[1] == expected_strength
        assert e.voices[0].rng_u32 == expected_rng


def test_masked_lanes_do_not_draw_and_program_regime_overrides_are_used():
    p = policy(e0=LaneRuleSpec(Rule.GATE_EDGE), e1=LaneRuleSpec(Rule.STOCHASTIC, rate_q15=32767))
    e = engine(p)
    e.voices[0].program.regime_specs[Regime.R1_ATTACK] = RegimeSpec(regime=Regime.R1_ATTACK, event_enable_mask=1)
    seed = e.voices[0].rng_u32
    e.note_on(0)
    assert e.step_voice(0).event_mask == 1
    assert e.voices[0].rng_u32 == seed
    p.enable_mask = 0
    e.note_on(0)
    assert e.step_voice(0).event_mask == 0 and e.voices[0].rng_u32 == seed


def test_authored_band_and_bitmap_selectors_reach_render_frame():
    e = engine(policy(
        e0=LaneRuleSpec(Rule.GATE_EDGE, target_band=0),
        e1=LaneRuleSpec(Rule.GATE_EDGE, target_band=4),
        e3=LaneRuleSpec(Rule.GATE_EDGE, band_mask=0),
        e5=LaneRuleSpec(Rule.GATE_EDGE, band_mask=0x15),
    ))
    e.note_on(0)
    frame = e.step_block().render_batch[0].frame
    assert frame.cmmm_events.active_mask == 0x2B
    assert frame.cmmm_events.impulse_band == 0
    assert frame.cmmm_events.noise_band == 4
    assert frame.cmmm_events.gate_mask == 0
    assert frame.cmmm_events.swallowtail_bands == 0x15


def test_authored_patterns_change_rendered_audio_deterministically():
    from symcrash_python_ap.vm.executor import vm_step_block
    from symcrash_python_ap.vm.state import VMState

    e = engine(policy(e0=LaneRuleSpec(Rule.PATTERN, target_band=2)), duration=12,
               patterns={0: EventPatternSpec([20000, 0], step_blocks=2)})
    e.note_on(0)
    frames = [e.step_block().render_batch[0].frame for _ in range(16)]
    assert [i for i, f in enumerate(frames) if f.cmmm_events.active_mask] == [0, 4, 8]
    def render(enabled):
        state, result = VMState.create(), []
        for frame in deepcopy(frames):
            if not enabled:
                frame.cmmm_events = None
            state, samples = vm_step_block(state, frame)
            result.extend(samples.tolist())
        return result
    assert render(True) == render(True)
    assert render(True) != render(False)


def test_direct_note_on_supplies_gate_without_mutating_the_input_sample():
    e = engine(policy(e0=LaneRuleSpec(Rule.GATE_EDGE)))
    gesture = GestureBasis(pressure_q15=10000)
    e.note_on(0, gesture)
    assert not gesture.gate and e.voices[0].gesture.gate
    assert e.step_voice(0).event_mask == 1


@pytest.mark.parametrize('checkpoint', ['pending_rise', 'pattern_midpoint', 'pending_fall'])
def test_snapshot_replay_preserves_edges_patterns_rng_and_authored_program(checkpoint):
    e = engine(policy(e0=LaneRuleSpec(Rule.GATE_EDGE, on_fall=True),
                      e1=LaneRuleSpec(Rule.STOCHASTIC, rate_q15=12000),
                      e4=LaneRuleSpec(Rule.PATTERN)), duration=20,
               patterns={0: EventPatternSpec([10000, 0, 20000], 2)})
    e.note_on(0)
    if checkpoint != 'pending_rise':
        for _ in range(5):
            e.step_voice(0)
    if checkpoint == 'pending_fall':
        e.note_off(0)
    snap = json.loads(json.dumps(e.voices[0].snapshot()))
    restored = VoiceEngine()
    restored.voices[0] = VoiceStateWithProgram.restore(snap, programs=e.programs)
    for _ in range(12):
        assert e.step_voice(0) == restored.step_voice(0)
        assert e.voices[0].rng_u32 == restored.voices[0].rng_u32


def test_old_snapshot_does_not_invent_a_rise_and_missing_program_is_explicit():
    e = engine(policy(e0=LaneRuleSpec(Rule.GATE_EDGE)))
    e.note_on(0)
    e.step_voice(0)
    snap = e.voices[0]._snapshot_legacy()
    for key in ('previous_gate', 'gate_rise_pending', 'gate_fall_pending'):
        snap.pop(key)
    with pytest.raises(ValueError, match='Program 60001'):
        VoiceStateWithProgram.restore(snap)
    restored = VoiceEngine()
    restored.voices[0] = VoiceStateWithProgram.restore(snap, programs=e.programs)
    assert restored.step_voice(0).event_mask == 0


def test_builtin_program_is_resolved_on_restore_and_registries_are_local():
    a, b = VoiceEngine(), VoiceEngine()
    a.allocate_voice(program_id=1001)
    snapshot = a.voices[0].snapshot()
    assert VoiceStateWithProgram.restore(snapshot).program is INSTRUMENT_PROGRAMS[1001]
    a.programs[55555] = InstrumentProgram()
    assert 55555 not in b.programs and 55555 not in INSTRUMENT_PROGRAMS


def test_program_graph_can_use_an_authored_transition_absent_from_default_graph():
    e = engine(policy(e0=LaneRuleSpec(Rule.GATE_EDGE)))
    program = e.voices[0].program
    program.regime_graph.regime_specs[Regime.R6_RECOVERY] = RegimeSpec(
        regime=Regime.R6_RECOVERY, event_enable_mask=1)
    program.regime_graph.edges.append(EdgeSpec(
        from_regime=Regime.R1_ATTACK, to_regime=Regime.R6_RECOVERY, min_dwell_blocks=0))
    e.note_on(0)
    frame = e.step_voice(0)
    assert frame.regime_id == Regime.R6_RECOVERY and frame.event_mask == 1


@pytest.mark.parametrize('kwargs', [dict(rule_type=99), dict(intent_lane=8), dict(threshold_q15=-1),
    dict(rate_q15=32768), dict(strength_mod_q15=40000), dict(pattern_id=-1), dict(target_band=6), dict(band_mask=256)])
def test_invalid_rule_fields_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LaneRuleSpec(**kwargs)


@pytest.mark.parametrize('kwargs', [dict(steps_q15=[]), dict(steps_q15=[-1]),
                                   dict(steps_q15=[32768]), dict(step_blocks=0)])
def test_invalid_patterns_are_rejected(kwargs):
    with pytest.raises(ValueError):
        EventPatternSpec(**kwargs)


def test_invalid_policy_and_missing_pattern_are_rejected_before_allocation():
    with pytest.raises(ValueError):
        EventPolicySpec(lane_rules=[])
    with pytest.raises(ValueError):
        EventPolicySpec(enable_mask=256)
    with pytest.raises(ValueError, match='Undefined event patterns'):
        engine(policy(e0=LaneRuleSpec(Rule.PATTERN, pattern_id=7)))


def test_get_regime_spec_creates_a_valid_fallback():
    assert InstrumentProgram().get_regime_spec(Regime.R6_RECOVERY).regime == Regime.R6_RECOVERY
