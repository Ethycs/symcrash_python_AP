"""Scheduling changes real execution, with independent state/audio assertions."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from symcrash_python_ap.frames import VoiceFlags, VoiceFramePacketV0b, VoiceFramePacketV1, CMMMEventPayload
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.scheduling import ExecutionPolicy, ExecutionReport
from symcrash_python_ap.vm.tables import COEFF_LEN, VM_K
from symcrash_python_ap.voice.scheduler import (
    VoiceSelector, BlockScheduler, ScheduleResult, VoiceScheduleInfo, HealthTelemetry,
)
from symcrash_python_ap.symbol.executor_scheduling import VoiceSelector as CompatSelector
from symcrash_python_ap.budgets.regime_budgets import RegimeBudget
from symcrash_python_ap.voice.orchestrator import BlockOrchestrator


def packet(voice=0, seq=1, **kw):
    return VoiceFramePacketV1(voice_id=voice, frame_seq=seq, flags=VoiceFlags.GATE,
                             gain_q15=32767, w_family_q15=[32767] * 8, **kw)


def prepared(ops=(), **kw):
    # OFF is a pure coefficient instrument: no oscillator injection obscures
    # the observable action of the scheduled operators.
    state = VMState.create(0)
    vm_step_block(state, packet(carrier_role_u8=0).to_voice_frame())
    state.coeff_state[:] = 1000
    frame = packet(seq=2, carrier_role_u8=0).to_voice_frame()
    frame.backbone_ops = list(ops)
    for name, value in kw.items():
        setattr(frame, name, value)
    return state, frame


def budget(ticks=8000, nodes=32, nonlinear=True):
    return RegimeBudget(2, 'test', ticks, nodes, nonlinear, 0)


def schedule(*ids, mid=True, top=True):
    return ScheduleResult([VoiceScheduleInfo(i, mid, top) for i in ids], 0, False,
                          HealthTelemetry(voices_serviced=len(ids)))


@pytest.mark.parametrize('selector_type', [VoiceSelector, CompatSelector])
@pytest.mark.parametrize('cap', [1, 3, 8])
def test_ring_is_fair_despite_counting_all_eligible_voices(selector_type, cap):
    frames = [packet(i) if i in (0, 2, 5, 7) else None for i in range(8)]
    selector = selector_type(v_max=8, cursor_fg=6)
    seen = []
    cursor = 6
    for _ in range(8):
        expected = []
        for _ in range(8):
            if len(expected) >= cap:
                break
            if cursor in (0, 2, 5, 7):
                expected.append(cursor)
            cursor = (cursor + 1) % 8
        selected, eligible = selector.select_voices(frames, cap)
        assert selected == expected
        assert selector.cursor_fg == cursor
        assert eligible == 4
        seen.extend(selected)
    assert set(seen) == {0, 2, 5, 7}


@pytest.mark.parametrize('selector_type', [VoiceSelector, CompatSelector])
def test_zero_or_negative_budget_cannot_select_voices(selector_type):
    selector = selector_type(v_max=4, cursor_fg=2)
    assert selector.compute_v_serviced(-1, 12) == 0
    assert selector.select_voices([packet(i) for i in range(4)], 0) == ([], 4)
    assert selector.cursor_fg == 2


@pytest.mark.parametrize('mid,top,expected', [
    (False, False, [1]), (True, False, [1, 4]),
    (False, True, [1, 15]), (True, True, [1, 4, 15]),
])
def test_due_flags_change_actual_coefficients_without_mutating_authored_frame(mid, top, expected):
    state, frame = prepared([1, 4, 15])
    control = copy.deepcopy(state)
    reference = copy.deepcopy(frame)
    reference.backbone_ops = expected
    report = ExecutionReport()
    vm_step_block(state, frame, policy=ExecutionPolicy(mid_due=mid, top_due=top), report=report)
    vm_step_block(control, reference)
    np.testing.assert_array_equal(state.coeff_state, control.coeff_state)
    assert frame.backbone_ops == [1, 4, 15]
    assert report.backbone_ops == tuple(expected)
    assert report.tail_ops == (2, 11, 23)


def test_budget_node_cap_reaches_dispatch_without_overwriting_persistent_mask():
    state, frame = prepared([15], tail_mode=2)
    report = ExecutionReport()
    vm_step_block(state, frame, policy=ExecutionPolicy(budget=budget(nodes=1)), report=report)
    assert report.node_mask == 1
    assert state.coeff_state[0] != 1000
    np.testing.assert_array_equal(state.coeff_state[8:], np.full(COEFF_LEN - 8, 1000))
    assert state.executor_latch.node_mask_u32 == 0xFFFFFFFF
    assert report.predicted_ticks == 576 + 16  # MIN tail + one-node Pressure.
    assert report.budget_fits


@pytest.mark.parametrize('ticks', [1, 8000])
def test_zero_node_mask_is_never_promoted_to_full_mask(ticks):
    state, frame = prepared([15], node_mask=0, tail_mode=2)
    report = ExecutionReport()
    vm_step_block(state, frame, policy=ExecutionPolicy(budget=budget(ticks)), report=report)
    assert report.node_mask == 0
    np.testing.assert_array_equal(state.coeff_state[:60 * VM_K], np.full(60 * VM_K, 1000))
    assert report.budget_fits == (ticks == 8000)


@pytest.mark.parametrize('due,budget_ticks', [(False, 8000), (True, 2400)])
def test_suppressed_or_budget_dropped_noise_never_advances_vm_rng(due, budget_ticks):
    state, frame = prepared()
    frame.cmmm_events = CMMMEventPayload(active_mask=2, noise_amount_q15=5000)
    before = state.prng_state
    report = ExecutionReport()
    vm_step_block(state, frame, policy=ExecutionPolicy(mid_due=due, budget=budget(budget_ticks)), report=report)
    assert state.prng_state == before
    assert report.event_ops == ()
    assert report.predicted_ticks == 2304
    assert report.budget_fits
    assert frame.cmmm_events.active_mask == 2


def test_impossible_budget_reports_real_safety_cost_and_still_enforces_tail():
    state, frame = prepared([4, 15])
    state.coeff_state[:] = 32767
    report = ExecutionReport()
    vm_step_block(state, frame, policy=ExecutionPolicy(budget=budget(1)), report=report)
    assert report.emergency and not report.budget_fits
    assert report.predicted_ticks == 1152
    assert report.tail_ops == (2, 19)
    assert not np.any(state.coeff_state[60 * VM_K:])


def test_wire_force_on_is_persistent_but_cannot_override_due_flags():
    state, frame = prepared()
    p = packet(seq=2, carrier_role_u8=0)
    p.event_mask = 2
    p.event_payload_u64[1] = (1 << 60) | (1 << 56) | (15 << 32)  # FORCE_ON Pressure
    frame = p.to_voice_frame()
    report = ExecutionReport()
    vm_step_block(state, frame, policy=ExecutionPolicy(top_due=False), report=report)
    assert report.backbone_ops == ()
    assert state.wire_state.forced_on == 1 << 15
    vm_step_block(state, packet(seq=3, carrier_role_u8=0).to_voice_frame(),
                  policy=ExecutionPolicy(), report=report)
    assert report.backbone_ops == (15,)
    assert state.coeff_state[0] != 1000


def test_budget_downgrade_cannot_be_undone_by_wire_tail_force_on():
    state, _ = prepared()
    p = packet(seq=2, carrier_role_u8=0)
    p.event_mask = 2
    p.event_payload_u64[1] = (1 << 60) | (1 << 56) | (11 << 32)
    report = ExecutionReport()
    vm_step_block(state, p.to_voice_frame(),
                  policy=ExecutionPolicy(budget=budget(nonlinear=False)), report=report)
    assert report.tail_ops == (2,)
    assert report.budget_fits
    assert state.wire_state.forced_on == 1 << 11


def test_held_render_has_audio_and_does_not_apply_new_commands_or_advance_state():
    state, frame = prepared([15])
    vm_step_block(state, frame)
    before = json.dumps(state.snapshot(), sort_keys=True)
    held = packet(seq=100, program_id=99).to_voice_frame()
    report = ExecutionReport()
    outputs = []
    for _ in range(4):
        _, out = vm_step_block(state, held, policy=ExecutionPolicy(serviced=False), report=report)
        outputs.append(out.copy())
        assert json.dumps(state.snapshot(), sort_keys=True) == before
    assert np.any(outputs[0])
    for out in outputs[1:]:
        np.testing.assert_array_equal(out, outputs[0])
    assert not report.serviced and report.predicted_ticks == 0


def test_rejected_frame_is_silent_even_when_requested_as_held_audio():
    state, _ = prepared()
    frame = packet(reject_code=1).to_voice_frame()
    before = json.dumps(state.snapshot(), sort_keys=True)
    _, out = vm_step_block(state, frame, policy=ExecutionPolicy(serviced=False))
    assert not np.any(out)
    assert json.dumps(state.snapshot(), sort_keys=True) == before


def test_legacy_adapter_services_ring_and_holds_every_other_vm():
    engine = BlockOrchestrator(n_voices=3)
    for vid in range(3):
        engine.update_frame(vid, VoiceFramePacketV0b(
            voice_id=vid, flags=VoiceFlags.GATE, gain_q15=20000))
    engine.step_block()  # Seed all three voices.
    engine.scheduler.b_compute = engine.scheduler.b_mem = 12
    for expected in (0, 1, 2, 0, 1, 2):
        before = {vid: json.dumps(state.snapshot(), sort_keys=True)
                  for vid, state in engine.vm_states.items()}
        left, right, result = engine.step_block()
        assert result.serviced_ids == [expected]
        assert any(left) and any(right)
        for vid, state in engine.vm_states.items():
            assert (json.dumps(state.snapshot(), sort_keys=True) == before[vid]) == (vid != expected)
            assert engine.execution_reports[vid].serviced == (vid == expected)
    assert [s.blocks_since_gate for s in engine.voice_slots] == [3, 3, 3]


def test_free_packet_voice_cannot_be_selected_from_a_stale_frame():
    engine = BlockOrchestrator(n_voices=1)
    engine.update_frame(0, packet())
    engine.step_block()
    engine.voice_free(0)
    left, right, result = engine.step_block()
    assert result.serviced_ids == []
    assert not any(left + right)
    assert not engine.vm_states


def test_native_host_holds_audio_voice_rng_and_arc_clocks():
    from symcrash_python_ap.synth_master import SynthMaster
    synth = SynthMaster()
    vid = synth.note_on(60, velocity=100)
    synth.render_block(schedule=schedule(vid))
    voice = synth.voice_engine.voices[vid]
    vm = synth._voice_slots[vid].vm_state
    before_voice = voice.snapshot()
    before_vm = json.dumps(vm.snapshot(), sort_keys=True)
    for _ in range(4):
        left, right, telemetry = synth.render_block(schedule=schedule())
        assert any(left) and any(right)
        assert voice.snapshot() == before_voice
        assert json.dumps(vm.snapshot(), sort_keys=True) == before_vm
        assert synth._voice_slots[vid].active
        assert telemetry.voices_serviced == 0
    synth.render_block(schedule=schedule(vid))
    assert voice.block_count == before_voice['block_count'] + 1


def test_native_top_suppression_happens_before_authored_random_draws():
    from symcrash_python_ap.voice.voice_engine import VoiceEngine, InstrumentProgram
    from symcrash_python_ap.voice.event_policy import EventPolicySpec, LaneRuleSpec, LaneRuleType
    rules = [LaneRuleSpec() for _ in range(8)]
    rules[4] = LaneRuleSpec(rule_type=LaneRuleType.STOCHASTIC, rate_q15=32767)
    engine = VoiceEngine()
    engine.programs[99] = InstrumentProgram(event_policy=EventPolicySpec(lane_rules=rules))
    engine.allocate_voice(0, program_id=99)
    engine.note_on(0)
    before = engine.voices[0].rng_u32
    result = engine.step_block(schedule=schedule(0, top=False))
    assert not result.render_batch[0].top_due
    assert engine.voices[0].rng_u32 == before
    assert not result.render_batch[0].frame.cmmm_events.active_mask & 16


def test_unserviced_first_note_waits_without_losing_its_start_event():
    from symcrash_python_ap.voice.voice_engine import VoiceEngine
    engine = VoiceEngine()
    engine.note_on(0)
    before = engine.voices[0].snapshot()
    result = engine.step_block(schedule=schedule())
    assert not result.render_batch
    assert engine.voices[0].snapshot() == before
    result = engine.step_block(schedule=schedule(0))
    assert result.render_batch[0].frame.is_starting


def test_invalid_native_schedule_fails_before_stepping_any_voice():
    from symcrash_python_ap.voice.voice_engine import VoiceEngine
    engine = VoiceEngine()
    engine.note_on(0)
    before = engine.voices[0].snapshot()
    with pytest.raises(ValueError, match='unique'):
        engine.step_block(schedule=schedule(0, 0))
    assert engine.voices[0].snapshot() == before


@pytest.mark.parametrize('jit', [False, True])
def test_scheduled_trace_matches_reference(jit):
    from symcrash_python_ap.vm import hotpath_jit
    previous = hotpath_jit.jit_enabled()
    try:
        hotpath_jit.set_jit_enabled(jit)
        state, frame = prepared([1, 15], tail_mode=2)
        frame.node_mask = 1
        for seq in range(2, 10):
            # Hold alternates with a one-node pressure pass; Tilt is disabled
            # by authored selection rather than by the scheduling policy.
            frame.backbone_ops = [15]
            frame.wire_data = replace(frame.wire_data, frame_seq=seq)
            vm_step_block(state, frame, policy=ExecutionPolicy(serviced=seq % 2 == 0))
        # Four applications of 1 + round((-983 * 32767)/32768)/32768.
        expected = 1000
        from symcrash_python_ap.fixed import RSHIFT_RNE
        for _ in range(4):
            expected += RSHIFT_RNE(expected * RSHIFT_RNE(-983 * 32767, 15), 15)
        assert state.coeff_state[0] == expected
        np.testing.assert_array_equal(state.coeff_state[8:], np.full(COEFF_LEN - 8, 1000))
    finally:
        hotpath_jit.set_jit_enabled(previous)


def test_report_only_observes_execution_and_does_not_change_audio():
    state, frame = prepared([12, 1, 15])
    other = copy.deepcopy(state)
    _, expected = vm_step_block(other, frame)
    expected = expected.copy()
    report = ExecutionReport()
    _, actual = vm_step_block(state, frame, report=report)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(state.coeff_state, other.coeff_state)
    assert report.serviced and report.backbone_ops == (1, 15)
    assert report.tail_ops == (2, 11, 12, 23)


def test_glue_runs_once_between_normalize_and_sheaf(monkeypatch):
    from symcrash_python_ap.vm import coefficient_tail, lanes
    from symcrash_python_ap.vm.decode import DecodedPlan, BackboneEntry
    state = VMState.create()
    state.coeff_state[:] = np.arange(COEFF_LEN, dtype=np.int16) * 4
    cfg = DecodedPlan(wire_tail_ops=[BackboneEntry(12), BackboneEntry(12)])
    order = []
    for name in ('hard_project', 'normalize_scale', 'sheaf_project'):
        original = getattr(coefficient_tail, name)
        def tracked(*args, _name=name, _original=original):
            order.append(_name)
            return _original(*args)
        monkeypatch.setattr(coefficient_tail, name, tracked)
    dispatch = lanes.dispatch_coeff_op
    def tracked_dispatch(*args):
        order.append('glue')
        return dispatch(*args)
    monkeypatch.setattr(lanes, 'dispatch_coeff_op', tracked_dispatch)
    coefficient_tail.apply_coefficient_tail(state, cfg)
    assert order == ['hard_project', 'normalize_scale', 'glue', 'sheaf_project', 'hard_project']


def test_native_host_applies_explicit_voice_budget_and_reports_impossible_fit():
    from symcrash_python_ap.synth_master import SynthMaster
    synth = SynthMaster()
    vid = synth.note_on(60, velocity=100)
    synth.render_block(voice_budget=budget(1))
    report = synth.last_execution_reports[vid]
    assert report.serviced and report.emergency and not report.budget_fits
    assert report.predicted_ticks == 1152
    assert not np.any(synth._voice_slots[vid].vm_state.coeff_state[60 * VM_K:])


def test_packet_adapter_isolates_ingestion_and_validates_slot_before_mutation():
    engine = BlockOrchestrator(n_voices=2)
    source = packet()
    engine.update_frame(0, source)
    source.w_family_q15[0] = 1
    source.flags = 0
    assert engine.voice_frames[0].w_family_q15[0] == 32767
    assert engine.voice_frames[0].flags == VoiceFlags.GATE
    with pytest.raises(ValueError, match='slot'):
        engine.update_frame(0, packet(voice=1))
    assert engine.voice_frames[0].voice_id == 0


def test_rejected_packet_cannot_erase_sequence_history_or_change_lifecycle():
    from symcrash_python_ap.voice.orchestrator import VoiceState
    engine = BlockOrchestrator(n_voices=1)
    engine.update_frame(0, packet(seq=10))
    engine.step_block()
    previous = json.dumps(engine.vm_states[0].snapshot(), sort_keys=True)
    rejected = packet(seq=20, reject_code=1)
    rejected.flags = VoiceFlags.GATE_FALL
    engine.update_frame(0, rejected)
    left, right, result = engine.step_block()
    assert not any(left + right) and result.serviced_ids == []
    assert engine.voice_slots[0].state == VoiceState.ACTIVE
    assert json.dumps(engine.vm_states[0].snapshot(), sort_keys=True) == previous
    stale = packet(seq=9)
    stale.flags = VoiceFlags.GATE_FALL
    engine.update_frame(0, stale)
    assert engine.voice_frames[0].reject_code == 1
    engine.update_frame(0, packet(seq=11))
    assert engine.step_block()[2].serviced_ids == [0]


def test_program_reassignment_resets_vm_but_retains_sample_for_allocation_fade():
    engine = BlockOrchestrator(n_voices=1)
    engine.update_frame(0, packet(seq=10))
    engine.step_block()
    old = engine.vm_states[0]
    regularity = engine.wavelet.get_regularity(0)
    last = regularity.prev_sample
    engine.update_frame(0, packet(seq=1, program_id=5))
    assert engine.vm_states[0] is not old
    assert regularity.prev_sample == last
    assert regularity.is_fading()
    engine.step_block()
    assert engine.vm_states[0].wire_state.program_id == 5


def test_external_reallocation_clears_stale_packet_and_sequence():
    engine = BlockOrchestrator(n_voices=1)
    engine.update_frame(0, packet(seq=100))
    engine.step_block()
    engine.voice_allocate(0, owner_id=10)
    assert engine.step_block()[2].serviced_ids == []
    engine.update_frame(0, packet(seq=1))
    assert engine.step_block()[2].serviced_ids == [0]


def test_rejected_zero_activity_cannot_free_a_releasing_voice():
    from symcrash_python_ap.voice.orchestrator import VoiceState
    engine = BlockOrchestrator(n_voices=1)
    engine.update_frame(0, packet(seq=1))
    engine.step_block()
    release = packet(seq=2, activity_q15=5000)
    release.flags = VoiceFlags.GATE_FALL
    engine.update_frame(0, release)
    before = json.dumps(engine.vm_states[0].snapshot(), sort_keys=True)
    engine.update_frame(0, packet(seq=3, reject_code=1))
    engine.step_block()
    assert engine.voice_slots[0].state == VoiceState.RELEASING
    assert json.dumps(engine.vm_states[0].snapshot(), sort_keys=True) == before


def reference_stress_masked_dispatch(original):
    """Independent scalar oracle for the stress trace's K=16 region.

    Its only backbone operations are Tilt, IntraMix and RotateNodes. The
    remaining K=32 regions use the unchanged, separately tested operators.
    """
    def dispatch(state, entry, mask=None):
        if mask != 0xFFFF:
            return original(state, entry, mask)
        values = [int(v) for v in state.coeff_state]
        result = values.copy()
        amount = entry.strength_q15
        if entry.op_id == 1:
            for index in range(128):
                node = index // 8
                gain = 32767 + ((16384 + node * 32767 // 32 - 32767) * amount) // 32767
                result[index] = max(-32768, min(32767, values[index] * gain // 32768))
        elif entry.op_id == 3:
            for index in range(128):
                if index % 8 not in (0, 7):
                    average = (values[index - 1] + 2 * values[index] + values[index + 1]) // 4
                    result[index] += (average - values[index]) * amount // 32768
        elif entry.op_id == 10:
            if amount >= 16384:
                result[:128] = values[120:128] + values[:120]
        else:
            raise AssertionError(f'Unexpected masked stress operator: {entry.op_id}')
        state.coeff_state[:] = result
    return dispatch


@pytest.mark.parametrize('jit', [False, True])
def test_stress_pattern_scope_matches_independent_scalar_reference(monkeypatch, jit):
    from symcrash_python_ap.vm import executor, hotpath_jit
    from tests.artifacts.golden_trace_stress_v0 import compute_trace_hashes
    previous = hotpath_jit.jit_enabled()
    try:
        hotpath_jit.set_jit_enabled(jit)
        actual = compute_trace_hashes()
        monkeypatch.setattr(executor, 'dispatch_coeff_op_crossfade',
                            reference_stress_masked_dispatch(executor.dispatch_coeff_op_crossfade))
        assert compute_trace_hashes() == actual
    finally:
        hotpath_jit.set_jit_enabled(previous)
