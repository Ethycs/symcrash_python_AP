"""Frozen packet conversion and versioned VM consumers, independent fixtures."""
import json

import numpy as np
import pytest

from symcrash_python_ap.frames import (
    VoiceFrame, VoiceFramePacketV0b as V0, VoiceFramePacketV1 as V1, VoiceFlags,
)
from symcrash_python_ap.wire_frame import decode_wire_events
from symcrash_python_ap.vm import VMState, vm_step_block
from symcrash_python_ap.vm.decode import decode_frame, CarrierRole
from symcrash_python_ap.vm.wire_events import begin_wire_block, apply_wire_controls


# Literal v0b fixture: note 72/-50 cents, independent bus values, two events.
FIXTURE = bytes.fromhex(
    '56460188 1234 0007 12345678 0000 0000 '
    '0003 03 01 48 00 ffce '
    '6000 5000 c000 2000 1000 '
    '0001 0002 0003 0004 0005 0006 0007 0008 '
    '01 03 4000 11 00 '
    '0064 0000 0000 0000 00c8 0000 0000 0000 '
    '0200050007000000 0000000000000000 0000000000000000 0000000000000000 '
    '4100030300070000 0000000000000000 0000000000000000 0000000000000000')


def packet(cls=V0, *, seq=1, events=(), **fields):
    p = cls(flags=VoiceFlags.GATE, frame_seq=seq, **fields)
    for lane, variant, strength, target_kind, target_id, pattern, p0, p1, p2 in events:
        p.event_mask |= 1 << lane
        p.event_strength_q15[lane] = strength
        # Independent bit assembly, without calling production payload packer.
        p.event_payload_u64[lane] = (lane << 60 | variant << 56 | target_kind << 40 |
            target_id << 32 | pattern << 24 | p0 << 16 | p1 << 8 | p2)
    return p


def event(lane, variant=0, strength=1000, *, target=0, node=0, pattern=0, p0=0, p1=0, p2=0):
    return lane, variant, strength, target, node, pattern, p0, p1, p2


def evolve(p, state=None, ops=()):
    frame = p.to_voice_frame()
    frame.base_mix_q15 = 0  # Isolate events and persistent coefficient evolution.
    frame.ops_enable = list(ops)
    state = state or VMState()
    return vm_step_block(state, frame)


def test_literal_v0_fixture_survives_every_conversion_without_aliasing():
    p = V0.from_bytes(FIXTURE)
    assert p.validate() == (True, None)
    assert (p.program_id, p.voice_id, p.frame_seq, p.note_u8, p.cents_i16) == (0x1234, 7, 0x12345678, 72, -50)
    frame = p.to_voice_frame()
    assert frame.note == 72 and frame.cents == -50 and frame.is_starting
    assert frame.wire_data.sendA_q15 == 8192 and frame.wire_data.sendB_q15 == 4096
    assert frame.wire_data.family_weights_q15 == tuple(range(1, 9))
    assert V0.from_voice_frame(frame).to_bytes() == FIXTURE
    p.event_payload_u64[0] = 0
    p.w_family_q15[0] = 30000
    assert V0.from_voice_frame(frame).to_bytes() == FIXTURE
    restored = V0.from_voice_frame(frame, program_id=0, frame_seq=0, note_u8=0, cents_i16=0)
    assert (restored.program_id, restored.frame_seq, restored.note_u8, restored.cents_i16) == (0, 0, 0, 0)


def test_literal_v1_header_and_roundtrip_preserve_carrier_buses():
    # Header inserted at offset 54. Remove v0 lane-4 injection from this fixture.
    base = bytearray(FIXTURE)
    base[2:4] = bytes([2, 148])
    base[54] = 1
    base[64:66] = bytes(2)
    base[104:112] = bytes(8)
    raw = bytes(base[:54]) + bytes.fromhex('02 02 04 01 fff6 2000 6000 0014') + bytes(base[54:])
    p = V1.from_bytes(raw)
    assert p.validate() == (True, None)
    frame = p.to_voice_frame()
    assert V1.from_voice_frame(frame).to_bytes() == raw
    cfg = decode_frame(frame)
    assert cfg.coeff_osc_params.cents == -40  # -50 -10 +20, exactly once.
    assert cfg.coeff_osc_params.pwm_enabled
    assert cfg.coeff_osc_params.pwm_width_q15 == 8192
    assert cfg.coeff_osc_params.legato_mode
    assert cfg.carrier_role == CarrierRole.EXCITER_ONLY  # Wire enum 2, VM enum 1.


@pytest.mark.parametrize('cls', [V0, V1])
@pytest.mark.parametrize('offset,value', [(0, 0), (2, 99), (3, 0), (14, 1), (21, 1), (20, 255), (50, 7)])
def test_corrupt_core_rejected_before_rendering(cls, offset, value):
    data = bytearray(cls().to_bytes())
    data[offset] = value
    with pytest.raises(ValueError):
        cls.from_bytes(bytes(data))


@pytest.mark.parametrize('cls', [V0, V1])
@pytest.mark.parametrize('extra', [-1, 1, 136])
def test_packet_length_must_be_exact(cls, extra):
    data = cls().to_bytes()
    bad = data[:extra] if extra < 0 else data + bytes(extra)
    with pytest.raises(ValueError, match='exactly'):
        cls.from_bytes(bad)


@pytest.mark.parametrize('field,value', [
    ('gain_q15', -1), ('gain_q15', 32768), ('note_u8', -1), ('pan_q15', 32768),
    ('w_family_q15', [1] * 7), ('event_strength_q15', [0] * 9),
    ('event_payload_u64', [-1] * 8), ('frame_seq', 2**32), ('tree_blend_q15', 65535),
])
def test_invalid_direct_packet_is_not_silently_coerced(field, value):
    p = V0(**{field: value})
    assert not p.validate()[0]
    with pytest.raises(ValueError):
        p.to_voice_frame()


def test_bad_event_is_ignored_individually_and_inactive_bus_is_not_executed():
    p = packet(events=[event(0), event(2, 4)])
    p.event_payload_u64[0] |= 1 << 60  # Wrong lane nibble.
    p.event_payload_u64[6] = 6 << 60  # Inactive slots must validate strictly.
    assert not p.validate()[0]
    frame = V0.from_bytes(p.to_bytes()).to_voice_frame()
    assert [e.lane for e in decode_wire_events(frame.wire_data)] == [2]
    state = VMState()
    begin_wire_block(state, frame)
    cfg = decode_frame(frame)
    apply_wire_controls(state, cfg)
    assert cfg.tree_blend_q15 == 32767
    assert cfg.event_instrs == []  # No legacy lane remapping.


@pytest.mark.parametrize('bad', [event(7), event(0, 15), event(1, node=24),
    event(1, p0=1), event(2, p2=1), event(0, target=4), event(4, target=3, node=32)])
def test_reserved_events_are_noops(bad):
    frame = packet(events=[bad]).to_voice_frame()
    assert decode_wire_events(frame.wire_data) == ()


def test_reserved_payload_strength_is_not_an_alternate_amplitude():
    p = packet(events=[event(0)])
    p.event_payload_u64[0] |= 255 << 48
    assert not p.validate()[0]
    assert not decode_wire_events(p.to_voice_frame().wire_data)


def test_v0_note_injection_cannot_silently_become_v1_carrier_command():
    p = packet(events=[event(4)])
    with pytest.raises(ValueError, match='lane 4'):
        V1.from_v0b(p)
    migrated = V1.from_v0b(p, drop_note_injection=True)
    assert migrated.event_mask == 0
    assert migrated.event_payload_u64[4] == migrated.event_strength_q15[4] == 0
    assert p.event_mask == 16
    with pytest.raises(ValueError, match='Cross-version'):
        V1.from_voice_frame(p.to_voice_frame())


def test_injection_then_backbone_then_pulse_order_is_audible():
    events = [event(4, target=3, node=3), event(0, target=3, node=3)]
    state, audio = evolve(packet(events=events), ops=['PressureRamp'])
    # PressureRamp is +10% on node 3. Pre-backbone 1000 becomes 1100;
    # post-backbone pulse adds 1000. Reversing either stage gives 2000/2200.
    assert state.coeff_state[24] == 2100
    assert np.count_nonzero(state.coeff_state) == 1
    assert np.any(audio)


def test_toggle_applies_before_backbone_persists_and_resets_on_retrigger():
    state, _ = evolve(packet(events=[event(4, target=3, node=3), event(1, 2, node=15)]), ops=['PressureRamp'])
    assert state.coeff_state[24] == 1000
    state, _ = evolve(packet(seq=2), state, ['PressureRamp'])
    assert state.coeff_state[24] == 1000
    state, _ = evolve(packet(seq=3, events=[event(1, 1, node=15)]), state, ['PressureRamp'])
    assert state.coeff_state[24] == 1100
    p = packet(seq=4, events=[event(4, target=3, node=3)])
    p.flags |= VoiceFlags.RETRIGGER
    state, _ = evolve(p, state, ['PressureRamp'])
    assert state.coeff_state[24] == 1100  # Fresh 1000, not 2100 from old state.
    assert state.wire_state.forced_off == state.wire_state.forced_on == 0


def test_duplicate_events_do_not_retrigger_and_stale_packet_does_not_mutate_state():
    p = packet(events=[event(4, target=3, node=3)])
    state, _ = evolve(p)
    state, _ = evolve(p, state)
    assert state.coeff_state[24] == 1000
    before = json.dumps(state.snapshot(), sort_keys=True)
    _, audio = evolve(packet(seq=0), state)
    assert not np.any(audio)
    assert json.dumps(state.snapshot(), sort_keys=True) == before


def test_packet_sequence_wrap_program_reset_and_snapshot_replay():
    p = packet(seq=0xFFFFFFFF, events=[event(1, 1, node=15), event(4, target=3, node=3)])
    state, _ = evolve(p)
    restored = VMState.restore(json.loads(json.dumps(state.snapshot())))
    next_packet = packet(seq=0)
    state, left = evolve(next_packet, state)
    restored, right = evolve(next_packet, restored)
    np.testing.assert_array_equal(left, right)
    assert state.wire_state.forced_on == 1 << 15
    state, _ = evolve(packet(seq=0, program_id=123), state)
    assert state.wire_state.forced_on == 0


def test_rejected_packet_returns_silence_without_advancing_any_state():
    state = VMState()
    before = json.dumps(state.snapshot(), sort_keys=True)
    _, audio = evolve(packet(reject_code=4, events=[event(0, 2)]), state)
    assert not np.any(audio)
    assert json.dumps(state.snapshot(), sort_keys=True) == before


def test_pattern_targeting_and_noise_do_not_consume_cmmm_rng():
    state, _ = evolve(packet(events=[event(0, 2, target=5, pattern=12)]))
    assert np.any(state.coeff_state[:8]) and not np.any(state.coeff_state[8:])
    assert state.prng_state == 0x12345678
    empty, _ = evolve(packet(events=[event(0, 2, target=5, pattern=255)]))
    assert not np.any(empty.coeff_state)


def test_tree_hold_continue_and_snap_preserve_continuous_bus_precedence():
    state = VMState()
    def control(seq, blend, events=()):
        frame = packet(seq=seq, tree_blend_q15=blend, events=events).to_voice_frame()
        begin_wire_block(state, frame)
        cfg = decode_frame(frame)
        apply_wire_controls(state, cfg)
        return cfg
    assert control(1, 4000).tree_blend_q15 == 4000
    assert control(2, 8000, [event(2, 2)]).tree_blend_q15 == 4000
    assert control(3, 12000).tree_blend_q15 == 4000
    assert control(4, 12000, [event(2, 1)]).tree_blend_q15 == 12000
    snap = control(5, 12000, [event(2, 4, p0=3)])
    assert (snap.tree_blend_q15, snap.tree_b) == (32767, 3)
    assert control(6, 6000).tree_blend_q15 == 6000


@pytest.mark.parametrize('model,waveform', [(0, 0), (1, 3), (2, 2), (3, 5)])
def test_v1_model_enum_mapping_and_pitch_buses_reach_vm(model, waveform):
    p = packet(V1, carrier_model_u8=model, note_u8=69, cents_i16=-20,
               detune_cents_i16=30, pitchbend_cents_i16=40)
    cfg = decode_frame(p.to_voice_frame())
    assert cfg.coeff_osc_params.waveform_type == waveform
    assert cfg.coeff_osc_params.cents == 50
    state, _ = vm_step_block(VMState(), p.to_voice_frame())
    assert (state.last_note, state.last_cents) == (69, 50)


def test_v1_off_and_noise_have_no_accidental_pitched_carrier():
    state, silence = vm_step_block(VMState(), packet(V1, carrier_role_u8=0).to_voice_frame())
    assert not np.any(silence) and not np.any(state.coeff_state)
    p = packet(V1, carrier_model_u8=5)
    state, noise = vm_step_block(VMState(), p.to_voice_frame())
    noise = noise.copy()  # VM output is a reusable block buffer.
    assert np.any(noise) and state.ccs_state.phase == 0
    assert not np.any(np.abs(noise.astype(np.int32)) >= 32767)
    assert state.prng_state != 0x12345678
    p.frame_seq += 1
    _, another = vm_step_block(state, p.to_voice_frame())
    assert not np.array_equal(noise, another)


def test_v1_lane4_never_injects_note_coefficients_or_overrides_header():
    p = packet(V1, carrier_role_u8=0, events=[event(4)])
    state, audio = vm_step_block(VMState(), p.to_voice_frame())
    assert not np.any(state.coeff_state) and not np.any(audio)


def test_explicit_emergency_operator_runs_in_tail_not_backbone():
    p = packet(events=[event(1, 1, node=19), event(0, target=3, node=31),
                       event(4, strength=5000, target=3, node=1)])
    state, _ = evolve(p)
    assert state.coeff_state[8] == 5000
    assert not np.any(state.coeff_state[240:])
    assert state.wire_state.forced_on == 1 << 19


def test_v1_waveform_and_pwm_updates_reach_held_note_without_phase_reset():
    p = packet(V1, carrier_model_u8=1, note_u8=69)
    changed, _ = vm_step_block(VMState(), p.to_voice_frame())
    sine = packet(V1, note_u8=69)
    control, _ = vm_step_block(VMState(), sine.to_voice_frame())
    sine.frame_seq = 2
    changed, output = vm_step_block(changed, sine.to_voice_frame())
    output = output.copy()
    control, expected = vm_step_block(control, sine.to_voice_frame())
    np.testing.assert_array_equal(output, expected)
    assert changed.mode_b_state.phase == control.mode_b_state.phase
    pulse = packet(V1, seq=3, carrier_model_u8=2, pw_q15=10000, note_u8=69)
    before = changed.mode_b_state.phase
    changed, _ = vm_step_block(changed, pulse.to_voice_frame())
    assert changed.mode_b_state.pwm_width_q15 == 10000
    assert changed.mode_b_state.pwm_enabled
    assert changed.mode_b_state.phase != before and changed.mode_b_state.phase != 0


def test_external_consumer_events_remain_versioned_and_lossless():
    p = packet(V1, events=[event(3, p0=2, p1=1), event(5, 2, p0=100, p1=4),
                           event(6, 1, target=2, node=5, p0=80, p1=3)])
    frame = p.to_voice_frame()
    events = decode_wire_events(frame.wire_data)
    assert [(e.lane, e.variant, e.param0, e.param1) for e in events] == [(3, 0, 2, 1), (5, 2, 100, 4), (6, 1, 80, 3)]
    assert V1.from_voice_frame(frame).to_bytes() == p.to_bytes()


def test_native_frame_serialization_keeps_its_pitch():
    frame = VoiceFrame(note=81, cents=-21)
    assert (V0.from_voice_frame(frame).note_u8, V0.from_voice_frame(frame).cents_i16) == (81, -21)


@pytest.mark.parametrize('weight', [0, 1, 17, 127, 30001, 32767])
def test_packet_family_weights_are_integer_exact_for_authored_and_toggled_ops(weight):
    p = packet(events=[event(1, 1, node=20)])
    p.w_family_q15[7] = weight
    frame = p.to_voice_frame()
    cfg = decode_frame(frame)
    state = VMState()
    begin_wire_block(state, frame)
    apply_wire_controls(state, cfg)
    assert [(e.op_id, e.strength_q15) for e in cfg.backbone_instrs] == [(20, weight)]
    frame.ops_enable = ['FoldPressure']
    assert decode_frame(frame).backbone_instrs[0].strength_q15 == weight


@pytest.mark.parametrize('transition', [0, 1, 2])
def test_note_transition_preserves_phase_only_when_requested(transition):
    state = VMState()
    first = packet(V1).to_voice_frame()
    begin_wire_block(state, first)
    state.ccs_state.phase = 123456
    state.mode_b_state.phase = 234567
    state.coeff_state[0] = 900
    state.wire_state.forced_off = 1 << 15
    p = packet(V1, seq=2, transition_kind_u8=transition)
    p.flags |= VoiceFlags.GATE_RISE
    begin_wire_block(state, p.to_voice_frame())
    assert state.wire_state.forced_off == 0
    assert state.ccs_state.phase == (123456 if transition else 0)
    assert state.mode_b_state.phase == (234567 if transition else 0)
    assert state.coeff_state[0] == (900 if transition else 0)


def test_packet_rendering_is_bit_exact_with_and_without_jit():
    from symcrash_python_ap.vm import hotpath_jit
    if not hotpath_jit.NUMBA_AVAILABLE:
        pytest.skip('Numba is unavailable')
    before = hotpath_jit.jit_enabled()
    rendered = []
    try:
        for enabled in [False, True]:
            hotpath_jit.set_jit_enabled(enabled)
            state = VMState()
            blocks = []
            for seq in range(1, 9):
                p = packet(V1, seq=seq, carrier_model_u8=seq % 4, note_u8=69,
                    events=[event(0, seq % 3, target=5, pattern=4)])
                p.w_family_q15[7] = 20000
                frame = p.to_voice_frame()
                frame.ops_enable = ['FoldPressure']
                state, block = vm_step_block(state, frame)
                blocks.append(block.copy())
            rendered.append(np.concatenate(blocks))
        np.testing.assert_array_equal(*rendered)
    finally:
        hotpath_jit.set_jit_enabled(before)
