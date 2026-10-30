"""Adapters for the frozen 136/148-byte VoiceFrame packets.

These packets are debug/replay contracts, distinct from compact UART updates.
Core corruption rejects the packet. Invalid or reserved event lanes remain in
the diagnostic data but are ignored independently by event consumers.
"""
from dataclasses import dataclass
import struct

from .frames import (
    VoiceFrame, VoiceFramePacketV0b, VoiceFramePacketV1, VoiceFlags,
    WireCarrierHeader, WireFrameData, VFP_MAGIC, VFP_VERSION, VFP_PACKET_LEN,
    VFP_V1_VERSION, VFP_V1_PACKET_LEN, NUM_TREES,
)
from .primitives import Q15_ONE, Q15_NEG_ONE, FAMILY_TABLE, midi_to_freq_hz


@dataclass(frozen=True)
class WireEvent:
    lane: int
    variant: int
    strength_q15: int
    target_kind: int
    target_id: int
    pattern_id: int
    param0: int
    param1: int
    param2: int


def decode_wire_events(wire: WireFrameData) -> tuple[WireEvent, ...]:
    """Ignore malformed/reserved lanes independently, preserving lane order."""
    result = []
    variants = (3, 3, 5, 1, 3 if wire.version == 1 else 4, 3, 3, 0)
    targets = ({0, 3, 5}, {0}, {0}, {0}, {0, 3, 5} if wire.version == 1 else {0},
               {0}, {0, 1, 2}, set())
    for lane in range(8):
        if not wire.event_mask & (1 << lane):
            continue
        payload = wire.event_payload_u64[lane]
        parts = VoiceFramePacketV0b.unpack_event_payload(payload)
        if parts['lane_id'] != lane or parts['strength_q8'] != 0:
            continue
        if parts['variant'] >= variants[lane] or parts['target_kind'] not in targets[lane]:
            continue
        event = WireEvent(lane, parts['variant'], wire.event_strength_q15[lane],
            parts['target_kind'], parts['target_id'], parts['pattern_id'],
            parts['param0_q8'], parts['param1_q8'], parts['param2_q8'])
        if event.target_kind == 3 and event.target_id >= 32:
            continue
        if lane == 1 and (event.target_id >= 24 or event.param0 or event.param1 or event.param2):
            continue
        if lane in (2, 3, 5, 6) and event.param2:
            continue
        if lane == 2 and event.param0 >= NUM_TREES:
            continue
        result.append(event)
    return tuple(result)


def validate_packet_bytes(data: bytes, version: int, length: int) -> None:
    if len(data) != length:
        raise ValueError(f"Packet length must be exactly {length} bytes, got {len(data)}")
    magic, actual_version, actual_length = struct.unpack_from('>HBB', data)
    if magic != VFP_MAGIC:
        raise ValueError(f"Invalid magic: 0x{magic:04X}")
    if actual_version != version:
        raise ValueError(f"Invalid version: {actual_version}, expected {version}")
    if actual_length != length:
        raise ValueError(f"Invalid packet_len: {actual_length}, expected {length}")


def validate_packet(packet, *, events=True):
    expected = ((VFP_V1_VERSION, VFP_V1_PACKET_LEN) if isinstance(packet, VoiceFramePacketV1)
                else (VFP_VERSION, VFP_PACKET_LEN))
    for name, value in [('magic', VFP_MAGIC), ('version', expected[0]),
                        ('packet_len', expected[1]), ('hdr_pad', 0), ('pad0', 0), ('pad1', 0)]:
        if getattr(packet, name) != value:
            return False, f"Invalid {name}: {getattr(packet, name)}, expected {value}"

    def integer(value, lo, hi):
        return isinstance(value, int) and not isinstance(value, bool) and lo <= value <= hi

    ranges = {'program_id': (0, 0xFFFF), 'voice_id': (0, 0xFFFF),
              'frame_seq': (0, 0xFFFFFFFF), 'reject_code': (0, 0xFFFF),
              'flags': (0, 0xFFFF), 'regime_id': (0, 15), 'degradation_level': (0, 255),
              'note_u8': (0, 127), 'cents_i16': (Q15_NEG_ONE, Q15_ONE),
              'pan_q15': (Q15_NEG_ONE, Q15_ONE), 'tree_a_id': (0, NUM_TREES - 1),
              'tree_b_id': (0, NUM_TREES - 1), 'event_mask': (0, 255)}
    for name in ('gain_q15', 'activity_q15', 'sendA_q15', 'sendB_q15', 'tree_blend_q15'):
        ranges[name] = (0, Q15_ONE)
    if isinstance(packet, VoiceFramePacketV1):
        ranges.update({'carrier_model_u8': (0, 5), 'carrier_role_u8': (0, 2),
                       'carrier_archetype_u8': (0, 4), 'transition_kind_u8': (0, 2),
                       'detune_cents_i16': (Q15_NEG_ONE, Q15_ONE),
                       'pitchbend_cents_i16': (Q15_NEG_ONE, Q15_ONE),
                       'pw_q15': (0, Q15_ONE), 'eta_target_q15': (0, Q15_ONE)})
    for name, (lo, hi) in ranges.items():
        if not integer(getattr(packet, name), lo, hi):
            return False, f"{name} out of range: expected integer in {lo}..{hi}"
    for name, hi in [('w_family_q15', Q15_ONE), ('event_strength_q15', Q15_ONE),
                     ('event_payload_u64', 0xFFFFFFFFFFFFFFFF)]:
        values = getattr(packet, name)
        if not isinstance(values, (list, tuple)) or len(values) != 8:
            return False, f"{name} must contain exactly 8 entries"
        if any(not integer(value, 0, hi) for value in values):
            return False, f"{name} entry out of range: expected integer in 0..{hi}"
    if events:
        for lane, (strength, payload) in enumerate(zip(packet.event_strength_q15, packet.event_payload_u64)):
            if not packet.event_mask & (1 << lane):
                if strength or payload:
                    return False, f"Inactive lane {lane} must have zero strength and payload"
            elif payload >> 60 != lane:
                return False, f"Lane {lane} payload has mismatched lane_id: {payload >> 60}"
            elif (payload >> 48) & 255:
                return False, f"Lane {lane} reserved strength_q8 must be zero"
    return True, None


def require_valid_core(packet):
    valid, reason = validate_packet(packet, events=False)
    if not valid:
        raise ValueError(reason)


def packet_to_voice_frame(packet) -> VoiceFrame:
    require_valid_core(packet)
    carrier = None
    if isinstance(packet, VoiceFramePacketV1):
        carrier = WireCarrierHeader(packet.carrier_model_u8, packet.carrier_role_u8,
            packet.carrier_archetype_u8, packet.transition_kind_u8, packet.detune_cents_i16,
            packet.pw_q15, packet.eta_target_q15, packet.pitchbend_cents_i16)
    wire = WireFrameData(
        version=packet.version, program_id=packet.program_id, frame_seq=packet.frame_seq,
        reject_code=packet.reject_code, regime_id=packet.regime_id,
        degradation_level=packet.degradation_level, activity_q15=packet.activity_q15,
        sendA_q15=packet.sendA_q15, sendB_q15=packet.sendB_q15,
        family_weights_q15=tuple(packet.w_family_q15), event_mask=packet.event_mask,
        event_strength_q15=tuple(packet.event_strength_q15),
        event_payload_u64=tuple(packet.event_payload_u64), carrier=carrier)
    # Keep base cents separate from the v1 detune/bend buses for exact replay.
    frame = VoiceFrame(voice_id=packet.voice_id, tree_id=packet.tree_a_id,
        flags=packet.flags, gain_q15=packet.gain_q15, pan_q15=packet.pan_q15,
        sendA_q15=packet.sendA_q15, sendB_q15=packet.sendB_q15,
        tree_a_id=packet.tree_a_id, tree_b_id=packet.tree_b_id,
        tree_blend_q15=packet.tree_blend_q15, note=packet.note_u8, cents=packet.cents_i16,
        f0_hz=midi_to_freq_hz(packet.note_u8, packet.cents_i16),
        is_starting=bool(packet.flags & (VoiceFlags.GATE_RISE | VoiceFlags.RETRIGGER)),
        legato_mode=bool(packet.flags & VoiceFlags.LEGATO), wire_data=wire,
        is_active=packet.reject_code == 0 and bool(packet.flags & (
            VoiceFlags.GATE | VoiceFlags.GATE_RISE | VoiceFlags.RETRIGGER) or packet.activity_q15))
    frame.families = {FAMILY_TABLE[i].name: weight / Q15_ONE
                      for i, weight in enumerate(packet.w_family_q15) if i in FAMILY_TABLE}
    if carrier is not None:
        frame.beta_target_q15 = carrier.eta_q15
        frame.f0_hz = midi_to_freq_hz(frame.note,
            frame.cents + carrier.detune_cents + carrier.pitchbend_cents)
    return frame


def voice_frame_to_packet(packet_type, frame, **overrides):
    if frame.harmonic_gains_q15 is not None or frame.harmonic_gain_bound_q15 != 32768:
        raise ValueError('Native harmonic modulation cannot be encoded by the frozen packet')
    if frame.osc_start_phase_q32 is not None:
        raise ValueError('Native oscillator start phase cannot be encoded by the frozen packet')
    if frame.fm_modulator_q15 is not None:
        raise ValueError('Native FM modulator samples cannot be encoded by the frozen packet')
    if frame.operator_strengths_q15 is not None:
        raise ValueError('Native per-operator graph weights cannot be encoded by the frozen family-bus packet')
    if frame.wire_data is not None and frame.event_commands is not None:
        raise ValueError('A frame cannot carry both wire_data and native event_commands')
    wire = frame.wire_data or frame.event_commands or WireFrameData()
    wanted_version = VFP_V1_VERSION if packet_type is VoiceFramePacketV1 else VFP_VERSION
    if (frame.wire_data is not None or frame.event_commands is not None) and wire.version != wanted_version:
        raise ValueError("Cross-version conversion changes lane 4 semantics; use an explicit migration")
    data = dict(program_id=wire.program_id, frame_seq=wire.frame_seq,
        reject_code=wire.reject_code, voice_id=frame.voice_id, flags=frame.flags,
        regime_id=wire.regime_id, degradation_level=wire.degradation_level,
        note_u8=frame.note, cents_i16=frame.cents, gain_q15=frame.gain_q15,
        pan_q15=frame.pan_q15, activity_q15=wire.activity_q15,
        sendA_q15=frame.sendA_q15, sendB_q15=frame.sendB_q15,
        w_family_q15=list(wire.family_weights_q15), tree_a_id=frame.tree_a_id,
        tree_b_id=frame.tree_b_id, tree_blend_q15=frame.tree_blend_q15,
        event_mask=wire.event_mask, event_strength_q15=list(wire.event_strength_q15),
        event_payload_u64=list(wire.event_payload_u64))
    data.update({key: value for key, value in overrides.items() if value is not None})
    if packet_type is VoiceFramePacketV1:
        c = wire.carrier or WireCarrierHeader(eta_q15=frame.beta_target_q15)
        data.update(carrier_model_u8=c.model, carrier_role_u8=c.role,
            carrier_archetype_u8=c.archetype, transition_kind_u8=c.transition,
            detune_cents_i16=c.detune_cents, pw_q15=c.pulse_width_q15,
            eta_target_q15=frame.beta_target_q15, pitchbend_cents_i16=c.pitchbend_cents)
    result = packet_type(**data)
    require_valid_core(result)
    return result
