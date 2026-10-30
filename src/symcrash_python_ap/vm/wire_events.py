"""Versioned frozen-packet controls and coefficient event stages.

Wire lane 4 injections precede the backbone; lane 0 pulses follow it.
Material/cloud lanes remain available on the frame for their owning consumer.
No wire lane is passed through the legacy VM lane-number-to-operator mapping.
"""
from dataclasses import dataclass, field

import numpy as np

from ..frames import VoiceFlags
from ..primitives import Q15_ONE, Q15_NEG_ONE, Q15_SCALE, RSHIFT_RNE
from ..wire_frame import WireEvent, decode_wire_events
from .tables import NODE_LEN
from ..tables.pattern_table import PATTERN_TABLE

TAIL_IDS = {2, 11, 12, 19, 23}


@dataclass
class WireExecutionState:
    program_id: int = -1
    frame_seq: int = -1
    forced_on: int = 0
    forced_off: int = 0
    tree_held: bool = False
    tree_blend_q15: int = 0
    carrier_role: int = -1
    carrier_model: int = -1
    pulse_width_q15: int = -1
    events: tuple[WireEvent, ...] = field(default_factory=tuple)


def begin_wire_block(state, frame) -> bool:
    """Process allocation and sequence before the ordinary decoder runs."""
    wire = getattr(frame, 'wire_data', None)
    if wire is None:
        commands = getattr(frame, 'event_commands', None)
        if commands is not None:
            if commands.version != 1 or commands.reject_code:
                raise ValueError('Native authored commands require a valid v0b event profile')
            if frame.is_starting:
                state.wire_state = WireExecutionState()
            state.wire_state.events = decode_wire_events(commands)
            state.wire_state.frame_seq = commands.frame_seq
        return True
    if getattr(frame, 'event_commands', None) is not None:
        raise ValueError('A frame cannot contain both a wire packet and native authored commands')
    current = state.wire_state
    if wire.reject_code:
        return False
    new_program = current.program_id != wire.program_id
    delta = (wire.frame_seq - current.frame_seq) & 0xFFFFFFFF
    if not new_program and current.frame_seq >= 0:
        if delta == 0:
            current.events = ()
            return True  # Held continuous buses; one-shots are consumed once.
        if delta >= 0x80000000:
            return False  # Stale packet: no state or RNG mutation.
    if new_program or frame.flags & (VoiceFlags.GATE_RISE | VoiceFlags.RETRIGGER):
        cents = frame.cents
        if wire.carrier is not None:
            cents += wire.carrier.detune_cents + wire.carrier.pitchbend_cents
        preserve_phase = not new_program and (frame.legato_mode or
            wire.carrier is not None and wire.carrier.transition in (1, 2))
        if preserve_phase:
            # A logical note still resets persistent command overrides, but
            # GLIDE/BRIDGE preserve the oscillator and evolving coefficients.
            state.wire_state = WireExecutionState()
        else:
            state.reset_for_note_on(note=frame.note, cents=cents)
        current = state.wire_state
        current.tree_blend_q15 = frame.tree_blend_q15
    current.program_id = wire.program_id
    current.frame_seq = wire.frame_seq
    current.events = decode_wire_events(wire)
    return True


def apply_wire_controls(state, cfg):
    from .decode import BackboneEntry, wire_op_strength
    current = state.wire_state
    base = {entry.op_id: entry for entry in cfg.backbone_instrs}
    for event in current.events:
        if event.lane == 1:
            bit = 1 << event.target_id
            enabled = bool(current.forced_on & bit or
                event.target_id in base and not current.forced_off & bit)
            turn_on = event.variant == 1 or event.variant == 0 and not enabled
            if turn_on:
                current.forced_on |= bit
                current.forced_off &= ~bit
            else:
                current.forced_off |= bit
                current.forced_on &= ~bit
        elif event.lane == 2:
            if event.param0:
                cfg.tree_b = event.param0
            if event.variant in (0, 1):
                current.tree_held = False
            elif event.variant == 2:
                current.tree_held = True
            else:
                current.tree_held = False
                cfg.tree_blend_q15 = 0 if event.variant == 3 else Q15_ONE
    if current.tree_held:
        cfg.tree_blend_q15 = current.tree_blend_q15
    current.tree_blend_q15 = cfg.tree_blend_q15
    effective = [entry for entry in cfg.backbone_instrs
                 if not current.forced_off & (1 << entry.op_id)]
    for op_id in range(24):
        if current.forced_on & (1 << op_id) and op_id not in base:
            effective.append(BackboneEntry(op_id=op_id,
                strength_q15=wire_op_strength(op_id, cfg.wire_family_weights_q15)))
    # Wire author order never allows tail operators to run in the backbone.
    cfg.wire_tail_ops = [entry for entry in effective if entry.op_id in TAIL_IDS]
    cfg.backbone_instrs = [entry for entry in effective if entry.op_id not in TAIL_IDS]
    cfg.wire_events = current.events
    cfg.wire_frame_seq = current.frame_seq


def _target_mask(event, mask):
    if event.target_kind == 0:
        return mask
    if event.target_kind == 3:
        return mask & (1 << event.target_id)
    pattern = PATTERN_TABLE.get(event.pattern_id)
    return mask & pattern.mask if pattern is not None else 0


def _noise_word(seed):
    """Local xorshift mixer; does not consume the Voice/CMMM transition RNG."""
    seed ^= (seed << 13) & 0xFFFFFFFF
    seed ^= seed >> 17
    seed ^= (seed << 5) & 0xFFFFFFFF
    return seed & 0xFFFFFFFF


def apply_wire_injections(state, cfg, *, before_backbone):
    events = [event for event in cfg.wire_events
              if event.lane == (4 if before_backbone else 0)]
    if before_backbone and cfg.wire_version != 1:
        return  # v1 carrier configuration is never a note injection.
    additions = np.zeros_like(state.coeff_state, dtype=np.int64)
    for event in events:
        if event.strength_q15 == 0:
            continue
        mask = _target_mask(event, cfg.mask_u32)
        for node in range(32):
            if not mask & (1 << node):
                continue
            start = node * NODE_LEN
            if event.lane == 0 and event.variant == 1:
                additions[start] += (32 - node) * event.strength_q15 // 32
            elif event.variant == 2:
                # Stable noise for this frame/target, independent of traversal,
                # service skips, invalid lanes and other voices' random state.
                for offset in range(NODE_LEN):
                    index = start + offset
                    seed = ((cfg.wire_frame_seq + 1) * 0x9E3779B9 ^
                            (index + 1) * 0x85EBCA6B ^ event.param0) & 0xFFFFFFFF
                    sample = (_noise_word(seed) & 0xFFFF) - Q15_SCALE
                    additions[index] += RSHIFT_RNE(sample * event.strength_q15, 15)
            elif event.lane == 4 and event.variant == 1:
                # Deterministic signed inharmonic excitation, authored by pitch
                # selector. The Voice owns any repeated duration/window.
                sign = -1 if (node + event.param0) & 1 else 1
                additions[start] += sign * event.strength_q15
            else:
                additions[start] += event.strength_q15
    state.coeff_state[:] = np.clip(state.coeff_state.astype(np.int64) + additions,
                                   Q15_NEG_ONE, Q15_ONE)
