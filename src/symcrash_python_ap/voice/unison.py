"""Deterministic live note groups using physical VoiceEngine slots."""
from dataclasses import dataclass
from math import isqrt

from ..primitives import Q15_ONE, RSHIFT_RNE

GOLDEN_Q32 = 0x9E3779B9


def round_ratio(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (2 * remainder > denominator or
                      (2 * remainder == denominator and quotient & 1))


def power_gain(count):
    gain = isqrt(Q15_ONE * Q15_ONE // count)
    # Compare the exact square to the midpoint, without runtime floating point.
    lhs, rhs = 4 * Q15_ONE * Q15_ONE, count * (2 * gain + 1) ** 2
    return gain + (lhs > rhs or (lhs == rhs and gain & 1))


@dataclass(frozen=True)
class UnisonMember:
    group_id: int
    leader_id: int
    index: int
    count: int
    detune_cents: int
    pan_q15: int
    gain_q15: int
    phase_q32: int

    def __post_init__(self):
        limits = {'group_id': (1, 0x7fffffffffffffff), 'leader_id': (0, 65534),
                  'count': (1, 8), 'index': (0, self.count - 1),
                  'detune_cents': (-100, 100), 'pan_q15': (-Q15_ONE, Q15_ONE),
                  'gain_q15': (1, Q15_ONE), 'phase_q32': (0, 0xffffffff)}
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'Invalid unison member {name}')
        if self.gain_q15 != power_gain(self.count) or self.phase_q32 != (self.index * GOLDEN_Q32) & 0xffffffff:
            raise ValueError('Invalid unison gain or initial phase')

    def apply(self, frame):
        frame.cents += self.detune_cents
        if self.detune_cents:
            from ..fixed import midi_to_phase_inc_fixed
            frame.f0_hz = midi_to_phase_inc_fixed(frame.note, frame.cents) * 48000 / 2**32
        frame.pan_q15 = max(-Q15_ONE, min(Q15_ONE, frame.pan_q15 + self.pan_q15))
        if self.count > 1:
            frame.gain_q15 = RSHIFT_RNE(frame.gain_q15 * self.gain_q15, 15)
        if frame.is_starting:
            frame.osc_start_phase_q32 = self.phase_q32
        frame.noise_seed_u32 ^= self.phase_q32


def members(group_id, slots, config):
    count = len(slots)
    for index in range(count):
        numerator, denominator = (0, 1) if count == 1 else (2 * index - count + 1, count - 1)
        yield UnisonMember(group_id, slots[0], index, count,
            round_ratio(config.unison_spread_cents * numerator, denominator),
            round_ratio(config.unison_width_q15 * numerator, denominator),
            power_gain(count), (index * GOLDEN_Q32) & 0xffffffff)


def group_slots(engine, voice_id):
    state = engine.voices.get(voice_id)
    if state is None or state.unison is None:
        return [voice_id]
    group_id = state.unison.group_id
    return [vid for vid, voice in engine.voices.items()
            if voice.unison is not None and voice.unison.group_id == group_id]


def allocate_note_group(engine, note, channel, explicit_id):
    """Reserve a whole plan before mutation; prefer the held note, free slots, tails."""
    from .voice_engine import VoicePhase
    first = engine._find_voice_by_note(note, channel) if explicit_id == 0xffff else explicit_id
    if first is not None and not 0 <= first < engine.max_voices:
        return []
    old = group_slots(engine, first) if first is not None else []
    preferred = ([first] + [vid for vid in old if vid != first]) if first is not None else []
    available = preferred + [vid for vid in range(engine.max_voices) if vid not in preferred
        and (vid not in engine.voices or engine.voices[vid].phase == VoicePhase.FREE)]
    available += [voice.voice_id for voice in sorted(engine.voices.values(),
        key=lambda voice: (-voice.phase_block, voice.voice_id))
        if voice.phase == VoicePhase.RELEASE and voice.voice_id not in available]
    if not available:
        return []
    preset = engine.selected_preset
    if preset is None and available[0] in engine.voices:
        preset = engine.voices[available[0]].compiled_preset
    config = getattr(preset, 'native_carrier', None)
    requested = config.unison_voices if config is not None else 1
    slots = available[:requested]
    group_id = 1 + max((voice.unison.group_id for voice in engine.voices.values()
                       if voice.unison is not None), default=0)
    plans = list(members(group_id, slots, config)) if requested > 1 else [None]
    # If a retrigger shrinks the group, release the surplus copies.
    for vid in old:
        if vid not in slots:
            voice = engine.voices[vid]
            if voice.engine_mode == 0:
                engine.note_off_simple(vid)
            else:
                engine.note_off(vid)
    for vid, plan in zip(slots, plans):
        voice = engine.voices.get(vid)
        if preset is not None and (voice is None or voice.compiled_preset != preset):
            engine.allocate_voice_from_preset(preset, vid)
        else:
            engine._prepare_control_voice(vid)
        engine.voices[vid].unison = plan
    return slots
