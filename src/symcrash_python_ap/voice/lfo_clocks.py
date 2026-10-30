"""Eight voice-owned Q32 phase clocks, with explicit compatible LFO sharing."""
from ..primitives import BLOCK_SIZE
from ..vm.harmonic_bank import LFOSyncMode

LFO_CLOCK_SLOTS = 8
PHASE_MASK = 0xffffffff


def validate_lfo_clock_phases(arc_phases):
    if (type(arc_phases) is not list or len(arc_phases) != LFO_CLOCK_SLOTS
            or any(type(phase) is not int or not 0 <= phase <= PHASE_MASK for phase in arc_phases)):
        raise ValueError('LFO owner requires eight unsigned Q32 phase slots')


def lfo_clock_groups(lfos, *, initial=False):
    """Validate the complete clock plan before changing any owner state."""
    groups = {}
    for lfo in lfos:
        if type(lfo.enabled) is not bool:
            raise ValueError('LFO enabled must be boolean')
        if type(lfo.arc_slot) is not int or not 0 <= lfo.arc_slot < LFO_CLOCK_SLOTS:
            raise ValueError('LFO arc_slot must address one of eight phase clocks')
        if (type(lfo.phase) is not int or not 0 <= lfo.phase <= PHASE_MASK
                or type(lfo.phase_inc) is not int or not 0 <= lfo.phase_inc <= PHASE_MASK):
            raise ValueError('LFO phase and increment must be unsigned Q32 integers')
        if (type(lfo.sync_mode) not in (int, LFOSyncMode)
                or lfo.sync_mode not in tuple(LFOSyncMode)):
            raise ValueError('Invalid LFO sync mode')
        if any(type(getattr(lfo, key)) is not int or getattr(lfo, key) < 0
               for key in ('elapsed_samples', 'delay_samples', 'ramp_samples')):
            raise ValueError('LFO envelope clocks must be nonnegative integers')
        members = groups.setdefault(lfo.arc_slot, [])
        if members:
            first = members[0]
            if ((first.phase_inc, first.sync_mode) != (lfo.phase_inc, lfo.sync_mode)
                    or (initial and first.phase != lfo.phase)):
                raise ValueError('LFOs sharing an arc_slot must agree on phase, increment and sync mode')
        members.append(lfo)
    return groups


def initialize_lfo_clocks(lfos, arc_phases):
    groups = lfo_clock_groups(lfos, initial=True)
    validate_lfo_clock_phases(arc_phases)
    for slot, members in groups.items():
        arc_phases[slot] = members[0].phase


def advance_lfo_clocks(lfos, arc_phases=None, block_samples=BLOCK_SIZE):
    groups = lfo_clock_groups(lfos, initial=arc_phases is None)
    if type(block_samples) is not int or block_samples < 0:
        raise ValueError('LFO clock step must be a nonnegative sample count')
    if not groups:
        return
    if arc_phases is not None:
        validate_lfo_clock_phases(arc_phases)
    for slot, members in groups.items():
        phase = members[0].phase if arc_phases is None else arc_phases[slot]
        active = any(lfo.enabled for lfo in members)
        end = (phase + members[0].phase_inc * block_samples) & PHASE_MASK if active else phase
        for lfo in members:
            lfo.phase = phase
            # Each enabled LFO advances its own delay/ramp once. Its shared
            # phase advances once per group, never once per member.
            lfo.advance_block(block_samples)
            lfo.phase = end
        if arc_phases is not None:
            arc_phases[slot] = end


def gate_lfo_clocks(lfos, arc_phases=None, *, is_legato=False):
    groups = lfo_clock_groups(lfos, initial=arc_phases is None)
    if arc_phases is not None:
        validate_lfo_clock_phases(arc_phases)
    for members in groups.values():
        for lfo in members:
            if arc_phases is None:
                lfo.on_gate(is_legato)
            else:
                lfo.on_gate_with_arc(arc_phases, is_legato)
