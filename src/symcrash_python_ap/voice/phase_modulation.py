"""Read lane-rate PM from voice-owned LFOs without advancing their clocks."""
from ..primitives import RSHIFT_RNE, SAT16, Q15_ONE, Q15_SCALE
from ..vm.harmonic_bank import LFOTarget, VM_K


def phase_lfo_lanes(lfos, arc_phases=None, depth_scale_q15=Q15_ONE):
    """Use explicit PHASE routes, or the first enabled LFO when none exist.

    Lane k samples time k within the current block (WSE-STD-210 option L1).
    The caller advances every LFO once after this read, including its envelope.
    Explicit disabled routes still select the routing policy, so disabling a
    phase LFO cannot accidentally select an unrelated pitch LFO.
    """
    routes = [(lfo, route.amount_q15) for lfo in lfos for route in lfo.routings
              if route.target == LFOTarget.PHASE]
    if not routes:
        first = next((lfo for lfo in lfos if lfo.enabled), None)
        # 32768 here means exact identity, not a serialized Q15 parameter.
        routes = [(first, Q15_SCALE)] if first is not None else []
    totals = [0] * VM_K
    for lfo, amount in routes:
        if not lfo.enabled:
            continue
        phase = lfo.phase if arc_phases is None else arc_phases[lfo.arc_slot]
        for k in range(VM_K):
            elapsed = lfo.elapsed_samples + k
            if elapsed < lfo.delay_samples:
                continue
            envelope = Q15_ONE
            if lfo.ramp_samples and elapsed < lfo.delay_samples + lfo.ramp_samples:
                envelope = ((elapsed - lfo.delay_samples) * Q15_ONE) // lfo.ramp_samples
            value = lfo.get_waveform_value((phase + k * lfo.phase_inc) & 0xffffffff)
            value = RSHIFT_RNE(value * envelope, 15)
            value = RSHIFT_RNE(value * lfo.depth_q15, 15)
            value = RSHIFT_RNE(value * depth_scale_q15, 15)
            totals[k] += RSHIFT_RNE(value * amount, 15)
    # Sum before clipping so equivalent routing order cannot alter playback.
    return [SAT16(value) for value in totals]
