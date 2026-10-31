"""Post-synthesis wire controls and independent stereo auxiliary returns.

Python material-hit profile: param0 is unsigned Q0.8, multiplied by the
authoritative Q15 event strength with nearest-even rounding. A burst targets
the material parameter for max(1, duration) output blocks, including arrival.
Overlaps add in wide integers and saturate once. MaterialEngine owns slew.
Auxiliary inputs are post-voice-gain/post-pan, before the main material insert.
"""
from copy import deepcopy
from dataclasses import dataclass, replace

from .fixed import RSHIFT_RNE, mul_q15_sat, sat16
from .frames import ParamFrame
from .material.material_engine import MaterialEngine
from .primitives import Q15_ONE, Q15_NEG_ONE
from .vm.hotpath_jit import jit_enabled


def _saturate_bus(values):
    # SAT16 is identity for these ordinary values; still detach the output.
    # Custom containers/scalars retain the reference's per-element behavior.
    if (jit_enabled() and type(values) is list
            and set(map(type, values)) <= {int}
            and (not values or min(values) >= Q15_NEG_ONE and max(values) <= Q15_ONE)):
        return values.copy()
    return [sat16(x) for x in values]


@dataclass(frozen=True)
class MaterialBurst:
    drive_q15: int
    bias_q15: int
    remaining: int


class PostMixControls:
    """Output-block clock for material bursts, independent of voice service.

    Input events must come from a serviced VM ExecutionReport, which owns
    packet acceptance and exactly-once delivery. Voice IDs and lane IDs define
    order for simultaneous global commands; the last K command wins.
    """
    def __init__(self):
        self.k_mode = None
        self.oversample_request = 0
        self.bursts = []
        self.unhandled_events = ()

    def step(self, params: ParamFrame, events=()) -> ParamFrame:
        unhandled = []
        for voice_id, event in sorted(events, key=lambda pair: (pair[0], pair[1].lane)):
            if event.lane == 3:
                if event.param0 <= 3:
                    self.k_mode = event.param0
                self.oversample_request = event.param1
                if event.param0 > 3 or event.param1:
                    # No oversample enum/rate table is frozen for this backend.
                    unhandled.append((voice_id, event))
            elif event.lane == 5:
                amount = RSHIFT_RNE(event.strength_q15 * event.param0, 8)
                if amount:
                    self.bursts.append(MaterialBurst(
                        amount if event.variant in (0, 2) else 0,
                        amount if event.variant in (1, 2) else 0,
                        max(1, event.param1)))
            else:
                unhandled.append((voice_id, event))
        self.unhandled_events = tuple(unhandled)
        result = replace(params,
            K_mode=params.K_mode if self.k_mode is None else self.k_mode,
            drive_q15=max(0, min(Q15_ONE, params.drive_q15 + sum(b.drive_q15 for b in self.bursts))),
            bias_q15=sat16(params.bias_q15 + sum(b.bias_q15 for b in self.bursts)))
        self.bursts = [replace(b, remaining=b.remaining - 1) for b in self.bursts if b.remaining > 1]
        return result


@dataclass
class AuxiliaryBus:
    engine: MaterialEngine
    material_id: int
    params: ParamFrame
    return_gain_q15: int


class AuxiliaryReturns:
    """Two explicitly configured effects buses; unconfigured buses have no return.

    Each configured engine runs once per output block, including silent inputs,
    so its delay/reverb tail survives note release and service skips. No dry
    voice gain, pan, master gain, or safety processing is applied a second time.
    """
    def __init__(self, block_size=64):
        self.block_size = block_size
        self.buses = {'A': None, 'B': None}
        self.last_inputs = self.empty_inputs()

    def empty_inputs(self):
        return {name: ([0] * self.block_size, [0] * self.block_size) for name in self.buses}

    def configure(self, name, material_id, params=None, return_gain_q15=Q15_ONE):
        if name not in self.buses:
            raise ValueError('Auxiliary bus must be A or B')
        if material_id is None:
            self.buses[name] = None
            return
        if not isinstance(material_id, int) or not 0 <= material_id <= 10:
            raise ValueError('Unknown auxiliary material program')
        if not isinstance(params, ParamFrame):
            raise TypeError('Auxiliary configuration requires a ParamFrame')
        if not isinstance(return_gain_q15, int) or not 0 <= return_gain_q15 <= Q15_ONE:
            raise ValueError('Auxiliary return gain must be unsigned Q15')
        existing = self.buses[name]
        engine = existing.engine if existing is not None else MaterialEngine(self.block_size)
        self.buses[name] = AuxiliaryBus(engine, material_id, deepcopy(params), return_gain_q15)

    def add_voice(self, inputs, left, right, send_a, send_b):
        for name, amount in (('A', send_a), ('B', send_b)):
            amount = max(0, min(Q15_ONE, int(amount)))
            if not amount:
                continue
            out_l, out_r = inputs[name]
            for i in range(self.block_size):
                out_l[i] += mul_q15_sat(int(left[i]), amount)
                out_r[i] += mul_q15_sat(int(right[i]), amount)

    def process(self, inputs):
        self.last_inputs = {name: (_saturate_bus(left), _saturate_bus(right))
                            for name, (left, right) in inputs.items()}
        out_l, out_r = [0] * self.block_size, [0] * self.block_size
        for name, bus in self.buses.items():
            if bus is None:
                continue
            left, right = self.last_inputs[name]
            left, right = bus.engine.step_block(
                left, right, bus.params, bus.material_id,
                filter_cutoff_q15=bus.params.filter_cutoff_q15,
                filter_reso_q15=bus.params.filter_reso_q15,
                filter_mode=bus.params.filter_mode)
            for i in range(self.block_size):
                out_l[i] += mul_q15_sat(left[i], bus.return_gain_q15)
                out_r[i] += mul_q15_sat(right[i], bus.return_gain_q15)
        return out_l, out_r
