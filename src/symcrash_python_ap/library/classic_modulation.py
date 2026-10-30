"""Native, note-relative modulation families from L1 (or legacy NodeParams).

Integer runtime laws are specified in WSE-IMPL-CLASSIC-MODULATION-v1.
No mutable clock is hidden here: the voice's serviced block count owns time.
"""
from dataclasses import dataclass, fields

from ..primitives import BLOCK_SIZE, SAMPLE_RATE_HZ, Q15_ONE, RSHIFT_RNE
from ..wavelet.sine_core import sin_cos_q15


UNITY = 1 << 15
HARMONICS = 32


@dataclass(frozen=True)
class ClassicModulation:
    vibrato_inc_q32: int = 0
    amp_depth_q15: int = 0
    timbre_depth_q15: int = 0
    delay_samples: int = 0
    ramp_samples: int = 0
    warble_inc_q32: int = 0
    warble_depth_q15: int = 0
    warble_harmonic: int = 4
    attack_samples: int = 0
    attack_amount_q15: int = 0
    attack_harmonics: tuple[int, ...] = (5, 6)

    def __post_init__(self):
        from .native_carrier import integer
        for name in ('vibrato_inc_q32', 'warble_inc_q32'):
            integer(getattr(self, name), 0, 1 << 31, name)
        for name in ('amp_depth_q15', 'timbre_depth_q15', 'warble_depth_q15', 'attack_amount_q15'):
            integer(getattr(self, name), 0, Q15_ONE, name)
        for name in ('delay_samples', 'ramp_samples', 'attack_samples'):
            integer(getattr(self, name), 0, SAMPLE_RATE_HZ * 1000000, name)
        integer(self.warble_harmonic, 1, HARMONICS, 'Warble harmonic')
        if type(self.attack_harmonics) is not tuple or len(self.attack_harmonics) > HARMONICS:
            raise ValueError('Attack harmonics must be a tuple of at most 32 distinct indices')
        for h in self.attack_harmonics:
            integer(h, 1, HARMONICS, 'Attack harmonic')
        if len(set(self.attack_harmonics)) != len(self.attack_harmonics):
            raise ValueError('Attack harmonic indices must be distinct')

    @classmethod
    def from_dict(cls, data):
        if type(data) is not dict or set(data) != {f.name for f in fields(cls)}:
            raise ValueError('Incomplete or unknown classic modulation fields')
        data = dict(data)
        if type(data['attack_harmonics']) is not list:
            raise ValueError('Serialized attack harmonics must be a list')
        data['attack_harmonics'] = tuple(data['attack_harmonics'])
        return cls(**data)

    @property
    def spectral(self):
        return bool(self.timbre_depth_q15 or self.warble_depth_q15 or
                    (self.attack_amount_q15 and self.attack_samples and self.attack_harmonics))

    def apply(self, frame, block):
        """Sample the start of a serviced block; never advance another owner."""
        if type(block) is not int or block < 0:
            raise ValueError('Classic modulation block must be a nonnegative integer')
        elapsed = block * BLOCK_SIZE
        v = 0
        if elapsed >= self.delay_samples:
            envelope = UNITY
            if self.ramp_samples:
                envelope = min(UNITY, (elapsed - self.delay_samples) * UNITY // self.ramp_samples)
            v = RSHIFT_RNE(sin_cos_q15((block * self.vibrato_inc_q32) & 0xffffffff)[0] * envelope, 15)
        amp = RSHIFT_RNE(v * self.amp_depth_q15, 15)
        if amp:
            frame.gain_q15 = max(0, min(Q15_ONE, RSHIFT_RNE(frame.gain_q15 * (UNITY + amp), 15)))
        if not self.spectral:
            return
        gains = [UNITY] * HARMONICS
        bounds = [UNITY] * HARMONICS
        timbre = RSHIFT_RNE(v * self.timbre_depth_q15, 15)
        gains[1] -= timbre
        gains[3] += timbre
        bounds[1] += self.timbre_depth_q15
        bounds[3] += self.timbre_depth_q15
        warble = RSHIFT_RNE(sin_cos_q15((block * self.warble_inc_q32) & 0xffffffff)[0] * self.warble_depth_q15, 15)
        gains[self.warble_harmonic - 1] += warble
        bounds[self.warble_harmonic - 1] += self.warble_depth_q15
        if self.attack_samples:
            attack = (self.attack_amount_q15 * max(0, self.attack_samples - elapsed)) // self.attack_samples
            for h in self.attack_harmonics:
                gains[h - 1] += attack
                bounds[h - 1] += self.attack_amount_q15
        frame.harmonic_gains_q15 = [max(0, gain) for gain in gains]
        frame.harmonic_gain_bound_q15 = max(bounds)


def extract_classic_modulation(ir, carrier_node, mode):
    from .native_carrier import finite
    values = {}
    for family in ('vibrato', 'body_warble', 'attack_texture'):
        value = getattr(ir.layers.L1_lift, family)
        if value is None and carrier_node is not None:
            value = getattr(carrier_node.params, family)
        if value is None:
            continue
        def fraction(number, name):
            return round(finite(number, 0, 1, name) * Q15_ONE)
        def duration(number, name):
            return round(finite(number, 0, 1e9, name) * SAMPLE_RATE_HZ / 1000)
        def rate(number, name):
            return round(finite(number, 0, SAMPLE_RATE_HZ / (2 * BLOCK_SIZE), name) * (1 << 32) * BLOCK_SIZE / SAMPLE_RATE_HZ)
        if family == 'vibrato':
            values.update(vibrato_inc_q32=rate(value.rate_hz, 'Vibrato rate'),
                amp_depth_q15=fraction(value.depth_amp, 'Vibrato amplitude'),
                timbre_depth_q15=fraction(value.depth_timbre, 'Vibrato timbre'),
                delay_samples=duration(value.delay_ms, 'Vibrato delay'),
                ramp_samples=duration(value.ramp_ms, 'Vibrato ramp'))
        elif family == 'body_warble':
            values.update(warble_inc_q32=rate(value.rate_hz, 'Warble rate'),
                warble_depth_q15=fraction(value.depth, 'Warble depth'), warble_harmonic=value.target_harmonic)
        else:
            if type(value.harmonics) not in (list, tuple):
                raise ValueError('Attack harmonics must be a list of indices')
            values.update(attack_samples=duration(value.bright_ms, 'Attack duration'),
                attack_amount_q15=fraction(value.bright_amount, 'Attack brightness'),
                attack_harmonics=tuple(value.harmonics))
    config = ClassicModulation(**values)
    if config.spectral and mode != 1:
        raise ValueError('Classic timbre, body warble and attack brightness require Mode B')
    return config if config.amp_depth_q15 or config.spectral else None
