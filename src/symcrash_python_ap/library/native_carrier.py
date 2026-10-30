"""Portable native carrier parameters shared by IR and capsule loading.

One authored carrier may expand into a live unison note group. Arbitrary L0
routing and externally loaded sample/wavetable/granular resources are separate.
"""
from dataclasses import asdict, dataclass, fields
import math
from typing import Optional

from ..primitives import Q15_ONE, Q15_HALF, Q16_ONE, ms_to_blocks
from .classic_modulation import ClassicModulation, extract_classic_modulation


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


def finite(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in {low}..{high}')
    return value


@dataclass(frozen=True)
class NativeCarrierConfig:
    version: int = 3
    classic_modulation: Optional[ClassicModulation] = None
    unison_voices: int = 1
    unison_spread_cents: int = 10
    unison_width_q15: int = Q15_ONE
    waveform_type: int = 3
    coeff_table_id: int = 1
    partial_phase_spread_q15: int = 0
    pwm_enabled: bool = False
    detune_cents: int = 0
    oeis_sequence_id: int = -1
    oeis_rolloff_db: float = -3.0
    noise_color: int = 0
    noise_level_q15: int = 0
    noise_attack_blocks: int = 1
    noise_decay_blocks: int = 45
    noise_seed_u32: int = 0
    fm_depth_q15: int = 0
    fm_source: int = 0
    drone_mode: bool = False
    legato_mode: bool = False
    fm_topo_a_id: Optional[int] = None
    fm_topo_b_id: Optional[int] = None
    fm_topo_blend_q15: int = 0
    filter_cutoff_q15: int = Q15_ONE
    filter_reso_q15: int = 0
    filter_mode: int = 0

    def __post_init__(self):
        integer(self.version, 3, 3, 'Native carrier version')
        if self.classic_modulation is not None and type(self.classic_modulation) is not ClassicModulation:
            raise ValueError('Invalid classic modulation configuration')
        integer(self.unison_voices, 1, 8, 'Unison voices')
        integer(self.unison_spread_cents, 0, 100, 'Unison spread cents')
        integer(self.unison_width_q15, 0, Q15_ONE, 'Unison width')
        for name, lo, hi in [('waveform_type', 0, 5), ('coeff_table_id', 0, 4),
                ('detune_cents', -32768, 32767), ('oeis_sequence_id', -1, 7),
                ('noise_color', 0, 2), ('noise_seed_u32', 0, 0xffffffff),
                ('noise_attack_blocks', 1, 0x7fffffff), ('noise_decay_blocks', 1, 0x7fffffff),
                ('fm_source', 0, 2), ('filter_mode', 0, 3)]:
            integer(getattr(self, name), lo, hi, name)
        for name in ('partial_phase_spread_q15', 'noise_level_q15', 'fm_depth_q15',
                     'fm_topo_blend_q15', 'filter_cutoff_q15', 'filter_reso_q15'):
            integer(getattr(self, name), 0, Q15_ONE, name)
        for name in ('pwm_enabled', 'drone_mode', 'legato_mode'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        for name in ('fm_topo_a_id', 'fm_topo_b_id'):
            if getattr(self, name) is not None:
                integer(getattr(self, name), 0, 7, name)
        if (self.fm_topo_a_id is None) != (self.fm_topo_b_id is None):
            raise ValueError('FM topology morph requires both endpoints')
        finite(self.oeis_rolloff_db, -120, 0, 'OEIS rolloff')

    def to_dict(self):
        data = asdict(self)
        if data['classic_modulation'] is not None:
            data['classic_modulation']['attack_harmonics'] = list(self.classic_modulation.attack_harmonics)
        return data

    @classmethod
    def from_dict(cls, data):
        unison = {'unison_voices', 'unison_spread_cents', 'unison_width_q15'}
        if type(data) is dict and type(data.get('version')) is int and data['version'] in (1, 2):
            missing = {'classic_modulation'} | (unison if data['version'] == 1 else set())
            if set(data) != {f.name for f in fields(cls)} - missing:
                raise ValueError('Incomplete or unknown legacy native carrier fields')
            defaults = dict(unison_voices=1, unison_spread_cents=10, unison_width_q15=Q15_ONE) if data['version'] == 1 else {}
            data = dict(data, **defaults, version=3, classic_modulation=None)
        if type(data) is not dict or set(data) != {f.name for f in fields(cls)}:
            raise ValueError('Incomplete or unknown native carrier fields')
        data = dict(data)
        if type(data['classic_modulation']) is dict:
            data['classic_modulation'] = ClassicModulation.from_dict(data['classic_modulation'])
        return cls(**data)

    def apply(self, frame):
        # Pitch and filter/LFO parameters are computed by the voice owner.
        for name, value in asdict(self).items():
            if name not in ('version', 'detune_cents', 'filter_cutoff_q15', 'filter_reso_q15', 'filter_mode',
                            'unison_voices', 'unison_spread_cents', 'unison_width_q15', 'classic_modulation'):
                setattr(frame, name, value)

    def validate_mode(self, mode):
        if self.classic_modulation is not None and self.classic_modulation.spectral and mode != 1:
            raise ValueError('Classic timbre, body warble and attack brightness require Mode B')


def extract_native_carrier(ir):
    from ..voice.lfo_clocks import lfo_clock_groups
    lfo_clock_groups(ir.modulators, initial=True)
    from ..instrument_ir import (extract_partials_q15, extract_waveform_type,
        extract_phase_spread_q15, extract_noise_params, extract_oeis_params)
    from .ir_extract import extract_fm6
    from ..vm.harmonic_bank import LFOTarget
    node = next((n for n in ir.layers.L0_graph.nodes if n.type == 'Osc'), None)
    fm6 = extract_fm6(ir)
    mode = 3 if fm6 is not None else (node.params.osc_mode if node and node.params.osc_mode is not None else 1)
    integer(mode, 0, 3, 'Oscillator mode')
    if mode == 2:
        raise ValueError('Sample-mode presets require an explicit sample resource loader')
    if mode == 3 and fm6 is None:
        raise ValueError('FM6 mode requires an FM6 node')
    amps = extract_partials_q15(ir) or []
    waveform = extract_waveform_type(ir)
    params = dict(waveform_type=waveform,
        coeff_table_id={0: 0, 1: 2, 2: 4, 3: 1, 4: 1, 5: 3}.get(waveform, 0),
        partial_phase_spread_q15=extract_phase_spread_q15(ir))
    width = Q15_HALF
    unison_node = next((n for n in ir.layers.L0_graph.nodes if n.type == ('FM6' if fm6 else 'Osc')), None)
    params['classic_modulation'] = extract_classic_modulation(ir, unison_node, mode)
    if unison_node is not None:
        p = unison_node.params
        params.update(unison_voices=p.unison_voices, unison_spread_cents=p.unison_spread,
                      unison_width_q15=round(finite(p.unison_width, 0, 1, 'Unison width') * Q15_ONE))
    if node:
        p = node.params
        width = p.pwm_width_q15
        if width == Q15_HALF and p.pw != .5:
            width = round(finite(p.pw, 0, 1, 'Pulse width') * Q15_ONE)
        finite(p.detune_cents, -32768, 32767, 'Detune cents')
        params.update(pwm_enabled=p.pwm_enable, detune_cents=round(p.detune_cents),
            fm_depth_q15=p.fm_depth_q15 if p.fm_enable else 0,
            fm_source=p.fm_source if p.fm_enable else 0,
            drone_mode=p.drone_mode, legato_mode=p.legato_mode)
    if not amps and any(r.target == LFOTarget.PWM_WIDTH for lfo in ir.modulators if lfo.enabled for r in lfo.routings):
        params['pwm_enabled'] = True
    if mode != 1 and params.get('fm_depth_q15') and params.get('fm_source'):
        raise ValueError('Native Osc FM/PM requires Mode B')
    integer(width, 0, Q15_ONE, 'PWM width')
    oeis = extract_oeis_params(ir)
    if oeis:
        params.update(oeis_sequence_id=oeis['sequence_id'], oeis_rolloff_db=oeis['rolloff_db'])
    if mode == 0 and (amps or oeis or params.get('pwm_enabled') or waveform in (2, 4)
                      or params['partial_phase_spread_q15']):
        raise ValueError('This authored spectrum/PWM requires Mode B; Mode A supports sine/square/saw/triangle LUTs')
    noise = extract_noise_params(ir)
    if noise:
        finite(noise['attack_ms'], 0, 1e9, 'Noise attack')
        finite(noise['decay_ms'], 0, 1e9, 'Noise decay')
        params.update(noise_color=noise['color'], noise_level_q15=noise['level_q15'],
            noise_attack_blocks=ms_to_blocks(noise['attack_ms']),
            noise_decay_blocks=ms_to_blocks(noise['decay_ms']), noise_seed_u32=noise['seed_u32'])
    if fm6 and any((fm6.topo_a, fm6.topo_b, fm6.topo_blend_q15)):
        params.update(fm_topo_a_id=fm6.topo_a, fm_topo_b_id=fm6.topo_b,
                      fm_topo_blend_q15=fm6.topo_blend_q15 or 0)
    morph = ir.layers.L1_lift.fm6_morph
    if morph is not None:
        if fm6 is None or morph.blend_source != 'static':
            raise ValueError('Native FM morph currently requires an FM6 node and static blend source')
        params.update(fm_topo_a_id=morph.topo_a, fm_topo_b_id=morph.topo_b,
                      fm_topo_blend_q15=morph.blend_q15)
    filt = next((n.params for n in ir.layers.L0_graph.nodes if n.type == 'Filter'), None)
    if filt is not None and (filt.filter_enable or filt.cutoff_hz is not None):
        cutoff = filt.filter_cutoff_q15
        if filt.cutoff_hz is not None:
            hz = finite(filt.cutoff_hz, 20, 20000, 'Filter cutoff Hz')
            cutoff = round(math.log(hz / 20) / math.log(1000) * Q15_ONE)
        reso = filt.filter_resonance_q15
        if filt.resonance:
            reso = round(finite(filt.resonance, 0, 1, 'Filter resonance') * Q15_ONE)
        modes = {'lp': 0, 'lpf': 0, 'hp': 1, 'hpf': 1, 'bp': 2, 'bpf': 2, 'notch': 3}
        params.update(filter_cutoff_q15=cutoff, filter_reso_q15=reso,
                      filter_mode=modes.get(filt.mode, filt.filter_type))
    return mode, amps, width, NativeCarrierConfig(**params)


def exact_amp_to_dict(ir):
    from .ir_extract import extract_amp
    amp = extract_amp(ir)
    if amp is None:
        return None
    data = dict(attack_ms=amp.a_ms, decay_ms=amp.d_ms, sustain=amp.s, release_ms=amp.r_ms)
    exact_amp_from_dict(data)
    return data


def exact_amp_from_dict(data):
    from ..envelope_scheduler import ADSRSpec
    if data is None:
        return None
    if type(data) is not dict or set(data) != {'attack_ms', 'decay_ms', 'sustain', 'release_ms'}:
        raise ValueError('Incomplete or unknown exact amplitude envelope')
    for name in ('attack_ms', 'decay_ms', 'release_ms'):
        finite(data[name], 0, 1e9, name)
    finite(data['sustain'], 0, 1, 'Sustain')
    return ADSRSpec(**data)
