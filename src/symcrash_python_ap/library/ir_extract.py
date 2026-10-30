"""
Single-source InstrumentIR extraction (synthesis-path dedup, step 7).

One walk of the L0 graph, shared by:
- CapsuleCompiler._extract_* (library/compiler.py, capsule path)
- compiled_preset_from_ir (library/preset_compiler.py, runtime path)
- render tools (tools/render/*, preview/offline renders)

Design rule: extractors return RAW values (present-or-None) and never
invent defaults — each consumer applies its own documented defaults, so
consolidating the walks cannot silently change behavior.

Dependency-light on purpose: imports only instrument_ir helpers and
primitives (no library/ internals) to stay cycle-free.
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional

from ..primitives import Q15_ONE, Q15_HALF

# Re-exported canonical name->id mapping (defined next to the IR schema)
from ..instrument_ir import (
    WAVEFORM_NAME_TO_ID,
    WAVEFORM_ID_DEFAULT,
    extract_waveform_type,
    extract_partials_q15,
)


# Waveform name -> default Q15 harmonic profile for the Mode B harmonic
# bank (single home; formerly preset_compiler._WAVEFORM_HARMONICS, matching
# the hand-built tables that used to live in synth_master.py).
WAVEFORM_HARMONICS: Dict[str, List[int]] = {
    'sine': [Q15_ONE, 0, 0, 0, 0, 0, 0, 0],
    'saw': [Q15_ONE, Q15_HALF, 10922, 8191, 6553, 5461, 4681, 4096],
    'square': [Q15_ONE, 0, 10922, 0, 6553, 0, 4681, 0],
    'pulse': [Q15_ONE, 0, 10922, 0, 6553, 0, 4681, 0],
    'tri': [Q15_ONE, 0, 3640, 0, 1310, 0, 668, 0],
    'triangle': [Q15_ONE, 0, 3640, 0, 1310, 0, 668, 0],
}


@dataclass(frozen=True)
class IRAmpParams:
    """Raw Amp-node envelope + dynamics (ms-domain, no unit conversion)."""
    a_ms: float
    d_ms: float
    s: float
    r_ms: float
    velocity_sens: float = 1.0
    key_scale_rate: float = 0.0


@dataclass(frozen=True)
class IRFM6Params:
    """
    Raw FM6-node parameters. Every field is exactly what the IR declares;
    None means "not specified" (consumer applies its default / lets the
    registry default stand).
    """
    algorithm: Optional[int] = None
    feedback: Optional[int] = None
    op_ratios: Optional[List[int]] = None    # Q16.16
    op_levels: Optional[List[int]] = None    # Q15
    topo_a: Optional[int] = None
    topo_b: Optional[int] = None
    topo_blend_q15: Optional[int] = None
    post_filter_hz: Optional[float] = None
    mod_index: Optional[float] = None        # PM depth scale (1.0 = legacy)


def extract_amp(ir) -> Optional[IRAmpParams]:
    """First Amp node with an ADSR, or None."""
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Amp" and node.params.adsr is not None:
            adsr = node.params.adsr
            return IRAmpParams(
                a_ms=adsr.a_ms,
                d_ms=adsr.d_ms,
                s=adsr.s,
                r_ms=adsr.r_ms,
                velocity_sens=float(getattr(node.params, 'velocity_sens', 1.0)),
                key_scale_rate=float(getattr(node.params, 'key_scale_rate', 0.0)),
            )
    return None


def extract_fm6(ir) -> Optional[IRFM6Params]:
    """First FM6 node's raw parameters, or None if the preset is not FM6."""
    for node in ir.layers.L0_graph.nodes:
        if node.type == "FM6":
            p = node.params
            return IRFM6Params(
                algorithm=p.algorithm,
                feedback=p.feedback,
                op_ratios=list(p.op_ratios) if p.op_ratios else None,
                op_levels=list(p.op_levels) if p.op_levels else None,
                topo_a=getattr(p, 'fm6_topo_a', None),
                topo_b=getattr(p, 'fm6_topo_b', None),
                topo_blend_q15=getattr(p, 'fm6_topo_blend_q15', None),
                post_filter_hz=getattr(p, 'post_filter_hz', None),
                mod_index=getattr(p, 'mod_index', None),
            )
    return None


def extract_pwm_width_q15(ir) -> Optional[int]:
    """PWM width from the first Osc node with pwm_enable, or None."""
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Osc" and node.params.pwm_enable:
            return node.params.pwm_width_q15
    return None


def extract_fx_sends_q15(ir) -> Optional[tuple[int, int]]:
    """First FXSend node, using the Python profile's unsigned Q15 levels.

    Like the other Simple-mode extractors, this does not execute arbitrary
    L0 graph topology. Levels are finite fractions in [0, 1], rounded to even.
    """
    import math
    for node in ir.layers.L0_graph.nodes:
        if node.type == 'FXSend':
            levels = (node.params.send_a, node.params.send_b)
            if any(not isinstance(x, (int, float)) or isinstance(x, bool)
                   or not math.isfinite(x) or not 0 <= x <= 1 for x in levels):
                raise ValueError('FXSend levels must be finite numbers in [0, 1]')
            return tuple(round(x * Q15_ONE) for x in levels)
    return None


def extract_harmonics_q15(ir) -> Optional[List[int]]:
    """
    Q15 harmonic profile for the Mode B harmonic bank, or None.

    Priority (matches compiled_preset_from_ir / synth_master behavior):
    1. Custom partials (`waveform: partials` + harmonics block) win.
    2. Otherwise the named waveform's default profile, if the name is known.
    """
    partials = extract_partials_q15(ir)
    if partials:
        return partials
    result = None
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Osc" and node.params.waveform:
            profile = WAVEFORM_HARMONICS.get(node.params.waveform.lower())
            if profile is not None:
                result = list(profile)  # last Osc wins (historical behavior)
    return result


# =============================================================================
# LFO MODULATOR SERIALIZATION (capsule SOUND_EXTRAS payload)
# =============================================================================

def lfo_to_dict(lfo) -> Dict:
    """Serialize an LFOState modulator to a JSON-friendly dict."""
    from ..primitives import SAMPLE_RATE_HZ
    raw = asdict(lfo)
    if set(vars(lfo)) != set(raw):
        raise ValueError('Undeclared LFO attributes cannot be serialized')
    _lfo_from_raw(raw)
    return {
        'raw_state_version': 1,
        'raw_state': raw,
        'id': lfo.id,
        'waveform': int(lfo.waveform),
        'sync_mode': int(lfo.sync_mode),
        'rate_hz': float(lfo.rate_hz),
        'depth_q15': int(lfo.depth_q15),
        'delay_ms': lfo.delay_samples * 1000.0 / SAMPLE_RATE_HZ,
        'ramp_ms': lfo.ramp_samples * 1000.0 / SAMPLE_RATE_HZ,
        'arc_slot': int(lfo.arc_slot),
        'routings': [
            {'target': int(r.target), 'amount_q15': int(r.amount_q15)}
            for r in lfo.routings
        ],
    }


def lfo_from_dict(d: Dict):
    """Rebuild an LFOState modulator from lfo_to_dict output."""
    if 'raw_state' in d or 'raw_state_version' in d:
        if type(d.get('raw_state_version')) is not int or d.get('raw_state_version') != 1:
            raise ValueError('Unsupported LFO raw-state version')
        return _lfo_from_raw(d.get('raw_state'))
    from ..vm.harmonic_bank import (
        LFOState, LFORouting, LFOWaveform, LFOSyncMode, LFOTarget,
    )
    routings = [
        LFORouting(target=LFOTarget(r['target']),
                   amount_q15=int(r['amount_q15']))
        for r in d.get('routings', [])
    ]
    lfo = LFOState(
        id=d.get('id', ''),
        waveform=LFOWaveform(d.get('waveform', 0)),
        sync_mode=LFOSyncMode(d.get('sync_mode', 0)),
        routings=routings,
        depth_q15=int(d.get('depth_q15', Q15_HALF)),
        arc_slot=int(d.get('arc_slot', 0)),
    )
    lfo.configure_lfo(
        rate_hz=float(d.get('rate_hz', 5.0)),
        depth_q15=int(d.get('depth_q15', Q15_HALF)),
        waveform=lfo.waveform,
        sync_mode=lfo.sync_mode,
        delay_ms=float(d.get('delay_ms', 0.0)),
        ramp_ms=float(d.get('ramp_ms', 0.0)),
        routings=routings,
        arc_slot=lfo.arc_slot,
    )
    return lfo


def _lfo_from_raw(raw):
    """Restore all declared LFO fields, including disabled and authored phase state."""
    import math
    from ..vm.harmonic_bank import LFOState, LFORouting, LFOWaveform, LFOSyncMode, LFOTarget
    if type(raw) is not dict or set(raw) != {item.name for item in fields(LFOState)}:
        raise ValueError('Incomplete or unknown LFO raw-state fields')
    defaults = LFOState()
    data = dict(raw)
    for name, value in data.items():
        if name in ('waveform', 'sync_mode', 'routings'):
            continue
        default = getattr(defaults, name)
        if type(default) is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = type(value) is type(default)
        if not valid:
            raise ValueError(f'Invalid LFO raw-state type: {name}')
    if type(data['routings']) is not list:
        raise ValueError('Invalid LFO routing list')
    routes = []
    for route in data['routings']:
        if (type(route) is not dict or set(route) != {'target', 'amount_q15'}
                or type(route['amount_q15']) is not int or type(route['target']) not in (int, LFOTarget)):
            raise ValueError('Invalid LFO routing record')
        routes.append(LFORouting(LFOTarget(route['target']), route['amount_q15']))
    if type(data['waveform']) not in (int, LFOWaveform) or type(data['sync_mode']) not in (int, LFOSyncMode):
        raise ValueError('Invalid LFO mode')
    data.update(waveform=LFOWaveform(data['waveform']), sync_mode=LFOSyncMode(data['sync_mode']), routings=routes)
    if (not 0 <= data['phase'] <= 0xffffffff or not 0 <= data['phase_inc'] <= 0xffffffff
            or min(data['delay_samples'], data['ramp_samples'], data['elapsed_samples']) < 0
            or not 0 <= data['current_depth_scale'] <= Q15_ONE or not 0 <= data['arc_slot'] < 8):
        raise ValueError('Invalid LFO phase, clock or arc slot')
    return LFOState(**data)


__all__ = [
    'WAVEFORM_HARMONICS',
    'WAVEFORM_NAME_TO_ID',
    'WAVEFORM_ID_DEFAULT',
    'IRAmpParams',
    'IRFM6Params',
    'extract_amp',
    'extract_fm6',
    'extract_pwm_width_q15',
    'extract_harmonics_q15',
    'extract_waveform_type',
    'extract_partials_q15',
    'lfo_to_dict',
    'lfo_from_dict',
]
