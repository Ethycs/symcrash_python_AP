"""Canonical, checked InstrumentIR authoring export and atomic file writes."""
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import tempfile


def _value(value, path='instrument_ir_v0', *, omit_none=False):
    from .. import instrument_ir as model

    if isinstance(value, Enum):
        return _value(value.value, path, omit_none=omit_none)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_value(item, f'{path}[{i}]', omit_none=omit_none) for i, item in enumerate(value)]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError(f'{path}: authoring map keys must be strings')
        return {key: _value(item, f'{path}.{key}', omit_none=omit_none) for key, item in value.items()}
    if not is_dataclass(value):
        raise ValueError(f'{path}: unsupported authoring value {type(value).__name__}')
    declared = {item.name for item in fields(value)}
    if set(vars(value)) - declared:
        raise ValueError(f'{path}: undeclared fields cannot be saved')
    result = {item.name: _value(getattr(value, item.name), f'{path}.{item.name}', omit_none=omit_none)
              for item in fields(value) if not omit_none or getattr(value, item.name) is not None}
    if isinstance(value, (model.RegimeTransition, model.TransitionRule)):
        result['from'] = result.pop('from_regime')
    if isinstance(value, model.NodeParams):
        # Explicit units avoid the author's legacy heuristic for small ratios.
        for source, target in (('op_ratios', 'op_ratios_q16'), ('op_levels', 'op_levels_q15')):
            if source in result:
                result[target] = result.pop(source)
    if isinstance(value, model.LFORouting):
        result['type'] = model.LFOTarget(value.target).name.lower()
        result.pop('target')
    if isinstance(value, model.LFOState):
        result['waveform'] = model.LFOWaveform(value.waveform).name.lower()
        result['sync_mode'] = model.LFOSyncMode(value.sync_mode).name.lower()
        result['targets'] = result.pop('routings')
    if isinstance(value, model.TreeTarget):
        for key in ('A', 'B'):
            result[key] = model.LEGACY_TREE_NAMES.get(result[key], result[key])
    if isinstance(value, (model.EmitTemplate, model.OverlayAdd)):
        canonical = {}
        for key, item in result['families'].items():
            name = model.LEGACY_FAMILY_NAMES.get(key, key)
            if name in canonical and canonical[name] != item:
                raise ValueError(f'{path}: conflicting family aliases for {name}')
            canonical[name] = item
        result['families'] = canonical
    if isinstance(value, model.InstrumentIR):
        # Legacy property fallbacks migrate to the effective Tower layers.
        for legacy, current, label in (
            (value._control_compat, value.layers.L1_lift.control, 'control'),
            (value._envelopes_compat, value.layers.L2_regimes.envelopes, 'envelopes'),
        ):
            if legacy and current and _value(legacy) != _value(current):
                raise ValueError(f'{path}: conflicting legacy and layered {label}')
        result.pop('_envelopes_compat', None)
        result.pop('_control_compat', None)
        if value.control is not None:
            result['layers']['L1_lift']['control'] = _value(value.control, path + '.control', omit_none=omit_none)
        result['layers']['L2_regimes']['envelopes'] = _value(value.envelopes, path + '.envelopes', omit_none=omit_none)
        result['modulators'] = {'lfos': result['modulators']}
    return result


def _difference(left, right, path='instrument_ir_v0'):
    if type(left) is dict and type(right) is dict:
        if left.keys() != right.keys():
            return path + ' (fields)'
        for key in left:
            difference = _difference(left[key], right[key], path + '.' + key)
            if difference:
                return difference
    elif type(left) is list and type(right) is list:
        if len(left) != len(right):
            return path + ' (length)'
        for i, (a, b) in enumerate(zip(left, right)):
            difference = _difference(a, b, f'{path}[{i}]')
            if difference:
                return difference
    elif left != right:
        return path
    return None


def instrument_to_dict(ir):
    """Return detached canonical authoring data, verifying every stored field."""
    from ..instrument_ir import InstrumentIR, load_instrument_from_dict

    if not isinstance(ir, InstrumentIR):
        raise TypeError('Expected InstrumentIR')
    expected = _value(ir)
    result = {'instrument_ir_v0': _value(ir, omit_none=True)}
    restored = load_instrument_from_dict(result)
    difference = _difference(expected, _value(restored))
    if difference:
        raise ValueError(f'InstrumentIR export would change {difference}')
    return result


def save_instrument(ir, path):
    """Serialize before touching the destination, then atomically replace it."""
    from .. import instrument_ir as model

    requested = Path(path)
    destination = requested.resolve()
    data = instrument_to_dict(ir)
    if requested.suffix.lower() == '.json':
        text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    else:
        if not model.HAS_YAML:
            raise ImportError('YAML support requires pyyaml. Install with: pip install pyyaml')
        text = model.yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                dir=destination.parent, prefix='.' + destination.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
