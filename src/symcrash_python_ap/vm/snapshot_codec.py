"""JSON value codec for the VM's explicitly registered oscillator state types.

Snapshots are an off-render-path checkpoint format. Loaded sample/atlas data
is embedded so restore never substitutes a default sound or requires a file.
No pickle, dynamic imports from payload names, or arbitrary object creation.
"""
from dataclasses import fields, is_dataclass
from functools import lru_cache
import math

import numpy as np


@lru_cache(maxsize=1)
def _types():
    from .harmonic_bank import ModeBState, HarmonicWeights
    from .fm6_state import FM6State, FMOperator
    from .fm6_morph_engine import FM6MorphState
    from .sample_source import SampleOscState
    from .wavetable_engine import WavetableState, WavetableAtlas
    from .granular_engine import GranularState, GrainState
    return {cls.__name__: cls for cls in (ModeBState, HarmonicWeights, FM6State,
        FMOperator, FM6MorphState, SampleOscState, WavetableState, WavetableAtlas,
        GranularState, GrainState)}


ARRAY_DTYPES = frozenset(('int16', 'int32', 'int64', 'uint16', 'uint32', 'uint64', 'float32', 'float64'))


def encode(value):
    """Return detached JSON values, preserving tuples and numeric array types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Non-finite snapshot value')
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.name not in ARRAY_DTYPES:
            raise ValueError(f'Unsupported snapshot array dtype: {value.dtype}')
        return {'kind': 'array', 'dtype': value.dtype.name, 'shape': list(value.shape),
                'data': encode(value.reshape(-1).tolist())}
    if isinstance(value, tuple):
        return {'kind': 'tuple', 'items': [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if is_dataclass(value) and _types().get(type(value).__name__) is type(value):
        known = {f.name for f in fields(value)}
        if set(vars(value)) - known:
            raise ValueError(f'Unregistered runtime fields in {type(value).__name__}')
        return {'kind': type(value).__name__,
                'fields': {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    raise TypeError(f'Unsupported snapshot value: {type(value).__name__}')


def decode(value, *, _depth=0):
    """Decode registered data only; malformed arrays fail before NumPy casts."""
    if _depth > 64:
        raise ValueError('Snapshot nesting is too deep')
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Non-finite snapshot value')
        return value
    if type(value) is list:
        return [decode(x, _depth=_depth + 1) for x in value]
    if type(value) is not dict:
        raise ValueError('Snapshot must contain JSON values')
    kind = value.get('kind')
    if kind == 'tuple' and set(value) == {'kind', 'items'} and type(value['items']) is list:
        return tuple(decode(x, _depth=_depth + 1) for x in value['items'])
    if kind == 'array' and set(value) == {'kind', 'dtype', 'shape', 'data'}:
        dtype, shape, data = value['dtype'], value['shape'], value['data']
        if (not isinstance(dtype, str) or dtype not in ARRAY_DTYPES
                or type(shape) is not list or len(shape) > 8
                or any(type(n) is not int or n < 0 for n in shape)
                or type(data) is not list or math.prod(shape) != len(data)):
            raise ValueError('Invalid snapshot array metadata')
        target = np.dtype(dtype)
        if target.kind in 'iu':
            limits = np.iinfo(target)
            if any(type(x) is not int or not limits.min <= x <= limits.max for x in data):
                raise ValueError('Snapshot array value outside its integer dtype')
        else:
            limit = np.finfo(target).max
            if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > limit for x in data):
                raise ValueError('Invalid floating snapshot array value')
        return np.array(data, dtype=target).reshape(shape)
    cls = _types().get(kind) if isinstance(kind, str) else None
    if cls is None or set(value) != {'kind', 'fields'} or type(value['fields']) is not dict:
        raise ValueError(f'Unknown or malformed snapshot kind: {kind!r}')
    if set(value['fields']) != {f.name for f in fields(cls)}:
        raise ValueError(f'Incomplete or unknown fields in {kind} snapshot')
    return cls(**{name: decode(item, _depth=_depth + 1) for name, item in value['fields'].items()})


OSCILLATOR_TYPES = {
    '_mode_b_state': 'ModeBState', '_fm6_state': 'FM6State',
    '_fm6_morph_state': 'FM6MorphState', '_sample_state': 'SampleOscState',
    '_wavetable_state': 'WavetableState', '_granular_state': 'GranularState',
}


def encode_oscillators(state):
    return {name: encode(getattr(state, name)) for name in OSCILLATOR_TYPES}


def decode_oscillators(data):
    if type(data) is not dict or set(data) != set(OSCILLATOR_TYPES):
        raise ValueError('Incomplete oscillator snapshot')
    restored = {}
    for name, expected in OSCILLATOR_TYPES.items():
        value = decode(data[name])
        if value is not None and type(value) is not _types()[expected]:
            raise ValueError(f'Incorrect oscillator snapshot type for {name}')
        restored[name] = value
    return restored
