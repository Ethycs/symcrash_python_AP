"""Versioned data-only codec for the WSE-STD-070 legacy tower model.

The existing 16-byte preset header is unchanged. Its body is SYTW followed by
canonical UTF-8 JSON. Explicit container tags preserve tuples, sets and typed
dictionary keys; arbitrary Python objects are never reconstructed.
"""
from dataclasses import fields
import json
import math
from typing import Any, Union, get_args, get_origin, get_type_hints


BODY_MAGIC = b'SYTW'
BODY_VERSION = 1
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_VALUE_DEPTH = 64


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


def _encode(value, depth=0):
    if depth > MAX_VALUE_DEPTH:
        raise ValueError('Preset value nesting exceeds codec limit')
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Preset values must be finite')
        return value
    if type(value) in (list, tuple, set):
        items = [_encode(item, depth + 1) for item in value]
        if isinstance(value, set):
            items.sort(key=_json)
        return {'type': type(value).__name__, 'items': items}
    if type(value) is dict:
        items = [[_encode(key, depth + 1), _encode(item, depth + 1)] for key, item in value.items()]
        items.sort(key=lambda pair: _json(pair[0]))
        return {'type': 'dict', 'items': items}
    raise ValueError(f'Unsupported preset value type: {type(value).__name__}')


def _decode(value, depth=0):
    if depth > MAX_VALUE_DEPTH:
        raise ValueError('Preset value nesting exceeds codec limit')
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if not isinstance(value, dict) or set(value) != {'type', 'items'} or not isinstance(value['items'], list):
        raise ValueError('Invalid typed preset value')
    kind, items = value['type'], value['items']
    if kind == 'dict':
        result = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('Invalid preset dictionary entry')
            key, item = (_decode(v, depth + 1) for v in pair)
            try:
                if key in result:
                    raise ValueError('Duplicate preset dictionary key')
                result[key] = item
            except TypeError as exc:
                raise ValueError('Unhashable preset dictionary key') from exc
        return result
    if kind in ('list', 'tuple', 'set'):
        decoded = [_decode(item, depth + 1) for item in items]
        if kind == 'list':
            return decoded
        if kind == 'tuple':
            return tuple(decoded)
        try:
            result = set(decoded)
        except TypeError as exc:
            raise ValueError('Unhashable preset set member') from exc
        if len(result) != len(decoded):
            raise ValueError('Duplicate preset set member')
        return result
    raise ValueError('Unknown preset container type')


def _matches(value, expected):
    """Validate the declared layer schema without coercing authored types."""
    if expected is Any:
        return True  # _encode/_decode separately enforce the data-only domain.
    origin, args = get_origin(expected), get_args(expected)
    if origin is Union:
        return any(_matches(value, choice) for choice in args)
    if origin is dict:
        return type(value) is dict and all(_matches(k, args[0]) and _matches(v, args[1]) for k, v in value.items())
    if origin in (list, set):
        return type(value) is origin and all(_matches(v, args[0]) for v in value)
    if origin is tuple:
        return type(value) is tuple and len(value) == len(args) and all(_matches(v, t) for v, t in zip(value, args))
    if expected is float:
        return type(value) in (float, int) and math.isfinite(value)
    return type(value) is expected


def _layer_types():
    # Lazy import: preset_model imports the codec only at the save/load boundary.
    from .preset_model import (
        L0PatchLayer, L1FamilyLayer, L2InstrumentLayer,
        L3CockpitLayer, L4ForgeLayer, L5LabLayer,
    )
    return (L0PatchLayer, L1FamilyLayer, L2InstrumentLayer, L3CockpitLayer, L4ForgeLayer, L5LabLayer)


def _validate_layer(layer, cls):
    if type(layer) is not cls:
        raise ValueError(f'Expected {cls.__name__}')
    for name, expected in get_type_hints(cls).items():
        if not _matches(getattr(layer, name), expected):
            raise ValueError(f'Invalid type for {cls.__name__}.{name}')
    if not layer.is_valid():
        raise ValueError(f'Invalid {cls.__name__}')


def encode_body(body):
    """Return deterministic bytes and the highest explicitly authored level."""
    from .preset_model import TowerLevel

    layers = []
    for level, cls in enumerate(_layer_types()):
        level = TowerLevel(level)
        if not body.is_explicit(level):
            continue
        layer = body.get_layer(level)
        _validate_layer(layer, cls)
        values = {f.name: getattr(layer, f.name) for f in fields(cls)}
        layers.append({'level': int(level), 'fields': _encode(values)})
    try:
        data = BODY_MAGIC + _json({'version': BODY_VERSION, 'layers': layers}).encode('utf-8')
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid preset body encoding') from exc
    if len(data) > MAX_BODY_BYTES:
        raise ValueError('Preset body exceeds codec size limit')
    highest = TowerLevel(layers[-1]['level']) if layers else TowerLevel.L0_PATCH
    return data, highest


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(text):
    raise ValueError(f'Non-finite JSON value: {text}')


def decode_body(data, highest):
    """Parse and validate authored layers; completion belongs to the model."""
    from .preset_model import PresetBody, TowerLevel

    if len(data) > MAX_BODY_BYTES:
        raise ValueError('Preset body exceeds codec size limit')
    if not data.startswith(BODY_MAGIC):
        raise ValueError('Unsupported preset body format')
    try:
        obj = json.loads(data[len(BODY_MAGIC):].decode('utf-8'), object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid preset body encoding') from exc
    if not isinstance(obj, dict) or set(obj) != {'version', 'layers'}:
        raise ValueError('Invalid preset body envelope')
    if type(obj['version']) is not int or obj['version'] != BODY_VERSION:
        raise ValueError('Unsupported preset body version')
    if not isinstance(obj['layers'], list) or len(obj['layers']) > 6:
        raise ValueError('Invalid preset layer list')
    body = PresetBody()
    seen = set()
    for record in obj['layers']:
        if not isinstance(record, dict) or set(record) != {'level', 'fields'}:
            raise ValueError('Invalid preset layer record')
        level = record['level']
        if type(level) is not int or level not in range(6) or level in seen:
            raise ValueError('Invalid or duplicate preset layer level')
        seen.add(level)
        cls = _layer_types()[level]
        values = _decode(record['fields'])
        if not isinstance(values, dict) or set(values) != {f.name for f in fields(cls)}:
            raise ValueError(f'Invalid fields for {cls.__name__}')
        layer = cls(**values)
        _validate_layer(layer, cls)
        body.set_layer(TowerLevel(level), layer)
    if body.highest_explicit_level() != highest:
        raise ValueError('Header tower_level does not match authored layers')
    return body
