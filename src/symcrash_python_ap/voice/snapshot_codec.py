"""Versioned, data-only checkpoints for VoiceEngine's per-voice time state.

Compiled presets and mutable envelope/LFO state are embedded and detached.
CMMM programs remain explicit external resources, resolved by program ID.
This Python format is separate from the frozen WSES binary container.
"""
from dataclasses import fields, is_dataclass
from enum import IntEnum
from functools import lru_cache
import math
from typing import Any, Union, get_args, get_origin, get_type_hints

import numpy as np


@lru_cache(maxsize=1)
def _types():
    from .voice_engine import (GestureBasis, IntentVector, ArcState, TreeMorphState,
        OverlaySpec, VoicePhase, EngineMode, DegradationLevel, TowerLevel)
    from .control_bindings import VoiceControls
    from .ir_events import CompiledEventRule, CompiledEventProgram, AuthoredEventState
    from .ir_regimes import (CompiledRegimeProgram, AuthoredRegimeState,
        RegimeNodePlan, RegimeEdgePlan, RegimeTemplatePlan, RegimeOverlayPlan)
    from ..frames import CMMMEventPayload, CarrierModel, CarrierRole, TransitionKind
    from ..primitives import Regime
    from ..envelope_scheduler import ADSRSpec, EnvelopeScheduler, EnvelopePhase
    from ..vm.harmonic_bank import LFOState, LFORouting, LFOWaveform, LFOSyncMode, LFOTarget
    from ..vm.state import CMMMemory
    from ..instrument_ir import ADSR, EnvelopeBinding, ControlConfig, TemperamentVector, SafetyConfig
    from ..library.preset_compiler import (CompiledPreset, BackboneOp, FM6Config,
        CarrierConfig, MaterialConfig, LatchDefaults, CmmmDefaults)
    from ..library.native_carrier import NativeCarrierConfig
    from ..library.classic_modulation import ClassicModulation
    from .unison import UnisonMember
    classes = (GestureBasis, IntentVector, ArcState, TreeMorphState, OverlaySpec,
        VoiceControls, CMMMEventPayload, VoicePhase, EngineMode, DegradationLevel,
        TowerLevel, Regime, CarrierModel, CarrierRole, TransitionKind,
        ADSRSpec, EnvelopeScheduler, EnvelopePhase, LFOState, LFORouting,
        LFOWaveform, LFOSyncMode, LFOTarget, ADSR, EnvelopeBinding, ControlConfig,
        TemperamentVector, SafetyConfig, CompiledPreset, BackboneOp, FM6Config,
        CarrierConfig, MaterialConfig, LatchDefaults, CmmmDefaults,
        CompiledEventRule, CompiledEventProgram, AuthoredEventState, CMMMemory,
        CompiledRegimeProgram, AuthoredRegimeState, RegimeNodePlan, RegimeEdgePlan,
        RegimeTemplatePlan, RegimeOverlayPlan, NativeCarrierConfig, UnisonMember, ClassicModulation)
    return {cls.__name__: cls for cls in classes}


def _matches(value, expected):
    if expected is Any:
        return True
    origin, args = get_origin(expected), get_args(expected)
    if origin is Union:
        return any(_matches(value, option) for option in args)
    if origin is list:
        return type(value) is list and all(_matches(v, args[0]) for v in value)
    if origin is dict:
        return type(value) is dict and all(_matches(k, args[0]) and _matches(v, args[1])
                                           for k, v in value.items())
    if origin is tuple:
        if type(value) is not tuple:
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches(v, args[0]) for v in value)
        return len(value) == len(args) and all(_matches(v, t) for v, t in zip(value, args))
    if expected is float:
        return type(value) in (int, float) and math.isfinite(value)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return type(value) is expected


@lru_cache(maxsize=None)
def _hints(cls):
    return get_type_hints(cls, localns=_types())


def _validate_fields(cls, data, *, exclude=()):
    expected = {f.name for f in fields(cls)} - set(exclude)
    if type(data) is not dict or set(data) != expected:
        raise ValueError(f'Incomplete or unknown fields in {cls.__name__} snapshot')
    hints = _hints(cls)
    for name, value in data.items():
        if not _matches(value, hints[name]):
            raise ValueError(f'Invalid snapshot type for {cls.__name__}.{name}')


def _validate_runtime(data):
    """Check dimensions used for indexing and envelope runtime invariants."""
    from .voice_engine import EngineMode, MAX_ARCS
    from ..envelope_scheduler import EnvelopeScheduler, ADSRSpec
    from ..primitives import Q15_ONE
    memory = data['transition_memory']
    if (any(not 0 <= getattr(memory, name) <= Q15_ONE for name in
            ('tension_q15', 'fatigue_q15', 'harmonic_memory_q15'))
            or memory.blocks_since_transition < 0
            or any(getattr(memory, name) < 0 for name in
                   ('TENSION_DECAY', 'FATIGUE_DECAY', 'HARMONIC_DECAY'))):
        raise ValueError('Invalid transition memory state')
    if (data['voice_id'] < 0 or not 0 <= data['channel'] < 16
            or data['engine_mode'] not in (int(EngineMode.SIMPLE), int(EngineMode.FULL_CMMM))
            or len(data['family_weights']) != 8 or len(data['intent'].lanes) != 8
            or len(data['dI_q15']) != 8 or len(data['arc_phase_q16']) != MAX_ARCS):
        raise ValueError('Invalid voice snapshot dimensions or identity')
    if data['engine_mode'] == int(EngineMode.SIMPLE) and data['compiled_preset'] is None:
        raise ValueError('Simple voice snapshot requires its compiled preset')
    runtime = data['authored_regime_state']
    runtime.__post_init__()
    preset = data['compiled_preset']
    if preset is not None and preset.native_carrier is not None:
        preset.native_carrier.validate_mode(preset.osc_mode)
    if data['unison'] is not None:
        data['unison'].__post_init__()
        if preset is None or preset.native_carrier is None or data['unison'].count > preset.native_carrier.unison_voices:
            raise ValueError('Unison member requires its authored carrier configuration')
    events = preset.event_program if preset is not None else None
    if (data['authored_event_state'].regime > 6 and (events is None
            or data['authored_event_state'].regime not in events.regime_templates)):
        raise ValueError('Unknown authored event regime identity')
    graph = preset.regime_program if preset is not None else None
    if graph is not None:
        graph.__post_init__()
        if ((runtime.node and runtime.node not in graph.nodes)
                or runtime.active_edge >= len(graph.edges)
                or any(index >= len(graph.edges) or not graph.edges[index].hysteresis_input
                       for index in runtime.latched_edges)):
            raise ValueError('Invalid authored regime state references')
        if runtime.fade_duration and (runtime.active_edge < 0
                or graph.edges[runtime.active_edge].duration != runtime.fade_duration
                or graph.edges[runtime.active_edge].target != runtime.node):
            raise ValueError('Invalid authored transition fade state')
        operators = {op for template in graph.templates.values() for op in template.ops}
        if any(name not in operators for weights in (runtime.source_weights, runtime.last_weights) for name in weights):
            raise ValueError('Unknown authored transition operator')
    elif runtime.node:
        raise ValueError('Authored regime state requires its compiled graph')
    if any(not 0 <= lfo.arc_slot < MAX_ARCS or lfo.elapsed_samples < 0
           for lfo in data['lfos']):
        raise ValueError('Invalid LFO snapshot slot or clock')
    from .lfo_clocks import lfo_clock_groups, validate_lfo_clock_phases
    validate_lfo_clock_phases(data['arc_phase_q16'])
    lfo_clock_groups(data['lfos'])
    if preset is not None:
        lfo_clock_groups(preset.lfos, initial=True)
    schedulers = [data['amp_envelope']] if data['amp_envelope'] is not None else []
    for entry in data['envelope_matrix'].values():
        if (not 2 <= len(entry) <= 5 or type(entry[0]) is not EnvelopeScheduler
                or type(entry[1]) is not list or any(type(t) is not int for t in entry[1])
                or (len(entry) > 2 and (type(entry[2]) is not int or not 0 <= entry[2] <= Q15_ONE))
                or (len(entry) > 3 and not _matches(entry[3], float))
                or (len(entry) > 4 and type(entry[4]) is not ADSRSpec)):
            raise ValueError('Invalid matrix envelope snapshot')
        schedulers.append(entry[0])
    for scheduler in schedulers:
        if (scheduler.phase_block < 0 or min(scheduler._attack_blocks, scheduler._decay_blocks,
                                             scheduler._release_blocks) < 1
                or not 0 <= scheduler.current_q15 <= Q15_ONE
                or not 0 <= scheduler.release_level_q15 <= Q15_ONE
                or (scheduler._sustain_override_q15 is not None
                    and not 0 <= scheduler._sustain_override_q15 <= Q15_ONE)):
            raise ValueError('Invalid envelope runtime snapshot')


def encode(value, _depth=0):
    """Encode only registered data, retaining enums, tuples and typed map keys."""
    if _depth > 64:
        raise ValueError('Snapshot nesting is too deep')
    if isinstance(value, IntEnum):
        if _types().get(type(value).__name__) is not type(value):
            # Integer-valued IDs need not carry their incidental enum type.
            return int(value)
        return {'kind': type(value).__name__, 'value': int(value)}
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, np.generic):
        return encode(value.item(), _depth + 1)
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        items = [encode(v, _depth + 1) for v in value]
        return items if type(value) is list else {'kind': 'tuple', 'items': items}
    if type(value) is dict:
        return {'kind': 'dict', 'items': [[encode(k, _depth + 1), encode(v, _depth + 1)]
                                         for k, v in value.items()]}
    if is_dataclass(value) and _types().get(type(value).__name__) is type(value):
        data = {f.name: getattr(value, f.name) for f in fields(value)}
        if set(vars(value)) - set(data):
            raise ValueError(f'Unregistered runtime fields in {type(value).__name__}')
        _validate_fields(type(value), data)
        return {'kind': type(value).__name__,
                'fields': {k: encode(v, _depth + 1) for k, v in data.items()}}
    raise ValueError(f'Unsupported snapshot value: {type(value).__name__}')


def decode(value, _depth=0):
    if _depth > 64:
        raise ValueError('Snapshot nesting is too deep')
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [decode(v, _depth + 1) for v in value]
    if type(value) is not dict or not isinstance(value.get('kind'), str):
        raise ValueError('Malformed snapshot value')
    kind = value['kind']
    if kind in ('tuple', 'dict') and set(value) == {'kind', 'items'} and type(value['items']) is list:
        if kind == 'tuple':
            return tuple(decode(v, _depth + 1) for v in value['items'])
        result = {}
        for pair in value['items']:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError('Malformed snapshot map entry')
            key, item = (decode(v, _depth + 1) for v in pair)
            if type(key) not in (int, str) and not isinstance(key, IntEnum):
                raise ValueError('Invalid snapshot map key')
            if key in result:
                raise ValueError('Duplicate snapshot map key')
            result[key] = item
        return result
    cls = _types().get(kind)
    if cls is None:
        raise ValueError(f'Unknown snapshot kind: {kind}')
    if issubclass(cls, IntEnum):
        if set(value) != {'kind', 'value'} or type(value['value']) is not int:
            raise ValueError('Invalid snapshot enum')
        return cls(value['value'])
    if set(value) != {'kind', 'fields'} or type(value['fields']) is not dict:
        raise ValueError('Invalid snapshot object')
    data = {k: decode(v, _depth + 1) for k, v in value['fields'].items()}
    if kind == 'NativeCarrierConfig':
        return cls.from_dict(data)
    _validate_fields(cls, data)
    result = cls(**{f.name: data[f.name] for f in fields(cls) if f.init})
    # Constructors may derive caches. Checkpoints retain the saved cache/phase,
    # including envelope times changed by live controls after construction.
    for name, item in data.items():
        object.__setattr__(result, name, item)
    return result


def snapshot_voice(state):
    from .voice_engine import VoiceState
    names = {f.name for f in fields(VoiceState)}
    if set(vars(state)) - names - {'program'}:
        raise ValueError('Unregistered runtime fields in VoiceState')
    data = {name: getattr(state, name) for name in sorted(names)}
    _validate_fields(VoiceState, data)
    _validate_runtime(data)
    return {'snapshot_version': 7, **{k: encode(v) for k, v in data.items()}}


def restore_voice(cls, snapshot, programs):
    from .voice_engine import (VoiceState, VoiceStateWithProgram,
                              INSTRUMENT_PROGRAMS, InstrumentProgram)
    if type(snapshot) is not dict or type(snapshot.get('snapshot_version')) is not int or snapshot['snapshot_version'] not in (1, 2, 3, 4, 5, 6, 7):
        raise ValueError('Unsupported voice snapshot version')
    if snapshot['snapshot_version'] == 1:
        # v1 predates authored event programs and their per-voice time state.
        # Migrate detached data; current-version snapshots require every field.
        from .ir_events import AuthoredEventState
        def migrate(value, depth=0):
            if depth > 64:
                raise ValueError('Snapshot nesting is too deep')
            if type(value) is list:
                return [migrate(x, depth + 1) for x in value]
            if type(value) is dict:
                result = {k: migrate(v, depth + 1) for k, v in value.items()}
                if result.get('kind') == 'CompiledPreset' and type(result.get('fields')) is dict:
                    result['fields'].setdefault('event_program', None)
                return result
            return value
        snapshot = migrate(snapshot)
        snapshot.setdefault('authored_event_state', encode(AuthoredEventState()))
    if snapshot['snapshot_version'] < 3:
        from ..vm.state import CMMMemory
        snapshot = dict(snapshot)
        snapshot.setdefault('transition_memory', encode(CMMMemory()))
    if snapshot['snapshot_version'] < 4:
        from .ir_regimes import AuthoredRegimeState
        def migrate_graph(value, depth=0):
            if depth > 64:
                raise ValueError('Snapshot nesting is too deep')
            if type(value) is list:
                return [migrate_graph(x, depth + 1) for x in value]
            if type(value) is dict:
                result = {k: migrate_graph(v, depth + 1) for k, v in value.items()}
                if result.get('kind') == 'CompiledPreset' and type(result.get('fields')) is dict:
                    result['fields'].setdefault('regime_program', None)
                return result
            return value
        snapshot = migrate_graph(snapshot)
        snapshot.setdefault('authored_regime_state', encode(AuthoredRegimeState()))
    if snapshot['snapshot_version'] < 5:
        def migrate_carrier(value, depth=0):
            if depth > 64:
                raise ValueError('Snapshot nesting is too deep')
            if type(value) is list:
                return [migrate_carrier(x, depth + 1) for x in value]
            if type(value) is dict:
                result = {k: migrate_carrier(v, depth + 1) for k, v in value.items()}
                if result.get('kind') == 'CompiledPreset' and type(result.get('fields')) is dict:
                    result['fields'].setdefault('native_carrier', None)
                return result
            return value
        snapshot = migrate_carrier(snapshot)
    if snapshot['snapshot_version'] < 6:
        snapshot = dict(snapshot)
        snapshot.setdefault('unison', None)
    if snapshot['snapshot_version'] < 7:
        def migrate_filter(value, depth=0):
            if depth > 64:
                raise ValueError('Snapshot nesting is too deep')
            if type(value) is list:
                return [migrate_filter(x, depth + 1) for x in value]
            if type(value) is dict:
                result = {k: migrate_filter(v, depth + 1) for k, v in value.items()}
                if result.get('kind') == 'CompiledPreset' and type(result.get('fields')) is dict:
                    result['fields'].setdefault('filter_enabled', None)
                return result
            return value
        snapshot = migrate_filter(snapshot)
    data = {k: decode(v) for k, v in snapshot.items() if k != 'snapshot_version'}
    _validate_fields(VoiceState, data)
    _validate_runtime(data)
    state = cls(**data)
    if isinstance(state, VoiceStateWithProgram):
        registry = INSTRUMENT_PROGRAMS if programs is None else programs
        if state.program_id in registry:
            state.program = registry[state.program_id]
        elif state.program_id == 0:
            state.program = InstrumentProgram()
        else:
            raise ValueError(f'Program {state.program_id} is required to restore this voice')
    return state
