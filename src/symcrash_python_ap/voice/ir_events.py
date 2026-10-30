"""Compile InstrumentIR event expressions into bounded integer wire commands.

Expressions are parsed once through a whitelist, never Python eval. Commands
use the frozen v0b lane meanings, independently of semantic CMMM event lanes.
"""
import ast
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Dict, List

from ..frames import WireFrameData, VoiceFramePacketV0b
from ..primitives import Q15_ONE, Q15_HALF, Q15_NEG_ONE, mul_q15_sat, sat16
from ..wire_frame import decode_wire_events


INTENTS = ('tone', 'bow', 'air', 'glitch', 'warp', 'snap', 'glue', 'calm')
GESTURES = ('pressure_q15', 'speed_q15', 'position_q15', 'tilt_q15', 'chaos_q15',
            'pitch_q15', 'velocity_q15', 'pan_q15', 'gate', 'gate_rise', 'gate_fall')
FLAGS = ('gate', 'gate_rise', 'gate_fall', 'regime_enter', 'regime_exit', 'dwell_done')
VARIABLES = frozenset((*FLAGS, 'block_index', 'age', 'energy_intent', 'energy_state',
    'risk', *(f'gesture.{n}' for n in GESTURES), *(f'intent.{n}_q15' for n in INTENTS)))
LANES = {'EVT_OP_PULSE': 0, 'EVT_OP_TOGGLE': 1, 'EVT_TREE_MORPH': 2,
         'EVT_K_MODE': 3, 'EVT_INJECT_NOTE': 4, 'EVT_MATERIAL_HIT': 5,
         'EVT_CLOUD_PUSH': 6, 'EVT_RESERVED': 7}
TARGETS = {'global': 0, 'voice_id': 1, 'scale_id': 2, 'band_id': 3,
           'band': 3, 'node': 3, 'pattern_id': 5}
_ALIASES = {'Gate rises': 'gate_rise', 'Gate falls': 'gate_fall'}
_BIN = {ast.Add: 'add', ast.Sub: 'sub', ast.LShift: 'shl', ast.RShift: 'shr',
        ast.BitAnd: 'band', ast.BitOr: 'bor', ast.Mod: 'mod'}
_CMP = {ast.Eq: 'eq', ast.NotEq: 'ne', ast.Lt: 'lt', ast.LtE: 'le', ast.Gt: 'gt', ast.GtE: 'ge'}
_FUNCTIONS = {'mul_q15': (2, 2), 'mul_q15_sat': (2, 2), 'sat16': (1, 1),
              'abs': (1, 1), 'min': (2, 8), 'max': (2, 8), 'clamp': (3, 3)}


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer in {low}..{high}')
    return value


@lru_cache(maxsize=None)
def _expression(source):
    if type(source) is not str or not source.strip() or len(source) > 2048:
        raise ValueError('Event expression must be nonempty and at most 2048 characters')
    try:
        tree = ast.parse(_ALIASES.get(source, source), mode='eval')
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f'Invalid event expression: {source!r}') from exc
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError('Event expression exceeds 128 syntax nodes')

    def visit(node, depth=0):
        if depth > 24:
            raise ValueError('Event expression is too deep')
        child = lambda value: visit(value, depth + 1)
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) is bool:
                return ('constant', int(value))
            if type(value) is float and -1.0 <= value <= 1.0:
                value = round(value * Q15_ONE)
            return ('constant', _integer(value, -(1 << 31), (1 << 31) - 1, 'Expression literal'))
        if isinstance(node, ast.Name):
            constants = {'always': 1, 'never': 0, 'true': 1, 'false': 0,
                         'Q15_ONE': Q15_ONE, 'Q15_HALF': Q15_HALF}
            if node.id in constants:
                return ('constant', constants[node.id])
            if node.id in VARIABLES:
                return ('variable', node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            name = node.value.id + '.' + node.attr
            if name in VARIABLES or node.value.id == 'macro' and node.attr.startswith('M_'):
                return ('variable', name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name == 'macro' and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                if node.args[0].id.startswith('M_'):
                    return ('variable', 'macro.' + node.args[0].id)
            if name in _FUNCTIONS and _FUNCTIONS[name][0] <= len(node.args) <= _FUNCTIONS[name][1]:
                return (name, *(child(arg) for arg in node.args))
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.Not, ast.USub, ast.UAdd):
            return ({ast.Not: 'not', ast.USub: 'neg', ast.UAdd: 'pos'}[type(node.op)], child(node.operand))
        if isinstance(node, ast.BoolOp):
            return ('and' if isinstance(node.op, ast.And) else 'or', *(child(v) for v in node.values))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            if isinstance(node.op, (ast.LShift, ast.RShift, ast.Mod)):
                right = child(node.right)
                low, high = (0, 31) if not isinstance(node.op, ast.Mod) else (1, (1 << 31) - 1)
                if right[0] != 'constant' or not low <= right[1] <= high:
                    raise ValueError('Event shifts/modulo require a bounded constant operand')
            return (_BIN[type(node.op)], child(node.left), child(node.right))
        if isinstance(node, ast.Compare) and all(type(op) in _CMP for op in node.ops):
            values = [node.left, *node.comparators]
            return ('and', *((_CMP[type(op)], child(values[i]), child(values[i + 1]))
                             for i, op in enumerate(node.ops)))
        raise ValueError(f'Unsupported event expression syntax: {ast.dump(node, include_attributes=False)}')
    return visit(tree.body)


def _variables(code):
    if code[0] == 'variable':
        return {code[1]}
    return set().union(*(_variables(x) for x in code[1:] if isinstance(x, tuple)))


def evaluate(source, context):
    """Evaluate compiled integer instructions. Missing inputs fail explicitly."""
    def run(code):
        op = code[0]
        if op == 'constant':
            return code[1]
        if op == 'variable':
            if code[1] not in context:
                raise ValueError(f'Missing event input: {code[1]}')
            value = context[code[1]]
            if type(value) not in (int, bool):
                raise ValueError(f'Event input must be integer: {code[1]}')
            return int(value)
        if op == 'and':
            return int(all(run(x) for x in code[1:]))
        if op == 'or':
            return int(any(run(x) for x in code[1:]))
        args = [run(x) for x in code[1:]]
        a = args[0]
        if op in ('mul_q15', 'mul_q15_sat'):
            return mul_q15_sat(a, args[1])
        if op == 'sat16': return sat16(a)
        if op == 'abs': return abs(a)
        if op == 'min': return min(args)
        if op == 'max': return max(args)
        if op == 'clamp': return max(args[1], min(args[2], a))
        if op == 'not': return int(not a)
        if op == 'neg': return -a
        if op == 'pos': return a
        b = args[1]
        if op == 'add': return a + b
        if op == 'sub': return a - b
        if op == 'shl': return a << b
        if op == 'shr': return a >> b
        if op == 'band': return a & b
        if op == 'bor': return a | b
        if op == 'mod': return a % b
        if op == 'eq': return int(a == b)
        if op == 'ne': return int(a != b)
        if op == 'lt': return int(a < b)
        if op == 'le': return int(a <= b)
        if op == 'gt': return int(a > b)
        if op == 'ge': return int(a >= b)
        raise AssertionError(op)
    return run(_expression(source))


@dataclass(frozen=True)
class CompiledEventRule:
    lane: int
    condition: str
    strength: str
    payload: int

    def __post_init__(self):
        _integer(self.lane, 0, 6, 'Event lane')
        _integer(self.payload, 0, (1 << 64) - 1, 'Event payload')
        _expression(self.condition)
        _expression(self.strength)
        values, payloads = [0] * 8, [0] * 8
        values[self.lane], payloads[self.lane] = Q15_ONE, self.payload
        wire = WireFrameData(event_mask=1 << self.lane, event_strength_q15=tuple(values),
                             event_payload_u64=tuple(payloads))
        events = decode_wire_events(wire)
        if len(events) != 1:
            raise ValueError('Unsupported or malformed event payload for its frozen wire lane')
        event = events[0]
        if event.target_kind != 5 and event.pattern_id:
            raise ValueError('Pattern ID requires a pattern target')
        if event.target_kind in (0, 5) and event.lane != 1 and event.target_id:
            raise ValueError('This event target does not consume a target ID')
        if event.lane == 6:
            raise ValueError('Cloud event execution is not implemented by the current output owner')
        if event.target_kind == 5:
            from ..tables.pattern_table import PATTERN_TABLE
            if event.pattern_id not in PATTERN_TABLE:
                raise ValueError('Event target pattern is not registered')
        if event.lane in (0, 4) and (event.param1 or event.param2):
            raise ValueError('Injection duration/jitter requires a future authored event profile')
        if event.lane == 0 and event.variant != 2 and event.param0:
            raise ValueError('Pulse shape parameter is not implemented by the current consumer')
        if event.lane == 4 and event.variant == 0 and event.param0:
            raise ValueError('Harmonic injection pitch selector is not implemented by the current consumer')
        if event.lane in (2, 3) and event.param1:
            raise ValueError('Tree slew/oversampling hints are not implemented by the current consumer')


@dataclass
class CompiledEventProgram:
    templates: Dict[str, List[CompiledEventRule]] = field(default_factory=dict)
    overlays: List[CompiledEventRule] = field(default_factory=list)
    macros: Dict[str, int] = field(default_factory=dict)
    regime_templates: Dict[int, str] = field(default_factory=dict)
    regime_dwell_blocks: Dict[int, int] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError('Unsupported authored event program version')
        if (type(self.templates) is not dict or any(type(name) is not str or type(items) is not list
                for name, items in self.templates.items()) or type(self.overlays) is not list
                or type(self.macros) is not dict or type(self.regime_templates) is not dict
                or type(self.regime_dwell_blocks) is not dict):
            raise ValueError('Invalid authored event program structure')
        for key, name in self.regime_templates.items():
            _integer(key, 0, 65535, 'Authored regime ID')
            if name not in self.templates:
                raise ValueError('Authored event regime references an undefined template')
        for key, value in self.regime_dwell_blocks.items():
            _integer(key, 0, 65535, 'Authored regime ID')
            _integer(value, 0, (1 << 31) - 1, 'Regime dwell')
        rules = [r for items in self.templates.values() for r in items] + self.overlays
        if len(rules) > 256 or any(type(r) is not CompiledEventRule for r in rules):
            raise ValueError('Authored event program must contain at most 256 compiled rules')
        for name, value in self.macros.items():
            if not isinstance(name, str) or not name.startswith('M_'):
                raise ValueError('Invalid authored macro identifier')
            _integer(value, Q15_NEG_ONE, Q15_ONE, 'Macro default')
        for rule in rules:
            for name in _variables(_expression(rule.condition)) | _variables(_expression(rule.strength)):
                if name.startswith('macro.') and name[6:] not in self.macros:
                    raise ValueError(f'Undeclared event macro: {name[6:]}')

    def emit(self, template, context, *, sequence=0, allowed_mask=0xff, departing=False):
        """Strongest candidate per wire lane; ties keep earlier template order."""
        inputs = {'macro.' + name: value for name, value in self.macros.items()}
        inputs.update(context)
        strengths, payloads, mask = [0] * 8, [0] * 8, 0
        candidates = [*self.templates.get(template, ()), *self.overlays]
        if departing:
            candidates = [rule for rule in candidates
                          if 'regime_exit' in _variables(_expression(rule.condition))]
        # Ascending lane order, then authored template/overlay order.
        for lane in range(7):
            if not allowed_mask & (1 << lane):
                continue
            for rule in candidates:
                if rule.lane != lane or not evaluate(rule.condition, inputs):
                    continue
                strength = max(0, min(Q15_ONE, evaluate(rule.strength, inputs)))
                if not mask & (1 << lane) or strength > strengths[lane]:
                    mask |= 1 << lane
                    strengths[lane], payloads[lane] = strength, rule.payload
        return WireFrameData(frame_seq=sequence & 0xffffffff, event_mask=mask,
            event_strength_q15=tuple(strengths), event_payload_u64=tuple(payloads))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if type(data) is not dict or set(data) != {'templates', 'overlays', 'macros', 'regime_templates', 'regime_dwell_blocks', 'version'}:
            raise ValueError('Incomplete authored event program')
        try:
            def ids(mapping):
                result = {}
                for key, value in mapping.items():
                    if type(key) not in (int, str) or str(int(key)) != str(key) or int(key) in result:
                        raise ValueError('Invalid or duplicate event regime key')
                    result[int(key)] = value
                return result
            return cls(templates={name: [CompiledEventRule(**r) for r in rules]
                                  for name, rules in data['templates'].items()},
                       overlays=[CompiledEventRule(**r) for r in data['overlays']],
                       macros=dict(data['macros']),
                       regime_templates=ids(data['regime_templates']),
                       regime_dwell_blocks=ids(data['regime_dwell_blocks']),
                       version=data['version'])
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError('Malformed authored event program') from exc


@dataclass
class AuthoredEventState:
    template: str = ''
    template_blocks: int = 0
    previous_gate: bool = False
    rise_pending: bool = False
    fall_pending: bool = False
    sequence: int = 0
    regime: int = -1
    note_blocks: int = 0
    macros: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.template) is not str or any(type(value) is not bool for value in
                (self.previous_gate, self.rise_pending, self.fall_pending)):
            raise ValueError('Invalid authored event state')
        _integer(self.template_blocks, 0, (1 << 63) - 1, 'Template clock')
        _integer(self.sequence, 0, 0xffffffff, 'Event sequence')
        _integer(self.regime, -1, 65535, 'Event regime')
        _integer(self.note_blocks, 0, (1 << 63) - 1, 'Note clock')
        if type(self.macros) is not dict:
            raise ValueError('Invalid event macro state')
        for name, value in self.macros.items():
            if type(name) is not str or not name.startswith('M_'):
                raise ValueError('Invalid event macro name')
            _integer(value, Q15_NEG_ONE, Q15_ONE, 'Event macro value')

    def set_macro(self, program, name, value):
        if name not in program.macros:
            raise ValueError(f'Undeclared event macro: {name}')
        self.macros[name] = _integer(value, Q15_NEG_ONE, Q15_ONE, 'Event macro value')

    def emit(self, program, template, context, *, gate, allowed_mask=0xff, regime=-1, entered=False):
        rise = self.rise_pending or gate and not self.previous_gate
        fall = self.fall_pending or not gate and self.previous_gate
        changed = template != self.template or regime != self.regime or rise or entered
        elapsed = 0 if changed else self.template_blocks + 1
        note_blocks = 0 if rise else self.note_blocks
        inputs = dict(context, gate=int(gate), gate_rise=int(rise), gate_fall=int(fall),
                      regime_enter=int(changed), regime_exit=0,
                      age=note_blocks * 4 // 3,
                      dwell_done=int(elapsed >= program.regime_dwell_blocks.get(regime, 0)))
        inputs.update({'gesture.gate': int(gate), 'gesture.gate_rise': int(rise),
                       'gesture.gate_fall': int(fall)})
        inputs.update({'macro.' + name: value for name, value in self.macros.items()})
        # Exit events belong to the departing template and merge ahead of entry.
        output = program.emit(template, inputs, sequence=self.sequence, allowed_mask=allowed_mask)
        if changed and self.template:
            exit_inputs = dict(inputs, regime_exit=1, regime_enter=0,
                age=self.note_blocks * 4 // 3,
                dwell_done=int(self.template_blocks + 1 >=
                               program.regime_dwell_blocks.get(self.regime, 0)))
            # Only explicit exit predicates are eligible in the departing template.
            outgoing = program.emit(self.template, exit_inputs, sequence=self.sequence,
                                    allowed_mask=allowed_mask, departing=True)
            strengths, payloads = list(output.event_strength_q15), list(output.event_payload_u64)
            mask = output.event_mask | outgoing.event_mask
            for lane in range(8):
                if outgoing.event_mask & (1 << lane) and (not output.event_mask & (1 << lane)
                        or outgoing.event_strength_q15[lane] >= strengths[lane]):
                    strengths[lane], payloads[lane] = outgoing.event_strength_q15[lane], outgoing.event_payload_u64[lane]
            output = WireFrameData(frame_seq=self.sequence, event_mask=mask,
                event_strength_q15=tuple(strengths), event_payload_u64=tuple(payloads))
        self.template_blocks = elapsed
        self.template, self.previous_gate = template, gate
        self.regime = regime
        self.rise_pending = self.fall_pending = False
        self.sequence = (self.sequence + 1) & 0xffffffff
        self.note_blocks = note_blocks + 1
        return output


def _rule(event, guard='always'):
    lane = LANES.get(event.lane, event.lane)
    _integer(lane, 0, 6, 'Frozen wire event lane')
    payload = event.payload_template
    if any(getattr(payload, key) is not None for key in
           ('wave_type', 'partials_N', 'pull_q8', 'carrier_model', 'carrier_role', 'carrier_archetype')):
        raise ValueError('Carrier configuration requires the v1 carrier header; it is not a v0b event payload')
    variant = payload.variant
    if type(variant) is str:
        aliases = {0: {'inject': 0, 'impulse': 0, 'harmonic': 1, 'noise': 2},
                   1: {'toggle': 0, 'enable': 1, 'disable': 2},
                   2: {'start': 0, 'continue': 1, 'hold': 2, 'snap_a': 3, 'snap_b': 4},
                   3: {'set': 0}, 4: {'harmonic': 0, 'inharmonic': 1, 'noise': 2},
                   5: {'drive': 0, 'bias': 1, 'combined': 2}, 6: {'add': 0, 'decay': 1, 'reset': 2}}
        variant = aliases.get(lane, {}).get(variant.lower(), variant)
    target = TARGETS.get(payload.target_kind, payload.target_kind) if payload.target_kind is not None else 0
    target_id = payload.target_id
    if payload.op is not None:
        op = str(payload.op).lower().replace('_', '')
        if lane == 0:
            variants = {'impulse': 0, 'globalpulse': 0, 'harmonicimpulse': 1, 'noiseband': 2}
            if op not in variants:
                raise ValueError(f'Operator {payload.op!r} has no frozen OP_PULSE variant')
            if variant not in (0, variants[op]):
                raise ValueError('Event operator and variant disagree')
            variant = variants[op]
        elif lane == 1:
            from ..primitives import OPERATOR_TABLE
            names = {entry.name.lower().replace('_', ''): key for key, entry in OPERATOR_TABLE.items()}
            target_id = names.get(op, payload.op)
        else:
            raise ValueError('An operator selector is supported only on OP_PULSE/OP_TOGGLE')
    for label, value, maximum in [('variant', variant, 15), ('target_kind', target, 15),
            ('target_id', target_id, 255), ('pattern_id', payload.pattern_id, 255),
            ('p0', payload.p0, 255), ('p1', payload.p1, 255), ('p2', payload.p2, 255)]:
        _integer(value, 0, maximum, label)
    packed = VoiceFramePacketV0b.pack_event_payload(lane_id=lane, variant=variant,
        target_kind=target, target_id=target_id, pattern_id=payload.pattern_id,
        param0_q8=payload.p0, param1_q8=payload.p1, param2_q8=payload.p2)
    guard = _ALIASES.get(guard, guard)
    condition = _ALIASES.get(event.condition, event.condition)
    return CompiledEventRule(lane, f'({guard}) and ({condition})', str(event.strength), packed)


def compile_ir_events(ir):
    """Return an immutable-rule program, or None when the IR authors no events."""
    l3 = ir.layers.L3_cmmm
    if not any(t.events for t in l3.emit_templates.values()) and not any(o.add.events for o in l3.overlays):
        return None
    emitter = ir.compile.frame_emitter_policy
    if emitter.voiceframe_version != 1 or emitter.lane_limit != 8:
        raise ValueError('Authored events currently require the eight-lane v0b packet profile (version 1)')
    if emitter.events.choose_order != 'lane_id_ascending_then_template_order':
        raise ValueError('Unsupported authored event selection order')
    if emitter.events.merge_policy != 'at most one payload per lane per block; strongest wins':
        raise ValueError('Unsupported authored event merge policy')
    for name, value in emitter.lanes.items():
        if name not in LANES or LANES[name] != value:
            raise ValueError('Authored lane assignments must match the frozen wire profile')
    templates = {name: [_rule(event) for event in template.events] for name, template in l3.emit_templates.items()}
    overlays = []
    for overlay in l3.overlays:
        if overlay.add.events:
            if overlay.suppression_rules:
                raise ValueError('Authored event suppression rules require an explicit supported event selector')
            guard = 'always' if overlay.kind == 'always' else overlay.enable_when
            overlays.extend(_rule(event, guard) for event in overlay.add.events)
    macros = {}
    for macro in ir.layers.L0_graph.macros:
        value = macro.default
        if type(value) not in (int, float) or not -1 <= value <= 1:
            raise ValueError('Authored event macro defaults must be normalized in -1..1')
        macros[macro.id] = round(value * Q15_ONE)
    regimes, dwell = {}, {}
    from ..primitives import Regime, ms_to_blocks
    from .ir_regimes import node_ids
    ids = node_ids(ir)
    for node in ir.layers.L2_regimes.regime_nodes:
        regimes[ids[node.id]] = node.emit_template
        if node.dwell_ms is not None:
            if type(node.dwell_ms) not in (int, float) or node.dwell_ms < 0:
                raise ValueError('Authored regime dwell must be nonnegative milliseconds')
            dwell[ids[node.id]] = ms_to_blocks(node.dwell_ms) if node.dwell_ms else 0
    return CompiledEventProgram(templates, overlays, macros, regimes, dwell)


def event_context(*, gesture=None, intent=None, velocity=0, gain=0, block=0, risk=0, dwell_done=False):
    """Shared input units for live, CMMM and offline event evaluation."""
    result = {f'gesture.{name}': int(getattr(gesture, name, 0)) for name in GESTURES}
    result['gesture.velocity_q15'] = int(velocity)
    for i, name in enumerate(INTENTS):
        result[f'intent.{name}_q15'] = int(intent.lanes[i]) if intent is not None else 0
    result.update(block_index=int(block), age=int(block) * 4 // 3,
                  energy_intent=int(velocity), energy_state=int(gain), risk=int(risk),
                  dwell_done=int(dwell_done))
    return result


def phase_template(program, phase):
    regime = phase_regime(phase)
    default = {0: 'T_IDLE', 1: 'T_ATTACK', 2: 'T_STABLE', 6: 'T_RECOVERY'}[regime]
    return program.regime_templates.get(regime, default)


def phase_regime(phase):
    return {'ATTACK': 1, 'DECAY': 2, 'SUSTAIN': 2, 'RELEASE': 6, 'FREE': 0}.get(phase, 2)
