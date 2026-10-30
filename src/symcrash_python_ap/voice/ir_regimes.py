"""Shared authored L2 graph execution for native preset and offline voices.

Node names are identities, not numeric CMMM categories. All clocks and runtime
weights are integers; expressions use the same bounded interpreter as events.
"""
from dataclasses import asdict, dataclass, field, fields
import ast
import math
from typing import Dict, List, Optional

from ..primitives import Q15_ONE, Regime, ms_to_blocks
from .ir_events import _expression, _variables, evaluate, _integer


def node_ids(ir):
    names = [node.id for node in ir.layers.L2_regimes.regime_nodes]
    if any(type(n) is not str or not n or n == 'any' for n in names) or len(set(names)) != len(names):
        raise ValueError('Regime node names must be nonempty, unique and not "any"')
    # Preserve old event IDs; distinct authored names never collapse by kind.
    result = {n: int(Regime[n]) for n in names if n in Regime.__members__}
    result.update({n: i + 7 for i, n in enumerate(sorted(set(names) - result.keys()))})
    if len(names) > 256:
        raise ValueError('At most 256 authored regime nodes are supported')
    return result


def expression(source):
    aliases = {'Gate rises': 'gate_rise', 'Gate falls': 'gate_fall',
               'energy': 'energy_state', 'pressure': 'gesture.pressure_q15',
               'chaos': 'gesture.chaos_q15'}
    if type(source) is not str:
        raise ValueError('Regime expressions must be strings')
    source = aliases.get(source.strip(), source.strip()) or 'always'
    # Token replacement avoids changing attribute names or macro identifiers.
    import re
    source = re.sub(r'(?<![\w.])(energy|pressure|chaos)\b', lambda m: aliases[m[0]], source)
    _expression(source)
    return source


def weight(value):
    if type(value) is str:
        return expression(value)
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Authored weights must be normalized in 0..1 or integer expressions')
    return str(round(value * Q15_ONE))


def tree_id(name):
    from ..primitives import TreeArchetype
    aliases = {'texture': 'NOISY', 'attack': 'TRANSIENT'}
    key = aliases.get(name, str(name).upper())
    if key not in TreeArchetype.__members__:
        raise ValueError(f'Unknown authored tree: {name}')
    return int(TreeArchetype[key])


@dataclass(frozen=True)
class RegimeNodePlan:
    name: str
    kind: str
    event_id: int
    template: str
    dwell_blocks: int = 0


@dataclass(frozen=True)
class RegimeEdgePlan:
    source: str
    target: str
    guard: str
    min_dwell: int = 0
    duration: int = 0
    overlay: str = ''
    hysteresis_input: str = ''
    enter: int = 0
    exit: int = 0
    high: bool = True


@dataclass(frozen=True)
class RegimeTemplatePlan:
    ops: List[str]
    families: Dict[str, str]
    tree_a: int = 0
    tree_b: int = 0
    blend: str = '0'
    mask: Optional[int] = None
    mask_ramp: int = 5
    tail: Optional[int] = None
    traction: int = Q15_ONE


@dataclass(frozen=True)
class RegimeOverlayPlan:
    guard: str
    families: Dict[str, str]


@dataclass
class CompiledRegimeProgram:
    nodes: Dict[str, RegimeNodePlan]
    edges: List[RegimeEdgePlan]
    templates: Dict[str, RegimeTemplatePlan]
    overlays: Dict[str, RegimeOverlayPlan]
    macros: Dict[str, int]
    start: str
    version: int = 1

    def __post_init__(self):
        from ..vm.decode import _op_name_to_id, _FAMILY_NAME_SLOT
        if (any(type(value) is not dict for value in (self.nodes, self.templates, self.overlays, self.macros))
                or type(self.edges) is not list or type(self.start) is not str
                or any(type(k) is not str for mapping in (self.nodes, self.templates, self.overlays) for k in mapping)):
            raise ValueError('Invalid compiled regime program structure')
        if type(self.version) is not int or self.version != 1 or self.start not in self.nodes:
            raise ValueError('Invalid authored regime program version or start node')
        if not 1 <= len(self.nodes) <= 256 or len(self.edges) > 1024:
            raise ValueError('Authored regime program exceeds graph bounds')
        ids = set()
        for name, node in self.nodes.items():
            if type(node) is not RegimeNodePlan or node.name != name or not name or name == 'any' or type(node.kind) is not str:
                raise ValueError('Invalid compiled regime node')
            _integer(node.event_id, 0, 65535, 'Regime event identity')
            _integer(node.dwell_blocks, 0, (1 << 31) - 1, 'Regime dwell')
            if node.template not in self.templates or node.event_id in ids:
                raise ValueError('Missing regime template or duplicate event identity')
            ids.add(node.event_id)
        expressions = []
        for template in self.templates.values():
            if type(template) is not RegimeTemplatePlan:
                raise ValueError('Invalid compiled regime template')
            if type(template.ops) is not list or any(type(op) is not str for op in template.ops):
                raise ValueError('Invalid compiled operator list')
            op_ids = [_op_name_to_id(op) for op in template.ops]
            if None in op_ids or len(set(op_ids)) != len(op_ids):
                raise ValueError('Unknown or duplicate template operator')
            _integer(template.tree_a, 0, 4, 'Tree A')
            _integer(template.tree_b, 0, 4, 'Tree B')
            if template.mask is not None:
                _integer(template.mask, 0, 0xffffffff, 'Node mask')
            if template.tail is not None:
                _integer(template.tail, 0, 3, 'Tail mode')
            _integer(template.mask_ramp, 0, (1 << 31) - 1, 'Mask ramp')
            _integer(template.traction, 0, Q15_ONE, 'Traction')
            expressions.append(template.blend)
        for item in [*self.templates.values(), *self.overlays.values()]:
            if type(item) not in (RegimeTemplatePlan, RegimeOverlayPlan) or type(item.families) is not dict:
                raise ValueError('Invalid compiled family map')
            direct_ids = set()
            for key, value in item.families.items():
                if type(key) is not str or key not in _FAMILY_NAME_SLOT and _op_name_to_id(key) is None:
                    raise ValueError(f'Unknown authored family/operator: {key}')
                op_id = _op_name_to_id(key)
                if op_id is not None:
                    if op_id in direct_ids:
                        raise ValueError('Duplicate aliases for a per-operator strength')
                    direct_ids.add(op_id)
                expressions.append(value)
        for overlay in self.overlays.values():
            if type(overlay) is not RegimeOverlayPlan:
                raise ValueError('Invalid compiled regime overlay')
            expressions.append(overlay.guard)
        for edge in self.edges:
            if (type(edge) is not RegimeEdgePlan or edge.source not in (*self.nodes, 'any')
                    or edge.target not in self.nodes or edge.overlay and edge.overlay not in self.overlays):
                raise ValueError('Invalid regime edge endpoint or overlay')
            _integer(edge.min_dwell, 0, (1 << 31) - 1, 'Minimum dwell')
            _integer(edge.duration, 0, (1 << 31) - 1, 'Transition duration')
            expressions.append(edge.guard)
            if edge.hysteresis_input:
                if _expression(edge.hysteresis_input)[0] != 'variable' or type(edge.high) is not bool:
                    raise ValueError('Hysteresis requires a scalar input')
                _integer(edge.enter, 0, Q15_ONE, 'Hysteresis enter')
                _integer(edge.exit, 0, Q15_ONE, 'Hysteresis exit')
                if (edge.high and edge.exit > edge.enter) or (not edge.high and edge.exit < edge.enter):
                    raise ValueError('Hysteresis thresholds are reversed')
                expressions.append(edge.hysteresis_input)
        for name, value in self.macros.items():
            if type(name) is not str or not name.startswith('M_'):
                raise ValueError('Invalid regime macro name')
            _integer(value, -Q15_ONE, Q15_ONE, 'Regime macro default')
        for source in expressions:
            for variable in _variables(_expression(source)):
                if variable.startswith('macro.') and variable[6:] not in self.macros:
                    raise ValueError(f'Undeclared regime macro: {variable}')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if type(data) is not dict or set(data) != {'nodes', 'edges', 'templates', 'overlays', 'macros', 'start', 'version'}:
            raise ValueError('Invalid compiled regime program structure')
        try:
            def record(kind, values):
                if type(values) is not dict or set(values) != {f.name for f in fields(kind)}:
                    raise ValueError(f'Incomplete or unknown {kind.__name__} fields')
                return kind(**values)
            return cls(nodes={k: record(RegimeNodePlan, v) for k, v in data['nodes'].items()},
                edges=[record(RegimeEdgePlan, v) for v in data['edges']],
                templates={k: record(RegimeTemplatePlan, v) for k, v in data['templates'].items()},
                overlays={k: record(RegimeOverlayPlan, v) for k, v in data['overlays'].items()},
                macros=dict(data['macros']), start=data['start'], version=data['version'])
        except (TypeError, AttributeError, KeyError) as exc:
            raise ValueError('Malformed compiled regime program') from exc

    def emission(self, node, context, forced_overlay=''):
        from ..vm.decode import _op_name_to_id, _canonical_op_name, _OP_FAMILY_SLOT, _FAMILY_NAME_SLOT
        template = self.templates[self.nodes[node].template]
        def key_for(name):
            op_id = _op_name_to_id(name)
            return ('op', op_id) if op_id is not None else ('family', _FAMILY_NAME_SLOT[name])
        values = {}
        for name, source in template.families.items():
            key = key_for(name)
            values[key] = values.get(key, 0) + max(0, min(Q15_ONE, evaluate(source, context)))
        values = {key: min(Q15_ONE, value) for key, value in values.items()}
        for name, overlay in self.overlays.items():
            if name == forced_overlay or evaluate(overlay.guard, context):
                for name, source in overlay.families.items():
                    key = key_for(name)
                    values[key] = values.get(key, 0) + evaluate(source, context)
        # Sum before clamping so serialized map order cannot change a mix of
        # positive and negative overlay expressions.
        values = {key: max(0, min(Q15_ONE, value)) for key, value in values.items()}
        strengths = {}
        for op in template.ops:
            direct = values.get(('op', _op_name_to_id(op)))
            slot = _OP_FAMILY_SLOT.get(_canonical_op_name(op))
            strengths[op] = direct if direct is not None else values.get(('family', slot), Q15_ONE)
        return strengths


@dataclass
class AuthoredRegimeState:
    node: str = ''
    blocks: int = 0
    note_blocks: int = 0
    previous_gate: bool = False
    rise_pending: bool = False
    fall_pending: bool = False
    latched_edges: List[int] = field(default_factory=list)
    source_weights: Dict[str, int] = field(default_factory=dict)
    last_weights: Dict[str, int] = field(default_factory=dict)
    fade_blocks: int = 0
    fade_duration: int = 0
    active_edge: int = -1

    def __post_init__(self):
        if type(self.node) is not str or any(type(v) is not bool for v in
                (self.previous_gate, self.rise_pending, self.fall_pending)):
            raise ValueError('Invalid authored regime state')
        for value in (self.blocks, self.note_blocks, self.fade_blocks, self.fade_duration):
            _integer(value, 0, (1 << 63) - 1, 'Regime clock')
        if self.fade_blocks > self.fade_duration:
            raise ValueError('Invalid regime fade clock')
        _integer(self.active_edge, -1, 1023, 'Active regime edge')
        if type(self.latched_edges) is not list or len(set(self.latched_edges)) != len(self.latched_edges):
            raise ValueError('Invalid regime hysteresis state')
        for index in self.latched_edges:
            _integer(index, 0, 1023, 'Latched regime edge')
        for values in (self.source_weights, self.last_weights):
            for name, value in values.items():
                if type(name) is not str:
                    raise ValueError('Invalid regime operator name')
                _integer(value, 0, Q15_ONE, 'Regime operator weight')

    def step(self, program, context, *, gate):
        rise = self.rise_pending or gate and not self.previous_gate
        fall = self.fall_pending or not gate and self.previous_gate
        if not self.node or rise:
            self.node, self.blocks, self.note_blocks = program.start, 0, 0
            self.latched_edges = []
            self.source_weights, self.last_weights = {}, {}
            self.fade_blocks = self.fade_duration = 0
            self.active_edge = -1
        if self.node not in program.nodes:
            raise ValueError('Regime state references a missing node')
        inputs = {'macro.' + name: value for name, value in program.macros.items()}
        inputs.update(context)
        inputs.update(gate=int(gate), gate_rise=int(rise), gate_fall=int(fall),
            regime_enter=int(self.blocks == 0), regime_exit=0,
            age=self.note_blocks * 4 // 3,
            dwell_done=int(self.blocks >= program.nodes[self.node].dwell_blocks))
        inputs.update({'gesture.gate': int(gate), 'gesture.gate_rise': int(rise), 'gesture.gate_fall': int(fall)})
        # A consumed Schmitt edge rearms only beyond its opposite threshold,
        # even while the voice visits another state in the graph.
        for index in list(self.latched_edges):
            edge = program.edges[index]
            value = evaluate(edge.hysteresis_input, inputs)
            if (edge.high and value < edge.exit) or (not edge.high and value > edge.exit):
                self.latched_edges.remove(index)
        for index, edge in enumerate(program.edges):
            if edge.source not in (self.node, 'any') or self.blocks < edge.min_dwell or index in self.latched_edges:
                continue
            if not evaluate(edge.guard, inputs):
                continue
            if edge.hysteresis_input:
                value = evaluate(edge.hysteresis_input, inputs)
                if not (value > edge.enter if edge.high else value < edge.enter):
                    continue
                self.latched_edges.append(index)
            self.source_weights = dict(self.last_weights or program.emission(self.node, inputs))
            self.node, self.blocks = edge.target, 0
            self.fade_blocks, self.fade_duration, self.active_edge = 0, edge.duration, index
            break  # At most one transition per serviced block, authored order.
        inputs.update(regime_enter=int(self.blocks == 0),
                      dwell_done=int(self.blocks >= program.nodes[self.node].dwell_blocks))
        overlay = (program.edges[self.active_edge].overlay if self.active_edge >= 0
                   and (self.blocks == 0 or self.fade_blocks < self.fade_duration) else '')
        output = program.emission(self.node, inputs, overlay)
        if self.fade_blocks < self.fade_duration:
            self.fade_blocks += 1
            names = dict.fromkeys([*self.source_weights, *output])
            output = {name: (self.source_weights.get(name, 0) * (self.fade_duration - self.fade_blocks)
                            + output.get(name, 0) * self.fade_blocks) // self.fade_duration for name in names}
        self.last_weights = dict(output)
        self.blocks += 1
        self.note_blocks += 1
        self.previous_gate, self.rise_pending, self.fall_pending = gate, False, False
        return output, inputs

    def apply(self, program, frame, weights, context):
        template = program.templates[program.nodes[self.node].template]
        frame.ops_enable = list(weights)
        frame.operator_strengths_q15 = dict(weights)
        frame.families = {key: value / Q15_ONE for key, value in weights.items()}
        frame.tree_a_id, frame.tree_b_id = template.tree_a, template.tree_b
        frame.tree_blend_q15 = max(0, min(Q15_ONE, evaluate(template.blend, context)))
        frame.cmmm_node_mask, frame.cmmm_tail_mode = template.mask, template.tail
        frame.cmmm_mask_ramp_blocks = template.mask_ramp
        frame.cmmm_traction_q15 = template.traction


def compile_ir_regimes(ir):
    l2, l3 = ir.layers.L2_regimes, ir.layers.L3_cmmm
    # Nodes alone are also used to name ADSR-phase event templates.
    if l2.regime_graph_spec is None and not l2.transitions:
        return None
    ids = node_ids(ir)
    if l3.transition_kernel:
        raise ValueError('L3 transition kernels require a future graph profile')
    if l2.envelope_overrides:
        raise ValueError('Per-regime envelope overrides require a future graph profile')
    nodes = {}
    for node in l2.regime_nodes:
        ms = node.dwell_ms
        if ms is not None and (type(ms) not in (int, float) or not math.isfinite(ms) or ms < 0):
            raise ValueError('Regime dwell must be finite nonnegative milliseconds')
        nodes[node.id] = RegimeNodePlan(node.id, node.kind, ids[node.id], node.emit_template,
                                        ms_to_blocks(ms) if ms else 0)
    templates = {}
    for name, template in l3.emit_templates.items():
        c = template.cmmm
        if c and type(c.node_mask) is int:
            _integer(c.node_mask, 0, 0xffffffff, 'Node mask')
        if c and isinstance(c.node_mask, str):
            if c.node_mask.startswith('pattern:') and int(c.node_mask[8:]) not in c.PATTERN_TABLE:
                raise ValueError('Unknown node-mask pattern')
            if c.node_mask.startswith('0x'):
                _integer(int(c.node_mask, 16), 0, 0xffffffff, 'Node mask')
        if c and type(c.node_mask) not in (str, int):
            raise ValueError('Invalid authored node mask')
        if c and (isinstance(c.tail_mode, str) and c.tail_mode.upper() not in c.TAIL_MODE_MAP):
            raise ValueError('Unknown authored tail mode')
        if c and isinstance(c.node_mask, str) and not c.node_mask.startswith(('pattern:', '0x')):
            raise ValueError('Unknown authored node mask')
        templates[name] = RegimeTemplatePlan(list(template.ops_enable),
            {key: weight(value) for key, value in template.families.items()},
            tree_id(template.trees.A), tree_id(template.trees.B), weight(template.trees.beta),
            c.get_node_mask_u32() if c else None, c.mask_ramp_blocks if c else 5,
            int(c.get_tail_mode_int()) if c else None, c.traction_q15 if c else Q15_ONE)
    overlays = {}
    for overlay in l3.overlays:
        if overlay.id in overlays or overlay.suppression_rules:
            raise ValueError('Duplicate overlays or unsupported suppression rules')
        overlays[overlay.id] = RegimeOverlayPlan(expression('always' if overlay.kind == 'always' else overlay.enable_when),
            {key: weight(value) for key, value in overlay.add.families.items()})
    edges = []
    raw = l2.regime_graph_spec.edges if l2.regime_graph_spec is not None else l2.transitions
    for item in raw:
        typed = l2.regime_graph_spec is not None
        source, target = (item.src_regime, item.dst_regime) if typed else (item.from_regime, item.to)
        guard = expression(item.guard if typed else item.when)
        transition = item.transition_type if typed else 'hard'
        if transition not in ('hard', 'soft'):
            raise ValueError(f'Unsupported authored transition type: {transition}')
        extra = {}
        if item.hysteresis:
            tree = ast.parse(guard, mode='eval').body
            if (not isinstance(tree, ast.Compare) or len(tree.ops) != 1
                    or not isinstance(tree.ops[0], (ast.Gt, ast.Lt))
                    or _expression(ast.unparse(tree.left))[0] != 'variable'):
                raise ValueError('Hysteresis requires a single scalar > or < threshold guard')
            extra = dict(hysteresis_input=ast.unparse(tree.left),
                enter=int(weight(item.hysteresis.enter)), exit=int(weight(item.hysteresis.exit)),
                high=isinstance(tree.ops[0], ast.Gt))
        if typed and item.overlay:
            overlay = next((o for o in l3.overlays if o.id == item.overlay), None)
            if overlay is not None and overlay.add.events:
                raise ValueError('Transition-forced event overlays require a future event profile')
        edges.append(RegimeEdgePlan(source, target, guard,
            item.min_dwell_blocks if typed else 0,
            item.duration_blocks if typed and transition == 'soft' else 0,
            (item.overlay or '') if typed else '', **extra))
    start = l2.regime_graph_spec.start_regime if l2.regime_graph_spec is not None else (
        'R0_IDLE' if 'R0_IDLE' in nodes else next(iter(nodes), ''))
    macros = {}
    for macro in ir.layers.L0_graph.macros:
        value = macro.default
        if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('Regime macro defaults must be normalized in -1..1')
        if macro.id in macros:
            raise ValueError('Duplicate regime macro')
        macros[macro.id] = round(value * Q15_ONE)
    return CompiledRegimeProgram(nodes, edges, templates, overlays, macros, start)
