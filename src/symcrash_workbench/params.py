"""
Parameter registry: OSC address <-> InstrumentIR field <-> preset YAML path.

Each Param carries three accessors:
  get(ir)              -> current value (None if the preset has no such target)
  set(ir, value)       -> mutate the in-memory IR (False if no target)
  yaml_apply(root, ir, value) -> write into the ruamel round-trip YAML doc
                                 (root = doc['instrument_ir_v0'])

The registry is the single source of truth: the OSC dispatcher, the
Open Stage Control layout generator, and YAML write-back all derive
from it. Adding a Param here adds a knob everywhere.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from symcrash_python_ap.instrument_ir import (
    ADSR,
    AXIS_OPERATORS,
    EmitTemplate,
    derive_operators_from_emphasis,
)
from symcrash_python_ap.primitives import Q16_ONE

AXES = list(AXIS_OPERATORS.keys())  # color, body, spread, snap, glue, warp, chaos

# Axis knobs write to these behavior states. Simple-form presets map
# envelope phase -> T_ATTACK / T_STABLE / T_RECOVERY in the render
# harness, so both states are audible in a single note.
AXIS_STATES = ["attack", "stable"]

# Mod-envelope bindings exposed by index (env1 = first binding in YAML order).
ENV_SLOTS = 2


@dataclass
class Param:
    address: str
    label: str
    lo: float
    hi: float
    group: str
    get: Callable[[Any], Optional[float]]
    set: Callable[[Any, float], bool]
    yaml_apply: Callable[[Any, Any, float], bool]
    log: bool = False
    decimals: int = 3


# ---------------------------------------------------------------- IR helpers

def _node(ir, type_):
    for n in ir.layers.L0_graph.nodes:
        if n.type == type_:
            return n
    return None


def _ensure_adsr(node):
    if node.params.adsr is None:
        node.params.adsr = ADSR()
    return node.params.adsr


def _ynode(root, ir, type_):
    """Find the YAML mapping for the first L0 node of a given type."""
    node = _node(ir, type_)
    if node is None:
        return None
    graph = root.get("graph")
    if graph is None or node.id not in graph:
        return None
    return graph[node.id]


# ---------------------------------------------------------------- factories

def _amp_param(field, ykey, label, lo, hi, log=False, decimals=3):
    def get(ir):
        node = _node(ir, "Amp")
        if node is None or node.params.adsr is None:
            return None
        return float(getattr(node.params.adsr, field))

    def set_(ir, v):
        node = _node(ir, "Amp")
        if node is None:
            return False
        setattr(_ensure_adsr(node), field, float(v))
        return True

    def yapply(root, ir, v):
        y = _ynode(root, ir, "Amp")
        if y is None:
            return False
        y[ykey] = round(float(v), 3)
        return True

    return Param(f"/amp/{ykey}", label, lo, hi, "amp", get, set_, yapply,
                 log=log, decimals=decimals)


def _amp_dyn_param(field, label, lo, hi):
    def get(ir):
        node = _node(ir, "Amp")
        return None if node is None else float(getattr(node.params, field))

    def set_(ir, v):
        node = _node(ir, "Amp")
        if node is None:
            return False
        setattr(node.params, field, float(v))
        return True

    def yapply(root, ir, v):
        y = _ynode(root, ir, "Amp")
        if y is None:
            return False
        y[field] = round(float(v), 3)
        return True

    return Param(f"/amp/{field}", label, lo, hi, "amp", get, set_, yapply)


def _fm6_op_param(i, kind):
    """kind: 'ratio' (Q16.16 in IR, float in YAML) or 'level' (Q15 / float)."""
    scale = 65536.0 if kind == "ratio" else 32767.0
    attr = "op_ratios" if kind == "ratio" else "op_levels"
    lo, hi = (0.25, 16.0) if kind == "ratio" else (0.0, 1.0)

    def get(ir):
        node = _node(ir, "FM6")
        if node is None or getattr(node.params, attr) is None:
            return None
        vals = getattr(node.params, attr)
        return vals[i] / scale if i < len(vals) else None

    def set_(ir, v):
        node = _node(ir, "FM6")
        if node is None:
            return False
        vals = getattr(node.params, attr)
        if vals is None:
            vals = [Q16_ONE] * 6 if kind == "ratio" else [0] * 6
            setattr(node.params, attr, vals)
        while len(vals) < 6:
            vals.append(Q16_ONE if kind == "ratio" else 0)
        vals[i] = int(round(float(v) * scale))
        return True

    def yapply(root, ir, v):
        y = _ynode(root, ir, "FM6")
        if y is None:
            return False
        if "operators" in y and i < len(y["operators"]):
            y["operators"][i][kind] = round(float(v), 4)
            return True
        legacy = "op_ratios" if kind == "ratio" else "op_levels"
        if legacy in y and i < len(y[legacy]):
            y[legacy][i] = round(float(v), 4)
            return True
        return False

    return Param(f"/fm/op{i + 1}/{kind}", f"OP{i + 1} {kind[:3].upper()}",
                 lo, hi, "fm6", get, set_, yapply,
                 log=(kind == "ratio"), decimals=4)


def _fm6_scalar_param(field, label, lo, hi, log=False, decimals=1,
                      cast=float):
    def get(ir):
        node = _node(ir, "FM6")
        if node is None:
            return None
        v = getattr(node.params, field)
        return None if v is None else float(v)

    def set_(ir, v):
        node = _node(ir, "FM6")
        if node is None:
            return False
        setattr(node.params, field, cast(v))
        return True

    def yapply(root, ir, v):
        y = _ynode(root, ir, "FM6")
        if y is None:
            return False
        y[field] = cast(v) if cast is int else round(float(v), 1)
        return True

    return Param(f"/fm/{field}", label, lo, hi, "fm6", get, set_, yapply,
                 log=log, decimals=decimals)


def _env_param(slot, field, ykey, label, lo, hi, log=False):
    """slot: 0-based index into ir.envelopes (YAML order)."""
    is_adsr = field in ("a_ms", "d_ms", "s", "r_ms")

    def get(ir):
        envs = getattr(ir, "envelopes", None) or []
        if slot >= len(envs):
            return None
        b = envs[slot]
        return float(getattr(b.adsr, field) if is_adsr else getattr(b, field))

    def set_(ir, v):
        envs = getattr(ir, "envelopes", None) or []
        if slot >= len(envs):
            return False
        b = envs[slot]
        if is_adsr:
            setattr(b.adsr, field, float(v))
        else:
            setattr(b, field, float(v))
        return True

    def yapply(root, ir, v):
        envs = getattr(ir, "envelopes", None) or []
        if slot >= len(envs):
            return False
        env_id = envs[slot].id
        ysec = root.get("envelopes")
        if ysec is None or env_id not in ysec:
            return False
        ysec[env_id][ykey] = round(float(v), 3)
        return True

    return Param(f"/env{slot + 1}/{ykey}", label, lo, hi, f"env{slot + 1}",
                 get, set_, yapply, log=log)


def _axis_param(state, axis):
    tname = f"T_{state.upper()}"

    def get(ir):
        tmpl = ir.layers.L3_cmmm.emit_templates.get(tname)
        if tmpl is None:
            return 0.0
        v = tmpl.families.get(axis, 0.0)
        return float(v) if isinstance(v, (int, float)) else 0.0

    def set_(ir, v):
        templates = ir.layers.L3_cmmm.emit_templates
        tmpl = templates.get(tname)
        if tmpl is None:
            tmpl = EmitTemplate()
            templates[tname] = tmpl
        numeric = {k: w for k, w in tmpl.families.items()
                   if isinstance(w, (int, float))}
        expected = derive_operators_from_emphasis(numeric)
        if float(v) <= 0.0005:
            tmpl.families.pop(axis, None)
        else:
            tmpl.families[axis] = round(float(v), 4)
        # Re-derive ops_enable unless the preset pinned an explicit
        # processing list (escape hatch) - never clobber that.
        if not tmpl.ops_enable or tmpl.ops_enable == expected:
            numeric = {k: w for k, w in tmpl.families.items()
                       if isinstance(w, (int, float))}
            tmpl.ops_enable = derive_operators_from_emphasis(numeric)
        return True

    def yapply(root, ir, v):
        behavior = root.get("behavior")
        if behavior is None:
            return False
        if state not in behavior or behavior[state] is None:
            behavior[state] = {"emphasis": {}}
        ystate = behavior[state]
        if "emphasis" not in ystate or ystate["emphasis"] is None:
            ystate["emphasis"] = {}
        if float(v) <= 0.0005:
            ystate["emphasis"].pop(axis, None)
        else:
            ystate["emphasis"][axis] = round(float(v), 4)
        return True

    return Param(f"/axis/{state}/{axis}", axis.upper(), 0.0, 1.0,
                 f"axis_{state}", get, set_, yapply)


def _filter_param(field, label, lo, hi, log=False):
    def get(ir):
        node = _node(ir, "Filter")
        if node is None:
            return None
        v = getattr(node.params, field)
        return None if v is None else float(v)

    def set_(ir, v):
        node = _node(ir, "Filter")
        if node is None:
            return False
        setattr(node.params, field, float(v))
        return True

    def yapply(root, ir, v):
        y = _ynode(root, ir, "Filter")
        if y is None:
            return False
        y[field] = round(float(v), 2)
        return True

    return Param(f"/filter/{field}", label, lo, hi, "filter",
                 get, set_, yapply, log=log)


# ---------------------------------------------------------------- registry

def build_registry() -> Dict[str, Param]:
    params: List[Param] = []

    params += [
        _amp_param("a_ms", "attack", "ATK", 0.5, 2000.0, log=True),
        _amp_param("d_ms", "decay", "DCY", 5.0, 4000.0, log=True),
        _amp_param("s", "sustain", "SUS", 0.0, 1.0),
        _amp_param("r_ms", "release", "REL", 5.0, 3000.0, log=True),
        _amp_dyn_param("velocity_sens", "VEL SENS", 0.0, 1.0),
        _amp_dyn_param("key_scale_rate", "KEY SCALE", 0.0, 1.5),
    ]

    for i in range(6):
        params.append(_fm6_op_param(i, "ratio"))
        params.append(_fm6_op_param(i, "level"))
    params += [
        _fm6_scalar_param("feedback", "FB", 0, 7, decimals=0, cast=int),
        _fm6_scalar_param("post_filter_hz", "POST LPF", 200.0, 20000.0,
                          log=True),
    ]

    for slot in range(ENV_SLOTS):
        params += [
            _env_param(slot, "a_ms", "attack", "ATK", 0.5, 2000.0, log=True),
            _env_param(slot, "d_ms", "decay", "DCY", 5.0, 4000.0, log=True),
            _env_param(slot, "s", "sustain", "SUS", 0.0, 1.0),
            _env_param(slot, "r_ms", "release", "REL", 5.0, 3000.0, log=True),
            _env_param(slot, "velocity_sens", "velocity_sens", "VEL SENS",
                       0.0, 1.0),
            _env_param(slot, "key_scale_rate", "key_scale_rate", "KEY SCALE",
                       0.0, 1.5),
        ]

    for state in AXIS_STATES:
        for axis in AXES:
            params.append(_axis_param(state, axis))

    params += [
        _filter_param("cutoff_hz", "CUTOFF", 50.0, 20000.0, log=True),
        _filter_param("resonance", "RESO", 0.0, 1.0),
        _filter_param("env_amount", "ENV AMT", 0.0, 1.0),
    ]

    return {p.address: p for p in params}


# Layout group order + captions (consumed by make_layout.py)
GROUPS = [
    ("amp", "AMP"),
    ("fm6", "FM6"),
    ("env1", "MOD ENV 1"),
    ("env2", "MOD ENV 2"),
    ("axis_attack", "AXES - ATTACK"),
    ("axis_stable", "AXES - STABLE"),
    ("filter", "FILTER"),
]
