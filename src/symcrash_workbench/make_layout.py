"""Generate the tabbed Open Stage Control workbench.

The parameter registry remains the single source of truth for scalar
controls. Structural surfaces use stable slot names defined by
``patchboard.py`` so the same generated session follows preset changes.

Regenerate after adding controls:

    pixi run python -m symcrash_workbench --write-layout
"""

import json
from pathlib import Path

from symcrash_python_ap.vm.fm_algorithms import list_algorithms

from .library import scan_preset_catalog
from .params import build_registry
from .patchboard import (
    BEHAVIOR_ADDRESS,
    GRAPH_ADDRESS,
    MAX_ENVELOPES,
    MAX_GRAPH_NODES,
    MAX_REGIMES,
    MOD_ADDRESS,
    MOD_TARGETS,
)

DEFAULT_LAYOUT_PATH = (Path(__file__).resolve().parents[2]
                       / "tools" / "workbench" / "symcrash-workbench.json")

_MAX_ROW = 8


def _knob(param):
    return {
        "type": "knob",
        "id": param.address.strip("/").replace("/", "_"),
        "label": param.label,
        "address": param.address,
        "range": {"min": param.lo, "max": param.hi},
        "logScale": bool(param.log),
        "decimals": param.decimals,
        "doubleTap": True,
        "expand": True,
    }


def _labeled(widget, label):
    """Give a control an explicit, version-independent visible caption."""
    widget_id = widget["id"]
    return {
        "type": "panel",
        "id": f"{widget_id}_field",
        "layout": "vertical",
        "innerPadding": False,
        "expand": True,
        "widgets": [
            {"type": "text", "id": f"{widget_id}_label", "value": label,
             "height": 30, "expand": False, "vertical": False},
            widget,
        ],
    }


def _labeled_knob(param):
    return _labeled(_knob(param), param.label)


def _caption(text, width=110):
    return {"type": "text", "id": f"cap_{_safe_id(text)}",
            "value": text, "width": width, "expand": False,
            "vertical": False}


def _safe_id(text):
    return "".join(char.lower() if char.isalnum() else "_" for char in text)


def _button(label, address, mode="push", default=None, width=90):
    widget = {"type": "button", "id": address.strip("/").replace("/", "_"),
              "label": label, "address": address, "mode": mode,
              "on": 1, "off": 0, "width": width, "expand": False}
    if default is not None:
        widget["default"] = default
    return widget


def _panel(widgets, height=None, layout="horizontal", panel_id=None):
    panel = {"type": "panel", "layout": layout, "innerPadding": True,
             "widgets": widgets, "expand": True}
    if panel_id:
        panel["id"] = panel_id
    if height is not None:
        panel["height"] = height
        panel["expand"] = False
    return panel


def _readonly(id_, address, default="--", height=42):
    return {
        "type": "input",
        "id": id_,
        "address": address,
        "default": default,
        "editable": False,
        "bypass": True,
        "align": "left",
        "height": height,
        "expand": True,
    }


def _intro(id_, text):
    return {
        "type": "html",
        "id": id_,
        "html": f"<p>{text}</p>",
        "height": 70,
        "expand": False,
        "border": False,
        "bypass": True,
    }


def _tab(id_, label, widgets):
    return {
        "type": "tab",
        "id": id_,
        "label": label,
        "layout": "vertical",
        "scroll": True,
        "widgets": widgets,
    }


def _preset_row():
    values = {}
    for entry in scan_preset_catalog():
        label = entry.label
        if label in values:
            label = f"{label} ({entry.stem})"
        values[label] = entry.stem
    dropdown = {
        "type": "dropdown",
        "id": "preset_load",
        "label": "PRESET",
        "address": "/preset/load",
        "values": values,
        "css": "option:first-child{display:none}",
        "expand": True,
    }
    return _panel([
        _caption("PRESET"),
        dropdown,
        _button("LIST", "/preset/list"),
    ], height=110, panel_id="preset_row")


def _perform_row():
    note = {"type": "knob", "id": "note", "label": "NOTE",
            "address": "/note", "range": {"min": 24, "max": 96},
            "decimals": 0, "default": 60, "expand": True}
    velocity = {"type": "knob", "id": "velocity", "label": "VEL",
                "address": "/velocity", "range": {"min": 0, "max": 1},
                "decimals": 2, "default": 0.85, "expand": True}
    duration = {"type": "knob", "id": "duration", "label": "DUR s",
                "address": "/duration", "range": {"min": 0.2, "max": 4},
                "logScale": True, "decimals": 2, "default": 0.8,
                "expand": True}
    return _panel([
        _caption("PERFORM"),
        _labeled(note, "NOTE"),
        _labeled(velocity, "VEL"),
        _labeled(duration, "DUR s"),
        _button("AUTO", "/autoplay", mode="toggle", default=1),
        _button("PLAY", "/play"),
        _button("SAVE", "/save"),
        _button("RELOAD", "/reload"),
    ], height=150, panel_id="perform_row")


def _keyboard_row():
    return _panel([
        _caption("KEYBOARD"),
        {
            "type": "keyboard",
            "id": "performance_keyboard",
            "address": "/stream/keyboard",
            "keys": 25,
            "start": 48,
            "on": 1,
            "off": 0,
            "toggles": False,
            "traversing": True,
            "split": False,
            "expand": True,
        },
    ], height=220, panel_id="keyboard_row")


_GROUP_CAPTIONS = {
    "amp": "AMP",
    "fm6": "FM6 OPERATORS",
    "env1": "MOD ENV 1",
    "env2": "MOD ENV 2",
    "axis_attack": "AXES - ATTACK",
    "axis_stable": "AXES - STABLE",
    "system": "SYSTEM MANIFOLD",
    "filter": "FILTER",
    "noise": "NOISE",
    "unison": "UNISON",
}


def _group_rows(registry, groups, exclude=()):
    excluded = set(exclude)
    rows = []
    for group in groups:
        params = [param for param in registry.values()
                  if param.group == group and param.address not in excluded]
        for index in range(0, len(params), _MAX_ROW):
            chunk = params[index:index + _MAX_ROW]
            caption = _GROUP_CAPTIONS[group]
            if index:
                caption += " (cont.)"
            rows.append(_panel([_caption(caption)]
                               + [_labeled_knob(param) for param in chunk],
                               panel_id=f"group_{group}_{index // _MAX_ROW}"))
    return rows


def _slot_readouts(prefix, address_prefix, count):
    return _panel([
        _readonly(f"{prefix}_slot_{i + 1}",
                  f"{address_prefix}/{prefix}{i + 1}")
        for i in range(count)
    ], layout="vertical", panel_id=f"{prefix}_slot_readouts")


def _graph_surface():
    labels = {f"NODE {i + 1}": f"node{i + 1}"
              for i in range(MAX_GRAPH_NODES)}
    outputs = {**labels, "AUDIO OUT": "audio"}
    return [
        _intro("patch_intro",
               "Drag cords from a node to its downstream node. Each graph "
               "node has one output; terminal nodes connect to AUDIO OUT. "
               "Cycles and inactive slots are rejected and snapped back."),
        _panel([
            _slot_readouts("node", "/editor/graph", MAX_GRAPH_NODES),
            {
                "type": "patchbay",
                "id": "graph_patchbay",
                "address": GRAPH_ADDRESS,
                "inputs": labels,
                "outputs": outputs,
                "css": ".nodes { width: 30% }",
                "height": 500,
                "expand": True,
            },
        ], height=560, panel_id="graph_surface"),
    ]


def _algorithm_control():
    values = {f"{algorithm_id:02d} · {name}": algorithm_id
              for algorithm_id, name in list_algorithms()}
    dropdown = {
        "type": "dropdown",
        "id": "fm_algorithm",
        "label": "ALGORITHM",
        "address": "/fm/algorithm",
        "values": values,
        "css": "option:first-child{display:none}",
        "expand": True,
    }
    return _panel([
        _caption("TOPOLOGY"),
        _labeled(dropdown, "ALGORITHM"),
        _readonly("fm_algorithm_name", "/editor/fm/name",
                  "No FM6 node in this preset"),
    ], height=120, panel_id="fm_algorithm_row")


def _fm_surface(registry):
    operators = {f"OP {i + 1}": f"op{i + 1}" for i in range(6)}
    return [
        _intro("fm_intro",
               "The cords are the selected DX7/WSE algorithm. Choose an "
               "algorithm to repatch; self-cords show active feedback."),
        _algorithm_control(),
        {
            "type": "patchbay",
            "id": "fm_topology_patchbay",
            "address": "/editor/fm/topology",
            "inputs": operators,
            "outputs": {**operators, "AUDIO OUT": "audio"},
            "bypass": True,
            "css": ".nodes { width: 28% }",
            "height": 430,
            "expand": False,
        },
        *_group_rows(registry, ["fm6"], exclude={"/fm/algorithm"}),
    ]


def _mod_surface(registry):
    inputs = {f"ENV {i + 1}": f"env{i + 1}"
              for i in range(MAX_ENVELOPES)}
    outputs = {target.replace("_", " "): target for target in MOD_TARGETS}
    return [
        _intro("mod_intro",
               "Each named envelope may drive any number of destinations. "
               "Empty envelope slots are inert and reject cords."),
        _panel([
            _slot_readouts("env", "/editor/mod", MAX_ENVELOPES),
            {
                "type": "patchbay",
                "id": "mod_patchbay",
                "address": MOD_ADDRESS,
                "inputs": inputs,
                "outputs": outputs,
                "css": ".nodes { width: 34% }",
                "height": 540,
                "expand": True,
            },
        ], height=600, panel_id="mod_surface"),
        *_group_rows(registry, ["env1", "env2"]),
    ]


def _behavior_surface(registry):
    states = {f"STATE {i + 1}": f"state{i + 1}"
              for i in range(MAX_REGIMES)}
    inputs = {**states, "ANY STATE": "any"}
    return [
        _intro("behavior_intro",
               "The behavior graph controls state transitions. Existing "
               "edge guards and hysteresis are preserved; a new edge uses "
               "Gate rises, Gate falls, or dwell_done as appropriate."),
        _panel([
            _button("MAKE DEFAULT GRAPH", "/editor/behavior/materialize",
                    width=190),
            _button("RESYNC", "/editor/resync", width=100),
        ], height=72, panel_id="behavior_actions"),
        _panel([
            _slot_readouts("state", "/editor/behavior", MAX_REGIMES),
            {
                "type": "patchbay",
                "id": "behavior_patchbay",
                "address": BEHAVIOR_ADDRESS,
                "inputs": inputs,
                "outputs": states,
                "css": ".nodes { width: 34% }",
                "height": 540,
                "expand": True,
            },
        ], height=600, panel_id="behavior_surface"),
        *_group_rows(registry, ["axis_attack", "axis_stable"]),
    ]


def _source_surface():
    return [
        _intro("source_intro",
               "This is the exact backing YAML, including comments. Scalar "
               "and patchboard edits are transactional: SAVE writes and "
               "bumps meta.version; RELOAD discards unsaved edits."),
        _panel([
            _caption("FILE"),
            _readonly("source_path", "/editor/preset/path"),
            _caption("VERSION", width=90),
            _readonly("source_version", "/editor/version", height=42),
            _caption("DIRTY", width=70),
            _readonly("source_dirty", "/editor/dirty", "0", height=42),
            _button("SAVE", "/save"),
            _button("RELOAD", "/reload"),
        ], height=92, panel_id="source_header"),
        _readonly("editor_status", "/editor/status", "Ready", height=52),
        {
            "type": "frame",
            "id": "yaml_source_editor",
            "label": False,
            "value": "http://127.0.0.1:7410/",
            "height": 760,
            "expand": True,
        },
    ]


def build_layout() -> dict:
    registry = build_registry()
    play = [
        _preset_row(),
        _perform_row(),
        _keyboard_row(),
        *_group_rows(registry, ["amp", "filter", "noise", "unison"]),
    ]
    source_tab = _tab("source_tab", "SOURCE", _source_surface())
    source_tab["detached"] = False
    tabs = [
        _tab("play_tab", "PLAY", play),
        _tab("patch_tab", "PATCH", _graph_surface()),
        _tab("fm6_tab", "FM6", _fm_surface(registry)),
        _tab("mod_tab", "MOD MAP", _mod_surface(registry)),
        _tab("behavior_tab", "BEHAVIOR", _behavior_surface(registry)),
        _tab("system_tab", "SYSTEM", [
            _intro("system_intro",
                   "These eleven controller-owned axes steer the hidden L1 "
                   "manifold live; they are distinct from per-state "
                   "operator emphasis on the BEHAVIOR tab."),
            *_group_rows(registry, ["system"]),
        ]),
        source_tab,
    ]
    return {
        "version": "1.27.3",
        "type": "session",
        "createdWith": "symcrash_workbench (generated - do not hand-edit; "
                       "re-run with --write-layout)",
        "content": {
            "type": "root",
            "id": "root",
            "value": 0,
            "tabs": tabs,
        },
    }


def write_layout(path=None) -> Path:
    path = Path(path) if path else DEFAULT_LAYOUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(build_layout(), file, indent=2)
        file.write("\n")
    return path
