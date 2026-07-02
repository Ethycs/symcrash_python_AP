"""
Generate the Open Stage Control session layout from the param registry.

The registry is the single source of truth: every Param becomes a knob,
grouped into rows per GROUPS, under a performance header (note/velocity/
duration, autoplay, PLAY / SAVE / RELOAD). Re-run after adding params:

    pixi run python -m symcrash_workbench --write-layout
"""

import json
from pathlib import Path

from .library import scan_preset_catalog
from .params import GROUPS, build_registry

DEFAULT_LAYOUT_PATH = (Path(__file__).resolve().parents[2]
                       / "tools" / "workbench" / "symcrash-workbench.json")

_MAX_ROW = 8  # chunk wide groups into rows of this many knobs


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


def _caption(text):
    return {"type": "text", "id": f"cap_{text.lower().replace(' ', '_')}",
            "value": text, "width": 90, "expand": False,
            "vertical": False}


def _button(label, address, mode="push", default=None, width=90):
    w = {"type": "button", "id": address.strip("/"), "label": label,
         "address": address, "mode": mode, "on": 1, "off": 0,
         "width": width, "expand": False}
    if default is not None:
        w["default"] = default
    return w


def _panel(widgets, height=None):
    p = {"type": "panel", "layout": "horizontal", "innerPadding": True,
         "widgets": widgets, "expand": True}
    if height is not None:
        p["height"] = height
        p["expand"] = False
    return p


def _preset_row():
    """
    Preset picker: a dropdown over the catalog scanned at layout
    generation time (values = filename stems, what /preset/load takes)
    plus a LIST button that asks the backend to re-send the catalog on
    the feedback port (/preset/item + /preset/count).
    """
    values = {}
    for entry in scan_preset_catalog():
        label = entry.label
        if label in values:  # duplicate display names: disambiguate
            label = f"{label} ({entry.stem})"
        values[label] = entry.stem
    dropdown = {
        "type": "dropdown",
        "id": "preset_load",
        "label": "PRESET",
        "address": "/preset/load",
        "values": values,
        "expand": True,
    }
    return _panel([
        _caption("PRESET"),
        dropdown,
        _button("LIST", "/preset/list"),
    ], height=110)


def _perform_row():
    return _panel([
        _caption("PERFORM"),
        {"type": "knob", "id": "note", "label": "NOTE", "address": "/note",
         "range": {"min": 24, "max": 96}, "decimals": 0, "default": 60,
         "expand": True},
        {"type": "knob", "id": "velocity", "label": "VEL",
         "address": "/velocity", "range": {"min": 0, "max": 1},
         "decimals": 2, "default": 0.85, "expand": True},
        {"type": "knob", "id": "duration", "label": "DUR s",
         "address": "/duration", "range": {"min": 0.2, "max": 4},
         "logScale": True, "decimals": 2, "default": 0.8, "expand": True},
        _button("AUTO", "/autoplay", mode="toggle", default=1),
        _button("PLAY", "/play"),
        _button("SAVE", "/save"),
        _button("RELOAD", "/reload"),
    ], height=150)


def build_layout() -> dict:
    registry = build_registry()
    rows = [_preset_row(), _perform_row()]

    for group, caption in GROUPS:
        group_params = [p for p in registry.values() if p.group == group]
        if not group_params:
            continue
        for i in range(0, len(group_params), _MAX_ROW):
            chunk = group_params[i:i + _MAX_ROW]
            label = caption if i == 0 else f"{caption} (cont.)"
            rows.append(_panel([_caption(label)] + [_knob(p) for p in chunk]))

    return {
        "version": "1.27.3",
        "type": "session",
        "createdWith": "symcrash_workbench (generated - do not hand-edit; "
                       "re-run with --write-layout)",
        "content": {
            "type": "root",
            "id": "root",
            "layout": "vertical",
            "scroll": True,
            "widgets": rows,
        },
    }


def write_layout(path=None) -> Path:
    path = Path(path) if path else DEFAULT_LAYOUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(build_layout(), f, indent=2)
        f.write("\n")
    return path
