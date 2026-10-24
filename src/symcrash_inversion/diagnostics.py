"""Structured diagnostics (spec §5.2, §15 G0): unsupported features are reported and
preserved, never silently dropped."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

SUPPORTED_NODE_TYPES = ("FM6", "Amp", "Filter")
UNSUPPORTED_FEATURE = "unsupported_feature"


@dataclass(frozen=True)
class Diagnostic:
    code: str                      # e.g. "unsupported_feature"
    path: str                      # JSON-pointer style path into instrument_ir_v0, e.g. "/graph/osc1/type"
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


def _ir(d: dict) -> dict:
    return d.get("instrument_ir_v0", d)


def check_supported(preset: dict) -> List[Diagnostic]:
    """Diagnostics for everything in a preset that the fm6-p0.v0 schema cannot represent.

    An empty list means the preset is fully covered by the schema. Unsupported content is
    never removed from the preset; callers keep it as immutable template context.
    """
    ir = _ir(preset)
    out: List[Diagnostic] = []
    graph = ir.get("graph") or {}
    has_fm6 = False
    for name, node in graph.items():
        ntype = (node or {}).get("type")
        if ntype == "FM6":
            has_fm6 = True
        if ntype not in SUPPORTED_NODE_TYPES:
            out.append(Diagnostic(UNSUPPORTED_FEATURE, f"/graph/{name}/type",
                                  f"graph node type {ntype!r} is outside the fm6-p0.v0 schema",
                                  {"node": name, "type": ntype}))
    if not has_fm6:
        out.append(Diagnostic(UNSUPPORTED_FEATURE, "/graph",
                              "no FM6 engine node; fm6-p0.v0 requires an FM6 engine", {}))
    for key in ("cmmm", "regimes", "temperament", "granular", "wavetable"):
        if key in ir:
            out.append(Diagnostic(UNSUPPORTED_FEATURE, f"/{key}",
                                  f"section {key!r} is outside the fm6-p0.v0 schema", {}))
    return out
