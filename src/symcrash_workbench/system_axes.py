"""Live bridge from the L1 system manifold to workbench VoiceFrames.

The production control path owns these coordinates in ``MacroRuntime`` and
applies its frozen block-rate slew laws.  The workbench renders the simpler
IRFrameBuilder path so that arbitrary IR edits can be re-latched every block;
this adapter projects macro motion onto the same seven operator-family axes
used by that path.  At the neutral ControlConfig defaults it is a bit-exact
no-op.
"""

from typing import Dict

from symcrash_python_ap.instrument_ir import (
    AXIS_OPERATORS,
    ControlConfig,
    canonical_axis,
    derive_operators_from_emphasis,
)
from symcrash_python_ap.macro_runtime import MacroRuntime
from symcrash_python_ap.primitives import Q15_ONE, SystemMacro


SYSTEM_AXIS_MACROS = {
    "energy": SystemMacro.ENERGY,
    "motion": SystemMacro.MOTION,
    "focus": SystemMacro.FOCUS,
    "brightness": SystemMacro.BRIGHTNESS,
    "density": SystemMacro.DENSITY,
    "risk": SystemMacro.RISK,
    "space": SystemMacro.SPACE,
    "material": SystemMacro.MATERIAL,
    "unity": SystemMacro.UNITY,
    "dream": SystemMacro.DREAM,
    "slip": SystemMacro.SLIP,
}

# Projection used only by the live IR workbench path.  It follows the existing
# Gesture -> Intent -> Family semantics: intensity recruits events, motion and
# spatial controls recruit transport/topology, spectral/material controls
# recruit color/body, and coherence recruits glue.
SYSTEM_AXIS_PROJECTION = {
    "energy": {"snap": 1.0},
    "motion": {"spread": 0.65, "warp": 0.35},
    "focus": {"color": 0.60, "glue": 0.40},
    "brightness": {"color": 1.0},
    "density": {"body": 1.0},
    "risk": {"chaos": 1.0},
    "space": {"spread": 1.0},
    "material": {"color": 0.50, "body": 0.50},
    "unity": {"glue": 1.0},
    "dream": {"warp": 1.0},
    "slip": {"spread": 1.0},
}

_NEUTRAL = ControlConfig()
_NEUTRAL_MACROS = _NEUTRAL.to_control_program().to_macro_dict()
SYSTEM_AXIS_NEUTRAL_Q15 = {
    name: _NEUTRAL_MACROS[macro_id]
    for name, macro_id in SYSTEM_AXIS_MACROS.items()
}


def _axis_strength(families: dict, axis: str) -> float:
    """Resolve canonical, legacy, and direct per-operator strengths."""
    total = 0.0
    for name, weight in families.items():
        if isinstance(name, str) and canonical_axis(name) == axis:
            try:
                total += float(weight)
            except (TypeError, ValueError):
                pass
    direct = []
    for op_name in AXIS_OPERATORS[axis]:
        if op_name in families:
            try:
                direct.append(float(families[op_name]))
            except (TypeError, ValueError):
                pass
    if direct:
        total = max(total, max(direct))
    return max(0.0, min(1.0, total))


class SystemAxisRuntime:
    """Slewed controller state and its live VoiceFrame projection."""

    def __init__(self, ir=None):
        self.runtime = MacroRuntime()
        self.reset_from_ir(ir)

    def reset_from_ir(self, ir=None) -> None:
        self.runtime = MacroRuntime()
        control = getattr(ir, "control", None) if ir is not None else None
        control = control or ControlConfig()
        self.runtime.apply_control_defaults(control.to_control_program())

    def _sync_targets(self, ir) -> None:
        control = getattr(ir, "control", None) or ControlConfig()
        values = control.to_control_program().to_macro_dict()
        for macro_id, value_q15 in values.items():
            self.runtime.set_system_macro(macro_id, value_q15)

    def projected_deltas(self) -> Dict[str, float]:
        """Return signed changes in the seven operator-family coordinates."""
        deltas = {axis: 0.0 for axis in AXIS_OPERATORS}
        for name, macro_id in SYSTEM_AXIS_MACROS.items():
            current = self.runtime.get_system_macro(macro_id)
            change = (current - SYSTEM_AXIS_NEUTRAL_Q15[name]) / Q15_ONE
            for axis, amount in SYSTEM_AXIS_PROJECTION[name].items():
                deltas[axis] += change * amount
        return deltas

    def apply(self, frame, ir):
        """Advance one block and project the live controls onto ``frame``."""
        self._sync_targets(ir)
        self.runtime.tick()
        deltas = self.projected_deltas()
        active = {axis: delta for axis, delta in deltas.items()
                  if abs(delta) > (0.5 / Q15_ONE)}
        if not active:
            return frame

        families = dict(getattr(frame, "families", None) or {})
        projected = {}
        for axis, delta in active.items():
            strength = max(0.0, min(1.0,
                                    _axis_strength(families, axis) + delta))
            projected[axis] = strength
            # A direct per-op entry has precedence in VM decode.  This makes
            # negative macro motion able to attenuate an authored family as
            # well as positive motion able to recruit it.
            for op_name in AXIS_OPERATORS[axis]:
                families[op_name] = strength

        ops_enable = list(getattr(frame, "ops_enable", None) or [])
        for op_name in derive_operators_from_emphasis(projected):
            if op_name not in ops_enable:
                ops_enable.append(op_name)
        frame.families = families
        frame.ops_enable = ops_enable
        return frame


__all__ = [
    "SYSTEM_AXIS_MACROS",
    "SYSTEM_AXIS_NEUTRAL_Q15",
    "SYSTEM_AXIS_PROJECTION",
    "SystemAxisRuntime",
]
