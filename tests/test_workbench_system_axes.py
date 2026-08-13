"""Focused coverage for the OSC-facing SystemMacro manifold bridge."""

import pytest

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.instrument_ir import ControlConfig
from symcrash_workbench.system_axes import (
    SYSTEM_AXIS_MACROS,
    SYSTEM_AXIS_PROJECTION,
    SystemAxisRuntime,
)


class _IR:
    def __init__(self):
        self.control = ControlConfig()


@pytest.mark.parametrize("name", list(SYSTEM_AXIS_MACROS))
def test_each_controller_axis_projects_to_its_compiled_family(name):
    """Every /system coordinate reaches at least one allowed operator."""
    ir = _IR()
    runtime = SystemAxisRuntime(ir)
    setattr(ir.control, name, 1.0)
    frame = VoiceFrame(ops_enable=[], families={})

    runtime.apply(frame, ir)

    for family in SYSTEM_AXIS_PROJECTION[name]:
        assert any(op in frame.ops_enable for op in {
            "color": ["Tilt"],
            "body": ["IntraMix"],
            "spread": ["DiffuseDown", "DiffuseUp"],
            "snap": ["Impulse", "GlobalPulse"],
            "glue": ["GluePair", "SheafProject"],
            "warp": ["BandShear", "SwapBands", "RotateNodes"],
            "chaos": ["FoldPressure"],
        }[family])


def test_neutral_system_defaults_are_frame_noop():
    """Adding the bridge cannot alter existing presets at neutral defaults."""
    ir = _IR()
    runtime = SystemAxisRuntime(ir)
    frame = VoiceFrame(
        ops_enable=["NormalizeScale", "FineClamp", "IntraMix"],
        families={"body": 0.6},
    )
    before_ops = list(frame.ops_enable)
    before_families = dict(frame.families)

    assert runtime.apply(frame, ir) is frame
    assert frame.ops_enable == before_ops
    assert frame.families == before_families


def test_below_neutral_axis_attenuates_an_authored_family():
    """Bipolar movement around neutral can reduce as well as add."""
    ir = _IR()
    ir.control.brightness = 0.0
    runtime = SystemAxisRuntime(ir)
    frame = VoiceFrame(ops_enable=["Tilt"], families={"color": 0.8})

    runtime.apply(frame, ir)

    assert 0.29 < frame.families["Tilt"] < 0.31
