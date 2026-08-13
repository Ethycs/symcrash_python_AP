"""
symcrash_workbench - OSC tuning workbench (Open Stage Control front-end).

Replaces the imgui knob panel with an off-the-shelf control surface:
Open Stage Control (https://openstagecontrol.ammd.net/) sends OSC to a
session backend that mutates a loaded InstrumentIR in memory, re-renders
through the full render harness (tools/render/render_preset.py), and
plays the result. Tuned values write back to the preset YAML with a
version bump (comment-preserving via ruamel.yaml).

Usage (canonical live entry - streaming mode: realtime audio, per-block
parameter latch, preset browsing over OSC):

    pixi run python -m symcrash_workbench dx_marimba_v0
    open-stage-control --load tools/workbench/symcrash-workbench.json \
        --send 127.0.0.1:7400 --osc-port 7500

Offline mode (debounced full re-render on each edit, the old default):

    pixi run python -m symcrash_workbench dx_marimba_v0 --offline

Preset library over OSC: /preset/list replies with the catalog
(presets/instruments/*.yaml plus capsule ids from the library index);
/preset/load <id-or-filename> live-swaps the session - mid-stream the
old note fades over one block and the new preset retriggers.

The controller-owned L1 manifold is available as normalized OSC targets:

    /system/energy  /system/motion  /system/focus  /system/brightness
    /system/density /system/risk    /system/space  /system/material
    /system/unity   /system/dream   /system/slip

These are block-latched and use the SystemMacro slew laws.  They are
separate from /axis/{attack,stable}/*, which edits the preset's structural
operator emphasis rather than live controller intent.

Phase C internals (see streaming.py): a sustained note rendered
block-by-block through the jit vm_step_block, atlas-mode sweeps, noise
exciter and N-voice unison/stereo latched from the live IR every block:

    pixi run python -m symcrash_workbench.streaming string_pad_v0          # report + demo WAV
    pixi run python -m symcrash_workbench.streaming string_pad_v0 --serve  # live OSC + audio

The same session API is the substrate for the WS3 block-coordinate
refiner: an optimizer is just another caller of WorkbenchSession.handle().
"""

from .session import WorkbenchSession
from .params import build_registry, GROUPS

__all__ = ["WorkbenchSession", "build_registry", "GROUPS",
           "StreamingVoice", "AxisSweep", "StreamingOSCBridge",
           "OSC_AVAILABLE"]

_STREAMING_NAMES = ("StreamingVoice", "AxisSweep", "StreamingOSCBridge",
                    "OSC_AVAILABLE")


def __getattr__(name):
    # Lazy: keeps `python -m symcrash_workbench.streaming` runpy-clean
    if name in _STREAMING_NAMES:
        from . import streaming
        return getattr(streaming, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
