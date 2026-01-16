"""
Symcrash Preset Editor - Multi-Level InstrumentIR Editor
========================================================

A visual editor for InstrumentIR presets with level-specific views:
- L0 Patch: DSP node graph (OSC→MIX→FILTER→AMP→FX)
- L1 Lift: Gesture→Intent mapping matrix
- L2 Regimes: State machine (R0-R6 with transitions)
- L3+ CMMM: Emit templates, overlays (tree/YAML view)

Built on imgui_bundle with imgui_node_editor for graph views.

Usage:
    from symcrash_preset_editor import PresetEditorApp
    app = PresetEditorApp()
    app.run()
"""

from .app import PresetEditorApp, main
from .views.preset_browser import PresetBrowser

__all__ = ["PresetEditorApp", "PresetBrowser", "main"]
