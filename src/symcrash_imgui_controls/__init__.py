"""
Symcrash Sound Design Workbench
================================

Dear ImGui-based sound design workbench for the Symcrash synthesizer.
Provides preset browsing, character control knobs, and render-and-play.

Usage:
    from symcrash_imgui_controls import main
    main()

Or run directly:
    python -m symcrash_imgui_controls
"""

from .state import (
    ControlState,
    MACRO_ATTACK,
    MACRO_DECAY,
    MACRO_SUSTAIN,
    MACRO_RELEASE,
    MACRO_CUTOFF,
    MACRO_RESO,
    MACRO_HARMONICS,
    MACRO_ETA,
    MACRO_DETUNE,
    MACRO_GAIN,
    MACRO_LABELS,
    NOTE_LABELS,
    Q15_MAX,
    Q15_HALF,
    CHARACTER_LABELS,
    CHARACTER_FIELDS,
    PLAYBACK_NOTES,
)

from .uart_sender import (
    UARTSender,
    find_available_ports,
)

from .controls import (
    render_knob,
    render_knob_row,
    render_note_button,
    render_note_row,
    render_preset_combo,
    render_tree_combo,
    render_connection_status,
    render_character_knob,
    render_character_knob_row,
    render_preset_file_combo,
)

from .app import (
    SymcrashImGuiApp,
    main,
)


__all__ = [
    # State
    'ControlState',
    'MACRO_ATTACK',
    'MACRO_DECAY',
    'MACRO_SUSTAIN',
    'MACRO_RELEASE',
    'MACRO_CUTOFF',
    'MACRO_RESO',
    'MACRO_HARMONICS',
    'MACRO_ETA',
    'MACRO_DETUNE',
    'MACRO_GAIN',
    'MACRO_LABELS',
    'NOTE_LABELS',
    'Q15_MAX',
    'Q15_HALF',
    'CHARACTER_LABELS',
    'CHARACTER_FIELDS',
    'PLAYBACK_NOTES',
    # UART
    'UARTSender',
    'find_available_ports',
    # Controls
    'render_knob',
    'render_knob_row',
    'render_note_button',
    'render_note_row',
    'render_preset_combo',
    'render_tree_combo',
    'render_connection_status',
    'render_character_knob',
    'render_character_knob_row',
    'render_preset_file_combo',
    # App
    'SymcrashImGuiApp',
    'main',
]
