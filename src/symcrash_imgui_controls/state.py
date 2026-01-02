"""
ControlState - Shared state for ImGui controls.

Mirrors the Pico2 hardware controller state:
- 10 macro values (potentiometers)
- 4 note states (buttons)
- Preset selection
- Connection status
"""

from dataclasses import dataclass, field
from typing import List


# Macro IDs (same as pico2_synth_bridge.py)
MACRO_ATTACK = 0
MACRO_DECAY = 1
MACRO_SUSTAIN = 2
MACRO_RELEASE = 3
MACRO_CUTOFF = 4
MACRO_RESO = 5
MACRO_HARMONICS = 6
MACRO_ETA = 7
MACRO_DETUNE = 8
MACRO_GAIN = 9

# Note assignments (same as pico2 control_map.c)
NOTE_C4 = 60
NOTE_D4 = 62
NOTE_E4 = 64
NOTE_F4 = 65

# Q15 constants
Q15_MAX = 32767
Q15_HALF = 16384


@dataclass
class ControlState:
    """
    Shared state for all ImGui controls.

    Mirrors the Pico2 hardware controller state so that
    the UI can function identically to the hardware.
    """

    # 10 macro values (Q15: 0-32767)
    # Index matches MACRO_* constants
    macro_values: List[int] = field(default_factory=lambda: [
        Q15_HALF,   # ATTACK - 50%
        Q15_HALF,   # DECAY - 50%
        Q15_HALF,   # SUSTAIN - 50%
        Q15_HALF,   # RELEASE - 50%
        Q15_HALF,   # CUTOFF - 50%
        0,          # RESO - 0%
        0,          # HARMONICS - 0%
        Q15_HALF,   # ETA - 50%
        Q15_HALF,   # DETUNE - center (no detune)
        Q15_HALF,   # GAIN - 50%
    ])

    # 4 note states (for toggle buttons)
    # True = note on, False = note off
    note_states: List[bool] = field(default_factory=lambda: [False] * 4)

    # Note numbers for each button
    note_numbers: List[int] = field(default_factory=lambda: [
        NOTE_C4, NOTE_D4, NOTE_E4, NOTE_F4
    ])

    # Preset selection (0-7)
    current_preset: int = 0

    # Tree selection (0-4)
    current_tree: int = 0

    # Connection state
    connected: bool = False
    port: str = ""

    def get_macro_float(self, macro_id: int) -> float:
        """Get macro value as 0.0-1.0 float."""
        if 0 <= macro_id < len(self.macro_values):
            return self.macro_values[macro_id] / Q15_MAX
        return 0.0

    def set_macro_float(self, macro_id: int, value: float) -> None:
        """Set macro value from 0.0-1.0 float."""
        if 0 <= macro_id < len(self.macro_values):
            self.macro_values[macro_id] = int(max(0, min(1, value)) * Q15_MAX)

    def toggle_note(self, button_idx: int) -> bool:
        """Toggle note state. Returns new state."""
        if 0 <= button_idx < len(self.note_states):
            self.note_states[button_idx] = not self.note_states[button_idx]
            return self.note_states[button_idx]
        return False

    def get_note_number(self, button_idx: int) -> int:
        """Get MIDI note number for button."""
        if 0 <= button_idx < len(self.note_numbers):
            return self.note_numbers[button_idx]
        return 60


# Macro labels for display
MACRO_LABELS = [
    "ATK",    # Attack
    "DCY",    # Decay
    "SUS",    # Sustain
    "REL",    # Release
    "CUT",    # Cutoff
    "RES",    # Resonance
    "HRM",    # Harmonics
    "ETA",    # Eta (dry/wet)
    "DET",    # Detune
    "GAIN",   # Gain
]

# Note labels for display
NOTE_LABELS = ["C4", "D4", "E4", "F4"]
