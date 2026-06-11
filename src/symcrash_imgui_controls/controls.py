"""
ImGui Control Widgets for Symcrash.

Provides knobs (using imgui_knobs) and buttons for synth control.
"""

from typing import Tuple
from imgui_bundle import imgui, imgui_knobs

from .state import Q15_MAX


def render_knob(
    label: str,
    value_q15: int,
    size: float = 50.0,
    format_str: str = "%.0f%%",
) -> Tuple[bool, int]:
    """
    Render a rotary knob control.

    Args:
        label: Knob label
        value_q15: Current value in Q15 (0-32767)
        size: Knob diameter in pixels
        format_str: Display format (receives 0-100 percentage)

    Returns:
        (changed, new_value_q15) tuple
    """
    # Convert Q15 to 0-100 for display
    value_pct = (value_q15 / Q15_MAX) * 100.0

    # Render knob
    changed, new_pct = imgui_knobs.knob(
        label,
        value_pct,
        0.0,
        100.0,
        0.0,  # default_value
        format_str,
        imgui_knobs.ImGuiKnobVariant_.wiper_dot,
        size,
        imgui_knobs.ImGuiKnobFlags_.no_input,
    )

    # Convert back to Q15
    new_q15 = int((new_pct / 100.0) * Q15_MAX)
    return changed, new_q15


def render_knob_row(
    labels: list,
    values_q15: list,
    size: float = 50.0,
) -> Tuple[bool, list, int]:
    """
    Render a row of knobs.

    Args:
        labels: List of knob labels
        values_q15: List of Q15 values
        size: Knob diameter

    Returns:
        (any_changed, new_values, changed_index) tuple
        changed_index is -1 if nothing changed
    """
    any_changed = False
    changed_idx = -1
    new_values = list(values_q15)

    for i, (label, value) in enumerate(zip(labels, values_q15)):
        if i > 0:
            imgui.same_line()

        changed, new_val = render_knob(label, value, size)
        if changed:
            any_changed = True
            changed_idx = i
            new_values[i] = new_val

    return any_changed, new_values, changed_idx


def render_note_button(
    label: str,
    is_on: bool,
    width: float = 60.0,
    height: float = 40.0,
) -> bool:
    """
    Render a note toggle button.

    Args:
        label: Button label (e.g., "C4")
        is_on: Current note state
        width: Button width
        height: Button height

    Returns:
        True if button was clicked
    """
    # Color based on state
    if is_on:
        # Green when note is on
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.7, 0.2, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.8, 0.3, 1.0))
        imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.6, 0.1, 1.0))
    else:
        # Dark gray when off
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.3, 0.3, 0.3, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.4, 0.4, 0.4, 1.0))
        imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.2, 0.2, 0.2, 1.0))

    clicked = imgui.button(label, imgui.ImVec2(width, height))

    imgui.pop_style_color(3)

    return clicked


def render_note_row(
    labels: list,
    states: list,
    width: float = 60.0,
    height: float = 40.0,
) -> Tuple[bool, int]:
    """
    Render a row of note buttons.

    Args:
        labels: List of note labels
        states: List of note states
        width: Button width
        height: Button height

    Returns:
        (any_clicked, clicked_index) tuple
        clicked_index is -1 if nothing clicked
    """
    clicked_idx = -1

    for i, (label, state) in enumerate(zip(labels, states)):
        if i > 0:
            imgui.same_line()

        if render_note_button(label, state, width, height):
            clicked_idx = i

    return clicked_idx >= 0, clicked_idx


def render_preset_combo(current: int, count: int = 8) -> Tuple[bool, int]:
    """
    Render preset selector combo box.

    Args:
        current: Current preset index
        count: Number of presets

    Returns:
        (changed, new_index) tuple
    """
    items = [f"Preset {i}" for i in range(count)]

    imgui.set_next_item_width(120)
    changed, new_idx = imgui.combo("##preset", current, items)

    return changed, new_idx


def render_tree_combo(current: int) -> Tuple[bool, int]:
    """
    Render tree selector combo box.

    Args:
        current: Current tree index

    Returns:
        (changed, new_index) tuple
    """
    trees = ["balanced", "harmonic", "texture", "impact", "drone"]

    imgui.set_next_item_width(100)
    changed, new_idx = imgui.combo("Tree", current, trees)

    return changed, new_idx


def render_character_knob(
    label: str,
    value: float,
    size: float = 45.0,
) -> Tuple[bool, float]:
    """
    Render a character control knob (0.0-1.0 range).

    Args:
        label: Knob label (e.g., "BRT")
        value: Current value (0.0-1.0)
        size: Knob diameter in pixels

    Returns:
        (changed, new_value) tuple with value in 0.0-1.0
    """
    value_pct = value * 100.0

    changed, new_pct = imgui_knobs.knob(
        label,
        value_pct,
        0.0,
        100.0,
        50.0,  # default_value at center
        "%.0f%%",
        imgui_knobs.ImGuiKnobVariant_.wiper_dot,
        size,
        imgui_knobs.ImGuiKnobFlags_.no_input,
    )

    return changed, new_pct / 100.0


def render_character_knob_row(
    labels: list,
    values: list,
    size: float = 45.0,
) -> Tuple[bool, list, int]:
    """
    Render a row of character knobs (0.0-1.0 range).

    Args:
        labels: List of knob labels
        values: List of float values (0.0-1.0)
        size: Knob diameter

    Returns:
        (any_changed, new_values, changed_index) tuple
        changed_index is -1 if nothing changed
    """
    any_changed = False
    changed_idx = -1
    new_values = list(values)

    for i, (label, value) in enumerate(zip(labels, values)):
        if i > 0:
            imgui.same_line()

        changed, new_val = render_character_knob(label, value, size)
        if changed:
            any_changed = True
            changed_idx = i
            new_values[i] = new_val

    return any_changed, new_values, changed_idx


def render_preset_file_combo(
    preset_names: list,
    current_idx: int,
) -> Tuple[bool, int]:
    """
    Render preset file selector combo.

    Args:
        preset_names: List of preset display names
        current_idx: Currently selected index

    Returns:
        (changed, new_index) tuple
    """
    imgui.set_next_item_width(220)
    changed, new_idx = imgui.combo("##preset_file", current_idx, preset_names)
    return changed, new_idx


def render_connection_status(connected: bool, port: str) -> None:
    """
    Render connection status indicator.

    Args:
        connected: Connection state
        port: Port name
    """
    if connected:
        imgui.text_colored(imgui.ImVec4(0.2, 0.8, 0.2, 1.0), f"Connected: {port}")
    else:
        imgui.text_colored(imgui.ImVec4(0.8, 0.2, 0.2, 1.0), "Disconnected")
