"""
Level Selector - Tower Level Tab Bar
====================================

Renders tabs for each tower level (L0-L5).
Levels above the current tower_level are shown as disabled.
"""

from enum import IntEnum
from imgui_bundle import imgui


class TowerLevel(IntEnum):
    """Tower levels per WSE-STD-070 §3."""
    L0_PATCH = 0      # Conventional DSP graph
    L1_LIFT = 1       # Gesture + Intent mapping
    L2_REGIMES = 2    # Regime state machine
    L3_COCKPIT = 3    # CMMM emit templates
    L4_FORGE = 4      # Full graph editing
    L5_LAB = 5        # AST / new operators

    @property
    def display_name(self) -> str:
        names = {
            TowerLevel.L0_PATCH: "L0 Patch",
            TowerLevel.L1_LIFT: "L1 Lift",
            TowerLevel.L2_REGIMES: "L2 Regimes",
            TowerLevel.L3_COCKPIT: "L3 Cockpit",
            TowerLevel.L4_FORGE: "L4 Forge",
            TowerLevel.L5_LAB: "L5 Lab",
        }
        return names.get(self, f"L{self.value}")

    @property
    def description(self) -> str:
        descs = {
            TowerLevel.L0_PATCH: "DSP graph: OSC → MIX → FILTER → AMP → FX",
            TowerLevel.L1_LIFT: "Gesture → Intent mapping, macro semantics",
            TowerLevel.L2_REGIMES: "Regime state machine (R0-R6), transitions",
            TowerLevel.L3_COCKPIT: "CMMM emit templates, overlays",
            TowerLevel.L4_FORGE: "Full graph editing, custom transitions",
            TowerLevel.L5_LAB: "New operators, AST manipulation",
        }
        return descs.get(self, "")


def render_level_selector(
    current_level: TowerLevel,
    authored_level: int = 0,
) -> TowerLevel:
    """
    Render tower level selector tabs.

    Args:
        current_level: Currently selected level
        authored_level: Highest explicitly authored level (from preset)

    Returns:
        Selected level (may be unchanged)
    """
    selected = current_level

    if imgui.begin_tab_bar("TowerLevelTabs"):
        for level in TowerLevel:
            # Levels above authored_level are available but shown differently
            is_authored = level.value <= authored_level

            # Style for unauthored levels
            if not is_authored:
                imgui.push_style_color(
                    imgui.Col_.tab,
                    imgui.ImVec4(0.2, 0.2, 0.2, 1.0)
                )
                imgui.push_style_color(
                    imgui.Col_.tab_hovered,
                    imgui.ImVec4(0.3, 0.3, 0.3, 1.0)
                )

            # Render tab
            label = level.display_name
            if not is_authored:
                label += " (empty)"

            # begin_tab_item returns (selected, p_open) tuple
            # Don't use set_selected flag - it causes flickering
            tab_result = imgui.begin_tab_item(label)
            if tab_result[0]:
                # Tab is selected/active
                selected = level
                imgui.end_tab_item()

            # Tooltip
            if imgui.is_item_hovered():
                imgui.set_tooltip(level.description)

            # Pop style
            if not is_authored:
                imgui.pop_style_color(2)

        imgui.end_tab_bar()

    return selected
