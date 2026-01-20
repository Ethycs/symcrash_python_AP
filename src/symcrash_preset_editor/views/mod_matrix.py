"""
Modulation Matrix View - Parameter Modulation Routing
=====================================================

Visual matrix editor for modulation routing:
- Sources: LFOs, Envelopes, Macros, Velocity, Key, Aftertouch
- Destinations: Filter cutoff, resonance, osc params, etc.
- Amount: Bipolar modulation depth per slot

Per WSE-STD-070: Modulation routing is part of L0/L1 abstraction.
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from imgui_bundle import imgui

from symcrash_python_ap.editor import EditorSession, SetParam
from symcrash_python_ap.instrument_ir import Macro


# Standard modulation sources
MOD_SOURCES = [
    ("None", "none"),
    ("LFO 1", "lfo1"),
    ("LFO 2", "lfo2"),
    ("Env 1 (Amp)", "env_amp"),
    ("Env 2 (Filter)", "env_filter"),
    ("Velocity", "velocity"),
    ("Key Track", "key_track"),
    ("Aftertouch", "aftertouch"),
    ("Mod Wheel", "mod_wheel"),
    ("Pitch Bend", "pitch_bend"),
    ("M_BRIGHT", "macro_bright"),
    ("M_GLITCH", "macro_glitch"),
    ("M_MORPH", "macro_morph"),
    ("M_TEXTURE", "macro_texture"),
]

# Standard modulation destinations
MOD_DESTINATIONS = [
    ("None", "none"),
    ("Filter Cutoff", "filter_cutoff"),
    ("Filter Resonance", "filter_resonance"),
    ("Osc 1 Pitch", "osc1_pitch"),
    ("Osc 2 Pitch", "osc2_pitch"),
    ("Osc Mix", "osc_mix"),
    ("PWM Width", "pwm_width"),
    ("FM Depth", "fm_depth"),
    ("Pan", "pan"),
    ("Volume", "volume"),
    ("LFO 1 Rate", "lfo1_rate"),
    ("LFO 2 Rate", "lfo2_rate"),
]

# Number of modulation slots
NUM_MOD_SLOTS = 8


@dataclass
class ModSlot:
    """A single modulation routing slot."""
    source: str = "none"
    destination: str = "none"
    amount: float = 0.0  # -1.0 to 1.0 bipolar


class ModMatrixView:
    """
    Modulation matrix editor view.

    Features:
    - 8 modulation slots
    - Source/destination dropdowns
    - Bipolar amount sliders
    - Quick macro editor
    """

    def __init__(self, session: EditorSession):
        self.session = session
        self._mod_slots: List[ModSlot] = [ModSlot() for _ in range(NUM_MOD_SLOTS)]

    def render(self, avail: imgui.ImVec2) -> None:
        """Render the modulation matrix view."""
        # Split into matrix and macros
        matrix_height = avail.y * 0.6
        macro_height = avail.y * 0.4 - 10

        # Modulation matrix
        imgui.begin_child("ModMatrix", imgui.ImVec2(avail.x, matrix_height), imgui.ChildFlags_.borders)
        self._render_mod_matrix()
        imgui.end_child()

        # Macro editor
        imgui.begin_child("MacroEditor", imgui.ImVec2(avail.x, macro_height), imgui.ChildFlags_.borders)
        self._render_macro_editor()
        imgui.end_child()

    def _render_mod_matrix(self) -> None:
        """Render the modulation routing matrix."""
        imgui.text("Modulation Matrix")
        imgui.separator()

        # Column headers
        imgui.columns(4, "ModMatrixColumns")
        imgui.set_column_width(0, 40)
        imgui.set_column_width(1, 150)
        imgui.set_column_width(2, 150)

        imgui.text("#")
        imgui.next_column()
        imgui.text("Source")
        imgui.next_column()
        imgui.text("Destination")
        imgui.next_column()
        imgui.text("Amount")
        imgui.next_column()

        imgui.separator()

        # Modulation slots
        source_names = [name for name, _ in MOD_SOURCES]
        dest_names = [name for name, _ in MOD_DESTINATIONS]

        for i, slot in enumerate(self._mod_slots):
            imgui.push_id(f"mod_slot_{i}")

            # Slot number
            imgui.text(f"{i + 1}")
            imgui.next_column()

            # Source selector
            source_idx = self._get_source_index(slot.source)
            imgui.set_next_item_width(-1)
            changed, new_idx = imgui.combo("##source", source_idx, source_names)
            if changed:
                slot.source = MOD_SOURCES[new_idx][1]
            imgui.next_column()

            # Destination selector
            dest_idx = self._get_dest_index(slot.destination)
            imgui.set_next_item_width(-1)
            changed, new_idx = imgui.combo("##dest", dest_idx, dest_names)
            if changed:
                slot.destination = MOD_DESTINATIONS[new_idx][1]
            imgui.next_column()

            # Amount slider (bipolar)
            imgui.set_next_item_width(-1)
            changed, new_amount = imgui.slider_float("##amount", slot.amount, -1.0, 1.0, "%.2f")
            if changed:
                slot.amount = new_amount
            imgui.next_column()

            imgui.pop_id()

        imgui.columns(1)

        imgui.spacing()
        imgui.text_disabled("Drag connections in the L0 graph to route audio signals.")
        imgui.text_disabled("Use this matrix for control-rate modulation routing.")

    def _render_macro_editor(self) -> None:
        """Render the macro parameter editor."""
        imgui.text("Macros")
        imgui.separator()

        macros = self.session.layers.L0_graph.macros

        if not macros:
            imgui.text_disabled("No macros defined.")
            imgui.spacing()
            if imgui.button("Add Default Macros"):
                self._add_default_macros()
            return

        # Render each macro
        for i, macro in enumerate(macros):
            imgui.push_id(f"macro_{i}")

            # Macro header
            expanded, _ = imgui.collapsing_header(f"{macro.name} ({macro.id})")

            if expanded:
                imgui.indent()

                # ID (read-only display)
                imgui.text(f"ID: {macro.id}")

                # Name editor
                imgui.set_next_item_width(200)
                changed, new_name = imgui.input_text("Name", macro.name, 64)
                if changed:
                    path = f"layers.L0_graph.macros[{i}].name"
                    self.session.execute(SetParam(path, new_name))

                # Range display
                imgui.text(f"Range: {macro.range[0]:.2f} - {macro.range[1]:.2f}")

                # Default value
                imgui.set_next_item_width(200)
                changed, new_default = imgui.slider_float(
                    "Default",
                    macro.default,
                    macro.range[0],
                    macro.range[1]
                )
                if changed:
                    path = f"layers.L0_graph.macros[{i}].default"
                    self.session.execute(SetParam(path, new_default))

                # Delete button
                imgui.spacing()
                if imgui.button("Delete Macro"):
                    self._delete_macro(i)

                imgui.unindent()

            imgui.pop_id()

        imgui.spacing()
        imgui.separator()

        # Add macro button
        if imgui.button("+ Add Macro"):
            imgui.open_popup("AddMacroPopup")

        self._render_add_macro_popup()

    def _render_add_macro_popup(self) -> None:
        """Render popup for adding a new macro."""
        if imgui.begin_popup("AddMacroPopup"):
            imgui.text("Add Macro")
            imgui.separator()

            # Preset macros
            preset_macros = [
                ("M_BRIGHT", "Brightness"),
                ("M_GLITCH", "Glitch"),
                ("M_MORPH", "Morph"),
                ("M_TEXTURE", "Texture"),
                ("M_DETUNE", "Detune"),
                ("M_ATTACK", "Attack"),
                ("M_DECAY", "Decay"),
                ("M_SUSTAIN", "Sustain"),
                ("M_RELEASE", "Release"),
            ]

            for macro_id, macro_name in preset_macros:
                # Check if already exists
                exists = any(m.id == macro_id for m in self.session.layers.L0_graph.macros)

                if exists:
                    imgui.text_disabled(f"{macro_name} (exists)")
                else:
                    if imgui.menu_item(macro_name)[0]:
                        self._add_macro(macro_id, macro_name)

            imgui.separator()
            if imgui.menu_item("Custom...")[0]:
                self._add_macro("M_CUSTOM", "Custom")

            imgui.end_popup()

    def _add_macro(self, macro_id: str, macro_name: str) -> None:
        """Add a new macro to the L0 graph."""
        # Generate unique ID if needed
        existing_ids = {m.id for m in self.session.layers.L0_graph.macros}
        final_id = macro_id
        counter = 1
        while final_id in existing_ids:
            final_id = f"{macro_id}_{counter}"
            counter += 1

        new_macro = Macro(
            id=final_id,
            name=macro_name,
            range=(0.0, 1.0),
            default=0.5
        )
        self.session.layers.L0_graph.macros.append(new_macro)

    def _delete_macro(self, index: int) -> None:
        """Delete a macro by index."""
        macros = self.session.layers.L0_graph.macros
        if 0 <= index < len(macros):
            macros.pop(index)

    def _add_default_macros(self) -> None:
        """Add a default set of macros."""
        default_macros = [
            ("M_BRIGHT", "Brightness", 0.5),
            ("M_GLITCH", "Glitch", 0.0),
            ("M_MORPH", "Morph", 0.5),
            ("M_TEXTURE", "Texture", 0.5),
        ]

        for macro_id, name, default in default_macros:
            self.session.layers.L0_graph.macros.append(
                Macro(id=macro_id, name=name, range=(0.0, 1.0), default=default)
            )

    def _get_source_index(self, source_id: str) -> int:
        """Get index of source in MOD_SOURCES list."""
        for i, (_, src_id) in enumerate(MOD_SOURCES):
            if src_id == source_id:
                return i
        return 0

    def _get_dest_index(self, dest_id: str) -> int:
        """Get index of destination in MOD_DESTINATIONS list."""
        for i, (_, dst_id) in enumerate(MOD_DESTINATIONS):
            if dst_id == dest_id:
                return i
        return 0
