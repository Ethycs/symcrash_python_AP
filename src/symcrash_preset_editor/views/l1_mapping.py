"""
L1 Mapping View - Gesture/Intent Matrix Editor
==============================================

Visual editor for L1 Lift layer:
- Gesture Basis (8 raw controller inputs)
- Intent Vector (8 semantic lanes)
- Gesture → Intent mapping matrix
- Intent → Family mapping
- Macro semantics

Per WSE-STD-GESTURE v0 (§1-3).
"""

from typing import Optional, Dict, List, Tuple
from imgui_bundle import imgui

from symcrash_python_ap.editor import EditorSession, SetParam


# Gesture basis fields (WSE-STD-GESTURE §1)
GESTURE_FIELDS = [
    ("Gate", "bool", "Note on/off gate"),
    ("Pressure", "Q15U", "Aftertouch / mod wheel"),
    ("Position", "Q15S", "CC74 / MPE slide"),
    ("Speed", "Q15U", "Derivative of Position"),
    ("Tilt", "Q15S", "MPE Y / CC1"),
    ("Chaos", "Q15U", "Macro M_GLITCH"),
    ("Pitch", "Q15S", "Pitch bend"),
    ("Pan", "Q15S", "CC10 / MPE X"),
]

# Intent lanes (WSE-STD-GESTURE §2)
INTENT_LANES = [
    ("I_TONE", "Pitched tone strength"),
    ("I_BOW", "Sustained pressure / bowing"),
    ("I_AIR", "Breathiness / noise desire"),
    ("I_GLITCH", "Artifact / glitch desire"),
    ("I_WARP", "Time/pitch warping desire"),
    ("I_SNAP", "Transient / attack desire"),
    ("I_GLUE", "Coherence / binding desire"),
    ("I_CALM", "Recovery / simplification desire"),
]

# Operator families (WSE-STD-GESTURE §8)
FAMILY_NAMES = [
    "F_RESERVED",
    "F_COLOR",
    "F_TRANSPORT",
    "F_LOCAL",
    "F_TOPOLOGY",
    "F_GLUE",
    "F_EVENTS",
    "F_CATASTROPHE",
]


class L1MappingView:
    """
    L1 Lift layer editor view.

    Features:
    - Gesture source binding table
    - Intent lane visualization
    - Mapping matrix (gesture → intent)
    - Family weight preview
    """

    def __init__(self, session: EditorSession):
        self.session = session

        # View state
        self._selected_intent: Optional[int] = None
        self._show_family_mapping = False

    def render(self, avail: imgui.ImVec2) -> None:
        """Render the L1 mapping view."""
        # Two columns: gesture/intent config, mapping matrix
        col_width = avail.x / 2 - 10

        imgui.begin_child("L1Left", imgui.ImVec2(col_width, avail.y - 30), imgui.ChildFlags_.borders)
        self._render_gesture_panel()
        imgui.separator()
        self._render_intent_panel()
        imgui.end_child()

        imgui.same_line()

        imgui.begin_child("L1Right", imgui.ImVec2(col_width, avail.y - 30), imgui.ChildFlags_.borders)
        self._render_mapping_matrix()
        imgui.separator()
        self._render_family_preview()
        imgui.end_child()

    def _render_gesture_panel(self) -> None:
        """Render gesture basis configuration."""
        imgui.text("Gesture Basis (8 Controller Inputs)")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Raw controller inputs mapped to Q15 fixed-point")

        imgui.separator()

        ir = self.session.ir
        gesture_basis = getattr(self.session.l1_lift, 'gesture_basis', {})

        # Table of gesture sources
        flags = (
            imgui.TableFlags_.borders_inner |
            imgui.TableFlags_.row_bg |
            imgui.TableFlags_.resizable
        )

        if imgui.begin_table("GestureBasisTable", 4, flags):
            imgui.table_setup_column("Field")
            imgui.table_setup_column("Type")
            imgui.table_setup_column("Source")
            imgui.table_setup_column("Range")
            imgui.table_headers_row()

            for field_name, field_type, desc in GESTURE_FIELDS:
                imgui.table_next_row()

                # Field name
                imgui.table_next_column()
                imgui.text(field_name)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(desc)

                # Type
                imgui.table_next_column()
                imgui.text_disabled(field_type)

                # Source binding
                imgui.table_next_column()
                source = gesture_basis.get(field_name, {})
                if isinstance(source, dict):
                    source_str = source.get('source', 'unbound')
                else:
                    source_str = str(source) if source else 'unbound'

                imgui.push_id(f"src_{field_name}")
                imgui.set_next_item_width(150)
                changed, new_src = imgui.input_text("##src", source_str, 256)
                if changed:
                    path = f"layers.L1_lift.gesture_basis.{field_name}.source"
                    self.session.execute(SetParam(path, new_src))
                imgui.pop_id()

                # Range
                imgui.table_next_column()
                if field_type == "Q15U":
                    imgui.text_disabled("[0, 32767]")
                elif field_type == "Q15S":
                    imgui.text_disabled("[-32768, 32767]")
                else:
                    imgui.text_disabled("[0, 1]")

            imgui.end_table()

    def _render_intent_panel(self) -> None:
        """Render intent vector configuration."""
        imgui.text("Intent Vector (8 Semantic Lanes)")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Semantic interpretation of gesture - what the voice wants to do")

        imgui.separator()

        # Intent lanes list
        for i, (lane_id, desc) in enumerate(INTENT_LANES):
            selected = self._selected_intent == i

            if imgui.selectable(f"{lane_id}", selected)[0]:
                self._selected_intent = i

            imgui.same_line(120)
            imgui.text_disabled(desc)

    def _render_mapping_matrix(self) -> None:
        """Render gesture → intent mapping matrix."""
        ir = self.session.ir
        intent_mapping = getattr(self.session.l1_lift, 'intent_mapping', {})
        mode = intent_mapping.get('mode', 'default') if isinstance(intent_mapping, dict) else 'default'

        imgui.text("Gesture → Intent Mapping")
        imgui.separator()

        # Mode selector
        modes = ["default", "custom"]
        current_idx = modes.index(mode) if mode in modes else 0

        imgui.set_next_item_width(150)
        changed, new_idx = imgui.combo("Mode", current_idx, modes)
        if changed:
            self.session.execute(SetParam("layers.L1_lift.intent_mapping.mode", modes[new_idx]))

        if mode == "default":
            imgui.text_disabled("Using WSE-STD-GESTURE default_gesture_to_intent()")
            imgui.spacing()

            # Show default mapping as read-only
            self._render_default_mapping_preview()

        else:
            imgui.text("Custom Mapping Expressions")
            imgui.spacing()

            custom_map = intent_mapping.get('custom_map', {}) if isinstance(intent_mapping, dict) else {}

            # Editable expressions for each intent lane
            for i, (lane_id, desc) in enumerate(INTENT_LANES):
                expr = custom_map.get(lane_id, "")

                imgui.push_id(f"expr_{lane_id}")
                imgui.text(f"{lane_id}:")
                imgui.same_line(80)
                imgui.set_next_item_width(imgui.get_content_region_avail().x - 20)

                changed, new_expr = imgui.input_text("##expr", expr, 512)
                if changed:
                    path = f"layers.L1_lift.intent_mapping.custom_map.{lane_id}"
                    self.session.execute(SetParam(path, new_expr))

                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"{desc}\nExpr: {expr if expr else '(empty)'}")

                imgui.pop_id()

    def _render_default_mapping_preview(self) -> None:
        """Show the default gesture→intent mapping as read-only."""
        # Per WSE-STD-GESTURE §3.2
        default_mappings = [
            ("I_TONE", "Gate ? mul_q15(Pressure, Q15_HALF) + Q15_HALF/4 : 0"),
            ("I_BOW", "Pressure"),
            ("I_AIR", "sat16((Speed >> 1) + max(0, Q15_HALF - Pressure) >> 1)"),
            ("I_GLITCH", "sat16(Chaos + (abs(Tilt) >> 1)) >> 1"),
            ("I_WARP", "sat16(abs(Pitch) + (abs(Position) >> 2))"),
            ("I_SNAP", "Gate && Speed > Q15_HALF ? Speed : 0"),
            ("I_GLUE", "max(0, Q15_ONE - Chaos)"),
            ("I_CALM", "(max(0, Q15_HALF - Pressure) + max(0, Q15_HALF - Speed)) >> 1"),
        ]

        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.6, 1.0))

        for lane_id, expr in default_mappings:
            imgui.bullet_text(f"{lane_id} = {expr}")

        imgui.pop_style_color()

    def _render_family_preview(self) -> None:
        """Render intent → family weight preview."""
        imgui.text("Intent → Family Weights")
        if imgui.is_item_hovered():
            imgui.set_tooltip("How intent lanes map to operator family weights")

        imgui.separator()

        ir = self.session.ir
        family_mapping = getattr(self.session.l1_lift, 'family_mapping', {})
        mode = family_mapping.get('mode', 'default') if isinstance(family_mapping, dict) else 'default'

        # Mode selector
        modes = ["default", "custom"]
        current_idx = modes.index(mode) if mode in modes else 0

        imgui.set_next_item_width(150)
        changed, new_idx = imgui.combo("Family Mode", current_idx, modes)
        if changed:
            self.session.execute(SetParam("layers.L1_lift.family_mapping.mode", modes[new_idx]))

        imgui.spacing()

        # Family weight bars (visualization)
        bar_width = imgui.get_content_region_avail().x - 100

        for i, family_name in enumerate(FAMILY_NAMES[1:], 1):  # Skip F_RESERVED
            # Mock weight value for visualization
            weight = 0.5 if i % 2 == 0 else 0.3

            imgui.text(f"{family_name}:")
            imgui.same_line(120)

            # Progress bar as weight visualization
            imgui.progress_bar(weight, imgui.ImVec2(bar_width, 16), f"{int(weight * 100)}%")
