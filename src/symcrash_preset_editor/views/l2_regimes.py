"""
L2 Regime View - State Machine Editor
=====================================

Visual editor for L2 Regime graph:
- Regime nodes (R0-R6): Idle, Attack, Stable, Air, Glitch, Catastrophe, Recovery
- Transitions with intent-driven guards and hysteresis
- Arc specifications for transition modulators

Uses imgui_node_editor for state machine visualization.
Per WSE-STD-GESTURE v0 (§4-6).
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from imgui_bundle import imgui, imgui_node_editor as ed

from symcrash_python_ap.editor import (
    EditorSession, SetParam, AddRegime, RemoveRegime,
    AddTransition, RemoveTransition
)


# Regime kinds and their colors
REGIME_COLORS = {
    "Idle": imgui.ImVec4(0.3, 0.3, 0.3, 1.0),          # Gray
    "Transient": imgui.ImVec4(0.8, 0.6, 0.2, 1.0),    # Orange
    "StableTone": imgui.ImVec4(0.2, 0.7, 0.3, 1.0),   # Green
    "NoiseTexture": imgui.ImVec4(0.5, 0.5, 0.7, 1.0), # Blue-gray
    "GlitchChatter": imgui.ImVec4(0.8, 0.3, 0.3, 1.0),# Red
    "Catastrophe": imgui.ImVec4(0.9, 0.2, 0.5, 1.0),  # Magenta
    "Recovery": imgui.ImVec4(0.3, 0.6, 0.8, 1.0),     # Cyan
}

# Edge type colors
EDGE_COLORS = {
    "HARD": imgui.ImVec4(0.9, 0.9, 0.9, 1.0),
    "SOFT": imgui.ImVec4(0.6, 0.8, 0.6, 1.0),
    "CHAOTIC": imgui.ImVec4(0.9, 0.5, 0.3, 1.0),
    "RECOVERY": imgui.ImVec4(0.3, 0.7, 0.9, 1.0),
}


@dataclass
class RegimeNodeState:
    """Runtime state for a regime node in the graph."""
    id: int                      # imgui_node_editor ID
    regime_id: str              # IR regime ID (e.g., "R0_IDLE")
    kind: str                   # RegimeKind
    input_pin_id: int
    output_pin_id: int
    position: Tuple[float, float]


class L2RegimeView:
    """
    L2 Regime state machine editor.

    Features:
    - State machine canvas with regime nodes
    - Transition arrows with guards
    - Properties panel for selected regime/transition
    - Add/remove regimes and transitions
    """

    def __init__(self, session: EditorSession):
        self.session = session
        self._context: Optional[ed.EditorContext] = None
        self._node_states: Dict[str, RegimeNodeState] = {}
        self._node_id_counter = 5000
        self._pin_id_counter = 6000
        self._link_id_counter = 7000

        # Selection
        self._selected_regime: Optional[str] = None
        self._selected_transition: Optional[Tuple[str, str]] = None

        self._init_context()
        self._sync_from_ir()

    def _init_context(self) -> None:
        """Initialize node editor context."""
        config = ed.Config()
        self._context = ed.create_editor(config)

    def _sync_from_ir(self) -> None:
        """Sync view state from InstrumentIR."""
        self._node_states.clear()
        ir = self.session.ir

        # Layout regime nodes in a circle
        import math
        nodes = self.session.l2_regimes.regime_nodes
        n = len(nodes)
        center_x, center_y = 400, 300
        radius = 200

        for i, regime in enumerate(nodes):
            angle = (2 * math.pi * i / n) - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            node_id = self._node_id_counter
            self._node_id_counter += 1
            input_pin = self._pin_id_counter
            self._pin_id_counter += 1
            output_pin = self._pin_id_counter
            self._pin_id_counter += 1

            self._node_states[regime.id] = RegimeNodeState(
                id=node_id,
                regime_id=regime.id,
                kind=regime.kind,
                input_pin_id=input_pin,
                output_pin_id=output_pin,
                position=(x, y)
            )

    def render(self, avail: imgui.ImVec2) -> None:
        """Render the L2 regime view."""
        # Split: graph canvas and properties
        graph_width = avail.x * 0.65
        props_width = avail.x * 0.35

        imgui.begin_child("L2GraphCanvas", imgui.ImVec2(graph_width, avail.y - 30), imgui.ChildFlags_.borders)
        self._render_state_machine()
        imgui.end_child()

        imgui.same_line()

        imgui.begin_child("L2Properties", imgui.ImVec2(props_width - 10, avail.y - 30), imgui.ChildFlags_.borders)
        self._render_properties_panel()
        imgui.end_child()

    def _render_state_machine(self) -> None:
        """Render the regime state machine graph."""
        ed.set_current_editor(self._context)
        ed.begin("L2Regimes")

        # Render regime nodes
        for regime_id, state in self._node_states.items():
            self._render_regime_node(state)

        # Render transitions
        self._render_transitions()

        # Handle interactions
        self._handle_interactions()

        ed.end()

    def _render_regime_node(self, state: RegimeNodeState) -> None:
        """Render a single regime node."""
        ir = self.session.ir

        # Find the regime in IR
        regime = None
        for r in self.session.l2_regimes.regime_nodes:
            if r.id == state.regime_id:
                regime = r
                break

        if regime is None:
            return

        color = REGIME_COLORS.get(regime.kind, imgui.ImVec4(0.4, 0.4, 0.4, 1.0))

        # Set position
        ed.set_node_position(ed.NodeId(state.id), imgui.ImVec2(*state.position))

        # Node style
        ed.push_style_color(ed.StyleColor.node_bg, color)
        ed.push_style_color(ed.StyleColor.node_border, imgui.ImVec4(0.8, 0.8, 0.8, 1.0))
        ed.push_style_var(ed.StyleVar.node_rounding, 20.0)  # Rounded for state machine

        ed.begin_node(ed.NodeId(state.id))

        # Input pin (incoming transitions)
        ed.begin_pin(ed.PinId(state.input_pin_id), ed.PinKind.input)
        imgui.text("")  # Empty - just need the pin
        ed.end_pin()

        imgui.same_line()

        # Node content
        imgui.begin_group()
        imgui.text(regime.id)
        imgui.text_disabled(regime.kind)

        if regime.dwell_ms:
            imgui.text_disabled(f"dwell: {regime.dwell_ms}ms")

        imgui.end_group()

        imgui.same_line()

        # Output pin (outgoing transitions)
        ed.begin_pin(ed.PinId(state.output_pin_id), ed.PinKind.output)
        imgui.text("")
        ed.end_pin()

        ed.end_node()
        ed.pop_style_var(1)
        ed.pop_style_color(2)

    def _render_transitions(self) -> None:
        """Render transition arrows between regimes."""
        ir = self.session.ir
        link_id = self._link_id_counter

        for trans in self.session.l2_regimes.transitions:
            src_regime = trans.from_regime
            dst_regime = trans.to

            # Handle "any" source
            if src_regime == "any":
                # Draw from all regimes
                for src_id in self._node_states:
                    if src_id != dst_regime:
                        self._draw_transition_link(src_id, dst_regime, trans, link_id)
                        link_id += 1
            else:
                src_state = self._node_states.get(src_regime)
                dst_state = self._node_states.get(dst_regime)

                if src_state and dst_state:
                    self._draw_transition_link(src_regime, dst_regime, trans, link_id)
                    link_id += 1

    def _draw_transition_link(self, src_id: str, dst_id: str, trans, link_id: int) -> None:
        """Draw a single transition link."""
        src_state = self._node_states.get(src_id)
        dst_state = self._node_states.get(dst_id)

        if not src_state or not dst_state:
            return

        # Get edge color based on type
        edge_type = getattr(trans, 'edge_type', 'SOFT')
        if isinstance(edge_type, str):
            color = EDGE_COLORS.get(edge_type, EDGE_COLORS["SOFT"])
        else:
            color = EDGE_COLORS["SOFT"]

        ed.link(
            ed.LinkId(link_id),
            ed.PinId(src_state.output_pin_id),
            ed.PinId(dst_state.input_pin_id),
            color,
            2.0
        )

    def _handle_interactions(self) -> None:
        """Handle user interactions."""
        # Node selection
        if ed.has_selection_changed():
            for regime_id, state in self._node_states.items():
                if ed.is_node_selected(ed.NodeId(state.id)):
                    self._selected_regime = regime_id
                    self._selected_transition = None
                    break
            else:
                self._selected_regime = None

        # Context menu for adding regimes
        if ed.show_background_context_menu():
            imgui.open_popup("AddRegimeMenu")

        if imgui.begin_popup("AddRegimeMenu"):
            imgui.text("Add Regime")
            imgui.separator()

            for kind in REGIME_COLORS.keys():
                if imgui.menu_item(kind)[0]:
                    self._add_regime(kind)

            imgui.end_popup()

        # Context menu for nodes
        node_id = ed.NodeId()
        if ed.show_node_context_menu(node_id):
            # Find which regime this is
            for regime_id, state in self._node_states.items():
                if state.id == node_id.get():
                    self._selected_regime = regime_id
                    imgui.open_popup("RegimeContextMenu")
                    break

        if imgui.begin_popup("RegimeContextMenu"):
            if imgui.menu_item("Delete Regime")[0]:
                if self._selected_regime:
                    self.session.execute(RemoveRegime(self._selected_regime))
                    self._sync_from_ir()
                    self._selected_regime = None

            imgui.separator()

            if imgui.begin_menu("Add Transition To..."):
                for regime_id in self._node_states:
                    if regime_id != self._selected_regime:
                        if imgui.menu_item(regime_id)[0]:
                            self.session.execute(
                                AddTransition(self._selected_regime, regime_id, "true")
                            )
                imgui.end_menu()

            imgui.end_popup()

    def _add_regime(self, kind: str) -> None:
        """Add a new regime."""
        # Generate unique ID
        existing = {r.id for r in self.session.l2_regimes.regime_nodes}
        base_id = f"R{len(existing)}_{kind.upper()}"
        regime_id = base_id
        counter = 1
        while regime_id in existing:
            regime_id = f"{base_id}_{counter}"
            counter += 1

        self.session.execute(AddRegime(regime_id, kind, f"T_{kind.upper()}"))
        self._sync_from_ir()

    def _render_properties_panel(self) -> None:
        """Render properties panel."""
        imgui.text("Properties")
        imgui.separator()

        if self._selected_regime:
            self._render_regime_properties()
        elif self._selected_transition:
            self._render_transition_properties()
        else:
            imgui.text_disabled("Select a regime or transition")

    def _render_regime_properties(self) -> None:
        """Render properties for selected regime."""
        ir = self.session.ir

        regime = None
        regime_idx = -1
        for i, r in enumerate(self.session.l2_regimes.regime_nodes):
            if r.id == self._selected_regime:
                regime = r
                regime_idx = i
                break

        if regime is None:
            return

        imgui.text(f"Regime: {regime.id}")
        imgui.separator()

        # Kind selector
        kinds = list(REGIME_COLORS.keys())
        current_idx = kinds.index(regime.kind) if regime.kind in kinds else 0

        changed, new_idx = imgui.combo("Kind", current_idx, kinds)
        if changed:
            path = f"layers.L2_regimes.regime_nodes[{regime_idx}].kind"
            self.session.execute(SetParam(path, kinds[new_idx]))

        # Emit template
        changed, new_template = imgui.input_text(
            "Emit Template",
            regime.emit_template,
            256
        )
        if changed:
            path = f"layers.L2_regimes.regime_nodes[{regime_idx}].emit_template"
            self.session.execute(SetParam(path, new_template))

        # Dwell time
        dwell = regime.dwell_ms or 0
        changed, new_dwell = imgui.input_int("Dwell (ms)", dwell)
        if changed:
            path = f"layers.L2_regimes.regime_nodes[{regime_idx}].dwell_ms"
            self.session.execute(SetParam(path, new_dwell if new_dwell > 0 else None))

        imgui.separator()
        imgui.text("Outgoing Transitions:")

        # List transitions from this regime
        for i, trans in enumerate(self.session.l2_regimes.transitions):
            if trans.from_regime == self._selected_regime:
                imgui.bullet_text(f"→ {trans.to}")
                imgui.same_line()
                imgui.text_disabled(f"({trans.when})")

    def _render_transition_properties(self) -> None:
        """Render properties for selected transition."""
        src, dst = self._selected_transition
        imgui.text(f"Transition: {src} → {dst}")
        imgui.separator()

        ir = self.session.ir

        # Find the transition
        trans = None
        trans_idx = -1
        for i, t in enumerate(self.session.l2_regimes.transitions):
            if t.from_regime == src and t.to == dst:
                trans = t
                trans_idx = i
                break

        if trans is None:
            return

        # Guard condition
        changed, new_when = imgui.input_text("Guard (when)", trans.when, 256)
        if changed:
            path = f"layers.L2_regimes.transitions[{trans_idx}].when"
            self.session.execute(SetParam(path, new_when))

        # Hysteresis
        if trans.hysteresis:
            imgui.text("Hysteresis:")
            imgui.indent()

            changed, new_enter = imgui.input_float(
                "Enter threshold",
                trans.hysteresis.enter
            )
            if changed:
                path = f"layers.L2_regimes.transitions[{trans_idx}].hysteresis.enter"
                self.session.execute(SetParam(path, new_enter))

            changed, new_exit = imgui.input_float(
                "Exit threshold",
                trans.hysteresis.exit
            )
            if changed:
                path = f"layers.L2_regimes.transitions[{trans_idx}].hysteresis.exit"
                self.session.execute(SetParam(path, new_exit))

            imgui.unindent()

        imgui.separator()

        # Delete button
        if imgui.button("Delete Transition"):
            self.session.execute(RemoveTransition(src, dst))
            self._selected_transition = None
