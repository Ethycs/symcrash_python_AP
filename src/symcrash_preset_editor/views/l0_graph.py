"""
L0 Graph View - DSP Node Graph Editor
=====================================

Visual editor for L0 conventional DSP graph:
- Nodes: OSC, NOISE, MIXER, FILTER, AMP, FX_SEND
- Edges: Signal flow connections
- Properties panel for selected node

Uses imgui_node_editor for the graph canvas.
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from imgui_bundle import imgui, imgui_node_editor as ed

from symcrash_python_ap.editor import EditorSession, SetParam
from symcrash_python_ap.instrument_ir import L0Node, L0Edge, NodeType


# Node colors by type
NODE_COLORS = {
    "Osc": imgui.ImVec4(0.2, 0.6, 0.2, 1.0),       # Green
    "Noise": imgui.ImVec4(0.6, 0.4, 0.2, 1.0),     # Orange
    "Mixer": imgui.ImVec4(0.3, 0.3, 0.6, 1.0),     # Blue
    "Filter": imgui.ImVec4(0.6, 0.2, 0.6, 1.0),    # Purple
    "Amp": imgui.ImVec4(0.6, 0.6, 0.2, 1.0),       # Yellow
    "FXSend": imgui.ImVec4(0.2, 0.6, 0.6, 1.0),    # Cyan
}


@dataclass
class NodeState:
    """Runtime state for a graph node."""
    id: int                    # imgui_node_editor ID
    ir_node: L0Node           # Reference to IR node
    input_pin_id: int         # Input pin ID
    output_pin_id: int        # Output pin ID
    position: Tuple[float, float] = (0, 0)


class L0GraphView:
    """
    L0 DSP graph editor view.

    Features:
    - Node canvas with drag-and-drop
    - Connection drawing between pins
    - Property panel for selected node
    - Add/remove nodes via context menu
    """

    def __init__(self, session: EditorSession):
        self.session = session
        self._context: Optional[ed.EditorContext] = None
        self._node_states: Dict[str, NodeState] = {}
        self._link_id_counter = 1000
        self._node_id_counter = 1
        self._pin_id_counter = 2000

        # Selection state
        self._selected_node_id: Optional[str] = None

        # Initialize context
        self._init_context()
        self._sync_from_ir()

    def _init_context(self) -> None:
        """Initialize node editor context."""
        config = ed.Config()
        self._context = ed.create_editor(config)

    def _sync_from_ir(self) -> None:
        """Sync view state from InstrumentIR."""
        self._node_states.clear()

        # Create node states from IR
        x_offset = 50
        y_offset = 100
        x_spacing = 200

        for i, node in enumerate(self.session.layers.L0_graph.nodes):
            node_id = self._node_id_counter
            self._node_id_counter += 1
            input_pin = self._pin_id_counter
            self._pin_id_counter += 1
            output_pin = self._pin_id_counter
            self._pin_id_counter += 1

            self._node_states[node.id] = NodeState(
                id=node_id,
                ir_node=node,
                input_pin_id=input_pin,
                output_pin_id=output_pin,
                position=(x_offset + i * x_spacing, y_offset)
            )

    def render(self, avail: imgui.ImVec2) -> None:
        """Render the L0 graph view."""
        # Split into graph and properties
        graph_width = avail.x * 0.7
        props_width = avail.x * 0.3

        # Graph canvas
        imgui.begin_child("L0GraphCanvas", imgui.ImVec2(graph_width, avail.y - 30), imgui.ChildFlags_.borders)
        self._render_graph_canvas()
        imgui.end_child()

        imgui.same_line()

        # Properties panel
        imgui.begin_child("L0Properties", imgui.ImVec2(props_width - 10, avail.y - 30), imgui.ChildFlags_.borders)
        self._render_properties_panel()
        imgui.end_child()

    def _render_graph_canvas(self) -> None:
        """Render the node graph canvas."""
        ed.set_current_editor(self._context)
        ed.begin("L0Graph")

        # Render nodes
        for node_id, state in self._node_states.items():
            self._render_node(state)

        # Render links (edges)
        self._render_links()

        # Handle interactions
        self._handle_interactions()

        ed.end()

    def _render_node(self, state: NodeState) -> None:
        """Render a single graph node."""
        node = state.ir_node
        node_color = NODE_COLORS.get(node.type, imgui.ImVec4(0.4, 0.4, 0.4, 1.0))

        # Set position (only first time)
        ed.set_node_position(ed.NodeId(state.id), imgui.ImVec2(*state.position))

        # Node style
        ed.push_style_color(ed.StyleColor.node_bg, node_color)
        ed.push_style_color(ed.StyleColor.node_border, imgui.ImVec4(0.8, 0.8, 0.8, 1.0))

        ed.begin_node(ed.NodeId(state.id))

        # Header
        imgui.text(f"{node.type}: {node.id}")

        # Input pin
        ed.begin_pin(ed.PinId(state.input_pin_id), ed.PinKind.input)
        imgui.text("-> In")
        ed.end_pin()

        # Output pin
        imgui.same_line(100)
        ed.begin_pin(ed.PinId(state.output_pin_id), ed.PinKind.output)
        imgui.text("Out ->")
        ed.end_pin()

        # Brief params
        if node.type == "Osc":
            wf = node.params.waveform or "sine"
            imgui.text_disabled(f"Wave: {wf}")
        elif node.type == "Filter":
            mode = node.params.mode or "lp"
            imgui.text_disabled(f"Mode: {mode}")

        ed.end_node()
        ed.pop_style_color(2)

    def _render_links(self) -> None:
        """Render connection links between nodes."""
        ir = self.session.ir
        link_id = self._link_id_counter

        for edge in self.session.l0_graph.edges:
            # Parse src/dst (format: "NODE_ID.out" or "NODE_ID.in")
            src_node_id = edge.src.split('.')[0]
            dst_node_id = edge.dst.split('.')[0]

            src_state = self._node_states.get(src_node_id)
            dst_state = self._node_states.get(dst_node_id)

            if src_state and dst_state:
                ed.link(
                    ed.LinkId(link_id),
                    ed.PinId(src_state.output_pin_id),
                    ed.PinId(dst_state.input_pin_id),
                    imgui.ImVec4(0.8, 0.8, 0.8, 1.0),
                    2.0
                )
                link_id += 1

    def _handle_interactions(self) -> None:
        """Handle user interactions with the graph."""
        # Check for node selection
        if ed.has_selection_changed():
            selected_nodes = []
            node_count = ed.get_selected_object_count()
            if node_count > 0:
                # Find selected node
                for node_id, state in self._node_states.items():
                    if ed.is_node_selected(ed.NodeId(state.id)):
                        self._selected_node_id = node_id
                        break
            else:
                self._selected_node_id = None

        # Context menu for adding nodes
        if ed.show_background_context_menu():
            imgui.open_popup("AddNodeMenu")

        if imgui.begin_popup("AddNodeMenu"):
            imgui.text("Add Node")
            imgui.separator()

            for node_type in ["Osc", "Noise", "Mixer", "Filter", "Amp", "FXSend"]:
                if imgui.menu_item(node_type)[0]:
                    self._add_node(node_type)

            imgui.end_popup()

    def _add_node(self, node_type: str) -> None:
        """Add a new node to the graph."""
        from symcrash_python_ap.instrument_ir import L0Node, NodeParams

        # Generate unique ID
        existing_ids = {n.id for n in self.session.layers.L0_graph.nodes}
        base_id = f"N_{node_type.upper()}"
        node_id = base_id
        counter = 1
        while node_id in existing_ids:
            node_id = f"{base_id}_{counter}"
            counter += 1

        # Create node
        new_node = L0Node(
            id=node_id,
            type=node_type,
            params=NodeParams()
        )

        # Add to IR (direct mutation for now - should use command)
        self.session.layers.L0_graph.nodes.append(new_node)
        self._sync_from_ir()

    def _render_properties_panel(self) -> None:
        """Render the properties panel for selected node."""
        imgui.text("Properties")
        imgui.separator()

        if self._selected_node_id is None:
            imgui.text_disabled("Select a node to edit properties")
            return

        state = self._node_states.get(self._selected_node_id)
        if state is None:
            return

        node = state.ir_node
        params = node.params

        imgui.text(f"Node: {node.id}")
        imgui.text(f"Type: {node.type}")
        imgui.separator()

        # Type-specific properties
        if node.type == "Osc":
            self._render_osc_properties(params)
        elif node.type == "Filter":
            self._render_filter_properties(params)
        elif node.type == "Amp":
            self._render_amp_properties(params)
        elif node.type == "Noise":
            self._render_noise_properties(params)

    def _render_osc_properties(self, params) -> None:
        """Render oscillator properties."""
        imgui.text("Oscillator")

        # Waveform selector
        waveforms = ["sine", "saw", "pulse", "tri"]
        current_wf = params.waveform or "sine"
        current_idx = waveforms.index(current_wf) if current_wf in waveforms else 0

        changed, new_idx = imgui.combo("Waveform", current_idx, waveforms)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.waveform"
            self.session.execute(SetParam(path, waveforms[new_idx]))

        # Detune
        changed, new_val = imgui.slider_float("Detune (cents)", params.detune_cents, -100.0, 100.0)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.detune_cents"
            self.session.execute(SetParam(path, new_val))

        # Pulse width (if pulse)
        if params.waveform == "pulse":
            changed, new_val = imgui.slider_float("Pulse Width", params.pw, 0.1, 0.9)
            if changed:
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.pw"
                self.session.execute(SetParam(path, new_val))

    def _render_filter_properties(self, params) -> None:
        """Render filter properties."""
        imgui.text("Filter")

        # Filter mode
        modes = ["lp", "hp", "bp", "notch"]
        current_mode = params.mode or "lp"
        current_idx = modes.index(current_mode) if current_mode in modes else 0

        changed, new_idx = imgui.combo("Mode", current_idx, modes)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.mode"
            self.session.execute(SetParam(path, modes[new_idx]))

        # Cutoff (Q15)
        cutoff_pct = params.filter_cutoff_q15 / 32767.0 * 100.0
        changed, new_pct = imgui.slider_float("Cutoff", cutoff_pct, 0.0, 100.0, "%.1f%%")
        if changed:
            new_q15 = int(new_pct / 100.0 * 32767)
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.filter_cutoff_q15"
            self.session.execute(SetParam(path, new_q15))

        # Resonance (Q15)
        reso_pct = params.filter_resonance_q15 / 32767.0 * 100.0
        changed, new_pct = imgui.slider_float("Resonance", reso_pct, 0.0, 100.0, "%.1f%%")
        if changed:
            new_q15 = int(new_pct / 100.0 * 32767)
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.filter_resonance_q15"
            self.session.execute(SetParam(path, new_q15))

    def _render_amp_properties(self, params) -> None:
        """Render amplifier properties."""
        imgui.text("Amplifier")

        if params.adsr:
            adsr = params.adsr

            changed, new_val = imgui.slider_float("Attack (ms)", adsr.a_ms, 0.0, 1000.0)
            if changed:
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.adsr.a_ms"
                self.session.execute(SetParam(path, new_val))

            changed, new_val = imgui.slider_float("Decay (ms)", adsr.d_ms, 0.0, 2000.0)
            if changed:
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.adsr.d_ms"
                self.session.execute(SetParam(path, new_val))

            changed, new_val = imgui.slider_float("Sustain", adsr.s, 0.0, 1.0)
            if changed:
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.adsr.s"
                self.session.execute(SetParam(path, new_val))

            changed, new_val = imgui.slider_float("Release (ms)", adsr.r_ms, 0.0, 5000.0)
            if changed:
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.adsr.r_ms"
                self.session.execute(SetParam(path, new_val))

    def _render_noise_properties(self, params) -> None:
        """Render noise generator properties."""
        imgui.text("Noise Generator")

        colors = ["white", "pink", "brown"]
        current_color = params.color or "white"
        current_idx = colors.index(current_color) if current_color in colors else 0

        changed, new_idx = imgui.combo("Color", current_idx, colors)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.color"
            self.session.execute(SetParam(path, colors[new_idx]))

    def _get_node_index(self) -> int:
        """Get index of selected node in IR list."""
        if self._selected_node_id is None:
            return -1

        for i, node in enumerate(self.session.layers.L0_graph.nodes):
            if node.id == self._selected_node_id:
                return i
        return -1
