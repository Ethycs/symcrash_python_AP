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
    "Osc": imgui.ImVec4(0.2, 0.6, 0.2, 1.0),       # Green - Mode A/B oscillator
    "FM6": imgui.ImVec4(0.8, 0.4, 0.2, 1.0),       # Orange-red - 6-op FM
    "Noise": imgui.ImVec4(0.5, 0.5, 0.3, 1.0),     # Olive
    "Mixer": imgui.ImVec4(0.3, 0.3, 0.6, 1.0),     # Blue
    "Filter": imgui.ImVec4(0.6, 0.2, 0.6, 1.0),    # Purple
    "Amp": imgui.ImVec4(0.6, 0.6, 0.2, 1.0),       # Yellow
    "FXSend": imgui.ImVec4(0.2, 0.6, 0.6, 1.0),    # Cyan
}

# All available node types for the add menu
NODE_TYPES = ["Osc", "FM6", "Noise", "Mixer", "Filter", "Amp", "FXSend"]

# Oscillator modes (maps to OSC_MODE in VM)
OSC_MODES = [
    ("Mode A (LUT)", 0),
    ("Mode B (Rotator)", 1),
    ("Sample", 2),
    ("FM6", 3),
]

# Waveform types
WAVEFORMS = ["sine", "saw", "pulse", "tri", "saw_down", "square"]

# Filter types
FILTER_TYPES = [
    ("Lowpass", 0),
    ("Highpass", 1),
    ("Bandpass", 2),
    ("Notch", 3),
]

# Noise colors
NOISE_COLORS = ["white", "pink", "brown"]


@dataclass
class NodeState:
    """Runtime state for a graph node."""
    id: int                    # imgui_node_editor ID
    ir_node: L0Node           # Reference to IR node
    input_pin_id: int         # Input pin ID
    output_pin_id: int        # Output pin ID
    position: Tuple[float, float] = (0, 0)


@dataclass
class LinkState:
    """Runtime state for a graph link/edge."""
    id: int                    # imgui_node_editor link ID
    edge_index: int           # Index in IR edges list
    src_node_id: str          # Source node ID
    dst_node_id: str          # Destination node ID


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
        self._link_states: Dict[int, LinkState] = {}  # link_id -> LinkState
        self._link_id_counter = 1000
        self._node_id_counter = 1
        self._pin_id_counter = 2000

        # Selection state
        self._selected_node_id: Optional[str] = None
        self._selected_link_id: Optional[int] = None

        # Pin ID to node ID mapping (for link creation)
        self._pin_to_node: Dict[int, str] = {}
        self._pin_is_output: Dict[int, bool] = {}

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
        self._link_states.clear()
        self._pin_to_node.clear()
        self._pin_is_output.clear()

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

            # Track pin to node mappings
            self._pin_to_node[input_pin] = node.id
            self._pin_to_node[output_pin] = node.id
            self._pin_is_output[input_pin] = False
            self._pin_is_output[output_pin] = True

        # Create link states from IR edges
        for i, edge in enumerate(self.session.layers.L0_graph.edges):
            src_node_id = edge.src.split('.')[0]
            dst_node_id = edge.dst.split('.')[0]

            link_id = self._link_id_counter
            self._link_id_counter += 1

            self._link_states[link_id] = LinkState(
                id=link_id,
                edge_index=i,
                src_node_id=src_node_id,
                dst_node_id=dst_node_id
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
            mode_names = {0: "A", 1: "B", 2: "Smp", 3: "FM6"}
            mode_str = mode_names.get(node.params.osc_mode, "A")
            imgui.text_disabled(f"{mode_str}: {wf}")
        elif node.type == "FM6":
            alg = node.params.algorithm or 1
            fb = node.params.feedback or 0
            imgui.text_disabled(f"Alg {alg}, FB {fb}")
        elif node.type == "Filter":
            mode = node.params.mode or "lp"
            imgui.text_disabled(f"Mode: {mode}")
        elif node.type == "Noise":
            color = node.params.color or "white"
            imgui.text_disabled(f"Color: {color}")

        ed.end_node()
        ed.pop_style_color(2)

    def _render_links(self) -> None:
        """Render connection links between nodes."""
        for link_id, link_state in self._link_states.items():
            src_state = self._node_states.get(link_state.src_node_id)
            dst_state = self._node_states.get(link_state.dst_node_id)

            if src_state and dst_state:
                # Highlight selected link
                if self._selected_link_id == link_id:
                    color = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)  # Gold for selected
                    thickness = 3.0
                else:
                    color = imgui.ImVec4(0.8, 0.8, 0.8, 1.0)
                    thickness = 2.0

                ed.link(
                    ed.LinkId(link_id),
                    ed.PinId(src_state.output_pin_id),
                    ed.PinId(dst_state.input_pin_id),
                    color,
                    thickness
                )

    def _handle_interactions(self) -> None:
        """Handle user interactions with the graph."""
        # Check for node selection
        if ed.has_selection_changed():
            self._selected_node_id = None
            self._selected_link_id = None

            # Find selected node
            for node_id, state in self._node_states.items():
                if ed.is_node_selected(ed.NodeId(state.id)):
                    self._selected_node_id = node_id
                    break

            # Find selected link
            for link_id in self._link_states.keys():
                if ed.is_link_selected(ed.LinkId(link_id)):
                    self._selected_link_id = link_id
                    break

        # Handle link creation (dragging from output to input)
        if ed.begin_create():
            start_pin_id = ed.PinId()
            end_pin_id = ed.PinId()

            if ed.query_new_link(start_pin_id, end_pin_id):
                start_id = start_pin_id.get()
                end_id = end_pin_id.get()

                # Validate: start must be output, end must be input
                start_is_output = self._pin_is_output.get(start_id, False)
                end_is_output = self._pin_is_output.get(end_id, True)

                if start_is_output and not end_is_output:
                    # Valid connection direction
                    if ed.accept_new_item(imgui.ImVec4(0.2, 0.8, 0.2, 1.0), 2.0):
                        src_node = self._pin_to_node.get(start_id)
                        dst_node = self._pin_to_node.get(end_id)
                        if src_node and dst_node:
                            self._add_edge(src_node, dst_node)
                elif not start_is_output and end_is_output:
                    # Reversed direction - swap and accept
                    if ed.accept_new_item(imgui.ImVec4(0.2, 0.8, 0.2, 1.0), 2.0):
                        src_node = self._pin_to_node.get(end_id)
                        dst_node = self._pin_to_node.get(start_id)
                        if src_node and dst_node:
                            self._add_edge(src_node, dst_node)
                else:
                    # Invalid (both inputs or both outputs)
                    ed.reject_new_item(imgui.ImVec4(0.8, 0.2, 0.2, 1.0), 2.0)

            ed.end_create()

        # Handle link/node deletion
        if ed.begin_delete():
            # Check for deleted links
            deleted_link_id = ed.LinkId()
            while ed.query_deleted_link(deleted_link_id):
                if ed.accept_deleted_item():
                    link_id = deleted_link_id.get()
                    self._delete_edge(link_id)

            # Check for deleted nodes
            deleted_node_id = ed.NodeId()
            while ed.query_deleted_node(deleted_node_id):
                if ed.accept_deleted_item():
                    node_ed_id = deleted_node_id.get()
                    self._delete_node_by_ed_id(node_ed_id)

            ed.end_delete()

        # Context menu for adding nodes (background)
        if ed.show_background_context_menu():
            imgui.open_popup("AddNodeMenu")

        # Context menu for nodes
        if ed.show_node_context_menu(ed.NodeId()):
            imgui.open_popup("NodeContextMenu")

        # Context menu for links
        if ed.show_link_context_menu(ed.LinkId()):
            imgui.open_popup("LinkContextMenu")

        # Add node popup
        if imgui.begin_popup("AddNodeMenu"):
            imgui.text("Add Node")
            imgui.separator()

            for node_type in NODE_TYPES:
                if imgui.menu_item(node_type)[0]:
                    self._add_node(node_type)

            imgui.end_popup()

        # Node context menu
        if imgui.begin_popup("NodeContextMenu"):
            if self._selected_node_id:
                imgui.text(f"Node: {self._selected_node_id}")
                imgui.separator()
                if imgui.menu_item("Delete Node", "Del")[0]:
                    self._delete_node(self._selected_node_id)
            imgui.end_popup()

        # Link context menu
        if imgui.begin_popup("LinkContextMenu"):
            if self._selected_link_id:
                link_state = self._link_states.get(self._selected_link_id)
                if link_state:
                    imgui.text(f"Connection: {link_state.src_node_id} -> {link_state.dst_node_id}")
                    imgui.separator()
                    if imgui.menu_item("Delete Connection", "Del")[0]:
                        self._delete_edge(self._selected_link_id)
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

    def _add_edge(self, src_node_id: str, dst_node_id: str) -> None:
        """Add a new edge between nodes."""
        # Check if edge already exists
        for edge in self.session.layers.L0_graph.edges:
            existing_src = edge.src.split('.')[0]
            existing_dst = edge.dst.split('.')[0]
            if existing_src == src_node_id and existing_dst == dst_node_id:
                return  # Edge already exists

        # Create new edge
        new_edge = L0Edge(
            src=f"{src_node_id}.out",
            dst=f"{dst_node_id}.in"
        )
        self.session.layers.L0_graph.edges.append(new_edge)
        self._sync_from_ir()

    def _delete_edge(self, link_id: int) -> None:
        """Delete an edge by link ID."""
        link_state = self._link_states.get(link_id)
        if link_state is None:
            return

        # Find and remove the edge from IR
        edges = self.session.layers.L0_graph.edges
        for i, edge in enumerate(edges):
            src_node = edge.src.split('.')[0]
            dst_node = edge.dst.split('.')[0]
            if src_node == link_state.src_node_id and dst_node == link_state.dst_node_id:
                edges.pop(i)
                break

        self._selected_link_id = None
        self._sync_from_ir()

    def _delete_node(self, node_id: str) -> None:
        """Delete a node and its connections."""
        # Remove all edges connected to this node
        edges = self.session.layers.L0_graph.edges
        edges_to_remove = []
        for i, edge in enumerate(edges):
            src_node = edge.src.split('.')[0]
            dst_node = edge.dst.split('.')[0]
            if src_node == node_id or dst_node == node_id:
                edges_to_remove.append(i)

        # Remove edges in reverse order to maintain indices
        for i in reversed(edges_to_remove):
            edges.pop(i)

        # Remove the node
        nodes = self.session.layers.L0_graph.nodes
        for i, node in enumerate(nodes):
            if node.id == node_id:
                nodes.pop(i)
                break

        self._selected_node_id = None
        self._sync_from_ir()

    def _delete_node_by_ed_id(self, ed_node_id: int) -> None:
        """Delete a node by its editor ID."""
        for node_id, state in self._node_states.items():
            if state.id == ed_node_id:
                self._delete_node(node_id)
                return

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
        elif node.type == "FM6":
            self._render_fm6_properties(params)
        elif node.type == "Filter":
            self._render_filter_properties(params)
        elif node.type == "Amp":
            self._render_amp_properties(params)
        elif node.type == "Noise":
            self._render_noise_properties(params)
        elif node.type == "Mixer":
            self._render_mixer_properties(params)
        elif node.type == "FXSend":
            self._render_fxsend_properties(params)

    def _render_osc_properties(self, params) -> None:
        """Render oscillator properties."""
        imgui.text("Oscillator")

        # OSC Mode selector
        osc_mode_names = [name for name, _ in OSC_MODES]
        current_mode_idx = params.osc_mode
        changed, new_idx = imgui.combo("OSC Mode", current_mode_idx, osc_mode_names)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.osc_mode"
            self.session.execute(SetParam(path, new_idx))

        imgui.separator()

        # Waveform selector
        current_wf = params.waveform or "sine"
        current_wf_idx = WAVEFORMS.index(current_wf) if current_wf in WAVEFORMS else 0

        changed, new_idx = imgui.combo("Waveform", current_wf_idx, WAVEFORMS)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.waveform"
            self.session.execute(SetParam(path, WAVEFORMS[new_idx]))

        # Detune
        changed, new_val = imgui.slider_float("Detune (cents)", params.detune_cents, -100.0, 100.0)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.detune_cents"
            self.session.execute(SetParam(path, new_val))

        imgui.separator()
        imgui.text("PWM")

        # PWM Enable
        changed, new_val = imgui.checkbox("PWM Enable", params.pwm_enable)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.pwm_enable"
            self.session.execute(SetParam(path, new_val))

        if params.pwm_enable:
            # PWM Width (Q0.15)
            pw_pct = params.pwm_width_q15 / 32767.0 * 100.0
            changed, new_pct = imgui.slider_float("Pulse Width", pw_pct, 10.0, 90.0, "%.1f%%")
            if changed:
                new_q15 = int(new_pct / 100.0 * 32767)
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.pwm_width_q15"
                self.session.execute(SetParam(path, new_q15))

        imgui.separator()
        imgui.text("Hard Sync")

        # Sync Enable
        changed, new_val = imgui.checkbox("Sync Enable", params.sync_enable)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.sync_enable"
            self.session.execute(SetParam(path, new_val))

        if params.sync_enable:
            # Sync Ratio (Q16.16)
            ratio_float = params.sync_ratio_q16 / 65536.0
            changed, new_ratio = imgui.slider_float("Sync Ratio", ratio_float, 0.5, 4.0, "%.2fx")
            if changed:
                new_q16 = int(new_ratio * 65536)
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.sync_ratio_q16"
                self.session.execute(SetParam(path, new_q16))

        imgui.separator()
        imgui.text("Unison")

        # Unison voices
        changed, new_val = imgui.slider_int("Unison Voices", params.unison_voices, 1, 8)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.unison_voices"
            self.session.execute(SetParam(path, new_val))

        if params.unison_voices > 1:
            # Unison spread
            changed, new_val = imgui.slider_int("Spread (cents)", params.unison_spread, 0, 50)
            if changed:
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.unison_spread"
                self.session.execute(SetParam(path, new_val))

        imgui.separator()
        imgui.text("Mode Flags")

        # Drone mode
        changed, new_val = imgui.checkbox("Drone Mode", params.drone_mode)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.drone_mode"
            self.session.execute(SetParam(path, new_val))

        # Legato mode
        changed, new_val = imgui.checkbox("Legato Mode", params.legato_mode)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.legato_mode"
            self.session.execute(SetParam(path, new_val))

    def _render_filter_properties(self, params) -> None:
        """Render filter properties."""
        imgui.text("Filter")

        # Filter Enable
        changed, new_val = imgui.checkbox("Filter Enable", params.filter_enable)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.filter_enable"
            self.session.execute(SetParam(path, new_val))

        # Filter type (using FILTER_TYPES)
        filter_type_names = [name for name, _ in FILTER_TYPES]
        current_type_idx = params.filter_type
        changed, new_idx = imgui.combo("Type", current_type_idx, filter_type_names)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.filter_type"
            self.session.execute(SetParam(path, new_idx))

        # Filter slope
        slope_names = ["6 dB/oct", "12 dB/oct", "24 dB/oct"]
        slope_map = {0: 2, 1: 0, 2: 1}  # Display index to actual value
        reverse_map = {2: 0, 0: 1, 1: 2}
        current_slope_idx = reverse_map.get(params.filter_slope, 1)
        changed, new_idx = imgui.combo("Slope", current_slope_idx, slope_names)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.filter_slope"
            self.session.execute(SetParam(path, slope_map[new_idx]))

        imgui.separator()

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

        imgui.separator()
        imgui.text("Modulation")

        # Envelope depth (Q1.15, signed)
        env_depth_pct = params.filter_env_depth_q15 / 32767.0 * 100.0
        changed, new_pct = imgui.slider_float("Env Depth", env_depth_pct, -100.0, 100.0, "%.1f%%")
        if changed:
            new_q15 = int(new_pct / 100.0 * 32767)
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.filter_env_depth_q15"
            self.session.execute(SetParam(path, new_q15))

        # Key tracking (Q0.15)
        key_track_pct = params.filter_key_track_q15 / 32767.0 * 100.0
        changed, new_pct = imgui.slider_float("Key Track", key_track_pct, 0.0, 100.0, "%.1f%%")
        if changed:
            new_q15 = int(new_pct / 100.0 * 32767)
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.filter_key_track_q15"
            self.session.execute(SetParam(path, new_q15))

        imgui.separator()

        # Drive (Q0.15)
        drive_pct = params.filter_drive_q15 / 32767.0 * 100.0
        changed, new_pct = imgui.slider_float("Drive", drive_pct, 0.0, 100.0, "%.1f%%")
        if changed:
            new_q15 = int(new_pct / 100.0 * 32767)
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.filter_drive_q15"
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

    def _render_fm6_properties(self, params) -> None:
        """Render FM6 6-operator FM synthesis properties."""
        imgui.text("FM6 Synthesis")

        # Algorithm selector (DX7 algorithms 1-32)
        current_alg = params.algorithm or 1
        changed, new_alg = imgui.slider_int("Algorithm", current_alg, 1, 32)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.algorithm"
            self.session.execute(SetParam(path, new_alg))

        # Feedback level (0-7)
        current_fb = params.feedback or 0
        changed, new_fb = imgui.slider_int("Feedback", current_fb, 0, 7)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.feedback"
            self.session.execute(SetParam(path, new_fb))

        imgui.separator()
        imgui.text("Operator Levels")

        # Operator levels (6 operators)
        op_levels = params.op_levels or [32767] * 6
        for i in range(6):
            level_pct = op_levels[i] / 32767.0 * 100.0
            changed, new_pct = imgui.slider_float(f"Op {i+1} Level", level_pct, 0.0, 100.0, "%.1f%%")
            if changed:
                new_levels = list(op_levels)
                new_levels[i] = int(new_pct / 100.0 * 32767)
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.op_levels"
                self.session.execute(SetParam(path, new_levels))

        imgui.separator()
        imgui.text("Operator Ratios")

        # Operator ratios (Q16.16 format, 65536 = 1.0x)
        op_ratios = params.op_ratios or [65536] * 6
        ratio_presets = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        for i in range(6):
            ratio_float = op_ratios[i] / 65536.0
            changed, new_ratio = imgui.slider_float(f"Op {i+1} Ratio", ratio_float, 0.5, 16.0, "%.2fx")
            if changed:
                new_ratios = list(op_ratios)
                new_ratios[i] = int(new_ratio * 65536)
                path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.op_ratios"
                self.session.execute(SetParam(path, new_ratios))

    def _render_mixer_properties(self, params) -> None:
        """Render mixer properties."""
        imgui.text("Mixer")
        imgui.text_disabled("Mix levels controlled by connections")

    def _render_fxsend_properties(self, params) -> None:
        """Render FX send properties."""
        imgui.text("FX Send")

        # Send A level
        changed, new_val = imgui.slider_float("Send A", params.send_a, 0.0, 1.0)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.send_a"
            self.session.execute(SetParam(path, new_val))

        # Send B level
        changed, new_val = imgui.slider_float("Send B", params.send_b, 0.0, 1.0)
        if changed:
            path = f"layers.L0_graph.nodes[{self._get_node_index()}].params.send_b"
            self.session.execute(SetParam(path, new_val))

    def _get_node_index(self) -> int:
        """Get index of selected node in IR list."""
        if self._selected_node_id is None:
            return -1

        for i, node in enumerate(self.session.layers.L0_graph.nodes):
            if node.id == self._selected_node_id:
                return i
        return -1
