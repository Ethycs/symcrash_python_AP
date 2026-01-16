"""
L3+ CMMM View - Emit Templates and YAML Editor
==============================================

Visual editor for L3-L5 CMMM layers:
- L3 Cockpit: Emit templates, overlays, transition kernel
- L4 Forge: Full graph editing (extended)
- L5 Lab: AST manipulation (code view)

Provides tree view for structured data and YAML editor for raw access.
Per WSE-STD-GESTURE v0 (§11, §13).
"""

from typing import Optional, Dict, Any
from imgui_bundle import imgui

from symcrash_python_ap.editor import EditorSession, SetParam
from .level_selector import TowerLevel

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class L3CmmmView:
    """
    L3+ CMMM editor view.

    Features:
    - Tree view for emit templates
    - Overlay configuration
    - Transition kernel rules
    - Raw YAML editor for advanced users
    """

    def __init__(self, session: EditorSession):
        self.session = session

        # View state
        self._selected_template: Optional[str] = None
        self._show_yaml_editor = False
        self._yaml_content = ""
        self._yaml_error = ""

    def render(self, avail: imgui.ImVec2, level: TowerLevel) -> None:
        """Render the appropriate L3+ view."""
        if level == TowerLevel.L3_COCKPIT:
            self._render_cockpit_view(avail)
        elif level == TowerLevel.L4_FORGE:
            self._render_forge_view(avail)
        elif level == TowerLevel.L5_LAB:
            self._render_lab_view(avail)

    # =========================================================================
    # L3 COCKPIT VIEW
    # =========================================================================

    def _render_cockpit_view(self, avail: imgui.ImVec2) -> None:
        """Render L3 Cockpit view (emit templates, overlays)."""
        # Two columns: template list and details
        list_width = avail.x * 0.3
        detail_width = avail.x * 0.7

        imgui.begin_child("L3TemplateList", imgui.ImVec2(list_width - 5, avail.y - 30), imgui.ChildFlags_.borders)
        self._render_template_list()
        imgui.end_child()

        imgui.same_line()

        imgui.begin_child("L3TemplateDetail", imgui.ImVec2(detail_width - 10, avail.y - 30), imgui.ChildFlags_.borders)
        self._render_template_detail()
        imgui.end_child()

    def _render_template_list(self) -> None:
        """Render list of emit templates."""
        imgui.text("Emit Templates")
        imgui.separator()

        cmmm = self.session.l3_cmmm
        emit_templates = getattr(cmmm, 'emit_templates', {}) if cmmm else {}

        if not emit_templates:
            imgui.text_disabled("No emit templates defined")
            return

        # Template list
        for template_id in emit_templates:
            selected = self._selected_template == template_id
            if imgui.selectable(template_id, selected)[0]:
                self._selected_template = template_id

        imgui.separator()

        # Overlays section
        imgui.text("Overlays")
        imgui.separator()

        overlays = getattr(cmmm, 'overlays', [])
        for overlay in overlays:
            overlay_id = overlay.get('id', 'unknown') if isinstance(overlay, dict) else 'unknown'
            imgui.bullet_text(overlay_id)

    def _render_template_detail(self) -> None:
        """Render detail view for selected emit template."""
        if self._selected_template is None:
            imgui.text_disabled("Select an emit template")
            return

        cmmm = self.session.l3_cmmm
        emit_templates = getattr(cmmm, 'emit_templates', {}) if cmmm else {}
        template = emit_templates.get(self._selected_template)

        if template is None:
            imgui.text(f"Template not found: {self._selected_template}")
            return

        imgui.text(f"Template: {self._selected_template}")
        imgui.separator()

        # Template is a dict with families, ops_enable, trees, events
        if isinstance(template, dict):
            self._render_template_dict(template)
        else:
            imgui.text_disabled("Template format not recognized")

    def _render_template_dict(self, template: dict) -> None:
        """Render a template dictionary."""
        # Families section
        if imgui.collapsing_header("Families", imgui.TreeNodeFlags_.default_open):
            families = template.get('families', {})
            if isinstance(families, dict):
                mode = families.get('mode', 'static')
                imgui.text(f"Mode: {mode}")

                if mode == 'static':
                    values = families.get('values', {})
                    for family, weight in values.items():
                        pct = (weight / 32767.0 * 100) if isinstance(weight, (int, float)) else 0
                        imgui.text(f"  {family}:")
                        imgui.same_line(150)
                        imgui.progress_bar(pct / 100.0, imgui.ImVec2(150, 16), f"{pct:.0f}%")

                elif mode == 'intent_plus_boost':
                    boost = families.get('boost', {})
                    imgui.text("Boost:")
                    for family, amount in boost.items():
                        imgui.text(f"  {family}: +{amount}")

                    suppress = families.get('suppress', {})
                    if suppress:
                        imgui.text("Suppress:")
                        for family, factor in suppress.items():
                            imgui.text(f"  {family}: x{factor}")

        # Operators section
        if imgui.collapsing_header("Operators", imgui.TreeNodeFlags_.default_open):
            ops = template.get('ops_enable', [])
            if ops:
                for op in ops:
                    imgui.bullet_text(str(op))
            else:
                imgui.text_disabled("No operators")

        # Trees section
        if imgui.collapsing_header("Trees", imgui.TreeNodeFlags_.default_open):
            trees = template.get('trees', {})
            if isinstance(trees, dict):
                tree_a = trees.get('A', trees.get('default', {}).get('A', 1))
                tree_b = trees.get('B', trees.get('default', {}).get('B', 1))
                beta = trees.get('beta', 0)

                imgui.text(f"Tree A: {tree_a}")
                imgui.text(f"Tree B: {tree_b}")
                imgui.text(f"Beta: {beta}")

                selection = trees.get('selection', '')
                if selection:
                    imgui.text_disabled(f"Selection: {selection}")

        # Events section
        if imgui.collapsing_header("Events"):
            events = template.get('events', [])
            if events:
                for i, event in enumerate(events):
                    if isinstance(event, dict):
                        lane = event.get('lane', 'unknown')
                        condition = event.get('condition', 'true')
                        imgui.text(f"{i+1}. {lane}")
                        imgui.text_disabled(f"   when: {condition}")
            else:
                imgui.text_disabled("No events")

    # =========================================================================
    # L4 FORGE VIEW
    # =========================================================================

    def _render_forge_view(self, avail: imgui.ImVec2) -> None:
        """Render L4 Forge view (full graph editing)."""
        imgui.text("L4 Forge - Extended Graph Editing")
        imgui.separator()

        imgui.text_disabled("L4 extends L2 regime editing with:")
        imgui.bullet_text("Custom transition types")
        imgui.bullet_text("Arc specifications")
        imgui.bullet_text("Graph structure modifications")

        imgui.spacing()

        # Show transition kernel
        if imgui.collapsing_header("Transition Kernel", imgui.TreeNodeFlags_.default_open):
            self._render_transition_kernel()

        # Show arc specifications
        if imgui.collapsing_header("Arc Specifications"):
            self._render_arc_specs()

    def _render_transition_kernel(self) -> None:
        """Render transition kernel rules."""
        cmmm = self.session.l3_cmmm
        kernel = getattr(cmmm, 'transition_kernel', []) if cmmm else []

        if not kernel:
            imgui.text_disabled("No transition kernel rules")
            return

        for i, rule in enumerate(kernel):
            if isinstance(rule, dict):
                name = rule.get('name', f'Rule {i+1}')
                from_r = rule.get('from', 'any')
                to_r = rule.get('to', '?')
                when = rule.get('when', 'true')
                priority = rule.get('priority', 0)

                if imgui.tree_node(f"{name} (P{priority})"):
                    imgui.text(f"From: {from_r} → To: {to_r}")
                    imgui.text(f"When: {when}")

                    trans_type = rule.get('type', 'deterministic')
                    imgui.text_disabled(f"Type: {trans_type}")

                    if trans_type == 'weighted':
                        choices = rule.get('choices', [])
                        imgui.text("Choices:")
                        for choice in choices:
                            to = choice.get('to', '?')
                            weight = choice.get('weight', 0)
                            imgui.text(f"  → {to}: {weight:.0%}")

                    imgui.tree_pop()

    def _render_arc_specs(self) -> None:
        """Render arc specifications."""
        arcs = getattr(self.session.l2_regimes, 'arcs', []) if self.session.l2_regimes else []

        if not arcs:
            imgui.text_disabled("No arc specifications")
            return

        for arc in arcs:
            if isinstance(arc, dict):
                arc_id = arc.get('id', '?')
                name = arc.get('name', 'unnamed')
                duration = arc.get('duration_blocks', 0)

                if imgui.tree_node(f"Arc {arc_id}: {name}"):
                    imgui.text(f"Duration: {duration} blocks")

                    boost = arc.get('family_boost', {})
                    if boost:
                        imgui.text("Family boost:")
                        for family, amount in boost.items():
                            imgui.text(f"  {family}: +{amount}")

                    events = arc.get('arc_events', 0)
                    if events:
                        imgui.text(f"Arc events: 0x{events:02X}")

                    curve = arc.get('gain_curve', 'linear')
                    imgui.text_disabled(f"Curve: {curve}")

                    imgui.tree_pop()

    # =========================================================================
    # L5 LAB VIEW
    # =========================================================================

    def _render_lab_view(self, avail: imgui.ImVec2) -> None:
        """Render L5 Lab view (AST manipulation, YAML editor)."""
        imgui.text("L5 Lab - Raw IR Editor")
        imgui.separator()

        imgui.text_disabled("L5 provides direct access to the InstrumentIR structure.")
        imgui.text_disabled("Use with caution - invalid edits may break the preset.")

        imgui.spacing()

        # Toggle YAML editor
        changed, self._show_yaml_editor = imgui.checkbox("Show YAML Editor", self._show_yaml_editor)

        if self._show_yaml_editor:
            self._render_yaml_editor(avail)
        else:
            self._render_ir_tree(avail)

    def _render_yaml_editor(self, avail: imgui.ImVec2) -> None:
        """Render raw YAML editor."""
        if not HAS_YAML:
            imgui.text_colored(imgui.ImVec4(1, 0.5, 0, 1), "PyYAML not installed")
            return

        # Convert IR to YAML
        if imgui.button("Refresh from IR"):
            self._yaml_content = self._ir_to_yaml()
            self._yaml_error = ""

        imgui.same_line()

        if imgui.button("Apply to IR"):
            self._apply_yaml_to_ir()

        if self._yaml_error:
            imgui.text_colored(imgui.ImVec4(1, 0.3, 0.3, 1), self._yaml_error)

        imgui.spacing()

        # Multi-line text editor
        editor_height = avail.y - 120
        changed, self._yaml_content = imgui.input_text_multiline(
            "##YAMLEditor",
            self._yaml_content,
            imgui.ImVec2(avail.x - 20, editor_height),
            imgui.InputTextFlags_.allow_tab_input
        )

    def _ir_to_yaml(self) -> str:
        """Convert InstrumentIR to YAML string."""
        if not HAS_YAML:
            return ""

        # Build dict representation
        data = {
            "name": self.session.name,
            "tower_level": self.session.tower_level,
        }

        # Add L0 graph
        l0 = self.session.l0_graph
        if l0:
            data["l0_graph"] = {
                "nodes": [
                    {"id": n.id, "type": n.type}
                    for n in l0.nodes
                ],
                "edges": [
                    {"src": e.src, "dst": e.dst}
                    for e in l0.edges
                ]
            }

        # Add L2 regimes
        if self.session.l2_regimes:
            data["l2_regimes"] = {
                "regime_nodes": [
                    {"id": r.id, "kind": r.kind, "emit_template": r.emit_template}
                    for r in self.session.l2_regimes.regime_nodes
                ],
                "transitions": [
                    {"from": t.from_regime, "to": t.to, "when": t.when}
                    for t in self.session.l2_regimes.transitions
                ]
            }

        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def _apply_yaml_to_ir(self) -> None:
        """Apply YAML changes to IR."""
        if not HAS_YAML:
            return

        try:
            data = yaml.safe_load(self._yaml_content)
            # For now, just validate it parses
            self._yaml_error = ""
            # TODO: Actually apply changes to IR
            self._yaml_error = "Apply not yet implemented - use individual commands"
        except yaml.YAMLError as e:
            self._yaml_error = f"YAML Error: {e}"

    def _render_ir_tree(self, avail: imgui.ImVec2) -> None:
        """Render IR as a tree view."""
        if imgui.tree_node("InstrumentIR"):
            imgui.text(f"name: {self.session.name}")
            imgui.text(f"tower_level: {self.session.tower_level}")

            l0 = self.session.l0_graph
            if imgui.tree_node("l0_graph"):
                imgui.text(f"nodes: {len(l0.nodes) if l0 else 0}")
                imgui.text(f"edges: {len(l0.edges) if l0 else 0}")
                imgui.tree_pop()

            l1 = self.session.l1_lift
            if imgui.tree_node("l1_lift"):
                imgui.text(f"gesture_basis: {len(getattr(l1, 'gesture_basis', {})) if l1 else 0}")
                imgui.text(f"macro_semantics: {len(getattr(l1, 'macro_semantics', {})) if l1 else 0}")
                imgui.tree_pop()

            l2 = self.session.l2_regimes
            if imgui.tree_node("l2_regimes"):
                imgui.text(f"regime_nodes: {len(l2.regime_nodes) if l2 else 0}")
                imgui.text(f"transitions: {len(l2.transitions) if l2 else 0}")
                imgui.tree_pop()

            l3 = self.session.l3_cmmm
            if l3:
                if imgui.tree_node("cmmm"):
                    imgui.text("(L3+ CMMM data)")
                    imgui.tree_pop()

            imgui.tree_pop()
