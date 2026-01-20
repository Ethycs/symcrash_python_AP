"""
Preset Editor Application - Main Window
=======================================

Multi-level InstrumentIR editor with tabbed views for each tower level.
"""

from typing import Optional, Callable
from pathlib import Path
from enum import IntEnum

from imgui_bundle import imgui, hello_imgui, immapp

from symcrash_python_ap.editor import EditorSession
from symcrash_python_ap.instrument_ir import InstrumentIR

from .views.level_selector import render_level_selector, TowerLevel
from .views.l0_graph import L0GraphView
from .views.l1_mapping import L1MappingView
from .views.l2_regimes import L2RegimeView
from .views.l3_cmmm import L3CmmmView
from .views.mod_matrix import ModMatrixView
from .views.preset_browser import PresetBrowser, render_preset_browser_popup


class PresetEditorApp:
    """
    Main preset editor application.

    Provides:
    - Level selector (L0-L5 tabs)
    - Level-specific views
    - Undo/redo toolbar
    - File operations
    - Command integration with EditorSession
    """

    def __init__(self):
        self.session: Optional[EditorSession] = None
        self.current_level: TowerLevel = TowerLevel.L0_PATCH

        # Level views (lazy initialized)
        self._l0_view: Optional[L0GraphView] = None
        self._l1_view: Optional[L1MappingView] = None
        self._l2_view: Optional[L2RegimeView] = None
        self._l3_view: Optional[L3CmmmView] = None
        self._mod_matrix_view: Optional[ModMatrixView] = None

        # L0 sub-tab selection (Graph vs Modulation)
        self._l0_tab: str = "graph"  # "graph" or "modulation"

        # File dialog state
        self._file_dialog_open = False
        self._save_dialog_open = False

        # Status message
        self._status_message = ""
        self._status_timer = 0.0

        # Preset browser
        self._preset_browser = PresetBrowser(on_load=self._on_browser_load)
        self._show_browser_popup = False
        self._show_browser_panel = True  # Show sidebar by default

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================

    def new_preset(self, name: str = "Untitled") -> None:
        """Create a new empty preset."""
        self.session = EditorSession.new(name)
        self._init_views()
        self._set_status(f"Created new preset: {name}")

    def open_preset(self, path: Path) -> bool:
        """Open a preset from file."""
        try:
            self.session = EditorSession.from_preset(path)
            self._init_views()
            self._set_status(f"Opened: {path.name}")
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._set_status(f"Error loading {path.name}: {e}", duration=10.0)
            return False

    def save_preset(self, path: Optional[Path] = None) -> bool:
        """Save current preset."""
        if self.session is None:
            return False
        try:
            self.session.save(path)
            self._set_status(f"Saved: {self.session.save_path.name}")
            return True
        except Exception as e:
            self._set_status(f"Error saving: {e}")
            return False

    def _init_views(self) -> None:
        """Initialize level views with current session."""
        if self.session is None:
            return

        self._l0_view = L0GraphView(self.session)
        self._l1_view = L1MappingView(self.session)
        self._l2_view = L2RegimeView(self.session)
        self._l3_view = L3CmmmView(self.session)
        self._mod_matrix_view = ModMatrixView(self.session)

    def _set_status(self, message: str, duration: float = 3.0) -> None:
        """Set status bar message."""
        self._status_message = message
        self._status_timer = duration

    def _on_browser_load(self, path: Path) -> None:
        """Callback when preset is loaded from browser."""
        self.open_preset(path)

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self) -> None:
        """Render the main editor window."""
        # Menu bar
        if imgui.begin_main_menu_bar():
            self._render_menu_bar()
            imgui.end_main_menu_bar()

        # Toolbar
        self._render_toolbar()

        # Level selector tabs
        new_level = render_level_selector(
            self.current_level,
            self.session.tower_level if self.session else 0
        )
        if new_level != self.current_level:
            self.current_level = new_level

        # Main content area with optional browser panel
        imgui.separator()

        avail = imgui.get_content_region_avail()

        # Browser sidebar (left panel)
        if self._show_browser_panel:
            browser_width = 250
            imgui.begin_child("BrowserPanel", imgui.ImVec2(browser_width, avail.y - 30), imgui.ChildFlags_.borders)
            imgui.text("Presets")
            imgui.separator()
            self._preset_browser.render(avail.y - 60)
            imgui.end_child()

            imgui.same_line()

            # Content area (reduced width)
            content_width = avail.x - browser_width - 10
            imgui.begin_child("ContentArea", imgui.ImVec2(content_width, avail.y - 30))
            if self.session is None:
                self._render_welcome()
            else:
                self._render_current_level()
            imgui.end_child()
        else:
            # Full width content
            if self.session is None:
                self._render_welcome()
            else:
                self._render_current_level()

        # Browser popup (File > Open)
        if self._show_browser_popup:
            render_preset_browser_popup(self._preset_browser, True)
            self._show_browser_popup = False
        else:
            render_preset_browser_popup(self._preset_browser, False)

        # Status bar
        self._render_status_bar()

    def _render_menu_bar(self) -> None:
        """Render the menu bar."""
        if imgui.begin_menu("File"):
            if imgui.menu_item("New", "Ctrl+N", False)[0]:
                self.new_preset()
            if imgui.menu_item("Open...", "Ctrl+O", False)[0]:
                self._show_browser_popup = True
            if imgui.menu_item("Save", "Ctrl+S", False, self.session is not None)[0]:
                self.save_preset()
            if imgui.menu_item("Save As...", "Ctrl+Shift+S", False, self.session is not None)[0]:
                self._save_dialog_open = True
            imgui.separator()
            if imgui.menu_item("Exit", "Alt+F4", False)[0]:
                pass  # Handled by hello_imgui
            imgui.end_menu()

        if imgui.begin_menu("Edit"):
            can_undo = self.session.can_undo() if self.session else False
            can_redo = self.session.can_redo() if self.session else False
            undo_desc = self.session.undo_description() if self.session else ""
            redo_desc = self.session.redo_description() if self.session else ""

            if imgui.menu_item(f"Undo {undo_desc}", "Ctrl+Z", False, can_undo)[0]:
                self.session.undo()
            if imgui.menu_item(f"Redo {redo_desc}", "Ctrl+Y", False, can_redo)[0]:
                self.session.redo()
            imgui.end_menu()

        if imgui.begin_menu("View"):
            # Preset browser toggle
            changed, self._show_browser_panel = imgui.menu_item(
                "Preset Browser", "Ctrl+B", self._show_browser_panel
            )
            imgui.separator()

            # Level selection
            for level in TowerLevel:
                selected = self.current_level == level
                if imgui.menu_item(level.display_name, f"F{level.value + 1}", selected)[0]:
                    self.current_level = level
            imgui.end_menu()

    def _render_toolbar(self) -> None:
        """Render the toolbar."""
        # Undo/Redo buttons
        can_undo = self.session.can_undo() if self.session else False
        can_redo = self.session.can_redo() if self.session else False

        imgui.begin_disabled(not can_undo)
        if imgui.button("Undo"):
            self.session.undo()
        imgui.end_disabled()

        imgui.same_line()

        imgui.begin_disabled(not can_redo)
        if imgui.button("Redo"):
            self.session.redo()
        imgui.end_disabled()

        imgui.same_line()
        imgui.text("|")  # Simple vertical separator
        imgui.same_line()

        # Preset name
        if self.session:
            dirty = "*" if self.session.is_dirty else ""
            imgui.text(f"{self.session.name}{dirty}")
            imgui.same_line()
            imgui.text_disabled(f"(L{self.session.tower_level})")

    def _render_welcome(self) -> None:
        """Render welcome screen when no preset is loaded."""
        avail = imgui.get_content_region_avail()

        # Center the welcome text
        imgui.set_cursor_pos_x(avail.x / 2 - 150)
        imgui.set_cursor_pos_y(avail.y / 2 - 50)

        imgui.text("Symcrash Preset Editor")
        imgui.spacing()
        imgui.text("No preset loaded.")
        imgui.spacing()

        if imgui.button("New Preset", imgui.ImVec2(120, 0)):
            self.new_preset()

        imgui.same_line()

        if imgui.button("Open Preset...", imgui.ImVec2(120, 0)):
            self._show_browser_popup = True

        imgui.spacing()
        imgui.spacing()

        if self._show_browser_panel:
            imgui.text_disabled("Or select a preset from the browser panel on the left.")
        else:
            imgui.text_disabled("Tip: View > Preset Browser to show the sidebar.")

    def _render_current_level(self) -> None:
        """Render the current level view."""
        avail = imgui.get_content_region_avail()

        if self.current_level == TowerLevel.L0_PATCH:
            # L0 has sub-tabs: Graph and Modulation
            if imgui.begin_tab_bar("L0SubTabs"):
                if imgui.begin_tab_item("DSP Graph")[0]:
                    self._l0_tab = "graph"
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Modulation / Macros")[0]:
                    self._l0_tab = "modulation"
                    imgui.end_tab_item()
                imgui.end_tab_bar()

            # Render sub-view
            sub_avail = imgui.get_content_region_avail()
            if self._l0_tab == "graph":
                if self._l0_view:
                    self._l0_view.render(sub_avail)
            else:
                if self._mod_matrix_view:
                    self._mod_matrix_view.render(sub_avail)

        elif self.current_level == TowerLevel.L1_LIFT:
            if self._l1_view:
                self._l1_view.render(avail)
        elif self.current_level == TowerLevel.L2_REGIMES:
            if self._l2_view:
                self._l2_view.render(avail)
        elif self.current_level in (TowerLevel.L3_COCKPIT, TowerLevel.L4_FORGE, TowerLevel.L5_LAB):
            if self._l3_view:
                self._l3_view.render(avail, self.current_level)

    def _render_status_bar(self) -> None:
        """Render status bar at bottom."""
        # Decrement timer
        if self._status_timer > 0:
            self._status_timer -= imgui.get_io().delta_time

        # Status bar
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(
            imgui.ImVec2(viewport.work_pos.x, viewport.work_pos.y + viewport.work_size.y - 22)
        )
        imgui.set_next_window_size(imgui.ImVec2(viewport.work_size.x, 22))

        flags = (
            imgui.WindowFlags_.no_decoration |
            imgui.WindowFlags_.no_move |
            imgui.WindowFlags_.no_saved_settings |
            imgui.WindowFlags_.no_focus_on_appearing |
            imgui.WindowFlags_.no_nav
        )

        imgui.begin("##StatusBar", flags=flags)

        if self._status_timer > 0 and self._status_message:
            imgui.text(self._status_message)
        else:
            if self.session:
                imgui.text_disabled(
                    f"Undo: {self.session.history.undo_count} | "
                    f"Redo: {self.session.history.redo_count}"
                )

        imgui.end()

    # =========================================================================
    # RUN
    # =========================================================================

    def run(self) -> None:
        """Run the editor application."""
        runner_params = hello_imgui.RunnerParams()
        runner_params.app_window_params.window_title = "Symcrash Preset Editor"
        runner_params.app_window_params.window_geometry.size = (1400, 900)

        runner_params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.no_default_window
        )

        # Main dockspace
        runner_params.imgui_window_params.enable_viewports = True

        def gui():
            # Create main dockspace
            viewport = imgui.get_main_viewport()
            imgui.set_next_window_pos(viewport.work_pos)
            imgui.set_next_window_size(viewport.work_size)
            imgui.set_next_window_viewport(viewport.id_)

            window_flags = (
                imgui.WindowFlags_.no_title_bar |
                imgui.WindowFlags_.no_collapse |
                imgui.WindowFlags_.no_resize |
                imgui.WindowFlags_.no_move |
                imgui.WindowFlags_.no_bring_to_front_on_focus |
                imgui.WindowFlags_.no_nav_focus |
                imgui.WindowFlags_.menu_bar
            )

            imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
            imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))

            imgui.begin("MainWindow", flags=window_flags)
            imgui.pop_style_var(3)

            self.render()

            imgui.end()

        runner_params.callbacks.show_gui = gui

        immapp.run(runner_params)


def main():
    """Entry point for the preset editor."""
    app = PresetEditorApp()
    app.run()


if __name__ == "__main__":
    main()
