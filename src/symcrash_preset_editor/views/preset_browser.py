"""
Preset Browser View - Folder-based preset selection
====================================================

Displays presets from the preset folder in a hierarchical tree.
Supports loading presets by clicking on them.
"""

from typing import Optional, Callable, Dict, List
from pathlib import Path
from dataclasses import dataclass, field

from imgui_bundle import imgui

# Default preset folder relative to project root
# preset_browser.py is at: src/symcrash_preset_editor/views/preset_browser.py
# Project root is 4 parents up: views -> symcrash_preset_editor -> src -> project_root
DEFAULT_PRESET_ROOT = Path(__file__).parent.parent.parent.parent / "presets"


@dataclass
class PresetEntry:
    """A preset file or folder entry."""
    name: str
    path: Path
    is_folder: bool
    children: List['PresetEntry'] = field(default_factory=list)


class PresetBrowser:
    """
    Preset browser panel.

    Features:
    - Tree view of preset folder structure
    - Category grouping (instruments/, macros/, etc.)
    - Double-click or button to load
    - Refresh button to rescan
    """

    def __init__(
        self,
        preset_root: Optional[Path] = None,
        on_load: Optional[Callable[[Path], None]] = None
    ):
        self.preset_root = preset_root or DEFAULT_PRESET_ROOT
        self.on_load = on_load

        # State
        self._selected_path: Optional[Path] = None
        self._tree: List[PresetEntry] = []
        self._filter_text = ""

        # Scan presets on init
        self.refresh()

    def refresh(self) -> None:
        """Rescan the preset folder."""
        self._tree = self._scan_folder(self.preset_root)

    def _scan_folder(self, folder: Path) -> List[PresetEntry]:
        """Recursively scan a folder for presets."""
        entries: List[PresetEntry] = []

        if not folder.exists():
            return entries

        # Get all items, sorted (folders first, then files)
        items = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

        for item in items:
            if item.is_dir():
                # Skip hidden folders
                if item.name.startswith('.'):
                    continue
                # Skip __pycache__ and similar
                if item.name.startswith('__'):
                    continue

                children = self._scan_folder(item)
                if children:  # Only include folders with content
                    entries.append(PresetEntry(
                        name=item.name,
                        path=item,
                        is_folder=True,
                        children=children
                    ))
            elif item.suffix.lower() in ('.yaml', '.yml'):
                entries.append(PresetEntry(
                    name=item.stem,  # Name without extension
                    path=item,
                    is_folder=False
                ))

        return entries

    def render(self, height: float = 0) -> Optional[Path]:
        """
        Render the preset browser panel.

        Args:
            height: Panel height (0 = use available space)

        Returns:
            Path if a preset was loaded this frame, else None
        """
        loaded_path: Optional[Path] = None

        # Header with refresh button
        if imgui.button("Refresh"):
            self.refresh()

        imgui.same_line()
        imgui.text_disabled(f"({self._count_presets(self._tree)} presets)")

        # Filter input
        imgui.set_next_item_width(-1)
        changed, self._filter_text = imgui.input_text_with_hint(
            "##Filter",
            "Filter presets...",
            self._filter_text,
            256
        )

        imgui.separator()

        # Tree view in scrolling region
        if height > 0:
            imgui.begin_child("PresetTree", imgui.ImVec2(0, height - 80), imgui.ChildFlags_.borders)
        else:
            imgui.begin_child("PresetTree", imgui.ImVec2(0, -60), imgui.ChildFlags_.borders)

        for entry in self._tree:
            result = self._render_entry(entry)
            if result:
                loaded_path = result

        imgui.end_child()

        # Selection info and load button
        imgui.separator()

        if self._selected_path:
            imgui.text(f"Selected: {self._selected_path.stem}")

            if imgui.button("Load Selected", imgui.ImVec2(-1, 0)):
                loaded_path = self._selected_path
                if self.on_load:
                    self.on_load(self._selected_path)
        else:
            imgui.text_disabled("Select a preset to load")

        return loaded_path

    def _render_entry(self, entry: PresetEntry) -> Optional[Path]:
        """Render a single tree entry. Returns path if loaded."""
        loaded_path: Optional[Path] = None

        # Apply filter
        if self._filter_text:
            if entry.is_folder:
                # For folders, check if any children match
                has_match = self._folder_has_match(entry, self._filter_text.lower())
                if not has_match:
                    return None
            else:
                if self._filter_text.lower() not in entry.name.lower():
                    return None

        if entry.is_folder:
            # Folder node
            flags = imgui.TreeNodeFlags_.open_on_arrow | imgui.TreeNodeFlags_.open_on_double_click

            # Auto-expand if filtering
            if self._filter_text:
                flags |= imgui.TreeNodeFlags_.default_open

            if imgui.tree_node_ex(f"{entry.name}/", flags):
                for child in entry.children:
                    result = self._render_entry(child)
                    if result:
                        loaded_path = result
                imgui.tree_pop()
        else:
            # Preset file (leaf node)
            is_selected = self._selected_path == entry.path

            flags = imgui.TreeNodeFlags_.leaf | imgui.TreeNodeFlags_.no_tree_push_on_open
            if is_selected:
                flags |= imgui.TreeNodeFlags_.selected

            imgui.tree_node_ex(entry.name, flags)

            # Single click = select
            if imgui.is_item_clicked():
                self._selected_path = entry.path

            # Double click = load
            if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                loaded_path = entry.path
                if self.on_load:
                    self.on_load(entry.path)

        return loaded_path

    def _folder_has_match(self, entry: PresetEntry, filter_lower: str) -> bool:
        """Check if folder contains any matching presets."""
        for child in entry.children:
            if child.is_folder:
                if self._folder_has_match(child, filter_lower):
                    return True
            else:
                if filter_lower in child.name.lower():
                    return True
        return False

    def _count_presets(self, entries: List[PresetEntry]) -> int:
        """Count total presets in tree."""
        count = 0
        for entry in entries:
            if entry.is_folder:
                count += self._count_presets(entry.children)
            else:
                count += 1
        return count


def render_preset_browser_popup(
    browser: PresetBrowser,
    open_popup: bool
) -> Optional[Path]:
    """
    Render preset browser as a modal popup.

    Args:
        browser: PresetBrowser instance
        open_popup: If True, open the popup this frame

    Returns:
        Path if a preset was selected, else None
    """
    loaded_path: Optional[Path] = None

    if open_popup:
        imgui.open_popup("Open Preset")

    # Center popup
    center = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size(imgui.ImVec2(400, 500), imgui.Cond_.appearing)

    if imgui.begin_popup_modal("Open Preset", flags=imgui.WindowFlags_.none)[0]:
        loaded_path = browser.render(400)

        imgui.separator()

        if imgui.button("Cancel", imgui.ImVec2(120, 0)):
            imgui.close_current_popup()

        # Close on load
        if loaded_path:
            imgui.close_current_popup()

        imgui.end_popup()

    return loaded_path
