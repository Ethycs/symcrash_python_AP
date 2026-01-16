"""
Editor Session - Managed InstrumentIR Editing
==============================================

Provides a high-level interface for editing InstrumentIR instances
using the command pattern with full undo/redo support.

Usage:
    from symcrash_python_ap.editor import EditorSession

    session = EditorSession.from_preset("my_preset.yaml")
    session.execute(SetParam("L1.intent_mapping.mode", "custom"))
    session.execute(AddRegime("R5_CATASTROPHE"))
    session.undo()  # Reverts AddRegime
    session.save("my_preset_modified.yaml")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union, Callable, Any
import copy

from .commands import EditorCommand, EditorResult, EditorResultStatus
from .history import CommandHistory, HistoryEntry


@dataclass
class SessionState:
    """
    Tracks session state for dirty detection and save management.
    """
    is_dirty: bool = False
    save_path: Optional[Path] = None
    last_save_hash: Optional[int] = None


class EditorSession:
    """
    Manages an InstrumentIR instance with command-based editing.

    Features:
    - Command pattern for all modifications
    - Full undo/redo support
    - Dirty state tracking
    - Save/load support (YAML or JSON)
    - Event callbacks for UI integration

    Per WSE-STD-070:
    - All edits go through typed primitives
    - All operations are pure and replayable
    - Determinism is guaranteed
    """

    def __init__(
        self,
        ir: 'InstrumentIR',
        max_history: int = 100,
        source_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize an editor session.

        Args:
            ir: The InstrumentIR to edit
            max_history: Maximum undo history depth
            source_path: Original file path (for save tracking)
        """
        from ..instrument_ir import InstrumentIR

        self._ir = ir
        self._history = CommandHistory(max_history)
        self._state = SessionState(
            save_path=Path(source_path) if source_path else None
        )

        # Callbacks for UI integration
        self._on_change_callbacks: List[Callable[['EditorSession'], None]] = []
        self._on_undo_callbacks: List[Callable[['EditorSession', str], None]] = []
        self._on_redo_callbacks: List[Callable[['EditorSession', str], None]] = []

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    @classmethod
    def from_preset(
        cls,
        path: Union[str, Path],
        max_history: int = 100
    ) -> 'EditorSession':
        """
        Create an editor session from a preset file.

        Args:
            path: Path to YAML or JSON preset file
            max_history: Maximum undo history depth

        Returns:
            EditorSession with loaded InstrumentIR
        """
        from ..instrument_ir import load_instrument

        path = Path(path)
        ir = load_instrument(path)
        return cls(ir, max_history, source_path=path)

    @classmethod
    def new(cls, name: str = "Untitled", max_history: int = 100) -> 'EditorSession':
        """
        Create a new editor session with a default InstrumentIR.

        Args:
            name: Name for the new instrument
            max_history: Maximum undo history depth

        Returns:
            EditorSession with new, empty InstrumentIR
        """
        from ..instrument_ir import (
            InstrumentIR, InstrumentMeta, IOConfig, Layers,
            L0Graph, L1Lift, L2Regimes, L3Cmmm
        )

        ir = InstrumentIR(
            meta=InstrumentMeta(name=name),
            io=IOConfig(),
            layers=Layers(
                L0_graph=L0Graph(),
                L1_lift=L1Lift(),
                L2_regimes=L2Regimes(),
                L3_cmmm=L3Cmmm()
            )
        )
        return cls(ir, max_history)

    # =========================================================================
    # COMMAND EXECUTION
    # =========================================================================

    def execute(self, command: EditorCommand) -> EditorResult:
        """
        Execute a command on the InstrumentIR.

        The command is recorded in history if successful.
        Triggers on_change callbacks.

        Args:
            command: The EditorCommand to execute

        Returns:
            EditorResult with status and message
        """
        result = command.execute(self._ir)

        if result.status == EditorResultStatus.SUCCESS:
            self._history.record(command, result)
            self._state.is_dirty = True
            self._notify_change()

        return result

    def execute_batch(self, commands: List[EditorCommand]) -> List[EditorResult]:
        """
        Execute multiple commands as a batch.

        All commands are executed in order. Each successful command
        is recorded individually in history.

        Args:
            commands: List of commands to execute

        Returns:
            List of results (one per command)
        """
        results = []
        for command in commands:
            result = self.execute(command)
            results.append(result)
            # Stop on first failure
            if not result.ok:
                break
        return results

    # =========================================================================
    # UNDO/REDO
    # =========================================================================

    def undo(self) -> Optional[EditorResult]:
        """
        Undo the most recent command.

        Returns:
            EditorResult if undo was performed, None if nothing to undo
        """
        entry = self._history.pop_undo()
        if entry is None:
            return None

        result = entry.command.undo(self._ir, entry.undo_data)

        if result.ok:
            self._history.push_redo(entry)
            self._state.is_dirty = True
            self._notify_undo(entry.description)

        return result

    def redo(self) -> Optional[EditorResult]:
        """
        Redo the most recently undone command.

        Returns:
            EditorResult if redo was performed, None if nothing to redo
        """
        entry = self._history.pop_redo()
        if entry is None:
            return None

        result = entry.command.execute(self._ir)

        if result.ok:
            self._history.record(entry.command, result)
            self._state.is_dirty = True
            self._notify_redo(entry.description)

        return result

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._history.can_undo()

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._history.can_redo()

    def undo_description(self) -> Optional[str]:
        """Get description of the next undo action."""
        return self._history.peek_undo()

    def redo_description(self) -> Optional[str]:
        """Get description of the next redo action."""
        return self._history.peek_redo()

    # =========================================================================
    # SAVE/LOAD
    # =========================================================================

    def save(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Save the InstrumentIR to file.

        Args:
            path: Path to save to. If None, uses the original source path.

        Returns:
            True if save succeeded
        """
        from ..instrument_ir import save_instrument

        save_path = Path(path) if path else self._state.save_path

        if save_path is None:
            raise ValueError("No save path specified and no source path available")

        save_instrument(self._ir, save_path)
        self._state.save_path = save_path
        self._state.is_dirty = False

        return True

    def save_as(self, path: Union[str, Path]) -> bool:
        """
        Save the InstrumentIR to a new path.

        Updates the session's source path to the new location.

        Args:
            path: New path to save to

        Returns:
            True if save succeeded
        """
        return self.save(path)

    def reload(self) -> bool:
        """
        Reload from the original source path.

        Discards all unsaved changes and clears history.

        Returns:
            True if reload succeeded, False if no source path
        """
        from ..instrument_ir import load_instrument

        if self._state.save_path is None:
            return False

        self._ir = load_instrument(self._state.save_path)
        self._history.clear()
        self._state.is_dirty = False
        self._notify_change()

        return True

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def ir(self) -> 'InstrumentIR':
        """Access the underlying InstrumentIR (read-only recommended)."""
        return self._ir

    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._state.is_dirty

    @property
    def save_path(self) -> Optional[Path]:
        """Get the current save path."""
        return self._state.save_path

    @property
    def name(self) -> str:
        """Get the instrument name."""
        if hasattr(self._ir, 'meta') and self._ir.meta:
            return getattr(self._ir.meta, 'name', 'Untitled') or 'Untitled'
        return 'Untitled'

    @property
    def tower_level(self) -> int:
        """Get the current tower level."""
        # tower_level may be on compile config or meta
        if hasattr(self._ir, 'compile') and self._ir.compile:
            level = getattr(self._ir.compile, 'tower_level', None)
            if level is not None:
                return level
        return 0

    @property
    def history(self) -> CommandHistory:
        """Access the command history."""
        return self._history

    @property
    def layers(self):
        """Access the layers container from IR."""
        return self._ir.layers

    @property
    def l0_graph(self):
        """Access L0 graph layer."""
        return self._ir.layers.L0_graph

    @property
    def l1_lift(self):
        """Access L1 lift layer."""
        return self._ir.layers.L1_lift

    @property
    def l2_regimes(self):
        """Access L2 regimes layer."""
        return self._ir.layers.L2_regimes

    @property
    def l3_cmmm(self):
        """Access L3 CMMM layer."""
        return self._ir.layers.L3_cmmm

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_change(self, callback: Callable[['EditorSession'], None]) -> None:
        """
        Register a callback for any changes.

        Args:
            callback: Function called after any successful command execution
        """
        self._on_change_callbacks.append(callback)

    def on_undo(self, callback: Callable[['EditorSession', str], None]) -> None:
        """
        Register a callback for undo operations.

        Args:
            callback: Function called after undo with (session, description)
        """
        self._on_undo_callbacks.append(callback)

    def on_redo(self, callback: Callable[['EditorSession', str], None]) -> None:
        """
        Register a callback for redo operations.

        Args:
            callback: Function called after redo with (session, description)
        """
        self._on_redo_callbacks.append(callback)

    def _notify_change(self) -> None:
        """Notify all change listeners."""
        for callback in self._on_change_callbacks:
            callback(self)

    def _notify_undo(self, description: str) -> None:
        """Notify all undo listeners."""
        for callback in self._on_undo_callbacks:
            callback(self, description)

    def _notify_redo(self, description: str) -> None:
        """Notify all redo listeners."""
        for callback in self._on_redo_callbacks:
            callback(self, description)

    # =========================================================================
    # UTILITY
    # =========================================================================

    def clone(self) -> 'EditorSession':
        """
        Create a copy of this session with independent history.

        Useful for "what-if" editing scenarios.

        Returns:
            New EditorSession with cloned IR
        """
        cloned_ir = copy.deepcopy(self._ir)
        return EditorSession(cloned_ir)

    def get_summary(self) -> dict:
        """
        Get a summary of the session state.

        Returns:
            Dictionary with session information
        """
        return {
            "name": self.name,
            "tower_level": self.tower_level,
            "is_dirty": self.is_dirty,
            "save_path": str(self._state.save_path) if self._state.save_path else None,
            "undo_available": self.can_undo(),
            "redo_available": self.can_redo(),
            "undo_count": self._history.undo_count,
            "redo_count": self._history.redo_count,
        }

    def __repr__(self) -> str:
        dirty = "*" if self.is_dirty else ""
        return f"EditorSession({self.name}{dirty}, L{self.tower_level}, {self._history})"
