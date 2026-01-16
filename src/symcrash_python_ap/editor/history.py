"""
Command History - Undo/Redo Stack Management
=============================================

Manages the history of editor commands for undo/redo operations.
Follows standard undo/redo semantics with configurable history depth.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict
from .commands import EditorCommand, EditorResult, EditorResultStatus


@dataclass
class HistoryEntry:
    """
    A single entry in the command history.

    Stores the command and its undo data for reversal.
    """
    command: EditorCommand
    undo_data: Dict[str, Any]

    @property
    def description(self) -> str:
        return self.command.description


class CommandHistory:
    """
    Manages command history for undo/redo operations.

    Uses two stacks:
    - undo_stack: Commands that have been executed and can be undone
    - redo_stack: Commands that have been undone and can be redone

    Per standard undo/redo semantics:
    - Executing a new command clears the redo stack
    - Undo moves commands from undo_stack to redo_stack
    - Redo moves commands from redo_stack to undo_stack
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize command history.

        Args:
            max_history: Maximum number of commands to keep in history.
                        Oldest commands are dropped when limit is reached.
        """
        self._undo_stack: List[HistoryEntry] = []
        self._redo_stack: List[HistoryEntry] = []
        self._max_history = max_history

    def record(self, command: EditorCommand, result: EditorResult) -> None:
        """
        Record a successfully executed command.

        Only commands with SUCCESS status are recorded (NO_OP and FAILURE are not).
        Clears the redo stack since a new action invalidates the redo path.

        Args:
            command: The command that was executed
            result: The result of execution (must contain undo_data)
        """
        if result.status != EditorResultStatus.SUCCESS:
            return

        if result.undo_data is None:
            return

        entry = HistoryEntry(command=command, undo_data=result.undo_data)
        self._undo_stack.append(entry)

        # Clear redo stack - new action invalidates redo path
        self._redo_stack.clear()

        # Enforce max history limit
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)

    def can_undo(self) -> bool:
        """Check if there are commands to undo."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if there are commands to redo."""
        return len(self._redo_stack) > 0

    def peek_undo(self) -> Optional[str]:
        """
        Get the description of the next command to undo without undoing it.

        Returns:
            Command description or None if nothing to undo
        """
        if not self._undo_stack:
            return None
        return self._undo_stack[-1].description

    def peek_redo(self) -> Optional[str]:
        """
        Get the description of the next command to redo without redoing it.

        Returns:
            Command description or None if nothing to redo
        """
        if not self._redo_stack:
            return None
        return self._redo_stack[-1].description

    def pop_undo(self) -> Optional[HistoryEntry]:
        """
        Pop the most recent command from undo stack.

        The caller is responsible for actually executing the undo
        and then calling push_redo() if successful.

        Returns:
            HistoryEntry or None if nothing to undo
        """
        if not self._undo_stack:
            return None
        return self._undo_stack.pop()

    def push_redo(self, entry: HistoryEntry) -> None:
        """
        Push an entry to the redo stack after successful undo.

        Args:
            entry: The entry that was just undone
        """
        self._redo_stack.append(entry)

    def pop_redo(self) -> Optional[HistoryEntry]:
        """
        Pop the most recent command from redo stack.

        The caller is responsible for actually re-executing the command
        and then calling record() if successful.

        Returns:
            HistoryEntry or None if nothing to redo
        """
        if not self._redo_stack:
            return None
        return self._redo_stack.pop()

    def clear(self) -> None:
        """Clear all history (both undo and redo stacks)."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def get_undo_descriptions(self, count: int = 10) -> List[str]:
        """
        Get descriptions of recent commands that can be undone.

        Args:
            count: Maximum number of descriptions to return

        Returns:
            List of descriptions, most recent first
        """
        return [
            entry.description
            for entry in reversed(self._undo_stack[-count:])
        ]

    def get_redo_descriptions(self, count: int = 10) -> List[str]:
        """
        Get descriptions of commands that can be redone.

        Args:
            count: Maximum number of descriptions to return

        Returns:
            List of descriptions, most recent first
        """
        return [
            entry.description
            for entry in reversed(self._redo_stack[-count:])
        ]

    @property
    def undo_count(self) -> int:
        """Number of commands that can be undone."""
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        """Number of commands that can be redone."""
        return len(self._redo_stack)

    def __repr__(self) -> str:
        return f"CommandHistory(undo={self.undo_count}, redo={self.redo_count})"
