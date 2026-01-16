"""
Editor Module - Command-Based Preset Editing
=============================================

Implements WSE-STD-070 §2 Editor Operations using the Command Pattern.

This module provides:
- EditorCommand: Abstract base for all editor operations
- CommandHistory: Undo/redo stack management
- EditorSession: Manages InstrumentIR with command-based editing
- Concrete commands for all WSE-STD-070 §2.1 primitives

Usage:
    from symcrash_python_ap.editor import EditorSession

    session = EditorSession.from_preset("my_preset.yaml")
    session.execute(SetParam("L1.intent_mapping.mode", "custom"))
    session.execute(AddRegime("R5_CATASTROPHE"))
    session.undo()  # Reverts AddRegime
    session.save("my_preset_modified.yaml")
"""

from .commands import (
    EditorCommand,
    EditorResult,
    # Structural commands
    SetTreeId,
    AddRegime,
    RemoveRegime,
    AddTransition,
    RemoveTransition,
    # Parametric commands
    SetParam,
    DeltaParam,
    SetBound,
    # Policy commands
    SetOperatorAllowed,
    SetOperatorWeight,
    SetBudget,
    # Meta commands
    SetTowerLevel,
    ProjectToLevel,
    ResetToSafe,
)
from .history import CommandHistory
from .session import EditorSession

__all__ = [
    # Core
    "EditorCommand",
    "EditorResult",
    "CommandHistory",
    "EditorSession",
    # Structural
    "SetTreeId",
    "AddRegime",
    "RemoveRegime",
    "AddTransition",
    "RemoveTransition",
    # Parametric
    "SetParam",
    "DeltaParam",
    "SetBound",
    # Policy
    "SetOperatorAllowed",
    "SetOperatorWeight",
    "SetBudget",
    # Meta
    "SetTowerLevel",
    "ProjectToLevel",
    "ResetToSafe",
]
