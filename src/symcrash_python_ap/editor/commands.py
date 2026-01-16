"""
Editor Commands - WSE-STD-070 §2 Editor Primitives
==================================================

Implements the Command Pattern for all allowed editor operations.
All commands are pure, replayable, and support undo/redo.

Per WSE-STD-070:
- Editors MUST operate using typed primitives, not raw mutation
- All editor ops MUST be pure and replayable
- Editors MUST NOT emit operators, events, or timing logic
- Editors MUST NOT mutate runtime state directly
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import copy


# =============================================================================
# RESULT TYPE
# =============================================================================

class EditorResultStatus(Enum):
    """Result status for editor commands."""
    SUCCESS = "success"
    FAILURE = "failure"
    NO_OP = "no_op"  # Command had no effect (e.g., setting same value)


@dataclass
class EditorResult:
    """
    Result of executing an editor command.

    Contains status, optional error message, and any data needed for undo.
    """
    status: EditorResultStatus
    message: str = ""
    undo_data: Optional[Dict[str, Any]] = None

    @classmethod
    def success(cls, message: str = "", undo_data: Optional[Dict[str, Any]] = None) -> 'EditorResult':
        return cls(EditorResultStatus.SUCCESS, message, undo_data)

    @classmethod
    def failure(cls, message: str) -> 'EditorResult':
        return cls(EditorResultStatus.FAILURE, message, None)

    @classmethod
    def no_op(cls, message: str = "") -> 'EditorResult':
        return cls(EditorResultStatus.NO_OP, message, None)

    @property
    def ok(self) -> bool:
        return self.status in (EditorResultStatus.SUCCESS, EditorResultStatus.NO_OP)


# =============================================================================
# BASE COMMAND
# =============================================================================

class EditorCommand(ABC):
    """
    Abstract base class for all editor commands.

    Commands follow the Command Pattern:
    - execute() applies the command to an InstrumentIR
    - undo() reverses the command using stored undo_data
    - Commands are pure and replayable

    Per WSE-STD-070 §2.1, editors may only emit these typed primitives.
    """

    @abstractmethod
    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        """
        Execute the command on the given InstrumentIR.

        Returns EditorResult with status and undo_data if successful.
        The undo_data should contain everything needed to reverse this command.
        """
        pass

    @abstractmethod
    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        """
        Undo the command using the stored undo_data.

        Returns EditorResult with status.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the command."""
        pass


# =============================================================================
# PATH UTILITIES
# =============================================================================

def _get_nested(obj: Any, path: str) -> Tuple[bool, Any]:
    """
    Get a value from a nested object using dot-separated path.

    Returns (success, value) tuple.

    Example:
        _get_nested(ir, "l2_regimes.regime_nodes") -> (True, [...])
    """
    parts = path.split('.')
    current = obj

    for part in parts:
        # Handle list indexing like "regime_nodes[0]"
        if '[' in part:
            name, idx_str = part.split('[')
            idx = int(idx_str.rstrip(']'))

            if hasattr(current, name):
                current = getattr(current, name)
            elif isinstance(current, dict) and name in current:
                current = current[name]
            else:
                return (False, None)

            if isinstance(current, list) and 0 <= idx < len(current):
                current = current[idx]
            else:
                return (False, None)
        elif hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return (False, None)

    return (True, current)


def _set_nested(obj: Any, path: str, value: Any) -> bool:
    """
    Set a value in a nested object using dot-separated path.

    Returns True if successful.
    """
    parts = path.split('.')
    current = obj

    # Navigate to parent
    for part in parts[:-1]:
        if '[' in part:
            name, idx_str = part.split('[')
            idx = int(idx_str.rstrip(']'))

            if hasattr(current, name):
                current = getattr(current, name)
            elif isinstance(current, dict) and name in current:
                current = current[name]
            else:
                return False

            if isinstance(current, list) and 0 <= idx < len(current):
                current = current[idx]
            else:
                return False
        elif hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False

    # Set final value
    final_part = parts[-1]
    if '[' in final_part:
        name, idx_str = final_part.split('[')
        idx = int(idx_str.rstrip(']'))

        if hasattr(current, name):
            lst = getattr(current, name)
        elif isinstance(current, dict) and name in current:
            lst = current[name]
        else:
            return False

        if isinstance(lst, list) and 0 <= idx < len(lst):
            lst[idx] = value
            return True
        return False
    elif hasattr(current, final_part):
        setattr(current, final_part, value)
        return True
    elif isinstance(current, dict):
        current[final_part] = value
        return True

    return False


# =============================================================================
# STRUCTURAL COMMANDS (WSE-STD-070 §2.1)
# =============================================================================

@dataclass
class SetTreeId(EditorCommand):
    """
    SET_TREE_ID(tree_id) - Set the primary tree selector.

    Per WSE-STD-070 §1.1: tree_id is the primary selector for synthesis identity.
    """
    tree_id: int

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        old_tree_id = ir.tree_id
        if old_tree_id == self.tree_id:
            return EditorResult.no_op(f"tree_id already {self.tree_id}")

        ir.tree_id = self.tree_id
        return EditorResult.success(
            f"Set tree_id to {self.tree_id}",
            {"old_tree_id": old_tree_id}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        ir.tree_id = undo_data["old_tree_id"]
        return EditorResult.success(f"Restored tree_id to {undo_data['old_tree_id']}")

    @property
    def description(self) -> str:
        return f"Set tree ID to {self.tree_id}"


@dataclass
class AddRegime(EditorCommand):
    """
    ADD_REGIME(regime_id) - Add a new regime node to L2.

    Creates a regime with default kind=STABLE_TONE and empty emit_template.
    """
    regime_id: str
    kind: str = "StableTone"
    emit_template: str = ""
    dwell_ms: Optional[int] = None

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        from ..instrument_ir import RegimeNode

        # Check if regime already exists
        for node in ir.l2_regimes.regime_nodes:
            if node.id == self.regime_id:
                return EditorResult.failure(f"Regime '{self.regime_id}' already exists")

        new_node = RegimeNode(
            id=self.regime_id,
            kind=self.kind,
            emit_template=self.emit_template,
            dwell_ms=self.dwell_ms
        )
        ir.l2_regimes.regime_nodes.append(new_node)

        return EditorResult.success(
            f"Added regime '{self.regime_id}'",
            {"regime_id": self.regime_id}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        regime_id = undo_data["regime_id"]
        ir.l2_regimes.regime_nodes = [
            n for n in ir.l2_regimes.regime_nodes if n.id != regime_id
        ]
        return EditorResult.success(f"Removed regime '{regime_id}'")

    @property
    def description(self) -> str:
        return f"Add regime '{self.regime_id}'"


@dataclass
class RemoveRegime(EditorCommand):
    """
    REMOVE_REGIME(regime_id) - Remove a regime node from L2.

    Also removes all transitions involving this regime.
    """
    regime_id: str

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        # Find the regime
        regime_node = None
        for i, node in enumerate(ir.l2_regimes.regime_nodes):
            if node.id == self.regime_id:
                regime_node = node
                break

        if regime_node is None:
            return EditorResult.failure(f"Regime '{self.regime_id}' not found")

        # Store the regime and its transitions for undo
        removed_transitions = [
            t for t in ir.l2_regimes.transitions
            if t.from_regime == self.regime_id or t.to == self.regime_id
        ]

        # Remove regime
        ir.l2_regimes.regime_nodes = [
            n for n in ir.l2_regimes.regime_nodes if n.id != self.regime_id
        ]

        # Remove associated transitions
        ir.l2_regimes.transitions = [
            t for t in ir.l2_regimes.transitions
            if t.from_regime != self.regime_id and t.to != self.regime_id
        ]

        return EditorResult.success(
            f"Removed regime '{self.regime_id}' and {len(removed_transitions)} transitions",
            {
                "regime_node": regime_node,
                "removed_transitions": removed_transitions
            }
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        ir.l2_regimes.regime_nodes.append(undo_data["regime_node"])
        ir.l2_regimes.transitions.extend(undo_data["removed_transitions"])
        return EditorResult.success(f"Restored regime '{self.regime_id}'")

    @property
    def description(self) -> str:
        return f"Remove regime '{self.regime_id}'"


@dataclass
class AddTransition(EditorCommand):
    """
    ADD_TRANSITION(src, dst) - Add a transition between regimes.
    """
    src: str
    dst: str
    when: str = "true"  # Guard condition

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        from ..instrument_ir import RegimeTransition

        # Verify both regimes exist
        regime_ids = {n.id for n in ir.l2_regimes.regime_nodes}
        if self.src not in regime_ids:
            return EditorResult.failure(f"Source regime '{self.src}' not found")
        if self.dst not in regime_ids:
            return EditorResult.failure(f"Destination regime '{self.dst}' not found")

        # Check if transition already exists
        for t in ir.l2_regimes.transitions:
            if t.from_regime == self.src and t.to == self.dst:
                return EditorResult.failure(
                    f"Transition from '{self.src}' to '{self.dst}' already exists"
                )

        new_transition = RegimeTransition(
            from_regime=self.src,
            to=self.dst,
            when=self.when
        )
        ir.l2_regimes.transitions.append(new_transition)

        return EditorResult.success(
            f"Added transition {self.src} -> {self.dst}",
            {"src": self.src, "dst": self.dst}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        ir.l2_regimes.transitions = [
            t for t in ir.l2_regimes.transitions
            if not (t.from_regime == undo_data["src"] and t.to == undo_data["dst"])
        ]
        return EditorResult.success(
            f"Removed transition {undo_data['src']} -> {undo_data['dst']}"
        )

    @property
    def description(self) -> str:
        return f"Add transition {self.src} -> {self.dst}"


@dataclass
class RemoveTransition(EditorCommand):
    """
    REMOVE_TRANSITION(src, dst) - Remove a transition between regimes.
    """
    src: str
    dst: str

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        # Find the transition
        transition = None
        for t in ir.l2_regimes.transitions:
            if t.from_regime == self.src and t.to == self.dst:
                transition = t
                break

        if transition is None:
            return EditorResult.failure(
                f"Transition from '{self.src}' to '{self.dst}' not found"
            )

        ir.l2_regimes.transitions = [
            t for t in ir.l2_regimes.transitions
            if not (t.from_regime == self.src and t.to == self.dst)
        ]

        return EditorResult.success(
            f"Removed transition {self.src} -> {self.dst}",
            {"transition": transition}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        ir.l2_regimes.transitions.append(undo_data["transition"])
        return EditorResult.success(
            f"Restored transition {self.src} -> {self.dst}"
        )

    @property
    def description(self) -> str:
        return f"Remove transition {self.src} -> {self.dst}"


# =============================================================================
# PARAMETRIC COMMANDS (WSE-STD-070 §2.1)
# =============================================================================

@dataclass
class SetParam(EditorCommand):
    """
    SET_PARAM(path, value) - Set a parameter by path.

    Path uses dot notation: "l1_lift.gesture_basis.pressure.range"
    """
    path: str
    value: Any

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        success, old_value = _get_nested(ir, self.path)
        if not success:
            return EditorResult.failure(f"Path '{self.path}' not found")

        if old_value == self.value:
            return EditorResult.no_op(f"Value at '{self.path}' unchanged")

        if not _set_nested(ir, self.path, self.value):
            return EditorResult.failure(f"Failed to set '{self.path}'")

        return EditorResult.success(
            f"Set {self.path} = {self.value}",
            {"path": self.path, "old_value": copy.deepcopy(old_value)}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        _set_nested(ir, undo_data["path"], undo_data["old_value"])
        return EditorResult.success(
            f"Restored {undo_data['path']} = {undo_data['old_value']}"
        )

    @property
    def description(self) -> str:
        return f"Set {self.path} = {self.value}"


@dataclass
class DeltaParam(EditorCommand):
    """
    DELTA_PARAM(path, delta) - Add delta to a numeric parameter.

    Useful for relative adjustments (e.g., +5 cents detune).
    """
    path: str
    delta: Union[int, float]

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        success, old_value = _get_nested(ir, self.path)
        if not success:
            return EditorResult.failure(f"Path '{self.path}' not found")

        if not isinstance(old_value, (int, float)):
            return EditorResult.failure(
                f"Value at '{self.path}' is not numeric: {type(old_value)}"
            )

        new_value = old_value + self.delta
        if not _set_nested(ir, self.path, new_value):
            return EditorResult.failure(f"Failed to set '{self.path}'")

        return EditorResult.success(
            f"Set {self.path} = {old_value} + {self.delta} = {new_value}",
            {"path": self.path, "old_value": old_value}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        _set_nested(ir, undo_data["path"], undo_data["old_value"])
        return EditorResult.success(
            f"Restored {undo_data['path']} = {undo_data['old_value']}"
        )

    @property
    def description(self) -> str:
        return f"Adjust {self.path} by {self.delta:+}"


@dataclass
class SetBound(EditorCommand):
    """
    SET_BOUND(path, min, max) - Set parameter bounds.

    For parameters that have range constraints.
    """
    path: str
    min_value: Union[int, float]
    max_value: Union[int, float]

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        # Bounds are typically stored as tuple or separate min/max fields
        success, old_value = _get_nested(ir, self.path)

        if success and isinstance(old_value, tuple) and len(old_value) == 2:
            # Path points to a range tuple
            new_value = (self.min_value, self.max_value)
            if old_value == new_value:
                return EditorResult.no_op(f"Bounds at '{self.path}' unchanged")

            if not _set_nested(ir, self.path, new_value):
                return EditorResult.failure(f"Failed to set bounds at '{self.path}'")

            return EditorResult.success(
                f"Set bounds at {self.path} = ({self.min_value}, {self.max_value})",
                {"path": self.path, "old_value": old_value}
            )

        # Try separate min/max paths
        min_path = f"{self.path}.min"
        max_path = f"{self.path}.max"

        success_min, old_min = _get_nested(ir, min_path)
        success_max, old_max = _get_nested(ir, max_path)

        if success_min and success_max:
            if old_min == self.min_value and old_max == self.max_value:
                return EditorResult.no_op(f"Bounds at '{self.path}' unchanged")

            _set_nested(ir, min_path, self.min_value)
            _set_nested(ir, max_path, self.max_value)

            return EditorResult.success(
                f"Set bounds at {self.path} = ({self.min_value}, {self.max_value})",
                {"path": self.path, "old_min": old_min, "old_max": old_max}
            )

        return EditorResult.failure(f"No bounds found at '{self.path}'")

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        path = undo_data["path"]

        if "old_value" in undo_data:
            _set_nested(ir, path, undo_data["old_value"])
        else:
            _set_nested(ir, f"{path}.min", undo_data["old_min"])
            _set_nested(ir, f"{path}.max", undo_data["old_max"])

        return EditorResult.success(f"Restored bounds at {path}")

    @property
    def description(self) -> str:
        return f"Set bounds at {self.path} = ({self.min_value}, {self.max_value})"


# =============================================================================
# POLICY COMMANDS (WSE-STD-070 §2.1)
# =============================================================================

@dataclass
class SetOperatorAllowed(EditorCommand):
    """
    SET_OPERATOR_ALLOWED(op_id, bool) - Enable/disable an operator in the allowlist.

    Controls which operators are permitted for synthesis.
    """
    op_id: str
    allowed: bool

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        # Operator allowlist is typically in cmmm or policy section
        if not hasattr(ir, 'operator_allowlist'):
            ir.operator_allowlist = {}

        old_allowed = ir.operator_allowlist.get(self.op_id, True)  # Default allowed
        if old_allowed == self.allowed:
            return EditorResult.no_op(
                f"Operator '{self.op_id}' already {'allowed' if self.allowed else 'disallowed'}"
            )

        ir.operator_allowlist[self.op_id] = self.allowed

        return EditorResult.success(
            f"Operator '{self.op_id}' {'allowed' if self.allowed else 'disallowed'}",
            {"op_id": self.op_id, "old_allowed": old_allowed}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        ir.operator_allowlist[undo_data["op_id"]] = undo_data["old_allowed"]
        return EditorResult.success(
            f"Restored operator '{undo_data['op_id']}' to "
            f"{'allowed' if undo_data['old_allowed'] else 'disallowed'}"
        )

    @property
    def description(self) -> str:
        return f"{'Allow' if self.allowed else 'Disallow'} operator '{self.op_id}'"


@dataclass
class SetOperatorWeight(EditorCommand):
    """
    SET_OPERATOR_WEIGHT(op_id, weight) - Set the weight of an operator.

    Weight affects operator selection probability in CMMM.
    """
    op_id: str
    weight: float

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        if not hasattr(ir, 'operator_weights'):
            ir.operator_weights = {}

        old_weight = ir.operator_weights.get(self.op_id, 1.0)  # Default weight
        if old_weight == self.weight:
            return EditorResult.no_op(
                f"Operator '{self.op_id}' weight unchanged"
            )

        ir.operator_weights[self.op_id] = self.weight

        return EditorResult.success(
            f"Set operator '{self.op_id}' weight to {self.weight}",
            {"op_id": self.op_id, "old_weight": old_weight}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        ir.operator_weights[undo_data["op_id"]] = undo_data["old_weight"]
        return EditorResult.success(
            f"Restored operator '{undo_data['op_id']}' weight to {undo_data['old_weight']}"
        )

    @property
    def description(self) -> str:
        return f"Set operator '{self.op_id}' weight to {self.weight}"


@dataclass
class SetBudget(EditorCommand):
    """
    SET_BUDGET(limit) - Set the compute budget limit.

    Per WSE-STD-070: Controls resource allocation for synthesis.
    """
    limit: int

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        old_budget = getattr(ir, 'budget', None)
        if old_budget == self.limit:
            return EditorResult.no_op(f"Budget unchanged at {self.limit}")

        ir.budget = self.limit

        return EditorResult.success(
            f"Set budget to {self.limit}",
            {"old_budget": old_budget}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        if undo_data["old_budget"] is None:
            if hasattr(ir, 'budget'):
                delattr(ir, 'budget')
        else:
            ir.budget = undo_data["old_budget"]
        return EditorResult.success(f"Restored budget to {undo_data['old_budget']}")

    @property
    def description(self) -> str:
        return f"Set budget to {self.limit}"


# =============================================================================
# META COMMANDS (WSE-STD-070 §2.1)
# =============================================================================

@dataclass
class SetTowerLevel(EditorCommand):
    """
    SET_TOWER_LEVEL(level) - Set the tower level (0-5).

    Per WSE-STD-070 §1.1: tower_level declares the highest level explicitly authored.
    """
    level: int

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        if not 0 <= self.level <= 5:
            return EditorResult.failure(f"Invalid tower level: {self.level} (must be 0-5)")

        old_level = getattr(ir, 'tower_level', 0)
        if old_level == self.level:
            return EditorResult.no_op(f"Tower level unchanged at L{self.level}")

        ir.tower_level = self.level

        return EditorResult.success(
            f"Set tower level to L{self.level}",
            {"old_level": old_level}
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        ir.tower_level = undo_data["old_level"]
        return EditorResult.success(f"Restored tower level to L{undo_data['old_level']}")

    @property
    def description(self) -> str:
        return f"Set tower level to L{self.level}"


@dataclass
class ProjectToLevel(EditorCommand):
    """
    PROJECT_TO_LEVEL(level) - Project IR down to target level.

    Per WSE-STD-070 §5.1:
    - Removes all data above target_level
    - Re-applies completion rules
    - MUST preserve playability

    Examples:
    - L4 -> L3: freeze graph, keep params
    - L3 -> L2: freeze envelopes, keep identity
    - L2 -> L0: collapse to playable patch
    """
    target_level: int

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        if not 0 <= self.target_level <= 5:
            return EditorResult.failure(
                f"Invalid target level: {self.target_level} (must be 0-5)"
            )

        current_level = getattr(ir, 'tower_level', 0)
        if current_level <= self.target_level:
            return EditorResult.no_op(
                f"Already at or below L{self.target_level}"
            )

        # Store full state for undo
        undo_state = {
            "old_tower_level": current_level,
            "old_cmmm": copy.deepcopy(ir.cmmm) if hasattr(ir, 'cmmm') else None,
            "old_l2_regimes": copy.deepcopy(ir.l2_regimes),
            "old_l1_lift": copy.deepcopy(ir.l1_lift) if hasattr(ir, 'l1_lift') else None,
        }

        # Project down - remove data above target level
        # Per WSE-STD-070 §4.1: Projection proceeds downward

        if self.target_level < 3 and hasattr(ir, 'cmmm'):
            # L3+ data frozen
            # Keep CMMM but mark as read-only in practice
            pass

        if self.target_level < 2:
            # L2 data (regimes) simplified
            # Keep only stable regimes
            ir.l2_regimes.regime_nodes = [
                n for n in ir.l2_regimes.regime_nodes
                if n.kind in ("Idle", "StableTone")
            ]
            ir.l2_regimes.transitions = []

        if self.target_level < 1 and hasattr(ir, 'l1_lift'):
            # L1 data (gestures/macros) collapsed
            ir.l1_lift.gesture_basis = {}
            ir.l1_lift.macro_semantics = {}

        ir.tower_level = self.target_level

        return EditorResult.success(
            f"Projected from L{current_level} to L{self.target_level}",
            undo_state
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        ir.tower_level = undo_data["old_tower_level"]
        ir.l2_regimes = undo_data["old_l2_regimes"]
        if undo_data["old_cmmm"] is not None:
            ir.cmmm = undo_data["old_cmmm"]
        if undo_data["old_l1_lift"] is not None:
            ir.l1_lift = undo_data["old_l1_lift"]

        return EditorResult.success(
            f"Restored to L{undo_data['old_tower_level']}"
        )

    @property
    def description(self) -> str:
        return f"Project to L{self.target_level}"


@dataclass
class ResetToSafe(EditorCommand):
    """
    RESET_TO_SAFE() - Panic projection to safe state.

    Per WSE-STD-070 §5.2:
    - Project to L2
    - Enable only SAFE operator set
    - Reset envelopes to neutral
    - Clear all event policies
    """

    def execute(self, ir: 'InstrumentIR') -> EditorResult:
        # Store full state for undo
        undo_state = {
            "old_tower_level": getattr(ir, 'tower_level', 0),
            "old_operator_allowlist": copy.deepcopy(getattr(ir, 'operator_allowlist', {})),
            "old_operator_weights": copy.deepcopy(getattr(ir, 'operator_weights', {})),
            "old_cmmm": copy.deepcopy(ir.cmmm) if hasattr(ir, 'cmmm') else None,
            "old_l2_regimes": copy.deepcopy(ir.l2_regimes),
        }

        # Project to L2
        ir.tower_level = 2

        # Enable only SAFE operator set
        # SAFE_SET typically includes basic operators
        SAFE_SET = {"identity", "gain", "mix", "add"}
        ir.operator_allowlist = {op: True for op in SAFE_SET}

        # Reset operator weights
        ir.operator_weights = {}

        # Clear event policies in CMMM if present
        if hasattr(ir, 'cmmm') and ir.cmmm:
            if hasattr(ir.cmmm, 'event_policies'):
                ir.cmmm.event_policies = []

        # Reset envelopes to neutral in L0 graph
        from ..instrument_ir import ADSR
        for node in ir.l0_graph.nodes:
            if node.params.adsr is not None:
                node.params.adsr = ADSR()  # Neutral defaults

        return EditorResult.success(
            "Reset to safe state (L2, SAFE operators only)",
            undo_state
        )

    def undo(self, ir: 'InstrumentIR', undo_data: Dict[str, Any]) -> EditorResult:
        ir.tower_level = undo_data["old_tower_level"]
        ir.operator_allowlist = undo_data["old_operator_allowlist"]
        ir.operator_weights = undo_data["old_operator_weights"]
        ir.l2_regimes = undo_data["old_l2_regimes"]
        if undo_data["old_cmmm"] is not None:
            ir.cmmm = undo_data["old_cmmm"]

        return EditorResult.success("Restored from safe state")

    @property
    def description(self) -> str:
        return "Reset to safe state"
