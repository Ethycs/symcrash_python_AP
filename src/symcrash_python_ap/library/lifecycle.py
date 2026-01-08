"""
IPLC Lifecycle State Machine (WSE-STD-IPLC v0 §2)
===================================================

Capsule lifecycle states and transition validation.

States:
- DRAFT: Authoring form (InstrumentIR graphs, loose JSON/YAML, editor graphs)
- COMPILED: Capsule bundle produced by compiler; ABI-checked; checksummed
- PUBLISHED: Stored in PresetLibrary with stable capsule_id and version
- APPLIED: Loaded into runtime registries/program selector (no runtime histories)
- RUNNING: Being exercised via VoiceFrames + VoiceEngine
- DEPRECATED: Still loadable but not default-recommended
- RETIRED: Not loadable unless explicit override (library policy)

Transition Rules (§2.2):
- No backward transitions except by producing a new version
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from .capsule import CapsuleV0


# =============================================================================
# LIFECYCLE STATES
# =============================================================================

class LifecycleState(IntEnum):
    """
    IPLC lifecycle states per §2.1.

    Capsule progresses through these states:
    DRAFT → COMPILED → PUBLISHED → APPLIED → RUNNING
                                             ↓
                                       DEPRECATED → RETIRED
    """
    DRAFT = 0       # Authoring form (InstrumentIR, editor graphs)
    COMPILED = 1    # Capsule bundle produced, ABI-checked, checksummed
    PUBLISHED = 2   # Stored in PresetLibrary with stable capsule_id
    APPLIED = 3     # Loaded into runtime registries
    RUNNING = 4     # Being exercised via VoiceFrames
    DEPRECATED = 5  # Still loadable but not recommended
    RETIRED = 6     # Not loadable unless explicit override


# =============================================================================
# TRANSITION TABLE
# =============================================================================

# Allowed state transitions per §2.2
# Key: from_state, Value: set of allowed to_states
ALLOWED_TRANSITIONS: Dict[LifecycleState, Set[LifecycleState]] = {
    LifecycleState.DRAFT: {LifecycleState.COMPILED},
    LifecycleState.COMPILED: {LifecycleState.PUBLISHED},
    LifecycleState.PUBLISHED: {LifecycleState.APPLIED, LifecycleState.DEPRECATED},
    LifecycleState.APPLIED: {LifecycleState.RUNNING},
    LifecycleState.RUNNING: set(),  # Terminal runtime state
    LifecycleState.DEPRECATED: {LifecycleState.RETIRED, LifecycleState.APPLIED},
    LifecycleState.RETIRED: set(),  # Terminal library state
}

# Human-readable transition triggers
TRANSITION_TRIGGERS: Dict[Tuple[LifecycleState, LifecycleState], str] = {
    (LifecycleState.DRAFT, LifecycleState.COMPILED): "compiler emits capsule bundle",
    (LifecycleState.COMPILED, LifecycleState.PUBLISHED): "library import transaction succeeds",
    (LifecycleState.PUBLISHED, LifecycleState.APPLIED): "runtime apply_capsule()",
    (LifecycleState.APPLIED, LifecycleState.RUNNING): "first VoiceFrame accepted",
    (LifecycleState.PUBLISHED, LifecycleState.DEPRECATED): "library metadata update",
    (LifecycleState.DEPRECATED, LifecycleState.APPLIED): "runtime apply_capsule()",
    (LifecycleState.DEPRECATED, LifecycleState.RETIRED): "library policy update",
}


# =============================================================================
# CAPSULE RECORD
# =============================================================================

@dataclass
class CapsuleRecord:
    """
    Capsule with lifecycle metadata per §1.2.

    A stored record includes:
    - capsule_id (from CapsuleV0.identity)
    - revision_id (monotonically increasing per capsule_id)
    - manifest (stored in capsule)
    - content hashes (CRCs stored in manifest)
    - storage location pointer
    - flags: deprecated/retired
    """
    capsule: CapsuleV0
    state: LifecycleState
    revision_id: int = 1
    storage_path: Optional[str] = None
    deprecated: bool = False
    retired: bool = False
    compatible: bool = True  # ABI compatibility flag
    imported_at: Optional[str] = None  # ISO timestamp
    notes: str = ""

    @property
    def capsule_id(self) -> str:
        """Get capsule_id from underlying capsule."""
        return self.capsule.identity.capsule_id

    @property
    def name(self) -> str:
        """Get name from underlying capsule."""
        return self.capsule.identity.name

    @property
    def kind(self):
        """Get kind from underlying capsule."""
        return self.capsule.identity.kind

    @property
    def tags(self) -> List[str]:
        """Get tags from underlying capsule."""
        return self.capsule.identity.tags

    def is_loadable(self, allow_retired: bool = False) -> bool:
        """
        Check if capsule can be loaded per §2.1.

        RETIRED capsules are not loadable unless explicit override.
        """
        if self.retired and not allow_retired:
            return False
        return True

    def is_recommended(self) -> bool:
        """Check if capsule is recommended (not deprecated, not retired)."""
        return not self.deprecated and not self.retired


# =============================================================================
# TRANSITION VALIDATION
# =============================================================================

def validate_transition(from_state: LifecycleState, to_state: LifecycleState) -> bool:
    """
    Check if transition is allowed per §2.2.

    Args:
        from_state: Current state
        to_state: Target state

    Returns:
        True if transition is allowed
    """
    allowed = ALLOWED_TRANSITIONS.get(from_state, set())
    return to_state in allowed


def get_transition_trigger(from_state: LifecycleState, to_state: LifecycleState) -> Optional[str]:
    """Get human-readable trigger for a transition."""
    return TRANSITION_TRIGGERS.get((from_state, to_state))


class TransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


def transition(
    record: CapsuleRecord,
    to_state: LifecycleState,
    allow_backward: bool = False,
) -> CapsuleRecord:
    """
    Transition capsule record to new state.

    Args:
        record: Current capsule record
        to_state: Target state
        allow_backward: If True, allow backward transitions (for version reset)

    Returns:
        New CapsuleRecord with updated state

    Raises:
        TransitionError: If transition is not allowed
    """
    from_state = record.state

    # Check if transition is valid
    if not validate_transition(from_state, to_state):
        if not allow_backward:
            trigger = get_transition_trigger(from_state, to_state)
            raise TransitionError(
                f"Invalid transition: {from_state.name} → {to_state.name}. "
                f"Expected trigger: {trigger or 'N/A'}"
            )

    # Create new record with updated state
    return CapsuleRecord(
        capsule=record.capsule,
        state=to_state,
        revision_id=record.revision_id,
        storage_path=record.storage_path,
        deprecated=record.deprecated,
        retired=record.retired,
        compatible=record.compatible,
        imported_at=record.imported_at,
        notes=record.notes,
    )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_draft_record(capsule: CapsuleV0) -> CapsuleRecord:
    """Create a new DRAFT capsule record."""
    return CapsuleRecord(
        capsule=capsule,
        state=LifecycleState.DRAFT,
        revision_id=0,  # Not yet assigned
    )


def create_compiled_record(capsule: CapsuleV0) -> CapsuleRecord:
    """Create a COMPILED capsule record (from compiler output)."""
    return CapsuleRecord(
        capsule=capsule,
        state=LifecycleState.COMPILED,
        revision_id=0,  # Not yet assigned
    )


def create_published_record(
    capsule: CapsuleV0,
    revision_id: int,
    storage_path: str,
    compatible: bool = True,
) -> CapsuleRecord:
    """Create a PUBLISHED capsule record (from library import)."""
    return CapsuleRecord(
        capsule=capsule,
        state=LifecycleState.PUBLISHED,
        revision_id=revision_id,
        storage_path=storage_path,
        compatible=compatible,
        imported_at=datetime.utcnow().isoformat(),
    )


# =============================================================================
# STATE QUERIES
# =============================================================================

def get_allowed_transitions(state: LifecycleState) -> List[LifecycleState]:
    """Get list of allowed next states from current state."""
    return list(ALLOWED_TRANSITIONS.get(state, set()))


def is_terminal_state(state: LifecycleState) -> bool:
    """Check if state is terminal (no outgoing transitions)."""
    return len(ALLOWED_TRANSITIONS.get(state, set())) == 0


def is_library_state(state: LifecycleState) -> bool:
    """Check if state is managed by library (PUBLISHED, DEPRECATED, RETIRED)."""
    return state in {LifecycleState.PUBLISHED, LifecycleState.DEPRECATED, LifecycleState.RETIRED}


def is_runtime_state(state: LifecycleState) -> bool:
    """Check if state is runtime (APPLIED, RUNNING)."""
    return state in {LifecycleState.APPLIED, LifecycleState.RUNNING}


__all__ = [
    'LifecycleState',
    'CapsuleRecord',
    'ALLOWED_TRANSITIONS',
    'TRANSITION_TRIGGERS',
    'validate_transition',
    'get_transition_trigger',
    'transition',
    'TransitionError',
    'create_draft_record',
    'create_compiled_record',
    'create_published_record',
    'get_allowed_transitions',
    'is_terminal_state',
    'is_library_state',
    'is_runtime_state',
]
