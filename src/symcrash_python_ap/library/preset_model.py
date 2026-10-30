"""
WSE-STD-070 — Presets, Editor Operations, and Tower Completion (v0)
====================================================================

This module implements the preset object model, editor primitives,
tower completion rules, and projection maps.

Key Guarantees:
1. Presets are valid even with only one layer present
2. Missing layers are filled deterministically (Tower Completion)
3. Editors operate via typed primitives, not raw mutation
4. Projection maps allow safe descent through tower levels
5. Same preset + same input = same output (determinism)

Architecture:
- L0 (Patch): MIDI maps, gains - always playable
- L1 (Family): Regime hints, IntentFrame - recognizable class
- L2 (Instrument): Params, bounds, VoiceFrame - identity preserved
- L3 (Cockpit): Weights, envelopes - no destructive edits
- L4 (Forge): Graph edits, regime graph - always projectable
- L5 (Lab): New ops/trees, AST - must compile down
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum, IntEnum
from copy import deepcopy
import struct
import zlib
import math


# =============================================================================
# CONSTANTS
# =============================================================================

# Preset magic bytes
PRESET_MAGIC = 0x50525354  # "PRST"

# Safe operator set (default when L3 missing)
SAFE_OPERATOR_SET = {
    'Identity', 'Tilt', 'FineClamp', 'IntraMix',
    'DiffuseUp', 'DiffuseDown', 'NormalizeScale',
    'GluePair', 'SheafProject',
}

# Default regime when L1 missing
DEFAULT_REGIME = 'STABLE'


# =============================================================================
# TOWER LEVELS (Section 3)
# =============================================================================

class TowerLevel(IntEnum):
    """
    Tower levels per WSE-STD-070 Section 3.

    | Level | Name       | Description                    |
    |-------|------------|--------------------------------|
    | L0    | Patch      | MIDI maps, gains - always playable |
    | L1    | Family     | Regime hints - recognizable class |
    | L2    | Instrument | Params, bounds - identity preserved |
    | L3    | Cockpit    | Weights, envelopes - no destructive edits |
    | L4    | Forge      | Graph edits - always projectable |
    | L5    | Lab        | New ops/trees - must compile down |
    """
    L0_PATCH = 0
    L1_FAMILY = 1
    L2_INSTRUMENT = 2
    L3_COCKPIT = 3
    L4_FORGE = 4
    L5_LAB = 5


# Tower level metadata
TOWER_LEVEL_INFO: Dict[TowerLevel, Dict[str, Any]] = {
    TowerLevel.L0_PATCH: {
        'name': 'Patch',
        'editor_primitives': ['MIDI maps', 'gains'],
        'ir_primitives': None,
        'engine_primitives': ['Voice alloc only'],
        'invariant': 'Always playable',
    },
    TowerLevel.L1_FAMILY: {
        'name': 'Family',
        'editor_primitives': ['Regime hints'],
        'ir_primitives': 'IntentFrame',
        'engine_primitives': ['Voice Engine'],
        'invariant': 'Recognizable class',
    },
    TowerLevel.L2_INSTRUMENT: {
        'name': 'Instrument',
        'editor_primitives': ['Params', 'bounds'],
        'ir_primitives': 'VoiceFrame',
        'engine_primitives': ['Symbol Engine'],
        'invariant': 'Identity preserved',
    },
    TowerLevel.L3_COCKPIT: {
        'name': 'Cockpit',
        'editor_primitives': ['Weights', 'envelopes'],
        'ir_primitives': 'Operator sets',
        'engine_primitives': ['Symbol + Safety'],
        'invariant': 'No destructive edits',
    },
    TowerLevel.L4_FORGE: {
        'name': 'Forge',
        'editor_primitives': ['Graph edits'],
        'ir_primitives': 'Regime graph',
        'engine_primitives': ['Compiler'],
        'invariant': 'Always projectable',
    },
    TowerLevel.L5_LAB: {
        'name': 'Lab',
        'editor_primitives': ['New ops/trees'],
        'ir_primitives': 'AST / meta',
        'engine_primitives': ['Offline tools'],
        'invariant': 'Must compile down',
    },
}


def get_tower_level_info(level: TowerLevel) -> Dict[str, Any]:
    """Get metadata for a tower level."""
    return TOWER_LEVEL_INFO[level]


# =============================================================================
# PRESET HEADER (Section 1.1)
# =============================================================================

@dataclass
class PresetHeader:
    """
    Preset header per WSE-STD-070 Section 1.1.

    Every preset MUST contain:
    - preset_id: u32
    - preset_version: u16
    - tower_level: u8 (highest level explicitly authored)
    - tree_id: u16 (primary selector for synthesis identity)
    - flags: u16
    - crc32: u32
    """
    preset_id: int
    preset_version: int = 1
    tower_level: TowerLevel = TowerLevel.L2_INSTRUMENT
    tree_id: int = 0
    flags: int = 0
    crc32: int = 0

    # Header size in bytes
    HEADER_SIZE = 16  # 4 + 2 + 1 + 1(pad) + 2 + 2 + 4

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            '>IHBBHHI',  # Big-endian: u32, u16, u8, u8, u16, u16, u32
            self.preset_id,
            self.preset_version,
            self.tower_level.value,
            0,  # padding
            self.tree_id,
            self.flags,
            self.crc32,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'PresetHeader':
        """Deserialize header from bytes."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} < {cls.HEADER_SIZE}")

        preset_id, preset_version, tower_level, padding, tree_id, flags = struct.unpack(
            '>IHBBHH', data[:12]
        )
        if padding != 0:
            raise ValueError('Preset header padding must be zero')
        crc32 = struct.unpack('>I', data[12:16])[0]

        return cls(
            preset_id=preset_id,
            preset_version=preset_version,
            tower_level=TowerLevel(tower_level),
            tree_id=tree_id,
            flags=flags,
            crc32=crc32,
        )


# =============================================================================
# TOWER LAYERS (Section 1.2)
# =============================================================================

@dataclass
class L0PatchLayer:
    """L0 Patch Layer - MIDI maps, gains."""
    midi_channel: int = 0
    gain: float = 1.0
    pan: float = 0.0
    transpose: int = 0
    velocity_curve: str = 'linear'

    # Patch is always playable
    def is_valid(self) -> bool:
        return True


@dataclass
class L1FamilyLayer:
    """L1 Family Layer - Regime hints, family classification."""
    family_name: str = 'DEFAULT'
    regimes: Set[str] = field(default_factory=lambda: {DEFAULT_REGIME})
    initial_regime: str = DEFAULT_REGIME

    # Family weights for morphing
    family_weights: Dict[str, float] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return len(self.regimes) > 0 and self.initial_regime in self.regimes


@dataclass
class L2InstrumentLayer:
    """L2 Instrument Layer - Params, bounds, identity."""
    # Parameter values (path -> value)
    params: Dict[str, Any] = field(default_factory=dict)

    # Parameter bounds (path -> (min, max))
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Operator allowlist
    allowed_operators: Set[str] = field(default_factory=lambda: set(SAFE_OPERATOR_SET))

    # Tree ADSR settings
    tree_adsr: Dict[str, float] = field(default_factory=lambda: {
        'attack': 0.01,
        'decay': 0.1,
        'sustain': 0.8,
        'release': 0.3,
    })

    def is_valid(self) -> bool:
        return len(self.allowed_operators) > 0


@dataclass
class L3CockpitLayer:
    """L3 Cockpit Layer - Weights, envelopes, runtime tuning."""
    # Operator weights (op_name -> weight)
    operator_weights: Dict[str, float] = field(default_factory=dict)

    # Envelope bounds (env_name -> (min, max))
    envelope_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Budget limits
    compute_budget: int = 1000
    memory_budget: int = 1000

    # Event policies (lane_id -> policy)
    event_policies: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.compute_budget > 0 and self.memory_budget > 0


@dataclass
class L4ForgeLayer:
    """L4 Forge Layer - Graph edits, regime graph."""
    # Regime graph (regime_id -> list of allowed transitions)
    regime_graph: Dict[str, List[str]] = field(default_factory=dict)

    # Graph edit history (for undo)
    edit_history: List[Dict[str, Any]] = field(default_factory=list)

    # Transition probabilities
    transition_probs: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return True  # Forge layer is always projectable


@dataclass
class L5LabLayer:
    """L5 Lab Layer - New ops/trees, experimental."""
    # Custom operator definitions
    custom_operators: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Custom tree definitions
    custom_trees: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # AST for compilation
    ast_data: Optional[Dict[str, Any]] = None

    def is_valid(self) -> bool:
        return True  # Lab must compile down


# =============================================================================
# PRESET BODY (Section 1.2)
# =============================================================================

@dataclass
class PresetBody:
    """
    Preset body per WSE-STD-070 Section 1.2.

    A preset MUST be valid even if only one layer is present.
    Missing layers MUST be filled by Tower Completion.
    """
    L0_patch: Optional[L0PatchLayer] = None
    L1_family: Optional[L1FamilyLayer] = None
    L2_instrument: Optional[L2InstrumentLayer] = None
    L3_cockpit: Optional[L3CockpitLayer] = None
    L4_forge: Optional[L4ForgeLayer] = None
    L5_lab: Optional[L5LabLayer] = None
    # (generated object, pristine copy). Identity detects replacement with an
    # equal value; equality detects in-place edits of nested lists/maps/sets.
    _completed_layers: Dict[TowerLevel, Tuple[Any, Any]] = field(
        default_factory=dict, init=False, repr=False, compare=False)

    def get_layer(self, level: TowerLevel) -> Optional[Any]:
        """Get layer by tower level."""
        mapping = {
            TowerLevel.L0_PATCH: self.L0_patch,
            TowerLevel.L1_FAMILY: self.L1_family,
            TowerLevel.L2_INSTRUMENT: self.L2_instrument,
            TowerLevel.L3_COCKPIT: self.L3_cockpit,
            TowerLevel.L4_FORGE: self.L4_forge,
            TowerLevel.L5_LAB: self.L5_lab,
        }
        return mapping.get(level)

    def set_layer(self, level: TowerLevel, layer: Any) -> None:
        """Set layer by tower level."""
        level = TowerLevel(level)
        self._completed_layers.pop(level, None)
        if level == TowerLevel.L0_PATCH:
            self.L0_patch = layer
        elif level == TowerLevel.L1_FAMILY:
            self.L1_family = layer
        elif level == TowerLevel.L2_INSTRUMENT:
            self.L2_instrument = layer
        elif level == TowerLevel.L3_COCKPIT:
            self.L3_cockpit = layer
        elif level == TowerLevel.L4_FORGE:
            self.L4_forge = layer
        elif level == TowerLevel.L5_LAB:
            self.L5_lab = layer

    def set_completed_layer(self, level: TowerLevel, layer: Any) -> None:
        """Install a generated default without making it authored data."""
        self.set_layer(level, layer)
        self._completed_layers[level] = (layer, deepcopy(layer))

    def mark_explicit(self, level: TowerLevel) -> None:
        """Record an explicit edit even when its value equals the default."""
        self._completed_layers.pop(level, None)

    def is_explicit(self, level: TowerLevel) -> bool:
        layer = self.get_layer(level)
        if layer is None:
            return False
        generated = self._completed_layers.get(level)
        return generated is None or layer is not generated[0] or layer != generated[1]

    def has_layer(self, level: TowerLevel) -> bool:
        """Check if layer is present."""
        return self.get_layer(level) is not None

    def highest_explicit_level(self) -> TowerLevel:
        """Get highest explicitly authored level."""
        for level in reversed(TowerLevel):
            if self.is_explicit(level):
                return level
        return TowerLevel.L0_PATCH


# =============================================================================
# EDITOR PRIMITIVES (Section 2)
# =============================================================================

class EditorOpType(Enum):
    """Editor operation types per WSE-STD-070 Section 2.1."""
    # Structural
    SET_TREE_ID = "set_tree_id"
    ADD_REGIME = "add_regime"
    REMOVE_REGIME = "remove_regime"
    ADD_TRANSITION = "add_transition"
    REMOVE_TRANSITION = "remove_transition"

    # Parametric
    SET_PARAM = "set_param"
    DELTA_PARAM = "delta_param"
    SET_BOUND = "set_bound"

    # Policy
    SET_OPERATOR_ALLOWED = "set_operator_allowed"
    SET_OPERATOR_WEIGHT = "set_operator_weight"
    SET_BUDGET = "set_budget"

    # Meta
    SET_TOWER_LEVEL = "set_tower_level"
    PROJECT_TO_LEVEL = "project_to_level"
    RESET_TO_SAFE = "reset_to_safe"


@dataclass
class EditorOp:
    """
    Editor operation per WSE-STD-070 Section 2.

    All editor ops MUST be pure and replayable.
    Editors MUST NOT emit operators, events, or timing logic.
    """
    op_type: EditorOpType
    args: Dict[str, Any] = field(default_factory=dict)

    # Timestamp for replay ordering
    timestamp: int = 0

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        """Validate operation arguments."""
        if not isinstance(self.op_type, EditorOpType) or not isinstance(self.args, dict):
            raise ValueError('Invalid editor operation')
        required_args = {
            EditorOpType.SET_TREE_ID: ['tree_id'],
            EditorOpType.ADD_REGIME: ['regime_id'],
            EditorOpType.REMOVE_REGIME: ['regime_id'],
            EditorOpType.ADD_TRANSITION: ['src', 'dst'],
            EditorOpType.REMOVE_TRANSITION: ['src', 'dst'],
            EditorOpType.SET_PARAM: ['path', 'value'],
            EditorOpType.DELTA_PARAM: ['path', 'delta'],
            EditorOpType.SET_BOUND: ['path', 'min', 'max'],
            EditorOpType.SET_OPERATOR_ALLOWED: ['op_id', 'allowed'],
            EditorOpType.SET_OPERATOR_WEIGHT: ['op_id', 'weight'],
            EditorOpType.SET_BUDGET: ['limit'],
            EditorOpType.SET_TOWER_LEVEL: ['level'],
            EditorOpType.PROJECT_TO_LEVEL: ['level'],
            EditorOpType.RESET_TO_SAFE: [],
        }

        for arg in required_args.get(self.op_type, []):
            if arg not in self.args:
                raise ValueError(f"Missing required argument '{arg}' for {self.op_type}")
        for name in ('path', 'op_id', 'regime_id', 'src', 'dst'):
            if name in self.args and (not isinstance(self.args[name], str) or not self.args[name]):
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('delta', 'min', 'max', 'weight'):
            if name in self.args:
                value = self.args[name]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f'{name} must be finite numeric data')
        if self.op_type == EditorOpType.SET_BOUND and self.args['min'] > self.args['max']:
            raise ValueError('Minimum bound exceeds maximum')
        if self.op_type == EditorOpType.SET_TREE_ID:
            if type(self.args['tree_id']) is not int or not 0 <= self.args['tree_id'] <= 0xFFFF:
                raise ValueError('tree_id must be a u16')
        if self.op_type == EditorOpType.SET_OPERATOR_ALLOWED and type(self.args['allowed']) is not bool:
            raise ValueError('allowed must be a boolean')
        if self.op_type == EditorOpType.SET_OPERATOR_WEIGHT and not 0 <= self.args['weight'] <= 1:
            raise ValueError('Operator weight must be in [0, 1]')
        if self.op_type == EditorOpType.SET_BUDGET:
            if type(self.args['limit']) is not int or self.args['limit'] <= 0:
                raise ValueError('Budget must be a positive integer')
        if self.op_type in (EditorOpType.SET_TOWER_LEVEL, EditorOpType.PROJECT_TO_LEVEL):
            self.args['level'] = TowerLevel(self.args['level'])


# Editor operation factory functions
def set_tree_id(tree_id: int) -> EditorOp:
    """Create SET_TREE_ID operation."""
    return EditorOp(EditorOpType.SET_TREE_ID, {'tree_id': tree_id})


def add_regime(regime_id: str) -> EditorOp:
    """Create ADD_REGIME operation."""
    return EditorOp(EditorOpType.ADD_REGIME, {'regime_id': regime_id})


def remove_regime(regime_id: str) -> EditorOp:
    """Create REMOVE_REGIME operation."""
    return EditorOp(EditorOpType.REMOVE_REGIME, {'regime_id': regime_id})


def add_transition(src: str, dst: str) -> EditorOp:
    """Create ADD_TRANSITION operation."""
    return EditorOp(EditorOpType.ADD_TRANSITION, {'src': src, 'dst': dst})


def remove_transition(src: str, dst: str) -> EditorOp:
    """Create REMOVE_TRANSITION operation."""
    return EditorOp(EditorOpType.REMOVE_TRANSITION, {'src': src, 'dst': dst})


def set_param(path: str, value: Any) -> EditorOp:
    """Create SET_PARAM operation."""
    return EditorOp(EditorOpType.SET_PARAM, {'path': path, 'value': value})


def delta_param(path: str, delta: float) -> EditorOp:
    """Create DELTA_PARAM operation."""
    return EditorOp(EditorOpType.DELTA_PARAM, {'path': path, 'delta': delta})


def set_bound(path: str, min_val: float, max_val: float) -> EditorOp:
    """Create SET_BOUND operation."""
    return EditorOp(EditorOpType.SET_BOUND, {'path': path, 'min': min_val, 'max': max_val})


def set_operator_allowed(op_id: str, allowed: bool) -> EditorOp:
    """Create SET_OPERATOR_ALLOWED operation."""
    return EditorOp(EditorOpType.SET_OPERATOR_ALLOWED, {'op_id': op_id, 'allowed': allowed})


def set_operator_weight(op_id: str, weight: float) -> EditorOp:
    """Create SET_OPERATOR_WEIGHT operation."""
    return EditorOp(EditorOpType.SET_OPERATOR_WEIGHT, {'op_id': op_id, 'weight': weight})


def set_budget(limit: int) -> EditorOp:
    """Create SET_BUDGET operation."""
    return EditorOp(EditorOpType.SET_BUDGET, {'limit': limit})


def set_tower_level(level: TowerLevel) -> EditorOp:
    """Create SET_TOWER_LEVEL operation."""
    return EditorOp(EditorOpType.SET_TOWER_LEVEL, {'level': level})


def project_to_level_op(level: TowerLevel) -> EditorOp:
    """Create PROJECT_TO_LEVEL operation."""
    return EditorOp(EditorOpType.PROJECT_TO_LEVEL, {'level': level})


def reset_to_safe_op() -> EditorOp:
    """Create RESET_TO_SAFE operation."""
    return EditorOp(EditorOpType.RESET_TO_SAFE, {})


# =============================================================================
# TOWER COMPLETION (Section 4)
# =============================================================================

def complete_l1_family(preset: 'Preset') -> L1FamilyLayer:
    """
    Complete missing L1 (Family) layer.

    Per Section 4.2:
    - Infer family from tree_id
    - Default regimes = {STABLE}
    """
    # Infer family from tree_id
    tree_id = preset.header.tree_id
    family_name = f"TREE_{tree_id}"

    return L1FamilyLayer(
        family_name=family_name,
        regimes={DEFAULT_REGIME},
        initial_regime=DEFAULT_REGIME,
    )


def complete_l2_instrument(preset: 'Preset') -> L2InstrumentLayer:
    """
    Complete missing L2 (Instrument) layer.

    Per Section 4.2:
    - Bind tree ADSR = neutral
    - Operator allowlist = SAFE_SET
    """
    return L2InstrumentLayer(
        params={},
        bounds={},
        allowed_operators=set(SAFE_OPERATOR_SET),
        tree_adsr={
            'attack': 0.01,
            'decay': 0.1,
            'sustain': 0.8,
            'release': 0.3,
        },
    )


def complete_l3_cockpit(preset: 'Preset') -> L3CockpitLayer:
    """
    Complete missing L3 (Cockpit) layer.

    Per Section 4.2:
    - Envelope bounds = conservative
    - Budget = nominal
    - All unsafe ops disabled
    """
    return L3CockpitLayer(
        operator_weights={op: 1.0 for op in SAFE_OPERATOR_SET},
        envelope_bounds={
            'attack': (0.001, 1.0),
            'decay': (0.01, 2.0),
            'sustain': (0.0, 1.0),
            'release': (0.01, 5.0),
        },
        compute_budget=1000,
        memory_budget=1000,
        event_policies={},
    )


def complete_l4_forge(preset: 'Preset') -> L4ForgeLayer:
    """
    Complete missing L4 (Forge) layer.

    Per Section 4.2:
    - Graph edits forbidden
    - Transitions fixed
    """
    # Get regimes from L1 if available
    regimes = {DEFAULT_REGIME}
    if preset.body.L1_family:
        regimes = preset.body.L1_family.regimes

    # Create fixed transitions (self-loops only)
    regime_graph = {r: [r] for r in regimes}

    return L4ForgeLayer(
        regime_graph=regime_graph,
        edit_history=[],
        transition_probs={},
    )


def complete_l5_lab(preset: 'Preset') -> L5LabLayer:
    """
    Complete missing L5 (Lab) layer.

    Per Section 4.2:
    - No-op (lab never implied)
    """
    return L5LabLayer(
        custom_operators={},
        custom_trees={},
        ast_data=None,
    )


def complete_tower(preset: 'Preset', target_level: TowerLevel = TowerLevel.L5_LAB) -> None:
    """
    Apply tower completion rules per Section 4.

    Completion always proceeds upward: Lk → Lk+1 → ... → target_level

    Normative rule: Completion MUST be pure, deterministic, and side-effect free.
    """
    completion_funcs = {
        TowerLevel.L1_FAMILY: complete_l1_family,
        TowerLevel.L2_INSTRUMENT: complete_l2_instrument,
        TowerLevel.L3_COCKPIT: complete_l3_cockpit,
        TowerLevel.L4_FORGE: complete_l4_forge,
        TowerLevel.L5_LAB: complete_l5_lab,
    }

    # Always ensure L0 exists
    if not preset.body.is_explicit(TowerLevel.L0_PATCH):
        preset.body.set_completed_layer(TowerLevel.L0_PATCH, L0PatchLayer())

    # Complete upward from L1 to target_level
    for level in TowerLevel:
        if level == TowerLevel.L0_PATCH:
            continue
        if level > target_level:
            break
        if not preset.body.is_explicit(level):
            func = completion_funcs.get(level)
            if func:
                preset.body.set_completed_layer(level, func(preset))


# =============================================================================
# PROJECTION MAPS (Section 5)
# =============================================================================

def project_l5_to_l4(preset: 'Preset') -> None:
    """Project from L5 (Lab) to L4 (Forge)."""
    # Remove L5 data
    preset.body.L5_lab = None


def project_l4_to_l3(preset: 'Preset') -> None:
    """Project from L4 (Forge) to L3 (Cockpit).

    Freeze graph, keep params.
    """
    # Remove L4/L5 data
    preset.body.L4_forge = None
    preset.body.L5_lab = None


def project_l3_to_l2(preset: 'Preset') -> None:
    """Project from L3 (Cockpit) to L2 (Instrument).

    Freeze envelopes, keep identity.
    """
    # Remove L3/L4/L5 data
    preset.body.L3_cockpit = None
    preset.body.L4_forge = None
    preset.body.L5_lab = None


def project_l2_to_l1(preset: 'Preset') -> None:
    """Project from L2 (Instrument) to L1 (Family)."""
    preset.body.L2_instrument = None
    preset.body.L3_cockpit = None
    preset.body.L4_forge = None
    preset.body.L5_lab = None


def project_l1_to_l0(preset: 'Preset') -> None:
    """Project from L1 (Family) to L0 (Patch).

    Collapse to playable patch.
    """
    preset.body.L1_family = None
    preset.body.L2_instrument = None
    preset.body.L3_cockpit = None
    preset.body.L4_forge = None
    preset.body.L5_lab = None


def project_to_level(preset: 'Preset', target_level: TowerLevel) -> None:
    """
    Project preset to target level per Section 5.1.

    Projection always proceeds downward: Lj → Lj-1 → ... → target_level

    Normative behavior:
    - Removes all data above target_level
    - Re-applies completion rules
    - MUST preserve playability
    """
    projection_funcs = [
        (TowerLevel.L5_LAB, project_l5_to_l4),
        (TowerLevel.L4_FORGE, project_l4_to_l3),
        (TowerLevel.L3_COCKPIT, project_l3_to_l2),
        (TowerLevel.L2_INSTRUMENT, project_l2_to_l1),
        (TowerLevel.L1_FAMILY, project_l1_to_l0),
    ]

    # Apply projections from highest to target
    for level, func in projection_funcs:
        if level > target_level:
            func(preset)

    # Update header
    preset.header.tower_level = target_level

    # Re-apply completion to ensure validity
    complete_tower(preset, target_level)


def reset_to_safe(preset: 'Preset') -> None:
    """
    Panic projection per Section 5.2.

    RESET_TO_SAFE() MUST:
    - Project to L2
    - Enable only SAFE operator set
    - Reset envelopes to neutral
    - Clear all event policies
    """
    # Project to L2
    project_to_level(preset, TowerLevel.L2_INSTRUMENT)

    # Ensure L2 exists and is safe
    if not preset.body.L2_instrument:
        preset.body.L2_instrument = complete_l2_instrument(preset)
    else:
        # Reset to safe operators only
        preset.body.L2_instrument.allowed_operators = set(SAFE_OPERATOR_SET)
        # Reset ADSR to neutral
        preset.body.L2_instrument.tree_adsr = {
            'attack': 0.01,
            'decay': 0.1,
            'sustain': 0.8,
            'release': 0.3,
        }
    preset.body.mark_explicit(TowerLevel.L2_INSTRUMENT)


# =============================================================================
# PRESET CLASS (Complete)
# =============================================================================

@dataclass
class Preset:
    """
    Complete preset per WSE-STD-070.

    A preset is a serialized description of an instrument at some Tower level.
    """
    header: PresetHeader
    body: PresetBody = field(default_factory=PresetBody)

    # Edit history for replay
    edit_history: List[EditorOp] = field(default_factory=list)

    def apply_op(self, op: EditorOp) -> None:
        """Apply an editor operation to this preset."""
        # Work on our own copy: subsequent caller changes must not alter the
        # preset or its recorded edit stream.
        op = deepcopy(op)
        op._validate()
        edited_level = {
            EditorOpType.ADD_REGIME: TowerLevel.L1_FAMILY,
            EditorOpType.REMOVE_REGIME: TowerLevel.L1_FAMILY,
            EditorOpType.SET_PARAM: TowerLevel.L2_INSTRUMENT,
            EditorOpType.DELTA_PARAM: TowerLevel.L2_INSTRUMENT,
            EditorOpType.SET_BOUND: TowerLevel.L2_INSTRUMENT,
            EditorOpType.SET_OPERATOR_ALLOWED: TowerLevel.L2_INSTRUMENT,
            EditorOpType.SET_OPERATOR_WEIGHT: TowerLevel.L3_COCKPIT,
            EditorOpType.SET_BUDGET: TowerLevel.L3_COCKPIT,
            EditorOpType.ADD_TRANSITION: TowerLevel.L4_FORGE,
            EditorOpType.REMOVE_TRANSITION: TowerLevel.L4_FORGE,
        }.get(op.op_type)
        if edited_level is not None:
            self.complete(max(self.header.tower_level, edited_level))

        if op.op_type == EditorOpType.SET_TREE_ID:
            self.header.tree_id = op.args['tree_id']

        elif op.op_type == EditorOpType.ADD_REGIME:
            if self.body.L1_family:
                self.body.L1_family.regimes.add(op.args['regime_id'])

        elif op.op_type == EditorOpType.REMOVE_REGIME:
            if self.body.L1_family:
                family = self.body.L1_family
                regime = op.args['regime_id']
                if family.regimes == {regime}:
                    raise ValueError('Cannot remove the final regime')
                family.regimes.discard(regime)
                if family.initial_regime == regime:
                    family.initial_regime = sorted(family.regimes)[0]

        elif op.op_type == EditorOpType.ADD_TRANSITION:
            if self.body.L4_forge:
                src, dst = op.args['src'], op.args['dst']
                if src not in self.body.L4_forge.regime_graph:
                    self.body.L4_forge.regime_graph[src] = []
                if dst not in self.body.L4_forge.regime_graph[src]:
                    self.body.L4_forge.regime_graph[src].append(dst)

        elif op.op_type == EditorOpType.REMOVE_TRANSITION:
            if self.body.L4_forge:
                src, dst = op.args['src'], op.args['dst']
                if src in self.body.L4_forge.regime_graph:
                    if dst in self.body.L4_forge.regime_graph[src]:
                        self.body.L4_forge.regime_graph[src].remove(dst)

        elif op.op_type == EditorOpType.SET_PARAM:
            if self.body.L2_instrument:
                self.body.L2_instrument.params[op.args['path']] = op.args['value']

        elif op.op_type == EditorOpType.DELTA_PARAM:
            if self.body.L2_instrument:
                path = op.args['path']
                delta = op.args['delta']
                current = self.body.L2_instrument.params.get(path, 0)
                self.body.L2_instrument.params[path] = current + delta

        elif op.op_type == EditorOpType.SET_BOUND:
            if self.body.L2_instrument:
                path = op.args['path']
                self.body.L2_instrument.bounds[path] = (op.args['min'], op.args['max'])

        elif op.op_type == EditorOpType.SET_OPERATOR_ALLOWED:
            if self.body.L2_instrument:
                op_id = op.args['op_id']
                if op.args['allowed']:
                    self.body.L2_instrument.allowed_operators.add(op_id)
                else:
                    if self.body.L2_instrument.allowed_operators == {op_id}:
                        raise ValueError('Cannot remove the final allowed operator')
                    self.body.L2_instrument.allowed_operators.discard(op_id)

        elif op.op_type == EditorOpType.SET_OPERATOR_WEIGHT:
            if self.body.L3_cockpit:
                self.body.L3_cockpit.operator_weights[op.args['op_id']] = op.args['weight']

        elif op.op_type == EditorOpType.SET_BUDGET:
            if self.body.L3_cockpit:
                self.body.L3_cockpit.compute_budget = op.args['limit']
                self.body.L3_cockpit.memory_budget = op.args['limit']

        elif op.op_type == EditorOpType.SET_TOWER_LEVEL:
            level = TowerLevel(op.args['level'])
            if level < self.body.highest_explicit_level():
                self.project(level)
            else:
                self.complete(level)
            self.body.mark_explicit(level)
            self.header.tower_level = level

        elif op.op_type == EditorOpType.PROJECT_TO_LEVEL:
            project_to_level(self, op.args['level'])

        elif op.op_type == EditorOpType.RESET_TO_SAFE:
            reset_to_safe(self)

        if edited_level is not None:
            self.body.mark_explicit(edited_level)
            self.header.tower_level = self.body.highest_explicit_level()
        # Recompute derived family/graph defaults after structural edits. This
        # preserves authored layers and never promotes a default to authored.
        present = [level for level in TowerLevel if self.body.has_layer(level)]
        self.complete(max(present, default=self.header.tower_level))
        self.edit_history.append(deepcopy(op))

    def complete(self, target_level: Optional[TowerLevel] = None) -> None:
        """Complete missing layers up to target level."""
        if target_level is None:
            target_level = self.header.tower_level
        complete_tower(self, target_level)

    def project(self, target_level: TowerLevel) -> None:
        """Project down to target level."""
        project_to_level(self, target_level)

    def is_valid(self) -> bool:
        """Check if preset is valid (playable)."""
        # Must have at least L0
        if not self.body.L0_patch:
            return False

        # All present layers must be valid
        for level in TowerLevel:
            layer = self.body.get_layer(level)
            if layer and hasattr(layer, 'is_valid'):
                if not layer.is_valid():
                    return False

        return True


# =============================================================================
# PRESET LIFECYCLE (Section 7)
# =============================================================================

def compute_crc32(data: bytes) -> int:
    """Compute CRC32 checksum."""
    return zlib.crc32(data) & 0xFFFFFFFF


def load_preset(data: bytes) -> Preset:
    """
    Load preset from bytes per Section 7.1.

    1. Verify CRC
    2. Parse header
    3. Apply explicit layers
    4. Complete missing layers
    5. Project if needed
    6. Activate
    """
    if len(data) < PresetHeader.HEADER_SIZE:
        raise ValueError("Data too short for preset")

    # Verify CRC (over body only)
    body_data = data[PresetHeader.HEADER_SIZE:]
    expected_crc = compute_crc32(body_data)
    stored_crc = struct.unpack('>I', data[12:16])[0]
    if stored_crc != expected_crc:
        raise ValueError(f"CRC mismatch: expected {expected_crc}, got {stored_crc}")

    header = PresetHeader.from_bytes(data[:PresetHeader.HEADER_SIZE])

    # Create preset with header
    preset = Preset(header=header)

    # Old files contained only the header. They remain readable when their
    # empty-body CRC is valid, but every new save uses the versioned codec.
    if body_data:
        from .tower_codec import decode_body
        preset.body = decode_body(body_data, header.tower_level)
    preset.complete()

    return preset


def save_preset(preset: Preset) -> bytes:
    """
    Save preset to bytes per Section 7.2.

    - Serialize only explicit layers
    - Record tower_level accordingly
    - Never serialize completed defaults unless explicitly edited
    """
    from .tower_codec import encode_body

    body_data, highest = encode_body(preset.body)
    header = replace(preset.header, tower_level=highest, crc32=compute_crc32(body_data))
    header_data = header.to_bytes()
    preset.header.tower_level = header.tower_level
    preset.header.crc32 = header.crc32

    return header_data + body_data


def create_preset(
    preset_id: int,
    tree_id: int = 0,
    tower_level: TowerLevel = TowerLevel.L2_INSTRUMENT,
) -> Preset:
    """Create a new preset with default completion."""
    header = PresetHeader(
        preset_id=preset_id,
        tree_id=tree_id,
        tower_level=tower_level,
    )
    preset = Preset(header=header)
    preset.complete()
    return preset


# =============================================================================
# DETERMINISM VALIDATION (Section 6)
# =============================================================================

def validate_determinism(
    preset1: Preset,
    preset2: Preset,
    ops1: List[EditorOp],
    ops2: List[EditorOp],
) -> bool:
    """
    Validate determinism guarantee per Section 6.

    Same preset + same ops => same result.
    """
    # Validation must not edit its inputs. Compare complete authored content,
    # not just tree ID/tower level (which missed all parameter differences).
    preset1, preset2 = deepcopy(preset1), deepcopy(preset2)
    for op in ops1:
        preset1.apply_op(op)
    for op in ops2:
        preset2.apply_op(op)

    return save_preset(preset1) == save_preset(preset2) and preset1.body == preset2.body


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'PRESET_MAGIC',
    'SAFE_OPERATOR_SET',
    'DEFAULT_REGIME',
    # Tower levels
    'TowerLevel',
    'TOWER_LEVEL_INFO',
    'get_tower_level_info',
    # Header
    'PresetHeader',
    # Layers
    'L0PatchLayer',
    'L1FamilyLayer',
    'L2InstrumentLayer',
    'L3CockpitLayer',
    'L4ForgeLayer',
    'L5LabLayer',
    # Body
    'PresetBody',
    # Editor ops
    'EditorOpType',
    'EditorOp',
    'set_tree_id',
    'add_regime',
    'remove_regime',
    'add_transition',
    'remove_transition',
    'set_param',
    'delta_param',
    'set_bound',
    'set_operator_allowed',
    'set_operator_weight',
    'set_budget',
    'set_tower_level',
    'project_to_level_op',
    'reset_to_safe_op',
    'project_to_level',
    'reset_to_safe',
    # Completion
    'complete_tower',
    'complete_l1_family',
    'complete_l2_instrument',
    'complete_l3_cockpit',
    'complete_l4_forge',
    'complete_l5_lab',
    # Projection
    'project_to_level',
    'reset_to_safe',
    # Preset
    'Preset',
    # Lifecycle
    'load_preset',
    'save_preset',
    'create_preset',
    'compute_crc32',
    # Determinism
    'validate_determinism',
]
