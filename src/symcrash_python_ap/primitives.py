"""
Named Primitives Reference (WSE-STD-PRIMITIVES v0)
===================================================

Consolidated definitions of all named constants, enums, and semantic primitives
used throughout the symcrash synthesis engine.

This module serves as the canonical source of truth for:
- Gesture basis (8 raw controller inputs)
- Intent lanes (8 semantic control axes)
- System macros (11 CMMM control axes)
- Operator families (8 operator categories)
- Operators (24 wavelet coefficient operators)
- Regimes (7 voice states)
- Event lanes (8 event channels)
- Tail modes (4 safety operator sets)
- FM archetypes (8 FM6 topologies)
- Tree archetypes (5 wavelet tree structures)

Usage:
    from symcrash_python_ap.primitives import (
        Gesture, Intent, SystemMacro, Family,
        Operator, Regime, EventLane, TailMode,
        FMArchetype, TreeArchetype
    )
"""

from enum import IntEnum, Enum
from typing import Dict, List, NamedTuple


# =============================================================================
# FIXED-POINT CONSTANTS (Q-format arithmetic)
# =============================================================================

# Q15 format: 1 sign bit + 15 fractional bits (range: -1.0 to ~0.99997)
Q15_ONE = 32767           # Maximum positive Q15 value (~1.0)
Q15_HALF = 16384          # 0.5 in Q15
Q15_QUARTER = 8192        # 0.25 in Q15
Q15_SCALE = 32768         # 2^15 - scaling divisor for Q15 multiplication
Q15_NEG_ONE = -32768      # Minimum Q15 value (-1.0)

# Q16 format: 16 fractional bits (unsigned, range: 0.0 to ~0.99998)
Q16_ONE = 65536           # 2^16 - phase accumulator full cycle
Q16_HALF = 32768          # 0.5 in Q16

# Q8 format: 8 fractional bits
Q8_ONE = 256              # 2^8

# Q32 format: for extended precision accumulators
Q32_SCALE = 0x80000000    # 2^31


# =============================================================================
# BLOCK AND TILE CONSTANTS
# =============================================================================

BLOCK_SIZE = 64           # Samples per audio block
BLOCK_SAMPLES = 64        # Alias for BLOCK_SIZE
TILE_SIZE = 512           # Samples per processing tile (8 blocks)
BLOCKS_PER_TILE = 8       # Blocks in a tile

# Coefficient vector dimensions
COEFF_LEN = 512           # 64 leaves × 8 k-slots (coefficient vector length)
NUM_LEAVES = 64           # Wavelet packet tree leaves
VM_K = 8                  # K time slots per block

# Sample rates
SAMPLE_RATE_HZ = 48000    # Default sample rate
BLOCK_RATE_HZ = SAMPLE_RATE_HZ // BLOCK_SIZE  # ~750 Hz block rate


# =============================================================================
# GESTURE BASIS (8 raw controller inputs)
# =============================================================================

class Gesture(IntEnum):
    """
    Raw gesture inputs from controllers (WSE-STD-GESTURE §1).

    These are the 8 raw inputs that feed into the intent mapping.
    All values are normalized to Q15 fixed-point at runtime.
    """
    GATE = 0          # bool: Note on/off
    PRESSURE = 1      # Q15U: Aftertouch or mod wheel
    POSITION = 2      # Q15S: CC74 or MPE slide
    SPEED = 3         # Q15U: derivative(Position)
    TILT = 4          # Q15S: MPE Y or CC1
    CHAOS = 5         # Q15U: Glitch macro
    PITCH = 6         # Q15S: Pitch bend
    PAN = 7           # Q15S: CC10 or MPE X


class GestureInfo(NamedTuple):
    """Gesture metadata."""
    id: int
    name: str
    type: str          # "bool", "Q15U", "Q15S"
    default_source: str


GESTURE_TABLE: Dict[int, GestureInfo] = {
    0: GestureInfo(0, "Gate", "bool", "note_on_off"),
    1: GestureInfo(1, "Pressure", "Q15U", "aftertouch_or_modwheel"),
    2: GestureInfo(2, "Position", "Q15S", "cc74_or_mpe_slide"),
    3: GestureInfo(3, "Speed", "Q15U", "derivative(Position)"),
    4: GestureInfo(4, "Tilt", "Q15S", "mpe_y_or_cc1"),
    5: GestureInfo(5, "Chaos", "Q15U", "macro(M_GLITCH)"),
    6: GestureInfo(6, "Pitch", "Q15S", "pitchbend"),
    7: GestureInfo(7, "Pan", "Q15S", "cc10_or_mpe_x"),
}


# =============================================================================
# INTENT LANES (8 semantic control axes)
# =============================================================================

class Intent(IntEnum):
    """
    Semantic intent lanes (WSE-STD-GESTURE §2).

    These are computed from raw gestures and drive family weights.
    The intent vector represents "what the voice wants to do".
    """
    I_TONE = 0        # Pitched tone strength
    I_BOW = 1         # Sustained pressure / bowing
    I_AIR = 2         # Breathiness / noise desire
    I_GLITCH = 3      # Artifact / glitch desire
    I_WARP = 4        # Time/pitch warping desire
    I_SNAP = 5        # Transient / attack desire
    I_GLUE = 6        # Coherence / binding desire
    I_CALM = 7        # Recovery / simplification desire


class IntentInfo(NamedTuple):
    """Intent lane metadata."""
    id: int
    name: str
    description: str


INTENT_TABLE: Dict[int, IntentInfo] = {
    0: IntentInfo(0, "Tone", "Pitched tone strength"),
    1: IntentInfo(1, "Bow", "Sustained pressure / bowing"),
    2: IntentInfo(2, "Air", "Breathiness / noise desire"),
    3: IntentInfo(3, "Glitch", "Artifact / glitch desire"),
    4: IntentInfo(4, "Warp", "Time/pitch warping desire"),
    5: IntentInfo(5, "Snap", "Transient / attack desire"),
    6: IntentInfo(6, "Glue", "Coherence / binding desire"),
    7: IntentInfo(7, "Calm", "Recovery / simplification desire"),
}


# =============================================================================
# SYSTEM MACROS (11 CMMM control axes)
# =============================================================================

class SystemMacro(IntEnum):
    """
    System macro IDs (WSE-STD-MACRO v0 §2).

    These are the internal state variables managed by the CMMM.
    IDs 0-10 are system macros, 11-12 are performance macros.
    """
    # System macros (0-10)
    ENERGY_STATE = 0      # Envelope-owned state (read-only to controller)
    ENERGY_INTENT = 1     # Controller → CMMM: playing intensity
    MOTION = 2            # Movement / modulation amount
    FOCUS = 3             # Attention / emphasis (spectral concentration)
    BRIGHTNESS = 4        # Timbral brightness (harmonic content)
    DENSITY = 5           # Body / thickness (low-mid presence)
    RISK = 6              # Chaos / instability (stochastic behavior)
    AGE = 7               # CMMM-owned integrator (time in regime)
    SPACE = 8             # Spatial / reverb character
    MATERIAL = 9          # Surface texture (wood=low, metal=mid, glass=high)
    UNITY = 10            # Coherence / blend

    # Performance macros (11-12)
    VIBRATO = 11          # Vibrato / tremolo depth
    LEGATO = 12           # Legato / portamento mode


class MacroInfo(NamedTuple):
    """System macro metadata."""
    id: int
    name: str
    owner: str           # "Envelope", "Controller", "CMMM"
    default_slew: str    # "NONE", "ONE_POLE", "NBLOCK"


MACRO_TABLE: Dict[int, MacroInfo] = {
    0: MacroInfo(0, "ENERGY_STATE", "Envelope", "NONE"),
    1: MacroInfo(1, "ENERGY_INTENT", "Controller", "ONE_POLE(2048)"),
    2: MacroInfo(2, "MOTION", "Controller", "ONE_POLE(2048)"),
    3: MacroInfo(3, "FOCUS", "Controller", "ONE_POLE(2048)"),
    4: MacroInfo(4, "BRIGHTNESS", "Controller", "ONE_POLE(2048)"),
    5: MacroInfo(5, "DENSITY", "Controller", "ONE_POLE(2048)"),
    6: MacroInfo(6, "RISK", "Controller", "NBLOCK(16)"),
    7: MacroInfo(7, "AGE", "CMMM", "NONE"),
    8: MacroInfo(8, "SPACE", "Controller", "ONE_POLE(2048)"),
    9: MacroInfo(9, "MATERIAL", "Controller", "NBLOCK(32)"),
    10: MacroInfo(10, "UNITY", "Controller", "NBLOCK(32)"),
    11: MacroInfo(11, "VIBRATO", "Controller", "ONE_POLE(4096)"),
    12: MacroInfo(12, "LEGATO", "Controller", "NONE"),
}


# =============================================================================
# OPERATOR FAMILIES (8 categories)
# =============================================================================

class Family(IntEnum):
    """
    Operator family IDs (WSE-STD-GESTURE §8).

    Families group operators by semantic purpose.
    Family weights are computed from intent lanes.
    """
    F_RESERVED = 0        # Reserved (not used)
    F_COLOR = 1           # Timbral coloring (Tilt, etc.)
    F_TRANSPORT = 2       # Energy transport (DiffuseUp/Down)
    F_LOCAL = 3           # Local structure (IntraMix)
    F_TOPOLOGY = 4        # Global topology (SwapBands, RotateNodes)
    F_GLUE = 5            # Coherence / binding (GluePair, SheafProject)
    F_EVENTS = 6          # Event injection (Impulse, NoiseBand)
    F_CATASTROPHE = 7     # Catastrophe operators (Fold, Cusp)


class FamilyInfo(NamedTuple):
    """Family metadata."""
    id: int
    name: str
    description: str
    operators: List[int]  # Operator IDs in this family


FAMILY_TABLE: Dict[int, FamilyInfo] = {
    0: FamilyInfo(0, "RESERVED", "Reserved", []),
    1: FamilyInfo(1, "COLOR", "Timbral coloring", [1]),  # Tilt
    2: FamilyInfo(2, "TRANSPORT", "Energy transport", [4, 5]),  # DiffuseUp/Down
    3: FamilyInfo(3, "LOCAL", "Local structure", [3]),  # IntraMix
    4: FamilyInfo(4, "TOPOLOGY", "Global topology", [6, 9, 10, 17]),  # BandShear, SwapBands, RotateNodes, QuasiRotate
    5: FamilyInfo(5, "GLUE", "Coherence / binding", [12, 23]),  # GluePair, SheafProject
    6: FamilyInfo(6, "EVENTS", "Event injection", [8, 13, 14]),  # GlobalPulse, Impulse, NoiseBand
    7: FamilyInfo(7, "CATASTROPHE", "Catastrophe operators", [20, 21, 22]),  # FoldPressure, CuspInjection, SwallowtailMask
}


# =============================================================================
# OPERATORS (24 wavelet coefficient operators)
# =============================================================================

class Operator(IntEnum):
    """
    Operator IDs (WSE-STD-OPERATORS v0).

    24 operators that transform the 256-coefficient wavelet vector.
    Organized by emission group (TAIL, BASE, MID, TOP).
    """
    # BASE group (always allowed)
    IDENTITY = 0
    TILT = 1
    INTRA_MIX = 3

    # TAIL group (always run for safety)
    FINE_CLAMP = 2
    NORMALIZE_SCALE = 11
    GLUE_PAIR = 12
    EMERGENCY_SIMPLIFY = 19
    SHEAF_PROJECT = 23

    # MID group (gated by mid_due)
    DIFFUSE_UP = 4
    DIFFUSE_DOWN = 5
    LOWRANK_MORPH = 7
    GLOBAL_PULSE = 8
    IMPULSE = 13
    NOISE_BAND = 14

    # TOP group (gated by top_due)
    BAND_SHEAR = 6
    SWAP_BANDS = 9
    ROTATE_NODES = 10
    PRESSURE_RAMP = 15
    GATE_MASK = 16
    QUASI_ROTATE = 17
    MORPH_MIX = 18
    FOLD_PRESSURE = 20
    CUSP_INJECTION = 21
    SWALLOWTAIL_MASK = 22


class OperatorType(Enum):
    """Operator type classification."""
    DIAG = "DIAG"        # Element-wise scaling (K-scaling)
    BANDED = "BANDED"    # Within/across band coupling (K-scaling)
    LOWRANK = "LOWRANK"  # Global UV^T patterns (L-scaling)
    PERMUTE = "PERMUTE"  # Coefficient reordering (L-scaling)
    PROJ = "PROJ"        # Energy projection/constraints (L-scaling)
    INJECT = "INJECT"    # Signal injection (K-scaling)
    MIX = "MIX"          # Operator blending


class OperatorGroup(Enum):
    """Emission group classification."""
    TAIL = "TAIL"        # Always run (safety)
    BASE = "BASE"        # Always allowed (minimal)
    MID = "MID"          # When mid_due=True
    TOP = "TOP"          # When top_due=True


class OperatorInfo(NamedTuple):
    """Operator metadata."""
    id: int
    name: str
    op_type: str         # OperatorType value
    group: str           # OperatorGroup value
    description: str


OPERATOR_TABLE: Dict[int, OperatorInfo] = {
    0: OperatorInfo(0, "Identity", "DIAG", "BASE", "Pass-through, no change"),
    1: OperatorInfo(1, "Tilt", "DIAG", "BASE", "Frequency-dependent gain (dB/octave)"),
    2: OperatorInfo(2, "FineClamp", "PROJ", "TAIL", "Caps fine-band energy relative to coarse"),
    3: OperatorInfo(3, "IntraMix", "BANDED", "BASE", "Tridiagonal mixing within each node"),
    4: OperatorInfo(4, "DiffuseUp", "BANDED", "MID", "Fine → coarse energy transport"),
    5: OperatorInfo(5, "DiffuseDown", "BANDED", "MID", "Coarse → fine structure transport"),
    6: OperatorInfo(6, "BandShear", "PERMUTE", "TOP", "Position-dependent rotation within bands"),
    7: OperatorInfo(7, "LowRankMorph", "LOWRANK", "MID", "Global coherence via UV^T"),
    8: OperatorInfo(8, "GlobalPulse", "INJECT", "MID", "Energy injection (coarse_heavy/mid_peak/bright)"),
    9: OperatorInfo(9, "SwapBands", "PERMUTE", "TOP", "Swap two entire bands"),
    10: OperatorInfo(10, "RotateNodes", "PERMUTE", "TOP", "Rotate node ordering within each band"),
    11: OperatorInfo(11, "NormalizeScale", "PROJ", "TAIL", "Per-band normalization (flat/pink/bright)"),
    12: OperatorInfo(12, "GluePair", "PROJ", "TAIL", "Sheaf compatibility between two bands"),
    13: OperatorInfo(13, "Impulse", "INJECT", "MID", "Sparse attack injection at band starts"),
    14: OperatorInfo(14, "NoiseBand", "INJECT", "MID", "Band-limited Gaussian noise injection"),
    15: OperatorInfo(15, "PressureRamp", "DIAG", "TOP", "Pressure modulation toward transition"),
    16: OperatorInfo(16, "GateMask", "DIAG", "TOP", "Binary mask (sturmian/thue_morse/random)"),
    17: OperatorInfo(17, "QuasiRotate", "PERMUTE", "TOP", "Quasiperiodic rotation (golden ratio)"),
    18: OperatorInfo(18, "MorphMix", "MIX", "TOP", "Convex combination of two operators"),
    19: OperatorInfo(19, "EmergencySimplify", "PROJ", "TAIL", "Hard projection to safe envelope"),
    20: OperatorInfo(20, "FoldPressure", "DIAG", "TOP", "Catastrophe fold setup"),
    21: OperatorInfo(21, "CuspInjection", "INJECT", "TOP", "Asymmetric perturbation for cusp"),
    22: OperatorInfo(22, "SwallowtailMask", "DIAG", "TOP", "3-parameter catastrophe gate"),
    23: OperatorInfo(23, "SheafProject", "PROJ", "TAIL", "Local-to-global consistency projection"),
}

# String name to ID mapping
OPERATOR_NAME_TO_ID: Dict[str, int] = {
    info.name: info.id for info in OPERATOR_TABLE.values()
}

# Also support uppercase/underscored names
OPERATOR_NAME_TO_ID.update({
    "IDENTITY": 0, "TILT": 1, "FINE_CLAMP": 2, "INTRA_MIX": 3,
    "DIFFUSE_UP": 4, "DIFFUSE_DOWN": 5, "BAND_SHEAR": 6, "LOWRANK_MORPH": 7,
    "GLOBAL_PULSE": 8, "SWAP_BANDS": 9, "ROTATE_NODES": 10, "NORMALIZE_SCALE": 11,
    "GLUE_PAIR": 12, "IMPULSE": 13, "NOISE_BAND": 14, "PRESSURE_RAMP": 15,
    "GATE_MASK": 16, "QUASI_ROTATE": 17, "MORPH_MIX": 18, "EMERGENCY_SIMPLIFY": 19,
    "FOLD_PRESSURE": 20, "CUSP_INJECTION": 21, "SWALLOWTAIL_MASK": 22, "SHEAF_PROJECT": 23,
})


# =============================================================================
# REGIMES (7 voice states)
# =============================================================================

class Regime(IntEnum):
    """
    Voice regime states (WSE-STD-GESTURE §4-6).

    The regime graph defines state machine transitions driven by intent.
    """
    R0_IDLE = 0           # Silent / waiting for note
    R1_ATTACK = 1         # Transient / attack ignition
    R2_STABLE = 2         # Stable tone / sustain
    R3_AIR = 3            # Noise texture / breath
    R4_GLITCH = 4         # Glitch chatter / artifacts
    R5_CATASTROPHE = 5    # Catastrophe / extreme chaos
    R6_RECOVERY = 6       # Recovery / healing


class RegimeKind(Enum):
    """Regime type classification."""
    IDLE = "Idle"
    TRANSIENT = "Transient"
    STABLE_TONE = "StableTone"
    NOISE_TEXTURE = "NoiseTexture"
    GLITCH_CHATTER = "GlitchChatter"
    CATASTROPHE = "Catastrophe"
    RECOVERY = "Recovery"


class RegimeInfo(NamedTuple):
    """Regime metadata."""
    id: int
    name: str
    kind: str
    description: str


REGIME_TABLE: Dict[int, RegimeInfo] = {
    0: RegimeInfo(0, "R0_IDLE", "Idle", "Silent / waiting for note"),
    1: RegimeInfo(1, "R1_ATTACK", "Transient", "Transient / attack ignition"),
    2: RegimeInfo(2, "R2_STABLE", "StableTone", "Stable tone / sustain"),
    3: RegimeInfo(3, "R3_AIR", "NoiseTexture", "Noise texture / breath"),
    4: RegimeInfo(4, "R4_GLITCH", "GlitchChatter", "Glitch chatter / artifacts"),
    5: RegimeInfo(5, "R5_CATASTROPHE", "Catastrophe", "Catastrophe / extreme chaos"),
    6: RegimeInfo(6, "R6_RECOVERY", "Recovery", "Recovery / healing"),
}


# =============================================================================
# EVENT LANES (8 channels)
# =============================================================================

class EventLane(IntEnum):
    """
    Event lane IDs (WSE-STD-VOICEFRAME v0).

    8 lanes for discrete events that trigger at block boundaries.
    """
    EVT_OP_PULSE = 0      # Operator pulse injection
    EVT_OP_TOGGLE = 1     # Operator toggle
    EVT_TREE_MORPH = 2    # Tree topology morph
    EVT_K_MODE = 3        # K degradation mode change
    EVT_CARRIER_CFG = 4   # Carrier configuration (was EVT_INJECT_NOTE)
    EVT_MATERIAL_HIT = 5  # Material excitation
    EVT_CLOUD_PUSH = 6    # Cloud coefficient push
    EVT_RESERVED = 7      # Reserved


class EventLaneInfo(NamedTuple):
    """Event lane metadata."""
    id: int
    name: str
    description: str


EVENT_LANE_TABLE: Dict[int, EventLaneInfo] = {
    0: EventLaneInfo(0, "EVT_OP_PULSE", "Operator pulse injection"),
    1: EventLaneInfo(1, "EVT_OP_TOGGLE", "Operator toggle"),
    2: EventLaneInfo(2, "EVT_TREE_MORPH", "Tree topology morph"),
    3: EventLaneInfo(3, "EVT_K_MODE", "K degradation mode change"),
    4: EventLaneInfo(4, "EVT_CARRIER_CFG", "Carrier configuration"),
    5: EventLaneInfo(5, "EVT_MATERIAL_HIT", "Material excitation"),
    6: EventLaneInfo(6, "EVT_CLOUD_PUSH", "Cloud coefficient push"),
    7: EventLaneInfo(7, "EVT_RESERVED", "Reserved"),
}


# =============================================================================
# TAIL MODES (4 safety operator sets)
# =============================================================================

class TailMode(IntEnum):
    """
    Tail mode IDs (WSE-STD-070 §10).

    Controls which safety operators run at the end of the operator chain.
    """
    FULL = 0              # FineClamp, NormalizeScale, SheafProject
    MID = 1               # FineClamp, NormalizeScale
    MIN = 2               # FineClamp only
    EMERGENCY = 3         # FineClamp, EmergencySimplify


class TailModeInfo(NamedTuple):
    """Tail mode metadata."""
    id: int
    name: str
    operators: List[int]  # Operator IDs


TAIL_MODE_TABLE: Dict[int, TailModeInfo] = {
    0: TailModeInfo(0, "FULL", [2, 11, 23]),       # FineClamp, NormalizeScale, SheafProject
    1: TailModeInfo(1, "MID", [2, 11]),            # FineClamp, NormalizeScale
    2: TailModeInfo(2, "MIN", [2]),                # FineClamp
    3: TailModeInfo(3, "EMERGENCY", [2, 19]),      # FineClamp, EmergencySimplify
}


def get_tail_operators(tail_mode: int) -> List[int]:
    """Get the operator IDs for a tail mode."""
    return TAIL_MODE_TABLE.get(tail_mode, TAIL_MODE_TABLE[0]).operators


# =============================================================================
# FM ARCHETYPES (8 FM6 topologies)
# =============================================================================

class FMArchetype(IntEnum):
    """
    FM6 archetype IDs (WSE-STD-FM6 v0).

    8 canonical FM operator topologies for 6-operator synthesis.
    """
    CHAIN = 0             # Serial 6-op chain (1→2→3→4→5→6)
    STACK = 1             # 5 modulators → 1 carrier
    DUAL = 2              # Two parallel 3-op chains
    SPLIT = 3             # Modulator tree → multiple carriers
    YPAIR = 4             # Y-branch structure
    CLOUD = 5             # Parallel modulators (thick)
    ADDITIVE = 6          # All carriers, no modulation
    FEEDBACK = 7          # Chain with enhanced feedback


class FMArchetypeInfo(NamedTuple):
    """FM archetype metadata."""
    id: int
    name: str
    description: str
    carriers: List[int]   # Which ops are carriers (1-indexed DX7 style)


FM_ARCHETYPE_TABLE: Dict[int, FMArchetypeInfo] = {
    0: FMArchetypeInfo(0, "CHAIN", "Serial 6-op chain", [6]),
    1: FMArchetypeInfo(1, "STACK", "5 modulators → 1 carrier", [6]),
    2: FMArchetypeInfo(2, "DUAL", "Two parallel 3-op chains", [3, 6]),
    3: FMArchetypeInfo(3, "SPLIT", "Modulator tree → multiple carriers", [4, 5, 6]),
    4: FMArchetypeInfo(4, "YPAIR", "Y-branch structure", [3, 6]),
    5: FMArchetypeInfo(5, "CLOUD", "Parallel modulators (thick)", [6]),
    6: FMArchetypeInfo(6, "ADDITIVE", "All carriers, no modulation", [1, 2, 3, 4, 5, 6]),
    7: FMArchetypeInfo(7, "FEEDBACK", "Chain with enhanced feedback", [6]),
}


# =============================================================================
# TREE ARCHETYPES (5 wavelet tree structures)
# =============================================================================

class TreeArchetype(IntEnum):
    """
    Wavelet tree archetype IDs.

    5 canonical tree structures for scaffold generation.
    """
    HARMONIC = 0          # Harmonic series emphasis
    BRIGHT = 1            # High-frequency emphasis
    HOLLOW = 2            # Suppressed mid-harmonics
    NOISY = 3             # Noise-like texture
    TRANSIENT = 4         # Attack/impact emphasis


class TreeArchetypeInfo(NamedTuple):
    """Tree archetype metadata."""
    id: int
    name: str
    description: str


TREE_ARCHETYPE_TABLE: Dict[int, TreeArchetypeInfo] = {
    0: TreeArchetypeInfo(0, "HARMONIC", "Harmonic series emphasis"),
    1: TreeArchetypeInfo(1, "BRIGHT", "High-frequency emphasis"),
    2: TreeArchetypeInfo(2, "HOLLOW", "Suppressed mid-harmonics"),
    3: TreeArchetypeInfo(3, "NOISY", "Noise-like texture"),
    4: TreeArchetypeInfo(4, "TRANSIENT", "Attack/impact emphasis"),
}


# =============================================================================
# K LADDER (degradation masks)
# =============================================================================

class KLadder(IntEnum):
    """
    K ladder pattern IDs (WSE-STD-070 §10).

    Canonical node masks for graceful degradation.
    """
    K32 = 0               # 0xFFFFFFFF - Full resolution
    K24 = 1               # 0x00FFFFFF - 24 nodes
    K20 = 2               # 0x000FFFFF - 20 nodes
    K16 = 3               # 0x0000FFFF - 16 nodes
    K12 = 4               # 0x00000FFF - 12 nodes
    K8 = 5                # 0x000000FF - 8 nodes
    K4 = 6                # 0x0000000F - 4 nodes
    K1 = 7                # 0x00000001 - 1 node


# Pattern ID → mask mapping
K_LADDER_MASKS: Dict[int, int] = {
    0: 0xFFFFFFFF,  # K=32
    1: 0x00FFFFFF,  # K=24
    2: 0x000FFFFF,  # K=20
    3: 0x0000FFFF,  # K=16
    4: 0x00000FFF,  # K=12
    5: 0x000000FF,  # K=8
    6: 0x0000000F,  # K=4
    7: 0x00000001,  # K=1
}


def get_k_mask(pattern_id: int) -> int:
    """Get the node mask for a K ladder pattern."""
    return K_LADDER_MASKS.get(pattern_id, 0xFFFFFFFF)


def get_k_from_mask(mask: int) -> int:
    """Get the K value (node count) from a mask."""
    return bin(mask).count('1')


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Fixed-point constants
    'Q15_ONE', 'Q15_HALF', 'Q15_QUARTER', 'Q15_SCALE', 'Q15_NEG_ONE',
    'Q16_ONE', 'Q16_HALF', 'Q8_ONE', 'Q32_SCALE',
    # Block/tile constants
    'BLOCK_SIZE', 'BLOCK_SAMPLES', 'TILE_SIZE', 'BLOCKS_PER_TILE',
    'COEFF_LEN', 'NUM_LEAVES', 'VM_K',
    'SAMPLE_RATE_HZ', 'BLOCK_RATE_HZ',
    # Gesture
    'Gesture', 'GestureInfo', 'GESTURE_TABLE',
    # Intent
    'Intent', 'IntentInfo', 'INTENT_TABLE',
    # System Macro
    'SystemMacro', 'MacroInfo', 'MACRO_TABLE',
    # Family
    'Family', 'FamilyInfo', 'FAMILY_TABLE',
    # Operator
    'Operator', 'OperatorType', 'OperatorGroup', 'OperatorInfo',
    'OPERATOR_TABLE', 'OPERATOR_NAME_TO_ID',
    # Regime
    'Regime', 'RegimeKind', 'RegimeInfo', 'REGIME_TABLE',
    # Event Lane
    'EventLane', 'EventLaneInfo', 'EVENT_LANE_TABLE',
    # Tail Mode
    'TailMode', 'TailModeInfo', 'TAIL_MODE_TABLE', 'get_tail_operators',
    # FM Archetype
    'FMArchetype', 'FMArchetypeInfo', 'FM_ARCHETYPE_TABLE',
    # Tree Archetype
    'TreeArchetype', 'TreeArchetypeInfo', 'TREE_ARCHETYPE_TABLE',
    # K Ladder
    'KLadder', 'K_LADDER_MASKS', 'get_k_mask', 'get_k_from_mask',
]
