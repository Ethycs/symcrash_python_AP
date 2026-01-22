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

# Additional Q16 constants
Q16_SCALE = 65536         # 2^16 - Q16 scaling factor (alias for Q16_ONE)

# Drone mode constants (vm/executor.py)
DRONE_MODE_CARRIER_MIX = 1638  # ~5% carrier blend in drone mode (Q15_ONE * 0.05)


# =============================================================================
# FIXED-POINT UTILITY FUNCTIONS
# =============================================================================

def clamp_q15(value: int) -> int:
    """
    Clamp value to Q15 signed range [-32768, 32767].

    This is the canonical clamping function for coefficient-domain operations.
    Use this instead of inline min(32767, max(-32768, x)) patterns.

    Args:
        value: Integer value to clamp

    Returns:
        Value clamped to Q15 range
    """
    return max(-32768, min(32767, value))


def clamp_q15u(value: int) -> int:
    """
    Clamp value to Q15 unsigned range [0, 32767].

    For gain, envelope, and other non-negative Q15 values.

    Args:
        value: Integer value to clamp

    Returns:
        Value clamped to [0, 32767]
    """
    return max(0, min(32767, value))


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
NODE_LEN = 8              # Coefficients per node (sub-block)
L_TOTAL_NODES = COEFF_LEN // NODE_LEN  # 64 total nodes (for mask operations)

# Sample rates
SAMPLE_RATE_HZ = 48000    # Default sample rate
BLOCK_RATE_HZ = SAMPLE_RATE_HZ // BLOCK_SIZE  # ~750 Hz block rate


# =============================================================================
# NODE ITERATION HELPERS
# =============================================================================

def iter_active_nodes(mask: int):
    """
    Iterate over active node indices based on a bitmask.

    This is the canonical iterator for operator dispatch loops that check
    node masks. Use this instead of inline for-loop + mask check patterns.

    Args:
        mask: 32-bit node mask (1 = node active)

    Yields:
        int: Index of each active node (0 to L_TOTAL_NODES-1)

    Example:
        for node in iter_active_nodes(mask):
            coeff_idx = node * NODE_LEN
            # process node...
    """
    for node in range(L_TOTAL_NODES):
        if mask & (1 << node):
            yield node


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


class EdgeTransitionType(Enum):
    """
    Edge transition types per Voice Engine Tower spec.

    Defines how the transition between regimes behaves.
    Canonical source: primitives.py (import from here)
    """
    HARD = "hard"                  # Immediate switch, no crossfade
    SOFT = "soft"                  # Smooth crossfade over duration_blocks
    CHAOTIC = "chaotic"            # Stochastic switching with jitter
    CORRIDOR_ENTRY = "corridor"    # Entry into degradation corridor
    RECOVERY = "recovery"          # Return from instability to stable


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
# OSCILLATOR MODES (6 synthesis modes)
# =============================================================================

class OscMode(IntEnum):
    """
    Oscillator mode IDs (WSE-STD-210).

    6 synthesis modes that generate coefficient vectors.
    Each mode has its own state structure and generation function.
    """
    MODE_A = 0            # LUT-based coefficient streaming (CCSState)
    MODE_B = 1            # Rotator/Harmonic Bank (ModeBState)
    MODE_SAMPLE = 2       # Sample playback via FWPT gateway (Phase 3)
    MODE_FM6 = 3          # 6-operator FM synthesis (WSE-STD-FM-TOPOLOGY-v0)
    MODE_WAVETABLE = 4    # Wavetable synthesis with 2D interpolation (Phase 4)
    MODE_GRANULAR = 5     # Granular synthesis (coefficient-domain)


class OscModeInfo(NamedTuple):
    """Oscillator mode metadata."""
    id: int
    name: str
    description: str
    state_class: str       # Name of state class


OSC_MODE_TABLE: Dict[int, OscModeInfo] = {
    0: OscModeInfo(0, "MODE_A", "LUT-based coefficient streaming", "CCSState"),
    1: OscModeInfo(1, "MODE_B", "Rotator/Harmonic Bank", "ModeBState"),
    2: OscModeInfo(2, "MODE_SAMPLE", "Sample playback via FWPT gateway", "SampleOscState"),
    3: OscModeInfo(3, "MODE_FM6", "6-operator FM synthesis", "FM6State"),
    4: OscModeInfo(4, "MODE_WAVETABLE", "Wavetable synthesis with 2D interpolation", "WavetableState"),
    5: OscModeInfo(5, "MODE_GRANULAR", "Granular synthesis (coefficient-domain)", "GranularState"),
}


# =============================================================================
# SYMBOL ENGINE MODES (WSE-STD-CMMM-ENRICHMENT Gap G1)
# =============================================================================

class SymbolMode(IntEnum):
    """
    Symbol Engine operating modes (semi-Markov).

    These modes control coefficient evolution rate. Note that IDLE, ACCENT,
    TRANSITION, BUDGET_DEGRADE, and RECOVER are already handled by the
    Regime FSM (R0-R6) + degradation ladder. These 4 modes fill the gaps.

    Reference: CMMM-ENRICHMENT-DESIGN.md §9.2
    """
    HOLD = 0              # Low-motion sustain, minimal coefficient evolution (~3%)
    EVOLVE = 1            # Normal "living" voice, full evolution (100%)
    PHRASE_QUANTIZE = 2   # Tempo-locked commits (100% but gated to beat)
    FREEZE = 3            # Stop evolution, optionally cache coefficients (0%)


class SymbolModeInfo(NamedTuple):
    """Symbol mode metadata."""
    id: int
    name: str
    evolution_rate_q15: int  # Evolution rate multiplier (Q15)
    description: str


# Evolution rate multiplier per mode (Q15)
# Q15_ONE = 32767 = 100%, 1024 = ~3%, 0 = 0%
SYMBOL_MODE_TABLE: Dict[int, SymbolModeInfo] = {
    0: SymbolModeInfo(0, "HOLD", 1024, "Low-motion sustain (~3% evolution)"),
    1: SymbolModeInfo(1, "EVOLVE", 32767, "Normal voice (100% evolution)"),
    2: SymbolModeInfo(2, "PHRASE_QUANTIZE", 32767, "Tempo-locked (100% but gated)"),
    3: SymbolModeInfo(3, "FREEZE", 0, "Frozen coefficients (0% evolution)"),
}


def get_symbol_mode_evolution_rate(mode: int) -> int:
    """Get evolution rate (Q15) for a symbol mode."""
    return SYMBOL_MODE_TABLE.get(mode, SYMBOL_MODE_TABLE[1]).evolution_rate_q15


# Regime → default symbol mode mapping
# Used when no explicit mode is requested
REGIME_TO_SYMBOL_MODE: Dict[int, int] = {
    0: SymbolMode.HOLD,      # R0_IDLE → HOLD (minimal evolution)
    1: SymbolMode.EVOLVE,    # R1_ATTACK → EVOLVE
    2: SymbolMode.EVOLVE,    # R2_STABLE → EVOLVE (default)
    3: SymbolMode.EVOLVE,    # R3_AIR → EVOLVE
    4: SymbolMode.EVOLVE,    # R4_GLITCH → EVOLVE
    5: SymbolMode.FREEZE,    # R5_CATASTROPHE → FREEZE (safety)
    6: SymbolMode.HOLD,      # R6_RECOVERY → HOLD
}


# =============================================================================
# TIMBRAL MODES (WSE-STD-CMMM-ENRICHMENT Gap G4)
# =============================================================================

class TimbralMode(IntEnum):
    """
    Timbral sub-states within STABLE_TONE regime for dynamic operator selection.

    These are hierarchical sub-states that select which operators to use
    within a given regime. Intent vector drives timbral mode selection.

    Reference: CMMM-ENRICHMENT-DESIGN.md §9.5
    """
    T0_PURE = 0       # Harmonic, clean
    T1_BRIGHT = 1     # High frequency emphasis
    T2_DARK = 2       # Low frequency emphasis
    T3_HOLLOW = 3     # Mid-frequency scoop
    T4_NASAL = 4      # Formant peak emphasis
    T5_BREATHY = 5    # Air + tone blend
    T6_GROWL = 6      # Modulation artifacts
    T7_CHAOS = 7      # Full catastrophe mode


class TimbralModeInfo(NamedTuple):
    """Timbral mode metadata."""
    id: int
    name: str
    description: str


TIMBRAL_MODE_TABLE: Dict[int, TimbralModeInfo] = {
    0: TimbralModeInfo(0, "T0_PURE", "Harmonic, clean"),
    1: TimbralModeInfo(1, "T1_BRIGHT", "High frequency emphasis"),
    2: TimbralModeInfo(2, "T2_DARK", "Low frequency emphasis"),
    3: TimbralModeInfo(3, "T3_HOLLOW", "Mid-frequency scoop"),
    4: TimbralModeInfo(4, "T4_NASAL", "Formant peak emphasis"),
    5: TimbralModeInfo(5, "T5_BREATHY", "Air + tone blend"),
    6: TimbralModeInfo(6, "T6_GROWL", "Modulation artifacts"),
    7: TimbralModeInfo(7, "T7_CHAOS", "Full catastrophe mode"),
}


# =============================================================================
# REGIME-SYMBOL OPERATOR TABLE (CMMM Enrichment Gap G4)
# =============================================================================
# Maps (regime_id, timbral_mode) to list of operator IDs.
# This enables dynamic operator selection based on CMMM state.
#
# Reference: CMMM-ENRICHMENT-DESIGN.md §3.5 (Per-Symbol Operator Selection)
#
# Operator IDs (from vm/tables.py OPERATOR_LANES):
#   0: Identity      1: Tilt         2: FineClamp     3: IntraMix
#   4: DiffuseUp     5: DiffuseDown  6: BandShear     7: LowRankMorph
#   8: GlobalPulse   9: SwapBands   10: RotateNodes  11: NormalizeScale
#  12: GluePair     13: Impulse     14: NoiseBand    15: PressureRamp
#  16: GateMask     17: QuasiRotate 18: MorphMix     19: EmergencySimplify
#  20: FoldPressure 21: CuspInjection 22: SwallowtailMask 23: SheafProject

REGIME_SYMBOL_OPS: Dict[Tuple[int, int], List[int]] = {
    # -------------------------------------------------------------------------
    # R0_IDLE: Minimal processing (voice inactive)
    # -------------------------------------------------------------------------
    (0, 0): [0],           # T0_PURE: Identity only
    (0, 1): [0, 1],        # T1_BRIGHT: Identity + Tilt (slight brightening)

    # -------------------------------------------------------------------------
    # R1_ATTACK: Transient burst (attack phase)
    # -------------------------------------------------------------------------
    (1, 0): [13, 8],       # T0_PURE: Impulse + GlobalPulse (sharp attack)
    (1, 1): [13, 4, 8],    # T1_BRIGHT: Impulse + DiffuseUp + GlobalPulse
    (1, 2): [13, 5, 8],    # T2_DARK: Impulse + DiffuseDown + GlobalPulse
    (1, 3): [13, 6, 8],    # T3_HOLLOW: Impulse + BandShear + GlobalPulse
    (1, 4): [13, 7, 8],    # T4_NASAL: Impulse + LowRankMorph + GlobalPulse
    (1, 5): [13, 14, 8],   # T5_BREATHY: Impulse + NoiseBand + GlobalPulse
    (1, 6): [13, 15, 20],  # T6_GROWL: Impulse + PressureRamp + FoldPressure
    (1, 7): [13, 21, 22],  # T7_CHAOS: Impulse + CuspInjection + SwallowtailMask

    # -------------------------------------------------------------------------
    # R2_STABLE: Sustained tone (main playback regime)
    # Most timbral variety happens here
    # -------------------------------------------------------------------------
    (2, 0): [1, 3, 12],    # T0_PURE: Tilt + IntraMix + GluePair (harmonic, clean)
    (2, 1): [1, 4, 5],     # T1_BRIGHT: Tilt + DiffuseUp + DiffuseDown (high emphasis)
    (2, 2): [1, 3, 5],     # T2_DARK: Tilt + IntraMix + DiffuseDown (low emphasis)
    (2, 3): [6, 10, 12],   # T3_HOLLOW: BandShear + RotateNodes + GluePair (mid scoop)
    (2, 4): [7, 12, 3],    # T4_NASAL: LowRankMorph + GluePair + IntraMix (formant peak)
    (2, 5): [14, 3, 12],   # T5_BREATHY: NoiseBand + IntraMix + GluePair (air + tone)
    (2, 6): [15, 20, 3],   # T6_GROWL: PressureRamp + FoldPressure + IntraMix (modulation)
    (2, 7): [21, 22, 8],   # T7_CHAOS: CuspInjection + SwallowtailMask + GlobalPulse

    # -------------------------------------------------------------------------
    # R3_AIR: Breathy sustain (noise texture regime)
    # -------------------------------------------------------------------------
    (3, 0): [14, 4],       # T0_PURE: NoiseBand + DiffuseUp
    (3, 1): [14, 4, 1],    # T1_BRIGHT: NoiseBand + DiffuseUp + Tilt
    (3, 2): [14, 5, 3],    # T2_DARK: NoiseBand + DiffuseDown + IntraMix
    (3, 3): [14, 6, 10],   # T3_HOLLOW: NoiseBand + BandShear + RotateNodes
    (3, 4): [14, 7, 12],   # T4_NASAL: NoiseBand + LowRankMorph + GluePair
    (3, 5): [14, 14, 4],   # T5_BREATHY: Double NoiseBand + DiffuseUp (more air)
    (3, 6): [14, 15, 20],  # T6_GROWL: NoiseBand + PressureRamp + FoldPressure
    (3, 7): [14, 21, 22],  # T7_CHAOS: NoiseBand + CuspInjection + SwallowtailMask

    # -------------------------------------------------------------------------
    # R4_GLITCH: Artifact regime
    # -------------------------------------------------------------------------
    (4, 0): [13, 14, 9],   # T0_PURE: Impulse + NoiseBand + SwapBands
    (4, 1): [13, 4, 9],    # T1_BRIGHT: Impulse + DiffuseUp + SwapBands
    (4, 2): [13, 5, 9],    # T2_DARK: Impulse + DiffuseDown + SwapBands
    (4, 3): [13, 6, 9],    # T3_HOLLOW: Impulse + BandShear + SwapBands
    (4, 4): [13, 7, 9],    # T4_NASAL: Impulse + LowRankMorph + SwapBands
    (4, 5): [14, 13, 9],   # T5_BREATHY: NoiseBand + Impulse + SwapBands
    (4, 6): [13, 15, 9],   # T6_GROWL: Impulse + PressureRamp + SwapBands
    (4, 7): [21, 22, 9],   # T7_CHAOS: CuspInjection + SwallowtailMask + SwapBands

    # -------------------------------------------------------------------------
    # R5_CATASTROPHE: Extreme chaos (safety regime)
    # -------------------------------------------------------------------------
    (5, 0): [21, 22, 15, 20],  # All modes use full catastrophe family
    (5, 1): [21, 22, 15, 20],
    (5, 2): [21, 22, 15, 20],
    (5, 3): [21, 22, 15, 20],
    (5, 4): [21, 22, 15, 20],
    (5, 5): [21, 22, 15, 20],
    (5, 6): [21, 22, 15, 20],
    (5, 7): [21, 22, 15, 20],

    # -------------------------------------------------------------------------
    # R6_RECOVERY: Return to calm (healing regime)
    # -------------------------------------------------------------------------
    (6, 0): [12, 23],      # T0_PURE: GluePair + SheafProject (stabilize)
    (6, 1): [12, 23, 1],   # T1_BRIGHT: + Tilt
    (6, 2): [12, 23, 5],   # T2_DARK: + DiffuseDown
    (6, 3): [12, 23, 6],   # T3_HOLLOW: + BandShear
    (6, 4): [12, 23, 7],   # T4_NASAL: + LowRankMorph
    (6, 5): [12, 23, 14],  # T5_BREATHY: + NoiseBand
    (6, 6): [12, 23, 3],   # T6_GROWL: + IntraMix
    (6, 7): [12, 23, 8],   # T7_CHAOS: + GlobalPulse
}


def get_operators_for_state(regime: int, timbral_mode: int) -> List[int]:
    """
    Get operator list for current (regime, timbral_mode) state.

    Args:
        regime: Regime enum value (0-6)
        timbral_mode: TimbralMode enum value (0-7)

    Returns:
        List of operator IDs for this state. Returns default [0] if not found.
    """
    key = (regime, timbral_mode)
    if key in REGIME_SYMBOL_OPS:
        return REGIME_SYMBOL_OPS[key]
    # Fallback: try regime-level default (timbral_mode=0)
    fallback_key = (regime, 0)
    if fallback_key in REGIME_SYMBOL_OPS:
        return REGIME_SYMBOL_OPS[fallback_key]
    # Ultimate fallback: Identity only
    return [0]


# =============================================================================
# INTENT → TIMBRAL MODE MAPPING (CMMM Enrichment Gap G4)
# =============================================================================
# Maps intent vector to timbral mode selection.
# Higher intent lanes push toward specific timbral modes.
#
# Reference: CMMM-ENRICHMENT-DESIGN.md §4.5 Phase P3

def compute_timbral_mode_from_intent(intent_lanes: List[int]) -> int:
    """
    Compute timbral mode from intent vector.

    Uses intent lane values to select appropriate timbral mode:
    - High I_TONE (0) + low others → T0_PURE
    - High I_AIR (2) → T5_BREATHY
    - High I_GLITCH (3) → T6_GROWL or T7_CHAOS
    - etc.

    Args:
        intent_lanes: List of 8 intent lane values (Q15)

    Returns:
        TimbralMode enum value (0-7)
    """
    if len(intent_lanes) < 4:
        return TimbralMode.T0_PURE

    i_tone = intent_lanes[0]     # Intent.I_TONE
    i_bow = intent_lanes[1]      # Intent.I_BOW
    i_air = intent_lanes[2]      # Intent.I_AIR
    i_glitch = intent_lanes[3]   # Intent.I_GLITCH

    # Thresholds in Q15
    HIGH_THRESH = 24576    # ~75% of Q15_ONE (32767)
    MED_THRESH = 16384     # ~50% of Q15_ONE
    LOW_THRESH = 8192      # ~25% of Q15_ONE

    # Priority-based selection (higher priority wins):
    # 1. Extreme chaos: I_GLITCH very high
    if i_glitch > HIGH_THRESH:
        return TimbralMode.T7_CHAOS

    # 2. Growl: I_GLITCH medium-high
    if i_glitch > MED_THRESH:
        return TimbralMode.T6_GROWL

    # 3. Breathy: I_AIR high
    if i_air > HIGH_THRESH:
        return TimbralMode.T5_BREATHY

    # 4. Nasal: I_BOW high (sustained pressure → formant emphasis)
    if i_bow > HIGH_THRESH and i_tone > MED_THRESH:
        return TimbralMode.T4_NASAL

    # 5. Hollow: Low tone, medium air
    if i_tone < LOW_THRESH and i_air > LOW_THRESH:
        return TimbralMode.T3_HOLLOW

    # 6. Bright: High tone, low air
    if i_tone > MED_THRESH and i_air < LOW_THRESH:
        return TimbralMode.T1_BRIGHT

    # 7. Dark: Medium tone, medium bow
    if i_tone > LOW_THRESH and i_bow > MED_THRESH and i_air < MED_THRESH:
        return TimbralMode.T2_DARK

    # 8. Default: Pure
    return TimbralMode.T0_PURE


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
    'Q16_ONE', 'Q16_HALF', 'Q16_SCALE', 'Q8_ONE', 'Q32_SCALE',
    'DRONE_MODE_CARRIER_MIX',
    # Fixed-point utility functions
    'clamp_q15', 'clamp_q15u',
    # Block/tile constants
    'BLOCK_SIZE', 'BLOCK_SAMPLES', 'TILE_SIZE', 'BLOCKS_PER_TILE',
    'COEFF_LEN', 'NUM_LEAVES', 'VM_K', 'NODE_LEN', 'L_TOTAL_NODES',
    'SAMPLE_RATE_HZ', 'BLOCK_RATE_HZ',
    # Node iteration helpers
    'iter_active_nodes',
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
    'Regime', 'RegimeKind', 'EdgeTransitionType', 'RegimeInfo', 'REGIME_TABLE',
    # Event Lane
    'EventLane', 'EventLaneInfo', 'EVENT_LANE_TABLE',
    # Tail Mode
    'TailMode', 'TailModeInfo', 'TAIL_MODE_TABLE', 'get_tail_operators',
    # FM Archetype
    'FMArchetype', 'FMArchetypeInfo', 'FM_ARCHETYPE_TABLE',
    # Tree Archetype
    'TreeArchetype', 'TreeArchetypeInfo', 'TREE_ARCHETYPE_TABLE',
    # Oscillator Mode
    'OscMode', 'OscModeInfo', 'OSC_MODE_TABLE',
    # K Ladder
    'KLadder', 'K_LADDER_MASKS', 'get_k_mask', 'get_k_from_mask',
    # Symbol Mode (CMMM Enrichment Gap G1)
    'SymbolMode', 'SymbolModeInfo', 'SYMBOL_MODE_TABLE',
    'get_symbol_mode_evolution_rate', 'REGIME_TO_SYMBOL_MODE',
    # Timbral Mode (CMMM Enrichment Gap G4)
    'TimbralMode', 'TimbralModeInfo', 'TIMBRAL_MODE_TABLE',
    # Regime-Symbol Ops (CMMM Enrichment Gap G4)
    'REGIME_SYMBOL_OPS', 'get_operators_for_state', 'compute_timbral_mode_from_intent',
]
