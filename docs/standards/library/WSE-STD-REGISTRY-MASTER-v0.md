# WSE-STD-REGISTRY-MASTER v0 — Central Registry (Truth Table)

**Status:** Normative (v0)
**Applies to:** All runtime registries, preset loading, instrument compilation
**Implements:** `src/symcrash_python_ap/library/central_registry.py`

---

## 0. Purpose

The **Central Registry** (`CentralRegistry`) is the **unified access point** for all runtime registries in the WSE Tower system. It consolidates:

- **ControlRegistry** — Cockpit macro controls (user-facing parameters)
- **VoiceEngineRegistry** — Engine-owned parameters (DSP configuration)
- **RegimeRegistry** — Regime graph specifications (L2 state machines)
- **TreeRegistry** — Tree family specifications (wavelet coefficient structures)
- **OperatorRegistry** — CMMM operator specifications (processing lanes)
- **PatternRegistry** — Node mask patterns (K ladder configurations)
- **Primitives/Constants** — Fixed-point constants, block sizes, etc.

**Design principle:** Single source of truth per domain, lazy initialization, deterministic hashing.

---

## 1. Registry Hierarchy

```
CentralRegistry (unified access point)
├── controls         → ControlRegistry (WSE-STD-CONTROLREGISTRY.md)
├── voice_engine     → VoiceEngineRegistry (WSE-STD-VOICEENGINEREGISTRY-v0.md)
├── regimes          → RegimeRegistry (§3 this document)
├── trees            → TreeRegistry (§4 this document)
├── operators        → OperatorRegistry (§5 this document)
├── patterns         → PatternRegistry (§6 this document)
└── constants        → primitives module (§7 this document)
```

**Usage:**

```python
from symcrash_python_ap.library.central_registry import get_central_registry

registry = get_central_registry()

# Access individual registries
regime = registry.regimes.get(1)
tree = registry.trees.get(TreeID.HARMONIC)
control = registry.controls.get_entry(ControlID.TIMBRE)
voice_param = registry.voice_engine.get(VoiceParamID.OSC_MODE)
operator = registry.operators.get(OperatorID.COLOR)
pattern = registry.patterns.get(0)

# Access primitives/constants
block_size = registry.constants.BLOCK_SIZE
q15_one = registry.constants.Q15_ONE

# Introspection
registry.list_all_regimes()
registry.list_all_trees()
registry.list_all_controls()

# Registry summary
print(registry.summary())
```

---

## 2. Sub-Registry References

### 2.1 ControlRegistry (Cockpit Macros)

**Specification:** [WSE-STD-CONTROLREGISTRY](../control/WSE-STD-CONTROLREGISTRY.md)

| Property | Value |
|----------|-------|
| Scope | Cockpit macro controls (user-facing) |
| Frozen IDs | 0-19 (v0) |
| Entries | 12 controls (TIMBRE, TEXTURE, DYNAMICS, MORPH, CHAOS, DETUNE, PITCH_BEND, MASTER_GAIN, ADSR timing) |
| Slew policies | INSTANT, NBLOCK, ONE_POLE |

**Summary of v0 controls:**

| ID | Name | Type | Default | Slew | Binding |
|----|------|------|---------|------|---------|
| 0 | TIMBRE | Q15U | 32767 | ONE_POLE | family_scale[COLOR] |
| 1 | TEXTURE | Q15U | 32767 | ONE_POLE | family_scale[LOCAL] |
| 2 | DYNAMICS | Q15U | 32767 | ONE_POLE | gain_scale_q15 |
| 3 | MORPH | Q15U | 0 | NBLOCK | tree_blend_q15 |
| 4 | CHAOS | Q15U | 0 | NBLOCK | family_scale[CATASTROPHE] |
| 5 | DETUNE | I16 | 0 | INSTANT | pitch_offset_cents |
| 6 | PITCH_BEND | I16 | 0 | INSTANT | pitch_offset_cents |
| 9 | MASTER_GAIN | Q15U | 32767 | ONE_POLE | global.master_gain_q15 |
| 16-19 | ADSR | U16/Q15U | varies | NBLOCK | attack/decay/sustain/release |

### 2.2 VoiceEngineRegistry (Engine Parameters)

**Specification:** [WSE-STD-VOICEENGINEREGISTRY-v0](../voice/WSE-STD-VOICEENGINEREGISTRY-v0.md)

| Property | Value |
|----------|-------|
| Scope | Engine-owned parameters (DSP config) |
| Domains | CORE (0-31), CMMM (32-63), CCS (64-127), IDWPT (128-159), PLANNER (160-191), DEBUG (240-255) |
| Entries | 60+ parameters |
| Persistence | PER_VOICE, PROGRAM, GLOBAL |

**Domain summary:**

| Domain | ID Range | Content |
|--------|----------|---------|
| CORE | 0-31 | program_id, regime_graph_id, tree_id, voice_class, seed_policy |
| CMMM | 32-63 | route_id, regime_bias, dwell_blocks, transition_policy |
| CCS | 64-127 | osc_mode, waveform, FM6 (80-96), filter (97-104), unison, PWM |
| IDWPT | 128-159 | k_request, tail_policy, tree_xfade, wet_enable |
| PLANNER | 160-191 | max_ticks_hint, allow_nonlinear_hint |
| DEBUG | 240-255 | bypass_idwpt, freeze_coeffs |

---

## 3. RegimeRegistry (Regime Graphs)

**Implementation:** `src/symcrash_python_ap/library/regime_registry.py`

Regime graphs define L2 state machines that control voice behavior over time.

### 3.1 Entry Format

```python
@dataclass
class RegimeNodeSpec:
    """Single node in a regime graph."""
    name: str                    # Node name (e.g., "ATTACK", "STABLE")
    is_terminal: bool = False    # True if this is a terminal state

@dataclass
class EdgeSpec:
    """Edge in a regime graph."""
    source: str                  # Source node name
    target: str                  # Target node name
    condition: str = "default"   # Transition condition

@dataclass
class RegimeGraphSpec:
    """Complete regime graph specification."""
    start_regime: str           # Initial node name
    nodes: List[RegimeNodeSpec] # All nodes
    edges: List[EdgeSpec]       # All edges
```

### 3.2 Standard Regime Graphs (v0)

| ID | Hash | Name | Nodes | Description |
|----|------|------|-------|-------------|
| 0 | (computed) | minimal | 2 | IDLE ↔ ACTIVE |
| 1 | (computed) | standard | 4 | IDLE → ATTACK → STABLE → RELEASE |
| 2 | (computed) | expressive | 5 | Adds GLITCH state for controlled instability |

**Minimal graph (ID 0):**
```
IDLE ──────► ACTIVE ──────► IDLE
     note_on         note_off
```

**Standard graph (ID 1):**
```
IDLE ──► ATTACK ──► STABLE ──► RELEASE ──► IDLE
    note_on    complete     note_off    complete
```

**Expressive graph (ID 2):**
```
IDLE ──► ATTACK ──► STABLE ──┬──► RELEASE ──► IDLE
                             │
                             ▼
                          GLITCH ──► STABLE
```

### 3.3 Registry Methods

```python
class RegimeRegistry:
    def register(self, spec: RegimeGraphSpec) -> int  # Returns graph ID
    def get(self, graph_id: int) -> Optional[RegimeGraphSpec]
    def get_by_hash(self, hash_str: str) -> Optional[RegimeGraphSpec]
    def list_all(self) -> List[Tuple[int, str, RegimeGraphSpec]]  # (id, hash, spec)
```

### 3.4 Deterministic Hashing

Regime graphs are hashed using SHA256 of their canonical representation:

```python
def compute_graph_hash(spec: RegimeGraphSpec) -> str:
    canonical = {
        "start": spec.start_regime,
        "nodes": sorted([(n.name, n.is_terminal) for n in spec.nodes]),
        "edges": sorted([(e.source, e.target, e.condition) for e in spec.edges])
    }
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()[:16]
```

---

## 4. TreeRegistry (Tree Families)

**Implementation:** `src/symcrash_python_ap/library/tree_registry.py`

Tree families define wavelet coefficient distribution structures for IDWPT synthesis.

### 4.1 Entry Format

```python
class TreeID(IntEnum):
    """Canonical tree family IDs."""
    BALANCED = 0    # Even distribution across frequencies
    HARMONIC = 1    # Emphasize low-frequency partials
    TEXTURE = 2     # Emphasize high-frequency detail
    IMPACT = 3      # Attack transient emphasis
    DRONE = 4       # Sustained low-frequency emphasis

class TreeCoeffProfile(Enum):
    """Coefficient weighting profile."""
    FLAT = "flat"           # Equal weights
    HARMONIC = "harmonic"   # 1/n decay
    PERCUSSIVE = "percussive"  # Attack emphasis
    SUSTAIN = "sustain"     # Decay emphasis

@dataclass
class TreeFamily:
    """Tree family specification."""
    id: TreeID
    name: str
    description: str
    coeff_profile: TreeCoeffProfile
    blend_mode: str = "linear"  # How to blend between trees
```

### 4.2 Canonical Trees (v0)

| ID | Name | Profile | Description |
|----|------|---------|-------------|
| 0 | BALANCED | FLAT | Even distribution, general-purpose |
| 1 | HARMONIC | HARMONIC | Rich tonal content, 1/n decay |
| 2 | TEXTURE | PERCUSSIVE | High-frequency detail, noise-like |
| 3 | IMPACT | PERCUSSIVE | Attack transients, percussion |
| 4 | DRONE | SUSTAIN | Sustained low frequencies |

### 4.3 Registry Methods

```python
class TreeRegistry:
    def register(self, family: TreeFamily) -> None
    def get(self, tree_id: int) -> Optional[TreeFamily]
    def list_all(self) -> List[TreeFamily]
```

---

## 5. OperatorRegistry (CMMM Operators)

**Implementation:** `src/symcrash_python_ap/library/central_registry.py`

CMMM operators are processing lanes that transform wavelet coefficients.

### 5.1 Entry Format

```python
class OperatorID(IntEnum):
    """Canonical operator IDs for CMMM lanes."""
    NORMALIZE = 0       # Energy normalization
    COLOR = 1           # Timbral coloring
    TRANSPORT = 2       # Coefficient transport
    LOCAL = 3           # Local detail injection
    GLOBAL = 4          # Global structure
    CATASTROPHE = 5     # Controlled instability
    TAIL_FINE = 6       # Fine tail adjustment
    TAIL_COARSE = 7     # Coarse tail adjustment

@dataclass
class OperatorSpec:
    """Specification for a CMMM operator."""
    id: OperatorID
    name: str
    description: str
    cost_cycles: int        # Approximate FPGA cycles
    is_tail: bool = False   # True if this is a tail operator
    lane_index: int = 0     # Which lane this operator runs in
```

### 5.2 Default Operators (v0)

| ID | Name | Description | Cost | Tail? |
|----|------|-------------|------|-------|
| 0 | Normalize | Energy normalization | 50 | No |
| 1 | Color | Timbral coloring via spectral shaping | 80 | No |
| 2 | Transport | Coefficient transport/blur | 120 | No |
| 3 | Local | Local detail injection | 60 | No |
| 4 | Global | Global structure shaping | 40 | No |
| 5 | Catastrophe | Controlled instability | 100 | No |
| 6 | TailFine | Fine-grained tail coefficient clamping | 30 | Yes |
| 7 | TailCoarse | Coarse tail coefficient zeroing | 20 | Yes |

### 5.3 Registry Methods

```python
class OperatorRegistry:
    def get(self, op_id: int) -> Optional[OperatorSpec]
    def get_by_name(self, name: str) -> Optional[OperatorSpec]
    def list_all(self) -> List[OperatorSpec]
    def get_tail_operators(self) -> List[OperatorSpec]
    def get_backbone_operators(self) -> List[OperatorSpec]
    def total_cost(self, op_ids: List[int]) -> int
```

---

## 6. PatternRegistry (Node Masks)

**Implementation:** `src/symcrash_python_ap/library/central_registry.py`

Node mask patterns define which wavelet leaf nodes are active (K ladder).

### 6.1 Entry Format

```python
@dataclass
class PatternSpec:
    """Specification for a leaf node mask pattern."""
    id: int
    name: str
    mask_u32: int           # 32-bit mask for leaves 0-31
    mask_u32_hi: int = 0    # 32-bit mask for leaves 32-63
    description: str = ""

    @property
    def mask_u64(self) -> int:
        """Get full 64-bit mask."""
        return (self.mask_u32_hi << 32) | self.mask_u32

    @property
    def active_count(self) -> int:
        """Count of active leaves (K value)."""
        return bin(self.mask_u64).count('1')
```

### 6.2 Default Patterns (K Ladder, v0)

| ID | Name | K | Mask (hex) | Description |
|----|------|---|------------|-------------|
| 0 | K64_FULL | 64 | 0xFFFFFFFF_FFFFFFFF | All 64 leaves active |
| 1 | K32_EVEN | 32 | 0x55555555_55555555 | 32 leaves (even indices) |
| 2 | K16_HARMONIC | 16 | 0x00000000_0000FFFF | 16 leaves (harmonics) |
| 3 | K8_CORE | 8 | 0x00000000_000000FF | 8 core leaves |
| 4 | K4_MINIMAL | 4 | 0x00000000_0000000F | 4 minimal leaves |
| 5 | K2_STEREO | 2 | 0x00000000_00000003 | 2 stereo leaves |
| 6 | K1_MONO | 1 | 0x00000000_00000001 | 1 mono leaf |

### 6.3 Registry Methods

```python
class PatternRegistry:
    def get(self, pattern_id: int) -> Optional[PatternSpec]
    def get_by_name(self, name: str) -> Optional[PatternSpec]
    def register(self, spec: PatternSpec) -> int
    def list_all(self) -> List[PatternSpec]
```

---

## 7. Primitives (Constants)

**Implementation:** `src/symcrash_python_ap/primitives.py`

Accessed via `registry.constants`.

### 7.1 Fixed-Point Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Q15_ONE | 32767 | Maximum Q0.15 value (≈1.0) |
| Q15_HALF | 16384 | Half Q0.15 value (≈0.5) |
| Q15_ZERO | 0 | Zero Q0.15 value |
| Q16_ONE | 65536 | Q16.16 value for 1.0 |

### 7.2 System Constants

| Constant | Value | Description |
|----------|-------|-------------|
| BLOCK_SIZE | 64 | Samples per processing block |
| NUM_LEAVES | 64 | Maximum wavelet leaves (K=64) |
| SAMPLE_RATE_HZ | 48000 | Audio sample rate |
| MAX_VOICES | 32 | Maximum polyphony |

---

## 8. Precedence and Ownership

### 8.1 Control Flow Precedence

```
User Input
    │
    ▼
CockpitEngine (ControlRegistry)
    │ Applies slew, computes bindings
    ▼
VoiceFrame (per-block emission)
    │
    ▼
VoiceEngine (VoiceEngineRegistry)
    │ Applies engine-owned params
    ▼
Planner (budget constraints)
    │ May override effective K/tail/ops
    ▼
Runtime Execution
```

### 8.2 Registry Ownership Boundaries

| Registry | Owner | Mutability |
|----------|-------|------------|
| ControlRegistry | CockpitEngine | Slewed per-block |
| VoiceEngineRegistry | VoiceEngine | Per-voice or program |
| RegimeRegistry | InstrumentIR | Immutable after registration |
| TreeRegistry | InstrumentIR | Immutable after registration |
| OperatorRegistry | Global | Frozen v0 |
| PatternRegistry | Global | Extensible |

---

## 9. Validation and Introspection

### 9.1 Preset Resource Validation

```python
errors = registry.validate_preset_resources(
    regime_graph_id=1,
    tree_a_id=TreeID.HARMONIC,
    tree_b_id=TreeID.TEXTURE,
    pattern_id=0
)
if errors:
    raise ValueError(f"Invalid preset: {errors}")
```

### 9.2 Registry Summary

```python
print(registry.summary())
# Output:
# === Central Registry Summary ===
#
# Regimes:   3 registered
# Trees:     5 registered
# Controls:  12 registered
# Operators: 8 registered
# Patterns:  7 registered
#
# Constants:
#   Q15_ONE = 32767
#   Q15_HALF = 16384
#   BLOCK_SIZE = 64
#   NUM_LEAVES = 64
#   SAMPLE_RATE_HZ = 48000
```

---

## 10. Relationship to Other Standards

| Standard | Relationship |
|----------|--------------|
| [WSE-STD-CONTROLREGISTRY](../control/WSE-STD-CONTROLREGISTRY.md) | Defines ControlRegistry entries (frozen v0) |
| [WSE-STD-VOICEENGINEREGISTRY-v0](../voice/WSE-STD-VOICEENGINEREGISTRY-v0.md) | Defines VoiceEngineRegistry entries |
| [WSE-STD-070](../architecture/WSE-STD-070.md) | Tower levels reference registry IDs |
| [WSE-STD-CAPSULE-COMPAT-v0](WSE-STD-CAPSULE-COMPAT-v0.md) | Preset/capsule uses registry for validation |
| [WSE-STD-020](../core/WSE-STD-020.md) | VoiceFrame ABI carries registry-bound values |

---

## 11. Conformance Checklist

An implementation conforms to WSE-STD-REGISTRY-MASTER v0 iff:

- [ ] Implements CentralRegistry with lazy-initialized sub-registries
- [ ] All sub-registries accessible via property access
- [ ] ControlRegistry uses frozen v0 IDs (0-19)
- [ ] VoiceEngineRegistry uses frozen v0 domain ranges
- [ ] RegimeRegistry computes deterministic hashes
- [ ] TreeRegistry pre-registers 5 canonical trees
- [ ] OperatorRegistry has 8 frozen v0 operators
- [ ] PatternRegistry has 7 default K-ladder patterns
- [ ] Primitives accessible via constants property
- [ ] validate_preset_resources() returns empty list for valid presets
- [ ] summary() produces human-readable output

---

## 12. Migration from Separate Documents

This document consolidates registry specifications. The following documents remain authoritative for their specific domains:

| Document | Status | Content |
|----------|--------|---------|
| WSE-STD-CONTROLREGISTRY | **Authoritative** | ControlRegistry entry format, slew algorithms |
| WSE-STD-VOICEENGINEREGISTRY-v0 | **Authoritative** | VoiceEngineRegistry entry format, domains |
| This document | **Authoritative** | CentralRegistry API, Regime/Tree/Operator/Pattern specs |

---

*WSE-STD-REGISTRY-MASTER v0 — Normative*
