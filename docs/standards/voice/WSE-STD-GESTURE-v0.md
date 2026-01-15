# WSE-STD-GESTURE v0 — Gesture Lifecycle & Regime FSM

**Status:** Normative (v0 frozen)
**Applies to:** VoiceEngine, CMMM integration, cockpit input processing
**Depends on:**

* WSE-STD-VOICEENGINEREGISTRY v0 (engine parameters)
* WSE-STD-CONTROLREGISTRY v0 (cockpit macros)
* WSE-STD-020 (VoiceFrame ABI)

---

## 0. Purpose

The **Gesture Lifecycle Standard** defines the complete data flow from physical controller input to per-voice DSP configuration. This standard specifies:

* **GestureBasis**: The 8-field raw controller input structure
* **IntentVector**: The 8-lane semantic intent representation
* **Gesture → Intent Mapping**: Canonical transformation functions
* **Regime FSM**: The 7-state voice lifecycle state machine (R0-R6)
* **Edge Specifications**: Transition conditions and hysteresis
* **Policy Specifications**: Family, tree, and event activation policies

**Non-goal:** Replacing the VoiceEngineRegistry or ControlRegistry. This standard defines the *behavioral layer* that sits between controller input and those registries.

---

## 1. GestureBasis Structure (Frozen v0)

The **GestureBasis** is the raw physical controller state, normalized to fixed-point Q15 representation.

```python
@dataclass
class GestureBasis:
    """Physical gesture inputs - raw controller state."""
    gate: bool = False           # Note on/off (boolean)
    pressure_q15: int = 0        # Aftertouch / bow pressure
    position_q15: int = 0        # Position on surface (bipolar)
    speed_q15: int = 0           # Movement speed
    tilt_q15: int = 0            # Surface tilt (bipolar)
    chaos_q15: int = 0           # Random/noise input
    pitch_q15: int = 0           # Pitch bend (bipolar)
    pan_q15: int = 0             # Stereo position (bipolar)
```

### 1.1 Field Specifications (Frozen v0)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `gate` | bool | 0..1 | Note on/off state |
| `pressure_q15` | Q15U | 0..32767 | Aftertouch / bow pressure |
| `position_q15` | Q15S | -32768..+32767 | Position on controller surface |
| `speed_q15` | Q15U | 0..32767 | Movement speed (magnitude) |
| `tilt_q15` | Q15S | -32768..+32767 | Surface tilt angle |
| `chaos_q15` | Q15U | 0..32767 | Random/noise component |
| `pitch_q15` | Q15S | -32768..+32767 | Pitch bend amount |
| `pan_q15` | Q15S | -32768..+32767 | Stereo position |

### 1.2 Extended Fields (Optional)

Implementations MAY extend GestureBasis with:

* `gate_rise: bool` — True on rising edge of gate (note-on instant)
* `gate_fall: bool` — True on falling edge of gate (note-off instant)

These MUST be computed from gate state transitions between blocks.

### 1.3 Factory Methods (SHOULD)

```python
@classmethod
def from_note_on(cls, velocity_q15: int, pitch_q15: int = 0,
                 pan_q15: int = 0) -> 'GestureBasis':
    """Create gesture from basic MIDI note-on."""
    return cls(
        gate=True,
        pressure_q15=velocity_q15,
        position_q15=0,
        speed_q15=velocity_q15,  # Initial speed from velocity
        tilt_q15=0,
        chaos_q15=0,
        pitch_q15=pitch_q15,
        pan_q15=pan_q15,
    )
```

---

## 2. IntentVector Structure (Frozen v0)

The **IntentVector** is the semantic interpretation of gesture — what the voice "wants to do".

### 2.1 Lane Definitions (Frozen v0)

```python
# Intent lane indices
I_TONE   = 0   # Pitched tone strength
I_BOW    = 1   # Bowing/sustain pressure
I_AIR    = 2   # Breathiness/noise desire
I_GLITCH = 3   # Glitch/artifact desire
I_WARP   = 4   # Time/pitch warping desire
I_SNAP   = 5   # Transient snap desire
I_GLUE   = 6   # Coherence/binding desire
I_CALM   = 7   # Recovery/simplification desire
```

### 2.2 IntentVector Dataclass (Frozen v0)

```python
@dataclass
class IntentVector:
    """Semantic intent lanes - what the voice wants to do."""
    lanes: List[int] = field(default_factory=lambda: [0] * 8)

    @property
    def tone(self) -> int: return self.lanes[I_TONE]
    @property
    def bow(self) -> int: return self.lanes[I_BOW]
    @property
    def air(self) -> int: return self.lanes[I_AIR]
    @property
    def glitch(self) -> int: return self.lanes[I_GLITCH]
    @property
    def warp(self) -> int: return self.lanes[I_WARP]
    @property
    def snap(self) -> int: return self.lanes[I_SNAP]
    @property
    def glue(self) -> int: return self.lanes[I_GLUE]
    @property
    def calm(self) -> int: return self.lanes[I_CALM]
```

### 2.3 Intent Lane Semantics

| Lane | Purpose | High Value Indicates |
|------|---------|---------------------|
| I_TONE | Pitched tone production | Strong pitched sustain desired |
| I_BOW | Sustained pressure | Continuous bowing/breath pressure |
| I_AIR | Breathiness/texture | Desire for noise, air, texture |
| I_GLITCH | Artifacts | Desire for glitches, clicks, artifacts |
| I_WARP | Time/pitch warping | Morphing, pitch bending, modulation |
| I_SNAP | Transients | Attack transients, snaps, plucks |
| I_GLUE | Coherence | Desire for stability and binding |
| I_CALM | Recovery | Desire to simplify, recover from chaos |

---

## 3. Gesture → Intent Mapping (Normative v0)

### 3.1 Default Mapping Function (MUST)

```python
def default_gesture_to_intent(g: GestureBasis) -> IntentVector:
    """Default mapping from GestureBasis to IntentVector."""
    intent = IntentVector()

    # I_TONE: high when gate is on and pressure is moderate
    if g.gate:
        intent.lanes[I_TONE] = mul_q15_sat(g.pressure_q15, Q15_HALF) + Q15_HALF // 2

    # I_BOW: direct from pressure
    intent.lanes[I_BOW] = g.pressure_q15

    # I_AIR: increases with speed and low pressure
    air_from_speed = g.speed_q15 >> 1
    air_from_low_pressure = max(0, Q15_HALF - g.pressure_q15)
    intent.lanes[I_AIR] = sat16(air_from_speed + (air_from_low_pressure >> 1))

    # I_GLITCH: from chaos and extreme tilt
    glitch_from_chaos = g.chaos_q15
    glitch_from_tilt = abs(g.tilt_q15) >> 1
    intent.lanes[I_GLITCH] = sat16(glitch_from_chaos + glitch_from_tilt) >> 1

    # I_WARP: from pitch bend and extreme position
    intent.lanes[I_WARP] = sat16(abs(g.pitch_q15) + (abs(g.position_q15) >> 2))

    # I_SNAP: high on gate-on with high speed
    if g.gate and g.speed_q15 > Q15_HALF:
        intent.lanes[I_SNAP] = g.speed_q15
    else:
        intent.lanes[I_SNAP] = 0

    # I_GLUE: inverse of chaos - coherence desire
    intent.lanes[I_GLUE] = max(0, Q15_ONE - g.chaos_q15)

    # I_CALM: low pressure and speed = desire for calm
    calm_pressure = max(0, Q15_HALF - g.pressure_q15)
    calm_speed = max(0, Q15_HALF - g.speed_q15)
    intent.lanes[I_CALM] = (calm_pressure + calm_speed) >> 1

    return intent
```

### 3.2 Mapping Rules (MUST)

1. All intent lane values MUST be clamped to `[0, Q15_ONE]`
2. Mapping MUST be deterministic (same gesture → same intent)
3. Implementations MAY provide alternative mappings but MUST provide `default_gesture_to_intent`

---

## 4. Regime FSM (Frozen v0)

### 4.1 Regime Enum (Frozen v0)

```python
class Regime(IntEnum):
    R0_IDLE           = 0  # Voice silent, no processing
    R1_ATTACK_IGNITION = 1  # Note-on transient burst
    R2_STABLE_TONE    = 2  # Sustained pitched tone
    R3_GRAIN_AIR      = 3  # Breathy, textured sustain
    R4_GLITCH_CHATTER = 4  # Intentional artifacts
    R5_CATASTROPHE    = 5  # Extreme morphing/chaos
    R6_RECOVERY_HEAL  = 6  # Return to stability
```

### 4.2 Regime Semantics

| Regime | Name | Purpose |
|--------|------|---------|
| R0 | Idle | Voice allocated but silent; minimal processing |
| R1 | Attack/Ignition | Initial transient on note-on; impulse + noise burst |
| R2 | Stable Tone | Sustained pitched sound; harmonic tree dominant |
| R3 | Grain/Air | Breathy sustain; texture tree dominant |
| R4 | Glitch/Chatter | Intentional artifacts; topology operators active |
| R5 | Catastrophe | Extreme chaos; all families enabled |
| R6 | Recovery/Heal | Return to stability from chaos; glue dominant |

### 4.3 Regime State Graph

```
          ┌─────────────────────────────────────────────┐
          │                                             │
          v                                             │
    ┌──────────┐     gate_on      ┌──────────────────┐ │
    │ R0_IDLE  │ ─────────────────► R1_ATTACK_IGNIT  │ │
    └──────────┘                  └────────┬─────────┘ │
          ^                                │           │
          │                    dwell > 4   │           │
          │         ┌──────────────────────┘           │
          │         v                                  │
          │  ┌──────────────┐                          │
          │  │ R2_STABLE    │◄─────────────────────┐   │
          │  │    TONE      │                      │   │
          │  └──────┬───────┘                      │   │
          │         │                              │   │
          │    air > thr     tone > thr            │   │
          │         │              │               │   │
          │         v              │               │   │
          │  ┌──────────────┐      │               │   │
          │  │ R3_GRAIN_AIR │──────┘               │   │
          │  └──────┬───────┘                      │   │
          │         │                              │   │
          │   glitch > thr                         │   │
          │         │                              │   │
          │         v                              │   │
          │  ┌──────────────────┐                  │   │
          │  │ R4_GLITCH_CHATTER│                  │   │
          │  └──────┬───────────┘                  │   │
          │         │                              │   │
          │   chaos > thr                          │   │
          │         │                              │   │
          │         v                              │   │
          │  ┌──────────────┐     calm > thr       │   │
          │  │R5_CATASTROPHE│─────────────────►────┼───┘
          │  └──────┬───────┘                      │
          │         │                              │
          │   calm > thr                           │
          │         │                              │
          │         v                              │
          │  ┌──────────────────┐    glue > thr    │
          │  │ R6_RECOVERY_HEAL │──────────────────┘
          │  └──────┬───────────┘
          │         │
          │   gate_off
          │         │
          └─────────┘
```

---

## 5. RegimeSpec (Frozen v0)

Each regime has a **RegimeSpec** defining its behavioral configuration.

### 5.1 RegimeSpec Structure

```python
@dataclass
class RegimeSpec:
    """Complete specification for a regime's behavior."""
    regime: Regime
    name: str = ""

    # Family Policy: base weights and caps
    family_base: List[int] = field(default_factory=lambda: [0] * 8)
    family_cap: List[int] = field(default_factory=lambda: [Q15_ONE] * 8)
    family_enable_mask: int = 0xFF  # Bitmask of enabled families

    # Tree Policy
    preferred_tree: int = 0
    tree_lock: bool = False  # If True, ignore intent-based tree selection
    morph_rate: int = 512

    # Event Policy
    event_enable_mask: int = 0xFF
    event_probability: List[int] = field(default_factory=lambda: [Q15_HALF] * 8)

    # Tail Policy
    normalize_curve: str = "pink"  # "flat", "pink", "bright"
    fineclamp_cap_ratio: float = 0.5
    slew_max_delta: int = 512

    # Tower level for this regime
    tower_level: TowerLevel = TowerLevel.L0_FULL
```

### 5.2 Default RegimeSpecs (Normative v0)

| Regime | family_enable_mask | preferred_tree | event_enable_mask | tower_level |
|--------|-------------------|----------------|-------------------|-------------|
| R0_IDLE | 0x00 | - | 0x00 | L4_SCALAR |
| R1_ATTACK | 0xFF | 3 (Transient) | 0x07 | L1_COARSE |
| R2_STABLE | 0xFF | 1 (Harmonic) | 0x00 | L0_FULL |
| R3_GRAIN | 0xFF | 2 (Texture) | 0x02 | L1_COARSE |
| R4_GLITCH | 0xFF | 2 (Texture) | 0x2B | L2_SPARSE |
| R5_CATASTROPHE | 0xFF | 2 (Texture) | 0xF0 | L2_SPARSE |
| R6_RECOVERY | 0xFF | 4 (Drone) | 0x80 | L1_COARSE |

---

## 6. Edge Specification (Frozen v0)

### 6.1 EdgeType Enum

```python
class EdgeType(IntEnum):
    HARD = 0           # Immediate switch
    SOFT = 1           # Gradual blend
    CHAOTIC = 2        # Turbulent transition
    CORRIDOR_ENTRY = 3 # Entering transitional regime
    RECOVERY = 4       # Returning to stability
```

### 6.2 HysteresisSpec (MUST)

```python
@dataclass
class HysteresisSpec:
    """Hysteresis configuration for edge transitions."""
    enter_margin_q15: int = 1000  # ~3% margin to enter
    exit_margin_q15: int = 500    # ~1.5% margin to exit
    debounce_blocks: int = 4      # Minimum blocks between transitions
    last_transition_block: int = 0
```

### 6.3 ConditionSpec

```python
@dataclass
class ConditionSpec:
    """Condition specification for edge activation."""
    intent_thresholds: Dict[int, int] = field(default_factory=dict)
    intent_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    gate_required: Optional[bool] = None
    gate_rise_required: bool = False
    gate_fall_required: bool = False

    def check(self, gesture: GestureBasis, intent: IntentVector) -> bool:
        """Check if condition is satisfied."""
        # Check gate conditions
        if self.gate_required is not None and gesture.gate != self.gate_required:
            return False
        if self.gate_rise_required and not getattr(gesture, 'gate_rise', False):
            return False
        if self.gate_fall_required and not getattr(gesture, 'gate_fall', False):
            return False

        # Check intent thresholds
        for lane_id, threshold in self.intent_thresholds.items():
            if intent.lanes[lane_id] < threshold:
                return False

        # Check intent ranges
        for lane_id, (lo, hi) in self.intent_ranges.items():
            if not (lo <= intent.lanes[lane_id] <= hi):
                return False

        return True
```

### 6.4 EdgeSpec

```python
@dataclass
class EdgeSpec:
    """Specification for a regime transition edge."""
    from_regime: Regime
    to_regime: Regime
    edge_type: EdgeType = EdgeType.SOFT
    condition: ConditionSpec = field(default_factory=ConditionSpec)
    hysteresis: HysteresisSpec = field(default_factory=HysteresisSpec)
    min_dwell_blocks: int = 0
    duration_blocks: int = 8
    priority: int = 0
    arc_id: Optional[int] = None
```

---

## 7. Arc Specification (Frozen v0)

Arcs are short-duration modulators applied during regime transitions.

### 7.1 ArcSpec Structure

```python
@dataclass
class ArcSpec:
    """Specification for arc modulators during transitions."""
    arc_id: int = 0
    name: str = "default"
    duration_blocks: int = 8
    family_boost: List[int] = field(default_factory=lambda: [0] * 8)
    arc_events: int = 0  # Bitmask of events to fire
    gain_curve: str = "linear"  # "linear", "ease_in", "ease_out", "s_curve"
    morph_rate_override: int = 0
```

### 7.2 Default Arcs (Normative v0)

| Arc ID | Name | Duration | Family Boost | Events |
|--------|------|----------|--------------|--------|
| 1 | attack_burst | 4 | F_COLOR, F_EVENTS | E_IMPULSE |
| 2 | catastrophe_entry | 16 | F_TOPOLOGY, F_EVENTS, F_CATASTROPHE | E_CUSP, E_FOLD |
| 3 | recovery_smooth | 32 | F_GLUE | E_SHEAF |

---

## 8. Family Weights (Frozen v0)

### 8.1 Family Enum

```python
F_RESERVED    = 0  # Reserved / identity
F_COLOR       = 1  # Timbral color (Tilt, FineClamp)
F_TRANSPORT   = 2  # Energy transport (DiffuseUp/Down, BandShear)
F_LOCAL       = 3  # Local mixing (IntraMix, LowRankMorph)
F_TOPOLOGY    = 4  # Structural (SwapBands, RotateNodes, Permute)
F_GLUE        = 5  # Coherence (GluePair, NormalizeScale)
F_EVENTS      = 6  # Impulses, noise (Impulse, NoiseBand)
F_CATASTROPHE = 7  # Emergency (EmergencySimplify, FoldPressure)
```

### 8.2 Intent → Family Mapping (MUST)

```python
def compute_family_weights(intent: IntentVector, regime: Regime) -> List[int]:
    """Compute family weights from intent vector and regime."""
    weights = [0] * 8

    # Base mappings (intent -> family)
    weights[F_COLOR] = (intent.tone + intent.bow) >> 1
    weights[F_TRANSPORT] = (intent.warp + intent.air) >> 1
    weights[F_LOCAL] = (intent.bow + intent.tone) >> 1
    weights[F_TOPOLOGY] = (intent.glitch + intent.warp) >> 1
    weights[F_GLUE] = intent.glue
    weights[F_EVENTS] = (intent.snap + intent.glitch) >> 1

    # F_CATASTROPHE only active in R5
    if regime == Regime.R5_CATASTROPHE:
        weights[F_CATASTROPHE] = Q15_ONE
    else:
        weights[F_CATASTROPHE] = 0

    # Apply regime-specific modifiers
    # (see §8.3 for full specification)

    return weights
```

### 8.3 Regime Modifiers (Normative v0)

| Regime | Modifications |
|--------|---------------|
| R1_ATTACK | F_EVENTS += Q15_HALF, F_COLOR += Q15_HALF/2 |
| R2_STABLE | F_COLOR += Q15_HALF/2, F_LOCAL += Q15_HALF/2, F_GLUE += Q15_HALF/2, F_TOPOLOGY >>= 1 |
| R3_GRAIN | F_TRANSPORT += Q15_HALF, F_EVENTS += Q15_HALF/2 |
| R4_GLITCH | F_TOPOLOGY += Q15_HALF, F_EVENTS += Q15_HALF |
| R6_RECOVERY | F_GLUE += Q15_HALF, F_CATASTROPHE = 0, F_TOPOLOGY >>= 1, F_EVENTS >>= 1 |

---

## 9. Event Lanes (Frozen v0)

### 9.1 Event Enum

```python
E_IMPULSE     = 0  # Single-sample impulse
E_NOISE       = 1  # Noise burst
E_PULSE       = 2  # Pressure ramp
E_GATE        = 3  # Gate mask
E_CUSP        = 4  # Cusp injection (catastrophe)
E_SWALLOWTAIL = 5  # Swallowtail mask (catastrophe)
E_FOLD        = 6  # Fold pressure (catastrophe)
E_SHEAF       = 7  # Sheaf projection (recovery)
```

### 9.2 Lane Rule Types

```python
class LaneRuleType(IntEnum):
    DISABLED = 0     # Lane disabled
    GATE_EDGE = 1    # On gate rise/fall
    THRESHOLD = 2    # When intent exceeds threshold
    STOCHASTIC = 3   # Random with probability
    PATTERN = 4      # Pattern-driven by arc
```

---

## 10. Policy Specifications

### 10.1 FamilyPolicySpec

```python
@dataclass
class FamilyPolicySpec:
    """Computes w_family[8] from intents + regime params."""
    enable_mask: int = 0xFF
    base_q15: List[int] = field(default_factory=lambda: [0] * 8)
    W_q15: List[int] = field(default_factory=lambda: [0] * 64)  # 8x8 matrix
    bias_q15: List[int] = field(default_factory=lambda: [0] * 8)
    caps_q15: List[int] = field(default_factory=lambda: [Q15_ONE] * 8)
    slew_tc_blocks: List[int] = field(default_factory=lambda: [4] * 8)
    selection_rule: FamilySelectionRule = FamilySelectionRule.BY_INTENT
```

### 10.2 TreePolicySpec

```python
@dataclass
class TreePolicySpec:
    """Tree selection and blending policy."""
    allowed_pairs: List[Tuple[int, int]] = field(default_factory=list)
    selection_rule: TreeSelectionRule = TreeSelectionRule.BY_INTENT
    blend_rule: TreeBlendRule = TreeBlendRule.INTENT_WARP
    blend_slew_tc: int = 8
    default_tree_a: int = 0
    default_tree_b: int = 0
    fixed_blend_q15: int = 0
```

### 10.3 EventPolicySpec

```python
@dataclass
class EventPolicySpec:
    """Event generation policy."""
    enable_mask: int = 0xFF
    lane_rules: List[LaneRuleSpec] = field(
        default_factory=lambda: [LaneRuleSpec() for _ in range(8)]
    )
```

---

## 11. RegimeGraph (Frozen v0)

### 11.1 Structure

```python
@dataclass
class RegimeGraph:
    """Complete regime graph with all edges and arc specifications."""
    edges: List[EdgeSpec] = field(default_factory=list)
    regime_specs: Dict[Regime, RegimeSpec] = field(default_factory=dict)
    arc_specs: Dict[int, ArcSpec] = field(default_factory=dict)

    def get_valid_transitions(self, from_regime: Regime) -> List[EdgeSpec]:
        """Get all edges leaving a regime, sorted by priority."""
        edges = [e for e in self.edges if e.from_regime == from_regime]
        return sorted(edges, key=lambda e: -e.priority)

    def check_transition(self, from_regime: Regime, intent: IntentVector,
                        dwell_blocks: int) -> Optional[Regime]:
        """Check if any transition should occur."""
        for edge in self.get_valid_transitions(from_regime):
            if edge.check_condition(intent, dwell_blocks):
                return edge.to_regime
        return None
```

---

## 12. Degradation Levels (Frozen v0)

```python
class DegradationLevel(IntEnum):
    NONE = 0       # Full processing
    LIGHT = 1      # Disable expensive transport ops
    HEAVY = 2      # Disable transport + topology
    EMERGENCY = 3  # Only tail stages (normalize + clamp)
```

---

## 13. Lifecycle Flow (Normative)

### 13.1 Per-Block Processing Order (MUST)

1. **Receive GestureBasis** from cockpit/controller
2. **Map Gesture → Intent** via `default_gesture_to_intent()` or custom mapper
3. **Check Regime Transitions** via `RegimeGraph.check_transition()`
4. **Apply Arc** if transition occurred (duration-limited modulation)
5. **Compute Family Weights** via `compute_family_weights(intent, regime)`
6. **Select Trees** via TreePolicySpec
7. **Generate Events** via EventPolicySpec
8. **Emit VoiceFrame** with computed parameters

### 13.2 Voice Allocation (MUST)

On NOTE_ON / allocator event:

1. Set regime to R1_ATTACK_IGNITION
2. Apply GestureBasis.from_note_on(velocity, pitch, pan)
3. Initialize dwell counter to 0
4. Apply RegimeSpec for R1

### 13.3 Voice Release (MUST)

On NOTE_OFF / release event:

1. Set gate = False, gate_fall = True
2. Transition to R6_RECOVERY_HEAL (if not already in R0)
3. Apply recovery arc
4. After recovery dwell, transition to R0_IDLE

---

## 14. Relationship to Other Standards

* **VoiceEngineRegistry** (§WSE-STD-VOICEENGINEREGISTRY-v0): Defines engine parameter IDs and ranges; this standard defines how those parameters are computed from gesture.
* **ControlRegistry** (§WSE-STD-CONTROLREGISTRY-v0): Defines cockpit macros; this standard sits between cockpit input and engine parameters.
* **VoiceFrame ABI** (§WSE-STD-020): This standard produces VoiceFrame emission; VoiceFrame carries the computed parameters to the DSP engine.
* **InstrumentIR** (§03 - InstrumentIR v0): L1 (Gesture Lift) and L2 (Regimes) of the tower correspond to this standard's GestureBasis→Intent and Regime FSM.

---

## 15. Conformance Checklist (v0)

An implementation conforms iff it:

* [ ] Implements GestureBasis 8-field structure (§1)
* [ ] Implements IntentVector 8-lane structure (§2)
* [ ] Provides `default_gesture_to_intent()` mapping (§3)
* [ ] Implements Regime enum R0-R6 (§4)
* [ ] Implements RegimeSpec per regime (§5)
* [ ] Implements EdgeSpec with hysteresis (§6)
* [ ] Implements ArcSpec for transitions (§7)
* [ ] Computes family weights per §8
* [ ] Implements event lanes E0-E7 (§9)
* [ ] Follows per-block processing order (§13.1)
* [ ] Produces deterministic results given identical inputs

---

## 16. Implementation Guide (Reference)

Recommended module structure:

* `gesture.py`
  * `GestureBasis` dataclass
  * `from_note_on()` factory

* `intent.py`
  * `IntentVector` dataclass
  * `default_gesture_to_intent()` function

* `regime.py`
  * `Regime` enum
  * `RegimeSpec` dataclass
  * `REGIME_SPECS` default table

* `regime_graph.py`
  * `EdgeSpec`, `ArcSpec`, `HysteresisSpec`
  * `RegimeGraph` class
  * `build_default_regime_graph()` factory

* `policy.py`
  * `FamilyPolicySpec`, `TreePolicySpec`, `EventPolicySpec`
  * Policy computation functions

---

## Appendix A: Q15 Fixed-Point Conventions

```python
Q15_ONE = 32767   # 1.0 in Q0.15
Q15_HALF = 16384  # 0.5 in Q0.15

def mul_q15_sat(a: int, b: int) -> int:
    """Multiply two Q15 values with saturation."""
    result = (a * b) >> 15
    return max(-32768, min(32767, result))

def sat16(x: int) -> int:
    """Saturate to signed 16-bit range."""
    return max(-32768, min(32767, x))
```

---

## Appendix B: Tree Semantics

| Tree ID | Name | Use Case |
|---------|------|----------|
| 0 | Default | Fallback tree |
| 1 | Harmonic | Pitched, tonal sounds |
| 2 | Texture | Noisy, breathy sounds |
| 3 | Transient | Attack, pluck sounds |
| 4 | Drone | Stable, ambient sounds |

Tree selection is driven by intent lanes per TreePolicySpec.

---

*End of WSE-STD-GESTURE v0*
