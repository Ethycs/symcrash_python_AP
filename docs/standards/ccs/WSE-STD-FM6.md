# WSE-STD-FM6 — 6-Operator FM Synthesis Standard

**Standard ID:** WSE-STD-FM6
**Title:** 6-Operator FM Synthesis with DX7-Compatible Algorithms
**Version:** 1.0
**Status:** DRAFT
**Scope:** Defines topology representation, HQ render kernel, and evaluation model for DX7-compatible 6-operator FM synthesis.
**Depends on:** WSE-STD-001 (Fixed-Point), WSE-STD-210 (SineCore, Phase Model)
**Related:** WSE-STD-BRAID (coefficient-domain braid modulation)

---

## 0. Design Invariants (MUST)

1. **Graph, Not Knot.** Algorithm topologies are represented as directed graphs with weighted edges, NOT knot-theoretic structures. DX7 algorithms are planar DAGs with at most one feedback edge.

2. **Topological Evaluation Order.** Operators MUST be evaluated in topological order: modulators compute before carriers. The single feedback path uses z^-1 block delay.

3. **Lane-Rate or Sample-Rate.** FM modulation occurs at lane rate (K samples per block) for lite mode, or sample rate for HQ mode.

4. **DX7 Compatibility.** All 32 original DX7 algorithms MUST be representable and produce equivalent spectra within implementation tolerances.

5. **Deterministic Fixed-Point.** All operations MUST be deterministic across platforms per WSE-STD-001.

---

## 1. Mathematical Foundation

### 1.1 Why Not Knot Theory?

DX7-style FM algorithms do NOT require knot-theoretic representation because:

| Property | Knot Theory | FM Algorithms |
|----------|-------------|---------------|
| **Dimensionality** | 3D embeddings | 2D planar graphs |
| **Crossings** | Essential feature | Never required |
| **Feedback loops** | Linked rings, trefoils | Single self-loop per algorithm |
| **Invariants needed** | Jones polynomial, etc. | Topological sort |
| **Computational cost** | O(n^3+) | O(n^2) adjacency |

**Key Insight:** A DX7 algorithm is a **DAG (Directed Acyclic Graph) with at most one marked feedback vertex**. This is topologically trivial and does not benefit from knot invariants.

### 1.2 Graph Representation

An FM algorithm is represented as:

```
G = (V, E, C, fb)
```

Where:
- `V = {Op1, Op2, ..., Op6}` — 6 operator vertices
- `E c V x V` — modulation edges (src modulates dst)
- `C c V` — carrier set (operators that sum to output)
- `fb in V | {nil}` — feedback operator (self-modulating)

### 1.3 Evaluation Order

Given graph G, compute topological order by:
1. Remove feedback edge (if present) to create DAG
2. Apply Kahn's algorithm or DFS-based sort
3. Modulators appear before their targets in order

**Invariant:** For any edge (u, v) in E (except feedback), u appears before v in eval_order.

### 1.4 FM State Machine (HQ Model)

FM synthesis is fundamentally a state machine:

```
theta_i[t+1] = theta_i[t] + omega_i + sum_{j in mods(i)} (I_ij * y_j[t]) + fb_i[t]
y_i[t] = A_i[t] * sin(theta_i[t])
```

Where:
- `theta_i` is phase accumulator for operator i
- `omega_i` is base frequency (ratio * fundamental)
- `I_ij` is modulation index from operator j to i
- `fb_i` is self-feedback contribution

---

## 2. Data Structures

### 2.1 FMOperator

Per-operator state and parameters:

```python
@dataclass
class FMOperator:
    """Single FM operator with DX7-compatible parameters."""

    # Identity
    op_id: int                    # 0-5 (operator index)

    # Frequency
    ratio_q16: int = 65536        # Frequency ratio (Q16: 65536 = 1.0x)
    detune_cents: int = 0         # Fine detune (-99 to +99 cents)

    # Level
    level_q15: int = 32767        # Output level (Q15)
    velocity_sens: int = 0        # Velocity sensitivity (0-7)

    # Feedback (only meaningful if this is feedback_op)
    feedback_q15: int = 0         # Self-modulation depth (Q15)

    # Phase state
    phase: int = 0                # Q32 phase accumulator

    # Envelope (DX7-style rate/level)
    env_rates: List[int] = field(default_factory=lambda: [99, 99, 99, 99])
    env_levels: List[int] = field(default_factory=lambda: [99, 99, 99, 0])
    env_phase: int = 0            # Current envelope segment (0-3)
    env_value_q15: int = 0        # Current envelope value
```

### 2.2 FMAlgorithm

Algorithm topology definition:

```python
@dataclass
class FMAlgorithm:
    """FM algorithm topology (DX7-compatible)."""

    alg_id: int                           # 1-32 (DX7 algorithm number)
    name: str                             # Human-readable name

    # Topology
    modulation_matrix: np.ndarray         # [6,6] bool: M[i,j]=True if Op_i modulates Op_j
    carrier_mask: int                     # Bitmask: bit i set if Op_i is a carrier
    feedback_op: int                      # Which op has feedback (-1 if none)

    # Derived (computed once at load time)
    eval_order: List[int]                 # Topological evaluation order
    modulator_lists: List[List[int]]      # Per-op list of modulators
```

### 2.3 FM6State

Runtime state for 6-operator FM voice:

```python
@dataclass
class FM6State:
    """Runtime state for 6-op FM synthesis."""

    operators: List[FMOperator]           # 6 operators
    algorithm: FMAlgorithm                # Current algorithm

    # Per-block intermediate storage
    op_outputs: np.ndarray                # [6, K] lane-rate operator outputs
    feedback_buffer: np.ndarray           # [K] previous block's feedback op output

    # Master controls
    master_level_q15: int = 32767         # Overall output scaling
    master_pitch_bend: int = 0            # Pitch bend (cents)
```

### 2.4 FM6HQVoiceState (HQ Mode)

Extended state for sample-accurate HQ rendering:

```python
@dataclass
class FM6HQVoiceState:
    """Per-voice state for HQ FM6 rendering."""

    # Operator phase accumulators (Q32 phase)
    theta_q32: List[int]          # [NUM_OPS] wraps mod 2^32

    # Feedback delay (z^-1) in sample domain (Q15 or Q31)
    fb_z1: List[int]              # [NUM_OPS] last output sample for feedback ops

    # Per-op envelopes (block or sample)
    env: List[EnvState]           # [NUM_OPS]

    # Cached per-block increments (Q32)
    base_inc_q32: List[int]       # [NUM_OPS] ratio * phase_inc, computed per note

    # Scratch: operator outputs for current block (Q15 or Q31)
    op_out: np.ndarray            # [NUM_OPS, BLOCK] store full block for modulation
```

---

## 3. Algorithm Table (32 DX7 Algorithms)

### 3.1 Encoding Format

Each algorithm is encoded as:
- `mod_matrix`: 36-bit packed (6x6 matrix, row-major)
- `carrier_mask`: 6-bit mask
- `feedback_op`: 3-bit (0-5 or 7 for none)

### 3.2 Example Algorithms

#### Algorithm 1: 6-Op Chain
```
[6] -> [5] -> [4] -> [3] -> [2] -> [1*] -> OUT
                                    |fb
```
- Carriers: {1}
- Modulators: 6->5, 5->4, 4->3, 3->2, 2->1
- Feedback: Op 1
- Eval order: [6, 5, 4, 3, 2, 1]

#### Algorithm 5: Dual Stack into Single Carrier
```
[6] -> [5] -> [4] --+
                    +---> [1] -> OUT
[3] -> [2] --------+
```
- Carriers: {1}
- Modulators: 6->5, 5->4, 4->1, 3->2, 2->1
- Feedback: Op 6 (typical)
- Eval order: [6, 5, 4, 3, 2, 1]

#### Algorithm 32: All Carriers (Additive)
```
[1] -> OUT
[2] -> OUT
[3] -> OUT
[4] -> OUT
[5] -> OUT
[6] -> OUT
```
- Carriers: {1, 2, 3, 4, 5, 6}
- Modulators: none
- Feedback: any (typically Op 6)
- Eval order: any

*(Full 32-algorithm table in Appendix A)*

---

## 4. Synthesis Algorithms

### 4.1 Lane-Rate Evaluation (Lite Mode)

```python
def fm6_generate_block(state: FM6State, base_phase_inc: int) -> np.ndarray:
    """
    Generate one block of FM audio (lane-rate).

    Args:
        state: FM6State with operators and algorithm
        base_phase_inc: Q32 phase increment for fundamental

    Returns:
        output_k: [K] lane-rate output samples (Q15)
    """
    alg = state.algorithm
    ops = state.operators
    K = 8  # Lane count

    # Clear operator outputs
    state.op_outputs.fill(0)

    # Evaluate in topological order
    for op_idx in alg.eval_order:
        op = ops[op_idx]

        # Compute operator phase increment
        op_phase_inc = (base_phase_inc * op.ratio_q16) >> 16

        # Apply modulation from all modulators
        mod_sum = np.zeros(K, dtype=np.int32)
        for mod_idx in alg.modulator_lists[op_idx]:
            mod_sum += state.op_outputs[mod_idx]

        # Apply self-feedback (if this is feedback op)
        if op_idx == alg.feedback_op:
            mod_sum += (state.feedback_buffer * op.feedback_q15) >> 15

        # Generate operator output for each k-slot
        for k in range(K):
            # Advance phase
            op.phase = (op.phase + op_phase_inc) & 0xFFFFFFFF

            # Apply modulation to phase
            mod_phase = (mod_sum[k] << 16) & 0xFFFFFFFF  # Scale to Q32
            effective_phase = (op.phase + mod_phase) & 0xFFFFFFFF

            # Compute sine (using SineCore from WSE-STD-210)
            sin_val, _ = sin_cos_q15(effective_phase)

            # Apply envelope and level
            output = RSHIFT_RNE(sin_val * op.level_q15, 15)
            output = RSHIFT_RNE(output * op.env_value_q15, 15)

            state.op_outputs[op_idx, k] = SAT16(output)

    # Store feedback for next block (z^-1 delay)
    if alg.feedback_op >= 0:
        state.feedback_buffer[:] = state.op_outputs[alg.feedback_op]

    # Sum carrier outputs
    output_k = np.zeros(K, dtype=np.int32)
    for c in range(6):
        if alg.carrier_mask & (1 << c):
            output_k += state.op_outputs[c]

    # Apply master level and saturate
    output_k = RSHIFT_RNE(output_k * state.master_level_q15, 15)
    return np.clip(output_k, -32768, 32767).astype(np.int16)
```

### 4.2 HQ Sample-Rate Evaluation

For high-quality rendering with true per-sample FM:

```python
def fm6_hq_render_block(
    state: FM6HQVoiceState,
    alg: FM6Algorithm,
    controls: FM6Controls,
    BLOCK: int = 64
) -> np.ndarray:
    """
    HQ FM6 render: sample-accurate phase integration.

    Returns:
        audio_block: [BLOCK] time-domain samples (Q15)
    """
    op_out = state.op_out  # [NUM_OPS, BLOCK]

    # Evaluate operators in topological order
    for op_idx in alg.topo_order:
        theta = state.theta_q32[op_idx]
        base = state.base_inc_q32[op_idx]

        for t in range(BLOCK):
            dtheta = base

            # FM from modulators (feedforward)
            for (j, depth_q15) in alg.mods[op_idx]:
                # Q15 * Q15 -> Q15
                fm_q15 = (op_out[j, t] * depth_q15) >> 15
                # Map to Q32 phase increment
                dtheta += fm_q15 * FM_PHASE_SCALE_Q32

            # Feedback (z^-1 sample)
            if alg.has_feedback[op_idx]:
                fb_q15 = (state.fb_z1[op_idx] * alg.fb_gain_q15[op_idx]) >> 15
                dtheta += fb_q15 * FM_PHASE_SCALE_Q32

            theta = (theta + dtheta) & 0xFFFFFFFF

            s_q15 = sin_q15(theta)
            a_q15 = env_gain_q15(state.env[op_idx], t)
            y_q15 = (s_q15 * a_q15) >> 15

            op_out[op_idx, t] = y_q15

        # Update persistent state
        state.theta_q32[op_idx] = theta
        if alg.has_feedback[op_idx]:
            state.fb_z1[op_idx] = op_out[op_idx, BLOCK - 1]

    # Sum carrier outputs
    audio_block = np.zeros(BLOCK, dtype=np.int32)
    for c in range(6):
        if alg.carrier_mask & (1 << c):
            audio_block += op_out[c]

    return np.clip(audio_block, -32768, 32767).astype(np.int16)
```

### 4.3 Converting to Coefficient Format

The lane-rate output `[K]` is expanded to coefficient format `[512]`:

```python
def fm6_to_coefficients(output_k: np.ndarray) -> np.ndarray:
    """
    Convert lane-rate FM output to CCS coefficient format.

    Per WSE-STD-210, this replicates across leaves for direct
    IDWPT synthesis (the FM signal becomes the carrier).
    """
    coeffs = np.zeros(512, dtype=np.int16)

    # Replicate each k-slot value across 64 leaves
    for k in range(8):
        for leaf in range(64):
            coeffs[leaf * 8 + k] = output_k[k]

    return coeffs
```

---

## 5. Fixed-Point Conventions

### 5.1 Format Summary

| Stage | Format | Range | Notes |
|-------|--------|-------|-------|
| Phase accumulator | Q32 wrap | 0..2^32-1 | Maps to 0..2pi |
| Phase increment | Q32 | 0..2^32-1 | Per sample |
| Operator output | Q15 | [-1, 1) | Primary format |
| Modulation depth | Q15 | [-1, 1) | Edge weight |
| sin/cos LUT | Q15 | [-1, 1) | 1024-4096 entries |
| Internal MAC | Q3.29 | [-8, 8) | Guard bits |

### 5.2 FM Phase Scaling

Define one global mapping for modulation depth to phase:

```python
# Converts (Q15 * Q15) -> Q32 phase increment
# dtheta_q32 += ((y_q15 * depth_q15) >> 15) * FM_PHASE_SCALE_Q32
FM_PHASE_SCALE_Q32 = 65536  # Tunable: depth=1.0 -> reasonable modulation index
```

**Requirement:** Scaling must be deterministic and overflow-safe.

### 5.3 Rounding and Saturation

Per WSE-STD-001:
- All right shifts use `RSHIFT_RNE` (round-to-nearest-even)
- All narrowing operations use `SAT16`
- Silent wrap is forbidden
- Overflow clamps to +/-32767

---

## 6. Feedback Stability

### 6.1 Feedback Depth Limits

DX7-compatible feedback uses integer levels 0-7, mapped to Q15:

| DX7 Level | feedback_q15 | Approximate beta |
|-----------|--------------|------------------|
| 0 | 0 | 0.0 |
| 1 | 4681 | ~0.143 |
| 2 | 9362 | ~0.286 |
| 3 | 14043 | ~0.429 |
| 4 | 18724 | ~0.571 |
| 5 | 23405 | ~0.714 |
| 6 | 28086 | ~0.857 |
| 7 | 32767 | ~1.0 |

### 6.2 Stability Analysis

For self-modulating operator with feedback depth beta:
- `beta < pi`: Bounded, produces rich harmonics
- `beta ~ pi`: Approaches maximum harmonic density
- `beta > pi`: Can become chaotic/noisy

DX7 level 7 (beta ~ 1.0) is well below the chaotic threshold, ensuring stability.

### 6.3 Implementation Requirements

Implementations MUST:
1. Limit feedback_q15 to range [0, 32767]
2. Use block-delay (z^-1) for feedback to ensure determinism
3. Apply saturation (SAT16) after feedback application

---

## 7. Topology Morphing

### 7.1 Canonical FM Archetypes

The 32 DX7 algorithms cluster into 8 structural families:

| ID | Name | Carriers | Structure | Character |
|----|------|----------|-----------|-----------|
| 0 | CHAIN | 1 | [6]->[5]->[4]->[3]->[2]->[1*] | Deep modulation, metallic |
| 1 | STACK | 1 | [6,5,4,3,2]->[1*] | Bright, aggressive |
| 2 | DUAL | 2 | [6]->[5]->[4*], [3]->[2]->[1*] | Dual voice, detuned |
| 3 | SPLIT | 3+ | [6]->[5]->[4*,3*,2*,1*] | Organ-like, spread |
| 4 | YPAIR | 2 | [6]->[5,4]->[3*], [2]->[1*] | Y-branch, complex |
| 5 | CLOUD | 1 | [6,5,4,3,2]->[1*] parallel | Thick, chorused |
| 6 | ADDITIVE | 6 | [6*,5*,4*,3*,2*,1*] | Pure additive |
| 7 | FEEDBACK | 1 | Chain with enhanced FB | Harsh, distorted |

### 7.2 Archetype Edge Lists

Each archetype is defined by a canonical edge list `(src, dst, weight_q15)`:

```python
ARCHETYPE_EDGES = {
    # 0: CHAIN - Serial 6-op chain
    0: {
        'edges': [
            (5, 4, 32767),  # Op6 -> Op5
            (4, 3, 32767),  # Op5 -> Op4
            (3, 2, 32767),  # Op4 -> Op3
            (2, 1, 32767),  # Op3 -> Op2
            (1, 0, 32767),  # Op2 -> Op1
        ],
        'carrier_mask': 0b000001,
        'feedback_op': 0,
    },

    # 1: STACK - 5 modulators into 1 carrier
    1: {
        'edges': [
            (5, 0, 32767),  # Op6 -> Op1
            (4, 0, 32767),  # Op5 -> Op1
            (3, 0, 32767),  # Op4 -> Op1
            (2, 0, 32767),  # Op3 -> Op1
            (1, 0, 32767),  # Op2 -> Op1
        ],
        'carrier_mask': 0b000001,
        'feedback_op': 5,
    },

    # 2: DUAL - Two parallel 3-op chains
    2: {
        'edges': [
            (5, 4, 32767),  # Op6 -> Op5
            (4, 3, 32767),  # Op5 -> Op4
            (2, 1, 32767),  # Op3 -> Op2
            (1, 0, 32767),  # Op2 -> Op1
        ],
        'carrier_mask': 0b001001,  # Op1, Op4
        'feedback_op': 5,
    },

    # 3: SPLIT - Modulator tree to multiple carriers
    3: {
        'edges': [
            (5, 4, 32767),  # Op6 -> Op5
            (4, 3, 32767),  # Op5 -> Op4
            (4, 2, 32767),  # Op5 -> Op3
            (4, 1, 32767),  # Op5 -> Op2
        ],
        'carrier_mask': 0b001111,  # Op1-4
        'feedback_op': 5,
    },

    # 4: YPAIR - Y-branch structure
    4: {
        'edges': [
            (5, 4, 32767),  # Op6 -> Op5
            (5, 3, 32767),  # Op6 -> Op4
            (4, 2, 32767),  # Op5 -> Op3
            (3, 2, 32767),  # Op4 -> Op3
            (1, 0, 32767),  # Op2 -> Op1
        ],
        'carrier_mask': 0b000101,  # Op1, Op3
        'feedback_op': 5,
    },

    # 5: CLOUD - Parallel modulators (thick)
    5: {
        'edges': [
            (5, 0, 32767),  # Op6 -> Op1
            (4, 0, 24576),  # Op5 -> Op1 (slightly reduced)
            (3, 0, 16384),  # Op4 -> Op1
            (2, 0, 16384),  # Op3 -> Op1
            (1, 0, 24576),  # Op2 -> Op1
        ],
        'carrier_mask': 0b000001,
        'feedback_op': 5,
    },

    # 6: ADDITIVE - All carriers, no modulation
    6: {
        'edges': [],
        'carrier_mask': 0b111111,  # All carriers
        'feedback_op': 5,
    },

    # 7: FEEDBACK - Chain with enhanced feedback flag
    7: {
        'edges': [
            (5, 4, 32767),
            (4, 3, 32767),
            (3, 2, 32767),
            (2, 1, 32767),
            (1, 0, 32767),
        ],
        'carrier_mask': 0b000001,
        'feedback_op': 0,
        'feedback_enhanced': True,  # 2x feedback depth
    },
}
```

### 7.3 DX7 -> Archetype Mapping

```python
DX7_TO_ARCHETYPE = {
    1: 0, 2: 0,                             # CHAIN
    3: 2, 4: 2, 9: 2, 10: 2, 11: 2,          # DUAL
    12: 2, 13: 2, 30: 2,
    5: 4, 6: 4, 20: 4,                       # YPAIR
    7: 3, 8: 3, 14: 3, 15: 3, 19: 3,         # SPLIT
    21: 3, 22: 3, 28: 3, 29: 3,
    16: 1, 17: 1,                            # STACK
    18: 5,                                    # CLOUD
    23: 6, 24: 6, 25: 6, 26: 6,              # ADDITIVE
    27: 6, 31: 6, 32: 6,
}
```

### 7.4 Morphing Parameters

| Parameter | ID | Type | Range | Default | Description |
|-----------|-----|------|-------|---------|-------------|
| `fm.topo_a_id` | 0x40 | u8 | 0-7 | 0 | Source archetype |
| `fm.topo_b_id` | 0x41 | u8 | 0-7 | 0 | Target archetype |
| `fm.topo_blend_q15` | 0x42 | i16 | 0-32767 | 0 | A/B blend (0=A, 32767=B) |
| `fm.dx7_compat_id` | 0x43 | u8 | 1-32 | 0 | DX7 algorithm (overrides topo_a) |

### 7.5 Blend Interpolation

#### Edge Weight Interpolation

```python
def blend_edge_weight(edge_a_q15: int, edge_b_q15: int, blend_q15: int) -> int:
    """Interpolate edge weight between A and B topologies."""
    Q15_ONE = 32767
    inv_blend = Q15_ONE - blend_q15
    result = (edge_a_q15 * inv_blend + edge_b_q15 * blend_q15) >> 15
    return result
```

#### Carrier Mask Interpolation

```python
def blend_carrier_level(is_carrier_a: bool, is_carrier_b: bool,
                        level_q15: int, blend_q15: int) -> int:
    """Blend carrier output level based on topology transition."""
    Q15_ONE = 32767
    if is_carrier_a and is_carrier_b:
        return level_q15  # Carrier in both - full level
    elif is_carrier_a and not is_carrier_b:
        return (level_q15 * (Q15_ONE - blend_q15)) >> 15  # Fading out
    elif not is_carrier_a and is_carrier_b:
        return (level_q15 * blend_q15) >> 15  # Fading in
    else:
        return 0  # Carrier in neither
```

#### Feedback Operator Transition

```python
def blend_feedback(fb_op_a: int, fb_op_b: int, blend_q15: int,
                   op_outputs: np.ndarray) -> np.ndarray:
    """Blend feedback from two different operators."""
    Q15_ONE = 32767
    if fb_op_a == fb_op_b:
        return op_outputs[fb_op_a]

    # Crossfade between feedback sources
    inv_blend = Q15_ONE - blend_q15
    fb_a = (op_outputs[fb_op_a] * inv_blend) >> 15
    fb_b = (op_outputs[fb_op_b] * blend_q15) >> 15
    return fb_a + fb_b
```

### 7.6 CMMM Integration

CMMM regime transitions can drive FM topology morphing:

| CMMM Regime | Suggested Archetype | Rationale |
|-------------|---------------------|-----------|
| IDLE (0) | ADDITIVE (6) | Clean, pure |
| SUSTAIN (1) | DUAL (2) | Rich, stable |
| ATTACK (2) | CHAIN (0) | Bright, punchy |
| DECAY (3) | SPLIT (3) | Complex decay |
| CHAOS (4) | FEEDBACK (7) | Harsh, unstable |
| MORPH (5) | Blend A<->B | Transitioning |

---

## 8. CCS Pipeline Integration

### 8.1 Oscillator Modes

```python
OSC_MODE_A = 0            # Coefficient LUT
OSC_MODE_B = 1            # Rotator/Harmonic Bank
OSC_MODE_SAMPLE = 2       # Sample playback
OSC_MODE_FM6 = 3          # 6-operator FM (lane-rate)
OSC_MODE_FM6_HQ = 4       # 6-operator FM (sample-rate HQ)
OSC_MODE_FM6_CCS_BRAID = 5  # FM6 + braid fabric (see WSE-STD-BRAID)
```

### 8.2 Pipeline Flow

```
FM6 Operators -> Coefficients -> Operators -> IDWPT -> Audio
     ^                              |
     +---- feedback (z^-1) ---------+
```

The FM6 engine generates coefficients that then pass through the standard CCS operator chain (TILT, COMPRESS, etc.) before IDWPT reconstruction.

### 8.3 FWPT Adapter (HQ Mode)

For HQ mode to remain CCS-compatible:

```python
def fm6_hq_with_coeff_output(state, alg, controls, need_coeff: bool = True):
    """HQ render with optional coefficient output."""
    # Generate HQ audio
    audio_block = fm6_hq_render_block(state, alg, controls)

    if need_coeff:
        # Project to coefficient space for CCS pipeline
        coeff_frame = fwpt_analyze_block(audio_block)
        return audio_block, coeff_frame
    else:
        return audio_block, None
```

### 8.4 VoiceFrame Parameters

```python
# Added to VoiceFrame for FM6 mode:
fm_algorithm_id: int = 1              # DX7 algorithm (1-32)
fm_op_ratios: List[int] = [65536]*6   # Frequency ratios (Q16)
fm_op_levels: List[int] = [32767]*6   # Output levels (Q15)
fm_feedback_depth: int = 0            # Feedback level (0-7)
fm_mode: int = OSC_MODE_FM6           # FM6 or FM6_HQ
```

### 8.5 Compute Budget / Degradation

| Tier | Description | Operations |
|------|-------------|------------|
| 0 (FULL) | HQ sample-rate FM6 + FWPT | ~3000 mults |
| 1 (MID) | Lane-rate FM6, full envelope | ~1500 mults |
| 2 (MIN) | Lane-rate FM6, block-rate env | ~800 mults |
| 3 (EMERGENCY) | Single carrier only | ~200 mults |

Degradation rules (deterministic):
1. If budget insufficient for HQ -> fall back to lane-rate
2. If still tight -> reduce envelope rate to block-rate
3. If emergency -> single carrier, disable modulators

---

## 9. Verification Requirements

### 9.1 Algorithm Topology Tests

For each algorithm 1-32:
1. Verify modulation_matrix matches DX7 specification
2. Verify carrier_mask matches DX7 specification
3. Verify eval_order is valid topological sort
4. Verify no cycles exist (except single feedback edge)

### 9.2 Spectral Equivalence Tests

Compare output spectrum against DX7 reference for:
- Algorithm 1: Bell-like timbres
- Algorithm 5: E.Piano patches
- Algorithm 22: Brass patches
- Algorithm 32: Additive organ

Tolerance: Fundamental and first 16 harmonics within +/-3 dB.

### 9.3 Feedback Stability Tests

For each feedback level 0-7:
1. Run 1000 blocks
2. Verify output RMS remains bounded
3. Verify no NaN or overflow

### 9.4 Phase Continuity Test (HQ)

- For a single operator (no mods), verify sample-to-sample continuity
- For FM2 and FM6 graphs, ensure no discontinuities at block boundaries

### 9.5 Determinism Test

```python
def test_determinism():
    """Same inputs must produce identical output."""
    state1 = create_fm6_state(alg_id=1)
    state2 = create_fm6_state(alg_id=1)

    for _ in range(100):
        out1 = fm6_generate_block(state1, base_phase_inc)
        out2 = fm6_generate_block(state2, base_phase_inc)
        assert np.array_equal(out1, out2)
```

---

## Appendix A: Full Algorithm Table

```python
ALGORITHMS = {
    1: FMAlgorithm(
        alg_id=1,
        name="6-Op Chain",
        modulation_matrix=np.array([
            [0,0,0,0,0,0],  # Op1: no modulators (except self-fb)
            [1,0,0,0,0,0],  # Op2: modulates Op1
            [0,1,0,0,0,0],  # Op3: modulates Op2
            [0,0,1,0,0,0],  # Op4: modulates Op3
            [0,0,0,1,0,0],  # Op5: modulates Op4
            [0,0,0,0,1,0],  # Op6: modulates Op5
        ], dtype=bool),
        carrier_mask=0b000001,  # Op1 is carrier
        feedback_op=0,          # Op1 has feedback
        eval_order=[5,4,3,2,1,0],
    ),
    # ... (remaining 31 algorithms follow same pattern)
}
```

---

## Appendix B: Archetype Visual Diagrams

### A.0 CHAIN
```
[Op6] -> [Op5] -> [Op4] -> [Op3] -> [Op2] -> [Op1*]
                                              |fb
```

### A.1 STACK
```
[Op6] -+
[Op5] -+
[Op4] -+---> [Op1*]
[Op3] -+      |fb
[Op2] -+
```

### A.2 DUAL
```
[Op6] -> [Op5] -> [Op4*]
[Op3] -> [Op2] -> [Op1*]
```

### A.3 SPLIT
```
              +---> [Op4*]
[Op6] -> [Op5] +---> [Op3*]
              +---> [Op2*]
              +---> [Op1*]
```

### A.4 YPAIR
```
        +---> [Op5] --+
[Op6] --+             +---> [Op3*]
        +---> [Op4] --+
              [Op2] -> [Op1*]
```

### A.5 CLOUD
```
[Op6] ===+
[Op5] ===+
[Op4] ===+===> [Op1*]   (weighted parallel)
[Op3] ===+
[Op2] ===+
```

### A.6 ADDITIVE
```
[Op6*]  [Op5*]  [Op4*]  [Op3*]  [Op2*]  [Op1*]
  |       |       |       |       |       |
  +-------+-------+-------+-------+-------+
                      |
                   OUTPUT
```

### A.7 FEEDBACK
```
[Op6] -> [Op5] -> [Op4] -> [Op3] -> [Op2] -> [Op1*]
                                              ||2x
```

---

## Appendix C: Comparison with Knot Theory

### C.1 When Knot Theory Would Be Relevant

Knot theory becomes meaningful for audio routing if:
1. Multiple interlinked feedback loops exist simultaneously
2. Routing has genuine 3D structure (cables crossing in physical space)
3. Analyzing topological invariants provides musical insight

### C.2 Why It Doesn't Apply Here

DX7 algorithms have:
- At most ONE feedback loop per algorithm
- No crossing edges (planar embedding always exists)
- No linking between separate cycles

The mathematical structure is simply: **DAG + optional self-loop**.

### C.3 Graph Invariants That ARE Useful

| Invariant | Musical Meaning |
|-----------|-----------------|
| **Path length** | Modulation depth (harmonic complexity) |
| **In-degree** | Number of modulators (spectral density) |
| **Carrier count** | Number of independent timbres summed |
| **Has feedback** | Self-modulation capability |

These are all standard graph properties computable in O(V+E) time.

---

**Document Version:** 1.0
**Last Updated:** 2026-01-13
**Author:** CCS Development Team
**Status:** DRAFT
