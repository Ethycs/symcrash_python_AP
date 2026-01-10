# WSE-STD-FM-TOPOLOGY-v0 — DX7-Style 6-Operator FM Algorithms

**Standard ID:** WSE-STD-FM-TOPOLOGY-v0
**Title:** 6-Operator FM Synthesis with Algorithm Topologies
**Status:** DRAFT
**Scope:** Defines the topology representation and evaluation model for DX7-compatible 6-operator FM synthesis.
**Depends on:** WSE-STD-210 (SineCore, Phase Model)

---

## 0. Design Invariants (MUST)

1. **Graph, Not Knot.** Algorithm topologies are represented as directed graphs with weighted edges, NOT knot-theoretic structures. DX7 algorithms are planar DAGs with at most one feedback edge.

2. **Topological Evaluation Order.** Operators MUST be evaluated in topological order: modulators compute before carriers. The single feedback path uses z⁻¹ block delay.

3. **Lane-Rate Modulation.** All FM modulation occurs at lane rate (K samples per block), consistent with WSE-STD-210 polyphase model.

4. **DX7 Compatibility.** All 32 original DX7 algorithms MUST be representable and produce equivalent spectra within implementation tolerances.

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
| **Computational cost** | O(n³+) | O(n²) adjacency |

**Key Insight:** A DX7 algorithm is a **DAG (Directed Acyclic Graph) with at most one marked feedback vertex**. This is topologically trivial and does not benefit from knot invariants.

### 1.2 Graph Representation

An FM algorithm is represented as:

```
G = (V, E, C, fb)
```

Where:
- `V = {Op1, Op2, ..., Op6}` — 6 operator vertices
- `E ⊆ V × V` — modulation edges (src modulates dst)
- `C ⊆ V` — carrier set (operators that sum to output)
- `fb ∈ V ∪ {∅}` — feedback operator (self-modulating)

### 1.3 Evaluation Order

Given graph G, compute topological order by:
1. Remove feedback edge (if present) to create DAG
2. Apply Kahn's algorithm or DFS-based sort
3. Modulators appear before their targets in order

**Invariant:** For any edge (u, v) in E (except feedback), u appears before v in eval_order.

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
    ratio_q16: int = 65536        # Frequency ratio (Q16: 65536 = 1.0×)
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

---

## 3. Algorithm Table (32 DX7 Algorithms)

### 3.1 Encoding Format

Each algorithm is encoded as:
- `mod_matrix`: 36-bit packed (6×6 matrix, row-major)
- `carrier_mask`: 6-bit mask
- `feedback_op`: 3-bit (0-5 or 7 for none)

### 3.2 Algorithm Definitions

#### Algorithm 1: 6-Op Chain
```
[6] → [5] → [4] → [3] → [2] → [1*] → OUT
                                ↺fb
```
- Carriers: {1}
- Modulators: 6→5, 5→4, 4→3, 3→2, 2→1
- Feedback: Op 1
- Eval order: [6, 5, 4, 3, 2, 1]

#### Algorithm 5: Dual Stack into Single Carrier
```
[6] → [5] → [4] ─┐
                 ├─→ [1] → OUT
[3] → [2] ───────┘
```
- Carriers: {1}
- Modulators: 6→5, 5→4, 4→1, 3→2, 2→1
- Feedback: Op 6 (typical)
- Eval order: [6, 5, 4, 3, 2, 1]

#### Algorithm 32: All Carriers (Additive)
```
[1] → OUT
[2] → OUT
[3] → OUT
[4] → OUT
[5] → OUT
[6] → OUT
```
- Carriers: {1, 2, 3, 4, 5, 6}
- Modulators: none
- Feedback: any (typically Op 6)
- Eval order: any

*(Full 32-algorithm table in Appendix A)*

---

## 4. Synthesis Algorithm

### 4.1 Per-Block Evaluation

```python
def fm6_generate_block(state: FM6State, base_phase_inc: int) -> np.ndarray:
    """
    Generate one block of FM audio.

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

    # Store feedback for next block (z⁻¹ delay)
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

### 4.2 Converting to Coefficient Format

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

## 5. Feedback Stability

### 5.1 Feedback Depth Limits

DX7-compatible feedback uses integer levels 0-7, mapped to Q15:

| DX7 Level | feedback_q15 | Approximate β |
|-----------|--------------|---------------|
| 0 | 0 | 0.0 |
| 1 | 4681 | ~0.143 |
| 2 | 9362 | ~0.286 |
| 3 | 14043 | ~0.429 |
| 4 | 18724 | ~0.571 |
| 5 | 23405 | ~0.714 |
| 6 | 28086 | ~0.857 |
| 7 | 32767 | ~1.0 |

### 5.2 Stability Analysis

For self-modulating operator with feedback depth β:
- `β < π`: Bounded, produces rich harmonics
- `β ≈ π`: Approaches maximum harmonic density
- `β > π`: Can become chaotic/noisy

DX7 level 7 (β ≈ 1.0) is well below the chaotic threshold, ensuring stability.

### 5.3 Implementation Requirement

Implementations MUST:
1. Limit feedback_q15 to range [0, 32767]
2. Use block-delay (z⁻¹) for feedback to ensure determinism
3. Apply saturation (SAT16) after feedback application

---

## 6. Integration with CCS Pipeline

### 6.1 Oscillator Mode

FM6 is integrated as oscillator mode 3:

```python
OSC_MODE_A = 0      # Coefficient LUT
OSC_MODE_B = 1      # Rotator/Harmonic Bank
OSC_MODE_SAMPLE = 2 # Sample playback
OSC_MODE_FM6 = 3    # 6-operator FM
```

### 6.2 Pipeline Flow

```
FM6 Operators → Coefficients → Operators → IDWPT → Audio
     ↑                              |
     └──── feedback (z⁻¹) ─────────┘
```

The FM6 engine generates coefficients that then pass through the standard CCS operator chain (TILT, COMPRESS, etc.) before IDWPT reconstruction.

### 6.3 VoiceFrame Parameters

```python
# Added to VoiceFrame for FM6 mode:
fm_algorithm_id: int = 1              # DX7 algorithm (1-32)
fm_op_ratios: List[int] = [65536]*6   # Frequency ratios (Q16)
fm_op_levels: List[int] = [32767]*6   # Output levels (Q15)
fm_feedback_depth: int = 0            # Feedback level (0-7)
```

---

## 7. Verification Requirements

### 7.1 Algorithm Topology Tests

For each algorithm 1-32:
1. Verify modulation_matrix matches DX7 specification
2. Verify carrier_mask matches DX7 specification
3. Verify eval_order is valid topological sort
4. Verify no cycles exist (except single feedback edge)

### 7.2 Spectral Equivalence Tests

Compare output spectrum against DX7 reference for:
- Algorithm 1: Bell-like timbres
- Algorithm 5: E.Piano patches
- Algorithm 22: Brass patches
- Algorithm 32: Additive organ

Tolerance: Fundamental and first 16 harmonics within ±3 dB.

### 7.3 Feedback Stability Tests

For each feedback level 0-7:
1. Run 1000 blocks
2. Verify output RMS remains bounded
3. Verify no NaN or overflow

---

## Appendix A: Full Algorithm Table

*(32 algorithms with modulation matrices, carrier masks, and eval orders)*

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
    # ... (remaining 31 algorithms)
}
```

---

## Appendix B: Comparison with Knot Theory

### B.1 When Knot Theory Would Be Relevant

Knot theory becomes meaningful for audio routing if:
1. Multiple interlinked feedback loops exist simultaneously
2. Routing has genuine 3D structure (cables crossing in physical space)
3. Analyzing topological invariants provides musical insight

### B.2 Why It Doesn't Apply Here

DX7 algorithms have:
- At most ONE feedback loop per algorithm
- No crossing edges (planar embedding always exists)
- No linking between separate cycles

The mathematical structure is simply: **DAG + optional self-loop**.

### B.3 Graph Invariants That ARE Useful

| Invariant | Musical Meaning |
|-----------|-----------------|
| **Path length** | Modulation depth (harmonic complexity) |
| **In-degree** | Number of modulators (spectral density) |
| **Carrier count** | Number of independent timbres summed |
| **Has feedback** | Self-modulation capability |

These are all standard graph properties computable in O(V+E) time.

---

## 8. Topology Morphing Extension

### 8.1 Overview

This section defines a continuous FM topology morphing system that factors the 32 DX7 algorithms into 8 canonical archetypes with A/B blend capability. CMMM can drive topology transitions via the blend parameter.

### 8.2 Canonical FM Archetypes

The 32 DX7 algorithms cluster into 8 structural families:

| ID | Name | Carriers | Structure | Character |
|----|------|----------|-----------|-----------|
| 0 | CHAIN | 1 | [6]→[5]→[4]→[3]→[2]→[1*] | Deep modulation, metallic |
| 1 | STACK | 1 | [6,5,4,3,2]→[1*] | Bright, aggressive |
| 2 | DUAL | 2 | [6]→[5]→[4*], [3]→[2]→[1*] | Dual voice, detuned |
| 3 | SPLIT | 3+ | [6]→[5]→[4*,3*,2*,1*] | Organ-like, spread |
| 4 | YPAIR | 2 | [6]→[5,4]→[3*], [2]→[1*] | Y-branch, complex |
| 5 | CLOUD | 1 | [6,5,4,3,2]→[1*] parallel | Thick, chorused |
| 6 | ADDITIVE | 6 | [6*,5*,4*,3*,2*,1*] | Pure additive |
| 7 | FEEDBACK | 1 | Chain with enhanced FB | Harsh, distorted |

### 8.3 Archetype Edge Lists

Each archetype is defined by a canonical edge list `(src, dst, weight_q15)`:

```python
ARCHETYPE_EDGES = {
    # 0: CHAIN - Serial 6-op chain
    0: {
        'edges': [
            (5, 4, 32767),  # Op6 → Op5
            (4, 3, 32767),  # Op5 → Op4
            (3, 2, 32767),  # Op4 → Op3
            (2, 1, 32767),  # Op3 → Op2
            (1, 0, 32767),  # Op2 → Op1
        ],
        'carrier_mask': 0b000001,
        'feedback_op': 0,
    },

    # 1: STACK - 5 modulators into 1 carrier
    1: {
        'edges': [
            (5, 0, 32767),  # Op6 → Op1
            (4, 0, 32767),  # Op5 → Op1
            (3, 0, 32767),  # Op4 → Op1
            (2, 0, 32767),  # Op3 → Op1
            (1, 0, 32767),  # Op2 → Op1
        ],
        'carrier_mask': 0b000001,
        'feedback_op': 5,
    },

    # 2: DUAL - Two parallel 3-op chains
    2: {
        'edges': [
            (5, 4, 32767),  # Op6 → Op5
            (4, 3, 32767),  # Op5 → Op4
            (2, 1, 32767),  # Op3 → Op2
            (1, 0, 32767),  # Op2 → Op1
        ],
        'carrier_mask': 0b001001,  # Op1, Op4
        'feedback_op': 5,
    },

    # 3: SPLIT - Modulator tree to multiple carriers
    3: {
        'edges': [
            (5, 4, 32767),  # Op6 → Op5
            (4, 3, 32767),  # Op5 → Op4
            (4, 2, 32767),  # Op5 → Op3
            (4, 1, 32767),  # Op5 → Op2
        ],
        'carrier_mask': 0b001111,  # Op1-4
        'feedback_op': 5,
    },

    # 4: YPAIR - Y-branch structure
    4: {
        'edges': [
            (5, 4, 32767),  # Op6 → Op5
            (5, 3, 32767),  # Op6 → Op4
            (4, 2, 32767),  # Op5 → Op3
            (3, 2, 32767),  # Op4 → Op3
            (1, 0, 32767),  # Op2 → Op1
        ],
        'carrier_mask': 0b000101,  # Op1, Op3
        'feedback_op': 5,
    },

    # 5: CLOUD - Parallel modulators (thick)
    5: {
        'edges': [
            (5, 0, 32767),  # Op6 → Op1
            (4, 0, 24576),  # Op5 → Op1 (slightly reduced)
            (3, 0, 16384),  # Op4 → Op1
            (2, 0, 16384),  # Op3 → Op1
            (1, 0, 24576),  # Op2 → Op1
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

### 8.4 DX7 → Archetype Mapping

Each DX7 algorithm maps to its closest archetype:

| DX7 Alg | Archetype | Notes |
|---------|-----------|-------|
| 1, 2 | CHAIN (0) | Serial chain variants |
| 3, 4, 9, 10, 11, 12, 13, 30 | DUAL (2) | Parallel chain variants |
| 5, 6, 20 | YPAIR (4) | Y-branch structures |
| 7, 8, 14, 15, 19, 21, 22, 28, 29 | SPLIT (3) | Multi-carrier splits |
| 16, 17 | STACK (1) | All-to-one modulation |
| 18 | CLOUD (5) | Thick parallel |
| 23, 24, 25, 26, 27, 31, 32 | ADDITIVE (6) | Mostly/all carriers |

```python
DX7_TO_ARCHETYPE = {
    1: 0, 2: 0,                           # CHAIN
    3: 2, 4: 2, 9: 2, 10: 2, 11: 2,        # DUAL
    12: 2, 13: 2, 30: 2,
    5: 4, 6: 4, 20: 4,                     # YPAIR
    7: 3, 8: 3, 14: 3, 15: 3, 19: 3,       # SPLIT
    21: 3, 22: 3, 28: 3, 29: 3,
    16: 1, 17: 1,                          # STACK
    18: 5,                                  # CLOUD
    23: 6, 24: 6, 25: 6, 26: 6,            # ADDITIVE
    27: 6, 31: 6, 32: 6,
}
```

### 8.5 Morphing Parameters

| Parameter | ID | Type | Range | Default | Description |
|-----------|-----|------|-------|---------|-------------|
| `fm.topo_a_id` | 0x40 | u8 | 0-7 | 0 | Source archetype |
| `fm.topo_b_id` | 0x41 | u8 | 0-7 | 0 | Target archetype |
| `fm.topo_blend_q15` | 0x42 | i16 | 0-32767 | 0 | A/B blend (0=A, 32767=B) |
| `fm.dx7_compat_id` | 0x43 | u8 | 1-32 | 0 | DX7 algorithm (overrides topo_a when nonzero) |

### 8.6 Blend Interpolation

#### 8.6.1 Edge Weight Interpolation

For each edge `(src, dst)`, compute blended weight:

```python
def blend_edge_weight(edge_a_q15: int, edge_b_q15: int, blend_q15: int) -> int:
    """Interpolate edge weight between A and B topologies."""
    # Linear interpolation: result = A*(1-blend) + B*blend
    Q15_ONE = 32767
    inv_blend = Q15_ONE - blend_q15
    result = (edge_a_q15 * inv_blend + edge_b_q15 * blend_q15) >> 15
    return result
```

#### 8.6.2 Carrier Mask Interpolation

Carrier membership is binary but can be smoothly faded:

```python
def blend_carrier_level(is_carrier_a: bool, is_carrier_b: bool,
                        level_q15: int, blend_q15: int) -> int:
    """Blend carrier output level based on topology transition."""
    Q15_ONE = 32767
    if is_carrier_a and is_carrier_b:
        return level_q15  # Carrier in both - full level
    elif is_carrier_a and not is_carrier_b:
        # Fading out as carrier
        return (level_q15 * (Q15_ONE - blend_q15)) >> 15
    elif not is_carrier_a and is_carrier_b:
        # Fading in as carrier
        return (level_q15 * blend_q15) >> 15
    else:
        return 0  # Carrier in neither
```

#### 8.6.3 Feedback Operator Transition

When feedback operator changes between topologies:

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

### 8.7 Blended Modulation Matrix

```python
def compute_blended_matrix(topo_a_id: int, topo_b_id: int,
                           blend_q15: int) -> np.ndarray:
    """Compute blended modulation matrix from two archetypes."""
    mat = np.zeros((6, 6), dtype=np.int16)

    edges_a = {(e[0], e[1]): e[2] for e in ARCHETYPE_EDGES[topo_a_id]['edges']}
    edges_b = {(e[0], e[1]): e[2] for e in ARCHETYPE_EDGES[topo_b_id]['edges']}

    # All possible edges from both topologies
    all_edges = set(edges_a.keys()) | set(edges_b.keys())

    for src, dst in all_edges:
        w_a = edges_a.get((src, dst), 0)
        w_b = edges_b.get((src, dst), 0)
        mat[src, dst] = blend_edge_weight(w_a, w_b, blend_q15)

    return mat
```

### 8.8 CMMM Integration

#### 8.8.1 Regime-Driven Topology

CMMM regime transitions can drive FM topology morphing:

| CMMM Regime | Suggested Archetype | Rationale |
|-------------|---------------------|-----------|
| IDLE (0) | ADDITIVE (6) | Clean, pure |
| SUSTAIN (1) | DUAL (2) | Rich, stable |
| ATTACK (2) | CHAIN (0) | Bright, punchy |
| DECAY (3) | SPLIT (3) | Complex decay |
| CHAOS (4) | FEEDBACK (7) | Harsh, unstable |
| MORPH (5) | Blend A↔B | Transitioning |

#### 8.8.2 Binding Targets

```python
CMMM_FM_TARGETS = {
    'fm.topo_blend_q15': 0x42,        # Primary morph control
    'fm.master_level_q15': 0x70,      # Output scaling
    'fm.feedback_global_q15': 0x71,   # Feedback amount
    'fm.op[0].level_q15': 0x58,       # Per-op level modulation
    'fm.op[1].level_q15': 0x59,
    'fm.op[2].level_q15': 0x5A,
    'fm.op[3].level_q15': 0x5B,
    'fm.op[4].level_q15': 0x5C,
    'fm.op[5].level_q15': 0x5D,
}
```

### 8.9 DX7 Compatibility Mode

When `fm.dx7_compat_id` is set (1-32):

1. Look up DX7 algorithm → archetype mapping
2. Set `topo_a_id` to matched archetype
3. Set `blend_q15 = 0` (pure A topology)
4. Load exact DX7 edge weights from `fm_algorithms.py`

This provides bit-exact DX7 compatibility while allowing morph-out transitions.

### 8.10 Archetype Visual Diagrams

#### A.0 CHAIN
```
[Op6] → [Op5] → [Op4] → [Op3] → [Op2] → [Op1*]
                                          ↺fb
```

#### A.1 STACK
```
[Op6] ─┐
[Op5] ─┤
[Op4] ─┼──→ [Op1*]
[Op3] ─┤     ↺fb
[Op2] ─┘
```

#### A.2 DUAL
```
[Op6] → [Op5] → [Op4*]
[Op3] → [Op2] → [Op1*]
```

#### A.3 SPLIT
```
              ┌──→ [Op4*]
[Op6] → [Op5] ┼──→ [Op3*]
              ├──→ [Op2*]
              └──→ [Op1*]
```

#### A.4 YPAIR
```
        ┌──→ [Op5] ──┐
[Op6] ──┤            ├──→ [Op3*]
        └──→ [Op4] ──┘
              [Op2] → [Op1*]
```

#### A.5 CLOUD
```
[Op6] ═══╗
[Op5] ═══╬═══→ [Op1*]   (weighted parallel)
[Op4] ═══╣
[Op3] ═══╣
[Op2] ═══╝
```

#### A.6 ADDITIVE
```
[Op6*]  [Op5*]  [Op4*]  [Op3*]  [Op2*]  [Op1*]
  │       │       │       │       │       │
  └───────┴───────┴───────┴───────┴───────┘
                      ↓
                    OUTPUT
```

#### A.7 FEEDBACK
```
[Op6] → [Op5] → [Op4] → [Op3] → [Op2] → [Op1*]
                                          ⟲2x
```

### 8.11 Example Morph Sequences

#### E.Piano → Brass Morph
```python
# Start: E.Piano (DX7 Alg 5 → YPAIR)
# End: Brass (DX7 Alg 16 → STACK)

frame.fm_topo_a_id = 4  # YPAIR
frame.fm_topo_b_id = 1  # STACK
frame.fm_topo_blend_q15 = cmmm_blend  # 0 → 32767 over transition
```

#### Additive → Chain Morph
```python
# Organ to Lead transition
frame.fm_topo_a_id = 6  # ADDITIVE
frame.fm_topo_b_id = 0  # CHAIN
frame.fm_topo_blend_q15 = attack_envelope  # Fast morph on attack
```

### 8.12 Resource Estimates

| Resource | Size | Notes |
|----------|------|-------|
| Archetype table | 320 bytes | 8 archetypes × 40 bytes |
| DX7 mapping | 32 bytes | 32 × 1 byte archetype ID |
| Blend state | 16 bytes | topo_a, topo_b, blend, cache |
| Per-voice | 256 bytes | Operators + outputs |

### 8.13 Conformance Requirements

#### 8.13.1 Required

- Support all 8 canonical archetypes
- Implement edge weight interpolation (§8.6.1)
- Implement carrier fade interpolation (§8.6.2)
- Support DX7 compatibility mode (1-32)
- Q15 blend parameter range [0, 32767]

#### 8.13.2 Optional

- Feedback operator crossfade (§8.6.3)
- CMMM binding targets (§8.8.2)
- Real-time archetype switching without glitches
- Custom archetype definition

---

**Document Version:** v0.2
**Last Updated:** 2026-01-10
**Author:** CCS Development Team
