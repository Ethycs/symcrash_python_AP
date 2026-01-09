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

**Document Version:** v0.1
**Last Updated:** 2026-01-09
**Author:** CCS Development Team
