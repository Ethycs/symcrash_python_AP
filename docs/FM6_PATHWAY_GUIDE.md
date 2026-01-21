# FM6 Pathway Guide

**Standard:** WSE-STD-FM6 v1.0
**Status:** DRAFT
**Depends on:** WSE-STD-001 (Fixed-Point), WSE-STD-210 (SineCore, Phase Model)

---

## Overview

FM6 implements DX7-compatible 6-operator FM synthesis within the CCS architecture. All 32 classic algorithms are supported with topological evaluation order, z^-1 block-delayed feedback, and Q15/Q32 fixed-point arithmetic.

**Design Invariants:**
1. **Graph, Not Knot** - Algorithm topologies are DAGs with at most one feedback edge
2. **Topological Evaluation** - Modulators compute before carriers
3. **Lane-Rate or Sample-Rate** - FM modulation at K=8 lane rate or 64-sample HQ mode
4. **DX7 Compatibility** - All 32 original algorithms produce equivalent spectra
5. **Deterministic Fixed-Point** - All operations per WSE-STD-001

---

## Architecture

```
VoiceFrame (fm_algorithm_id, fm_op_ratios, fm_op_levels, fm_feedback_depth)
    |
    v
executor.py (OSC_MODE_FM6 = 3)
    |
    v
fm6_engine.py::fm6_generate_coeffs()
    |
    +---> fm_algorithms.py::get_algorithm()  --> FMAlgorithm (topology)
    |
    +---> fm6_state.py::FM6State             --> 6 FMOperators + feedback buffer
    |
    v
fm6_generate_block()
    |
    +---> Topological evaluation (modulators before carriers)
    |     - Sum modulation from upstream operators
    |     - Apply self-feedback (z^-1 delayed)
    |     - Generate sine via sin_cos_q15()
    |     - Scale by level and envelope
    |
    +---> Sum carrier outputs
    |
    v
fm6_to_coefficients()
    |
    v
carrier_512[COEFF_LEN]  --> CCS Operators --> IDWPT --> Audio
```

## Files

| File | Purpose |
|------|---------|
| `vm/fm_algorithms.py` | 32 DX7 algorithm definitions with modulation matrices |
| `vm/fm6_state.py` | FMOperator, FM6State dataclasses |
| `vm/fm6_engine.py` | Block generation, coefficient conversion |
| `vm/state.py` | OSC_MODE_FM6 constant, lazy fm6_state property |
| `vm/executor.py` | Mode routing and VoiceFrame parameter extraction |
| `frames.py` | VoiceFrame FM6 parameters |

---

## Matrix Routing System

The core of FM6 is the **modulation matrix** - a 6x6 boolean array that defines operator connections.

### Modulation Matrix

```python
modulation_matrix: np.ndarray  # [6,6] bool: M[i,j]=True if Op_i modulates Op_j
```

**Reading the matrix:** Row `i`, column `j` being `True` means Operator `i` modulates Operator `j`.

**Example - Algorithm 1 (6-Op Chain):**
```
Matrix M[src, dst]:
         Op1  Op2  Op3  Op4  Op5  Op6  (destinations)
    Op1   .    .    .    .    .    .
    Op2   X    .    .    .    .    .   <- Op2 modulates Op1
    Op3   .    X    .    .    .    .   <- Op3 modulates Op2
    Op4   .    .    X    .    .    .   <- Op4 modulates Op3
    Op5   .    .    .    X    .    .   <- Op5 modulates Op4
    Op6   .    .    .    .    X    .   <- Op6 modulates Op5
(sources)

Signal flow: [Op6] -> [Op5] -> [Op4] -> [Op3] -> [Op2] -> [Op1*] -> OUT
                                                          ^feedback
```

### Carrier Mask

A 6-bit bitmask indicating which operators contribute to output:

```python
carrier_mask: int  # Bitmask: bit i set if Op_i is a carrier

# Example: carrier_mask = 0b000001 means only Op1 is a carrier
# Example: carrier_mask = 0b111111 means all 6 operators are carriers (additive)
```

### Evaluation Order

Operators are evaluated in **topological order** - all modulators compute before their targets:

```python
eval_order: List[int]  # Topological sort order

# Algorithm 1: eval_order = [5, 4, 3, 2, 1, 0]
# Meaning: Op6 first, then Op5, ..., finally Op1 (the carrier)
```

### FMAlgorithm Data Structure

```python
@dataclass
class FMAlgorithm:
    alg_id: int                      # 1-32 (DX7 algorithm number)
    name: str                        # Human-readable name
    modulation_matrix: np.ndarray    # [6,6] bool - routing matrix
    carrier_mask: int                # Bitmask of carrier operators
    feedback_op: int                 # Which op has self-feedback (-1 if none)
    eval_order: List[int]            # Topological sort (modulators first)

    def get_modulators(self, op_idx: int) -> List[int]:
        """Get list of operators that modulate the given operator."""

    def get_carriers(self) -> List[int]:
        """Get list of carrier operator indices."""

    def is_carrier(self, op_idx: int) -> bool:
        """Check if operator is a carrier (contributes to output)."""

    def has_feedback(self, op_idx: int) -> bool:
        """Check if operator has self-feedback."""
```

---

## All 32 DX7 Algorithms

### Chain Family (Deep Modulation)

**Algorithm 1: 6-Op Chain**
```
[Op6] -> [Op5] -> [Op4] -> [Op3] -> [Op2] -> [Op1*]
                                              ^fb
Carriers: {1}  |  Feedback: Op1  |  Eval: [5,4,3,2,1,0]
```

**Algorithm 2: 5-Op Chain + FB on Op2**
```
[Op6] -> [Op5] -> [Op4] -> [Op3] -> [Op2] -> [Op1*]
                                    ^fb
Carriers: {1}  |  Feedback: Op2  |  Eval: [5,4,3,2,1,0]
```

### Dual Chain Family

**Algorithm 3: 4-Op + 2-Op Chains**
```
[Op6] -> [Op5] -> [Op4] -> [Op3*]
              [Op2] -> [Op1*]
               ^fb
Carriers: {1,3}  |  Feedback: Op2  |  Eval: [5,4,3,1,2,0]
```

**Algorithm 4: 3-Op + 3-Op Chains**
```
[Op6] -> [Op5] -> [Op4*]
[Op3] -> [Op2] -> [Op1*]
                   ^fb
Carriers: {1,4}  |  Feedback: Op1  |  Eval: [5,4,2,1,3,0]
```

### E.Piano / Stack Family

**Algorithm 5: E.Piano (Dual Stack)**
```
[Op6] -> [Op5] -> [Op4] --+
 ^fb                      |
[Op3] -> [Op2] ---------->+--> [Op1*]
```
Carriers: {1}  |  Feedback: Op6  |  Eval: [5,4,2,1,3,0]

**Algorithm 6: Parallel Stacks**
```
[Op6] --> [Op3] -> [Op2] --+
 ^fb  [Op5] -> [Op4] ------->+--> [Op1*]
```
Carriers: {1}  |  Feedback: Op6  |  Eval: [5,4,2,1,3,0]

### Split Modulator Family

**Algorithm 7: 3-Op Stack + 2 Carriers**
```
[Op6] -> [Op5] -> [Op4] -> [Op1*]
 ^fb              [Op3*]
                  [Op2*]
```
Carriers: {1,2,3}  |  Feedback: Op6  |  Eval: [5,4,3,2,1,0]

**Algorithm 8: Split Modulator**
```
[Op6] -> [Op5] -+-> [Op4*]
 ^fb            +-> [Op3*]
                    [Op2*]
                    [Op1*]
```
Carriers: {1,2,3,4}  |  Feedback: Op6  |  Eval: [5,4,3,2,1,0]

### Brass Family (Parallel Modulators)

**Algorithm 16: Brass (5 into 1)**
```
[Op6] -+
 ^fb   |
[Op5] -+
[Op4] -+--> [Op1*]
[Op3] -+
[Op2] -+
```
Carriers: {1}  |  Feedback: Op6  |  Eval: [5,4,3,2,1,0]

**Algorithm 22: Brass (Split 3)**
```
[Op6] -> [Op5] -+-> [Op4*]
 ^fb            +-> [Op3*]
                +-> [Op2*]
                    [Op1*]
```
Carriers: {1,2,3,4}  |  Feedback: Op6  |  Eval: [5,4,3,2,1,0]

### Additive Family

**Algorithm 32: Additive (6 Carriers)**
```
[Op6*]  [Op5*]  [Op4*]  [Op3*]  [Op2*]  [Op1*]
  |       |       |       |       |       |
  +-------+-------+-------+-------+-------+
                      |
                   OUTPUT
```
Carriers: {1,2,3,4,5,6}  |  Feedback: Op6  |  Eval: any

### Complete Algorithm Table

| Alg | Name | Carriers | Feedback | Character |
|-----|------|----------|----------|-----------|
| 1 | 6-Op Chain | {1} | Op1 | Metallic, deep modulation |
| 2 | 5-Op Chain | {1} | Op2 | Similar, different FB position |
| 3 | 4-Op + 2-Op | {1,3} | Op2 | Dual voice |
| 4 | 3-Op + 3-Op | {1,4} | Op1 | Balanced dual |
| 5 | E.Piano | {1} | Op6 | Classic DX7 E.Piano |
| 6 | Parallel Stacks | {1} | Op6 | Complex modulation |
| 7 | 3-Op + 2 Carriers | {1,2,3} | Op6 | Organ-like |
| 8 | Split Modulator | {1,2,3,4} | Op6 | Wide spread |
| 9 | Dual 3-Op | {1,4} | Op3 | Detuned dual |
| 10 | Three 2-Op Pairs | {1,3,5} | Op4 | Organ drawbars |
| 11 | Three 2-Op (FB6) | {1,3,5} | Op6 | Variation |
| 12 | 4+2 Split | {1,3} | Op6 | Complex dual |
| 13 | 3+3 Arrangement | {1,4} | Op6 | Balanced |
| 14 | 4-Op + 2 Carriers | {1,2,3} | Op6 | Thick |
| 15 | 2-Op + 4 Carriers | {1,3,4,5} | Op6 | Mostly additive |
| 16 | Brass (5 into 1) | {1} | Op6 | Aggressive brass |
| 17 | 5 Modulators | {1} | Op6 | Similar to 16 |
| 18 | 3 Stacks into 1 | {1} | Op6 | Thick brass |
| 19 | Split to 3 | {1,2,3,4} | Op6 | Organ/brass |
| 20 | 2+2 into 2 | {1,2,4} | Op4 | Complex |
| 21 | 2 Mod Pairs | {1,2,3,4} | Op6 | Wide |
| 22 | Brass (Split 3) | {1,2,3,4} | Op6 | Classic brass |
| 23-25 | 1 Mod + 5 Carriers | {1-5} | Op6 | Near-additive |
| 26-27 | 2 Mods + 4 Carriers | varies | varies | Mixed |
| 28 | 3-Op + 3 Carriers | {1,2,3,4} | Op6 | Organ-like |
| 29 | 2+2+2 Pairs | {1,2,3,5} | Op6 | Drawbar organ |
| 30 | 2x 3-Op Chains | {1,4} | Op6 | Dual timbre |
| 31 | 1 Mod + 5 Carriers | {1-5} | Op6 | Near-additive |
| 32 | Additive | {1-6} | Op6 | Pure additive |

---

## Block Generation Algorithm

### Lane-Rate Evaluation

```python
def fm6_generate_block(state: FM6State, base_phase_inc: int) -> np.ndarray:
    alg = get_algorithm(state.algorithm_id)

    # Step 1: Evaluate operators in TOPOLOGICAL ORDER
    for op_idx in alg.eval_order:
        op = state.operators[op_idx]

        # Compute phase increment with frequency ratio
        op_phase_inc = (base_phase_inc * op.ratio_q16) >> 16

        # Step 2: Sum modulation from all UPSTREAM operators (matrix routing)
        mod_sum = np.zeros(K, dtype=np.int32)
        for mod_idx in alg.get_modulators(op_idx):  # <-- MATRIX LOOKUP
            mod_sum += state.op_outputs[mod_idx]

        # Step 3: Apply self-feedback (z^-1 block delay)
        if alg.feedback_op == op_idx and op.feedback_q15 > 0:
            mod_sum += (state.feedback_buffer * op.feedback_q15) >> 15

        # Step 4: Generate samples for each k-slot
        for k in range(K):
            op.phase = (op.phase + op_phase_inc) & Q32_MASK

            # Route modulation to phase
            mod_phase = (mod_sum[k] << 16) & Q32_MASK
            effective_phase = (op.phase + mod_phase) & Q32_MASK

            sin_val, _ = sin_cos_q15(effective_phase)
            output = (sin_val * op.level_q15 * op.env_value_q15) >> 30
            state.op_outputs[op_idx, k] = saturate(output)

    # Step 5: Store feedback for next block
    if alg.feedback_op >= 0:
        state.feedback_buffer[:] = state.op_outputs[alg.feedback_op]

    # Step 6: Sum CARRIER outputs (carrier mask routing)
    output_k = np.zeros(K, dtype=np.int32)
    for c in alg.get_carriers():  # <-- CARRIER MASK LOOKUP
        output_k += state.op_outputs[c]

    return output_k
```

### Routing Flow Diagram

```
                    MODULATION MATRIX
                    ┌─────────────────┐
                    │  [6,6] bool     │
                    │  M[src,dst]     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         v                   v                   v
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │  Op 6   │───────>│  Op 5   │───────>│  Op 4   │
    │(phase)  │ mod    │(phase)  │ mod    │(phase)  │
    │(level)  │        │(level)  │        │(level)  │
    └─────────┘        └─────────┘        └─────────┘
                                               │ mod
                                               v
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │  Op 3   │<───────│  Op 2   │<───────│  Op 1   │<─┐ fb
    │(phase)  │ mod    │(phase)  │ mod    │(phase)  │──┘
    │(level)  │        │(level)  │        │(level)  │
    └─────────┘        └─────────┘        └────┬────┘
                                               │
                                          CARRIER MASK
                                          (0b000001)
                                               │
                                               v
                                           OUTPUT
```

---

## Operator State

### FMOperator

```python
@dataclass
class FMOperator:
    op_id: int = 0                    # 0-5 (operator index)
    ratio_q16: int = 65536            # Frequency ratio (Q16: 65536 = 1.0x)
    level_q15: int = 32767            # Output level (Q15)
    feedback_q15: int = 0             # Self-modulation depth (Q15)
    phase: int = 0                    # Q32 phase accumulator
    env_value_q15: int = 32767        # Envelope value (Q15)
```

### FM6State

```python
@dataclass
class FM6State:
    operators: List[FMOperator]       # Always 6 operators
    algorithm_id: int = 1             # 1-32 (DX7 algorithm)
    op_outputs: np.ndarray            # [6, K] intermediate storage
    feedback_buffer: np.ndarray       # [K] z^-1 delay buffer
    master_level_q15: int = 32767     # Output scaling
```

---

## Fixed-Point Arithmetic

> **See also:** [MACROS-AND-CONTROLS.md](standards/library/MACROS-AND-CONTROLS.md) for complete fixed-point value types (Q15U, I16, U16, Q15S, Q16.16) and the normative [WSE-STD-001](standards/core/WSE-STD-001.md) fixed-point specification.

| Type | Range | Usage |
|------|-------|-------|
| Q15 | -32768 to 32767 | Levels, sine output, feedback |
| Q16 | 0 to 65536+ | Frequency ratios (65536 = 1.0) |
| Q32 | 0 to 2^32-1 | Phase accumulators |

### Key Operations

```python
# Frequency ratio application (Q16)
phase_inc = (base_phase_inc * ratio_q16) >> 16

# Level scaling (Q15 * Q15 -> Q15)
output = (sin_val * level_q15) >> 15

# Modulation to phase (Q15 -> Q32)
mod_phase = (mod_q15 << 16) & Q32_MASK

# Round-to-nearest-even right shift
def _rshift_rne(val, shift):
    half = 1 << (shift - 1)
    result = (val + half) >> shift
    if (val & ((1 << shift) - 1)) == half:
        result = result & ~1  # Round to even on tie
    return result
```

---

## Feedback System

### DX7 Feedback Mapping

DX7 feedback levels 0-7 map to Q15:

| Level | Q15 Value | Approx Beta |
|-------|-----------|-------------|
| 0 | 0 | 0 |
| 1 | 4681 | 0.143 |
| 2 | 9362 | 0.286 |
| 3 | 14043 | 0.429 |
| 4 | 18724 | 0.571 |
| 5 | 23405 | 0.714 |
| 6 | 28086 | 0.857 |
| 7 | 32767 | 1.0 |

Level 7 (beta ~ 1.0) is well below the chaotic threshold (pi), ensuring stability.

### Feedback Implementation

```python
# Feedback uses z^-1 (one block delay) for determinism
if alg.feedback_op == op_idx and op.feedback_q15 > 0:
    feedback_scaled = (state.feedback_buffer * op.feedback_q15) >> 15
    mod_sum += feedback_scaled

# Store current output for next block
state.feedback_buffer[:] = state.op_outputs[alg.feedback_op]
```

---

## Topology Morphing (8 Archetypes)

The 32 DX7 algorithms cluster into 8 structural families:

| ID | Name | Structure | Character |
|----|------|-----------|-----------|
| 0 | CHAIN | [6]->[5]->[4]->[3]->[2]->[1*] | Deep modulation, metallic |
| 1 | STACK | [6,5,4,3,2]->[1*] | Bright, aggressive |
| 2 | DUAL | [6]->[5]->[4*], [3]->[2]->[1*] | Dual voice, detuned |
| 3 | SPLIT | [6]->[5]->[4*,3*,2*,1*] | Organ-like, spread |
| 4 | YPAIR | [6]->[5,4]->[3*], [2]->[1*] | Y-branch, complex |
| 5 | CLOUD | [6,5,4,3,2]->[1*] parallel | Thick, chorused |
| 6 | ADDITIVE | [1*,2*,3*,4*,5*,6*] | Pure additive |
| 7 | FEEDBACK | Chain with enhanced FB | Harsh, distorted |

### Archetype Edge Weights

Each archetype uses weighted edges (Q15) for smooth morphing:

```python
# CHAIN archetype edges (src, dst, weight_q15)
CHAIN_EDGES = [
    (5, 4, 32767),  # Op6 -> Op5 (full weight)
    (4, 3, 32767),  # Op5 -> Op4
    (3, 2, 32767),  # Op4 -> Op3
    (2, 1, 32767),  # Op3 -> Op2
    (1, 0, 32767),  # Op2 -> Op1
]

# CLOUD archetype edges (parallel weighted)
CLOUD_EDGES = [
    (5, 0, 32767),  # Op6 -> Op1 (full)
    (4, 0, 24576),  # Op5 -> Op1 (0.75)
    (3, 0, 16384),  # Op4 -> Op1 (0.5)
    (2, 0, 16384),  # Op3 -> Op1 (0.5)
    (1, 0, 24576),  # Op2 -> Op1 (0.75)
]
```

### Blend Interpolation

```python
def blend_edge_weight(edge_a_q15: int, edge_b_q15: int, blend_q15: int) -> int:
    """Interpolate edge weight between A and B topologies."""
    inv_blend = 32767 - blend_q15
    return (edge_a_q15 * inv_blend + edge_b_q15 * blend_q15) >> 15
```

---

## CCS Pipeline Integration

### Oscillator Modes

```python
OSC_MODE_A = 0            # Coefficient LUT
OSC_MODE_B = 1            # Rotator/Harmonic Bank
OSC_MODE_SAMPLE = 2       # Sample playback
OSC_MODE_FM6 = 3          # 6-operator FM (lane-rate)
OSC_MODE_FM6_HQ = 4       # 6-operator FM (sample-rate HQ)
```

### Pipeline Flow

```
FM6 Operators -> Coefficients -> CCS Operators -> IDWPT -> Audio
     ^                              |
     +---- feedback (z^-1) ---------+
```

FM6 output flows through the same coefficient operators as Mode A/B, allowing spectral shaping of FM tones.

### VoiceFrame Parameters

```python
frame = VoiceFrame(
    note=60,
    coeff_osc_mode=OSC_MODE_FM6,  # 3
    fm_algorithm_id=5,            # DX7 algorithm 1-32
    fm_op_ratios=[65536, 65536, 131072, 65536, 65536, 65536],  # Q16
    fm_op_levels=[32767, 16384, 32767, 0, 0, 0],               # Q15
    fm_feedback_depth=3,          # DX7 level 0-7
)
```

---

## Preset Configurations

### Simple 2-Op FM Bass

```python
frame = VoiceFrame(
    coeff_osc_mode=OSC_MODE_FM6,
    fm_algorithm_id=1,  # Chain
    fm_op_ratios=[65536, 65536, 0, 0, 0, 0],  # 1:1 ratio
    fm_op_levels=[32767, 24576, 0, 0, 0, 0],  # Carrier full, mod 75%
    fm_feedback_depth=0,
)
```

### Bright FM Lead

```python
frame = VoiceFrame(
    coeff_osc_mode=OSC_MODE_FM6,
    fm_algorithm_id=1,
    fm_op_ratios=[65536, 131072, 196608, 0, 0, 0],  # 1:2:3
    fm_op_levels=[32767, 20000, 10000, 0, 0, 0],
    fm_feedback_depth=2,
)
```

### Additive Organ

```python
frame = VoiceFrame(
    coeff_osc_mode=OSC_MODE_FM6,
    fm_algorithm_id=32,  # All carriers
    fm_op_ratios=[65536, 131072, 196608, 262144, 327680, 393216],  # 1-6x
    fm_op_levels=[32767, 24576, 16384, 12288, 8192, 4096],  # Decay
    fm_feedback_depth=0,
)
```

### Classic E.Piano

```python
state = FM6State.create()
configure_electric_piano(state)
# - Algorithm 5 (dual stack)
# - Op2 ratio 14x for bell-like attack
```

### DX7 Brass

```python
state = FM6State.create()
configure_brass(state)
# - Algorithm 22 (split to 3 carriers)
# - Slight carrier detune for thickness
# - 2:1 modulator ratio for brass edge
```

---

## Compute Budget / Degradation

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

## FPGA Resource Estimate

| Resource | Count | Notes |
|----------|-------|-------|
| Multipliers | 18 per voice | 6 ops x 3 mults (ratio, level, mod) |
| Phase accumulators | 6 per voice | 32-bit each |
| Op outputs | 48 bytes/voice | 6 ops x K x 16-bit |
| Feedback buffer | 8 bytes/voice | K x 16-bit |
| Sine LUT | 4KB shared | 2048 entries |

With time-multiplexing at 48kHz block rate (750Hz lane rate), a single multiplier bank can serve all operators sequentially.

---

## Testing

```bash
pixi run python -m pytest tests/test_fm6_synthesis.py
```

Tests verify:
- All 32 algorithms defined with valid topology
- Modulation matrix matches DX7 specification
- Carrier mask matches DX7 specification
- Eval order is valid topological sort
- Single operator produces clean sine
- Modulation adds harmonics
- Feedback stable at all levels 0-7
- Presets produce audio
- Pitch tracking across MIDI range
- Determinism (same inputs = identical output)

---

## Known Issues

- 5 FM6 presets failed to compile due to a serialization issue with FM6 operator ratios exceeding int16 range (see CLAUDE.md)

---

## Relationship to Mode B FM

Mode B has a simpler FM feedback path:
- Single modulator extracted from coefficients
- Feedback after wavelet operators
- Used for carrier modulation in harmonic bank

FM6 is independent:
- 6 operators with internal feedback
- Feedback before coefficient conversion
- Used as direct FM synthesis engine

Both can coexist - FM6 for DX7-style sounds, Mode B for CCS-native FM.

---

**Document Version:** 1.1
**Last Updated:** 2026-01-19
**Author:** CCS Development Team
**Status:** DRAFT
