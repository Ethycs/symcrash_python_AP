# FM6 Pathway Guide

## Overview

FM6 implements DX7-compatible 6-operator FM synthesis within the CCS architecture. All 32 classic algorithms are supported with topological evaluation order, z^-1 block-delayed feedback, and Q15/Q32 fixed-point arithmetic.

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
carrier_512[COEFF_LEN]  --> CCS operators --> IDWPT --> Audio
```

## Files

| File | Purpose |
|------|---------|
| `vm/fm_algorithms.py` | 32 DX7 algorithm definitions |
| `vm/fm6_state.py` | FMOperator, FM6State dataclasses |
| `vm/fm6_engine.py` | Block generation, coefficient conversion |
| `vm/state.py` | OSC_MODE_FM6 constant, lazy fm6_state property |
| `vm/executor.py` | Mode routing and VoiceFrame parameter extraction |
| `frames.py` | VoiceFrame FM6 parameters |

## Data Flow

### 1. VoiceFrame Parameters

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

### 2. Algorithm Topology (fm_algorithms.py)

Each algorithm is defined by:

```python
@dataclass
class FMAlgorithm:
    alg_id: int                      # 1-32
    name: str                        # e.g., "6-op chain"
    modulation_matrix: np.ndarray    # [6,6] bool - who modulates whom
    carrier_mask: int                # Bitmask of carrier operators
    feedback_op: int                 # Which op has self-modulation (-1 if none)
    eval_order: List[int]            # Topological sort (modulators first)
```

**Example - Algorithm 1 (6-op chain):**
```
[Op6] -> [Op5] -> [Op4] -> [Op3] -> [Op2] -> [Op1] -> OUT
                                              ^
                                              |
                                           feedback
```

- Carriers: {0} (op1)
- Modulators: {1,2,3,4,5} (op2-6)
- Feedback: op 0
- Eval order: [5,4,3,2,1,0]

### 3. Operator State (fm6_state.py)

```python
@dataclass
class FMOperator:
    op_id: int = 0                    # 0-5
    ratio_q16: int = 65536            # Frequency ratio (1.0 = 65536)
    level_q15: int = 32767            # Output level
    feedback_q15: int = 0             # Self-modulation depth
    phase: int = 0                    # Q32 phase accumulator
    env_value_q15: int = 32767        # Envelope (simplified)
```

```python
@dataclass
class FM6State:
    operators: List[FMOperator]       # Always 6
    algorithm_id: int = 1             # 1-32
    op_outputs: np.ndarray            # [6, VM_K] intermediate storage
    feedback_buffer: np.ndarray       # [VM_K] z^-1 delay
    master_level_q15: int = 32767     # Output scaling
```

### 4. Block Generation (fm6_engine.py)

```python
def fm6_generate_block(state: FM6State, base_phase_inc: int) -> np.ndarray:
    alg = get_algorithm(state.algorithm_id)

    # Evaluate in topological order (modulators before carriers)
    for op_idx in alg.eval_order:
        op = state.operators[op_idx]

        # Compute phase increment with ratio
        op_phase_inc = op.compute_phase_inc(base_phase_inc)

        # Sum modulation from all upstream operators
        mod_sum = sum(state.op_outputs[m] for m in alg.get_modulators(op_idx))

        # Add self-feedback (z^-1 delayed)
        if alg.feedback_op == op_idx and op.feedback_q15 > 0:
            mod_sum += state.feedback_buffer * op.feedback_q15 >> 15

        # Generate output for each k-slot
        for k in range(VM_K):
            op.phase = (op.phase + op_phase_inc) & Q32_MASK
            mod_phase = (mod_sum[k] << 16) & Q32_MASK
            effective_phase = (op.phase + mod_phase) & Q32_MASK

            sin_val, _ = sin_cos_q15(effective_phase)
            output = (sin_val * op.level_q15) >> 15
            state.op_outputs[op_idx, k] = output

    # Store feedback for next block
    if alg.feedback_op >= 0:
        state.feedback_buffer[:] = state.op_outputs[alg.feedback_op]

    # Sum carrier outputs
    output_k = sum(state.op_outputs[c] for c in alg.get_carriers())

    return output_k  # [VM_K] int16
```

### 5. Coefficient Conversion

FM6 output is replicated across all 64 leaves:

```python
def fm6_to_coefficients(output_k: np.ndarray) -> np.ndarray:
    coeffs = np.zeros(COEFF_LEN, dtype=np.int16)  # 64 * VM_K = 256

    for k in range(VM_K):
        for leaf in range(64):
            coeffs[leaf * VM_K + k] = output_k[k]

    return coeffs
```

This produces a "full-band" FM signal that the IDWPT reconstructs directly.

## Fixed-Point Arithmetic

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

## DX7 Feedback Mapping

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

Level 7 corresponds to beta ~ 1.0, which is well below the chaotic threshold (pi).

## Preset Configurations

```python
# Electric Piano (Algorithm 5)
configure_electric_piano(state)
# - Algorithm 5 (dual stack)
# - Op2 ratio 14x for bell-like attack

# Brass (Algorithm 22)
configure_brass(state)
# - Algorithm 22 (split to 3 carriers)
# - All ratios 1x for harmonic spectrum

# Bell (Algorithm 1)
configure_bell(state)
# - Algorithm 1 (6-op chain)
# - Inharmonic ratios (1.4, 2.0, 3.1, 4.4, 5.7)

# Organ (Algorithm 32)
configure_organ(state)
# - Algorithm 32 (all carriers - additive)
# - Harmonic ratios (1, 2, 3, 4, 5, 6)
```

## FPGA Resource Estimate

| Resource | Count | Notes |
|----------|-------|-------|
| Multipliers | 18 per voice | 6 ops x 3 mults (ratio, level, mod) |
| Phase accumulators | 6 per voice | 32-bit each |
| Op outputs | 48 bytes/voice | 6 ops x VM_K x 16-bit |
| Feedback buffer | 8 bytes/voice | VM_K x 16-bit |
| Sine LUT | 4KB shared | 2048 entries |

With time-multiplexing at 48kHz block rate (750Hz lane rate), a single multiplier bank can serve all operators sequentially.

## Integration with CCS Pipeline

FM6 integrates as an alternative coefficient source:

```
Mode A (LUT):      ccs_generate_coeffs_polyphase() -> carrier_512
Mode B (Rotator):  mode_b_generate_coeffs()        -> carrier_512
Mode FM6:          fm6_generate_coeffs()           -> carrier_512
                                                         |
                                                         v
                                              CCS Operators (TILT, COMPRESS, etc.)
                                                         |
                                                         v
                                                      IDWPT
                                                         |
                                                         v
                                                   Audio Output
```

FM6 output flows through the same coefficient operators as Mode A/B, allowing spectral shaping of FM tones.

## Testing

```bash
pixi run python tests/test_fm6_synthesis.py
```

Tests verify:
- All 32 algorithms defined with valid topology
- State creation and configuration
- Single operator produces clean sine
- Modulation adds harmonics
- Feedback stable at all levels 0-7
- Presets produce audio
- Executor integration
- Coefficient format correct
- Pitch tracking across MIDI range

## Common Patterns

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
