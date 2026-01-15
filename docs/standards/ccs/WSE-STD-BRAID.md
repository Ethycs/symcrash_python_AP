# WSE-STD-BRAID — Sparse Braid Network for Coefficient-Domain Modulation

**Standard ID:** WSE-STD-BRAID
**Title:** Sparse Braid Network (Givens Rotation Fabric) for CCS
**Version:** 1.0
**Status:** DRAFT
**Scope:** Defines coefficient-domain modulation via sparse 2x2 Givens rotations ("braid crossings") as an alternative/complement to traditional FM.
**Depends on:** WSE-STD-001 (Fixed-Point), WSE-STD-FM6 (FM Topology), WSE-STD-210 (CCS Mode B)
**Supersedes:** WSE-UPGRADE-BRAID-v0, WSE-UPGRADE-BRAID-FM6-v0

---

## 0. Executive Summary

This document defines a **coefficient-domain modulation architecture** based on sparse braid networks. Instead of traditional phase modulation (FM6), this method applies 2x2 Givens rotations directly to coefficient pairs.

**Key Properties:**
- FPGA-optimized: O(K) crossings instead of O(64xK) operations per block
- Streaming-friendly: Local 2x2 operations, no global memory access
- Energy-preserving: Unitary rotations maintain coefficient energy
- Novel sound: Different spectral character than traditional FM

**Primary Use Cases:**
- Coefficient-native FM6 integration (no IDWPT bypass)
- Post-FM spectral processing
- Hybrid FM6+Braid synthesis

---

## 1. Architectural Requirements

### R1 — No Bypass

When used with FM6, the combined system MUST NOT bypass IDWPT. All audio output MUST be produced via the standard reconstruction path:

```
coeff_vector -> IDWPT -> audio_block
```

### R2 — FM6 as CCS Mode

FM6+Braid MUST be implemented as a CCS mode that:
1. Generates (or receives) a coefficient vector
2. Applies braid modulation fabric
3. Passes to IDWPT

### R3 — Deterministic Fixed-Point

All braid operations MUST be deterministic across platforms:
- Fixed-point arithmetic (per WSE-STD-001)
- Defined rounding and saturation
- No denormals
- LUT-based trig or CORDIC with fixed iteration count

### R4 — Matched Transform Pair

If using forward WPT projection:
- The forward WPT MUST be paired and normalized consistently with IDWPT
- Round-trip error MUST be bounded and testable

---

## 2. Mathematical Foundation

### 2.1 Givens Rotation as Primitive

A 2x2 Givens rotation is:

```
G(theta) = [cos(theta)   sin(theta) ]
           [-sin(theta)  cos(theta)]
```

**Properties:**
- **Unitary:** G(theta)^T G(theta) = I (energy-preserving)
- **Invertible:** G(theta)^-1 = G(-theta)
- **Commutative composition:** G(theta1) G(theta2) = G(theta1 + theta2)
- **Hardware cost:** 4 multiplies + 2 adds (or 3 mults with lifting)

### 2.2 Braid Group Interpretation

The braid group Bn on n strands has generators sigma_i where sigma_i swaps strands i and i+1. In our coefficient context:

- **Strands** = coefficient values (64 leaves x K slots = 256-512 values)
- **Crossings** = Givens rotations between coefficient pairs
- **Braid word** = Sequence of crossings defining the algorithm

**Key insight:** A sparse set of crossings creates controlled spectral mixing.

### 2.3 Connection to FM Synthesis

Traditional FM: `sin(wc*t + beta*sin(wm*t))` creates sidebands at wc +/- n*wm

Braid rotation: Mixing coefficient pairs (leaf i, leaf j) with angle theta:
- Transfers energy between frequency bands i and j
- theta controls "modulation depth"
- Multiple crossings create compound spectral effects

**Not mathematically equivalent, but sonically related.**

### 2.4 Energy Conservation Proof

For unitary matrix U (like Givens rotation):
```
||Ux||^2 = (Ux)^T(Ux) = x^T U^T U x = x^T I x = ||x||^2
```

Therefore ||c'||^2 = ||c||^2 after any sequence of Givens rotations.

---

## 3. Crossing Primitives

### 3.1 Standard Crossing (4-multiply)

Default crossing SHALL be an orthonormal 2x2 rotation applied to lanes i, j:

```python
def braid_crossing_q15(
    c_i: int, c_j: int,
    sin_theta_q15: int, cos_theta_q15: int
) -> Tuple[int, int]:
    """
    Apply 2x2 Givens rotation to coefficient pair.

    Args:
        c_i, c_j: Input coefficients (Q1.15)
        sin_theta_q15, cos_theta_q15: Rotation angle (Q1.15)

    Returns:
        (c_i', c_j'): Rotated coefficients (Q1.15)
    """
    # Compute products in s32 (WSE-STD-001 Section 7)
    ci_cos = RSHIFT_RNE(c_i * cos_theta_q15, 15)
    ci_sin = RSHIFT_RNE(c_i * sin_theta_q15, 15)
    cj_cos = RSHIFT_RNE(c_j * cos_theta_q15, 15)
    cj_sin = RSHIFT_RNE(c_j * sin_theta_q15, 15)

    # Combine with saturation (WSE-STD-001 Section 5)
    c_i_prime = SAT16(ci_cos + cj_sin)
    c_j_prime = SAT16(cj_cos - ci_sin)

    return c_i_prime, c_j_prime
```

### 3.2 Lifting Crossing (3-multiply)

If DSP resources are tight, a crossing MAY be implemented as 2-3 shear/lifting steps:

```python
def braid_crossing_lifting(
    c_i: int, c_j: int,
    alpha_q15: int, beta_q15: int
) -> Tuple[int, int]:
    """
    Lifting-based crossing (3 mults instead of 4).

    Implements: G = [[1, alpha], [0, 1]] @ [[1, 0], [beta, 1]] @ [[1, alpha], [0, 1]]
    """
    # Lift step 1
    c_j_temp = SAT16(c_j + RSHIFT_RNE(c_i * alpha_q15, 15))
    # Lift step 2
    c_i_temp = SAT16(c_i + RSHIFT_RNE(c_j_temp * beta_q15, 15))
    # Lift step 3
    c_j_prime = SAT16(c_j_temp + RSHIFT_RNE(c_i_temp * alpha_q15, 15))

    return c_i_temp, c_j_prime
```

---

## 4. Braid Network Architecture

### 4.1 Network Structure

```
                     Coefficient Stream [256]
                              |
           +------------------+------------------+
           v                  v                  v
      +---------+        +---------+        +---------+
      |Crossing |        |Crossing |        |Crossing |
      |  (i,j)  |        |  (k,l)  |        |  (m,n)  |
      |   t1    |        |   t2    |        |   t3    |
      +----+----+        +----+----+        +----+----+
           |                  |                  |
           +------------------+------------------+
                              |
           +------------------+------------------+
           v                  v                  v
      +---------+        +---------+        +---------+
      |Crossing |        |Crossing |        |Crossing |
      |  (a,b)  |        |  (c,d)  |        |  (e,f)  |
      |   t4    |        |   t5    |        |   t6    |
      +----+----+        +----+----+        +----+----+
           |                  |                  |
           +------------------+------------------+
                              |
                              v
                     Output Coefficients [256]
```

### 4.2 Crossing Instruction

```python
@dataclass
class CrossInstruction:
    """Single crossing instruction in braid program."""

    lane_i: int              # First lane index (0..COEFF_LEN-1)
    lane_j: int              # Second lane index (0..COEFF_LEN-1)
    angle_src: AngleSource   # Source for angle parameter
    angle_param: int         # LUT index or depth scalar (Q1.15)
    flags: int               # Bitfield: swap, sign, clamp mode, bypass

class AngleSource(IntEnum):
    CONST_LUT = 0    # Fixed angle from LUT
    ENV = 1          # From envelope
    LFO = 2          # From LFO
    MODBUS = 3       # From modulation bus
    FB = 4           # From feedback path
    CONTROL = 5      # From control frame
```

### 4.3 Braid Program Format

```python
@dataclass
class BraidProgram:
    """Complete braid modulation program."""

    program_id: int                      # Unique identifier
    name: str                            # Human-readable name
    num_crossings: int                   # Number of crossings (K)
    crossings: List[CrossInstruction]    # Ordered crossing list
    num_layers: int                      # Pipeline depth hint
    flags: int                           # Program-level flags
```

### 4.4 Schedule Execution

Within a block, apply crossings in program order:

```python
def braid_fabric_apply(
    coeffs: np.ndarray,
    program: BraidProgram,
    angles_q15: np.ndarray,
) -> np.ndarray:
    """
    Execute braid program on coefficient vector.

    Args:
        coeffs: Input coefficients [COEFF_LEN] (Q1.15)
        program: Braid program with crossing schedule
        angles_q15: Current angle values [num_crossings] (Q1.15)

    Returns:
        coeffs': Transformed coefficients [COEFF_LEN] (Q1.15)
    """
    c = coeffs.copy()

    for k in range(program.num_crossings):
        cross = program.crossings[k]
        theta_q15 = angles_q15[k]

        # Get sin/cos from LUT (256 entries, Q1.15)
        sin_theta = SIN_LUT_Q15[theta_q15 >> 7]  # Top 9 bits as index
        cos_theta = COS_LUT_Q15[theta_q15 >> 7]

        # Apply crossing
        c[cross.lane_i], c[cross.lane_j] = braid_crossing_q15(
            c[cross.lane_i], c[cross.lane_j],
            sin_theta, cos_theta
        )

    return c
```

---

## 5. Schedule Design

### 5.1 Crossing Types

| Type | Pair Selection | Use Case |
|------|----------------|----------|
| **Neighbor** | (i, i+1) | Local spectral smoothing |
| **Harmonic** | (i, 2*i) | Octave relationships |
| **Subband** | (leaf_i, leaf_j) across k-slots | Cross-polyphase mixing |
| **Ratio-based** | Based on FM operator ratios | DX7-like topology |

### 5.2 Predefined Programs

| ID | Name | Crossings | Description |
|----|------|-----------|-------------|
| 0 | IDENTITY | 0 | No modulation (bypass) |
| 1 | NEIGHBOR_8 | 8 | Adjacent lane pairs |
| 2 | HARMONIC_16 | 16 | Octave relationships |
| 3 | DX_CHAIN | 24 | Approximates DX7 Alg 1 |
| 4 | DX_DUAL | 20 | Approximates DX7 Alg 5 |
| 5 | SCATTER_32 | 32 | Wide spectral mixing |

### 5.3 Neighbor Schedule

```python
SCHEDULE_NEIGHBOR = BraidProgram(
    program_id=1,
    name="NEIGHBOR_8",
    num_crossings=8,
    crossings=[
        # Layer 0: Even pairs
        CrossInstruction(0, 1, AngleSource.CONTROL, 0, 0),
        CrossInstruction(2, 3, AngleSource.CONTROL, 1, 0),
        CrossInstruction(4, 5, AngleSource.CONTROL, 2, 0),
        CrossInstruction(6, 7, AngleSource.CONTROL, 3, 0),
        # Layer 1: Odd pairs
        CrossInstruction(1, 2, AngleSource.CONTROL, 4, 0),
        CrossInstruction(3, 4, AngleSource.CONTROL, 5, 0),
        CrossInstruction(5, 6, AngleSource.CONTROL, 6, 0),
        CrossInstruction(7, 8, AngleSource.CONTROL, 7, 0),
    ],
    num_layers=2,
    flags=0,
)
```

### 5.4 Harmonic Schedule

```python
SCHEDULE_HARMONIC = BraidProgram(
    program_id=2,
    name="HARMONIC_16",
    num_crossings=16,
    crossings=[
        # Fundamental <-> 2nd harmonic
        CrossInstruction(0, 1, AngleSource.CONTROL, 0, 0),
        CrossInstruction(0, 2, AngleSource.CONTROL, 1, 0),
        # Carrier <-> Modulator relationships
        CrossInstruction(0, 4, AngleSource.CONTROL, 2, 0),
        CrossInstruction(1, 5, AngleSource.CONTROL, 3, 0),
        # Cross-operator mixing
        CrossInstruction(8, 16, AngleSource.CONTROL, 4, 0),
        # ... (continues)
    ],
    num_layers=4,
    flags=0,
)
```

### 5.5 DX7-Mapped Schedule

Map DX7 algorithm topology to braid crossings:

```python
def dx7_to_braid_schedule(alg_id: int) -> BraidProgram:
    """
    Convert DX7 algorithm to equivalent braid schedule.

    Mapping:
    - Each modulation edge -> set of crossings
    - Operator ratio -> which lanes to cross
    - Feedback -> crossing chain with delayed angle
    """
    alg = get_algorithm(alg_id)  # From WSE-STD-FM6
    crossings = []

    for op_idx in alg.eval_order:
        for mod_idx in alg.get_modulators(op_idx):
            # Create crossings for this modulation relationship
            carrier_lanes = get_lanes_for_ratio(alg.operators[op_idx].ratio_q16)
            mod_lanes = get_lanes_for_ratio(alg.operators[mod_idx].ratio_q16)

            for c_lane in carrier_lanes[:2]:  # Limit per pair
                for m_lane in mod_lanes[:2]:
                    crossings.append(CrossInstruction(
                        lane_i=c_lane,
                        lane_j=m_lane,
                        angle_src=AngleSource.CONTROL,
                        angle_param=len(crossings),
                        flags=0,
                    ))

    return BraidProgram(
        program_id=alg_id + 100,
        name=f"DX7_Alg{alg_id}_Braid",
        num_crossings=len(crossings),
        crossings=crossings,
        num_layers=len(alg.eval_order),
        flags=0,
    )
```

---

## 6. FM6 Integration

### 6.1 Pipeline

Per voice, per block:

```
+----------------------------------------------------------+
| 1. FM6 Physics Render (time-domain carrier block)        |
|    x[0..N-1] = FM6_Physics(voice_state, controls, N)     |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
| 2. Projection to coefficient space:                      |
|    - Preferred: c = WPT_Forward(x)                       |
|    - Alternative: c = BasisProject(x)                    |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
| 3. CCS Braid Fabric Modulation                           |
|    c' = BraidFabric(c, braid_program, control_sources)   |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
| 4. IDWPT Reconstruction                                  |
|    y = IDWPT(c')                                         |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
| 5. Mix/output                                            |
|    mix += y * voice_gain                                 |
+----------------------------------------------------------+
```

### 6.2 OSC_MODE

```python
OSC_MODE_FM6_CCS_BRAID = 5   # FM6 + braid fabric mode
```

### 6.3 VoiceFrame Parameters

```python
# Additions to VoiceFrame for braid mode:
osc_mode: int = OSC_MODE_FM6_CCS_BRAID
fm6_params: FM6Params                  # From WSE-STD-FM6
braid_program_id: int = 0              # Index into braid program ROM
braid_depth_q15: int = 0               # Global crossing depth (Q1.15)
braid_rate_q15: int = 0                # Modulation rate
braid_sources: List[int] = []          # Control signals for crossings
```

### 6.4 FM6 Integration Rules

**R1: Physics must remain physics**

FM6 oscillator/phase integration MUST remain in the physics layer. Braid fabric MUST NOT attempt to generate oscillations from silence.

**R2: Where FM routing lives**

FM routing/topology can be expressed as braid programs:

| FM6 Concept | Braid Equivalent |
|-------------|------------------|
| Algorithm (topology) | Braid crossing schedule |
| Modulation depth | Crossing angle magnitude |
| Feedback | Looped control source feeding angle |
| Operator ratio | Which lanes participate in crossing |

**R3: Compatibility with existing FM6 algorithms**

A "DX-style" algorithm can be approximated by choosing crossings that mix modulator energy into carrier-related lanes. Exact equivalence is NOT required; perceptual similarity is the goal.

---

## 7. Projection Requirements

### 7.1 Forward WPT Analysis (Preferred)

Implement `WPT_Forward` matched to IDWPT:
- Same wavelet family
- Same levels
- Same boundary handling
- Consistent normalization

```python
def wpt_forward(x: np.ndarray, headroom_bits: int = 2) -> np.ndarray:
    """
    Forward WPT analysis (matched to IDWPT).

    Args:
        x: Time-domain block [64] (Q1.15)
        headroom_bits: Normalization headroom

    Returns:
        c: Coefficient vector [64] (Q1.15)

    MUST satisfy: ||IDWPT(WPT_Forward(x)) - x|| < tolerance
    """
    return fwpt_jit(x.tolist(), headroom_bits=headroom_bits)
```

### 7.2 Basis Projection (Alternative)

If IDWPT is via fixed synthesis matrix B, projection MAY be:
- **Orthonormal:** `c = B^T x`
- **Frame:** `c = argmin ||B c - x||^2` (precomputed pseudoinverse)

---

## 8. Compute Budget & Degradation

### 8.1 Tier Definitions

| Tier | Description | Max Crossings | Angle LUT |
|------|-------------|---------------|-----------|
| 0 (FULL) | FM6 + WPT + K crossings + IDWPT | 32 | 256 |
| 1 (MID) | Reduced crossings, coarser LUT | 16 | 64 |
| 2 (MIN) | Identity (no braid), keep projection | 0 | N/A |
| 3 (EMERGENCY) | Single carrier projected | N/A | N/A |

### 8.2 Operation Counts

| Tier | Approximate Mults |
|------|-------------------|
| 0 | ~1500 |
| 1 | ~800 |
| 2 | ~400 |
| 3 | ~200 |

### 8.3 Degradation Determinism

Degradation MUST be deterministic and consistent:
- Same budget -> same tier
- Same tier -> same crossings enabled
- Tier transitions produce no glitches (crossfade optional)

---

## 9. FPGA Resource Analysis

### 9.1 Comparison: FM6 vs Braid Network

| Metric | Traditional FM6 | Braid (32 crossings) |
|--------|-----------------|---------------------|
| Multiplies/block | ~1536 | ~128 |
| Memory access | 6 basis frame lookups | Streaming (local) |
| Latency | O(6 x 64) | O(num_layers) |
| DSP usage | High | Low |
| Parallelizable | Per-operator | Per-layer |

### 9.2 Crossing Primitive Resources

Per 2x2 Givens rotation:
- 4 DSP multiplies (or 3 with lifting)
- 2 adders
- 2 registers (intermediate storage)
- Optional: sin/cos LUT (256 entries, shared)

### 9.3 Full Network Resources

For K=32 crossings, 4 layers:

| Resource | Count | Notes |
|----------|-------|-------|
| DSP | 8 (time-multiplexed) | 4 mults x 2 active |
| BRAM | 0.5 KB | sin/cos LUT |
| Registers | 64 | Coefficient buffer |
| LUTs | ~200 | Control logic |

### 9.4 Throughput Analysis

**Target:** Process 64-sample block in block period (1.33ms @ 48kHz)

With 32 crossings, 4 layers, pipeline depth 1:
- 32 crossings x 4 cycles/crossing = 128 cycles
- At 100 MHz: 1.28 us per block
- **Margin:** 1000x faster than block period

---

## 10. Sound Characteristics

### 10.1 Spectral Differences from Traditional FM

| Aspect | Traditional FM | Braid Network |
|--------|---------------|---------------|
| Sideband creation | Phase modulation | Energy redistribution |
| Frequency accuracy | Exact fc +/- n*fm | Approximate (lane-based) |
| Dynamic response | Instant | Slightly smoothed |
| Aliasing | Possible | Naturally band-limited |
| Character | Metallic, sharp | Warmer, more organic |

### 10.2 Best Use Cases

**Well suited for:**
- Pad/texture synthesis
- Evolving timbres
- Hybrid with other CCS modes
- Novel spectral effects

**Less suited for:**
- Classic DX7 E.Piano (use traditional FM6)
- Precise bell tones
- FM basses with exact harmonics

### 10.3 Hybrid FM6+Braid

Blend traditional FM6 with braid network:

```python
def hybrid_fm_braid(
    fm6_coeffs: np.ndarray,
    braid_state: BraidState,
    blend_q15: int,
) -> np.ndarray:
    """
    Blend traditional FM6 with braid network.

    blend_q15: 0 = pure FM6, 32767 = pure braid
    """
    # Apply braid network
    braid_coeffs = braid_fabric_apply(
        fm6_coeffs,
        braid_state.program,
        braid_state.angles
    )

    # Blend
    inv_blend = 32767 - blend_q15
    output = np.zeros_like(fm6_coeffs)
    for i in range(len(fm6_coeffs)):
        output[i] = SAT16(
            RSHIFT_RNE(fm6_coeffs[i] * inv_blend, 15) +
            RSHIFT_RNE(braid_coeffs[i] * blend_q15, 15)
        )

    return output
```

---

## 11. Verification & Conformance

### 11.1 Crossing Primitive Tests

1. **Energy preservation:** |x'|^2 + |y'|^2 = |x|^2 + |y|^2 (within 1 LSB)
2. **Inverse operation:** G(-theta) x G(theta) = identity
3. **Composition:** G(theta1 + theta2) = G(theta1) x G(theta2)
4. **Edge cases:** theta=0, theta=pi/2, theta=pi

```python
def test_crossing_invertibility():
    """Verify crossing with theta then -theta recovers input."""
    for _ in range(1000):
        c_i = np.random.randint(-32768, 32767)
        c_j = np.random.randint(-32768, 32767)
        theta = np.random.randint(0, 32767)

        sin_t, cos_t = sin_cos_q15(theta << 17)
        c_i_rot, c_j_rot = braid_crossing_q15(c_i, c_j, sin_t, cos_t)

        # Inverse rotation
        sin_neg, cos_neg = sin_cos_q15((-theta) << 17 & 0xFFFFFFFF)
        c_i_rec, c_j_rec = braid_crossing_q15(c_i_rot, c_j_rot, sin_neg, cos_neg)

        # Allow 1 LSB error from rounding
        assert abs(c_i - c_i_rec) <= 1
        assert abs(c_j - c_j_rec) <= 1
```

### 11.2 Network Tests

1. **Determinism:** Same input + schedule + angles -> same output
2. **Saturation bounds:** No overflow with max-amplitude inputs
3. **Layer independence:** Layers can be evaluated in parallel

```python
def test_determinism():
    """Same input/program must produce identical output."""
    coeffs = np.random.randint(-32768, 32767, 256, dtype=np.int16)
    program = BRAID_PROGRAMS[1]
    angles = np.full(program.num_crossings, 8192, dtype=np.int16)

    result1 = braid_fabric_apply(coeffs, program, angles)
    result2 = braid_fabric_apply(coeffs, program, angles)

    assert np.array_equal(result1, result2)
```

### 11.3 Round-Trip Transform Test

```python
def test_transform_roundtrip():
    """Verify WPT_Forward and IDWPT are matched inverses."""
    for _ in range(1000):
        c = np.random.randint(-32768, 32767, 64, dtype=np.int16)
        x = idwpt(c)
        c_hat = wpt_forward(x)
        # Tolerance: 1 LSB per coefficient
        assert np.abs(c - c_hat).max() <= 1
```

### 11.4 Audio Quality Tests

1. **No DC drift:** Average output ~ 0 over 1000 blocks
2. **No aliasing:** No energy above Nyquist
3. **Stable amplitude:** RMS bounded under modulation

### 11.5 Conformance Checklist

A build is **WSE-STD-BRAID conformant** if:

- [ ] R1: No IDWPT bypass when braid is active
- [ ] R2: Braid implemented as CCS mode stage
- [ ] R3: Fixed-point, deterministic braid math
- [ ] R4: Forward/inverse transform pairing verified
- [ ] Braid program microcode implemented
- [ ] Degradation ladder implemented and tested
- [ ] All tests pass

---

## 12. Reference Data

### 12.1 Sin/Cos LUT (256 entries)

```python
import math
SIN_LUT_Q15 = np.array([
    int(32767 * math.sin(2 * math.pi * i / 256))
    for i in range(256)
], dtype=np.int16)

COS_LUT_Q15 = np.array([
    int(32767 * math.cos(2 * math.pi * i / 256))
    for i in range(256)
], dtype=np.int16)
```

### 12.2 Fixed-Point Formats

| Stage | Format | Range | Notes |
|-------|--------|-------|-------|
| Coefficient lanes | Q1.15 | [-1, 1) | Primary format |
| Internal MAC | Q3.29 | [-8, 8) | Guard bits |
| Angle LUT | Q1.15 | [-1, 1) | sin/cos values |
| Angle param | Q1.15 | [0, 1) | Depth/index |

---

## 13. Future Extensions (Informative)

### 13.1 Audio-Rate Angle Modulation

For sample-accurate FM character, angles could update at lane-rate (K times per block) instead of block-rate. Requires:
- Angle interpolation logic
- Increased control bandwidth
- Higher compute cost

### 13.2 Adaptive Crossing Schedules

Runtime selection based on:
- Audio content (spectral analysis)
- Performance budget
- Patch-defined heuristics

### 13.3 Braid Feedback Loops

Direct coefficient-domain feedback where crossing output feeds back to crossing angles:

```
c' = BraidFabric(c, program, angles(c))
```

Creates coefficient-domain self-modulation analogous to FM feedback.

---

**Document Version:** 1.0
**Last Updated:** 2026-01-13
**Author:** CCS Development Team
**Status:** DRAFT
