# WSE-UPGRADE-BRAID-FM6 v0.9

**Title:** FM6 as a CCS Mode via Braid (Butterfly) Modulation in Coefficient Space
**Status:** Draft (Implementable)
**Applies to:** VoiceEngine / CCS subsystem, WPT/IDWPT pipeline, deterministic block renderer

---

## 0. Intent

Upgrade FM6 so it is **coefficient-native** and **IDWPT-compatible** by:

1. **eliminating FM6 bypass** of reconstruction, and
2. implementing FM6 modulation/routing as a **braid (butterfly) fabric** acting on coefficient lanes within CCS.

This preserves FM6's audible character (sidebands from phase/FM physics) while making the system compatible with WSE's coefficient pipeline and compute degradation ladder.

---

## 1. Scope

### In scope

* FM6 block rendering (64-sample blocks or engine's base block)
* Projection into coefficient space (WPT forward or basis projection)
* Braid modulation fabric (sparse 2×2 crossings)
* IDWPT-only output path (no FM6 special-casing at output stage)
* Deterministic fixed-point implementation
* Compute-budget degradation for FM6/CCS mode

### Out of scope

* Full DX7 algorithm catalog parity
* High-rate sample-accurate audio-rate braid parameter modulation beyond block-rate (optional extension)
* New wavelet families / re-normalization research

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **FM6 Physics Layer** | Operator oscillators, phase accumulation, feedback delay, envelopes |
| **Coefficient Space** | The vector space of coefficients consumed by IDWPT for reconstruction |
| **Braid / Butterfly Fabric** | A scheduled sequence of sparse 2×2 invertible mixers ("crossings") acting on coefficient lanes |
| **Crossing** | A 2×2 rotation/shear mixing between two lanes parameterized by an angle/strength |
| **Mode** | A CCS execution mode defining how a voice block is generated and modulated prior to reconstruction |
| **Projection** | Mapping from time-domain block to coefficient vector (preferred: forward WPT analysis paired to IDWPT) |

---

## 3. Architectural Requirements

### R1 — No bypass

FM6 MUST NOT bypass IDWPT. All audio output MUST be produced via the standard reconstruction path:

```
coeff_vector → IDWPT → audio_block
```

### R2 — FM6 as CCS mode

FM6 MUST be implemented as a CCS mode that:

1. generates (or receives) a coefficient vector, then
2. applies braid modulation fabric, then
3. passes to IDWPT.

### R3 — Deterministic fixed-point

All braid operations MUST be deterministic across platforms:

* fixed-point arithmetic (per WSE-STD-001)
* defined rounding and saturation
* no denormals
* LUT-based trig or CORDIC with fixed iteration count

### R4 — Matched transform pair

If using forward WPT projection:

* the forward WPT MUST be paired and normalized consistently with IDWPT such that round-trip error is bounded and testable.

---

## 4. High-Level Pipeline

Per voice, per block:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. FM6 Physics Render (time-domain carrier block)              │
│    x[0..N-1] = FM6_Physics(voice_state, controls, N)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Projection to coefficient space (choose one):               │
│    • Preferred (A): c = WPT_Forward(x)                         │
│    • Alternative (B): c = BasisProject(x)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CCS Braid Fabric Modulation                                 │
│    c' = BraidFabric(c, braid_program, control_sources)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. IDWPT Reconstruction                                        │
│    y = IDWPT(c')                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Mix/output                                                  │
│    mix += y * voice_gain                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Model

### 5.1 VoiceFrame additions (logical)

FM6 mode shall declare:

```python
osc_mode = FM6_CCS_BRAID     # New mode constant
fm6_params: FM6Params        # Existing FM6 parameters
braid_program_id: int        # Index into braid program ROM
braid_depth_q15: int         # Global crossing depth (Q1.15)
braid_rate_q15: int          # Modulation rate (block-rate by default)
braid_sources: List[int]     # Optional: which control signals drive crossings
```

### 5.2 Braid Program ROM (microcode)

A braid program is a list of `Cross` instructions:

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

Programs SHOULD be shareable across patches; patches reference them by ID.

### 5.3 Braid Program Format

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

---

## 6. Braid Fabric Semantics

### 6.1 Crossing primitive (norm-preserving default)

Default crossing SHALL be an orthonormal 2×2 rotation applied to lanes `i,j`:

```
[c_i']   [cos θ   sin θ ] [c_i]
[c_j'] = [-sin θ  cos θ ] [c_j]
```

**Fixed-point implementation (per WSE-STD-001):**

```python
def braid_crossing_q15(
    c_i: int, c_j: int,
    sin_theta_q15: int, cos_theta_q15: int
) -> Tuple[int, int]:
    """
    Apply 2×2 Givens rotation to coefficient pair.

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

### 6.2 Optional low-mult "lifting" crossing

If DSP resources are tight, a crossing MAY be implemented as 2–3 shear/lifting steps (still invertible) with better multiplier economy:

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

### 6.3 Schedule execution

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

Crossings SHOULD be sparse and local (small K), and MAY be pipelined or time-multiplexed.

---

## 7. FM6 Integration Rules

### 7.1 Physics must remain physics

FM6 oscillator/phase integration MUST remain in the physics layer. Braid fabric MUST NOT attempt to generate oscillations from silence.

**Rationale:** The braid fabric is a modulation/routing stage, not a sound source. FM6's carrier generation, phase accumulation, and envelope logic remain unchanged.

### 7.2 Where FM routing lives

FM routing/topology SHOULD be expressed as braid programs:

| FM6 Concept | Braid Equivalent |
|-------------|------------------|
| Algorithm (topology) | Braid crossing schedule |
| Modulation depth | Crossing angle magnitude |
| Feedback | Looped control source feeding angle |
| Operator ratio | Which lanes participate in crossing |

### 7.3 Compatibility with existing FM6 algorithms

* A "DX-style" algorithm can be approximated by choosing crossings that mix modulator energy into carrier-related lanes.
* Exact equivalence is NOT required; perceptual similarity is the goal.

**Mapping guideline:**

```python
def dx7_algorithm_to_braid(alg_id: int) -> BraidProgram:
    """
    Map DX7 algorithm to braid program.

    Strategy:
    - For each modulation edge (mod_op → carrier_op):
      - Create crossings between coefficient lanes corresponding to
        mod_op's frequency ratio and carrier_op's frequency ratio
    - Feedback edge maps to FB angle source
    """
    alg = get_algorithm(alg_id)
    crossings = []

    for op_idx in alg.eval_order:
        for mod_idx in alg.get_modulators(op_idx):
            # Identify lane pairs based on frequency ratios
            op_lanes = ratio_to_lanes(alg.operators[op_idx].ratio_q16)
            mod_lanes = ratio_to_lanes(alg.operators[mod_idx].ratio_q16)

            for op_lane in op_lanes[:2]:  # Limit crossings per pair
                for mod_lane in mod_lanes[:2]:
                    crossings.append(CrossInstruction(
                        lane_i=op_lane,
                        lane_j=mod_lane,
                        angle_src=AngleSource.CONTROL,
                        angle_param=0,
                        flags=0,
                    ))

    return BraidProgram(
        program_id=alg_id + 100,  # Offset from custom programs
        name=f"DX7_Alg{alg_id}_Braid",
        num_crossings=len(crossings),
        crossings=crossings,
        num_layers=len(alg.eval_order),
        flags=0,
    )
```

---

## 8. Projection Requirements

### 8.1 Preferred: Forward WPT analysis

Implement `WPT_Forward` matched to IDWPT:

* same wavelet family
* same levels
* same boundary handling
* consistent normalization

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

### 8.2 Alternative: Basis projection

If IDWPT is implemented via a fixed synthesis matrix/basis `B` (explicitly or implicitly), projection MAY be:

* **Orthonormal:** `c = Bᵀ x`
* **Frame:** `c = argmin ||B c - x||²` (precomputed pseudoinverse if feasible)

**Note:** This method is only safe if basis definition is *exactly* aligned with IDWPT.

```python
def basis_project(x: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    """
    Project time-domain block onto IDWPT basis.

    Args:
        x: Time-domain block [64] (Q1.15)
        basis_matrix: IDWPT synthesis basis [64, 64] (precomputed)

    Returns:
        c: Coefficient vector [64] (Q1.15)
    """
    # For orthonormal basis: c = Bᵀ x
    c = np.zeros(64, dtype=np.int16)
    for i in range(64):
        acc = 0
        for j in range(64):
            acc += RSHIFT_RNE(x[j] * basis_matrix[j, i], 15)
        c[i] = SAT16(acc)
    return c
```

---

## 9. Compute Budget and Degradation Ladder

FM6_CCS_BRAID MUST support compute degradation per WSE-STD-DEGRADE:

| Tier | Description | Operations |
|------|-------------|------------|
| **0 (FULL)** | FM6 physics + forward WPT + K crossings + IDWPT | ~1500 mults |
| **1 (MID)** | Reduce K crossings; coarser angle LUT | ~800 mults |
| **2 (MIN)** | Disable braid (identity), keep projection + IDWPT | ~400 mults |
| **3 (EMERGENCY)** | Single carrier projected (bypass FM6 physics) | ~200 mults |

### 9.1 Crossing count by tier

| Tier | Max Crossings (K) | Angle LUT Size |
|------|-------------------|----------------|
| 0 | 32 | 256 |
| 1 | 16 | 64 |
| 2 | 0 (identity) | N/A |
| 3 | N/A | N/A |

### 9.2 Degradation determinism

Degradation MUST be deterministic and consistent:

* Same budget → same tier
* Same tier → same crossings enabled
* Tier transitions produce no glitches (crossfade optional)

---

## 10. Determinism & Numeric Policy

### 10.1 Fixed-point formats

| Stage | Format | Range | Notes |
|-------|--------|-------|-------|
| Coefficient lanes | Q1.15 | [-1, 1) | Primary format |
| Internal MAC | Q3.29 | [-8, 8) | Guard bits for accumulation |
| Angle LUT | Q1.15 | [-1, 1) | sin/cos values |
| Angle param | Q1.15 | [0, 1) | Depth/index |

### 10.2 Rounding policy

Per WSE-STD-001:

* All right shifts use `RSHIFT_RNE` (round-to-nearest-even)
* No floor or truncation rounding in bit-exact paths

### 10.3 Saturation policy

Per WSE-STD-001:

* All narrowing operations use `SAT16`
* Silent wrap is forbidden
* Overflow clamps to ±32767

---

## 11. Conformance Tests

### T1 — No bypass (structural)

```python
def test_no_fm6_bypass():
    """Assert FM6 output comes from IDWPT, not direct audio."""
    # Inspect executor code path
    assert osc_mode == OSC_MODE_FM6_CCS_BRAID implies output == lane_ccs_render()
    # No special-case branch for FM6 audio output
```

### T2 — Round-trip transform (core)

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

### T3 — FM6 projection consistency

```python
def test_fm6_projection():
    """Verify FM6 physics block survives projection round-trip."""
    state = FM6State(algorithm_id=1)
    x = fm6_generate_block_full(state, base_phase_inc=89478485)
    c = wpt_forward(x)
    y = idwpt(c)
    # Tolerance: -60 dB error
    error_rms = np.sqrt(np.mean((x - y).astype(float)**2))
    signal_rms = np.sqrt(np.mean(x.astype(float)**2))
    assert error_rms / signal_rms < 0.001  # < -60 dB
```

### T4 — Braid invertibility sanity (local)

```python
def test_crossing_invertibility():
    """Verify crossing with θ then -θ recovers input."""
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

### T5 — Determinism

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

### T6 — Degradation determinism

```python
def test_degradation_determinism():
    """Same budget produces same degradation path."""
    for budget in [100, 50, 25, 10]:
        tier1 = compute_degradation_tier(budget)
        tier2 = compute_degradation_tier(budget)
        assert tier1 == tier2

        if tier1 < 2:  # Has braid crossings
            k1 = get_max_crossings_for_tier(tier1)
            k2 = get_max_crossings_for_tier(tier1)
            assert k1 == k2
```

---

## 12. Conformance Checklist

A build is **WSE-UPGRADE-BRAID-FM6 conformant** if:

- [ ] **R1:** FM6 bypass removed; IDWPT is the only output path
- [ ] **R2:** FM6 implemented as CCS mode with braid fabric stage
- [ ] **R3:** Fixed-point, deterministic braid math (LUT/CORDIC defined)
- [ ] **R4:** Forward/inverse transform pairing verified (T2, T3)
- [ ] Braid program microcode implemented and patch-addressable
- [ ] Degradation ladder implemented and tested (T6)
- [ ] All tests T1–T6 pass

---

## 13. Implementation Notes (non-normative)

### 13.1 FPGA optimization

* A single **time-multiplexed** butterfly unit + lane RAM is the most compact implementation
* Keep braid schedules **static wiring**, varying only angles for best timing closure
* Store a small `sin/cos` LUT (256 entries); avoid runtime trig
* Start with K=8–32 crossings per block; tune perceptually

### 13.2 Python reference implementation

```python
# Reference sin/cos LUT (256 entries, Q1.15)
SIN_LUT_Q15 = np.array([
    int(32767 * math.sin(2 * math.pi * i / 256))
    for i in range(256)
], dtype=np.int16)

COS_LUT_Q15 = np.array([
    int(32767 * math.cos(2 * math.pi * i / 256))
    for i in range(256)
], dtype=np.int16)
```

### 13.3 Suggested initial braid programs

| ID | Name | Crossings | Description |
|----|------|-----------|-------------|
| 0 | IDENTITY | 0 | No modulation (bypass) |
| 1 | NEIGHBOR_8 | 8 | Adjacent lane pairs |
| 2 | HARMONIC_16 | 16 | Octave relationships |
| 3 | DX_CHAIN | 24 | Approximates DX7 Alg 1 |
| 4 | DX_DUAL | 20 | Approximates DX7 Alg 5 |
| 5 | SCATTER_32 | 32 | Wide spectral mixing |

---

## 14. Future Extensions (informative)

### 14.1 Audio-rate angle modulation

For sample-accurate FM character, angles could update at lane-rate (K times per block) instead of block-rate. This requires:

* Angle interpolation logic
* Increased control bandwidth
* Higher compute cost

### 14.2 Adaptive crossing schedules

Runtime selection of crossing schedule based on:

* Audio content (spectral analysis)
* Performance budget
* Patch-defined heuristics

### 14.3 Braid feedback loops

Direct coefficient-domain feedback where crossing output feeds back to crossing angles:

```
c' = BraidFabric(c, program, angles(c))
```

This creates coefficient-domain self-modulation analogous to FM feedback.

---

**Document Version:** v0.9
**Last Updated:** 2026-01-11
**Author:** CCS Development Team
**Status:** Draft (Implementable)
