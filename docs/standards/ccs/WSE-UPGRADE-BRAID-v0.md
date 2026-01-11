# WSE-UPGRADE-BRAID-v0 — Sparse Braid Network for FM Synthesis

**Standard ID:** WSE-UPGRADE-BRAID-v0
**Title:** Sparse Braid Network Approach to Coefficient-Domain FM
**Status:** PROPOSAL
**Scope:** Alternative FM synthesis architecture using sparse 2x2 Givens rotations ("braid crossings") in coefficient domain.
**Depends on:** WSE-STD-001 (Fixed-Point), WSE-STD-FM-TOPOLOGY-v0, WSE-STD-210 (CCS Mode B)
**Supersedes:** None (upgrade path, not replacement)

---

## 0. Executive Summary

This document proposes an **alternative FM synthesis architecture** based on sparse braid networks. Instead of traditional phase modulation (current FM6 approach), this method applies 2x2 Givens rotations directly to coefficient pairs.

**Key Properties:**
- FPGA-optimized: O(K) crossings instead of O(64×K) operations per block
- Streaming-friendly: Local 2x2 operations, no global memory access
- Energy-preserving: Unitary rotations maintain coefficient energy
- Novel sound: Different spectral character than traditional FM

**Status:** Research proposal. Not a replacement for current FM6, but a potential future synthesis mode.

---

## 1. Motivation

### 1.1 Current FM6 Architecture

The current coefficient-native FM6 (per WSE-STD-FM-TOPOLOGY-v0) uses:

```python
# Per operator, per k-slot:
for op_idx in eval_order:
    basis = get_basis_for_ratio(base_phase_inc, op.ratio_q16)
    for k in range(K):
        eff_phase = op.phase + k * phase_inc + mod_sum[k]
        sin_phi, cos_phi = sin_cos_q15(eff_phase)
        for leaf in range(64):
            coeff[leaf,k] = cos_phi * basis.Usin[leaf,k] + sin_phi * basis.Ucos[leaf,k]
```

**Cost:** 6 operators × K slots × 64 leaves × ~6 ops = ~1536 multiplies per block

### 1.2 Braid Network Insight

FM synthesis creates sidebands by phase modulation. But in coefficient domain, we can achieve spectral mixing through **Givens rotations** between coefficient pairs:

```
[x']   [cos θ   sin θ] [x]
[y'] = [-sin θ  cos θ] [y]
```

A sparse network of such rotations, applied to carefully chosen coefficient pairs, can create FM-like spectral effects with far fewer operations.

---

## 2. Mathematical Foundation

### 2.1 Givens Rotation as FM Primitive

A 2x2 Givens rotation is:

```
G(θ) = [cos θ   sin θ]
       [-sin θ  cos θ]
```

**Properties:**
- **Unitary:** G(θ)ᵀG(θ) = I (energy-preserving)
- **Invertible:** G(θ)⁻¹ = G(-θ)
- **Commutative composition:** G(θ₁)G(θ₂) = G(θ₁+θ₂)
- **Hardware cost:** 4 multiplies + 2 adds (or 3 mults with Golub-Van Loan trick)

### 2.2 Braid Group Interpretation

The braid group Bₙ on n strands has generators σᵢ where σᵢ swaps strands i and i+1. In our coefficient context:

- **Strands** = coefficient values (64 leaves × K slots = 256-512 values)
- **Crossings** = Givens rotations between adjacent coefficient pairs
- **Braid word** = Sequence of crossings defining the FM algorithm

The key insight: **A sparse set of crossings creates controlled spectral mixing**.

### 2.3 Connection to FM Synthesis

Traditional FM: `sin(ωc t + β sin(ωm t))` creates sidebands at ωc ± n·ωm

Braid rotation: Mixing coefficient pairs (leaf i, leaf j) with angle θ:
- Transfers energy between frequency bands i and j
- θ controls "modulation depth"
- Multiple crossings create compound spectral effects

**Not mathematically equivalent, but sonically related.**

---

## 3. Architecture

### 3.1 Braid Network Structure

```
                     Coefficient Stream [256]
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
      ┌─────────┐        ┌─────────┐        ┌─────────┐
      │Crossing │        │Crossing │        │Crossing │
      │  (i,j)  │        │  (k,l)  │        │  (m,n)  │
      │   θ₁    │        │   θ₂    │        │   θ₃    │
      └────┬────┘        └────┬────┘        └────┬────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
      ┌─────────┐        ┌─────────┐        ┌─────────┐
      │Crossing │        │Crossing │        │Crossing │
      │  (a,b)  │        │  (c,d)  │        │  (e,f)  │
      │   θ₄    │        │   θ₅    │        │   θ₆    │
      └────┬────┘        └────┬────┘        └────┬────┘
           │                  │                  │
           └──────────────────┴──────────────────┘
                              │
                              ▼
                     Output Coefficients [256]
```

### 3.2 Crossing Schedule

A **crossing schedule** defines which coefficient pairs interact and in what order:

```python
@dataclass
class CrossingSchedule:
    """Defines the braid network topology."""

    # List of crossings: (coeff_i, coeff_j, angle_param_id)
    crossings: List[Tuple[int, int, int]]

    # Number of crossing layers (for pipelining)
    num_layers: int

    # Static or dynamic angles
    static_angles: Optional[np.ndarray]  # [num_crossings] Q15
```

### 3.3 Crossing Types

| Type | Pair Selection | Use Case |
|------|----------------|----------|
| **Neighbor** | (i, i+1) | Local spectral smoothing |
| **Harmonic** | (i, 2*i) | Octave relationships |
| **Subband** | (leaf_i, leaf_j) across k-slots | Cross-polyphase mixing |
| **Ratio-based** | Based on FM operator ratios | DX7-like topology |

---

## 4. Implementation

### 4.1 Single Crossing Primitive

```python
def braid_crossing_q15(x: int, y: int, sin_theta: int, cos_theta: int) -> Tuple[int, int]:
    """
    Apply 2x2 Givens rotation to coefficient pair.

    Args:
        x, y: Input coefficients (Q15)
        sin_theta, cos_theta: Rotation angle (Q15)

    Returns:
        (x', y'): Rotated coefficients (Q15)

    Per WSE-STD-001:
        x' = cos(θ)*x + sin(θ)*y
        y' = -sin(θ)*x + cos(θ)*y
    """
    # Compute products in s32/s64
    x_cos = RSHIFT_RNE(x * cos_theta, 15)
    x_sin = RSHIFT_RNE(x * sin_theta, 15)
    y_cos = RSHIFT_RNE(y * cos_theta, 15)
    y_sin = RSHIFT_RNE(y * sin_theta, 15)

    # Combine with saturation
    x_prime = SAT16(x_cos + y_sin)
    y_prime = SAT16(y_cos - x_sin)

    return x_prime, y_prime
```

### 4.2 Full Braid Network

```python
def braid_network_apply(
    coeffs: np.ndarray,
    schedule: CrossingSchedule,
    angles_q15: np.ndarray,
) -> np.ndarray:
    """
    Apply braid network to coefficient array.

    Args:
        coeffs: Input coefficients [COEFF_LEN] (Q15)
        schedule: Crossing schedule defining topology
        angles_q15: Current angle values [num_crossings] (Q15)

    Returns:
        output: Transformed coefficients [COEFF_LEN] (Q15)
    """
    output = coeffs.copy()

    for crossing_idx, (i, j, _) in enumerate(schedule.crossings):
        theta_q15 = angles_q15[crossing_idx]
        sin_theta, cos_theta = sin_cos_q15(theta_q15 << 17)  # Scale to Q32 phase

        output[i], output[j] = braid_crossing_q15(
            output[i], output[j],
            sin_theta, cos_theta
        )

    return output
```

### 4.3 Angle Modulation (FM Equivalent)

```python
def compute_braid_angles(
    state: BraidFMState,
    base_phase_inc: int,
) -> np.ndarray:
    """
    Compute crossing angles from FM-like parameters.

    Maps operator ratios and levels to braid angles.
    """
    angles = np.zeros(state.num_crossings, dtype=np.int16)

    for crossing_idx in range(state.num_crossings):
        # Base angle from crossing "frequency"
        base_angle = state.crossing_base_angles[crossing_idx]

        # Modulation from upstream crossings (like FM feedback)
        mod_sum = 0
        for mod_idx in state.crossing_modulators[crossing_idx]:
            mod_sum += state.crossing_outputs[mod_idx]

        # Scale modulation by depth
        mod_angle = RSHIFT_RNE(mod_sum * state.mod_depth_q15, 15)

        # Total angle with phase advancement
        total_angle = (base_angle + mod_angle + state.phase) & 0xFFFF
        angles[crossing_idx] = total_angle

        # Store output for modulation chain
        state.crossing_outputs[crossing_idx] = angles[crossing_idx]

    # Advance phase
    state.phase = (state.phase + base_phase_inc) & 0xFFFFFFFF

    return angles
```

---

## 5. FPGA Resource Analysis

### 5.1 Comparison: Current FM6 vs Braid Network

| Metric | Current FM6 | Braid (32 crossings) |
|--------|-------------|---------------------|
| **Multiplies/block** | ~1536 | ~128 |
| **Memory access** | 6 basis frame lookups | Streaming (local) |
| **Latency** | O(6 × 64) | O(num_layers) |
| **DSP usage** | High | Low |
| **Parallelizable** | Per-operator | Per-layer |

### 5.2 Crossing Primitive Resources

Per 2x2 Givens rotation:
- **4 DSP multiplies** (or 3 with Golub-Van Loan)
- **2 adders**
- **2 registers** (intermediate storage)
- **Optional:** sin/cos LUT (256 entries, shared)

### 5.3 Full Network Resources

For K=32 crossings, 4 layers:

| Resource | Count | Notes |
|----------|-------|-------|
| DSP | 8 (time-multiplexed) | 4 mults × 2 active |
| BRAM | 0.5 KB | sin/cos LUT |
| Registers | 64 | Coefficient buffer |
| LUTs | ~200 | Control logic |

### 5.4 Throughput Analysis

**Target:** Process 64-sample block in block period (1.33ms @ 48kHz)

With 32 crossings, 4 layers, pipeline depth 1:
- 32 crossings × 4 cycles/crossing = 128 cycles
- At 100 MHz: 1.28 μs per block
- **Margin:** 1000× faster than block period

---

## 6. Crossing Schedule Design

### 6.1 Neighbor Schedule (Smoothing)

```python
SCHEDULE_NEIGHBOR = CrossingSchedule(
    crossings=[
        # Layer 0: Even pairs
        (0, 1, 0), (2, 3, 1), (4, 5, 2), (6, 7, 3),
        # ... continues for all adjacent pairs
        # Layer 1: Odd pairs
        (1, 2, 4), (3, 4, 5), (5, 6, 6), (7, 8, 7),
        # ...
    ],
    num_layers=2,
)
```

### 6.2 Harmonic Schedule (FM-like)

```python
SCHEDULE_HARMONIC = CrossingSchedule(
    crossings=[
        # Fundamental ↔ 2nd harmonic
        (0, 1, 0),   # Leaf 0 ↔ Leaf 1
        (0, 2, 1),   # Leaf 0 ↔ Leaf 2
        # Carrier ↔ Modulator relationships
        (0, 4, 2),   # 1:1 ↔ 1:5 (like Op1 ↔ Op5)
        (1, 5, 3),   # Similar for other k-slots
        # Cross-operator mixing
        (8, 16, 4),  # Wider spectral mixing
        # ...
    ],
    num_layers=4,
)
```

### 6.3 DX7-Mapped Schedule

Map DX7 algorithm topology to braid crossings:

```python
def dx7_to_braid_schedule(alg_id: int) -> CrossingSchedule:
    """
    Convert DX7 algorithm to equivalent braid schedule.

    Mapping:
    - Each modulation edge → set of crossings
    - Operator ratio → which leaves to cross
    - Feedback → crossing chain with delayed angle
    """
    alg = get_algorithm(alg_id)
    crossings = []

    for op_idx in alg.eval_order:
        for mod_idx in alg.get_modulators(op_idx):
            # Create crossings for this modulation relationship
            carrier_leaves = get_leaves_for_ratio(alg.operators[op_idx].ratio_q16)
            mod_leaves = get_leaves_for_ratio(alg.operators[mod_idx].ratio_q16)

            for c_leaf in carrier_leaves:
                for m_leaf in mod_leaves:
                    crossings.append((c_leaf, m_leaf, len(crossings)))

    return CrossingSchedule(crossings=crossings, num_layers=len(alg.eval_order))
```

---

## 7. Sound Characteristics

### 7.1 Spectral Differences from Traditional FM

| Aspect | Traditional FM | Braid Network |
|--------|---------------|---------------|
| **Sideband creation** | Phase modulation | Energy redistribution |
| **Frequency accuracy** | Exact fc ± n·fm | Approximate (leaf-based) |
| **Dynamic response** | Instant | Slightly smoothed |
| **Aliasing** | Possible | Naturally band-limited |
| **Character** | Metallic, sharp | Warmer, more organic |

### 7.2 Use Cases

**Best suited for:**
- Pad/texture synthesis
- Evolving timbres
- Hybrid with other CCS modes
- Novel spectral effects

**Less suited for:**
- Classic DX7 E.Piano (use traditional FM6)
- Precise bell tones
- FM basses with exact harmonics

### 7.3 Hybrid Approach

Combine braid network with traditional FM6:

```python
def hybrid_fm_braid(
    state: FM6State,
    braid_state: BraidFMState,
    blend_q15: int,
) -> np.ndarray:
    """
    Blend traditional FM6 with braid network.

    blend_q15: 0 = pure FM6, 32767 = pure braid
    """
    # Generate traditional FM6 coefficients
    fm6_coeffs = np.zeros(COEFF_LEN, dtype=np.int16)
    fm6_generate_coeffs(state, fm6_coeffs, base_phase_inc)

    # Apply braid network
    braid_coeffs = braid_network_apply(fm6_coeffs, braid_state.schedule, braid_state.angles)

    # Blend
    inv_blend = 32767 - blend_q15
    for i in range(COEFF_LEN):
        output[i] = SAT16(
            RSHIFT_RNE(fm6_coeffs[i] * inv_blend, 15) +
            RSHIFT_RNE(braid_coeffs[i] * blend_q15, 15)
        )

    return output
```

---

## 8. Integration with CCS Pipeline

### 8.1 Proposed OSC_MODE

```python
OSC_MODE_A = 0        # Coefficient LUT
OSC_MODE_B = 1        # Rotator/Harmonic Bank
OSC_MODE_SAMPLE = 2   # Sample playback
OSC_MODE_FM6 = 3      # 6-operator FM (current)
OSC_MODE_BRAID = 4    # Braid network (proposed)
```

### 8.2 Pipeline Position

Option A: Replace coefficient generation
```
BraidNetwork → Operators → IDWPT → Audio
```

Option B: Post-coefficient processing
```
FM6/ModeB → BraidNetwork → Operators → IDWPT → Audio
```

**Recommendation:** Option B for maximum flexibility.

### 8.3 VoiceFrame Parameters

```python
# Proposed additions to VoiceFrame:
braid_schedule_id: int = 0        # Predefined schedule (0-7)
braid_depth_q15: int = 0          # Global modulation depth
braid_feedback_q15: int = 0       # Crossing chain feedback
braid_angle_offsets: List[int]    # Per-crossing angle offsets
```

---

## 9. Verification Requirements

### 9.1 Crossing Primitive Tests

1. **Energy preservation:** |x'|² + |y'|² = |x|² + |y|² (within 1 LSB)
2. **Inverse operation:** G(-θ) × G(θ) = identity
3. **Composition:** G(θ₁+θ₂) = G(θ₁) × G(θ₂)
4. **Edge cases:** θ=0, θ=π/2, θ=π

### 9.2 Network Tests

1. **Determinism:** Same input + schedule + angles → same output
2. **Saturation bounds:** No overflow with max-amplitude inputs
3. **Layer independence:** Layers can be evaluated in parallel

### 9.3 Audio Quality Tests

1. **No DC drift:** Average output ≈ 0 over 1000 blocks
2. **No aliasing:** No energy above Nyquist
3. **Stable amplitude:** RMS bounded under modulation

---

## 10. Implementation Roadmap

### Phase 1: Python Prototype
- [ ] Implement `braid_crossing_q15()` primitive
- [ ] Implement `braid_network_apply()` with static schedule
- [ ] Test energy preservation and determinism
- [ ] Create basic crossing schedules (neighbor, harmonic)

### Phase 2: Audio Evaluation
- [ ] Render test tones and compare with FM6
- [ ] Evaluate spectral characteristics
- [ ] Tune crossing schedules for musical sounds
- [ ] Create A/B comparison presets

### Phase 3: CCS Integration
- [ ] Add OSC_MODE_BRAID to executor
- [ ] Add VoiceFrame parameters
- [ ] Implement hybrid FM6+Braid mode
- [ ] Create preset examples

### Phase 4: FPGA Optimization
- [ ] Design pipelined crossing primitive
- [ ] Implement time-multiplexed network
- [ ] Verify resource usage on target FPGA
- [ ] Benchmark throughput

---

## 11. Appendix A: Mathematical Derivation

### A.1 Givens Rotation in Wavelet Domain

Given wavelet coefficients c = [c₀, c₁, ..., c₆₃] representing signal x(t), applying Givens rotation G(θ) to pair (cᵢ, cⱼ) produces:

```
c'ᵢ = cos(θ)cᵢ + sin(θ)cⱼ
c'ⱼ = -sin(θ)cᵢ + cos(θ)cⱼ
```

In time domain, this corresponds to mixing the frequency bands represented by leaves i and j.

### A.2 Relation to FFT Butterfly

An FFT butterfly is:
```
X[k] = A[k] + W^k B[k]
X[k+N/2] = A[k] - W^k B[k]
```

Where W = e^(-2πi/N). For real signals with θ = 2πk/N, this is equivalent to a Givens rotation.

### A.3 Energy Conservation Proof

For unitary matrix U (like Givens rotation):
```
||Ux||² = (Ux)ᵀ(Ux) = xᵀUᵀUx = xᵀIx = ||x||²
```

Therefore ||c'||² = ||c||² after any sequence of Givens rotations.

---

## 12. Appendix B: Preset Examples

### B.1 Braid Pad

```yaml
preset:
  preset_id: "braid_pad_v0"
  osc_mode: BRAID

braid:
  schedule_id: 2          # HARMONIC schedule
  depth_q15: 16384        # 50% modulation
  feedback_q15: 4096      # Light feedback

env:
  amp:
    attack_ms: 500
    decay_ms: 1000
    sustain: 0.7
    release_ms: 2000
```

### B.2 Hybrid FM+Braid Lead

```yaml
preset:
  preset_id: "hybrid_lead_v0"
  osc_mode: FM6

fm6:
  algorithm_id: 1
  # ... FM6 params ...

braid:
  enabled: true
  blend_q15: 8192         # 25% braid
  schedule_id: 1          # NEIGHBOR schedule
```

---

**Document Version:** v0.1
**Last Updated:** 2026-01-11
**Author:** CCS Development Team
**Status:** PROPOSAL - Pending experimental validation
