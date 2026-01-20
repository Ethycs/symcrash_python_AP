# Operator Reference

Complete reference for the 24 operators in the symcrash wavelet synthesis engine.

## Overview

- **Coefficient Vector:** 256 samples per voice
- **Band Structure:** 6 dyadic bands (coarse to fine)
- **Node Structure:** 32 nodes, 8 coefficients each
- **Fixed-Point:** Q15 (int16) for operator parameters, Q16.16 for FM6 ratios

### Synthesis Modes

The CCS pipeline supports multiple synthesis modes via `osc_mode`:

| Mode | Value | Description | Operators Used |
|------|-------|-------------|----------------|
| Mode A (LUT) | 0 | Wavetable lookup | Full operator set |
| Mode B (Rotator) | 1 | Harmonic rotator | Full operator set |
| Sample | 2 | Sample playback | Reduced set |
| FM6 | 3 | 6-operator FM | FM6-specific + WPT projection |

For FM6 mode (osc_mode=3), the pipeline uses:
1. FM6 physics engine (differential equation kernel)
2. WPT projection to coefficient space
3. Standard operators for wavelet processing
4. IDWPT reconstruction

### Band Layout

| Band | Nodes | Coefficients | Character |
|------|-------|--------------|-----------|
| 0 | 0 | 8 | Fundamental structure |
| 1 | 1-2 | 16 | Low-mid body |
| 2 | 3-6 | 32 | Mid presence |
| 3 | 7-12 | 48 | High-mid detail |
| 4 | 13-20 | 64 | Fine texture |
| 5 | 21-31 | 88 | Ultrafine/transients |

---

## Emission Groups

Operators are grouped by when they run during scheduling:

| Group | When | Operators |
|-------|------|-----------|
| **TAIL** | Always (safety) | FineClamp, NormalizeScale, GluePair, SheafProject, EmergencySimplify |
| **BASE** | Always (minimal) | Identity, Tilt, IntraMix |
| **MID** | When `mid_due=True` | DiffuseUp, DiffuseDown, LowRankMorph, GlobalPulse, Impulse, NoiseBand |
| **TOP** | When `top_due=True` | BandShear, SwapBands, RotateNodes, PressureRamp, GateMask, QuasiRotate, MorphMix, FoldPressure, CuspInjection, SwallowtailMask |

Under load, TOP operators are throttled first, then MID. TAIL and BASE always run.

---

## Operator Categories

| Type | Description | Scaling |
|------|-------------|---------|
| **DIAG** | Element-wise scaling | K (cheap to budget) |
| **BANDED** | Within/across band coupling | K |
| **LOWRANK** | Global UV^T patterns | L (expensive) |
| **PERMUTE** | Coefficient reordering | L (expensive) |
| **PROJ** | Energy projection/constraints | L |
| **INJECT** | Signal injection | K |
| **MIX** | Operator blending | Depends on operands |

K = active nodes (degradable), L = total nodes (fixed at 32)

---

## Complete Operator Table

| ID | Name | Type | Group | Parameters | Description |
|----|------|------|-------|------------|-------------|
| 0 | Identity | DIAG | BASE | - | Pass-through, no change |
| 1 | Tilt | DIAG | BASE | `tilt_db` (3.0) | Frequency-dependent gain (dB/octave). Positive boosts fine scales |
| 2 | FineClamp | PROJ | TAIL | `cap_ratio` (0.5) | Caps fine-band energy relative to coarse bands |
| 3 | IntraMix | BANDED | BASE | `mix_strength` (0.15) | Tridiagonal mixing within each node |
| 4 | DiffuseUp | BANDED | MID | `gamma` (0.2) | Fine → coarse energy transport via downsampling |
| 5 | DiffuseDown | BANDED | MID | `gamma` (0.15) | Coarse → fine structure transport via upsampling |
| 6 | BandShear | PERMUTE | TOP | `shear_factor` (1) | Position-dependent rotation within bands |
| 7 | LowRankMorph | LOWRANK | MID | `rank` (2), `seed` (42) | Global coherence via UV^T decomposition |
| 8 | GlobalPulse | INJECT | MID | `pulse_shape` ("coarse_heavy") | Energy injection. Shapes: "coarse_heavy", "mid_peak", "bright" |
| 9 | SwapBands | PERMUTE | TOP | `band_a` (1), `band_b` (3) | Swap two entire bands (dramatic timbre flip) |
| 10 | RotateNodes | PERMUTE | TOP | `rotation` (1) | Rotate node ordering within each band |
| 11 | NormalizeScale | PROJ | TAIL | `target_curve` ("pink") | Per-band normalization. Curves: "flat", "pink", "bright" |
| 12 | GluePair | PROJ | TAIL | `band_a` (2), `band_b` (3), `eta` (0.3) | Sheaf compatibility: a'=(1-eta)a + eta*b |
| 13 | Impulse | INJECT | MID | `positions`, `amp` (0.5) | Sparse attack injection at band starts |
| 14 | NoiseBand | INJECT | MID | `band_id` (4), `amp` (0.1), `seed` (123) | Band-limited Gaussian noise injection |
| 15 | PressureRamp | DIAG | TOP | `ramp_rate` (0.1) | Pressure modulation toward regime transition |
| 16 | GateMask | DIAG | TOP | `pattern` ("sturmian"), `seed` (42) | Binary mask. Patterns: "sturmian", "thue_morse", "random_sparse" |
| 17 | QuasiRotate | PERMUTE | TOP | `period_ratio` (1.618) | Quasiperiodic rotation (golden ratio default) |
| 18 | MorphMix | MIX | TOP | `op_a` (3), `op_b` (7), `weight` (0.5) | Convex combination: T = (1-w)*A + w*B |
| 19 | EmergencySimplify | PROJ | TAIL | - | Hard projection to safe envelope. Kills fine bands |
| 20 | FoldPressure | DIAG | TOP | `fold_band` (2), `pressure` (0.3) | Catastrophe fold setup |
| 21 | CuspInjection | INJECT | TOP | `cusp_strength` (0.4) | Asymmetric perturbation for cusp transition |
| 22 | SwallowtailMask | DIAG | TOP | - | 3-parameter catastrophe gate with triangular wave |
| 23 | SheafProject | PROJ | TAIL | `num_sections` (4) | Local-to-global consistency projection |

---

## Operator Details

### TAIL Operators (Always Run)

#### FineClamp (ID: 2)
Safety projection that prevents fine bands from dominating.
```yaml
- op: "FINE_CLAMP"
  strength_q15: 16384    # 0.5
  param0_q15: 16384      # cap_ratio
```

#### NormalizeScale (ID: 11)
Pushes each band toward a target energy distribution.
```yaml
- op: "NORMALIZE_SCALE"
  strength_q15: 32767    # Full strength
  # target_curve: "pink" (default)
```

#### GluePair (ID: 12)
Enforces coherence between two bands.
```yaml
- op: "GLUE_PAIR"
  strength_q15: 9830     # eta = 0.3
  param0_q15: 2          # band_a
  param1_q15: 3          # band_b
```

#### SheafProject (ID: 23)
Projects onto compatible configurations across overlapping sections.
```yaml
- op: "SHEAF_PROJECT"
  strength_q15: 32767
  param0_q15: 4          # num_sections
```

#### EmergencySimplify (ID: 19)
Last-resort fallback that kills fine bands and normalizes.
```yaml
- op: "EMERGENCY_SIMPLIFY"
  strength_q15: 32767    # Full effect
```

---

### BASE Operators (Always Allowed)

#### Identity (ID: 0)
Pass-through, useful for placeholder or bypass.
```yaml
- op: "IDENTITY"
```

#### Tilt (ID: 1)
Applies frequency-dependent gain. Positive values boost high frequencies.
```yaml
- op: "TILT"
  strength_q15: 14746    # 0.45
  param0_q15: 9830       # tilt_db ~3.0
```

#### IntraMix (ID: 3)
Spreads energy locally within each node using tridiagonal mixing.
```yaml
- op: "INTRA_MIX"
  strength_q15: 4915     # 0.15
```

---

### MID Operators (Gated by mid_due)

#### DiffuseUp (ID: 4)
Transports energy from fine to coarse bands.
```yaml
- op: "DIFFUSE_UP"
  strength_q15: 6554     # gamma = 0.2
```

#### DiffuseDown (ID: 5)
Transports structure from coarse to fine bands.
```yaml
- op: "DIFFUSE_DOWN"
  strength_q15: 4915     # gamma = 0.15
```

#### LowRankMorph (ID: 7)
Imprints global coherent patterns via low-rank UV^T decomposition.
```yaml
- op: "LOWRANK_MORPH"
  strength_q15: 6554     # 0.2
  param0_q15: 2          # rank
  param1_q15: 42         # seed
```

#### GlobalPulse (ID: 8)
Injects energy emphasizing low/mid bands.
```yaml
- op: "GLOBAL_PULSE"
  strength_q15: 8192     # 0.25
  # pulse_shape: "coarse_heavy" | "mid_peak" | "bright"
```

#### Impulse (ID: 13)
Sparse injection at the start of each band (attack transients).
```yaml
- op: "IMPULSE"
  strength_q15: 16384    # amp = 0.5
```

#### NoiseBand (ID: 14)
Injects Gaussian noise into a specific band.
```yaml
- op: "NOISE_BAND"
  strength_q15: 3277     # amp = 0.1
  param0_q15: 4          # band_id
  param1_q15: 123        # seed
```

---

### TOP Operators (Gated by top_due)

#### BandShear (ID: 6)
Rotates coefficients position-dependently within bands.
```yaml
- op: "BAND_SHEAR"
  strength_q15: 32767
  param0_q15: 1          # shear_factor
```

#### SwapBands (ID: 9)
Swaps two entire bands for dramatic timbre changes.
```yaml
- op: "SWAP_BANDS"
  strength_q15: 32767
  param0_q15: 1          # band_a
  param1_q15: 3          # band_b
```

#### RotateNodes (ID: 10)
Rotates node ordering within each band.
```yaml
- op: "ROTATE_NODES"
  strength_q15: 32767
  param0_q15: 1          # rotation amount
```

#### PressureRamp (ID: 15)
Diagonal scaling toward regime transition.
```yaml
- op: "PRESSURE_RAMP"
  strength_q15: 3277     # ramp_rate = 0.1
```

#### GateMask (ID: 16)
Applies deterministic binary mask patterns.
```yaml
- op: "GATE_MASK"
  strength_q15: 32767
  # pattern: "sturmian" | "thue_morse" | "random_sparse"
  param0_q15: 42         # seed
```

#### QuasiRotate (ID: 17)
Quasiperiodic rotation using golden ratio.
```yaml
- op: "QUASI_ROTATE"
  strength_q15: 32767
  param0_q15: 53019      # period_ratio ~1.618 in Q15
```

#### MorphMix (ID: 18)
Blends two operators with a weight.
```yaml
- op: "MORPH_MIX"
  strength_q15: 16384    # weight = 0.5
  param0_q15: 3          # op_a (IntraMix)
  param1_q15: 7          # op_b (LowRankMorph)
```

#### FoldPressure (ID: 20)
Concentrates energy for catastrophe fold bifurcation.
```yaml
- op: "FOLD_PRESSURE"
  strength_q15: 9830     # pressure = 0.3
  param0_q15: 2          # fold_band
```

#### CuspInjection (ID: 21)
Injects asymmetric perturbation for cusp transitions.
```yaml
- op: "CUSP_INJECTION"
  strength_q15: 13107    # cusp_strength = 0.4
```

#### SwallowtailMask (ID: 22)
Complex 3-parameter catastrophe gate pattern.
```yaml
- op: "SWALLOWTAIL_MASK"
  strength_q15: 32767
```

---

## Backbone Examples

### Minimal (Low CPU)
```yaml
backbone:
  - op: "TILT"
    strength_q15: 14746
  - op: "INTRA_MIX"
    strength_q15: 4915
```

### Standard
```yaml
backbone:
  - op: "TILT"
    strength_q15: 14746
  - op: "INTRA_MIX"
    strength_q15: 8192
  - op: "DIFFUSE_DOWN"
    strength_q15: 6554
  - op: "DIFFUSE_UP"
    strength_q15: 4915
```

### Rich (Higher CPU)
```yaml
backbone:
  - op: "TILT"
    strength_q15: 14746
  - op: "INTRA_MIX"
    strength_q15: 8192
  - op: "DIFFUSE_DOWN"
    strength_q15: 6554
  - op: "LOWRANK_MORPH"
    strength_q15: 4915
  - op: "BAND_SHEAR"
    strength_q15: 3277
```

### Glitch/Chaos
```yaml
backbone:
  - op: "TILT"
    strength_q15: 14746
  - op: "GATE_MASK"
    strength_q15: 16384
  - op: "SWAP_BANDS"
    strength_q15: 32767
  - op: "CUSP_INJECTION"
    strength_q15: 13107
```

---

## Operator Name to ID Mapping

```python
{
    "IDENTITY": 0,
    "TILT": 1,
    "FINE_CLAMP": 2,
    "INTRA_MIX": 3,
    "DIFFUSE_UP": 4,
    "DIFFUSE_DOWN": 5,
    "BAND_SHEAR": 6,
    "LOWRANK_MORPH": 7,
    "GLOBAL_PULSE": 8,
    "SWAP_BANDS": 9,
    "ROTATE_NODES": 10,
    "NORMALIZE_SCALE": 11,
    "GLUE_PAIR": 12,
    "IMPULSE": 13,
    "NOISE_BAND": 14,
    "PRESSURE_RAMP": 15,
    "GATE_MASK": 16,
    "QUASI_ROTATE": 17,
    "MORPH_MIX": 18,
    "EMERGENCY_SIMPLIFY": 19,
    "FOLD_PRESSURE": 20,
    "CUSP_INJECTION": 21,
    "SWALLOWTAIL_MASK": 22,
    "SHEAF_PROJECT": 23,
}
```

---

## Runtime Operator Control (WSE-STD-070)

Operators are subject to runtime mask and tail mode control via CMMM events.

### Node Masks and K

Each operator respects the current node mask from the executor latch:

- **K-scaling operators** (DIAG, BANDED, INJECT): Only process active nodes in the mask
- **L-scaling operators** (PERMUTE, LOWRANK, PROJ): Touch all nodes but respect mask for energy calculations

During mask transitions, `dispatch_coeff_op_crossfade()` blends operator outputs between start and target masks.

### Operator Lane Assignment

```python
from symcrash_python_ap.vm.tables import Lane, get_operator_lane

lane_info = get_operator_lane(op_id=7)  # LowRankMorph
# → Lane.LOWRANK
```

| Lane | Operators | Scaling |
|------|-----------|---------|
| DIAG | 0, 1, 15, 16, 20, 22 | K |
| BANDED | 3, 4, 5 | K |
| PERMUTE | 6, 9, 10, 17 | L |
| LOWRANK | 7 | L |
| PROJ | 2, 11, 12, 19, 23 | L |
| INJECT | 8, 13, 14, 21 | K |

### Tail Operators

Tail operators (IDs 2, 11, 19, 23) are controlled by the tail mode latch:

| TailMode | Operators Enabled |
|----------|-------------------|
| FULL (0) | FineClamp, NormalizeScale, SheafProject |
| MID (1) | FineClamp, NormalizeScale |
| MIN (2) | FineClamp |
| EMERGENCY (3) | FineClamp, EmergencySimplify |

CMMM controls tail mode via the `0x31 TAILMODE_SET` event payload.

---

## Related Documentation

- [PRESET-AUTHORING-GUIDE.md](PRESET-AUTHORING-GUIDE.md) - Main authoring guide
- [MACROS-AND-CONTROLS.md](MACROS-AND-CONTROLS.md) - Control system
- [BUDGETS-AND-DEGRADATION.md](BUDGETS-AND-DEGRADATION.md) - Performance budgets
- [../architecture/WSE-STD-070.md](../architecture/WSE-STD-070.md) - Tower completion and CMMM events
