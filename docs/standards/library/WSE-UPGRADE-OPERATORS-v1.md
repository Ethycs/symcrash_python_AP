# WSE-UPGRADE-OPERATORS-v1 — Operator Library Extension

**Standard ID:** WSE-UPGRADE-OPERATORS-v1
**Title:** Operator Library Extension (v0 → v1)
**Version:** 1.0
**Status:** DRAFT
**Scope:** Defines 6 new operators (IDs 24-29) extending WSE-OPLIB-24 v0 to WSE-OPLIB-30 v1
**Depends on:** WSE-STD-001 (Fixed-Point), WSE-STD-030 (Operators), OPERATORS.md
**Related:** WSE-STD-PRESET-v0 (preset compatibility), operator_costs.py (budget integration)

---

## 0. Motivation

The existing 24-operator library covers core wavelet coefficient manipulation but lacks:
1. **Vocal/Speech sounds** — No formant synthesis capability
2. **Spectral freezing** — No ability to hold/smear spectral content
3. **Resonance/Delay** — No comb filtering for pitched resonance
4. **Brightness enhancement** — No harmonic exciter
5. **Pitch manipulation** — No non-integer harmonic stretching
6. **Transient shaping** — No attack/sustain separation

This upgrade adds 6 musically essential operators while maintaining backward compatibility with v0 presets.

---

## 1. Compatibility

### 1.1 Version Transition

| Library Version | Operator IDs | Presets Targeting |
|-----------------|--------------|-------------------|
| WSE-OPLIB-24 v0 | 0-23 | Existing presets |
| WSE-OPLIB-30 v1 | 0-29 | v0 presets + new |

### 1.2 Backward Compatibility

- All v0 operator IDs (0-23) remain unchanged
- v0 presets compile and execute identically
- New operators are additive (no semantic changes to existing)
- Preset compiler defaults to v0 when no new operators are used

### 1.3 Preset Version Declaration

```yaml
# New preset using v1 operators MUST declare:
compat:
  operator_lib_id: "WSE-OPLIB-30 v1"
```

---

## 2. New Operator Specifications

### 2.1 Formant (ID: 24)

**Category:** BANDED
**Group:** MID
**Musical Purpose:** Vowel/voice formant synthesis

Applies frequency-dependent gain curves to create vowel formants.
Uses 3-5 resonant peaks mapped to coefficient bands.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `formant_id` | u8 | 0-9 | 0 | Preset vowel (0=A, 1=E, 2=I, 3=O, 4=U, 5-9=custom) |
| `formant_shift_q15` | i16 | -32767..32767 | 0 | Formant frequency shift (+/- octave) |
| `resonance_q15` | i16 | 0..32767 | 16384 | Resonance/Q (0=flat, 32767=sharp peaks) |

#### Formant Presets (Coefficient Band Mapping)

| ID | Vowel | Band 0 | Band 1 | Band 2 | Band 3 | Band 4 | Band 5 |
|----|-------|--------|--------|--------|--------|--------|--------|
| 0 | A | 1.0 | 0.9 | 0.7 | 0.8 | 0.3 | 0.1 |
| 1 | E | 0.8 | 0.6 | 1.0 | 0.9 | 0.5 | 0.2 |
| 2 | I | 0.5 | 0.3 | 0.8 | 1.0 | 0.9 | 0.4 |
| 3 | O | 1.0 | 0.8 | 0.4 | 0.3 | 0.2 | 0.1 |
| 4 | U | 1.0 | 0.6 | 0.2 | 0.15 | 0.1 | 0.05 |

#### YAML Example

```yaml
- op: "FORMANT"
  strength_q15: 24576      # 0.75
  param0_q15: 0            # formant_id = A
  param1_q15: 0            # no shift
```

#### Algorithm (Pseudocode)

```python
def formant_op(coeffs: np.ndarray, params: FormantParams) -> np.ndarray:
    """Apply formant resonance to coefficient bands."""
    # Get formant curve for selected vowel
    curve = FORMANT_CURVES[params.formant_id]

    # Apply frequency shift (shift curve index)
    shift_bands = (params.formant_shift_q15 * 6) >> 15  # Map to band offset

    # Apply per-band gain with resonance
    for band_idx in range(6):
        band_start, band_end = BAND_RANGES[band_idx]

        # Get formant gain for this band (with shift)
        src_band = (band_idx - shift_bands) % 6
        gain_q15 = curve[src_band]

        # Apply resonance sharpening (Q factor)
        # High resonance = more contrast between peaks and valleys
        if params.resonance_q15 > 16384:
            # Sharpen: boost peaks, cut valleys
            gain_q15 = RSHIFT_RNE(gain_q15 * params.resonance_q15, 14)

        # Apply to band
        for i in range(band_start, band_end):
            coeffs[i] = RSHIFT_RNE(coeffs[i] * gain_q15, 15)

    return coeffs
```

#### Cost

```python
OperatorCost(
    op_id=24,
    op_name="Formant",
    category=OpCategory.BANDED,
    ticks_base=24,
    ticks_per_active_node=16,
    ticks_per_total_node=0,
    nonlinear=False,
    is_tail=False,
    drop_priority=100,  # Important for voice presets
)
```

---

### 2.2 SpectralFreeze (ID: 25)

**Category:** PROJ
**Group:** MID
**Musical Purpose:** Spectral hold/smear effect (freeze, blur)

Holds coefficient magnitudes while allowing phase to evolve.
Creates "frozen" or time-smeared timbres.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `freeze_amount_q15` | i16 | 0..32767 | 0 | Freeze blend (0=bypass, 32767=full freeze) |
| `decay_rate_q15` | i16 | 0..32767 | 1638 | Frozen magnitude decay per block (0=hold forever) |

#### Algorithm (Pseudocode)

```python
@dataclass
class SpectralFreezeState:
    """Per-voice state for spectral freeze."""
    frozen_magnitudes: np.ndarray  # [256] stored |coeff| values
    is_frozen: bool = False

def spectral_freeze_op(coeffs: np.ndarray, params: FreezeParams,
                       state: SpectralFreezeState) -> np.ndarray:
    """Hold spectral content while phase evolves."""

    # Update frozen magnitudes (IIR low-pass on magnitude)
    current_mags = np.abs(coeffs)

    if params.freeze_amount_q15 > 0:
        # Blend toward frozen
        blend = params.freeze_amount_q15
        inv_blend = Q15_ONE - blend

        if not state.is_frozen:
            # Capture current spectrum
            state.frozen_magnitudes = current_mags.copy()
            state.is_frozen = True
        else:
            # Apply decay to frozen magnitudes
            state.frozen_magnitudes = RSHIFT_RNE(
                state.frozen_magnitudes * (Q15_ONE - params.decay_rate_q15), 15
            )

        # Output = blend(current, frozen)
        target_mags = RSHIFT_RNE(
            current_mags * inv_blend + state.frozen_magnitudes * blend, 15
        )

        # Apply magnitude while preserving sign
        for i in range(len(coeffs)):
            sign = 1 if coeffs[i] >= 0 else -1
            coeffs[i] = sign * target_mags[i]

    return coeffs
```

#### YAML Example

```yaml
- op: "SPECTRAL_FREEZE"
  strength_q15: 24576      # 0.75 freeze amount
  param0_q15: 819          # 2.5% decay per block
```

#### Cost

```python
OperatorCost(
    op_id=25,
    op_name="SpectralFreeze",
    category=OpCategory.PROJ,
    ticks_base=64,
    ticks_per_active_node=16,  # Magnitude capture
    ticks_per_total_node=8,    # State update
    nonlinear=True,            # Uses abs()
    is_tail=False,
    drop_priority=180,  # Effect operator
)
```

---

### 2.3 Comb (ID: 26)

**Category:** BANDED
**Group:** MID
**Musical Purpose:** Comb filtering for metallic/pitched resonance

Applies comb filter response in coefficient domain.
Creates flanging, chorus, or metallic resonance effects.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `comb_period_q15` | i16 | 1638..32767 | 8192 | Comb spacing (maps to coefficient offset) |
| `feedback_q15` | i16 | 0..32767 | 16384 | Resonance/feedback (0=single, 32767=infinite) |
| `mix_q15` | i16 | 0..32767 | 16384 | Dry/wet mix |

#### Algorithm (Pseudocode)

```python
def comb_op(coeffs: np.ndarray, params: CombParams,
            delay_buffer: np.ndarray) -> np.ndarray:
    """Apply comb filter in coefficient domain."""

    # Compute delay offset (period maps to coefficient spacing)
    delay_samples = (params.comb_period_q15 * 64) >> 15  # 2-64 coeff delay
    delay_samples = max(2, min(64, delay_samples))

    output = np.zeros_like(coeffs)

    for i in range(len(coeffs)):
        # Feedforward: current + delayed
        delayed_idx = (i - delay_samples) % len(coeffs)
        delayed_val = delay_buffer[delayed_idx]

        # Apply feedback
        fb_contribution = RSHIFT_RNE(delayed_val * params.feedback_q15, 15)

        # Mix: dry + wet*comb
        dry = RSHIFT_RNE(coeffs[i] * (Q15_ONE - params.mix_q15), 15)
        wet = RSHIFT_RNE((coeffs[i] + fb_contribution) * params.mix_q15, 15)
        output[i] = SAT16(dry + wet)

        # Update delay buffer
        delay_buffer[i] = coeffs[i]

    return output
```

#### YAML Example

```yaml
- op: "COMB"
  strength_q15: 32767      # Full effect
  param0_q15: 8192         # comb_period (0.25 range)
  param1_q15: 20480        # feedback (0.625)
```

#### Cost

```python
OperatorCost(
    op_id=26,
    op_name="Comb",
    category=OpCategory.BANDED,
    ticks_base=32,
    ticks_per_active_node=24,  # Delay + feedback
    ticks_per_total_node=0,
    nonlinear=False,
    is_tail=False,
    drop_priority=170,
)
```

---

### 2.4 Exciter (ID: 27)

**Category:** DIAG
**Group:** MID
**Musical Purpose:** Harmonic exciter/enhancer for brightness

Generates and adds upper harmonics to existing content.
Creates "air" and presence without harsh EQ boost.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `drive_q15` | i16 | 0..32767 | 8192 | Excitation drive (harmonic generation) |
| `mix_q15` | i16 | 0..32767 | 8192 | Excited signal mix |
| `band_focus` | u8 | 3-5 | 4 | Target band for excitation (3=mid, 4=high, 5=air) |

#### Algorithm (Pseudocode)

```python
def exciter_op(coeffs: np.ndarray, params: ExciterParams) -> np.ndarray:
    """Generate upper harmonics for brightness enhancement."""

    # Get source band (one below target)
    src_band = params.band_focus - 1
    src_start, src_end = BAND_RANGES[src_band]

    # Get target band
    dst_start, dst_end = BAND_RANGES[params.band_focus]

    # Generate harmonics via soft clipping of source
    for i, src_i in enumerate(range(src_start, src_end)):
        if i >= (dst_end - dst_start):
            break
        dst_i = dst_start + i

        # Soft clip source to generate harmonics
        src_val = coeffs[src_i]
        driven = RSHIFT_RNE(src_val * params.drive_q15, 14)  # 2x headroom

        # Soft saturation: tanh approximation
        if driven > 16384:
            excited = 16384 + ((driven - 16384) >> 2)
        elif driven < -16384:
            excited = -16384 + ((driven + 16384) >> 2)
        else:
            excited = driven

        # Mix into target band
        existing = coeffs[dst_i]
        mixed = RSHIFT_RNE(existing * (Q15_ONE - params.mix_q15), 15)
        mixed += RSHIFT_RNE(excited * params.mix_q15, 15)
        coeffs[dst_i] = SAT16(mixed)

    return coeffs
```

#### YAML Example

```yaml
- op: "EXCITER"
  strength_q15: 12288      # 0.375
  param0_q15: 16384        # drive (0.5)
  param1_q15: 8192         # mix (0.25)
```

#### Cost

```python
OperatorCost(
    op_id=27,
    op_name="Exciter",
    category=OpCategory.DIAG,
    ticks_base=24,
    ticks_per_active_node=16,
    ticks_per_total_node=0,
    nonlinear=True,  # Soft clip
    is_tail=False,
    drop_priority=150,
)
```

---

### 2.5 HarmonicStretch (ID: 28)

**Category:** LOWRANK
**Group:** TOP
**Musical Purpose:** Non-integer harmonic stretching for inharmonicity

Shifts harmonic relationships for bell-like or prepared piano timbres.
Essential for realistic acoustic instrument modeling.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `stretch_factor_q15` | i16 | 0..32767 | 0 | Stretch amount (0=harmonic, 32767=2x inharmonic) |
| `stretch_curve` | u8 | 0-3 | 0 | Curve type (0=linear, 1=quadratic, 2=sqrt, 3=exp) |

#### Algorithm (Pseudocode)

```python
def harmonic_stretch_op(coeffs: np.ndarray, params: StretchParams) -> np.ndarray:
    """Apply inharmonic stretching to coefficient positions."""

    output = np.zeros_like(coeffs)

    # Compute stretch mapping for each coefficient
    for i in range(len(coeffs)):
        # Normalized position (0-1)
        pos_norm = i / len(coeffs)

        # Apply stretch curve
        if params.stretch_curve == 0:  # Linear
            stretch = pos_norm * params.stretch_factor_q15
        elif params.stretch_curve == 1:  # Quadratic (bells)
            stretch = (pos_norm ** 2) * params.stretch_factor_q15
        elif params.stretch_curve == 2:  # Sqrt (gentle)
            stretch = (pos_norm ** 0.5) * params.stretch_factor_q15
        else:  # Exponential
            stretch = ((2 ** pos_norm) - 1) * params.stretch_factor_q15

        # Compute target position (fractional)
        target_pos = i + (stretch * i) / Q15_ONE

        # Interpolate to output
        idx_lo = int(target_pos) % len(coeffs)
        idx_hi = (idx_lo + 1) % len(coeffs)
        frac = target_pos - int(target_pos)

        output[idx_lo] += RSHIFT_RNE(coeffs[i] * (Q15_ONE - int(frac * Q15_ONE)), 15)
        output[idx_hi] += RSHIFT_RNE(coeffs[i] * int(frac * Q15_ONE), 15)

    return output
```

#### YAML Example

```yaml
- op: "HARMONIC_STRETCH"
  strength_q15: 32767      # Full effect
  param0_q15: 3277         # 10% stretch
  param1_q15: 1            # quadratic curve (bell-like)
```

#### Cost

```python
OperatorCost(
    op_id=28,
    op_name="HarmonicStretch",
    category=OpCategory.LOWRANK,
    ticks_base=64,
    ticks_per_active_node=0,
    ticks_per_total_node=32,  # Global resampling
    nonlinear=True,  # Uses power functions
    is_tail=False,
    drop_priority=240,  # Expensive, artistic
)
```

---

### 2.6 TransientShape (ID: 29)

**Category:** DIAG
**Group:** MID
**Musical Purpose:** Attack/sustain separation and shaping

Separates transient and steady-state components for independent processing.
Enables attack boost/cut without affecting sustain.

#### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `attack_gain_q15` | i16 | 0..65534 | 32767 | Attack multiplier (32767=unity, >32767=boost) |
| `sustain_gain_q15` | i16 | 0..32767 | 32767 | Sustain multiplier |
| `threshold_q15` | i16 | 0..32767 | 8192 | Transient detection threshold |

#### Algorithm (Pseudocode)

```python
@dataclass
class TransientState:
    """Per-voice state for transient detection."""
    envelope_q15: np.ndarray  # [256] smoothed magnitude
    attack_flag: np.ndarray   # [256] bool: currently in attack

def transient_shape_op(coeffs: np.ndarray, params: TransientParams,
                       state: TransientState) -> np.ndarray:
    """Shape attack and sustain independently."""

    ATTACK_FAST = 26214   # ~0.8 fast follower
    RELEASE_SLOW = 3277   # ~0.1 slow release

    for i in range(len(coeffs)):
        mag = abs(coeffs[i])

        # Envelope follower (attack fast, release slow)
        if mag > state.envelope_q15[i]:
            # Attack: fast rise
            state.envelope_q15[i] += RSHIFT_RNE(
                (mag - state.envelope_q15[i]) * ATTACK_FAST, 15
            )
            state.attack_flag[i] = True
        else:
            # Release: slow decay
            state.envelope_q15[i] += RSHIFT_RNE(
                (mag - state.envelope_q15[i]) * RELEASE_SLOW, 15
            )
            state.attack_flag[i] = False

        # Detect transient: current > envelope + threshold
        is_transient = mag > state.envelope_q15[i] + params.threshold_q15

        # Apply gain based on transient state
        if is_transient or state.attack_flag[i]:
            gain = params.attack_gain_q15
        else:
            gain = params.sustain_gain_q15

        # Apply (handle >32767 for boost)
        if gain > 32767:
            coeffs[i] = SAT16(RSHIFT_RNE(coeffs[i] * gain, 15))
        else:
            coeffs[i] = RSHIFT_RNE(coeffs[i] * gain, 15)

    return coeffs
```

#### YAML Example

```yaml
- op: "TRANSIENT_SHAPE"
  strength_q15: 32767      # Full effect
  param0_q15: 49151        # attack_gain = 1.5x boost
  param1_q15: 24576        # sustain_gain = 0.75
```

#### Cost

```python
OperatorCost(
    op_id=29,
    op_name="TransientShape",
    category=OpCategory.DIAG,
    ticks_base=32,
    ticks_per_active_node=24,  # Envelope + detection
    ticks_per_total_node=0,
    nonlinear=True,  # abs(), comparisons
    is_tail=False,
    drop_priority=130,
)
```

---

## 3. Summary Table

| ID | Name | Category | Group | Musical Use | Priority |
|----|------|----------|-------|-------------|----------|
| 24 | Formant | BANDED | MID | Vowels, voice synthesis | 100 |
| 25 | SpectralFreeze | PROJ | MID | Freeze, smear, hold | 180 |
| 26 | Comb | BANDED | MID | Metallic resonance, flange | 170 |
| 27 | Exciter | DIAG | MID | Air, brightness, presence | 150 |
| 28 | HarmonicStretch | LOWRANK | TOP | Bells, inharmonicity | 240 |
| 29 | TransientShape | DIAG | MID | Attack/sustain control | 130 |

---

## 4. Implementation Priority

### Phase 1: High Value (Implement First)
1. **Formant (24)** — Enables vocal/choir presets, unique capability
2. **TransientShape (29)** — Attack shaping is fundamental for expression

### Phase 2: Medium Value
3. **Exciter (27)** — Brightness without harsh EQ
4. **Comb (26)** — Metallic and resonant textures

### Phase 3: Specialized
5. **SpectralFreeze (25)** — Effect/experimental
6. **HarmonicStretch (28)** — Acoustic modeling (bells, piano)

---

## 5. Preset Examples

### 5.1 Choir Pad (using Formant)

```yaml
preset:
  preset_id: "choir-pad-v1"
  preset_version: "v1"

compat:
  operator_lib_id: "WSE-OPLIB-30 v1"

backbone:
  - op: "TILT"
    strength_q15: 9830       # Warm tilt
  - op: "FORMANT"
    strength_q15: 24576      # Strong formant
    param0_q15: 0            # Vowel A
  - op: "DIFFUSE_DOWN"
    strength_q15: 6554
```

### 5.2 Punchy Synth (using TransientShape)

```yaml
backbone:
  - op: "TILT"
    strength_q15: 14746
  - op: "TRANSIENT_SHAPE"
    strength_q15: 32767
    param0_q15: 49151        # Attack boost 1.5x
    param1_q15: 24576        # Sustain cut 0.75
  - op: "NORMALIZE_SCALE"
    strength_q15: 32767
```

### 5.3 Metallic Bell (using HarmonicStretch + Comb)

```yaml
backbone:
  - op: "HARMONIC_STRETCH"
    strength_q15: 32767
    param0_q15: 4915         # 15% stretch
    param1_q15: 1            # Quadratic curve
  - op: "COMB"
    strength_q15: 16384
    param0_q15: 10922        # Medium spacing
    param1_q15: 24576        # High feedback
```

---

## 6. Verification Requirements

### 6.1 Unit Tests

For each new operator (24-29):
1. Identity test: strength_q15=0 → output equals input
2. Full strength test: strength_q15=32767 → expected transform
3. Parameter bounds test: min/max params produce valid output
4. Saturation test: no overflow/wrap on extreme input

### 6.2 Integration Tests

1. **Backward compatibility**: All v0 presets compile and produce identical output
2. **New operator preset**: Each new operator works in full preset context
3. **Degradation**: New operators respect drop_priority under budget pressure

### 6.3 Audio Tests

1. **Formant**: A/E/I/O/U vowels audibly distinct
2. **SpectralFreeze**: Sustained note → frozen timbre on freeze_amount>0
3. **Comb**: Pitched resonance at expected frequency
4. **Exciter**: Increased brightness without harshness
5. **HarmonicStretch**: Bell-like inharmonicity on stretch>0
6. **TransientShape**: Attack boost audible on percussive sounds

---

## 7. Migration Notes

### 7.1 Updating Existing Code

```python
# operator_costs.py additions:
OPERATOR_COSTS[24] = OperatorCost(op_id=24, op_name="Formant", ...)
OPERATOR_COSTS[25] = OperatorCost(op_id=25, op_name="SpectralFreeze", ...)
OPERATOR_COSTS[26] = OperatorCost(op_id=26, op_name="Comb", ...)
OPERATOR_COSTS[27] = OperatorCost(op_id=27, op_name="Exciter", ...)
OPERATOR_COSTS[28] = OperatorCost(op_id=28, op_name="HarmonicStretch", ...)
OPERATOR_COSTS[29] = OperatorCost(op_id=29, op_name="TransientShape", ...)
```

### 7.2 Name-to-ID Mapping Extension

```python
OPERATOR_NAME_TO_ID.update({
    "FORMANT": 24,
    "SPECTRAL_FREEZE": 25,
    "COMB": 26,
    "EXCITER": 27,
    "HARMONIC_STRETCH": 28,
    "TRANSIENT_SHAPE": 29,
})
```

### 7.3 State Management

SpectralFreeze, Comb, and TransientShape require per-voice state:

```python
@dataclass
class ExtendedVoiceState:
    """Extended state for v1 operators."""
    # SpectralFreeze
    freeze_magnitudes: np.ndarray = None  # [256]
    freeze_active: bool = False

    # Comb
    comb_delay_buffer: np.ndarray = None  # [256]

    # TransientShape
    transient_envelope: np.ndarray = None  # [256]
    transient_attack_flag: np.ndarray = None  # [256] bool
```

---

## 8. Future Considerations

### 8.1 Potential v2 Operators (IDs 30+)

| ID | Name | Purpose |
|----|------|---------|
| 30 | Bitcrush | Lo-fi, retro textures |
| 31 | RingMod | Ring modulation |
| 32 | Granular | Micro-slice manipulation |
| 33 | Wavefold | Wave folding distortion |

### 8.2 Operator Chaining

Future standard may define operator chains as first-class presets:

```yaml
chains:
  - id: "vocal_chain"
    ops: [24, 27, 11]  # Formant → Exciter → Normalize
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-14
**Author:** CCS Development Team
**Status:** DRAFT
