# WSE-STD-MATERIAL v0 — Material Engine & Analog Bus

**Status:** DRAFT → FROZEN
**Applies to:** Python reference VM + FPGA material processor
**Depends on:** WSE-STD-001 (Q1.15 arithmetic), WSE-STD-020 (VoiceFrame), WSE-STD-030 (Tail Safety)

---

## 0. Purpose & Scope

This standard defines the **Material Engine**: a post-symbolic physical modeling layer that applies analog-inspired processing to mixed audio.

**Core Principle:**
> "Analog sound is not about topology or geometry — it is about **low-dimensional nonlinear dynamical systems** under strict energy and bandwidth constraints."

The Material Engine is the boundary between **symbolic structure** (CMMM/wavelet) and **physical behavior** (tape, catastrophe, texture).

**What this standard defines:**
- 6 normative material programs (material_id 0-5)
- State machine specifications per program
- Parameter slewing protocol for click-free transitions
- Safety contract (bounded output, no wrap/NaN)
- Bus placement and topology
- Determinism requirements (bit-exact Q1.15)

**What this standard does NOT define:**
- Symbolic layer structure (see WSE-STD-200 CMMM)
- Wavelet synthesis (see WSE-STD-120 Polyphase)
- Budget/scheduling (see WSE-STD-DEGRADE)

---

## 1. Architectural Placement

### 1.1 Pipeline Location (MUST)

```
Symbol Engine (CMMM)
    ↓ (coefficient evolution)
Wavelet Engine (IDWPT)
    ↓ (audio reconstruction)
Voice Mix Stage
    ↓ (sum all voices → stereo)
━━━━━━━━━━━━━━━━━━━━━━━━
MATERIAL ENGINE ← HERE
━━━━━━━━━━━━━━━━━━━━━━━━
    ↓ (analog processing)
Safety/Limiter (Tail Stage)
    ↓ (bounded output)
Audio Output
```

**Rule (MUST):** Material Engine processes **after** voice mix, **before** safety/limiter.

### 1.2 Bus Topology (v0)

**Single stereo stream** (cheapest, most "ensemble glue").

```
Input:  L[64], R[64]  (mixed voices, Q1.15)
Output: L'[64], R'[64] (material-processed, Q1.15)
```

**Rationale:**
- Cheapest (one stereo stream, not per-voice)
- Most "ensemble" feel (intermodulation across voices)
- Easiest to oversample once (if needed)
- Allows global tape/saturation "glue"

**Rule (MUST):** Material Engine operates on **stereo pairs** with independent per-channel state.

### 1.3 Block-Rate Processing (MUST)

- Block size `B = 64` samples
- Sample rate `fs = 48000` Hz
- Block rate `fb = 750` Hz (48000 / 64)
- Parameter updates at block rate only

---

## 2. Material Program Interface

### 2.1 Universal Form (MUST)

Every material program is a **stateful dynamical system**:

```
(state', output) = MaterialProgram(state, input, params)
```

Where:
- `state`: Internal memory (per-channel L/R)
- `input`: Audio block L[64], R[64] (Q1.15)
- `params`: MaterialParams (slewed, block-rate)
- `output`: Processed audio L'[64], R'[64] (Q1.15)

### 2.2 Program Selection (v0)

Material programs selected via `material_id` (6-bit selector):

| material_id | Name | Category | State Type |
|-------------|------|----------|------------|
| 0 | Clean Glue | Transparent | None |
| 1 | Tape Bus | Hysteresis + Tone | TapeState |
| 2 | Rank Grit | Order Statistics | RankState |
| 3 | Tropical Cut | Hard Threshold | None |
| 4 | Catastrophe Fold | A2 Bifurcation | CatastropheState |
| 5 | Catastrophe Cusp | A3 Bifurcation | CatastropheState |

**Rule (MUST):** Implementation MUST support all 6 programs with bit-exact behavior.

**Default (MUST):** `material_id=0` (Clean Glue) when unspecified.

### 2.3 Per-Block Switching (MAY)

Material programs MAY be switched per-block.

**If supported:**
- State MUST be preserved across switches (each program has independent state)
- Switching MUST NOT cause clicks (parameter slewing still applies)
- Previous program state MUST remain available if switched back

---

## 3. Fixed-Point Format (Q1.15)

**All material processing uses WSE-STD-001 Q1.15 arithmetic.**

### 3.1 Data Types (MUST)

| Quantity | Type | Range | Notes |
|----------|------|-------|-------|
| Audio samples | `i16 Q1.15` | [-1.0, 1.0] | ±32767 |
| Parameters | `i16 Q1.15` | [-1.0, 1.0] | drive, bias, pressure, tone, cut |
| State variables | `i16 Q1.15` | [-1.0, 1.0] | magnetization, slow_state |
| Indices/counters | `u8` or `u16` | Integer | Buffer indices, regime IDs |

### 3.2 Arithmetic Rules (MUST)

**Multiply:**
```
MUL_Q15(a, b) = SAT16(RSHIFT_RNE((s64)a × (s64)b, 15))
```

**Add/Subtract:**
```
ADD_Q15(a, b) = SAT16((s32)a + (s32)b)
```

**Saturation:**
```
SAT16(x) = CLAMP(x, -32768, 32767)
```

**Rounding:** Round-to-nearest, ties-to-even (RNE).

**Reference:** WSE-STD-001 for complete arithmetic specification.

---

## 4. Material State Specification

### 4.1 State Structures (MUST)

**TapeState** (for material_id=1):
```
struct TapeState {
    i16 magnetization_q15;    // Current magnetization state
}
```

**CatastropheState** (for material_id=4,5):
```
struct CatastropheState {
    i16 slow_state_q15;       // Slow dynamical variable
    u8  regime_id;            // Bifurcation regime (0-255)
}
```

**RankState** (for material_id=2):
```
struct RankState {
    i16 buffer[17];           // Circular sample buffer
    u8  index;                // Write position (0-16)
}
```

### 4.2 Per-Channel State (MUST)

All stateful programs MUST maintain **independent state for L and R channels**.

```
MaterialEngineState {
    TapeState tape_L, tape_R;
    CatastropheState cat_L, cat_R;
    RankState rank_L, rank_R;

    // Tone filter state (shared across programs)
    i16 lp_state_L_q15;
    i16 lp_state_R_q15;
}
```

### 4.3 State Initialization (MUST)

**Default initial state (all zeros):**
- `magnetization_q15 = 0`
- `slow_state_q15 = 0`
- `regime_id = 0`
- `buffer[] = {0, 0, ..., 0}`
- `index = 0`
- `lp_state_L/R_q15 = 0`

**Rule (MUST):** State initialization MUST be deterministic and independent of platform.

---

## 5. Parameter Slewing (MUST)

### 5.1 MaterialParams Structure

```
struct MaterialParams {
    i16 drive_q15;       // Drive/energy into material (0 to 32767)
    i16 tone_q15;        // Tone control (0=dark, 32767=bright)
    i16 bias_q15;        // DC bias offset (-32768 to 32767)
    i16 pressure_q15;    // Catastrophe pressure parameter
    i16 cut_q15;         // Tropical cut threshold (0 to 32767)
}
```

### 5.2 Slew Update Law (MUST)

Parameters MUST slew toward target using:

```
param_new = param_old + RSHIFT_RNE((target - param_old) × rate, 15)
```

Where `rate` is a Q0.15 slew coefficient (default `rate = 256` for gentle slew).

**Normative slew_from() behavior** (per block):
```python
def slew_from(current: MaterialParams, target: ParamFrame, rate: int = 256):
    """
    Slew current parameters toward target.

    rate: Q0.15 slew rate (256 = ~0.0078 per block = gentle)
    Higher rate → faster slew, lower rate → slower slew
    """
    current.drive_q15 = slew_q15(current.drive_q15, target.drive_q15, rate)
    current.tone_q15 = slew_q15(current.tone_q15, target.tone_q15, rate)
    current.bias_q15 = slew_q15(current.bias_q15, target.bias_q15, rate)
    current.pressure_q15 = slew_q15(current.pressure_q15, target.pressure_q15, rate)
    current.cut_q15 = slew_q15(current.cut_q15, target.cut_q15, rate)
```

**Rule (MUST):** All parameter changes MUST slew (no instantaneous jumps).

**Purpose:** Click-free transitions, smooth analog feel.

---

## 6. Material Program Specifications

### 6.1 Program 0: Clean Glue (Transparent)

**Category:** Memoryless nonlinearity
**State:** None
**Parameters Used:** None (fully transparent except gentle clip)

**Algorithm:**
```
For each sample n in [0, 63]:
    if |L[n]| < 30000:
        L'[n] = L[n]
    else:
        L'[n] = soft_clip_q15(L[n], threshold=30000)

    (same for R channel)
```

**Purpose:** Transparent pass-through with protective soft clipping only near digital maximum.

**Complexity:** ~1 tick per sample (negligible).

---

### 6.2 Program 1: Tape Bus (Hysteresis + Tone)

**Category:** Stateful nonlinearity + LTI filter
**State:** `TapeState` (magnetization_q15)
**Parameters Used:** `drive_q15`, `bias_q15`, `tone_q15`

**Hysteresis Model (normative):**
```
m[n+1] = m[n] + λ(tanh(drive × x[n] + bias) - m[n]) × (1 - ρ × m[n]²)

Where:
    λ (feedback rate) = min(32767, drive + 16384)  // 0.5 to 1.5 range
    ρ (saturation) = drive >> 2                    // Gentle saturation factor
```

**Tone Filter (one-pole lowpass):**
```
y_tone[n] = α × y_raw[n] + (1 - α) × lp_state[n-1]
lp_state[n] = y_tone[n]

Where:
    α = tone_q15  // tone=0 → full LP, tone=32767 → bypass
```

**Full Algorithm:**
```
For each sample n:
    1. driven = MUL_Q15(x[n], drive) + bias
    2. target = tanh_q15(driven)
    3. m_sq = MUL_Q15(m, m)
    4. damp = 32767 - MUL_Q15(ρ, m_sq)
    5. delta = MUL_Q15(target - m, λ) × MUL_Q15(delta_raw, damp)
    6. m = SAT16(m + (delta >> 1))
    7. y_raw = m
    8. y = tone_filter(y_raw)  // Apply one-pole lowpass
```

**Complexity:** ~20 ticks per sample.

**Safety:** Output bounded by tanh saturation + soft clip in tone filter.

---

### 6.3 Program 2: Rank Grit (Order Statistics)

**Category:** Nonlinear temporal filter
**State:** `RankState` (circular buffer[17], index)
**Parameters Used:** `drive_q15`

**Algorithm:**
```
Window size:
    W = 17 if drive < 16384 else 9  // Drive controls window (gritty vs smooth)

For each sample n:
    1. buffer[index] = x[n]
    2. index = (index + 1) % W
    3. sorted = sort(buffer[0:W])
    4. trimmed = sorted[2:-2]  // Drop top/bottom 2 outliers
    5. filtered = mean(trimmed)
    6. y[n] = MUL_Q15(filtered, drive) + MUL_Q15(x[n], 32767 - drive)  // Dry/wet mix
```

**Purpose:** Rank-based smoothing/grit. Median-like behavior with trimmed mean robustness.

**Complexity:** ~40 ticks per sample (sort dominates).

**Safety:** Output bounded by input range (trimmed mean preserves bounds).

---

### 6.4 Program 3: Tropical Cut (Hard Threshold)

**Category:** Memoryless threshold
**State:** None
**Parameters Used:** `cut_q15`

**Algorithm:**
```
For each sample n:
    if |x[n]| < cut:
        y[n] = 0
    else:
        y[n] = x[n]
```

**Purpose:** Hard gate/cut for aggressive dynamics. Tropical algebra (min/max operations).

**Complexity:** ~1 tick per sample.

**Safety:** Output bounded by input (subset).

---

### 6.5 Program 4: Catastrophe Fold (A2 Bifurcation)

**Category:** Stateful nonlinearity (catastrophe theory)
**State:** `CatastropheState` (slow_state_q15)
**Parameters Used:** `pressure_q15`

**Mathematical Model:**

Fold catastrophe potential:
```
V(x) = x³ + a×x

Where:
    x = slow state variable
    a = pressure (control parameter)
```

**Dynamics:**
```
x' = -∂V/∂x = -3x² - a

Discretized update:
    s[n+1] = s[n] + rate × (-s³[n] - pressure × s[n] + input_scaled[n])

Where:
    rate = 16384  (Q1.15 = 0.5, integration rate)
    input_scaled = x[n] >> 2  (reduce input amplitude)
```

**Output:**
```
y[n] = soft_clip_q15(s[n])
```

**Purpose:** Single-parameter bifurcation. "Snap" behavior as pressure crosses critical value.

**Complexity:** ~15 ticks per sample.

**Safety:** Soft clip ensures bounded output.

---

### 6.6 Program 5: Catastrophe Cusp (A3 Bifurcation)

**Category:** Stateful nonlinearity (catastrophe theory)
**State:** `CatastropheState` (slow_state_q15)
**Parameters Used:** `pressure_q15`, `bias_q15`

**Mathematical Model:**

Cusp catastrophe potential:
```
V(x) = x⁴ + a×x² + b×x

Where:
    x = slow state variable
    a = pressure (control parameter 1)
    b = bias (control parameter 2)
```

**Dynamics:**
```
x' = -∂V/∂x = -4x³ - 2a×x - b

Discretized update:
    grad = -4×s³[n] - 2×pressure×s[n] - bias
    s[n+1] = s[n] + rate × (grad + input_scaled[n])
```

**Output:**
```
y[n] = soft_clip_q15(s[n])
```

**Purpose:** Dual-parameter bifurcation. Complex hysteresis behaviors with pressure+bias control.

**Complexity:** ~20 ticks per sample.

**Safety:** Soft clip ensures bounded output.

---

## 7. Tape Bus Detailed Specification (material_id=1)

### 7.1 Hysteresis Model (normative)

**State equation:**
```
m[n+1] = m[n] + λ × (tanh(drive × x[n] + bias) - m[n]) × (1 - ρ × m[n]²)

Where:
    λ (feedback rate) = min(32767, drive + 16384)
        // drive=0 → λ=0.5, drive=16384 → λ=1.0, drive=32767 → λ=1.5

    ρ (saturation factor) = drive >> 2
        // Higher drive → stronger saturation limiting
```

**Physical interpretation:**
- `m`: Magnetic state (tape magnetization)
- `λ`: How fast tape responds to signal
- `ρ`: Magnetic saturation strength
- `tanh(drive × x + bias)`: Target magnetization for current input
- `(1 - ρ × m²)`: Saturation damping (slows response near ±1)

### 7.2 Tone Control (normative)

**One-pole lowpass filter:**
```
y[n] = α × x[n] + (1 - α) × z[n-1]
z[n] = y[n]

Where:
    α = tone_q15  // Q1.15 mix coefficient
    // tone=0 → full lowpass (dark)
    // tone=32767 → bypass (bright)
```

### 7.3 Full Tape Sample Processing

```
def tape_sample(x_in: i16, state: TapeState, params: MaterialParams) -> i16:
    # 1. Drive and bias
    driven = MUL_Q15(x_in, params.drive_q15)
    driven = SAT16(driven + params.bias_q15)

    # 2. Tanh saturation
    target = tanh_q15(driven)

    # 3. Compute λ and ρ
    lam = min(32767, params.drive_q15 + 16384)
    rho = params.drive_q15 >> 2

    # 4. Hysteresis update
    m_sq = MUL_Q15(state.m, state.m)
    damp = 32767 - MUL_Q15(rho, m_sq)
    damp = max(0, damp)

    delta_raw = target - state.m
    delta = MUL_Q15(delta_raw, lam)
    delta = MUL_Q15(delta, damp)

    state.m = SAT16(state.m + (delta >> 1))

    # 5. Return magnetization
    return state.m
```

Then apply tone filter to magnetization output.

### 7.4 Tape Bus Properties

- **Hysteresis:** Output depends on input history (via state.m)
- **Saturation:** Bounded by tanh + magnetic saturation (1 - ρ×m²)
- **Bias:** Asymmetric response (simulate DC bias in tape)
- **Tone:** Post-saturation filtering (simulate HF loss)

---

## 8. Catastrophe Theory Programs

### 8.1 Mathematical Foundation

Catastrophe theory studies discontinuous changes in system behavior from smooth changes in control parameters.

**Elementary catastrophes (Thom's classification):**
- A2 (Fold): 1 control parameter
- A3 (Cusp): 2 control parameters
- A4 (Swallowtail): 3 control parameters (future)

### 8.2 Fold Catastrophe (A2) - material_id=4

**Potential function:**
```
V(x; a) = x³ + a×x
```

**Critical points:** Solve ∂V/∂x = 3x² + a = 0
- If a > 0: No real critical points (smooth)
- If a < 0: Two critical points at x = ±√(-a/3) (bifurcation)

**Dynamics (gradient descent with input drive):**
```
s[n+1] = s[n] + rate × (-3×s[n]² - pressure - fast[n])

Where:
    fast[n] = x[n] >> 2  // Scaled input
    rate = 16384 (Q1.15 = 0.5)
```

**Bifurcation behavior:**
- pressure > 0: Single stable state
- pressure < 0: "Snap" between two stable states
- Hysteresis in response to input

### 8.3 Cusp Catastrophe (A3) - material_id=5

**Potential function:**
```
V(x; a, b) = x⁴ + a×x² + b×x
```

**Dynamics:**
```
s[n+1] = s[n] + rate × (-4×s[n]³ - 2×pressure×s[n] - bias + fast[n])
```

**Bifurcation behavior:**
- Richer than fold: cusp point in (pressure, bias) space
- Three stable states possible
- Complex hysteresis patterns

### 8.4 Catastrophe Safety (MUST)

Both catastrophe programs MUST:
- Apply `soft_clip_q15()` to output state
- Prevent state divergence via saturation
- Maintain bounded slow_state_q15 ∈ [-32768, 32767]

---

## 9. Safety Contract

### 9.1 Material Output Postconditions (MUST)

Material Engine output MUST satisfy:

1. **Magnitude Bound:** `|y[n]| ≤ 32767` for all n
2. **No Wrap:** No integer overflow/underflow in output
3. **No NaN/Undefined:** All samples must be valid i16
4. **Deterministic:** Same input + state → same output
5. **Energy Bound:** RMS(y) ≤ RMS(x) × material_gain_factor (program-dependent)

**Enforcement:** Each program uses internal clipping/saturation to guarantee bounds.

### 9.2 Tail Safety Integration

Material output feeds into **Tail Stage** (WSE-STD-030) which enforces:
- FineClamp: Hard boundary enforcement
- NormalizeScale: Energy normalization (if nonlinear allowed)
- SheafProject: Wavelet-domain projection (if FULL tail)

**Rule (MUST):** Material Engine does NOT replace tail safety. Tail stage is final enforcement layer.

---

## 10. Control Plane Integration

### 10.1 Parameter Sources

**Block-rate control (ParamFrame):**
- Drive, tone, bias, pressure, cut
- Updated at 750 Hz (48000 / 64)
- Slewed for smooth transitions

**Voice-level influence (VoiceFrame):**
- `traction_q15`: Indirectly scales material response (via Symbol Engine)
- `gear_q15`: Energy/commitment (influences drive calculation)
- See WSE-STD-020 for VoiceFrame specification

### 10.2 Event Lane 3: EVT_MATERIAL

**Purpose:** Material engine burst/override events

**Payload:** `(material_id, param_burst_flags, param_values[])`

**Behavior:** Allows instant material program switch or parameter jump (bypassing slew).

**Reference:** WSE-STD-020 §4.3 Event Command Bus

### 10.3 UART-B Configuration

**Transport:** UART-B (configuration/telemetry)

**Commands:**
- SET_MATERIAL_PROGRAM(material_id)
- SET_MATERIAL_PARAMS(drive, tone, bias, ...)
- GET_MATERIAL_STATE()

**Rule (MUST):** Audio execution MUST NOT stall on UART-B traffic.

**Reference:** WSE-STD-090 Dual UART Transport

---

## 11. Determinism Requirements

### 11.1 Bit-Exact Arithmetic (MUST)

All material processing MUST use:
- Q1.15 fixed-point (no floats)
- WSE-STD-001 arithmetic rules (SAT16, MUL_Q15, etc.)
- Round-to-nearest, ties-to-even

**Cross-platform guarantee:** Same input + state + params → bit-identical output on Python, FPGA, or any conformant implementation.

### 11.2 State Determinism (MUST)

- State initialization: all zeros (deterministic)
- State updates: deterministic arithmetic only
- No platform-specific behavior (no `rand()`, `time()`, etc.)

### 11.3 Parameter Determinism (MUST)

- Slew formula is exact (no float approximations)
- rate=256 is normative default
- No adaptive/automatic parameter changes

---

## 12. Material Program Selection Protocol

### 12.1 Routing (MUST)

```
material_id selector → program dispatch:
    0 → clean_glue()
    1 → tape_bus()
    2 → rank_grit()
    3 → tropical_cut()
    4 → catastrophe_fold()
    5 → catastrophe_cusp()
    default → clean_glue()  // Safe fallback
```

### 12.2 State Preservation (MUST)

When switching between programs:
- Each program's state MUST remain intact
- Switching back to program N restores state from when it was last active
- No state bleed between programs

**Example:** Switch sequence `1 → 4 → 1`
- TapeState preserved when switching to Fold
- CatastropheState accumulates while Fold active
- TapeState restored (continues from previous magnetization) when switching back

### 12.3 Per-Block Switching Latency

Material program changes take effect **at block boundaries** (not sample-accurate).

**Rule (MUST):** material_id sampled once per block at block start.

---

## 13. Future Extensions (Informative)

### 13.1 Stochastic Processes (Archetype E)

Future material programs MAY include:
- Hiss injection (band-limited noise)
- Grain texture (Poisson process)
- Random walk bias (slow drift)

**Requirements if added:**
- Deterministic PRNG with seed management
- Energy-bounded noise (safety contract)
- No correlation between L/R unless musically desired

### 13.2 Time-Varying Linear (Archetype C)

Future programs MAY include:
- Wow/flutter (slow pitch modulation)
- Chorus-lite (short delay modulation)
- EQ drift (time-varying filter coefficients)

**Requirements if added:**
- Delay lines with bounded length
- Smooth modulation (no zipper noise)
- Deterministic LFO/modulation sources

### 13.3 Alternative Placements

v0 specifies **time-domain bus** placement. Future versions MAY specify:

- **Subband placement:** Process bands 0-2, 3-5 separately (multiband saturation)
- **Per-voice lightweight:** Tiny per-voice material (micro-drift, noise injection)

These would require separate standard revisions.

---

## 14. Conformance Checklist

An implementation conforms to WSE-STD-MATERIAL v0 iff it:

1. ✅ Implements all 6 material programs (0-5) with bit-exact behavior
2. ✅ Uses Q1.15 fixed-point per WSE-STD-001
3. ✅ Maintains per-channel (L/R) state
4. ✅ Implements parameter slewing with rate=256 default
5. ✅ Places material processing post-mix, pre-safety
6. ✅ Satisfies safety contract (bounded output, no wrap/NaN)
7. ✅ Provides deterministic, cross-platform behavior
8. ✅ Supports per-block material_id switching
9. ✅ Preserves state across program switches
10. ✅ Integrates with ParamFrame and MaterialParams
11. ✅ Applies tape hysteresis model exactly (§7.1)
12. ✅ Applies catastrophe dynamics exactly (§8.2, §8.3)

---

## 15. Example Configurations

### 15.1 Clean Mix Bus (material_id=0)

**Use Case:** Transparent processing with protective clipping only.

**Parameters:** None required (gentle soft clip at 30000 threshold).

**State:** None.

**Cost:** Negligible (~64 ticks per block).

---

### 15.2 Tape Glue (material_id=1)

**Use Case:** Warm analog tape saturation with tone control.

**Parameters:**
- `drive = 16384` (Q1.15 = 0.5, moderate saturation)
- `bias = 0` (symmetric response)
- `tone = 24576` (Q1.15 = 0.75, gentle HF rolloff)

**State:** TapeState with magnetization tracking input history.

**Cost:** ~1280 ticks per block (20 ticks/sample).

**Character:** Soft saturation, gentle compression, warm tone.

---

### 15.3 Brutal Fold (material_id=4)

**Use Case:** Aggressive fold distortion for percussive/aggressive sounds.

**Parameters:**
- `pressure = -16384` (Q1.15 = -0.5, negative = bifurcation region)

**State:** Catastrophe slow state with snap behavior.

**Cost:** ~960 ticks per block (15 ticks/sample).

**Character:** Sharp transitions, hysteresis, "snap" on transients.

---

## 16. Implementation Notes

### 16.1 FPGA Considerations

**Resource budget:**
- Clean/Tropical: <100 LUTs (trivial)
- Tape: ~500 LUTs (tanh lookup, multiply-adds)
- Rank: ~800 LUTs (sorting network)
- Catastrophe: ~400 LUTs (polynomial evaluation)

**Memory:**
- TapeState: 2 × i16 (4 bytes)
- RankState: 2 × (17 × i16 + u8) (70 bytes)
- Total material state: <150 bytes per stereo pair

**Timing:** All programs fit in 64-sample block budget at 100 MHz.

### 16.2 Python Reference Implementation

**File:** `src/symcrash_python_ap/material/material_engine.py`

This file is the **normative reference** for v0 material programs.

---

## 17. Relationship to Other Standards

| Standard | Relationship |
|----------|--------------|
| WSE-STD-001 | Material uses Q1.15 arithmetic rules |
| WSE-STD-020 | Material params from VoiceFrame (traction, gear) + ParamFrame |
| WSE-STD-030 | Material output feeds into Tail Safety stage |
| WSE-STD-070 | Material settings at Tower L2/L3 levels |
| WSE-STD-090 | Material control via UART-B + Event Lane 3 |
| WSE-STD-120 | Material receives stable polyphase IDWPT output |
| WSE-STD-200 | CMMM owns traction_q15 (material response control) |

---

## 18. Glossary

- **Material:** Physical/analog layer (post-symbolic processing)
- **Hysteresis:** History-dependent nonlinear response
- **Catastrophe:** Bifurcation-based nonlinearity (fold, cusp)
- **Q1.15:** Signed 16-bit fixed-point format (1 integer bit, 15 fractional bits)
- **Slew:** Gradual parameter change for click-free transitions
- **Tail Safety:** Final enforcement stage ensuring bounded output
- **Bus:** Shared processing path (all voices mixed to single stereo stream)
- **Dynamical System:** State-based system: `(state', output) = f(state, input, params)`

---

**End of WSE-STD-MATERIAL v0**

---

**Revision History:**
- v0 (2026-01-05): Initial freeze
  - 6 material programs specified
  - Tape hysteresis normative model
  - Catastrophe A2/A3 dynamics
  - Safety contract defined
