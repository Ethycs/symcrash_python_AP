# WSE-STD-210 — CCS Oscillator Enrichment (Mode A LUT + Mode B Rotator)

**Standard ID:** WSE-STD-210
**Title:** CCS Oscillator Enrichment (Mode A LUT + Mode B Rotator/Harmonic Bank)
**Status:** SPECIFIED (normative)
**Scope:** Defines oscillator generation for CCS synthesis where **IDWPT is the only synthesis filterbank**, with two compliant oscillator front-ends: **Mode A (coeff LUT)** and **Mode B (rotator/harmonic bank)**.
**Targets:** CPU reference implementation + FPGA bit-exact implementations.

---

## 0. Design Invariants (MUST)

1. **IDWPT is the synthesizer.** All periodic oscillator audio is produced by feeding coefficient frames into IDWPT and assembling output samples using the polyphase interleave rule (Section 4).
2. **Phase math is Q32 everywhere.** One shared Q32 phase model is used for all runtime oscillators and all offline generation/compression paths.
3. **K semantics are “within-block sample offset.”** If `K=8`, then `k=0..7` indexes the within-block sample offset, not an arbitrary “phase dimension.”
4. **Determinism is mandatory.** Rounding, saturation, and phase stepping are fully specified (Sections 2, 5).

---

## 1. Terms, Sizes, and Fixed-Point Formats

### 1.1 Constants

* `N` = block length in samples. **Default:** 64.
* `K` = polyphase lanes / within-block offsets. **Default:** 8.
* `S = N/K` = samples per lane per block. **Default:** 8.
* `LEAVES` = number of WPT leaves. **Fixed:** 64.
* `COEFF_LEN = LEAVES * K` = flattened coefficient count. **Default:** 512.

### 1.2 Phase formats

* `phase_32`: **uint32** Q32 phase accumulator representing `[0, 1)` turns.
* `phase_inc_32`: **uint32** Q32 per-sample phase increment.

### 1.3 Audio sample and coefficient formats

* Time-domain audio samples: **int16** (Q15-ish signed PCM).
* Coefficients stored in frames/LUTs: **int16**.
* Internal MAC/accumulators: **int32** or wider (implementation-defined) as long as output matches rounding/saturation requirements.

### 1.4 Rounding and saturation primitives (MUST)

* `SAT16(x)`: clamp to `[-32768, 32767]`.
* `RSHIFT_RNE(x, b)`: right shift by `b` with **round-to-nearest-even**.
  (CPU and FPGA MUST match. If an implementation cannot do exact RNE efficiently, it MUST emulate it.)

---

## 2. SineCore (Foundation Service)

SineCore is a **mandatory** service for WSE-STD-210 compliance.

### 2.1 SineCore API (MUST)

* `sin_cos_q15(phase_32) -> (sin_q15, cos_q15)`

  * `sin_q15 ≈ round(sin(2π·phase) * 32767)`
  * `cos_q15 ≈ round(cos(2π·phase) * 32767)`
  * `cos` MUST be computed as `sin(phase + 0x40000000)` modulo 2³² (exact quarter-turn).
* `sin_cos_inc_q15(phase_inc_32) -> (sinΔ_q15, cosΔ_q15)`

  * Same mapping, used for NCO recurrence.

### 2.2 Accuracy / spur guidance (SHOULD)

For a mid-band sine (e.g., 440 Hz @ 48 kHz), SineCore quantization spurs SHOULD be below **-60 dBc** in the oscillator output when used in Mode B with H=1 and no extra modulation.

### 2.3 NCO rotator recurrence (OPTIONAL but RECOMMENDED)

Implementations MAY maintain `(sinφ, cosφ)` via:

* `sin(φ+Δ)=sinφ·cosΔ + cosφ·sinΔ`
* `cos(φ+Δ)=cosφ·cosΔ - sinφ·sinΔ`
  using `RSHIFT_RNE` and `SAT16` at each step.

---

## 3. Oscillator Modes (Normative)

WSE-STD-210 defines two oscillator front-ends. Implementations MAY support one or both; if both are supported, mode switching MUST follow Section 8.

### 3.1 Mode A — Coefficient Phase LUT (CCS-LUT)

**Runtime mapping (MUST):**

* LUT has shape `[LEAVES, K, P]` with `P` power-of-two recommended.
* Compute:

  * `p = phase_32 >> (32 - log2(P))`
  * `p_next = (p + 1) & (P-1)`
  * `frac = phase_32 & ((1 << (32 - log2(P))) - 1)` (remaining bits)
* Then:

  * `coeff_frame = lerp(LUT[:,:,p], LUT[:,:,p_next], frac)` elementwise
  * Lerp MUST be deterministic fixed-point:

    * `val = v0 + ((v1 - v0) * frac + half) >> frac_bits`
      with RNE-equivalent behavior.

**Mode A generation constraint (MUST):**
Mode A LUTs MUST be generated using the same Q32 phase model as runtime:

* For each `(p,k)`:

  * `phase0 = p*(2^32/P) + k*phase_inc_32`
  * build block: `x[n] = Wave(phase0 + n*phase_inc_32, params)`
  * `LUT[:,k,p] = FWPT(x_block)`
    No `cycle_len` rounding schemes are allowed if they can drift vs Q32 runtime stepping.

### 3.2 Mode B — Rotator/Harmonic Bank (CCS-ROT)

Mode B generates coefficients dynamically from trig weights supplied by SineCore.

WSE-STD-210 defines **two compliant Mode B profiles**:

#### 3.2.1 Mode B2 (Harmonic Basis Bank) — **NORMATIVE**

This profile is REQUIRED if you claim support for **PWM modulation** and/or **audio-rate FM/PM** in Mode B.

**Precomputed data (per pitch anchor):**

* `DC_frame[LEAVES,K]`: coefficients that reconstruct a constant signal of amplitude +1.0 (Q15) when passed through IDWPT+interleave (Section 4).
* For each harmonic `h = 1..Hmax_anchor`:

  * `Ucos_h[LEAVES,K]`: coefficients that reconstruct `cos(2πh·φ)` for that pitch anchor under the system’s phase stepping model.
  * `Usin_h[LEAVES,K]`: coefficients that reconstruct `sin(2πh·φ)` for that pitch anchor.

**Runtime waveform definition (per oscillator instance):**
A waveform is defined by Q15 harmonic weights:

* `dc_q15`
* arrays `a_h_q15` (sine weights), `b_h_q15` (cos weights) for `h=1..H_active`

Time-domain target form:
[
x(\phi) = dc + \sum_{h=1}^{H} \big(b_h \cos(h\phi) + a_h \sin(h\phi)\big)
]

**Runtime coefficient synthesis (MUST):**
For each lane `k`:

1. Compute lane phase:

   * `φ_k = phase_32 + k*phase_inc_32` (mod 2³²)
2. Compute `(sin(hφ_k), cos(hφ_k))` via SineCore for each active harmonic `h`.
3. Compute weighted harmonic contribution coefficients using the identity:
   [
   b_h\cos(h\phi) + a_h\sin(h\phi)
   = \Re{(b_h - j a_h),e^{j h\phi}}
   ]
   which expands into basis mixing:

* Let `cos = cos(hφ_k)`, `sin = sin(hφ_k)` (both Q15)
* Let `b=b_h_q15`, `a=a_h_q15` (Q15)
* Compute two Q15 scalars:

  * `wcos = b*cos - a*sin`  (Q15, using RSHIFT_RNE)
  * `wsin = b*sin + a*cos`  (Q15)
    Then:
    [
    coeff[:,k] = dc_q15 \cdot DC_frame[:,k] ;+;\sum_{h=1}^{H} \Big( wcos\cdot Ucos_h[:,k] + wsin\cdot Usin_h[:,k] \Big)
    ]
    All multiplies use int32 accumulators and are rounded/saturated deterministically to int16 coefficients.

#### 3.2.2 Mode B1 (Compressed Wave-Specific Rotator) — OPTIONAL

This is the “compress Mode A LUT into harmonics” profile. It is allowed, but PWM-by-weight is NOT guaranteed to be correct unless you explicitly precompute width-dependent compressed frames.

Mode B1 stores per waveform:
[
coeff(\phi)=D + \sum_{h=1}^{H}\big(C_h\cos(h\phi)+S_h\sin(h\phi)\big)
]
where `C_h,S_h,D` are derived by FFT compression across phase bins (Section 6.1).

---

## 4. Polyphase Synthesis (IDWPT + Interleave)

Given a coefficient frame `coeff_frame[LEAVES,K]` for one block:

1. For each lane `k=0..K-1`:

   * run `IDWPT(coeff_frame[:,k])` producing `y_k[0..N-1]`.

2. Assemble time-domain block output `out[0..N-1]` by the **canonical interleave rule (MUST)**:

* `out[k::K] = y_k[0::K]`

This rule is mandatory for the “shifted-block / within-block offset” semantics used by WSE-STD-210.

---

## 5. Pitch Anchors and Frame Atlases (Mode B)

### 5.1 Pitch dependence (MUST)

For Mode B, `DC_frame`, `Ucos_h`, `Usin_h` depend on pitch (phase_inc) because the WPT leaf responses depend on frequency relative to block size.

### 5.2 Anchors (MUST)

Implementations MUST use an atlas keyed by `(wave_id, anchor_id)` where:

* `anchor_id` is typically octave or coarse semitone bins (implementation-defined but fixed).
* Each anchor provides basis frames for that pitch region.

### 5.3 Inter-anchor interpolation (SHOULD)

Implementations SHOULD interpolate between adjacent anchors for tuning smoothness. If interpolation is used, it MUST be deterministic fixed-point.

---

## 6. Offline Generation and Compression

### 6.1 Mode A → Mode B1 FFT compression (OPTIONAL)

Given Mode A LUT `series[p] = LUT[leaf,k,p]` for `p=0..P-1`, define FFT:

* `X = FFT(series)` using convention `X[h] = Σ series[p]·e^{-j2πhp/P}`

Scaling rules (MUST):

* `D = round(Re(X[0]) / P)`
* For `h=1..Hmax` (and `h < P/2`):

  * `C_h = round( 2·Re(X[h]) / P )`
  * `S_h = round( -2·Im(X[h]) / P )`
    Then apply `SAT16`.

Rounding MUST use the standard’s RNE rule (Section 1.4). The offline compressor must document and match it.

### 6.2 Mode B2 basis generation (NORMATIVE)

For each anchor pitch and harmonic `h`:

* Generate time-domain reference blocks using the system’s Q32 phase stepping model:

  * `x_cos[n] = cos(2πh·(phase0 + n·phase_inc))`
  * `x_sin[n] = sin(2πh·(phase0 + n·phase_inc))`
* Produce basis frames:

  * `Ucos_h[:,k] = FWPT( x_cos shifted by k samples )`
  * `Usin_h[:,k] = FWPT( x_sin shifted by k samples )`
* Generate `DC_frame` similarly from constant `x[n]=1.0` reference blocks.

This basis generation MUST use the same FWPT/IDWPT pair and the same K semantics as runtime.

---

## 7. Classical Oscillator Coverage (Mode B2)

### 7.1 Bandlimiting (MUST)

For sample rate `fs` and oscillator fundamental `f0`, harmonic cap:
[
H \le \left\lfloor \frac{fs/2}{f0} \right\rfloor
]
Implementations SHOULD apply a safety margin (e.g., 0.9× Nyquist) when PWM and/or FM is active.

### 7.2 Harmonic weights for classic shapes (RECOMMENDED defaults)

All weights are Q15.

* **Sine:** `H=1`, `a_1=1`, `b_1=0`, `dc=0`.
* **Saw:** for `h>=1`, `a_h = 1/h`, `b_h = 0`, `dc=0` (phase convention can swap a/b; choose one convention and keep it consistent).
* **Square:** odd harmonics only, `a_h = 1/h`, `b_h=0`.
* **Triangle:** odd only, `a_h = 1/h^2` with alternating sign, `b_h=0`.

(Exact phase convention MUST be consistent with the basis generation; if you choose cosine-phase saw, use `b_h` instead.)

---

## 8. Modulation and Events

### 8.1 PWM (Mode B2, MUST if advertised)

PWM is implemented by updating harmonic weights `(a_h(d), b_h(d), dc(d))` as a function of duty cycle `d`.

* **Duty cycle format:** `width_q15 ∈ [0..32767]` representing `d ∈ [0..1)`.
* Implementations MAY:

  * compute weights analytically using trig (via SineCore), or
  * use width-binned tables with deterministic interpolation.

PWM MUST update at **lane rate** (k-slot) or faster; implementations MUST document which (see 8.4).

### 8.2 Hard Sync (MUST if advertised)

Hard sync is treated as an event that introduces a waveform discontinuity. To prevent aliasing, a bandlimited correction MUST be applied.

* Discontinuity amplitude MUST be computed as:

  * `Δ = x_after - x_before`
    where `x_before` is waveform value immediately before reset, `x_after` immediately after reset.

**Allowed correction path:**

* Audio-domain BLEP/minBLEP injection post-IDWPT is compliant and RECOMMENDED.
* Coefficient-domain BLEP is OPTIONAL.

Fractional timing:

* Implementations SHOULD support fractional-delay BLEP kernels (e.g., 16 bins).

### 8.3 FM/PM (MUST if advertised)

FM/PM modifies phase. WSE-STD-210 requires explicit Q-format mapping:

* `mod_sample_q15`: int16 in [-32768, 32767]
* `fm_depth_q15`: int16 in [0, 32767] interpreted as “fraction of full turn at full-scale mod”
* Phase modulation in Q32:

  * `phase_mod_32 = round( (mod_sample_q15 * fm_depth_q15) * 2^(32-15-15) )`
  * Then `φ' = φ + phase_mod_32`

FM/PM SHOULD update at lane rate or faster (see 8.4). If audio-rate FM is required, implementations MUST update at sample rate or use an equivalent higher-rate strategy.

### 8.4 Lane-rate vs microstep-rate updates (MUST specify)

Given polyphase interleave mapping:

* lane `k` maps to output samples: `k, k+K, k+2K, ...`

When applying modulation “per lane,” the modulator sample selection MUST be specified:

* Option L1: `mod_k = mod_buffer[k]` (lowest latency)
* Option L2: `mod_k = mean(mod_buffer[k::K])` (smoother)

An implementation MUST choose one option and keep it consistent across CPU/FPGA.

---

## 9. Mode Switching (A ↔ B) (MUST if both modes supported)

Switching between Mode A and Mode B MUST be click-free:

* Preserve phase continuity.
* Crossfade either:

  * coefficient frames in coeff space, or
  * audio output in time domain,
    over **2–4 blocks** (implementation-defined but fixed).

---

## 10. Wire/Frame Parameters (Normative Fields)

### 10.1 Oscillator mode

* `osc_mode`:

  * `0` = Mode A
  * `1` = Mode B (B2 basis bank unless explicitly flagged B1)

### 10.2 Mode B parameters (minimum)

* `H_request` (uint8): requested harmonic count (actual `H_active` is bandlimited)
* `pwm_width_q15` (int16): duty (if PWM enabled)
* `sync_enable` (bool)
* `fm_depth_q15` (int16)
* `unison_voices` (uint8)
* `unison_spread_cents` (int16)

(Implementations may extend; these are the minimum for “classic osc enrichment.”)

---

## 11. Compliance Tests (MUST)

### 11.1 Sine correctness (H=1)

* f0 error < **0.1%** across a test set of pitches (recommended threshold).
* Boundary/intra ratio < **3.0** (mandatory continuity threshold).
* Largest spur < **-60 dBc** (recommended).

### 11.2 Classic waves harmonic behavior

For saw/square/triangle:

* expected harmonic rolloff and harmonic selection (odd-only where applicable) must match within tolerance (implementation-defined).

### 11.3 PWM sweep

A PWM sweep MUST not exhibit zippering or obvious block-rate combing beyond the defined spur threshold.

### 11.4 Sync

Hard sync output MUST show controlled aliasing and no raw discontinuity click (BLEP correction must be verifiable).

### 11.5 Determinism

Two runs with identical inputs MUST produce bit-identical outputs.

---

## 12. What Replaces What (Normative Relationship Summary)

* **SineCore** is the universal trig service and defines phase conventions.
* **Mode A** is a coeff LUT indexed by phase; SineCore is used for correct LUT generation and testing, not runtime.
* **Mode B2** uses SineCore at runtime to rotate harmonic basis frames; it replaces large phase LUTs and enables PWM/FM by weight updates.
* **Mode B1** is an optional compressed representation of Mode A for fixed waveforms; it is not inherently PWM-by-weight unless width-dependent frames are provided.

---

## Appendix A — Reference Resource Accounting (Informative)

* One frame `[64,K]` int16 @ K=8 is `64*8*2 = 1024 bytes`.
* Mode B2 basis memory per anchor:

  * `DC_frame`: 1 KB
  * per harmonic: `Ucos_h` + `Usin_h`: 2 KB/h
  * total ≈ `(2H + 1) * 1KB`

---

If you want this emitted as an actual repo file (`docs/WSE-STD-210.md`) with your project’s header conventions, tell me your preferred header block format (or point me at an existing WSE-STD markdown file in your tree) and I’ll match it exactly.
