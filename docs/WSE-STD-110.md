# WSE-STD-110 — First-Class Sine & Cosine Core Specification

**Status:** Draft v1.0 (implementable)
**Scope:** Defines the *canonical* high-quality `sin`/`cos` primitive for WSE (Wave/Symbol Engine) used by Baby Fourier carrier, modulation, and any phasor-based operators.

---

## 1. Purpose

WSE-STD-110 specifies a deterministic, instrument-grade sine/cosine implementation that:

* preserves **phase continuity** across blocks and voices
* avoids **cycle kinks**, **block-rate spurs**, and **deterministic alias-like lines** introduced by the oscillator itself
* supports **quadrature output** (`sin` and `cos`) from a single phase input
* is **FPGA-native**: fixed-point, ROM-based, predictable timing, time-mux friendly

This is a first-class primitive: it has a stable interface, conformance tests, and clear reset/latency contracts.

---

## 2. Definitions

* **Phase**: unsigned integer representing a fraction of one turn (cycle).
* **Quadrature**: `cos(θ)` and `sin(θ)` produced from the same θ; `cos(θ)=sin(θ+π/2)`.
* **Q-format**: fixed-point; e.g. Q1.15 signed is `int16` in [-32768, 32767] representing [-1, +0.99997].

---

## 3. Normative Requirements

### 3.1 Correctness and Continuity

**R-110-1 (Phase continuity):**
Given a phase accumulator `phase[n+1]=phase[n]+phase_inc` (mod 2^PHASE_BITS), the output must be continuous w.r.t. phase evolution. No internal “block reset” behavior is permitted.

**R-110-2 (Determinism):**
For fixed inputs (`phase`, configuration), outputs (`sin`, `cos`) must be bit-exact across runs/builds.

**R-110-3 (Symmetry correctness):**
The implementation must preserve quadrant symmetries:

* `sin(θ)` odd symmetry
* `cos(θ)` even symmetry
  within the quantization implied by output format.

### 3.2 Audio-Quality Constraints

**R-110-4 (No phase-domain kinks):**
The output waveform must not contain repeatable derivative discontinuities caused by coarse internal quantization (e.g., “8-bit parabola” artifacts).

**R-110-5 (No block-rate behavior):**
The primitive must not reference block boundaries or sample indices. Any spurs at multiples of block rate attributable to this module are nonconforming.

### 3.3 Interface Completeness

**R-110-6 (Quadrature output):**
Must provide `sin` and `cos` outputs, or provide `sin` plus a guaranteed method to obtain `cos` via a phase offset constant.

**R-110-7 (Configurable precision):**
Must support at least two LUT sizes and a defined interpolation method (see §5).

---

## 4. External Interface

### 4.1 Inputs

* `phase` : `uPHASE_BITS` (recommended PHASE_BITS = 32)
* `mode` : enum

  * `SINE_ONLY`
  * `SINE_COS` (required supported mode)
* `enable` : 1-bit (if 0, outputs may be held, but internal ROM access behavior is implementation-defined)
* Optional:

  * `phase_offset` : `uPHASE_BITS` (for arbitrary phase shift; default 0)

### 4.2 Outputs

* `sin_out` : signed Q format (default Q1.15, `i16`)
* `cos_out` : signed Q format (same as `sin_out`)
* Optional:

  * `valid` : 1-bit (for pipelined implementations)
  * `phase_passthru` : `phase + phase_offset` (debug/verification)

### 4.3 Latency

* **Combinational** or **pipelined** allowed.
* If pipelined, latency must be **fixed** and documented as `LAT_CYCLES`.
* `valid` must align to outputs if latency > 0.

---

## 5. Reference Implementation Class

WSE-STD-110 defines one *normative* algorithm family:

## 5.1 Quarter-Wave LUT + Linear Interpolation (Normative)

### 5.1.1 LUT

* Store quarter-wave values `LUT[i] ≈ sin( (i / (N-1)) * (π/2) )`
* LUT size `N = 2^ADDR_BITS`, minimum supported:

  * **Profile A:** `ADDR_BITS=10` (N=1024)
  * **Profile B:** `ADDR_BITS=11` (N=2048)
* LUT values stored as signed Q format:

  * default: Q1.15 (`int16`)

### 5.1.2 Phase Decomposition (PHASE_BITS recommended 32)

Let `phase_eff = phase + phase_offset` (mod 2^PHASE_BITS)

* Quadrant: `q = phase_eff[PHASE_BITS-1 : PHASE_BITS-2]` ∈ {0,1,2,3}
* Within-quadrant angle bits are the remaining high bits.
* Define:

  * `addr` = next `ADDR_BITS` below quadrant bits
  * `frac` = next `FRAC_BITS` below `addr`
* **Requirement:** `FRAC_BITS >= 6` (recommended 8–10)

### 5.1.3 Address Mirroring and Sign

Map to quarter-wave index `i`:

* For q=0: use `i = addr`
* For q=1: use `i = (N-1) - addr`
* For q=2: use `i = addr`, then negate result
* For q=3: use `i = (N-1) - addr`, then negate result

### 5.1.4 Linear Interpolation

Fetch:

* `a = LUT[i]`
* `b = LUT[i+1]` (with `LUT[N] = LUT[N-1]` or clamp at end)

Compute:

* `y = a + ((b - a) * frac >> FRAC_BITS)`

Apply sign per quadrant.

### 5.1.5 Cosine

Compute cosine using the same sine core:

* `cos(phase) = sin(phase + 2^(PHASE_BITS-2))`
  (i.e., add quarter turn constant)

**Requirement:** cosine must be bit-identical to this definition (modulo the same core).

---

## 6. Numerical Formats and Ranges

### 6.1 Output Format

Default output format: **Q1.15**

* Range: [-32768, 32767]
* Interpretation: `value / 32768`

### 6.2 Intermediate Precision

To avoid bias/spurs:

* `b-a` must be computed at ≥17 bits signed
* multiplication `(b-a)*frac` must retain ≥ (17 + FRAC_BITS) bits before shifting
* round-to-nearest is recommended; truncation is allowed but must pass THD/spur tests (§9)

### 6.3 DC Bias Requirement

Mean value over a full cycle must be within:

* **|mean| ≤ 1 LSB** (for Q1.15) for a pure sine driven by uniform phase steps over a full cycle.

---

## 7. Operating Profiles (Compliance Targets)

### Profile A (Default)

* PHASE_BITS = 32
* ADDR_BITS = 10 (1024 LUT)
* FRAC_BITS = 8
* OUT = Q1.15

### Profile B (Higher quality)

* PHASE_BITS = 32
* ADDR_BITS = 11 (2048 LUT)
* FRAC_BITS = 9–10
* OUT = Q1.15 or Q1.17 internally, downshift to Q1.15

---

## 8. FPGA Implementation Notes (Normative constraints)

**R-110-8 (No per-block hooks):**
No logic may depend on “sample index in block” or “block start.”

**R-110-9 (Time-mux safe):**
Core must operate correctly when invoked sequentially for different voices in successive cycles. No hidden per-voice state is permitted in the sine core itself.

**R-110-10 (ROM determinism):**
LUT contents must be build-locked (generated deterministically, committed as a ROM init file or generated by deterministic script).

---

## 9. Conformance Tests (Required)

A device/implementation is conformant to WSE-STD-110 only if it passes all tests for at least Profile A.

### 9.1 Symmetry Tests

* For random phases θ:

  * `sin(θ) ≈ -sin(θ+π)` within ≤2 LSB
  * `cos(θ) ≈ cos(-θ)` within ≤2 LSB

### 9.2 Continuity Test

* Sweep phase across the wrap boundary and ensure no discontinuity larger than what is implied by the phase step.

### 9.3 Spectral Purity Test (Instrument-grade sanity)

Generate a steady tone for multiple frequencies (e.g., 55 Hz, 220 Hz, 880 Hz) and measure:

* No fixed spurs at block-rate multiples attributable to the sine core.
* THD must be below a defined target for Profile A:

  * **Recommended acceptance:** THD ≤ -70 dBFS for midband tones (this is feasible with 1024+interp in Q15).
    (If your measurement harness differs, treat this as a target; the *hard* requirement is “no deterministic kinks/spurs.”)

### 9.4 DC Bias Test

* Over an integer number of cycles, mean must be ≤ 1 LSB (Q1.15).

---

## 10. Reset and State Contract

The sine core itself is **stateless** aside from pipeline registers.

* Reset affects only pipeline/valid registers.
* Phase accumulation is external (carrier engine).
  **This standard does not define phase accumulator behavior** other than expecting `phase` to be valid.

---

## 11. Integration Requirements for WSE

### 11.1 Carrier Engine Requirements (when using WSE-STD-110)

* Carrier must maintain per-voice `phase` and `phase_inc`.
* Pitch changes must be applied via:

  * phase_inc slew, or
  * dual-render crossfade
    but **never** by resetting phase mid-note unless explicitly requested.

### 11.2 Use in Modulation / FX

* Any module needing quadrature must use WSE-STD-110 outputs rather than duplicating sine approximations.

---

## 12. Non-Normative Extensions (Allowed)

These do not affect conformance if the normative mode remains available.

* “Character sine” modes (parabola, low-bit, etc.) as optional flavors
* Higher output precision (Q1.23/Q1.31) internal, then dither/downshift
* Higher-order interpolation (cubic) if it remains deterministic and passes tests

---

## 13. Deliverables Checklist (what you should have in the repo/RTL)

* `sine_lut.mem` (or equivalent) for Profile A (and optionally B)
* Deterministic LUT generator script (optional but recommended)
* Unit tests for symmetry/DC/continuity
* Audio test harness (offline) that produces spectra for 3–5 test tones
* A single exported module/API: `SineCosCore`

---

If you want, I’ll also write **WSE-STD-110A** (the “verification appendix”) with exact numeric thresholds for your specific Q format and your measurement tooling, so you can gate commits on it.
