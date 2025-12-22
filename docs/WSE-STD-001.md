## WSE-STD-001 — Fixed-Point Arithmetic & Numeric Hygiene (v0)

**Status:** NORMATIVE
**Applies to:** Voice Engine numeric outputs, Symbol Engine operator math, Wavelet Engine coefficient math, Safety Engine math
**Goal:** Python reference and FPGA implementation produce **bit-equivalent** results for all “bit-exact” paths.

---

### 1. Scope and design rules

1. This standard defines **integer widths, Q formats, rounding, saturation, and shift semantics** for all arithmetic used in block processing.
2. Anything not specified here **MUST NOT** be relied on for matching (e.g., float math, platform-dependent overflow).
3. If an algorithm needs non-bit-exact functions (e.g., `tanh`, `atan2`), it **MUST** be explicitly marked **NON-BIT-EXACT** and isolated behind a switch or separate standard.

---

### 2. Primitive numeric types

All bit-exact paths use signed two’s-complement integers.

| Name          |     Bits | Meaning                                                                 |
| ------------- | -------: | ----------------------------------------------------------------------- |
| `s16`         |       16 | signed sample / coefficient lane                                        |
| `s32`         |       32 | signed accumulator / intermediate                                       |
| `s64`         |       64 | **required** for products and wide accumulations in the reference model |
| `u16/u32/u64` | unsigned | masks, payloads, IDs                                                    |

**Normative rule:** In the **Python reference**, every multiply **MUST** be computed in `s64` (or Python int), then reduced to the target width with the rounding/saturation rules below.

---

### 3. Canonical Q formats (v0)

These are the only “standard” Q formats in v0.

| Format   | Type  | Range               | Use                                                 |
| -------- | ----- | ------------------- | --------------------------------------------------- |
| `Q1.15`  | `s16` | [-1, 1)             | gains, strengths, normalized coefficients (typical) |
| `Q8.8`   | `s16` | [-128, 127.996)     | slow rates, UI-ish values, Hz-ish coarse values     |
| `Q0.16`  | `u16` | [0, 0.99998)        | unipolar controls                                   |
| `Q16.16` | `s32` | [-32768, 32767.999) | long accumulators / envelopes if needed             |

**Canonical constants:**

* `ONE_Q15 = 0x7FFF` (≈ 0.999969482)
* `ZERO = 0`
* `MIN_S16 = -32768`, `MAX_S16 = 32767`

---

### 4. Shifts and sign rules

1. Right shift `>>>` is **arithmetic shift** (sign-extended).
2. Left shift `<<` is logical shift (introduces zeros) but can overflow; any overflow behavior is controlled by explicit saturation rules below.
3. Shifts **MUST** be performed on a sufficiently wide type to avoid undefined behavior (e.g., don’t shift a 16-bit value left into overflow without specifying saturate).

---

### 5. Saturation (clamp) rules

#### 5.1 Saturate to `s16`

`SAT16(x)` maps any integer `x` to the range [-32768, 32767]:

* if `x > 32767` → `32767`
* if `x < -32768` → `-32768`
* else → `x`

#### 5.2 Saturate to `s32`

`SAT32(x)` maps any integer `x` to the range [-2^31, 2^31-1].

**Normative rule:** Any intermediate that might exceed its container width **MUST** be saturated at a specified boundary. Silent wraparound is **forbidden** in bit-exact paths.

---

### 6. Rounding modes

Bit-exact arithmetic uses **round-to-nearest, ties-to-even** (banker’s rounding) when reducing precision after a right shift.

#### 6.1 `RSHIFT_RNE(x, k)`

Compute `(x / 2^k)` rounded to nearest integer, ties-to-even.

Let:

* `q = x >> k` (arithmetic shift)
* `r = x & ((1<<k)-1)` for x ≥ 0
  For x < 0, use a symmetric remainder model in the reference (Python int handles this cleanly if you implement via absolute value; see below).

Define:

* `half = 1 << (k-1)`
* `inc = 1` if remainder is strictly greater than half
* `inc = 0` if remainder is strictly less than half
* if remainder == half: `inc = 1` **only if** `q` is odd (tie-to-even)

Then:

* result = `q + inc` for x ≥ 0
* for x < 0, apply the same rule to `abs(x)` and restore sign (this avoids Python’s floor-division quirks).

**Normative rule:** All “multiply-then-shift” reductions **MUST** use this rounding mode unless explicitly marked otherwise.

---

### 7. Canonical multiply primitives

#### 7.1 Q15 multiply

`MUL_Q15(a_q15, b_q15) -> s16`:

* Inputs: `a, b` are `s16` interpreted as `Q1.15`
* Compute wide product: `p = (s64)a * (s64)b`  (this is `Q2.30`)
* Reduce back to `Q1.15` via `RSHIFT_RNE(p, 15)`
* Saturate to `s16` via `SAT16`

So:

* `MUL_Q15(a,b) = SAT16( RSHIFT_RNE( (s64)a*(s64)b, 15) )`

#### 7.2 Q15 multiply to Q31

Sometimes you want a wider output (accumulating gains, etc.):

`MUL_Q15_TO_Q31(a_q15, b_q15) -> s32`:

* `p = (s64)a*(s64)b` (`Q2.30`)
* shift left by 1 to `Q1.31`: `p2 = p << 1`
* saturate to s32: `SAT32(p2)`

(Only use if you really need `Q1.31`.)

---

### 8. Canonical add/mix primitives

#### 8.1 Add two Q15 values

`ADD_Q15(a,b) = SAT16((s32)a + (s32)b)`

#### 8.2 Weighted mix (crossfade)

Given `x,y` in Q15 and `t` in Q15 where `t=0` → x, `t=ONE` → y:

`MIX_Q15(x,y,t) = ADD_Q15( MUL_Q15(x, ONE-t), MUL_Q15(y, t) )`

**Normative rule:** All tree morphs, coefficient blends, and OLA-style blends in bit-exact mode must use this (or an explicitly defined equivalent).

---

### 9. Absolute value and peak detection

`ABS16(x)`:

* If `x == -32768`, return `32767` (saturating abs)
* Else return `abs(x)`

This avoids the “abs(minint) overflows” trap.

---

### 10. Division (rare; avoid if possible)

Division is expensive and tricky to match. In v0 bit-exact paths:

1. Integer division **MUST** be truncation toward zero (C/Verilog-style).
2. If you need reciprocal scaling, prefer:

   * fixed-point reciprocal LUT + multiply, or
   * shift-based approximations with defined error bounds.

If division is used, the exact formula and rounding MUST be specified per use site.

---

### 11. Square root and trig: allowed status

#### 11.1 SQRT

If you compute RMS/energy or magnitudes:

* `sqrt_approx` **MUST** be either:

  * exact integer sqrt (deterministic), or
  * a specified LUT/iteration with frozen steps.

Anything else is NON-BIT-EXACT.

#### 11.2 `tanh`, `atan2`, sin/cos

These are **NON-BIT-EXACT by default** unless you commit to:

* a LUT (size specified),
* or a fixed polynomial (coeffs + evaluation order specified),
* or a fixed CORDIC configuration.

**v0 recommendation:** Keep these out of the bit-exact core. If you want them, create **WSE-STD-001A (Approx Functions)** later.

---

### 12. Overflow and intermediate width rules (hard requirement)

For any operator implementation:

1. All products **MUST** be computed in at least `s32`, and in the reference model in `s64`.
2. Accumulators that sum more than 2 terms of `s16` **SHOULD** use `s32` or `s64`.
3. Any narrowing (e.g., `s32 -> s16`) **MUST** specify:

   * right shift amount (if any)
   * rounding (RNE ties-to-even)
   * saturation (`SAT16`)

Silent truncation is forbidden.

---

### 13. Canonical “operator skeleton” pattern (for bit-exact ops)

Every operator in the Symbol Engine that maps `coeffs[256] -> coeffs'[256]` MUST follow:

1. Read inputs as `s16` (Q15 unless explicitly stated)
2. Compute intermediate values in `s32`/`s64`
3. Apply any scaling via `MUL_Q15`, reductions via `RSHIFT_RNE`
4. Narrow with explicit `SAT16`
5. Write output as `s16`

This ensures portability.

---

### 14. Conformance tests (mandatory)

Any implementation claiming conformance to WSE-STD-001 MUST provide:

1. Unit tests for:

   * `RSHIFT_RNE` (positive & negative tie cases)
   * `MUL_Q15` edge cases (e.g., `-32768 * -32768`, `32767*32767`, mixed signs)
   * `ABS16(-32768)`
2. Operator-level “golden vectors”:

   * fixed input coefficient arrays
   * fixed op parameters
   * fixed expected output arrays (hash accepted if too large)

---

### 15. What this standard does *not* decide

* Operator ordering / tail ordering (that’s STD-030)
* PRNG (STD-040)
* Degradation policy (STD-050)
* Frame ABI / UART packing (STD-020)
* Tree semantics (STD-010)

---

If you want, paste (or point me to) the exact functions you currently use in your Python reference for `mul_q15`, rounding, and saturation, and I’ll produce a **conformance diff**: “your code matches STD-001” vs “needs edits” with a short patch list.
