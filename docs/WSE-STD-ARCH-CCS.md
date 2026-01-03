# WSE-ARCH-v0: Carrier-Coefficient Synthesis (No Forward Pass in Voice Path)

## 0. Goals and Non-Goals

### Goals

1. **Phase-stable voices** with classic synth expectations (saw/square/sine, pitch bend, glide).
2. **Wavelet-native expressivity** via operator algebra (CMMM/Symbol engine).
3. **Deterministic FPGA execution**: wavelet+symbol are “automatic transmissions”; all timing in voice engine.
4. **FWPT optional**: allowed for external audio conditioning, but never required for voices.

### Non-Goals (v0)

* Full audio→instrument inverse/mimic in real time.
* Per-voice FWPT analysis.
* Perfect psychoacoustic loudness matching (we do deterministic RMS/peak rules first).

---

# 1. Engine Breakdown and Time Ownership

**Block size**: `N = 64` samples
**Block rate**: everything executes exactly once per block.

### 1.1 Voice Engine (time-aware)

Owns:

* note lifecycle (gate rise/fall)
* pitch / glide / portamento
* ADSR / macro envelopes
* regime graph + scheduling into operator intents
* voice allocation + degrade policy

Emits **VoiceFrameV0b** per active voice per block (or sparse deltas via UART).

### 1.2 Symbol Engine (time-agnostic “transmission”)

Owns:

* coefficient state update `c ← Q( T(c) + b )`
* operator application order
* legality clamps (Q projection, tail ordering)
* tail ops (mandatory)

Consumes VoiceFrame, outputs **coeff_final[256]** per voice (or per lane).

### 1.3 Wavelet Engine (purely mechanical)

Owns:

* `IDWPT(coeff_final) → block_audio[64]`
* optional *numerical hygiene* only (format, saturate, deterministic rounding)
* **No musical time, no voice resets**.

### 1.4 Safety / Mix Engine (block-time, not note-time)

Owns:

* summing voices
* anti-clip / limiter / DC block
* *final* de-click/OLA (only for boundary invisibility)
* stereo mode

**Rule**: if something is about “note intent” it is Voice. If it is about “block boundary invisibility”, it is Safety/Wavelet.

---

# 2. New Voice Path: Coefficient Carrier Oscillator

## 2.1 Core voice render equation

Per voice `v` at block `k`:

1. **Phase update** (Voice):
   [
   \phi_{k+1} = \phi_k + \Delta\phi(f_k)\pmod{1}
   ]
   where (f_k) already includes glide.

2. **Carrier coefficient synthesis** (Wavelet-wavetable):
   [
   c^{(0)}_{k} = \mathrm{CoeffTable}[wave_id]\big(\phi_k\big)
   ]
   `c^(0)_k` is 256 coefficients representing the voice’s base waveform over this block.

3. **Operator deformation** (Symbol/CMMM):
   [
   c_k = \mathrm{Tail}\Big(Q\big(\mathcal{O}(c^{(0)}_k;; \text{VoiceFrame}_k)\big)\Big)
   ]
   Where `O` is backbone ops + event ops, and Tail is mandatory ordering.

4. **Reconstruction** (Wavelet):
   [
   x_k = \mathrm{IDWPT}(c_k)
   ]

5. **Amplitude** (Voice emits gain_q15, applied in Mix/Safety):
   [
   y_k = \mathrm{gain}_k \cdot x_k
   ]

### Key consequence

You never do: `carrier → FWPT → mask → IDWPT` in the voice path.
So you eliminate the “phase-variant mask” combing failure mode.

---

# 3. Data Interfaces

## 3.1 VoiceFrameV0b (already in your plan)

Keep your 136-byte packet. In voice path, the important fields are:

* `note_u8`, `cents_i16` (pitch)
* `gain_q15` (amp)
* `tree_a_id`, `tree_b_id`, `tree_blend_q15` (optional; see below)
* `w_family_q15[8]` (macro weights into operator resolver)
* `event_mask`, `event_strength_q15[8]`, `event_payload_u64[8]`
* `degradation_level`
* `regime_id`

### v0 freeze rule

* For the hybrid carrier, **TreeID is the primary selector** (you said this).
* Operator engine must be able to run with events missing: deterministic defaults.

## 3.2 CoeffTable contract (new)

You need a compact “wavelet wavetable” library:

**CoeffTable entry**:

* `wave_id` (u8)
* `L` phase steps (e.g. 64 or 128)
* `coeff[phase_index][256]` in `i16` or `i18`

**Lookup rule**:

* nearest-neighbor or linear interpolation across phase_index
* interpolation must be bit-exact specified

**Normalization**:

* each table entry must be normalized by a canonical rule (section 6).

---

# 4. Tail Ordering and Safety Predicate (v0 freeze)

## 4.1 Tail ordering (mandatory, reserved budget)

Tail runs after all author/operator ops, per voice, per block, unless voice is fully bypassed.

**Tail sequence (recommended v0):**

1. `FineClamp` (hard bound coefficients to safe range)
2. `NormalizeScale` (RMS/peak normalization in coefficient space)
3. `GluePair` (optional, if enabled by policy; stabilizer)
4. `SheafProject` (structural legality projection)
5. `EmergencySimplify` (conditional; only if safety predicate fires)

This matches your earlier list and preserves “tail ops mandatory overhead reserved outside budget.”

## 4.2 Tail safety predicate

Let `c` be the post-operator coefficient vector.

Define cheap invariants:

* `E = sum(c_i^2)` (energy proxy)
* `P = max(|c_i|)` (peak)
* `ΔE = |E - E_prev|`
* `ΔP = |P - P_prev|`
* `S = roughness proxy` (optional: sum |c_i - c_{i-1}|)

**Predicate fires if any:**

* `P > P_max`
* `E > E_max`
* `ΔE > ΔE_max` (block-to-block jerk too high)
* `ΔP > ΔP_max`
* or scheduler says `degradation_level == 4` AND `cpu_bar > 1.0`

**Action**:

* Always run Clamp + Normalize + Sheaf.
* Run EmergencySimplify only if predicate fires (and log `emergency_fired++`).

This gives you deterministic guardrails without “safety windowing”.

---

# 5. Degradation / Budget Model

You already have the skeleton: reserve tail cost, distribute remaining budget, degrade cloud voices.

### v0 freeze behavior

* Foreground voices: if skipped, **freeze** state (do not advance RNG or regime graph).
* Cloud voices: if skipped, **decay** activity + optionally simplify ops (do not advance RNG).

**Scheduler contract**:

* `degradation_level` governs:

  * max operator chain length
  * lowrank rank cap
  * event frequency gating
  * service period for top/mid layers

---

# 6. Canonical Normalization Rules (for “expected synth feel”)

You asked earlier: “canonical way to normalize waveform volumes?”

### 6.1 Table-time normalization (offline)

When building `CoeffTable[wave_id][phase]`:

For each phase frame:

* Compute reconstructed block `x = IDWPT(c)`
* Let `rms = sqrt(mean(x^2))`
* Scale coefficients by `target_rms / rms` (target_rms fixed, e.g. -18 dBFS in q15)
* Clamp coefficients after scaling to avoid overflow

This ensures every waveform has the same “nominal loudness” before envelopes.

### 6.2 Runtime normalization (tail)

`NormalizeScale` should be weaker and deterministic:

* enforce `rms` within `[rms_min, rms_max]`
* apply a **slow** correction (slewed) so you don’t add pumping

---

# 7. Note Transitions: Glide vs Crossfade (v0)

You asked: “Do we distinguish separate tones vs slide?”

Yes. Distinguish them in **Voice** and keep engines simple.

### 7.1 Glide (pitch-continuous)

* phase increments change smoothly (portamento)
* coefficient carrier is sampled at the new phase each block
* **no crossfade between notes**; it’s one oscillator moving

### 7.2 Retrigger / separate tone

* if gate rises while prior voice is still active, choose:

  * allocate new voice (poly) **or**
  * retrigger same voice but **do not reset** wavelet/wavelet regularity state in wavelet engine
* use **amplitude envelope** de-click (short attack) in Voice

**Rule**: crossfade is for *voice stealing* and *program changes*, not for normal pitch motion.

---

# 8. Implementation Guide

## Phase A — Python prototype refactor (fast win)

### A1) Remove FWPT from voice path

* Add `CoeffTable` module and load precomputed coefficient wavetables.
* Replace “render carrier audio → FWPT” with “render coeffs directly from phase”.

### A2) Keep FWPT only for conditioning mode

* External audio path can still do: audio→FWPT→operators→IDWPT.

### A3) Enforce “no resets” in wavelet engine state

* Never reset `prev_sample`-like state on voice_on unless voice truly inactive AND you also ramp amplitude from 0.

### A4) Add tail predicate + metrics

* Implement E/P/ΔE/ΔP tracking per voice.

### A5) Add a test that reproduces the comb and proves it’s gone

* Carrier-only with ETA-like deformation, steady state, sweep pitch.
* Compare spectrum: no 750 Hz tooth spacing.

## Phase B — FPGA architecture (clean mapping)

### B1) Memory layout

* CoeffTable stored in BRAM (or external RAM) indexed by:

  * `wave_id`
  * `phase_index`
  * `coeff_idx (0..255)`
* 64 phase steps × 256 coeff × 2 bytes = 32 KB per waveform.
* You can start with **4 waveforms** = 128 KB (fits in external, not small BRAM).
  v0 practical: **16 phase steps** + interpolation (8 KB per waveform) to fit.

### B2) Execution pipeline per block

1. Voice engine emits frames (UART→FIFO).
2. For each serviced voice:

   * read coeff base vector (stream from table)
   * apply symbol ops in fixed order
   * tail ops
   * IDWPT
3. Mix voices
4. Safety tail

### B3) Determinism requirements

* Fixed-point formats + rounding identical to Python reference.
* Operator ordering is normative; no reordering.
* PRNG per voice, draw order fixed inside Symbol engine.

---

# 9. “What’s left” (architecture closure checklist, v0)

These are the remaining **spec items** you still must freeze to avoid divergence:

1. **CoeffTable format**

* phase resolution L
* interpolation rule
* coefficient ordering canonical (leaf/band ordering)

2. **NormalizeScale exact method**

* RMS window definition (block RMS vs coefficient RMS proxy)
* slew constant

3. **Tail predicate constants**

* `P_max, E_max, ΔE_max, ΔP_max`

4. **Degrade table**

* what is dropped at each D level (rank caps, event gating)

5. **UART command set v0**

* abs vs delta updates (you deferred, fine)
* how program_id / voice_id routing works with 2 UARTs

---

# 10. Quick recommendation on your “bang for buck” question

If you want the **biggest improvement per line of code** versus trying to window harder:

✅ **Move to carrier-as-coefficients (this spec).**
It removes the structural cause of combing: *phase-variant block masking*.

Everything else (OLA, Hann, longer crossfade) is trying to sand a fundamentally phase-variant transform into behaving like a phase-invariant one. It helps—but you’ll keep chasing edge cases.

---

## Two small questions (so I can lock constants without blocking you)

1. For the wavetable phase resolution `L`: do you want **16** (cheap) or **64** (cleaner), assuming linear interpolation?
2. How many “classic” base waveforms do you want in v0: **4** (sine/tri/saw/square) or **8** (add pulse, noise-ish, etc.)?

If you don’t answer, I’ll assume **L=16** and **4 waveforms** for a first FPGA-friendly v0.
