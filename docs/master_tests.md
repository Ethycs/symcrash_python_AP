Below is a **master test specification** written as an engineering QA document.
It is organized **by engine**, focuses on **failure modes / gotchas**, and defines **what must be true** for the architecture you’ve converged on.

This is not a feature checklist — it is a **“will this instrument survive real use?”** spec.

You can hand this to:

* yourself in 3 months,
* a collaborator,
* or use it as acceptance criteria before freezing FPGA bitstreams.

---

# Master Test Specification

**Symbolic → Wavelet → Analog Audio Pipeline**

**Scope:**
Covers correctness, timing, safety, musicality, and architectural integrity of each engine.
Assumes **block size = 64 samples**, block-rate orchestration, and FPGA execution.

---

## GLOBAL INVARIANTS (apply to the whole system)

These are non-negotiable. Any failure here is a **hard bug**.

### G0. Block Invisibility

* **Requirement:** No audible discontinuity attributable to block boundaries.
* **Test:**

  * Sustain a static tone for ≥60 seconds.
  * Sweep parameters slowly and abruptly.
  * Record output; inspect waveform and spectrogram for periodic artifacts at block rate (750 Hz).
* **Gotchas:**

  * Tree switches without coefficient crossfade
  * Event injection aligned to block edges without smoothing
  * Safety limiter engaging discretely per block

---

### G1. Determinism Under Load

* **Requirement:** Same inputs → same output bit-for-bit (except PRNG-controlled ops).
* **Test:**

  * Run identical MIDI/gesture sequence twice.
  * Diff output buffers.
* **Allowed nondeterminism:**

  * Explicit PRNG lanes only (NoiseBand, stochastic GateMask).
* **Gotchas:**

  * Hidden counters in Symbol Engine
  * Voice Engine using wall-clock instead of block count
  * Floating-point leakage in Python reference vs FPGA

---

### G2. Graceful Degradation

* **Requirement:** When resources are exceeded, sound degrades musically, not catastrophically.
* **Test:**

  * Max voices + max events + max operator chains.
  * Observe: no silence, no NaNs, no runaway noise.
* **Expected behavior:**

  * Drop higher layers first (events → backbone complexity → exotic ops).
* **Gotchas:**

  * All-or-nothing operator application
  * Symbol Engine assuming “full budget” availability

---

## ENGINE 1 — VOICE ENGINE

*(The only place musical time is allowed to exist)*

### V1. Musical Time Ownership

* **Requirement:**

  * Voice Engine owns **all** timing: ADSR, regimes, phase, alternation, scheduling.
  * Symbol Engine must be time-agnostic.
* **Test:**

  * Disable Symbol Engine entirely; Voice Engine still emits valid VoiceFrames.
* **Gotchas:**

  * `local_block % N` anywhere outside Voice Engine
  * Timing logic inside operator programs

---

### V2. ADSR Correctness (Amplitude + Tree)

* **Requirement:**

  * ADSR envelopes are continuous across blocks.
  * Note-off during attack/decay behaves correctly.
* **Test:**

  * Rapid note retriggering.
  * Long release tails under heavy modulation.
* **Gotchas:**

  * Tree ADSR implemented inside wavelet generator
  * Envelope stepping at block boundaries

---

### V3. Regime Graph Stability

* **Requirement:**

  * Regime transitions exhibit hysteresis.
  * Every regime has a recovery path.
* **Test:**

  * Hold pressure at threshold boundaries.
  * Oscillate control values around regime edges.
* **Gotchas:**

  * Regime flapping every block
  * No “exit” from catastrophe regimes

---

### V4. Voice Independence

* **Requirement:**

  * Each voice is logically independent.
  * Shared engines must not leak state between voices.
* **Test:**

  * Play identical notes on different voices with different gestures.
  * Kill one voice mid-catastrophe; others continue unaffected.
* **Gotchas:**

  * Shared PRNG without per-voice salt
  * Shared temporary buffers

---

### V5. VoiceFrame Contract Integrity

* **Requirement:**

  * Every block emits a complete, self-contained VoiceFrame.
* **Test:**

  * Drop random VoiceFrames → silence only for those voices.
* **Gotchas:**

  * Implicit state assumed in Symbol Engine
  * Partial frames interpreted as valid

---

## ENGINE 2 — SYMBOL ENGINE

*(Automatic transmission: algebra, legality, smoothing)*

### S1. Statelessness w.r.t. Musical Time

* **Requirement:**

  * Symbol Engine cannot know block count, note age, or phase.
* **Test:**

  * Replay same VoiceFrame repeatedly → converges to fixed coefficient behavior.
* **Gotchas:**

  * Internal counters
  * Phase-based operator logic

---

### S2. Operator Legality Enforcement

* **Requirement:**

  * All outputs satisfy sparsity, energy, rank, and band constraints.
* **Test:**

  * Apply worst-case operator chains continuously.
* **Gotchas:**

  * Accumulating low-rank ops into dense behavior
  * Permutations creating unintended energy concentration

---

### S3. Coefficient State Stability

* **Requirement:**

  * ( c_{t+1} = Q(Tc_t + b) ) never diverges.
* **Test:**

  * Long-run simulation (minutes) with static VoiceFrame.
* **Gotchas:**

  * Injection bias not counteracted by Q
  * Numeric overflow in fixed-point arithmetic

---

### S4. Event Lane Isolation

* **Requirement:**

  * Events affect only intended bands/nodes.
* **Test:**

  * Trigger impulses in single lanes while monitoring others.
* **Gotchas:**

  * Incorrect index mapping
  * Off-by-one in leaf targeting

---

### S5. Emergency Simplify Correctness

* **Requirement:**

  * EmergencySimplify must always produce a legal, audible state.
* **Test:**

  * Force-trigger simplify under max load repeatedly.
* **Gotchas:**

  * Simplify muting output
  * Simplify introducing DC or silence

---

## ENGINE 3 — WAVELET ENGINE

*(Pure mechanical realization)*

### W1. No Musical Semantics

* **Requirement:**

  * Wavelet engine cannot know “note”, “voice”, “regime”.
* **Test:**

  * Feed random coefficients → deterministic audio.
* **Gotchas:**

  * Hidden assumptions about coefficient meaning

---

### W2. Perfect Reconstruction (Internal)

* **Requirement:**

  * Forward + inverse (where used) are numerically stable.
* **Test:**

  * Round-trip test within fixed-point tolerance.
* **Gotchas:**

  * Boundary handling errors
  * Lifting step truncation artifacts

---

### W3. Tree Interpolation Hygiene

* **Requirement:**

  * Tree switching produces no clicks.
* **Test:**

  * Rapid alternation between extreme trees.
* **Gotchas:**

  * Interpolating *after* synthesis instead of coefficients
  * Mismatched normalization across trees

---

### W4. Time-Multiplexing Safety

* **Requirement:**

  * Multiple voices sharing hardware do not bleed.
* **Test:**

  * Stress test with alternating voices per block.
* **Gotchas:**

  * Incomplete buffer clears
  * Pipeline hazards in FPGA scheduling

---

## ENGINE 4 — MATERIAL / ANALOG ENGINE

*(Intentional nonlinearity)*

### M1. Energy-Dependent Behavior

* **Requirement:**

  * Nonlinear response depends smoothly on signal energy.
* **Test:**

  * Fade signal from silence → max level.
* **Gotchas:**

  * Sudden regime jumps
  * Nonlinearities triggering safety repeatedly

---

### M2. Memory / Hysteresis Control

* **Requirement:**

  * Any stateful behavior has bounded memory and reset paths.
* **Test:**

  * Silence input after heavy saturation; output must recover.
* **Gotchas:**

  * Latent DC buildup
  * State never decays

---

### M3. Stereo Integrity

* **Requirement:**

  * Stereo modes preserve phase relationships as designed.
* **Test:**

  * Mono compatibility test
  * Mid/Side collapse test
* **Gotchas:**

  * Asymmetric distortion
  * Channel drift

---

## ENGINE 5 — SAFETY ENGINE

*(Physics + user protection)*

### F1. Never Silence the Instrument

* **Requirement:**

  * Safety may compress or color, but never hard-mute unless explicitly commanded.
* **Test:**

  * Drive system into extreme overload.
* **Gotchas:**

  * Limiter gain → zero
  * Slew limiter freezing output

---

### F2. Click and Spike Suppression

* **Requirement:**

  * All fast discontinuities are smoothed.
* **Test:**

  * Hard operator toggles every block.
* **Gotchas:**

  * Slew limiting only per-sample, not per-block
  * No lookahead for limiter

---

### F3. Alias Policy Correctness (K modes)

* **Requirement:**

  * Aliasing character changes as designed, never uncontrolled.
* **Test:**

  * Sweep K modes under bright signals.
* **Gotchas:**

  * Aliasing interacting destructively with material stage

---

## CROSS-ENGINE INTEGRATION TESTS

### I1. “Automatic Transmission” Test

* **Goal:** Prove that Symbol + Wavelet behave like a transmission.
* **Test:**

  * Slowly vary a single gesture (pressure).
  * Confirm smooth change in sound without explicit operator switching.
* **Failure mode:** Audible gear “clunks”.

---

### I2. Voice Dropout Test

* **Goal:** Killing voices is safe.
* **Test:**

  * Rapidly allocate/free voices.
* **Expected:** No pops, no DC, no timing glitch.

---

### I3. Long-Form Musical Test

* **Goal:** Prove compositional viability.
* **Test:**

  * 5–10 minute autonomous performance using only Voice Engine trajectories.
* **Watch for:**

  * Drift, fatigue, collapse, repetition.

---

## FINAL ACCEPTANCE CRITERIA

The system is **ready** when:

* ✅ No engine violates its time ownership.
* ✅ No audible block-rate artifacts exist.
* ✅ Any failure degrades musically.
* ✅ Voice Engine alone can “play” the instrument.
* ✅ Symbol + Wavelet engines can be reasoned about as math, not music.
* ✅ The instrument feels *analog* under continuous control.

---

### One-sentence architectural litmus test

> *If I rip out the UI and drive only the Voice Engine with curves, the instrument should still perform a convincing piece of music.*

If you want next, I can:

* turn this into a **Python pytest-style harness**, or
* derive a **minimum set of FPGA assertions** you can embed in simulation.
