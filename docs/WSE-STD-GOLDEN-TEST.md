Below is a **self-contained “golden standards” document** you can drop into your repo as a WSE standard. It’s written to be practical: what to test, why, how to structure artifacts, and when to regenerate.

---

# WSE-STD-GOLDEN-TESTS v0 — Golden Standards for Bit-Exact Audio + ABI Stability

## 0. Purpose

This standard defines the **minimum golden test suite** required to keep the Wavelet/Symbol Engine (WSE) pipeline:

* **bit-exact** (same inputs → identical output bits),
* **deterministic** (no hidden state leakage, PRNG discipline),
* **ABI-stable** (VoiceFrame semantics don’t drift),
* **bounded** (budget/degradation remains predictable),
* **safe to refactor** (aggressive optimization without “quiet musical changes”).

This is the test harness that lets you move fast on **presets** without fear.

## 1. Terminology

* **Block**: 64 samples at 48 kHz (or configured); the only time quantum.
* **VoiceFrame**: ABI command object (WSE-STD-020).
* **VM**: Hybrid VM executor implementing lane schedule (WSE-STD-VM-HYBRID).
* **Golden Trace**: Frozen sequence of frames + frozen output hashes.
* **Artifact**: A versioned file containing golden hashes + regeneration audit.

## 2. Golden Philosophy (Normative)

### 2.1 What must be frozen

You MUST freeze artifacts for things that can change “quietly”:

* lane semantics ordering
* phase continuity rules
* PRNG advancement rules
* tail legality semantics
* budget/degradation ladder outputs
* operator outputs for key operators
* baseline “musical sound” traces

### 2.2 What must *not* be golden

You SHOULD NOT golden-test:

* performance timing (use profiling, not hashing)
* floating-point DSP output (goldens assume integer-only bit exactness)
* UI rendering behavior

### 2.3 Regeneration policy (Normative)

Golden artifacts MUST NOT be regenerated silently.

Any regeneration MUST:

* print old vs new hash
* record a timestamped audit report
* include git commit hash
* include version metadata (VM, VoiceFrame, sine LUT, etc.)

## 3. Required Golden Test Categories

### Category A — Determinism & Bit-Exact Invariants (Unit/Property)

**Goal:** Prove the VM is deterministic at the block level.

#### A1. Integer-only execution

* No floats in execution path (decode + lanes + tail).
* If floats exist in authoring code, they MUST compile to integer tables before runtime.

**Tests:**

* `test_no_floats_in_vm_path` (static grep / runtime guard)
* `test_bit_exact_output_across_runs`

#### A2. Phase continuity

* Phase MUST advance monotonically across blocks.
* Phase MUST NOT reset unless explicitly requested (e.g., retrigger policy).

**Tests:**

* `test_phase_monotonic`
* `test_phase_not_reset_between_blocks`

#### A3. PRNG discipline

* PRNG MUST advance **only** when an op that requires randomness runs (e.g., `NoiseBand`).
* PRNG MUST be voice-local (no shared global draws).

**Tests:**

* `test_prng_only_advances_on_noise`
* `test_different_voices_produce_different_prng`
* `test_snapshot_restore_determinism`

#### A4. Overflow/saturation policy

* Q15 operations MUST clamp deterministically.
* Saturation symmetry required (+x and -x behave symmetrically).

**Tests:**

* `test_vm_overflow_*` suite (existing)

---

### Category B — Golden Audio Traces (ABI Regression)

**Goal:** Freeze “musically meaningful” and “semantic stress” sequences.

#### B1. Golden Trace: Content (musical)

Example: **Brass stab** (classic sound).

**Artifact:**

* `golden_trace_v0.py` with:

  * frozen block hashes
  * final trace hash
  * version header
  * regenerate audit

**Test:**

* `test_golden_trace_matches_frozen`

#### B2. Golden Trace: Stress (semantic coverage)

Example: **riser → noise gating → permute motion → hit → resolve**

Must exercise:

* Noise PRNG gating
* Permute ops
* K ladder change (pattern_id)
* Tail ladder change
* Event lanes
* Smooth ramps

**Artifact:**

* `golden_trace_stress_v0.py`
* `test_golden_trace_stress.py` with coverage assertions

**Normative:** Stress trace MUST include explicit coverage tests, so it cannot silently stop exercising the features.

---

### Category C — Golden Degradation Plans (Boundedness Regression)

**Goal:** Freeze budget/degradation semantics so “real-time guarantees” can’t drift.

This is *not audio*. This is plan-level regression.

#### C1. Plan signature

A **PlanSignature** is the minimal deterministic output of the degradation step:

```
PlanSignature {
  regime_id
  voice_class
  requested_ops
  requested_tail_mode
  requested_pattern_id (or requested mask)

  final_K
  final_pattern_id
  final_tail_mode
  final_backbone_ops
  final_event_ops
  predicted_ticks
  uses_nonlinear
  is_emergency
  steps_taken
}
```

#### C2. Required scenarios (minimum set)

MUST include at least:

* STABLE_TONE with K=32 request
* TRANSIENT with nonlinear request in a nonlinear-disallowed budget
* RESIDUAL with aggressive request
* IDLE budget sanity
* CARRIER_ONLY budget sanity
* EMERGENCY fallback baseline

**Artifact:**

* `golden_degradation_v0.py` containing frozen hashes of the scenario set.

**Test:**

* `test_golden_degradation_matches_frozen`

**Normative:** Any change to degradation ladder requires explicit regen + audit.

---

### Category D — Operator Semantics Goldens (Micro-tests)

**Goal:** Prevent “optimization drift” in core operators.

Pick 3–6 operators that are mission-critical:

* FineClamp (2)
* NormalizeScale (11)
* SheafProject (23)
* RotateNodes (10) or BandShear (6)
* LowRankMorph (7) (optional)
* NoiseBand (14) (optional; but PRNG semantics already covered)

#### D1. Canonical inputs

Define fixed test coefficient vectors:

* all zeros
* impulse in low band
* impulse in high band
* random-but-fixed (seeded) coefficients
* “worst-case” near-saturation coefficients

Run each operator at:

* K=32, K=16, K=1 masks (or pattern_id 0,2,6)

Freeze the output hash (or full vector hash).

**Tests:**

* `test_op_fineclamp_golden_*`
* `test_op_normalize_golden_*`
* etc.

---

### Category E — ABI / Decode / Completion Goldens

**Goal:** VoiceFrame completion rules and decoding must not drift.

#### E1. Completion behavior

Given a sequence of VF_DELTA frames omitting certain fields:

* verify latch semantics match spec
* verify omitted harmonic amps latch correctly (your earlier fix)

**Tests:**

* `test_decode_field_completion_*`
* `test_harmonics_latch_when_omitted`

#### E2. Lane ordering invariance

Lane schedule is a semantic contract:
`CARRIER_RENDER → INJECT → BACKBONE → TAIL → IDWPT_RENDER → MIX_GAIN`

**Tests:**

* `test_lane_schedule_order_constant` (simple assertion against a constant table)

---

## 4. Artifact Format and Placement

### 4.1 Directory layout (recommended)

```
pipeline_tests/
  artifacts/
    golden_trace_v0.py
    golden_trace_stress_v0.py
    golden_degradation_v0.py
    regen_reports/
      .gitkeep
  test_artifacts/
    test_golden_trace.py
    test_golden_trace_stress.py
    test_golden_degradation.py
```

### 4.2 Artifact headers (Normative)

Each golden artifact MUST include:

* trace/degradation version
* VM standard version
* VoiceFrame standard version
* sine LUT version
* a short intent string

### 4.3 Hashing rule (Normative)

A block hash MUST be computed over the **exact output bytes** (little-endian, fixed dtype).
The final trace hash MUST be computed over the concatenation of all block hashes.

---

## 5. Regeneration Procedure (Normative)

Golden regeneration MUST:

1. Print VERSION_INFO
2. Print OLD TRACE HASH
3. Print NEW TRACE HASH
4. Print STATUS: UNCHANGED or CHANGED
5. Write an audit report to `regen_reports/regen_YYYYMMDD_HHMMSS.txt`
6. Include git commit hash (or “unknown” if not available)
7. Require an explicit `--regenerate` flag

---

## 6. “Ready to Ship” Checklist

A build is “golden clean” iff:

* All determinism tests pass
* Both golden traces match frozen hashes
* Degradation golden matches frozen hashes
* Operator goldens pass for core ops
* No float usage in runtime path
* Regen reports exist for any updated artifacts

---

## 7. What this buys you (practical outcome)

If this spec is implemented, you can:

* refactor the VM aggressively
* change internal representations
* optimize for FPGA mapping
* add new presets constantly

…without losing:

* bit-exact replay
* click-free behavior
* PRNG correctness
* bounded real-time guarantees
* musical identity of existing presets

---

## Appendix A — Recommended “Future” Golden Additions (Optional)

* **Carrier role toggle golden** (AUDIBLE ↔ EXCITER_ONLY) once `GoldenFrame` exposes it
* **Multi-voice interference trace** (two voices detuned + shared mixer)
* **Budget stress trace** (many voices competing → ensure deterministic degradation ordering)

---

If you want, I can also write a **Preset Development Standard** (WSE-STD-PRESETS v0) that tells you how to author L0 classic sounds using your operator set (tilt/intramix/diffuse + tail), and how to map “producer language” (brightness/body/punch/air) to specific operator families. That’s the bridge from this test foundation to “now ship a killer preset pack.”
