Yep — this addition is _compatible_ with all the choices you’ve already locked, and it’s a great use of your primitives. The key is exactly what you wrote: **treat classic wave types as coefficient-space targets/injections**, not as a parallel audio synth.

Below is the **formal forward-mode v0 spec addendum** for a **Harmonic Target Injector** that plugs into your existing pipeline with minimal new state and zero new architecture.

---

## Addendum: Harmonic Target Injector v0 (Baby Fourier)

### 0) Where it lives in the pipeline

- **Voice Engine / Tower IR**: decides _how harmonic_ the voice should be, selects wave type, sets “pull” and “dial” parameters.
- **Symbol Executor**: applies the harmonic influence as part of $b_t$ / injection in **Phase B INJECT**, then runs the **mandatory tail**.
- **Wavelet Engine**: unchanged (still renders $c_t$ with β-tree morph).

So you keep one coherent system: **coefficients remain the shared state**.

---

### 1) New primitive: `HarmonicTargetGenerator`

Per voice, per block (or every N blocks), generate a sparse or dense target vector $c_{harm}$ in coefficient space (or leaf-energy space).

#### Inputs (per voice)

- `note_u8`, `cents_i16` (from packet)
- `wave_type` (enum)
- `partials_N` (u8, recommended ≤ 16)
- `brightness/tilt` (q15)
- `phase_rand` (q15) (optional)
- `harmonic_dial \eta$ (q15) 0..1 (how much harmonic component to mix in)
- `pull_strength \alpha$ (q15) 0..1 (how strongly to converge toward target)

#### Output

- either dense $c_{harm}[256]$ (expensive), OR
- **recommended v0:** sparse (leaf_mask, coeff_mask, amp_q15[targets]) to write into $b_t$

You don’t need exact phase reconstruction to get the “classic wave feel”; you mainly need **band energy structure** and stable pitch anchoring.

---

### 2) How it enters the Symbol update

You already have a clean place for this:

#### Option v0 (recommended): treat as an injection command

Use your existing lane:

- **Lane 4 `EVT_INJECT_NOTE`**
    - `variant` encodes _harmonic injection mode_
    - params encode wave type / N / brightness / pull
    - targeting uses pattern/band id

Then Symbol Executor does (in Phase B INJECT):

$$
b_t \leftarrow b_t ;+; \alpha (c_{harm} - c_t) \quad \text{(applied only on targeted coeffs)}
$$

Followed by the normal:

- backbone ops
- tail Q (always)

This gives you the “pull toward harmonic skeleton” without introducing a new engine.

---

### 3) Event payload semantics for harmonic injection (Lane 4 extension)

#### Lane 4 — EVT_INJECT_NOTE (extended for harmonic target)

**variant**

- 0 harmonic target pull (sine/saw/pulse/tri)
- 1 harmonic + phase drift (adds controlled randomness)
- 2 harmonic-noise hybrid (adds band-limited noise floor)

**params**

- `param0_q8`: wave_type (enum)
- `param1_q8`: partials_N (1..16)
- `param2_q8`: pull_strength $\alpha$ in q8 (0..255)

**strength**

- `event_strength_q15[4]` is the harmonic dial $\eta$ (0..1)
    - $\eta$ = “how much harmonic target participates”
    - $\alpha$ = “how hard we converge per block”

This separates “amount present” vs “convergence speed.”

---

### 4) Wave types (v0 set)

Wave type enum:

- 0 SINE
- 1 SAW
- 2 PULSE (50% default)
- 3 TRIANGLE
- 4 NOISE (optional: handled by existing NoiseBand; keep here reserved)

Amplitude law per partial ($k = 1..N$):

- **SINE**: only $k=1$
- **SAW**: $a_k = 1/k$
- **PULSE**: $a_k = 1/k$ with even/odd weighting (v0: odd-only for “square-like”; later you can add duty cycle)
- **TRIANGLE**: odd-only with $a_k = 1/k^2$

Brightness/tilt multiplies ($a_k$) by a simple exponential rolloff (cheap in fixed point):

- precompute a tiny LUT for $k \mapsto w_k(\text{brightness})$

---

### 5) Mapping partials → leaves (cheap “harmonic to band”)

You don’t want an FFT. You want a deterministic mapping from frequency to leaf index.

Define a compile-time table:

- `leaf_center_hz[i]` for $i=0..31$, for each tree family/order you support (or for canonical low→high ordering)

Then for each partial frequency:  
$$
f_k = k \cdot f_0
$$
Pick leaf:  
$$
i_k = \arg\min_i |leaf\_center\_hz[i] - f_k|
$$

In hardware, don’t do argmin; do:

- a monotone threshold table `leaf_edges_hz[i]`
- find the bucket (simple compare chain or binary search)

So you just bucket partials into leaves.

#### Coeff index inside leaf

v0 simplest:

- write energy into coeff $j=0$ (or distribute across a fixed mask like $[0,1]$ for a bit of width)
- later you can use $j$ to encode phase-ish texture, but it’s optional

---

### 6) Per-block algorithm (FPGA-friendly)

Per voice per block (if harmonic injection active):

1. Compute $f_0$ from `note_u8 + cents` using a small LUT (or compute phase increment directly)
2. For $k=1..N$:
    - compute $f_k = k f_0$ (integer multiply)
    - map to leaf $i_k$
    - amplitude $a_k$ from wave type law × brightness rolloff
3. Build a sparse target mask of leaves (often only N leaves)
4. Apply injection:
    - for each targeted leaf:
        - $\Delta = (target\_amp - current\_coeff)$ (or just $target\_amp$ if you keep it purely additive)
        - $b += \alpha \Delta$
5. Proceed with the normal operator pipeline and mandatory tail

This uses only:

- small LUTs
- integer multiplies
- sparse writes

---

### 7) Minimal new per-voice state

You only need **one** of these, depending on how “oscillator-like” you want it.

#### v0 minimal (block-stationary target)

- **no phase state** required
- target depends only on note/brightness/dial
- gives stable pitch anchoring without “oscillator phase continuity”

This is the cheapest and usually enough for “classic wave character” inside your system.

#### v0+ (adds analog drift / mild motion)

Add:

- `phase_acc_u32` (per voice)
- optional `drift_u16` (slowly varying)
- used only to rotate which coeff indices get written (tiny motion), not to generate samples

Still very cheap and doesn’t become a second synth.

---

### 8) Why this is a good decision given your primitives

- keeps **one state**: coefficients
- leverages your existing **INJECT lane** and **tail legality**
- improves playability (stable harmonic anchors) without building a VCO/filter/audio-rate path
- FPGA cost is tiny: partial bucketing + sparse injection

---

### 9) Where it fits in Tower IR

At Tower levels:

- **L0**: “Wave Type” + “Harmonic” + “Brightness” macros compile to lane 4 injections + family weights
- **L2/L3**: regimes decide when harmonic dial is strong vs weak
- **L4**: events can trigger harmonic snaps (short $\alpha$ bursts) on attacks

No new editor vocabulary needed.

---

If you want the next crank: I can add this to the **canonical packet diagram** as a “Lane 4 variant table” entry (so it’s formally part of the ABI), and I can propose a default mapping for your four instruments (e.g., Static Organ uses strong harmonic dial; Grass Wind uses almost none; Glitch Violin uses moderate dial with chaotic overlays).