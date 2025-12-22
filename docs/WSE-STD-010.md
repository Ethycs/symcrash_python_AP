## WSE-STD-010 — Tree Library Contract (v0)

**Status:** NORMATIVE (MUST/SHOULD language)
**Scope:** Defines the *canonical* wavelet packet tree “library” used by Voice/Symbol/Wavelet engines. Freezes: IDs, invariants, leaf ordering, band semantics, and morph legality.

This standard exists to prevent Python/FPGA/editor divergence in:

* leaf/band indexing,
* event targeting,
* tree morph behavior,
* per-tree gain/normalization.

---

# 0. Definitions

### 0.1 Block and coefficient layout

* **Block size**: `B = 64` audio samples per block.
* **Leaf count**: `L = 32` leaves (bands) per block.
* **Coefficients per leaf**: `K = 8`.
* **Coefficient vector**: `c ∈ Z^(256)` in canonical order.

**Canonical packing (MUST):**

* `leaf i` occupies coefficient indices `[8*i + 0 .. 8*i + 7]`.
* `i = 0..31` in **canonical leaf order** (see §1).

### 0.2 Tree

A **Tree** is a discrete schedule + metadata that maps a coefficient vector to audio via inverse lifting (and optionally forward analysis). Trees are addressed by **tree_id**.

---

# 1. Canonical leaf ordering and frequency semantics

### 1.1 Canonical leaf order (MUST)

The library defines a single canonical leaf order:

* **Leaf 0 = lowest-frequency band**
* **Leaf 31 = highest-frequency band**

This is the only ordering exposed to:

* editor,
* VoiceFrame event payload targeting,
* Symbol operator “band” arguments.

**Internal implementation ordering MAY differ**, but MUST be mapped bijectively to/from canonical order.

### 1.2 Band boundary semantics (MUST)

A tree MUST define (or imply) a band boundary table:

* `f_lo[i], f_hi[i]` for leaves `i=0..31`, in Hz (or fixed-point Hz).
* Bands MUST cover `[0, Fs/2]` monotonically and without gaps:

  * `f_lo[0] = 0`
  * `f_hi[31] = Fs/2`
  * `f_lo[i] < f_hi[i]`
  * `f_hi[i] = f_lo[i+1]` (exact adjacency is recommended; tolerances allowed only if explicitly stated).

### 1.3 Fs dependence (MUST)

All boundary computations are relative to sample rate `Fs`.
If the system supports multiple `Fs`, each tree MUST be valid for all supported `Fs` or the library MUST provide per-`Fs` variants.

---

# 2. Tree identity and descriptor

### 2.1 tree_id (MUST)

* `tree_id` is a 16-bit unsigned identifier.
* `tree_id = 0` MUST be reserved for **TREE_BYPASS_DEFAULT** (the default stable tree used when nothing is specified).

### 2.2 Descriptor (normative schema)

Each tree_id maps to a **TreeDescriptor**:

```c
typedef struct {
  uint16_t tree_id;              // unique key
  uint8_t  family_id;            // CDF53_WPT=1, etc.
  uint8_t  leaf_count;           // MUST be 32 in v0
  uint8_t  coeffs_per_leaf;      // MUST be 8 in v0
  uint8_t  leaf_order_id;        // MUST be CANON_LOW_TO_HIGH in v0

  // Reconstruction normalization
  int16_t  recon_gain_q15;       // unity calibration for this tree
  int16_t  coeff_energy_gain_q15;// optional: maps coeff energy to audio RMS estimate

  // Frequency mapping (optional if implied by family+params)
  uint16_t f_lo_q10_6[32];       // optional
  uint16_t f_hi_q10_6[32];       // optional

  // Schedule / topology payload
  // For FPGA-friendly operation, this MUST be representable as a fixed schedule table.
  uint16_t schedule_words[/*fixed N*/];

  // Hash for toolchain + bit-exact agreement
  uint32_t descriptor_crc32;
} TreeDescriptor;
```

Notes:

* `schedule_words[]` format is implementation-defined **but MUST be frozen** within the tree library and stable across Python/FPGA for the same `tree_id`.
* The library MAY omit `f_lo/f_hi` if derivable deterministically from `(family_id, params, Fs)`.

---

# 3. Validity invariants (MUST)

A tree is **valid** iff all of the following hold:

### 3.1 Structural invariants

* `leaf_count == 32`
* `coeffs_per_leaf == 8`
* canonical leaf order mapping exists and is bijective.

### 3.2 Numerical safety invariants

* The inverse reconstruction MUST be bounded under unit coefficient input:

  * There exists a published constant `G_tree` such that:

    * If `|c[j]| ≤ Cmax` for all j, then `|x[n]| ≤ G_tree * Cmax` for all n in the block.
* `recon_gain_q15` MUST be chosen so that a reference coefficient pattern yields near-unity output level (see §4).

### 3.3 Determinism invariants

* For fixed `(tree_id, Fs)` and fixed `c`, the reconstruction MUST be bit-exact under WSE-STD-001 arithmetic rules.
* The descriptor CRC MUST match at load time; mismatch MUST cause deterministic fallback to `TREE_BYPASS_DEFAULT`.

---

# 4. Unity and calibration

### 4.1 Recon gain semantics (MUST)

`recon_gain_q15` is applied as a final scalar to the reconstructed block:

* `x_out[n] = mul_q15(recon_gain_q15, x_raw[n])`

This gain MUST be part of the tree descriptor, not an external “preset tweak”.

### 4.2 Reference calibration procedure (SHOULD)

Each tree SHOULD be calibrated by a standard test vector:

* single-leaf excitation at multiple leaves,
* broadband excitation,
* ensure consistent perceived level across trees (within tolerance).

The library SHOULD record the calibration method (even just as doc text) so future trees are comparable.

---

# 5. Tree parameterization

### 5.1 Family IDs (v0)

At minimum, v0 MUST support:

* `family_id = 1`: **CDF 5/3 wavelet packet tree** (lifting-based)

Other families MAY be added later but MUST NOT change semantics of existing tree_ids.

### 5.2 Beta trees (if used) (MAY / MUST if present)

If you support “β trees” (a continuous parameter controlling band allocation/topology):

* A **beta tree** is not a tree_id by itself; it MUST compile to a tree_id in the library OR compile to a **TreeDescriptor-equivalent schedule** whose CRC can be computed.
* If you allow runtime-generated beta schedules, they MUST still satisfy all invariants in §3.

---

# 6. Tree morph contract

Tree morphing is a first-class behavior and MUST be standardized.

### 6.1 Morph legality (MUST)

The library MUST provide a **morph whitelist**:

* `allowed_morph(tree_a, tree_b) -> bool`

If a morph is not allowed:

* the engine MUST refuse it deterministically (replace with nearest allowed or hold tree_a).
* “Nearest” policy is implementation-defined, but MUST be deterministic and documented.

### 6.2 Morph semantics (MUST)

Morphing between two trees MUST choose exactly one of these normative modes in v0:

**Mode M1 — Coefficient-space blend (RECOMMENDED for your architecture):**

1. Generate descriptors/schedules for `tree_a` and `tree_b`.
2. Use a canonical coefficient order for both.
3. Blend coefficients before inverse reconstruction:

   * `c_blend = (1-α) * c_a + α * c_b` (fixed-point blend; see STD-001)
4. Run inverse lifting once on `c_blend`.

**Mode M2 — Audio crossfade:**

1. Reconstruct audio separately: `x_a = Inv(tree_a, c)` and `x_b = Inv(tree_b, c)`
2. Crossfade audio:

   * `x = (1-α)*x_a + α*x_b`
3. Apply OLA/safety as usual.

**v0 MUST pick exactly one as default.**
If both are supported, a `morph_mode` field MUST be explicit and versioned.

### 6.3 Morph continuity (MUST)

Tree morph MUST be click-safe under normal operation:

* Either by coefficient blending continuity (M1),
* Or by audio crossfade + OLA (M2).

If morph causes clipping or instability, Safety tail may clamp, but behavior MUST remain deterministic.

### 6.4 Morph blend parameter

* `α` (“tree_blend”) is Q0.16 or Q1.15 (choose one and freeze globally).
* `α=0` means “pure tree_a”, `α=max` means “pure tree_b”.

---

# 7. Leaf targeting and mapping rules

### 7.1 Canonical targeting (MUST)

All external targeting (events, UI, editor, presets) MUST target canonical leaf IDs `0..31`.

### 7.2 Internal mapping (MUST)

If the engine stores leaves in a different physical order, it MUST provide:

* `phys = map_canon_to_phys(tree_id, canon_leaf)`
* `canon = map_phys_to_canon(tree_id, phys_leaf)`

Both MUST be bijections.

### 7.3 Tree switches and mapping stability (MUST)

When a tree changes (switch or morph), any persistent state indexed by leaf MUST be interpreted in canonical order, then remapped to physical order for the active tree.

This prevents “leaf meaning drift” when switching trees.

---

# 8. API surface the engines rely on

The tree library MUST provide these functions (conceptually; could be ROM tables in FPGA):

1. `get_tree_descriptor(tree_id) -> TreeDescriptor*`
2. `validate_tree(tree_id, Fs) -> bool`
3. `get_band_limits(tree_id, Fs, leaf) -> (f_lo, f_hi)`
4. `allowed_morph(tree_a, tree_b) -> bool`
5. `map_canon_to_phys(tree_id, leaf) -> leaf_phys`
6. `map_phys_to_canon(tree_id, leaf_phys) -> leaf`

---

# 9. Defaulting and failure behavior

### 9.1 Default tree (MUST)

If any subsystem requests an invalid tree_id, the system MUST substitute `TREE_BYPASS_DEFAULT` deterministically.

### 9.2 Missing metadata (MUST)

If frequency boundaries are not explicitly stored, they MUST be derived deterministically from the descriptor, family_id, and Fs.

### 9.3 Version mismatch (MUST)

If descriptor CRC or library version mismatch occurs:

* reject the descriptor,
* fall back to default tree_id,
* signal a status flag (for UI/debug) but keep audio running.

---

# 10. What this standard does NOT decide (explicitly out of scope)

* Operator algebra semantics (STD-030).
* Event payload formats (STD-020).
* Fixed-point math rules (STD-001).
* Voice engine timing (STD-060).

---

## v0 Freeze Summary (the “can’t change later” bits)

* `B=64`, `L=32`, `K=8`, coeff packing per leaf.
* Canonical leaf order low→high.
* Tree IDs + descriptor CRC enforcement.
* Morph whitelist + chosen morph semantic mode.
* Canonical targeting + bijective mapping functions.
* Deterministic fallback behavior.

---

If you answer just one thing, we can lock this completely:

**For v0, do you want tree morph to be M1 (coefficient-space blend) or M2 (audio crossfade)?**

I’ll assume **M1** by default because it matches your “automatic transmission” wavelet engine and keeps the tail/safety story clean.
