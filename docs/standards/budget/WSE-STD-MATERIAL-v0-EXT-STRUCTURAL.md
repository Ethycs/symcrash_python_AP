# WSE-STD-MATERIAL v0 Extension: Structural Failure Programs

**Status:** DRAFT
**Extends:** WSE-STD-MATERIAL v0
**Applies to:** Material Engine programs 6-7

---

## 0. Purpose

This extension adds two structural mechanics-inspired material programs:

| material_id | Name | Model | Character |
|-------------|------|-------|-----------|
| 6 | Steel Yield | Bouc-Wen hysteresis (n=1) | "Metal under load" |
| 7 | Brittle Fracture | Bouc-Wen + damage events | "Cracking/breaking" |

These programs model **material memory** - the sound "remembers" how hard it was pushed.

---

## 1. Program 6: Steel Yield (Bouc-Wen Hysteresis)

### 1.1 Physical Model

Bouc-Wen models smooth hysteresis used in structural engineering for steel under cyclic loading.

**Governing equation (continuous):**
```
dz/dt = A*dx/dt - beta*|dx/dt|*|z|*z - gamma*dx/dt*|z|^2
y = k*x + alpha*z
```

For n=1 (simplified, FPGA-friendly):
```
dz/dt = A*dx/dt - beta*|dx/dt|*z - gamma*dx/dt*|z|
```

### 1.2 State

```c
struct SteelState {
    i16 z_q15;           // Hysteresis state (internal "strain")
    i16 x_prev_q15;      // Previous sample (for dx calculation)
}
```

### 1.3 Parameters

| Parameter | Field | Range | Default | Meaning |
|-----------|-------|-------|---------|---------|
| Stiffness | `stiffness_q15` | 0-32767 | 24576 | Elastic response (k) |
| Yield | `yield_q15` | 0-32767 | 16384 | Yield threshold (affects beta/gamma) |
| Hysteresis | `hysteresis_q15` | 0-32767 | 8192 | Loop width (alpha blend) |

### 1.4 Algorithm (Q1.15 Bit-Exact)

```python
def steel_yield_sample(x: i16, state: SteelState, params: SteelParams) -> i16:
    """
    Bouc-Wen hysteresis (n=1 simplified).

    Models: elastic-plastic deformation with memory.
    Sound: "metal under load", yielding compression.
    """
    # 1. Compute velocity (dx)
    dx = SAT16(x - state.x_prev)
    state.x_prev = x

    # 2. Bouc-Wen coefficients from params
    # A = stiffness (how fast z follows x)
    # beta, gamma control hysteresis shape
    A = params.stiffness_q15

    # Scale beta/gamma inversely with yield (lower yield = more hysteresis)
    yield_inv = 32767 - params.yield_q15  # High yield = less nonlinearity
    beta = MUL_Q15(yield_inv, 8192)   # ~0.25 at max
    gamma = MUL_Q15(yield_inv, 4096)  # ~0.125 at max

    # 3. Bouc-Wen update: dz = A*dx - beta*|dx|*z - gamma*dx*|z|
    abs_dx = abs(dx)
    abs_z = abs(state.z)

    term1 = MUL_Q15(A, dx)                          # A * dx
    term2 = MUL_Q15(MUL_Q15(beta, abs_dx), state.z) # beta * |dx| * z
    term3 = MUL_Q15(MUL_Q15(gamma, dx), abs_z)      # gamma * dx * |z|

    dz = SAT16(term1 - term2 - term3)

    # 4. Integrate z (slew-limited for stability)
    # Rate controls how fast hysteresis responds
    rate = 4096  # Q1.15 = 0.125 (stable integration)
    state.z = SAT16(state.z + MUL_Q15(dz, rate))

    # 5. Output: y = k*x + alpha*z
    # k = linear elastic component
    # alpha = hysteresis blend
    k = MUL_Q15(params.stiffness_q15, 24576)  # ~0.75 * stiffness
    alpha = params.hysteresis_q15

    y_elastic = MUL_Q15(k, x)
    y_hysteresis = MUL_Q15(alpha, state.z)

    return SAT16(y_elastic + y_hysteresis)
```

### 1.5 Sound Character

- **Low drive**: Stiff, "resisting" tone
- **High drive**: Yielding compression that *gives* rather than clips
- **Cyclic input**: Hysteresis loop - different path on return
- **Memory**: Permanent set accumulates slowly

### 1.6 Complexity

~25 ticks per sample (6 MUL_Q15, 3 abs, saturations)

---

## 2. Program 7: Brittle Fracture (Bouc-Wen + Damage Events)

### 2.1 Physical Model

Extends Steel Yield with:
- **Damage accumulation**: Stress cycles increase damage state
- **Micro-fracture events**: When damage crosses thresholds, inject transients
- **Stiffness degradation**: Material "softens" as damage increases

### 2.2 State

```c
struct FractureState {
    // Bouc-Wen base state
    i16 z_q15;           // Hysteresis state
    i16 x_prev_q15;      // Previous sample

    // Damage tracking
    i16 damage_q15;      // Accumulated damage (0 = pristine, 32767 = failed)
    u8  cooldown;        // Blocks until next crack event allowed
    u8  crack_pending;   // Crack event queued for this block
}
```

### 2.3 Parameters

| Parameter | Field | Range | Default | Meaning |
|-----------|-------|-------|---------|---------|
| Stiffness | `stiffness_q15` | 0-32767 | 24576 | Initial elastic response |
| Yield | `yield_q15` | 0-32767 | 16384 | Yield threshold |
| Brittleness | `brittleness_q15` | 0-32767 | 8192 | Damage accumulation rate |
| Crack Intensity | `crack_q15` | 0-32767 | 12288 | Fracture event amplitude |

### 2.4 Algorithm (Q1.15 Bit-Exact)

```python
def brittle_fracture_sample(x: i16, state: FractureState, params: FractureParams,
                            sample_idx: int) -> i16:
    """
    Bouc-Wen + damage accumulation + micro-fracture events.

    Models: material fatigue, cracking, structural failure.
    Sound: "metal breaking", transient pops, degrading stiffness.
    """
    # 1. Compute effective stiffness (degrades with damage)
    # As damage increases, stiffness drops
    damage_factor = 32767 - (state.damage >> 1)  # 50% stiffness loss at max damage
    effective_stiffness = MUL_Q15(params.stiffness_q15, damage_factor)

    # 2. Run Bouc-Wen with degraded stiffness
    dx = SAT16(x - state.x_prev)
    state.x_prev = x

    A = effective_stiffness
    yield_inv = 32767 - params.yield_q15
    beta = MUL_Q15(yield_inv, 8192)
    gamma = MUL_Q15(yield_inv, 4096)

    abs_dx = abs(dx)
    abs_z = abs(state.z)

    term1 = MUL_Q15(A, dx)
    term2 = MUL_Q15(MUL_Q15(beta, abs_dx), state.z)
    term3 = MUL_Q15(MUL_Q15(gamma, dx), abs_z)

    dz = SAT16(term1 - term2 - term3)
    state.z = SAT16(state.z + MUL_Q15(dz, 4096))

    # 3. Accumulate damage from stress cycles
    # Damage increases when |z| is high (material under stress)
    stress_energy = MUL_Q15(abs_z, abs_z)  # z^2 proxy for stress energy
    damage_rate = MUL_Q15(params.brittleness_q15, stress_energy)
    damage_rate = damage_rate >> 8  # Slow accumulation

    state.damage = min(32767, state.damage + damage_rate)

    # 4. Check for crack event
    crack_impulse = 0
    if state.cooldown > 0:
        state.cooldown -= 1
    else:
        # Probability of crack increases with damage
        # Deterministic: crack when damage crosses thresholds
        damage_threshold = 8192 + (sample_idx & 0x1FFF)  # Pseudo-random threshold
        if state.damage > damage_threshold and abs_z > (params.yield_q15 >> 1):
            # CRACK EVENT
            crack_impulse = MUL_Q15(params.crack_q15,
                                    (16384 + (state.damage >> 2)))  # Louder with damage
            # Sign follows stress direction
            if state.z < 0:
                crack_impulse = -crack_impulse

            state.cooldown = 8  # 8 samples minimum between cracks

    # 5. Output: hysteresis + crack impulse
    y_elastic = MUL_Q15(MUL_Q15(effective_stiffness, 24576), x)
    y_hysteresis = MUL_Q15(params.hysteresis_q15, state.z)
    y_base = SAT16(y_elastic + y_hysteresis)

    return SAT16(y_base + crack_impulse)
```

### 2.5 Sound Character

- **Fresh material**: Stiff, minimal cracking
- **Under stress**: Yielding with occasional micro-pops
- **High damage**: Softened response, frequent crackle
- **Failure region**: Dense transients, material "giving way"

### 2.6 Complexity

~35 ticks per sample (Bouc-Wen base + damage tracking + event logic)

---

## 3. Damage Reset Protocol

### 3.1 Reset Triggers

Damage state MAY be reset on:
- Note-off (fresh material per note)
- Program change
- Explicit reset command (EVT_MATERIAL with reset flag)

### 3.2 Persistence Options

**Per-note damage (default):**
- Damage resets on note-off
- Each note starts "fresh"

**Persistent damage (optional):**
- Damage persists across notes
- Material "wears out" over performance
- Reset only on explicit command

### 3.3 YAML Configuration

```yaml
compile:
  material:
    program_id: 7  # Brittle Fracture
    damage_mode: "per_note"  # or "persistent"
    initial_damage_q15: 0
```

---

## 4. Musical Applications

### 4.1 Program 6: Steel Yield

| Use Case | Settings | Character |
|----------|----------|-----------|
| "Metal pad" | stiffness=high, yield=high | Stiff, resisting |
| "Bending bass" | stiffness=mid, yield=low | Yielding warmth |
| "Stressed drone" | stiffness=low, hysteresis=high | Creaking movement |

### 4.2 Program 7: Brittle Fracture

| Use Case | Settings | Character |
|----------|----------|-----------|
| "Cracking ice" | brittleness=high, crack=high | Dense pops |
| "Metal fatigue" | brittleness=mid, crack=low | Occasional snaps |
| "Structural fail" | brittleness=high, damage_mode=persistent | Progressive breakdown |

---

## 5. Safety Contract

### 5.1 Bouc-Wen Stability

The n=1 Bouc-Wen is inherently bounded:
- |z| cannot grow unbounded (beta/gamma terms provide damping)
- Output is sum of bounded terms
- SAT16 enforces hard limits

### 5.2 Damage Bounds

- `damage_q15` clamped to [0, 32767]
- At max damage, stiffness drops to 50% (still bounded)
- Crack impulses scaled by `crack_q15` parameter

### 5.3 Event Rate Limiting

- `cooldown` prevents burst of cracks (min 8 samples between)
- Block-rate event budget still applies (WSE-STD-030)

---

## 6. FPGA Resource Estimate

| Program | LUTs | DSP | State Bytes |
|---------|------|-----|-------------|
| Steel Yield (6) | ~400 | 6 MUL | 4 (z, x_prev) |
| Brittle Fracture (7) | ~550 | 8 MUL | 6 (z, x_prev, damage, cooldown) |

Both fit comfortably in material budget (~1000 LUTs total for material stage).

---

## 7. Example Preset Configuration

```yaml
# Steel Yield bass
compile:
  material:
    program_id: 6
    stiffness_q15: 28672    # 0.875 - stiff
    yield_q15: 12288        # 0.375 - yields under pressure
    hysteresis_q15: 10922   # 0.33 - moderate loop width

# Brittle percussion
compile:
  material:
    program_id: 7
    stiffness_q15: 24576    # 0.75
    yield_q15: 16384        # 0.5
    brittleness_q15: 16384  # 0.5 - moderate damage rate
    crack_q15: 20480        # 0.625 - audible cracks
    damage_mode: "per_note"
```

---

## 8. Conformance

An implementation of programs 6-7 conforms iff:

1. Implements Bouc-Wen (n=1) with bit-exact Q1.15 arithmetic
2. Maintains per-channel state (L/R independent)
3. Implements damage accumulation for program 7
4. Rate-limits crack events (cooldown >= 8 samples)
5. Supports damage_mode selection (per_note/persistent)
6. Satisfies safety contract (bounded output)

---

**End of WSE-STD-MATERIAL v0 Extension: Structural Failure**
