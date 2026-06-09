# Vocabulary Reference
#
# Sound-engineering terms used throughout presets AND internal code.
# Legacy jargon (CONSTRAINT, CALM, T1, tone, energy, brightness, density, …)
# is accepted on load for backward compatibility but maps to the canonical
# 7-axis control vocabulary listed below.

## Control Axes (the 7-axis vector)

A preset shapes a sound with **seven continuous control axes**. Each axis maps
1:1 onto one internal operator family and recruits that family's operators
*continuously* — any nonzero weight engages them, and the weight sets their
strength. There are no thresholds. Weight 1.0 = full, 0.0 = off.

| Axis   | Sonic meaning                            | Operators recruited                 | Family        |
|--------|------------------------------------------|-------------------------------------|---------------|
| color  | Spectral tilt / timbral brightness       | Tilt                                | F_COLOR       |
| body   | Within-band weight and thickness         | IntraMix                            | F_LOCAL       |
| spread | Cross-band energy redistribution / width | DiffuseDown, DiffuseUp              | F_TRANSPORT   |
| warp   | Structural reshaping / permutation       | BandShear, SwapBands, RotateNodes  | F_TOPOLOGY    |
| glue   | Coherence, stability, settling           | GluePair, SheafProject             | F_GLUE        |
| snap   | Transient / energy injection             | Impulse, GlobalPulse               | F_EVENTS      |
| chaos  | Controlled extreme transformations       | FoldPressure                       | F_CATASTROPHE |

An 8th family, **TAIL** (NormalizeScale + FineClamp), is always-on safety and is
not author-facing — it is applied to every behavior state automatically.

### Example

```yaml
  emphasis: {color: 0.6, body: 0.8, glue: 1.0}
```

`emphasis:` is the behavior-state key. It accepts the 7 canonical axes above and
the legacy emphasis names below; you do **not** list operators — the axes derive
them. An explicit `processing:` list is an optional escape hatch (L5/Lab) that
overrides derivation.

### Legacy emphasis names (accepted on load → canonical axis)

| Legacy emphasis | → Axis  |
|-----------------|---------|
| tone            | color   |
| clarity         | glue    |
| stability       | glue    |
| settling        | glue    |
| energy          | snap    |
| body / spread / warp / chaos | (unchanged) |

Older presets that write `emphasis: {tone: 0.7, stability: 1.0}` still load —
those names fold into the canonical axes (here `color` and `glue`).

## Tree Names (wavelet packet families)

Trees control wavelet packet decomposition — how the sound's frequency content
is analyzed and resynthesized. This axis is orthogonal to the control vector.

| Canonical Name | Legacy Alias | Band Emphasis     | Use Case               |
|----------------|--------------|-------------------|------------------------|
| balanced       | T0           | even              | Default / neutral      |
| harmonic       | T1           | low (fundamental) | Pitched / tonal sounds |
| texture        | T2           | high (detail)     | Noise / texture        |
| impact         | T3           | mid (transient)   | Attacks / percussion   |
| drone          | T4           | low (sustained)   | Sustained tones        |

### Example

```yaml
  trees: {A: harmonic, B: texture, blend: 0.3}
```

## Character knobs (legacy — aliases of the control axes)

The cockpit "character" knobs are a UI convenience that alias onto the same 7
axes. Authoring a separate `character:` block is **deprecated** — prefer one
`emphasis:` vector. The aliases:

| Character knob | → Axis  | Notes                          |
|----------------|---------|--------------------------------|
| brightness     | color   |                                |
| density        | body    |                                |
| energy         | snap    |                                |
| risk           | chaos   |                                |
| dream          | warp    | deprecated, no-op              |
| slip           | spread  | deprecated, no-op              |

Runtime SystemMacro IDs that are *not* control axes — `energy_state`, `motion`,
`focus`, `age`, `space`, `material`, `unity`, `vibrato`, `legato` — remain as
the cockpit / gesture control ABI (see WSE-STD-MACRO, WSE-STD-GESTURE). They are
unchanged by the 7-axis collapse.

## Partials (harmonic amplitudes)

Osc nodes with `waveform: partials` specify harmonic amplitudes.
Use normalized floats (0.0–1.0) instead of Q15 integers (0–32767).

### Example

```yaml
  partials: [0.943, 1.0, 0.211, 0.731, 0.112, 0.187, 0.036, 0.037]
```

## Complete YAML ↔ Internal Mapping

| YAML Surface                     | Internal Representation                  | Category  |
|----------------------------------|------------------------------------------|-----------|
| `audio:`                         | `io:`                                    | Section   |
| `id:`                            | `instrument_id:`                         | Meta      |
| `seed:`                          | `seed_u32:`                              | Meta      |
| `graph:`                         | `layers.L0_graph`                        | Section   |
| `output:`                        | `L0Edge`                                 | Graph     |
| `attack:`                        | `a_ms:`                                  | Envelope  |
| `decay:`                         | `d_ms:`                                  | Envelope  |
| `sustain:`                       | `s:`                                     | Envelope  |
| `release:`                       | `r_ms:`                                  | Envelope  |
| `controls:`                      | `macros:`                                | Section   |
| `character:` (deprecated alias)  | `control:`                               | Section   |
| `emphasis:` (7-axis vector)      | `families:`                              | Behavior  |
| `processing:` (optional)         | `ops_enable:` (else derived from axes)   | Behavior  |
| `blend:`                         | `beta:`                                  | Trees     |
| `resolution:`                    | `node_mask:`                             | CMMM      |
| `safety:`                        | `tail_mode:`                             | CMMM      |
| `ramp:`                          | `mask_ramp_blocks:`                      | CMMM      |
| floats in partials               | Q15 integers in partials                 | Graph/Osc |
