# Vocabulary Reference
#
# Sound-engineering terms used throughout presets AND internal code.
# Legacy jargon (CONSTRAINT, CALM, T1, energy_intent, etc.) is accepted
# on load for backward compatibility but translated to the canonical
# sound-engineering names listed below.

## Emphasis Families (behavior state weights)

Emphasis families control which operator groups get priority
during each behavior state. Weight 1.0 = full emphasis, 0.0 = off.

| Canonical Name | Legacy Alias  | Sonic Meaning                                |
|----------------|---------------|----------------------------------------------|
| stability      | CONSTRAINT    | Keeps output bounded and structurally safe   |
| settling       | CALM          | Recovery / damping, reduces activity         |
| tone           | COLOR         | Spectral shaping, timbral coloring (Tilt)    |
| body           | LOCAL         | Within-band mixing, weight and thickness     |
| spread         | TRANSPORT     | Cross-band energy redistribution             |
| clarity        | COHERENCE     | Band coherence, prevents artifacts (Glue)    |
| energy         | EXCITE        | Energy injection, attack transients          |
| warp           | TOPOLOGY      | Structural reshaping, permutations           |
| chaos          | CATASTROPHE   | Controlled extreme transformations           |

### Example

```yaml
  emphasis: {stability: 1.0, settling: 0.5, tone: 0.7}
```

## Tree Names (wavelet packet families)

Trees control wavelet packet decomposition — how the sound's
frequency content is analyzed and resynthesized.

| Canonical Name | Legacy Alias | Band Emphasis     | Use Case              |
|----------------|--------------|-------------------|-----------------------|
| balanced       | T0           | even              | Default / neutral     |
| harmonic       | T1           | low (fundamental) | Pitched / tonal sounds|
| texture        | T2           | high (detail)     | Noise / texture       |
| impact         | T3           | mid (transient)   | Attacks / percussion  |
| drone          | T4           | low (sustained)   | Sustained tones       |

### Example

```yaml
  trees: {A: harmonic, B: texture, blend: 0.3}
```

## Cockpit Controls (SystemMacro)

Semantic control axes that shape sound character. Values are
floats (0.0–1.0) in presets, Q15 unsigned (0–32767) at runtime.

| ID | Name       | Meaning                                          |
|----|------------|--------------------------------------------------|
| 0  | energy_state | Envelope-owned (not settable by controllers)   |
| 1  | energy     | Playing intensity / attack character              |
| 2  | motion     | Movement / modulation amount (gesture tracking)   |
| 3  | focus      | Attention / emphasis (spectral concentration)     |
| 4  | brightness | Timbral brightness (harmonic content)             |
| 5  | density    | Body / thickness (low-mid presence)               |
| 6  | risk       | Chaos / instability (stochastic behavior)         |
| 7  | age        | Integrator: accumulated note age                  |
| 8  | space      | Spatial / reverb character                        |
| 9  | material   | Surface texture (wood=low, metal=mid, glass=high) |
| 10 | unity      | Coherence / blend (how much voices blend)         |
| 11 | vibrato    | Performance macro: LFO modulation depth           |
| 12 | legato     | Performance macro: smooth pitch transitions       |
| 13 | dream      | Entropy freedom, fine-scale scramble permission   |
| 14 | slip       | Cross-scale transport / diffusion amount          |

### Example (character section in preset YAML)

```yaml
  character:
    brightness: 0.7
    density: 0.6
    energy: 0.5
    risk: 0.01
    dream: 0.0
    slip: 0.0
```

## Partials (harmonic amplitudes)

Osc nodes with `waveform: partials` specify harmonic amplitudes.
Use normalized floats (0.0–1.0) instead of Q15 integers (0–32767).

### Example

```yaml
  partials: [0.943, 1.0, 0.211, 0.731, 0.112, 0.187, 0.036, 0.037]
```

## Complete YAML ↔ Internal Mapping

| YAML Surface       | Internal Representation       | Category      |
|---------------------|-------------------------------|---------------|
| `audio:`            | `io:`                         | Section       |
| `id:`               | `instrument_id:`              | Meta          |
| `seed:`             | `seed_u32:`                   | Meta          |
| `graph:`            | `layers.L0_graph`             | Section       |
| `output:`           | `L0Edge`                      | Graph         |
| `attack:`           | `a_ms:`                       | Envelope      |
| `decay:`            | `d_ms:`                       | Envelope      |
| `sustain:`          | `s:`                          | Envelope      |
| `release:`          | `r_ms:`                       | Envelope      |
| `controls:`         | `macros:`                     | Section       |
| `character:`        | `control:`                    | Section       |
| `processing:`       | `ops_enable:`                 | Behavior      |
| `emphasis:`         | `families:`                   | Behavior      |
| `blend:`            | `beta:`                       | Trees         |
| `resolution:`       | `node_mask:`                  | CMMM          |
| `safety:`           | `tail_mode:`                  | CMMM          |
| `ramp:`             | `mask_ramp_blocks:`           | CMMM          |
| floats in partials  | Q15 integers in partials      | Graph/Osc     |
