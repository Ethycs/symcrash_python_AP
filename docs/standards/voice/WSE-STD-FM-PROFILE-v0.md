# WSE-STD-FM-PROFILE v0 — FM/PM Synthesis Profile

**Standard ID:** WSE-STD-FM-PROFILE v0
**Title:** CMMM Profile for FM/PM (Frequency/Phase Modulation) Synthesis
**Status:** Normative (v0 draft)
**Applies to:** Voice Engine, FM6 Engine, Preset Compiler
**Depends on:**
* WSE-STD-020 v1 (VoiceFrame ABI)
* WSE-STD-VM-HYBRID v0 (VM execution semantics)
* WSE-STD-ENVELOPE-SUITE v0 (Envelope generators)
* WSE-STD-ENVELOPE-MATRIX v0 (Matrix envelope routing)
* WSE-STD-001 (Fixed-Point arithmetic)

---

## 0. Scope

This document defines the **FM_PM** CMMM profile for DX7-style FM synthesis. This profile covers:

- 6-operator FM synthesis (FM6 engine)
- 32 classic DX7 algorithms
- Per-operator envelopes via Matrix Envelope System
- Feedback on designated operators
- Carrier/modulator relationships

This standard does **not** define:
* Linear FM (through-zero) — see future extensions
* More than 6 operators — see future FM8/FM12 profiles
* Real-time algorithm switching — operators are static per preset

---

## 1. Profile Header

| Property | Value |
|----------|-------|
| Profile ID | `FM_PM` |
| VM Lanes Used | CARRIER_RENDER (FM6), INJECT, TAIL, IDWPT_RENDER, MIX_GAIN |
| Oscillator Mode | MODE_FM6 (3) |
| Budget Regimes | MAIN, RESIDUAL |
| Determinism | Phase accumulators are continuous; feedback is deterministic |

---

## 2. FM6 Architecture

### 2.1 Operator Structure

Each of the 6 operators contains:

```python
@dataclass
class FM6Operator:
    ratio_q16: int           # Frequency ratio (65536 = 1.0)
    level_q15: int           # Output level (0-32767)
    env_value_q15: int       # Envelope modulation (from matrix)
    velocity_sens_q15: int   # Velocity sensitivity
    phase_q32: int           # Phase accumulator
```

### 2.2 Algorithm Matrix

Algorithms define the modulation routing:
- **Matrix M[i,j]**: True if operator i modulates operator j
- **Carrier mask**: Which operators contribute to audio output
- **Feedback**: Self-modulation on designated operators

Example (Algorithm 1 - DX7):
```
OP6 → OP5 → OP4 → OP3 → OP2 → OP1 [carrier]
                              ↑
                              └── feedback
```

### 2.3 All 32 Algorithms

| Alg | Carriers | Modulators | Feedback | Pattern |
|-----|----------|------------|----------|---------|
| 1 | 1 | 2,3,4,5,6 | 6 | Serial stack |
| 2 | 1 | 2,3,4,5,6 | 2 | Serial with branch |
| 3 | 1,4 | 2,3,5,6 | 6 | Parallel carriers |
| ... | ... | ... | ... | ... |
| 19 | 1,2,3,4 | 5,6 | 6 | Jump Brass (4 carriers) |
| 32 | 1,2,3,4,5,6 | - | 6 | All carriers (additive) |

See [FM6_PATHWAY_GUIDE.md](../../FM6_PATHWAY_GUIDE.md) for complete algorithm documentation.

---

## 3. InstrumentIR Fields

### 3.1 FM6 Node Specification

```yaml
instrument_ir_v0:
  meta:
    instrument_id: "fm-brass-v0"
    name: "FM Brass"

  layers:
    L0_graph:
      nodes:
        - id: "fm6"
          type: "FM6"
          params:
            algorithm: 19           # DX7 algorithm number (1-32)
            feedback: 4             # Feedback amount (0-7)

            operators:
              - ratio: 1.0          # Op1: fundamental
                level: 1.0
              - ratio: 1.003        # Op2: slight detune
                level: 1.0
              - ratio: 0.997        # Op3: slight detune
                level: 1.0
              - ratio: 1.006        # Op4: slight detune
                level: 0.875
              - ratio: 1.0          # Op5: modulator
                level: 0.333
              - ratio: 1.0          # Op6: modulator
                level: 0.25

  envelopes:
    # Carrier envelope
    - id: "carrier_env"
      adsr: { a_ms: 8, d_ms: 150, s: 0.7, r_ms: 100 }
      targets: [FM_OP1, FM_OP2, FM_OP3, FM_OP4]

    # Modulator envelope (faster decay for brightness transient)
    - id: "mod_env"
      adsr: { a_ms: 8, d_ms: 50, s: 0.3, r_ms: 100 }
      targets: [FM_OP5, FM_OP6]
```

### 3.2 Operator Parameter Ranges

| Field | Type | Range | Q Format | Description |
|-------|------|-------|----------|-------------|
| `ratio` | float | 0.5-16.0 | Q16 (65536=1.0) | Frequency ratio |
| `level` | float | 0.0-1.0 | Q15 (32767=1.0) | Output level |
| `detune_cents` | int | -100 to +100 | I16 | Fine tuning |
| `velocity_sens` | float | 0.0-1.0 | Q15 | Velocity sensitivity |

### 3.3 Algorithm Parameters

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `algorithm` | int | 1-32 | DX7 algorithm number |
| `feedback` | int | 0-7 | Feedback amount (0=off, 7=max) |
| `output_level` | float | 0.0-1.0 | Master output scaling |

---

## 4. Control Manifold

### 4.1 Macro Controls

| Macro | Range | Default | Mapping |
|-------|-------|---------|---------|
| BRIGHTNESS | 0-32767 | 32767 | Modulator levels (Op5, Op6) |
| DYNAMICS | 0-32767 | 32767 | Overall output gain |
| ATTACK | 0-65535 | 10 | All envelope attack times |
| RELEASE | 0-65535 | 200 | All envelope release times |

### 4.2 BRIGHTNESS Mapping

The BRIGHTNESS macro scales modulator operator levels:

```python
# Modulators are operators that don't contribute to carrier mask
for op_idx in range(6):
    if not (carrier_mask & (1 << op_idx)):
        # This is a modulator
        effective_level = (base_level * brightness_q15) >> 15
```

---

## 5. State Machine

### 5.1 Regime Graph

```
IDLE ──note_on──> ATTACK ──complete──> SUSTAIN ──note_off──> RELEASE ──complete──> IDLE
```

### 5.2 Note Lifecycle

**Note On:**
1. All matrix envelopes trigger
2. Phase accumulators reset to 0 (optional: configurable)
3. Velocity scales operator levels per `velocity_sens`

**Note Off:**
1. All matrix envelopes enter release
2. Voice remains active until all envelopes reach zero

**Retrigger:**
1. Envelopes restart from current level (no click)
2. Phase accumulators continue (pitch stable)

---

## 6. Compiler Rules

### 6.1 IR → VoiceFrame Compilation

```python
def compile_fm6(ir: InstrumentIR) -> Dict[VoiceParamID, int]:
    fm6_node = ir.get_node_by_type("FM6")

    defaults = {
        VoiceParamID.OSC_MODE: 3,  # MODE_FM6
        VoiceParamID.FM6_ALGORITHM: fm6_node.params.algorithm,
        VoiceParamID.FM6_FEEDBACK: fm6_node.params.feedback,
    }

    # Operator ratios (Q16)
    for i, op in enumerate(fm6_node.params.operators):
        ratio_q16 = int(op.ratio * 65536)
        defaults[VoiceParamID[f'FM6_OP{i}_RATIO']] = ratio_q16

        level_q15 = int(op.level * 32767)
        defaults[VoiceParamID[f'FM6_OP{i}_LEVEL']] = level_q15

    return defaults
```

### 6.2 Matrix Envelope Compilation

```python
def compile_fm6_envelopes(ir: InstrumentIR) -> Dict[EnvelopeTarget, EnvelopeScheduler]:
    matrix = {}

    for binding in ir.envelopes:
        scheduler = ADSR(
            attack_ms=binding.adsr.a_ms,
            decay_ms=binding.adsr.d_ms,
            sustain=binding.adsr.s,
            release_ms=binding.adsr.r_ms,
        )

        for target_name in binding.targets:
            target = EnvelopeTarget[target_name]
            matrix[target] = scheduler

    return matrix
```

---

## 7. FM6 Engine Implementation

### 7.1 Per-Sample Processing

```python
def fm6_sample(state: FM6State, phase_inc_base: int) -> int:
    output = 0
    op_outputs = [0] * 6

    # Process in topological order (modulators before carriers)
    for op_idx in state.eval_order:
        op = state.operators[op_idx]

        # Compute phase increment
        phase_inc = (phase_inc_base * op.ratio_q16) >> 16

        # Add modulation from other operators
        for mod_idx in range(6):
            if state.mod_matrix[mod_idx][op_idx]:
                phase_inc += (op_outputs[mod_idx] * op.level_q15) >> 15

        # Add feedback (self-modulation)
        if state.feedback_mask & (1 << op_idx):
            phase_inc += (state.feedback_buffer[op_idx] * state.feedback_amount) >> 15

        # Advance phase and compute sine
        op.phase_q32 = (op.phase_q32 + phase_inc) & 0xFFFFFFFF
        op_output = sine_q15(op.phase_q32 >> 17)  # Top 15 bits → table index

        # Apply envelope
        op_output = (op_output * op.env_value_q15) >> 15
        op_outputs[op_idx] = op_output

        # Update feedback buffer
        state.feedback_buffer[op_idx] = op_output

        # Accumulate carriers
        if state.carrier_mask & (1 << op_idx):
            output += op_output

    return clamp_q15(output)
```

### 7.2 Fixed-Point Considerations

| Value | Format | Notes |
|-------|--------|-------|
| Phase accumulator | Q32 | Wraps at 2^32 |
| Frequency ratio | Q16 | 65536 = 1.0 |
| Sine output | Q15 | -32768 to +32767 |
| Operator level | Q15 | 0 to 32767 |
| Feedback | Q15 | Scaled before addition |

---

## 8. Degradation Semantics

### 8.1 Degradation Priority

1. **Reduce feedback** → Simpler tone
2. **Reduce modulator levels** → Less brightness
3. **Drop modulators** → Fewer operators computed
4. **Drop carriers** → Reduced polyphony

### 8.2 Preserved Under Degradation

- **Pitch accuracy**: Phase accumulators maintained
- **Carrier identity**: Core carriers never dropped
- **Envelope shape**: Attack/release contour preserved

---

## 9. Tests

### 9.1 Golden Trace Requirements

- [ ] Algorithm 1 (serial stack) - classic FM tone
- [ ] Algorithm 19 (jump brass) - 4 carrier brass
- [ ] Algorithm 32 (all carriers) - additive synthesis
- [ ] Feedback sweep (0 to 7)
- [ ] Velocity response curve
- [ ] Carrier/modulator envelope separation

### 9.2 Stress Trace Requirements

- [ ] High polyphony (8+ voices)
- [ ] Rapid note retriggering
- [ ] Extreme ratios (0.5x to 16x)
- [ ] Maximum feedback

### 9.3 Existing Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_fm6_yaml_pipeline.py` | IR → VoiceFrame flow |
| `test_fm6_engine.py` | Core FM6 computation |
| `test_fm_feedback.py` | Feedback stability |
| `test_fm6_pipeline.py` | End-to-end synthesis |

---

## 10. Common Preset Patterns

### 10.1 Brass (Algorithm 19)

```yaml
algorithm: 19
feedback: 4
operators:
  - { ratio: 1.0, level: 1.0 }      # Carrier 1
  - { ratio: 1.003, level: 1.0 }    # Carrier 2 (detune)
  - { ratio: 0.997, level: 1.0 }    # Carrier 3 (detune)
  - { ratio: 1.006, level: 0.875 }  # Carrier 4
  - { ratio: 1.0, level: 0.333 }    # Modulator
  - { ratio: 1.0, level: 0.25 }     # Modulator (feedback)

envelopes:
  - id: "carrier"
    adsr: { a_ms: 8, d_ms: 150, s: 0.7, r_ms: 100 }
    targets: [FM_OP1, FM_OP2, FM_OP3, FM_OP4]
  - id: "mod"
    adsr: { a_ms: 8, d_ms: 50, s: 0.3, r_ms: 100 }
    targets: [FM_OP5, FM_OP6]
```

### 10.2 Electric Piano (Algorithm 5)

```yaml
algorithm: 5
feedback: 0
operators:
  - { ratio: 1.0, level: 1.0 }
  - { ratio: 14.0, level: 0.2 }     # High harmonic
  - { ratio: 1.0, level: 0.8 }
  - { ratio: 1.0, level: 0.3 }
  # ...

envelopes:
  - id: "carrier"
    adsr: { a_ms: 0, d_ms: 800, s: 0.0, r_ms: 200 }
    targets: [FM_OP1, FM_OP3]
  - id: "mod"
    adsr: { a_ms: 0, d_ms: 200, s: 0.0, r_ms: 50 }
    targets: [FM_OP2, FM_OP4, FM_OP5, FM_OP6]
```

### 10.3 Bell (Algorithm 4)

```yaml
algorithm: 4
feedback: 0
operators:
  - { ratio: 1.0, level: 1.0 }
  - { ratio: 3.5, level: 0.6 }      # Inharmonic
  - { ratio: 4.5, level: 0.4 }      # Inharmonic
  - { ratio: 14.0, level: 0.2 }     # Shimmer
  # ...
```

---

## 11. Conformance Checklist

A conforming FM_PM profile implementation MUST:

- [x] Implement FM6 engine with 6 operators
- [x] Support all 32 DX7 algorithms
- [x] Implement feedback on designated operators
- [x] Route matrix envelopes to FM_OP1-6 targets
- [x] Use Q16 frequency ratios (65536 = 1.0)
- [x] Use Q15 levels and envelope values
- [x] Support velocity sensitivity per operator
- [x] Maintain phase continuity on retrigger
- [ ] Produce hash-frozen golden traces
- [ ] Pass stress tests under polyphony

A conforming implementation MAY:

- Support algorithm morphing
- Implement operator waveform selection (beyond sine)
- Add per-operator filter
- Support more than 6 operators

---

## 12. References

- [FM6_PATHWAY_GUIDE.md](../../FM6_PATHWAY_GUIDE.md) - Complete FM6 documentation
- [fm6_engine.py](../../../src/symcrash_python_ap/fm6/fm6_engine.py) - FM6 engine implementation
- [envelope_targets.py](../../../src/symcrash_python_ap/envelope_targets.py) - FM_OP targets
- [WSE-STD-ENVELOPE-MATRIX-v0.md](../control/WSE-STD-ENVELOPE-MATRIX-v0.md) - Matrix envelope system
- [WSE-STD-ENVELOPE-SUITE-v0.md](../control/WSE-STD-ENVELOPE-SUITE-v0.md) - ADSR implementation

---

**Document Version:** 0.1
**Last Updated:** 2026-01-20
**Author:** CCS Development Team
**Status:** NORMATIVE DRAFT
