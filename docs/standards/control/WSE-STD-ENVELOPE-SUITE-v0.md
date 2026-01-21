# WSE-STD-ENVELOPE-SUITE v0 — Unified Envelope System

**Standard ID:** WSE-STD-ENVELOPE-SUITE v0
**Title:** Unified Envelope Generators, Followers, and Modulators
**Status:** Normative (v0 draft)
**Applies to:** Voice synthesis, modulation, dynamics processing
**Depends on:** WSE-STD-001 (Fixed-Point), WSE-STD-ENVELOPE-MATRIX v0

---

## 0. Scope

This standard defines a unified suite of envelope generators and signal processors:

| Component | Type | Purpose |
|-----------|------|---------|
| **ADSR** | Generator | Classic attack/decay/sustain/release (note-triggered) |
| **AR** | Generator | Attack-release (percussive, no sustain) |
| **Decay** | Generator | One-shot exponential decay |
| **EnvelopeFollower** | Processor | Tracks input amplitude with asymmetric attack/release |
| **Slew** | Processor | Smoothing filter for control signals |
| **LFO** | Modulator | Low-frequency oscillator with multiple waveforms |

All components operate at **block rate (750 Hz)** and use **Q15 fixed-point** arithmetic.

---

## 1. Design Principles

### 1.1 Block-Rate Operation

All envelope components tick at block rate (48kHz / 64 = 750 Hz):
- Sufficient resolution for musical envelopes
- Minimal CPU overhead
- Deterministic behavior

### 1.2 Integer-Only Arithmetic

All internal state is integer for determinism:
- No floating-point in tick() functions
- Q15 format for levels and coefficients
- Q16 format for phase accumulators

### 1.3 Unified Interface

All generators share common patterns:
- `trigger()` / `release()` for note events
- `tick()` returns Q15 output
- `is_active` property for voice stealing
- `reset()` for state clearing

---

## 2. ADSR Envelope

### 2.1 Specification

```python
@dataclass
class ADSR:
    attack_ms: float = 10.0      # Time to rise from 0 to peak
    decay_ms: float = 100.0      # Time to fall from peak to sustain
    sustain: float = 0.5         # Sustain level (0.0 to 1.0)
    release_ms: float = 100.0    # Time to fall from current to 0
```

### 2.2 State Machine

```
IDLE ──trigger()──> ATTACK ──peak reached──> DECAY ──sustain reached──> SUSTAIN
  ^                                                                        |
  |                                                                        |
  +──────────────────────── release() ─────────────────────────────────────+
                               |
                               v
                            RELEASE ──zero reached──> IDLE
```

### 2.3 Segment Shapes

| Phase | Shape | Formula |
|-------|-------|---------|
| ATTACK | Linear | `level = t / attack_blocks * Q15_ONE` |
| DECAY | Linear | `level = Q15_ONE - (Q15_ONE - sustain) * t / decay_blocks` |
| SUSTAIN | Constant | `level = sustain_q15` |
| RELEASE | Linear | `level = release_level * (1 - t / release_blocks)` |

### 2.4 Usage

```python
env = ADSR(attack_ms=10, decay_ms=100, sustain=0.7, release_ms=200)
env.trigger()  # Note on

for block in range(total_blocks):
    gain_q15 = env.tick()
    if block == note_off_block:
        env.release()  # Note off
```

---

## 3. AR Envelope (Attack-Release)

### 3.1 Specification

```python
@dataclass
class AR:
    attack_ms: float = 10.0      # Time to rise from 0 to peak
    release_ms: float = 200.0    # Time to fall from peak to 0
```

### 3.2 Behavior

- No sustain phase - immediately enters release after attack
- Useful for percussive sounds, one-shot triggers
- Simpler than ADSR, lower overhead

### 3.3 State Machine

```
IDLE ──trigger()──> ATTACK ──peak reached──> RELEASE ──zero reached──> IDLE
```

---

## 4. Decay Envelope (Exponential)

### 4.1 Specification

```python
@dataclass
class Decay:
    decay_ms: float = 100.0      # Time constant (~37% level)
    floor: float = 0.0           # Minimum output level (0.0 to 1.0)
```

### 4.2 Behavior

- Starts at Q15_ONE, decays exponentially toward floor
- True exponential curve: `level = level * coeff` each block
- Useful for hi-hats, FM modulator envelopes

### 4.3 Formula

```python
# Each tick:
level_q15 = (level_q15 * decay_coeff) >> 15

# Where decay_coeff = Q15_ONE - (Q15_ONE / decay_blocks)
```

---

## 5. Envelope Follower

### 5.1 Specification

```python
@dataclass
class EnvelopeFollower:
    attack_ms: float = 5.0       # Rise time when input increases
    release_ms: float = 100.0    # Fall time when input decreases
    mode: str = "PEAK"           # "PEAK" or "RMS"
```

### 5.2 Algorithm

```python
def tick(self, input_q15: int) -> int:
    mag = abs(input_q15)

    if mag > level:
        # Attack: fast rise
        step = (mag - level) * attack_coeff >> 15
        level += max(1, step)
    else:
        # Release: slow decay
        step = (level - mag) * release_coeff >> 15
        level -= max(1, step)

    return level
```

### 5.3 Use Cases

| Application | Attack | Release | Mode |
|-------------|--------|---------|------|
| Sidechain compression | 1-5 ms | 50-200 ms | PEAK |
| Auto-wah | 5-20 ms | 100-500 ms | PEAK |
| Level meter | 1 ms | 300 ms | RMS |
| Dynamics | 5 ms | 100 ms | PEAK |

### 5.4 Block Processing

```python
# Process a block of samples
level = follower.tick_block(samples)

# Or process block-rate signal
level = follower.tick(block_peak_q15)
```

---

## 6. Slew Limiter

### 6.1 Specification

```python
@dataclass
class Slew:
    rise_ms: float = 10.0        # Time to rise from 0 to full scale
    fall_ms: float = 10.0        # Time to fall from full scale to 0
    mode: str = "LINEAR"         # "LINEAR" or "EXP"
```

### 6.2 Modes

**LINEAR Mode:**
- Constant slew rate
- Reaches target in exactly `rise_ms` or `fall_ms`
- No zipper noise, predictable timing

**EXP Mode:**
- Exponential smoothing (one-pole filter)
- Asymptotically approaches target
- Smoother, more natural feel

### 6.3 Usage

```python
slew = Slew(rise_ms=10, fall_ms=10, mode="LINEAR")

# Smooth a parameter change
for block in range(100):
    smoothed = slew.tick(target_value)

# Immediate jump (bypass slew)
slew.set_immediate(value)
```

---

## 7. LFO (Low Frequency Oscillator)

### 7.1 Specification

```python
@dataclass
class LFO:
    rate_hz: float = 1.0         # Oscillation frequency (0.01 to 50 Hz)
    shape: LFOShape = SINE       # Waveform shape
    unipolar: bool = False       # True: 0 to 32767, False: -32768 to +32767
    phase_offset: float = 0.0    # Initial phase (0.0 to 1.0)
```

### 7.2 Waveform Shapes

| Shape | ID | Description | Output Range |
|-------|-----|-------------|--------------|
| SINE | 0 | Smooth sine wave | [-32767, +32767] |
| TRIANGLE | 1 | Linear ramp up/down | [-32767, +32767] |
| SAW_UP | 2 | Ramp from -1 to +1 | [-32767, +32767] |
| SAW_DOWN | 3 | Ramp from +1 to -1 | [-32767, +32767] |
| SQUARE | 4 | Binary +1/-1 | {-32767, +32767} |
| SAMPLE_HOLD | 5 | Random, held per cycle | [-32767, +32767] |

### 7.3 Phase Accumulator

```python
# Phase is Q16 (0 to 65535 = 0 to 2pi)
phase_inc = (rate_hz * 65536) / BLOCKS_PER_SECOND

# Each tick:
phase = (phase + phase_inc) & 0xFFFF
```

### 7.4 Sync and Reset

```python
lfo.sync(0.0)      # Reset to phase 0
lfo.sync(0.25)     # Reset to 90 degrees
lfo.set_rate(5.0)  # Change rate dynamically
```

---

## 8. Integration with Matrix Envelope System

### 8.1 Envelope Target Routing

The envelope suite components can route to any `EnvelopeTarget`:

```yaml
envelopes:
  - id: "amp_env"
    type: "ADSR"
    adsr: { a_ms: 10, d_ms: 100, s: 0.7, r_ms: 200 }
    targets: [GAIN]

  - id: "mod_env"
    type: "Decay"
    decay_ms: 50
    targets: [FM_OP5, FM_OP6]

  - id: "vibrato"
    type: "LFO"
    rate_hz: 5.0
    shape: "SINE"
    targets: [PITCH]

  - id: "sidechain"
    type: "EnvelopeFollower"
    attack_ms: 5
    release_ms: 100
    source: "EXTERNAL"
    targets: [GAIN]
```

### 8.2 Modulation Depth

LFO and follower outputs can be scaled before routing:

```python
# In voice frame emitter:
lfo_output = lfo.tick()
scaled = (lfo_output * depth_q15) >> 15
env_targets_q15[EnvelopeTarget.PITCH] = scaled
```

---

## 9. Common Patterns

### 9.1 FM Brass with Brightness Decay

```python
# Carrier envelope: standard ADSR
carrier_env = ADSR(attack_ms=8, decay_ms=150, sustain=0.7, release_ms=100)

# Modulator envelope: fast decay for brightness transient
mod_env = Decay(decay_ms=50, floor=0.3)

# On note-on:
carrier_env.trigger()
mod_env.trigger()

# Each block:
carrier_level = carrier_env.tick()
mod_level = mod_env.tick()
```

### 9.2 Auto-Wah Effect

```python
# Follow input amplitude
follower = EnvelopeFollower(attack_ms=10, release_ms=200)

# Each block:
input_level = get_block_peak()
filter_cutoff = follower.tick(input_level)
```

### 9.3 Vibrato with Delayed Onset

```python
lfo = LFO(rate_hz=5.0, shape=LFOShape.SINE)
delay_env = ADSR(attack_ms=500, decay_ms=0, sustain=1.0, release_ms=100)

# On note-on:
delay_env.trigger()

# Each block:
vibrato = lfo.tick()
depth = delay_env.tick()
pitch_mod = (vibrato * depth) >> 15
```

### 9.4 Ducking / Sidechain Compression

```python
follower = EnvelopeFollower(attack_ms=2, release_ms=100)
slew = Slew(rise_ms=5, fall_ms=50, mode="EXP")

# Each block:
sidechain_level = follower.tick(sidechain_input)
duck_amount = Q15_ONE - sidechain_level
smoothed = slew.tick(duck_amount)
output_gain = smoothed
```

---

## 10. Performance Characteristics

### 10.1 CPU Cost per Tick

| Component | Approx. Operations |
|-----------|-------------------|
| ADSR | 5-10 |
| AR | 4-8 |
| Decay | 3-5 |
| EnvelopeFollower | 8-12 |
| Slew | 4-8 |
| LFO (SINE) | 10-15 (table lookup) |
| LFO (others) | 5-10 |

### 10.2 Memory per Instance

| Component | Bytes |
|-----------|-------|
| ADSR | ~64 |
| AR | ~40 |
| Decay | ~32 |
| EnvelopeFollower | ~32 |
| Slew | ~40 |
| LFO | ~48 |

---

## 11. Conformance

A conforming implementation MUST:

1. Operate at block rate (750 Hz at 48kHz/64)
2. Use Q15 output format (0 to 32767 for unipolar, -32768 to +32767 for bipolar)
3. Support `trigger()`, `tick()`, and `reset()` methods for generators
4. Provide `is_active` property for voice management
5. Use integer-only arithmetic in tick functions

A conforming implementation MAY:

- Add curve shaping parameters (exponential attack/decay)
- Support sample-rate operation for high-precision applications
- Provide additional LFO waveforms
- Implement envelope looping

---

## 12. References

- [envelope_suite.py](../../../src/symcrash_python_ap/envelope_suite.py) - Implementation
- [WSE-STD-ENVELOPE-MATRIX-v0.md](WSE-STD-ENVELOPE-MATRIX-v0.md) - Target routing
- [test_envelope_suite.py](../../../tests/test_envelope_suite.py) - Test coverage

---

**Document Version:** 0.1
**Last Updated:** 2026-01-20
**Author:** CCS Development Team
**Status:** DRAFT
