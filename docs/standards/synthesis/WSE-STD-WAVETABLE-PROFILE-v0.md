# WSE-STD-WAVETABLE-PROFILE v0

**Status:** Draft
**Version:** 0.1
**Date:** 2026-01-20
**Cross-ref:** WSE-STD-210, WSE-STD-CMM-ROADMAP-v0, TOWER-CONFORMANCE-ROADMAP

---

## 1. Abstract

This document specifies the Wavetable Synthesis Profile for the SymCrash Audio Pipeline. Wavetable synthesis enables smooth timbral morphing through 2D interpolation across a grid of single-cycle waveforms (atlas).

---

## 2. Scope

This profile covers:

1. **Wavetable Atlas** - 2D grid structure for waveform storage
2. **Bilinear Interpolation** - Smooth transitions between waveforms
3. **Frame Scanning** - Automatic evolution through atlas frames
4. **Coefficient Generation** - Integration with CCS pipeline
5. **IR Specification** - `WavetableSpec` in InstrumentIR L1 layer

---

## 3. Conformance Levels

| Level | Requirements | Use Case |
|-------|--------------|----------|
| **W0** | Static wavetable (single frame, no morphing) | Basic wavetable oscillator |
| **W1** | W0 + frame scanning | Evolving timbres |
| **W2** | W1 + 2D position morphing | Full timbral control |
| **W3** | W2 + modulation routing | Expressive wavetable |

---

## 4. Wavetable Atlas Structure

### 4.1 Dimensional Layout

```
Atlas: [num_positions × num_frames × frame_size]
       ↑              ↑              ↑
       X-axis         Y-axis         samples per cycle
       (morph)        (evolution)    (typically 256)
```

### 4.2 Coordinate System

| Axis | Range | Description |
|------|-------|-------------|
| **Position** (X) | 0.0 - 1.0 | Timbral morph target |
| **Frame** (Y) | 0.0 - 1.0 | Evolution stage |
| **Phase** | 0.0 - 1.0 | Sample position within cycle |

### 4.3 Data Format

```python
@dataclass
class WavetableAtlas:
    data: np.ndarray          # Shape: [positions, frames, samples]
    num_positions: int        # X dimension (1-16 typical)
    num_frames: int           # Y dimension (16-256 typical)
    frame_size: int           # Samples per cycle (256 or 512)
    name: str                 # Human-readable identifier
    base_note: int = 60       # MIDI note for 1:1 playback
```

### 4.4 Standard Dimensions

| Preset Type | Positions | Frames | Frame Size |
|-------------|-----------|--------|------------|
| Basic | 1 | 4-16 | 256 |
| Standard | 1 | 32-64 | 256 |
| Morphing | 2-4 | 32 | 256 |
| High-Res | 1 | 256 | 512 |

---

## 5. Bilinear Interpolation

### 5.1 Algorithm

For position `p` (Q16), frame `f` (Q16), and phase `φ` (Q16):

```
1. Extract integer indices:
   p_int = p >> 16
   f_int = f >> 16

2. Extract fractional parts:
   p_frac = p & 0xFFFF
   f_frac = f & 0xFFFF

3. Get 4 corner samples at phase φ:
   s00 = atlas[p_int,     f_int,     φ]
   s01 = atlas[p_int,     f_int + 1, φ]
   s10 = atlas[p_int + 1, f_int,     φ]
   s11 = atlas[p_int + 1, f_int + 1, φ]

4. Interpolate along frame axis:
   s0 = lerp(s00, s01, f_frac)
   s1 = lerp(s10, s11, f_frac)

5. Interpolate along position axis:
   result = lerp(s0, s1, p_frac)
```

### 5.2 Linear Interpolation (Q16)

```python
def lerp_q16(a: int, b: int, frac: int) -> int:
    """Linear interpolation with Q16 fraction (0-65535)."""
    return a + (((b - a) * frac) >> 16)
```

### 5.3 Phase Interpolation

Sub-sample phase interpolation is also performed:

```python
phase_scaled = (phase_q16 * frame_size) >> 16
phase_int = phase_scaled % frame_size
phase_frac = ((phase_q16 * frame_size) & 0xFFFF)

# Interpolate between adjacent samples
s_current = atlas[..., phase_int]
s_next = atlas[..., (phase_int + 1) % frame_size]
s_interpolated = lerp_q16(s_current, s_next, phase_frac)
```

---

## 6. Frame Scanning

### 6.1 Scanning Rate

Frame position advances automatically during playback:

```python
frame_rate_q16 = (rate_hz * num_frames * 65536) / sample_rate
frame_q16 += frame_rate_q16  # Per sample
```

### 6.2 Loop Modes

| Mode | Behavior |
|------|----------|
| **loop** | Wrap to start when reaching end |
| **pingpong** | Reverse direction at boundaries |
| **oneshot** | Stop at end frame |

---

## 7. Wavetable State

### 7.1 State Structure

```python
@dataclass
class WavetableState:
    atlas: WavetableAtlas      # Reference to atlas
    phase_q32: int = 0         # Current phase (32-bit for precision)
    phase_inc_q32: int = 0     # Phase increment (from pitch)
    position_q16: int = 0      # X morph position
    frame_q16: int = 0         # Y frame position
    frame_rate_q16: int = 0    # Frame scanning rate
    is_playing: bool = False
```

### 7.2 Pitch Calculation

```python
# Phase increment from frequency
phase_inc_q32 = (freq_hz * (1 << 32)) / sample_rate

# From MIDI note
freq_hz = 440.0 * 2^((midi_note - 69) / 12)
```

---

## 8. Coefficient Generation

### 8.1 Generation Pipeline

```
WavetableState → wavetable_generate_block() → [64] audio samples
                                             ↓
                                      FWPT (analysis)
                                             ↓
                                   [64] coefficient leaves
                                             ↓
                                   Replicate across K slots
                                             ↓
                                      [512] output
```

### 8.2 Function Signature

```python
def wavetable_generate_coeffs(
    state: WavetableState,
    output_512: np.ndarray
) -> None:
    """
    Generate 512 coefficients from wavetable playback.

    Args:
        state: Wavetable oscillator state
        output_512: Output array [512] int16 to fill
    """
```

### 8.3 Coefficient Layout

```
output_512[leaf * 8 + k] = coeff[leaf] for k in 0..7
```

Where `leaf` is 0-63 and `k` is the time slot index.

---

## 9. InstrumentIR Integration

### 9.1 WavetableSpec

```python
@dataclass
class WavetableSpec:
    # Atlas reference
    atlas_name: str = "BasicWaves"
    atlas_index: Optional[int] = None

    # Initial position (0.0-1.0)
    position: float = 0.0
    frame: float = 0.0

    # Frame scanning
    frame_rate_hz: float = 0.0    # 0 = static
    frame_loop: bool = True

    # Modulation routing
    position_mod_source: Optional[str] = None
    frame_mod_source: Optional[str] = None
    position_mod_depth: float = 0.0
    frame_mod_depth: float = 0.0

    # Morphing
    morph_time_ms: float = 0.0
```

### 9.2 IR Placement

`WavetableSpec` is placed in L1Lift as a synthesis mode configuration:

```python
@dataclass
class L1Lift:
    # ... other fields ...
    wavetable_spec: Optional[WavetableSpec] = None
```

### 9.3 Example YAML

```yaml
instrument_id: wavetable_pad
name: Evolving Pad
layers:
  L1_lift:
    wavetable_spec:
      atlas_name: HarmonicSeries
      position: 0.0
      frame: 0.0
      frame_rate_hz: 0.1     # Slow evolution
      frame_loop: true
      frame_mod_source: macro:AGE
      frame_mod_depth: 0.5
```

---

## 10. Wavetable Bank

### 10.1 Bank Structure

```python
@dataclass
class WavetableBank:
    atlases: List[WavetableAtlas]

    def get(self, index: int) -> Optional[WavetableAtlas]
    def get_by_name(self, name: str) -> Optional[WavetableAtlas]
    def add(self, atlas: WavetableAtlas) -> int
```

### 10.2 Default Bank Contents

| Index | Name | Description |
|-------|------|-------------|
| 0 | BasicWaves | Sine, Triangle, Saw, Square |
| 1 | HarmonicSeries | Progressive harmonic addition |

---

## 11. Modulation Targets

### 11.1 Valid Modulation Sources

| Source Type | Example | Description |
|-------------|---------|-------------|
| **LFO** | `LFO1`, `LFO2` | Low-frequency oscillator |
| **Envelope** | `ENV1`, `ENV2` | ADSR envelope |
| **Macro** | `macro:MOTION` | System macro value |
| **Gesture** | `gesture:TILT` | Raw gesture input |

### 11.2 Modulation Application

```python
# Apply modulation to position
mod_position = position_q16 + ((mod_value_q15 * num_positions) >> 15)
mod_position = clamp(mod_position, 0, (num_positions - 1) << 16)

# Apply modulation to frame
mod_frame = frame_q16 + ((mod_value_q15 * num_frames) >> 15)
mod_frame = clamp(mod_frame, 0, (num_frames - 1) << 16)
```

---

## 12. Factory Atlases

### 12.1 BasicWaves

4 frames morphing through fundamental waveforms:

| Frame | Waveform |
|-------|----------|
| 0 | Sine |
| 1 | Triangle |
| 2 | Sawtooth |
| 3 | Square |

### 12.2 HarmonicSeries

Progressive harmonic addition (bandlimited):

```
Frame 0: Fundamental only
Frame N: Fundamental + harmonics 1..N
```

Default: 32 frames, up to 32 harmonics.

---

## 13. Performance Considerations

### 13.1 Interpolation Cost

| Operation | Cost |
|-----------|------|
| Phase lookup | O(1) |
| Frame interpolation | 4 samples + 2 lerps |
| Position interpolation | 2 lerps |
| Full bilinear | 4 lookups + 4 lerps |

### 13.2 Memory Usage

```
Atlas size = positions × frames × frame_size × 2 bytes

Example:
  1 position × 64 frames × 256 samples × 2 = 32 KB
  4 positions × 32 frames × 256 samples × 2 = 64 KB
```

### 13.3 Optimization: Pre-computed Coefficients

Future enhancement: Store pre-transformed coefficients:

```python
coeff_atlas[positions, frames, 64 leaves]
```

Eliminates FWPT from hot path for looping wavetables.

---

## 14. Oscillator Mode Integration

### 14.1 Mode Constant

```python
class OscMode(IntEnum):
    MODE_WAVETABLE = 4    # Wavetable synthesis with 2D interpolation
```

### 14.2 State Selection

```python
OSC_MODE_TABLE = {
    4: OscModeInfo(4, "MODE_WAVETABLE", "Wavetable synthesis", "WavetableState"),
}
```

---

## 15. Test Requirements

### 15.1 Unit Tests

- Atlas creation from arrays and waveforms
- Bilinear interpolation accuracy at corners and midpoints
- Frame scanning advances correctly
- Coefficient generation produces valid output

### 15.2 Integration Tests

- Full pipeline: bank → state → coefficients
- Morphing produces different timbres
- Modulation routing works correctly

### 15.3 Golden Traces

| Trace | Description |
|-------|-------------|
| `wt_basic_440hz.bin` | Basic sine wavetable at 440 Hz |
| `wt_morph_sweep.bin` | Position sweep 0→1 |
| `wt_frame_scan.bin` | Frame scanning at 1 Hz |

---

## 16. Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| OscMode enum | ✅ | `primitives.py` |
| WavetableAtlas | ✅ | `vm/wavetable_engine.py` |
| WavetableBank | ✅ | `vm/wavetable_engine.py` |
| WavetableState | ✅ | `vm/wavetable_engine.py` |
| Bilinear interpolation | ✅ | `vm/wavetable_engine.py` |
| Coefficient generation | ✅ | `vm/wavetable_engine.py` |
| WavetableSpec (IR) | ✅ | `instrument_ir.py` |
| Unit tests | ✅ | `tests/test_wavetable_engine.py` |
| Executor integration | 🔲 | Pending |
| Golden traces | 🔲 | Pending |

---

## 17. References

- WSE-STD-210: VM Hybrid Architecture
- WSE-STD-CMM-ROADMAP-v0: Synthesis Profile Roadmap
- TOWER-CONFORMANCE-ROADMAP: Tower Implementation Status
- `vm/sample_source.py`: Reference coefficient generation pattern

---

## Changelog

### v0.1 (2026-01-20)
- Initial draft
- Core atlas and interpolation implemented
- WavetableSpec added to InstrumentIR
- 29 unit tests passing
