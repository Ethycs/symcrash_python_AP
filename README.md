# Symcrash Python AP

A deterministic, real-time audio synthesis engine with hardware control support.

## Quick Start

```bash
# Install dependencies
pixi install

# Run tests
pixi run pytest pipeline_tests/ -q

# Start virtual synth with GUI controls
pixi run virtual-synth

# Start live synth with hardware (Pico2)
pixi run live-synth
```

## Project Structure

```
symcrash_python_AP/
├── src/                          # Main source code
│   ├── symcrash_python_ap/       # Core synthesis engine
│   ├── symcrash_imgui_controls/  # GUI control interface
│   └── symcrash_pico2_controls/  # Hardware control (C/CMake)
├── pipeline_tests/               # Pipeline validation tests
├── audio_generation_tests/       # Audio output tests & analysis
├── tools/                        # Development and demo scripts
├── docs/                         # Architecture documentation (WSE-STD-*)
└── audio_out/                    # Generated audio files
```

## Test Organization

### Pipeline Tests (`pipeline_tests/`)

Core pipeline validation - run frequently during development:

```bash
# Run all pipeline tests
pixi run pytest pipeline_tests/ -q

# Run specific test category
pixi run pytest pipeline_tests/unit/ -v
pixi run pytest pipeline_tests/integration/ -v
pixi run pytest pipeline_tests/conformance/ -v
```

| Directory | Purpose |
|-----------|---------|
| `unit/` | Isolated component tests (oscillator, budgets, VM) |
| `integration/` | Cross-component tests (voice engine, wavelet engine) |
| `conformance/` | Standards compliance (WSE-STD-030, UART protocol) |
| `artifacts/` | Golden trace regression tests |

### Audio Generation Tests (`audio_generation_tests/`)

Tests that produce audio files for listening and analysis:

```bash
# Run audio tests (generates WAV files)
pixi run pytest audio_generation_tests/ -v

# Run specific category
pixi run pytest audio_generation_tests/unit/carrier/ -v
pixi run pytest audio_generation_tests/integration/pipeline/ -v
```

| Directory | Purpose |
|-----------|---------|
| `unit/carrier/` | Carrier wave generation and lifting |
| `unit/wavelet/` | Wavelet packet tree synthesis |
| `unit/waveform/` | Basic waveform generation |
| `unit/safety/` | Click prevention and OLA |
| `unit/presets/` | Preset rendering |
| `integration/pipeline/` | Full synthesis pipeline |
| `integration/wet_path/` | Effect/material path |
| `demos/` | Demo scripts (Fur Elise, electronic, etc.) |
| `analysis/` | Audio analysis tools |

Output files go to `audio_generation_tests/audio_out/` or `audio_out/`.

## Audio I/O

Centralized audio file handling with automatic timestamping:

```python
from symcrash_python_ap.audio_io import write_audio
import numpy as np

# Generate some audio
L = np.sin(np.linspace(0, 440 * 2 * np.pi, 48000)).astype(np.int16) * 16384
R = L.copy()

# Write with automatic timestamp
path = write_audio("my_sound", L, R)
# → audio_out/my_sound-20251219-114532.wav

# Write to subdirectory
write_audio("test_tone", L, R, subdirectory="tests")
# → audio_out/tests/test_tone-20251219-114532.wav

# Mono audio
write_audio("mono_tone", L)
```

The `write_audio` function:
- Automatically creates output directories
- Adds timestamps to filenames (configurable)
- Handles stereo/mono automatically
- Returns the output path for reference

## GUI and Hardware Controls

### Pixi Tasks

| Task | Command | Description |
|------|---------|-------------|
| `live-synth` | `pixi run live-synth` | Hardware control via Pico2 pots (console) |
| `live-synth-imgui` | `pixi run live-synth-imgui` | Hardware + ImGui visual feedback |
| `virtual-synth` | `pixi run virtual-synth` | No hardware, ImGui knobs for control |
| `virtual-synth-mp` | `pixi run virtual-synth-mp` | Multiprocess synth (lower latency) |
| `imgui-controls` | `pixi run imgui-controls` | Standalone ImGui panel (UART to hardware) |

### Command Line Options

```bash
# Virtual synth with GUI (no hardware needed)
python tools/pico2_live_synth.py --no-hardware --imgui

# Hardware synth with specific port
python tools/pico2_live_synth.py COM3

# Hardware synth with GUI overlay
python tools/pico2_live_synth.py --imgui

# Start at different note
python tools/pico2_live_synth.py --no-hardware --imgui --note 48
```

### Hardware Controls (Pico2)

When connected to Pico2 hardware:

| Pot | Parameter | Range | Control |
|-----|-----------|-------|---------|
| 0 | Attack | 0-665ms | ATK_BLOCKS |
| 1 | Decay | 0-1333ms | DCY_BLOCKS |
| 2 | Sustain | 0-100% | SUS_Q15 |
| 3 | Release | 0-2000ms | REL_BLOCKS |
| 4 | Cutoff | 0-100% | CHAOS |
| 5 | Resonance | 0-100% | TEXTURE |
| 6 | Harmonics | 0-100% | RESERVED_7 |
| 7 | Eta | 0-100% (dry/wet) | MORPH |
| 8 | Detune | ±2 octaves | DETUNE |
| 9 | Gain | 0-100% | MASTER_GAIN |

### Keyboard Controls (Console)

| Key | Action |
|-----|--------|
| SPACE | Toggle note on/off |
| UP/DOWN | Change note (+/- 1 semitone) |
| s | Print status |
| q | Quit |

### ImGui Controls

The ImGui window provides:
- 10 rotary knobs for all parameters
- Visual feedback of current values
- Note trigger buttons
- Preset selection
- Connection status

## Tools Directory

Development and analysis scripts in `tools/`:

| Category | Examples |
|----------|----------|
| Live Synth | `pico2_live_synth.py`, `virtual_synth_mp.py` |
| Analysis | `analyze_spectrum.py`, `check_carrier_purity.py` |
| Debugging | `debug_crossfade.py`, `trace_carrier_path.py` |
| Benchmarks | `bench_idwpt.py`, `bench_vm.py` |
| Testing | `check_discontinuity.py`, `compare_wet_dry.py` |

## Architecture Overview

The synthesis pipeline consists of five main engines:

1. **Voice Engine** - Musical intent, ADSR envelopes, voice allocation
2. **Symbol Engine** - Operator scheduling, coefficient computation
3. **Wavelet Engine** - Audio reconstruction via inverse DWT
4. **Material Engine** - Nonlinear processing, effects
5. **Safety Engine** - Click prevention, limiting, OLA windowing

All processing uses Q15 fixed-point arithmetic for determinism.

## Documentation

Technical standards in `docs/`:

| Document | Topic |
|----------|-------|
| WSE-STD-001 | Numeric Determinism (Q15 rules) |
| WSE-STD-010 | Tree Library |
| WSE-STD-020 | UART Protocol |
| WSE-STD-030 | Symbol Engine Executor |
| WSE-STD-050 | Degradation Semantics |
| WSE-STD-060 | Time Ownership |
| WSE-STD-070 | Anti-Click Operator |
| WSE-STD-VM-HYBRID | Virtual Machine Design |

## Dependencies

- Python >= 3.11
- numpy >= 2.3
- sounddevice >= 0.5
- imgui-bundle >= 1.92
- pyserial >= 3.5
- numba >= 0.60
- scipy >= 1.16
- pytest >= 9.0

Install via Pixi:
```bash
pixi install
```

## License

[Add license information]
