# Symcrash Python AP

A deterministic, real-time audio synthesis engine with hardware control support.

The target is the documented OrangeCrab ECP5-25F with 128 MB LPDDR3 and
128 KB BRAM. The current Mode A image makes combined oscillator storage
**exceed that target's memory limits** and needs replacement. Run `pixi run check-hardware-resources`
to check their packed storage; it currently exits with a failure. Passing the
Python audio tests does not establish hardware conformance. The
[hardware requirement mapping](symcrash_docs/05%20-%20Standards/01%20-%20Architecture/WSE-IMPL-HARDWARE-CONSTRAINTS-v1.md)
links the existing normative budgets and the unresolved implementation work.

## Quick Start

```bash
# Install dependencies
pixi install

# Prepare Mode A tables and Mode B source anchors before audio startup
pixi run python tools/prepare_mode_a_tables.py
pixi run python tools/prepare_mode_b_basis.py

# Run tests
pixi run pytest tests/ -q

# Start virtual synth with GUI controls
pixi run virtual-synth

# Start live synth with hardware (Pico2)
pixi run live-synth
```

Mode B uses a validated, read-only basis image under `presets/.build/resources/`.
Its 128 fixed source anchors occupy 3,124,940 bytes. Between-anchor lookup
interpolates coefficients while retaining the exact accepted Q32 phase. Hosts
prepare it automatically during initialization if absent. Preparation must
finish before starting audio callbacks. Subsequent startups validate and map
the existing image; pitch changes do not run forward wavelet transforms.
The [resource profile](symcrash_docs/05%20-%20Standards/06%20-%20CCS/WSE-IMPL-BASIS-RESOURCE-v3.md)
describes arithmetic, regeneration, measured audio changes and unresolved
hardware processing/streaming costs.

Mode A uses an exact integer-cent pitch image: 63,508 tables in about
1.94 GiB, built in 87.7 seconds locally. Hosts prepare both images before playback.
Its tables follow the shared Q32 pitch law and cap source harmonics below
0.9 Nyquist; cache misses only create views of immutable data. The
[pitch/resource profile](symcrash_docs/05%20-%20Standards/06%20-%20CCS/WSE-IMPL-MODE-A-RESOURCE-v2.md)
records the arithmetic correction and its unacceptable target storage cost.

## Project Structure

```
symcrash_python_AP/
├── src/                          # Main source code
│   ├── symcrash_python_ap/       # Core synthesis engine
│   ├── symcrash_imgui_controls/  # GUI control interface
│   └── symcrash_pico2_controls/  # Hardware control (C/CMake)
├── tests/                        # Unit, integration, conformance, audio tests
├── tools/                        # Development and demo scripts
├── presets/                      # Preset YAML source + ignored .build library
├── symcrash_docs/                 # Documentation submodule (WSE-STD-*)
└── audio_out/                    # Generated audio files
```

`presets/` is the only preset root. Tracked YAML source lives in its category
folders; compiled capsules, indexes, caches, and previews are regenerated in
`presets/.build/`. See [`presets/README.md`](presets/README.md).

## Test Organization

### Pipeline Tests (`tests/`)

Core pipeline validation - run frequently during development:

```bash
# Run all pipeline tests
pixi run pytest tests/ -q

# Run specific test category
pixi run pytest tests/unit/ -v
pixi run pytest tests/integration/ -v
pixi run pytest tests/conformance/ -v
```

| Directory | Purpose |
|-----------|---------|
| `unit/` | Isolated component tests (oscillator, budgets, VM) |
| `integration/` | Cross-component tests (voice engine, wavelet engine) |
| `conformance/` | Standards compliance (WSE-STD-030, UART protocol) |
| `artifacts/` | Golden trace regression tests |

### Audio quality and preset rendering

```bash
# Check audio quality and spectral behavior
pixi run pytest tests/quality/ -v
pixi run pytest tests/spectral/ -v

# Render a preset for listening
pixi run python tools/render/render_preset.py presets/instruments/dx_marimba_v0.yaml --note 60 --duration 2.5
```

Generated audio goes under the ignored `audio_out/` directory.

## Audio I/O

Centralized audio file handling with automatic timestamping:

```python
from symcrash_python_ap.audio_io import write_audio
import numpy as np

# Generate some audio
L = np.rint(16384 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.int16)
R = L.copy()

# Write with automatic timestamp
metadata = write_audio("my_sound", L, R)
print(metadata.filepath)
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
- Returns metadata including the output filepath

## GUI and Hardware Controls

### Pixi Tasks

| Task | Command | Description |
|------|---------|-------------|
| `live-synth` | `pixi run live-synth` | Hardware control via Pico2 pots (console) |
| `live-synth-imgui` | `pixi run live-synth-imgui` | Hardware + ImGui visual feedback |
| `virtual-synth` | `pixi run virtual-synth` | No hardware, ImGui knobs for control |
| `virtual-synth-mp` | `pixi run virtual-synth-mp` | Multiprocess synth with ImGui controls |
| `imgui-controls` | `pixi run imgui-controls` | Standalone ImGui panel (UART to hardware) |

### Command Line Options

```bash
# Virtual synth with GUI (no hardware needed)
python tools/synth/pico2_live_synth.py --no-hardware --imgui

# Hardware synth with specific port
python tools/synth/pico2_live_synth.py COM3

# Hardware synth with GUI overlay
python tools/synth/pico2_live_synth.py --imgui

# Start at different note
python tools/synth/pico2_live_synth.py --no-hardware --imgui --note 48
```

### Hardware Controls (Pico2)

When connected to Pico2 hardware:

| Pot | Parameter | Range | Control |
|-----|-----------|-------|---------|
| 0 | Attack | 1.33-667ms | ATK_BLOCKS |
| 1 | Decay | 1.33-1333ms | DCY_BLOCKS |
| 2 | Sustain | 0-100% | SUS_Q15 |
| 3 | Release | 1.33-2000ms | REL_BLOCKS |
| 4 | Chaos family control | 0-100% | CHAOS |
| 5 | Local texture family control | 0-100% | TEXTURE |
| 6 | Reserved (inactive) | — | RESERVED_7 |
| 7 | Tree morph | 0-100% (A/B blend) | MORPH |
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
| Live Synth | `synth/pico2_live_synth.py`, `synth/virtual_synth_mp.py` |
| Presets | `lint_presets.py`, `render/compile_presets.py` |
| Rendering | `render/render_preset.py`, `render/render_all_presets.py` |
| Benchmarks | `benchmarks/bench_vm.py`, `benchmarks/bench_osc_modes.py` |
| Audio comparison | `gym/timbre_distance.py`, `gym/dexed_reference.py` |

## Architecture Overview

The synthesis pipeline consists of five main engines:

1. **Voice Engine** - Musical intent, ADSR envelopes, voice allocation
2. **Symbol Engine** - Operator scheduling, coefficient computation
3. **Wavelet Engine** - Audio reconstruction via inverse DWT
4. **Material Engine** - Nonlinear processing, effects
5. **Safety Engine** - Click prevention, limiting, OLA windowing

The synthesis audio paths use fixed-point arithmetic. Preset authoring and some control-rate conversions use floating point. Material chorus and SVF playback use frozen integer lookup tables; extended material programs have frozen output-hash tests.

## Documentation

Current verification and remaining gaps: [conformance report](symcrash_docs/07%20-%20Status%20Reports/CONFORMANCE-2026-10-30.md).

Technical standards in `symcrash_docs/05 - Standards/`:

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

- Python >= 3.11 (the Pixi environment pins Python 3.14)
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

[MIT](LICENSE)
