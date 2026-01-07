## Pipeline Tests Migration

**Date**: 2026-01-03
**Status**: Phase 1 Complete

---

## Summary

Migrated 43+ tools from `tools/` into organized `pipeline_tests/` structure with pytest-based test modules. Consolidated duplicate functionality and created reusable test helpers.

---

## New Directory Structure

```
pipeline_tests/
├── spectral/           # FFT and frequency domain analysis
│   ├── __init__.py
│   └── test_spectrum_analysis.py
│
├── quality/            # Signal integrity and artifact detection
│   ├── __init__.py
│   ├── test_discontinuity.py
│   ├── test_carrier_purity.py
│   └── test_combing_artifacts.py
│
├── conformance/        # Standards compliance and roundtrip tests
│   ├── __init__.py
│   └── test_wavelet_roundtrip.py
│
├── generation/         # Test audio generation helpers
│   └── __init__.py
│
├── integration/        # Existing integration tests
├── unit/              # Existing unit tests
└── debug/             # Existing debug tests
```

---

## Migrated Test Modules

### 1. test_spectrum_analysis.py (spectral/)

**Consolidates**: 10 spectrum analysis tools from tools/
- `analyze_spectrum.py`
- `analyze_spectrum_windowed.py`
- `trace_carrier_path.py`
- `compare_carrier_test.py`
- And others...

**Key Features**:
- `render_pipeline()` - Unified full-pipeline rendering
- `analyze_spectrum()` - Windowed FFT with Blackman window
- Harmonic purity verification
- Non-harmonic artifact detection (-60 dB threshold)
- Noise floor measurement

**Tests**:
- `test_pure_carrier_sine()` - No non-harmonic content
- `test_carrier_harmonic_decay()` - 2nd harmonic <-60 dB
- `test_wet_path_25_percent()` - 25% wet purity (≤2 peaks)
- `test_wet_path_100_percent()` - 100% wet purity (≤5 peaks)
- `test_50_50_mix()` - Mixed signal quality
- `test_windowing_reduces_leakage()` - Windowing validation

**Run**:
```bash
pixi run pytest pipeline_tests/spectral/test_spectrum_analysis.py -v
```

---

### 2. test_discontinuity.py (quality/)

**Consolidates**: 18 quality/purity tools from tools/
- `check_discontinuity.py`
- `check_pure_sine.py`
- `trace_block_boundary.py`
- `analyze_dc_offset.py`
- And others...

**Key Features**:
- `compute_expected_accel()` - ω² × amplitude for sine waves
- `detect_discontinuities()` - Acceleration-based detection
- `render_scenario()` - VM rendering with parameter ramping
- Second-order difference analysis (acceleration)
- Block boundary continuity checking

**Tests**:
- `test_pure_carrier_no_discontinuities()` - Carrier continuity
- `test_wet_path_steady_state()` - Wet path continuity
- `test_eta_ramp()` - ETA ramping smoothness
- `test_gain_ramp()` - Gain ramping smoothness
- `test_block_boundaries_carrier()` - Zero boundary discontinuities

**Run**:
```bash
pixi run pytest pipeline_tests/quality/test_discontinuity.py -v
```

---

### 3. test_carrier_purity.py (quality/)

**Consolidates**:
- `test_carrier_purity.py`
- `check_carrier_purity.py`
- `check_raw_sine.py`

**Key Features**:
- `generate_carrier_vm()` - VM-level carrier generation
- `analyze_spectrum_with_window()` - Hann-windowed spectrum analysis
- `check_block_boundaries()` - Boundary continuity verification
- Sideband detection within ±50 Hz of fundamental
- Harmonic content analysis

**Tests**:
- `test_carrier_has_no_sidebands()` - No frequency splatter
- `test_carrier_pure_sine_no_harmonics()` - <-60 dB harmonics
- `test_block_boundaries_continuous()` - No clicks at boundaries
- `test_avg_boundary_diff_within_tolerance()` - Smooth transitions
- `test_purity_at_frequency()` - Parametrized frequency sweep

**Run**:
```bash
pixi run pytest pipeline_tests/quality/test_carrier_purity.py -v
```

---

### 4. test_combing_artifacts.py (quality/)

**Consolidates**:
- `test_combing.py`
- `analyze_mix_artifacts.py`
- `analyze_multiblock_mix.py`

**Key Features**:
- `generate_vm_audio()` - VM rendering with beta_target control
- `analyze_non_harmonic_content()` - Comb detection via FFT
- `detect_sample_discontinuities()` - Large jump detection
- Non-harmonic peak counting
- Block boundary discontinuity analysis

**Tests**:
- `test_dry_no_non_harmonic_peaks()` - Pure carrier
- `test_full_wet_limited_artifacts()` - ≤5 non-harmonic peaks
- `test_mixed_signal_purity()` - Parametrized wet amounts
- `test_dry_purity_at_frequency()` - Frequency sweep

**Run**:
```bash
pixi run pytest pipeline_tests/quality/test_combing_artifacts.py -v
```

---

### 5. test_wavelet_roundtrip.py (conformance/)

**Consolidates**:
- `test_roundtrip.py`
- `test_idwpt_purity.py`

**Key Features**:
- `compute_roundtrip_error()` - FWPT->IDWPT error analysis
- Test data generators (sine, DC, offset, actual carrier)
- Headroom bits effect testing
- Max error and RMS error metrics

**Tests**:
- `test_pure_sine_roundtrip()` - ≤2 LSB max error
- `test_offset_sine_roundtrip()` - DC offset handling
- `test_dc_roundtrip()` - Pure DC preservation
- `test_zero_roundtrip()` - Exact zero roundtrip
- `test_unity_mask_matches_passthrough()` - Mask equivalence
- `test_headroom_bits_accuracy()` - Headroom 0-4 accuracy
- `test_roundtrip_at_frequency()` - Parametrized freq sweep

**Run**:
```bash
pixi run pytest pipeline_tests/conformance/test_wavelet_roundtrip.py -v
```

---

## Test Patterns & Best Practices

### Pytest Structure
All test modules follow standard pytest conventions:
- `class Test*` for organization
- `def test_*()` for individual tests
- `@pytest.mark.parametrize` for data-driven tests
- `@pytest.mark.manual` for manual inspection tests (not in CI)

### Audio Output
All test modules use centralized `audio_io`:
```python
from symcrash_python_ap.audio_io import write_audio

meta = write_audio(
    "test_name",
    samples,
    timestamp=True,
    subdirectory="test_category"
)
```

Output goes to `audio_out/<subdirectory>/` with automatic timestamping.

### Manual Inspection Tests
Tests marked with `@pytest.mark.manual` generate audio files for manual analysis:
```bash
# Generate spectrum test files
pixi run pytest pipeline_tests/spectral/test_spectrum_analysis.py::test_generate_spectrum_test_files -v -s

# Generate discontinuity test files
pixi run pytest pipeline_tests/quality/test_discontinuity.py::test_generate_discontinuity_test_files -v -s
```

These are excluded from CI but useful for debugging.

---

## Reusable Test Helpers

### Rendering Helpers

**Full Pipeline** (UnifiedOutputStage):
```python
from pipeline_tests.spectral.test_spectrum_analysis import render_pipeline

samples = render_pipeline(
    note=60,
    beta_target_q15=16384,  # 50% wet
    duration_blocks=1500,
)
```

**VM-Level** (vm_step_block):
```python
from pipeline_tests.quality.test_discontinuity import render_scenario

samples = render_scenario(
    note=60,
    f0_hz=261.63,
    eta_q15=32767,
    gain_q15=24576,
    num_blocks=100,
    ramp_eta=True,  # Ramp from 0 to eta_q15
)
```

### Analysis Helpers

**Spectrum Analysis**:
```python
from pipeline_tests.spectral.test_spectrum_analysis import analyze_spectrum

result = analyze_spectrum(samples, use_window=True, threshold_db=-60)
# Returns: fundamental_freq, harmonics, non_harmonic peaks, noise_floor_db, is_pure
```

**Discontinuity Detection**:
```python
from pipeline_tests.quality.test_discontinuity import detect_discontinuities, compute_expected_accel

expected = compute_expected_accel(peak_amplitude=24576, f0_hz=261.63)
result = detect_discontinuities(samples, threshold_lsb=24, expected_accel=expected)
# Returns: max_boundary_accel, max_within_block_accel, num_violations, details
```

**Carrier Purity**:
```python
from pipeline_tests.quality.test_carrier_purity import analyze_spectrum_with_window

result = analyze_spectrum_with_window(samples)
# Returns: fundamental_freq, sidebands, harmonics
```

---

## Migration Status

### ✅ Completed (Phase 1)

| Category | Files Migrated | New Test Module |
|----------|----------------|-----------------|
| Spectral Analysis | 10 | test_spectrum_analysis.py |
| Quality/Purity | 18 | test_discontinuity.py, test_carrier_purity.py, test_combing_artifacts.py |
| Wavelet Roundtrip | 2 | test_wavelet_roundtrip.py |
| **Total** | **30** | **5 modules** |

### 📋 Remaining Work (Phase 2)

**To Migrate**:
- [ ] ETA/beta mixing tests (8 files) → integration/
- [ ] Debug tools (20 files) → debug/ or consolidate
- [ ] Benchmark tools (12 files) → integration/
- [ ] Generation tools (15 files) → generation/ helpers

**To Keep in tools/**:
- Live synth tools (pico2_live_synth.py, standalone_synth.py)
- Interactive tools (imgui_controls.py)
- Device testing utilities

**Audio I/O Updates**:
- [ ] Update remaining 34 tools to use `write_audio()`
- [x] Updated: save_and_play.py, generate_wet_test.py, generate_carrier_only_test.py, test_synth_prerender.py

---

## Running All Tests

```bash
# Run all spectral tests
pixi run pytest pipeline_tests/spectral/ -v

# Run all quality tests
pixi run pytest pipeline_tests/quality/ -v

# Run all conformance tests
pixi run pytest pipeline_tests/conformance/ -v

# Run everything
pixi run pytest pipeline_tests/ -v

# Run with coverage
pixi run pytest pipeline_tests/ --cov=symcrash_python_ap --cov-report=html
```

---

## Key Metrics & Thresholds

### Spectral Purity
- **Carrier (dry)**: No non-harmonic peaks above -60 dB
- **25% wet**: ≤2 non-harmonic peaks
- **50% wet**: ≤4 non-harmonic peaks
- **100% wet**: ≤5 non-harmonic peaks
- **Noise floor**: <-60 dB for carrier, <-40 dB for wet

### Discontinuity Detection
- **Carrier**: Zero discontinuities (threshold: -60 dB)
- **Wet steady-state**: <5 boundary violations (threshold: -40 dB)
- **ETA ramp**: <10 violations during transition
- **Block boundaries**: Max diff ≤ expected sine diff × 1.1

### Wavelet Roundtrip
- **Pure sine**: ≤2 LSB max error, <1.0 RMS
- **With DC offset**: ≤5 LSB max error, <2.0 RMS
- **Small amplitude**: ≤1 LSB max error
- **Pure DC**: Exact preservation (≤1 LSB)

### Carrier Purity
- **Harmonics**: <-60 dB for pure sine
- **Sidebands**: None within ±50 Hz of fundamental
- **Block boundaries**: Continuous (within 10% of expected diff)

---

## Benefits of Migration

1. **Unified Testing Framework**: All tests use pytest conventions
2. **Reusable Helpers**: Common rendering and analysis functions
3. **Organized by Purpose**: Spectral, quality, conformance categories
4. **Automated CI**: Tests run in continuous integration
5. **Manual Inspection**: @pytest.mark.manual for debugging
6. **Centralized Audio I/O**: Consistent file output with timestamping
7. **Documentation**: Self-documenting test names and docstrings
8. **Reduced Duplication**: 30 tools → 5 consolidated modules

---

## Next Steps

1. **Phase 2 Migration**:
   - Create test_eta_effects.py for ETA/beta mixing tests
   - Consolidate debug tools into test_debug_scenarios.py
   - Migrate benchmarks to integration/

2. **Audio I/O Completion**:
   - Update remaining 34 tools to use write_audio()
   - Remove direct wav.open() and scipy.io.wavfile usage

3. **Documentation**:
   - Update README with pipeline_tests structure
   - Create CONTRIBUTING guide for adding new tests

4. **CI Integration**:
   - Add pipeline_tests to GitHub Actions
   - Set up coverage reporting
   - Create performance regression tracking
