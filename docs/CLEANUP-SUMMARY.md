# Tools Cleanup Summary

**Date**: 2026-01-03
**Action**: Removed redundant test_* files after integration into pipeline_tests/

---

## Removed Files (11 total)

### ✅ Integrated into pipeline_tests/quality/test_carrier_purity.py
- `tools/test_carrier_purity.py`

### ✅ Integrated into pipeline_tests/quality/test_combing_artifacts.py
- `tools/test_combing.py`

### ✅ Integrated into pipeline_tests/conformance/test_wavelet_roundtrip.py
- `tools/test_roundtrip.py`
- `tools/test_idwpt_purity.py`
- `tools/test_phase_delay.py`
- `tools/test_effect_mode.py`

### ✅ Integrated into pipeline_tests/spectral/test_spectrum_analysis.py
- `tools/test_wet_only.py` - Pure wet spectrum (covered by test_wet_path_100_percent)
- `tools/test_low_eta.py` - Low ETA spectrum (covered by parametrized wet tests)
- `tools/test_eta_spectrum.py` - ETA spectrum sweep (covered by parametrized tests)
- `tools/test_eta_effect.py` - ETA mix effect (covered by wet path tests)
- `tools/test_eta_steady.py` - ETA steady-state (covered by stability tests)

---

## Remaining Test Files (22 total)

### Device/Audio Testing (7 files) - Keep for hardware debugging
- `test_audio_simple.py` - Simple audio playback test
- `test_callback_sine.py` - Callback sine generation
- `test_callback_synth.py` - Callback-based synthesis
- `test_default_audio.py` - Default device selection
- `test_sd_fresh.py` - Fresh sounddevice verification
- `test_synth_device4.py` - Device-specific synthesis
- `test_wasapi.py` - WASAPI device testing

### Live Synthesis (4 files) - Keep for interactive testing
- `test_queue_synth.py` - Queue-based synthesis
- `test_live_render.py` - Live rendering test
- `test_live_synth_flow.py` - Live synth data flow
- `test_synth_prerender.py` - Pre-rendered synthesis
- `test_synth_simple.py` - Simple synthesis test

### Low-Level Unit Tests (6 files) - Keep for implementation verification
- `test_crossfade.py` - Equal-power crossfade testing
- `test_sincos.py` - Sincos interpolation accuracy
- `test_sine_fast.py` - Fast sine generation
- `test_safety_dc.py` - DC safety filter
- `test_separated_transforms.py` - Separated transform implementation
- `test_vm_direct.py` - Direct VM execution

### Debug/Investigation (3 files) - Keep for troubleshooting
- `test_eta_ccs.py` - ETA with CCS integration
- `test_eta_init.py` - ETA initialization
- `test_eta_toggle.py` - ETA toggle behavior
- `test_vm_eta_effect.py` - VM ETA effects

---

## Before/After Comparison

| Category | Before | After | Removed |
|----------|--------|-------|---------|
| test_* files in tools/ | 33 | 22 | 11 |
| Integrated into pipeline_tests/ | 0 | 5 modules | N/A |
| Total test coverage | Scattered | Organized | Improved |

---

## What Was Consolidated

### Spectral Analysis
**Before**: 5 separate scripts testing ETA spectrum, wet-only, low ETA
**After**: 1 unified module with parametrized tests covering 0-100% wet

### Carrier Purity
**Before**: 1 standalone script
**After**: Integrated with block boundary checks and frequency sweeps

### Combing Detection
**Before**: 1 standalone script
**After**: Parametrized tests for dry/wet/mixed signals

### Wavelet Roundtrip
**Before**: 4 separate scripts (roundtrip, purity, phase delay, effect mode)
**After**: 1 comprehensive module with 12+ test cases

---

## Remaining Work Categories

### Should Stay in tools/
These files serve specific purposes not suitable for pytest:
- **Device testing** - Hardware-specific, interactive
- **Live synthesis** - Real-time playback, interactive
- **Debug scripts** - One-off investigations

### Could Be Migrated (Future)
Consider migrating these to pipeline_tests/ in Phase 2:
- `test_crossfade.py` → pipeline_tests/unit/test_crossfade.py
- `test_sincos.py` → pipeline_tests/unit/test_sincos.py
- `test_safety_dc.py` → pipeline_tests/unit/test_safety_dc.py
- `test_vm_direct.py` → pipeline_tests/unit/test_vm_execution.py

---

## Benefits of Cleanup

1. **Reduced Duplication**: 11 redundant files removed
2. **Better Organization**: Related tests consolidated into logical modules
3. **Pytest Integration**: All integrated tests now run in CI
4. **Easier Maintenance**: Single source of truth for each test category
5. **Improved Coverage**: Parametrized tests cover more cases than originals
6. **Clear Separation**: tools/ now focused on interactive/hardware tools

---

## Running Integrated Tests

```bash
# Run all integrated tests
pixi run pytest pipeline_tests/ -v

# Run specific categories
pixi run pytest pipeline_tests/spectral/ -v      # Spectrum analysis
pixi run pytest pipeline_tests/quality/ -v       # Purity & continuity
pixi run pytest pipeline_tests/conformance/ -v   # Roundtrip & standards

# Generate test audio for manual inspection
pixi run pytest pipeline_tests/spectral/test_spectrum_analysis.py::test_generate_spectrum_test_files -v -s
```

---

## Migration Mapping

| Original Tool | New Test Module | Test Method |
|---------------|-----------------|-------------|
| test_carrier_purity.py | quality/test_carrier_purity.py | test_carrier_has_no_sidebands() |
| test_combing.py | quality/test_combing_artifacts.py | test_dry_no_non_harmonic_peaks() |
| test_roundtrip.py | conformance/test_wavelet_roundtrip.py | test_pure_sine_roundtrip() |
| test_idwpt_purity.py | conformance/test_wavelet_roundtrip.py | test_roundtrip_at_frequency() |
| test_phase_delay.py | conformance/test_wavelet_roundtrip.py | test_headroom_bits_accuracy() |
| test_effect_mode.py | conformance/test_wavelet_roundtrip.py | test_unity_mask_matches_passthrough() |
| test_wet_only.py | spectral/test_spectrum_analysis.py | test_wet_path_100_percent() |
| test_low_eta.py | spectral/test_spectrum_analysis.py | test_wet_path_25_percent() |
| test_eta_spectrum.py | spectral/test_spectrum_analysis.py | Parametrized wet tests |
| test_eta_effect.py | spectral/test_spectrum_analysis.py | test_50_50_mix() |
| test_eta_steady.py | quality/test_combing_artifacts.py | test_full_wet_limited_artifacts() |

---

## Verification

All removed files have equivalent or superior coverage in pipeline_tests/:

✅ **Spectral purity** - More comprehensive with windowing
✅ **Carrier quality** - Added block boundary checks
✅ **Combing detection** - Parametrized across wet amounts
✅ **Roundtrip accuracy** - Extended with headroom and frequency sweeps
✅ **Phase delay** - Covered by roundtrip error analysis
✅ **ETA effects** - Parametrized 0-100% wet coverage

No functionality was lost in the migration.
