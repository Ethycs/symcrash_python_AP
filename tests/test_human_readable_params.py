"""
Tests for human-readable parameter parsing.

Verifies that float alternatives for Q15/Q16 parameters work correctly
and that legacy integer format remains backward-compatible.
"""

import pytest
from symcrash_python_ap.instrument_ir import (
    pct_to_q15, q15_to_pct, float_or_q15,
    _auto_detect_ratios, _auto_detect_levels,
    load_instrument,
)
from symcrash_python_ap.library.preset_compiler import (
    FM6Config, MaterialConfig, LatchDefaults, CarrierConfig,
    CmmmDefaults, BackboneOp, PresetCompiler, validate_preset,
)
from symcrash_python_ap.primitives import Q15_ONE, Q16_ONE, Q15_HALF


# =============================================================================
# float_or_q15 helper
# =============================================================================

class TestFloatOrQ15Helper:
    """Test float_or_q15() conversion helper."""

    def test_float_key_converts_to_q15(self):
        result = float_or_q15({'beta_target': 0.5}, 'beta_target', 'beta_target_q15')
        assert result == 16384  # round(0.5 * 32767)

    def test_q15_key_passes_through(self):
        result = float_or_q15({'beta_target_q15': 8192}, 'beta_target', 'beta_target_q15')
        assert result == 8192

    def test_float_takes_priority_over_q15(self):
        result = float_or_q15(
            {'beta_target': 0.5, 'beta_target_q15': 9999},
            'beta_target', 'beta_target_q15'
        )
        assert result == 16384  # float wins

    def test_default_when_neither_key(self):
        result = float_or_q15({}, 'beta_target', 'beta_target_q15', 42)
        assert result == 42

    def test_zero_float(self):
        result = float_or_q15({'x': 0.0}, 'x', 'x_q15')
        assert result == 0

    def test_one_float(self):
        result = float_or_q15({'x': 1.0}, 'x', 'x_q15')
        assert result == Q15_ONE  # 32767

    def test_clamps_negative(self):
        result = float_or_q15({'x': -0.5}, 'x', 'x_q15')
        assert result == 0

    def test_clamps_over_one(self):
        result = float_or_q15({'x': 1.5}, 'x', 'x_q15')
        assert result == Q15_ONE


# =============================================================================
# pct_to_q15 rounding
# =============================================================================

class TestPctToQ15Rounding:
    """Test that pct_to_q15 uses round() for exact round-trip."""

    def test_half_rounds_correctly(self):
        # 0.5 * 32767 = 16383.5, should round to 16384
        assert pct_to_q15(0.5) == 16384

    def test_quarter_rounds_correctly(self):
        # 0.25 * 32767 = 8191.75, should round to 8192
        assert pct_to_q15(0.25) == 8192

    def test_zero(self):
        assert pct_to_q15(0.0) == 0

    def test_one(self):
        assert pct_to_q15(1.0) == Q15_ONE

    def test_roundtrip_common_fractions(self):
        """Common fractions should round-trip within 1 LSB."""
        for frac in [0.125, 0.1875, 0.25, 0.5, 0.75]:
            q15 = pct_to_q15(frac)
            back = q15_to_pct(q15)
            assert abs(back - frac) < 0.001, f"Round-trip failed for {frac}: {q15} -> {back}"


# =============================================================================
# FM6 ratio auto-detection
# =============================================================================

class TestFM6RatioAutoDetect:
    """Test auto-detection heuristic for op_ratios."""

    def test_legacy_q16_integers_unchanged(self):
        raw = [65536, 131072, 196608, 262144, 327680, 917504]
        result = _auto_detect_ratios(raw)
        assert result == raw  # Passed through unchanged

    def test_float_ratios_converted(self):
        raw = [1.0, 2.0, 3.0, 4.0, 5.0, 14.0]
        result = _auto_detect_ratios(raw)
        assert result == [65536, 131072, 196608, 262144, 327680, 917504]

    def test_small_integer_ratios_treated_as_float(self):
        raw = [1, 2, 3, 4, 5, 14]
        result = _auto_detect_ratios(raw)
        assert result == [65536, 131072, 196608, 262144, 327680, 917504]

    def test_mixed_float_int_treated_as_float(self):
        raw = [1.0, 2, 3.5, 4, 5, 14]
        result = _auto_detect_ratios(raw)
        assert result[0] == 65536   # 1.0 * 65536
        assert result[2] == 229376  # 3.5 * 65536

    def test_inharmonic_float_ratio(self):
        result = _auto_detect_ratios([3.5])
        assert result == [229376]

    def test_half_ratio(self):
        result = _auto_detect_ratios([0.5])
        assert result == [32768]

    def test_empty_list(self):
        assert _auto_detect_ratios([]) == []


# =============================================================================
# FM6 level auto-detection
# =============================================================================

class TestFM6LevelAutoDetect:
    """Test auto-detection heuristic for op_levels."""

    def test_legacy_q15_integers_unchanged(self):
        raw = [32767, 24576, 16384, 0, 0, 20480]
        result = _auto_detect_levels(raw)
        assert result == raw

    def test_float_levels_converted(self):
        raw = [1.0, 0.75, 0.5, 0.0, 0.0, 0.625]
        result = _auto_detect_levels(raw)
        assert result[0] == 32767  # round(1.0 * 32767)
        assert result[1] == 24575  # round(0.75 * 32767)
        assert result[2] == 16384  # round(0.5 * 32767)
        assert result[3] == 0
        assert result[5] == 20479  # round(0.625 * 32767)

    def test_all_zero_integers_unchanged(self):
        raw = [0, 0, 0, 0, 0, 0]
        result = _auto_detect_levels(raw)
        assert result == raw  # All int, no float detected

    def test_empty_list(self):
        assert _auto_detect_levels([]) == []


# =============================================================================
# FM6Config float alternatives
# =============================================================================

class TestFM6ConfigFloat:
    """Test FM6Config.from_dict() with float ratios/levels."""

    def test_float_ratios_in_fm6config(self):
        d = {
            'algorithm': 5,
            'feedback': 3,
            'op_ratios': [1.0, 2.0, 3.5, 1.0, 1.0, 14.0],
            'op_levels': [1.0, 0.75, 0.5, 0.0, 0.0, 0.625],
        }
        cfg = FM6Config.from_dict(d)
        assert cfg.algorithm_id == 5
        assert cfg.feedback_depth == 3
        assert cfg.op_ratios[0] == 65536
        assert cfg.op_ratios[2] == 229376
        assert cfg.op_levels[0] == 32767

    def test_legacy_integer_fm6config(self):
        d = {
            'algorithm': 1,
            'op_ratios': [65536, 131072, 196608, 262144, 327680, 917504],
            'op_levels': [32767, 24576, 16384, 0, 0, 20480],
        }
        cfg = FM6Config.from_dict(d)
        assert cfg.op_ratios == [65536, 131072, 196608, 262144, 327680, 917504]
        assert cfg.op_levels == [32767, 24576, 16384, 0, 0, 20480]


# =============================================================================
# BackboneOp float params
# =============================================================================

class TestBackboneFloatParams:
    """Test backbone operator float parameter alternatives."""

    def test_param0_float(self):
        compiler = PresetCompiler()
        preset = {
            'preset': {'preset_id': 'test'},
            'backbone': [
                {'op': 'TILT', 'strength': 0.45, 'param0': 0.3}
            ],
        }
        result = compiler.compile(preset)
        assert result.backbone[0].strength_q15 == round(0.45 * Q15_ONE)
        assert result.backbone[0].param0_q15 == round(0.3 * Q15_ONE)

    def test_param1_float(self):
        compiler = PresetCompiler()
        preset = {
            'preset': {'preset_id': 'test'},
            'backbone': [
                {'op': 'NOISE_BAND', 'param0': 0.5, 'param1': 0.25}
            ],
        }
        result = compiler.compile(preset)
        assert result.backbone[0].param0_q15 == round(0.5 * Q15_ONE)
        assert result.backbone[0].param1_q15 == round(0.25 * Q15_ONE)

    def test_legacy_q15_params_still_work(self):
        compiler = PresetCompiler()
        preset = {
            'preset': {'preset_id': 'test'},
            'backbone': [
                {'op': 'TILT', 'strength_q15': 14746, 'param0_q15': 9830}
            ],
        }
        result = compiler.compile(preset)
        assert result.backbone[0].strength_q15 == 14746
        assert result.backbone[0].param0_q15 == 9830


# =============================================================================
# MaterialConfig float alternatives
# =============================================================================

class TestMaterialConfigFloat:
    """Test MaterialConfig float alternatives in preset_compiler."""

    def test_beta_start_float(self):
        cfg = MaterialConfig.from_dict({'beta_start': 0.0})
        assert cfg.beta_start_q15 == 0

    def test_beta_target_float(self):
        cfg = MaterialConfig.from_dict({'beta_target': 0.5})
        assert cfg.beta_target_q15 == 16384

    def test_legacy_q15_still_works(self):
        cfg = MaterialConfig.from_dict({'beta_target_q15': 24576})
        assert cfg.beta_target_q15 == 24576

    def test_float_overrides_q15(self):
        cfg = MaterialConfig.from_dict({'beta_target': 0.75, 'beta_target_q15': 9999})
        assert cfg.beta_target_q15 == round(0.75 * Q15_ONE)


# =============================================================================
# CmmmDefaults traction float
# =============================================================================

class TestCmmmTractionFloat:
    """Test CmmmDefaults traction float alternative."""

    def test_traction_float(self):
        cmmm = CmmmDefaults.from_dict({'traction': 0.75})
        assert cmmm.traction_q15 == round(0.75 * Q15_ONE)

    def test_traction_q15_still_works(self):
        cmmm = CmmmDefaults.from_dict({'traction_q15': 16384})
        assert cmmm.traction_q15 == 16384

    def test_traction_default(self):
        cmmm = CmmmDefaults.from_dict({})
        assert cmmm.traction_q15 == Q15_ONE


# =============================================================================
# LatchDefaults float alternatives
# =============================================================================

class TestLatchDefaultsFloat:
    """Test LatchDefaults float alternatives."""

    def test_tree_blend_float(self):
        latch = LatchDefaults.from_dict({'tree_blend': 0.5})
        assert latch.tree_blend_q15 == 16384

    def test_tree_blend_q15_still_works(self):
        latch = LatchDefaults.from_dict({'tree_blend_q15': 8192})
        assert latch.tree_blend_q15 == 8192


# =============================================================================
# CarrierConfig pulse_width float
# =============================================================================

class TestCarrierConfigFloat:
    """Test CarrierConfig pulse_width float alternative."""

    def test_pulse_width_float(self):
        cfg = CarrierConfig.from_dict({'pulse_width': 0.3})
        assert cfg.pulse_width_q15 == round(0.3 * Q15_ONE)

    def test_pulse_width_q15_still_works(self):
        cfg = CarrierConfig.from_dict({'pulse_width_q15': 16384})
        assert cfg.pulse_width_q15 == 16384


# =============================================================================
# Backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Test that legacy integer formats still work identically."""

    def test_legacy_fm6_preset_loads(self):
        """Load an existing FM6 preset in legacy integer format."""
        from pathlib import Path
        preset_path = Path(__file__).parent.parent / 'presets' / 'instruments' / 'fm_brass_v0.yaml'
        if not preset_path.exists():
            pytest.skip("fm_brass_v0.yaml not found")
        ir = load_instrument(str(preset_path))
        # Verify the known values from fm_brass_v0.yaml
        fm6_node = None
        for node in ir.layers.L0_graph.nodes:
            if node.type == 'FM6':
                fm6_node = node
                break
        assert fm6_node is not None, "FM6 node not found"
        assert fm6_node.params.op_ratios[0] == 65536  # 1.0x
        assert fm6_node.params.op_levels[0] == 32767   # Full

    def test_legacy_material_config_loads(self):
        """MaterialConfig with legacy Q15 keys parses correctly."""
        cfg = MaterialConfig.from_dict({
            'beta_start_q15': 0,
            'beta_target_q15': 8192,
            'beta_ramp_blocks': 16,
        })
        assert cfg.beta_start_q15 == 0
        assert cfg.beta_target_q15 == 8192
        assert cfg.beta_ramp_blocks == 16


# =============================================================================
# Round-trip tests
# =============================================================================

class TestRoundTrip:
    """Test round-trip: float → Q15/Q16 → verify values match."""

    def test_material_roundtrip(self):
        """beta_target: 0.25 → Q15 → matches round(0.25 * 32767)."""
        cfg = MaterialConfig.from_dict({'beta_target': 0.25})
        assert cfg.beta_target_q15 == round(0.25 * 32767)

    def test_fm6_ratio_roundtrip(self):
        """ratio 2.0 → Q16 → exactly 131072."""
        result = _auto_detect_ratios([2.0])
        assert result[0] == 131072

    def test_fm6_ratio_inharmonic_roundtrip(self):
        """ratio 3.5 → Q16 → exactly 229376."""
        result = _auto_detect_ratios([3.5])
        assert result[0] == 229376

    def test_detune_ratio_roundtrip(self):
        """Detune ratios should round-trip correctly."""
        # Op2 in fm_brass_v0: 65726 → 65726/65536 = 1.002899...
        original = 65726
        as_float = original / 65536.0
        back = round(as_float * 65536)
        assert back == original


# =============================================================================
# Preset validation
# =============================================================================

class TestPresetValidation:
    """Test validate_preset() helper for instrument_ir_v0 format."""

    def test_valid_preset_no_errors(self):
        data = {
            'instrument_ir_v0': {
                'layers': {
                    'L0_graph': {
                        'nodes': [
                            {'id': 'osc1', 'type': 'Osc', 'params': {}},
                            {'id': 'amp1', 'type': 'Amp', 'params': {
                                'adsr': {'a_ms': 10, 'd_ms': 100, 's': 0.7, 'r_ms': 200}
                            }},
                        ],
                        'edges': [{'src': 'osc1', 'dst': 'amp1'}],
                        'macros': [{'id': 'M_BRIGHT', 'default': 0.5}],
                    }
                }
            }
        }
        errors = validate_preset(data)
        assert errors == []

    def test_missing_node_in_edge(self):
        data = {
            'instrument_ir_v0': {
                'layers': {
                    'L0_graph': {
                        'nodes': [{'id': 'osc1', 'type': 'Osc', 'params': {}}],
                        'edges': [{'src': 'osc1', 'dst': 'missing_amp'}],
                    }
                }
            }
        }
        errors = validate_preset(data)
        assert any("missing_amp" in e for e in errors)

    def test_macro_default_out_of_range(self):
        data = {
            'instrument_ir_v0': {
                'layers': {
                    'L0_graph': {
                        'nodes': [],
                        'edges': [],
                        'macros': [{'id': 'M_BAD', 'default': 1.5}],
                    }
                }
            }
        }
        errors = validate_preset(data)
        assert any("outside [0.0, 1.0]" in e for e in errors)

    def test_fm6_wrong_operator_count(self):
        data = {
            'instrument_ir_v0': {
                'layers': {
                    'L0_graph': {
                        'nodes': [{
                            'id': 'fm6', 'type': 'FM6',
                            'params': {'op_ratios': [1.0, 2.0, 3.0]}  # Only 3
                        }],
                        'edges': [],
                    }
                }
            }
        }
        errors = validate_preset(data)
        assert any("expected 6" in e for e in errors)

    def test_negative_adsr(self):
        data = {
            'instrument_ir_v0': {
                'layers': {
                    'L0_graph': {
                        'nodes': [{
                            'id': 'amp1', 'type': 'Amp',
                            'params': {'adsr': {'a_ms': -5.0, 'd_ms': 100, 's': 0.7, 'r_ms': 200}}
                        }],
                        'edges': [],
                    }
                }
            }
        }
        errors = validate_preset(data)
        assert any("negative" in e for e in errors)
