"""
Parameter Registry Conformance Tests
=====================================

Tests for the Closure Appendix v0.1 Parameter Registry implementation.
"""

import pytest
from src.symcrash_python_ap.parameter_registry import (
    # Constants
    Q15_MIN,
    Q15_MAX,
    Q15_SCALE,
    Q15_ONE,
    Q8_MIN,
    Q8_MAX,
    U8_MIN,
    U8_MAX,
    # Enums
    ParamDomain,
    ParamType,
    UpdateMode,
    Persistence,
    # Classes
    ParameterDef,
    ParameterState,
    ParameterRegistry,
    ParameterBank,
    # Fixed-point functions
    q15_mul,
    q15_add,
    q15_sub,
    q15_from_float,
    q15_to_float,
    q8_from_float,
    q8_to_float,
    # Defaults
    CANONICAL_DEFAULTS,
    # Global registry
    get_registry,
)


# =============================================================================
# CONSTANT TESTS
# =============================================================================

class TestConstants:
    """Test fixed-point constants."""

    def test_q15_range(self):
        """Q15 range should be [-32768, 32767]."""
        assert Q15_MIN == -32768
        assert Q15_MAX == 32767
        assert Q15_ONE == 32767
        assert Q15_SCALE == 32768

    def test_q8_range(self):
        """Q8 range should be [-128, 127]."""
        assert Q8_MIN == -128
        assert Q8_MAX == 127

    def test_u8_range(self):
        """U8 range should be [0, 255]."""
        assert U8_MIN == 0
        assert U8_MAX == 255


# =============================================================================
# PARAMETER DEFINITION TESTS
# =============================================================================

class TestParameterDef:
    """Test parameter definition structure."""

    def test_parameter_creation(self):
        """Parameter definition should store all fields."""
        param = ParameterDef(
            param_id=0x0001,
            name="gain_q15",
            domain=ParamDomain.VOICE,
            dtype=ParamType.Q15,
            min_value=0,
            max_value=Q15_MAX,
            default_value=Q15_ONE,
        )
        assert param.param_id == 0x0001
        assert param.name == "gain_q15"
        assert param.domain == ParamDomain.VOICE
        assert param.dtype == ParamType.Q15
        assert param.default_value == Q15_ONE

    def test_validate_value_clamping(self):
        """validate_value should clamp to [min, max]."""
        param = ParameterDef(
            param_id=1, name="test",
            domain=ParamDomain.VOICE, dtype=ParamType.Q15,
            min_value=0, max_value=100,
            default_value=50,
        )

        # Within range
        assert param.validate_value(50) == 50

        # Below min
        assert param.validate_value(-10) == 0

        # Above max
        assert param.validate_value(200) == 100

    def test_type_range_q15(self):
        """type_range should return correct range for Q15."""
        param = ParameterDef(
            param_id=1, name="test",
            domain=ParamDomain.VOICE, dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=0,
        )
        min_val, max_val = param.type_range()
        assert min_val == Q15_MIN
        assert max_val == Q15_MAX

    def test_type_range_bool(self):
        """type_range should return [0, 1] for bool."""
        param = ParameterDef(
            param_id=1, name="test",
            domain=ParamDomain.VOICE, dtype=ParamType.BOOL,
            min_value=0, max_value=1,
            default_value=0,
        )
        min_val, max_val = param.type_range()
        assert min_val == 0
        assert max_val == 1


# =============================================================================
# PARAMETER REGISTRY TESTS
# =============================================================================

class TestParameterRegistry:
    """Test parameter registry operations."""

    def test_canonical_parameters_registered(self):
        """Registry should have canonical parameters pre-registered."""
        registry = ParameterRegistry()

        # Check some canonical parameters exist
        assert registry.get(0x0001) is not None  # gain_q15
        assert registry.get_by_name("gain_q15") is not None
        assert registry.get_by_name("activity_q15") is not None
        assert registry.get_by_name("pan_q15") is not None

    def test_register_custom_parameter(self):
        """Should be able to register custom parameters."""
        registry = ParameterRegistry()

        custom = ParameterDef(
            param_id=0xFFFF,
            name="custom_param",
            domain=ParamDomain.VOICE,
            dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=0,
        )
        registry.register(custom)

        assert registry.get(0xFFFF) is not None
        assert registry.get_by_name("custom_param") is not None

    def test_duplicate_id_raises(self):
        """Registering duplicate ID should raise."""
        registry = ParameterRegistry()

        # gain_q15 already has ID 0x0001
        with pytest.raises(ValueError):
            registry.register(ParameterDef(
                param_id=0x0001,
                name="duplicate",
                domain=ParamDomain.VOICE,
                dtype=ParamType.Q15,
                min_value=0, max_value=Q15_MAX,
                default_value=0,
            ))

    def test_duplicate_name_raises(self):
        """Registering duplicate name should raise."""
        registry = ParameterRegistry()

        # gain_q15 already exists
        with pytest.raises(ValueError):
            registry.register(ParameterDef(
                param_id=0xFFFE,
                name="gain_q15",
                domain=ParamDomain.VOICE,
                dtype=ParamType.Q15,
                min_value=0, max_value=Q15_MAX,
                default_value=0,
            ))

    def test_list_by_domain(self):
        """list_by_domain should return only matching parameters."""
        registry = ParameterRegistry()

        voice_params = registry.list_by_domain(ParamDomain.VOICE)
        symbol_params = registry.list_by_domain(ParamDomain.SYMBOL)
        safety_params = registry.list_by_domain(ParamDomain.SAFETY)

        assert len(voice_params) > 0
        assert len(symbol_params) > 0
        assert len(safety_params) > 0

        # Check all returned params are in correct domain
        for param in voice_params:
            assert param.domain == ParamDomain.VOICE

    def test_get_nonexistent(self):
        """Getting nonexistent parameter should return None."""
        registry = ParameterRegistry()
        assert registry.get(0x9999) is None
        assert registry.get_by_name("nonexistent") is None


# =============================================================================
# PARAMETER BANK TESTS
# =============================================================================

class TestParameterBank:
    """Test parameter bank state management."""

    def test_initialized_with_defaults(self):
        """Bank should be initialized with default values."""
        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        # gain_q15 default is Q15_ONE
        assert bank.get_by_name("gain_q15") == Q15_ONE

        # regime_id default is 2 (REG_STABLE)
        assert bank.get_by_name("regime_id") == 2

    def test_abs_update_sets_value(self):
        """ABS update should set value directly."""
        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        param = registry.get_by_name("gain_q15")
        result = bank.apply_abs(param.param_id, 16000)

        assert result == 16000
        assert bank.get_by_name("gain_q15") == 16000

    def test_abs_update_clamps(self):
        """ABS update should clamp to parameter range."""
        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        param = registry.get_by_name("gain_q15")  # min=0, max=Q15_MAX

        # Below min
        result = bank.apply_abs(param.param_id, -1000)
        assert result == 0

        # Above max
        result = bank.apply_abs(param.param_id, 50000)
        assert result == Q15_MAX

    def test_delta_update_adds(self):
        """DELTA update should add to current value."""
        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        param = registry.get_by_name("pan_q15")

        # Start at 0
        bank.apply_abs(param.param_id, 0)

        # Add delta
        result = bank.apply_delta(param.param_id, 1000)
        assert result == 1000

        # Add more
        result = bank.apply_delta(param.param_id, 500)
        assert result == 1500

    def test_delta_update_clamps(self):
        """DELTA update should clamp to range."""
        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        param = registry.get_by_name("gain_q15")  # min=0, max=Q15_MAX

        bank.apply_abs(param.param_id, Q15_MAX - 100)

        # Delta that would exceed max
        result = bank.apply_delta(param.param_id, 500)
        assert result == Q15_MAX

    def test_delta_ignored_on_uninitialized(self):
        """DELTA on uninitialized parameter should be ignored."""
        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        # Create a new parameter that might be uninitialized
        custom = ParameterDef(
            param_id=0xFFF0,
            name="custom_uninit",
            domain=ParamDomain.VOICE,
            dtype=ParamType.Q15,
            min_value=0, max_value=Q15_MAX,
            default_value=0,
        )
        registry.register(custom)

        # Bank was created before custom param, so it wasn't initialized
        # Actually, since we initialize defaults in constructor, let's test
        # the semantic by marking as uninitialized manually
        bank._state[0xFFF0] = ParameterState(value=0, initialized=False)

        result = bank.apply_delta(0xFFF0, 100)
        assert result is None  # Should be ignored

        # After ABS, DELTA should work
        bank.apply_abs(0xFFF0, 50)
        result = bank.apply_delta(0xFFF0, 100)
        assert result == 150

    def test_reset_to_default(self):
        """reset_to_default should restore default value."""
        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        param = registry.get_by_name("gain_q15")

        # Change value
        bank.apply_abs(param.param_id, 5000)
        assert bank.get_by_name("gain_q15") == 5000

        # Reset
        bank.reset_to_default(param.param_id)
        assert bank.get_by_name("gain_q15") == Q15_ONE

    def test_reset_all_defaults(self):
        """reset_all_defaults should restore all parameters."""
        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        # Change several values
        bank.apply_abs(registry.get_by_name("gain_q15").param_id, 5000)
        bank.apply_abs(registry.get_by_name("pan_q15").param_id, 10000)

        # Reset all
        bank.reset_all_defaults()

        assert bank.get_by_name("gain_q15") == Q15_ONE
        assert bank.get_by_name("pan_q15") == 0

    def test_to_dict(self):
        """to_dict should export state as name->value dict."""
        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        bank.apply_abs(registry.get_by_name("gain_q15").param_id, 12345)

        state_dict = bank.to_dict()

        assert "gain_q15" in state_dict
        assert state_dict["gain_q15"] == 12345


# =============================================================================
# FIXED-POINT ARITHMETIC TESTS
# =============================================================================

class TestFixedPointArithmetic:
    """Test fixed-point arithmetic per Closure Appendix B."""

    def test_q15_mul_basic(self):
        """q15_mul should multiply with proper scaling."""
        # 0.5 * 0.5 = 0.25
        half = Q15_SCALE // 2  # 16384
        result = q15_mul(half, half)
        # Expected: ~8192 (0.25 in q15)
        assert abs(result - 8192) < 2

    def test_q15_mul_one(self):
        """Multiplying by ~1.0 should preserve value."""
        value = 10000
        one = Q15_ONE  # ~1.0
        result = q15_mul(value, one)
        # Should be close to original (slight loss due to q15_one < 1.0)
        assert abs(result - value) < 10

    def test_q15_mul_negative(self):
        """q15_mul should handle negative values."""
        result = q15_mul(16384, -16384)  # 0.5 * -0.5 = -0.25
        assert result < 0
        assert abs(result + 8192) < 2

    def test_q15_mul_saturation(self):
        """q15_mul should saturate on overflow."""
        result = q15_mul(Q15_MAX, Q15_MAX)
        assert result <= Q15_MAX

    def test_q15_add_basic(self):
        """q15_add should add with saturation."""
        assert q15_add(100, 200) == 300
        assert q15_add(-100, -200) == -300

    def test_q15_add_saturation(self):
        """q15_add should saturate."""
        assert q15_add(Q15_MAX, 100) == Q15_MAX
        assert q15_add(Q15_MIN, -100) == Q15_MIN

    def test_q15_sub_basic(self):
        """q15_sub should subtract with saturation."""
        assert q15_sub(300, 100) == 200
        assert q15_sub(-100, 100) == -200

    def test_q15_sub_saturation(self):
        """q15_sub should saturate."""
        assert q15_sub(Q15_MIN, 100) == Q15_MIN
        assert q15_sub(Q15_MAX, -100) == Q15_MAX

    def test_q15_from_float(self):
        """q15_from_float should convert correctly."""
        assert q15_from_float(0.0) == 0
        assert abs(q15_from_float(0.5) - 16384) < 2
        assert abs(q15_from_float(-0.5) - (-16384)) < 2
        assert q15_from_float(1.0) == Q15_MAX  # Clamped
        assert q15_from_float(-1.0) == Q15_MIN  # Clamped

    def test_q15_to_float(self):
        """q15_to_float should convert correctly."""
        assert q15_to_float(0) == 0.0
        assert abs(q15_to_float(16384) - 0.5) < 0.001
        assert abs(q15_to_float(-16384) - (-0.5)) < 0.001

    def test_q8_from_float(self):
        """q8_from_float should convert correctly."""
        assert q8_from_float(0.0) == 0
        assert abs(q8_from_float(0.5) - 64) < 2
        assert q8_from_float(1.0) == Q8_MAX  # Clamped

    def test_q8_to_float(self):
        """q8_to_float should convert correctly."""
        assert q8_to_float(0) == 0.0
        assert abs(q8_to_float(64) - 0.5) < 0.01


# =============================================================================
# CANONICAL DEFAULTS TESTS
# =============================================================================

class TestCanonicalDefaults:
    """Test canonical defaults per Closure Appendix C.2."""

    def test_canonical_defaults_exist(self):
        """Canonical defaults should have required fields."""
        assert "regime_id" in CANONICAL_DEFAULTS
        assert "gear_strength" in CANONICAL_DEFAULTS
        assert "gain" in CANONICAL_DEFAULTS
        assert "tree_id" in CANONICAL_DEFAULTS
        assert "tree_blend" in CANONICAL_DEFAULTS
        assert "event_mask" in CANONICAL_DEFAULTS

    def test_regime_default_is_stable(self):
        """Default regime should be REG_STABLE (2)."""
        assert CANONICAL_DEFAULTS["regime_id"] == 2

    def test_gain_default_is_one(self):
        """Default gain should be 1.0 (Q15_ONE)."""
        assert CANONICAL_DEFAULTS["gain"] == Q15_ONE


# =============================================================================
# GLOBAL REGISTRY TESTS
# =============================================================================

class TestGlobalRegistry:
    """Test global registry singleton."""

    def test_get_registry_returns_same_instance(self):
        """get_registry should return same instance."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_global_registry_has_params(self):
        """Global registry should have canonical params."""
        registry = get_registry()
        assert registry.get_by_name("gain_q15") is not None
