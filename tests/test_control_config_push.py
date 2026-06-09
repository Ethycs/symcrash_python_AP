"""
Test ControlConfig → MacroRuntime push at preset load.

Verifies that when a preset with character defaults is loaded,
the MacroRuntime receives the correct initial values.
"""

import pytest
from symcrash_python_ap.instrument_ir import ControlConfig
from symcrash_python_ap.primitives import SystemMacro


class TestControlConfigPush:
    """Test that ControlConfig from presets reaches MacroRuntime."""

    def test_control_config_to_control_program(self):
        """ControlConfig.to_control_program() produces correct Q15 values."""
        config = ControlConfig(brightness=0.9, density=0.5, energy=0.6, risk=0.1)
        program = config.to_control_program()
        assert program.brightness_q15 == int(0.9 * 32767)
        assert program.density_q15 == int(0.5 * 32767)
        assert program.energy_q15 == int(0.6 * 32767)
        assert program.risk_q15 == int(0.1 * 32767)

    def test_macro_runtime_apply_control_defaults(self):
        """apply_control_defaults() sets both target and current values."""
        from symcrash_python_ap.macro_runtime import MacroRuntime

        runtime = MacroRuntime()
        config = ControlConfig(brightness=0.8, density=0.3)
        program = config.to_control_program()
        runtime.apply_control_defaults(program)

        # BRIGHTNESS should reflect the preset's value
        brightness_q15 = runtime.get_system_macro(SystemMacro.BRIGHTNESS)
        expected = int(0.8 * 32767)
        assert brightness_q15 == expected, \
            f"BRIGHTNESS: expected {expected}, got {brightness_q15}"

        # DENSITY should reflect the preset's value
        density_q15 = runtime.get_system_macro(SystemMacro.DENSITY)
        expected_d = int(0.3 * 32767)
        assert density_q15 == expected_d, \
            f"DENSITY: expected {expected_d}, got {density_q15}"

    def test_preset_compiler_parses_character_section(self):
        """PresetCompiler extracts control_config from character: section."""
        from symcrash_python_ap.library.preset_compiler import PresetCompiler

        preset_dict = {
            'preset': {'preset_id': 'TEST', 'preset_version': 'v0'},
            'compat': {
                'voiceframe_std': 'WSE-STD-020 v1',
                'vm_std': 'WSE-STD-VM-HYBRID v0',
                'operator_lib_id': 'WSE-OPLIB-24 v0',
            },
            'latch_defaults': {},
            'env': {'amp': {'attack_ms': 10, 'decay_ms': 100, 'sustain': 0.7, 'release_ms': 100}},
            'material': {},
            'backbone': [],
            'character': {
                'brightness': 0.9,
                'density': 0.4,
                'energy': 0.7,
            },
        }
        compiler = PresetCompiler()
        compiled = compiler.compile(preset_dict)
        assert compiled.control_config is not None
        assert compiled.control_config.brightness == 0.9
        assert compiled.control_config.density == 0.4
        assert compiled.control_config.energy == 0.7

    def test_preset_without_character_has_no_config(self):
        """Presets without character: section have control_config=None."""
        from symcrash_python_ap.library.preset_compiler import PresetCompiler

        preset_dict = {
            'preset': {'preset_id': 'TEST', 'preset_version': 'v0'},
            'compat': {
                'voiceframe_std': 'WSE-STD-020 v1',
                'vm_std': 'WSE-STD-VM-HYBRID v0',
                'operator_lib_id': 'WSE-OPLIB-24 v0',
            },
            'latch_defaults': {},
            'env': {'amp': {'attack_ms': 10, 'decay_ms': 100, 'sustain': 0.7, 'release_ms': 100}},
            'material': {},
            'backbone': [],
        }
        compiler = PresetCompiler()
        compiled = compiler.compile(preset_dict)
        assert compiled.control_config is None

    def test_allocate_voice_pushes_control_defaults(self):
        """allocate_voice_from_preset() pushes ControlConfig to MacroRuntime."""
        from symcrash_python_ap.voice.voice_engine import VoiceEngine
        from symcrash_python_ap.macro_runtime import MacroRuntime
        from symcrash_python_ap.library.preset_compiler import PresetCompiler

        # Build a preset with character defaults
        preset_dict = {
            'preset': {'preset_id': 'TEST', 'preset_version': 'v0'},
            'compat': {
                'voiceframe_std': 'WSE-STD-020 v1',
                'vm_std': 'WSE-STD-VM-HYBRID v0',
                'operator_lib_id': 'WSE-OPLIB-24 v0',
            },
            'latch_defaults': {},
            'env': {'amp': {'attack_ms': 10, 'decay_ms': 100, 'sustain': 0.7, 'release_ms': 100}},
            'material': {},
            'backbone': [],
            'character': {
                'brightness': 0.75,
            },
        }
        compiler = PresetCompiler()
        compiled = compiler.compile(preset_dict)

        # Create engine with MacroRuntime
        runtime = MacroRuntime()
        engine = VoiceEngine(max_voices=4, macro_runtime=runtime)

        # Allocate voice — this should push ControlConfig to MacroRuntime
        voice_id = engine.allocate_voice_from_preset(compiled)
        assert voice_id is not None

        # Verify MacroRuntime has the preset's brightness value
        brightness_q15 = runtime.get_system_macro(SystemMacro.BRIGHTNESS)
        expected = int(0.75 * 32767)
        assert brightness_q15 == expected, \
            f"BRIGHTNESS after allocation: expected {expected}, got {brightness_q15}"
