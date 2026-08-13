"""
Test FM6 YAML Pipeline Integration
===================================

Verifies that FM6 parameters flow correctly through:
1. instrument_ir_v0 YAML -> InstrumentIR
2. InstrumentIR -> CapsuleCompiler -> Voice param defaults
3. PresetCompiler fm6 section -> CompiledPreset -> VoiceFrame

Per WSE-STD-FM-TOPOLOGY-v0 and WSE-STD-PRESET v0.
"""

import pytest
from pathlib import Path

# Test imports
from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.preset_compiler import PresetCompiler, FM6Config
from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter
from symcrash_python_ap.voice.voice_engine_registry import VoiceParamID


# Find the presets directory
PRESET_DIR = Path(__file__).parent.parent / "presets" / "instruments"


class TestInstrumentIRFM6Extraction:
    """Test FM6 parameter extraction from instrument_ir_v0 YAML."""

    @pytest.fixture
    def brass_stab_ir(self):
        """Load brass_stab_v0.yaml as InstrumentIR."""
        yaml_path = PRESET_DIR / "brass_stab_v0.yaml"
        if not yaml_path.exists():
            pytest.skip(f"Preset not found: {yaml_path}")
        return load_instrument(str(yaml_path))

    @pytest.fixture
    def jump_brass_ir(self):
        """Load jump_brass_v0.yaml as InstrumentIR."""
        yaml_path = PRESET_DIR / "jump_brass_v0.yaml"
        if not yaml_path.exists():
            pytest.skip(f"Preset not found: {yaml_path}")
        return load_instrument(str(yaml_path))

    def test_brass_stab_has_fm6_node(self, brass_stab_ir):
        """Verify brass_stab has an FM6 node in L0_graph."""
        nodes = brass_stab_ir.layers.L0_graph.nodes
        fm6_nodes = [n for n in nodes if n.type == "FM6"]

        assert len(fm6_nodes) >= 1, "brass_stab should have an FM6 node"

    def test_brass_stab_fm6_params(self, brass_stab_ir):
        """Verify FM6 parameters are correctly parsed."""
        nodes = brass_stab_ir.layers.L0_graph.nodes
        fm6_node = next((n for n in nodes if n.type == "FM6"), None)
        assert fm6_node is not None

        params = fm6_node.params

        # Algorithm 22 is classic brass
        assert params.algorithm == 38  # legacy WSE id (old alg 22 topology preserved)

        # Should have 6 operator ratios
        assert params.op_ratios is not None
        assert len(params.op_ratios) == 6

        # Should have 6 operator levels
        assert params.op_levels is not None
        assert len(params.op_levels) == 6

        # Feedback should be set
        assert params.feedback is not None

    def test_jump_brass_detune(self, jump_brass_ir):
        """Verify jump brass has carrier detune for thick sound."""
        nodes = jump_brass_ir.layers.L0_graph.nodes
        fm6_node = next((n for n in nodes if n.type == "FM6"), None)
        assert fm6_node is not None

        ratios = fm6_node.params.op_ratios
        assert ratios is not None

        # Op1 should be anchor at 1.0 (65536)
        assert ratios[0] == 65536

        # Op2 should be detuned up (> 65536)
        assert ratios[1] > 65536

        # Op3 should be detuned down (< 65536)
        assert ratios[2] < 65536


class TestCapsuleCompilerFM6Extraction:
    """Test CapsuleCompiler extraction of FM6 to voice params."""

    @pytest.fixture
    def compiler(self):
        """Create CapsuleCompiler."""
        return CapsuleCompiler()

    @pytest.fixture
    def brass_stab_ir(self):
        """Load brass_stab_v0.yaml as InstrumentIR."""
        yaml_path = PRESET_DIR / "brass_stab_v0.yaml"
        if not yaml_path.exists():
            pytest.skip(f"Preset not found: {yaml_path}")
        return load_instrument(str(yaml_path))

    def test_extracts_osc_mode_fm6(self, compiler, brass_stab_ir):
        """Verify OSC_MODE is set to 3 (FM6)."""
        defaults = compiler._extract_voice_param_defaults_from_ir(brass_stab_ir)

        assert VoiceParamID.OSC_MODE in defaults
        assert defaults[VoiceParamID.OSC_MODE] == 3

    def test_extracts_algorithm(self, compiler, brass_stab_ir):
        """Verify FM6_ALGORITHM is extracted."""
        defaults = compiler._extract_voice_param_defaults_from_ir(brass_stab_ir)

        assert VoiceParamID.FM6_ALGORITHM in defaults
        assert defaults[VoiceParamID.FM6_ALGORITHM] == 38  # legacy WSE id

    def test_extracts_feedback(self, compiler, brass_stab_ir):
        """Verify FM6_FEEDBACK is extracted."""
        defaults = compiler._extract_voice_param_defaults_from_ir(brass_stab_ir)

        assert VoiceParamID.FM6_FEEDBACK in defaults
        assert defaults[VoiceParamID.FM6_FEEDBACK] >= 0
        assert defaults[VoiceParamID.FM6_FEEDBACK] <= 7

    def test_extracts_op_ratios(self, compiler, brass_stab_ir):
        """Verify all 6 operator ratios are extracted."""
        defaults = compiler._extract_voice_param_defaults_from_ir(brass_stab_ir)

        ratio_params = [
            VoiceParamID.FM6_OP0_RATIO,
            VoiceParamID.FM6_OP1_RATIO,
            VoiceParamID.FM6_OP2_RATIO,
            VoiceParamID.FM6_OP3_RATIO,
            VoiceParamID.FM6_OP4_RATIO,
            VoiceParamID.FM6_OP5_RATIO,
        ]

        for param_id in ratio_params:
            assert param_id in defaults, f"Missing {param_id}"
            # Ratios should be positive Q16 values
            assert defaults[param_id] > 0

    def test_extracts_op_levels(self, compiler, brass_stab_ir):
        """Verify all 6 operator levels are extracted."""
        defaults = compiler._extract_voice_param_defaults_from_ir(brass_stab_ir)

        level_params = [
            VoiceParamID.FM6_OP0_LEVEL,
            VoiceParamID.FM6_OP1_LEVEL,
            VoiceParamID.FM6_OP2_LEVEL,
            VoiceParamID.FM6_OP3_LEVEL,
            VoiceParamID.FM6_OP4_LEVEL,
            VoiceParamID.FM6_OP5_LEVEL,
        ]

        for param_id in level_params:
            assert param_id in defaults, f"Missing {param_id}"
            # Levels should be Q15 (0-32767)
            assert 0 <= defaults[param_id] <= 32767


class TestPresetCompilerFM6:
    """Test PresetCompiler fm6 section parsing."""

    def test_parses_fm6_section(self):
        """Verify fm6 section is parsed into FM6Config."""
        preset_dict = {
            'preset': {'preset_id': 'test'},
            'compat': {},
            'fm6': {
                'algorithm_id': 19,
                'feedback_depth': 3,
                'op_ratios': [65536, 66106, 64966, 66486, 65536, 65536],
                'op_levels': [32767, 32767, 32767, 28672, 10922, 8192],
            }
        }

        compiler = PresetCompiler()
        compiled = compiler.compile(preset_dict)

        assert compiled.fm6 is not None
        assert compiled.osc_mode == 3

        assert compiled.fm6.algorithm_id == 19
        assert compiled.fm6.feedback_depth == 3
        assert len(compiled.fm6.op_ratios) == 6
        assert len(compiled.fm6.op_levels) == 6

    def test_sets_osc_mode_when_fm6_present(self):
        """Verify osc_mode is set to 3 when fm6 section exists."""
        preset_dict = {
            'preset': {'preset_id': 'test'},
            'compat': {},
            'fm6': {
                'algorithm': 1  # Alternate key
            }
        }

        compiler = PresetCompiler()
        compiled = compiler.compile(preset_dict)

        assert compiled.osc_mode == 3

    def test_no_fm6_section(self):
        """Verify fm6 is None when section not present."""
        preset_dict = {
            'preset': {'preset_id': 'test'},
            'compat': {},
        }

        compiler = PresetCompiler()
        compiled = compiler.compile(preset_dict)

        assert compiled.fm6 is None
        assert compiled.osc_mode == 0


class TestVoiceFrameEmitterFM6:
    """Test VoiceFrameEmitter population of FM6 fields."""

    def test_emits_fm6_fields(self):
        """Verify VoiceFrame has FM6 fields when preset has fm6 config."""
        preset_dict = {
            'preset': {'preset_id': 'test'},
            'compat': {},
            'latch_defaults': {},
            'fm6': {
                'algorithm_id': 22,
                'feedback_depth': 4,
                'mod_index': 2.0,
                'op_ratios': [65536, 65602, 65470, 65668, 65536, 131072],
                'op_levels': [32767, 24576, 24576, 20480, 20480, 16384],
            }
        }

        compiler = PresetCompiler()
        compiled = compiler.compile(preset_dict)

        emitter = VoiceFrameEmitter(compiled)
        emitter.note_on(60)
        frame = emitter.emit_frame()

        assert frame.coeff_osc_mode == 3
        assert frame.fm_algorithm_id == 22  # passthrough of the inline input above
        assert frame.fm_feedback_depth == 4
        assert frame.fm_mod_index_q16 == 2 * 65536
        assert frame.fm_op_ratios is not None
        assert len(frame.fm_op_ratios) == 6
        assert frame.fm_op_levels is not None
        assert len(frame.fm_op_levels) == 6

    def test_no_fm6_fields_without_config(self):
        """Verify VoiceFrame has None FM6 fields when no fm6 config."""
        preset_dict = {
            'preset': {'preset_id': 'test'},
            'compat': {},
            'latch_defaults': {},
        }

        compiler = PresetCompiler()
        compiled = compiler.compile(preset_dict)

        emitter = VoiceFrameEmitter(compiled)
        emitter.note_on(60)
        frame = emitter.emit_frame()

        assert frame.coeff_osc_mode == 0
        assert frame.fm_algorithm_id is None
        assert frame.fm_op_ratios is None
        assert frame.fm_op_levels is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
