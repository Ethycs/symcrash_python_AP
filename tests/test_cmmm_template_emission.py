"""
Tests for CMMM Template Emission
================================

Verifies that VoiceFrameEmitter correctly emits CMMM fields from preset
to VoiceFrame per WSE-STD-070 §10.

Tests:
1. Default CMMM values when preset has no cmmm section
2. CMMM values from preset.cmmm flow to VoiceFrame
3. Pattern table reference resolution (pattern:N)
4. Tail mode name resolution (FULL, MID, MIN, EMERGENCY)
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Mock the preset structures for testing
@dataclass
class MockMaterial:
    """Mock material params."""
    beta_start_q15: int = 0
    beta_target_q15: int = 24576
    beta_ramp_blocks: int = 10


@dataclass
class MockLatch:
    """Mock latch params."""
    tree_a_id: int = 0
    tree_b_id: int = 0
    tree_blend_q15: int = 0


@dataclass
class MockCmmmDefaults:
    """Mock CMMM defaults."""
    node_mask_u32: int = 0xFFFFFFFF
    mask_ramp_blocks: int = 5
    tail_mode: int = 0
    traction_q15: int = 32767


@dataclass
class MockCompiledPreset:
    """Mock compiled preset for testing."""
    material: MockMaterial = field(default_factory=MockMaterial)
    latch: MockLatch = field(default_factory=MockLatch)
    harmonic_amps_q15: List[int] = field(default_factory=lambda: [32767, 0, 0, 0, 0, 0, 0, 0])
    amp_env: Optional[object] = None
    cmmm: Optional[MockCmmmDefaults] = None
    osc_mode: int = 0
    fm6: Optional[object] = None
    envelope_bindings: Optional[list] = None

    def backbone_ops(self) -> List[str]:
        return ["TILT", "COMPRESS_K"]

    def backbone_families(self) -> Dict[str, float]:
        # Must match the CompiledPreset contract: op name (lowercase) ->
        # strength (0.0-1.0). VoiceEngine.step_voice_simple() calls .values()
        # on this, so returning a list breaks emission.
        return {"tilt": 1.0, "compress_k": 0.5}


class TestCmmmTemplateEmission:
    """Test CMMM template emission from preset to VoiceFrame."""

    def test_default_cmmm_values_when_no_cmmm_section(self):
        """When preset has no cmmm section, VoiceFrame should have defaults."""
        # Import the real emitter
        from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter

        # Create preset without cmmm
        preset = MockCompiledPreset(cmmm=None)

        emitter = VoiceFrameEmitter(preset=preset)
        emitter.note_on(60, 32767)

        frame = emitter.emit_frame()

        # Default values when no cmmm
        assert frame.cmmm_node_mask is None
        assert frame.cmmm_mask_ramp_blocks == 5
        assert frame.cmmm_tail_mode is None
        assert frame.cmmm_traction_q15 == 32767

    def test_cmmm_values_flow_to_voiceframe(self):
        """CMMM values from preset should flow to VoiceFrame."""
        from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter

        # Create preset with custom CMMM
        cmmm = MockCmmmDefaults(
            node_mask_u32=0x0000FFFF,  # K=16
            mask_ramp_blocks=10,
            tail_mode=1,  # MID
            traction_q15=16384,
        )
        preset = MockCompiledPreset(cmmm=cmmm)

        emitter = VoiceFrameEmitter(preset=preset)
        emitter.note_on(60, 32767)

        frame = emitter.emit_frame()

        assert frame.cmmm_node_mask == 0x0000FFFF
        assert frame.cmmm_mask_ramp_blocks == 10
        assert frame.cmmm_tail_mode == 1
        assert frame.cmmm_traction_q15 == 16384

    def test_pattern_table_resolution(self):
        """Pattern table references should be resolved correctly."""
        from symcrash_python_ap.library.preset_compiler import CmmmDefaults

        # Pattern table (WSE-STD-DEGRADE-v0 §3.1 frozen):
        # 0=K32, 1=K24, 2=K16, 3=K12, 4=K8, 5=K4, 6=K1
        test_cases = [
            ("pattern:0", 0xFFFFFFFF),  # K=32
            ("pattern:2", 0x0000FFFF),  # K=16
            ("pattern:4", 0x000000FF),  # K=8
            ("pattern:6", 0x00000001),  # K=1
        ]

        for pattern_str, expected_mask in test_cases:
            cmmm = CmmmDefaults.from_dict({'node_mask': pattern_str})
            assert cmmm.node_mask_u32 == expected_mask, f"Failed for {pattern_str}"

    def test_tail_mode_name_resolution(self):
        """Tail mode names should be resolved to integers."""
        from symcrash_python_ap.library.preset_compiler import CmmmDefaults

        test_cases = [
            ("FULL", 0),
            ("MID", 1),
            ("MIN", 2),
            ("EMERGENCY", 3),
        ]

        for mode_name, expected_int in test_cases:
            cmmm = CmmmDefaults.from_dict({'tail_mode': mode_name})
            assert cmmm.tail_mode == expected_int, f"Failed for {mode_name}"

    def test_cmmm_persists_across_blocks(self):
        """CMMM values should persist across multiple blocks."""
        from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter

        cmmm = MockCmmmDefaults(
            node_mask_u32=0x00000FFF,  # K=12
            mask_ramp_blocks=3,
            tail_mode=2,  # MIN
            traction_q15=8192,
        )
        preset = MockCompiledPreset(cmmm=cmmm)

        emitter = VoiceFrameEmitter(preset=preset)
        emitter.note_on(60, 32767)

        # Emit multiple frames
        for i in range(5):
            frame = emitter.emit_frame()
            assert frame.cmmm_node_mask == 0x00000FFF
            assert frame.cmmm_mask_ramp_blocks == 3
            assert frame.cmmm_tail_mode == 2
            assert frame.cmmm_traction_q15 == 8192

    def test_hex_string_node_mask(self):
        """Hex string node masks should be parsed correctly."""
        from symcrash_python_ap.library.preset_compiler import CmmmDefaults

        # Test hex string parsing
        cmmm = CmmmDefaults.from_dict({'node_mask': '0x00FF00FF'})
        assert cmmm.node_mask_u32 == 0x00FF00FF

    def test_integer_node_mask(self):
        """Integer node masks should pass through directly."""
        from symcrash_python_ap.library.preset_compiler import CmmmDefaults

        cmmm = CmmmDefaults.from_dict({'node_mask': 0xABCDEF01})
        assert cmmm.node_mask_u32 == 0xABCDEF01

    def test_cmmm_defaults_from_none(self):
        """CmmmDefaults.from_dict(None) should return defaults."""
        from symcrash_python_ap.library.preset_compiler import CmmmDefaults

        cmmm = CmmmDefaults.from_dict(None)
        assert cmmm.node_mask_u32 == 0xFFFFFFFF
        assert cmmm.mask_ramp_blocks == 5
        assert cmmm.tail_mode == 0
        assert cmmm.traction_q15 == 32767


class TestCmmmDefaultsFromDict:
    """Test CmmmDefaults.from_dict parsing."""

    def test_full_dict(self):
        """All fields should be parsed from dict."""
        from symcrash_python_ap.library.preset_compiler import CmmmDefaults

        d = {
            'node_mask': 0x0000FFFF,
            'mask_ramp_blocks': 7,
            'tail_mode': 'MID',
            'traction_q15': 20000,
        }

        cmmm = CmmmDefaults.from_dict(d)

        assert cmmm.node_mask_u32 == 0x0000FFFF
        assert cmmm.mask_ramp_blocks == 7
        assert cmmm.tail_mode == 1
        assert cmmm.traction_q15 == 20000

    def test_partial_dict_uses_defaults(self):
        """Missing fields should use defaults."""
        from symcrash_python_ap.library.preset_compiler import CmmmDefaults

        d = {'mask_ramp_blocks': 12}

        cmmm = CmmmDefaults.from_dict(d)

        assert cmmm.node_mask_u32 == 0xFFFFFFFF  # default
        assert cmmm.mask_ramp_blocks == 12       # from dict
        assert cmmm.tail_mode == 0               # default
        assert cmmm.traction_q15 == 32767        # default


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
