"""
Unit tests for Preset Compiler and related modules.

Tests:
- EnvelopeScheduler ADSR generation
- PresetCompiler YAML/JSON parsing
- VoiceFrameEmitter frame generation
- MacroRuntime slew filters
"""

import pytest
import numpy as np
from pathlib import Path

from symcrash_python_ap.envelope_scheduler import (
    EnvelopeScheduler, ADSRSpec, EnvelopePhase, ms_to_blocks,
    BLOCKS_PER_SECOND, Q15_ONE
)
from symcrash_python_ap.library.preset_compiler import (
    PresetCompiler, CompiledPreset, BackboneOp, PresetCompileError
)
from symcrash_python_ap.voiceframe_emitter import (
    VoiceFrameEmitter, midi_to_hz
)
from symcrash_python_ap.macro_runtime import (
    MacroRuntime, SystemMacro, MACROMAP_LITE_V0, clamp_q15
)


# =============================================================================
# ENVELOPE SCHEDULER TESTS
# =============================================================================

class TestEnvelopeScheduler:
    """Tests for EnvelopeScheduler."""

    def test_envelope_starts_idle(self):
        """Envelope starts in IDLE phase."""
        env = EnvelopeScheduler(spec=ADSRSpec())
        assert env.phase == EnvelopePhase.IDLE
        assert env.tick() == 0

    def test_trigger_starts_attack(self):
        """trigger_note() starts attack phase."""
        env = EnvelopeScheduler(spec=ADSRSpec(attack_ms=10.0))
        env.trigger_note()
        assert env.phase == EnvelopePhase.ATTACK

    def test_attack_rises_to_max(self):
        """Attack phase rises to Q15_ONE."""
        # 10ms at 750 blocks/sec = ~7 blocks
        env = EnvelopeScheduler(spec=ADSRSpec(attack_ms=10.0))
        env.trigger_note()

        # Tick through attack phase (need enough blocks)
        gains = []
        for _ in range(15):  # More than attack_blocks
            gains.append(env.tick())

        # Should have rising gains
        assert gains[0] < gains[-1]
        # Final gain should be at or near max (after attack completes)
        # Attack is ~7 blocks, so by block 15 we should be past it
        assert gains[-1] >= Q15_ONE - 2000, f"Expected near max, got {gains[-1]}"

    def test_decay_falls_to_sustain(self):
        """Decay phase falls to sustain level."""
        env = EnvelopeScheduler(spec=ADSRSpec(
            attack_ms=1.0,
            decay_ms=100.0,
            sustain=0.5
        ))
        env.trigger_note()

        # Tick through attack
        for _ in range(10):
            env.tick()

        # Should be in decay or sustain
        assert env.phase in (EnvelopePhase.DECAY, EnvelopePhase.SUSTAIN)

        # Tick more
        for _ in range(100):
            env.tick()

        # Should be at sustain
        assert env.phase == EnvelopePhase.SUSTAIN
        assert abs(env.current_q15 - int(0.5 * Q15_ONE)) < 500

    def test_release_falls_to_zero(self):
        """Release phase falls to zero."""
        env = EnvelopeScheduler(spec=ADSRSpec(
            attack_ms=1.0,
            decay_ms=10.0,
            sustain=0.5,
            release_ms=50.0
        ))
        env.trigger_note()

        # Get to sustain
        for _ in range(50):
            env.tick()

        # Release
        env.release_note()
        assert env.phase == EnvelopePhase.RELEASE

        # Tick through release
        for _ in range(100):
            env.tick()

        # Should be idle with zero output
        assert env.phase == EnvelopePhase.IDLE
        assert env.tick() == 0

    def test_ms_to_blocks(self):
        """ms_to_blocks converts correctly."""
        # At 750 blocks/sec:
        # 100ms = 75 blocks
        blocks = ms_to_blocks(100.0)
        assert blocks == 75

        # Minimum is 1 block
        assert ms_to_blocks(0.1) >= 1


# =============================================================================
# PRESET COMPILER TESTS
# =============================================================================

class TestPresetCompiler:
    """Tests for PresetCompiler."""

    def test_compile_minimal_preset(self):
        """Compile a minimal preset dict."""
        compiler = PresetCompiler()
        preset_dict = {
            'preset': {
                'preset_id': 'TEST',
                'preset_version': 'v0',
            },
            'compat': {
                'voiceframe_std': 'WSE-STD-020 v1',
                'vm_std': 'WSE-STD-VM-HYBRID v0',
            },
            'backbone': [
                {'op': 'TILT', 'strength_q15': 16384},
            ],
        }

        compiled = compiler.compile(preset_dict)

        assert compiled.preset_id == 'TEST'
        assert len(compiled.backbone) == 1
        assert compiled.backbone[0].op == 'TILT'
        assert compiled.backbone[0].strength_q15 == 16384

    def test_compile_with_envelope(self):
        """Compile preset with envelope specification."""
        compiler = PresetCompiler()
        preset_dict = {
            'preset': {'preset_id': 'TEST'},
            'env': {
                'amp': {
                    'attack_ms': 5.0,
                    'decay_ms': 100.0,
                    'sustain': 0.6,
                    'release_ms': 80.0,
                }
            },
        }

        compiled = compiler.compile(preset_dict)

        assert compiled.amp_env is not None
        assert compiled.amp_env.attack_ms == 5.0
        assert compiled.amp_env.sustain == 0.6

    def test_validate_detects_errors(self):
        """Validation catches missing fields."""
        compiler = PresetCompiler()

        # Missing preset section
        errors = compiler.validate({})
        assert len(errors) > 0
        assert any('preset' in e.lower() for e in errors)

    def test_backbone_ops_returns_list(self):
        """backbone_ops() returns list of op names."""
        compiler = PresetCompiler()
        preset_dict = {
            'preset': {'preset_id': 'TEST'},
            'backbone': [
                {'op': 'TILT'},
                {'op': 'DIFFUSE_DOWN'},
            ],
        }

        compiled = compiler.compile(preset_dict)
        ops = compiled.backbone_ops()

        assert ops == ['TILT', 'DIFFUSE_DOWN']


# =============================================================================
# VOICEFRAME EMITTER TESTS
# =============================================================================

class TestVoiceFrameEmitter:
    """Tests for VoiceFrameEmitter."""

    def test_midi_to_hz_a4(self):
        """MIDI note 69 = 440 Hz."""
        hz = midi_to_hz(69)
        assert abs(hz - 440.0) < 0.01

    def test_midi_to_hz_octaves(self):
        """Octaves double frequency."""
        a4 = midi_to_hz(69)
        a5 = midi_to_hz(81)  # +12 semitones
        assert abs(a5 / a4 - 2.0) < 0.01

    def test_emit_frame_produces_voiceframe(self):
        """emit_frame() produces VoiceFrame with correct fields."""
        compiler = PresetCompiler()
        preset_dict = {
            'preset': {'preset_id': 'TEST'},
            'env': {'amp': {'attack_ms': 5.0}},
            'backbone': [{'op': 'TILT'}],
        }
        compiled = compiler.compile(preset_dict)

        emitter = VoiceFrameEmitter(preset=compiled)
        emitter.note_on(note=60, velocity_q15=32767)

        frame = emitter.emit_frame()

        assert frame.note == 60
        assert frame.f0_hz > 0
        assert 'tilt' in [op.lower() for op in frame.ops_enable]

    def test_envelope_controls_gain(self):
        """Envelope controls frame gain_q15."""
        compiler = PresetCompiler()
        preset_dict = {
            'preset': {'preset_id': 'TEST'},
            'env': {
                'amp': {
                    'attack_ms': 1.0,
                    'decay_ms': 10.0,
                    'sustain': 0.5,
                    'release_ms': 10.0,
                }
            },
        }
        compiled = compiler.compile(preset_dict)

        emitter = VoiceFrameEmitter(preset=compiled)
        emitter.note_on(note=60, velocity_q15=32767)

        # First frame should have lower gain (attack starting)
        first_frame = emitter.emit_frame()

        # Get to sustain
        for _ in range(50):
            emitter.emit_frame()

        # Release
        emitter.note_off()

        # Tick release
        for _ in range(50):
            frame = emitter.emit_frame()

        # After release, should be very quiet
        assert frame.gain_q15 < 1000

    def test_eta_override(self):
        """set_eta() overrides preset beta ramp."""
        compiler = PresetCompiler()
        preset_dict = {
            'preset': {'preset_id': 'TEST'},
            'material': {
                'beta_start_q15': 0,
                'beta_target_q15': 32767,
                'beta_ramp_blocks': 100,
            },
        }
        compiled = compiler.compile(preset_dict)

        emitter = VoiceFrameEmitter(preset=compiled)
        emitter.note_on(note=60, velocity_q15=32767)

        # Without override, first frame should have low eta (from ramp start)
        frame1 = emitter.emit_frame()
        assert frame1.beta_target_q15 < 5000  # Near start of ramp

        # Set override
        emitter.set_eta(16384)  # 0.5

        # With override, eta should be exactly the override value
        frame2 = emitter.emit_frame()
        assert frame2.beta_target_q15 == 16384

        # Clear override
        emitter.set_eta(None)

        # Should return to ramp (now at block 2)
        frame3 = emitter.emit_frame()
        assert frame3.beta_target_q15 != 16384  # Should be ramp value

    def test_detune_changes_frequency(self):
        """set_detune() shifts frequency."""
        compiler = PresetCompiler()
        preset_dict = {'preset': {'preset_id': 'TEST'}}
        compiled = compiler.compile(preset_dict)

        emitter = VoiceFrameEmitter(preset=compiled)
        emitter.note_on(note=69, velocity_q15=32767)  # A4 = 440 Hz

        # Without detune
        frame1 = emitter.emit_frame()
        assert abs(frame1.f0_hz - 440.0) < 0.1

        # Detune by +1200 cents (1 octave up)
        emitter.set_detune(1200)
        frame2 = emitter.emit_frame()
        assert abs(frame2.f0_hz - 880.0) < 0.1  # Should be 880 Hz

        # Detune by -1200 cents (1 octave down)
        emitter.set_detune(-1200)
        frame3 = emitter.emit_frame()
        assert abs(frame3.f0_hz - 220.0) < 0.1  # Should be 220 Hz

    def test_detune_in_cents_reflects_in_frame(self):
        """Frame cents field includes detune."""
        compiler = PresetCompiler()
        preset_dict = {'preset': {'preset_id': 'TEST'}}
        compiled = compiler.compile(preset_dict)

        emitter = VoiceFrameEmitter(preset=compiled)
        emitter.note_on(note=60, velocity_q15=32767, cents=50)  # +50 cents

        # Add detune
        emitter.set_detune(100)  # +100 cents

        frame = emitter.emit_frame()
        assert frame.cents == 150  # 50 + 100

    def test_reset_clears_overrides(self):
        """reset() clears eta override and detune."""
        compiler = PresetCompiler()
        preset_dict = {'preset': {'preset_id': 'TEST'}}
        compiled = compiler.compile(preset_dict)

        emitter = VoiceFrameEmitter(preset=compiled)
        emitter.note_on(note=60, velocity_q15=32767)

        # Set overrides
        emitter.set_eta(16384)
        emitter.set_detune(1200)

        # Verify they're set
        assert emitter.get_eta() == 16384
        assert emitter.get_detune() == 1200

        # Reset
        emitter.reset()

        # Overrides should be cleared
        assert emitter._eta_override_q15 is None
        assert emitter._detune_cents == 0


# =============================================================================
# MACRO RUNTIME TESTS
# =============================================================================

class TestMacroRuntime:
    """Tests for MacroRuntime."""

    def test_set_preset_macro(self):
        """set_macro updates target for mapped system macro."""
        runtime = MacroRuntime()
        runtime.set_macro('BRIGHT', 24576)

        # Affine mapping: (a * in) >> 15 = (32767 * 24576) >> 15 = 24575
        # (off-by-one due to Q15 rounding)
        assert abs(runtime.macro_target_q15[SystemMacro.BRIGHTNESS] - 24576) <= 1

    def test_slew_one_pole(self):
        """1-pole slew smoothly approaches target."""
        runtime = MacroRuntime()
        runtime.set_macro('BRIGHT', 32767)

        values = []
        for _ in range(100):
            runtime.tick()
            values.append(runtime.get_system_macro(SystemMacro.BRIGHTNESS))

        # Should approach target
        assert values[-1] > values[0]
        assert values[-1] > 30000  # Should be close to target

    def test_energy_state_is_protected(self):
        """energy_state cannot be set by controllers."""
        runtime = MacroRuntime()
        runtime.set_system_macro(SystemMacro.ENERGY_STATE, 32767)

        # Should not have changed
        assert runtime.macro_target_q15[SystemMacro.ENERGY_STATE] == 0

    def test_set_energy_state_internal(self):
        """set_energy_state() works for internal use."""
        runtime = MacroRuntime()
        runtime.set_energy_state(16384)

        assert runtime.macro_curr_q15[SystemMacro.ENERGY_STATE] == 16384

    def test_clamp_q15(self):
        """clamp_q15 restricts to valid range."""
        assert clamp_q15(-100) == 0
        assert clamp_q15(50000) == 32767
        assert clamp_q15(16384) == 16384


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestPresetCompilerIntegration:
    """Integration tests for preset compiler pipeline."""

    def test_brass_stab_yaml_compiles(self):
        """brass_stab_v0.yaml compiles successfully."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        compiler = PresetCompiler()
        preset_path = Path(__file__).parent.parent.parent / 'src' / 'symcrash_python_ap' / 'presets' / 'brass_stab_v0.yaml'

        if not preset_path.exists():
            pytest.skip(f"Preset file not found: {preset_path}")

        compiled = compiler.compile_file(str(preset_path))

        assert compiled.preset_id == 'BRASS_STAB_V0'
        assert compiled.amp_env is not None
        assert len(compiled.backbone) >= 3

    def test_emitter_produces_frames_from_preset(self):
        """VoiceFrameEmitter produces valid frames from compiled preset."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        compiler = PresetCompiler()
        preset_path = Path(__file__).parent.parent.parent / 'src' / 'symcrash_python_ap' / 'presets' / 'brass_stab_v0.yaml'

        if not preset_path.exists():
            pytest.skip(f"Preset file not found: {preset_path}")

        compiled = compiler.compile_file(str(preset_path))
        emitter = VoiceFrameEmitter(preset=compiled)
        emitter.note_on(note=57, velocity_q15=32767)  # A3

        frames = [emitter.emit_frame() for _ in range(10)]

        assert len(frames) == 10
        assert all(f.note == 57 for f in frames)
        assert all(len(f.ops_enable) > 0 for f in frames)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
