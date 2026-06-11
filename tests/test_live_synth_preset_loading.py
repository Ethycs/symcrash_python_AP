"""
Live Synth Preset Loading Tests
===============================

Covers the live-synth revival (packet E1):

1. PresetCompiler accepts instrument_ir_v0 YAML (the only preset format left
   in the repo) by routing through load_instrument -> compiled_preset_from_ir.
2. CompiledPreset.backbone_families() contract: dict of op name -> strength
   (VoiceEngine.step_voice_simple() calls .values() on it).
3. Pico2LiveSynth constructs with an instrument_ir_v0 preset in virtual mode
   and _render_block() produces audio.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
MARIMBA_YAML = REPO_ROOT / "presets" / "instruments" / "dx_marimba_v0.yaml"

# tools/synth is not a package; make pico2_live_synth importable
_tools_synth = str(REPO_ROOT / "tools" / "synth")
if _tools_synth not in sys.path:
    sys.path.insert(0, _tools_synth)


class TestPresetCompilerInstrumentIR:
    """PresetCompiler must accept instrument_ir_v0 format."""

    def test_compile_file_instrument_ir_v0(self):
        from symcrash_python_ap.library.preset_compiler import (
            CompiledPreset, PresetCompiler,
        )

        compiler = PresetCompiler()
        preset = compiler.compile_file(str(MARIMBA_YAML))

        assert isinstance(preset, CompiledPreset)
        assert preset.preset_id == "dx-marimba-v0"

        # FM6 section extracted -> FM6 oscillator mode
        assert preset.osc_mode == 3
        assert preset.fm6 is not None
        assert preset.fm6.algorithm_id == 5
        assert len(preset.fm6.op_ratios) == 6
        assert len(preset.fm6.op_levels) == 6

        # Amp envelope extracted from the Amp node
        assert preset.amp_env is not None
        assert preset.amp_env.decay_ms == 500.0

        # Matrix envelope bindings carried over (mod_env -> FM_OP2/FM_OP4)
        assert preset.envelope_bindings
        assert preset.envelope_bindings[0].targets == ["FM_OP2", "FM_OP4"]

    def test_backbone_families_returns_dict(self):
        """VoiceEngine.step_voice_simple() calls .values() on this."""
        from symcrash_python_ap.library.preset_compiler import PresetCompiler

        compiler = PresetCompiler()
        preset = compiler.compile_file(str(MARIMBA_YAML))

        families = preset.backbone_families()
        assert isinstance(families, dict)

    def test_compiled_preset_from_ir_direct(self):
        from symcrash_python_ap.instrument_ir import load_instrument
        from symcrash_python_ap.library.preset_compiler import (
            CompiledPreset, compiled_preset_from_ir,
        )

        ir = load_instrument(str(MARIMBA_YAML))
        preset = compiled_preset_from_ir(ir)
        assert isinstance(preset, CompiledPreset)
        assert preset.osc_mode == 3

        # Material beta ramp from compile.material (blend/target/ramp -> beta)
        assert preset.material.beta_target_q15 == 0
        assert preset.material.beta_ramp_blocks == 1


class TestPico2LiveSynthIRPreset:
    """Acceptance: live synth loads instrument_ir_v0 presets and renders."""

    def test_construct_and_render_with_ir_preset(self):
        import numpy as np
        from pico2_live_synth import Pico2LiveSynth

        synth = Pico2LiveSynth(
            virtual_mode=True,
            preset_path=str(MARIMBA_YAML),
        )
        synth.note_on(60, 100)

        for _ in range(10):
            synth._render_block()

        assert synth.blocks_rendered == 10

        # Skip the first blocks (attack) then verify non-silent audio
        for _ in range(6):
            left, right = synth.audio_queue.get_nowait()
        rms = float(np.sqrt(np.mean(np.asarray(left, dtype=np.float64) ** 2)))
        assert rms > 100.0, f"Expected audible output, got RMS {rms}"

    def test_construct_and_render_default_preset(self):
        """The built-in _MinimalPreset path must still work."""
        from pico2_live_synth import Pico2LiveSynth

        synth = Pico2LiveSynth(virtual_mode=True)
        synth.note_on(60, 100)

        for _ in range(5):
            synth._render_block()

        assert synth.blocks_rendered == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
