"""
SynthMaster smoke coverage (Phase 0 of the synthesis-path dedup).

SynthMaster previously had zero test coverage. These tests pin the
observable contract before the CompiledPreset rewiring (plan step 9):
note_on -> render produces nonzero, deterministic, correctly-sized audio,
for both the no-preset fallback and the YAML preset path.
"""

from pathlib import Path

import pytest

from symcrash_python_ap.synth_master import SynthMaster, SynthConfig

PROJECT_ROOT = Path(__file__).parent.parent
SAW_LEAD_YAML = PROJECT_ROOT / "presets" / "instruments" / "saw_lead_v0.yaml"
FM_BELL_YAML = PROJECT_ROOT / "presets" / "instruments" / "fm_bell_v0.yaml"

RENDER_SEC = 0.25


def _render(synth: SynthMaster, note: int = 60) -> tuple:
    synth.note_on(note, velocity=100)
    L, R = synth.render(RENDER_SEC)
    return L, R


def _expected_samples(synth: SynthMaster) -> int:
    blocks_per_sec = synth.config.sample_rate // synth.config.block_size
    return int(RENDER_SEC * blocks_per_sec) * synth.config.block_size


class TestSynthMasterSmoke:
    def test_note_on_render_nonzero(self):
        synth = SynthMaster()
        L, R = _render(synth)
        assert len(L) == _expected_samples(synth)
        assert len(R) == len(L)
        assert any(s != 0 for s in L), "render produced silence"

    def test_render_deterministic(self):
        a = _render(SynthMaster())
        b = _render(SynthMaster())
        assert a == b, "two fresh SynthMaster renders differ"

    def test_note_on_returns_voice_and_note_off_releases(self):
        synth = SynthMaster()
        voice_id = synth.note_on(60, velocity=100)
        assert voice_id is not None
        synth.note_off(note=60)  # must not raise
        L, _ = synth.render(RENDER_SEC)
        assert len(L) == _expected_samples(synth)

    @pytest.mark.skipif(not SAW_LEAD_YAML.exists(), reason="preset missing")
    def test_yaml_preset_render_nonzero(self):
        synth = SynthMaster()
        assert synth.load_preset_from_yaml(str(SAW_LEAD_YAML))
        L, _ = _render(synth)
        assert any(s != 0 for s in L)

    @pytest.mark.skipif(not FM_BELL_YAML.exists(), reason="preset missing")
    def test_fm6_yaml_preset_render_nonzero(self):
        synth = SynthMaster()
        assert synth.load_preset_from_yaml(str(FM_BELL_YAML))
        L, _ = _render(synth)
        assert any(s != 0 for s in L)

    @pytest.mark.skipif(not SAW_LEAD_YAML.exists(), reason="preset missing")
    def test_yaml_preset_render_deterministic(self):
        def once():
            synth = SynthMaster()
            synth.load_preset_from_yaml(str(SAW_LEAD_YAML))
            return _render(synth)

        assert once() == once()
