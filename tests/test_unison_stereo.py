"""
Unison / stereo width tests (Packet E5, handoff 2026-06-11).

Covers the chain: preset YAML `unison:` schema -> NodeParams ->
extract_unison_params -> render harness N-voice detuned/golden-angle
stereo render (tools/render/render_preset.py).

Invariants pinned here:
- Schema parse (nested block + flat fields, clamping).
- Determinism: two unison renders are bit-identical.
- Mono compatibility: no unison authored -> 1-D mono output, bit-exact
  legacy path (ModeBState.phase_offset_q32 default 0 keeps reset at
  phase 0).
- Width: a unison render has L != R (correlation < mono's 1.0, nonzero
  side energy); width=0 collapses pan to center.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent

# tools/render is not a package; make render_preset importable
_tools_render = str(REPO_ROOT / "tools" / "render")
if _tools_render not in sys.path:
    sys.path.insert(0, _tools_render)

from render_preset import render_preset_note  # noqa: E402

from symcrash_python_ap.instrument_ir import (  # noqa: E402
    load_instrument,
    load_instrument_from_dict,
    extract_unison_params,
)
from symcrash_python_ap.vm.harmonic_bank import ModeBState  # noqa: E402

SUPER_SAW_YAML = REPO_ROOT / "presets" / "instruments" / "super_saw_v0.yaml"
MARIMBA_YAML = REPO_ROOT / "presets" / "instruments" / "dx_marimba_v0.yaml"
SAW_LEAD_YAML = REPO_ROOT / "presets" / "instruments" / "saw_lead_v0.yaml"


def _preset_dict(osc_extra: dict) -> dict:
    """Minimal simple-form preset with extra keys on the Osc node."""
    osc = {'type': 'Osc', 'waveform': 'saw', 'output': 'amp1'}
    osc.update(osc_extra)
    return {
        'instrument_ir_v0': {
            'meta': {'id': 'unison-test', 'name': 'U', 'version': 1, 'seed': 7},
            'audio': {'sample_rate': 48000, 'block_size': 64},
            'graph': {
                'osc1': osc,
                'amp1': {'type': 'Amp', 'attack': 2.0, 'decay': 60.0,
                         'sustain': 0.7, 'release': 40.0},
            },
        }
    }


# =============================================================================
# Schema -> IR -> extractor
# =============================================================================

class TestUnisonSchema:
    """`unison:` in preset YAML reaches NodeParams and the extractor."""

    def test_nested_block(self):
        ir = load_instrument_from_dict(_preset_dict(
            {'unison': {'voices': 5, 'spread_cents': 18, 'width': 0.75}}))
        assert extract_unison_params(ir) == {
            'voices': 5, 'spread_cents': 18, 'width': 0.75}

    def test_flat_fields(self):
        ir = load_instrument_from_dict(_preset_dict(
            {'unison_voices': 3, 'unison_spread': 12, 'unison_width': 0.5}))
        assert extract_unison_params(ir) == {
            'voices': 3, 'spread_cents': 12, 'width': 0.5}

    def test_clamping(self):
        ir = load_instrument_from_dict(_preset_dict(
            {'unison': {'voices': 99, 'spread_cents': 30, 'width': 2.5}}))
        params = extract_unison_params(ir)
        assert params['voices'] == 8
        assert params['width'] == 1.0

    def test_super_saw_preset_carries_unison(self):
        ir = load_instrument(str(SUPER_SAW_YAML))
        assert extract_unison_params(ir) == {
            'voices': 7, 'spread_cents': 25, 'width': 1.0}

    def test_absent_unison_extracts_none(self):
        ir = load_instrument_from_dict(_preset_dict({}))
        assert extract_unison_params(ir) is None
        assert extract_unison_params(load_instrument(str(MARIMBA_YAML))) is None

    def test_single_voice_is_mono(self):
        # voices: 1 is not unison — mono path
        ir = load_instrument_from_dict(_preset_dict(
            {'unison': {'voices': 1, 'spread_cents': 25}}))
        assert extract_unison_params(ir) is None


# =============================================================================
# Render: determinism, mono compatibility, stereo width
# =============================================================================

@pytest.fixture(scope="module")
def unison_render():
    ir = load_instrument(str(SUPER_SAW_YAML))
    return render_preset_note(ir, note=57, duration_sec=0.4)


class TestUnisonRender:

    def test_stereo_shape(self, unison_render):
        assert unison_render.ndim == 2
        assert unison_render.shape[0] == 2
        assert unison_render.dtype == np.int16

    def test_determinism(self, unison_render):
        ir = load_instrument(str(SUPER_SAW_YAML))
        again = render_preset_note(ir, note=57, duration_sec=0.4)
        assert np.array_equal(unison_render, again)

    def test_width_metrics(self, unison_render):
        L = unison_render[0].astype(np.float64)
        R = unison_render[1].astype(np.float64)
        assert np.any(L != R), "unison render must have L != R"
        corr = np.corrcoef(L, R)[0, 1]
        assert corr < 0.95, f"L/R correlation {corr:.4f} should be < mono's 1.0"
        side_rms = np.sqrt((((L - R) / 2.0) ** 2).mean())
        mid_rms = np.sqrt((((L + R) / 2.0) ** 2).mean())
        assert mid_rms > 0
        assert side_rms > 0.05 * mid_rms, \
            f"side energy too small (side={side_rms:.1f}, mid={mid_rms:.1f})"

    def test_zero_width_collapses_to_center(self):
        ir = load_instrument_from_dict(_preset_dict(
            {'unison': {'voices': 5, 'spread_cents': 20, 'width': 0.0}}))
        out = render_preset_note(ir, note=60, duration_sec=0.3)
        assert out.ndim == 2
        # width=0 -> all copies pan center (equal-power), L == R exactly
        assert np.array_equal(out[0], out[1])

    def test_no_clipping(self, unison_render):
        # Peak guard keeps the unison mix inside int16 without hard clip
        assert int(np.abs(unison_render.astype(np.int32)).max()) <= 32000


class TestMonoCompatibility:
    """No unison authored -> the legacy mono path, bit-exact."""

    def test_mono_shape_and_determinism(self):
        ir = load_instrument(str(SAW_LEAD_YAML))
        a = render_preset_note(ir, note=60, duration_sec=0.3)
        b = render_preset_note(ir, note=60, duration_sec=0.3)
        assert a.ndim == 1
        assert a.dtype == np.int16
        assert np.array_equal(a, b)

    def test_mode_b_default_reset_zeros_phase(self):
        # phase_offset_q32 default 0 keeps the legacy zero-phase reset
        mb = ModeBState()
        mb.phase = 12345
        mb.reset(note=60, phase_inc=1 << 20)
        assert mb.phase == 0

    def test_mode_b_phase_offset_applied_on_reset(self):
        mb = ModeBState()
        mb.phase_offset_q32 = 0x9E3779B9
        mb.reset(note=60, phase_inc=1 << 20)
        assert mb.phase == 0x9E3779B9
