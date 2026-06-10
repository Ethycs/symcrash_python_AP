"""
Tests for engine-native envelope dynamics (WSE-STD-PRESET-AUTHORING 3.3):

- velocity_sens on matrix envelope bindings (velocity -> envelope output,
  e.g. velocity -> FM modulation index) and on the Amp node (velocity ->
  gain curve: scale = 1 - sens + sens*velocity)
- key_scale_rate (DX7-style EG rate scaling): envelope times scale by
  2^(-ks * (note - 60) / 24), so higher notes ring shorter

Rendered through tools/render/render_preset.py, which mirrors the
voice_engine semantics (same formulas at the same stages).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "render"))

from render_preset import render_preset_note  # noqa: E402
from symcrash_python_ap.instrument_ir import load_instrument_from_dict  # noqa: E402
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas  # noqa: E402

FS = 48000


def _mallet_dict(amp_velocity_sens=None, amp_key_scale_rate=None,
                 strike_velocity_sens=0.0):
    amp = {
        'type': 'Amp',
        'attack': 2.0,
        'decay': 400.0,
        'sustain': 0.0,
        'release': 60.0,
    }
    if amp_velocity_sens is not None:
        amp['velocity_sens'] = amp_velocity_sens
    if amp_key_scale_rate is not None:
        amp['key_scale_rate'] = amp_key_scale_rate
    return {
        'instrument_ir_v0': {
            'meta': {'id': 'test-mallet', 'name': 'Test Mallet'},
            'graph': {
                'fm6': {
                    'type': 'FM6',
                    'output': 'amp1',
                    'algorithm': 5,
                    'feedback': 0,
                    'operators': [
                        {'ratio': 1.0, 'level': 0.4},
                        {'ratio': 2.0, 'level': 0.6},
                        {'ratio': 1.0, 'level': 0.0},
                        {'ratio': 1.0, 'level': 0.0},
                        {'ratio': 1.0, 'level': 0.0},
                        {'ratio': 1.0, 'level': 0.0},
                    ],
                },
                'amp1': amp,
            },
            'envelopes': {
                'strike_env': {
                    'attack': 0.5,
                    'decay': 40.0,
                    'sustain': 0.0,
                    'release': 30.0,
                    'velocity_sens': strike_velocity_sens,
                    'targets': ['FM_OP2'],
                },
            },
        }
    }


def _render(data, note=60, velocity=1.0, duration=1.0):
    clear_basis_atlas()
    ir = load_instrument_from_dict(data)
    return np.asarray(
        render_preset_note(ir, note=note, duration_sec=duration,
                           velocity_q15=int(velocity * 32767)),
        dtype=np.float64)


def _high_band_fraction(s, f0, lo_ms=2, hi_ms=60):
    seg = s[int(lo_ms / 1000 * FS): int(hi_ms / 1000 * FS)]
    seg = (seg - seg.mean()) * np.hanning(len(seg))
    sp = np.abs(np.fft.rfft(seg)) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / FS)
    hi = sp[fr > 2.5 * f0].sum()
    return float(hi / max(sp.sum(), 1e-12))


def _ring_ms(s):
    env = np.abs(s)
    peak = env.max() or 1.0
    above = np.where(env > peak * 0.05)[0]
    return (above[-1] / FS * 1000) if len(above) else 0.0


class TestParsing:

    def test_binding_dynamics_fields(self):
        ir = load_instrument_from_dict(_mallet_dict(strike_velocity_sens=0.7))
        binding = ir.envelopes[0]
        assert binding.velocity_sens == pytest.approx(0.7)
        assert binding.key_scale_rate == 0.0

    def test_amp_dynamics_fields(self):
        ir = load_instrument_from_dict(
            _mallet_dict(amp_velocity_sens=0.5, amp_key_scale_rate=0.8))
        amp = next(n for n in ir.layers.L0_graph.nodes if n.type == "Amp")
        assert amp.params.velocity_sens == pytest.approx(0.5)
        assert amp.params.key_scale_rate == pytest.approx(0.8)


class TestVelocitySens:

    def test_velocity_drives_modulation_index(self):
        """With velocity_sens on the modulator envelope, harder hits must be
        spectrally brighter (more FM sidebands), not just louder."""
        data = _mallet_dict(strike_velocity_sens=1.0)
        soft = _render(data, velocity=0.35)
        hard = _render(data, velocity=1.0)
        f0 = 261.6
        hb_soft = _high_band_fraction(soft, f0)
        hb_hard = _high_band_fraction(hard, f0)
        assert hb_hard > hb_soft * 1.5, (
            f"high-band fraction soft={hb_soft:.4f} hard={hb_hard:.4f} — "
            "velocity is not reaching the modulation index"
        )

    def test_amp_velocity_sens_compresses_dynamics(self):
        """Amp velocity_sens < 1 narrows the loudness range across velocity."""
        raw = _mallet_dict()            # default sens = 1.0 (raw velocity)
        soft_raw = _render(raw, velocity=0.4)
        hard_raw = _render(raw, velocity=1.0)
        ratio_raw = np.abs(soft_raw).max() / max(np.abs(hard_raw).max(), 1e-9)

        comp = _mallet_dict(amp_velocity_sens=0.5)
        soft_c = _render(comp, velocity=0.4)
        hard_c = _render(comp, velocity=1.0)
        ratio_c = np.abs(soft_c).max() / max(np.abs(hard_c).max(), 1e-9)

        assert ratio_raw == pytest.approx(0.4, abs=0.08)
        assert ratio_c > ratio_raw + 0.15, (
            f"velocity curve not compressing: raw={ratio_raw:.2f} "
            f"sens0.5={ratio_c:.2f} (expected ~0.7)"
        )


class TestKeyScaling:

    def test_amp_key_scaling_shortens_high_notes(self):
        """key_scale_rate=1: a note an octave up rings audibly shorter."""
        data = _mallet_dict(amp_key_scale_rate=1.0)
        low = _render(data, note=48, duration=1.2)
        high = _render(data, note=72, duration=1.2)
        ring_low = _ring_ms(low)
        ring_high = _ring_ms(high)
        assert ring_low > ring_high * 1.5, (
            f"key scaling inactive: ring(48)={ring_low:.0f}ms "
            f"ring(72)={ring_high:.0f}ms"
        )

    def test_no_key_scaling_by_default(self):
        data = _mallet_dict()
        low = _render(data, note=48, duration=1.2)
        high = _render(data, note=72, duration=1.2)
        assert abs(_ring_ms(low) - _ring_ms(high)) < 100, (
            "default (ks=0) should ring equally across the keyboard"
        )
