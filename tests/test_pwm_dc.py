"""
Regression tests for the PWM DC component (packet E7a).

The pulse Fourier series contains a DC term (2d - 1) for duty cycle d.
compute_pwm_weights / compute_pwm_harmonics used to emit it, so any
non-50% pulse rode on a duty-dependent DC offset, and a width LFO
(e.g. analog_strings_v0's 2 Hz PWM motion) turned the moving DC into
low-frequency rumble at the LFO rate. The oscillator output is
AC-coupled by design: the DC term must be omitted while the AC
spectrum (a_h = (2/pi h) * sin(pi h d)) is preserved.
"""

import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas, compute_pwm_weights
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.waveform_matrix import compute_pwm_harmonics

FS = 48000
BLOCK_SIZE = 64

DC_BOUND_DBFS = -60.0


def _render_pulse(note=57, width_q15=15401, blocks=375, sweep_hz=None):
    """Render a sustained PULSE note via the harmonic-bank PWM path.

    width 15401 ~ 47% duty (the analog_strings_v0 cookbook setting).
    If sweep_hz is set, the width is swept sinusoidally like a PWM LFO.
    """
    clear_basis_atlas()
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)
    out = []
    for i in range(blocks):
        w = width_q15
        if sweep_hz:
            t = i * BLOCK_SIZE / FS
            w = int(16384 + 0.235 * 32767 * np.sin(2 * np.pi * sweep_hz * t))
            w = max(1638, min(31130, w))
        frame = VoiceFrame(
            note=note,
            coeff_osc_mode=1,  # Mode B
            gain_q15=32767,
            H_request=16,
            waveform_type=2,  # PULSE
            pwm_width_q15=w,
            pwm_enabled=True,
        )
        state, output = vm_step_block(state, frame)
        # vm_step_block reuses its output buffer — copy each block
        out.append(np.array(output, dtype=np.int16, copy=True))
    return np.concatenate(out).astype(np.float64) / 32768.0


def _dc_dbfs(x):
    return 20.0 * np.log10(max(abs(float(np.mean(x))), 1e-12))


def test_static_pulse_dc_below_minus_60_dbfs():
    """Sustained 47% pulse: DC must stay below -60 dBFS (was ~-24 dBFS)."""
    x = _render_pulse(width_q15=15401)
    assert _dc_dbfs(x) < DC_BOUND_DBFS
    # The pulse must still actually sound (AC spectrum preserved)
    assert np.sqrt(np.mean(x**2)) > 0.05


def test_narrow_pulse_dc_below_minus_60_dbfs():
    """25% duty is the worst historical case (DC was ~-6 dBFS)."""
    x = _render_pulse(width_q15=8192)
    assert _dc_dbfs(x) < DC_BOUND_DBFS
    assert np.sqrt(np.mean(x**2)) > 0.05


def test_pwm_sweep_has_no_lf_rumble():
    """2 Hz width LFO (analog_strings_v0): no sub-20 Hz rumble from moving DC."""
    x = _render_pulse(blocks=750, sweep_hz=2.0)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / FS)
    lf = np.sqrt(np.sum(np.abs(spec[(freqs > 0) & (freqs < 20.0)]) ** 2))
    lf_dbfs = 20.0 * np.log10(max(lf / len(x) * 2.0, 1e-12))
    assert lf_dbfs < DC_BOUND_DBFS  # was ~-6.6 dBFS before the fix


def test_pwm_weights_have_zero_dc_but_intact_harmonics():
    """compute_pwm_weights: dc = 0 for any width; a_h unchanged."""
    for width in (3276, 8192, 15401, 16384, 24576, 31130):
        w = compute_pwm_weights(width)
        assert w.dc_q15 == 0, f"width {width} leaked dc={w.dc_q15}"
        d = width / 32768.0
        # Fundamental: a_1 = (2/pi) * sin(pi*d) in Q15
        expected = int(round((2.0 / np.pi) * np.sin(np.pi * d) * 32767))
        assert abs(int(w.a_h_q15[0]) - expected) <= 1


def test_pwm_harmonics_have_zero_dc():
    """compute_pwm_harmonics (waveform_matrix path): dc = 0 for any duty."""
    for width in (3276, 8192, 15401, 24576):
        dc, a_h, b_h = compute_pwm_harmonics(width)
        assert dc == 0
        assert np.any(a_h != 0)  # spectrum still present
