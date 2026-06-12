"""Test PWM synthesis in Mode B."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, mode_b_generate_coeffs, get_basis_for_note, VM_K, COEFF_LEN
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)


def _render_pwm(duty_q15, num_blocks=50):
    """Render PWM at given duty cycle; return (samples, measured_duty, expected_duty)."""
    basis = get_basis_for_note(60)
    mb_state = ModeBState()
    mb_state.reset(note=60, phase_inc=phase_inc)
    mb_state.weights.set_pwm(duty_q15=duty_q15, H=32)
    mb_state.H_active = 32

    all_samples = []
    for _ in range(num_blocks):
        coeffs = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, coeffs)
        audio = idwpt_jit(coeffs.reshape(64, VM_K)[:, 0].tolist(), headroom_bits=1)
        all_samples.extend(audio)
        mb_state.advance_block()

    samples = np.array(all_samples[500:], dtype=np.int16)

    # Better duty cycle measurement: find zero crossings
    # and measure time between rising and falling edges
    crossings = []
    for i in range(len(samples) - 1):
        if samples[i] <= 0 and samples[i+1] > 0:
            crossings.append((i, 'rise'))
        elif samples[i] >= 0 and samples[i+1] < 0:
            crossings.append((i, 'fall'))

    # Measure duty from crossing pairs
    duties_measured = []
    for i in range(len(crossings) - 2):
        if crossings[i][1] == 'rise' and crossings[i+1][1] == 'fall':
            high_time = crossings[i+1][0] - crossings[i][0]
            if i + 2 < len(crossings) and crossings[i+2][1] == 'rise':
                period = crossings[i+2][0] - crossings[i][0]
                if period > 0:
                    duties_measured.append(high_time / period)

    measured_duty = np.mean(duties_measured) if duties_measured else 0

    expected_duty = duty_q15 / 32768.0

    return samples, measured_duty, expected_duty


def _harmonic_magnitude(samples, harmonic):
    """Magnitude of the h-th harmonic of f0 in the rendered audio."""
    n = len(samples)
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64) * np.hanning(n)))
    bin_ = int(round(harmonic * freq_hz * n / 48000))
    return spectrum[max(0, bin_ - 2):bin_ + 3].max()


def test_pwm_spectrum_tracks_duty():
    """PWM harmonic spectrum should track duty cycle: a_h ~ sin(pi*h*D)/h.

    The engine's PWM is AC-coupled (DC term deliberately omitted, see
    compute_pwm_harmonics) and harmonics are sine-phase-aligned, so the
    time-domain shape is not a rectangle; the duty-cycle signature lives
    in the magnitude spectrum (nulls where sin(pi*h*D) == 0).
    """
    # 50% duty: even harmonics null, H3/H1 ~ 1/3
    samples_50, _, _ = _render_pwm(16384, num_blocks=50)
    h1 = _harmonic_magnitude(samples_50, 1)
    assert h1 > 0, "PWM render produced silence"
    assert _harmonic_magnitude(samples_50, 2) / h1 < 0.05, "50% duty: H2 should be nulled"
    assert _harmonic_magnitude(samples_50, 4) / h1 < 0.05, "50% duty: H4 should be nulled"
    assert abs(_harmonic_magnitude(samples_50, 3) / h1 - 1.0 / 3.0) < 0.1, (
        "50% duty: H3/H1 should be ~1/3"
    )

    # 25% duty: H4 null, H2 strong (theory sqrt(2)/2 ~ 0.71)
    samples_25, _, _ = _render_pwm(8192, num_blocks=50)
    h1 = _harmonic_magnitude(samples_25, 1)
    assert h1 > 0, "PWM render produced silence"
    assert _harmonic_magnitude(samples_25, 4) / h1 < 0.05, "25% duty: H4 should be nulled"
    assert _harmonic_magnitude(samples_25, 2) / h1 > 0.4, "25% duty: H2 should be strong"


def main():
    print("PWM SYNTHESIS TEST")
    print("=" * 50)

    print(f"\n{'Duty (Q15)':<12} {'Expected':>10} {'Measured':>10} {'Error':>10}")
    print("-" * 50)

    duties = [8192, 16384, 24576, 4096, 28672]  # 25%, 50%, 75%, 12.5%, 87.5%

    for duty in duties:
        samples, measured, expected = _render_pwm(duty)
        error = abs(measured - expected) * 100
        status = "[ok]" if error < 5 else "[!]"
        print(f"{duty:<12} {expected:>9.1%} {measured:>9.1%} {error:>9.1f}% {status}")

    # Visual check - show waveform shape
    print("\n" + "=" * 50)
    print("WAVEFORM SAMPLES (50% duty)")
    print("=" * 50)

    samples_50, _, _ = _render_pwm(16384)
    print("First 20 samples:", list(samples_50[:20]))
    print("Samples 80-100:", list(samples_50[80:100]))

    # Debug: check the harmonic weights
    print("\n" + "=" * 50)
    print("PWM HARMONIC WEIGHTS DEBUG")
    print("=" * 50)

    from symcrash_python_ap.vm.waveform_matrix import compute_pwm_harmonics

    for duty in [8192, 16384, 24576]:
        dc, a_h, b_h = compute_pwm_harmonics(duty, H_max=8)
        duty_pct = duty / 32768.0 * 100
        print(f"\nDuty={duty_pct:.0f}%: dc={dc}")
        print(f"  a_h[1:8] = {list(a_h[:8])}")

    # Compare to direct calculation
    print("\n" + "=" * 50)
    print("REFERENCE: Direct PWM synthesis (no wavelet)")
    print("=" * 50)

    from symcrash_python_ap.wavelet.sine_core import sin_cos_q15
    import math

    def direct_pwm(duty_q15, num_samples=2000):
        duty = duty_q15 / 32768.0
        H = 32
        samples = []
        phase = 0
        dc = int((2 * duty - 1) * 32767)

        for _ in range(num_samples):
            val = dc
            for h in range(1, H + 1):
                amp = (2 * 32767 / (math.pi * h)) * math.sin(math.pi * h * duty)
                h_phase = (phase * h) & 0xFFFFFFFF
                s, _ = sin_cos_q15(h_phase)
                val += int((amp * s) / 32767)
            samples.append(np.clip(val, -32768, 32767))
            phase = (phase + phase_inc) & 0xFFFFFFFF

        return np.array(samples, dtype=np.int16)

    for duty in [8192, 16384, 24576]:
        samples = direct_pwm(duty)
        expected = duty / 32768.0
        # Use DC level as proxy for duty cycle
        measured_dc = np.mean(samples) / 32767.0
        # DC should be (2*duty - 1), so duty = (dc + 1) / 2
        measured_duty = (measured_dc + 1) / 2
        print(f"Duty {duty}: expected={expected:.1%}, measured_duty={measured_duty:.1%}")

    # Better test: compare DC offsets
    print("\n" + "=" * 50)
    print("DC OFFSET TEST (correct metric for PWM)")
    print("=" * 50)

    print(f"\n{'Duty':<10} {'Expected DC':>12} {'Mode B DC':>12} {'Direct DC':>12}")
    print("-" * 50)

    for duty in [4096, 8192, 16384, 24576, 28672]:
        # Expected DC: 2D - 1
        expected_dc = 2 * (duty / 32768.0) - 1

        # Mode B
        samples_modeb, _, _ = _render_pwm(duty, num_blocks=30)
        dc_modeb = np.mean(samples_modeb) / 32767.0

        # Direct
        samples_direct = direct_pwm(duty)
        dc_direct = np.mean(samples_direct) / 32767.0

        print(f"{duty:<10} {expected_dc:>12.3f} {dc_modeb:>12.3f} {dc_direct:>12.3f}")


if __name__ == "__main__":
    main()
