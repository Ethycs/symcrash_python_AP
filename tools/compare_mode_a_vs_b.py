"""Compare Mode A vs Mode B for 50 Hz artifact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, PHASE_MASK
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, HarmonicWeights, VM_K, mode_b_generate_coeffs, ModeBState
)
from symcrash_python_ap.vm.coeff_lut_streaming import (
    StreamingCoeffLUT, get_lut_for_note, ccs_generate_coeffs_polyphase, CCSState
)
from symcrash_python_ap.fixed import Q15_ONE

BLOCK_SIZE = 64
SAMPLE_RATE = 48000
NUM_LEAVES = 64

print("=" * 70)
print("MODE A vs MODE B COMPARISON")
print(f"VM_K={VM_K}")
print("=" * 70)


def render_mode_a(note, num_blocks=100):
    """Mode A: LUT-based synthesis."""
    phase_inc = freq_to_phase_inc(440.0 * (2.0 ** ((note - 69) / 12.0)), SAMPLE_RATE)

    ccs_state = CCSState()
    ccs_state.note = note
    ccs_state.wave_id = 0  # SINE
    ccs_state.phase_inc = phase_inc

    all_samples = []

    for _ in range(num_blocks):
        coeffs = np.zeros(64 * VM_K, dtype=np.int16)
        ccs_generate_coeffs_polyphase(ccs_state, coeffs)

        coeff_2d = coeffs.reshape(64, VM_K)
        audio = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        all_samples.extend(audio.tolist())

        ccs_state.advance_block()

    return np.array(all_samples, dtype=np.float64)


def render_mode_b_sine(note, num_blocks=100):
    """Mode B: Harmonic bank synthesis with pure sine."""
    phase_inc = freq_to_phase_inc(440.0 * (2.0 ** ((note - 69) / 12.0)), SAMPLE_RATE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_sine()  # Pure sine
    mb_state.H_active = 1

    basis = generate_basis_frames(note, H_max=8)

    all_samples = []

    for _ in range(num_blocks):
        coeffs = np.zeros(64 * VM_K, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, coeffs)

        coeff_2d = coeffs.reshape(64, VM_K)
        audio = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        all_samples.extend(audio.tolist())

        mb_state.advance_block()

    return np.array(all_samples, dtype=np.float64)


def render_mode_b_violin(note, num_blocks=100):
    """Mode B: Harmonic bank synthesis with violin harmonics."""
    phase_inc = freq_to_phase_inc(440.0 * (2.0 ** ((note - 69) / 12.0)), SAMPLE_RATE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)

    # Violin-like harmonics
    violin_harmonics = [Q15_ONE, 24576, 16384, 8192, 3673, 2457, 1229, 614]
    mb_state.weights.set_custom(violin_harmonics)
    mb_state.H_active = len(violin_harmonics)

    basis = generate_basis_frames(note, H_max=8)

    all_samples = []

    for _ in range(num_blocks):
        coeffs = np.zeros(64 * VM_K, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, coeffs)

        coeff_2d = coeffs.reshape(64, VM_K)
        audio = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        all_samples.extend(audio.tolist())

        mb_state.advance_block()

    return np.array(all_samples, dtype=np.float64)


def analyze_50hz(samples, label):
    """Analyze 50 Hz artifact level relative to fundamental."""
    n = len(samples)
    hann = np.hanning(n)
    windowed = samples * hann
    spectrum = np.abs(rfft(windowed)) * (2.0 / np.mean(hann))

    # Find fundamental (440 Hz)
    idx_fund = int(440 * n / SAMPLE_RATE)
    amp_fund = spectrum[idx_fund] if idx_fund < len(spectrum) else 1

    # Find 50 Hz
    idx_50 = int(50 * n / SAMPLE_RATE)
    amp_50 = spectrum[idx_50]

    rel_50 = 20 * np.log10(amp_50 / amp_fund) if amp_fund > 0 else -100

    print(f"{label:40s}: 50Hz = {rel_50:+6.1f} dB")
    return rel_50


print("\n--- Mode A vs Mode B: Pure Sine ---")
samples_a = render_mode_a(69)
samples_b = render_mode_b_sine(69)

analyze_50hz(samples_a, "Mode A (LUT sine)")
analyze_50hz(samples_b, "Mode B (Harmonic bank sine)")

print("\n--- Mode B: Violin Harmonics ---")
samples_violin = render_mode_b_violin(69)
analyze_50hz(samples_violin, "Mode B (Violin harmonics)")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
If Mode A and Mode B pure sine are similar, the issue is NOT in Mode B's
coefficient generation but in the IDWPT itself.

If violin harmonics have worse artifact than pure sine, H5 (2200 Hz) is
contributing disproportionately due to its 2.93 cycles/block boundary error.
""")
