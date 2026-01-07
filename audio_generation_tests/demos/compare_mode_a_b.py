"""Compare Mode A and Mode B spectral quality."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, mode_b_advance_phase,
    ModeBState, clear_basis_atlas
)
from symcrash_python_ap.vm.coeff_lut_streaming import get_lut_for_note, CCSState, ccs_generate_coeffs_polyphase
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15, PHASE_MASK
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_from_flat

SAMPLE_RATE = 48000
BLOCK_SIZE = 64

clear_basis_atlas()

print("=" * 70)
print("Mode A vs Mode B Spectral Comparison")
print("=" * 70)

# Generate 1 second of audio from both modes
num_blocks = SAMPLE_RATE // BLOCK_SIZE
phase_inc = freq_to_phase_inc(440.0, SAMPLE_RATE)

# Mode B
print("\nGenerating Mode B audio...")
clear_basis_atlas()
basis = generate_basis_frames(69, H_max=16)
state_b = ModeBState()
state_b.reset(note=69, phase_inc=phase_inc)
state_b.weights.set_sine()
state_b.H_active = 1

mode_b_samples = []
for _ in range(num_blocks):
    output_coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(state_b, basis, output_coeffs)
    audio = idwpt_polyphase_from_flat(output_coeffs, K=8, headroom_bits=1)
    mode_b_samples.extend(audio.tolist())
    mode_b_advance_phase(state_b)

mode_b_samples = np.array(mode_b_samples, dtype=np.float64)

# Mode A
print("Generating Mode A audio...")
lut = get_lut_for_note(0, 69)  # wave_id=0 is sine
state_a = CCSState()
state_a.reset(note=69, phase_inc=phase_inc, wave_id=0)

mode_a_samples = []
for _ in range(num_blocks):
    output_coeffs = np.zeros((64, 8), dtype=np.int16)
    lut.get_subband_samples(state_a.phase, output_coeffs)
    flat_coeffs = output_coeffs.flatten()
    audio = idwpt_polyphase_from_flat(flat_coeffs, K=8, headroom_bits=1)
    mode_a_samples.extend(audio.tolist())
    state_a.advance_block()

mode_a_samples = np.array(mode_a_samples, dtype=np.float64)

# Reference sine
print("Generating reference sine...")
ref_phase = 0
ref_samples = []
for _ in range(len(mode_b_samples)):
    ref_samples.append(sine_q15(ref_phase))
    ref_phase = (ref_phase + phase_inc) & PHASE_MASK
ref_samples = np.array(ref_samples, dtype=np.float64)

# Normalize signals to same RMS for fair comparison
def normalize(samples):
    rms = np.sqrt(np.mean(samples ** 2))
    if rms > 0:
        return samples * (10000.0 / rms)
    return samples

mode_a_norm = normalize(mode_a_samples)
mode_b_norm = normalize(mode_b_samples)
ref_norm = normalize(ref_samples)

# FFT analysis function
def analyze_spectrum(samples, samples_norm, ref_norm, name, fundamental_hz=440.0):
    N = 8192
    chunk = samples[:N]
    window = np.hanning(N)
    spectrum = np.abs(np.fft.rfft(chunk * window))
    freqs = np.fft.rfftfreq(N, 1/SAMPLE_RATE)
    max_val = np.max(spectrum)
    spectrum_db = 20 * np.log10(spectrum / max_val + 1e-12)

    # Find spurs (non-harmonic peaks)
    harmonics = [fundamental_hz * (h+1) for h in range(50)]
    peaks = []
    for i in range(2, len(spectrum_db) - 2):
        if (spectrum_db[i] > spectrum_db[i-1] and
            spectrum_db[i] > spectrum_db[i+1] and
            spectrum_db[i] > -80):
            is_harmonic = any(abs(freqs[i] - h) < 10 for h in harmonics)
            if not is_harmonic:
                peaks.append((freqs[i], spectrum_db[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)
    worst_spur = peaks[0][1] if peaks else -100

    # RMS error after normalization
    rms_err = np.sqrt(np.mean((samples_norm - ref_norm)**2))

    print(f"\n{name}:")
    print(f"  RMS: {np.sqrt(np.mean(samples**2)):.0f}")
    print(f"  RMS error vs ref (normalized): {rms_err:.1f}")
    print(f"  Worst spur: {worst_spur:.1f} dB")
    if peaks:
        print(f"  Top 5 spurs:")
        for i, (freq, db) in enumerate(peaks[:5]):
            print(f"    {freq:.1f} Hz: {db:.1f} dB")

    return worst_spur, rms_err

print("\n" + "=" * 70)
print("Spectral Analysis")
print("=" * 70)

spur_b, err_b = analyze_spectrum(mode_b_samples, mode_b_norm, ref_norm, "Mode B")
spur_a, err_a = analyze_spectrum(mode_a_samples, mode_a_norm, ref_norm, "Mode A")
spur_ref, err_ref = analyze_spectrum(ref_samples, ref_norm, ref_norm, "Reference (direct sine)")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"  Mode A: worst spur {spur_a:.1f} dB, RMS error {err_a:.1f}")
print(f"  Mode B: worst spur {spur_b:.1f} dB, RMS error {err_b:.1f}")
print(f"  Reference: worst spur {spur_ref:.1f} dB")
print(f"\n  Note: Spurs < -60 dB are generally inaudible")
print(f"  Both modes should have similar RMS error vs reference")
