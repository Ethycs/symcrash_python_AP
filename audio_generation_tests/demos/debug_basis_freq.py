"""Debug basis frame frequency encoding."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, clear_basis_atlas, VM_K, NUM_LEAVES, BLOCK_SIZE, PHASE_MASK
)
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sine_q15
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit, fwpt_jit

SAMPLE_RATE = 48000

clear_basis_atlas()

print("=" * 70)
print("Basis Frame Frequency Analysis")
print("=" * 70)

# Generate basis at A440
anchor_note = 69
basis = generate_basis_frames(anchor_note, H_max=16)

print(f"\nAnchor: MIDI {anchor_note}, Freq: {basis.anchor_freq_hz} Hz")
print(f"Phase inc for this freq: {freq_to_phase_inc(basis.anchor_freq_hz, SAMPLE_RATE)}")

# Test: reconstruct Usin_1 and Ucos_1 for k=0
print(f"\n\nReconstructing basis frames via IDWPT:")
print("-" * 60)

for k in range(3):  # Just first 3 k-slots
    # Get coefficients for h=1 (fundamental)
    sin_coeffs = basis.Usin_h[0, :, k].tolist()
    cos_coeffs = basis.Ucos_h[0, :, k].tolist()

    # IDWPT to time domain
    sin_time = np.array(idwpt_jit(sin_coeffs, headroom_bits=1), dtype=np.int16)
    cos_time = np.array(idwpt_jit(cos_coeffs, headroom_bits=1), dtype=np.int16)

    print(f"\nk={k}:")
    print(f"  Usin (first 8): {sin_time[:8].tolist()}")
    print(f"  Ucos (first 8): {cos_time[:8].tolist()}")

    # Check what these should be
    # The basis was generated with:
    #   sin_unrolled = [sin(n * phase_inc) for n in range(64)]
    #   sin_rolled = sin_unrolled[k:] + sin_unrolled[:k]
    #   Usin[k] = FWPT(sin_rolled)
    # So IDWPT(Usin[k]) should give back sin_rolled

    phase_inc = freq_to_phase_inc(basis.anchor_freq_hz, SAMPLE_RATE)
    expected_sin_unrolled = [sine_q15((n * phase_inc) & PHASE_MASK) for n in range(64)]
    expected_sin_rolled = expected_sin_unrolled[k:] + expected_sin_unrolled[:k]

    print(f"  Expected sin (first 8): {expected_sin_rolled[:8]}")

    # Compute error
    err = [sin_time[i] - expected_sin_rolled[i] for i in range(64)]
    max_err = max(abs(e) for e in err)
    rms_err = np.sqrt(np.mean([e**2 for e in err]))
    print(f"  Max error: {max_err}, RMS error: {rms_err:.1f}")

# Now let's verify the frequency by looking at zero crossings
print(f"\n\nFrequency verification via zero crossings:")
print("-" * 60)

# Reconstruct a longer signal using the basis
from symcrash_python_ap.vm.harmonic_bank import mode_b_generate_coeffs, ModeBState
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_from_flat

state = ModeBState()
phase_inc = freq_to_phase_inc(440.0, SAMPLE_RATE)
state.reset(note=69, phase_inc=phase_inc)
state.weights.set_sine()
state.H_active = 1

# Generate enough samples for frequency measurement
from symcrash_python_ap.vm.harmonic_bank import mode_b_advance_phase

samples = []
for _ in range(100):  # 100 blocks = ~0.13 seconds
    output_coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(state, basis, output_coeffs)
    audio = idwpt_polyphase_from_flat(output_coeffs, K=8, headroom_bits=1)
    samples.extend(audio.tolist())
    mode_b_advance_phase(state)

samples = np.array(samples, dtype=np.float64)

# Find zero crossings (rising edge)
crossings = []
for i in range(1, len(samples)):
    if samples[i-1] < 0 and samples[i] >= 0:
        # Linear interpolation to find exact crossing
        t = -samples[i-1] / (samples[i] - samples[i-1])
        crossings.append(i - 1 + t)

if len(crossings) >= 2:
    periods = np.diff(crossings)
    avg_period = np.mean(periods)
    measured_freq = SAMPLE_RATE / avg_period

    print(f"  Zero crossings found: {len(crossings)}")
    print(f"  Average period: {avg_period:.2f} samples")
    print(f"  Measured frequency: {measured_freq:.2f} Hz")
    print(f"  Expected frequency: 440.00 Hz")
    print(f"  Error: {measured_freq - 440:.4f} Hz ({(measured_freq/440 - 1)*100:.4f}%)")

    # Check period stability
    print(f"\n  Period stability:")
    print(f"    Min period: {min(periods):.2f} samples")
    print(f"    Max period: {max(periods):.2f} samples")
    print(f"    Std dev: {np.std(periods):.4f} samples")

    # FFT of period lengths - this will show if there's block-rate jitter
    if len(periods) > 8:
        period_fft = np.abs(np.fft.rfft(periods - avg_period))
        period_freqs = np.fft.rfftfreq(len(periods), avg_period / SAMPLE_RATE)

        # Find dominant modulation frequency
        peak_idx = np.argmax(period_fft[1:]) + 1  # Skip DC
        print(f"\n  Period jitter analysis:")
        print(f"    Dominant jitter frequency: {period_freqs[peak_idx]:.1f} Hz")
        print(f"    (Block rate would be {750.0:.1f} Hz)")
