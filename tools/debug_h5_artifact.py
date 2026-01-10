"""Debug H5 artifact - verify root cause and explore mitigations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, mode_b_generate_coeffs, ModeBState,
    clear_basis_atlas, BLOCK_SIZE, SAMPLE_RATE, VM_K, K_STRIDE, NUM_LEAVES
)
from symcrash_python_ap.vm.tables import COEFF_LEN
from symcrash_python_ap.wavelet.idwpt_streaming import idwpt_polyphase_block
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc


def analyze_artifact_for_freq(base_freq_hz, harmonics, num_blocks=2000):
    """Analyze artifact at a specific base frequency."""
    note_offset = 12 * np.log2(base_freq_hz / 440)
    note = int(69 + note_offset)
    actual_freq = 440 * (2 ** ((note - 69) / 12))

    phase_inc = freq_to_phase_inc(actual_freq, SAMPLE_RATE)

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)
    mb_state.weights.set_custom(harmonics)
    mb_state.H_active = len(harmonics)

    all_samples = []

    for _ in range(num_blocks):
        carrier = np.zeros(COEFF_LEN, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, carrier)

        coeff_2d = carrier.reshape(NUM_LEAVES, VM_K)
        block_samples = idwpt_polyphase_block(coeff_2d, headroom_bits=1)
        all_samples.extend(block_samples.tolist())
        mb_state.advance_block()

    samples = np.array(all_samples, dtype=np.float64)

    # Windowed FFT
    n = len(samples)
    hann = np.hanning(n)
    windowed = samples * hann
    window_scale = 2.0 / np.mean(hann)

    spectrum = np.abs(rfft(windowed)) * window_scale
    freqs = rfftfreq(n, 1/SAMPLE_RATE)

    # Find strongest artifact frequency below 200 Hz
    low_mask = (freqs > 10) & (freqs < 200)
    low_freqs = freqs[low_mask]
    low_spectrum = spectrum[low_mask]

    if len(low_spectrum) > 0:
        max_idx = np.argmax(low_spectrum)
        artifact_freq = low_freqs[max_idx]
        artifact_amp = low_spectrum[max_idx]

        idx_fund = int(actual_freq * n / SAMPLE_RATE)
        fund_amp = spectrum[idx_fund]

        rel_db = 20 * np.log10(artifact_amp / fund_amp) if fund_amp > 0 else -100

        return artifact_freq, rel_db, actual_freq
    return 0, -100, actual_freq


print("=" * 60)
print("H5 ARTIFACT ANALYSIS")
print("=" * 60)

# Verify H5 beat frequency calculation
print("\n--- H5 Beat Frequency at Different Pitches ---")
block_rate = SAMPLE_RATE / BLOCK_SIZE

for note in range(48, 85, 3):  # C3 to C6
    freq = 440 * (2 ** ((note - 69) / 12))
    h5_freq = freq * 5
    cycles_per_block = h5_freq / block_rate
    frac = cycles_per_block % 1
    beat = block_rate * frac if frac < 0.5 else block_rate * (1 - frac)
    print(f"Note {note:3d} ({freq:7.1f} Hz): H5={h5_freq:7.1f} Hz, beat={beat:5.1f} Hz")

# Test at frequencies where H5 beat should be minimal
print("\n--- Finding Pitches with Minimal H5 Beat ---")
print("(Looking for where H5 cycles/block is near integer)")

best_notes = []
for note in range(36, 96):
    freq = 440 * (2 ** ((note - 69) / 12))
    h5_freq = freq * 5
    cycles_per_block = h5_freq / block_rate
    frac = cycles_per_block % 1
    beat = block_rate * frac if frac < 0.5 else block_rate * (1 - frac)
    if beat < 30:  # Less than 30 Hz beat
        best_notes.append((note, freq, beat, frac))
        print(f"Note {note:3d} ({freq:7.1f} Hz): beat={beat:5.1f} Hz (frac={frac:.4f})")

# Test rendering at a "good" frequency vs 440 Hz
if best_notes:
    print("\n--- Artifact Test at Good vs Bad Frequencies ---")

    # Test at 440 Hz (known bad)
    harmonics = [32767, 32767, 32767, 32767, 32767]  # H1-H5 equal
    art_freq, art_db, actual = analyze_artifact_for_freq(440, harmonics)
    print(f"A4 ({actual:.1f} Hz): artifact at {art_freq:.1f} Hz = {art_db:.1f} dB")

    # Test at a good frequency
    if len(best_notes) > 0:
        good_note, good_freq, good_beat, _ = best_notes[len(best_notes)//2]
        art_freq, art_db, actual = analyze_artifact_for_freq(good_freq, harmonics)
        print(f"Note {good_note} ({actual:.1f} Hz): artifact at {art_freq:.1f} Hz = {art_db:.1f} dB")

# The root cause analysis
print("\n" + "=" * 60)
print("ROOT CAUSE ANALYSIS")
print("=" * 60)
print("""
The 50 Hz artifact at A4 (440 Hz) is caused by:

1. H5 = 2200 Hz
2. Block rate = 750 Hz
3. 2200 / 750 = 2.933... cycles per block
4. Fractional part = 0.933
5. Beat = 750 * (1 - 0.933) = 50 Hz

The FWPT basis frames are computed for exact phase positions.
When H5's phase doesn't align with block boundaries, the
wavelet coefficients exhibit a slow 50 Hz modulation.

This is an aliasing artifact inherent to block-based wavelet
synthesis when the harmonic frequency doesn't divide evenly
into the block size.

POSSIBLE MITIGATIONS:
1. Use a block size that makes more frequencies "good" (e.g., 128)
2. Use overlap-add with crossfade between blocks
3. Apply a soft rolloff to harmonics that have high beat frequencies
4. Choose different sample rate (48000 -> 44100)
""")

# Test block size effect (theoretical)
print("\n--- Block Size Effect (Theoretical) ---")
for block_size in [32, 64, 128, 256]:
    block_rate_test = SAMPLE_RATE / block_size
    h5_freq = 2200
    cycles_per_block = h5_freq / block_rate_test
    frac = cycles_per_block % 1
    beat = block_rate_test * frac if frac < 0.5 else block_rate_test * (1 - frac)
    print(f"Block size {block_size:3d}: block_rate={block_rate_test:6.1f} Hz, H5 beat={beat:5.1f} Hz")
