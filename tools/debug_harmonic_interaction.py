"""Debug multi-harmonic interaction causing 50 Hz artifact."""
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
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc


def render_with_harmonics(harmonics, num_blocks=2000):
    """Render with specific harmonic profile."""
    note = 69  # A4 = 440 Hz
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

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

    return np.array(all_samples, dtype=np.float64)


def analyze_50hz(samples, name):
    """Analyze 50 Hz component with windowed FFT."""
    # Use Hann window
    n = len(samples)
    hann = np.hanning(n)
    windowed = samples * hann
    window_scale = 2.0 / np.mean(hann)

    spectrum = np.abs(rfft(windowed)) * window_scale
    freqs = rfftfreq(n, 1/SAMPLE_RATE)

    idx_50 = int(50 * n / SAMPLE_RATE)
    idx_440 = int(440 * n / SAMPLE_RATE)

    amp_50 = spectrum[idx_50]
    amp_440 = spectrum[idx_440]

    rel_db = 20 * np.log10(amp_50 / amp_440) if amp_440 > 0 else -100

    print(f"{name}: 50 Hz = {rel_db:.1f} dB relative to 440 Hz")
    return rel_db


print("=" * 60)
print("MULTI-HARMONIC INTERACTION TEST")
print("=" * 60)

# Test 1: Pure H1 only
print("\n--- Single Harmonic Tests ---")
analyze_50hz(render_with_harmonics([32767]), "H1 only")

# Test 2: H2 only
analyze_50hz(render_with_harmonics([0, 32767]), "H2 only")

# Test 3: H1 + H2
print("\n--- Two Harmonic Combinations ---")
analyze_50hz(render_with_harmonics([32767, 32767]), "H1 + H2")
analyze_50hz(render_with_harmonics([32767, 16384]), "H1 + H2(50%)")
analyze_50hz(render_with_harmonics([32767, 0, 32767]), "H1 + H3")
analyze_50hz(render_with_harmonics([32767, 0, 0, 32767]), "H1 + H4")

# Test 4: Progressive harmonic addition
print("\n--- Progressive Harmonic Addition ---")
for num_h in range(1, 9):
    harmonics = [32767] * num_h
    name = f"H1-H{num_h} (all equal)"
    analyze_50hz(render_with_harmonics(harmonics), name)

# Test 5: Violin profile
print("\n--- Violin Profile ---")
violin_harmonics = [30890, 32767, 6926, 23969, 3673, 6138, 1164, 1228]
analyze_50hz(render_with_harmonics(violin_harmonics), "Violin profile")

# Test 6: Just the strong harmonics from violin
print("\n--- Violin Strong Harmonics Only ---")
analyze_50hz(render_with_harmonics([30890, 32767]), "Violin H1+H2")
analyze_50hz(render_with_harmonics([30890, 32767, 0, 23969]), "Violin H1+H2+H4")
analyze_50hz(render_with_harmonics([30890, 32767, 6926, 23969]), "Violin H1-H4")

# Test 7: Check if it's related to specific harmonic phase relationships
print("\n--- Phase Relationship Test ---")
# H1 and H2 beat at: |2f - f| = f = 440 Hz (no 50 Hz beat)
# H4 and H2 beat at: |4f - 2f| = 2f = 880 Hz (no 50 Hz beat)
# So direct harmonic beats shouldn't create 50 Hz
print("Theoretical beat frequencies:")
print("  H1 + H2 beat: 440 Hz")
print("  H2 + H4 beat: 880 Hz")
print("  H1 + H4 beat: 1320 Hz")
print("  -> No 50 Hz from simple harmonic beats")

# The 50 Hz must come from the block processing interacting with
# the incommensurate relationship between 440 Hz and block rate
print("\n--- Block/Frequency Relationship ---")
block_rate = SAMPLE_RATE / BLOCK_SIZE
cycles_per_block = 440 / block_rate
print(f"Block rate: {block_rate} Hz")
print(f"440 Hz cycles per block: {cycles_per_block:.4f}")
print(f"Fractional part: {cycles_per_block % 1:.4f}")
print(f"Beat pattern: {block_rate * (cycles_per_block % 1):.1f} Hz")

# For 440 Hz: 440/750 = 0.5867 cycles/block
# 750 * 0.5867 = 440 Hz (just the fundamental)
# But with multiple harmonics, the phase error accumulates differently
print(f"\nPhase error per block for each harmonic:")
for h in range(1, 9):
    h_freq = 440 * h
    h_cycles_per_block = h_freq / block_rate
    frac = h_cycles_per_block % 1
    beat = block_rate * frac if frac < 0.5 else block_rate * (1 - frac)
    print(f"  H{h} ({h_freq} Hz): {h_cycles_per_block:.4f} cycles/block, frac={frac:.4f}, beat={beat:.1f} Hz")
