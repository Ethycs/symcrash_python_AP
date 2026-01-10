"""Debug multi-harmonic 50Hz artifact."""
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

def render_with_harmonics(harmonic_amps, num_blocks=100):
    """Render with specific harmonic profile."""
    note = 69  # A4 = 440 Hz
    freq_hz = 440.0
    phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

    clear_basis_atlas()
    basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

    mb_state = ModeBState()
    mb_state.reset(note=note, phase_inc=phase_inc)

    # Set custom harmonics
    mb_state.weights.set_custom(harmonic_amps)
    mb_state.H_active = len(harmonic_amps)

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
    """Check for 50 Hz component."""
    spectrum = np.abs(rfft(samples))
    freqs = rfftfreq(len(samples), 1/SAMPLE_RATE)

    # Find 50 Hz component
    idx_50 = int(50 * len(samples) / SAMPLE_RATE)
    idx_440 = int(440 * len(samples) / SAMPLE_RATE)

    amp_50 = spectrum[idx_50]
    amp_440 = spectrum[idx_440]

    rel_db = 20 * np.log10(amp_50 / amp_440) if amp_440 > 0 else -100

    print(f'{name}: 50 Hz = {rel_db:.1f} dB relative to 440 Hz')

# Test 1: Pure sine (H1 only)
print("Test 1: Pure sine (H1 only)")
samples = render_with_harmonics([32767])
analyze_50hz(samples, "H1 only")

# Test 2: H1 + H2
print("\nTest 2: H1 + H2")
samples = render_with_harmonics([32767, 32767])
analyze_50hz(samples, "H1 + H2")

# Test 3: Full violin profile
print("\nTest 3: Full violin profile")
violin_harmonics = [30890, 32767, 6926, 23969, 3673, 6138, 1164, 1228]
samples = render_with_harmonics(violin_harmonics)
analyze_50hz(samples, "Full violin")

# Test 4: H1 + H4 only (body resonance)
print("\nTest 4: H1 + H4 only")
samples = render_with_harmonics([32767, 0, 0, 32767])
analyze_50hz(samples, "H1 + H4")
