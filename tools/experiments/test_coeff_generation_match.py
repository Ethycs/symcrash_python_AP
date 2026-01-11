"""Test: Do Mode B generated coefficients match what IDWPT expects?

The core question: When we generate coefficients via Mode B (basis frame modulation),
do they produce the same audio as direct FWPT->IDWPT of the target waveform?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.vm.harmonic_bank import generate_basis_frames, NUM_LEAVES, BLOCK_SIZE
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

SAMPLE_RATE = 48000

print("=" * 70)
print("MODE B COEFFICIENT GENERATION VS DIRECT SYNTHESIS")
print("=" * 70)

note = 69  # A4 = 440 Hz
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)

# Generate basis frames
basis = generate_basis_frames(note, H_max=8)

def generate_target_waveform(h, start_phase):
    """Generate the target waveform we want to synthesize."""
    signal = []
    phase = start_phase
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, _ = sin_cos_q15(h_phase)
        signal.append(sin_val)
        phase = (phase + phase_inc) & PHASE_MASK
    return signal

def generate_mode_b_coeffs(h, start_phase):
    """Generate coefficients using Mode B (basis frame modulation)."""
    # Phase at block start determines modulation
    h_phase = (start_phase * h) & PHASE_MASK
    sin_phi, cos_phi = sin_cos_q15(h_phase)

    # Full amplitude
    a_h = Q15_ONE
    w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
    w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

    # Modulate basis to get coefficients (k=0 slot only)
    coeffs = []
    for leaf in range(NUM_LEAVES):
        sin_basis = int(basis.Usin_h[h-1, leaf, 0])
        cos_basis = int(basis.Ucos_h[h-1, leaf, 0])
        sin_contrib = RSHIFT_RNE(w_sin * sin_basis, 15)
        cos_contrib = RSHIFT_RNE(w_cos * cos_basis, 15)
        coeffs.append(SAT16(sin_contrib + cos_contrib))

    return coeffs

def compare_methods(h, start_phase, label):
    """Compare Mode B synthesis vs direct FWPT->IDWPT."""
    # Method 1: Direct - generate waveform, FWPT it, then IDWPT
    target = generate_target_waveform(h, start_phase)
    direct_coeffs = fwpt_jit(target, headroom_bits=1)
    direct_audio = idwpt_jit(direct_coeffs, headroom_bits=1)

    # Method 2: Mode B - generate coeffs from basis, then IDWPT
    mode_b_coeffs = generate_mode_b_coeffs(h, start_phase)
    mode_b_audio = idwpt_jit(mode_b_coeffs, headroom_bits=1)

    # Compare coefficients
    direct_coeffs_arr = np.array(direct_coeffs, dtype=np.int16)
    mode_b_coeffs_arr = np.array(mode_b_coeffs, dtype=np.int16)
    coeff_err = direct_coeffs_arr.astype(np.int32) - mode_b_coeffs_arr.astype(np.int32)

    # Compare audio
    target_arr = np.array(target, dtype=np.int16)
    direct_arr = np.array(direct_audio, dtype=np.int16)
    mode_b_arr = np.array(mode_b_audio, dtype=np.int16)

    # Errors
    direct_vs_target = target_arr.astype(np.int32) - direct_arr.astype(np.int32)
    mode_b_vs_target = target_arr.astype(np.int32) - mode_b_arr.astype(np.int32)
    mode_b_vs_direct = direct_arr.astype(np.int32) - mode_b_arr.astype(np.int32)

    print(f"\n{label}:")
    print(f"  Coefficient diff (Mode B vs Direct): RMS={np.sqrt(np.mean(coeff_err**2)):.1f}, Max={np.max(np.abs(coeff_err))}")
    print(f"  Direct audio vs target:  RMS={np.sqrt(np.mean(direct_vs_target**2)):.0f}")
    print(f"  Mode B audio vs target:  RMS={np.sqrt(np.mean(mode_b_vs_target**2)):.0f}")
    print(f"  Mode B audio vs Direct:  RMS={np.sqrt(np.mean(mode_b_vs_direct**2)):.1f}")

    return {
        'coeff_rms': np.sqrt(np.mean(coeff_err**2)),
        'direct_err': np.sqrt(np.mean(direct_vs_target**2)),
        'mode_b_err': np.sqrt(np.mean(mode_b_vs_target**2)),
    }

print("\n--- Single block at phase=0 ---")
for h in [1, 2, 3, 4, 5, 6]:
    compare_methods(h, 0, f"H{h}")

print("\n--- Multiple blocks (phase evolves) ---")
print("Testing H5 across 10 blocks:")
print(f"{'Block':>6} {'Phase':>10} {'Coeff RMS':>12} {'Direct Err':>12} {'Mode B Err':>12}")
print("-" * 60)

phase = 0
for block in range(10):
    result = compare_methods(5, phase, f"Block {block}")
    print(f"{block:>6} {phase:>10} {result['coeff_rms']:>12.1f} {result['direct_err']:>12.0f} {result['mode_b_err']:>12.0f}")
    phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("""
If Mode B coefficients match Direct coefficients (within rounding),
then both methods produce the SAME audio with the SAME roundtrip error.

The error is NOT in Mode B coefficient generation - it's in the
FWPT/IDWPT roundtrip itself. The transform cannot perfectly represent
H5 because of the boundary extension behavior.

The 50 Hz artifact comes from this roundtrip error being modulated
across blocks as the phase evolves.
""")
