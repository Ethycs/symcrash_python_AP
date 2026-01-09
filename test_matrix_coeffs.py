"""Test matrix coefficient methods."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.matrix_coeffs import (
    WaveformCoeffBank, HarmonicCoeffMatrix,
    get_waveform_coeff_bank, get_harmonic_coeff_matrix,
    clear_matrix_coeff_cache
)
from symcrash_python_ap.vm.waveform_matrix import WaveformType
from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, mode_b_generate_coeffs, get_basis_for_note, PHASE_MASK
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc
from symcrash_python_ap.fixed import Q15_ONE

print("=== Testing Matrix Coefficient Methods ===\n")

# Clear caches
clear_matrix_coeff_cache()

note = 60
freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)

# =============================================================================
# Test 1: WaveformCoeffBank (direct c-vector coefficients)
# =============================================================================
print("1. Generating WaveformCoeffBank for square wave...")
coeff_bank = WaveformCoeffBank.generate(WaveformType.SQUARE, phase_inc, num_bins=256)
print(f"   Shape: {coeff_bank.coeffs.shape}")
print(f"   Memory: {coeff_bank.coeffs.nbytes / 1024:.1f} KB")

# =============================================================================
# Test 2: HarmonicCoeffMatrix (harmonic basis coefficients)
# =============================================================================
print("\n2. Generating HarmonicCoeffMatrix...")
harm_matrix = HarmonicCoeffMatrix.generate(phase_inc, H_max=32, num_phase_bins=64)
print(f"   Shape: {harm_matrix.basis.shape}")
print(f"   Memory: {harm_matrix.basis.nbytes / 1024:.1f} KB")

# =============================================================================
# Test 3: Compare output quality
# =============================================================================
print("\n3. Comparing output quality (square wave)...")

def analyze_method(name, get_coeffs_fn, num_blocks=50):
    """Generate blocks and analyze quality."""
    all_samples = []
    phase = 0

    for _ in range(num_blocks):
        coeffs = get_coeffs_fn(phase)
        audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)
        all_samples.extend(audio)
        phase = (phase + 64 * phase_inc) & PHASE_MASK

    samples = np.array(all_samples, dtype=np.int16)

    # Skip settling
    samples = samples[1000:]

    # Boundary analysis
    boundary_diffs = []
    within_diffs = []
    for i in range(num_blocks - 2):
        idx = (i + 1) * 64 - 1 - 1000 // 64 * 64
        if 0 <= idx < len(samples) - 1:
            boundary_diffs.append(abs(int(samples[idx + 1]) - int(samples[idx])))

    for i in range(min(1000, len(samples) - 1)):
        if i % 64 != 63:
            within_diffs.append(abs(int(samples[i + 1]) - int(samples[i])))

    avg_boundary = np.mean(boundary_diffs) if boundary_diffs else 0
    avg_within = np.mean(within_diffs) if within_diffs else 1
    ratio = avg_boundary / avg_within if avg_within > 0 else 0

    # Flat region analysis
    top_mask = samples > 0.8 * np.max(samples)
    top_std = np.std(samples[top_mask]) if np.sum(top_mask) > 0 else 0

    print(f"   {name}:")
    print(f"     Boundary/Within ratio: {ratio:.2f}x")
    print(f"     Flat region std: {top_std:.1f}")

    return samples

# Method 1: WaveformCoeffBank
def get_bank_coeffs(phase):
    return coeff_bank.sample(phase)

samples_bank = analyze_method("WaveformCoeffBank", get_bank_coeffs)

# Method 2: HarmonicCoeffMatrix with square wave weights
a_h_square = np.zeros(32, dtype=np.int16)
b_h_square = np.zeros(32, dtype=np.int16)
for h in range(1, 33, 2):  # Odd harmonics
    a_h_square[h-1] = int(Q15_ONE / h)

def get_harmonic_coeffs(phase):
    return harm_matrix.synthesize(phase, a_h_square, b_h_square)

samples_harmonic = analyze_method("HarmonicCoeffMatrix", get_harmonic_coeffs)

# Method 3: Original Mode B harmonic bank
mb_state = ModeBState()
mb_state.reset(note=60, phase_inc=phase_inc)
mb_state.weights.set_square(H=32)
mb_state.H_active = 32
basis = get_basis_for_note(60)

def get_original_coeffs(phase):
    mb_state.phase = phase
    coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(mb_state, basis, coeffs)
    return coeffs.reshape(64, 8)[:, 0]

samples_original = analyze_method("Original Mode B", get_original_coeffs)

# =============================================================================
# Test 4: Visual comparison of first block
# =============================================================================
print("\n4. First block sample values:")

phase = 0
c_bank = coeff_bank.sample(phase)
c_harm = harm_matrix.synthesize(phase, a_h_square, b_h_square)
c_orig = get_original_coeffs(phase)

audio_bank = idwpt_jit(c_bank.tolist(), headroom_bits=1)
audio_harm = idwpt_jit(c_harm.tolist(), headroom_bits=1)
audio_orig = idwpt_jit(c_orig.tolist(), headroom_bits=1)

print("   Sample 0-7:")
print(f"     CoeffBank:  {list(audio_bank[:8])}")
print(f"     HarmMatrix: {list(audio_harm[:8])}")
print(f"     Original:   {list(audio_orig[:8])}")

print("\n   Sample 60-63 (end of block):")
print(f"     CoeffBank:  {list(audio_bank[60:64])}")
print(f"     HarmMatrix: {list(audio_harm[60:64])}")
print(f"     Original:   {list(audio_orig[60:64])}")

print("\n=== Test Complete ===")
