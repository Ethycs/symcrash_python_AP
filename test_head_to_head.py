"""Head-to-head comparison: Mode B vs WaveformCoeffBank vs HarmonicCoeffMatrix.
Focus on block banding artifacts."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.matrix_coeffs import (
    WaveformCoeffBank, HarmonicCoeffMatrix,
    clear_matrix_coeff_cache
)
from symcrash_python_ap.vm.waveform_matrix import WaveformType
from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, mode_b_generate_coeffs, get_basis_for_note, PHASE_MASK,
    clear_basis_atlas
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc
from symcrash_python_ap.fixed import Q15_ONE

print("=" * 70)
print("HEAD-TO-HEAD: Mode B vs WaveformCoeffBank vs HarmonicCoeffMatrix")
print("=" * 70)

# Clear caches for fresh comparison
clear_matrix_coeff_cache()
clear_basis_atlas()

note = 60
freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)
num_blocks = 100

print(f"\nFrequency: {freq_hz} Hz (C4)")
print(f"Blocks: {num_blocks} (= {num_blocks * 64 / 48000:.2f}s)")

# =============================================================================
# Setup methods
# =============================================================================

print("\n--- Setting up methods ---")

# 1. Original Mode B
print("1. Mode B (harmonic bank with basis frames)")
mb_state = ModeBState()
mb_state.reset(note=60, phase_inc=phase_inc)
mb_state.weights.set_square(H=32)
mb_state.H_active = 32
basis = get_basis_for_note(60)

# 2. WaveformCoeffBank (c-vector coefficients)
print("2. WaveformCoeffBank (precomputed c-vector FWPT)")
coeff_bank = WaveformCoeffBank.generate(WaveformType.SQUARE, phase_inc, num_bins=256)

# 3. HarmonicCoeffMatrix (harmonic basis in coeff domain)
print("3. HarmonicCoeffMatrix (harmonic basis coefficients)")
harm_matrix = HarmonicCoeffMatrix.generate(phase_inc, H_max=32, num_phase_bins=64)

# Square wave harmonic weights for HarmonicCoeffMatrix
a_h_square = np.zeros(32, dtype=np.int16)
b_h_square = np.zeros(32, dtype=np.int16)
for h in range(1, 33, 2):
    a_h_square[h-1] = int(Q15_ONE / h)

# =============================================================================
# Generate samples
# =============================================================================

print("\n--- Generating samples ---")

def generate_all(name, get_coeffs_fn):
    all_samples = []
    phase = 0
    for block_idx in range(num_blocks):
        coeffs = get_coeffs_fn(phase)
        audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)
        all_samples.extend(audio)
        phase = (phase + 64 * phase_inc) & PHASE_MASK
    return np.array(all_samples, dtype=np.int16)

def get_modeb(phase):
    mb_state.phase = phase
    coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(mb_state, basis, coeffs)
    return coeffs.reshape(64, 8)[:, 0]

def get_coeffbank(phase):
    return coeff_bank.sample(phase)

def get_harmmatrix(phase):
    return harm_matrix.synthesize(phase, a_h_square, b_h_square)

samples_modeb = generate_all("Mode B", get_modeb)
samples_bank = generate_all("CoeffBank", get_coeffbank)
samples_matrix = generate_all("HarmMatrix", get_harmmatrix)

print(f"Generated {len(samples_modeb)} samples each")

# =============================================================================
# Block Banding Analysis
# =============================================================================

print("\n" + "=" * 70)
print("BLOCK BANDING ANALYSIS")
print("=" * 70)

def analyze_banding(name, samples):
    """Analyze block-related artifacts."""
    # Skip first 1000 samples
    s = samples[1000:]

    # 1. Boundary discontinuity
    boundary_diffs = []
    within_diffs = []
    for i in range(len(s) - 1):
        diff = abs(int(s[i+1]) - int(s[i]))
        if (i + 1000) % 64 == 63:  # At block boundary
            boundary_diffs.append(diff)
        elif (i + 1000) % 64 == 31:  # Middle of block
            within_diffs.append(diff)

    avg_boundary = np.mean(boundary_diffs) if boundary_diffs else 0
    avg_within = np.mean(within_diffs) if within_diffs else 1
    ratio = avg_boundary / avg_within

    # 2. Per-block energy variation (banding shows as consistent energy pattern)
    block_energies = []
    for i in range(len(s) // 64):
        block = s[i*64:(i+1)*64]
        block_energies.append(np.mean(block.astype(np.float64)**2))

    energy_std = np.std(block_energies)

    # 3. FFT to find 750 Hz component (block rate = 48000/64 = 750 Hz)
    # If there's banding, we'll see energy at 750 Hz
    fft = np.abs(np.fft.rfft(s[:4096]))
    freqs = np.fft.rfftfreq(4096, 1/48000)

    # Find 750 Hz bin
    idx_750 = np.argmin(np.abs(freqs - 750))
    idx_fund = np.argmin(np.abs(freqs - freq_hz))

    band_750 = fft[idx_750]
    fund_energy = fft[idx_fund]
    band_ratio = band_750 / fund_energy * 100 if fund_energy > 0 else 0

    # 4. Flat region quality
    max_val = np.max(np.abs(s))
    high_mask = s > 0.7 * max_val
    high_std = np.std(s[high_mask]) if np.sum(high_mask) > 0 else 0

    print(f"\n{name}:")
    print(f"  Boundary/Within ratio: {ratio:.2f}x {'[!]' if ratio > 1.5 else '[ok]'}")
    print(f"  Block energy std:      {energy_std:.1f}")
    print(f"  750 Hz (block rate):   {band_ratio:.2f}% of fundamental {'[!]' if band_ratio > 1 else '[ok]'}")
    print(f"  Flat region std:       {high_std:.1f}")

    return {
        'boundary_ratio': ratio,
        'energy_std': energy_std,
        'band_750_pct': band_ratio,
        'flat_std': high_std
    }

results = {}
results['Mode B'] = analyze_banding("Mode B (harmonic bank)", samples_modeb)
results['CoeffBank'] = analyze_banding("WaveformCoeffBank", samples_bank)
results['HarmMatrix'] = analyze_banding("HarmonicCoeffMatrix", samples_matrix)

# =============================================================================
# Summary Table
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

print(f"\n{'Metric':<25} {'Mode B':>12} {'CoeffBank':>12} {'HarmMatrix':>12}")
print("-" * 65)
print(f"{'Boundary/Within':<25} {results['Mode B']['boundary_ratio']:>11.2f}x {results['CoeffBank']['boundary_ratio']:>11.2f}x {results['HarmMatrix']['boundary_ratio']:>11.2f}x")
print(f"{'Block energy std':<25} {results['Mode B']['energy_std']:>12.1f} {results['CoeffBank']['energy_std']:>12.1f} {results['HarmMatrix']['energy_std']:>12.1f}")
print(f"{'750 Hz (% fund)':<25} {results['Mode B']['band_750_pct']:>11.2f}% {results['CoeffBank']['band_750_pct']:>11.2f}% {results['HarmMatrix']['band_750_pct']:>11.2f}%")
print(f"{'Flat region std':<25} {results['Mode B']['flat_std']:>12.1f} {results['CoeffBank']['flat_std']:>12.1f} {results['HarmMatrix']['flat_std']:>12.1f}")

# Best in each category
print("\n" + "-" * 65)
print("BEST IN CATEGORY:")
metrics = ['boundary_ratio', 'energy_std', 'band_750_pct', 'flat_std']
metric_names = ['Boundary smoothness', 'Energy consistency', '750 Hz rejection', 'Flat region quality']

for metric, mname in zip(metrics, metric_names):
    best = min(results.keys(), key=lambda k: results[k][metric])
    print(f"  {mname}: {best}")

# =============================================================================
# Waveform preview
# =============================================================================

print("\n" + "=" * 70)
print("WAVEFORM SAMPLE VALUES (block boundary)")
print("=" * 70)

# Look at samples around a block boundary
boundary_idx = 1000 + 63  # End of a block (after skipping 1000)
print(f"\nSamples around block boundary at index ~{boundary_idx}:")
print(f"{'idx':>6} {'Mode B':>10} {'CoeffBank':>10} {'HarmMatrix':>10}")
print("-" * 45)

for i in range(boundary_idx - 3, boundary_idx + 5):
    marker = " <-- boundary" if i == boundary_idx else ""
    print(f"{i:6d} {samples_modeb[i]:10d} {samples_bank[i]:10d} {samples_matrix[i]:10d}{marker}")
