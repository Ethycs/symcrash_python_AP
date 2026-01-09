"""Compare square wave quality: Mode B harmonic bank vs WaveformCoeffBank."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.matrix_coeffs import WaveformCoeffBank
from symcrash_python_ap.vm.waveform_matrix import WaveformType
from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, mode_b_generate_coeffs, get_basis_for_note, PHASE_MASK
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc

print("=== Square Wave Quality Comparison ===\n")

note = 60
freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)

# Generate both
coeff_bank = WaveformCoeffBank.generate(WaveformType.SQUARE, phase_inc, num_bins=256)

mb_state = ModeBState()
mb_state.reset(note=60, phase_inc=phase_inc)
mb_state.weights.set_square(H=32)
mb_state.H_active = 32
basis = get_basis_for_note(60)

# Generate 100 blocks each
def generate_samples(name, get_coeffs_fn, num_blocks=100):
    all_samples = []
    phase = 0
    for _ in range(num_blocks):
        coeffs = get_coeffs_fn(phase)
        audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)
        all_samples.extend(audio)
        phase = (phase + 64 * phase_inc) & PHASE_MASK
    return np.array(all_samples, dtype=np.int16)

def get_bank_coeffs(phase):
    return coeff_bank.sample(phase)

def get_modeb_coeffs(phase):
    mb_state.phase = phase
    coeffs = np.zeros(512, dtype=np.int16)
    mode_b_generate_coeffs(mb_state, basis, coeffs)
    return coeffs.reshape(64, 8)[:, 0]

samples_bank = generate_samples("CoeffBank", get_bank_coeffs)
samples_modeb = generate_samples("Mode B", get_modeb_coeffs)

# Analyze the "flat top" region more carefully
# Skip first 1000 samples, look at samples in the "high" region
samples_bank = samples_bank[1000:]
samples_modeb = samples_modeb[1000:]

print("=== Sample-by-sample comparison in flat region ===")
print("(Looking for one complete period around sample 500)\n")

# Find a region where both should be "flat" (high part of square)
# Period is ~183 samples, so half period is ~91 samples

# Look at samples 500-550 which should be in a flat region
print("Sample range 500-520:")
print(f"{'idx':>5} {'CoeffBank':>10} {'Mode B':>10} {'Diff':>8}")
print("-" * 40)
for i in range(500, 520):
    diff = abs(int(samples_bank[i]) - int(samples_modeb[i]))
    print(f"{i:5d} {samples_bank[i]:10d} {samples_modeb[i]:10d} {diff:8d}")

# Compute running variance in flat regions
print("\n=== Running variance in high regions ===")

def compute_local_variance(samples, window=10):
    """Compute variance in sliding window for samples > 80% of max."""
    max_val = np.max(samples)
    variances = []
    for i in range(len(samples) - window):
        if all(samples[i:i+window] > 0.7 * max_val):
            variances.append(np.var(samples[i:i+window]))
    return variances

var_bank = compute_local_variance(samples_bank)
var_modeb = compute_local_variance(samples_modeb)

print(f"CoeffBank - Flat region windows: {len(var_bank)}")
if var_bank:
    print(f"  Mean variance: {np.mean(var_bank):.1f}")
    print(f"  Max variance:  {np.max(var_bank):.1f}")

print(f"\nMode B - Flat region windows: {len(var_modeb)}")
if var_modeb:
    print(f"  Mean variance: {np.mean(var_modeb):.1f}")
    print(f"  Max variance:  {np.max(var_modeb):.1f}")

# The key metric: How "flat" are the flat parts?
print("\n=== Flatness comparison ===")

# For a perfect square wave, the flat regions should have near-zero variance
# For bandlimited, there's Gibbs ringing
# The question: which is "cleaner" to the ear?

high_bank = samples_bank[samples_bank > 0.8 * np.max(samples_bank)]
high_modeb = samples_modeb[samples_modeb > 0.8 * np.max(samples_modeb)]

print(f"CoeffBank high region: n={len(high_bank)}, std={np.std(high_bank):.1f}")
print(f"Mode B high region:    n={len(high_modeb)}, std={np.std(high_modeb):.1f}")

# Actually show what the values look like
print("\n=== Distribution of values in high region ===")
print("CoeffBank unique values (top 10 most common):")
unique_bank, counts_bank = np.unique(high_bank, return_counts=True)
for v, c in sorted(zip(counts_bank, unique_bank), reverse=True)[:10]:
    print(f"  {c:5d}: {int(unique_bank[np.where(counts_bank == v)[0][0]])}")

print("\nMode B unique values (top 10 most common):")
unique_modeb, counts_modeb = np.unique(high_modeb, return_counts=True)
for v, c in sorted(zip(counts_modeb, unique_modeb), reverse=True)[:10]:
    print(f"  {c:5d}: {int(unique_modeb[np.where(counts_modeb == v)[0][0]])}")
