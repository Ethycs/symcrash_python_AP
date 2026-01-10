"""Debug and fix the interleave formula."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.fft import rfft, rfftfreq

from symcrash_python_ap.vm.harmonic_bank import (
    generate_basis_frames, BLOCK_SIZE, SAMPLE_RATE, VM_K, K_STRIDE, NUM_LEAVES
)
from symcrash_python_ap.vm.tables import COEFF_LEN
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.fixed import RSHIFT_RNE, SAT16

# Parameters
freq_hz = 440.0
note = 69
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
H = 5

print("=" * 70)
print("INTERLEAVE FORMULA ANALYSIS")
print("=" * 70)

# Generate basis frames
basis = generate_basis_frames(anchor_note=note, H_max=8, k_stride=K_STRIDE)

print(f"\nK_STRIDE = {K_STRIDE}")
print(f"This means basis frame for slot k is generated with phase offset of k samples")

print("""
Current interleave:
    out[k::K] = idwpt(slot_k)[0::K]

This means:
    out[0] = idwpt(slot_0)[0]    # sample at phase offset 0
    out[1] = idwpt(slot_1)[0]    # sample at phase offset 1 (from basis)
    out[2] = idwpt(slot_2)[0]    # sample at phase offset 2 (from basis)
    out[3] = idwpt(slot_3)[0]    # sample at phase offset 3 (from basis)
    out[4] = idwpt(slot_0)[4]    # sample 4 from slot 0
    out[5] = idwpt(slot_1)[4]    # sample 4 from slot 1 (basis offset 1)
    ...

The basis frame for slot k assumes the output sample at position n has
time offset k + n samples from block start.

So:
    idwpt(slot_k)[n] should be used for output sample k + n

For out[k::K] = idwpt(slot_k)[0::K], we're taking:
    out[k + i*K] = idwpt(slot_k)[i*K]

This maps:
    out[0] = idwpt(slot_0)[0]    # correct: time 0
    out[1] = idwpt(slot_1)[0]    # want time 1, basis has offset 1, so this is time 1 -> correct!
    out[4] = idwpt(slot_0)[4]    # want time 4, slot 0 has offset 0, sample 4 is time 4 -> correct!
    out[5] = idwpt(slot_1)[4]    # want time 5, slot 1 has offset 1, sample 4 is time 5 -> correct!

Wait, this should be correct...
""")

# Let's verify by generating a known sine and checking reconstruction
print("\n" + "=" * 70)
print("VERIFICATION: Generate sine directly and check reconstruction")
print("=" * 70)

def generate_reference_h5_sine(num_samples, start_phase=0):
    """Generate reference H5 sine wave."""
    samples = np.zeros(num_samples, dtype=np.float64)
    phase = start_phase
    for i in range(num_samples):
        h_phase = (phase * 5) & PHASE_MASK
        sin_val, cos_val = sin_cos_q15(h_phase)
        # Use sine only for clean test
        samples[i] = sin_val
        phase = (phase + phase_inc) & PHASE_MASK
    return samples


def generate_h5_coeffs_for_block(base_phase, a_h=32767):
    """Generate H5 coefficients for all K slots at a given phase."""
    h = 5
    coeffs = np.zeros((NUM_LEAVES, VM_K), dtype=np.int16)

    for k in range(VM_K):
        phi = base_phase
        h_phase = (phi * h) & PHASE_MASK
        sin_phi, cos_phi = sin_cos_q15(h_phase)
        w_sin = RSHIFT_RNE(a_h * cos_phi, 15)
        w_cos = RSHIFT_RNE(a_h * sin_phi, 15)

        for leaf in range(NUM_LEAVES):
            sin_basis = int(basis.Usin_h[h-1, leaf, k])
            cos_basis = int(basis.Ucos_h[h-1, leaf, k])
            sin_contrib = RSHIFT_RNE(w_sin * sin_basis, 15)
            cos_contrib = RSHIFT_RNE(w_cos * cos_basis, 15)
            coeffs[leaf, k] = SAT16(sin_contrib + cos_contrib)

    return coeffs


# Test different interleave formulas
def interleave_current(slot_outputs):
    """Current formula: out[k::K] = audio_k[0::K]"""
    K = len(slot_outputs)
    out = np.zeros(64, dtype=np.int16)
    for k in range(K):
        out[k::K] = slot_outputs[k][0::K]
    return out


def interleave_alternative(slot_outputs):
    """Alternative formula: out[k::K] = audio_k[k::K]"""
    K = len(slot_outputs)
    out = np.zeros(64, dtype=np.int16)
    for k in range(K):
        out[k::K] = slot_outputs[k][k::K]
    return out


def interleave_direct(slot_outputs):
    """Direct: out[n] = audio_{n%K}[n]"""
    K = len(slot_outputs)
    out = np.zeros(64, dtype=np.int16)
    for n in range(64):
        k = n % K
        out[n] = slot_outputs[k][n]
    return out


# Generate one block and compare
print("\nGenerating one block of H5...")
phase = 0
coeffs = generate_h5_coeffs_for_block(phase)

# Get IDWPT outputs for each slot
slot_outputs = []
for k in range(VM_K):
    coeffs_k = coeffs[:, k].astype(np.int16).tolist()
    audio_k = np.array(idwpt_jit(coeffs_k, headroom_bits=1), dtype=np.int16)
    slot_outputs.append(audio_k)

# Compare different interleaves
out_current = interleave_current(slot_outputs)
out_alt = interleave_alternative(slot_outputs)
out_direct = interleave_direct(slot_outputs)

# Generate reference
ref = generate_reference_h5_sine(64, start_phase=phase)

print(f"\nFirst 16 samples comparison:")
print(f"Reference:        {ref[:16].astype(int).tolist()}")
print(f"Current formula:  {out_current[:16].tolist()}")
print(f"Alternative:      {out_alt[:16].tolist()}")
print(f"Direct:           {out_direct[:16].tolist()}")

# Compute errors
err_current = np.sqrt(np.mean((ref - out_current.astype(np.float64))**2))
err_alt = np.sqrt(np.mean((ref - out_alt.astype(np.float64))**2))
err_direct = np.sqrt(np.mean((ref - out_direct.astype(np.float64))**2))

print(f"\nRMS Error vs Reference:")
print(f"  Current formula: {err_current:.1f}")
print(f"  Alternative:     {err_alt:.1f}")
print(f"  Direct:          {err_direct:.1f}")

# =============================================================================
# Test over multiple blocks with each formula
# =============================================================================

print("\n" + "=" * 70)
print("MULTI-BLOCK TEST WITH DIFFERENT INTERLEAVES")
print("=" * 70)

def render_blocks(interleave_func, num_blocks=100):
    """Render multiple blocks with a specific interleave function."""
    phase = 0
    all_samples = []

    for _ in range(num_blocks):
        coeffs = generate_h5_coeffs_for_block(phase)

        slot_outputs = []
        for k in range(VM_K):
            coeffs_k = coeffs[:, k].astype(np.int16).tolist()
            audio_k = np.array(idwpt_jit(coeffs_k, headroom_bits=1), dtype=np.int16)
            slot_outputs.append(audio_k)

        out = interleave_func(slot_outputs)
        all_samples.extend(out.tolist())

        phase = (phase + BLOCK_SIZE * phase_inc) & PHASE_MASK

    return np.array(all_samples, dtype=np.float64)


def analyze_50hz(samples, name):
    """Analyze 50 Hz artifact."""
    n = len(samples)
    hann = np.hanning(n)
    windowed = samples * hann
    spectrum = np.abs(rfft(windowed)) * (2.0 / np.mean(hann))
    freqs = rfftfreq(n, 1/SAMPLE_RATE)

    idx_50 = int(50 * n / SAMPLE_RATE)
    idx_h5 = int(2200 * n / SAMPLE_RATE)

    amp_50 = spectrum[idx_50]
    amp_h5 = spectrum[idx_h5] if idx_h5 < len(spectrum) else 1

    rel_db = 20 * np.log10(amp_50 / amp_h5) if amp_h5 > 0 else -100
    print(f"{name}: 50 Hz = {rel_db:.1f} dB relative to H5")


print("\nTesting interleave formulas over 100 blocks:")
samples_current = render_blocks(interleave_current)
analyze_50hz(samples_current, "Current [k::K] = audio[0::K]")

samples_alt = render_blocks(interleave_alternative)
analyze_50hz(samples_alt, "Alt [k::K] = audio[k::K]")

samples_direct = render_blocks(interleave_direct)
analyze_50hz(samples_direct, "Direct out[n] = audio[n%K][n]")
