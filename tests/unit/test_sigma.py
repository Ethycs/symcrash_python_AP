"""Test different mollifier windows on Gibbs ringing."""
import sys
import math
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, mode_b_generate_coeffs, get_basis_for_note
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc
from symcrash_python_ap.fixed import Q15_ONE, SAT16

freq_hz = 261.63
phase_inc = freq_to_phase_inc(freq_hz, 48000)
basis = get_basis_for_note(60)

def window_none(h, H):
    return 1.0

def window_sigma(h, H):
    """Lanczos sigma: sinc(h/H)"""
    if h >= H: return 0.0
    x = math.pi * h / H
    return math.sin(x) / x

def window_sigma2(h, H):
    """Fejer: sinc^2(h/H) - stronger smoothing"""
    if h >= H: return 0.0
    x = math.pi * h / H
    s = math.sin(x) / x
    return s * s

def window_raised_cos(h, H):
    """Raised cosine: 0.5 + 0.5*cos(pi*h/H)"""
    if h >= H: return 0.0
    return 0.5 + 0.5 * math.cos(math.pi * h / H)

def window_gaussian(h, H, alpha=2.5):
    """Gaussian: exp(-0.5 * (alpha*h/H)^2)"""
    return math.exp(-0.5 * (alpha * h / H) ** 2)

def test_window(window_fn, name, num_blocks=100):
    mb_state = ModeBState()
    mb_state.reset(note=60, phase_inc=phase_inc)

    # Manually set weights with window
    H = 32
    mb_state.weights.dc_q15 = 0
    mb_state.weights.a_h_q15.fill(0)
    mb_state.weights.b_h_q15.fill(0)
    for h in range(1, H + 1):
        if h % 2 == 1:  # Odd harmonics
            amplitude = (1.0 / h) * window_fn(h, H)
            mb_state.weights.a_h_q15[h - 1] = SAT16(int(amplitude * Q15_ONE))
    mb_state.weights.H_active = H
    mb_state.H_active = H

    all_samples = []
    for _ in range(num_blocks):
        coeffs = np.zeros(512, dtype=np.int16)
        mode_b_generate_coeffs(mb_state, basis, coeffs)
        audio = idwpt_jit(coeffs.reshape(64, 8)[:, 0].tolist(), headroom_bits=1)
        all_samples.extend(audio)
        mb_state.advance_block()

    samples = np.array(all_samples[1000:], dtype=np.int16)
    max_val = np.max(np.abs(samples))
    high_mask = samples > 0.7 * max_val
    flat_std = np.std(samples[high_mask]) if np.sum(high_mask) > 0 else 0
    return flat_std, np.max(samples)

print("MOLLIFIER WINDOW COMPARISON")
print("=" * 50)
print(f"{'Window':<20} {'Flat Std':>10} {'Peak':>10}")
print("-" * 50)

windows = [
    (window_none, "None (raw)"),
    (window_sigma, "Sigma (sinc)"),
    (window_sigma2, "Sigma^2 (Fejer)"),
    (window_raised_cos, "Raised Cosine"),
    (window_gaussian, "Gaussian"),
]

baseline_std = None
for fn, name in windows:
    std, peak = test_window(fn, name)
    if baseline_std is None:
        baseline_std = std
    ratio = baseline_std / std if std > 0 else 0
    print(f"{name:<20} {std:>10.1f} {peak:>10}  ({ratio:.1f}x)")

# Compare to direct synthesis (no wavelet)
print("\n" + "=" * 50)
print("REFERENCE: Direct harmonic synthesis (no wavelet)")
print("=" * 50)

from symcrash_python_ap.wavelet.sine_core import sin_cos_q15

def direct_synth(window_fn, num_samples=5400):
    """Generate square wave by direct harmonic sum."""
    samples = []
    H = 32
    phase = 0
    for _ in range(num_samples):
        val = 0
        for h in range(1, H + 1, 2):  # Odd harmonics
            weight = (1.0 / h) * window_fn(h, H)
            h_phase = (phase * h) & 0xFFFFFFFF
            s, _ = sin_cos_q15(h_phase)
            val += int(s * weight)
        samples.append(np.clip(val, -32768, 32767))
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return np.array(samples, dtype=np.int16)

for fn, name in windows:
    samples = direct_synth(fn)
    max_val = np.max(np.abs(samples))
    high_mask = samples > 0.7 * max_val
    flat_std = np.std(samples[high_mask]) if np.sum(high_mask) > 0 else 0
    print(f"{name:<20} flat_std={flat_std:.1f}")

