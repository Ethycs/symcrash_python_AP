"""Debug: Is coefficient saturation causing the Mode B divergence?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.wavelet.sine_core import freq_to_phase_inc, sin_cos_q15, PHASE_MASK
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.vm.harmonic_bank import generate_basis_frames, NUM_LEAVES, BLOCK_SIZE
from symcrash_python_ap.fixed import Q15_ONE, RSHIFT_RNE, SAT16

SAMPLE_RATE = 48000

print("=" * 70)
print("SATURATION ANALYSIS")
print("=" * 70)

note = 69
freq_hz = 440.0
phase_inc = freq_to_phase_inc(freq_hz, SAMPLE_RATE)
basis = generate_basis_frames(note, H_max=8)

for h in range(1, 9):
    usin = basis.Usin_h[h-1, :, 0]
    ucos = basis.Ucos_h[h-1, :, 0]

    sat_sin = np.sum(np.abs(usin) >= 32767)
    sat_cos = np.sum(np.abs(ucos) >= 32767)

    print(f"H{h}: Usin range [{np.min(usin):>7}, {np.max(usin):>6}] sat={sat_sin:>2}, "
          f"Ucos range [{np.min(ucos):>7}, {np.max(ucos):>6}] sat={sat_cos:>2}")

print("\n" + "=" * 70)
print("H5 SATURATED LEAVES DETAIL")
print("=" * 70)

h = 5
usin = basis.Usin_h[h-1, :, 0]
ucos = basis.Ucos_h[h-1, :, 0]

print(f"\nSaturated sin leaves:")
for leaf in range(NUM_LEAVES):
    if abs(usin[leaf]) >= 32767:
        print(f"  Leaf {leaf}: Usin={usin[leaf]}, Ucos={ucos[leaf]}")

print(f"\nSaturated cos leaves:")
for leaf in range(NUM_LEAVES):
    if abs(ucos[leaf]) >= 32767:
        print(f"  Leaf {leaf}: Usin={usin[leaf]}, Ucos={ucos[leaf]}")

# Test: what if we use headroom_bits=2 instead of 1?
print("\n" + "=" * 70)
print("TEST: MORE HEADROOM BITS")
print("=" * 70)

# Regenerate basis with more headroom
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit as fwpt

# Generate H5 basis with different headroom
for headroom in [1, 2, 3]:
    sin_block = []
    cos_block = []
    phase = 0
    for n in range(BLOCK_SIZE):
        h_phase = (phase * h) & PHASE_MASK
        sin_val, cos_val = sin_cos_q15(h_phase)
        sin_block.append(sin_val)
        cos_block.append(cos_val)
        phase = (phase + phase_inc) & PHASE_MASK

    sin_coeffs = np.array(fwpt(sin_block, headroom_bits=headroom), dtype=np.int16)
    cos_coeffs = np.array(fwpt(cos_block, headroom_bits=headroom), dtype=np.int16)

    sat_sin = np.sum(np.abs(sin_coeffs) >= 32767)
    sat_cos = np.sum(np.abs(cos_coeffs) >= 32767)

    print(f"headroom_bits={headroom}: sin range [{np.min(sin_coeffs):>7}, {np.max(sin_coeffs):>6}] sat={sat_sin:>2}, "
          f"cos range [{np.min(cos_coeffs):>7}, {np.max(cos_coeffs):>6}] sat={sat_cos:>2}")

print("\n" + "=" * 70)
print("ROOT CAUSE ANALYSIS")
print("=" * 70)
print("""
The basis coefficients for H5 are SATURATING (hitting +/-32767).

When we modulate:
    coeff = w_sin*Usin + w_cos*Ucos

If Usin or Ucos is at +/-32767, the linear combination can't produce
the correct intermediate values because the original coefficient
information was clipped.

However, the saturation count shows:
- H1: 0 saturated
- H5: 1 saturated sin, 2 saturated cos

This doesn't explain the RMS=2850 error with only 3 saturated coefficients.

Let me check if the issue is QUANTIZATION rather than saturation...
""")

print("\n" + "=" * 70)
print("QUANTIZATION ANALYSIS")
print("=" * 70)

# The real issue: even without saturation, the FWPT quantizes to int16.
# The MODE B modulation assumes perfect linear combination, but:
#   FWPT(sin(h*theta)*cos(h*phi)) != FWPT(sin(h*theta)) * cos(h*phi)
#
# Because the FWPT has internal quantization at each level!

# Let's verify by using float precision FWPT
def fwpt_float(x):
    """Float-precision FWPT for comparison."""
    from symcrash_python_ap.wavelet.fast_lifting import FastCDF53

    def forward_float(x):
        n = len(x) // 2
        if n == 0:
            return x, []
        s = [0.0] * n
        d = [0.0] * n

        for i in range(n):
            left = x[2 * i]
            right = x[2 * i + 2] if 2 * i + 2 < len(x) else x[2 * i]
            avg = (left + right) / 2
            d[i] = x[2 * i + 1] - avg

        for i in range(n):
            d_left = d[i - 1] if i > 0 else d[0]
            d_right = d[i]
            update = (d_left + d_right) / 4
            s[i] = x[2 * i] + update

        return s, d

    queue = [list(x)]
    for _ in range(6):
        next_queue = []
        for node in queue:
            if len(node) >= 2:
                s, d = forward_float(node)
                next_queue.append(s)
                next_queue.append(d)
            else:
                next_queue.append(node)
        queue = next_queue

    return [node[0] for node in queue]

# Generate sin and cos at phase 0
sin_block = []
cos_block = []
phase = 0
for n in range(BLOCK_SIZE):
    h_phase = (phase * h) & PHASE_MASK
    sin_val, cos_val = sin_cos_q15(h_phase)
    sin_block.append(float(sin_val))
    cos_block.append(float(cos_val))
    phase = (phase + phase_inc) & PHASE_MASK

# Float FWPT of sin and cos
sin_coeffs_float = np.array(fwpt_float(sin_block))
cos_coeffs_float = np.array(fwpt_float(cos_block))

# Now test: at phase=0.125, what SHOULD the coefficients be?
test_phase = int(0.125 * (1 << 32))
h_phase = (test_phase * h) & PHASE_MASK
sin_phi, cos_phi = sin_cos_q15(h_phase)
sin_phi_f = sin_phi / 32767.0
cos_phi_f = cos_phi / 32767.0

print(f"At phase 0.125:")
print(f"  h*phi = {h}*0.125 = {h*0.125} cycles")
print(f"  sin(h*phi) = {sin_phi_f:.4f}, cos(h*phi) = {cos_phi_f:.4f}")

# Generate target waveform at this phase
target_block = []
phase = test_phase
for n in range(BLOCK_SIZE):
    h_phase = (phase * h) & PHASE_MASK
    sin_val, _ = sin_cos_q15(h_phase)
    target_block.append(float(sin_val))
    phase = (phase + phase_inc) & PHASE_MASK

# Float FWPT of target
target_coeffs_float = np.array(fwpt_float(target_block))

# Mode B prediction (using float basis)
mode_b_float = cos_phi_f * sin_coeffs_float + sin_phi_f * cos_coeffs_float

# Compare
err = target_coeffs_float - mode_b_float
print(f"\nFloat-precision comparison:")
print(f"  Mode B prediction error (float): RMS={np.sqrt(np.mean(err**2)):.1f}")

# Now with integer quantization
sin_coeffs_int = np.array(fwpt(sin_block, headroom_bits=1), dtype=np.int16).astype(np.float64)
cos_coeffs_int = np.array(fwpt(cos_block, headroom_bits=1), dtype=np.int16).astype(np.float64)
target_coeffs_int = np.array(fwpt([int(x) for x in target_block], headroom_bits=1), dtype=np.int16).astype(np.float64)

mode_b_int = cos_phi_f * sin_coeffs_int + sin_phi_f * cos_coeffs_int
err_int = target_coeffs_int - mode_b_int
print(f"  Mode B prediction error (int16): RMS={np.sqrt(np.mean(err_int**2)):.1f}")

print("\nThe float error should be ~0 if the trig identity works.")
print("If float error is high, the FWPT's boundary behavior breaks the identity.")
