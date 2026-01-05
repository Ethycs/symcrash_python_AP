"""Three diagnostic tests to identify the 750 Hz comb source."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from scipy.fft import rfft, rfftfreq

# Clear cache
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()

from symcrash_python_ap.vm.coeff_lut_streaming import (
    get_lut_for_note, BLOCK_SIZE, NUM_LEAVES, DEFAULT_K
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15

SAMPLE_RATE = 48000
f0 = 440.0
K = 8

print("=" * 60)
print("DIAGNOSTIC TEST SUITE FOR 750 Hz COMB")
print("=" * 60)
print()

lut = get_lut_for_note(wave_id=0, midi_note=69)
phase_inc = int((f0 / SAMPLE_RATE) * (1 << 32))

TARGET_RMS_Q15 = 16384
Q15_ONE = 32767

def generate_ground_truth(n_samples):
    """Generate continuous ground truth sine."""
    output = np.empty(n_samples, dtype=np.int16)
    phase = 0
    for n in range(n_samples):
        sample = sine_q15(phase)
        sample = (sample * TARGET_RMS_Q15) // Q15_ONE
        output[n] = np.clip(sample, -32768, 32767)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return output

# Generate 1 second
duration = 1.0
n_samples = int(duration * SAMPLE_RATE)
n_blocks = n_samples // BLOCK_SIZE
valid_len = n_blocks * BLOCK_SIZE

ground_truth = generate_ground_truth(n_samples)

# ============================================================
# TEST 1: Measure spur level correctly (dBc, not dBFS)
# ============================================================
print("TEST 1: SPUR LEVEL (dBc)")
print("-" * 40)

# Generate CCS output
ccs_output = np.zeros(n_samples, dtype=np.int16)
phase_32 = 0
for block_idx in range(n_blocks):
    subband_output = np.zeros((NUM_LEAVES, K), dtype=np.int16)
    lut.get_subband_samples(phase_32, subband_output)

    idwpt_outputs = []
    for k in range(K):
        coeffs_k = [subband_output[l, k] for l in range(NUM_LEAVES)]
        audio_k = idwpt_jit(coeffs_k, headroom_bits=1)
        idwpt_outputs.append(audio_k)

    block_start = block_idx * BLOCK_SIZE
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            ccs_output[block_start + k + m * K] = idwpt_outputs[k][m * K]

    phase_32 = (phase_32 + phase_inc * BLOCK_SIZE) & 0xFFFFFFFF

# FFT (no window for clean measurement)
ccs_fft = rfft(ccs_output[:valid_len].astype(np.float64))
freqs = rfftfreq(valid_len, 1 / SAMPLE_RATE)

# Find carrier amplitude
fund_idx = np.argmin(np.abs(freqs - f0))
A_carrier = np.abs(ccs_fft[fund_idx])

print(f"Carrier (440 Hz) amplitude: {A_carrier:.0f}")
print()

# Measure spurs relative to carrier (dBc)
spur_freqs = [310, 1190, 6310, 7190, 12310, 18310]
print("Spur levels (dBc = 20*log10(A_spur/A_carrier)):")
for freq in spur_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    A_spur = np.abs(ccs_fft[idx])
    dBc = 20 * np.log10(A_spur / A_carrier + 1e-10)
    print(f"  {freq:5d} Hz: A={A_spur:8.0f}, dBc={dBc:+.1f}")

print()

# ============================================================
# TEST 2: Boundary delta test
# ============================================================
print("TEST 2: BOUNDARY DELTA TEST")
print("-" * 40)

# Value jump at each block boundary
delta_0 = []  # y[b*64] - y[b*64-1]
delta_1 = []  # slope jump

for b in range(1, n_blocks):
    boundary = b * BLOCK_SIZE
    # Value jump
    d0 = int(ccs_output[boundary]) - int(ccs_output[boundary - 1])
    delta_0.append(d0)

    # Slope jump
    if boundary >= 2:
        slope_before = int(ccs_output[boundary - 1]) - int(ccs_output[boundary - 2])
        slope_after = int(ccs_output[boundary]) - int(ccs_output[boundary - 1])
        d1 = slope_after - slope_before
        delta_1.append(d1)

delta_0 = np.array(delta_0)
delta_1 = np.array(delta_1)

print(f"Value jump (delta_0) at boundaries:")
print(f"  Max: {np.abs(delta_0).max()}")
print(f"  RMS: {np.sqrt(np.mean(delta_0**2)):.2f}")
print(f"  Mean: {np.mean(delta_0):.2f}")
print()

print(f"Slope jump (delta_1) at boundaries:")
print(f"  Max: {np.abs(delta_1).max()}")
print(f"  RMS: {np.sqrt(np.mean(delta_1**2)):.2f}")
print(f"  Mean: {np.mean(delta_1):.2f}")
print()

# Compare to ground truth boundaries
gt_delta_0 = []
gt_delta_1 = []
for b in range(1, n_blocks):
    boundary = b * BLOCK_SIZE
    d0 = int(ground_truth[boundary]) - int(ground_truth[boundary - 1])
    gt_delta_0.append(d0)
    if boundary >= 2:
        slope_before = int(ground_truth[boundary - 1]) - int(ground_truth[boundary - 2])
        slope_after = int(ground_truth[boundary]) - int(ground_truth[boundary - 1])
        d1 = slope_after - slope_before
        gt_delta_1.append(d1)

gt_delta_0 = np.array(gt_delta_0)
gt_delta_1 = np.array(gt_delta_1)

print(f"Ground truth boundary deltas (for comparison):")
print(f"  Value jump RMS: {np.sqrt(np.mean(gt_delta_0**2)):.2f}")
print(f"  Slope jump RMS: {np.sqrt(np.mean(gt_delta_1**2)):.2f}")
print()

# Difference between CCS and GT at boundaries
diff_delta_0 = delta_0 - gt_delta_0
diff_delta_1 = delta_1 - gt_delta_1
print(f"CCS - GT boundary delta difference:")
print(f"  Value jump diff RMS: {np.sqrt(np.mean(diff_delta_0**2)):.2f}")
print(f"  Slope jump diff RMS: {np.sqrt(np.mean(diff_delta_1**2)):.2f}")
print()

# ============================================================
# TEST 3: No-LUT control (direct FWPT/IDWPT on ground truth)
# ============================================================
print("TEST 3: NO-LUT CONTROL")
print("-" * 40)

no_lut_output = np.zeros(n_samples, dtype=np.int16)
phase_32 = 0

for block_idx in range(n_blocks):
    block_start = block_idx * BLOCK_SIZE

    # For each k, take shifted block from ground truth, FWPT, IDWPT
    idwpt_outputs = []
    for k in range(K):
        # Get ground truth block shifted by k samples
        gt_block = ground_truth[block_start + k : block_start + k + BLOCK_SIZE]
        if len(gt_block) < BLOCK_SIZE:
            gt_block = np.concatenate([gt_block, np.zeros(BLOCK_SIZE - len(gt_block), dtype=np.int16)])

        # FWPT -> IDWPT round trip
        coeffs = fwpt_jit(gt_block.tolist(), headroom_bits=1)
        recon = idwpt_jit(coeffs, headroom_bits=1)
        idwpt_outputs.append(recon)

    # Polyphase interleave
    for k in range(K):
        for m in range(BLOCK_SIZE // K):
            no_lut_output[block_start + k + m * K] = idwpt_outputs[k][m * K]

# FFT of no-LUT output
no_lut_fft = rfft(no_lut_output[:valid_len].astype(np.float64))
A_carrier_nolut = np.abs(no_lut_fft[fund_idx])

print(f"No-LUT carrier amplitude: {A_carrier_nolut:.0f}")
print()
print("No-LUT spur levels (dBc):")
for freq in spur_freqs:
    idx = np.argmin(np.abs(freqs - freq))
    A_spur = np.abs(no_lut_fft[idx])
    dBc = 20 * np.log10(A_spur / A_carrier_nolut + 1e-10)
    print(f"  {freq:5d} Hz: A={A_spur:8.0f}, dBc={dBc:+.1f}")

print()

# No-LUT boundary deltas
nolut_delta_0 = []
for b in range(1, n_blocks):
    boundary = b * BLOCK_SIZE
    d0 = int(no_lut_output[boundary]) - int(no_lut_output[boundary - 1])
    nolut_delta_0.append(d0)
nolut_delta_0 = np.array(nolut_delta_0)

diff_nolut = nolut_delta_0 - gt_delta_0
print(f"No-LUT boundary delta diff vs GT:")
print(f"  RMS: {np.sqrt(np.mean(diff_nolut**2)):.2f}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print("If No-LUT has similar spurs → problem is block stitching/IDWPT")
print("If No-LUT is clean → problem is LUT interpolation")
