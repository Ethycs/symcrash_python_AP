"""Check for derivative discontinuities in carrier output - full pipeline."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.wavelet.sine_core import sine_q15_vectorized

Q15_ONE = 32767
BLOCK_SAMPLES = 64

# Simulate carrier at 440Hz, 48kHz
phase_inc = int(440 * (2**32) / 48000)
print(f"Phase increment: {phase_inc}")

samples_per_cycle = int(48000 / 440)
print(f"Samples per cycle: {samples_per_cycle}")

# SAW wave harmonics (1/n amplitude)
harmonic_amps = [Q15_ONE // (h+1) for h in range(8)]
print(f"Harmonic amps: {harmonic_amps}")
total_amp = sum(harmonic_amps)
print(f"Total amp sum: {total_amp}")

# Generate multiple cycles
num_samples = samples_per_cycle * 5
phases_32bit = np.arange(num_samples, dtype=np.int64) * phase_inc

# Accumulate like lanes.py does
carrier_samples = np.zeros(num_samples, dtype=np.int32)

for h in range(len(harmonic_amps)):
    amp = harmonic_amps[h]
    if amp == 0:
        continue

    h_phases = (phases_32bit * (h + 1)) & 0xFFFFFFFF
    phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
    sin_vals = sine_q15_vectorized(phase_norm)
    carrier_samples += (sin_vals * amp) >> 15

# Normalize (this is where integer division happens)
DC_HEADROOM = 30983
divisor = max(total_amp, Q15_ONE)

# TRUNCATING division (original - causes spikes)
carrier_truncated = (carrier_samples * DC_HEADROOM) // divisor

# ROUNDED division (proposed fix)
half_div = divisor // 2
carrier_rounded = (carrier_samples * DC_HEADROOM + np.sign(carrier_samples) * half_div) // divisor

# Use truncated for analysis
carrier_normalized = carrier_truncated

# Compute derivatives
d1 = np.diff(carrier_normalized)
d2 = np.diff(d1)

# Also compute for rounded
d1_r = np.diff(carrier_rounded)
d2_r = np.diff(d1_r)

print(f"\nNormalized carrier stats:")
print(f"  Range: {carrier_normalized.min()} to {carrier_normalized.max()}")

print(f"\nFirst derivative (d1) stats:")
print(f"  Std: {d1.std():.1f}")
print(f"  Max: {d1.max()}, Min: {d1.min()}")

print(f"\nSecond derivative (d2) stats:")
print(f"  Std: {d2.std():.1f}")
print(f"  Max: {d2.max()}, Min: {d2.min()}")

# Find spikes in d2
threshold = d2.std() * 3
spikes = np.where(np.abs(d2) > threshold)[0]
print(f"\nSpike count (|d2| > {threshold:.0f}): {len(spikes)}")

if len(spikes) > 0:
    # Check cycle positions
    cycle_positions = (spikes % samples_per_cycle) / samples_per_cycle

    # Count spikes per cycle
    spikes_per_cycle = len(spikes) / (num_samples / samples_per_cycle)
    print(f"Spikes per cycle: {spikes_per_cycle:.1f}")

    # Group by position
    bins = np.round(cycle_positions * 12) / 12
    unique_bins, counts = np.unique(bins, return_counts=True)
    print(f"Position clusters (cycle fraction): {dict(zip(unique_bins[:10], counts[:10]))}")

# Look at the raw accumulation BEFORE normalization
d1_raw = np.diff(carrier_samples)
d2_raw = np.diff(d1_raw)

print("\n=== RAW (before normalization) ===")
print(f"Second derivative (d2_raw) stats:")
print(f"  Std: {d2_raw.std():.1f}")
print(f"  Max: {d2_raw.max()}, Min: {d2_raw.min()}")

threshold_raw = d2_raw.std() * 3
spikes_raw = np.where(np.abs(d2_raw) > threshold_raw)[0]
print(f"Spike count: {len(spikes_raw)}")
if len(spikes_raw) > 0:
    spikes_per_cycle_raw = len(spikes_raw) / (num_samples / samples_per_cycle)
    print(f"Spikes per cycle: {spikes_per_cycle_raw:.1f}")

# Check each harmonic individually
print("\n=== PER-HARMONIC ANALYSIS ===")
for h in range(len(harmonic_amps)):
    amp = harmonic_amps[h]
    if amp == 0:
        continue

    h_phases = (phases_32bit * (h + 1)) & 0xFFFFFFFF
    phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
    sin_vals = sine_q15_vectorized(phase_norm)
    h_contrib = (sin_vals * amp) >> 15

    d2_h = np.diff(np.diff(h_contrib))
    threshold_h = d2_h.std() * 3 if d2_h.std() > 0 else 1
    spikes_h = np.where(np.abs(d2_h) > threshold_h)[0]

    if len(spikes_h) > 0:
        print(f"H{h+1}: {len(spikes_h) / (num_samples / samples_per_cycle):.1f} spikes/cycle, d2 max={d2_h.max()}")
