#!/usr/bin/env python3
"""Trace through carrier render path step by step."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.wavelet.sine_core import sine_q15_vectorized
from symcrash_python_ap.vm.tables import BLOCK_SAMPLES

# Simulate lane_carrier_render step by step

Q15_ONE = 32767
f0 = 440.0
sample_rate = 48000
phase_inc = int(f0 * (2**32) / sample_rate)
harmonic_amps = [32767, 0, 0, 0, 0, 0, 0, 0]  # Pure sine

print("=" * 60)
print("Tracing carrier render path")
print("=" * 60)

# Render multiple blocks to get good FFT resolution
all_samples = []
phase = 0

for block in range(100):
    # Step 1: Generate phases
    sample_indices = np.arange(BLOCK_SAMPLES, dtype=np.int64)
    base_phases = (phase + sample_indices * phase_inc) & 0xFFFFFFFF
    phase = (phase + BLOCK_SAMPLES * phase_inc) & 0xFFFFFFFF

    # Step 2: Accumulate harmonics
    carrier_samples = np.zeros(BLOCK_SAMPLES, dtype=np.int32)
    total_amp = sum(harmonic_amps)
    if total_amp == 0:
        total_amp = Q15_ONE

    for h in range(len(harmonic_amps)):
        amp = harmonic_amps[h]
        if amp == 0:
            continue

        h_phases = (base_phases * (h + 1)) & 0xFFFFFFFF
        phase_norm = ((h_phases >> 16) & 0xFFFF).astype(np.int32)
        sin_vals = sine_q15_vectorized(phase_norm)
        carrier_samples += (sin_vals * amp) >> 15

    if block == 0:
        print(f"\nAfter harmonic synthesis (block 0):")
        print(f"  First 10: {carrier_samples[:10].tolist()}")
        print(f"  Max: {carrier_samples.max()}, Min: {carrier_samples.min()}")

    # Step 3: Normalize and add headroom
    DC_HEADROOM = 30983
    divisor = max(total_amp, Q15_ONE)
    carrier_samples = (carrier_samples * DC_HEADROOM) // divisor

    if block == 0:
        print(f"\nAfter headroom scaling:")
        print(f"  First 10: {carrier_samples[:10].tolist()}")
        print(f"  Max: {carrier_samples.max()}, Min: {carrier_samples.min()}")

    # Step 4: DC blocking filter
    dc_coeff = 32686  # ~0.9975
    dc_x1 = 0 if block == 0 else dc_x1_save
    dc_y1 = 0 if block == 0 else dc_y1_save

    output = np.zeros(BLOCK_SAMPLES, dtype=np.int32)
    for i in range(BLOCK_SAMPLES):
        x = int(carrier_samples[i])
        y = x - dc_x1 + ((dc_y1 * dc_coeff) >> 15)
        dc_x1 = x
        dc_y1 = y
        output[i] = y

    dc_x1_save = dc_x1
    dc_y1_save = dc_y1

    if block == 0:
        print(f"\nAfter DC filter:")
        print(f"  First 10: {output[:10].tolist()}")
        print(f"  Max: {output.max()}, Min: {output.min()}")

    all_samples.extend(output.tolist())

# FFT analysis
samples = np.array(all_samples, dtype=np.float64)
fft = np.abs(np.fft.rfft(samples))
freqs = np.fft.rfftfreq(len(samples), 1/sample_rate)
fund_idx = np.argmax(fft[10:]) + 10
fund_mag = fft[fund_idx]

print(f"\n" + "=" * 60)
print("Final carrier spectrum:")
print(f"  Fundamental: {freqs[fund_idx]:.1f} Hz")
for h in [2, 3, 4, 5]:
    idx = np.argmin(np.abs(freqs - freqs[fund_idx] * h))
    db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
    print(f"  H{h}: {db:+.1f} dB")

# Compare with running through actual lane_carrier_render
print(f"\n" + "=" * 60)
print("Comparison with actual lane_carrier_render:")
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_carrier_render
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame

frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=0, partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)
state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

actual_samples = []
for _ in range(100):
    cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=state.eta_prev_q15)
    carrier = lane_carrier_render(state, cfg)
    actual_samples.extend(carrier.tolist())
    state.eta_prev_q15 = cfg.eta_curr_q15

actual_arr = np.array(actual_samples, dtype=np.float64)
fft2 = np.abs(np.fft.rfft(actual_arr))
fund_idx2 = np.argmax(fft2[10:]) + 10
fund_mag2 = fft2[fund_idx2]

print(f"  Fundamental: {freqs[fund_idx2]:.1f} Hz")
for h in [2, 3, 4, 5]:
    idx = np.argmin(np.abs(freqs - freqs[fund_idx2] * h))
    db = 20 * np.log10(fft2[idx] / fund_mag2 + 1e-10)
    print(f"  H{h}: {db:+.1f} dB")
