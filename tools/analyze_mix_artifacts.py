#!/usr/bin/env python3
"""Deep analyze mixing artifacts."""
import sys
sys.path.insert(0, 'src')
import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_carrier_render, lane_idwpt_render, lane_mix_gain, _sincos_batch_q15, _ALPHA_Q15
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Warm up for eta stabilization
NUM_WARMUP = 32
NUM_BLOCKS = 128
BLOCK_SIZE = 64

frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=16384,  # 50/50
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

# Warm up (let eta stabilize)
for _ in range(NUM_WARMUP):
    cfg = decode_frame(frame, DEFAULT_TABLES)
    carrier = lane_carrier_render(state, cfg).copy()
    wet = lane_idwpt_render(state, cfg, carrier)
    _ = lane_mix_gain(cfg, carrier, wet)

# Collect and analyze
carrier_all = []
wet_all = []
mixed_all = []
simple_mix_all = []
dry_weight_all = []
wet_weight_all = []

for i in range(NUM_BLOCKS):
    cfg = decode_frame(frame, DEFAULT_TABLES)
    carrier = lane_carrier_render(state, cfg).copy()
    wet = lane_idwpt_render(state, cfg, carrier).copy()
    mixed = lane_mix_gain(cfg, carrier, wet).copy()

    # Replicate what lane_mix_gain does but simplified
    eta_delta = cfg.eta_curr_q15 - cfg.eta_prev_q15
    eta_vals = cfg.eta_prev_q15 + ((_ALPHA_Q15 * eta_delta) >> 15)

    phases = ((eta_vals.astype(np.int64) * 0x40000000) >> 15).astype(np.uint32)
    dry_weights, wet_weights = _sincos_batch_q15(phases)

    # Simple mixing (no gain)
    carrier_i32 = carrier.astype(np.int32)
    wet_i32 = wet.astype(np.int32)
    simple_mix = (carrier_i32 * dry_weights + wet_i32 * wet_weights) >> 15

    carrier_all.append(carrier)
    wet_all.append(wet)
    mixed_all.append(mixed)
    simple_mix_all.append(simple_mix.astype(np.int16))
    dry_weight_all.append(dry_weights)
    wet_weight_all.append(wet_weights)

carrier_arr = np.concatenate(carrier_all)
wet_arr = np.concatenate(wet_all)
mixed_arr = np.concatenate(mixed_all)
simple_mix_arr = np.concatenate(simple_mix_all)
dry_weights_arr = np.concatenate(dry_weight_all)
wet_weights_arr = np.concatenate(wet_weight_all)

print("Dry/wet weights analysis:")
print(f"  dry weights: min={dry_weights_arr.min()}, max={dry_weights_arr.max()}, mean={dry_weights_arr.mean():.1f}")
print(f"  wet weights: min={wet_weights_arr.min()}, max={wet_weights_arr.max()}, mean={wet_weights_arr.mean():.1f}")
print(f"  sum: min={(dry_weights_arr + wet_weights_arr).min()}, max={(dry_weights_arr + wet_weights_arr).max()}")

# Check equal power property: dry^2 + wet^2 should = 32767
power_sum = (dry_weights_arr.astype(np.int64)**2 + wet_weights_arr.astype(np.int64)**2) >> 15
print(f"  equal-power (dry^2+wet^2)>>15: min={power_sum.min()}, max={power_sum.max()}")

# FFT analysis
def spectrum_peaks(arr, name, sr=48000):
    fft = np.abs(np.fft.rfft(arr.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(arr), 1/sr)
    fund_idx = np.argmax(fft[1:]) + 1
    fund_mag = fft[fund_idx]

    print(f"\n{name}:")
    peaks = np.argsort(fft)[-15:][::-1]
    for idx in peaks:
        if fft[idx] > fund_mag * 0.0001:  # Above -80 dB
            db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
            if db > -60:
                print(f"  {freqs[idx]:8.1f} Hz: {db:6.1f} dB")

spectrum_peaks(carrier_arr, "Carrier")
spectrum_peaks(simple_mix_arr, "Simple mix (no gain)")
spectrum_peaks(mixed_arr, "Full mixed (with gain)")

# Check weights spectrum - do they have the artifact frequency?
print("\nWeight spectrum (to check for oscillation):")
fft_dry = np.abs(np.fft.rfft(dry_weights_arr.astype(np.float64)))
fft_wet = np.abs(np.fft.rfft(wet_weights_arr.astype(np.float64)))
freqs = np.fft.rfftfreq(len(dry_weights_arr), 1/48000)

peaks = np.argsort(fft_dry)[-5:][::-1]
print("Dry weight spectrum peaks:")
for idx in peaks:
    if fft_dry[idx] > 100:
        print(f"  {freqs[idx]:8.1f} Hz: {fft_dry[idx]:.0f}")
