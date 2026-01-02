#!/usr/bin/env python3
"""Check if the carrier is really generating a pure sine."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.lanes import lane_carrier_render
from symcrash_python_ap.vm.tables import DEFAULT_TABLES

# Create test frame
frame = VoiceFrame(
    voice_id=0, note=69, f0_hz=440.0, gain_q15=32767,
    beta_target_q15=0,
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],  # Pure sine
    is_active=True,
)

state = VMState.create(0)
state.reset_for_note_on(f0_hz=440.0)

cfg = decode_frame(frame, DEFAULT_TABLES)

print("DecodedPlan carrier params:")
print(f"  harmonic_amps_q15: {cfg.carrier_params.harmonic_amps_q15}")
print(f"  phase_inc: {cfg.carrier_params.phase_inc}")

# Render carrier for multiple blocks
samples = []
for _ in range(100):
    cfg = decode_frame(frame, DEFAULT_TABLES, prev_eta_q15=state.eta_prev_q15)
    carrier = lane_carrier_render(state, cfg)
    samples.extend(carrier.tolist())
    state.eta_prev_q15 = cfg.eta_curr_q15

samples = np.array(samples, dtype=np.float64)

# Analyze spectrum
fft = np.abs(np.fft.rfft(samples))
freqs = np.fft.rfftfreq(len(samples), 1/48000)
fund_idx = np.argmax(fft[10:]) + 10
fund_mag = fft[fund_idx]
fund_freq = freqs[fund_idx]

print(f"\nCarrier spectrum (should be pure 440Hz sine):")
print(f"  Fundamental: {fund_freq:.1f} Hz")

for h in [2, 3, 4, 5]:
    idx = np.argmin(np.abs(freqs - fund_freq * h))
    db = 20 * np.log10(fft[idx] / fund_mag + 1e-10)
    print(f"  H{h}: {db:+.1f} dB")

# Check the raw sine values
print(f"\nFirst block carrier values (should be smooth sine):")
print(f"  First 10: {samples[:10].astype(int).tolist()}")
print(f"  Max: {np.max(samples):.0f}, Min: {np.min(samples):.0f}")

# Generate expected pure sine for comparison
expected = [32767 * 0.9456 * np.sin(2 * np.pi * 440 * i / 48000) for i in range(100)]
print(f"\nExpected pure sine first 10: {[int(x) for x in expected[:10]]}")
