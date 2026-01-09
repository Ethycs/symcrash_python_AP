"""Run 100 blocks and analyze."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.fft import rfft, rfftfreq
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.vm import vm_step_block

print("=== 100 Block Analysis ===\n")

# Create state
state = VMState.create(voice_id=0)
state.reset_for_note_on(note=60, cents=0)

frame = VoiceFrame(
    voice_id=0,
    note=60,
    cents=0,
    gain_q15=32767,
    beta_target_q15=32767,
)

# Run 100 blocks
outputs = []
for i in range(100):
    state, output = vm_step_block(state, frame, DEFAULT_TABLES)
    outputs.append(output.copy())

audio = np.concatenate(outputs).astype(np.float64)

# Skip first 10 blocks (latency warmup)
skip = 10 * 64
audio_stable = audio[skip:]

print(f"Total samples: {len(audio)}")
print(f"Stable samples (after block 10): {len(audio_stable)}")
print(f"Audio range: [{audio.min():.0f}, {audio.max():.0f}]")
print(f"Audio mean: {audio.mean():.1f}")

# Block boundary jumps (after warmup)
print("\nBlock boundary jumps (blocks 10-20):")
jumps = []
for i in range(10, 20):
    idx = i * 64
    jump = abs(audio[idx] - audio[idx - 1])
    jumps.append(jump)
    print(f"  Block {i-1} -> {i}: jump={jump:.0f}")

print(f"\nJump stats (blocks 10-100):")
all_jumps = []
for i in range(10, 100):
    idx = i * 64
    jump = abs(audio[idx] - audio[idx - 1])
    all_jumps.append(jump)
print(f"  Max: {max(all_jumps):.0f}")
print(f"  Mean: {np.mean(all_jumps):.0f}")
print(f"  Median: {np.median(all_jumps):.0f}")

# FFT analysis
SAMPLE_RATE = 48000
audio_ac = audio_stable - np.mean(audio_stable)
spectrum = rfft(audio_ac)
freqs = rfftfreq(len(audio_ac), 1.0/SAMPLE_RATE)
mag_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

# Find peak
mask = freqs > 50
spec_masked = np.abs(spectrum).copy()
spec_masked[~mask] = 0
peak_idx = np.argmax(spec_masked)
peak_freq = freqs[peak_idx]

print(f"\nFFT Analysis:")
print(f"  Peak frequency: {peak_freq:.1f} Hz (expected 261.6 Hz)")
print(f"  Peak magnitude: {mag_db[peak_idx]:.1f} dB")

# Check 750 Hz comb
print(f"\nComb artifact check (750 Hz spacing):")
for mult in range(1, 8):
    comb_freq = 750.0 * mult
    if comb_freq > 5000:
        break
    idx = np.argmin(np.abs(freqs - comb_freq))
    comb_db = mag_db[idx]
    rel = comb_db - mag_db[peak_idx]
    status = "WARN" if rel > -40 else "OK"
    print(f"  {comb_freq:.0f} Hz: {comb_db:.1f} dB ({rel:+.1f} rel) [{status}]")

# Check harmonics of fundamental
print(f"\nHarmonics of 261.6 Hz:")
expected = 261.63
for h in range(1, 6):
    h_freq = expected * h
    idx = np.argmin(np.abs(freqs - h_freq))
    h_db = mag_db[idx]
    rel = h_db - mag_db[peak_idx]
    marker = " <- EXPECTED PEAK" if h == 1 else ""
    print(f"  H{h} ({h_freq:.1f} Hz): {h_db:.1f} dB ({rel:+.1f} rel){marker}")
