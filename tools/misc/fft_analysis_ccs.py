"""FFT analysis of CCS output to identify spurs."""

import numpy as np
import sys
sys.path.insert(0, 'src')

# Clear LUT cache to ensure fresh generation
import symcrash_python_ap.vm.coeff_lut_streaming as lut_mod
lut_mod._NOTE_LUT_CACHE.clear()
print(f"LUT cache cleared. DEFAULT_P = {lut_mod.DEFAULT_P}")

from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame

SAMPLE_RATE = 48000

# Generate 440 Hz sine via CCS
frame = VoiceFrame(
    voice_id=0,
    note=69,
    cents=0,
    gain_q15=32767,
    pan_q15=0,
    tree_a_id=0,
    tree_b_id=0,
    tree_blend_q15=0,
    is_active=True,
    beta_target_q15=0,
    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
)

state = VMState.create(voice_id=0)
state.reset_for_note_on(note=69, cents=0)

print("Generating 1 second of 440 Hz sine via CCS...")
samples = []
num_blocks = int(1.0 * SAMPLE_RATE / 64)

for _ in range(num_blocks):
    state, output = vm_step_block(state, frame, DEFAULT_TABLES)
    samples.extend(output.tolist())

samples = np.array(samples, dtype=np.float64)
print(f"Generated {len(samples)} samples")
print(f"RMS: {np.sqrt(np.mean(samples**2)):.1f}")
print(f"Peak: {np.max(np.abs(samples)):.0f}")

# Apply window
window = windows.blackmanharris(len(samples))
windowed = samples * window

# FFT
spectrum = np.abs(rfft(windowed))
freqs = rfftfreq(len(samples), 1 / SAMPLE_RATE)

# Convert to dB (relative to max)
spectrum_db = 20 * np.log10(spectrum / spectrum.max() + 1e-10)

# Find fundamental
fund_idx = np.argmax(spectrum[10:2000]) + 10
fund_freq = freqs[fund_idx]
fund_db = spectrum_db[fund_idx]

print(f"\n=== FFT ANALYSIS ===")
print(f"Fundamental: {fund_freq:.2f} Hz at {fund_db:.1f} dB (normalized to 0 dB)")

# Find all peaks above -80 dB
print(f"\n=== ALL PEAKS ABOVE -80 dB ===")
peaks = []
for i in range(1, len(spectrum) - 1):
    if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
        if spectrum_db[i] > -80 and freqs[i] > 20:
            peaks.append((freqs[i], spectrum_db[i]))

# Sort by magnitude
peaks.sort(key=lambda x: -x[1])

print(f"Found {len(peaks)} peaks above -80 dB")
print("\nTop 30 peaks:")
print(f"{'Freq (Hz)':>10} {'dB':>8} {'Ratio to 440':>14} {'Note':>20}")
print("-" * 55)

for freq, db in peaks[:30]:
    ratio = freq / 440.0
    # Check if it's a harmonic
    is_harmonic = False
    for h in range(1, 20):
        if abs(freq - 440 * h) < 5:
            is_harmonic = True
            note = f"H{h} (harmonic)"
            break
    if not is_harmonic:
        # Check for common spur patterns
        if abs(ratio - round(ratio)) < 0.02:
            note = f"~{round(ratio)}x fundamental"
        else:
            note = "SPUR"
    print(f"{freq:10.2f} {db:8.1f} {ratio:14.4f} {note:>20}")

# Analyze spur distribution
print("\n=== SPUR ANALYSIS ===")
harmonic_freqs = [440 * h for h in range(1, 20)]
spurs = []
for freq, db in peaks:
    is_near_harmonic = any(abs(freq - hf) < 10 for hf in harmonic_freqs)
    if not is_near_harmonic and db > -60:
        spurs.append((freq, db))

print(f"Non-harmonic spurs above -60 dB: {len(spurs)}")

if spurs:
    # Look for patterns in spur spacing
    spur_freqs = sorted([f for f, _ in spurs])
    if len(spur_freqs) > 1:
        spacings = np.diff(spur_freqs)
        print(f"\nSpur frequency spacings (Hz):")
        print(f"  Min: {spacings.min():.1f}")
        print(f"  Max: {spacings.max():.1f}")
        print(f"  Mean: {spacings.mean():.1f}")
        print(f"  Std: {spacings.std():.1f}")

        # Common spur spacing indicates block-rate artifacts (48000/64 = 750 Hz)
        # or phase quantization artifacts
        block_rate = SAMPLE_RATE / 64
        print(f"\n  Block rate: {block_rate:.1f} Hz")
        print(f"  Block rate / K=8: {block_rate/8:.1f} Hz")

# Look at low frequency content
print("\n=== LOW FREQUENCY ANALYSIS (< 400 Hz) ===")
low_freq_peaks = [(f, d) for f, d in peaks if f < 400 and d > -70]
print(f"Found {len(low_freq_peaks)} peaks below 400 Hz above -70 dB:")
for freq, db in sorted(low_freq_peaks)[:20]:
    print(f"  {freq:7.1f} Hz: {db:+.1f} dB")
