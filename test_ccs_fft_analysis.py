import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

print('=== CCS Path: Comprehensive FFT Analysis ===\n')

# Generate long signal for good frequency resolution
NUM_BLOCKS = 100
SAMPLE_RATE = 48000

state = VMState.create(voice_id=0)
state.reset_for_note_on(note=60, cents=0)

frame = VoiceFrame(
    voice_id=0,
    note=60,  # Middle C = 261.63 Hz
    cents=0,
    gain_q15=32767,  # Full gain
    beta_target_q15=32767,  # Full carrier (base_mix)
)

print(f'Generating {NUM_BLOCKS} blocks ({NUM_BLOCKS*64} samples, {NUM_BLOCKS*64/SAMPLE_RATE:.2f} sec)...')

all_audio = []
for block_idx in range(NUM_BLOCKS):
    state, output = vm_step_block(state, frame, DEFAULT_TABLES)
    all_audio.append(output)

audio_arr = np.concatenate(all_audio).astype(np.float32)

print(f'Audio generated: {len(audio_arr)} samples')
print(f'RMS: {np.sqrt(np.mean(audio_arr**2)):.1f}')
print(f'Range: {audio_arr.min():.0f} to {audio_arr.max():.0f}\n')

# FFT Analysis
print('=== FFT Spectrum Analysis ===\n')

# Compute FFT
spectrum = rfft(audio_arr)
freqs = rfftfreq(len(audio_arr), 1.0/SAMPLE_RATE)
magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

# Find peak
peak_idx = np.argmax(np.abs(spectrum))
peak_freq = freqs[peak_idx]
peak_db = magnitude_db[peak_idx]

expected_freq = 261.63  # Middle C

print(f'Peak Analysis:')
print(f'  Expected: {expected_freq:.2f} Hz')
print(f'  Found:    {peak_freq:.2f} Hz')
print(f'  Error:    {abs(peak_freq - expected_freq):.2f} Hz')
print(f'  Magnitude: {peak_db:.1f} dB\n')

# Find all significant peaks (> -40 dB relative to peak)
threshold_db = peak_db - 40
peak_indices = np.where(magnitude_db > threshold_db)[0]
peak_freqs = freqs[peak_indices]
peak_mags = magnitude_db[peak_indices]

print(f'Significant Peaks (>{threshold_db:.1f} dB):')
for freq, mag in zip(peak_freqs[:20], peak_mags[:20]):  # First 20
    rel_db = mag - peak_db
    if freq < 5000:  # Only show up to 5 kHz
        print(f'  {freq:7.2f} Hz: {mag:6.1f} dB ({rel_db:+.1f} dB rel)')

# Check for expected harmonics
print(f'\nHarmonic Analysis:')
for h in range(1, 11):
    harmonic_freq = expected_freq * h
    if harmonic_freq > SAMPLE_RATE / 2:
        break

    # Find closest bin
    idx = np.argmin(np.abs(freqs - harmonic_freq))
    harmonic_db = magnitude_db[idx]
    rel_db = harmonic_db - peak_db

    marker = ""
    if h == 1:
        marker = " <- FUNDAMENTAL"
    elif abs(rel_db) < 3:
        marker = " <- STRONG"
    elif rel_db < -60:
        marker = " (noise floor)"

    print(f'  H{h:2d} ({harmonic_freq:7.1f} Hz): {harmonic_db:6.1f} dB ({rel_db:+6.1f} dB){marker}')

# Check for non-harmonic artifacts (combing)
print(f'\nNon-Harmonic Artifact Check:')
comb_spacing = 750.0  # Legacy combing spacing
for mult in [1, 2, 3, 4]:
    comb_freq = comb_spacing * mult
    if comb_freq < expected_freq * 0.8 or comb_freq > expected_freq * 1.2:
        # Only check frequencies away from fundamental
        idx = np.argmin(np.abs(freqs - comb_freq))
        comb_db = magnitude_db[idx]
        rel_db = comb_db - peak_db

        status = "[OK]" if rel_db < -40 else "[WARN]"
        print(f'  {comb_freq:.1f} Hz: {comb_db:.1f} dB ({rel_db:+.1f} dB) {status}')

# THD calculation
print(f'\nTHD Analysis:')
fundamental_power = np.abs(spectrum[peak_idx])**2
harmonic_power = 0
for h in range(2, 11):
    h_freq = expected_freq * h
    if h_freq > SAMPLE_RATE / 2:
        break
    h_idx = np.argmin(np.abs(freqs - h_freq))
    harmonic_power += np.abs(spectrum[h_idx])**2

thd_percent = 100 * np.sqrt(harmonic_power / (fundamental_power + 1e-10))
print(f'  THD: {thd_percent:.2f}%')

if thd_percent < 1.0:
    print('  [EXCELLENT] Very clean sine')
elif thd_percent < 5.0:
    print('  [GOOD] Low distortion')
else:
    print(f'  [INFO] {thd_percent:.1f}% harmonic content')

# Block boundary artifact check
print(f'\nBlock Boundary Continuity:')
max_jump = 0
jumps = []
for i in range(1, NUM_BLOCKS):
    boundary_idx = i * 64
    jump = abs(audio_arr[boundary_idx] - audio_arr[boundary_idx - 1])
    max_jump = max(max_jump, jump)
    jumps.append(jump)

avg_jump = np.mean(jumps)
print(f'  Max jump: {max_jump:.1f} samples')
print(f'  Avg jump: {avg_jump:.1f} samples')
print(f'  Std dev:  {np.std(jumps):.1f} samples')

if max_jump < 500:
    print('  [OK] Smooth block boundaries')
elif max_jump < 2000:
    print('  [ACCEPTABLE] Minor discontinuities')
else:
    print('  [WARN] Large discontinuities detected')

print('\n=== SUMMARY ===')
freq_error = abs(peak_freq - expected_freq)
if freq_error < 5.0 and max_jump < 500 and rel_db < -40:
    print('[SUCCESS] CCS produces clean, continuous audio!')
    print(f'  - Frequency accurate: {peak_freq:.2f} Hz')
    print(f'  - Phase continuous: max jump {max_jump:.0f}')
    print(f'  - No combing artifacts')
else:
    print('[PARTIAL] CCS working but with issues:')
    if freq_error > 5.0:
        print(f'  - Frequency error: {freq_error:.2f} Hz')
    if max_jump > 500:
        print(f'  - Discontinuities: {max_jump:.0f} samples')
