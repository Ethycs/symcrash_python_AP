import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.coeff_osc import CoeffOscState, coeff_osc_generate_coeffs
from symcrash_python_ap.vm.coeff_tables_fast import get_fast_coeff_table
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15
import numpy as np

print('Testing coefficient-domain carrier with multiple blocks...\n')

# Create CoeffOsc state
state = CoeffOscState()
state.table_id = 60  # MIDI note 60 (Middle C)

# Compute phase increment for MIDI 60
freq_hz = 261.63
sample_rate = 48000
phase_inc = int((freq_hz / sample_rate) * (1 << 32))
state.base_phase_inc = phase_inc
state.target_phase_inc = phase_inc

print(f'MIDI 60 = {freq_hz:.2f} Hz')
print(f'Phase increment: {phase_inc} (0x{phase_inc:08X})')

# Get table
table = get_fast_coeff_table()

# Generate 10 blocks
print('\nGenerating 10 blocks (640 samples)...')
all_audio = []
coeffs_buffer = np.zeros(64, dtype=np.int16)

for block in range(10):
    # Generate coefficients
    coeff_osc_generate_coeffs(state, table, coeffs_buffer)

    # Reconstruct audio
    audio = idwpt_jit(coeffs_buffer.tolist(), headroom_bits=1)
    all_audio.extend(audio)

    if block == 0:
        print(f'  Block 0 coeffs[0:5]: {coeffs_buffer[0:5]}')
        print(f'  Block 0 audio[0:5]: {audio[0:5]}')

# Analyze full signal
all_audio_arr = np.array(all_audio, dtype=np.float32)

print(f'\nTotal samples: {len(all_audio_arr)}')
print(f'Audio range: {all_audio_arr.min():.0f} to {all_audio_arr.max():.0f}')

# FFT analysis
spectrum = np.fft.rfft(all_audio_arr)
freqs = np.fft.rfftfreq(len(all_audio_arr), 1.0/sample_rate)
peak_idx = np.argmax(np.abs(spectrum))
peak_freq = freqs[peak_idx]

freq_error = abs(peak_freq - freq_hz)

print(f'\nFFT Analysis:')
print(f'  Expected: {freq_hz:.2f} Hz')
print(f'  Peak at:  {peak_freq:.2f} Hz')
print(f'  Error:    {freq_error:.2f} Hz')

if freq_error < 5.0:
    print('\n[SUCCESS] Coefficient-domain carrier working correctly!')
else:
    print(f'\n[FAIL] Frequency error: {freq_error:.2f} Hz')

# Check block continuity
print(f'\nBlock continuity check:')
for i in range(1, 10):
    boundary_idx = i * 64
    jump = abs(all_audio_arr[boundary_idx] - all_audio_arr[boundary_idx - 1])
    print(f'  Block {i-1} -> {i}: jump = {jump:.1f} samples')
    if jump > 1000:
        print(f'    [WARN] Large discontinuity!')
