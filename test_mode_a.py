import sys
sys.path.insert(0, 'src')

from symcrash_python_ap.vm.coeff_osc_mode_a import (
    CoeffOscStateModeA, get_mode_a_table, generate_coeffs_mode_a
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit
import numpy as np

print('Testing Mode A: Table-Orbit Oscillator\n')

# Create state
state = CoeffOscStateModeA()

# Compute phase increment for MIDI 60 (261.63 Hz)
freq_hz = 261.63
sample_rate = 48000
phase_inc = int((freq_hz / sample_rate) * (1 << 32))

state.base_phase_inc = phase_inc
state.target_phase_inc = phase_inc
state.table_id = 0  # Sine table

print(f'MIDI 60 = {freq_hz:.2f} Hz')
print(f'Phase increment: 0x{phase_inc:08X}\n')

# Get table
table = get_mode_a_table(0)  # Sine
print(f'Table: {table.name}')
print(f'  mag[0] = {table.mag_q15[0]} (first coeff magnitude)')
print(f'  stride[0] = {table.stride_q16[0]} (Q16 format, = 1.0x)\n')

# Generate 10 blocks
print('Generating 10 blocks (640 samples)...')
all_audio = []
coeffs_buffer = np.zeros(64, dtype=np.int16)

for block in range(10):
    # Generate 64 leaf coefficients
    generate_coeffs_mode_a(state, table, coeffs_buffer)

    # Reconstruct audio via IDWPT
    audio = idwpt_jit(coeffs_buffer.tolist(), headroom_bits=1)
    all_audio.extend(audio)

    if block == 0:
        print(f'  Block 0 coeffs[0:5]: {coeffs_buffer[0:5]}')
        print(f'  Block 0 audio[0:5]: {audio[0:5]}')
        print(f'  Block 0 phases[0:5]: {state.leaf_phases[0:5]}')

# Analyze full signal
all_audio_arr = np.array(all_audio, dtype=np.float32)

print(f'\nTotal samples: {len(all_audio_arr)}')
print(f'Audio range: {all_audio_arr.min():.0f} to {all_audio_arr.max():.0f}')

# Check block continuity
print(f'\nBlock continuity check:')
max_jump = 0
for i in range(1, 10):
    boundary_idx = i * 64
    jump = abs(all_audio_arr[boundary_idx] - all_audio_arr[boundary_idx - 1])
    max_jump = max(max_jump, jump)
    if jump > 500:
        print(f'  Block {i-1} -> {i}: jump = {jump:.1f} samples [WARN]')

if max_jump < 500:
    print(f'  Max jump: {max_jump:.1f} samples [OK - continuous!]')

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

if freq_error < 5.0 and max_jump < 500:
    print('\n========================================')
    print('[SUCCESS] Mode A working correctly!')
    print('  - Phase-continuous coefficients')
    print('  - Correct frequency reconstruction')
    print('  - No block discontinuities')
    print('========================================')
else:
    print('\n[FAIL] Mode A has issues')
