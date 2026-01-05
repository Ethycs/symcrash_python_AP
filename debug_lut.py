"""Debug CoeffLUT output with sin/cos basis."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15
from symcrash_python_ap.vm.coeff_lut import get_coeff_lut

def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int((freq_hz / sample_rate) * (1 << 32))

# Get the CoeffLUT
lut = get_coeff_lut(0)

print("CoeffLUT sin/cos basis:")
print(f"  c_sin range: [{lut.c_sin.min()}, {lut.c_sin.max()}]")
print(f"  c_sin RMS: {np.sqrt(np.mean(lut.c_sin.astype(float)**2)):.1f}")
print(f"  c_cos range: [{lut.c_cos.min()}, {lut.c_cos.max()}]")
print(f"  c_cos RMS: {np.sqrt(np.mean(lut.c_cos.astype(float)**2)):.1f}")

# IDWPT of c_sin should give sin(ωt) at reference frequency
recon_sin = np.array(idwpt_jit(lut.c_sin.tolist(), headroom_bits=1), dtype=np.int16)
print(f"\nIDWPT(c_sin): range=[{recon_sin.min()}, {recon_sin.max()}], RMS={np.sqrt(np.mean(recon_sin.astype(float)**2)):.1f}")

# IDWPT of c_cos should give cos(ωt) at reference frequency
recon_cos = np.array(idwpt_jit(lut.c_cos.tolist(), headroom_bits=1), dtype=np.int16)
print(f"IDWPT(c_cos): range=[{recon_cos.min()}, {recon_cos.max()}], RMS={np.sqrt(np.mean(recon_cos.astype(float)**2)):.1f}")

# FFT of IDWPT(c_sin) to find reference frequency
from scipy.fft import fft
N = 64
fft_sin = np.abs(fft(recon_sin.astype(np.float32)))[:N//2]
freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
peak_idx = np.argmax(fft_sin)
ref_freq = freqs[peak_idx]
print(f"Reference frequency (from IDWPT(c_sin)): {ref_freq:.2f} Hz")

# Now simulate phase rotation at C4 (261.63 Hz)
phase_inc = midi_to_phase_inc(60)
phase = 0

print("\n\nPhase rotation simulation at C4 (261.63 Hz):")
for block in range(5):
    # Get coefficients via phase rotation
    coeffs = lut.get_coeffs_for_phase(phase)

    # IDWPT
    output = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)

    phase_deg = (phase / (1 << 32)) * 360
    print(f"  Block {block}: phase={phase_deg:6.1f} deg, output range=[{output.min():6d}, {output.max():6d}]")

    # Advance phase
    phase = (phase + phase_inc * 64) & 0xFFFFFFFF

# The problem: coefficients are for a DIFFERENT frequency than runtime frequency
# The table stores coeffs for ~750 Hz (one cycle per 64 samples at 48kHz)
# But we want 261.63 Hz

# Let's check what frequency the CoeffLUT was generated at
cycles_per_block_ref = 1.0  # We set this in generate_from_waveform
ref_freq_expected = cycles_per_block_ref * 48000 / 64
print(f"\nExpected reference frequency: {ref_freq_expected:.2f} Hz")
print(f"Actual reference frequency: {ref_freq:.2f} Hz")
print(f"\nThe issue: CoeffLUT is generated at {ref_freq:.0f} Hz, but we want 261.63 Hz")
print("Phase rotation CHANGES the phase offset, not the frequency!")
