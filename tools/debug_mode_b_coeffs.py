"""Debug Mode B coefficient generation directly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from symcrash_python_ap.vm.harmonic_bank import (
    ModeBState, HarmonicWeights, get_basis_for_note,
    mode_b_generate_coeffs
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

# Setup Mode B state
weights = HarmonicWeights()
weights.set_custom([30890, 32767, 6926, 23969, 3673, 6138, 1164, 1228])

print("Harmonic weights (a_h):")
for i in range(8):
    print(f"  H{i+1}: {weights.a_h_q15[i]}")
print(f"  H_active: {weights.H_active}")

# Create Mode B state
mb_state = ModeBState()
mb_state.weights = weights
mb_state.H_active = 8
mb_state.H_bandlimit = 32

# Get phase_inc for A4 (440 Hz)
from symcrash_python_ap.fixed import midi_to_phase_inc_fixed
phase_inc = midi_to_phase_inc_fixed(69, 0)
print(f"\nPhase inc for A4: {phase_inc}")

mb_state.reset(note=69, phase_inc=phase_inc)

# Get basis frames
basis = get_basis_for_note(69)
print(f"\nBasis frames: H_max={basis.H_max}, anchor_note={basis.anchor_note}")

# Generate coefficients
output_coeffs = np.zeros(512, dtype=np.int16)
mode_b_generate_coeffs(mb_state, basis, output_coeffs)

print(f"\nOutput coefficients (first 64 = first leaf across all k-slots):")
print(f"  Leaf 0: {list(output_coeffs[0:8])}")
print(f"  Leaf 1: {list(output_coeffs[8:16])}")
print(f"  Leaf 2: {list(output_coeffs[16:24])}")

print(f"\nCoeff stats: min={output_coeffs.min()}, max={output_coeffs.max()}, mean={output_coeffs.mean():.1f}")

# Now run IDWPT on first k-slot to see what audio we get
leaf_coeffs = output_coeffs.reshape(64, 8)[:, 0].astype(int).tolist()
print(f"\nLeaf coeffs for k=0: {leaf_coeffs[:8]}...")

audio = idwpt_jit(leaf_coeffs)
print(f"\nIDWPT output (first 16): {audio[:16]}")
print(f"Audio stats: min={min(audio)}, max={max(audio)}")

# FFT of the IDWPT output
from numpy.fft import rfft, rfftfreq
audio_np = np.array(audio, dtype=np.float64)
spectrum = np.abs(rfft(audio_np))
freqs = rfftfreq(len(audio), 1/48000)

print(f"\nSpectrum peaks from single-block IDWPT:")
peak_idx = np.argsort(spectrum)[-5:][::-1]
for i in peak_idx:
    print(f"  {freqs[i]:.1f} Hz: {spectrum[i]:.0f}")
