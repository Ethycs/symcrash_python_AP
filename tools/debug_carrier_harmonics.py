"""Debug Mode B custom harmonics rendering."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.decode import decode_frame
from symcrash_python_ap.vm.tables import Tables

# Test harmonics
harmonic_amps = [
    30890,  # H1
    32767,  # H2 - should be strongest
    6926,   # H3
    23969,  # H4
    3673,   # H5
    6138,   # H6
    1164,   # H7
    1228,   # H8
]

print("Input harmonic_amps:", harmonic_amps)

# Create frame with Mode B
frame = VoiceFrame(
    note=69,  # A4
    coeff_osc_mode=1,  # Mode B with custom harmonics
    H_request=8,
    gain_q15=32767,
    beta_target_q15=0,  # Pure carrier for testing
    partial_amps_q15=harmonic_amps,
)

print(f"\nFrame partial_amps_q15: {frame.partial_amps_q15}")
print(f"Frame coeff_osc_mode: {frame.coeff_osc_mode}")

# Decode
tables = Tables()
cfg = decode_frame(frame, tables, prev_eta_q15=0, prev_gain_q15=16384)

print(f"\nDecoded coeff_osc_params.custom_harmonics: {cfg.coeff_osc_params.custom_harmonics}")
print(f"Decoded coeff_osc_params.osc_mode: {cfg.coeff_osc_params.osc_mode}")

# Create state and render with CCS path
state = VMState.create(voice_id=0)
state.reset_for_note_on(note=69)

# Render multiple blocks to get stable output
all_output = []
for _ in range(10):
    state, output = vm_step_block(state, frame)
    all_output.extend(output.tolist())

output = np.array(all_output[-640:], dtype=np.int16)  # Last 10 blocks

print(f"\nOutput samples (first 10): {output[:10]}")
print(f"Output peak: {np.max(np.abs(output))}")

# Check Mode B state
print(f"\nMode B state:")
print(f"  H_active: {state.mode_b_state.H_active}")
print(f"  weights.a_h_q15[:8]: {list(state.mode_b_state.weights.a_h_q15[:8])}")

# FFT to check spectrum
from numpy.fft import rfft, rfftfreq
spectrum = np.abs(rfft(output.astype(np.float64)))
freqs = rfftfreq(len(output), 1/48000)

print("\nSpectrum peaks (should show H1-H8 at 440, 880, 1320, etc.):")
# Expected frequencies for A4 harmonics
expected_freqs = [440 * i for i in range(1, 9)]
print(f"Expected: {expected_freqs}")

peak_idx = np.argsort(spectrum)[-15:][::-1]
for i in peak_idx:
    if freqs[i] > 50:
        # Check which harmonic this is
        for h, ef in enumerate(expected_freqs, 1):
            if abs(freqs[i] - ef) < 50:
                print(f"  {freqs[i]:.1f} Hz (H{h}): {spectrum[i]:.0f}")
                break
        else:
            print(f"  {freqs[i]:.1f} Hz (SPUR): {spectrum[i]:.0f}")
