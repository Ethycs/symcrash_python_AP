"""
Test FastCoeffTable2D and coeff_osc_generate_coeffs.

Validates:
1. FastCoeffTable2D generates non-zero coefficients
2. coeff_osc_generate_coeffs produces non-zero output
3. IDWPT of generated coefficients produces correct frequency
4. Phase continuity across blocks
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.vm.coeff_tables_fast import (
    FastCoeffTable2D,
    get_fast_coeff_table,
    generate_coeffs_for_note_phase,
    midi_to_phase_inc,
)
from symcrash_python_ap.vm.coeff_osc import (
    CoeffOscState,
    coeff_osc_generate_coeffs,
)
from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit


def test_fast_table_basic():
    """Test that FastCoeffTable2D generates non-zero coefficients."""
    print("Test: FastCoeffTable2D Basic Generation")
    print("=" * 60)

    table = get_fast_coeff_table()

    # Get coefficients for C4 (MIDI 60), phase 0
    c_sin, c_cos = table.get_coeffs(60, 0)

    print(f"  c_sin shape: {c_sin.shape}")
    print(f"  c_sin range: [{c_sin.min()}, {c_sin.max()}]")
    print(f"  c_sin non-zero: {np.count_nonzero(c_sin)}")
    print(f"  c_cos shape: {c_cos.shape}")
    print(f"  c_cos range: [{c_cos.min()}, {c_cos.max()}]")
    print(f"  c_cos non-zero: {np.count_nonzero(c_cos)}")

    # IDWPT to get audio
    audio_sin = np.array(idwpt_jit(c_sin.tolist(), headroom_bits=1), dtype=np.int16)
    audio_cos = np.array(idwpt_jit(c_cos.tolist(), headroom_bits=1), dtype=np.int16)

    print(f"\n  IDWPT(c_sin) range: [{audio_sin.min()}, {audio_sin.max()}]")
    print(f"  IDWPT(c_cos) range: [{audio_cos.min()}, {audio_cos.max()}]")

    if c_sin.max() > 0 or c_sin.min() < 0:
        print("[OK] FastCoeffTable2D generates non-zero coefficients\n")
    else:
        print("[FAIL] FastCoeffTable2D generates zero coefficients!\n")
        return False

    return True


def test_coeff_osc_generate():
    """Test coeff_osc_generate_coeffs produces non-zero output."""
    print("Test: coeff_osc_generate_coeffs")
    print("=" * 60)

    table = get_fast_coeff_table()

    # Create oscillator state for C4
    state = CoeffOscState()
    state.table_id = 60  # MIDI note C4
    state.base_phase_inc = midi_to_phase_inc(60)
    state.target_phase_inc = state.base_phase_inc
    state.node_phases[0] = 0  # Start at phase 0

    print(f"  table_id: {state.table_id}")
    print(f"  phase_inc: {state.base_phase_inc}")

    # Generate coefficients
    output = np.zeros(64, dtype=np.int16)
    coeff_osc_generate_coeffs(state, table, output)

    print(f"\n  Output range: [{output.min()}, {output.max()}]")
    print(f"  Output non-zero: {np.count_nonzero(output)}")
    print(f"  Output RMS: {np.sqrt(np.mean(output.astype(float)**2)):.1f}")

    # Check phase advanced
    print(f"\n  Phase after 1 block: {state.node_phases[0]}")

    # IDWPT to audio
    audio = np.array(idwpt_jit(output.tolist(), headroom_bits=1), dtype=np.int16)
    print(f"  IDWPT output range: [{audio.min()}, {audio.max()}]")
    print(f"  IDWPT output RMS: {np.sqrt(np.mean(audio.astype(float)**2)):.1f}")

    if output.max() > 0 or output.min() < 0:
        print("[OK] coeff_osc_generate_coeffs produces non-zero coefficients\n")
    else:
        print("[FAIL] coeff_osc_generate_coeffs produces zero coefficients!\n")
        return False

    return True


def test_frequency_accuracy():
    """Test that IDWPT of generated coefficients produces correct frequency."""
    print("Test: Frequency Accuracy")
    print("=" * 60)

    table = get_fast_coeff_table()

    # Test C4 (261.63 Hz)
    state = CoeffOscState()
    state.table_id = 60
    state.base_phase_inc = midi_to_phase_inc(60)
    state.target_phase_inc = state.base_phase_inc
    state.node_phases[0] = 0

    all_audio = []
    for block in range(32):
        output = np.zeros(64, dtype=np.int16)
        coeff_osc_generate_coeffs(state, table, output)
        audio = np.array(idwpt_jit(output.tolist(), headroom_bits=1), dtype=np.int16)
        all_audio.append(audio)

    full_audio = np.concatenate(all_audio)

    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]

        expected = 261.63
        error = abs(peak_freq - expected)

        print(f"  Expected: {expected:.2f} Hz")
        print(f"  Measured: {peak_freq:.2f} Hz")
        print(f"  Error: {error:.2f} Hz")

        if error < 20:
            print("[OK] Frequency accurate within 20 Hz\n")
            return True
        else:
            print("[WARN] Frequency error > 20 Hz\n")
            return False

    except ImportError:
        print("  (scipy not available for FFT)\n")
        return True


def test_phase_continuity():
    """Test phase continuity across blocks."""
    print("Test: Phase Continuity")
    print("=" * 60)

    table = get_fast_coeff_table()

    state = CoeffOscState()
    state.table_id = 60
    state.base_phase_inc = midi_to_phase_inc(60)
    state.target_phase_inc = state.base_phase_inc
    state.node_phases[0] = 0

    all_audio = []
    phases = []

    for block in range(20):
        phases.append(state.node_phases[0])

        output = np.zeros(64, dtype=np.int16)
        coeff_osc_generate_coeffs(state, table, output)
        audio = np.array(idwpt_jit(output.tolist(), headroom_bits=1), dtype=np.int16)
        all_audio.append(audio)

        if block < 5:
            phase_deg = (state.node_phases[0] / (1 << 32)) * 360
            print(f"  Block {block}: phase={phase_deg:6.1f} deg, range=[{audio.min():6d}, {audio.max():6d}]")

    # Check phase is advancing (handle 32-bit wraparound)
    phase_deltas = []
    for i in range(len(phases)-1):
        delta = (phases[i+1] - phases[i]) & 0xFFFFFFFF
        # Handle signed interpretation
        if delta > 0x80000000:
            delta = delta - 0x100000000
        phase_deltas.append(delta)
    avg_delta = sum(phase_deltas) / len(phase_deltas)
    expected_delta = state.base_phase_inc * 64

    print(f"\n  Avg phase delta: {avg_delta}")
    print(f"  Expected delta: {expected_delta}")

    # Check block boundary discontinuities
    max_jump = 0
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        max_jump = max(max_jump, jump)

    print(f"  Max boundary jump: {max_jump}")

    if max_jump < 5000:
        print("[OK] Boundary discontinuities acceptable\n")
    else:
        print("[WARN] Large boundary discontinuities\n")

    if abs(avg_delta - expected_delta) < 1000:
        print("[OK] Phase advancing correctly\n")
        return True
    else:
        print("[FAIL] Phase not advancing correctly!\n")
        return False


def test_debug_phase_rotation():
    """Debug the phase rotation math."""
    print("Test: Phase Rotation Debug")
    print("=" * 60)

    from symcrash_python_ap.wavelet.sine_core import sine_q15, cosine_q15
    from symcrash_python_ap.fixed import RSHIFT_RNE

    table = get_fast_coeff_table()

    # Get coefficients for MIDI 60, phase 0
    c_sin, c_cos = table.get_coeffs(60, 0)

    print(f"  c_sin[0:5]: {c_sin[:5]}")
    print(f"  c_cos[0:5]: {c_cos[:5]}")

    # Test phase rotation at phase 0
    phase = 0
    cos_phi = cosine_q15(phase)
    sin_phi = sine_q15(phase)

    print(f"\n  At phase=0:")
    print(f"    cos_phi: {cos_phi}")
    print(f"    sin_phi: {sin_phi}")

    # Manual phase rotation for first coefficient
    val_sin = RSHIFT_RNE(cos_phi * int(c_sin[0]), 15)
    val_cos = RSHIFT_RNE(sin_phi * int(c_cos[0]), 15)
    val = val_sin + val_cos

    print(f"    val_sin: {val_sin}")
    print(f"    val_cos: {val_cos}")
    print(f"    total: {val}")

    # Test at phase 90 degrees
    phase = 0x40000000  # 90 degrees
    cos_phi = cosine_q15(phase)
    sin_phi = sine_q15(phase)

    print(f"\n  At phase=90 deg:")
    print(f"    cos_phi: {cos_phi}")
    print(f"    sin_phi: {sin_phi}")

    val_sin = RSHIFT_RNE(cos_phi * int(c_sin[0]), 15)
    val_cos = RSHIFT_RNE(sin_phi * int(c_cos[0]), 15)
    val = val_sin + val_cos

    print(f"    val_sin: {val_sin}")
    print(f"    val_cos: {val_cos}")
    print(f"    total: {val}")

    print()


if __name__ == '__main__':
    test_fast_table_basic()
    test_debug_phase_rotation()
    test_coeff_osc_generate()
    test_phase_continuity()
    test_frequency_accuracy()
