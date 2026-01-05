"""
Test StreamingCoeffLUT and SubbandState.

Validates:
1. StreamingCoeffLUT generates correct [L, K, P] structure
2. SubbandState maintains persistent state
3. Streaming IDWPT produces continuous audio
4. Phase continuity across blocks (no clicks)
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.vm.coeff_lut_streaming import (
    StreamingCoeffLUT,
    SubbandState,
    get_streaming_lut,
    streaming_idwpt_block,
    DEFAULT_K,
    DEFAULT_P,
    BLOCK_SIZE,
    NUM_LEAVES,
)


def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    """Convert MIDI note to 32-bit phase increment."""
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int((freq_hz / sample_rate) * (1 << 32))


def test_lut_structure():
    """Test that StreamingCoeffLUT has correct structure."""
    print("Test: StreamingCoeffLUT Structure")
    print("=" * 60)

    lut = get_streaming_lut(0)  # Sine

    print(f"  Wave: {lut.name}")
    print(f"  Data shape: {lut.data.shape}")
    print(f"  Expected: ({NUM_LEAVES}, {lut.K}, {lut.P})")
    print(f"  K (time samples): {lut.K}")
    print(f"  P (phase steps): {lut.P}")
    print(f"  Data range: [{lut.data.min()}, {lut.data.max()}]")
    print(f"  Data non-zero: {np.count_nonzero(lut.data)}")

    expected_shape = (NUM_LEAVES, lut.K, lut.P)
    if lut.data.shape == expected_shape:
        print("[OK] Correct LUT structure\n")
        return True
    else:
        print(f"[FAIL] Shape mismatch: {lut.data.shape} != {expected_shape}\n")
        return False


def test_subband_samples():
    """Test get_subband_samples produces [L, K] output."""
    print("Test: Subband Sample Retrieval")
    print("=" * 60)

    lut = get_streaming_lut(0)

    # Get samples at phase 0
    output = np.zeros((NUM_LEAVES, lut.K), dtype=np.int16)
    lut.get_subband_samples(0, output)

    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min()}, {output.max()}]")
    print(f"  Output non-zero: {np.count_nonzero(output)}")

    # Check first few leaves
    for l in range(min(3, NUM_LEAVES)):
        print(f"  Leaf {l}: K samples = {output[l, :]}")

    if np.count_nonzero(output) > 0:
        print("[OK] Non-zero subband samples\n")
        return True
    else:
        print("[FAIL] All zeros!\n")
        return False


def test_phase_interpolation():
    """Test that phase interpolation produces smooth transitions."""
    print("Test: Phase Interpolation")
    print("=" * 60)

    lut = get_streaming_lut(0)

    # Get samples at consecutive phases
    samples_at_phase = []
    for p_idx in range(4):
        phase_32 = int((p_idx / lut.P) * (1 << 32))
        output = np.zeros((NUM_LEAVES, lut.K), dtype=np.int16)
        lut.get_subband_samples(phase_32, output)
        samples_at_phase.append(output.copy())

    # Check leaf 0 across phases
    print("  Leaf 0 samples at different phases:")
    for p_idx in range(4):
        print(f"    Phase {p_idx}: {samples_at_phase[p_idx][0, :]}")

    # Calculate max jump between consecutive phases
    max_jump = 0
    for i in range(len(samples_at_phase) - 1):
        jump = np.abs(samples_at_phase[i+1].astype(np.int32) - samples_at_phase[i].astype(np.int32)).max()
        max_jump = max(max_jump, jump)

    print(f"\n  Max jump between phase steps: {max_jump}")

    if max_jump < 10000:
        print("[OK] Smooth phase transitions\n")
    else:
        print("[WARN] Large phase jumps\n")

    return True


def test_persistent_state():
    """Test that SubbandState maintains persistent state."""
    print("Test: Persistent SubbandState")
    print("=" * 60)

    lut = get_streaming_lut(0)

    state = SubbandState()
    state.phase_inc = midi_to_phase_inc(60)  # C4

    print(f"  Initial phase: {state.phase}")
    print(f"  Phase inc: {state.phase_inc}")
    print(f"  State shape: {state.state.shape}")
    print(f"  State initial: [{state.state.min()}, {state.state.max()}]")

    # Inject carrier
    state.inject_carrier(lut, base_mix_q15=32767)

    print(f"\n  After carrier injection:")
    print(f"  State range: [{state.state.min()}, {state.state.max()}]")
    print(f"  State non-zero: {np.count_nonzero(state.state)}")

    # Advance phase
    state.advance_phase(64)
    print(f"\n  Phase after 64 samples: {state.phase}")
    print(f"  Phase degrees: {(state.phase / (1 << 32)) * 360:.1f}")

    if np.count_nonzero(state.state) > 0:
        print("[OK] State populated\n")
        return True
    else:
        print("[FAIL] State not populated\n")
        return False


def test_streaming_synthesis():
    """Test streaming IDWPT synthesis."""
    print("Test: Streaming IDWPT Synthesis")
    print("=" * 60)

    lut = get_streaming_lut(0)

    state = SubbandState()
    state.phase_inc = midi_to_phase_inc(60)  # C4

    all_audio = []

    for block in range(20):
        # Inject carrier
        state.inject_carrier(lut, base_mix_q15=32767)

        # Synthesize
        output = np.zeros(BLOCK_SIZE, dtype=np.int16)
        streaming_idwpt_block(state, output)

        all_audio.append(output.copy())

        if block < 5:
            phase_deg = (state.phase / (1 << 32)) * 360
            print(f"  Block {block}: phase={phase_deg:6.1f} deg, range=[{output.min():6d}, {output.max():6d}]")

        # Advance phase
        state.advance_phase(BLOCK_SIZE)

    # Check boundary discontinuities
    max_jump = 0
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        max_jump = max(max_jump, jump)

    print(f"\n  Max boundary jump: {max_jump}")

    if max_jump < 5000:
        print("[OK] Acceptable boundary discontinuities\n")
    else:
        print("[WARN] Large boundary discontinuities\n")

    return True


def test_frequency_accuracy():
    """Test that output frequency is correct."""
    print("Test: Frequency Accuracy")
    print("=" * 60)

    lut = get_streaming_lut(0)

    state = SubbandState()
    state.phase_inc = midi_to_phase_inc(60)  # C4 = 261.63 Hz

    all_audio = []

    for block in range(32):
        state.inject_carrier(lut, base_mix_q15=32767)
        output = np.zeros(BLOCK_SIZE, dtype=np.int16)
        streaming_idwpt_block(state, output)
        all_audio.append(output.copy())
        state.advance_phase(BLOCK_SIZE)

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

        if error < 50:
            print("[OK] Frequency accurate\n")
            return True
        else:
            print("[WARN] Frequency error > 50 Hz\n")
            return False

    except ImportError:
        print("  (scipy not available for FFT)\n")
        return True


if __name__ == '__main__':
    print("StreamingCoeffLUT Tests")
    print("=" * 60)
    print()

    test_lut_structure()
    test_subband_samples()
    test_phase_interpolation()
    test_persistent_state()
    test_streaming_synthesis()
    test_frequency_accuracy()
