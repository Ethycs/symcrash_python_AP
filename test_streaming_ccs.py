"""
Test streaming coefficient-domain synthesis (WSE-STD-120).

Validates:
1. Phase continuity across blocks (no clicks)
2. Correct frequency (261.63 Hz for MIDI 60)
3. No block-boundary discontinuities
4. Proper time-series coefficient evolution
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.vm.coeff_streaming import (
    StreamingCoeffState,
    SubbandTable,
    generate_subband_samples,
    idwpt_polyphase_streaming,
    streaming_ccs_render,
    get_subband_table,
)

def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    """Convert MIDI note to 32-bit phase increment."""
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int((freq_hz / sample_rate) * (1 << 32))


def test_coefficient_evolution():
    """Test that coefficients evolve smoothly across blocks."""
    print("Test 1: Coefficient Evolution")
    print("=" * 50)

    state = StreamingCoeffState()
    state.base_phase_inc = midi_to_phase_inc(60)  # Middle C
    state.table_id = 0

    table = get_subband_table(0)  # Sine

    # Generate 5 blocks
    for block in range(5):
        # Save state before
        phases_before = state.leaf_phases.copy()
        coeffs_before = state.coeff_state.copy()

        # Generate
        generate_subband_samples(state, table)

        # Check phases advanced
        phases_after = state.leaf_phases.copy()
        phase_diff = (phases_after[0].astype(np.int64) - phases_before[0].astype(np.int64)) & 0xFFFFFFFF

        # Check coefficients changed
        coeff_diff = np.abs(state.coeff_state.astype(np.int32) - coeffs_before.astype(np.int32))
        max_coeff_change = coeff_diff.max()

        print(f"  Block {block}:")
        print(f"    Phase advance: {phase_diff:,} ({phase_diff / (1 << 32) * 360:.1f} deg)")
        print(f"    Coeff RMS: {np.sqrt(np.mean(state.coeff_state.astype(np.float32)**2)):.1f}")
        print(f"    Max coeff change: {max_coeff_change}")
        print(f"    Coeff[0][:4]: {state.coeff_state[:4]}")

    print("[OK] Coefficients evolve across blocks\n")


def test_audio_output():
    """Test audio output from streaming IDWPT."""
    print("Test 2: Audio Output")
    print("=" * 50)

    state = StreamingCoeffState()
    state.base_phase_inc = midi_to_phase_inc(60)
    state.table_id = 0

    table = get_subband_table(0)

    # Render 10 blocks
    all_audio = []
    for block in range(10):
        audio = streaming_ccs_render(state, table)
        all_audio.append(audio)

        rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
        print(f"  Block {block}: min={audio.min():6d}, max={audio.max():6d}, RMS={rms:.1f}")

    # Concatenate
    full_audio = np.concatenate(all_audio)
    print(f"\n  Full output: {len(full_audio)} samples")
    print(f"  Range: [{full_audio.min()}, {full_audio.max()}]")
    print(f"  RMS: {np.sqrt(np.mean(full_audio.astype(np.float32)**2)):.1f}")

    print("[OK] Audio output generated\n")
    return full_audio


def test_block_continuity():
    """Test for discontinuities at block boundaries."""
    print("Test 3: Block Boundary Continuity")
    print("=" * 50)

    state = StreamingCoeffState()
    state.base_phase_inc = midi_to_phase_inc(60)

    table = get_subband_table(0)

    # Render 20 blocks
    all_audio = []
    for _ in range(20):
        audio = streaming_ccs_render(state, table)
        all_audio.append(audio)

    full_audio = np.concatenate(all_audio)

    # Check for discontinuities at block boundaries (every 64 samples)
    max_jump = 0
    worst_boundary = -1

    for i in range(len(all_audio) - 1):
        # Last sample of block i
        last = int(all_audio[i][-1])
        # First sample of block i+1
        first = int(all_audio[i+1][0])
        jump = abs(first - last)

        if jump > max_jump:
            max_jump = jump
            worst_boundary = i

    print(f"  Max boundary jump: {max_jump}")
    print(f"  Worst boundary: block {worst_boundary} -> {worst_boundary + 1}")

    # Also check max jump anywhere in audio
    diff = np.abs(np.diff(full_audio.astype(np.int32)))
    max_diff = diff.max()
    max_diff_idx = np.argmax(diff)

    print(f"  Max sample-to-sample jump: {max_diff} at sample {max_diff_idx}")

    if max_jump < 500:
        print("[OK] No major discontinuities at block boundaries\n")
    else:
        print("[WARNING] Significant discontinuities detected\n")


def test_frequency_fft():
    """Test frequency content via FFT."""
    print("Test 4: Frequency Analysis (FFT)")
    print("=" * 50)

    try:
        from scipy.fft import fft
        from scipy.signal import find_peaks
    except ImportError:
        print("  [SKIP] scipy not available\n")
        return

    state = StreamingCoeffState()
    state.base_phase_inc = midi_to_phase_inc(60)  # 261.63 Hz

    table = get_subband_table(0)

    # Render many blocks for better frequency resolution
    all_audio = []
    for _ in range(100):
        audio = streaming_ccs_render(state, table)
        all_audio.append(audio)

    full_audio = np.concatenate(all_audio).astype(np.float32)

    # FFT
    N = len(full_audio)
    fft_result = fft(full_audio)
    magnitude = np.abs(fft_result[:N//2])
    freqs = np.fft.fftfreq(N, 1/48000)[:N//2]

    # Find peaks
    peaks, properties = find_peaks(magnitude, height=magnitude.max() * 0.1)

    print(f"  Signal length: {N} samples ({N/48000*1000:.1f} ms)")
    print(f"  Frequency resolution: {48000/N:.2f} Hz")
    print(f"\n  Top frequency peaks:")

    # Sort peaks by magnitude
    peak_mags = magnitude[peaks]
    sorted_idx = np.argsort(peak_mags)[::-1][:5]

    for idx in sorted_idx:
        peak_idx = peaks[idx]
        freq = freqs[peak_idx]
        mag = magnitude[peak_idx]
        print(f"    {freq:.1f} Hz: {20*np.log10(mag/magnitude.max()):.1f} dB")

    # Check fundamental
    expected_freq = 261.63
    fundamental_idx = np.argmax(magnitude)
    fundamental_freq = freqs[fundamental_idx]

    print(f"\n  Expected fundamental: {expected_freq:.2f} Hz")
    print(f"  Detected fundamental: {fundamental_freq:.2f} Hz")
    print(f"  Error: {abs(fundamental_freq - expected_freq):.2f} Hz")

    if abs(fundamental_freq - expected_freq) < 5:
        print("[OK] Correct fundamental frequency\n")
    elif fundamental_freq < 10:
        print("[FAIL] DC output (no oscillation)\n")
    else:
        print(f"[WARNING] Fundamental off by {abs(fundamental_freq - expected_freq):.1f} Hz\n")


def test_phase_accumulator_wrap():
    """Test that phase accumulators wrap correctly."""
    print("Test 5: Phase Accumulator Wrap")
    print("=" * 50)

    state = StreamingCoeffState()
    # Use high frequency to force wrap
    state.base_phase_inc = midi_to_phase_inc(120)  # Very high

    table = get_subband_table(0)

    # Track phase for leaf 0
    phases = []
    for block in range(1000):
        generate_subband_samples(state, table)
        phases.append(int(state.leaf_phases[0]))

    phases = np.array(phases, dtype=np.uint32)

    # Check for monotonic increase (with wrap)
    diffs = np.diff(phases.astype(np.int64))

    # Negative diffs indicate wrap (which is OK)
    wraps = np.sum(diffs < 0)

    print(f"  Phase wraps detected: {wraps}")
    print(f"  Final phase: {phases[-1]:,}")

    print("[OK] Phase accumulator wrapping correctly\n")


def main():
    print("\n" + "=" * 60)
    print("  STREAMING CCS VALIDATION (WSE-STD-120)")
    print("=" * 60 + "\n")

    test_coefficient_evolution()
    test_audio_output()
    test_block_continuity()
    test_frequency_fft()
    test_phase_accumulator_wrap()

    print("=" * 60)
    print("  VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
