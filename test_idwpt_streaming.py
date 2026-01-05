"""
Test streaming IDWPT implementation.

Validates:
1. Basic functionality (produces output)
2. Comparison with block IDWPT
3. Block boundary continuity
4. Frequency correctness
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.wavelet.idwpt_streaming import (
    IDWPTStreamingState,
    IDWPTBlockState,
    IDWPTCrossfadeState,
    idwpt_streaming_step,
    idwpt_streaming_block,
    idwpt_block_with_state,
    idwpt_with_crossfade,
    interpolate_coeffs_for_block,
)
from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
from symcrash_python_ap.wavelet.sine_core import sine_q15


def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    freq_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int((freq_hz / sample_rate) * (1 << 32))


def generate_sine_block(phase_start: int, phase_inc: int, num_samples: int = 64) -> tuple:
    audio = np.zeros(num_samples, dtype=np.int16)
    phase = phase_start
    for i in range(num_samples):
        audio[i] = sine_q15(phase)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return audio, phase


def test_streaming_basic():
    """Test basic streaming IDWPT functionality."""
    print("Test: Streaming IDWPT Basic")
    print("=" * 60)

    state = IDWPTStreamingState()

    # Generate a simple sine and get its coefficients
    phase_inc = midi_to_phase_inc(60)
    audio, _ = generate_sine_block(0, phase_inc)
    coeffs = np.array(fwpt_jit(audio.tolist(), headroom_bits=1), dtype=np.int16)

    print(f"  Input coefficients: range=[{coeffs.min()}, {coeffs.max()}]")

    # Process through streaming IDWPT
    output = idwpt_streaming_block(state, coeffs, headroom_bits=1)

    print(f"  Output: range=[{output.min()}, {output.max()}]")
    print(f"  Output RMS: {np.sqrt(np.mean(output.astype(np.float32)**2)):.1f}")
    print(f"  First 8 samples: {output[:8]}")

    # Compare with block IDWPT
    block_output = idwpt_jit(coeffs.tolist(), headroom_bits=1)
    block_arr = np.array(block_output, dtype=np.int16)

    print(f"\n  Block IDWPT: range=[{block_arr.min()}, {block_arr.max()}]")

    # Error
    error = np.abs(output.astype(np.int32) - block_arr.astype(np.int32))
    print(f"  Max error vs block: {error.max()}")

    print("[OK] Streaming IDWPT produces output\n")


def test_multi_block_continuity():
    """Test continuity across multiple blocks."""
    print("Test: Multi-Block Continuity (Streaming)")
    print("=" * 60)

    state = IDWPTStreamingState()
    phase_inc = midi_to_phase_inc(60)
    phase = 0

    all_audio = []

    for block in range(10):
        # Generate reference audio and get coefficients
        ref_audio, phase = generate_sine_block(phase, phase_inc)
        coeffs = np.array(fwpt_jit(ref_audio.tolist(), headroom_bits=1), dtype=np.int16)

        # Streaming IDWPT
        output = idwpt_streaming_block(state, coeffs, headroom_bits=1)
        all_audio.append(output)

        if block < 3:
            print(f"  Block {block}: range=[{output.min():6d}, {output.max():6d}]")

    # Check boundary continuity
    max_jump = 0
    worst_block = -1
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        if jump > max_jump:
            max_jump = jump
            worst_block = i

    print(f"\n  Max boundary jump: {max_jump} (at block {worst_block})")

    # Compare with block IDWPT (no state)
    phase = 0
    all_audio_block = []
    for block in range(10):
        ref_audio, phase = generate_sine_block(phase, phase_inc)
        coeffs = fwpt_jit(ref_audio.tolist(), headroom_bits=1)
        output = idwpt_jit(coeffs, headroom_bits=1)
        all_audio_block.append(np.array(output, dtype=np.int16))

    max_jump_block = 0
    for i in range(len(all_audio_block) - 1):
        jump = abs(int(all_audio_block[i][-1]) - int(all_audio_block[i+1][0]))
        max_jump_block = max(max_jump_block, jump)

    print(f"  Block IDWPT max boundary jump: {max_jump_block}")

    if max_jump < max_jump_block:
        print("[OK] Streaming reduces boundary jumps!\n")
    else:
        print("[INFO] Streaming boundary jumps similar to block\n")


def test_block_with_state():
    """Test block IDWPT with persistent state."""
    print("Test: Block IDWPT with State")
    print("=" * 60)

    state = IDWPTBlockState()
    phase_inc = midi_to_phase_inc(60)
    phase = 0

    all_audio = []

    for block in range(10):
        # Generate reference audio and get coefficients
        ref_audio, phase = generate_sine_block(phase, phase_inc)
        coeffs = fwpt_jit(ref_audio.tolist(), headroom_bits=1)

        # Block IDWPT with state
        output = idwpt_block_with_state(state, coeffs, headroom_bits=1)
        output_arr = np.array(output, dtype=np.int16)
        all_audio.append(output_arr)

        if block < 3:
            print(f"  Block {block}: range=[{output_arr.min():6d}, {output_arr.max():6d}]")

    # Check boundary continuity
    max_jump = 0
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        max_jump = max(max_jump, jump)

    print(f"\n  Max boundary jump: {max_jump}")

    # FFT analysis
    full_audio = np.concatenate(all_audio)
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.2f} Hz (expected 261.63 Hz)")
    except ImportError:
        pass

    print("[OK] Block with state test complete\n")


def test_ccs_simulation():
    """
    Simulate full CCS pipeline:
    CoeffLUT lookup → streaming IDWPT → audio

    Uses the new sin/cos basis CoeffLUT with phase rotation.
    """
    print("Test: CCS Simulation (CoeffLUT + Streaming IDWPT)")
    print("=" * 60)

    from symcrash_python_ap.vm.coeff_lut import get_coeff_lut

    lut = get_coeff_lut(0)  # Sine
    state = IDWPTStreamingState()
    phase_inc = midi_to_phase_inc(60)
    phase = 0

    all_audio = []

    for block in range(20):
        # Get coefficients from LUT using 32-bit phase (new interface)
        coeffs = lut.get_coeffs_for_phase(phase)

        # Streaming IDWPT
        output = idwpt_streaming_block(state, coeffs, headroom_bits=1)
        all_audio.append(output)

        if block < 3:
            phase_deg = (phase / (1 << 32)) * 360
            print(f"  Block {block}: phase={phase_deg:.1f} deg, range=[{output.min():6d}, {output.max():6d}]")

        # Advance phase for next block (64 samples)
        phase = (phase + phase_inc * 64) & 0xFFFFFFFF

    # Check boundary continuity
    max_jump = 0
    worst_block = -1
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        if jump > max_jump:
            max_jump = jump
            worst_block = i

    print(f"\n  Max boundary jump: {max_jump} (at block {worst_block})")

    # FFT analysis
    full_audio = np.concatenate(all_audio)
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.2f} Hz (expected 261.63 Hz)")

        if abs(peak_freq - 261.63) < 10 and max_jump < 1000:
            print("[OK] CCS simulation successful!\n")
        else:
            print("[WARNING] CCS simulation has issues\n")
    except ImportError:
        print("[OK] CCS simulation complete (no scipy for FFT)\n")


def test_crossfade_continuity():
    """Test IDWPT with crossfade for boundary continuity."""
    print("Test: IDWPT with Crossfade")
    print("=" * 60)

    state = IDWPTCrossfadeState()
    phase_inc = midi_to_phase_inc(60)
    phase = 0

    all_audio = []
    all_audio_nocrossfade = []

    for block in range(20):
        # Generate reference audio and get coefficients
        ref_audio, phase = generate_sine_block(phase, phase_inc)
        coeffs = np.array(fwpt_jit(ref_audio.tolist(), headroom_bits=1), dtype=np.int16)

        # IDWPT with crossfade
        output = idwpt_with_crossfade(state, coeffs, headroom_bits=1, crossfade_samples=8)
        all_audio.append(output)

        # Plain block IDWPT for comparison
        output_plain = np.array(idwpt_jit(coeffs.tolist(), headroom_bits=1), dtype=np.int16)
        all_audio_nocrossfade.append(output_plain)

        if block < 3:
            print(f"  Block {block}: crossfade range=[{output.min():6d}, {output.max():6d}]")

    # Reset phase for second pass
    phase = 0

    # Check boundary continuity - with crossfade
    max_jump_cf = 0
    worst_block_cf = -1
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        if jump > max_jump_cf:
            max_jump_cf = jump
            worst_block_cf = i

    # Check boundary continuity - no crossfade
    max_jump_plain = 0
    for i in range(len(all_audio_nocrossfade) - 1):
        jump = abs(int(all_audio_nocrossfade[i][-1]) - int(all_audio_nocrossfade[i+1][0]))
        max_jump_plain = max(max_jump_plain, jump)

    print(f"\n  Max boundary jump (with crossfade): {max_jump_cf} (at block {worst_block_cf})")
    print(f"  Max boundary jump (no crossfade): {max_jump_plain}")
    print(f"  Improvement: {max_jump_plain - max_jump_cf}")

    # FFT analysis
    full_audio = np.concatenate(all_audio)
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.2f} Hz (expected 261.63 Hz)")

        # Check for harmonics/artifacts
        noise_floor = np.median(fft_result)
        num_peaks = np.sum(fft_result > 10 * noise_floor)
        print(f"  Significant peaks above noise: {num_peaks}")
    except ImportError:
        pass

    if max_jump_cf < max_jump_plain:
        print("[OK] Crossfade reduces boundary discontinuities!\n")
    else:
        print("[INFO] Crossfade effect minimal\n")


def test_ccs_with_crossfade():
    """
    Full CCS pipeline with crossfade:
    CoeffLUT lookup → crossfade IDWPT → audio

    Uses the new sin/cos basis CoeffLUT with phase rotation.
    """
    print("Test: CCS with Crossfade (CoeffLUT + Crossfade IDWPT)")
    print("=" * 60)

    from symcrash_python_ap.vm.coeff_lut import get_coeff_lut

    lut = get_coeff_lut(0)  # Sine
    state = IDWPTCrossfadeState()
    phase_inc = midi_to_phase_inc(60)
    phase = 0

    all_audio = []

    for block in range(20):
        # Get coefficients from LUT using 32-bit phase (new interface)
        coeffs = lut.get_coeffs_for_phase(phase)

        # IDWPT with crossfade
        output = idwpt_with_crossfade(state, coeffs, headroom_bits=1, crossfade_samples=8)
        all_audio.append(output)

        if block < 3:
            phase_deg = (phase / (1 << 32)) * 360
            print(f"  Block {block}: phase={phase_deg:.1f} deg, range=[{output.min():6d}, {output.max():6d}]")

        # Advance phase for next block (64 samples)
        phase = (phase + phase_inc * 64) & 0xFFFFFFFF

    # Check boundary continuity
    max_jump = 0
    worst_block = -1
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        if jump > max_jump:
            max_jump = jump
            worst_block = i

    print(f"\n  Max boundary jump: {max_jump} (at block {worst_block})")

    # FFT analysis
    full_audio = np.concatenate(all_audio)
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.2f} Hz (expected 261.63 Hz)")

        # Success criteria
        freq_ok = abs(peak_freq - 261.63) < 10
        jump_ok = max_jump < 1000

        if freq_ok and jump_ok:
            print("[OK] CCS with crossfade successful!\n")
        elif freq_ok:
            print(f"[WARNING] Frequency OK but boundary jump {max_jump} > 1000\n")
        else:
            print("[WARNING] CCS with crossfade has issues\n")
    except ImportError:
        print("[OK] CCS with crossfade complete (no scipy for FFT)\n")


if __name__ == '__main__':
    test_streaming_basic()
    test_multi_block_continuity()
    test_block_with_state()
    test_crossfade_continuity()
    test_ccs_simulation()
    test_ccs_with_crossfade()
