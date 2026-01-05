"""
Test the correct streaming interpretation:

Instead of 4 time samples per leaf being passed to IDWPT at once,
we call IDWPT 4 times per block, each time with the "current" sample
from each leaf's ring buffer.

This gives us:
- 64 leaves
- Each leaf has a ring buffer of K samples
- Per block: K calls to IDWPT, each producing 64/K = 16 samples
- Total: 64 samples output

Wait, that's not right either. IDWPT(64 coeffs) produces 64 samples.
If we call it K times, we get 64*K samples.

Let me think again...

The decimation factor of the filterbank is 1:1 for leaves.
64 input samples -> 64 leaf coefficients
64 leaf coefficients -> 64 output samples

So "time samples per leaf" must mean something else.
Perhaps it's about the POLYPHASE structure of the filter itself?

In a polyphase filterbank:
- The synthesis filter h[n] is split into K polyphase components
- h_k[m] = h[k + mK] for k = 0..K-1
- Each component operates at decimated rate

For IDWPT with 64 leaves and K=4 polyphase components:
- We'd have 64*4 = 256 filter taps total
- Each leaf's 4 "time samples" are the 4 polyphase contributions

This matches the coeff_state[256] = [64 leaves][4 polyphase] layout!

Let me test this interpretation by implementing proper polyphase IDWPT.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
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


def test_phase_interpolation():
    """
    Test smooth interpolation between phase-indexed coefficient tables.

    This is the approach used by coeff_osc.py:
    - Store sin/cos basis coefficients for each phase step
    - Interpolate between them using current phase
    - This gives smooth coefficient evolution without regeneration
    """
    print("Test: Phase Interpolation Approach")
    print("=" * 60)

    phase_inc = midi_to_phase_inc(60)

    # Generate coefficient tables for a few phase steps
    num_phase_steps = 8
    sin_tables = []
    cos_tables = []

    for p in range(num_phase_steps):
        # Phase for sin (0, 45, 90, ... degrees)
        phase_32 = int((p / num_phase_steps) * (1 << 32))
        audio_sin, _ = generate_sine_block(phase_32, phase_inc)

        # Phase for cos (shifted 90 degrees)
        phase_cos = (phase_32 + 0x40000000) & 0xFFFFFFFF
        audio_cos, _ = generate_sine_block(phase_cos, phase_inc)

        c_sin = np.array(fwpt_jit(audio_sin.tolist(), headroom_bits=1), dtype=np.float32)
        c_cos = np.array(fwpt_jit(audio_cos.tolist(), headroom_bits=1), dtype=np.float32)

        sin_tables.append(c_sin)
        cos_tables.append(c_cos)

    sin_tables = np.array(sin_tables)  # [num_phase_steps, 64]
    cos_tables = np.array(cos_tables)

    # Now synthesize by interpolating between table entries
    print("\nSynthesizing 10 blocks using table interpolation...")

    all_audio = []
    phase = 0

    for block in range(10):
        # Interpolate coefficients for each of 64 output samples
        block_audio = np.zeros(64, dtype=np.int16)

        for i in range(64):
            # Current phase in [0, 1) range
            phase_frac = (phase / (1 << 32))

            # Which table entries to interpolate
            phase_idx = phase_frac * num_phase_steps
            idx0 = int(phase_idx) % num_phase_steps
            idx1 = (idx0 + 1) % num_phase_steps
            alpha = phase_idx - int(phase_idx)

            # Interpolate between table entries
            c_sin = sin_tables[idx0] * (1 - alpha) + sin_tables[idx1] * alpha
            c_cos = cos_tables[idx0] * (1 - alpha) + cos_tables[idx1] * alpha

            # Phase rotation: c = cos(phi)*c_sin + sin(phi)*c_cos
            cos_phi = np.cos(2 * np.pi * phase_frac)
            sin_phi = np.sin(2 * np.pi * phase_frac)

            coeffs = (cos_phi * c_sin + sin_phi * c_cos).astype(np.int16)

            # IDWPT to get one audio sample... but IDWPT gives 64 samples!
            # This is the problem - we can't get sample-by-sample output from standard IDWPT

            # For now, just compute the first sample
            audio = idwpt_jit(coeffs.tolist(), headroom_bits=1)
            block_audio[i] = audio[0]  # Only take first sample (WRONG but illustrative)

            # Advance phase
            phase = (phase + phase_inc) & 0xFFFFFFFF

        all_audio.append(block_audio)
        rms = np.sqrt(np.mean(block_audio.astype(np.float32)**2))
        print(f"  Block {block}: min={block_audio.min():6d}, max={block_audio.max():6d}, RMS={rms:.1f}")

    full_audio = np.concatenate(all_audio)

    # FFT analysis
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"\n  FFT peak: {peak_freq:.1f} Hz (expected 261.63 Hz)")
    except ImportError:
        pass


def test_proper_streaming():
    """
    The CORRECT streaming approach:

    Instead of generating coefficients that evolve,
    we generate AUDIO that evolves, and use FWPT to get coefficients.

    Wait, that's backwards. Let me think again...

    The REAL insight: The "coefficient oscillator" should generate
    coefficient TIME SERIES that, when fed through IDWPT, produce
    the desired audio time series.

    For a sinusoid at frequency f:
    - Audio: sin(2*pi*f*t)
    - Coefficients: ???

    The coefficients are NOT simply the Fourier transform.
    They're the wavelet packet coefficients, which depend on
    both frequency AND time/phase.

    The solution: Store coefficients as a function of PHASE,
    not just magnitude. As the oscillator advances, we look up
    coefficients from a phase-indexed table and interpolate.

    This is exactly what coeff_tables_fast.py does!
    The issue was in HOW we use the table.
    """
    print("\n\nTest: Correct Streaming via Phase-Indexed Tables")
    print("=" * 60)

    from symcrash_python_ap.vm.coeff_tables_fast import (
        FastCoeffTable2D,
        generate_coeffs_for_note_phase,
    )

    # Use the fast table
    table = FastCoeffTable2D()

    phase_inc = midi_to_phase_inc(60)
    phase = 0

    all_audio = []

    for block in range(10):
        # Get phase index (0-63) from current phase
        phase_idx = (phase >> 26) & 0x3F

        # Get sin/cos coefficients for this phase
        c_sin, c_cos = table.get_coeffs(60, phase_idx)

        # Phase rotation
        phase_frac = phase / (1 << 32)
        cos_phi = np.cos(2 * np.pi * phase_frac)
        sin_phi = np.sin(2 * np.pi * phase_frac)

        coeffs = (cos_phi * c_sin.astype(np.float32) + sin_phi * c_cos.astype(np.float32))
        coeffs_int = np.clip(coeffs, -32768, 32767).astype(np.int16)

        # IDWPT
        audio = idwpt_jit(coeffs_int.tolist(), headroom_bits=1)
        audio_arr = np.array(audio, dtype=np.int16)

        all_audio.append(audio_arr)

        rms = np.sqrt(np.mean(audio_arr.astype(np.float32)**2))
        print(f"  Block {block}: phase_idx={phase_idx:2d}, min={audio_arr.min():6d}, max={audio_arr.max():6d}, RMS={rms:.1f}")

        # Advance phase for next block (64 samples)
        phase = (phase + phase_inc * 64) & 0xFFFFFFFF

    full_audio = np.concatenate(all_audio)

    # Check for block boundary discontinuities
    max_jump = 0
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        if jump > max_jump:
            max_jump = jump
    print(f"\n  Max block boundary jump: {max_jump}")

    # FFT analysis
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.1f} Hz (expected 261.63 Hz)")

        # Check for DC
        dc_level = np.abs(fft_result[0])
        ac_level = np.abs(fft_result[peak_idx])
        print(f"  DC level: {dc_level:.1f}, AC level: {ac_level:.1f}, ratio: {dc_level/ac_level:.2f}")

    except ImportError:
        pass


if __name__ == '__main__':
    # test_phase_interpolation()  # Skip - illustrative but wrong
    test_proper_streaming()
