"""
Test direct synthesis: Generate audio, get coefficients via FWPT,
modify coefficients, reconstruct via IDWPT.

This validates the IDEAL path before adding oscillator complexity.
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


def test_continuous_synthesis():
    """
    Synthesize a continuous sine wave by:
    1. Generating reference audio blocks
    2. FWPT to get coefficients
    3. IDWPT to reconstruct

    This should be perfect (since FWPT/IDWPT are inverses).
    The key is maintaining phase continuity in step 1.
    """
    print("Test: Continuous FWPT/IDWPT Synthesis")
    print("=" * 60)

    phase_inc = midi_to_phase_inc(60)
    phase = 0

    all_reference = []
    all_reconstructed = []

    for block in range(20):
        # Generate reference audio (continuous phase)
        ref_audio, phase = generate_sine_block(phase, phase_inc)
        all_reference.append(ref_audio)

        # FWPT -> IDWPT roundtrip
        coeffs = fwpt_jit(ref_audio.tolist(), headroom_bits=1)
        recon = idwpt_jit(coeffs, headroom_bits=1)
        recon_arr = np.array(recon, dtype=np.int16)
        all_reconstructed.append(recon_arr)

        # Error
        error = np.abs(ref_audio.astype(np.int32) - recon_arr.astype(np.int32))

        if block < 5:
            print(f"  Block {block}: ref_range=[{ref_audio.min():6d},{ref_audio.max():6d}], "
                  f"recon_range=[{recon_arr.min():6d},{recon_arr.max():6d}], "
                  f"max_error={error.max()}")

    full_ref = np.concatenate(all_reference)
    full_recon = np.concatenate(all_reconstructed)

    # Check boundary continuity in reconstructed signal
    max_jump_ref = 0
    max_jump_recon = 0
    for i in range(len(all_reference) - 1):
        jump_ref = abs(int(all_reference[i][-1]) - int(all_reference[i+1][0]))
        jump_recon = abs(int(all_reconstructed[i][-1]) - int(all_reconstructed[i+1][0]))
        max_jump_ref = max(max_jump_ref, jump_ref)
        max_jump_recon = max(max_jump_recon, jump_recon)

    print(f"\n  Max boundary jump (reference): {max_jump_ref}")
    print(f"  Max boundary jump (reconstructed): {max_jump_recon}")

    # FFT analysis
    try:
        from scipy.fft import fft
        N = len(full_recon)
        fft_result = np.abs(fft(full_recon.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"\n  FFT peak (reconstructed): {peak_freq:.2f} Hz (expected 261.63 Hz)")

        # Show top 5 peaks
        sorted_idx = np.argsort(fft_result)[::-1][:5]
        print("  Top 5 frequency peaks:")
        for idx in sorted_idx:
            print(f"    {freqs[idx]:.1f} Hz: {20*np.log10(fft_result[idx]/fft_result[peak_idx]):.1f} dB")

    except ImportError:
        print("  (scipy not available for FFT)")


def test_coefficient_persistence():
    """
    Test what happens when we PERSIST coefficients across blocks
    instead of regenerating them.

    This simulates the streaming architecture where coefficients
    are state that evolves, not snapshots.
    """
    print("\n\nTest: Persistent Coefficients (Wrong Approach)")
    print("=" * 60)

    phase_inc = midi_to_phase_inc(60)
    phase = 0

    # Get initial coefficients from first block
    first_audio, phase = generate_sine_block(phase, phase_inc)
    persistent_coeffs = fwpt_jit(first_audio.tolist(), headroom_bits=1)

    all_audio = []
    all_audio.append(first_audio)  # First block is correct

    # Use same coefficients for subsequent blocks
    for block in range(1, 10):
        recon = idwpt_jit(persistent_coeffs, headroom_bits=1)
        recon_arr = np.array(recon, dtype=np.int16)
        all_audio.append(recon_arr)

        if block < 5:
            print(f"  Block {block}: range=[{recon_arr.min():6d},{recon_arr.max():6d}]")

    full_audio = np.concatenate(all_audio)

    # Check boundaries
    max_jump = 0
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        max_jump = max(max_jump, jump)

    print(f"\n  Max boundary jump: {max_jump}")
    print("  (This is WRONG because coefficients are static)")


def test_coefficient_evolution():
    """
    Test evolving coefficients by interpolating between
    per-block FWPT snapshots.

    This is closer to what the oscillator should do.
    """
    print("\n\nTest: Interpolated Coefficient Evolution")
    print("=" * 60)

    phase_inc = midi_to_phase_inc(60)

    # Pre-generate several reference blocks and their coefficients
    num_refs = 16
    ref_coeffs = []
    phase = 0

    for i in range(num_refs):
        audio, phase = generate_sine_block(phase, phase_inc)
        coeffs = np.array(fwpt_jit(audio.tolist(), headroom_bits=1), dtype=np.float32)
        ref_coeffs.append(coeffs)

    ref_coeffs = np.array(ref_coeffs)  # [num_refs, 64]

    # Now synthesize by interpolating between reference coefficients
    all_audio = []
    phase = 0

    for block in range(20):
        # Which reference blocks to interpolate between?
        # Use floating-point block index
        block_frac = (block % num_refs) + 0.5  # Centered in block
        idx0 = int(block_frac) % num_refs
        idx1 = (idx0 + 1) % num_refs
        alpha = block_frac - int(block_frac)

        # Interpolate coefficients
        coeffs_interp = ref_coeffs[idx0] * (1 - alpha) + ref_coeffs[idx1] * alpha
        coeffs_int = np.clip(coeffs_interp, -32768, 32767).astype(np.int16)

        # IDWPT
        audio = idwpt_jit(coeffs_int.tolist(), headroom_bits=1)
        audio_arr = np.array(audio, dtype=np.int16)
        all_audio.append(audio_arr)

        if block < 5:
            print(f"  Block {block}: idx={idx0}->{idx1} (a={alpha:.2f}), range=[{audio_arr.min():6d},{audio_arr.max():6d}]")

    full_audio = np.concatenate(all_audio)

    # Check boundaries
    max_jump = 0
    for i in range(len(all_audio) - 1):
        jump = abs(int(all_audio[i][-1]) - int(all_audio[i+1][0]))
        max_jump = max(max_jump, jump)

    print(f"\n  Max boundary jump: {max_jump}")

    # FFT
    try:
        from scipy.fft import fft
        N = len(full_audio)
        fft_result = np.abs(fft(full_audio.astype(np.float32)))[:N//2]
        freqs = np.fft.fftfreq(N, 1/48000)[:N//2]
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        print(f"  FFT peak: {peak_freq:.2f} Hz")
    except ImportError:
        pass


# test_correct_oscillator was deleted: it exercised FastCoeffTable2D from
# vm/coeff_tables_fast.py, which was removed when the quadrature coefficient
# table oscillator was replaced by StreamingCoeffLUT (commit d9332ee,
# "StreamingCoeffLUT into VM CCS Path"); the streaming path is covered by
# tests/conformance/test_ccs_conformance.py.


if __name__ == '__main__':
    test_continuous_synthesis()
    test_coefficient_persistence()
    test_coefficient_evolution()
