"""Verify the CCS path is producing clean sine waves."""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm.tables import DEFAULT_TABLES
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.coeff_lut_streaming import get_lut_for_note


def test_lut_directly():
    """Test the LUT produces correct frequency."""
    print("=" * 60)
    print("LUT DIRECT TEST")
    print("=" * 60)

    lut = get_lut_for_note(wave_id=0, midi_note=69)
    print(f"LUT target frequency: {lut.target_freq_hz:.2f} Hz")
    print(f"LUT K (time samples): {lut.K}")
    print(f"LUT P (phase positions): {lut.P}")

    # Get coefficients at phase=0
    subband_output = np.zeros((64, lut.K), dtype=np.int16)
    lut.get_subband_samples(phase_32=0, output=subband_output)

    # Check k=0 coefficients - which leaves have energy?
    coeffs_k0 = subband_output[:, 0]
    print(f"\nCoeffs k=0: min={coeffs_k0.min()}, max={coeffs_k0.max()}")
    print(f"Non-zero leaves: {np.count_nonzero(coeffs_k0)}")

    # Find which leaves have the most energy
    leaf_energy = np.abs(coeffs_k0)
    top_leaves = np.argsort(leaf_energy)[-5:][::-1]
    print("Top 5 leaves by energy:")
    for leaf in top_leaves:
        print(f"  Leaf {leaf}: {coeffs_k0[leaf]}")

    # Run IDWPT on k=0 coefficients
    from symcrash_python_ap.wavelet.fast_lifting import idwpt_jit

    audio = idwpt_jit(coeffs_k0.tolist(), headroom_bits=1)
    audio = np.array(audio, dtype=np.int16)

    print(f"\nIDWPT output: min={audio.min()}, max={audio.max()}")

    # FFT of single block
    spectrum = np.abs(rfft(audio.astype(float)))
    freqs = rfftfreq(64, 1 / 48000)

    peak_idx = np.argmax(spectrum[1:]) + 1
    peak_freq = freqs[peak_idx]
    print(f"Single block peak frequency: {peak_freq:.1f} Hz")

    # Now test using multiple blocks with polyphase
    print("\n--- POLYPHASE MULTI-BLOCK TEST ---")
    from symcrash_python_ap.vm.coeff_lut_streaming import SubbandState, streaming_idwpt_block

    state = SubbandState()
    state.phase_inc = int((440.0 / 48000) * (1 << 32))

    all_samples = []
    for _ in range(100):
        # Get subband samples for current phase
        lut.get_subband_samples(state.phase, state.state)

        # Polyphase IDWPT
        output = np.zeros(64, dtype=np.int16)
        streaming_idwpt_block(state, output, headroom_bits=1)
        all_samples.extend(output.tolist())

        # Advance phase
        state.phase = (state.phase + state.phase_inc * 64) & 0xFFFFFFFF

    all_samples = np.array(all_samples, dtype=np.int16)

    # FFT of multi-block output
    window = windows.blackman(len(all_samples))
    windowed = all_samples.astype(float) * window
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(all_samples), 1 / 48000)

    # Find peak
    fund_idx = np.argmax(spectrum[10:]) + 10
    fund_freq = freqs[fund_idx]
    print(f"Multi-block peak frequency: {fund_freq:.2f} Hz (expected 440 Hz)")

    print()


def main():
    # Generate 200 blocks at 440 Hz (note 69)
    frame = VoiceFrame(
        voice_id=0,
        note=69,
        cents=0,
        gain_q15=32767,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
        beta_target_q15=0,
        partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
    )
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=69, cents=0)

    # Check CCS state
    print(f"CCS State note: {state.ccs_state.note}")
    print(f"CCS State wave_id: {state.ccs_state.wave_id}")
    print(f"CCS State phase: {state.ccs_state.phase}")

    samples = []
    for _ in range(200):
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        samples.extend(output.tolist())

    samples = np.array(samples, dtype=np.int16)
    print(f"Generated {len(samples)} samples")
    print(f"RMS: {np.sqrt(np.mean(samples.astype(float)**2)):.1f}")
    print(f"Peak: {np.max(np.abs(samples))}")

    # FFT analysis
    window = windows.blackman(len(samples))
    windowed = samples.astype(float) * window
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(samples), 1 / 48000)

    # Find fundamental
    fund_idx = np.argmax(spectrum[10:2000]) + 10
    fund_freq = freqs[fund_idx]
    fund_mag = spectrum[fund_idx]
    fund_db = 20 * np.log10(fund_mag + 1e-10)

    print(f"\nFundamental: {fund_freq:.2f} Hz (expected 440 Hz)")
    print(f"Fundamental magnitude: {fund_db:.1f} dB")

    # Check harmonics
    print("\nHarmonic analysis (relative to fundamental):")
    for h in range(1, 8):
        expected = 440 * h
        search_start = int(expected * 0.95 * len(freqs) / 24000)
        search_end = int(expected * 1.05 * len(freqs) / 24000)
        if search_end < len(spectrum):
            h_idx = search_start + np.argmax(spectrum[search_start:search_end])
            h_freq = freqs[h_idx]
            h_mag = spectrum[h_idx]
            h_db = 20 * np.log10(h_mag / fund_mag + 1e-10)
            print(f"  H{h}: {h_freq:.1f} Hz, {h_db:+.1f} dB")

    # Check for non-harmonic content (sidebands)
    print("\nNon-harmonic content check:")
    sidebands = []
    for i, (f, m) in enumerate(zip(freqs, spectrum)):
        if f < 50 or m < fund_mag * 0.001:  # Skip DC and very weak
            continue
        # Check if near any harmonic
        is_harmonic = False
        for h in range(1, 20):
            expected = fund_freq * h
            if abs(f - expected) < expected * 0.03:  # Within 3%
                is_harmonic = True
                break
        if not is_harmonic:
            db = 20 * np.log10(m / fund_mag + 1e-10)
            if db > -60:  # Only report significant peaks
                sidebands.append((f, db))

    if sidebands:
        print(f"  Found {len(sidebands)} non-harmonic peaks above -60 dB:")
        for f, db in sidebands[:10]:
            print(f"    {f:.1f} Hz: {db:+.1f} dB")
    else:
        print("  No significant non-harmonic content detected")

    # Summary
    print("\n" + "=" * 60)
    freq_error = abs(fund_freq - 440.0)
    if freq_error < 2.0 and len(sidebands) < 50:
        print("RESULT: Clean sine wave at correct frequency")
    elif freq_error < 2.0:
        print(f"RESULT: Correct frequency but {len(sidebands)} spectral artifacts")
    else:
        print(f"RESULT: Frequency error {freq_error:.1f} Hz")


if __name__ == "__main__":
    test_lut_directly()
    main()
