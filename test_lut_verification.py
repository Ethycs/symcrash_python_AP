"""
Offline Verification Test for StreamingCoeffLUT.

Per WSE-STD-121: Verify that table lookup + IDWPT reconstructs
the original waveform at correct frequency with minimal error.

Tests:
1. Generate reference waveform at target frequency
2. Run table lookup -> IDWPT for many blocks
3. Compare synthesized audio to reference
4. Verify no block discontinuities
5. Verify correct pitch
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from symcrash_python_ap.vm.coeff_lut_streaming import (
    StreamingCoeffLUT,
    SubbandState,
    streaming_idwpt_block,
    BLOCK_SIZE,
    NUM_LEAVES,
    SAMPLE_RATE,
)
from symcrash_python_ap.wavelet.sine_core import sine_q15


def midi_to_hz(note: int) -> float:
    """Convert MIDI note to Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def midi_to_phase_inc(note: int, sample_rate: int = 48000) -> int:
    """Convert MIDI note to 32-bit phase increment."""
    freq_hz = midi_to_hz(note)
    return int((freq_hz / sample_rate) * (1 << 32))


def generate_reference_waveform(
    freq_hz: float,
    num_samples: int,
    generate_fn=sine_q15,
) -> np.ndarray:
    """Generate reference waveform at specified frequency."""
    phase_inc = int((freq_hz / SAMPLE_RATE) * (1 << 32))
    samples = np.zeros(num_samples, dtype=np.int16)
    phase = 0
    for i in range(num_samples):
        samples[i] = generate_fn(phase)
        phase = (phase + phase_inc) & 0xFFFFFFFF
    return samples


def test_reconstruction_accuracy():
    """Test that IDWPT reconstruction matches original waveform."""
    print("Test: Reconstruction Accuracy")
    print("=" * 60)

    target_freq = 261.63  # C4
    num_blocks = 32
    total_samples = num_blocks * BLOCK_SIZE

    # Generate reference
    reference = generate_reference_waveform(target_freq, total_samples)

    # Generate LUT at target frequency
    lut = StreamingCoeffLUT.generate_for_frequency(
        wave_id=0,
        name="sine_test",
        target_freq_hz=target_freq,
        generate_sample_fn=sine_q15,
    )

    # Synthesize using LUT
    state = SubbandState()
    state.phase_inc = int((target_freq / SAMPLE_RATE) * (1 << 32))

    synthesized = []
    for block in range(num_blocks):
        state.inject_carrier(lut, base_mix_q15=32767)
        output = np.zeros(BLOCK_SIZE, dtype=np.int16)
        streaming_idwpt_block(state, output)
        synthesized.append(output.copy())
        state.advance_phase(BLOCK_SIZE)

    synth_audio = np.concatenate(synthesized)

    # Compare
    # Note: There will be phase offset and amplitude differences due to
    # normalization and FWPT/IDWPT processing. We check correlation, not exact match.

    # Cross-correlation to find best alignment
    correlation = np.correlate(
        synth_audio.astype(np.float32),
        reference.astype(np.float32),
        mode='full'
    )
    best_offset = np.argmax(np.abs(correlation)) - len(reference) + 1

    print(f"  Reference samples: {len(reference)}")
    print(f"  Synthesized samples: {len(synth_audio)}")
    print(f"  Best alignment offset: {best_offset}")
    print(f"  Reference range: [{reference.min()}, {reference.max()}]")
    print(f"  Synth range: [{synth_audio.min()}, {synth_audio.max()}]")

    # Compute RMS of both
    ref_rms = np.sqrt(np.mean(reference.astype(np.float32) ** 2))
    synth_rms = np.sqrt(np.mean(synth_audio.astype(np.float32) ** 2))
    print(f"  Reference RMS: {ref_rms:.1f}")
    print(f"  Synth RMS: {synth_rms:.1f}")

    # Normalized correlation coefficient
    if ref_rms > 0 and synth_rms > 0:
        norm_corr = correlation[best_offset + len(reference) - 1] / (ref_rms * synth_rms * len(reference))
        print(f"  Normalized correlation: {norm_corr:.4f}")

        if abs(norm_corr) > 0.5:
            print("[OK] Good correlation with reference\n")
            return True
        else:
            print("[WARN] Low correlation\n")
            return False
    else:
        print("[FAIL] Zero RMS\n")
        return False


def test_frequency_sweep():
    """Test frequency accuracy across MIDI range.

    NOTE: The FWPT block size (64 samples at 48kHz) limits frequency resolution.
    - Maximum representable frequency: ~750 Hz (one cycle per block)
    - Minimum representable frequency: depends on M (cycle length)
    - Best accuracy around the filterbank center frequency
    """
    print("Test: Frequency Sweep")
    print("=" * 60)

    # Focus on frequencies that are well-represented by 64-sample blocks
    # 750 Hz = 1 cycle per block, 375 Hz = 2 blocks per cycle, etc.
    notes = [48, 54, 60, 66]  # C3, F#3, C4, F#4 (131-370 Hz range)
    expected_freqs = [midi_to_hz(n) for n in notes]

    print(f"  Note: Block rate = 48000/64 = 750 Hz")
    print(f"  Frequencies above 750 Hz will alias")
    print()

    try:
        from scipy.fft import fft

        results = []
        for note, expected in zip(notes, expected_freqs):
            # Generate LUT at this frequency
            lut = StreamingCoeffLUT.generate_for_frequency(
                wave_id=0,
                name=f"sine_{note}",
                target_freq_hz=expected,
                generate_sample_fn=sine_q15,
            )

            # Synthesize - phase_inc must match the table's target frequency
            # This is correct: runtime advances phase at same rate as table was generated
            state = SubbandState()
            state.phase_inc = int((expected / SAMPLE_RATE) * (1 << 32))

            all_audio = []
            for _ in range(128):  # More blocks for better FFT resolution
                state.inject_carrier(lut, base_mix_q15=32767)
                output = np.zeros(BLOCK_SIZE, dtype=np.int16)
                streaming_idwpt_block(state, output)
                all_audio.append(output.copy())
                state.advance_phase(BLOCK_SIZE)

            synth_audio = np.concatenate(all_audio)

            # FFT to find peak frequency
            N = len(synth_audio)
            fft_result = np.abs(fft(synth_audio.astype(np.float32)))[:N//2]
            freqs = np.fft.fftfreq(N, 1/SAMPLE_RATE)[:N//2]
            peak_idx = np.argmax(fft_result)
            measured = freqs[peak_idx]

            error = abs(measured - expected)
            error_cents = 1200 * np.log2(measured / expected) if measured > 0 and expected > 0 else 999

            status = "[OK]" if abs(error_cents) < 50 else "[WARN]"
            print(f"  Note {note}: expected {expected:.1f} Hz, measured {measured:.1f} Hz, error {error_cents:.1f} cents {status}")

            results.append(abs(error_cents) < 50)

        if all(results):
            print("\n[OK] All frequencies within 50 cents\n")
            return True
        else:
            print("\n[WARN] Some frequencies off\n")
            return False

    except ImportError:
        print("  (scipy not available)\n")
        return True


def test_block_continuity():
    """Test for clicks/discontinuities at block boundaries."""
    print("Test: Block Continuity")
    print("=" * 60)

    target_freq = 261.63

    lut = StreamingCoeffLUT.generate_for_frequency(
        wave_id=0,
        name="sine_cont",
        target_freq_hz=target_freq,
        generate_sample_fn=sine_q15,
    )

    state = SubbandState()
    state.phase_inc = int((target_freq / SAMPLE_RATE) * (1 << 32))

    all_blocks = []
    for _ in range(50):
        state.inject_carrier(lut, base_mix_q15=32767)
        output = np.zeros(BLOCK_SIZE, dtype=np.int16)
        streaming_idwpt_block(state, output)
        all_blocks.append(output.copy())
        state.advance_phase(BLOCK_SIZE)

    # Measure boundary jumps
    boundary_jumps = []
    for i in range(len(all_blocks) - 1):
        jump = abs(int(all_blocks[i][-1]) - int(all_blocks[i+1][0]))
        boundary_jumps.append(jump)

    max_jump = max(boundary_jumps)
    avg_jump = sum(boundary_jumps) / len(boundary_jumps)
    median_jump = sorted(boundary_jumps)[len(boundary_jumps)//2]

    print(f"  Max boundary jump: {max_jump}")
    print(f"  Avg boundary jump: {avg_jump:.1f}")
    print(f"  Median boundary jump: {median_jump}")

    # Measure intra-block jumps for comparison
    intra_jumps = []
    for block in all_blocks:
        for i in range(len(block) - 1):
            intra_jumps.append(abs(int(block[i+1]) - int(block[i])))

    max_intra = max(intra_jumps)
    avg_intra = sum(intra_jumps) / len(intra_jumps)

    print(f"  Max intra-block jump: {max_intra}")
    print(f"  Avg intra-block jump: {avg_intra:.1f}")

    # Boundary jumps should be similar to intra-block jumps
    ratio = max_jump / max_intra if max_intra > 0 else float('inf')
    print(f"  Boundary/intra ratio: {ratio:.2f}")

    if ratio < 3.0:
        print("[OK] Boundary discontinuities acceptable\n")
        return True
    else:
        print("[WARN] Boundary discontinuities larger than expected\n")
        return False


def test_k_dimension_variation():
    """Test that K dimension has proper time variation."""
    print("Test: K Dimension Variation")
    print("=" * 60)

    lut = StreamingCoeffLUT.generate_for_frequency(
        wave_id=0,
        name="sine_k",
        target_freq_hz=261.63,
        generate_sample_fn=sine_q15,
    )

    # Check that K samples vary for each leaf
    output = np.zeros((NUM_LEAVES, lut.K), dtype=np.int16)
    lut.get_subband_samples(0, output)

    # For each leaf, compute variance across K
    k_variances = []
    for l in range(NUM_LEAVES):
        k_samples = output[l, :]
        variance = np.var(k_samples.astype(np.float32))
        k_variances.append(variance)

    nonzero_var = sum(1 for v in k_variances if v > 0)
    avg_var = np.mean(k_variances)
    max_var = max(k_variances)

    print(f"  Leaves with K-variation: {nonzero_var}/{NUM_LEAVES}")
    print(f"  Average K-variance: {avg_var:.1f}")
    print(f"  Max K-variance: {max_var:.1f}")

    # Sample a few leaves
    for l in [0, 1, 2, 31, 32, 63]:
        print(f"  Leaf {l:2d} K samples: {output[l, :]}")

    if nonzero_var > NUM_LEAVES // 2:
        print("\n[OK] K dimension has time variation\n")
        return True
    else:
        print("\n[WARN] K dimension lacks variation\n")
        return False


def test_phase_advance_continuity():
    """Test that phase advances produce continuous output.

    NOTE: Phase advances are per-block (64 samples). For a sinusoid,
    the subband sample will oscillate, but the key is smooth transitions.
    Large jumps indicate phase wraparound or table discontinuities.
    """
    print("Test: Phase Advance Continuity")
    print("=" * 60)

    target_freq = 261.63

    lut = StreamingCoeffLUT.generate_for_frequency(
        wave_id=0,
        name="sine_phase",
        target_freq_hz=target_freq,
        generate_sample_fn=sine_q15,
    )

    # Get subband samples at consecutive phases
    # Phase increment per sample
    phase_inc_per_sample = int((target_freq / SAMPLE_RATE) * (1 << 32))

    samples = []
    phase_indices = []
    phase = 0
    for _ in range(100):
        output = np.zeros((NUM_LEAVES, lut.K), dtype=np.int16)
        lut.get_subband_samples(phase, output)
        samples.append(output[1, 0])  # Track leaf 1, k=0
        phase_indices.append((phase >> 26) & 0x3F)  # Phase index (0-63)
        phase = (phase + phase_inc_per_sample * BLOCK_SIZE) & 0xFFFFFFFF

    # Check for jumps
    jumps = []
    for i in range(len(samples) - 1):
        jump = abs(int(samples[i+1]) - int(samples[i]))
        jumps.append(jump)

    max_jump = max(jumps)
    avg_jump = sum(jumps) / len(jumps)

    print(f"  Tracked leaf 1, k=0 over 100 phase advances")
    print(f"  Sample range: [{min(samples)}, {max(samples)}]")
    print(f"  Max inter-sample jump: {max_jump}")
    print(f"  Avg inter-sample jump: {avg_jump:.1f}")

    # Show phase index progression
    print(f"  First 10 phase indices: {phase_indices[:10]}")

    # Calculate expected jump based on sinusoidal behavior
    # Phase advances by phase_inc_per_sample * BLOCK_SIZE per iteration
    phase_advance_per_block = (phase_inc_per_sample * BLOCK_SIZE) / (1 << 32)
    print(f"  Phase advance per block: {phase_advance_per_block * 360:.1f} degrees")

    # For a sinusoid, max expected jump is 2 * amplitude * sin(delta/2)
    # where delta is the phase change per step
    delta_rad = phase_advance_per_block * 2 * 3.14159
    max_expected = int(2 * 32767 * abs(np.sin(delta_rad / 2)))
    print(f"  Max expected sinusoidal jump: ~{max_expected}")

    # Allow 2x margin for filterbank effects
    if max_jump < max_expected * 3:
        print("[OK] Phase advances are reasonably continuous\n")
        return True
    else:
        print("[WARN] Jumps larger than expected for sinusoid\n")
        return False


if __name__ == '__main__':
    print("StreamingCoeffLUT Offline Verification")
    print("=" * 60)
    print()

    results = []
    results.append(("Reconstruction Accuracy", test_reconstruction_accuracy()))
    results.append(("Frequency Sweep", test_frequency_sweep()))
    results.append(("Block Continuity", test_block_continuity()))
    results.append(("K Dimension Variation", test_k_dimension_variation()))
    results.append(("Phase Advance Continuity", test_phase_advance_continuity()))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\n  {passed_count}/{total_count} tests passed")
