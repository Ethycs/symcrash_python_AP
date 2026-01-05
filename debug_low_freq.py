"""Debug low-frequency accuracy issue."""
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
from scipy.fft import fft


def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def analyze_frequency(note: int):
    """Analyze frequency accuracy for a MIDI note."""
    expected = midi_to_hz(note)
    print(f"\n{'='*60}")
    print(f"Note {note}: Expected {expected:.2f} Hz")
    print(f"{'='*60}")

    # Calculate cycle parameters
    samples_per_cycle = SAMPLE_RATE / expected
    blocks_per_cycle = samples_per_cycle / BLOCK_SIZE
    print(f"  Samples per cycle: {samples_per_cycle:.1f}")
    print(f"  Blocks per cycle: {blocks_per_cycle:.2f}")

    # Generate LUT
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('symcrash_python_ap.vm.coeff_lut_streaming')
    logger.setLevel(logging.DEBUG)

    lut = StreamingCoeffLUT.generate_for_frequency(
        wave_id=0,
        name=f"sine_{note}",
        target_freq_hz=expected,
        generate_sample_fn=sine_q15,
        S=16,  # Higher oversampling
    )

    # Synthesize
    state = SubbandState()
    state.phase_inc = int((expected / SAMPLE_RATE) * (1 << 32))

    print(f"\n  Phase inc: {state.phase_inc}")
    print(f"  Phase inc per block: {state.phase_inc * BLOCK_SIZE}")
    print(f"  Cycles per block: {(state.phase_inc * BLOCK_SIZE) / (1 << 32):.4f}")

    all_audio = []
    for block_idx in range(64):  # More blocks for low freq
        state.inject_carrier(lut, base_mix_q15=32767)
        output = np.zeros(BLOCK_SIZE, dtype=np.int16)
        streaming_idwpt_block(state, output)
        all_audio.append(output.copy())

        if block_idx < 3:
            phase_idx = (state.phase >> 26) & 0x3F
            print(f"  Block {block_idx}: phase_idx={phase_idx}, audio range=[{output.min()}, {output.max()}]")

        state.advance_phase(BLOCK_SIZE)

    synth_audio = np.concatenate(all_audio)

    # FFT
    N = len(synth_audio)
    fft_result = np.abs(fft(synth_audio.astype(np.float32)))[:N//2]
    freqs = np.fft.fftfreq(N, 1/SAMPLE_RATE)[:N//2]

    # Find peaks
    peak_idx = np.argmax(fft_result)
    measured = freqs[peak_idx]

    # Also show top 5 peaks
    top_indices = np.argsort(fft_result)[-5:][::-1]
    print(f"\n  Top 5 frequency peaks:")
    for idx in top_indices:
        print(f"    {freqs[idx]:.1f} Hz: magnitude {fft_result[idx]:.0f}")

    error_cents = 1200 * np.log2(measured / expected) if measured > 0 and expected > 0 else 999
    print(f"\n  Measured: {measured:.2f} Hz")
    print(f"  Error: {error_cents:.1f} cents")


if __name__ == '__main__':
    analyze_frequency(48)  # C3 = 130.8 Hz (problematic)
    analyze_frequency(60)  # C4 = 261.6 Hz (works well)
