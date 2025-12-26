"""Debug block boundary artifacts in FWPT->IDWPT."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np

from symcrash_python_ap.wavelet.carrier import CarrierOscillator
from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude
from symcrash_python_ap.audio_io import AudioWriter

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)

Q15_ONE = 32767
BLOCK_SIZE = 64

def generate_partial_amps(wave_type: WaveType, num_partials: int = 8):
    return [get_partial_amplitude(wave_type, k + 1) for k in range(num_partials)]

def main():
    print("=" * 60)
    print("BLOCK BOUNDARY DEBUG")
    print("=" * 60)

    note = 69
    saw_partials = generate_partial_amps(WaveType.SAW, 8)

    carrier = CarrierOscillator()
    carrier.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier.gate_on(0)

    tree_set = PacketTreeSet()

    # Render 10 blocks and look at boundaries
    print("\nBlock boundary analysis:")
    print("Looking at last sample of block N and first sample of block N+1")

    prev_carrier_last = None
    prev_wet_last = None

    carrier_samples = []
    wet_samples = []

    for block_idx in range(20):
        x_c = carrier.render_block(0, BLOCK_SIZE)
        x_w = tree_set.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)

        carrier_samples.extend(x_c)
        wet_samples.extend(x_w)

        if prev_carrier_last is not None:
            carrier_jump = abs(x_c[0] - prev_carrier_last)
            wet_jump = abs(x_w[0] - prev_wet_last)

            print(f"Block {block_idx-1}->{block_idx}: Carrier jump={carrier_jump:5d}, Wet jump={wet_jump:5d}")

        prev_carrier_last = x_c[-1]
        prev_wet_last = x_w[-1]

    # Look at the actual waveform around boundaries
    print("\n\nDetailed boundary inspection (blocks 9-10):")

    # Get the samples around block boundary (samples 576-640, blocks 9-10)
    boundary_start = 9 * BLOCK_SIZE - 5
    boundary_end = 10 * BLOCK_SIZE + 5

    print(f"Carrier samples {boundary_start} to {boundary_end}:")
    print(carrier_samples[boundary_start:boundary_end])

    print(f"\nWet samples {boundary_start} to {boundary_end}:")
    print(wet_samples[boundary_start:boundary_end])

    # Compute difference
    diff = [abs(c - w) for c, w in zip(carrier_samples, wet_samples)]

    print(f"\nMax carrier-wet difference: {max(diff)}")
    print(f"Mean carrier-wet difference: {np.mean(diff):.2f}")

    # Check if the issue is cumulative
    print("\n\nCumulative error over time:")
    for i in range(0, len(carrier_samples), BLOCK_SIZE * 10):
        block_num = i // BLOCK_SIZE
        chunk_diff = diff[i:i+BLOCK_SIZE]
        if chunk_diff:
            print(f"Block {block_num}: max_diff={max(chunk_diff):5d}, mean_diff={np.mean(chunk_diff):.1f}")

    # Save just carrier for comparison
    print("\n\nSaving pure carrier (no FWPT/IDWPT):")
    carrier2 = CarrierOscillator()
    carrier2.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier2.gate_on(0)

    pure_carrier = []
    for _ in range(750):
        x_c = carrier2.render_block(0, BLOCK_SIZE)
        pure_carrier.extend(x_c)

    path = audio_io.write_audio(
        "pure_carrier_no_wavelet", pure_carrier, pure_carrier,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
