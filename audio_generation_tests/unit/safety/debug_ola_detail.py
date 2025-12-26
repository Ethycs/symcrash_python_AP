"""Debug OLA in detail - inspect what's happening sample by sample."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import math

from symcrash_python_ap.wavelet.carrier import CarrierOscillator
from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet
from symcrash_python_ap.safety.anti_click import AntiClickOperator, AntiClickConfig, hann_window
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
    print("OLA DETAIL DEBUG")
    print("=" * 60)

    note = 69
    saw_partials = generate_partial_amps(WaveType.SAW, 8)
    tree_set = PacketTreeSet()

    # Create carrier and warm it up
    carrier = CarrierOscillator()
    carrier.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier.gate_on(0)

    # Warm up
    for _ in range(10):
        carrier.render_block(0, BLOCK_SIZE)

    # Create anticlick with OLA
    anticlick = AntiClickOperator(AntiClickConfig(
        block_size=BLOCK_SIZE,
        sample_rate=48000,
        wet_latency_blocks=5,
        bloom_time_s=0.001,  # Very fast bloom
        ola_K=16,
        equal_power=True
    ))

    ac_state = anticlick.init_state()
    anticlick.note_on(ac_state)
    anticlick.set_wet_target(ac_state, 0.5)

    # Get the windows
    K = 16
    w_in = hann_window(K)
    w_out = 1.0 - w_in
    print(f"\nOLA Windows (K={K}):")
    print(f"  w_in:  {w_in[:5]}... (should go from 0 to 1)")
    print(f"  w_out: {w_out[:5]}... (should go from 1 to 0)")
    print(f"  Sum:   {(w_in + w_out)[:5]} (should be all 1s)")

    # Render a few blocks and inspect the boundaries
    print("\n" + "=" * 60)
    print("Block-by-block OLA inspection:")
    print("=" * 60)

    prev_mixed = None
    prev_output = None

    for block_idx in range(20):
        x_c = carrier.render_block(0, BLOCK_SIZE)
        x_w = tree_set.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)

        c_np = np.array(x_c, dtype=np.float32) / 32768.0
        w_np = np.array(x_w, dtype=np.float32) / 32768.0

        # Compute what the mix would be (before OLA)
        beta = 0.5
        a_weight = math.cos(beta * math.pi / 2)
        b_weight = math.sin(beta * math.pi / 2)
        mixed_before_ola = a_weight * c_np + b_weight * w_np
        mixed_before_ola = np.clip(mixed_before_ola, -1.0, 1.0).astype(np.float32)

        # Run through anticlick (which applies OLA)
        output = anticlick.step(ac_state, c_np, w_np)

        if block_idx >= 1:  # After first block
            # Check the OLA boundary region
            print(f"\nBlock {block_idx-1} -> {block_idx} boundary:")

            # The tail saved from previous block (should be prev_mixed[-16:])
            # Current block OLA region: output[:16]

            # Calculate expected OLA output
            if prev_mixed is not None:
                expected_ola = w_out * prev_mixed[-K:] + w_in * mixed_before_ola[:K]
                actual_ola = output[:K]

                # Compare
                ola_error = np.abs(expected_ola - actual_ola)
                max_error = np.max(ola_error)

                print(f"  Max OLA error: {max_error:.6f}")

                # Check if there's a jump at the boundary
                if prev_output is not None:
                    prev_end = prev_output[-1]
                    curr_start = output[0]
                    jump = abs(curr_start - prev_end)
                    print(f"  Boundary jump: {jump:.6f} (prev_end={prev_end:.4f}, curr_start={curr_start:.4f})")

                    # Compare to what it would be without OLA
                    if prev_mixed is not None:
                        no_ola_jump = abs(mixed_before_ola[0] - prev_mixed[-1])
                        print(f"  No-OLA jump would be: {no_ola_jump:.6f}")

        prev_mixed = mixed_before_ola.copy()
        prev_output = output.copy()

    # Now let's trace through exactly what OLA does with perfect input
    print("\n" + "=" * 60)
    print("Testing OLA with IDENTICAL carrier and wet (should be noise-free):")
    print("=" * 60)

    carrier2 = CarrierOscillator()
    carrier2.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier2.gate_on(0)

    anticlick2 = AntiClickOperator(AntiClickConfig(
        block_size=BLOCK_SIZE,
        sample_rate=48000,
        wet_latency_blocks=5,
        bloom_time_s=0.001,
        ola_K=16,
        equal_power=True
    ))

    ac_state2 = anticlick2.init_state()
    anticlick2.note_on(ac_state2)
    anticlick2.set_wet_target(ac_state2, 0.5)

    samples_identical = []
    for _ in range(750):
        x_c = carrier2.render_block(0, BLOCK_SIZE)
        c_np = np.array(x_c, dtype=np.float32) / 32768.0

        # Pass IDENTICAL signal as both carrier and wet
        output = anticlick2.step(ac_state2, c_np, c_np.copy())

        output_q15 = (output * 32768.0).astype(np.int32)
        output_q15 = np.clip(output_q15, -32768, 32767)
        samples_identical.extend(output_q15.tolist())

    path = audio_io.write_audio(
        "ola_identical_inputs", samples_identical, samples_identical,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"Saved: {path}")
    print("If this is clean, OLA is fine - the issue is carrier/wet differences")
    print("If this is dirty, OLA itself is broken")

    # Test with different wavelet roundtrip that might introduce differences
    print("\n" + "=" * 60)
    print("Comparing carrier vs wet sample-by-sample:")
    print("=" * 60)

    carrier3 = CarrierOscillator()
    carrier3.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier3.gate_on(0)
    tree_set3 = PacketTreeSet()

    total_diff = 0
    max_diff = 0
    samples = 0

    for block_idx in range(100):
        x_c = carrier3.render_block(0, BLOCK_SIZE)
        x_w = tree_set3.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)

        c_np = np.array(x_c, dtype=np.float32) / 32768.0
        w_np = np.array(x_w, dtype=np.float32) / 32768.0

        diff = np.abs(c_np - w_np)
        total_diff += np.sum(diff)
        max_diff = max(max_diff, np.max(diff))
        samples += BLOCK_SIZE

        if block_idx < 5:
            print(f"Block {block_idx}: max_diff={np.max(diff):.6f}, mean_diff={np.mean(diff):.6f}")

    print(f"\nOverall: max_diff={max_diff:.6f}, mean_diff={total_diff/samples:.6f}")

if __name__ == "__main__":
    main()
