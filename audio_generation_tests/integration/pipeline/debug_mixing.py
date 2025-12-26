"""Debug the mixing stage to find where noise is introduced."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np

from symcrash_python_ap.wavelet.carrier import CarrierOscillator
from symcrash_python_ap.wavelet.wavelet_packet_trees import PacketTreeSet
from symcrash_python_ap.safety.anti_click import AntiClickOperator, AntiClickConfig
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
    print("MIXING DEBUG - Isolating the noise source")
    print("=" * 60)

    note = 69  # A4
    saw_partials = generate_partial_amps(WaveType.SAW, 8)

    # Create carrier
    carrier = CarrierOscillator()
    carrier.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier.gate_on(0)

    # Create tree set for wet path
    tree_set = PacketTreeSet()

    # Create anticlick operator
    anticlick = AntiClickOperator(AntiClickConfig(
        block_size=BLOCK_SIZE,
        sample_rate=48000,
        wet_latency_blocks=5,
        bloom_time_s=0.05,
        ola_K=16,
        equal_power=True
    ))

    # Initialize anticlick state
    ac_state = anticlick.init_state()
    anticlick.note_on(ac_state)

    # Warm up carrier
    for _ in range(10):
        carrier.render_block(0, BLOCK_SIZE)

    # Get stable carrier block
    x_carrier = carrier.render_block(0, BLOCK_SIZE)
    print(f"\n1. Carrier block (first 10): {x_carrier[:10]}")
    print(f"   Peak: {max(abs(s) for s in x_carrier)}")

    # Get wet block via effect_mode
    x_wet = tree_set.effect_mode(tree_id=0, carrier_audio=x_carrier, mask_q15=None)
    print(f"\n2. Wet block (first 10): {x_wet[:10]}")
    print(f"   Peak: {max(abs(s) for s in x_wet)}")

    # Check if they're similar
    diff = [abs(a - b) for a, b in zip(x_carrier, x_wet)]
    print(f"   Max diff from carrier: {max(diff)}")

    # Now test the anticlick mixing
    print("\n3. ANTICLICK MIXING TEST:")

    # Convert to float [-1, 1]
    carrier_np = np.array(x_carrier, dtype=np.float32) / 32768.0
    wet_np = np.array(x_wet, dtype=np.float32) / 32768.0

    print(f"   carrier_np range: [{carrier_np.min():.4f}, {carrier_np.max():.4f}]")
    print(f"   wet_np range: [{wet_np.min():.4f}, {wet_np.max():.4f}]")

    # Set wet target to 0.5
    anticlick.set_wet_target(ac_state, 0.5)

    # Step through several blocks
    for i in range(10):
        x_c = carrier.render_block(0, BLOCK_SIZE)
        x_w = tree_set.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)

        c_np = np.array(x_c, dtype=np.float32) / 32768.0
        w_np = np.array(x_w, dtype=np.float32) / 32768.0

        output = anticlick.step(ac_state, c_np, w_np)

        # Convert back to Q15 for analysis
        output_q15 = (output * 32768.0).astype(np.int32)
        output_q15 = np.clip(output_q15, -32768, 32767)

        peak = int(np.max(np.abs(output_q15)))
        rms = np.sqrt(np.mean(output_q15.astype(float)**2))

        print(f"   Block {i}: beta={ac_state.beta:.3f}, Peak={peak:5d}, RMS={rms:7.1f}, wet_ready={ac_state.wet_ready}")

    # Now render full 1 second with manual mixing (bypass anticlick)
    print("\n4. MANUAL MIXING TEST (bypass anticlick):")

    carrier2 = CarrierOscillator()
    carrier2.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier2.gate_on(0)

    samples_manual_mix = []
    for _ in range(750):
        x_c = carrier2.render_block(0, BLOCK_SIZE)
        x_w = tree_set.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)

        # Simple 50/50 mix (no anticlick)
        for i in range(BLOCK_SIZE):
            mixed = (x_c[i] + x_w[i]) // 2
            samples_manual_mix.append(mixed)

    path = audio_io.write_audio(
        "manual_mix_50_50", samples_manual_mix, samples_manual_mix,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"   Saved: {path}")

    # Compare to just carrier through effect_mode roundtrip
    print("\n5. PURE WET (carrier -> FWPT -> IDWPT only):")

    carrier3 = CarrierOscillator()
    carrier3.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier3.gate_on(0)

    samples_pure_wet = []
    for _ in range(750):
        x_c = carrier3.render_block(0, BLOCK_SIZE)
        x_w = tree_set.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)
        samples_pure_wet.extend(x_w)

    path = audio_io.write_audio(
        "pure_wet_roundtrip", samples_pure_wet, samples_pure_wet,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"   Saved: {path}")

    print("\nCheck these files:")
    print("  - manual_mix_50_50.wav (simple average, no anticlick)")
    print("  - pure_wet_roundtrip.wav (just FWPT->IDWPT)")

if __name__ == "__main__":
    main()
