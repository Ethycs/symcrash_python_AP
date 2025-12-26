"""Debug OLA (Overlap-Add) as the noise source."""

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
    print("OLA DEBUG - Testing with and without OLA")
    print("=" * 60)

    note = 69
    saw_partials = generate_partial_amps(WaveType.SAW, 8)
    tree_set = PacketTreeSet()

    # Test 1: AntiClick WITH OLA (ola_K=16) - current behavior
    print("\n1. With OLA (ola_K=16):")
    carrier1 = CarrierOscillator()
    carrier1.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier1.gate_on(0)

    anticlick_with_ola = AntiClickOperator(AntiClickConfig(
        block_size=BLOCK_SIZE,
        sample_rate=48000,
        wet_latency_blocks=5,
        bloom_time_s=0.05,
        ola_K=16,  # OLA enabled
        equal_power=True
    ))

    ac_state1 = anticlick_with_ola.init_state()
    anticlick_with_ola.note_on(ac_state1)
    anticlick_with_ola.set_wet_target(ac_state1, 0.5)

    samples_with_ola = []
    for _ in range(750):
        x_c = carrier1.render_block(0, BLOCK_SIZE)
        x_w = tree_set.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)

        c_np = np.array(x_c, dtype=np.float32) / 32768.0
        w_np = np.array(x_w, dtype=np.float32) / 32768.0

        output = anticlick_with_ola.step(ac_state1, c_np, w_np)
        output_q15 = (output * 32768.0).astype(np.int32)
        output_q15 = np.clip(output_q15, -32768, 32767)
        samples_with_ola.extend(output_q15.tolist())

    path = audio_io.write_audio(
        "anticlick_with_ola", samples_with_ola, samples_with_ola,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"   Saved: {path}")

    # Test 2: AntiClick WITHOUT OLA (ola_K=0)
    print("\n2. Without OLA (ola_K=0):")
    carrier2 = CarrierOscillator()
    carrier2.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier2.gate_on(0)

    anticlick_no_ola = AntiClickOperator(AntiClickConfig(
        block_size=BLOCK_SIZE,
        sample_rate=48000,
        wet_latency_blocks=5,
        bloom_time_s=0.05,
        ola_K=0,  # OLA disabled
        equal_power=True
    ))

    ac_state2 = anticlick_no_ola.init_state()
    anticlick_no_ola.note_on(ac_state2)
    anticlick_no_ola.set_wet_target(ac_state2, 0.5)

    samples_no_ola = []
    for _ in range(750):
        x_c = carrier2.render_block(0, BLOCK_SIZE)
        x_w = tree_set.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)

        c_np = np.array(x_c, dtype=np.float32) / 32768.0
        w_np = np.array(x_w, dtype=np.float32) / 32768.0

        output = anticlick_no_ola.step(ac_state2, c_np, w_np)
        output_q15 = (output * 32768.0).astype(np.int32)
        output_q15 = np.clip(output_q15, -32768, 32767)
        samples_no_ola.extend(output_q15.tolist())

    path = audio_io.write_audio(
        "anticlick_no_ola", samples_no_ola, samples_no_ola,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"   Saved: {path}")

    # Test 3: Just _mix2 without OLA (direct mix test)
    print("\n3. Direct _mix2 only (no OLA, no state):")
    carrier3 = CarrierOscillator()
    carrier3.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier3.gate_on(0)

    samples_direct_mix = []
    for _ in range(750):
        x_c = carrier3.render_block(0, BLOCK_SIZE)
        x_w = tree_set.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)

        c_np = np.array(x_c, dtype=np.float32) / 32768.0
        w_np = np.array(x_w, dtype=np.float32) / 32768.0

        # Direct equal-power mix at beta=0.5
        import math
        beta = 0.5
        a_weight = math.cos(beta * math.pi / 2)
        b_weight = math.sin(beta * math.pi / 2)
        mixed = a_weight * c_np + b_weight * w_np
        mixed = np.clip(mixed, -1.0, 1.0)

        output_q15 = (mixed * 32768.0).astype(np.int32)
        output_q15 = np.clip(output_q15, -32768, 32767)
        samples_direct_mix.extend(output_q15.tolist())

    path = audio_io.write_audio(
        "direct_equal_power_mix", samples_direct_mix, samples_direct_mix,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"   Saved: {path}")

    print("\nCompare these files:")
    print("  - anticlick_with_ola.wav (OLA enabled)")
    print("  - anticlick_no_ola.wav (OLA disabled)")
    print("  - direct_equal_power_mix.wav (no anticlick at all)")

if __name__ == "__main__":
    main()
