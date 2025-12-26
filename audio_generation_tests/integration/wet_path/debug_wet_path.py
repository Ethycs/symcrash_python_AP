"""Debug the wet path to find where noise is introduced."""

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
Q15_HALF = 16384
BLOCK_SIZE = 64

def generate_partial_amps(wave_type: WaveType, num_partials: int = 8):
    return [get_partial_amplitude(wave_type, k + 1) for k in range(num_partials)]

def analyze_signal(name, samples):
    """Analyze signal quality."""
    arr = np.array(samples, dtype=np.float64)
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr**2))
    zero_crossings = np.sum(np.diff(np.sign(arr)) != 0)

    print(f"  {name}:")
    print(f"    Peak: {peak}, RMS: {rms:.1f}, Zero crossings: {zero_crossings}")
    print(f"    First 10 samples: {samples[:10]}")

def main():
    print("=" * 60)
    print("WET PATH DEBUG")
    print("=" * 60)

    # Create carrier oscillator
    carrier = CarrierOscillator()
    saw_partials = generate_partial_amps(WaveType.SAW, 8)

    note = 69  # A4 (440 Hz)
    carrier.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier.gate_on(0)

    # Render several blocks to stabilize
    for _ in range(5):
        carrier.render_block(0, BLOCK_SIZE)

    # Get a stable carrier block
    x_carrier = carrier.render_block(0, BLOCK_SIZE)

    print("\n1. CARRIER OUTPUT (before wet path):")
    analyze_signal("Carrier", x_carrier)

    # Create packet tree set
    tree_set = PacketTreeSet()

    # Test effect_mode with identity mask (should return ~same as input)
    print("\n2. EFFECT MODE - Identity (no mask):")
    x_wet_identity = tree_set.effect_mode(
        tree_id=0,
        carrier_audio=x_carrier,
        mask_q15=None
    )
    analyze_signal("Wet (identity)", x_wet_identity)

    # Check reconstruction error
    diff = [abs(a - b) for a, b in zip(x_carrier, x_wet_identity)]
    print(f"    Max reconstruction error: {max(diff)}")
    print(f"    Mean reconstruction error: {np.mean(diff):.1f}")

    # Test with explicit identity mask (all Q15_ONE)
    print("\n3. EFFECT MODE - Explicit identity mask:")
    identity_mask = [Q15_ONE] * 64
    x_wet_explicit = tree_set.effect_mode(
        tree_id=0,
        carrier_audio=x_carrier,
        mask_q15=identity_mask
    )
    analyze_signal("Wet (explicit identity)", x_wet_explicit)

    # Test just FWPT -> IDWPT roundtrip
    print("\n4. MANUAL FWPT -> IDWPT ROUNDTRIP:")
    analyzer = tree_set.analyzers[0]
    synth = tree_set.synthesizers[0]

    # Scale down (same as effect_mode does)
    HEADROOM_BITS = 3
    scaled_input = [s >> HEADROOM_BITS for s in x_carrier]
    print(f"    Scaled input (>>3): {scaled_input[:10]}")

    # FWPT
    coeff_nodes = analyzer.analyze(scaled_input)
    print(f"    Num coeff nodes: {len(coeff_nodes)}")
    print(f"    First node (first 4): {coeff_nodes[0][:4] if coeff_nodes else 'empty'}")

    # Check total coefficient energy
    total_coeff_energy = sum(sum(c**2 for c in node) for node in coeff_nodes)
    total_input_energy = sum(s**2 for s in scaled_input)
    print(f"    Input energy: {total_input_energy}")
    print(f"    Coeff energy: {total_coeff_energy}")

    # IDWPT
    result = synth.synthesize_full(coeff_nodes, None)
    print(f"    Synth output (first 10): {result[:10]}")

    # Scale back up
    output = []
    for s in result:
        scaled_up = s << HEADROOM_BITS
        if scaled_up > 32767:
            scaled_up = 32767
        elif scaled_up < -32768:
            scaled_up = -32768
        output.append(scaled_up)

    analyze_signal("Manual roundtrip", output)

    # Now render 1 second and save for visual inspection
    print("\n5. RENDERING 1 SECOND FOR VISUAL INSPECTION...")

    carrier2 = CarrierOscillator()
    carrier2.configure_voice(0, note, harmonic_amps=saw_partials)
    carrier2.gate_on(0)

    carrier_samples = []
    wet_samples = []

    for _ in range(750):  # 1 second at 48kHz/64
        x_c = carrier2.render_block(0, BLOCK_SIZE)
        x_w = tree_set.effect_mode(tree_id=0, carrier_audio=x_c, mask_q15=None)

        carrier_samples.extend(x_c)
        wet_samples.extend(x_w)

    # Save both
    carrier_path = audio_io.write_audio(
        "debug_carrier_only", carrier_samples, carrier_samples,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"Saved carrier: {carrier_path}")

    wet_path = audio_io.write_audio(
        "debug_wet_only", wet_samples, wet_samples,
        subdirectory="debug", timestamp=False
    ).filepath
    print(f"Saved wet: {wet_path}")

    # Analysis
    print("\nFinal analysis:")
    analyze_signal("Carrier (full)", carrier_samples[:64])
    analyze_signal("Wet (full)", wet_samples[:64])

if __name__ == "__main__":
    main()
