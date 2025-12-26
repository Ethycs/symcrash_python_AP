"""
Debug Brass Stab
================

Trace through exactly what happens when rendering brass_stab preset.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # for test_config

from test_config import FS, BLOCK_SIZE, BLOCKS_PER_SEC

from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.audio_io import AudioWriter
from symcrash_python_ap.presets import load_preset
from symcrash_python_ap.wavelet.harmonic_injector import WaveType, get_partial_amplitude

# Constants
Q15_ONE = 32767
Q15_HALF = 16384

OUTPUT_DIR = str(Path(__file__).parent.parent / "audio_out")
audio_io = AudioWriter(output_dir=OUTPUT_DIR)

WAVEFORM_MAP = {
    "sine": WaveType.SINE,
    "saw": WaveType.SAW,
    "sawtooth": WaveType.SAW,
    "square": WaveType.PULSE,
    "pulse": WaveType.PULSE,
    "triangle": WaveType.TRIANGLE,
    "tri": WaveType.TRIANGLE,
}


def generate_partial_amps(wave_type: WaveType, num_partials: int = 16) -> list:
    """Generate partial amplitude list for a given waveform type."""
    return [get_partial_amplitude(wave_type, k + 1) for k in range(num_partials)]


def midi_to_freq(note: int) -> float:
    """Convert MIDI note to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def main():
    print("=" * 70)
    print("BRASS STAB DEBUG TRACE")
    print("=" * 70)

    # Load preset
    preset_name = "brass_stab"
    ir = load_preset(preset_name)

    print(f"\nPreset: {ir.meta.name}")
    print(f"\nL0 Graph Nodes:")
    for node in ir.layers.L0_graph.nodes:
        print(f"  - {node.id}: {node.type}")
        if hasattr(node.params, 'waveform'):
            print(f"      waveform: {node.params.waveform}")
        if hasattr(node.params, 'detune_cents'):
            print(f"      detune_cents: {node.params.detune_cents}")
        if hasattr(node.params, 'adsr') and node.params.adsr:
            adsr = node.params.adsr
            print(f"      adsr: A={adsr.a_ms}ms D={adsr.d_ms}ms S={adsr.s} R={adsr.r_ms}ms")

    # Extract waveforms and generate partial amps
    waveform_names = []
    detune_values = []
    for node in ir.layers.L0_graph.nodes:
        if node.type == "Osc" and hasattr(node.params, 'waveform') and node.params.waveform:
            waveform_names.append(node.params.waveform)
            detune = getattr(node.params, 'detune_cents', 0.0) or 0.0
            detune_values.append(detune)

    print(f"\nExtracted waveforms: {waveform_names}")
    print(f"Extracted detunes: {detune_values}")

    # Generate partial amps for saw
    saw_partials = generate_partial_amps(WaveType.SAW, 8)
    print(f"\nSaw partial amps: {saw_partials}")

    # Create pipeline and render a sustained note
    print("\n" + "=" * 70)
    print("RENDERING TEST")
    print("=" * 70)

    pipeline = CarrierLiftPipeline()

    note = 69  # A4 = 440 Hz
    duration_sec = 2.0
    num_blocks = int(duration_sec * BLOCKS_PER_SEC)

    print(f"\nNote: {note} ({midi_to_freq(note):.1f} Hz)")
    print(f"Duration: {duration_sec}s ({num_blocks} blocks)")

    # Start note
    pipeline.voice_on(0, note=note)

    # Check carrier state BEFORE any rendering
    print("\n--- Before first render ---")
    carrier_state = pipeline.carrier.states.get(0)
    if carrier_state:
        print(f"Carrier harmonic_amps_q15: {carrier_state.harmonic_amps_q15}")
        print(f"Carrier phase_inc: {carrier_state.phase_inc}")
    else:
        print("No carrier state found!")

    samples_L = []
    samples_R = []

    for block_idx in range(num_blocks):
        # Simple envelope
        if block_idx < 10:
            env = block_idx / 10.0
        else:
            env = 1.0

        frame = VoiceFrame(
            voice_id=0,
            note=note,
            f0_hz=midi_to_freq(note),
            gain_q15=int(Q15_HALF * env),
            beta_target_q15=0,  # Pure carrier
            partial_amps_q15=saw_partials,
        )

        # Render
        L, R = pipeline.render_block(frame)
        samples_L.extend(L)
        samples_R.extend(R)

        # Debug first block
        if block_idx == 0:
            print("\n--- After first render ---")
            print(f"Frame partial_amps_q15: {frame.partial_amps_q15[:8]}")
            carrier_state = pipeline.carrier.states.get(0)
            if carrier_state:
                print(f"Carrier harmonic_amps_q15: {carrier_state.harmonic_amps_q15}")
            print(f"First 8 L samples: {L[:8]}")
            print(f"L block peak: {max(abs(s) for s in L)}")

    # Release
    pipeline.voice_off(0)

    # Analyze output
    arr = np.array(samples_L, dtype=np.float64)
    peak = int(np.max(np.abs(arr)))
    rms = np.sqrt(np.mean(arr ** 2))

    print(f"\n--- Output Analysis ---")
    print(f"Peak: {peak} ({100*peak/32767:.1f}%)")
    print(f"RMS: {rms:.0f}")

    # FFT
    chunk_start = len(samples_L) // 3
    chunk_end = 2 * len(samples_L) // 3
    chunk = np.array(samples_L[chunk_start:chunk_end], dtype=np.float64)
    if np.max(np.abs(chunk)) > 0:
        chunk = chunk / np.max(np.abs(chunk))

    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / FS)

    low_mask = (freqs > 200) & (freqs < 600)
    if np.any(low_mask):
        low_spectrum = spectrum.copy()
        low_spectrum[~low_mask] = 0
        fund_idx = np.argmax(low_spectrum)
        f0 = freqs[fund_idx]
        fund_power = spectrum[fund_idx]

        print(f"\nFFT Analysis:")
        print(f"  Fundamental: {f0:.1f} Hz")

        for h in range(1, 9):
            h_freq = f0 * h
            h_mask = np.abs(freqs - h_freq) < 20
            if np.any(h_mask):
                h_power = np.max(spectrum[h_mask])
                ratio = h_power / fund_power if fund_power > 0 else 0
                expected = saw_partials[h-1] / Q15_ONE
                print(f"  H{h} ({h_freq:.0f}Hz): {ratio:.3f} (expected ~{expected:.3f})")

    # Save
    metadata = audio_io.write_audio(
        "debug_brass_stab",
        samples_L, samples_R,
        subdirectory="debug",
        timestamp=False
    )
    print(f"\nSaved: {metadata.filepath}")


if __name__ == "__main__":
    main()
