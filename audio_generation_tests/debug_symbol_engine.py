"""Debug script to trace Symbol Engine coefficient flow."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from symcrash_python_ap.voice.voice_engine import VoiceEngine, GestureBasis
from symcrash_python_ap.symbol.symbol_engine import SymbolExecutor
from symcrash_python_ap.wavelet.wavelet_engine import WaveletEngine
from symcrash_python_ap.frames import VoiceFrame, ParamFrame
from symcrash_python_ap.fixed import sat16, Q15_ONE

# Constants
FS = 48000
BLOCK_SIZE = 64
Q15_HALF = 0x4000


def create_harmonic_coeffs(frequency: float = 220.0, amplitude: int = 0x5000) -> list:
    """Create harmonic coefficient pattern."""
    coeffs = [0] * 256
    nyquist = FS / 2
    band_width = nyquist / 32

    for harmonic in range(1, 16):
        freq = frequency * harmonic
        if freq >= nyquist * 0.95:
            break

        leaf_idx = int(freq / band_width)
        leaf_idx = max(0, min(31, leaf_idx))
        harm_amp = amplitude // harmonic
        node_start = leaf_idx * 8

        for i in range(8):
            mod = 1.0 + 0.15 * np.sin(2 * np.pi * i / 8 + harmonic * 0.7)
            coeff_val = int(harm_amp * mod)
            coeffs[node_start + i] = sat16(coeffs[node_start + i] + coeff_val)

    return coeffs


def analyze_coefficients(name: str, coeffs: list):
    """Analyze coefficient distribution."""
    print(f"\n{name}:")

    # Per-band analysis (6 bands of varying sizes as per operator_library)
    # Actually the code uses 32 leaves of 8 coefficients each
    n_leaves = 32
    node_size = 8

    print(f"  Total coeffs: {len(coeffs)}")
    print(f"  Range: [{min(coeffs)}, {max(coeffs)}]")
    print(f"  Non-zero: {sum(1 for c in coeffs if c != 0)}")
    print(f"  RMS: {np.sqrt(sum(c*c for c in coeffs) / len(coeffs)):.1f}")

    # Energy per leaf
    leaf_energies = []
    for leaf in range(n_leaves):
        start = leaf * node_size
        end = start + node_size
        energy = sum(c*c for c in coeffs[start:end])
        leaf_energies.append(energy)

    # Find active leaves
    active_leaves = [(i, e) for i, e in enumerate(leaf_energies) if e > 0]
    if active_leaves:
        print(f"  Active leaves: {len(active_leaves)}")
        # Show top 5 by energy
        sorted_leaves = sorted(active_leaves, key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 leaves: {[(leaf, int(np.sqrt(e))) for leaf, e in sorted_leaves]}")
    else:
        print("  No active leaves!")


def analyze_audio(name: str, audio: list):
    """Analyze audio output."""
    print(f"\n{name}:")
    print(f"  Samples: {len(audio)}")
    print(f"  Range: [{min(audio)}, {max(audio)}]")
    print(f"  RMS: {np.sqrt(sum(s*s for s in audio) / len(audio)):.1f}")

    # Simple FFT to see frequency content
    audio_arr = np.array(audio, dtype=float)
    if len(audio_arr) >= 64:
        fft = np.fft.rfft(audio_arr[:64])
        magnitudes = np.abs(fft)

        # Find dominant frequencies
        top_bins = np.argsort(magnitudes)[::-1][:5]
        freqs = top_bins * FS / 64
        print(f"  Dominant freq bins: {list(top_bins)}")
        print(f"  Approx freqs (Hz): {[f'{f:.0f}' for f in freqs]}")


def main():
    print("=" * 70)
    print("SYMBOL ENGINE COEFFICIENT FLOW ANALYSIS")
    print("=" * 70)

    # Initialize engines
    voice_engine = VoiceEngine(max_voices=4)
    symbol_engine = SymbolExecutor(n_voices=4)
    wavelet_engine = WaveletEngine(block_size=BLOCK_SIZE)

    # Allocate voice
    vid = voice_engine.allocate_voice(program_id=1001)
    gesture = GestureBasis(gate=True, pressure_q15=Q15_HALF)
    voice_engine.note_on(vid, gesture=gesture)

    # Create harmonic coefficients
    harmonic_coeffs = create_harmonic_coeffs(frequency=220.0, amplitude=0x5000)
    analyze_coefficients("Harmonic coefficients (what we want)", harmonic_coeffs)

    # Get voice state and check Symbol Engine's internal state
    state = symbol_engine.get_voice_state(vid)
    analyze_coefficients("Symbol Engine initial state", state.c_q15)

    # Run voice engine step
    abi_frame = voice_engine.step_voice(vid)
    if not abi_frame:
        print("ERROR: No ABI frame returned")
        return

    print(f"\nVoice Engine ABI frame:")
    print(f"  gain_q15: {abi_frame.gain_q15}")
    print(f"  tree_a_id: {abi_frame.tree_a_id}, tree_b_id: {abi_frame.tree_b_id}")

    # Create VoiceFrame for Symbol Engine
    vf = VoiceFrame(
        voice_id=abi_frame.voice_id,
        tree_a_id=abi_frame.tree_a_id,
        tree_b_id=abi_frame.tree_b_id,
        tree_blend_q15=abi_frame.tree_blend_q15,
        gain_q15=abi_frame.gain_q15,
        tilt_env_q15=0x6000,
        density_env_q15=0x5000,
        material_id=0,
    )

    # Get Symbol Engine output
    symbol_coeffs = symbol_engine.step_voice(vf, program_id=0)
    analyze_coefficients("Symbol Engine output (after operators)", list(symbol_coeffs))

    # Combine with harmonics (current approach)
    combined_coeffs = [sat16(s + h) for s, h in zip(symbol_coeffs, harmonic_coeffs)]
    analyze_coefficients("Combined coefficients (symbol + harmonic)", combined_coeffs)

    # Render with wavelet engine - symbol output only
    vf.coeff_q15 = list(symbol_coeffs)
    audio_symbol_only = wavelet_engine.render_voice(vf)
    analyze_audio("Audio from Symbol Engine only", audio_symbol_only)

    # Render with wavelet engine - combined
    vf.coeff_q15 = combined_coeffs
    audio_combined = wavelet_engine.render_voice(vf)
    analyze_audio("Audio from combined coefficients", audio_combined)

    # Render with wavelet engine - harmonics only
    vf.coeff_q15 = harmonic_coeffs
    audio_harmonics_only = wavelet_engine.render_voice(vf)
    analyze_audio("Audio from harmonics only", audio_harmonics_only)

    # Check what the wavelet engine synthesis does
    print("\n" + "=" * 70)
    print("WAVELET ENGINE SYNTHESIS ANALYSIS")
    print("=" * 70)

    # The wavelet engine uses frequency-band modulation
    # Each leaf represents a frequency band
    # Let's see what frequencies we expect from our harmonic coeffs

    print("\nExpected frequency bands from harmonics (220 Hz fundamental):")
    nyquist = FS / 2
    band_width = nyquist / 32

    for harmonic in range(1, 8):
        freq = 220.0 * harmonic
        leaf_idx = int(freq / band_width)
        leaf_start_freq = leaf_idx * band_width
        leaf_end_freq = (leaf_idx + 1) * band_width
        print(f"  Harmonic {harmonic}: {freq:.0f} Hz -> leaf {leaf_idx} ({leaf_start_freq:.0f}-{leaf_end_freq:.0f} Hz)")

    # What the wavelet engine actually uses for synthesis
    print("\nWavelet Engine synthesis uses carrier at CENTER of each band.")
    print("This may not match the actual harmonic frequencies!")

    # Let's check the actual implementation
    print("\nCheck PacketSynthesizer.synthesize() implementation:")
    print("  It generates: carrier = sin(omega * i + phase_offset)")
    print("  where omega = 2 * pi * freq_center / sample_rate")
    print("  This creates a sine wave at the band center, NOT the actual harmonic freq.")


if __name__ == "__main__":
    main()
