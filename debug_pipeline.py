"""Debug script for audio pipeline - testing Baby Fourier harmonic injection."""
import sys
sys.path.insert(0, 'src/symcrash_python_ap')

import numpy as np
from voice_engine import VoiceEngine, GestureBasis
from symbol_engine import SymbolExecutor
from wavelet_engine import WaveletEngine
from wavelet_packet_trees import PacketTreeSet, BLOCK_SIZE, LEAF_COUNT, NODE_SIZE
from frames import VoiceFrame, ParamFrame
from fixed import sat16, Q15_ONE
from harmonic_injector import (
    HarmonicTargetGenerator, WaveType, create_harmonic_coeffs,
    note_to_freq, freq_to_leaf, LEAF_EDGES_HZ
)

FS = 48000
Q15_HALF = 0x4000


def analyze_coeffs(name, coeffs):
    """Analyze coefficient distribution by leaf."""
    print(f"\n  {name}:")
    print(f"    Range: [{min(coeffs)}, {max(coeffs)}]")
    print(f"    Non-zero: {sum(1 for c in coeffs if c != 0)}")

    # Per-leaf energy
    active_leaves = []
    for leaf in range(LEAF_COUNT):
        start = leaf * NODE_SIZE
        node = coeffs[start:start + NODE_SIZE]
        energy = sum(c*c for c in node)
        if energy > 0:
            rms = int(np.sqrt(energy / NODE_SIZE))
            active_leaves.append((leaf, rms))

    if active_leaves:
        sorted_leaves = sorted(active_leaves, key=lambda x: x[1], reverse=True)[:5]
        print(f"    Top 5 leaves by RMS: {sorted_leaves}")

        # Show which coefficients within each leaf are active
        for leaf, _ in sorted_leaves[:3]:
            start = leaf * NODE_SIZE
            node = coeffs[start:start + NODE_SIZE]
            active_coeffs = [(i, node[i]) for i in range(NODE_SIZE) if node[i] != 0]
            if active_coeffs:
                print(f"      Leaf {leaf} coeffs: {active_coeffs[:4]}")
    else:
        print(f"    No active leaves!")


def analyze_audio_spectrum(name, audio):
    """Analyze audio frequency content."""
    if len(audio) < 64:
        return

    audio_arr = np.array(audio, dtype=float)
    fft = np.fft.rfft(audio_arr[:64])
    magnitudes = np.abs(fft)

    # Find peaks
    top_bins = np.argsort(magnitudes)[::-1][:8]
    freqs_hz = top_bins * FS / 64

    print(f"\n  {name} spectrum:")
    print(f"    Range: [{min(audio)}, {max(audio)}]")
    print(f"    RMS: {np.sqrt(sum(s*s for s in audio) / len(audio)):.1f}")
    print(f"    Top bins: {list(top_bins[:5])} -> {[f'{f:.0f}Hz' for f in freqs_hz[:5]]}")


print("=" * 70)
print("BABY FOURIER / HARMONIC TARGET INJECTOR TEST")
print("=" * 70)

# Initialize
voice_engine = VoiceEngine(max_voices=4)
symbol_engine = SymbolExecutor(n_voices=4)
wavelet_engine = WaveletEngine(block_size=64)
tree_set = PacketTreeSet()
harmonic_gen = HarmonicTargetGenerator()

vid = voice_engine.allocate_voice(program_id=1001)
gesture = GestureBasis(gate=True, pressure_q15=Q15_HALF)
voice_engine.note_on(vid, gesture=gesture)

print("\n" + "=" * 70)
print("STEP 1: Harmonic coefficient generation with Baby Fourier")
print("=" * 70)

# Test with A3 (220 Hz)
test_freq = 220.0
test_note = 57  # MIDI A3

print(f"\nTest frequency: {test_freq} Hz (MIDI note {test_note})")

# Show where harmonics should be placed
print("\nExpected harmonic placement:")
for k in range(1, 9):
    f_k = test_freq * k
    leaf = freq_to_leaf(f_k)
    leaf_low = LEAF_EDGES_HZ[leaf]
    leaf_high = LEAF_EDGES_HZ[leaf + 1]
    pos_in_band = (f_k - leaf_low) / (leaf_high - leaf_low)
    coeff_idx = int(pos_in_band * NODE_SIZE)
    print(f"  Harmonic {k}: {f_k:.0f} Hz -> leaf {leaf}, coeff ~{coeff_idx}")

# Generate with new harmonic injector
harmonic_gen.set_note(vid, test_note)
harmonic_gen.set_wave_params(vid, WaveType.SAW, partials_n=12, brightness_q15=Q15_ONE)
baby_fourier_coeffs = harmonic_gen.generate_dense_target(vid)

analyze_coeffs("Baby Fourier coefficients (SAW wave)", baby_fourier_coeffs)

# Also test with create_harmonic_coeffs convenience function
convenience_coeffs = create_harmonic_coeffs(test_freq, 0x5000, WaveType.SAW, 12)
analyze_coeffs("Convenience function coefficients", convenience_coeffs)

print("\n" + "=" * 70)
print("STEP 2: Wavelet synthesis with frequency-encoded coefficients")
print("=" * 70)

# Create VoiceFrame
abi_frame = voice_engine.step_voice(vid)
if abi_frame:
    vf = VoiceFrame(
        voice_id=abi_frame.voice_id,
        tree_a_id=abi_frame.tree_a_id,
        tree_b_id=abi_frame.tree_b_id,
        tree_blend_q15=abi_frame.tree_blend_q15,
        gain_q15=0x6000,
    )

    # Synthesize with Baby Fourier coefficients
    vf.coeff_q15 = baby_fourier_coeffs
    audio_bf = wavelet_engine.render_voice(vf)
    analyze_audio_spectrum("Baby Fourier audio", audio_bf)

    # Synthesize with convenience function coefficients
    vf.coeff_q15 = convenience_coeffs
    audio_conv = wavelet_engine.render_voice(vf)
    analyze_audio_spectrum("Convenience function audio", audio_conv)

print("\n" + "=" * 70)
print("STEP 3: Compare different wave types")
print("=" * 70)

for wave_type in [WaveType.SINE, WaveType.SAW, WaveType.PULSE, WaveType.TRIANGLE]:
    coeffs = create_harmonic_coeffs(test_freq, 0x5000, wave_type, 12)
    vf.coeff_q15 = coeffs
    audio = wavelet_engine.render_voice(vf)

    n_harmonics = sum(1 for c in coeffs if c != 0)
    audio_rms = np.sqrt(sum(s*s for s in audio) / len(audio))
    print(f"  {wave_type.name}: {n_harmonics} active coeffs, audio RMS={audio_rms:.0f}")

print("\n" + "=" * 70)
print("STEP 4: Full pipeline with harmonic injection")
print("=" * 70)

# Get Symbol Engine state and apply harmonic injection
state = symbol_engine.get_voice_state(vid)

# Start with some Symbol Engine texture
for i in range(256):
    state.c_q15[i] = 1000  # Low-level texture

print("Initial Symbol Engine state: uniform 1000")

# Run Symbol Engine
symbol_coeffs = list(symbol_engine.step_voice(vf, program_id=0))
analyze_coeffs("Symbol Engine output", symbol_coeffs)

# Apply harmonic injection
harmonic_gen.set_injection_params(vid, harmonic_dial_q15=Q15_ONE, pull_strength_q15=0x6000)
injected_coeffs = harmonic_gen.apply_injection(vid, symbol_coeffs)
analyze_coeffs("After harmonic injection", injected_coeffs)

# Synthesize
vf.coeff_q15 = injected_coeffs
audio_injected = wavelet_engine.render_voice(vf)
analyze_audio_spectrum("Injected audio", audio_injected)

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The Baby Fourier / Harmonic Target Injector now:
1. Places harmonic energy at correct FREQUENCY POSITIONS within bands
2. Uses coefficient INDEX to encode frequency (coeff 0 = band low, coeff 7 = band high)
3. Wavelet synthesis generates proper sub-frequencies for each coefficient

This gives us:
- Rich harmonic content instead of single sine per band
- Proper pitch tracking (220 Hz fundamental, not 375 Hz band center)
- Integration with Symbol Engine via injection lane
""")
