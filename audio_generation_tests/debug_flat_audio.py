"""Debug why audio sounds flat."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from voice_engine import VoiceEngine, GestureBasis
from symbol_engine import SymbolExecutor
from wavelet_engine import WaveletEngine
from frames import VoiceFrame
from fixed import sat16
import numpy as np

print("=" * 60)
print("DIAGNOSIS: Symbol Engine NormalizeScale destroys harmonics")
print("=" * 60)
print()
print("The issue: inject_note_coefficients() modifies Symbol Engine's")
print("internal c_q15 state. But step_voice() runs apply_tail() which")
print("includes NormalizeScale - this caps energy per-band to a 'pink'")
print("target curve, destroying the harmonic structure we injected.")
print()

# Solution: Inject harmonics AFTER Symbol Engine, directly into VoiceFrame.coeff_q15
print("SOLUTION: Inject harmonics directly into VoiceFrame.coeff_q15")
print("          after Symbol Engine processing, before Wavelet Engine.")
print()

# Demo the fix
voice_engine = VoiceEngine(max_voices=4)
symbol_engine = SymbolExecutor(n_voices=4)
wavelet_engine = WaveletEngine(block_size=64)

vid = voice_engine.allocate_voice(program_id=1001)
gesture = GestureBasis(gate=True, pressure_q15=16384)
voice_engine.note_on(vid, gesture=gesture)


def create_harmonic_coeffs(frequency=220.0, amplitude=0x5000, sample_rate=48000):
    """Create harmonic coefficient pattern directly."""
    coeffs = [0] * 256
    nyquist = sample_rate / 2
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


# Run a block
abi_frame = voice_engine.step_voice(vid)

# Get Symbol Engine output (flat/normalized)
vf = VoiceFrame(
    voice_id=abi_frame.voice_id,
    tree_a_id=abi_frame.tree_a_id,
    tree_b_id=abi_frame.tree_b_id,
    tree_blend_q15=abi_frame.tree_blend_q15,
    gain_q15=abi_frame.gain_q15,
    coeff_q15=[0] * 256,
)
symbol_coeffs = symbol_engine.step_voice(vf, program_id=0)

print("Symbol Engine output (normalized to pink curve):")
print(f"  max coeff: {max(abs(c) for c in symbol_coeffs)}")
print(f"  first 16: {list(symbol_coeffs[:16])}")

# Now create our own harmonic coeffs and ADD to symbol output
harmonic_coeffs = create_harmonic_coeffs(frequency=220.0, amplitude=0x4000)
print()
print("Harmonic injection coefficients:")
print(f"  max coeff: {max(abs(c) for c in harmonic_coeffs)}")
print(f"  first 16: {harmonic_coeffs[:16]}")

# Combine: symbol output + harmonic injection
combined_coeffs = [sat16(s + h) for s, h in zip(symbol_coeffs, harmonic_coeffs)]
print()
print("Combined coefficients (symbol + harmonics):")
print(f"  max coeff: {max(abs(c) for c in combined_coeffs)}")
print(f"  first 16: {combined_coeffs[:16]}")

# Render with combined coeffs
vf.coeff_q15 = combined_coeffs
audio = wavelet_engine.render_voice(vf)

crossings = sum(1 for i in range(1, len(audio)) if audio[i] * audio[i-1] < 0)
print()
print("Audio output with combined coeffs:")
print(f"  max amplitude: {max(abs(s) for s in audio)}")
print(f"  zero crossings: {crossings}")
print()

# Compare to just using symbol coeffs
vf.coeff_q15 = list(symbol_coeffs)
audio_flat = wavelet_engine.render_voice(vf)
crossings_flat = sum(1 for i in range(1, len(audio_flat)) if audio_flat[i] * audio_flat[i-1] < 0)

print("Audio output with ONLY symbol coeffs (flat sound):")
print(f"  max amplitude: {max(abs(s) for s in audio_flat)}")
print(f"  zero crossings: {crossings_flat}")
print()

print("=" * 60)
print("FIX: Update render_instrument() to inject harmonics AFTER")
print("     symbol_engine.step_voice(), not into symbol engine state.")
print("=" * 60)
