"""Debug script for audio pipeline."""
import sys
sys.path.insert(0, 'src/symcrash_python_ap')

from voice_engine import VoiceEngine, GestureBasis
from symbol_engine import SymbolExecutor
from wavelet_engine import WaveletEngine
from frames import VoiceFrame, ParamFrame

# Initialize
voice_engine = VoiceEngine(max_voices=4)
symbol_engine = SymbolExecutor(n_voices=4)
wavelet_engine = WaveletEngine(block_size=64)

vid = voice_engine.allocate_voice(program_id=1001)
gesture = GestureBasis(gate=True, pressure_q15=16384)
voice_engine.note_on(vid, gesture=gesture)

# Inject initial coefficients into Symbol Engine
# This is what gives the sound its initial energy
state = symbol_engine.get_voice_state(vid)
for i in range(256):
    # Simple harmonic-ish initial state
    state.c_q15[i] = 8000 if i < 32 else 2000
print(f"Injected initial coefficients into voice {vid}")

# Run a few blocks
for i in range(10):
    abi_frame = voice_engine.step_voice(vid)
    if abi_frame:
        print(f"Block {i}: gain={abi_frame.gain_q15}, tree_a={abi_frame.tree_a_id}, tree_b={abi_frame.tree_b_id}")

        # Create VoiceFrame
        vf = VoiceFrame(
            voice_id=abi_frame.voice_id,
            tree_a_id=abi_frame.tree_a_id,
            tree_b_id=abi_frame.tree_b_id,
            tree_blend_q15=abi_frame.tree_blend_q15,
            gain_q15=abi_frame.gain_q15,
        )

        # Symbol engine
        coeffs = symbol_engine.step_voice(vf, program_id=0)
        non_zero_coeffs = sum(1 for c in coeffs if c != 0)
        print(f"  Coeffs: min={min(coeffs)}, max={max(coeffs)}, non-zero={non_zero_coeffs}")

        vf.coeff_q15 = list(coeffs)

        # Wavelet engine
        audio = wavelet_engine.render_voice(vf)
        non_zero_audio = sum(1 for a in audio if a != 0)
        print(f"  Audio: min={min(audio)}, max={max(audio)}, non-zero={non_zero_audio}")
    else:
        print(f"Block {i}: no frame")
